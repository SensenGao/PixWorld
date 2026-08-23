#!/usr/bin/env python
"""PixWorld stage 1: fine-tune Wan2.2-TI2V-5B into a pixel-space multi-view 3D generator.

Run it with ``train/train_l2p.sh``, which sets the environment this file assumes (gsplat
JIT warm-up, NCCL, allocator).

One step draws a scene of ``V`` input views plus ``K`` held-out novel views, noises the
input views at a sigma from :class:`~schedule.SigmaSchedule`, runs the transformer on 13
channels per view to predict ``x0``, and -- when the sigma is below ``--gs_sigma_hi`` --
lifts to a 3D Gaussian field and renders the target views for the render, depth and
perceptual terms.

FSDP2 shards the transformer and the pixel decoder; the Gaussian encoder/decoder/head are
replicated in fp32 and synchronised by an explicit gradient all-reduce each step.
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

# Everything this script needs lives beside it -- no imports from anywhere else.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry.cameras import create_raymaps  # noqa: E402
from data.dataset import (MultiViewDataset, collate_mv,  # noqa: E402
                                   gather_targets, source_sampler, val_scenes)
from objectives.geometry_loss import (GEOMETRY_BACKENDS, GEOMETRY_DISTANCES,  # noqa: E402
                                    GeometryPerceptualLoss)
from objectives.losses import (anchor_loss, depth_logit_loss,  # noqa: E402
                             gs_sigma_weight, render_photo_loss)
from models.dit import PixWorldMV, WAN22_5B_PIXEL_CONFIG, install_rope_precast
from models.gaussian_head import PixelAlignedGaussianHead, depth_tv  # noqa: E402
from models.gs_lift import GaussianDecoder, GaussianEncoder, gs_lift  # noqa: E402
from geometry.render import prune_opacity, render_views  # noqa: E402
from diffusion.sampling import sample  # noqa: E402
from diffusion.schedule import SigmaSchedule  # noqa: E402
from utils.cuda_utils import warm_cusolver  # noqa: E402
from objectives.perceptual import build_lpips  # noqa: E402
from data.text_encoder import (DEFAULT_NEG, DEFAULT_PROMPTS,  # noqa: E402
                          MV_PREFIX, TextEncoder)

try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
except ImportError:                                                      # pragma: no cover
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.checkpoint.state_dict import (  # noqa: E402
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    set_optimizer_state_dict)
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402


def get_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Wan2.2-TI2V-5B into a pixel-space multi-view 3D generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("paths")
    g.add_argument("--data_root", required=True,
                   help="dataset root holding index.jsonl (see docs/DATASET.md)")
    g.add_argument("--index", default="index.jsonl")
    g.add_argument("--wan_path", required=True,
                   help="Wan2.2-TI2V-5B-Diffusers directory: the transformer is the "
                        "cold-start backbone and the text_encoder is used throughout")
    g.add_argument("--out_dir", required=True)
    g.add_argument("--resume", default="",
                   help="checkpoint to continue from; empty means cold-start from --wan_path")
    g.add_argument("--model_config", default="",
                   help="optional JSON (inline or a file path) overriding the transformer "
                        "config, for training a smaller variant or for smoke tests")
    g.add_argument("--wan_init", type=int, default=1,
                   help="load Wan2.2's pretrained transformer at a cold start. 0 trains "
                        "the transformer from scratch, which is what smoke tests use.")

    g = p.add_argument_group("views and resolution")
    g.add_argument("--views", type=int, default=16, help="input views V")
    g.add_argument("--k_novel", type=int, default=16, help="held-out novel views K")
    g.add_argument("--n_target", type=int, default=16,
                   help="how many of the V+K views are rendered and scored; keeps the "
                        "render cost independent of K")
    g.add_argument("--both_ends_prob", type=float, default=1.0)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--width", type=int, default=832)
    g.add_argument("--batch_size", type=int, default=1, help="scenes per rank")
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--source_weights", default="",
                   help='JSON map of source tag -> fraction, e.g. \'{"a":0.7,"b":0.3}\'. '
                        "Empty means sample proportionally to pool size.")

    g = p.add_argument_group("noise schedule")
    g.add_argument("--shift", type=float, default=16.0,
                   help="schedule shift; higher pushes more mass to high noise")
    g.add_argument("--sigma_lo_frac", type=float, default=0.2,
                   help="fraction of steps below --gs_sigma_hi, i.e. the 3D branch's step "
                        "rate. Negative disables stratification.")
    g.add_argument("--gs_sigma_hi", type=float, default=0.5,
                   help="noise level below which the 3D Gaussian branch trains")
    g.add_argument("--pure_noise_prob", type=float, default=0.1)
    g.add_argument("--pure_clean_prob", type=float, default=0.05,
                   help="fraction of steps at sigma=0 exactly: pure feed-forward "
                        "reconstruction, no diffusion. Carved out of the low band, so it "
                        "must not exceed --sigma_lo_frac.")
    g.add_argument("--i2mv_frac", type=float, default=0.5,
                   help="fraction of steps that pin view 0 clean as an image anchor")
    g.add_argument("--cond_drop_prob", type=float, default=0.1,
                   help="caption dropout, which is what makes guidance possible at sampling")
    g.add_argument("--gamma_ip", type=float, default=0.1,
                   help="extra independent noise mixed into the noising direction only")

    g = p.add_argument_group("losses")
    g.add_argument("--render_w", type=float, default=1.0)
    g.add_argument("--render_lpips_w", type=float, default=0.1)
    g.add_argument("--depth_w", type=float, default=1.0)
    g.add_argument("--depth_tv_w", type=float, default=0.0)
    g.add_argument("--opacity_w", type=float, default=0.0)
    g.add_argument("--anchor_w_in_gs", type=float, default=1.0)
    g.add_argument("--anchor_lpips_in_gs", type=int, default=0)
    g.add_argument("--lpips_weight", type=float, default=0.1)
    g.add_argument("--lpips_gate", type=float, default=0.7)
    g.add_argument("--use_lpips", type=int, default=1)
    g.add_argument("--geo_loss", default="none", choices=list(GEOMETRY_BACKENDS),
                   help="geometry perception loss: score the renders in the feature space "
                        "of a frozen multi-view 3D foundation model. OFF by default -- it "
                        "costs a second ~1B-parameter forward and neither backbone ships "
                        "with PixWorld.")
    g.add_argument("--geo_w", type=float, default=0.05,
                   help="weight of the geometry perception loss")
    g.add_argument("--geo_gate", type=float, default=0.7,
                   help="only apply the geometry loss below this noise level. The paper "
                        "gates the perceptual and geometry terms together at a clean-"
                        "signal fraction of 0.3, i.e. sigma < 0.7. With the default "
                        "--gs_sigma_hi 0.5 there is no render above 0.5 anyway, so this "
                        "only bites if you widen the 3D band past 0.7.")
    g.add_argument("--geo_weights", default="",
                   help="backbone weights: Pi3's model.safetensors, or facebook/VGGT-1B")
    g.add_argument("--geo_repo", default="",
                   help="path to the backbone's source checkout, if not importable")
    g.add_argument("--geo_resolution", type=int, default=224,
                   help="short side the views are resized to for the backbone")
    g.add_argument("--geo_taps", type=int, default=1,
                   help="how many depths of the backbone to compare. 1 is the backbone's "
                        "final block -- its full geometric representation with the "
                        "decoding stage omitted. More than one only makes sense with "
                        "--geo_dist cosine.")
    g.add_argument("--geo_dist", default="mse", choices=list(GEOMETRY_DISTANCES),
                   help="mse on raw features (default, paired with --geo_taps 1) or cosine "
                        "on normalised tokens, which is what the paper specifies. Token "
                        "norms grow with depth, so an mse over several taps is dominated "
                        "by the deepest one -- keep --geo_taps 1 with mse.")
    g.add_argument("--geo_max_views", type=int, default=0,
                   help="cap the views fed to the backbone (0 = all). Views are "
                        "subsampled evenly, keeping the cross-view structure.")
    g.add_argument("--gs_sigma_weight", default="inv_sigma",
                   choices=["inv_sigma", "one_minus_sigma", "none"])
    g.add_argument("--gs_sigma_min", type=float, default=0.05)
    g.add_argument("--sigma_min_convert", type=float, default=0.05)
    g.add_argument("--prune_thr", type=float, default=0.01,
                   help="opacity threshold when exporting a scene (export only, never in "
                        "the training path)")

    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=5e-5)
    g.add_argument("--lr_min", type=float, default=5e-6)
    g.add_argument("--lr_warmup", type=int, default=2000,
                   help="linear ramp from 0")
    g.add_argument("--lr_decay", default="cosine", choices=["cosine", "linear", "none"])
    g.add_argument("--max_steps", type=int, default=50000)
    g.add_argument("--weight_decay", type=float, default=0.01)
    g.add_argument("--gs_wd", type=float, default=1e-6)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--ema_decay", type=float, default=0.999)
    g.add_argument("--n_edge_blocks", type=int, default=5,
                   help="train the first and last N transformer blocks; the middle stays "
                        "frozen and keeps Wan's video prior")
    g.add_argument("--train_patchify", type=int, default=1)
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("memory")
    g.add_argument("--ckpt_blocks", type=int, default=1,
                   help="activation-checkpoint the transformer blocks and the Gaussian "
                        "encoder")
    g.add_argument("--ckpt_gs_dec", type=int, default=1, help="loss-exact")
    g.add_argument("--ckpt_gs_head", type=int, default=1, help="-15.0 GiB, bitwise")
    g.add_argument("--ckpt_lpips", type=int, default=1, help="-25.3 GiB, loss-exact")
    g.add_argument("--gs_head_chunk", type=int, default=0,
                   help="NOT bitwise")
    g.add_argument("--lpips_chunk", type=int, default=0,
                   help="NOT loss-exact on GPU; last resort only")
    g.add_argument("--rope_precast", type=int, default=1,
                   help="required when blocks are both FSDP-wrapped and checkpointed")
    g.add_argument("--gs_depth_max", type=float, default=50.0,
                   help="cap on the predicted depth; 0 disables")

    g = p.add_argument_group("logging, checkpoints, sampling")
    g.add_argument("--log_every", type=int, default=20)
    g.add_argument("--save_every", type=int, default=2000)
    g.add_argument("--keep_ckpts", type=int, default=6)
    g.add_argument("--save_optim", type=int, default=1,
                   help="save optimizer state so a restart is not a cold-Adam restart")
    g.add_argument("--sample_every", type=int, default=2000)
    g.add_argument("--sample_steps", type=int, default=50)
    g.add_argument("--sample_cfg", type=float, default=5.0)
    g.add_argument("--sample_rescale", type=float, default=0.7)
    g.add_argument("--sample_sigma_switch", type=float, default=0.5)
    g.add_argument("--sample_shift", type=float, default=16.0,
                   help="must match --shift for the multi-view branch, so sampling "
                        "follows the schedule the model was trained on")
    g.add_argument("--n_val_scenes", type=int, default=2)
    g.add_argument("--n_val_prompts", type=int, default=2)
    g.add_argument("--export_scene", type=int, default=1,
                   help="also write the Gaussian field of each sampled scene as .npz")
    g.add_argument("--max_nan_skips", type=int, default=20)
    g.add_argument("--wandb", type=int, default=0)
    g.add_argument("--wandb_project", default="pixworld")
    g.add_argument("--wandb_name", default="")
    return p.parse_args()


# --------------------------------------------------------------------------- helpers --
def is_dist():
    return dist.is_available() and dist.is_initialized()


def lr_at(step, args):
    """Linear warm-up, then the chosen decay to ``--lr_min``."""
    if step < args.lr_warmup:
        return args.lr * (step + 1) / max(1, args.lr_warmup)
    if args.lr_decay == "none":
        return args.lr
    t = (step - args.lr_warmup) / max(1, args.max_steps - args.lr_warmup)
    t = min(max(t, 0.0), 1.0)
    if args.lr_decay == "linear":
        return args.lr_min + (args.lr - args.lr_min) * (1 - t)
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * t))


def save_image(t, path):
    """``t`` ``[3, H, W]`` in ``[-1, 1]`` -> PNG."""
    from PIL import Image
    arr = ((t.float().clamp(-1, 1).permute(1, 2, 0).cpu() + 1) * 127.5).round().clamp(0, 255)
    Image.fromarray(arr.byte().numpy()).save(path)


def export_scene_npz(scene, cameras, path, thr=0.01):
    """Write a pruned Gaussian field plus its cameras, for offline viewing."""
    g = prune_opacity(scene.detach().float(), thr).cpu().numpy()
    np.savez_compressed(path, gaussians=g, cameras=cameras.detach().float().cpu().numpy())
    return g.shape[0]


def main():
    args = get_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    is_main = rank == 0

    def p0(*a):
        if is_main:
            print(*a, flush=True)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)

    # Before any large allocation: see cuda_utils for why this matters.
    warm_cusolver(dev, verbose=p0)

    # ---------------------------------------------------------------- the model --
    cfg = dict(WAN22_5B_PIXEL_CONFIG)
    if args.model_config:
        raw = (open(args.model_config).read() if os.path.isfile(args.model_config)
               else args.model_config)
        over = json.loads(raw)
        cfg.update({k: (tuple(v) if k == "patch_size" else v) for k, v in over.items()})
        p0(f"[pixworld] transformer config overridden: {over}")
    model = PixWorldMV(cfg)
    gs_enc = GaussianEncoder()
    gs_dec = GaussianDecoder(ctx_dim=model.inner_dim)
    gs_head = PixelAlignedGaussianHead(
        feat_ch=64, checkpoint=bool(args.ckpt_gs_head), gs_chunk=args.gs_head_chunk,
        depth_max=args.gs_depth_max)
    p0(f"[pixworld] depth cap: "
       + (f"depth <= {args.gs_depth_max:g}" if args.gs_depth_max > 0 else "disabled"))

    start_step = 0
    opt_state = None
    if args.resume:
        start_step, opt_state = _load_checkpoint(
            args.resume, model, gs_enc, gs_dec, gs_head, p0)
    elif args.wan_init:
        _cold_start_from_wan(model, args.wan_path, p0)
        # The lift trunk is initialised from the pixel decoder so that at step 0 it
        # reproduces the decoder's own features exactly.
        gs_enc.init_from_detailer(model.dip_head)
        gs_dec.init_from_detailer(model.dip_head)
        p0("[pixworld] Gaussian trunk initialised from the pixel decoder")
    else:
        p0("[pixworld] --wan_init 0: the transformer starts from scratch")
    train_idx = model.set_trainable(args.n_edge_blocks, bool(args.train_patchify))
    tr, tot = model.num_trainable()
    p0(f"[pixworld] transformer blocks trained: {sorted(train_idx)}")
    p0(f"[pixworld] trainable {tr / 1e6:.1f}M / {tot / 1e6:.1f}M model params "
       f"({100 * tr / tot:.1f}%)")

    model.gradient_checkpointing = bool(args.ckpt_blocks)
    if args.rope_precast:
        # Must happen before fully_shard: it wraps a submodule's forward.
        install_rope_precast(model, dtype=torch.bfloat16, verbose=p0)

    model = model.to(dev)
    for m in (gs_enc, gs_dec, gs_head):
        m.to(dev).float().requires_grad_(True)
    gs_enc.gradient_checkpointing = bool(args.ckpt_blocks)
    gs_dec.gradient_checkpointing = bool(args.ckpt_gs_dec)

    mesh = init_device_mesh("cuda", (world,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in model.transformer.blocks:
        fully_shard(blk, mesh=mesh, mp_policy=mp)
    fully_shard(model.dip_head, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    p0(f"[pixworld] FSDP2: {len(model.transformer.blocks) + 2} units over {world} ranks; "
       f"the Gaussian stack is replicated fp32")

    params = [q for q in model.parameters() if q.requires_grad]
    gs_params = list(gs_head.parameters()) + list(gs_dec.parameters()) + list(gs_enc.parameters())
    opt_model = torch.optim.AdamW(params, lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    opt_gs = torch.optim.AdamW(gs_params, lr=args.lr,
                               weight_decay=args.gs_wd, betas=(0.9, 0.95))
    if opt_state is not None:
        try:
            if opt_state.get("model") is not None:
                set_optimizer_state_dict(
                    model, opt_model, optim_state_dict=opt_state["model"],
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            if opt_state.get("gs") is not None:
                opt_gs.load_state_dict(opt_state["gs"])
            p0("[pixworld] optimizer state restored")
        except Exception as e:                                       # noqa: BLE001
            p0(f"[pixworld] optimizer state NOT restored ({type(e).__name__}: {e}); "
               "continuing with a fresh AdamW")
        del opt_state

    trainable_names = {n for n, q in model.named_parameters() if q.requires_grad}
    ema = {n: q.detach().float().clone()
           for n, q in model.named_parameters() if q.requires_grad}
    for mod, pref in ((gs_head, "gs_head."), (gs_dec, "gs_dec."), (gs_enc, "gs_enc.")):
        for n, q in mod.named_parameters():
            ema[pref + n] = q.detach().float().clone()
    _maybe_restore_ema(args.resume, ema, p0)

    # ------------------------------------------------------------------- data --
    weights = json.loads(args.source_weights) if args.source_weights else None
    ds = MultiViewDataset(
        args.data_root, split="train", v=args.views, k_novel=args.k_novel,
        n_target=args.n_target, both_ends_prob=args.both_ends_prob,
        height=args.height, width=args.width, index=args.index, verbose=is_main)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.workers, drop_last=True,
        pin_memory=True, collate_fn=collate_mv, persistent_workers=args.workers > 0,
        sampler=source_sampler(ds, weights=weights, seed=args.seed * 7919 + rank))
    p0(f"[pixworld] {len(ds):,} train scenes; source mix: "
       f"{weights if weights else 'proportional to pool size'}")

    text_enc = TextEncoder(args.wan_path, dev, dtype=torch.bfloat16)
    lpips_fn = None
    if args.use_lpips:
        lpips_fn = build_lpips(dev, checkpoint=bool(args.ckpt_lpips), chunk=args.lpips_chunk)
        p0(f"[pixworld] LPIPS on: pixel w={args.lpips_weight} gate<{args.lpips_gate}, "
           f"render w={args.render_lpips_w}")

    geo_fn = GeometryPerceptualLoss(
        backend=args.geo_loss, weights=args.geo_weights, repo=args.geo_repo,
        resolution=args.geo_resolution, n_taps=args.geo_taps,
        max_views=args.geo_max_views, distance=args.geo_dist, device=dev, verbose=p0)
    if geo_fn.enabled:
        p0(f"[pixworld] geometry perception weight {args.geo_w}")

    sch = SigmaSchedule(shift=args.shift, lo_frac=args.sigma_lo_frac,
                        band_split=args.gs_sigma_hi,
                        pure_noise_prob=args.pure_noise_prob,
                        pure_clean_prob=args.pure_clean_prob)
    p0("[pixworld] " + sch.describe())

    val = None
    if is_main and args.sample_every > 0 and args.n_val_scenes > 0:
        try:
            val = val_scenes(args.data_root, args.n_val_scenes, v=args.views,
                             k_novel=args.k_novel, n_target=args.n_target,
                             height=args.height, width=args.width, index=args.index)
        except Exception as e:                                       # noqa: BLE001
            p0(f"[pixworld] no validation scenes ({type(e).__name__}: {e}); "
               "sampling text-to-3D only")

    run = None
    if args.wandb and is_main:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_name or None,
                         config=vars(args))

    def encode(texts, drop_prob=0.0):
        if drop_prob > 0:
            texts = ["" if random.random() < drop_prob else t for t in texts]
        return text_enc(texts)

    neg_emb = encode([DEFAULT_NEG])

    # ------------------------------------------------------------------- loop --
    ctx = _Trainer(args, model, gs_enc, gs_dec, gs_head, opt_model, opt_gs, ema, params,
                   gs_params, trainable_names, lpips_fn, geo_fn, sch, encode, neg_emb,
                   dev, is_main, p0, run)
    ctx.fit(dl, start_step, val)

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


# ------------------------------------------------------------------ weight loading --
def _cold_start_from_wan(model, wan_path, p0):
    """Load Wan2.2's pretrained transformer into the pixel model.

    Everything that has a matching shape is copied: all 30 blocks, the condition embedder
    and ``scale_shift_table`` -- 822 of 823 tensors.  The only mismatch is the patch
    embedding, which cannot match by construction (Wan patchifies 48 latent channels at
    stride 2; this model patchifies 13 pixel channels at stride 32).  It keeps its
    initialisation: RGB columns random, the ten conditioning columns exactly zero.

    """
    from diffusers import WanTransformer3DModel
    p0(f"[pixworld] cold start from {wan_path}")
    wan = WanTransformer3DModel.from_pretrained(
        wan_path, subfolder="transformer", torch_dtype=torch.float32)
    info = model.load_wan_backbone(wan.state_dict())
    del wan
    p0(f"[pixworld] Wan backbone: {info['copied']} tensors copied, "
       f"{len(info['shape_mismatch'])} shape-mismatched {info['shape_mismatch']}, "
       f"{info['missing_in_src']} absent from the source")
    if info["copied"] < 800 or info["shape_mismatch"] != ["patch_embedding.weight"]:
        raise RuntimeError(
            f"unexpected Wan load: copied={info['copied']}, "
            f"mismatched={info['shape_mismatch']}, missing={info['missing_in_src']}. "
            f"Expected ~822 copied and exactly ['patch_embedding.weight'] mismatched. "
            f"Is --wan_path really a Wan2.2-TI2V-5B-Diffusers directory?")


def _load_checkpoint(path, model, gs_enc, gs_dec, gs_head, p0):
    """Restore weights (and optionally optimizer state) from a training checkpoint."""
    p0(f"[pixworld] resuming from {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    bad = [n for n, t in sd.items()
           if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all()]
    if bad:
        raise RuntimeError(
            f"{path} holds {len(bad)} non-finite tensors, first {bad[:5]}. Refusing to "
            f"resume: this checkpoint is already broken and training it further will not "
            f"recover it.")
    gs_sd = {"gs_enc.": {}, "gs_dec.": {}, "gs_head.": {}}
    core = {}
    for n, t in sd.items():
        for pref in gs_sd:
            if n.startswith(pref):
                gs_sd[pref][n[len(pref):]] = t
                break
        else:
            core[n] = t
    missing, unexpected = model.load_state_dict(core, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in {path}: {unexpected[:8]}")
    if missing:
        p0(f"[pixworld] {len(missing)} model tensors absent from the checkpoint "
           f"(kept at init): {missing[:5]}")
    for pref, mod in (("gs_enc.", gs_enc), ("gs_dec.", gs_dec), ("gs_head.", gs_head)):
        if gs_sd[pref]:
            mod.load_state_dict(gs_sd[pref], strict=True)
            p0(f"[pixworld] restored {pref[:-1]}")
        else:
            p0(f"[pixworld] {pref[:-1]} absent from the checkpoint, initialising fresh")
    step = int(ck.get("step", 0)) if isinstance(ck, dict) else 0
    if not step:
        m = re.search(r"step(\d+)", os.path.basename(path))
        step = int(m.group(1)) if m else 0
    opt_state = ck.get("optim") if isinstance(ck, dict) else None
    p0(f"[pixworld] resumed at step {step}" + (" (with optimizer state)" if opt_state else ""))
    return step, opt_state


def _maybe_restore_ema(resume, ema, p0):
    """Restore the EMA from the ``model_ema_step*.pt`` beside the checkpoint.

    The saved payload is wrapped in a ``{"model": ..., "step": ...}`` dict, and the live
    EMA entries for model parameters are **DTensors** (they mirror the sharded parameter),
    so a full tensor cannot simply be copied into one -- it has to be redistributed onto
    the same mesh and placement first.    """
    if not resume:
        return
    cand = re.sub(r"model_step", "model_ema_step", resume)
    if cand == resume or not os.path.isfile(cand):
        p0("[pixworld] no EMA file beside the checkpoint; EMA starts from the live weights")
        return
    raw = torch.load(cand, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    try:
        from torch.distributed.tensor import DTensor, distribute_tensor
    except ImportError:                                              # pragma: no cover
        from torch.distributed._tensor import DTensor, distribute_tensor
    n_ok, n_shape = 0, 0
    for n, t in sd.items():
        tgt = ema.get(n)
        if tgt is None:
            continue
        if tuple(tgt.shape) != tuple(t.shape):
            n_shape += 1
            continue
        if isinstance(tgt, DTensor):
            ema[n] = distribute_tensor(t.float().to(tgt.device),
                                       tgt.device_mesh, tgt.placements)
        else:
            tgt.copy_(t.float().to(tgt.device))
        n_ok += 1
    p0(f"[pixworld] EMA restored from {os.path.basename(cand)} "
       f"({n_ok}/{len(ema)} tensors" + (f", {n_shape} shape mismatches" if n_shape else "")
       + ")")
    if n_ok == 0:
        raise RuntimeError(
            f"{cand} matched none of the {len(ema)} EMA entries. Resuming with a "
            f"live-weight EMA would silently discard the averaged model; refusing. "
            f"Checkpoint keys look like: {list(sd)[:3]}")



# ------------------------------------------------------------------------ trainer --
class _Trainer:
    """The step loop."""

    def __init__(self, args, model, gs_enc, gs_dec, gs_head, opt_model, opt_gs, ema,
                 params, gs_params, trainable_names, lpips_fn, geo_fn, sch, encode,
                 neg_emb, dev, is_main, p0, run):
        self.args, self.model = args, model
        self.gs_enc, self.gs_dec, self.gs_head = gs_enc, gs_dec, gs_head
        self.opt_model, self.opt_gs, self.ema = opt_model, opt_gs, ema
        self.params, self.gs_params = params, gs_params
        self.trainable_names, self.lpips_fn, self.sch = trainable_names, lpips_fn, sch
        self.geo_fn = geo_fn
        self.encode, self.neg_emb = encode, neg_emb
        self.dev, self.is_main, self.p0 = dev, is_main, p0
        self.wandb_run = run
        self.nan_skips = 0

    # ----------------------------------------------------------------- saving --
    def save(self, step):
        args, model = self.args, self.model
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        full_sd = get_model_state_dict(model, options=opts)
        optim_sd = None
        if args.save_optim:
            optim_sd = {"model": get_optimizer_state_dict(model, self.opt_model,
                                                          options=opts),
                        "gs": self.opt_gs.state_dict() if self.is_main else None}
        if self.is_main:
            # Frozen tensors go to bf16.
            for n, t in full_sd.items():
                if n not in self.trainable_names and t.is_floating_point():
                    full_sd[n] = t.to(torch.bfloat16)
            for mod, pref in ((self.gs_head, "gs_head."), (self.gs_dec, "gs_dec."),
                              (self.gs_enc, "gs_enc.")):
                for n, q in mod.named_parameters():
                    full_sd[pref + n] = q.detach().cpu()
                for n, b in mod.named_buffers():
                    full_sd[pref + n] = b.detach().cpu()
            payload = {"model": full_sd, "step": step, "args": vars(args)}
            if optim_sd is not None:
                payload["optim"] = optim_sd
            _atomic_save(payload, os.path.join(args.out_dir, f"model_step{step}.pt"))

        ema_sd = {}
        for n, t in self.ema.items():
            ft = t.full_tensor() if hasattr(t, "full_tensor") else t
            if self.is_main:
                ema_sd[n] = (ft.cpu() if n.split(".")[0] in ("gs_head", "gs_dec", "gs_enc")
                             else ft.cpu().to(torch.bfloat16))
            del ft
        if self.is_main:
            _atomic_save({"model": ema_sd, "step": step},
                         os.path.join(args.out_dir, f"model_ema_step{step}.pt"))
            self.p0(f"saved model_step{step}.pt + model_ema_step{step}.pt")
            if args.keep_ckpts > 0:
                _rotate(args.out_dir, args.keep_ckpts, self.p0)
        if is_dist():
            dist.barrier()

    # --------------------------------------------------------------- sampling --
    def _reshard_all(self):
        """Force every FSDP unit back to its sharded state.

        FSDP2 leaves a unit all-gathered after a forward, and in that state a parameter is
        a plain local tensor rather than a DTensor.  Copying a DTensor into it then fails.
        The EMA swap touches parameters directly, outside any forward, so it has to
        normalise the state first -- before the swap, after sampling, and after the
        restore.
        """
        for mm in [self.model, self.model.dip_head] + list(self.model.transformer.blocks):
            if hasattr(mm, "reshard"):
                mm.reshard()

    def write_samples(self, step, val):
        """Sample with the EMA weights, then put the live weights back.

        This runs on **every** rank; only rank 0 writes files.
        """
        args = self.args
        base = (os.path.join(args.out_dir, "samples", f"step{step}") if self.is_main
                else None)
        if base:
            os.makedirs(base, exist_ok=True)
        self._reshard_all()
        backup = {n: q.detach().clone() for n, q in self.model.named_parameters()
                  if q.requires_grad}
        gs_backup = {}
        with torch.no_grad():
            for n, q in self.model.named_parameters():
                if n in self.ema:
                    q.copy_(self.ema[n].to(q.dtype))
            for mod, pref in ((self.gs_head, "gs_head."), (self.gs_dec, "gs_dec."),
                              (self.gs_enc, "gs_enc.")):
                for n, q in mod.named_parameters():
                    gs_backup[pref + n] = q.detach().clone()
                    q.copy_(self.ema[pref + n].to(q.dtype))
        try:
            self.model.eval()
            prompts = DEFAULT_PROMPTS[:args.n_val_prompts]
            for i, pr in enumerate(prompts):
                emb = self.encode([MV_PREFIX + pr])
                x, _ = sample(self.model, self.gs_enc, self.gs_dec, self.gs_head, emb,
                              self.neg_emb, cameras=None, n_views=1, height=args.height,
                              width=args.width, cfg=args.sample_cfg,
                              steps=args.sample_steps, shift=args.sample_shift,
                              cfg_rescale=args.sample_rescale, seed=1234 + i,
                              device=self.dev, sigma_switch=0.0)
                _dump_views(x, os.path.join(base, "t2i"), f"{i:02d}")
            for i, sc in enumerate(val or []):
                emb = self.encode([MV_PREFIX + sc["text"]])
                cams = sc["cameras"].to(self.dev)
                for grp, ref in (("t2mv", None), ("i2mv", sc["image"][:, 0].to(self.dev))):
                    x, scene = sample(
                        self.model, self.gs_enc, self.gs_dec, self.gs_head, emb,
                        self.neg_emb, cameras=cams, ref_img=ref, n_views=cams.shape[0],
                        height=args.height, width=args.width, cfg=args.sample_cfg,
                        steps=args.sample_steps, shift=args.sample_shift,
                        cfg_rescale=args.sample_rescale, seed=4321 + i, device=self.dev,
                        sigma_switch=args.sample_sigma_switch)
                    if base:
                        _dump_views(x, os.path.join(base, grp), f"{i:02d}")
                    if base and scene is not None and args.export_scene:
                        n = export_scene_npz(scene[0], cams,
                                             os.path.join(base, grp, f"{i:02d}_scene.npz"),
                                             args.prune_thr)
                        self.p0(f"  {grp}[{i}] scene exported: {n:,} gaussians")
        finally:
            self._reshard_all()
            with torch.no_grad():
                for n, q in self.model.named_parameters():
                    if n in backup:
                        q.copy_(backup[n])
                for mod, pref in ((self.gs_head, "gs_head."), (self.gs_dec, "gs_dec."),
                                  (self.gs_enc, "gs_enc.")):
                    for n, q in mod.named_parameters():
                        q.copy_(gs_backup[pref + n])
            self._reshard_all()
            self.model.train()
        if base:
            self.p0(f"[pixworld] sampled -> {base}")

    # ------------------------------------------------------------------- step --
    def fit(self, dl, start_step, val):
        args = self.args
        self.model.train()
        it = iter(dl)
        t_last = time.time()
        step = start_step
        while step < args.max_steps:
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl)
                b = next(it)
            log = self.train_step(b, step)
            step += 1
            if log is not None:
                self.update_ema()
            if args.log_every > 0 and step % args.log_every == 0:
                dt = (time.time() - t_last) / args.log_every
                t_last = time.time()
                self.log(step, log, dt)
            if args.save_every > 0 and step % args.save_every == 0:
                self.save(step)
            if args.sample_every > 0 and step % args.sample_every == 0:
                self.write_samples(step, val)      # every rank: FSDP forward is collective
                if is_dist():
                    dist.barrier()
        if args.save_every <= 0 or step % args.save_every != 0:
            self.save(step)
        self.p0(f"[pixworld] done at step {step}")

    def train_step(self, b, step):
        args, dev = self.args, self.dev
        x0 = b["image"].to(dev, non_blocking=True).float()
        B, _, V, H, W = x0.shape
        cams = b["cameras"].to(dev, non_blocking=True).float()

        # Same draw on every rank.
        rng = random.Random(args.seed * 1000003 + step)
        i2mv = rng.random() < args.i2mv_frac

        texts = [MV_PREFIX + t for t in b["text"]]
        ehs = self.encode(texts, drop_prob=args.cond_drop_prob)

        rays = torch.stack([create_raymaps(b["cameras"][i], H, W) for i in range(B)])
        rays = torch.nan_to_num(rays.permute(0, 2, 1, 3, 4).float()).to(dev)
        cond = torch.zeros(B, 3, V, H, W, device=dev)
        cmask = torch.zeros(B, 1, V, H, W, device=dev)
        if i2mv:
            cond[:, :, 0] = x0[:, :, 0]
            cmask[:, :, 0] = 1.0

        sigma = self.sch.sample(B, device=dev).view(B, 1, 1, 1, 1)
        sig = sigma.view(B)
        noise = torch.randn_like(x0)
        noise_in = noise + args.gamma_ip * torch.randn_like(x0)
        x_t = ((1 - sigma) * x0 + sigma * noise_in).float()
        if i2mv:
            x_t[:, :, 0] = x0[:, :, 0].float()
            ts = (sig.view(B, 1) * 1000.0).repeat(1, V)
            ts[:, 0] = 0.0
        else:
            ts = sig * 1000.0
        x_cat = torch.cat([x_t, rays, cond, cmask], dim=1)
        del rays, cond, cmask, noise, noise_in

        gs_on = bool((sig < args.gs_sigma_hi).all())
        self.opt_model.zero_grad(set_to_none=True)
        self.opt_gs.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if gs_on:
                x0_hat, taps = self.model(x_cat, ts, ehs, return_taps=True)
            else:
                x0_hat, taps = self.model(x_cat, ts, ehs), None

        glog = {}
        if gs_on:
            scene, dpt = gs_lift(self.gs_enc, self.gs_dec, self.gs_head, taps, cams,
                                 return_depth=True)
            tgt_img, tgt_cams = gather_targets(
                b["image"], b["novel_image"], b["cameras"], b["novel_cameras"],
                b["target_gather"])
            tgt01 = (tgt_img.to(dev, non_blocking=True).float() + 1) / 2
            tgt_cams = tgt_cams.to(dev)
            renders, depths = render_views(scene, tgt_cams, H, W, bg_mode="random")
            t_mse, t_lp = render_photo_loss(
                renders, tgt01.transpose(1, 2), sig,
                self.lpips_fn if args.render_lpips_w > 0 else None,
                args.gs_sigma_weight, args.gs_sigma_min)
            geo_loss, geo_log = None, {}
            if (self.geo_fn.enabled and args.geo_w > 0
                    and bool((sig < args.geo_gate).all())):
                # Same tensors the photometric term sees.
                geo_loss, geo_log = self.geo_fn(renders, tgt01.transpose(1, 2))
                # Weighted on the same sigma curve as the other geometry terms, so the
                # whole 3D band fades together as the input gets noisier.
                geo_loss = geo_loss * gs_sigma_weight(
                    sig, args.gs_sigma_weight, args.gs_sigma_min).mean()
            del renders, tgt01
            d_loss, d_valid = depth_logit_loss(
                dpt, b["depth"].to(dev, non_blocking=True), cams, sig,
                args.gs_sigma_weight, args.gs_sigma_min)
            a_loss, llog = anchor_loss(
                x0_hat, x_t, x0, sigma, args.sigma_min_convert,
                self.lpips_fn if args.anchor_lpips_in_gs else None,
                args.lpips_weight, args.lpips_gate)
            loss = args.render_w * (t_mse + args.render_lpips_w * t_lp) + args.depth_w * d_loss
            if args.anchor_w_in_gs > 0:
                loss = loss + args.anchor_w_in_gs * a_loss
            if args.depth_tv_w > 0:
                loss = loss + args.depth_tv_w * depth_tv(depths.flatten(0, 1))
            if args.opacity_w > 0:
                loss = loss + args.opacity_w * scene[..., 3].mean()
            if geo_loss is not None:
                loss = loss + args.geo_w * geo_loss
            del depths
            # Keep the FSDP collective pattern identical on every rank -- see the module
            # docstring.  Exactly zero gradient, but autograd still traverses the decoder.
            loss = loss + x0_hat.float().sum() * 0.0
            glog = {"render_mse": float(t_mse), "render_lpips": float(t_lp),
                    "depth": float(d_loss), "depth_valid": float(d_valid),
                    "opacity": float(scene[..., 3].mean()),
                    "gaussians": float(scene.shape[1])}
            glog.update({k: float(v) for k, v in geo_log.items()})
            del scene, dpt
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, llog = anchor_loss(x0_hat, x_t, x0, sigma, args.sigma_min_convert,
                                         self.lpips_fn, args.lpips_weight, args.lpips_gate)

        loss.backward()

        # The Gaussian stack is replicated, not sharded, so its gradients need an explicit
        # all-reduce.  It runs on EVERY rank, hence the zero-fill.
        for q in self.gs_params:
            if q.grad is None:
                q.grad = torch.zeros_like(q)
            if is_dist():
                dist.all_reduce(q.grad, op=dist.ReduceOp.AVG)

        # Clip FIRST, then test.  clip_grad_norm_ computes a GLOBAL norm, so one
        # non-finite gradient makes the norm NaN and multiplies EVERY gradient by NaN.
        gn = torch.nn.utils.clip_grad_norm_(self.params, args.grad_clip)
        gn_gs = torch.nn.utils.clip_grad_norm_(self.gs_params, args.grad_clip)
        gnf = float(gn.full_tensor() if hasattr(gn, "full_tensor") else gn)
        ggf = float(gn_gs.full_tensor() if hasattr(gn_gs, "full_tensor") else gn_gs)
        bad = (not math.isfinite(float(loss))) or (not math.isfinite(gnf)) \
            or (not math.isfinite(ggf))
        badt = torch.tensor([1.0 if bad else 0.0], device=dev)
        if is_dist():
            dist.all_reduce(badt, op=dist.ReduceOp.MAX)     # every rank must agree
        if float(badt) > 0:
            self.nan_skips += 1
            self.p0(f"[pixworld] step {step}: non-finite loss/grad "
                    f"(loss={float(loss):.4g} gn={gnf:.4g} gn_gs={ggf:.4g}) -- step "
                    f"dropped ({self.nan_skips}/{args.max_nan_skips})")
            self.opt_model.zero_grad(set_to_none=True)
            self.opt_gs.zero_grad(set_to_none=True)
            if self.nan_skips >= args.max_nan_skips:
                raise RuntimeError(
                    f"{self.nan_skips} non-finite steps; stopping rather than training "
                    f"into a broken model.")
            return None

        lr_now = lr_at(step, args)
        for o in (self.opt_model, self.opt_gs):
            for g in o.param_groups:
                g["lr"] = lr_now
            o.step()

        out = {"loss": float(loss), "sigma": float(sig.mean()), "gn": gnf, "gn_gs": ggf,
               "lr": lr_now, "gs_on": float(gs_on), "i2mv": float(i2mv)}
        out.update(glog)
        for k in ("anchor_mse", "lpips"):
            if k in llog:
                out[k] = float(llog[k])
        return out

    @torch.no_grad()
    def update_ema(self):
        d = self.args.ema_decay
        for n, q in self.model.named_parameters():
            if n in self.ema:
                self.ema[n].mul_(d).add_(q.detach().float(), alpha=1 - d)
        for mod, pref in ((self.gs_head, "gs_head."), (self.gs_dec, "gs_dec."),
                          (self.gs_enc, "gs_enc.")):
            for n, q in mod.named_parameters():
                self.ema[pref + n].mul_(d).add_(q.detach().float(), alpha=1 - d)

    def log(self, step, log, dt):
        mem = torch.cuda.max_memory_allocated() / 2 ** 30
        memt = torch.tensor([mem], device=self.dev)
        if is_dist():
            dist.all_reduce(memt, op=dist.ReduceOp.MAX)
        torch.cuda.reset_peak_memory_stats()
        if log is None:
            return
        parts = [f"step {step}", f"loss {log['loss']:.4f}", f"sig {log['sigma']:.3f}",
                 f"lr {log['lr']:.2e}", f"gn {log['gn']:.2f}"]
        if log.get("gs_on"):
            parts += [f"render {log['render_mse']:.4f}",
                      f"lpips {log['render_lpips']:.4f}",
                      f"depth {log['depth']:.4f}({log['depth_valid']:.2f})"]
            if "geo" in log:
                parts += [f"geo {log['geo']:.4f}"]
        else:
            parts += [f"anchor {log.get('anchor_mse', float('nan')):.4f}"]
        parts += [f"{dt:.2f}s/it", f"mem {float(memt):.1f}G"]
        self.p0("  ".join(parts))
        if self.wandb_run is not None:
            self.wandb_run.log({**log, "step": step, "sec_per_step": dt,
                          "mem/peak_gib": float(memt)}, step=step)


def _dump_views(x, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    for v in range(x.shape[2]):
        save_image(x[0, :, v], os.path.join(out_dir, f"{prefix}_v{v:02d}.png"))


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rotate(out_dir, keep, p0):
    steps = sorted({int(m.group(1)) for f in os.listdir(out_dir)
                    if (m := re.match(r"model_(?:ema_)?step(\d+)\.pt$", f))})
    for s in steps[:-keep]:
        for f in (f"model_step{s}.pt", f"model_ema_step{s}.pt"):
            q = os.path.join(out_dir, f)
            if os.path.isfile(q):
                os.remove(q)
                p0(f"rotated away {f}")


if __name__ == "__main__":
    main()
