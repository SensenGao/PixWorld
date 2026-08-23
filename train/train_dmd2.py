#!/usr/bin/env python
"""PixWorld stage 2: distil the 50-step model into a 4-step one (DMD2).

Run it with ``train/train_dmd2.sh``.  It starts from a stage-1 checkpoint and needs no
ground-truth images in any loss -- only the dataset's cameras and captions.

Three copies of the same architecture, all initialised from that checkpoint:

* **generator** (student) -- trainable, produces a sample in 4 steps.
* **fake score** (critic) -- trainable, denoises the generator's own samples.
* **real score** (teacher) -- frozen, defines the distribution to match.

The generator is pushed along the difference between the two scores; the critic chases the
generator.  One cycle is ``1 + --dis_per_gen`` steps, and ``--max_steps`` counts both, so
the default 10000 steps is 2000 generator updates.

There is no discriminator, so no real pixel enters any loss here.
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
import torch.nn.functional as F

# Everything this script needs lives beside it -- no imports from anywhere else.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry.cameras import create_raymaps  # noqa: E402
from data.dataset import (MultiViewDataset, collate_mv,  # noqa: E402
                                   source_sampler, val_scenes)
from diffusion.dmd_core import (TASK_WEIGHTS, TASKS, TASKS_INPUT_CAM,  # noqa: E402
                               Generator, ScoreCondEmbeds, add_noise, gen_forward,
                               pin_view0, real_score_x0, rollout, score_cond10, score_x0)
from objectives.dmd_losses import (critic_v_loss, depth_reg_multiscale,  # noqa: E402
                                 dmd_loss)
from diffusion.dmd_sampling import sample_few_step  # noqa: E402
from diffusion.dmd_schedule import GenSchedule  # noqa: E402
from models.dit import (PixWorldMV, WAN22_5B_PIXEL_CONFIG,  # noqa: E402
                                 install_rope_precast)
from models.gaussian_head import PixelAlignedGaussianHead  # noqa: E402
from models.gs_lift import GaussianDecoder, GaussianEncoder  # noqa: E402
from geometry.render import prune_opacity  # noqa: E402
from utils.cuda_utils import warm_cusolver  # noqa: E402
from data.text_encoder import MV_PREFIX, TextEncoder  # noqa: E402

try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
except ImportError:                                                      # pragma: no cover
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.checkpoint.state_dict import (  # noqa: E402
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict)
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402


def get_args():
    p = argparse.ArgumentParser(
        description="Distil a PixWorld stage-1 model into a few-step generator (DMD2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("paths")
    g.add_argument("--data_root", required=True)
    g.add_argument("--index", default="index.jsonl")
    g.add_argument("--wan_path", required=True, help="for the UMT5 text encoder")
    g.add_argument("--teacher", required=True,
                   help="stage-1 checkpoint; seeds the student, the critic AND the frozen "
                        "teacher. It must contain a trained Gaussian stack -- distillation "
                        "cannot learn the lift from scratch.")
    g.add_argument("--out_dir", required=True)
    g.add_argument("--resume", default="", help="a DMD checkpoint to continue from")
    g.add_argument("--model_config", default="",
                   help="JSON overriding the transformer config (must match the teacher)")

    g = p.add_argument_group("the few-step schedule")
    g.add_argument("--n_gen_steps", type=int, default=4)
    g.add_argument("--gen_shift", type=float, default=3.0,
                   help="places the step sigmas; 3.0 gives 1.0 / 0.9 / 0.75 / 0.5")
    g.add_argument("--gen_sigmas", default="",
                   help="explicit descending sigma list, overriding --gen_shift")
    g.add_argument("--gs_step", type=int, default=-1,
                   help="first step that lifts to 3D; -1 means only the last")
    g.add_argument("--step_last_w", type=float, default=0.4,
                   help="probability of choosing the final (3D) step for a DMD update; "
                        "the rest share the remainder. Negative means uniform.")
    g.add_argument("--dmd_sigma_hi", type=float, default=0.98)
    g.add_argument("--dmd_sigma_lo", type=float, default=0.02)
    g.add_argument("--dmd_narrow_prob", type=float, default=0.1)

    g = p.add_argument_group("views and data")
    g.add_argument("--views", type=int, default=16)
    g.add_argument("--k_novel", type=int, default=16)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--width", type=int, default=832)
    g.add_argument("--batch_size", type=int, default=1)
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--source_weights", default="")
    g.add_argument("--i2mv_frac", type=float, default=0.5)
    g.add_argument("--cond_drop_prob", type=float, default=0.1)

    g = p.add_argument_group("losses")
    g.add_argument("--task_w", default=",".join(str(w) for w in TASK_WEIGHTS),
                   help="relative weight of " + " / ".join(TASKS))
    g.add_argument("--real_cfg", type=float, default=3.0,
                   help="guidance on the FROZEN teacher only. The student inherits it, "
                        "which is why inference then runs at cfg 1.")
    g.add_argument("--neg_prompt", default="",
                   help="negative prompt for teacher guidance; empty uses the multi-view "
                        "prefix alone")
    g.add_argument("--consistency_w", type=float, default=0.1,
                   help="MSE between the 3D render and the transformer's own RGB")
    g.add_argument("--depth_reg_w", type=float, default=0.01,
                   help="multi-scale smoothness prior on rendered depth")
    g.add_argument("--opacity_w", type=float, default=0.01)
    g.add_argument("--dmd_w_eps", type=float, default=1e-6)
    g.add_argument("--dmd_w_max", type=float, default=0.0)
    g.add_argument("--sigma_min_convert", type=float, default=0.01)
    g.add_argument("--n_tok_embed", type=int, default=32)

    g = p.add_argument_group("optimisation")
    g.add_argument("--dis_per_gen", type=int, default=4,
                   help="critic updates per generator update")
    g.add_argument("--lr_gen", type=float, default=1e-6)
    g.add_argument("--lr_dis", type=float, default=5e-7)
    g.add_argument("--max_steps", type=int, default=10000,
                   help="counts BOTH phases, so at --dis_per_gen 4 one fifth are "
                        "generator updates")
    g.add_argument("--warmup_steps", type=int, default=-1, help="-1 means max_steps/10")
    g.add_argument("--decay_steps", type=int, default=-1, help="-1 means max_steps/2")
    g.add_argument("--weight_decay", type=float, default=1e-6)
    g.add_argument("--gs_wd", type=float, default=1e-6)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.95)
    g.add_argument("--n_edge_blocks", type=int, default=5)
    g.add_argument("--train_patchify", type=int, default=1)
    g.add_argument("--dis_train_all", type=int, default=1,
                   help="train every critic parameter rather than the generator's subset")
    g.add_argument("--ema_decay", type=float, default=0.0,
                   help="0 disables the EMA")
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("memory")
    g.add_argument("--ckpt_blocks", type=int, default=1)
    g.add_argument("--ckpt_gs_dec", type=int, default=1)
    g.add_argument("--ckpt_gs_head", type=int, default=1)
    g.add_argument("--gs_head_chunk", type=int, default=0)
    g.add_argument("--rope_precast", type=int, default=1)
    g.add_argument("--gs_depth_max", type=float, default=50.0)
    g.add_argument("--score_bf16", type=int, default=1,
                   help="hold the frozen teacher in bf16: a no-op under FSDP mixed "
                        "precision, but it halves the shard")

    g = p.add_argument_group("logging, checkpoints, sampling")
    g.add_argument("--log_every", type=int, default=20)
    g.add_argument("--log_gen_every", type=int, default=2,
                   help="also log on every Nth GENERATOR update")
    g.add_argument("--save_every", type=int, default=1000)
    g.add_argument("--keep_ckpts", type=int, default=4)
    g.add_argument("--save_optim", type=int, default=1)
    g.add_argument("--sample_every", type=int, default=1000)
    g.add_argument("--n_val_scenes", type=int, default=2)
    g.add_argument("--n_val_prompts", type=int, default=2)
    g.add_argument("--export_scene", type=int, default=1)
    g.add_argument("--prune_thr", type=float, default=0.01)
    g.add_argument("--max_nan_skips", type=int, default=20)
    g.add_argument("--nan_probe", type=int, default=1)
    g.add_argument("--wandb", type=int, default=0)
    g.add_argument("--wandb_project", default="pixworld-dmd2")
    g.add_argument("--wandb_name", default="")
    return p.parse_args()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def lr_scale(step, warm, decay, mx):
    """Linear warm-up, flat plateau, linear decay to zero.

    """
    if step < warm:
        return (step + 1) / max(warm, 1)
    if step < decay:
        return 1.0
    return max(0.0, (mx - step) / max(mx - decay, 1))


def save_image(t, path):
    from PIL import Image
    arr = ((t.float().clamp(-1, 1).permute(1, 2, 0).cpu() + 1) * 127.5).round().clamp(0, 255)
    Image.fromarray(arr.byte().numpy()).save(path)


def _dump_views(x, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    for v in range(x.shape[2]):
        save_image(x[0, :, v], os.path.join(out_dir, f"{prefix}_v{v:02d}.png"))


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _build_stack(cfg, args, dev, feat_ch=64):
    """One transformer plus its Gaussian stack, on the device, untrained."""
    model = PixWorldMV(cfg)
    gs_enc = GaussianEncoder()
    gs_dec = GaussianDecoder(ctx_dim=model.inner_dim)
    gs_head = PixelAlignedGaussianHead(
        feat_ch=feat_ch, checkpoint=bool(args.ckpt_gs_head),
        gs_chunk=args.gs_head_chunk, depth_max=args.gs_depth_max)
    return model, gs_enc, gs_dec, gs_head


def _split_teacher(path, p0):
    """Load a stage-1 checkpoint and split it into transformer / Gaussian parts."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    bad = [n for n, t in sd.items()
           if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all()]
    if bad:
        raise RuntimeError(f"{path} holds {len(bad)} non-finite tensors, first {bad[:5]}")
    parts = {"gs_enc.": {}, "gs_dec.": {}, "gs_head.": {}}
    core = {}
    for n, t in sd.items():
        for pref in parts:
            if n.startswith(pref):
                parts[pref][n[len(pref):]] = t
                break
        else:
            core[n] = t
    if not parts["gs_dec."] or not parts["gs_head."]:
        raise RuntimeError(
            f"{path} has no trained Gaussian stack (gs_dec/gs_head keys are absent). "
            f"Distillation starts from a trained lift; it cannot learn one from the "
            f"score difference alone.")
    p0(f"[dmd2] teacher: {len(core)} transformer tensors, "
       f"{sum(len(v) for v in parts.values())} Gaussian tensors")
    return core, parts


def _load_into(model, gs_enc, gs_dec, gs_head, core, parts, p0, tag):
    missing, unexpected = model.load_state_dict(core, strict=False)
    if unexpected:
        raise RuntimeError(f"{tag}: unexpected checkpoint keys {unexpected[:8]}")
    if missing:
        p0(f"[dmd2] {tag}: {len(missing)} tensors kept at init: {missing[:4]}")
    for pref, mod in (("gs_enc.", gs_enc), ("gs_dec.", gs_dec), ("gs_head.", gs_head)):
        if parts[pref]:
            mod.load_state_dict(parts[pref], strict=True)
        elif pref == "gs_enc.":
            gs_enc.init_from_detailer(model.dip_head)
            p0("[dmd2] gs_enc absent from the teacher; initialised from the pixel decoder")


def _shard(model, mesh, mp):
    for blk in model.transformer.blocks:
        fully_shard(blk, mesh=mesh, mp_policy=mp)
    fully_shard(model.dip_head, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)


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
    warm_cusolver(dev, verbose=p0)

    cfg = dict(WAN22_5B_PIXEL_CONFIG)
    if args.model_config:
        raw = (open(args.model_config).read() if os.path.isfile(args.model_config)
               else args.model_config)
        over = json.loads(raw)
        cfg.update({k: (tuple(v) if k == "patch_size" else v) for k, v in over.items()})
        p0(f"[dmd2] transformer config overridden: {over}")

    sched = GenSchedule(n_steps=args.n_gen_steps, shift=args.gen_shift,
                        sigmas=[float(s) for s in args.gen_sigmas.split(",")]
                        if args.gen_sigmas else None,
                        sigma_hi=args.dmd_sigma_hi, sigma_lo=args.dmd_sigma_lo,
                        narrow_prob=args.dmd_narrow_prob, gs_step=args.gs_step)
    step_w = sched.step_weights(args.step_last_w)
    task_w = [float(x) for x in args.task_w.split(",")]
    if len(task_w) != len(TASKS):
        raise ValueError(f"--task_w needs {len(TASKS)} values for {TASKS}")
    if args.sigma_min_convert > args.dmd_sigma_lo:
        raise ValueError(
            f"--sigma_min_convert {args.sigma_min_convert} > --dmd_sigma_lo "
            f"{args.dmd_sigma_lo}: the critic's velocity floor would clip inside the band "
            f"it actually trains on.")
    cycle = 1 + args.dis_per_gen
    warm = args.warmup_steps if args.warmup_steps >= 0 else args.max_steps // 10
    decay = args.decay_steps if args.decay_steps >= 0 else args.max_steps // 2
    p0(f"[dmd2] {sched.describe()}")
    p0(f"[dmd2] step weights {[round(w, 3) for w in step_w]} -> "
       f"{100 * sum(w for k, w in enumerate(step_w) if sched.renders_at(k)):.0f}% of "
       f"generator updates carry a 3D gradient")
    p0(f"[dmd2] cycle {cycle} (1 generator + {args.dis_per_gen} critic); "
       f"{args.max_steps} total = {args.max_steps // cycle} generator updates")

    # ------------------------------------------------------------- networks --
    core, parts = _split_teacher(args.teacher, p0)
    gen_model, gs_enc, gs_dec, gs_head = _build_stack(cfg, args, dev)
    _load_into(gen_model, gs_enc, gs_dec, gs_head, core, parts, p0, "generator")
    fake_model, _fe, _fd, _fh = _build_stack(cfg, args, dev)
    _load_into(fake_model, _fe, _fd, _fh, core, parts, p0, "critic")
    real_model, _re, _rd, _rh = _build_stack(cfg, args, dev)
    _load_into(real_model, _re, _rd, _rh, core, parts, p0, "teacher")
    del _fe, _fd, _fh, _re, _rd, _rh, core, parts

    gen_model.set_trainable(args.n_edge_blocks, bool(args.train_patchify))
    gen_model.gradient_checkpointing = bool(args.ckpt_blocks)
    if args.dis_train_all:
        fake_model.requires_grad_(True)
    else:
        fake_model.set_trainable(args.n_edge_blocks, bool(args.train_patchify))
    fake_model.gradient_checkpointing = bool(args.ckpt_blocks)
    real_model.requires_grad_(False).eval()
    real_model.gradient_checkpointing = False

    if args.rope_precast:
        for m in (gen_model, fake_model, real_model):
            install_rope_precast(m, dtype=torch.bfloat16, verbose=None)
        p0("[dmd2] rope pre-cast to bfloat16 on all three networks")

    gen_model = gen_model.to(dev)
    fake_model = fake_model.to(dev)
    real_model = real_model.to(dev)
    if args.score_bf16:
        real_model = real_model.to(torch.bfloat16)
    for m in (gs_enc, gs_dec, gs_head):
        m.to(dev).float().requires_grad_(True)
    gs_enc.gradient_checkpointing = bool(args.ckpt_blocks)
    gs_dec.gradient_checkpointing = bool(args.ckpt_gs_dec)

    embeds = ScoreCondEmbeds(text_dim=cfg["text_dim"], n_steps=sched.n,
                             n_tok=args.n_tok_embed).to(dev).float()

    mesh = init_device_mesh("cuda", (world,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for m in (gen_model, fake_model, real_model):
        _shard(m, mesh, mp)
    gen = Generator(gen_model, gs_enc, gs_dec, gs_head)
    p0(f"[dmd2] three FSDP2-sharded transformers over {world} ranks; the Gaussian stack "
       f"and the score-conditioning tokens are replicated fp32")

    g_sharded = [q for q in gen_model.parameters() if q.requires_grad]
    gs_params = list(gs_head.parameters()) + list(gs_dec.parameters()) + list(gs_enc.parameters())
    d_sharded = [q for q in fake_model.parameters() if q.requires_grad]
    d_repl = list(embeds.parameters())
    mk = lambda ps, wd: torch.optim.AdamW(ps, lr=args.lr_gen, weight_decay=wd,
                                          betas=(args.beta1, args.beta2))
    opt_g_dit, opt_g_gs = mk(g_sharded, args.weight_decay), mk(gs_params, args.gs_wd)
    opt_d_dit, opt_d_emb = mk(d_sharded, args.weight_decay), mk(d_repl, args.weight_decay)
    OPTS_G, OPTS_D = (opt_g_dit, opt_g_gs), (opt_d_dit, opt_d_emb)
    ALL_OPTS = OPTS_G + OPTS_D
    p0(f"[dmd2] trainable: generator {len(g_sharded)} sharded + {len(gs_params)} "
       f"replicated | critic {len(d_sharded)} sharded + {len(d_repl)} replicated")

    start_step = 0
    if args.resume:
        start_step = _resume(args, gen_model, gs_enc, gs_dec, gs_head, fake_model,
                             embeds, opt_g_dit, opt_g_gs, opt_d_dit, opt_d_emb, p0)

    ema = {}
    if args.ema_decay > 0:
        ema = {n: q.detach().float().clone()
               for n, q in gen_model.named_parameters() if q.requires_grad}
        for mod, pref in ((gs_head, "gs_head."), (gs_dec, "gs_dec."), (gs_enc, "gs_enc.")):
            for n, q in mod.named_parameters():
                ema[pref + n] = q.detach().float().clone()
        p0(f"[dmd2] EMA on, decay {args.ema_decay}")

    # ----------------------------------------------------------------- data --
    weights = json.loads(args.source_weights) if args.source_weights else None
    ds = MultiViewDataset(args.data_root, split="train", v=args.views,
                          k_novel=args.k_novel, n_target=args.views,
                          both_ends_prob=0.0, height=args.height, width=args.width,
                          index=args.index, verbose=is_main)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.workers, drop_last=True,
        pin_memory=True, collate_fn=collate_mv, persistent_workers=args.workers > 0,
        sampler=source_sampler(ds, weights=weights, seed=args.seed * 7919 + rank))

    text_enc = TextEncoder(args.wan_path, dev, dtype=torch.bfloat16)
    neg_base = text_enc([args.neg_prompt or MV_PREFIX])
    p0(f"[dmd2] teacher guidance cfg={args.real_cfg} against "
       f"{(args.neg_prompt or MV_PREFIX)!r}; the student then samples at cfg=1")

    val = None
    if is_main and args.sample_every > 0 and args.n_val_scenes > 0:
        try:
            val = val_scenes(args.data_root, args.n_val_scenes, v=args.views,
                             k_novel=args.k_novel, n_target=args.views,
                             height=args.height, width=args.width, index=args.index)
        except Exception as e:                                       # noqa: BLE001
            p0(f"[dmd2] no validation scenes ({type(e).__name__}: {e})")

    run = None
    if args.wandb and is_main:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_name or None,
                         config=vars(args))

    # ----------------------------------------------------------------- loop --
    def clip_(ps, mx):
        return torch.nn.utils.clip_grad_norm_(ps, mx if mx > 0 else float("inf"))

    it = iter(dl)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(dl)
            return next(it)

    step = start_step
    gen_updates = 0
    nan_skips = 0
    t_last = time.time()
    gen.train()
    fake_model.train()

    while step < args.max_steps:
        is_gen = (step % cycle) == 0
        # Rank-uniform control-flow draws from a step-seeded RNG.
        srng = random.Random(args.seed * 1000003 + step)
        k_step = srng.choices(range(sched.n), step_w, k=1)[0]
        gs_on = sched.renders_at(k_step)
        # Below the 3D step there is no 3D quantity to distil, so the only well-defined
        # task is the 2D one.
        task = TASKS[srng.choices(range(len(TASKS)), task_w, k=1)[0]] if gs_on else "mv_2d"
        i2mv = srng.random() < args.i2mv_frac
        input_cam = task in TASKS_INPUT_CAM
        # The DMD noise level is drawn rank-LOCALLY.
        sigma_d = sched.dmd_sigma(k_step,
                                  random.Random(args.seed * 104729 + step * 977 + rank))

        b = next_batch()
        x_gt = b["image"].to(dev, non_blocking=True)
        cams_d = b["cameras"].to(dev)
        B, _, V, H, W = x_gt.shape
        texts = [MV_PREFIX + t for t in b["text"]]
        if args.cond_drop_prob > 0:
            texts = ["" if random.random() < args.cond_drop_prob else t for t in texts]
        with torch.no_grad():
            ehs = text_enc(texts)
        neg_ehs = neg_base.expand(B, -1, -1)

        rays = torch.stack([create_raymaps(cams_d[i], H, W) for i in range(B)])
        rays = torch.nan_to_num(rays.permute(0, 2, 1, 3, 4).float())
        cond_img = torch.zeros(B, 3, V, H, W, device=dev)
        cmask = torch.zeros(B, 1, V, H, W, device=dev)
        ref = None
        if i2mv:
            cond_img[:, :, 0] = x_gt[:, :, 0]
            cmask[:, :, 0] = 1.0
            ref = x_gt[:, :, 0]
        cond10 = torch.cat([rays, cond_img, cmask], dim=1)
        del rays

        if input_cam:
            render_cams = cams_d
            # Reuse the generator's own ray map bitwise: these cameras were already
            # rooted at identity by the dataset's joint normalisation.
            sc10 = cond10
        else:
            render_cams = b["novel_cameras"].to(dev)
            sc10 = score_cond10(render_cams, H, W, renormalize=True)
        R = render_cams.shape[1]
        pin_ref = ref if (i2mv and input_cam) else None
        tgt_mask = cond_rgb = None
        if pin_ref is not None:
            tgt_mask = torch.zeros(B, 1, R, 1, 1, device=dev)
            tgt_mask[:, :, 0] = 1.0

        sig_k = torch.full((B,), sched.sigmas[k_step], device=dev, dtype=torch.float32)
        sig_d = torch.full((B,), sigma_d, device=dev, dtype=torch.float32)
        glog = {}
        for o in ALL_OPTS:
            o.zero_grad(set_to_none=True)

        if is_gen:
            x_t = rollout(gen, sched, k_step, cond10, ehs, cams_d, H, W,
                          i2mv=i2mv, ref=ref, device=dev)
            # At the 3D step the render always happens, even for task mv_2d, because the
            # consistency / depth / opacity terms read it.
            out = gen_forward(gen, x_t, cond10, sig_k, ehs, cams_d, render_cams, H, W,
                              need_3d=gs_on, bg_mode="random", i2mv=i2mv)
            del x_t
            x0_2d = out["x0_2d"]
            rgb_3d = out["rgb_3d"] if gs_on else None
            x_fake = x0_2d if task == "mv_2d" else rgb_3d
            if pin_ref is not None:
                cond_rgb = torch.zeros_like(x_fake)
                cond_rgb[:, :, 0] = pin_ref

            noise = torch.randn_like(x_fake)
            x_noisy = pin_view0(add_noise(x_fake.detach(), sig_d, noise), pin_ref)
            with torch.no_grad():
                x0_real = real_score_x0(real_model, x_noisy, sc10, sig_d, ehs, neg_ehs,
                                        cfg=args.real_cfg,
                                        view0_clean=pin_ref is not None)
                x0_fake = score_x0(fake_model, x_noisy, sc10, sig_d,
                                   embeds(ehs, task, k_step),
                                   view0_clean=pin_ref is not None)
            loss, dlg = dmd_loss(x_fake, x0_real, x0_fake, mask=tgt_mask,
                                 cond_rgb=cond_rgb, eps=args.dmd_w_eps,
                                 w_max=args.dmd_w_max)
            del x0_real, x0_fake, x_noisy, noise
            glog["dmd"] = float(loss)
            glog.update({k: float(v) for k, v in dlg.items()})

            if gs_on and input_cam and args.consistency_w > 0:
                l_cons = F.mse_loss(rgb_3d, x0_2d)
                loss = loss + args.consistency_w * l_cons
                glog["consistency"] = float(l_cons)
            if gs_on and args.depth_reg_w > 0:
                l_dep = depth_reg_multiscale(out["depth01"])
                loss = loss + args.depth_reg_w * l_dep
                glog["depth_reg"] = float(l_dep)
            if gs_on and args.opacity_w > 0:
                l_op = out["scene"][..., 3].mean()
                loss = loss + args.opacity_w * l_op
                glog["opacity"] = float(l_op)
            # Keep the FSDP collective pattern rank-uniform: on mv_3d_novel nothing else
            # reaches the pixel decoder, so its unit would otherwise skip its reduction.
            loss = loss + x0_2d.float().sum() * 0.0
            del out, x0_2d, rgb_3d

            loss.backward()
            for q in gs_params:
                if q.grad is None:
                    q.grad = torch.zeros_like(q)
                if is_dist():
                    dist.all_reduce(q.grad, op=dist.ReduceOp.AVG)
            gn = float(_full(clip_(g_sharded, args.grad_clip)))
            gn2 = float(_full(clip_(gs_params, args.grad_clip)))
            opts, phase = OPTS_G, "gen"
            base_lr = args.lr_gen
        else:
            with torch.no_grad():
                x_t = rollout(gen, sched, k_step, cond10, ehs, cams_d, H, W,
                              i2mv=i2mv, ref=ref, device=dev)
                need_3d = gs_on and task != "mv_2d"
                out = gen_forward(gen, x_t, cond10, sig_k, ehs, cams_d, render_cams,
                                  H, W, need_3d=need_3d, bg_mode="random", i2mv=i2mv,
                                  use_checkpoint=False)
                x_fake = (out["rgb_3d"] if need_3d else out["x0_2d"]).detach()
                del x_t, out
                noise = torch.randn_like(x_fake)
                x_noisy = pin_view0(add_noise(x_fake, sig_d, noise), pin_ref)
            view_mask = None
            if pin_ref is not None:
                view_mask = torch.ones(B, 1, R, 1, 1, device=dev)
                view_mask[:, :, 0] = 0.0
            x0_fake_pred = score_x0(fake_model, x_noisy, sc10, sig_d,
                                    embeds(ehs, task, k_step),
                                    view0_clean=pin_ref is not None)
            loss = critic_v_loss(x0_fake_pred, x_noisy, x_fake, noise, sig_d,
                                 sigma_min=args.sigma_min_convert, view_mask=view_mask)
            glog["critic"] = float(loss)
            del x0_fake_pred, x_noisy, noise, x_fake

            loss.backward()
            for q in d_repl:
                if q.grad is None:
                    q.grad = torch.zeros_like(q)
                if is_dist():
                    dist.all_reduce(q.grad, op=dist.ReduceOp.AVG)
            gn = float(_full(clip_(d_sharded, args.grad_clip)))
            gn2 = float(_full(clip_(d_repl, args.grad_clip)))
            opts, phase = OPTS_D, "dis"
            base_lr = args.lr_dis

        bad = (not math.isfinite(float(loss))) or (not math.isfinite(gn)) \
            or (not math.isfinite(gn2))
        badt = torch.tensor([1.0 if bad else 0.0], device=dev)
        if is_dist():
            dist.all_reduce(badt, op=dist.ReduceOp.MAX)
        if float(badt) > 0:
            nan_skips += 1
            p0(f"[dmd2] step {step} ({phase}): non-finite loss/grad -- dropped "
               f"({nan_skips}/{args.max_nan_skips})")
            for o in ALL_OPTS:
                o.zero_grad(set_to_none=True)
            step += 1
            if nan_skips >= args.max_nan_skips:
                raise RuntimeError(f"{nan_skips} non-finite steps; stopping.")
            continue

        f = lr_scale(step, warm, decay, args.max_steps)
        lr_now = base_lr * f
        for o in opts:
            for grp in o.param_groups:
                grp["lr"] = lr_now
            o.step()
        step += 1
        if is_gen:
            gen_updates += 1
            if ema:
                with torch.no_grad():
                    d = args.ema_decay
                    for n, q in gen_model.named_parameters():
                        if n in ema:
                            ema[n].mul_(d).add_(q.detach().float(), alpha=1 - d)
                    for mod, pref in ((gs_head, "gs_head."), (gs_dec, "gs_dec."),
                                      (gs_enc, "gs_enc.")):
                        for n, q in mod.named_parameters():
                            ema[pref + n].mul_(d).add_(q.detach().float(), alpha=1 - d)

        do_log = ((args.log_every > 0 and step % args.log_every == 0)
                  or (is_gen and args.log_gen_every > 0
                      and gen_updates % args.log_gen_every == 0))
        if do_log:
            dt = time.time() - t_last
            t_last = time.time()
            mem = torch.cuda.max_memory_allocated() / 2 ** 30
            memt = torch.tensor([mem], device=dev)
            if is_dist():
                dist.all_reduce(memt, op=dist.ReduceOp.MAX)
            torch.cuda.reset_peak_memory_stats()
            parts = [f"step {step}", f"[{phase}]", f"k={k_step}", f"task={task}",
                     f"sig_d {sigma_d:.3f}", f"lr {lr_now:.2e}",
                     # BOTH norms: the sharded transformer and the replicated stack are
                     # clipped independently, and on a 3D step they can differ by orders
                     # of magnitude. Printing only the first hides a dead branch.
                     f"gn {gn:.3f}/{gn2:.3f}"]
            parts += [f"{k} {v:.4f}" for k, v in glog.items()]
            parts += [f"{dt:.2f}s", f"mem {float(memt):.1f}G"]
            p0("  ".join(parts))
            if run is not None:
                run.log({f"{phase}/{k}": v for k, v in glog.items()}
                        | {"step": step, "gen_updates": gen_updates, "lr": lr_now,
                           "gn": gn, "gn_replicated": gn2,
                           "mem/peak_gib": float(memt)}, step=step)

        if args.save_every > 0 and step % args.save_every == 0:
            _save(args, step, gen_updates, gen_model, gs_enc, gs_dec, gs_head, fake_model,
                  embeds, ema, opt_g_dit, opt_g_gs, opt_d_dit, opt_d_emb, is_main, p0)
        if args.sample_every > 0 and step % args.sample_every == 0:
            _sample(args, step, gen, gen_model, sched, text_enc, val, dev, is_main, p0)
            if is_dist():
                dist.barrier()

    if args.save_every <= 0 or step % args.save_every != 0:
        _save(args, step, gen_updates, gen_model, gs_enc, gs_dec, gs_head, fake_model,
              embeds, ema, opt_g_dit, opt_g_gs, opt_d_dit, opt_d_emb, is_main, p0)
    p0(f"[dmd2] done at step {step} ({gen_updates} generator updates)")
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def _full(t):
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def _save(args, step, gen_updates, model, gs_enc, gs_dec, gs_head, fake_model, embeds,
          ema, opt_g_dit, opt_g_gs, opt_d_dit, opt_d_emb, is_main, p0):
    """Write the student, the critic and the training state.

    Only ``model_step*.pt`` is needed for inference; the other two exist so a run can be
    resumed without restarting distillation.
    """
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    trainable = {n for n, q in model.named_parameters() if q.requires_grad}
    sd = get_model_state_dict(model, options=opts)
    crit = get_model_state_dict(fake_model, options=opts)
    og = get_optimizer_state_dict(model, opt_g_dit, options=opts) if args.save_optim else None
    od = get_optimizer_state_dict(fake_model, opt_d_dit, options=opts) if args.save_optim else None
    if is_main:
        for n, t in sd.items():
            if n not in trainable and t.is_floating_point():
                sd[n] = t.to(torch.bfloat16)
        for mod, pref in ((gs_head, "gs_head."), (gs_dec, "gs_dec."), (gs_enc, "gs_enc.")):
            for n, q in mod.named_parameters():
                sd[pref + n] = q.detach().cpu()
            for n, bb in mod.named_buffers():
                sd[pref + n] = bb.detach().cpu()
        _atomic_save({"model": sd, "step": step, "args": vars(args)},
                     os.path.join(args.out_dir, f"model_step{step}.pt"))
        _atomic_save({"fake_score": crit, "embeds": embeds.state_dict(), "step": step},
                     os.path.join(args.out_dir, f"critic_step{step}.pt"))
        state = {"step": step, "gen_updates": gen_updates}
        if args.save_optim:
            state.update(opt_g_dit=og, opt_d_dit=od,
                         opt_g_gs=opt_g_gs.state_dict(), opt_d_emb=opt_d_emb.state_dict())
        _atomic_save(state, os.path.join(args.out_dir, f"dmd_state_step{step}.pt"))
    if ema:
        ema_sd = {}
        for n, t in ema.items():
            ft = _full(t)
            if is_main:
                ema_sd[n] = (ft.cpu() if n.split(".")[0] in ("gs_head", "gs_dec", "gs_enc")
                             else ft.cpu().to(torch.bfloat16))
            del ft
        if is_main:
            _atomic_save({"model": ema_sd, "step": step},
                         os.path.join(args.out_dir, f"model_ema_step{step}.pt"))
    if is_main:
        p0(f"[dmd2] saved step {step}")
        if args.keep_ckpts > 0:
            _rotate(args.out_dir, args.keep_ckpts, p0)
    if is_dist():
        dist.barrier()


def _rotate(out_dir, keep, p0):
    pat = re.compile(r"(?:model|model_ema|critic|dmd_state)_step(\d+)\.pt$")
    steps = sorted({int(m.group(1)) for f in os.listdir(out_dir) if (m := pat.match(f))})
    for s in steps[:-keep]:
        for f in (f"model_step{s}.pt", f"model_ema_step{s}.pt",
                  f"critic_step{s}.pt", f"dmd_state_step{s}.pt"):
            q = os.path.join(out_dir, f)
            if os.path.isfile(q):
                os.remove(q)


def _resume(args, model, gs_enc, gs_dec, gs_head, fake_model, embeds,
            opt_g_dit, opt_g_gs, opt_d_dit, opt_d_emb, p0):
    """Restore a DMD run: student, critic and optimizer state."""
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    core = {n: t for n, t in sd.items()
            if not n.startswith(("gs_enc.", "gs_dec.", "gs_head."))}
    set_model_state_dict(model, model_state_dict=core, options=opts)
    for pref, mod in (("gs_enc.", gs_enc), ("gs_dec.", gs_dec), ("gs_head.", gs_head)):
        part = {n[len(pref):]: t for n, t in sd.items() if n.startswith(pref)}
        if part:
            mod.load_state_dict(part, strict=True)
    step = int(ck.get("step", 0))
    cpath = args.resume.replace("model_step", "critic_step")
    if os.path.isfile(cpath):
        cck = torch.load(cpath, map_location="cpu", weights_only=False)
        # The critic is restored through set_model_state_dict.
        set_model_state_dict(fake_model, model_state_dict=cck["fake_score"], options=opts)
        embeds.load_state_dict(cck["embeds"])
        p0("[dmd2] critic restored")
    spath = args.resume.replace("model_step", "dmd_state_step")
    if os.path.isfile(spath):
        st = torch.load(spath, map_location="cpu", weights_only=False)
        step = int(st.get("step", step))
        try:
            if st.get("opt_g_dit") is not None:
                set_optimizer_state_dict(model, opt_g_dit,
                                         optim_state_dict=st["opt_g_dit"], options=opts)
                set_optimizer_state_dict(fake_model, opt_d_dit,
                                         optim_state_dict=st["opt_d_dit"], options=opts)
                opt_g_gs.load_state_dict(st["opt_g_gs"])
                opt_d_emb.load_state_dict(st["opt_d_emb"])
                p0("[dmd2] optimizer state restored")
        except Exception as e:                                       # noqa: BLE001
            p0(f"[dmd2] optimizer state NOT restored ({type(e).__name__}: {e})")
    p0(f"[dmd2] resumed at step {step}")
    return step


def _sample(args, step, gen, gen_model, sched, text_enc, val, dev, is_main, p0):
    """Sample with the few-step sampler. Runs on every rank; only rank 0 writes."""
    base = os.path.join(args.out_dir, "samples", f"step{step}") if is_main else None
    if base:
        os.makedirs(base, exist_ok=True)
    for mm in [gen_model, gen_model.dip_head] + list(gen_model.transformer.blocks):
        if hasattr(mm, "reshard"):
            mm.reshard()
    gen.eval()
    try:
        for i, sc in enumerate(val or []):
            emb = text_enc([MV_PREFIX + sc["text"]])
            cams = sc["cameras"].to(dev)
            for grp, ref in (("t2mv", None), ("i2mv", sc["image"][:, 0].to(dev))):
                x, scene = sample_few_step(gen, sched, emb, cams, height=args.height,
                                           width=args.width, ref_img=ref, cfg=1.0,
                                           seed=4321 + i, device=dev)
                if base:
                    _dump_views(x, os.path.join(base, grp), f"{i:02d}")
                    if scene is not None and args.export_scene:
                        g = prune_opacity(scene[0].detach().float(), args.prune_thr)
                        np.savez_compressed(
                            os.path.join(base, grp, f"{i:02d}_scene.npz"),
                            gaussians=g.cpu().numpy(),
                            cameras=cams.detach().float().cpu().numpy())
    finally:
        gen.train()
    if base:
        p0(f"[dmd2] sampled -> {base}")


if __name__ == "__main__":
    main()
