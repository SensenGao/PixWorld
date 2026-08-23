#!/usr/bin/env python
"""PixWorld inference: text or image -> an explorable 3D Gaussian scene.

One script drives both released checkpoints; the ``config.json`` beside the weights says
which is which.  Every run needs a camera path plus a prompt, a reference image, or both.

    export PIXWORLD=$(python -c "from modelscope import snapshot_download; \\
        print(snapshot_download('SensenGao/PixWorld'))")
    export FEW=$PIXWORLD/PixWorld-L2P-Wan5B-4steps/PixWorld-L2P-Wan5B-4steps.safetensors

    python infer/infer.py --ckpt $FEW \\
        --wan_path weights/Wan2.2-TI2V-5B-Diffusers \\
        --cameras infer/examples/poses/t2mv_ridge.json \\
        --prompt "a rocky mountain ridge at sunset above the sea" \\
        --out out/ridge

Add ``--image photo.jpg`` for image-to-3D, or ``--mode recon`` with ``--views_dir`` to
reconstruct from views you already have.  ``--help`` lists everything; see
``infer/README.md``.

Each run writes the views, a contact sheet, a video rendered from the Gaussian field along
the camera path, and the field itself as ``.ply`` and ``.npz``.
"""
import argparse
import json
import os
import sys
import time

import torch

# Everything this script needs lives beside it -- no imports from anywhere else.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry.cameras import load_cameras_json, save_cameras_json  # noqa: E402
from diffusion.dmd_core import Generator  # noqa: E402
from diffusion.dmd_sampling import sample_few_step  # noqa: E402
from diffusion.dmd_schedule import GenSchedule  # noqa: E402
from utils.export import (save_grid, save_image, save_npz, save_ply,  # noqa: E402
                             save_video)
from models.dit import PixWorldMV, WAN22_5B_PIXEL_CONFIG  # noqa: E402
from models.gaussian_head import PixelAlignedGaussianHead  # noqa: E402
from models.gs_lift import GaussianDecoder, GaussianEncoder  # noqa: E402
from geometry.render import render_path, render_views, resample_cameras  # noqa: E402
from diffusion.sampling import reconstruct, sample  # noqa: E402
from geometry.trajectories import TRAJECTORIES, make_trajectory  # noqa: E402
from data.text_encoder import DEFAULT_NEG, MV_PREFIX, TextEncoder  # noqa: E402


def get_args():
    p = argparse.ArgumentParser(
        description="Generate a 3D Gaussian scene from text, an image, or posed views.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True, help="a PixWorld checkpoint")
    p.add_argument("--wan_path", required=True, help="for the UMT5 text encoder")
    p.add_argument("--out", required=True)

    g = p.add_argument_group("what to generate")
    g.add_argument("--prompt", default="", help="text prompt (without any prefix)")
    g.add_argument("--image", default="", help="reference image for image-to-3D")
    g.add_argument("--data_root", default="", help="dataset root, to reuse real cameras")
    g.add_argument("--scene", default="", help="scene id in --data_root; empty means the "
                                               "first val scene")
    g.add_argument("--index", default="index.jsonl")
    g.add_argument("--mode", default="auto", choices=["auto", "generate", "recon"],
                   help="'recon' reconstructs from the scene's own views in a single "
                        "forward pass at sigma=0 instead of generating. Needs "
                        "--data_root/--scene. 'auto' picks recon when a scene is given "
                        "and no prompt.")
    g.add_argument("--views_dir", default="",
                   help="directory of posed views to reconstruct from, one image per "
                        "camera in --cameras, sorted by filename. This is the "
                        "dataset-free way to run reconstruction; see "
                        "infer/examples/recon_bedroom/.")
    g.add_argument("--cameras", default="",
                   help="path to a camera JSON (see infer/examples/poses/). Overrides "
                        "--trajectory.")
    g.add_argument("--trajectory", default="dolly", choices=sorted(TRAJECTORIES),
                   help="synthetic camera path, used when no dataset scene is given")
    g.add_argument("--views", type=int, default=16)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--width", type=int, default=832)
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("sampler")
    g.add_argument("--steps", type=int, default=0,
                   help="0 auto-detects from the checkpoint: distilled models use their "
                        "own 4-step schedule, others 50 steps")
    g.add_argument("--cfg", type=float, default=-1.0,
                   help="-1 auto: 5.0 for the multi-step model, 1.0 for a distilled one "
                        "(which already has guidance baked in)")
    g.add_argument("--cfg_rescale", type=float, default=0.7)
    g.add_argument("--shift", type=float, default=16.0,
                   help="multi-step sampler only. 16 is the multi-view training schedule; "
                        "8 is the single-image one.")
    g.add_argument("--sigma_switch", type=float, default=0.5,
                   help="multi-step only: noise level below which the sampler renders "
                        "the Gaussian field instead of the 2D prediction. 0 disables the "
                        "3D path entirely.")
    g.add_argument("--gen_shift", type=float, default=3.0, help="distilled sampler only")
    g.add_argument("--few_step", default="auto", choices=["auto", "0", "1"],
                   help="whether this checkpoint is the distilled 4-step model. 'auto' "
                        "reads it from the checkpoint's config, which is what you want; "
                        "set it explicitly only for a bare state dict with no config.")
    g.add_argument("--negative", default="",
                   help="negative prompt for classifier-free guidance; empty uses "
                        "DEFAULT_NEG. "
                        "Ignored by the distilled model, which runs at cfg 1.")

    g = p.add_argument_group("outputs")
    g.add_argument("--video_frames", type=int, default=81,
                   help="frames in the rendered camera video. The 16 input cameras are "
                        "resampled to this many, so the motion is the same path, just "
                        "smoother. 0 disables the video.")
    g.add_argument("--video_round_trip", action="store_true",
                   help="fly the path out and back (0 -> 15 -> 0) instead of one way, so "
                        "the video loops seamlessly")
    g.add_argument("--video_chunk", type=int, default=16,
                   help="views rasterised per batch; lower this if a long video OOMs")
    g.add_argument("--fps", type=int, default=12)
    g.add_argument("--ply", type=int, default=1)
    g.add_argument("--ply_max_points", type=int, default=2_000_000,
                   help="keep only the most opaque N in the .ply; 16 views at 480x832 is "
                        "12.8M Gaussians, more than most viewers will open")
    g.add_argument("--npz", type=int, default=1)
    g.add_argument("--prune", type=float, default=0.01, help="opacity threshold for export")
    g.add_argument("--bg", default="white", choices=["white", "black", "random"])
    return p.parse_args()


def load_model(ckpt_path, device, verbose=print, few_step="auto"):
    """Build the model from a checkpoint and report which kind it is."""
    if ckpt_path.endswith(".safetensors"):
        # safetensors stores tensors and nothing else, so the settings that decide HOW to
        # sample (above all: whether this is the distilled model) travel in a config.json
        # beside the weights.
        from safetensors.torch import load_file
        sd = load_file(ckpt_path)
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                saved = json.load(f)
        else:
            saved = {}
            verbose(f"[pixworld] WARNING: no config.json beside {os.path.basename(ckpt_path)}; "
                    "assuming the 50-step model. If this is the 4-step checkpoint, pass "
                    "--few_step 1.")
    else:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        # A released checkpoint carries `config` -- model and sampler settings only.  A raw
        # training checkpoint carries `args`, the whole namespace; read either.
        saved = (ck.get("config") or ck.get("args") or {}) if isinstance(ck, dict) else {}
    cfg = dict(WAN22_5B_PIXEL_CONFIG)
    over = saved.get("transformer") or saved.get("model_config")
    if over:
        if isinstance(over, str):
            over = json.loads(open(over).read() if os.path.isfile(over) else over)
        cfg.update({k: (tuple(v) if k == "patch_size" else v) for k, v in over.items()})
        verbose(f"[pixworld] transformer config from checkpoint: {over}")

    model = PixWorldMV(cfg)
    gs_enc = GaussianEncoder()
    gs_dec = GaussianDecoder(ctx_dim=model.inner_dim)
    gs_head = PixelAlignedGaussianHead(
        feat_ch=64, checkpoint=False,
        depth_max=float(saved.get("gs_depth_max", 50.0)))

    parts = {"gs_enc.": {}, "gs_dec.": {}, "gs_head.": {}}
    core = {}
    for n, t in sd.items():
        for pref in parts:
            if n.startswith(pref):
                parts[pref][n[len(pref):]] = t
                break
        else:
            core[n] = t
    missing, unexpected = model.load_state_dict(core, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in {ckpt_path}: {unexpected[:8]}")
    if missing:
        verbose(f"[pixworld] {len(missing)} transformer tensors absent: {missing[:4]}")
    for pref, mod in (("gs_enc.", gs_enc), ("gs_dec.", gs_dec), ("gs_head.", gs_head)):
        if not parts[pref]:
            raise RuntimeError(
                f"{ckpt_path} has no {pref[:-1]} weights. This checkpoint cannot produce "
                f"3D output.")
        mod.load_state_dict(parts[pref], strict=True)

    distilled = saved.get("variant") == "few_step" or "n_gen_steps" in saved
    if few_step in ("0", "1"):
        distilled = few_step == "1"
    for m in (model, gs_enc, gs_dec, gs_head):
        m.to(device).eval().requires_grad_(False)
    model.to(torch.bfloat16)
    for m in (gs_enc, gs_dec, gs_head):
        m.float()                       # geometry stays fp32
    verbose(f"[pixworld] loaded {'few-step (distilled)' if distilled else 'multi-step'} "
            f"model from {os.path.basename(ckpt_path)}")
    return model, gs_enc, gs_dec, gs_head, distilled, saved


def load_reference(path, height, width):
    """Read an image and cover-resize + centre-crop it, exactly as the dataset does."""
    from PIL import Image
    import numpy as np
    from geometry.cameras import crop_geom
    im = Image.open(path).convert("RGB")
    nw, nh, left, top = crop_geom(im.size[0], im.size[1], width, height)
    im = im.resize((nw, nh), Image.BICUBIC).crop((left, top, left + width, top + height))
    x = torch.from_numpy(np.asarray(im, dtype="float32")) / 127.5 - 1.0
    return x.permute(2, 0, 1)


def main():
    args = get_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)

    # ---- cameras (and possibly a reference frame) come first, so we know V ----
    ref_img = None
    scene_text = ""
    scene_views = None
    if args.data_root:
        from data.dataset import MultiViewDataset
        ds = MultiViewDataset(args.data_root, split="all", v=args.views, k_novel=0,
                              n_target=args.views, both_ends_prob=0.0,
                              height=args.height, width=args.width, index=args.index,
                              uniform_frames=True, verbose=False)
        idx = 0
        if args.scene:
            ids = [r["scene"] for r in ds.rows]
            if args.scene not in ids:
                raise SystemExit(f"scene {args.scene!r} not in {args.data_root}")
            idx = ids.index(args.scene)
        item = ds[idx]
        cameras = item["cameras"]
        scene_text = item["text"]
        scene_views = item["image"].permute(1, 0, 2, 3)          # [V,3,H,W]
        if args.image == "scene":
            ref_img = item["image"][:, 0]
        print(f"[pixworld] cameras from scene {item['scene']!r} "
              f"({cameras.shape[0]} views)")
    elif args.cameras:
        cameras = load_cameras_json(args.cameras)
        if args.views_dir:
            names = sorted(f for f in os.listdir(args.views_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
            if len(names) != cameras.shape[0]:
                raise SystemExit(
                    f"--views_dir has {len(names)} images but --cameras has "
                    f"{cameras.shape[0]} cameras; they must correspond one to one, "
                    "in filename order")
            scene_views = torch.stack([
                load_reference(os.path.join(args.views_dir, n), args.height, args.width)
                for n in names])
            print(f"[pixworld] {len(names)} posed views from {args.views_dir}")
    else:
        cameras = make_trajectory(args.trajectory, n_views=args.views)
        print(f"[pixworld] synthetic {args.trajectory} trajectory, {args.views} views")

    if args.image and args.image != "scene":
        ref_img = load_reference(args.image, args.height, args.width)
        save_image(ref_img, os.path.join(args.out, "reference.png"))

    # Reconstruction is text-conditioned only nominally: at sigma = 0 the clean views
    # already determine the output, so a prompt is optional here and the empty string --
    # the model's own trained unconditional -- is what it gets.  Generation still needs one.
    will_recon = args.mode == "recon" or (
        args.mode == "auto" and scene_views is not None and not args.prompt)
    prompt = args.prompt or scene_text
    if not prompt and not will_recon:
        raise SystemExit("give a --prompt, or a --data_root scene that carries a caption "
                         "(reconstruction needs neither -- add --mode recon)")

    # ---- text first, then free the encoder, then the transformer ----
    t0 = time.time()
    text_enc = TextEncoder(args.wan_path, dev, dtype=torch.bfloat16)
    emb = text_enc([MV_PREFIX + prompt if prompt else ""])
    neg = text_enc([args.negative or DEFAULT_NEG])
    text_enc.free()
    print(f"[pixworld] prompt encoded in {time.time() - t0:.1f}s; text encoder freed")

    model, gs_enc, gs_dec, gs_head, distilled, saved = load_model(
        args.ckpt, dev, few_step=args.few_step)
    cfg = args.cfg if args.cfg >= 0 else (1.0 if distilled else 5.0)

    mode = args.mode
    if mode == "auto":
        mode = "recon" if (scene_views is not None and not args.prompt) else "generate"
    if mode == "recon" and scene_views is None:
        raise SystemExit("--mode recon needs views to reconstruct: either "
                         "--cameras with --views_dir, or --data_root with a scene")

    t0 = time.time()
    if mode == "recon":
        # One forward at sigma = 0. No sampler, no guidance, no step count -- the views
        # are already known, so there is nothing to denoise.
        print(f"[pixworld] reconstruction: {scene_views.shape[0]} posed views -> "
              f"one forward at sigma=0 -> gaussians")
        x_r, _depth, scene = reconstruct(model, gs_enc, gs_dec, gs_head,
                                         scene_views.to(dev), cameras.to(dev), emb,
                                         bg_mode=args.bg)
        x = x_r.unsqueeze(0).permute(0, 2, 1, 3, 4)
        # How closely the field re-renders the views it was built from.
        mse = ((x_r.clamp(-1, 1) - scene_views.to(dev)) / 2).square().mean().item()
        print(f"[pixworld] re-render error on the input views: {mse:.5f} (mse)")
    elif distilled and args.steps <= 0:
        sched = GenSchedule(n_steps=int(saved.get("n_gen_steps", 4)),
                            shift=float(saved.get("gen_shift", args.gen_shift)),
                            gs_step=int(saved.get("gs_step", -1)))
        print(f"[pixworld] {sched.describe()}")
        gen = Generator(model, gs_enc, gs_dec, gs_head)
        x, scene = sample_few_step(gen, sched, emb, cameras.to(dev), height=args.height,
                                   width=args.width, ref_img=ref_img, neg_emb=neg,
                                   cfg=cfg, seed=args.seed, device=dev)
    else:
        steps = args.steps if args.steps > 0 else 50
        print(f"[pixworld] {steps}-step sampler, cfg {cfg}, "
              f"render-as-x0 below sigma {args.sigma_switch}")
        x, scene = sample(model, gs_enc, gs_dec, gs_head, emb, neg,
                          cameras=cameras.to(dev), ref_img=ref_img,
                          n_views=cameras.shape[0], height=args.height, width=args.width,
                          cfg=cfg, steps=steps, shift=args.shift,
                          cfg_rescale=args.cfg_rescale, seed=args.seed, device=dev,
                          sigma_switch=args.sigma_switch, bg_mode=args.bg)
    dt = time.time() - t0
    print(f"[pixworld] sampled {cameras.shape[0]} views in {dt:.2f}s")

    views = x[0].permute(1, 0, 2, 3)                          # [V,3,H,W]
    for v in range(views.shape[0]):
        save_image(views[v], os.path.join(args.out, "views", f"v{v:02d}.png"))
    save_grid(views, os.path.join(args.out, "grid.png"))

    meta = {"prompt": prompt, "checkpoint": os.path.abspath(args.ckpt),
            "distilled": bool(distilled), "cfg": cfg, "seed": args.seed,
            "views": int(cameras.shape[0]), "height": args.height, "width": args.width,
            "trajectory": ("dataset" if args.data_root else
                           (os.path.basename(args.cameras) if args.cameras
                            else args.trajectory)),
            "seconds": round(dt, 3)}

    if scene is None:
        print("[pixworld] no Gaussian field was produced -- the sampler never reached the "
              "3D stage. With --sigma_switch 0 that is expected; otherwise raise --steps.")
    else:
        meta["gaussians"] = int(scene.shape[1])
        if args.npz:
            n = save_npz(scene[0], cameras, os.path.join(args.out, "scene.npz"), args.prune)
            print(f"[pixworld] scene.npz: {n:,} gaussians")
        if args.ply:
            n = save_ply(scene[0], os.path.join(args.out, "scene.ply"), args.prune,
                         args.ply_max_points)
            print(f"[pixworld] scene.ply: {n:,} gaussians")
        if args.video_frames > 0:
            path_cams = resample_cameras(cameras.cpu(), args.video_frames,
                                         round_trip=args.video_round_trip).to(dev)
            imgs = render_path(scene, path_cams, args.height, args.width,
                               bg_mode=args.bg, chunk=args.video_chunk)
            out_mp4 = save_video(imgs * 2 - 1, os.path.join(args.out, "sweep.mp4"),
                                 fps=args.fps)
            meta["video"] = {"frames": int(path_cams.shape[0]),
                             "round_trip": bool(args.video_round_trip),
                             "fps": int(args.fps)}
            print(f"[pixworld] video: {path_cams.shape[0]} frames"
                  f"{' (round trip)' if args.video_round_trip else ''}"
                  f" -> {os.path.basename(out_mp4)}")

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[pixworld] done -> {args.out}")


if __name__ == "__main__":
    main()
