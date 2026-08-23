#!/usr/bin/env python
"""Run every case in ``infer/examples/inputs.json`` through both checkpoints.

The examples directory ships **inputs only** -- camera paths, reference images and the
reconstruction's posed views -- so this is what turns them into outputs.  Each checkpoint
(~10 GiB) is loaded once and run over all six cases.

    python infer/make_examples.py \\
        --ckpt_multi weights/PixWorld-L2P-Wan5B.safetensors \\
        --ckpt_few   weights/PixWorld-L2P-Wan5B-4steps.safetensors \\
        --wan_path   weights/Wan2.2-TI2V-5B-Diffusers \\
        --examples   infer/examples --out out/examples

Every case names its own camera path, so nothing here invents a trajectory.
"""
import argparse
import json
import os
import sys
import time

import torch

# Everything this script needs lives beside it -- no imports from anywhere else.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry.cameras import load_cameras_json  # noqa: E402
from diffusion.dmd_core import Generator  # noqa: E402
from diffusion.dmd_sampling import sample_few_step  # noqa: E402
from diffusion.dmd_schedule import GenSchedule  # noqa: E402
from utils.export import (save_grid, save_image, save_npz,  # noqa: E402
                    save_ply, save_video)
from geometry.render import render_path, resample_cameras  # noqa: E402
from diffusion.sampling import reconstruct, sample  # noqa: E402
from data.text_encoder import DEFAULT_NEG, MV_PREFIX, TextEncoder  # noqa: E402

from infer import load_model, load_reference  # noqa: E402


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_multi", required=True, help="the 50-step base checkpoint")
    p.add_argument("--ckpt_few", required=True, help="the 4-step distilled checkpoint")
    p.add_argument("--wan_path", required=True, help="Wan2.2-TI2V-5B, for its text encoder")
    p.add_argument("--examples", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "examples"), help="directory holding inputs.json")
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="", help="comma-separated subset of case names")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)

    g = p.add_argument_group("sampler")
    g.add_argument("--steps", type=int, default=50, help="base model only")
    g.add_argument("--cfg", type=float, default=5.0, help="base model only")
    g.add_argument("--shift", type=float, default=16.0,
                   help="multi-view shift. 16 is the multi-view training schedule; "
                        "8 is the single-image one.")
    g.add_argument("--sigma_switch", type=float, default=0.5)
    g.add_argument("--cfg_rescale", type=float, default=0.7)

    g = p.add_argument_group("outputs")
    g.add_argument("--video_frames", type=int, default=81, help="0 disables the video")
    g.add_argument("--video_round_trip", type=int, default=1,
                   help="fly the path out and back, so the clip loops")
    g.add_argument("--video_chunk", type=int, default=16)
    g.add_argument("--fps", type=int, default=24)
    g.add_argument("--save_views", type=int, default=1)
    g.add_argument("--ply", type=int, default=0, help="also export .ply (large)")
    g.add_argument("--npz", type=int, default=0,
                   help="also export .npz. 16 views at 480x832 is 12.8M gaussians, "
                        "roughly 1 GB per scene -- off by default.")
    return p.parse_args()


def load_case(name, spec, ex_dir, height, width, device):
    """Resolve one inputs.json entry into the tensors the samplers want."""
    cams = load_cameras_json(os.path.join(ex_dir, spec["cameras"])).to(device)
    ref = None
    if spec.get("reference"):
        ref = load_reference(os.path.join(ex_dir, spec["reference"]),
                             height, width).to(device)
    views = None
    if spec["task"] == "reconstruction":
        d = os.path.join(ex_dir, spec["images"])
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png")))
        if len(files) != cams.shape[0]:
            raise SystemExit(f"{name}: {len(files)} images but {cams.shape[0]} cameras")
        views = torch.stack([load_reference(os.path.join(d, f), height, width)
                             for f in files]).to(device)
    return cams, ref, views


def main():
    args = get_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with open(os.path.join(args.examples, "inputs.json")) as f:
        cases = json.load(f)
    if args.only:
        want = set(args.only.split(","))
        cases = {k: v for k, v in cases.items() if k in want}
        if not cases:
            raise SystemExit(f"--only matched nothing; have {sorted(cases)}")
    os.makedirs(args.out, exist_ok=True)

    # Encode every prompt up front, then free the text encoder before any model loads --
    # UMT5 and a 5B transformer do not need to be resident at the same time.
    # Generation needs a prompt.  Reconstruction does not -- at sigma = 0 the clean views
    # determine the output -- so it gets the empty string, which is the model's own
    # trained unconditional.
    missing = [k for k, v in cases.items()
               if not v.get("prompt") and v["task"] != "reconstruction"]
    if missing:
        raise SystemExit(f"inputs.json: no prompt for {', '.join(missing)}")
    te = TextEncoder(args.wan_path, dev, dtype=torch.bfloat16)
    embs = {k: te([MV_PREFIX + v["prompt"] if v.get("prompt") else ""])
            for k, v in cases.items()}
    neg = te([DEFAULT_NEG])
    te.free()
    print(f"[examples] encoded {len(embs)} prompts; text encoder freed", flush=True)

    loaded = {k: load_case(k, v, args.examples, args.height, args.width, dev)
              for k, v in cases.items()}
    results = {}

    for tag, ckpt in (("50step", args.ckpt_multi), ("4step", args.ckpt_few)):
        model, gs_enc, gs_dec, gs_head, distilled, saved = load_model(ckpt, dev)
        gen = Generator(model, gs_enc, gs_dec, gs_head) if distilled else None
        sched = GenSchedule(n_steps=int(saved.get("n_gen_steps", 4)),
                            shift=float(saved.get("gen_shift", 3.0)),
                            gs_step=int(saved.get("gs_step", -1))) if distilled else None
        for name, spec in cases.items():
            cams, ref, views = loaded[name]
            emb = embs.get(name)
            d = os.path.join(args.out, name, tag)
            os.makedirs(d, exist_ok=True)
            torch.cuda.synchronize()
            t0 = time.time()
            if spec["task"] == "reconstruction":
                # One forward pass at sigma = 0.  No sampler, no guidance, and identical
                # for both checkpoints -- the step count never enters this path.
                imgs, _depth, scene = reconstruct(model, gs_enc, gs_dec, gs_head, views,
                                                  cams, emb, bg_mode="white")
                x = imgs.permute(1, 0, 2, 3).unsqueeze(0)
            elif distilled:
                x, scene = sample_few_step(gen, sched, emb, cams, height=args.height,
                                           width=args.width, ref_img=ref, cfg=1.0,
                                           seed=int(spec.get("seed", 0)), device=dev)
            else:
                x, scene = sample(model, gs_enc, gs_dec, gs_head, emb, neg, cameras=cams,
                                  ref_img=ref, n_views=cams.shape[0], height=args.height,
                                  width=args.width, cfg=args.cfg, steps=args.steps,
                                  shift=args.shift, cfg_rescale=args.cfg_rescale,
                                  seed=int(spec.get("seed", 0)), device=dev,
                                  sigma_switch=args.sigma_switch)
            torch.cuda.synchronize()
            dt = time.time() - t0

            out_views = x[0].permute(1, 0, 2, 3)
            if args.save_views:
                for v in range(out_views.shape[0]):
                    save_image(out_views[v], os.path.join(d, f"v{v:02d}.png"))
            save_grid(out_views, os.path.join(d, "grid.png"))
            n_g = int(scene.shape[1]) if scene is not None else 0
            if scene is not None:
                if args.npz:
                    save_npz(scene[0], cams, os.path.join(d, "scene.npz"))
                if args.ply:
                    save_ply(scene[0], os.path.join(d, "scene.ply"), max_points=2_000_000)
                if args.video_frames > 0:
                    path = resample_cameras(cams.cpu(), args.video_frames,
                                            round_trip=bool(args.video_round_trip)).to(dev)
                    frames = render_path(scene, path, args.height, args.width,
                                         bg_mode="white", chunk=args.video_chunk)
                    save_video(frames * 2 - 1, os.path.join(d, "sweep.mp4"), fps=args.fps)
            results.setdefault(name, {})[tag] = {
                "seconds": round(dt, 3), "gaussians": n_g,
                "task": spec["task"], "cameras": spec["cameras"]}
            print(f"[examples] {name:18s} {tag:7s} {dt:6.2f}s  {n_g:,} gaussians",
                  flush=True)
        del model, gs_enc, gs_dec, gs_head, gen
        torch.cuda.empty_cache()

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[examples] done -> {args.out}")


if __name__ == "__main__":
    main()
