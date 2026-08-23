#!/usr/bin/env python
"""Convert a training checkpoint into the released PixWorld format.

The released file carries **weights and nothing else**.  A training checkpoint also holds
the optimizer state, the step counter and the full argument namespace -- learning rate,
data paths, run names -- none of which belongs in a published artefact.  What survives is
the state dict plus a small ``config`` block, and everything in that block is required to
instantiate the model or to drive its sampler; there is no training bookkeeping in it.

The released model drops four modules that are never reached by the multi-view forward
pass::

    transformer.proj_out.*      the original latent output head, replaced by the decoder
    proj_out_rest.*             a video-only second output head
    patch_embedding_rest.*      a video-only second patch embedding
    out_refine.*                a zero-initialised residual conv
    dip_pure                    a mode flag with only one possible value

Together they are ~85M parameters of frozen, unused weight in every checkpoint.  Dropping
them changes nothing numerically and makes the released model straightforward to read.

Usage::

    python tools/convert_checkpoint.py in.pt out.pt [--ema in_ema.pt] [--fp16]

``--ema`` overlays an EMA file, which is normally what you want to publish: the EMA
weights are what the training samples were drawn from.
"""
import argparse
import json
import os
import sys

import torch

DEAD_PREFIXES = ("transformer.proj_out.", "out_refine.", "patch_embedding_rest.",
                 "proj_out_rest.")
DEAD_EXACT = ("dip_pure",)
GS_PREFIXES = ("gs_enc.", "gs_dec.", "gs_head.")


def is_dead(name):
    return name.startswith(DEAD_PREFIXES) or name in DEAD_EXACT


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--ema", default="", help="EMA file to overlay onto the base weights")
    p.add_argument("--distilled", action="store_true",
                   help="mark this as a few-step model so inference auto-detects it")
    p.add_argument("--n_gen_steps", type=int, default=4)
    p.add_argument("--gen_shift", type=float, default=3.0)
    p.add_argument("--gs_depth_max", type=float, default=50.0)
    p.add_argument("--safetensors", action="store_true",
                   help="write weights as .safetensors and the config as a sibling "
                        "config.json. safetensors holds tensors and nothing else -- it "
                        "cannot carry pickled Python objects at all -- which is what you "
                        "want for anything published.")
    p.add_argument("--fp16", action="store_true",
                   help="store frozen tensors as fp16 instead of bf16 (smaller, and "
                        "fine for inference; the Gaussian stack always stays fp32)")
    args = p.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    if not isinstance(sd, dict):
        raise SystemExit(f"{args.src} does not hold a state dict")
    n_over = 0
    if args.ema:
        ema = torch.load(args.ema, map_location="cpu", weights_only=False)
        ema = ema.get("model", ema) if isinstance(ema, dict) else ema
        for n, t in ema.items():
            if n in sd and tuple(sd[n].shape) == tuple(t.shape):
                sd[n] = t
                n_over += 1
        print(f"overlaid {n_over}/{len(ema)} EMA tensors")
        if n_over == 0:
            raise SystemExit(
                f"{args.ema} matched none of the base keys. Publishing the base weights "
                f"while believing they are the EMA is exactly the mistake this check is "
                f"here to prevent.")

    out, dropped = {}, []
    for n, t in sd.items():
        if is_dead(n):
            dropped.append(n)
            continue
        if torch.is_tensor(t) and t.is_floating_point():
            if not torch.isfinite(t).all():
                raise SystemExit(f"{n} holds non-finite values; refusing to publish it")
            t = t.float() if n.startswith(GS_PREFIXES) else \
                t.to(torch.float16 if args.fp16 else torch.bfloat16)
        out[n] = t

    have_gs = {p_: any(n.startswith(p_) for n in out) for p_ in GS_PREFIXES}
    if not all(have_gs.values()):
        print(f"WARNING: missing Gaussian modules {[k for k, v in have_gs.items() if not v]}"
              f" -- this checkpoint cannot produce 3D output", file=sys.stderr)

    # Only what is needed to build the model and run its sampler.  `gs_depth_max` bounds
    # the Gaussian head's depth activation and therefore changes the geometry; the few-step
    # fields define the sampler ladder.  Nothing else is carried.
    cfg = {"gs_depth_max": args.gs_depth_max}
    if args.distilled:
        cfg.update(n_gen_steps=args.n_gen_steps, gen_shift=args.gen_shift, gs_step=-1)
    train_cfg = (ck.get("args") or {}).get("model_config", "") if isinstance(ck, dict) else ""
    if train_cfg:
        # A non-default transformer shape is architecture, so it has to travel with the
        # weights or they cannot be loaded at all.
        cfg["transformer"] = json.loads(
            open(train_cfg).read() if os.path.isfile(train_cfg) else train_cfg)
    if args.safetensors:
        from safetensors.torch import save_file
        # safetensors refuses shared storage, and every tensor must be contiguous.
        flat = {k: v.contiguous().clone() for k, v in out.items() if torch.is_tensor(v)}
        skipped = [k for k, v in out.items() if not torch.is_tensor(v)]
        if skipped:
            raise SystemExit(f"cannot write non-tensor entries to safetensors: {skipped}")
        tmp = args.dst + ".tmp"
        save_file(flat, tmp)
        os.replace(tmp, args.dst)
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.dst)), "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=1)
        print(f"  config written beside the weights: {cfg_path}")
    else:
        payload = {"model": out, "config": cfg}
        tmp = args.dst + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, args.dst)

    n_par = sum(t.numel() for t in out.values() if torch.is_tensor(t))
    n_drop = sum(sd[n].numel() for n in dropped if torch.is_tensor(sd[n]))
    print(f"{args.src}\n  -> {args.dst}")
    print(f"  kept    {len(out):4d} tensors, {n_par / 1e6:8.1f}M params")
    print(f"  dropped {len(dropped):4d} tensors, {n_drop / 1e6:8.1f}M params  {dropped[:4]}")
    print(f"  {'few-step distilled' if args.distilled else 'multi-step'}, "
          f"{os.path.getsize(args.dst) / 2 ** 30:.2f} GiB")
    print(f"  config: {json.dumps(cfg)}")
    print(f"  dropped: optimizer state, step counter and the training argument namespace")


if __name__ == "__main__":
    main()
