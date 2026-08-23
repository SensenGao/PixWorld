#!/usr/bin/env python
"""Build (or validate) a PixWorld ``index.jsonl`` from a directory of scenes.

See ``docs/DATASET.md`` for the format.  Two input shapes are supported:

``--format json``
    Each scene directory holds a ``meta.json`` with ``frames``, ``intrinsics`` and
    ``w2c`` (and optionally ``text``, ``depth``, ``depth_scale``).

``--format re10k``
    Each scene has a RealEstate10K-style ``.txt``: one line per frame,
    ``timestamp fx fy cx cy 0 0 <12 w2c values>``, with intrinsics already normalised.
    Images are matched to lines **by order**, after sorting the image filenames.

``--check`` validates an existing index instead of writing one.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def is_val(scene, val_mod=20):
    return int(hashlib.md5(scene.encode()).hexdigest(), 16) % val_mod == 0


def _images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def from_json(scene_dir, root, caption):
    meta = json.load(open(os.path.join(scene_dir, "meta.json")))
    rec = {"frames": meta["frames"], "intrinsics": meta["intrinsics"], "w2c": meta["w2c"]}
    rec["text"] = meta.get("text") or caption or ""
    if meta.get("depth"):
        rec["depth"] = meta["depth"]
        rec["depth_scale"] = float(meta.get("depth_scale", 0.0))
    if meta.get("k_novel"):
        rec["k_novel"] = int(meta["k_novel"])
    return rec


def from_re10k(scene_dir, root, caption):
    txts = glob.glob(os.path.join(scene_dir, "*.txt"))
    if not txts:
        raise ValueError("no .txt camera file")
    lines = [l.split() for l in open(txts[0]) if l.strip()]
    # The first line of a RealEstate10K file is the source URL, not a frame.
    lines = [p for p in lines if len(p) >= 19]
    img_dir = os.path.join(scene_dir, "rgb")
    if not os.path.isdir(img_dir):
        img_dir = scene_dir
    imgs = _images(img_dir)
    if len(imgs) != len(lines):
        raise ValueError(f"{len(imgs)} images but {len(lines)} camera lines")
    rel = os.path.relpath(img_dir, scene_dir)
    frames, intr, w2c = [], [], []
    for name, p in zip(imgs, lines):
        frames.append(os.path.join(rel, name) if rel != "." else name)
        intr.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
        w2c.append([float(v) for v in p[7:19]])
    return {"frames": frames, "intrinsics": intr, "w2c": w2c, "text": caption or ""}


def validate(rec, root, deep=True):
    """Return a list of problems with one record."""
    bad = []
    n = len(rec["frames"])
    if n == 0:
        return ["no frames"]
    if len(rec["intrinsics"]) != n or len(rec["w2c"]) != n:
        bad.append(f"length mismatch: frames {n}, intrinsics "
                   f"{len(rec['intrinsics'])}, w2c {len(rec['w2c'])}")
    if rec.get("depth") is not None and len(rec["depth"]) != n:
        bad.append(f"depth has {len(rec['depth'])} entries for {n} frames")
    if not rec.get("text"):
        bad.append("empty caption")
    for k in rec["intrinsics"][:1] + rec["intrinsics"][-1:]:
        if len(k) != 4:
            bad.append(f"intrinsics row has {len(k)} values, expected 4")
        elif not (0.05 < k[0] < 10 and 0.05 < k[1] < 10):
            bad.append(f"fx/fy = {k[0]:.3f}/{k[1]:.3f} -- these must be normalised by "
                       f"the image size, not in pixels")
        elif not (0.2 < k[2] < 0.8 and 0.2 < k[3] < 0.8):
            bad.append(f"principal point {k[2]:.3f},{k[3]:.3f} is far off centre")
    m = np.asarray(rec["w2c"], dtype=np.float64)
    if m.shape != (n, 12):
        bad.append(f"w2c shape {m.shape}, expected ({n}, 12)")
    elif not np.isfinite(m).all():
        bad.append("non-finite pose values")
    else:
        R = m.reshape(n, 3, 4)[:, :, :3]
        det = np.linalg.det(R)
        if np.abs(det - 1).max() > 1e-2:
            bad.append(f"rotation determinant ranges {det.min():.4f}..{det.max():.4f}; "
                       f"expected 1. Are these camera-to-world, or scaled?")
        t = m.reshape(n, 3, 4)[:, :, 3]
        if np.linalg.norm(t, axis=-1).max() < 1e-6:
            bad.append("all translations are zero -- the camera never moves")
    if deep:
        sdir = os.path.join(root, rec["root"])
        for f in (rec["frames"][0], rec["frames"][-1]):
            if not os.path.isfile(os.path.join(sdir, f)):
                bad.append(f"missing image {f}")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="DATA_ROOT")
    p.add_argument("--out", default="", help="output index path; default <root>/index.jsonl")
    p.add_argument("--scenes", default="scenes", help="subdirectory holding the scenes")
    p.add_argument("--format", default="json", choices=["json", "re10k"])
    p.add_argument("--captions", default="",
                   help="optional JSON or JSONL mapping scene id -> caption")
    p.add_argument("--source", default="", help="source tag; default is the parent folder")
    p.add_argument("--val_mod", type=int, default=20, help="1 scene in N goes to val")
    p.add_argument("--min_frames", type=int, default=32)
    p.add_argument("--check", action="store_true", help="validate an existing index")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    out = args.out or os.path.join(args.root, "index.jsonl")

    if args.check:
        n_ok = n_bad = 0
        with open(out) as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                probs = validate(rec, args.root)
                if probs:
                    n_bad += 1
                    if n_bad <= 20:
                        print(f"{out}:{lineno} scene {rec.get('scene')!r}")
                        for q in probs:
                            print(f"    {q}")
                else:
                    n_ok += 1
        print(f"\n{n_ok} scenes OK, {n_bad} with problems")
        return 1 if n_bad else 0

    captions = {}
    if args.captions:
        if args.captions.endswith(".jsonl"):
            for line in open(args.captions):
                d = json.loads(line)
                cap = d.get("caption") or d.get("text") or ""
                if isinstance(cap, dict):
                    cap = cap.get("long_caption") or cap.get("short_caption") or ""
                captions[d["scene"]] = cap.strip()
        else:
            captions = {k: (v if isinstance(v, str) else
                            (v.get("long_caption") or v.get("short_caption") or "")).strip()
                        for k, v in json.load(open(args.captions)).items()}
        print(f"loaded {len(captions):,} captions")

    base = os.path.join(args.root, args.scenes)
    if not os.path.isdir(base):
        raise SystemExit(f"{base} is not a directory (use --scenes to point elsewhere)")
    scene_dirs = sorted(d for d in glob.glob(os.path.join(base, "*", "*"))
                        if os.path.isdir(d)) or \
                 sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))

    reader = {"json": from_json, "re10k": from_re10k}[args.format]
    n_written = n_skipped = 0
    reasons = {}
    with open(out + ".tmp", "w") as f:
        for d in scene_dirs:
            if args.limit and n_written >= args.limit:
                break
            sid = os.path.basename(d)
            src = args.source or os.path.basename(os.path.dirname(d))
            try:
                rec = reader(d, args.root, captions.get(sid, ""))
            except Exception as e:                                   # noqa: BLE001
                reasons[type(e).__name__] = reasons.get(type(e).__name__, 0) + 1
                n_skipped += 1
                continue
            rec.update(scene=sid, source=src, root=os.path.relpath(d, args.root),
                       split="val" if is_val(sid, args.val_mod) else "train")
            if len(rec["frames"]) < args.min_frames:
                reasons["too_short"] = reasons.get("too_short", 0) + 1
                n_skipped += 1
                continue
            probs = validate(rec, args.root, deep=False)
            if probs:
                reasons[probs[0][:40]] = reasons.get(probs[0][:40], 0) + 1
                n_skipped += 1
                continue
            f.write(json.dumps(rec) + "\n")
            n_written += 1
    os.replace(out + ".tmp", out)
    print(f"wrote {n_written:,} scenes -> {out}")
    if n_skipped:
        print(f"skipped {n_skipped:,}: {reasons}")
    print(f"validate it with:  python tools/build_index.py --root {args.root} --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
