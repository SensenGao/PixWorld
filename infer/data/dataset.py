"""Posed multi-view dataset.

Reads a plain local directory: one JSON Lines index plus per-scene image folders.  There
is no database, no object store and no credential anywhere in this file -- see
``docs/DATASET.md`` for the format and ``tools/build_index.py`` for a converter.

Each sample is one scene, from which ``V + K`` frames are drawn along the clip:

* ``V`` of them are **input** views the model sees,
* ``K`` are **held-out novel** views used only as rendering targets,
* and ``n_target`` of the ``V + K`` are actually rendered and scored, so the render cost
  does not grow with ``K``.

Novel positions are always interior, which has two consequences worth knowing: every novel
view is an interpolation rather than an extrapolation, and the temporally first and last
frames are always inputs -- so "both ends" is well defined as gather indices ``0`` and
``V-1``.

Cameras for all ``V + K`` frames are normalised **jointly**: re-rooted to input frame 0 and
divided by the largest input translation.  Novel views ride the same transform, so their
translation norm may exceed 1.  Depth, when present, is divided by that same scale, which
is what puts it in the same units the Gaussian head predicts.
"""
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image

from geometry.cameras import (adjust_intrinsics_crop, build_cameras, crop_geom,
                       normalize_cameras_joint, quaternion_to_matrix)

__all__ = ["MultiViewDataset", "collate_mv", "gather_targets", "source_sampler",
           "val_scenes", "load_index"]

DEPTH_ENCODING = "uint16_png_div65535_times_scale"


# ------------------------------------------------------------------------- index --
def is_val(scene, val_mod=20):
    """Stable per-scene held-out flag: ``md5(scene) % val_mod == 0``.

    Used only when the index does not carry an explicit ``split``.  Hashing the scene id
    means a scene appearing in two source pools always lands on the same side.
    """
    return int(hashlib.md5(scene.encode()).hexdigest(), 16) % val_mod == 0


def load_index(root, index="index.jsonl", split="train", sources=None,
               min_frames_for=None, val_mod=20, verbose=True):
    """Read ``index.jsonl`` into a list of scene records.

    Args:
        root: dataset root; every ``root`` field in the index is relative to it.
        split: ``"train"``, ``"val"`` or ``"all"``.
        sources: keep only these source tags (``None`` keeps all).
        min_frames_for: ``callable(record) -> int`` minimum frame count.

    Returns:
        ``(records, source_tags)``, two parallel lists.
    """
    path = os.path.join(root, index)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. See docs/DATASET.md for the expected layout, or build it "
            f"with tools/build_index.py.")
    recs, tags, dropped = [], [], {"caption": 0, "short": 0, "split": 0, "source": 0}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tag = r.get("source", "default")
            if sources is not None and tag not in sources:
                dropped["source"] += 1
                continue
            if not r.get("text"):
                dropped["caption"] += 1
                continue
            frames = r["frames"]
            n = len(frames)
            if len(r["intrinsics"]) != n or len(r["w2c"]) != n:
                raise ValueError(
                    f"{path}:{lineno} scene {r.get('scene')!r}: frames/intrinsics/w2c "
                    f"lengths differ ({n}/{len(r['intrinsics'])}/{len(r['w2c'])})")
            dep = r.get("depth")
            if dep is not None and len(dep) != n:
                raise ValueError(
                    f"{path}:{lineno} scene {r.get('scene')!r}: depth has {len(dep)} "
                    f"entries for {n} frames")
            if dep is not None and r.get("depth_encoding", DEPTH_ENCODING) != DEPTH_ENCODING:
                raise ValueError(
                    f"{path}:{lineno}: depth_encoding {r.get('depth_encoding')!r} is not "
                    f"supported (only {DEPTH_ENCODING!r})")
            mf = min_frames_for(r) if min_frames_for else 1
            if n < mf:
                dropped["short"] += 1
                continue
            sp = r.get("split") or ("val" if is_val(r["scene"], val_mod) else "train")
            if split != "all" and sp != split:
                dropped["split"] += 1
                continue
            r.setdefault("depth", None)
            r.setdefault("depth_scale", 0.0)
            recs.append(r)
            tags.append(tag)
    if verbose:
        by = {}
        for t in tags:
            by[t] = by.get(t, 0) + 1
        summary = ", ".join(f"{k}={v:,}" for k, v in sorted(by.items())) or "(none)"
        print(f"[pixworld] {path} split={split}: {len(recs):,} scenes  [{summary}]  "
              f"dropped: {dropped}", flush=True)
    return recs, tags


# ----------------------------------------------------------------------- dataset --
class MultiViewDataset(torch.utils.data.Dataset):
    """Posed multi-view scenes with held-out novel views.

    Args:
        root: dataset root directory.
        split: ``"train"``, ``"val"`` or ``"all"``.
        v: number of input views.
        k_novel: number of held-out novel views.  A scene may override this with its own
            ``k_novel`` field (a 24-frame clip cannot spare 16 novels on top of 16 inputs).
        n_target: how many of the ``V + K`` views the render loss is computed on.
        both_ends_prob: probability that the temporally first and last views are forced
            into the target set.
        height, width: output resolution; frames are cover-resized and centre-cropped, and
            the intrinsics are adjusted with the same integer geometry.
        sources: restrict to these source tags.
        uniform_frames: exact ``linspace`` frame positions instead of jittered ones.
            Defaults to True for training and
            is always True for val.
        max_retries: how many times a failed scene is re-drawn before giving up.  Re-draws
            stay **within the same source**, so an I/O problem in one pool cannot quietly
            shift the realised source mixture.

    Each item is a dict::

        image         [3, V, H, W]  fp32 in [-1, 1]
        cameras       [V, 11]       fp32, jointly normalised, row 0 is the identity pose
        novel_image   [3, K, H, W]  fp32 in [-1, 1]
        novel_cameras [K, 11]       fp32, same transform as `cameras`
        depth         [V, H, W]     fp32, normalised z-depth; 0.0 means "no ground truth"
        novel_depth   [K, H, W]     fp32
        target_gather [n_target]    int64 indices into cat([image, novel_image], dim=1)
        text          str           the RAW caption, with no prompt prefix
        scene, source str

    ``text`` is raw: the trainer adds the multi-view prompt prefix.    """

    def __init__(self, root, split="train", v=16, k_novel=16, n_target=16,
                 both_ends_prob=1.0, height=480, width=832, index="index.jsonl",
                 sources=None, uniform_frames=None, val_mod=20, max_retries=64,
                 verbose=True):
        assert v >= 1 and k_novel >= 0
        self.root = root
        self.V = int(v)
        self.K = int(k_novel)
        self.h, self.w = int(height), int(width)
        self.n_target = int(n_target)
        self.both_ends_prob = float(both_ends_prob)
        self.is_train = (split == "train")
        self.uniform_frames = True if uniform_frames is None else bool(uniform_frames)
        self.max_retries = int(max_retries) if self.is_train else 8
        assert self.n_target > 0, "n_target must be positive"
        assert self.n_target >= 2 or self.both_ends_prob == 0.0, \
            "both_ends needs room for two forced targets"

        def _min_frames(r):
            return self.V + int(r.get("k_novel") or self.K)

        self.rows, self.src = load_index(
            root, index=index, split=split, sources=sources,
            min_frames_for=_min_frames, val_mod=val_mod, verbose=verbose)
        if not self.rows:
            raise RuntimeError(
                f"MultiViewDataset: no usable scenes for split={split!r} under {root!r}")
        for r in self.rows:
            k = int(r.get("k_novel") or self.K)
            if self.n_target > self.V + k:
                raise ValueError(
                    f"scene {r['scene']!r}: n_target {self.n_target} > V+K {self.V + k}")
        self._by_src = {}
        for j, t in enumerate(self.src):
            self._by_src.setdefault(t, []).append(j)

    def __len__(self):
        return len(self.rows)

    # ------------------------------------------------------------------ frames --
    def _pick_indices(self, n, k):
        """``V + k`` strictly increasing frame indices spanning the whole clip."""
        total = self.V + k
        base = np.linspace(0, n - 1, total)
        if self.uniform_frames or not self.is_train or n <= total:
            idx = np.round(base).astype(int)
        else:
            gap = (n - 1) / (total - 1)
            idx = np.clip(np.round(base + (np.random.rand(total) - 0.5) * gap),
                          0, n - 1).astype(int)
        for j in range(1, total):                       # enforce strictly increasing
            if idx[j] <= idx[j - 1]:
                idx[j] = min(idx[j - 1] + 1, n - 1)
        return idx.tolist()

    def _novel_positions(self, total, k):
        """``k`` interior positions among ``total``, sorted.

        Interior means positions ``0`` and ``total-1`` are always inputs: position 0 is the
        normalisation root and the image-conditioning reference, and keeping both ends as
        inputs makes every novel view an interpolation.
        """
        if not self.is_train:
            return [((j + 1) * total) // (k + 1) for j in range(k)]
        assert total - 2 >= k, f"cannot place {k} interior novel views among {total}"
        return sorted(random.sample(range(1, total - 1), k))

    def _sample_targets(self, v, k):
        """Which of the ``V + K`` views the render loss is computed on, as int64 indices
        into ``cat([image, novel_image], dim=1)``."""
        forced = [0, v - 1] if (v >= 2 and random.random() < self.both_ends_prob) else []
        rest = [p for p in range(v + k) if p not in forced]
        pick = forced + random.sample(rest, self.n_target - len(forced))
        return torch.tensor(sorted(pick), dtype=torch.long)

    # ------------------------------------------------------------------ loading --
    def _resize_crop(self, im):
        """Cover-resize + centre-crop, sharing :func:`~cameras.crop_geom` with the
        intrinsics adjustment."""
        w, h = im.size
        nw, nh, left, top = crop_geom(w, h, self.w, self.h)
        im = im.resize((nw, nh), Image.BICUBIC).crop((left, top, left + self.w, top + self.h))
        x = torch.from_numpy(np.asarray(im, dtype=np.float32)) / 127.5 - 1.0
        return x.permute(2, 0, 1)                       # [3,H,W]

    def _load_depth(self, scene_dir, relpath, depth_scale, src_w, src_h):
        """One depth map -> ``[H, W]`` fp32 in the RGB crop frame; zeros mean "no data".

        The crop box comes from the **RGB** source size, not the depth file's own, because
        depth and RGB share an aspect ratio and must share a crop.  Resampling is NEAREST
        so the values stay metric.
        """
        if not relpath:
            return torch.zeros(self.h, self.w, dtype=torch.float32)
        try:
            d16 = np.array(Image.open(os.path.join(scene_dir, relpath)))
            if d16.dtype != np.uint16:
                raise ValueError(f"depth {relpath}: dtype {d16.dtype}, expected uint16")
            d = d16.astype(np.float32) / 65535.0 * float(depth_scale)
        except Exception:
            return torch.zeros(self.h, self.w, dtype=torch.float32)   # masked, not fatal
        nw, nh, left, top = crop_geom(src_w, src_h, self.w, self.h)
        im = Image.fromarray(d, mode="F").resize((nw, nh), Image.NEAREST)
        im = im.crop((left, top, left + self.w, top + self.h))
        out = torch.from_numpy(np.array(im, dtype=np.float32))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)

    def _build_cams(self, r, idx, src_wh):
        """Selected views -> **un-normalised** cameras ``[len(idx), 11]``.

        Intrinsics are crop-adjusted here; normalisation happens jointly afterwards,
        because the root and the scale must come from the input subset, which this helper
        does not know about.
        """
        w2c, intr = [], []
        for j, (sw, sh) in zip(idx, src_wh):
            m = torch.as_tensor(np.asarray(r["w2c"][j], dtype=np.float32)).reshape(3, 4)
            w2c.append(m)
            intr.append(adjust_intrinsics_crop(*[float(x) for x in r["intrinsics"][j]],
                                               sw, sh, self.w, self.h))
        return build_cameras(torch.stack(w2c, 0),
                             torch.tensor(intr, dtype=torch.float32))

    def _load_scene(self, r):
        scene_dir = os.path.join(self.root, r["root"])
        frames = r["frames"]
        k = int(r.get("k_novel") or self.K)
        idx = self._pick_indices(len(frames), k)
        nov = self._novel_positions(len(idx), k)
        nset = set(nov)
        inp = [p for p in range(len(idx)) if p not in nset]

        views, src_wh = [], []
        for j in idx:
            im = Image.open(os.path.join(scene_dir, frames[j])).convert("RGB")
            src_wh.append(im.size)                      # (w, h) BEFORE resize
            views.append(self._resize_crop(im))
        img_all = torch.stack(views, dim=1)             # [3, V+K, H, W]

        cams_all = self._build_cams(r, idx, src_wh)     # [V+K, 11] un-normalised
        cameras, novel_cameras = normalize_cameras_joint(cams_all[inp], cams_all[nov])
        # Validate before returning: a degenerate pose must not reach the caller.
        if not (torch.isfinite(cameras).all() and torch.isfinite(novel_cameras).all()):
            raise ValueError(f"non-finite cameras for scene {r.get('scene', '?')}")

        # Depth shares the cameras' scale.  Recompute it from the INPUT rows exactly as
        # normalize_cameras_joint does, so the two can never drift apart.
        sel = cams_all[inp].float()
        c2w = torch.eye(4, dtype=torch.float32).repeat(len(inp), 1, 1)
        c2w[:, :3, :3] = quaternion_to_matrix(sel[:, 0:4])
        c2w[:, :3, 3] = sel[:, 4:7]
        rel = (torch.inverse(c2w[:1]) @ c2w)[:, :3, :]
        t_scale = float(rel[:, :3, 3].norm(dim=-1).max()) + 1e-2

        drel = r.get("depth")
        dmaps = torch.stack(
            [self._load_depth(scene_dir, (drel[j] if drel else ""),
                              r.get("depth_scale", 0.0), *src_wh[p])
             for p, j in enumerate(idx)], dim=0) / t_scale          # [V+K, H, W]

        return {
            "image": img_all[:, inp], "cameras": cameras,
            "novel_image": img_all[:, nov], "novel_cameras": novel_cameras,
            "depth": dmaps[inp], "novel_depth": dmaps[nov],
            "target_gather": self._sample_targets(len(inp), len(nov)),
            "text": r["text"], "scene": r.get("scene", "?"),
            "source": r.get("source", "default"),
        }

    def __getitem__(self, i):
        tag = self.src[i]
        same = self._by_src.get(tag)
        last = None
        for k in range(self.max_retries):
            try:
                return self._load_scene(self.rows[i])
            except Exception as e:                      # noqa: BLE001
                last = e
                if k % 8 == 7:
                    print(f"[pixworld] {tag}: {k + 1} consecutive scene failures, "
                          f"last={e!r}", flush=True)
                time.sleep(min(0.05 * 2 ** min(k, 4), 1.0))
                # Re-draw within the SAME source: a uniform re-draw over the pooled rows
                # would quietly pull the realised mixture toward the largest pool.
                if self.is_train and same:
                    i = same[random.randrange(len(same))]
        raise RuntimeError(
            f"MultiViewDataset: too many consecutive failures in source {tag!r}; "
            f"last={last!r}")


# ---------------------------------------------------------------------- batching --
def collate_mv(batch):
    """Stack a list of items.  Every sample in a batch must share the same ``K``."""
    ks = {b["novel_image"].shape[1] for b in batch}
    if len(ks) != 1:
        raise ValueError(
            f"batch mixes novel-view counts {sorted(ks)}; scenes with different k_novel "
            f"cannot be stacked. Use batch size 1, or group scenes by k_novel.")
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "cameras": torch.stack([b["cameras"] for b in batch], dim=0),
        "novel_image": torch.stack([b["novel_image"] for b in batch], dim=0),
        "novel_cameras": torch.stack([b["novel_cameras"] for b in batch], dim=0),
        "depth": torch.stack([b["depth"] for b in batch], dim=0),
        "novel_depth": torch.stack([b["novel_depth"] for b in batch], dim=0),
        "target_gather": torch.stack([b["target_gather"] for b in batch], dim=0),
        "text": [b["text"] for b in batch],
        "scene": [b["scene"] for b in batch],
        "source": [b["source"] for b in batch],
    }


def gather_targets(image, novel_image, cameras, novel_cameras, target_gather):
    """Select the render-target views out of the concatenated input+novel sets.

    Args:
        image: ``[B, 3, V, H, W]``; novel_image: ``[B, 3, K, H, W]``.
        cameras: ``[B, V, 11]``; novel_cameras: ``[B, K, 11]``.
        target_gather: ``[B, T]`` int64 into ``[0, V+K)``.

    Returns:
        ``(target_image [B, 3, T, H, W], target_cameras [B, T, 11])``.
    """
    allimg = torch.cat([image, novel_image], dim=2)          # [B,3,V+K,H,W]
    allcam = torch.cat([cameras, novel_cameras], dim=1)      # [B,V+K,11]
    B, T = target_gather.shape
    gi = target_gather.to(allimg.device).view(B, 1, T, 1, 1).expand(
        B, allimg.shape[1], T, allimg.shape[3], allimg.shape[4])
    gc = target_gather.to(allcam.device).view(B, T, 1).expand(B, T, allcam.shape[2])
    return torch.gather(allimg, 2, gi), torch.gather(allcam, 1, gc)


def source_sampler(dataset, weights=None, num_samples=None, seed=None):
    """A :class:`~torch.utils.data.WeightedRandomSampler` that realises a target source mix.

    Args:
        weights: ``{source_tag: fraction}``.  Tags absent from the dataset are dropped and
            the rest are renormalised.  ``None`` means "proportional to pool size", i.e.
            plain uniform sampling over rows.
    """
    n_by = {}
    for t in dataset.src:
        n_by[t] = n_by.get(t, 0) + 1
    if weights is None:
        weights = {t: n / len(dataset.src) for t, n in n_by.items()}
    have = {t: float(w) for t, w in weights.items() if t in n_by and w > 0}
    if not have:
        raise ValueError(f"no requested source is present; dataset has {sorted(n_by)}, "
                         f"weights name {sorted(weights)}")
    tot = sum(have.values())
    have = {t: w / tot for t, w in have.items()}
    w = torch.tensor([have.get(t, 0.0) / n_by[t] for t in dataset.src], dtype=torch.double)
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(int(seed))
    return torch.utils.data.WeightedRandomSampler(
        w, num_samples=int(num_samples or len(dataset)), replacement=True, generator=gen)


def val_scenes(root, n, v=16, k_novel=16, height=480, width=832, index="index.jsonl",
               sources=None, n_target=16):
    """Load ``n`` deterministic validation scenes, evenly spaced through the val split."""
    ds = MultiViewDataset(root, split="val", v=v, k_novel=k_novel, n_target=n_target,
                          both_ends_prob=0.0, height=height, width=width, index=index,
                          sources=sources, verbose=False)
    L = len(ds)
    if not 0 < n <= L:
        raise ValueError(f"val_scenes: asked for {n} but the val split has {L}")
    return [ds[(k * L) // n] for k in range(n)]
