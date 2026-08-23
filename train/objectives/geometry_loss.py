"""Geometry perception loss.

Photometric and perceptual losses compare rendered views to ground truth **one image at a
time**.  Neither can tell whether the sixteen views agree about the *same 3D scene* -- a
model that renders each view plausibly but inconsistently scores well on both.

This loss closes that gap by scoring the renders in the feature space of a frozen
**multi-view 3D foundation model**.  Those models attend across views, so their features
encode where surfaces are, not just what each image looks like; matching them pushes the
Gaussian field toward the geometry the foundation model infers from the ground truth.

Two backends are supported, both frozen and both external:

``pi3``
    Pi3 (https://github.com/yyfz/Pi3).  Features are taken from the **decoder**, the
    cross-view stack that its point and camera heads read -- not from its DINOv2 encoder,
    which is per-image appearance.  Its own ``decode()`` only returns the last two blocks,
    so the decoder loop is re-run here to tap four depths instead.

``vggt``
    VGGT (https://github.com/facebookresearch/vggt).  Features come from the
    **aggregator**, again the cross-view part its DPT heads read.

``none``
    Disabled, and this is the **default**.  The loss costs a second ~1B-parameter forward
    with gradients, and neither repository is a dependency of PixWorld -- you have to
    install one and point at its weights before it will run.

Both backends emit 2048-dimensional tokens with five leading special tokens, so the taps
are directly comparable.  Features are L2-normalised per token before the distance, which
makes the loss scale-free across backends -- a weight of 0.05 means the same thing for
either one.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GEOMETRY_BACKENDS", "GEOMETRY_DISTANCES", "GeometryPerceptualLoss",
           "build_geometry_loss"]

GEOMETRY_BACKENDS = ("none", "pi3", "vggt")

#: ``mse`` on raw features (the shipped default, paired with a single tap), or ``cosine``
#: on L2-normalised tokens (what the PixWorld paper specifies).
GEOMETRY_DISTANCES = ("cosine", "mse")

_PATCH = 14                 # both backbones use a 14-pixel patch


def _round_to_patch(v, patch=_PATCH, minimum=2):
    return max(minimum, int(round(v / patch))) * patch


class GeometryPerceptualLoss(nn.Module):
    """Feature-space distance between rendered and ground-truth views.

    Args:
        backend: ``"none"``, ``"pi3"`` or ``"vggt"``.
        weights: path to the backbone weights.  For ``pi3`` a ``.safetensors`` or ``.pt``
            state dict; for ``vggt`` either a local file or a HuggingFace repo id
            (``facebook/VGGT-1B``).
        repo: path to the backbone's source checkout, prepended to ``sys.path``.  Not
            needed if the package is already importable.
        resolution: short-side resolution the views are resized to before the backbone.
            Rounded to a multiple of 14.  224 keeps the loss affordable; the backbones
            were trained nearer 518, so raising this makes the features sharper and the
            step much more expensive.
        n_taps: how many depths of the backbone to compare.  The default is **1** -- the
            backbone's final block, i.e. its full geometric representation with the
            decoding stage omitted.  Both backends can expose more (VGGT's aggregator
            returns one per alternating-attention block; the Pi3 decoder is tapped at
            evenly spaced cross-view blocks), but more than one tap only makes sense with
            ``distance="cosine"`` -- see :data:`GEOMETRY_DISTANCES`.
        max_views: cap on how many views are fed to the backbone.  Views are subsampled
            evenly, which keeps the cross-view structure while cutting the cost.  ``0``
            uses all of them.
        distance: ``"mse"`` on raw features (the default) or ``"cosine"`` on L2-normalised
            tokens, which is what the PixWorld paper specifies.  See
            :data:`GEOMETRY_DISTANCES`.
        dtype: the backbone runs in this dtype.

    Forward:
        ``render01``, ``gt01``: ``[B, V, 3, H, W]`` in ``[0, 1]``.
        Returns ``(loss, log_dict)``.  With ``backend="none"`` the loss is an exact zero
        that still carries a gradient path, so the caller needs no special case.
    """

    def __init__(self, backend="none", weights="", repo="", resolution=224, n_taps=1,
                 max_views=0, distance="mse", dtype=torch.bfloat16, device="cuda",
                 verbose=print):
        super().__init__()
        if backend not in GEOMETRY_BACKENDS:
            raise ValueError(f"geometry backend {backend!r} not in {GEOMETRY_BACKENDS}")
        if distance not in GEOMETRY_DISTANCES:
            raise ValueError(f"geometry distance {distance!r} not in {GEOMETRY_DISTANCES}")
        self.backend = backend
        self.distance = distance
        self.resolution = int(resolution)
        self.n_taps = int(n_taps)
        self.max_views = int(max_views)
        self.dtype = dtype
        self.model = None
        if backend == "none":
            return
        # Build the backbone inside a forked RNG: constructing a ~1B-parameter model
        # would otherwise shift every subsequent random draw in training.
        with torch.random.fork_rng(devices=[] if device == "cpu" else [device]):
            self.model = (_load_pi3 if backend == "pi3" else _load_vggt)(
                weights, repo, device, dtype, verbose)
        self.model.eval().requires_grad_(False)
        n = sum(p.numel() for p in self.model.parameters()) / 1e6
        if verbose:
            verbose(f"[pixworld] geometry perception loss: {backend} "
                    f"({n:.0f}M params, frozen, {dtype}), {distance} distance, "
                    f"short side {self.resolution}, {self.n_taps} tap(s)"
                    + (f", at most {self.max_views} views" if self.max_views else ""))
            if distance == "mse" and self.n_taps > 1:
                verbose(f"[pixworld] NOTE: an mse distance over {self.n_taps} taps is "
                        "dominated by the deepest one through magnitude alone (token "
                        "norms grow with depth). Consider --geo_taps 1, or use the "
                        "cosine distance.")

    @property
    def enabled(self):
        return self.model is not None

    def _prepare(self, x):
        """``[B, V, 3, H, W]`` in ``[0, 1]`` -> resized to a multiple of 14, in ``dtype``."""
        B, V, C, H, W = x.shape
        short, long = (H, W) if H <= W else (W, H)
        s = self.resolution / short
        h = _round_to_patch(H * s)
        w = _round_to_patch(W * s)
        y = F.interpolate(x.reshape(B * V, C, H, W).float(), size=(h, w),
                          mode="bilinear", align_corners=False, antialias=True)
        return y.reshape(B, V, C, h, w).to(self.dtype)

    def _subsample(self, x):
        V = x.shape[1]
        if not self.max_views or V <= self.max_views:
            return x
        idx = torch.linspace(0, V - 1, self.max_views).round().long().to(x.device)
        return x.index_select(1, idx)

    def _features(self, x):
        """``[B, V, 3, h, w]`` -> a list of ``[N, tokens, C]`` patch-token tensors."""
        if self.backend == "pi3":
            return _pi3_features(self.model, x, self.n_taps)
        return _vggt_features(self.model, x, self.n_taps)

    def forward(self, render01, gt01):
        if self.model is None:
            # An exact zero that still depends on the input, so the caller can add it
            # unconditionally without changing the graph in the disabled case.
            return render01.sum() * 0.0, {}
        r = self._subsample(self._prepare(render01))
        g = self._subsample(self._prepare(gt01))
        f_r = self._features(r)
        with torch.no_grad():
            f_g = self._features(g)
        loss = render01.new_zeros(())
        per_tap = []
        for a, b in zip(f_r, f_g):
            if self.distance == "cosine":
                # 0 for identical features, 1 for orthogonal, 2 for opposed -- and
                # independent of the channel count, so one weight fits either backbone.
                a = F.normalize(a.float(), dim=-1)
                b = F.normalize(b.float().detach(), dim=-1)
                d = (1.0 - (a * b).sum(dim=-1)).mean()
            else:
                d = F.mse_loss(a.float(), b.float().detach())
            per_tap.append(d.detach())
            loss = loss + d
        loss = loss / max(len(f_r), 1)
        log = {"geo": loss.detach()}
        for i, d in enumerate(per_tap):
            log[f"geo_tap{i}"] = d
        return loss, log


# ------------------------------------------------------------------- backends --
def _add_repo(repo):
    import sys
    if repo and repo not in sys.path:
        if not os.path.isdir(repo):
            raise FileNotFoundError(f"geometry backbone source directory not found: {repo}")
        sys.path.insert(0, repo)


def _load_state(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict"):
        if isinstance(ck, dict) and k in ck:
            return ck[k]
    return ck


def _load_pi3(weights, repo, device, dtype, verbose):
    _add_repo(repo)
    try:
        from pi3.models.pi3 import Pi3
    except ImportError as e:
        raise ImportError(
            "the pi3 geometry backend needs the Pi3 source on the path. Clone "
            "https://github.com/yyfz/Pi3 and pass --geo_repo /path/to/Pi3 (or install "
            "it), then --geo_weights /path/to/model.safetensors."
        ) from e
    m = Pi3()
    if not weights:
        raise ValueError("--geo_weights is required for the pi3 backend "
                         "(its model.safetensors)")
    missing, unexpected = m.load_state_dict(_load_state(weights), strict=False)
    if verbose:
        verbose(f"[pixworld] Pi3 weights: {len(missing)} missing, "
                f"{len(unexpected)} unexpected")
    if len(missing) > 20:
        raise RuntimeError(
            f"{weights} left {len(missing)} Pi3 tensors uninitialised (first "
            f"{missing[:4]}). Refusing to score geometry against a randomly initialised "
            f"backbone -- it would look like a working loss and teach nothing.")
    return m.to(device).to(dtype)


def _load_vggt(weights, repo, device, dtype, verbose):
    _add_repo(repo)
    try:
        from vggt.models.aggregator import Aggregator
    except ImportError as e:
        raise ImportError(
            "the vggt geometry backend needs the VGGT source on the path. Clone "
            "https://github.com/facebookresearch/vggt and pass --geo_repo /path/to/vggt "
            "(or install it), then --geo_weights facebook/VGGT-1B or a local checkpoint."
        ) from e
    m = Aggregator(img_size=518, patch_size=_PATCH, embed_dim=1024)
    if not weights:
        raise ValueError("--geo_weights is required for the vggt backend "
                         "(facebook/VGGT-1B, or a local checkpoint)")
    if os.path.isfile(weights):
        sd = _load_state(weights)
    elif os.sep in weights or weights.endswith((".safetensors", ".pt", ".pth", ".bin")):
        # It looks like a path, so say so plainly rather than letting the HuggingFace
        # client reject it as a malformed repo id.
        raise FileNotFoundError(f"geometry backbone weights not found: {weights}")
    else:                                       # a HuggingFace repo id
        from huggingface_hub import hf_hub_download
        sd = _load_state(hf_hub_download(weights, "model.safetensors"))
    # The published checkpoint is the whole VGGT model; only the aggregator is needed.
    sd = {k[len("aggregator."):]: v for k, v in sd.items() if k.startswith("aggregator.")} \
        or sd
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if verbose:
        verbose(f"[pixworld] VGGT aggregator weights: {len(missing)} missing, "
                f"{len(unexpected)} unexpected")
    if len(missing) > 20:
        raise RuntimeError(
            f"{weights} left {len(missing)} VGGT tensors uninitialised (first "
            f"{missing[:4]}). Refusing to score geometry against a randomly initialised "
            f"backbone.")
    return m.to(device).to(dtype)


def _even_taps(n_blocks, n_taps, cross_view_only=False):
    """``n_taps`` evenly spaced block indices, shallow to deep, always including the last.

    With ``cross_view_only`` each index is snapped forward onto an odd block.  That is for
    Pi3, whose decoder alternates per-view (even) and cross-view (odd) attention.
    """
    n_taps = max(1, min(int(n_taps), n_blocks // (2 if cross_view_only else 1)))
    out = []
    for k in range(1, n_taps + 1):
        i = round(k * n_blocks / n_taps) - 1
        if cross_view_only and i % 2 == 0:
            i = min(i + 1, n_blocks - 1)
        if not out or i > out[-1]:
            out.append(i)
    return out


def _pi3_tap_indices(n_blocks, n_taps):
    """Evenly spaced cross-view blocks of the Pi3 decoder, shallow to deep.

    The decoder alternates per-view (even index) and cross-view (odd index) attention, so
    every tap is snapped to a cross-view block.  For the shipped 36-block decoder and four
    taps this gives blocks 9, 17, 27 and 35.
    """
    return _even_taps(n_blocks, n_taps, cross_view_only=True)


def _pi3_features(model, x, n_taps=4):
    """Pi3's cross-view decoder tokens, at ``n_taps`` depths.

    Its encoder is DINOv2 -- per-image appearance.  The decoder is where views talk to
    each other and it is what the point and camera heads read, so that is the tap.

    Pi3's own ``decode()`` hard-codes a single tap (it concatenates the final two blocks
    and returns that), so the loop is reproduced here with the tap points opened up.  The
    body below mirrors ``Pi3.decode`` exactly -- register tokens, the rotary position
    offset for the special tokens, and the per-view/cross-view reshape alternation.
    """
    B, V, C, H, W = x.shape
    xn = (x - model.image_mean.to(x.dtype)) / model.image_std.to(x.dtype)
    hidden = model.encoder(xn.reshape(B * V, C, H, W), is_training=True)
    if isinstance(hidden, dict):
        hidden = hidden["x_norm_patchtokens"]

    BV = B * V
    hidden = hidden.reshape(BV, hidden.shape[1], -1)
    reg = model.register_token.repeat(B, V, 1, 1).reshape(
        BV, *model.register_token.shape[-2:])
    hidden = torch.cat([reg, hidden], dim=1)
    hw = hidden.shape[1]

    pos = model.position_getter(BV, H // model.patch_size, W // model.patch_size,
                                hidden.device)
    if model.patch_start_idx > 0:
        # The special tokens get no rotary position, exactly as Pi3 does it.
        pos = pos + 1
        pos_special = torch.zeros(BV, model.patch_start_idx, 2,
                                  device=hidden.device, dtype=pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)

    n_blocks = len(model.decoder)
    taps = set(_pi3_tap_indices(n_blocks, n_taps))
    out = []
    for i in range(n_blocks):
        if i % 2 == 0:                                  # per-view attention
            pos = pos.reshape(BV, hw, -1)
            hidden = hidden.reshape(BV, hw, -1)
        else:                                           # cross-view attention
            pos = pos.reshape(B, V * hw, -1)
            hidden = hidden.reshape(B, V * hw, -1)
        hidden = model.decoder[i](hidden, xpos=pos)
        if i in taps:
            out.append(hidden.reshape(BV, hw, -1)[:, model.patch_start_idx:])
    return out


def _vggt_features(model, x, n_taps=4):
    """VGGT's aggregator tokens, at ``n_taps`` depths.

    """
    import importlib
    mod = importlib.import_module(type(model).__module__)

    B, S, C_in, H, W = x.shape
    images = (x - model._resnet_mean.to(x.dtype)) / model._resnet_std.to(x.dtype)
    patch_tokens = model.patch_embed(images.view(B * S, C_in, H, W))
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]

    tokens = torch.cat([mod.slice_expand_and_flatten(model.camera_token, B, S),
                        mod.slice_expand_and_flatten(model.register_token, B, S),
                        patch_tokens], dim=1)
    _, P, C = tokens.shape

    pos = None
    if getattr(model, "rope", None) is not None:
        pos = model.position_getter(B * S, H // model.patch_size, W // model.patch_size,
                                    device=tokens.device)
        if model.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, model.patch_start_idx, 2,
                                      device=tokens.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

    taps = set(_even_taps(model.aa_block_num, n_taps))
    frame_idx = global_idx = 0
    out = []
    for blk in range(model.aa_block_num):
        for attn_type in model.aa_order:
            if attn_type == "frame":
                tokens, frame_idx, _ = model._process_frame_attention(
                    tokens, B, S, P, C, frame_idx, pos=pos)
            elif attn_type == "global":
                tokens, global_idx, _ = model._process_global_attention(
                    tokens, B, S, P, C, global_idx, pos=pos)
            else:
                raise ValueError(f"unknown attention type {attn_type!r}")
        if blk in taps:
            out.append(tokens.view(B, S, P, C).flatten(0, 1)[:, model.patch_start_idx:])
    return out


def build_geometry_loss(backend="none", **kw):
    """Convenience constructor; returns a disabled loss for ``backend="none"``."""
    return GeometryPerceptualLoss(backend=backend, **kw)


if __name__ == "__main__":  # pragma: no cover - self-test (CPU, no backbone needed)
    # Tap selection: evenly spaced, deepest last, and for Pi3 always on a cross-view block.
    for n, k in ((36, 4), (36, 2), (36, 1), (24, 4)):
        t = _pi3_tap_indices(n, k)
        assert len(t) == k and t == sorted(t) and t[-1] == n - 1, (n, k, t)
        assert all(i % 2 == 1 for i in t), (n, k, t)
        print(f"  pi3  {n:2d} blocks, {k} taps -> {t}")
    for n, k in ((24, 4), (24, 2), (12, 4)):
        t = _even_taps(n, k)
        assert len(t) == k and t == sorted(t) and t[-1] == n - 1, (n, k, t)
        print(f"  vggt {n:2d} blocks, {k} taps -> {t}")

    # Disabled backend: an exact zero that still carries a gradient path, so the trainer
    # needs no special case.
    g = GeometryPerceptualLoss("none")
    assert not g.enabled
    r = torch.rand(1, 4, 3, 64, 96, requires_grad=True)
    loss, log = g(r, torch.rand(1, 4, 3, 64, 96))
    assert float(loss) == 0.0 and log == {}
    loss.backward()
    assert r.grad is not None and float(r.grad.abs().sum()) == 0.0
    print("  disabled backend -> exact zero, zero gradient, no special case needed")

    # Resizing snaps to the patch grid while preserving aspect.
    g.resolution = 224
    for h, w in ((480, 832), (512, 512), (360, 640)):
        y = g._prepare(torch.rand(1, 2, 3, h, w))
        assert y.shape[-2] % _PATCH == 0 and y.shape[-1] % _PATCH == 0, y.shape
        assert min(y.shape[-2:]) == 224, y.shape
        print(f"  {h}x{w} -> {tuple(y.shape[-2:])} (both multiples of {_PATCH})")

    try:
        GeometryPerceptualLoss("nope")
        raise SystemExit("an unknown backend must be rejected")
    except ValueError:
        print("  unknown backend rejected")
    print("geometry_loss self-test OK")
