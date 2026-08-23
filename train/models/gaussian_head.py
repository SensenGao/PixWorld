"""Pixel-aligned 3D Gaussian head.

Turns per-pixel decoder features plus a camera into ``num_points_per_pixel`` Gaussians
per pixel.  Each Gaussian is placed along its own pixel ray, so the head predicts a
*depth* rather than a free-floating position.

Per-point channel order out of ``gs_proj`` is::

    [ sh 27 | uv_offset 2 | depth 1 | opacity 1 | scales 3 | rotations 4 ] = 38

and the head returns them in the renderer's layout ``[xyz | opacity | scales |
rotations | sh]`` (see :mod:`render`).

Notes:

* ``lrs_mul`` is a per-channel learning-rate multiplier folded into the convolution
  weights, so a single optimizer LR can drive channels with very different natural
  scales (the ``uv_offset`` channels move 100x more slowly than the rest).
* ``gs_proj`` ships **zero-initialised** with a hand-chosen bias, so at step 0 every
  Gaussian has depth 1, opacity 0.1, identity rotation, a one-pixel footprint, and
  all-zero spherical harmonics.  Renders start uniform grey; colour is learned entirely
  from the render loss.
* Scales are expressed as a multiple of the **pixel footprint at that depth**
  (``sqrt((fx/w)^2 + (fy/h)^2) * depth``), so a Gaussian's size is resolution- and
  distance-aware instead of being an absolute world length.
* ``log(depth)`` *is* the raw pre-activation logit, because the activation is ``exp``.
  A log-space depth loss can therefore read the returned depth directly.
* Everything runs fp32 with autocast disabled.

Activation checkpointing (``checkpoint=True``) recomputes the head in the backward pass.
``gs_chunk`` splits the leading ``N = B * V`` axis to cap the backward recompute
transient; it is off by default.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.autograd import Function

from geometry.cameras import create_rays

__all__ = [
    "SH_C0",
    "rgb_to_sh0",
    "inverse_sigmoid",
    "trunc_exp",
    "depth_tv",
    "PixelAlignedGaussianHead",
]

SH_C0 = 0.28209479177387814


def rgb_to_sh0(rgb01):
    """``[0, 1]`` RGB -> the degree-0 SH coefficient that renders back to that colour
    (gsplat adds the ``+0.5`` offset when it evaluates the SH)."""
    return (rgb01 - 0.5) / SH_C0


def inverse_sigmoid(x):
    return math.log(x / (1 - x))


class _TruncExp(Function):
    """``exp`` with the backward pass clamped to ``exp(+-10)``.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-10, 10))


trunc_exp = _TruncExp.apply


def depth_tv(depths):
    """Per-image max-normalised total variation of a rendered depth map.

    This is a piecewise-smoothness **prior**, not supervision.  ``depths`` is
    ``[N, 1, h, w]``.
    """
    depths = depths / (depths.flatten(1, -1).max(dim=-1)[0][
        :, None, None, None].detach() + 1e-3)
    depths_dx = depths.diff(dim=-1)
    depths_dy = depths.diff(dim=-2)
    return depths_dx.abs().mean() + depths_dy.abs().mean()


class PixelAlignedGaussianHead(nn.Module):
    """``(features, cameras)`` -> ``num_points_per_pixel`` Gaussians per pixel.

    Args:
        feat_ch: channel count of the incoming decoder features.
        sh_degree: spherical-harmonic degree; 2 gives the 27 SH channels the renderer
            expects.
        num_points_per_pixel: Gaussians emitted per pixel (``P``).
        scale_range: ``(lo, hi)`` multiples of the pixel footprint a Gaussian may take.
        depth_max: upper bound on the predicted depth; ``0`` leaves ``exp`` unbounded.
            Implemented as ``exp(min(logit, log(depth_max)))``, so the gradient is the
            uncapped one below the cap and zero above it.
        checkpoint: recompute the head in the backward pass instead of retaining its
            activations.  Bitwise identical.  Ignored when grad is disabled.
        gs_chunk: split the leading ``N`` axis into chunks of this many views while
            checkpointing, to cap the recompute transient.  ``<= 0`` means one chunk.
            **Not** bitwise -- see the module docstring.

    Forward:
        ``feats [N, C, h, w]``, ``cameras [N, 11]`` -> ``gaussians [N, P*h*w, 38]``
        fp32, plus ``depth [N, P, h, w]`` fp32 when ``return_depth=True``.  Nothing is
        detached inside.
    """

    def __init__(self, feat_ch=64, sh_degree=2, num_points_per_pixel=2,
                 scale_range=(0, 16), checkpoint=False, gs_chunk=0, depth_max=0.0):
        super().__init__()
        self.depth_max = float(depth_max)
        self._logit_max = math.log(self.depth_max) if self.depth_max > 0 else None
        self.sh_degree = sh_degree
        self.num_points_per_pixel = num_points_per_pixel
        self.scale_range = scale_range
        self.gaussian_channels = [3 * (sh_degree + 1) ** 2, 2, 1, 1, 3, 4]
        self.checkpoint = bool(checkpoint)
        self.gs_chunk = int(gs_chunk)

        self.adapter = nn.Sequential(
            nn.Conv2d(feat_ch, 128, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.SiLU(),
        )
        self.gs_proj = nn.Conv2d(
            128, num_points_per_pixel * sum(self.gaussian_channels), 3, 1, 1)

        lrs_mul = torch.Tensor(
            [1] * 3 +                                  # sh0
            [0.5] * 3 * ((sh_degree + 1) ** 2 - 1) +   # remaining sh
            [0.01] * 2 +                               # uv_offset
            [1] * 1 +                                  # depth
            [1] * 1 +                                  # opacity
            [1] * 3 +                                  # scales
            [1] * 4                                    # rotations
        ).repeat(num_points_per_pixel)
        self.register_buffer("lrs_mul", lrs_mul / lrs_mul.max(), persistent=True)

        with torch.no_grad():
            self.gs_proj.weight.data.zero_()
            self.gs_proj.bias = nn.Parameter(torch.Tensor(
                [0.0] * 3 * (sh_degree + 1) ** 2 +   # sh: zero -> grey at step 0
                [0.0] * 2 +                          # uv_offset
                [math.log(1)] * 1 +                  # depth -> 1
                [inverse_sigmoid(0.1)] * 1 +         # opacity -> 0.1
                [inverse_sigmoid((1 - scale_range[0])
                                 / (scale_range[1] - scale_range[0]))] * 3 +
                [1.0, 0.0, 0.0, 0.0]                 # identity rotation
            ).repeat(num_points_per_pixel) / self.lrs_mul)

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _head(self, feats, cameras, return_depth=False):
        feats = feats.to(torch.float32)
        cameras = cameras.to(torch.float32)
        N, _, h, w = feats.shape
        P = self.num_points_per_pixel

        x = self.adapter(feats)
        local = F.conv2d(x, self.gs_proj.weight * self.lrs_mul[:, None, None, None],
                         self.gs_proj.bias * self.lrs_mul, stride=1, padding=1)
        local = local.unflatten(1, (P, -1)).permute(0, 1, 3, 4, 2)   # [N,P,h,w,38]

        features, uv_offset, depth, opacity, scales, rotations = local.split(
            self.gaussian_channels, dim=-1)

        rays_o, rays_d = create_rays(
            cameras[:, None].repeat(1, P, 1), h, w, uv_offset=uv_offset)

        if self._logit_max is not None:
            depth = depth.clamp(max=self._logit_max)
        depth = trunc_exp(depth)
        xyz = rays_o + depth * rays_d
        opacity = torch.sigmoid(opacity)

        fx, fy = cameras[:, 7:9].split([1, 1], dim=-1)
        fx, fy = fx / w, fy / h
        pixel_size = torch.sqrt(fx.pow(2) + fy.pow(2))[:, None, None, None] * depth
        scales = (torch.sigmoid(scales)
                  * (self.scale_range[1] - self.scale_range[0])
                  + self.scale_range[0]) * pixel_size
        rotations = F.normalize(rotations, dim=-1)

        params = torch.cat([xyz, opacity, scales, rotations, features], dim=-1)
        out = params.flatten(1, 3)                                   # [N, P*h*w, 38]
        if return_depth:
            return out, depth.squeeze(-1)                            # [N,P,h,w]
        return out

    def forward(self, feats, cameras, return_depth=False):
        # Under no_grad (sampling, inference) nothing is retained anyway, so skip the
        # recompute entirely.
        if not (self.checkpoint and torch.is_grad_enabled()):
            return self._head(feats, cameras, return_depth=return_depth)

        n = feats.shape[0]
        step = n if self.gs_chunk <= 0 else min(self.gs_chunk, n)
        outs, dpts = [], []
        for i in range(0, n, step):
            f_i, c_i = feats[i:i + step], cameras[i:i + step]

            def _run(f, c):
                # Always ask for depth so the checkpointed segment has one signature;
                # the caller's `return_depth` decides what is handed back below.
                return self._head(f, c, return_depth=True)

            o_i, d_i = torch.utils.checkpoint.checkpoint(
                _run, f_i, c_i, use_reentrant=False)
            outs.append(o_i)
            dpts.append(d_i)

        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        if return_depth:
            dpt = torch.cat(dpts, dim=0) if len(dpts) > 1 else dpts[0]
            return out, dpt
        return out
