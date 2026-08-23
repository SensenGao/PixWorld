"""Differentiable 3D Gaussian rendering (gsplat).

A scene is a single tensor of shape ``[N, 38]`` (or ``[B, N, 38]``) in a fixed layout::

    [ xyz 3 | opacity 1 | scales 3 | rotations 4 | sh 27 ]

with ``sh_degree = 2`` (``(2 + 1) ** 2 * 3 = 27`` coefficients, coefficient-major, and
reshaped to ``(-1, 9, 3)`` at render time).

:func:`render_views` is the entry point used by training and inference: it takes scene
parameters plus cameras in the 11-vector convention of :mod:`cameras` and returns
RGB in ``[0, 1]`` together with alpha-composited depth.

Two conventions are fixed here:

* The OpenGL -> COLMAP axis flip is applied to a **clone** of the caller's ``c2w``.
* ``packed=False`` is passed to gsplat explicitly.  gsplat 1.5 flipped the default to
  ``True``, whose ``backgrounds`` shape assertion rejects a per-view background.  The
  unpacked path produces bit-identical renders.

:class:`GaussianRendererWithCheckpoint` renders view by view under ``no_grad`` in the
forward pass and re-renders per view with grad enabled in the backward pass, so peak
activation memory is ``O(1)`` views instead of ``O(V)``.
"""

import torch
import torch.nn.functional as F

from gsplat import rasterization

from geometry.cameras import quaternion_to_matrix


class GaussianRendererWithCheckpoint(torch.autograd.Function):
    """Gradient-checkpointed multi-view rasterization: forward renders
    view-by-view under no_grad, backward re-renders per view with grad enabled
    and accumulates into the gaussian tensors — O(1 view) of rasterizer
    activations instead of O(V)."""

    @staticmethod
    def render(xyz, feature, scale, rotation, opacity, test_c2w, test_intr,
               W, H, sh_degree, near_plane, far_plane, backgrounds):
        test_w2c = test_c2w.float().inverse().unsqueeze(0)  # (1, 4, 4)
        test_intr_i = torch.zeros(3, 3, device=test_intr.device)
        test_intr_i[0, 0] = test_intr[0]
        test_intr_i[1, 1] = test_intr[1]
        test_intr_i[0, 2] = test_intr[2]
        test_intr_i[1, 2] = test_intr[3]
        test_intr_i[2, 2] = 1
        test_intr_i = test_intr_i.unsqueeze(0)  # (1, 3, 3)
        rendering, alpha, _ = rasterization(
            xyz, rotation, scale, opacity, feature,
            test_w2c, test_intr_i, W, H, sh_degree=sh_degree,
            near_plane=near_plane, far_plane=far_plane,
            render_mode="RGB+D",
            backgrounds=backgrounds[None],
            # packed=False: gsplat 1.5 flipped the default to True, whose
            # backgrounds shape-assert rejects per-view bg
            packed=False,
            rasterize_mode="classic")  # (1, H, W, 4)
        return rendering

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr,
                W, H, sh_degree, near_plane, far_plane, backgrounds):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_c2ws,
                              test_intr, backgrounds)
        ctx.W = W
        ctx.H = H
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        with torch.no_grad():
            V, _ = test_intr.shape
            renderings = torch.zeros(V, H, W, 4, device=xyz.device)
            for iv in range(V):
                renderings[iv:iv + 1] = GaussianRendererWithCheckpoint.render(
                    xyz, feature, scale, rotation, opacity,
                    test_c2ws[iv], test_intr[iv], W, H, sh_degree,
                    near_plane, far_plane, backgrounds[iv])
        renderings = renderings.requires_grad_()
        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        (xyz, feature, scale, rotation, opacity, test_c2ws, test_intr,
         backgrounds) = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        sh_degree = ctx.sh_degree
        near_plane = ctx.near_plane
        far_plane = ctx.far_plane
        with torch.enable_grad():
            V, _ = test_intr.shape
            for iv in range(V):
                rendering = GaussianRendererWithCheckpoint.render(
                    xyz, feature, scale, rotation, opacity,
                    test_c2ws[iv], test_intr[iv], W, H, sh_degree,
                    near_plane, far_plane, backgrounds[iv])
                rendering.backward(grad_output[iv:iv + 1])
        return (xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad,
                None, None, None, None, None, None, None, None)


def gaussian_render(gaussian_params, test_c2ws, test_intr, W, H, near_plane=0.01,
                    far_plane=1000, use_checkpoint=False, sh_degree=2,
                    bg_mode="random"):
    """gaussian_params [B,N,38] tensor (or list of [N,38]) in the fixed render
    layout [xyz|opacity|scales|rotations|sh27]; test_c2ws [B,V,4,4] OpenGL c2w;
    test_intr [B,V,4] PIXEL intrinsics (fx,fy,cx,cy). Returns [B,V,H,W,4]
    stacked (rgb 3 in [0,1] + depth 1), per-view random/white/black background,
    gsplat render_mode='RGB+D'."""
    if not torch.is_grad_enabled():
        use_checkpoint = False

    # opengl2colmap — on a CLONE, never mutate the caller's tensor
    test_c2ws = test_c2ws.clone()
    test_c2ws[:, :, :3, 1:3] *= -1

    device = test_intr.device
    B, V, _ = test_intr.shape

    renderings = []

    for ib in range(B):
        if bg_mode == "random":
            backgrounds = torch.rand(V, 3, device=device)
        elif bg_mode == "white":
            backgrounds = torch.ones(V, 3, device=device)
        elif bg_mode == "black":
            backgrounds = torch.zeros(V, 3, device=device)
        else:
            raise ValueError(f"Invalid background mode: {bg_mode}")

        xyz_i, opacity_i, scale_i, rotation_i, feature_i = \
            gaussian_params[ib].float().split(
                [3, 1, 3, 4, (sh_degree + 1) ** 2 * 3], dim=-1)

        opacity_i = opacity_i.squeeze(-1)
        feature_i = feature_i.reshape(-1, (sh_degree + 1) ** 2, 3)

        if use_checkpoint:
            renderings.append(GaussianRendererWithCheckpoint.apply(
                xyz_i, feature_i, scale_i, rotation_i, opacity_i,
                test_c2ws[ib], test_intr[ib], W, H, sh_degree,
                near_plane, far_plane, backgrounds))
        else:
            rendering = torch.zeros(V, H, W, 4, device=device)
            for iv in range(V):
                rendering[iv:iv + 1] = GaussianRendererWithCheckpoint.render(
                    xyz_i, feature_i, scale_i, rotation_i, opacity_i,
                    test_c2ws[ib][iv], test_intr[ib][iv], W, H, sh_degree,
                    near_plane, far_plane, backgrounds[iv])
            renderings.append(rendering)

    renderings = torch.stack(renderings, dim=0)  # (B, V, H, W, 4)
    return torch.cat([renderings[..., :3].clamp(0, 1), renderings[..., 3:]], dim=-1)


@torch.amp.autocast(device_type="cuda", enabled=False)
def render_views(scene_params, cameras, height, width, bg_mode, use_checkpoint=True):
    """scene_params [B,N,38], cameras [B,K,11] (11-vector: real-first c2w quat,
    t, NORMALISED intrinsics) -> (imgs [B,K,3,H,W] in [0,1],
    depths [B,K,1,H,W]). Builds pixel intrinsics (fx*W, fy*H, cx*W, cy*H);
    fp32, autocast off, near_plane 0.01."""
    cameras = cameras.to(torch.float32)
    B, K, _ = cameras.shape

    test_c2ws = torch.eye(4, device=cameras.device)[None][None].repeat(
        B, K, 1, 1).float()
    test_c2ws[:, :, :3, :3] = quaternion_to_matrix(cameras[:, :, :4])
    test_c2ws[:, :, :3, 3] = cameras[:, :, 4:7]

    fx, fy, cx, cy = cameras[:, :, 7:11].split([1, 1, 1, 1], dim=-1)
    test_intr = torch.cat([fx * width, fy * height, cx * width, cy * height], dim=-1)

    out = gaussian_render(scene_params, test_c2ws, test_intr, width, height,
                          near_plane=0.01, far_plane=1000,
                          use_checkpoint=use_checkpoint, sh_degree=2,
                          bg_mode=bg_mode)  # [B,K,H,W,4]
    imgs = out[..., :3].permute(0, 1, 4, 2, 3).contiguous()
    depths = out[..., 3:].permute(0, 1, 4, 2, 3).contiguous()
    return imgs, depths


def prune_opacity(scene_params, thr=0.01):
    """[N,38] or [1,N,38] -> [M,38]: keep gaussians with (activated) opacity >
    thr. Batch-1 inference helper for scene export/sweep."""
    p = scene_params
    if p.dim() == 3:
        assert p.shape[0] == 1, f"prune_opacity is a batch-1 helper, got {p.shape}"
        p = p[0]
    return p[p[:, 3] > thr]


def _slerp(q0, q1, t):
    """Spherical interpolation of unit real-first quats (sign already aligned);
    lerp fallback when nearly parallel. Returns a unit quat."""
    dot = (q0 * q1).sum().clamp(-1.0, 1.0)
    if dot > 0.9995:
        q = (1.0 - t) * q0 + t * q1
    else:
        omega = torch.acos(dot)
        so = torch.sin(omega)
        q = (torch.sin((1.0 - t) * omega) / so) * q0 + (torch.sin(t * omega) / so) * q1
    return F.normalize(q, dim=-1)


def interp_cameras(cameras, n_between):
    """cameras [V,11] -> [(V-1)*n_between+V, 11]: insert n_between cameras
    between each consecutive pair — slerp on the quat (real-first, sign-aligned
    so interpolation takes the short arc), lerp on translation + intrinsics.
    Original rows pass through verbatim. For sweep rendering."""
    cams = cameras.float()
    V = cams.shape[0]
    if V <= 1:
        return cams.clone()
    rows = []
    for i in range(V - 1):
        rows.append(cams[i])
        q0 = F.normalize(cams[i, :4], dim=-1)
        q1 = F.normalize(cams[i + 1, :4], dim=-1)
        if (q0 * q1).sum() < 0:
            q1 = -q1
        for s in range(1, n_between + 1):
            t = s / float(n_between + 1)
            rest = (1.0 - t) * cams[i, 4:] + t * cams[i + 1, 4:]
            rows.append(torch.cat([_slerp(q0, q1, t), rest]))
    rows.append(cams[V - 1])
    return torch.stack(rows, dim=0)


def resample_cameras(cameras, n_frames, round_trip=False):
    """``cameras`` ``[V, 11]`` -> ``[n_frames, 11]`` sampled evenly along the same path.

    Where :func:`interp_cameras` inserts a fixed number of cameras *between each pair*
    (so the count is ``(V-1)*n_between+V`` and you cannot ask for 81), this resamples the
    path at an arbitrary frame count.  Rotation is slerped and translation/intrinsics are
    lerped between the two bracketing input cameras, so with
    ``n_frames == (V-1)*k+V`` the two agree.

    Args:
        cameras: ``[V, 11]`` the path to follow, in temporal order.
        n_frames: how many frames to emit.  16 input views -> 81 frames is a normal ask.
        round_trip: if ``True`` the path runs ``0 -> V-1 -> 0`` instead of ``0 -> V-1``,
            which returns the camera to where it started so the video loops seamlessly.
            The turnaround and the start are each visited once -- no duplicated frame.

    Returns:
        ``[n_frames, 11]`` on the input device and dtype-promoted to float32.
    """
    cams = cameras.float()
    V = cams.shape[0]
    n = max(int(n_frames), 1)
    if V <= 1:
        return cams[:1].clone().repeat(n, 1)
    if n == 1:
        return cams[:1].clone()
    span = float(V - 1)
    rows = []
    for i in range(n):
        if round_trip:
            # i/n rather than i/(n-1): frame n is where frame 0 is, so leaving it out
            # is what makes the loop close without showing the same frame twice.
            u = (i / n) * 2.0 * span
            if u > span:
                u = 2.0 * span - u
        else:
            u = (i / (n - 1)) * span
        j = int(u)
        if j > V - 2:                    # u == span exactly on the last frame
            j = V - 2
        t = u - j
        q0 = F.normalize(cams[j, :4], dim=-1)
        q1 = F.normalize(cams[j + 1, :4], dim=-1)
        if (q0 * q1).sum() < 0:
            q1 = -q1
        rest = (1.0 - t) * cams[j, 4:] + t * cams[j + 1, 4:]
        rows.append(torch.cat([_slerp(q0, q1, t), rest]))
    return torch.stack(rows, dim=0)


def render_path(scene, cameras, height, width, bg_mode="white", chunk=16):
    """Render a long camera path in chunks, returning ``[T, 3, H, W]`` in ``[0, 1]``.

    :func:`render_views` rasterises every requested view before returning, so an 81-frame
    sweep over a 12M-gaussian scene peaks at 81 full-resolution buffers at once.  This
    splits the path into ``chunk``-sized pieces and moves each to CPU as it lands, which
    keeps the peak flat in the number of frames.  Output is bitwise identical to a single
    :func:`render_views` call -- rasterisation of one view does not depend on the others.
    """
    outs = []
    with torch.no_grad():
        for s in range(0, cameras.shape[0], max(int(chunk), 1)):
            part = cameras[s:s + max(int(chunk), 1)].unsqueeze(0)
            imgs, _ = render_views(scene, part, height, width, bg_mode=bg_mode,
                                   use_checkpoint=False)
            outs.append(imgs[0].cpu())
    return torch.cat(outs, dim=0)


if __name__ == "__main__":  # pragma: no cover - GPU self-test
    assert torch.cuda.is_available(), "the render self-test needs a GPU"
    dev = "cuda"
    torch.manual_seed(0)

    # random gaussians in front of a near-identity OpenGL camera (looks -z)
    N = 4096
    xyz = torch.randn(N, 3, device=dev) * 0.25 + torch.tensor(
        [0.0, 0.0, -1.5], device=dev)
    opacity = torch.rand(N, 1, device=dev) * 0.8 + 0.1
    scales = torch.rand(N, 3, device=dev) * 0.02 + 0.005
    rot = F.normalize(torch.randn(N, 4, device=dev), dim=-1)
    sh = torch.randn(N, 27, device=dev) * 0.3
    params = torch.cat([xyz, opacity, scales, rot, sh], dim=-1)[None]  # [1,N,38]

    cams = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5],
         [0.99939083, 0.0, 0.03489950, 0.0, 0.06, 0.01, 0.03, 1.0, 1.0, 0.5, 0.5]],
        device=dev)[None]  # [1,2,11]
    H_, W_ = 96, 160

    # 1) shapes / finite / rgb in [0,1] / depth positive
    with torch.no_grad():
        imgs, depths = render_views(params, cams, H_, W_, bg_mode="white",
                                    use_checkpoint=False)
    assert imgs.shape == (1, 2, 3, H_, W_) and depths.shape == (1, 2, 1, H_, W_)
    assert torch.isfinite(imgs).all() and torch.isfinite(depths).all()
    assert imgs.min() >= 0 and imgs.max() <= 1, "rgb out of [0,1]"
    assert depths.min() >= 0 and depths.max() > 0.1, "depth not positive"
    print(f"render 1) shapes/finite/range OK  depth max {depths.max().item():.3f} "
          f"(gaussians at ~1.5)")

    # 2) bg modes differ on the same scene
    with torch.no_grad():
        w1, _ = render_views(params, cams, H_, W_, bg_mode="white",
                             use_checkpoint=False)
        b1, _ = render_views(params, cams, H_, W_, bg_mode="black",
                             use_checkpoint=False)
        r1, _ = render_views(params, cams, H_, W_, bg_mode="random",
                             use_checkpoint=False)
    assert (w1 - b1).abs().mean() > 1e-3, "white vs black bg must differ"
    assert (r1 - w1).abs().mean() > 1e-4, "random vs white bg must differ"
    print("render 2) bg modes differ OK")

    # 3) gaussian_render does NOT mutate the caller's c2ws (clone contract)
    c2ws = torch.eye(4, device=dev)[None][None].repeat(1, 2, 1, 1)
    c2ws[0, :, :3, :3] = quaternion_to_matrix(cams[0, :, :4])
    c2ws[0, :, :3, 3] = cams[0, :, 4:7]
    intr_px = torch.cat([cams[..., 7:8] * W_, cams[..., 8:9] * H_,
                         cams[..., 9:10] * W_, cams[..., 10:11] * H_], dim=-1)
    keep = c2ws.clone()
    with torch.no_grad():
        out = gaussian_render(params, c2ws, intr_px, W_, H_, bg_mode="white")
    assert out.shape == (1, 2, H_, W_, 4)
    assert torch.equal(c2ws, keep), "gaussian_render mutated caller c2ws"
    print("render 3) c2ws clone OK")

    # 4) backward through rasterization, both plain and checkpoint paths
    for ck in (False, True):
        p = params.clone().requires_grad_(True)
        imgs, depths = render_views(p, cams, H_, W_, bg_mode="white",
                                    use_checkpoint=ck)
        loss = imgs.mean() + 0.01 * depths.mean()
        loss.backward()
        g = p.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
            f"no grad through rasterization (checkpoint={ck})"
    print("render 4) backward OK (checkpoint False/True)")

    # 5) prune_opacity drops rows
    pp = params[0].clone()
    pp[: N // 2, 3] = 0.001
    kept = prune_opacity(pp, 0.01)
    assert kept.shape == (N - N // 2, 38), f"prune kept {kept.shape}"
    kept2 = prune_opacity(params, thr=2.0)  # batch-1 form, nothing survives
    assert kept2.shape == (0, 38)
    print("render 5) prune_opacity OK")

    # 6) interp_cameras: count, normalized quats, endpoints match inputs
    c3 = torch.stack([cams[0, 0], cams[0, 1],
                      torch.tensor([0.99756405, 0.0, 0.0697565, 0.0,
                                    0.12, 0.02, 0.05, 1.0, 1.0, 0.5, 0.5],
                                   device=dev)]).cpu()
    sw = interp_cameras(c3, 2)
    assert sw.shape == (3 + 2 * 2, 11)
    assert torch.equal(sw[0], c3[0]) and torch.equal(sw[3], c3[1]) \
        and torch.equal(sw[6], c3[2]), "endpoints must match inputs"
    assert (sw[:, :4].norm(dim=-1) - 1).abs().max() < 1e-5, "quats not normalized"
    c3b = c3.clone()
    c3b[1, :4] *= -1  # sign-flip mid quat: slerp must still take the short arc
    sw2 = interp_cameras(c3b, 1)
    assert sw2.shape == (5, 11)
    assert (sw2[:, :4].norm(dim=-1) - 1).abs().max() < 1e-5
    assert torch.isfinite(sw2).all()
    print("render 6) interp_cameras OK")

    # 7) resample_cameras: agrees with interp_cameras wherever the counts coincide,
    #    keeps unit quats, and the round trip returns to where it started.
    c16 = torch.cat([F.normalize(torch.randn(16, 4), dim=-1),
                     torch.randn(16, 3) * 0.3, torch.rand(16, 4)], dim=-1)
    c16[c16[:, 0] < 0] *= -1
    for k in (0, 1, 3, 5):
        n = (16 - 1) * k + 16
        d = (interp_cameras(c16, k) - resample_cameras(c16, n)).abs().max()
        assert d < 1e-5, f"resample != interp at n_between={k}: {d:.2e}"
    r81 = resample_cameras(c16, 81)
    assert r81.shape == (81, 11)
    assert (r81[-1] - c16[-1]).abs().max() < 1e-5, "one-way must end on the last camera"
    rt = resample_cameras(c16, 81, round_trip=True)
    assert rt.shape == (81, 11)
    assert (rt[0] - c16[0]).abs().max() < 1e-5, "round trip must start at camera 0"
    # The exact invariant of an out-and-back path is mirror symmetry in the path
    # parameter: frame i and frame n-i sit at the same point along the polyline.
    n = rt.shape[0]
    mirror = (rt[1:] - rt.flip(0)[:-1]).abs().max()
    assert mirror < 1e-5, f"round trip is not mirror-symmetric: {mirror:.2e}"
    # ...and the turnaround is where the path parameter reaches the LAST input camera,
    # which is the middle frame.
    turn = (rt[:, 4:7] - c16[-1, 4:7]).norm(dim=-1).argmin()
    assert abs(int(turn) - n // 2) <= 1, f"turnaround at {int(turn)}, expected ~{n // 2}"
    assert (rt[:, :4].norm(dim=-1) - 1).abs().max() < 1e-5
    print("render 7) resample_cameras + round trip OK")

    # 8) render_path: chunking must not change a single pixel.
    long_cams = resample_cameras(cams[0].cpu(), 9).to(dev)
    with torch.no_grad():
        whole, _ = render_views(params, long_cams.unsqueeze(0), H_, W_,
                                bg_mode="white", use_checkpoint=False)
    for c in (1, 2, 4, 16):
        piece = render_path(params, long_cams, H_, W_, bg_mode="white", chunk=c)
        d = (piece - whole[0].cpu()).abs().max()
        assert d == 0, f"render_path(chunk={c}) differs by {d:.2e}"
    print("render 8) render_path chunking is exact OK")

    print("render self-test OK")
