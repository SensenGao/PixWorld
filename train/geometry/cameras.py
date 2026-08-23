"""Camera conventions, ray maps and pose normalisation.

Every camera is an 11-vector::

    [qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy]

* ``q`` is a **real-first** quaternion of the **camera-to-world** rotation expressed in
  **OpenGL** coordinates (x right, y up, camera looks down -z).
* ``t`` is the camera-to-world translation (the camera centre in world coordinates).
* ``fx, fy, cx, cy`` are intrinsics **normalised by the image size** (``fx / W``,
  ``fy / H``, ``cx / W``, ``cy / H``), already adjusted for any resize/crop.

The network never sees the 11-vector directly.  It sees a **ray map**: a 6-channel
per-pixel image ``[rays_d | rays_o - (rays_o . rays_d) rays_d]``.  The second triple is
the component of the ray origin perpendicular to the ray direction, which is a
Pluecker-style encoding: sliding the camera centre along its own ray leaves the map
unchanged, so the network sees the *pose* rather than an arbitrary choice of origin.

All functions are pure ``torch`` and run on CPU or GPU.  Outputs are fp32.
"""
import torch
import torch.nn.functional as F

__all__ = [
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "standardize_quaternion",
    "build_cameras",
    "normalize_cameras",
    "normalize_cameras_joint",
    "create_rays",
    "create_raymaps",
    "crop_geom",
    "adjust_intrinsics_crop",
]


# --------------------------------------------------------------------- quaternions --
def quaternion_to_matrix(quaternions):
    """Real-first quaternion ``[..., 4]`` -> rotation matrix ``[..., 3, 3]``.

    The ``2 / |q|^2`` scaling makes this robust to un-normalised inputs.
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def standardize_quaternion(quaternions):
    """Flip the sign so the real part is non-negative (``q`` and ``-q`` are the same
    rotation)."""
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x):
    """``sqrt(max(0, x))`` with a zero subgradient at ``x == 0``."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion(matrix):
    """Rotation matrix ``[..., 3, 3]`` -> real-first quaternion ``[..., 4]``.

    Computes all four ``r/i/j/k``-scaled candidates and gathers the best-conditioned one,
    so it stays stable for every rotation including ``trace ~ -1``.
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    # Floor at 0.1: if q_abs is small that candidate will not be picked anyway.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    indices = q_abs.argmax(dim=-1, keepdim=True)
    gather_indices = indices.unsqueeze(-1).expand(list(batch_dim) + [1, 4])
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)


# ------------------------------------------------------------------ building poses --
def build_cameras(w2c, intr_norm):
    """World-to-camera matrices + normalised intrinsics -> cameras ``[V, 11]``.

    Args:
        w2c: ``[V, 3, 4]`` row-major **OpenCV** world-to-camera matrices.
        intr_norm: ``[V, 4]`` normalised ``(fx, fy, cx, cy)``.

    ``c2w = inv(w2c)``, then the y/z **columns** are flipped (an OpenCV -> OpenGL change
    of the *camera* basis; the world frame is untouched, so cross-view consistency
    survives).  The inverse runs in fp64.
    """
    w2c = torch.as_tensor(w2c, dtype=torch.float64)
    V = w2c.shape[0]
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(V, 1, 4)
    c2w = torch.inverse(torch.cat([w2c, bottom], dim=1))[:, :3, :].clone()
    c2w[:, :3, 1:3] *= -1
    q = matrix_to_quaternion(c2w[:, :3, :3])
    intr = torch.as_tensor(intr_norm, dtype=torch.float64).reshape(V, 4)
    return torch.cat([q, c2w[:, :3, 3], intr], dim=-1).float()


def normalize_cameras(cameras):
    """``[V, 11]`` -> ``[V, 11]``: re-root every pose to frame 0 (so ``c2w_0 = I``) and
    divide all translations by ``max_i ||t_i|| + 1e-2``.

    This removes the arbitrary world frame and the arbitrary scene scale, so ray maps
    live in one bounded distribution across scenes.
    """
    V = cameras.shape[0]
    dev = cameras.device
    c2w = torch.eye(4, dtype=torch.float32, device=dev).repeat(V, 1, 1)
    c2w[:, :3, :3] = quaternion_to_matrix(cameras[:, 0:4].float())
    c2w[:, :3, 3] = cameras[:, 4:7].float()
    rel = (torch.inverse(c2w[:1]) @ c2w)[:, :3, :]
    T_norm = rel[:, :3, 3].norm(dim=-1).max()
    t = rel[:, :3, 3] / (T_norm + 1e-2)
    q = matrix_to_quaternion(rel[:, :3, :3])
    return torch.cat([q, t, cameras[:, 7:].float()], dim=-1)


def cameras_are_normalised(cameras, tol=1e-3):
    """True if ``cameras`` already look like :func:`normalize_cameras` output.

    The discriminating invariant is **frame 0 is the identity pose**.  Raw world-frame or
    COLMAP poses essentially never satisfy it; anything that has been through
    :func:`normalize_cameras` always does.

    The largest translation is not required to equal 1.  Normalisation divides by
    ``max|t| + 1e-2``, so the result is ``T / (T + 0.01)``; only a loose upper bound is
    asserted.
    """
    c = cameras.float()
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=c.device)
    if (c[0, :7].abs() - ident.abs()).abs().max() > tol:
        return False                      # frame 0 is not the identity -> world frame
    return float(c[:, 4:7].norm(dim=-1).max()) <= 1.0 + 1e-2


def load_cameras_json(path, normalize="auto", verbose=print):
    """Read a camera path written by :func:`save_cameras_json` -> ``[V, 11]``.

    Args:
        path: a JSON file holding ``{"cameras": [[11 floats], ...]}`` or a bare ``[V, 11]``
            list.  The 11-vector is ``[qw qx qy qz | tx ty tz | fx fy cx cy]``, with the
            quaternion real-first and the intrinsics normalised by image width/height.
        normalize: ``"auto"`` normalises only if the file is not already normalised (the
            model is trained on normalised cameras and cannot use raw world poses);
            ``"always"`` / ``"never"`` force it.

    Returns:
        ``[V, 11]`` float32 on CPU.
    """
    import json
    with open(path) as f:
        blob = json.load(f)
    rows = blob["cameras"] if isinstance(blob, dict) else blob
    cams = torch.tensor(rows, dtype=torch.float32)
    if cams.dim() != 2 or cams.shape[1] != 11:
        raise ValueError(f"{path}: expected [V, 11] cameras, got {tuple(cams.shape)}")
    already = cameras_are_normalised(cams)
    do = {"always": True, "never": False}.get(normalize, not already)
    if do:
        cams = normalize_cameras(cams)
    verbose(f"[pixworld] {cams.shape[0]} cameras from {path} "
            f"({'already normalised' if already else 'raw'}"
            f"{', normalised on load' if do else ''})")
    return cams


def save_cameras_json(cameras, path, extra=None):
    """Write ``[V, 11]`` cameras as JSON, in the form :func:`load_cameras_json` reads."""
    import json
    import os
    blob = {"format": "pixworld-cameras-v1",
            "layout": "qw qx qy qz | tx ty tz | fx fy cx cy",
            "normalised": bool(cameras_are_normalised(cameras)),
            "cameras": [[round(float(v), 8) for v in row] for row in cameras]}
    if extra:
        blob.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(blob, f, indent=1)
    return path


def normalize_cameras_joint(cameras, novel_cameras):
    """Normalise input and held-out novel cameras **together**.

    Args:
        cameras: ``[V, 11]`` input views.
        novel_cameras: ``[K, 11]`` held-out novel views.

    Returns:
        ``(input_n [V, 11], novel_n [K, 11])``.

    Every pose -- input *and* novel -- is re-rooted to **input frame 0**, and every
    translation is divided by the max over the **input** translations only.  The input
    rows therefore come out bitwise-equal to ``normalize_cameras(cameras)`` while the
    novel views ride the same rigid+scale transform.  Novel ``|t|`` may exceed 1; that is
    expected and fine.
    """
    V = cameras.shape[0]
    dev = cameras.device
    allc = torch.cat([cameras, novel_cameras], dim=0).float()
    N = allc.shape[0]
    c2w = torch.eye(4, dtype=torch.float32, device=dev).repeat(N, 1, 1)
    c2w[:, :3, :3] = quaternion_to_matrix(allc[:, 0:4])
    c2w[:, :3, 3] = allc[:, 4:7]
    rel = (torch.inverse(c2w[:1]) @ c2w)[:, :3, :]      # root = INPUT frame 0
    T_norm = rel[:V, :3, 3].norm(dim=-1).max()          # scale from INPUT frames only
    t = rel[:, :3, 3] / (T_norm + 1e-2)
    q = matrix_to_quaternion(rel[:, :3, :3])
    out = torch.cat([q, t, allc[:, 7:]], dim=-1)
    return out[:V], out[V:]


# ------------------------------------------------------------------------- ray maps --
def create_rays(cameras, h, w, uv_offset=None):
    """cameras ``[..., 11]`` -> ``(rays_o, rays_d)``, each ``[..., h, w, 3]``.

    Rays are in the world frame, ``rays_d`` is unit length, both are fp32.  The pinhole
    model is OpenGL at pixel centres::

        x = (i + 0.5 - cx) / fx      y = -(j + 0.5 - cy) / fy      z = -1

    with ``fx, cx`` scaled by ``w`` and ``fy, cy`` by ``h``.

    Args:
        uv_offset: optional ``[..., h, w, 2]`` per-pixel offsets **in pixels**, added to
            ``(i + 0.5, j + 0.5)`` *before* the intrinsics; ``+u`` right, ``+v`` down.
            ``None`` is exactly equivalent to zeros.  The Gaussian head uses this to give
            each splat a sub-pixel position.
    """
    prefix_shape = cameras.shape[:-1]
    cams = cameras.reshape(-1, 11).float()
    N = cams.shape[0]
    R = quaternion_to_matrix(cams[:, :4])
    fx, fy, cx, cy = cams[:, 7:].chunk(4, -1)
    fx, cx = fx * w, cx * w
    fy, cy = fy * h, cy * h
    inds = torch.arange(0, h * w, device=cams.device).expand(N, h * w)
    i = inds % w + 0.5
    j = torch.div(inds, w, rounding_mode="floor") + 0.5
    if uv_offset is not None:
        uv = uv_offset.reshape(N, h * w, 2).float()
        i = i + uv[..., 0]
        j = j + uv[..., 1]
    xs = (i - cx) / fx
    ys = -(j - cy) / fy
    zs = -torch.ones_like(xs)
    directions = torch.stack((xs, ys, zs), dim=-1)
    rays_d = F.normalize(directions @ R.transpose(-1, -2), dim=-1)
    rays_o = cams[:, 4:7][:, None, :].expand_as(rays_d)
    rays_o = rays_o.reshape(*prefix_shape, h, w, 3)
    rays_d = rays_d.reshape(*prefix_shape, h, w, 3)
    return rays_o, rays_d


def create_raymaps(cameras, h, w):
    """cameras ``[V, 11]`` -> ray maps ``[V, 6, h, w]`` fp32.

    Channels are ``[rays_d | rays_o - (rays_o . rays_d) rays_d]``, channels-first.  The
    map is produced at full image resolution; the patch embedding pools it downstream.

    """
    rays_o, rays_d = create_rays(cameras, h, w)
    o_perp = rays_o - (rays_o * rays_d).sum(dim=-1, keepdim=True) * rays_d
    return torch.cat([rays_d, o_perp], dim=-1).movedim(-1, -3).contiguous().float()


# --------------------------------------------------------------- resize/crop geometry --
def crop_geom(src_w, src_h, dst_w, dst_h):
    """The exact integer cover-resize + centre-crop geometry used by the dataset.

    Returns ``(nw, nh, left, top)``: the integer resized size (clamped to at least the
    destination size) and the integer crop offsets.  The image crop and the intrinsics
    fix-up **must** share these numbers or ``K`` drifts away from the pixels.
    """
    s = max(dst_w / src_w, dst_h / src_h)
    nw = max(dst_w, round(src_w * s))
    nh = max(dst_h, round(src_h * s))
    left = (nw - dst_w) // 2
    top = (nh - dst_h) // 2
    return nw, nh, left, top


def adjust_intrinsics_crop(fx, fy, cx, cy, src_w, src_h, dst_w, dst_h):
    """Normalised-to-source intrinsics -> normalised-to-crop, sharing :func:`crop_geom`.

    Anisotropic rounding (``nw / dst_w`` vs ``nh / dst_h``) is handled per axis, and the
    ``cx, cy`` shift is kept even when the source has ``cx = cy = 0.5`` (it becomes
    ~0.5006 after an off-by-one crop).
    """
    nw, nh, left, top = crop_geom(src_w, src_h, dst_w, dst_h)
    fx2 = fx * nw / dst_w
    fy2 = fy * nh / dst_h
    cx2 = (cx * nw - left) / dst_w
    cy2 = (cy * nh - top) / dst_h
    return fx2, fy2, cx2, cy2


if __name__ == "__main__":  # pragma: no cover - self-test
    torch.manual_seed(0)
    V, H, W = 8, 30, 52

    # Random proper rotations via QR, independent of the quaternion code under test.
    A = torch.randn(V, 3, 3, dtype=torch.float64)
    Q, Rr = torch.linalg.qr(A)
    Q = Q * torch.diagonal(Rr, dim1=-2, dim2=-1).sign().unsqueeze(-2)
    Q[torch.linalg.det(Q) < 0, :, 0] *= -1

    # 1) quaternion round trip
    q = matrix_to_quaternion(Q.float())
    assert torch.allclose(quaternion_to_matrix(q), Q.float(), atol=1e-5), "quat round trip"

    # 2) build_cameras: c2w = inv(w2c) with the y/z columns flipped
    t_w2c = torch.randn(V, 3, 1, dtype=torch.float64)
    w2c = torch.cat([Q, t_w2c], dim=-1)
    intr = torch.tensor([0.9, 1.2, 0.5, 0.5]).expand(V, 4)
    cams = build_cameras(w2c, intr)
    assert cams.shape == (V, 11) and cams.dtype == torch.float32
    R_gl = Q.transpose(-1, -2).clone().float()
    R_gl[:, :3, 1:3] *= -1
    t_gl = (-Q.transpose(-1, -2) @ t_w2c).squeeze(-1).float()
    assert torch.allclose(quaternion_to_matrix(cams[:, :4]), R_gl, atol=1e-5), "build R"
    assert torch.allclose(cams[:, 4:7], t_gl, atol=1e-4), "build t"
    assert torch.equal(cams[:, 7:], intr.float()), "build intrinsics passthrough"

    # 3) normalize: frame 0 becomes exact identity, translations bounded by ~1
    ncams = normalize_cameras(cams)
    assert torch.allclose(ncams[0, :4], torch.tensor([1.0, 0, 0, 0]), atol=1e-5), "frame0 q"
    assert ncams[0, 4:7].abs().max() < 1e-5, "frame0 t"
    assert torch.equal(ncams[:, 7:], cams[:, 7:]), "normalize keeps intrinsics"

    # 3b) joint normalisation leaves the input rows bitwise-equal to the solo version
    inp_n, nov_n = normalize_cameras_joint(cams[:5], cams[5:])
    assert torch.equal(inp_n, normalize_cameras(cams[:5])), "joint != solo on inputs"

    # 4) raymap invariants
    rm = create_raymaps(ncams, H, W)
    assert rm.shape == (V, 6, H, W) and rm.dtype == torch.float32
    d, op = rm[:, :3], rm[:, 3:]
    assert (d.pow(2).sum(1) - 1).abs().max() < 1e-5, "|d| != 1"
    assert (d * op).sum(1).abs().max() < 1e-5, "o_perp not orthogonal to d"

    # 5) an identity camera matches the analytic pinhole grid
    idc = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0.9, 1.2, 0.5, 0.5]])
    rmi = create_raymaps(idc, H, W)
    jj, ii = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    exp = F.normalize(torch.stack([(ii + 0.5 - 0.5 * W) / (0.9 * W),
                                   -(jj + 0.5 - 0.5 * H) / (1.2 * H),
                                   -torch.ones_like(ii)], dim=0), dim=0)
    assert torch.allclose(rmi[0, :3], exp, atol=1e-5), "identity-cam pinhole mismatch"
    assert rmi[0, 3:].abs().max() == 0, "identity-cam o_perp must be exactly zero"

    # 6) Pluecker invariance: sliding the origin along the centre ray changes nothing
    slid = ncams.clone()
    look = quaternion_to_matrix(slid[:, :4])[:, :, 2]
    slid[:, 4:7] -= 0.37 * look
    rm2 = create_raymaps(slid, H, W)
    cj, ci = H // 2, W // 2
    assert (rm[:, :, cj, ci] - rm2[:, :, cj, ci]).abs().max() < 2e-2, "Pluecker invariance"

    # 7) uv_offset=None is exactly uv_offset=zeros
    o_none, d_none = create_rays(ncams, H, W)
    o_zero, d_zero = create_rays(ncams, H, W, torch.zeros(V, H, W, 2))
    assert torch.equal(o_none, o_zero) and torch.equal(d_none, d_zero), "uv None != zeros"

    # 8) shifting uv by (+1, 0) == shifting cx by one pixel
    uv10 = torch.zeros(V, H, W, 2)
    uv10[..., 0] = 1.0
    o_uv, d_uv = create_rays(ncams, H, W, uv10)
    cams_cx = ncams.clone()
    cams_cx[:, 9] -= 1.0 / W
    o_cx, d_cx = create_rays(cams_cx, H, W)
    assert torch.allclose(d_uv, d_cx, atol=1e-5), "uv(+1,0) != cx-1 directions"
    assert torch.equal(o_uv, o_cx), "uv offset must not move ray origins"

    # 9) crop geometry and intrinsics stay consistent
    nw, nh, left, top = crop_geom(1280, 720, 832, 480)
    assert nw >= 832 and nh >= 480 and left >= 0 and top >= 0
    fx2, fy2, cx2, cy2 = adjust_intrinsics_crop(0.5, 0.5, 0.5, 0.5, 1280, 720, 832, 480)
    assert 0.3 < cx2 < 0.7 and 0.3 < cy2 < 0.7, "principal point drifted off centre"

    print("cameras self-test OK")
