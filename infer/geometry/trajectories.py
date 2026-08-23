"""Synthetic camera trajectories.

The model consumes ``V`` posed views.  When you have a scene that is not a problem -- the
poses come with it.  When you have only a prompt or a single image there are no poses at
all, so one has to be invented, and what you invent determines what the model is being
asked to produce: a forward dolly asks for a corridor, an orbit asks for an object, a
sideways pan asks for a wall.

Everything here returns cameras in the convention of :mod:`cameras`
(``[V, 11]``), already passed through :func:`~cameras.normalize_cameras`, so view
0 is the identity pose and translations are bounded by 1 -- the distribution the model was
trained on.

Scale matters more than it looks.  Translations live in normalised units where 1.0 is the
largest camera displacement in the clip, so ``radius`` and ``distance`` here are *relative
to the scene*, not metres.  Values much above 1 push the trajectory outside anything the
model saw during training.
"""
import math

import torch

from geometry.cameras import matrix_to_quaternion, normalize_cameras

__all__ = ["look_at", "cameras_from_c2w", "orbit", "dolly", "spiral", "pan",
           "TRAJECTORIES", "make_trajectory"]

#: Default field of view, as normalised intrinsics: ``fx`` by width, ``fy`` by height,
#: which is why the two differ for a 832x480 frame.
#:
#: They give **square pixels**: ``fx * W == fy * H``.
DEFAULT_INTRINSICS = (0.535, 0.9273, 0.5, 0.5)


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """Build an OpenGL camera-to-world matrix looking from ``eye`` at ``target``.

    Returns ``[4, 4]``.  In the OpenGL convention the camera looks down its own ``-z``,
    so the third column of the rotation is ``-forward``, not ``forward``.
    """
    eye = torch.as_tensor(eye, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    up = torch.as_tensor(up, dtype=torch.float32)
    fwd = target - eye
    n = fwd.norm()
    if n < 1e-8:
        raise ValueError("look_at: eye and target coincide")
    fwd = fwd / n
    right = torch.cross(fwd, up, dim=0)
    if right.norm() < 1e-6:                     # looking straight along `up`
        up = torch.tensor([0.0, 0.0, 1.0])
        right = torch.cross(fwd, up, dim=0)
    right = right / right.norm()
    true_up = torch.cross(right, fwd, dim=0)
    m = torch.eye(4, dtype=torch.float32)
    m[:3, 0] = right
    m[:3, 1] = true_up
    m[:3, 2] = -fwd                             # OpenGL: the camera looks down -z
    m[:3, 3] = eye
    return m


def cameras_from_c2w(c2w, intrinsics=DEFAULT_INTRINSICS, normalize=True):
    """``[V, 4, 4]`` camera-to-world matrices -> ``[V, 11]`` cameras."""
    c2w = torch.as_tensor(c2w, dtype=torch.float32).reshape(-1, 4, 4)
    q = matrix_to_quaternion(c2w[:, :3, :3])
    t = c2w[:, :3, 3]
    k = torch.tensor(intrinsics, dtype=torch.float32).expand(c2w.shape[0], 4)
    cams = torch.cat([q, t, k], dim=-1)
    return normalize_cameras(cams) if normalize else cams


def orbit(n_views=16, radius=0.6, elevation=0.0, arc_degrees=90.0, center=(0.0, 0.0, -1.0),
          intrinsics=DEFAULT_INTRINSICS):
    """Sweep around a point, always looking at it.  Good for object-centred prompts."""
    cams = []
    for i in range(n_views):
        a = math.radians(arc_degrees) * (i / max(n_views - 1, 1) - 0.5)
        eye = (center[0] + radius * math.sin(a),
               center[1] + radius * math.sin(math.radians(elevation)),
               center[2] + radius * math.cos(a))
        cams.append(look_at(eye, center))
    return cameras_from_c2w(torch.stack(cams), intrinsics)


def dolly(n_views=16, distance=0.8, look_ahead=4.0, drift=(0.0, 0.0),
          intrinsics=DEFAULT_INTRINSICS):
    """Move forward along ``-z``, looking ahead.  The natural choice for interiors.

    ``drift`` adds a small ``(x, y)`` displacement over the clip, which keeps the
    trajectory from being a pure zoom.
    """
    cams = []
    for i in range(n_views):
        s = i / max(n_views - 1, 1)
        eye = (drift[0] * s, drift[1] * s, -distance * s)
        cams.append(look_at(eye, (drift[0] * s, drift[1] * s, eye[2] - look_ahead)))
    return cameras_from_c2w(torch.stack(cams), intrinsics)


def spiral(n_views=16, radius=0.25, distance=0.6, turns=1.0, look_ahead=4.0,
           intrinsics=DEFAULT_INTRINSICS):
    """Move forward while circling.  Gives the most parallax per view of the three."""
    cams = []
    for i in range(n_views):
        s = i / max(n_views - 1, 1)
        a = 2 * math.pi * turns * s
        eye = (radius * math.sin(a), radius * (math.cos(a) - 1.0) * 0.5, -distance * s)
        cams.append(look_at(eye, (0.0, 0.0, eye[2] - look_ahead)))
    return cameras_from_c2w(torch.stack(cams), intrinsics)


def pan(n_views=16, extent=0.8, look_ahead=4.0, intrinsics=DEFAULT_INTRINSICS):
    """Translate sideways with the view direction fixed. A pure-parallax trajectory."""
    cams = []
    for i in range(n_views):
        s = i / max(n_views - 1, 1) - 0.5
        eye = (extent * s, 0.0, 0.0)
        cams.append(look_at(eye, (extent * s, 0.0, -look_ahead)))
    return cameras_from_c2w(torch.stack(cams), intrinsics)


TRAJECTORIES = {"orbit": orbit, "dolly": dolly, "spiral": spiral, "pan": pan}


def make_trajectory(name, n_views=16, **kw):
    """Look up a trajectory by name.  See :data:`TRAJECTORIES` for the options."""
    if name not in TRAJECTORIES:
        raise ValueError(f"unknown trajectory {name!r}; choose from {sorted(TRAJECTORIES)}")
    return TRAJECTORIES[name](n_views=n_views, **kw)


if __name__ == "__main__":  # pragma: no cover - self-test
    from geometry.cameras import create_raymaps, quaternion_to_matrix

    for name in sorted(TRAJECTORIES):
        cams = make_trajectory(name, n_views=16)
        assert cams.shape == (16, 11), (name, cams.shape)
        assert torch.isfinite(cams).all(), name
        # view 0 is the normalisation root -> exactly the identity pose
        assert torch.allclose(cams[0, :4], torch.tensor([1.0, 0, 0, 0]), atol=1e-5), name
        assert cams[0, 4:7].abs().max() < 1e-5, name
        # translations bounded by 1, and the trajectory actually moves
        tn = cams[:, 4:7].norm(dim=-1)
        assert tn.max() < 1.0, (name, float(tn.max()))
        assert tn.max() > 0.3, (name, float(tn.max()))
        # unit quaternions, and consecutive views are genuinely distinct poses.
        # Note dolly and pan hold the rotation fixed on purpose, so the test is on the
        # pose as a whole, not on the rotation alone.
        assert (cams[:, :4].norm(dim=-1) - 1).abs().max() < 1e-5, name
        assert (cams[1:, :7] - cams[:-1, :7]).abs().max() > 1e-4, name
        rm = create_raymaps(cams, 48, 80)
        assert torch.isfinite(rm).all() and rm.shape == (16, 6, 48, 80), name
        assert (rm[:, :3].pow(2).sum(1) - 1).abs().max() < 1e-5, name
        rot = float((cams[:, :4] - cams[0, :4]).abs().max())
        print(f"  {name:8s} |t| max {float(tn.max()):.3f}  rotation span {rot:.3f}"
              + ("   (translation only, by design)" if rot < 1e-5 else ""))

    # The ray at the principal point must equal the direction the camera was aimed.
    c2w = look_at((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    cams = cameras_from_c2w(c2w[None], normalize=False)
    rm = create_raymaps(cams, 64, 64)
    centre = rm[0, :3, 32, 32]
    assert torch.allclose(centre, torch.tensor([0.0, 0.0, -1.0]), atol=2e-2), centre
    # and off-axis: looking down +x
    c2w = look_at((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    R = quaternion_to_matrix(cameras_from_c2w(c2w[None], normalize=False)[:, :4])[0]
    assert torch.allclose(R[:, 2], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-5), R[:, 2]
    print("trajectories self-test OK")
