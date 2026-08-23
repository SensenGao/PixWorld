"""The student generator, the score models, and the rollout that connects them."""
import torch
import torch.nn as nn

from geometry.cameras import create_raymaps, normalize_cameras
from models.gs_lift import gs_lift
from geometry.render import render_views

__all__ = ["TASKS", "TASK_WEIGHTS", "TASKS_INPUT_CAM", "Generator", "ScoreCondEmbeds",
           "add_noise", "make_timesteps", "pin_view0", "score_x0", "real_score_x0",
           "gen_forward", "rollout", "score_cond10"]

#: What the DMD loss is computed on.
#:
#: ``mv_2d`` distils the transformer's own multi-view RGB; ``mv_3d`` distils the Gaussian
#: render at the input cameras; ``mv_3d_novel`` distils the render at held-out cameras.
TASKS = ("mv_2d", "mv_3d", "mv_3d_novel")

#: Relative frequency of each task.  ``mv_3d`` is weighted 3x.
TASK_WEIGHTS = (1.0, 3.0, 1.0)

#: Tasks whose render cameras are the input cameras.
TASKS_INPUT_CAM = ("mv_2d", "mv_3d")


def add_noise(x0, sigma, noise):
    """The flow-matching forward process: ``(1 - sigma) * x0 + sigma * noise``."""
    if torch.is_tensor(sigma) and sigma.dim() > 0:
        sigma = sigma.view(-1, 1, 1, 1, 1)
    return (1.0 - sigma) * x0 + sigma * noise


def make_timesteps(sig, n_views, i2mv):
    """Build the timestep argument.

    Text-conditioned rows feed one sigma per sample, shape ``[B]``.  Image-conditioned
    rows feed a per-view vector whose first entry is 0, because view 0 carries the clean
    reference.  The model keys off ``timestep.dim()`` to tell the two apart.
    """
    B = sig.shape[0]
    if i2mv:
        ts = (sig.view(B, 1).float() * 1000.0).repeat(1, n_views).clone()
        ts[:, 0] = 0.0
        return ts
    return sig.float() * 1000.0


def pin_view0(x_noisy, ref):
    """Overwrite view 0 with the clean reference image.

    Applied by the caller rather than inside :func:`score_x0`.
    """
    if ref is None:
        return x_noisy
    out = x_noisy.clone()
    out[:, :, 0] = ref
    return out


class ScoreCondEmbeds(nn.Module):
    """Learned task and step tokens, appended to the **critic's** text stream only.

    The transformer's cross-attention has no positional embedding on the text stream, so
    appending 64 tokens to the usual 226 is invisible to every other consumer.

    They are not applied to the teacher.    """

    def __init__(self, text_dim=4096, n_steps=4, n_tok=32, tasks=TASKS, std=1e-3):
        super().__init__()
        self.n_tok = int(n_tok)
        self.task = nn.ParameterDict(
            {t: nn.Parameter(torch.randn(1, n_tok, text_dim) * std) for t in tasks})
        self.step = nn.ParameterList(
            [nn.Parameter(torch.randn(1, n_tok, text_dim) * std) for _ in range(n_steps)])

    def forward(self, ehs, task, k):
        B = ehs.shape[0]
        return torch.cat([ehs,
                          self.task[task].to(ehs.dtype).expand(B, -1, -1),
                          self.step[k].to(ehs.dtype).expand(B, -1, -1)], dim=1)


def score_x0(score_model, x_noisy, cond10, sig, ehs, view0_clean=False, autocast=True):
    """Run a transformer as a score model: ``(noisy RGB, cameras, text) -> x0`` in fp32."""
    ts = make_timesteps(sig, x_noisy.shape[2], view0_clean)
    x_cat = torch.cat([x_noisy, cond10], dim=1)
    if autocast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return score_model(x_cat, ts, ehs).float()
    return score_model(x_cat, ts, ehs).float()


def real_score_x0(real_model, x_noisy, cond10, sig, ehs_pos, ehs_neg, cfg=3.0,
                  view0_clean=False):
    """The frozen teacher, with classifier-free guidance.

    Guidance is applied in x0 space.  That is provably the same tensor as guiding in
    velocity space, because ``v = (x_t - x0) / sigma`` is affine in ``x0`` at fixed
    ``(x_t, sigma)``.
    """
    x0_pos = score_x0(real_model, x_noisy, cond10, sig, ehs_pos, view0_clean=view0_clean)
    if cfg == 1.0 or ehs_neg is None:
        return x0_pos
    x0_neg = score_x0(real_model, x_noisy, cond10, sig, ehs_neg, view0_clean=view0_clean)
    return x0_neg + cfg * (x0_pos - x0_neg)


class Generator:
    """The four modules that make up the student, as one object.

    A plain container rather than an ``nn.Module``.
    """

    __slots__ = ("model", "gs_enc", "gs_dec", "gs_head")

    def __init__(self, model, gs_enc, gs_dec, gs_head):
        self.model, self.gs_enc = model, gs_enc
        self.gs_dec, self.gs_head = gs_dec, gs_head

    def train(self, mode=True):
        for m in (self.model, self.gs_enc, self.gs_dec, self.gs_head):
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        for m in (self.model, self.gs_enc, self.gs_dec, self.gs_head):
            yield from m.parameters()


def gen_forward(gen, x_t, cond10, sig, ehs, cams_in, render_cams, height, width,
                need_3d=True, bg_mode="white", return_depth=False, i2mv=False,
                use_checkpoint=None):
    """One generator denoising step.

    Args:
        x_t: ``[B, 3, V, H, W]`` the current iterate, fp32, in ``[-1, 1]``.
        cond10: ``[B, 10, V, H, W]`` = ``[raymap 6 | cond_rgb 3 | cond_mask 1]`` at the
            **input** cameras.
        sig: ``[B]`` this step's noise level.
        cams_in: ``[B, V, 11]`` where the Gaussians are lifted *from*.
        render_cams: ``[B, R, 11]`` where they are rendered *to*.

    Returns a dict with ``x0_2d``, ``taps``, and -- when ``need_3d`` -- ``scene``,
    ``render01``, ``depth01``, ``rgb_3d`` and optionally ``dpt_pred``.

    The transformer runs under bf16 autocast; the Gaussian branch runs **outside** it in
    fp32.
    """
    B, _, V, H, W = x_t.shape
    assert cond10.shape[1] == 10, f"cond10 has {cond10.shape[1]} channels, expected 10"
    ts = make_timesteps(sig, V, i2mv)
    x_cat = torch.cat([x_t, cond10], dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x0_2d, taps = gen.model(x_cat, ts, ehs, return_taps=True)
    out = {"x0_2d": x0_2d.float(), "taps": taps}
    if not need_3d:
        return out
    if return_depth:
        scene, dpt = gs_lift(gen.gs_enc, gen.gs_dec, gen.gs_head, taps, cams_in,
                             return_depth=True)
    else:
        scene, dpt = gs_lift(gen.gs_enc, gen.gs_dec, gen.gs_head, taps, cams_in), None
    if use_checkpoint is None:
        use_checkpoint = torch.is_grad_enabled()
    imgs, depths = render_views(scene, render_cams, height, width, bg_mode=bg_mode,
                                use_checkpoint=use_checkpoint)
    out.update(scene=scene, dpt_pred=dpt, render01=imgs, depth01=depths,
               rgb_3d=(imgs.float() * 2 - 1).permute(0, 2, 1, 3, 4))
    return out


@torch.no_grad()
def rollout(gen, sched, k_target, cond10, ehs, cams_in, height, width,
            i2mv=False, ref=None, noise0=None, generator=None, device=None):
    """Run generator steps ``0 .. k_target-1`` under ``no_grad``.

    Returns the iterate that step ``k_target`` will consume.  Each intermediate step takes
    its ``x0`` from the transformer's own RGB below ``gs_step``, and from the 3D render at
    or above it, then re-noises to the next sigma with fresh noise.

    **The iterate is always at the input cameras**, in both branches.  It has to be: it is
    what the next forward denoises, so it must match ``cond10``'s ray map frame for frame.
    Novel cameras enter only in the graded step, and only as the render target.
    """
    B, V = cams_in.shape[0], cams_in.shape[1]
    dev = device or cams_in.device
    if noise0 is None:
        noise0 = torch.randn(B, 3, V, height, width, device=dev, dtype=torch.float32,
                             generator=generator)
    x_t = noise0                                   # sigma_0 == 1, so x_t IS the noise
    for i in range(k_target):
        if i2mv and ref is not None:
            x_t = x_t.clone()
            x_t[:, :, 0] = ref
        sig = torch.full((B,), sched.sigmas[i], device=dev, dtype=torch.float32)
        need_3d = sched.renders_at(i)
        out = gen_forward(gen, x_t, cond10, sig, ehs, cams_in, cams_in, height, width,
                          need_3d=need_3d, bg_mode="white", i2mv=i2mv,
                          use_checkpoint=False)
        x0 = out["rgb_3d"] if need_3d else out["x0_2d"]
        del out
        noise = torch.randn(x0.shape, device=dev, dtype=torch.float32, generator=generator)
        x_t = add_noise(x0, sched.sigmas_next[i], noise)
    if i2mv and ref is not None:
        x_t = x_t.clone()
        x_t[:, :, 0] = ref
    return x_t


def score_cond10(render_cams, height, width, cond_img=None, cond_mask=None,
                 renormalize=True):
    """The 10 conditioning channels the score models see at the **render** cameras.

    Args:
        render_cams: ``[B, R, 11]``.

    Returns ``[B, 10, R, H, W]``.

    When the render cameras are held-out novel views, they are **re-rooted to their own
    first camera** before the ray map is built.  The Gaussians are still rendered
    with the original, jointly normalised cameras -- only the score model's conditioning
    is re-rooted.
    """
    B, R, _ = render_cams.shape
    cams = render_cams.float()
    if renormalize:
        cams = torch.stack([normalize_cameras(cams[b]) for b in range(B)], dim=0)
    rays = torch.stack([create_raymaps(cams[b], height, width) for b in range(B)], dim=0)
    rays = torch.nan_to_num(rays).permute(0, 2, 1, 3, 4).float()          # [B,6,R,H,W]
    dev = rays.device
    if cond_img is None:
        cond_img = torch.zeros(B, 3, R, height, width, device=dev)
    if cond_mask is None:
        cond_mask = torch.zeros(B, 1, R, height, width, device=dev)
    return torch.cat([rays, cond_img, cond_mask], dim=1)
