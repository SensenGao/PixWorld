"""Multi-step sampling.

The sampler is a first-order flow-matching (Euler) integrator with classifier-free
guidance, plus one thing that is specific to this model: **render-as-x0**.

Once the noise level drops below ``sigma_switch``, instead of stepping on the network's
own image prediction, the model lifts its prediction to a 3D Gaussian field, renders that
field at the input cameras, and steps on the *renders*.  From that point on the trajectory
is constrained to be exactly 3D-consistent.  The final
output **is** the last render.

Setting ``sigma_switch <= 0`` disables the lift and reproduces a plain Euler sampler
bitwise.

For distilled few-step models see :func:`dmd_sampling.sample_few_step`.
"""
import contextlib

import torch

from geometry.cameras import create_raymaps
from models.gs_lift import gs_lift
from diffusion.schedule import shift_sigma

__all__ = ["sample", "reconstruct"]


@torch.no_grad()
def reconstruct(model, gs_enc, gs_dec, gs_head, images, cameras, text_emb,
                render_cameras=None, sigma=0.0, bg_mode="white"):
    """Reconstruct a scene from **clean** posed views -- no diffusion at all.

    This is the other half of what the model does.  Generation starts from noise; here the
    views are already known, so they are fed at sigma = 0 and a single forward pass
    lifts them to a Gaussian field, which is then rendered.

    Args:
        images: [V, 3, H, W] in [-1, 1] -- the observed views.
        cameras: [V, 11] -- their poses.
        render_cameras: [R, 11] to render at; defaults to the input cameras.

    Returns (renders [R, 3, H, W] in [-1, 1], depth [R, 1, H, W],
    scene [1, N, 38]).
    """
    from geometry.render import render_views
    device = images.device
    V, _, H, W = images.shape
    x0 = images.float().unsqueeze(0).permute(0, 2, 1, 3, 4)          # [1,3,V,H,W]
    cams_b = cameras.float().to(device).unsqueeze(0)
    raymap = create_raymaps(cameras.float().cpu(), H, W).to(device)
    cond = torch.zeros(V, 4, H, W, device=device)                    # cond_rgb + cond_mask
    cond10 = torch.cat([raymap, cond], dim=1).permute(1, 0, 2, 3).unsqueeze(0)

    # sigma = 0 means x_t IS x0, bitwise: no noise is added anywhere.
    x_t = (1.0 - sigma) * x0 + sigma * torch.randn_like(x0) if sigma > 0 else x0
    ts = torch.full((1,), sigma * 1000.0, device=device, dtype=torch.float32)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
           else contextlib.nullcontext())
    with amp:
        _x0_hat, taps = model(torch.cat([x_t, cond10], dim=1), ts, text_emb,
                              return_taps=True)
    scene = gs_lift(gs_enc, gs_dec, gs_head, taps, cams_b)
    tgt = (render_cameras if render_cameras is not None else cameras)
    tgt = tgt.float().to(device).unsqueeze(0)
    imgs, depth = render_views(scene, tgt, H, W, bg_mode=bg_mode, use_checkpoint=False)
    return imgs[0] * 2 - 1, depth[0], scene


@torch.no_grad()
def sample(model, gs_enc, gs_dec, gs_head, text_emb, neg_emb=None, cameras=None,
           ref_img=None, n_views=1, height=480, width=832, cfg=5.0, steps=50, shift=8.0,
           cfg_rescale=0.7, seed=42, device="cuda", sigma_switch=0.5, bg_mode="white"):
    """Sample a scene.

    Args:
        text_emb: ``[1, L, 4096]`` prompt embedding.
        neg_emb: ``[1, L, 4096]`` negative embedding, or None to disable guidance.
        cameras: ``[V, 11]`` target cameras, or None for a plain text-to-image sample
            (which never lifts, since there is no geometry to lift to).
        ref_img: ``[3, H, W]`` in ``[-1, 1]``, the image-to-3D reference.  It is pinned
            into view 0 at every step, so the anchor never drifts.
        cfg: guidance scale.  1.0 disables guidance.
        cfg_rescale: pulls the guided prediction's standard deviation back toward the
            conditional one.
        sigma_switch: the noise level below which the sampler switches to rendering.

    Returns:
        ``(x [1, 3, V, H, W]`` in ``[-1, 1]``, ``scene [1, N, 38]`` or None``)``.
    """
    sigma_min_convert = 0.05
    device = torch.device(device)
    V = n_views
    if cameras is not None:
        assert tuple(cameras.shape) == (V, 11), \
            f"cameras {tuple(cameras.shape)} != ({V}, 11)"
        raymap = create_raymaps(cameras.float().cpu(), height, width).to(device)
        cams_b = cameras.float().to(device).unsqueeze(0)
    else:
        raymap = torch.zeros(V, 6, height, width, device=device)
        cams_b = None
    cond_img = torch.zeros(V, 3, height, width, device=device)
    cond_mask = torch.zeros(V, 1, height, width, device=device)
    ref3 = None
    if ref_img is not None:
        ref3 = ref_img.reshape(3, height, width).to(device).float()
        cond_img[0] = ref3
        cond_mask[0] = 1.0
    cond = torch.cat([raymap, cond_img, cond_mask], dim=1).permute(1, 0, 2, 3).unsqueeze(0)

    sigmas = shift_sigma(torch.linspace(1, 0, steps + 1, device=device), shift)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, 3, V, height, width, device=device, dtype=torch.float32, generator=g)
    if ref3 is not None:
        x[:, :, 0] = ref3
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
           else contextlib.nullcontext())

    def _timestep(sigma):
        if ref3 is not None:
            # The reference view is clean, so it gets t = 0 while the rest get the
            # current level.  The transformer's AdaLN sees the max; the decoder's FiLM
            # sees the true per-view value and passes the anchor through untouched.
            t_ = (sigma * 1000.0).repeat(V).view(1, V).clone()
            t_[:, 0] = 0.0
            return t_
        return (sigma * 1000.0).view(1)

    def _predict_x0(x_in, sigma, want_taps=False):
        ts = _timestep(sigma)
        taps = None
        if want_taps:
            x0, taps = model(torch.cat([x_in, cond], dim=1), ts, text_emb, return_taps=True)
            x0 = x0.float()
        else:
            x0 = model(torch.cat([x_in, cond], dim=1), ts, text_emb).float()
        if cfg != 1.0 and neg_emb is not None:
            x0_neg = model(torch.cat([x_in, cond], dim=1), ts, neg_emb).float()
            x0_c = x0_neg + cfg * (x0 - x0_neg)
            if cfg_rescale > 0:
                std_pos = x0.std(dim=(1, 2, 3, 4), keepdim=True)
                std_cfg = x0_c.std(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1e-6)
                x0_c = cfg_rescale * (x0_c * std_pos / std_cfg) + (1 - cfg_rescale) * x0_c
            x0 = x0_c
        return x0, taps

    scene = None
    with amp:
        for i in range(steps):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            dt = sigma_next - sigma
            lift = (cams_b is not None) and (float(sigma) < sigma_switch)
            x0, taps = _predict_x0(x, sigma, want_taps=lift)
            if lift:
                from geometry.render import render_views
                scene = gs_lift(gs_enc, gs_dec, gs_head, taps, cams_b)
                renders, _ = render_views(scene, cams_b, height, width, bg_mode=bg_mode,
                                          use_checkpoint=False)
                x0 = (renders.float() * 2 - 1).permute(0, 2, 1, 3, 4)
            v = (x - x0) / sigma.clamp(min=sigma_min_convert)
            x = x + dt * v
            if ref3 is not None:
                x[:, :, 0] = ref3
    return x, scene
