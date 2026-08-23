"""Few-step sampling with a distilled model.

This is the inference counterpart of :func:`dmd_core.rollout` -- the same
dispatch, so the sampler and the training trajectory are literally the same function.  At
the default schedule that is three multi-view steps followed by one 3D Gaussian step whose
render is the output.

Guidance defaults to **off**.  A DMD2 student has the teacher's guidance baked into its
weights by the distillation target.
"""
import torch

from geometry.cameras import create_raymaps
from diffusion.dmd_core import add_noise, gen_forward

__all__ = ["sample_few_step"]


@torch.no_grad()
def sample_few_step(gen, sched, text_emb, cameras, height=480, width=832,
                    ref_img=None, neg_emb=None, cfg=1.0, seed=0, device="cuda"):
    """Sample a scene in ``sched.n`` steps.

    Args:
        gen: a :class:`~dmd_core.Generator`.
        sched: a :class:`~dmd_schedule.GenSchedule`.
        text_emb: ``[1, L, 4096]``.
        cameras: ``[V, 11]`` target cameras.
        ref_img: ``[3, H, W]`` in ``[-1, 1]`` for image-to-3D; pinned into view 0 at every
            step.
        cfg: guidance scale; 1.0 (the default) disables it.

    Returns ``(x [1, 3, V, H, W]`` in ``[-1, 1]``, ``scene [1, N, 38])``.
    """
    device = torch.device(device)
    V = cameras.shape[0]
    cams_b = cameras.float().to(device).unsqueeze(0)
    # Built on device, matching how training builds its ray maps.
    raymap = create_raymaps(cams_b[0], height, width)                     # [V,6,H,W]
    cond_img = torch.zeros(V, 3, height, width, device=device)
    cond_mask = torch.zeros(V, 1, height, width, device=device)
    ref3 = None
    if ref_img is not None:
        ref3 = ref_img.reshape(3, height, width).to(device).float()
        cond_img[0] = ref3
        cond_mask[0] = 1.0
    cond10 = torch.cat([raymap, cond_img, cond_mask], dim=1).permute(1, 0, 2, 3).unsqueeze(0)

    g = torch.Generator(device=device).manual_seed(int(seed))
    x_t = torch.randn(1, 3, V, height, width, device=device, dtype=torch.float32,
                      generator=g)
    scene = None
    for i in range(sched.n):
        if ref3 is not None:
            x_t = x_t.clone()
            x_t[:, :, 0] = ref3
        sig = torch.full((1,), sched.sigmas[i], device=device, dtype=torch.float32)
        need_3d = sched.renders_at(i)
        key = "rgb_3d" if need_3d else "x0_2d"
        out = gen_forward(gen, x_t, cond10, sig, text_emb, cams_b, cams_b, height, width,
                          need_3d=need_3d, bg_mode="white", i2mv=ref3 is not None,
                          use_checkpoint=False)
        if cfg != 1.0 and neg_emb is not None:
            out_n = gen_forward(gen, x_t, cond10, sig, neg_emb, cams_b, cams_b, height,
                                width, need_3d=need_3d, bg_mode="white",
                                i2mv=ref3 is not None, use_checkpoint=False)
            x0 = out_n[key] + cfg * (out[key] - out_n[key])
        else:
            x0 = out[key]
        if need_3d:
            scene = out["scene"]
        if i + 1 < sched.n:
            noise = torch.randn(x0.shape, device=device, dtype=torch.float32, generator=g)
            x_t = add_noise(x0, sched.sigmas_next[i], noise)
        else:
            x_t = x0                      # the last step is never re-noised
    if ref3 is not None:
        x_t = x_t.clone()
        x_t[:, :, 0] = ref3
    return x_t, scene
