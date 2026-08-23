"""DMD2 objectives.

Three networks are involved:

* the **generator** (student), which produces a sample in a few steps;
* the **real score** (teacher), frozen -- it defines the distribution to match;
* the **fake score** (critic), trained -- it tracks the *generator's own* distribution.

The generator is pushed along the difference between the two scores.  The critic is
trained on the generator's samples and never sees a real image at all.

The adversarial branch of DMD2 is not implemented; the generator is trained on the score
difference alone.
"""
import torch
import torch.nn.functional as F

__all__ = ["dmd_loss", "critic_v_loss", "depth_reg_fn", "depth_reg_multiscale"]


def dmd_loss(x_fake, x0_real, x0_fake, mask=None, cond_rgb=None, eps=1e-6, w_max=0.0):
    """The distribution-matching loss on the generator's sample.

    Args:
        x_fake: ``[B, 3, R, H, W]`` the generator's sample, **with** grad.
        x0_real: the frozen teacher's ``x0`` on ``noise(x_fake)``, no grad.
        x0_fake: the trained critic's ``x0`` on the same noised tensor, no grad.
        mask: ``[B, 1, R, 1, 1]``, 1 where a view is the image-conditioning reference.
        cond_rgb: ``[B, 3, R, H, W]`` reference pixels at those views.

    Returns ``(loss, log_dict)``.

    **The algebra.**  DMD is usually written in velocity space as

    .. code-block:: text

        target = G - sigma * (v_real - v_fake)

    With ``v = (x_t - x0_hat) / sigma`` evaluated at the *same* ``(x_t, sigma)`` for both
    scores, ``sigma * (v_real - v_fake) = x0_fake - x0_real`` exactly, so

    .. code-block:: text

        target = G + (x0_real - x0_fake)

    -- no division by sigma anywhere, and therefore no sigma clamp to tune.  The loss is
    a plain MSE against that stop-gradient target, so ``dL/dG = w * (x0_fake - x0_real)``:
    the gradient points from the critic's opinion toward the teacher's.

    **The weight.**  ``w = 1 / |G - x0_real|.mean()`` per sample makes the loss scale-free
    in how far the sample currently is from the teacher's reconstruction.  ``eps`` guards a
    zero denominator.

    At an image-conditioned view the target is replaced by the reference pixels, so that
    view is pulled toward the actual conditioning image rather than toward a score
    difference that does not apply to it.
    """
    with torch.no_grad():
        ref = x0_real
        tgt = x_fake.detach() + (x0_real - x0_fake)
        if mask is not None:
            ref = ref * (1 - mask) + cond_rgb * mask
            tgt = tgt * (1 - mask) + cond_rgb * mask
        w = 1.0 / ((x_fake.detach() - ref).abs().mean(dim=(1, 2, 3, 4)) + eps)
        if w_max > 0:
            w = w.clamp(max=w_max)
    loss = (F.mse_loss(x_fake, tgt, reduction="none") * w.view(-1, 1, 1, 1, 1)).mean()
    return loss, {"dmd_w": w.mean().detach(),
                  "dmd_gap": (x_fake.detach() - ref).abs().mean().detach()}


def critic_v_loss(x0_fake_pred, x_noisy, x_fake_detached, noise, sigma,
                  sigma_min=0.05, view_mask=None):
    """The critic's denoising loss: velocity regression on the generator's own samples.

    This is the critic's entire training signal.  It is taken in velocity space:
    ``v = (x_t - x0) / sigma``, which puts a ``1/sigma**2`` weight on the x0 error.

    ``sigma_min`` floors the divisor.

    ``view_mask`` ``[B, 1, R, 1, 1]`` marks views that count.  It exists only to drop a
    pinned image-conditioning view: that view's pixels were overwritten with the clean
    reference, so ``(x_noisy - x0) / sigma`` there is no longer the velocity of
    ``(noise, x_fake)``.
    """
    sc = sigma.view(-1, 1, 1, 1, 1).clamp(min=sigma_min) if torch.is_tensor(sigma) \
        else max(float(sigma), sigma_min)
    v_pred = (x_noisy - x0_fake_pred) / sc
    tgt = noise - x_fake_detached
    if view_mask is None:
        return F.mse_loss(v_pred, tgt)
    per = F.mse_loss(v_pred, tgt, reduction="none") * view_mask
    return per.sum() / (view_mask.expand_as(per).sum() + 1e-8)


def depth_reg_fn(depths):
    """Per-image max-normalised total variation of a rendered depth map. ``[N, 1, H, W]``."""
    depths = depths / (depths.flatten(1, -1).max(dim=-1)[0][:, None, None, None].detach()
                       + 1e-3)
    return depths.diff(dim=-1).abs().mean() + depths.diff(dim=-2).abs().mean()


def depth_reg_multiscale(depths, kernels=(4, 2, 1), weights=(4, 2, 1)):
    """The same smoothness prior at 1/4, 1/2 and full resolution, weighted 4:2:1.

    Accepts ``[B, R, 1, H, W]`` or ``[N, 1, H, W]``.    """
    if depths.dim() == 5:
        depths = depths.flatten(0, 1)
    wt = [w / sum(weights) for w in weights]
    tot = depths.new_zeros(())
    for ks, w in zip(kernels, wt):
        d = F.avg_pool2d(depths, kernel_size=ks, stride=ks) if ks > 1 else depths
        tot = tot + depth_reg_fn(d) * w
    return tot
