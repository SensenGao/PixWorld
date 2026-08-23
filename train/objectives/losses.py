"""Training objectives.

Two bands, mutually exclusive per step, decided by the noise level:

* **High noise** (``sigma >= gs_sigma_hi``): only the pixel reconstruction term
  (:func:`anchor_loss`) runs.  The Gaussian lift is not evaluated at all.
* **Low noise** (``sigma < gs_sigma_hi``): the model lifts to Gaussians, renders the
  target views, and pays :func:`render_photo_loss` and :func:`depth_logit_loss` on top of
  the reconstruction term.

The two bands are weighted on **different** powers of sigma:

* the pixel term carries an intrinsic ``1 / max(sigma, sigma_min)**2`` -- it is a
  velocity-space MSE, and converting an x0 prediction to velocity divides by sigma;
* every Gaussian-branch term is weighted ``1 / max(sigma, sigma_min)`` instead.

so as the input gets noisier the geometry terms fade faster than the pixel term.
"""
import torch
import torch.nn.functional as F

__all__ = ["anchor_loss", "gs_sigma_weight", "render_photo_loss", "depth_logit_loss"]


def anchor_loss(x0_hat, x_t, x0, sigma, sigma_min=0.05,
                lpips_fn=None, lpips_w=0.1, lpips_gate=0.7):
    """Velocity-space reconstruction loss on the predicted image.

    Args:
        x0_hat, x_t, x0: ``[B, 3, V, H, W]``.
        sigma: ``[B, 1, 1, 1, 1]``.
        lpips_fn: optional perceptual loss, applied only where ``sigma < lpips_gate``.

    Returns ``(loss, log_dict)``.

    The model predicts ``x0``, but the loss is taken in velocity space::

        pred_v = (x_t - x0_hat) / max(sigma, sigma_min)

    which is algebraically ``mean((x0_hat - x0)**2) / max(sigma, sigma_min)**2``.  The floor at
    ``sigma_min`` caps that weight at 400.    """
    B, _, V, H, W = x0.shape
    sig = sigma.view(B)
    sigma_c = sigma.clamp(min=sigma_min)
    pred_v = (x_t - x0_hat).float() / sigma_c
    target_v = (x_t - x0).float() / sigma_c
    mse_b = (pred_v - target_v).square().mean(dim=(1, 2, 3, 4))
    loss = mse_b.mean()
    log = {"anchor_mse": loss.detach(), "mse_b": mse_b.detach(), "sigma_b": sig.detach(),
           "mse_raw_b": (x0_hat.float() - x0.float()).square().mean(dim=(1, 2, 3, 4)).detach()}
    if lpips_fn is not None and lpips_w > 0:
        gate = sig < lpips_gate
        if gate.any():
            n = int(gate.sum())
            a = x0_hat[gate].transpose(1, 2).reshape(n * V, 3, H, W).float().clamp(-1, 1)
            b = x0[gate].transpose(1, 2).reshape(n * V, 3, H, W).float()
            lp = lpips_fn(a, b).view(n, V).mean(dim=1)
            w_t = 1.0 / sig[gate].clamp(min=sigma_min).square().float()
            lpips_loss = lpips_w * (lp * w_t).sum() / B
            loss = loss + lpips_loss
            log["lpips"] = lpips_loss.detach()
    return loss, log


def gs_sigma_weight(sig, mode="inv_sigma", sigma_min=0.05):
    """The sigma weighting shared by every Gaussian-branch term.

    ``inv_sigma`` -- ``1 / max(sigma, sigma_min)``, capped at 20 by the floor.  This is
    the shipped setting and it applies identically to the render MSE, the render LPIPS and
    the depth loss, so the three cannot drift apart.

    ``one_minus_sigma`` -- ``1 - sigma``, clamped to ``[0.1, 1.0]``.
    ``none`` disables the weighting entirely.
    """
    if mode == "inv_sigma":
        return (1.0 / sig.clamp(min=sigma_min)).float()
    if mode == "one_minus_sigma":
        return (1.0 - sig).clamp(0.1, 1.0).float()
    if mode == "none":
        return torch.ones_like(sig).float()
    raise ValueError(f"unknown gs_sigma_weight mode {mode!r}")


def render_photo_loss(renders, gt01, sig, lpips_fn=None,
                      weight_mode="inv_sigma", sigma_min=0.05):
    """Multi-scale photometric loss on rendered views.

    Args:
        renders: ``[B, K, 3, H, W]`` in ``[0, 1]`` (the rasteriser's output range).
        gt01: ``[B, K, 3, H, W]`` in ``[0, 1]``.
        sig: ``[B]``.

    Returns ``(mse_loss, lpips_loss)``, unweighted by ``render_w`` -- the caller combines
    them.

    The pyramid is average-pooled at strides 4, 2 and 1 with weights 4/7, 2/7, 1/7, so
    most of the signal comes from the **coarsest** level.  Both inputs are mapped to
    ``[-1, 1]`` first, which is
    the range LPIPS expects.
    """
    B, K = renders.shape[:2]
    t_w = gs_sigma_weight(sig, weight_mode, sigma_min)
    pr = renders.flatten(0, 1).float() * 2 - 1
    gt = gt01.flatten(0, 1).float() * 2 - 1
    mse_l = renders.new_zeros(())
    lp_l = renders.new_zeros(())
    for ks, w in zip((4, 2, 1), (4 / 7.0, 2 / 7.0, 1 / 7.0)):
        pr_s = F.avg_pool2d(pr, ks, ks) if ks > 1 else pr
        gt_s = F.avg_pool2d(gt, ks, ks) if ks > 1 else gt
        m = F.mse_loss(pr_s, gt_s, reduction="none").flatten(1).mean(1)
        mse_l = mse_l + w * (m.view(B, K).mean(1) * t_w).mean()
        if lpips_fn is not None:
            lp = lpips_fn(pr_s, gt_s).view(B, K).mean(1)
            lp_l = lp_l + w * (lp * t_w).mean()
    return mse_l, lp_l


def depth_logit_loss(dpt_pred, depth_gt_z, cams, sig,
                     weight_mode="inv_sigma", sigma_min=0.05):
    """Log-space depth supervision on the Gaussian head's own depth prediction.

    Args:
        dpt_pred: ``[B*V, P, H, W]`` -- the per-Gaussian **ray distance** the head emits.
        depth_gt_z: ``[B, V, H, W]`` -- ground-truth **z-depth** (along the optical axis),
            already in the same normalised units as the camera translations.  ``0`` marks
            a pixel with no ground truth.
        cams: ``[B, V, 11]``.

    Returns ``(loss, valid_fraction)``.

    Two conversions matter here.  The head predicts distance **along the ray** while a
    depth sensor records distance **along the optical axis**, so the ground truth is
    multiplied by ``sqrt(x^2 + y^2 + 1)`` in normalised camera coordinates.  And the loss
    is taken in **log** space, because ``log(depth)`` is exactly the head's raw
    pre-activation logit (the activation is ``exp``) -- so the gradient reaching the head
    is well conditioned across the whole depth range instead of being dominated by distant
    pixels.

    A scene with no depth annotation contributes exactly ``0.0`` with an exactly-zero
    gradient, so mixing annotated and unannotated data needs no special handling.
    """
    B, V, H, W = depth_gt_z.shape
    dev = dpt_pred.device
    c = cams.reshape(B * V, 11).float()
    fx, fy = c[:, 7] * W, c[:, 8] * H
    cx, cy = c[:, 9] * W, c[:, 10] * H
    gy, gx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32) + 0.5,
                            torch.arange(W, device=dev, dtype=torch.float32) + 0.5,
                            indexing="ij")
    x = (gx[None] - cx.view(-1, 1, 1)) / fx.view(-1, 1, 1)
    y = (gy[None] - cy.view(-1, 1, 1)) / fy.view(-1, 1, 1)
    n = torch.sqrt(x * x + y * y + 1.0)                     # z-depth -> ray distance
    t_gt = depth_gt_z.reshape(B * V, H, W).float().to(dev) * n
    valid = (t_gt > 1e-4).unsqueeze(1)
    per = F.smooth_l1_loss(dpt_pred.float().clamp(min=1e-6).log(),
                           t_gt.clamp(min=1e-3).log().unsqueeze(1).expand_as(dpt_pred),
                           beta=0.1, reduction="none")
    v = valid.float().expand_as(per)
    base = (per * v).sum() / (v.sum() + 1e-6)
    t_w = gs_sigma_weight(sig, weight_mode, sigma_min).mean()
    return base * t_w, valid.float().mean().detach()
