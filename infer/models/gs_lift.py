"""The 3D Gaussian lift trunk: encoder, decoder, and the lift entry point.

The Gaussian branch has its **own** encoder/decoder, structurally disjoint from the RGB
detailer.  Both are initialised from a trained detailer, so at the moment of the fork the
lift reproduces the detailer's features exactly, but from then on render and depth
gradients can only reach the shared trunk through the DiT context -- never through the RGB
head.

Everything here runs fp32 with autocast disabled.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.checkpoint import checkpoint

from models.blocks import conv_block, sigma_embed, up_block

__all__ = ["GaussianEncoder", "GaussianDecoder", "gs_lift"]


class GaussianEncoder(nn.Module):
    """A private copy of the detailer's encoder half, with its own FiLM.

    Forward:
        ``x_t [N, 3, H, W]`` (any dtype), ``sig [N]`` -> dict of ``e1..e5`` and
        ``b_pooled``, all fp32.
    """

    def __init__(self, ch=(64, 128, 256, 512, 1024), emb_dim=256, hidden=512):
        super().__init__()
        c0, c1, c2, c3, c4 = ch
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = conv_block(3, c0)       # H
        self.enc2 = conv_block(c0, c1)      # H/2
        self.enc3 = conv_block(c1, c2)      # H/4
        self.enc4 = conv_block(c2, c3)      # H/8
        self.enc5 = conv_block(c3, c4)      # H/16
        self.film_stages = (c0, c1, c2, c3, c4)         # e1..e5, FiLM applied post-conv
        self.film_emb_dim = emb_dim
        self.film_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * sum(self.film_stages)))
        with torch.no_grad():
            self.film_mlp[-1].weight.zero_()
            self.film_mlp[-1].bias.zero_()
        self.gradient_checkpointing = False

    def _film(self, sig):
        emb = sigma_embed(sig, self.film_emb_dim).to(self.film_mlp[0].weight.dtype)
        gb = self.film_mlp(emb)
        out, o = [], 0
        for c in self.film_stages:
            g = gb[:, o:o + c].view(-1, c, 1, 1)
            b = gb[:, o + c:o + 2 * c].view(-1, c, 1, 1)
            out.append((g, b))
            o += 2 * c
        return out

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, x_t, sig):
        x_t = x_t.float()
        fm = self._film(sig.float())

        def m(h, i):
            g, b = fm[i]
            return h * (1 + g) + b

        def stage(conv, h, i):
            if self.gradient_checkpointing and self.training:
                return torch.utils.checkpoint.checkpoint(
                    lambda z, _c=conv, _i=i: m(_c(z), _i), h, use_reentrant=False)
            return m(conv(h), i)

        e1 = stage(self.enc1, x_t, 0)
        e2 = stage(self.enc2, self.pool(e1), 1)
        e3 = stage(self.enc3, self.pool(e2), 2)
        e4 = stage(self.enc4, self.pool(e3), 3)
        e5 = stage(self.enc5, self.pool(e4), 4)
        return {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5,
                "b_pooled": self.pool(e5)}

    @torch.no_grad()
    def init_from_detailer(self, det):
        """Copy ``enc1..enc5`` and the matching FiLM rows out of a trained detailer.

        The encoder stages are the **leading** rows of the detailer's FiLM output layout,
        so the slice starts at 0.  After this call the encoder reproduces the detailer's
        own ``e1..e5`` and ``b_pooled`` for identical fp32 inputs.
        """
        for name in ("enc1", "enc2", "enc3", "enc4", "enc5"):
            getattr(self, name).load_state_dict(getattr(det, name).state_dict())
        self.film_mlp[0].load_state_dict(det.film_mlp[0].state_dict())
        n = 2 * sum(det.film_stages[:5])
        assert n == 2 * sum(self.film_stages), (det.film_stages, self.film_stages)
        self.film_mlp[-1].weight.copy_(det.film_mlp[-1].weight[:n])
        self.film_mlp[-1].bias.copy_(det.film_mlp[-1].bias[:n])


class GaussianDecoder(nn.Module):
    """A private copy of the detailer's decoder half, with its own FiLM.

    Consumes the encoder taps plus the DiT context and emits per-pixel features for
    :class:`~gaussian_head.PixelAlignedGaussianHead`.

    Forward:
        ``taps`` dict with ``e1..e5``, ``b_pooled``, ``ctx``, ``sig``
        -> ``[N, ch[0], H, W]`` fp32.

    Set ``gradient_checkpointing = True`` to recompute each decoder stage in the backward
    pass.  When the flag is off, or in eval, the plain path runs and the result is
    bitwise the same.
    """

    def __init__(self, ctx_dim=3072, ch=(64, 128, 256, 512, 1024), emb_dim=256, hidden=512):
        super().__init__()
        c0, c1, c2, c3, c4 = ch
        self.bottleneck = nn.Sequential(nn.Conv2d(c4 + ctx_dim, c4, 1), nn.SiLU())
        self.up5 = up_block(c4, c4); self.dec5 = conv_block(c4 + c4, c3)
        self.up4 = up_block(c3, c3); self.dec4 = conv_block(c3 + c3, c2)
        self.up3 = up_block(c2, c2); self.dec3 = conv_block(c2 + c2, c1)
        self.up2 = up_block(c1, c1); self.dec2 = conv_block(c1 + c1, c0)
        self.up1 = up_block(c0, c0); self.dec1 = conv_block(c0 + c0, c0)
        # Channels of each stage's OUTPUT: b -> c4, d5 -> c3, d4 -> c2, d3 -> c1,
        # d2 -> c0, d1 -> c0.  FiLM is applied post-conv.
        self.film_stages = (c4, c3, c2, c1, c0, c0)
        self.film_emb_dim = emb_dim
        self.film_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * sum(self.film_stages)))
        with torch.no_grad():
            self.film_mlp[-1].weight.zero_()
            self.film_mlp[-1].bias.zero_()
        self.gradient_checkpointing = False

    def _film(self, sig):
        emb = sigma_embed(sig, self.film_emb_dim).to(self.film_mlp[0].weight.dtype)
        gb = self.film_mlp(emb)
        out, o = [], 0
        for c in self.film_stages:
            g = gb[:, o:o + c].view(-1, c, 1, 1)
            b = gb[:, o + c:o + 2 * c].view(-1, c, 1, 1)
            out.append((g, b))
            o += 2 * c
        return out

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, taps):
        e1, e2, e3, e4, e5 = (taps[k].float() for k in ("e1", "e2", "e3", "e4", "e5"))
        b_pooled = taps["b_pooled"].float()
        ctx = taps["ctx"].float()
        # FiLM is a two-layer MLP on a sigma embedding -- kilobytes -- and every stage
        # needs its slice, so it is computed once outside the checkpoints.
        fm = self._film(taps["sig"].float())

        def m(h, i):
            g, b = fm[i]
            return h * (1 + g) + b

        if not (self.gradient_checkpointing and self.training):
            b_ = m(self.bottleneck(torch.cat([b_pooled, ctx], dim=1)), 0)
            d5 = m(self.dec5(torch.cat([self.up5(b_), e5], dim=1)), 1)
            d4 = m(self.dec4(torch.cat([self.up4(d5), e4], dim=1)), 2)
            d3 = m(self.dec3(torch.cat([self.up3(d4), e3], dim=1)), 3)
            d2 = m(self.dec2(torch.cat([self.up2(d3), e2], dim=1)), 4)
            d1 = m(self.dec1(torch.cat([self.up1(d2), e1], dim=1)), 5)
            return d1

        def stage(conv, up, i):
            def fn(x, skip):
                h = conv(torch.cat([up(x), skip], dim=1)) if up is not None \
                    else conv(torch.cat([x, skip], dim=1))
                g, b = fm[i]
                return h * (1 + g) + b
            return fn

        b_ = checkpoint(stage(self.bottleneck, None, 0), b_pooled, ctx, use_reentrant=False)
        d5 = checkpoint(stage(self.dec5, self.up5, 1), b_, e5, use_reentrant=False)
        d4 = checkpoint(stage(self.dec4, self.up4, 2), d5, e4, use_reentrant=False)
        d3 = checkpoint(stage(self.dec3, self.up3, 3), d4, e3, use_reentrant=False)
        d2 = checkpoint(stage(self.dec2, self.up2, 4), d3, e2, use_reentrant=False)
        d1 = checkpoint(stage(self.dec1, self.up1, 5), d2, e1, use_reentrant=False)
        return d1

    @torch.no_grad()
    def init_from_detailer(self, det):
        """Copy the decoder half and the matching FiLM rows out of a trained detailer.

        The decoder stages are the **trailing** contiguous rows of the detailer's FiLM
        layout, so the slice starts after the five encoder stages.  After this call the
        decoder reproduces the detailer's own ``d1`` bitwise for identical fp32 inputs.
        """
        for name in ("bottleneck", "up5", "dec5", "up4", "dec4", "up3", "dec3",
                     "up2", "dec2", "up1", "dec1"):
            getattr(self, name).load_state_dict(getattr(det, name).state_dict())
        self.film_mlp[0].load_state_dict(det.film_mlp[0].state_dict())
        det_stages = det.film_stages
        off = 2 * sum(det_stages[:5])
        n = 2 * sum(det_stages[5:])
        assert n == 2 * sum(self.film_stages), (det_stages, self.film_stages)
        self.film_mlp[-1].weight.copy_(det.film_mlp[-1].weight[off:off + n])
        self.film_mlp[-1].bias.copy_(det.film_mlp[-1].bias[off:off + n])


def gs_lift(gs_enc, gs_dec, gs_head, taps, cameras, return_depth=False):
    """Lift DiT taps to a 3D Gaussian scene.

    Args:
        taps: ``{"ctx", "x_t", "sig"}`` as returned by
            :meth:`dit.PixWorldMV.forward` with ``return_taps=True``.
            ``ctx`` is live; the Gaussian branch runs its own encoder on the raw ``x_t``.
        cameras: ``[B, V, 11]``.

    Returns:
        ``scene [B, V*P*H*W, 38]``, and per-Gaussian ``depth [B*V, P, H, W]`` when
        ``return_depth``.

    There is no ``detach`` anywhere, but the graph is structurally disjoint from the RGB
    head: render and depth gradients reach the shared trunk through ``ctx`` only.
    """
    B, V = cameras.shape[0], cameras.shape[1]
    assert cameras.shape[-1] == 11, f"cameras {tuple(cameras.shape)}"
    assert "e1" not in taps, "taps must carry no detailer tensor (the encoder is forked)"
    enc = gs_enc(taps["x_t"], taps["sig"])          # private e1..e5, b_pooled
    enc["ctx"], enc["sig"] = taps["ctx"], taps["sig"]
    gd1 = gs_dec(enc)                               # [B*V, 64, H, W] fp32
    assert gd1.shape[0] == B * V, (gd1.shape, B, V)
    cams = cameras.reshape(B * V, 11).float()
    if return_depth:
        gs, dpt = gs_head(gd1, cams, return_depth=True)
    else:
        gs, dpt = gs_head(gd1, cams), None
    assert gs.dim() == 3 and gs.shape[0] == B * V and gs.shape[-1] == 38, tuple(gs.shape)
    scene = gs.reshape(B, V * gs.shape[1], 38)
    return (scene, dpt) if return_depth else scene
