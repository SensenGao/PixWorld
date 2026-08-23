"""The pixel decoder ("detailer").

The diffusion transformer works on a 32x-downsampled token grid.  This U-Net turns those
tokens back into full-resolution RGB.  It is a plain encoder/decoder with skip
connections, conditioned two ways:

* **DiT context** is injected at the bottleneck, which sits at exactly ``H/32`` -- the
  native token grid -- so the context needs no interpolation at all.
* **Noise level** is injected as zero-initialised FiLM after every stage, so the FiLM
  starts as an identity map.

The FiLM parameters for all eleven stages (``e1..e5``, ``b``, ``d5..d1``) live in one
linear layer; the per-stage gains and biases are contiguous slices of its output, in that
order.  The Gaussian lift trunk (:mod:`gs_lift`) copies matching slices out
of this layout when it initialises itself from a trained detailer, so the order is part of
the checkpoint contract -- do not reorder ``film_stages``.
"""
import torch
import torch.nn as nn

from models.blocks import conv_block, sigma_embed, up_block

__all__ = ["DetailerUNet"]


class DetailerUNet(nn.Module):
    """``(x_t, ctx, sigma)`` -> RGB prediction.

    Args:
        ctx_dim: channel count of the DiT context injected at the bottleneck.
        ch: channels of the five encoder stages ``(c0..c4)``.
        emb_dim: width of the sinusoidal sigma embedding.
        hidden: hidden width of the FiLM MLP.

    Forward:
        ``x_t [N, 3, H, W]``, ``ctx [N, ctx_dim, H/32, W/32]``, ``sig [N]``
        -> ``[N, 3, H, W]``.

    The output is a **direct** prediction (x0 under the training objective used here),
    not a residual on ``x_t``.
    """

    def __init__(self, ctx_dim=3072, ch=(64, 128, 256, 512, 1024), emb_dim=256, hidden=512):
        super().__init__()
        c0, c1, c2, c3, c4 = ch
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = conv_block(3, c0)       # H
        self.enc2 = conv_block(c0, c1)      # H/2
        self.enc3 = conv_block(c1, c2)      # H/4
        self.enc4 = conv_block(c2, c3)      # H/8
        self.enc5 = conv_block(c3, c4)      # H/16
        # Bottleneck at H/32 == the DiT token grid: ctx is concatenated in natively.
        self.bottleneck = nn.Sequential(nn.Conv2d(c4 + ctx_dim, c4, 1), nn.SiLU())
        self.up5 = up_block(c4, c4); self.dec5 = conv_block(c4 + c4, c3)
        self.up4 = up_block(c3, c3); self.dec4 = conv_block(c3 + c3, c2)
        self.up3 = up_block(c2, c2); self.dec3 = conv_block(c2 + c2, c1)
        self.up2 = up_block(c1, c1); self.dec2 = conv_block(c1 + c1, c0)
        self.up1 = up_block(c0, c0); self.dec1 = conv_block(c0 + c0, c0)
        self.out_conv = nn.Conv2d(c0, 3, 1)

        # FiLM over all eleven stages, applied POST-conv.  Order is part of the
        # checkpoint contract (see the module docstring).
        self.film_stages = (c0, c1, c2, c3, c4, c4, c3, c2, c1, c0, c0)
        self.film_emb_dim = emb_dim
        self.film_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * sum(self.film_stages)))
        with torch.no_grad():
            self.film_mlp[-1].weight.zero_()
            self.film_mlp[-1].bias.zero_()

    def _film(self, sig):
        emb = sigma_embed(sig, self.film_emb_dim).to(self.film_mlp[0].weight.dtype)
        gb = self.film_mlp(emb)
        out, o = [], 0
        for c in self.film_stages:
            g = gb[:, o:o + c]
            b = gb[:, o + c:o + 2 * c]
            out.append((g.view(-1, c, 1, 1), b.view(-1, c, 1, 1)))
            o += 2 * c
        return out

    def forward(self, x_t, ctx, sig):
        fm = self._film(sig)

        def m(h, i):
            g, b = fm[i]
            return h * (1 + g.to(h.dtype)) + b.to(h.dtype)

        e1 = m(self.enc1(x_t), 0)
        e2 = m(self.enc2(self.pool(e1)), 1)
        e3 = m(self.enc3(self.pool(e2)), 2)
        e4 = m(self.enc4(self.pool(e3)), 3)
        e5 = m(self.enc5(self.pool(e4)), 4)
        b_ = self.pool(e5)
        b_ = m(self.bottleneck(torch.cat([b_, ctx], dim=1)), 5)
        d5 = m(self.dec5(torch.cat([self.up5(b_), e5], dim=1)), 6)
        d4 = m(self.dec4(torch.cat([self.up4(d5), e4], dim=1)), 7)
        d3 = m(self.dec3(torch.cat([self.up3(d4), e3], dim=1)), 8)
        d2 = m(self.dec2(torch.cat([self.up2(d3), e2], dim=1)), 9)
        d1 = m(self.dec1(torch.cat([self.up1(d2), e1], dim=1)), 10)
        return self.out_conv(d1)
