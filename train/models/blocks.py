"""Small shared building blocks for the pixel decoder and the Gaussian lift trunk."""
import math

import torch
import torch.nn as nn

__all__ = ["conv_block", "up_block", "sigma_embed"]


def conv_block(cin, cout):
    """3x3 conv + SiLU, no normalisation."""
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.SiLU())


def up_block(cin, cout):
    """Learnable resize-conv: nearest x2 followed by a 3x3 conv."""
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                         nn.Conv2d(cin, cout, 3, padding=1))


def sigma_embed(sig, dim=256):
    """Sinusoidal embedding of ``sigma * 1000`` (the timestep convention).

    ``sig`` is ``[N]``; the result is ``[N, dim]`` fp32.
    """
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=sig.device, dtype=torch.float32)
                      * (math.log(10000.0) / (half - 1)))
    a = sig.float().view(-1, 1) * 1000.0 * freqs.view(1, -1)
    return torch.cat([a.sin(), a.cos()], dim=1)
