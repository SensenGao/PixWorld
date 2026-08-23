"""The noise schedule.

Flow matching interpolates linearly between data and noise::

    x_t = (1 - sigma) * x0 + sigma * noise

so ``sigma = 0`` is a clean image and ``sigma = 1`` is pure noise.

``sigma`` is drawn by mapping a uniform ``u`` through

.. math:: \\sigma = \\frac{s\\,u}{1 + (s - 1) u}

with shift ``s`` (16 for the multi-view branch, 8 for single-image).  The draw is
stratified in ``u``: with probability ``lo_frac`` the step lands below ``band_split``,
otherwise above it.  ``pure_noise_prob`` of steps are forced to ``sigma = 1`` (taken from
the high band) and ``pure_clean_prob`` to ``sigma = 0`` (carved out of the low band), so
``lo_frac`` keeps its meaning either way.
"""
import torch

__all__ = ["shift_sigma", "inv_shift_sigma", "SigmaSchedule"]


def shift_sigma(u, shift):
    """``u`` in ``(0, 1)`` -> ``sigma`` in ``(0, 1)``, skewed toward high noise."""
    return shift * u / (1.0 + (shift - 1.0) * u)


def inv_shift_sigma(sigma, shift):
    """Exact inverse of :func:`shift_sigma`."""
    return sigma / (shift - sigma * (shift - 1.0))


class SigmaSchedule:
    """Draws training noise levels.

    Args:
        shift: the schedule shift ``s``.  16 for the multi-view branch.
        lo_frac: fraction of steps placed in ``[0, band_split)``.  Negative disables
            stratification and falls back to a plain shifted-uniform draw.
        band_split: the ``sigma`` that separates the two bands.  Should equal the noise
            level below which the Gaussian branch trains.
        pure_noise_prob: fraction of steps forced to ``sigma = 1``.  These are spent
            inside the high band, so the realised rate is
            ``pure_noise_prob * (1 - lo_frac)`` -- 8% at the defaults, not 10%.
        pure_clean_prob: fraction of steps forced to ``sigma = 0`` (carved out of the low
            band, so it must not exceed ``lo_frac``).
    """

    def __init__(self, shift=16.0, lo_frac=0.2, band_split=0.5,
                 pure_noise_prob=0.1, pure_clean_prob=0.05):
        if lo_frac >= 0.0 and pure_clean_prob > lo_frac:
            raise ValueError(
                f"pure_clean_prob {pure_clean_prob} > lo_frac {lo_frac}: sigma=0 is carved "
                "out of the low band, so it cannot exceed it.  For pure reconstruction "
                "only, use lo_frac=1.0 with pure_clean_prob=1.0.")
        self.shift = float(shift)
        self.lo_frac = float(lo_frac)
        self.band_split = float(band_split)
        self.pure_noise_prob = float(pure_noise_prob)
        self.pure_clean_prob = float(pure_clean_prob)
        self.u_split = inv_shift_sigma(self.band_split, self.shift)

    def sample(self, batch_size, device="cpu", generator=None):
        """Draw ``sigma`` for a batch.  Returns ``[B]`` fp32 in ``[0, 1]``."""
        def rand():
            return torch.rand(batch_size, device=device, generator=generator)

        u = rand()
        lo_band = None
        if self.lo_frac >= 0.0:
            lo_band = rand() < self.lo_frac
            u = torch.where(lo_band, u * self.u_split,
                            self.u_split + u * (1.0 - self.u_split))
        sigma = shift_sigma(u, self.shift)

        if self.pure_noise_prob > 0:
            pure = rand() < self.pure_noise_prob
            if lo_band is not None:
                # sigma == 1 lives in the high band by construction, so spend the budget
                # there.  Overwriting low-band rows would silently shrink the low band to
                # lo_frac * (1 - pure_noise_prob).
                pure = pure & (~lo_band)
            sigma = torch.where(pure, torch.ones_like(sigma), sigma)

        if self.pure_clean_prob > 0:
            if lo_band is not None:
                r = self.pure_clean_prob / max(self.lo_frac, 1e-12)
                clean = lo_band & (rand() < r)
            else:
                clean = rand() < self.pure_clean_prob
            sigma = torch.where(clean, torch.zeros_like(sigma), sigma)
        return sigma

    def describe(self):
        """One-line human-readable summary, for the training log."""
        if self.lo_frac < 0:
            return (f"sigma: plain shift={self.shift:g}, "
                    f"pure_noise={self.pure_noise_prob:.0%}, "
                    f"pure_clean={self.pure_clean_prob:.0%}")
        return (f"sigma: shift={self.shift:g}, {self.lo_frac:.0%} in "
                f"[0,{self.band_split:g}) / {1 - self.lo_frac:.0%} in "
                f"[{self.band_split:g},1] (u_split={self.u_split:.4f}), "
                f"pure_noise={self.pure_noise_prob:.0%} (high band), "
                f"pure_clean={self.pure_clean_prob:.0%} (carved from low band)")


if __name__ == "__main__":  # pragma: no cover - self-test
    torch.manual_seed(0)
    for s in (1.0, 3.0, 8.0, 16.0):
        u = torch.rand(10000)
        assert torch.allclose(inv_shift_sigma(shift_sigma(u, s), s), u, atol=1e-5), s

    sch = SigmaSchedule()
    sig = sch.sample(2_000_000)
    p_lo = float((sig < sch.band_split).float().mean())
    p_one = float((sig == 1).float().mean())
    p_zero = float((sig == 0).float().mean())
    print(sch.describe())
    print(f"  P(sigma < {sch.band_split}) = {p_lo:.4f}   (target {sch.lo_frac})")
    tgt_one = sch.pure_noise_prob * (1 - sch.lo_frac)   # confined to the high band
    print(f"  P(sigma == 1)      = {p_one:.4f}   (target {tgt_one})")
    print(f"  P(sigma == 0)      = {p_zero:.4f}   (target {sch.pure_clean_prob})")
    print(f"  median | sigma<split = {sig[(sig < sch.band_split) & (sig > 0)].median():.4f}")
    assert abs(p_lo - sch.lo_frac) < 3e-3, p_lo
    assert abs(p_one - tgt_one) < 3e-3, p_one
    assert abs(p_zero - sch.pure_clean_prob) < 3e-3, p_zero
    assert sig.min() >= 0 and sig.max() <= 1
    print("schedule self-test OK")
