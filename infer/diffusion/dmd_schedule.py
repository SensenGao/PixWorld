"""The few-step generator schedule.

The distilled student walks a **fixed, short** sigma ladder.  At the default of four
steps with shift 3 that ladder is exactly::

    k=0  sigma 1.00   multi-view RGB step
    k=1  sigma 0.90   multi-view RGB step
    k=2  sigma 0.75   multi-view RGB step
    k=3  sigma 0.50   3D Gaussian step -- lift, render, and the render is the output

Only the last step lifts to 3D.

The schedule also decides the **noise band** each DMD step is evaluated on.  The band limits
are declared in ``sigma`` and converted to ``u`` internally.
"""
import random

__all__ = ["shift_sigma", "inv_shift_sigma", "GenSchedule"]


def shift_sigma(u, shift):
    """``u`` in ``(0, 1)`` -> ``sigma``, skewed toward high noise for ``shift > 1``."""
    return shift * u / (1.0 + (shift - 1.0) * u)


def inv_shift_sigma(sigma, shift):
    """Exact inverse of :func:`shift_sigma`."""
    return sigma / (shift - sigma * (shift - 1.0))


class GenSchedule:
    """The student's fixed step ladder, and the noise bands the DMD loss is drawn from.

    Args:
        n_steps: number of generator steps.
        shift: schedule shift used to place the step sigmas.
        sigmas: explicit descending sigma list, overriding ``shift`` entirely.  Must start
            at exactly 1.0 -- the rollout begins from pure noise, and any other value
            silently makes the first iterate not-noise.
        sigma_hi, sigma_lo: the DMD band limits, **in sigma**.
        narrow_prob: probability of drawing from the narrow per-step band instead of the
            full one.
        gs_step: first step index that lifts to 3D.  Negative means "only the last".
    """

    def __init__(self, n_steps=4, shift=3.0, sigmas=None, sigma_hi=0.98, sigma_lo=0.02,
                 narrow_prob=0.1, gs_step=-1):
        self.n = int(n_steps)
        self.shift = float(shift)
        self.narrow_prob = float(narrow_prob)
        self.u = [1.0 - k / self.n for k in range(self.n)]
        if sigmas is not None:
            sg = [float(s) for s in sigmas]
            if len(sg) != self.n:
                raise ValueError(f"{len(sg)} sigmas for {self.n} steps")
            if any(a <= b for a, b in zip(sg, sg[1:])):
                raise ValueError(f"sigmas must be strictly descending, got {sg}")
            self.sigmas = sg
        else:
            self.sigmas = [shift_sigma(u, self.shift) for u in self.u]
        if abs(self.sigmas[0] - 1.0) > 1e-9:
            raise ValueError(
                f"sigmas[0] must be exactly 1.0 (the rollout starts from pure noise), "
                f"got {self.sigmas[0]}")
        # The final step is never re-noised: its output is the sample.
        self.sigmas_next = self.sigmas[1:] + [0.0]
        self.gs_step = (self.n - 1) if gs_step < 0 else int(gs_step)
        self.u_hi = inv_shift_sigma(float(sigma_hi), self.shift)
        self.u_lo = inv_shift_sigma(float(sigma_lo), self.shift)

    def renders_at(self, k):
        """Does step ``k`` lift to 3D and render?"""
        return k >= self.gs_step

    def layout(self):
        """``[(kind, sigma), ...]`` for logging."""
        return [("3DGS" if self.renders_at(k) else "MV-2D", self.sigmas[k])
                for k in range(self.n)]

    def step_weights(self, last_w=0.4):
        """How often each step is chosen for a DMD update.

        ``last_w < 0`` gives a uniform draw.  Otherwise the final step -- the only one
        that produces a 3D sample -- gets ``last_w`` and the rest share the remainder.
        The default 0.4 means 40% of generator updates carry a Gaussian gradient.
        """
        if last_w < 0:
            return [1.0 / self.n] * self.n
        if self.n == 1:
            return [1.0]
        rest = (1.0 - last_w) / (self.n - 1)
        return [rest] * (self.n - 1) + [float(last_w)]

    def dmd_u_range(self, k, rng=random):
        """The ``(u_lo, u_hi)`` interval this step's DMD noise is drawn from, low first.

        With probability ``narrow_prob`` the band is narrowed to roughly the interval this
        step actually occupies; otherwise it spans the whole usable range.
        """
        if not 0 <= k < self.n:
            raise IndexError(f"step {k} out of range for {self.n} steps")
        narrow = rng.random() <= self.narrow_prob
        hi = min(self.u_hi, self.u[k]) if narrow else self.u_hi
        lo = max(self.u_lo, self.u[k + 1]) if k + 1 < self.n else self.u_lo
        lo = min(lo, hi - 1e-6)          # keep the interval open when the bands degenerate
        return lo, hi

    def dmd_sigma(self, k, rng=random):
        """Draw one DMD noise level for step ``k`` -- uniform in ``u``, mapped to sigma."""
        lo, hi = self.dmd_u_range(k, rng)
        return float(shift_sigma(lo + (hi - lo) * rng.random(), self.shift))

    def describe(self):
        parts = [f"{kind}@{s:.3f}" for kind, s in self.layout()]
        return (f"{self.n} steps (shift {self.shift:g}): " + " -> ".join(parts)
                + f" | DMD band u in [{self.u_lo:.4f}, {self.u_hi:.4f}]")
