"""CUDA warm-up helpers."""
import torch

__all__ = ["warm_cusolver", "warm_gsplat"]


class _WarmInverse(torch.autograd.Function):
    """A no-op reduction whose *backward* runs a 4x4 inverse.

    The backward is the point: it executes on the device's autograd worker thread, which
    is the thread the renderer's per-view ``c2w.inverse()`` will use.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.dev = x.device
        ctx.shape = x.shape
        return x.sum()

    @staticmethod
    def backward(ctx, g):
        torch.eye(4, device=ctx.dev, dtype=torch.float32).inverse()
        return g.expand(ctx.shape).clone()          # d(sum)/dx is ones shaped like x


def warm_cusolver(device, verbose=print):
    """Force cuSOLVER handle creation now, on the main **and** the autograd thread.

    cuSOLVER handles are pooled per (device, thread).  The renderer inverts a 4x4
    camera-to-world matrix per view inside its backward pass.

    Returns True only if both threads really succeeded.  Never raises: if the warm-up
    itself is broken the caller should still see the real failure later, with its real
    traceback.
    """
    if not torch.cuda.is_available():
        return False
    ok_main = ok_bwd = False
    try:
        torch.eye(4, device=device, dtype=torch.float32).inverse()
        ok_main = True
    except Exception as e:                                          # noqa: BLE001
        verbose(f"[pixworld] cuSOLVER warm-up (main thread) FAILED: {type(e).__name__}: {e}")
    try:
        x = torch.zeros(1, device=device, dtype=torch.float32, requires_grad=True)
        _WarmInverse.apply(x).backward()
        ok_bwd = True
    except Exception as e:                                          # noqa: BLE001
        verbose(f"[pixworld] cuSOLVER warm-up (autograd thread) FAILED: "
                f"{type(e).__name__}: {e}")
    if ok_main and ok_bwd and verbose:
        verbose("[pixworld] cuSOLVER handles warmed (main + autograd thread)")
    return ok_main and ok_bwd


def warm_gsplat(verbose=print):
    """Trigger gsplat's CUDA JIT compile once, in this process.

    gsplat compiles its extension on the first ``rasterization()`` call, not at import.
    Call this from the launcher in a single process **before** spawning ranks, or at
    least once per node on a cold cache.
    """
    try:
        from gsplat.cuda._backend import _C                         # noqa: F401
        if verbose:
            verbose("[pixworld] gsplat CUDA extension warmed")
        return True
    except Exception as e:                                          # noqa: BLE001
        if verbose:
            verbose(f"[pixworld] gsplat warm-up failed: {type(e).__name__}: {e}")
        return False
