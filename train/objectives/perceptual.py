"""LPIPS wrapper with fp32 forcing and optional activation checkpointing."""
import torch

__all__ = ["build_lpips"]


class _FP32LPIPS(torch.nn.Module):
    """Run LPIPS in fp32 regardless of the surrounding autocast context."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, a, b):
        with torch.autocast("cuda", enabled=False):
            return self.fn(a.float(), b.float())


class _CheckpointedLPIPS(torch.nn.Module):
    """Recompute LPIPS in the backward pass instead of retaining VGG's feature stack.

    VGG keeps its whole feature stack for backward, and the loss is called once per
    pyramid level per rendered view.  The module has no trainable parameters, so the only
    thing checkpointing changes is *when* the forward happens.

    ``chunk`` additionally splits the batch.  LPIPS is exactly per-sample    separable (VGG has no batch norm and the reductions are per-sample), so splitting
    computes the same function.

    **``chunk`` is not loss-exact on GPU.** Changing the batch dimension makes cuDNN pick a
    different convolution algorithm, and TF32 carries ~10 mantissa bits.  Leave it at 0 unless you are actually out of memory.
    """

    def __init__(self, fn, enabled=True, chunk=0):
        super().__init__()
        self.fn = fn
        self.enabled = bool(enabled)
        self.chunk = int(chunk)

    def _one(self, a, b):
        if not (self.enabled and torch.is_grad_enabled()
                and (a.requires_grad or b.requires_grad)):
            return self.fn(a, b)
        return torch.utils.checkpoint.checkpoint(self.fn, a, b, use_reentrant=False)

    def forward(self, a, b):
        n, c = a.shape[0], self.chunk
        if c <= 0 or n <= c:
            return self._one(a, b)
        return torch.cat([self._one(a[i:i + c], b[i:i + c])
                          for i in range(0, n, c)], dim=0)


def build_lpips(device, checkpoint=True, chunk=0, net="vgg"):
    """Build the perceptual loss used by both the pixel and the render terms."""
    import lpips
    fn = _FP32LPIPS(lpips.LPIPS(net=net).to(device).eval())
    fn = _CheckpointedLPIPS(fn, enabled=checkpoint, chunk=chunk)
    fn.requires_grad_(False)
    return fn
