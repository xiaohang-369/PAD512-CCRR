"""Teacher resize: native forward with a deterministic first-order CUDA VJP.

PyTorch 2.6 has no deterministic CUDA antialiased-bilinear backward. Keep
its native forward pixels, but evaluate the same linear resize's adjoint on
CPU when an input gradient is needed under strict determinism. This costs
host/device transfers; it is not a straight-through gradient or a new filter.
The frozen PIL dataset preprocessing is separate and unchanged.
"""

from pathlib import Path
import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from ccrr.config import sha256


POLICY = "native-bilinear-aa-forward-cpu-vjp-v1"


def teacher_input_sources():
    folder = Path(__file__).parent
    return {name: sha256(folder / name) for name in ("pad.py", "resize.py")}


def _native(x, size):
    return F.interpolate(
        x, size=size, mode="bilinear", align_corners=False, antialias=True
    )


class _CPUResizeVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size):
        ctx.input_shape = tuple(x.shape)
        ctx.size = size
        return _native(x, size)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Resize is linear, so its Jacobian does not depend on image values.
        # Recompute on zeros instead of retaining/copying patient image pixels.
        with torch.enable_grad():
            reference = torch.zeros(
                ctx.input_shape,
                dtype=grad_output.dtype,
                device="cpu",
                requires_grad=True,
            )
            result = _native(reference, ctx.size)
            (grad_input,) = torch.autograd.grad(
                result, reference, grad_output.detach().cpu()
            )
        return grad_input.to(grad_output.device), None


def teacher_resize(x, size):
    size = tuple(size)
    if tuple(x.shape[-2:]) == size:
        # Identity resize otherwise registers a CUDA AA backward unnecessarily.
        return x
    if (
        x.is_cuda
        and x.requires_grad
        and torch.is_grad_enabled()
        and torch.are_deterministic_algorithms_enabled()
    ):
        return _CPUResizeVJP.apply(x, size)
    return _native(x, size)
