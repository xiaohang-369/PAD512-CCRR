"""Private latent conversion shared by learning, source coding and evaluation.

The step is the frozen Profile.step(level). Rounding is the original signed
half-away-from-zero STE; this module deliberately preserves its arithmetic.
Range and source-shape validation remain at the codec boundary, because neural
training may inspect out-of-range candidates before rejecting publication.
Purifier feature quantization has a separate range and uses ste_round directly.
"""

from ccrr.tensor_ops import ste_round


def quantize_latent(z, step):
    """Return integer-valued latent with the existing straight-through gradient."""
    return ste_round(z / step)


def dequantize_latent(n, step):
    """Map authenticated integers (or training surrogates) back to INN units."""
    return n * step
