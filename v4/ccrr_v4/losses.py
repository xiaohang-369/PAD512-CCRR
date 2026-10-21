"""Fixed-author-code loss arithmetic, not the paper's differing equations.

Derived from upstream main.py; see retained author LICENSE.txt.
VGG feature tensors must come from the separately verified author VGG weights.
"""

import torch
from torch.nn import functional as F


def legacy_resize_bilinear(x, size):
    """TF1 resize_images bilinear, align_corners=False, NO half-pixel centers."""
    oh, ow = size
    h, w = x.shape[-2:]
    yy = torch.arange(oh, device=x.device, dtype=x.dtype) * (h / oh)
    xx = torch.arange(ow, device=x.device, dtype=x.dtype) * (w / ow)
    y0, x0 = yy.floor().long(), xx.floor().long()
    y1, x1 = (y0 + 1).clamp_max(h - 1), (x0 + 1).clamp_max(w - 1)
    wy = (yy - y0).view(1, 1, -1, 1)
    wx = (xx - x0).view(1, 1, 1, -1)
    a = x[:, :, y0[:, None], x0[None, :]]
    b = x[:, :, y0[:, None], x1[None, :]]
    c = x[:, :, y1[:, None], x0[None, :]]
    d = x[:, :, y1[:, None], x1[None, :]]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy


def source_terms(x, g, rgb, *, distributed=False):
    y = (x * x.new_tensor([0.2989, 0.5870, 0.1140])[None, :, None, None]).sum(
        1, keepdim=True
    )

    def tv(t):
        h, w = t.shape[-2:]
        return (
            (t[..., 1:, :] - t[..., :-1, :]).abs().sum((1, 2, 3))
            + (t[..., :, 1:] - t[..., :, :-1]).abs().sum((1, 2, 3))
        ).mean() / (h * w)

    # min distance to the 256 levels. Minimum ties use equal-gradient averaging,
    # as TensorFlow reduce_min does; use the two neighboring grid levels to avoid N*256*H*W.
    lo = (((g + 1) * 127.5).floor()).clamp(0, 255)
    hi = (lo + 1).clamp(0, 255)
    # Upstream creates its grid with NumPy float64 and TF then casts to the
    # latent tensor dtype. Computing the grid in float32 changes boundary values.
    levels = (torch.arange(256, device=g.device, dtype=torch.float64) / 127.5 - 1).to(
        g.dtype
    )
    quant = torch.minimum(
        (g - levels[lo.long()]).abs(), (g - levels[hi.long()]).abs()
    ).mean()
    light_delta = g.sub(y).abs()
    threshold = g.new_tensor(70 / 127.0)
    variation_delta = tv(g) - tv(y)
    if distributed:
        import torch.distributed as dist
        from torch.distributed.nn.functional import all_reduce

        # The author's abs is AFTER the global batch mean, not rank-local abs.
        variation_delta = all_reduce(variation_delta) / dist.get_world_size()
    return dict(
        mse=F.mse_loss(rgb, x),
        variation=variation_delta.abs(),
        # TF maximum chooses its first (constant) operand at equality.
        lightness=torch.where(light_delta > threshold, light_delta, threshold).mean()
        - threshold,
        quantization=quant,
    )


def author_loss(
    x, g, rgb, target_features, gray_features, stage, weights, *, distributed=False
):
    if stage not in (1, 2):
        raise ValueError("Author reference has exactly two stages")
    if target_features is None or gray_features is None:
        raise ValueError("Author VGG features are mandatory; no silent substitute")
    terms = source_terms(x, g, rgb, distributed=distributed)
    terms["vgg"] = F.mse_loss(gray_features, target_features.detach())
    total = (
        weights["reconstruction"] * terms["mse"]
        + weights["vgg"] * terms["vgg"]
        + weights["lightness"] * terms["lightness"]
        + weights["variation"] * terms["variation"]
        + weights["quantization"] * terms["quantization"]
    )
    return total, terms
