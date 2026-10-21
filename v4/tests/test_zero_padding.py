"""No-copy SAME identity and full E/D output/gradient regression against old code."""

import math

import pytest
import torch
from torch.nn import functional as F

from ccrr_v4 import original


def legacy_same_pad(x, kernel, stride):
    h, w = x.shape[-2:]
    ph = max((math.ceil(h / stride) - 1) * stride + kernel - h, 0)
    pw = max((math.ceil(w / stride) - 1) * stride + kernel - w, 0)
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


@pytest.mark.parametrize("transposed", [False, True])
def test_zero_padding_reuses_storage_and_preserves_gradient(transposed):
    x = torch.randn(2, 256, 8, 12)
    if transposed:
        x = x.transpose(-1, -2)
    x.requires_grad_()
    result = original.same_pad(x, 1, 1)
    legacy = legacy_same_pad(x, 1, 1)
    assert result is x
    assert result.data_ptr() == x.data_ptr()
    assert legacy.data_ptr() != x.data_ptr()
    assert torch.equal(result, legacy)
    upstream = torch.randn_like(x)
    assert torch.equal(
        torch.autograd.grad(result, x, upstream)[0],
        torch.autograd.grad(legacy, x, upstream)[0],
    )


def test_full_ed_forward_input_and_all_parameter_gradients_unchanged(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(17)
        model = original.OriginalGrayscale()
        x = torch.randn(2, 3, 8, 12, requires_grad=True)
        g, rgb = model(x)
        # Both outputs participate, so every original E/D parameter is checked.
        loss = (rgb - x).square().mean() + g.square().mean()
        params = tuple(model.parameters())
        gradients = torch.autograd.grad(loss, (x, *params))
        monkeypatch.setattr(original, "same_pad", legacy_same_pad)
        g_old, rgb_old = model(x)
        loss_old = (rgb_old - x).square().mean() + g_old.square().mean()
        gradients_old = torch.autograd.grad(loss_old, (x, *params))
        assert torch.equal(g, g_old)
        assert torch.equal(rgb, rgb_old)
        assert torch.equal(loss, loss_old)
        assert len(params) == 45
        assert all(torch.equal(a, b) for a, b in zip(gradients, gradients_old))
    finally:
        torch.set_num_threads(old_threads)
