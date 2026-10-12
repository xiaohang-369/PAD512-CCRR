"""Checker regression with synthetic features, not real teacher evidence."""

import os
from pathlib import Path
import runpy
import pytest
import torch
from torch import nn
from torch.nn import functional as F


class SyntheticIdentity(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.anchor = nn.Parameter(
            torch.zeros((), dtype=torch.float64), requires_grad=False
        )
        self.scale = scale

    def forward(self, x, rects):
        # Float64 also exposes the old CPU float32 expected-value bug without CUDA.
        signal = x.to(self.anchor).mean((1, 2, 3))[:, None]
        axis = torch.linspace(0.25, 1.25, 2048, device=x.device, dtype=signal.dtype)
        features = signal * axis + (2 - axis)
        return {
            "embedding": self.scale * F.normalize(features, dim=1),
            "spatial": features[:, :, None, None].expand(-1, -1, 16, 16),
            "coverage": x.new_ones((1, 1, 16, 16)),
            "logits": features[:, :863],
            "transform": rects,
        }


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_identity_checker_device_dtype_and_normalization_gate(scale):
    script = Path(__file__).parents[1] / "scripts/check_teacher_interface.py"
    check = runpy.run_path(str(script))["check_inputs"]
    model = (
        SyntheticIdentity(scale).to(os.environ.get("CCRR_TEST_DEVICE", "cpu")).eval()
    )
    if scale != 1.0:
        with pytest.raises(AssertionError, match="not close"):
            check(model, "resnet50")
        return
    cases = check(model, "resnet50")
    assert len(cases) == 4
    assert {c["domain"] for c in cases} == {"rgb", "gray"}
    assert all(c["finite"] and c["input_gradient_l1"] > 0 for c in cases)
    assert all(c["frozen_parameter_gradients_absent"] for c in cases)
