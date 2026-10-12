"""Synthetic resize equivalence; set CCRR_TEST_DEVICE=cuda for target checks."""

import os
from pathlib import Path
import json
import pytest
import torch
from torch.nn import functional as F
from ccrr.config import sha256
from ccrr.data import pad
from ccrr.data.resize import (
    POLICY,
    _CPUResizeVJP,
    teacher_input_sources,
    teacher_resize,
)
from ccrr.runtime import preflight


@pytest.fixture(autouse=True)
def strict_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn)


def native(x, size):
    return F.interpolate(
        x, size=size, mode="bilinear", align_corners=False, antialias=True
    )


@pytest.mark.parametrize("size", [(5, 7), (17, 19)])
def test_cpu_vjp_matches_native_and_finite_differences(size):
    x = torch.randn(1, 1, 9, 13, dtype=torch.float64, requires_grad=True)
    reference = native(x, size)
    actual = _CPUResizeVJP.apply(x, size)
    assert torch.equal(actual, reference)
    probe = torch.randn_like(reference)
    a = torch.autograd.grad(actual, x, probe)[0]
    b = torch.autograd.grad(reference, x, probe)[0]
    assert torch.equal(a, b)
    assert torch.autograd.gradcheck(
        lambda value: _CPUResizeVJP.apply(value, size), (x,)
    )


@pytest.mark.parametrize("channels", [1, 3])
def test_identity_resize_preserves_pixels_and_input_gradient(channels):
    device = os.environ.get("CCRR_TEST_DEVICE", "cpu")
    x = torch.rand(1, channels, 512, 512, device=device, requires_grad=True)
    crop = x[..., 64:448, :]
    y = teacher_resize(crop, (384, 512))
    assert y is crop
    assert torch.equal(y, native(crop.detach(), (384, 512)))
    grad = torch.autograd.grad(y.sum(), x)[0]
    expected = torch.zeros_like(x)
    expected[..., 64:448, :] = 1
    assert torch.equal(grad, expected)


@pytest.mark.parametrize("size", [(5, 7), (17, 19)])
def test_input_vjp_matches_native_forward_directional_derivative(size):
    device = os.environ.get("CCRR_TEST_DEVICE", "cpu")
    x = torch.randn(1, 1, 9, 13, device=device, dtype=torch.float64, requires_grad=True)
    direction = torch.randn_like(x)
    result = teacher_resize(x, size)
    probe = torch.randn_like(result)
    gradient = torch.autograd.grad(result, x, probe)[0]
    with torch.no_grad():
        eps = 1e-4
        finite_difference = (
            (native(x + eps * direction, size) - native(x - eps * direction, size))
            * probe
        ).sum() / (2 * eps)
    torch.testing.assert_close(
        (gradient * direction).sum(), finite_difference, atol=1e-7, rtol=1e-6
    )


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("rect", [(0, 0, 512, 512), (0, 64, 512, 384)])
def test_panderm_resize_native_forward_cpu_vjp_and_repeatability(channels, rect):
    device = os.environ.get("CCRR_TEST_DEVICE", "cpu")
    x = torch.rand(1, channels, 512, 512, device=device, requires_grad=True)
    actual, mask, trans = pad.adapt(x, [rect], 224, panderm=True)
    # The no-grad path uses the original native device forward.
    with torch.no_grad():
        native_result, native_mask, native_trans = pad.adapt(
            x, [rect], 224, panderm=True
        )
    assert torch.equal(actual, native_result)
    assert torch.equal(mask, native_mask) and trans == native_trans
    probe = torch.randn_like(actual)
    grad = torch.autograd.grad(actual, x, probe)[0]
    reference = x.detach().cpu().requires_grad_(True)
    expected, _, _ = pad.adapt(reference, [rect], 224, panderm=True)
    cpu_grad = torch.autograd.grad(expected, reference, probe.cpu())[0]
    torch.testing.assert_close(grad.cpu(), cpu_grad, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    repeated, _, _ = pad.adapt(x, [rect], 224, panderm=True)
    repeated_grad = torch.autograd.grad(repeated, x, probe)[0]
    assert torch.equal(actual, repeated)
    assert torch.equal(grad, repeated_grad)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()


def test_interface_evidence_rejects_stale_teacher_preprocessing(tmp_path):
    path = tmp_path / "interface.json"
    name = "efficientnet_b0"
    weights = {"sha256": "fixture-digest"}
    report = {
        "passed": True,
        "model": name,
        **weights,
        "adapter_source_sha256": sha256(
            Path(preflight.__file__).parents[1] / "teachers/adapters.py"
        ),
        "teacher_input_sources": teacher_input_sources(),
        "teacher_resize_policy": POLICY,
        "device": "cpu",
        "torch": "2.6.0",
        "parameters_and_buffers_unchanged": True,
        "forward_backward": [
            {
                "domain": domain,
                "finite": True,
                "input_gradient_l1": 1,
                "frozen_parameter_gradients_absent": True,
            }
            for domain in ("rgb", "gray", "rgb", "gray")
        ],
    }

    def accepted(value):
        path.write_text(json.dumps(value))
        config = {
            "device": "cpu",
            "interface_reports": {name: {"path": str(path), "sha256": sha256(path)}},
        }
        return preflight.interface_evidence(
            weights, name, config, {"packages": {"torch": "2.6.0"}}
        )

    assert accepted(report)
    for key in ("pad.py", "resize.py"):
        stale = {
            **report,
            "teacher_input_sources": {
                **report["teacher_input_sources"],
                key: "0" * 64,
            },
        }
        assert not accepted(stale)
    assert not accepted({**report, "teacher_resize_policy": "old"})
    assert not accepted(
        {k: v for k, v in report.items() if k != "teacher_input_sources"}
    )
