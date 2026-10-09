import math
import pytest
import torch
from ccrr.training.losses import js_divergence
from ccrr.calibration import probes


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_js_identical_disjoint_and_extreme_logits(dtype):
    a = torch.tensor(
        [[0, -torch.inf, -torch.inf], [1e4, -1e4, 0]], dtype=dtype, requires_grad=True
    )
    b = torch.tensor(
        [[-torch.inf, 0, -torch.inf], [-1e4, 1e4, 0]], dtype=dtype, requires_grad=True
    )
    assert torch.equal(js_divergence(a, a), torch.zeros(2, dtype=dtype))
    torch.testing.assert_close(js_divergence(a, b), torch.ones(2, dtype=dtype))
    js_divergence(a, b).sum().backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_js_random_matches_probability_definition_and_symmetry():
    torch.manual_seed(43)
    a, b = (
        torch.randn(100, 6, dtype=torch.float64),
        torch.randn(100, 6, dtype=torch.float64),
    )
    p, q = a.softmax(-1), b.softmax(-1)
    m = (p + q) / 2
    expected = ((p * (p / m).log()).sum(-1) + (q * (q / m).log()).sum(-1)) / (
        2 * math.log(2)
    )
    actual = js_divergence(a, b)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual, js_divergence(b, a))
    torch.testing.assert_close(actual, js_divergence(a + 20, b - 30))
    assert ((actual >= 0) & (actual <= 1)).all()


def test_js_gradcheck_near_equal_stability_and_authoritative_probe_import():
    torch.manual_seed(1)
    a = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(js_divergence, (a, b))
    value = js_divergence(a, a.detach() + 1e-8 * b)
    assert ((value >= 0) & (value <= 1)).all()
    value.sum().backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    assert probes.js_divergence is js_divergence


def test_probe_absolute_thresholds_keep_normalized_units():
    def samples(js):
        return {
            "success": [True] * 64,
            "maximum": [0] * 64,
            "rmse": [0] * 64,
            "js": [js] * 32,
        }

    passing = {k: samples(0.001) for k in range(5)}
    assert probes.labels(passing)["kstar"] == 4
    failing = {k: samples(0.0011) for k in range(5)}
    assert probes.labels(failing)["kstar"] == 0
