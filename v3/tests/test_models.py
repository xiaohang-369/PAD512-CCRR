import pytest
import torch

from ccrr_v3.models import (
    Separator,
    GaussianMixture,
    quantize,
    weighted_mean,
    ClinicalHeads,
    FrozenSupervisor,
    freeze,
)
from ccrr_v3.learning import core_objective


def small(variant="separation"):
    return Separator(
        input_channels=8, width=8, code_channels=4, embedding_dim=6, variant=variant
    )


def test_quantization_hard_ties_and_gradient():
    x = torch.tensor([-1.5, -0.5, 0.49, 0.5, 1.5], requires_grad=True)
    y = quantize(x)
    assert y.tolist() == [-1, 0, 0, 1, 2]
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_mask_complements_hard_eval_and_input_gradient():
    m = small().eval()
    x = torch.randn(2, 8, 4, 4, requires_grad=True)
    v = torch.ones(2, 1, 4, 4)
    a, b = m(x, v), m(x, v)
    assert set(a["mask"].unique().tolist()) <= {0, 1}
    assert torch.equal(a["q_sem"] + a["q_link"], a["q"])
    assert torch.equal(a["semantic"], b["semantic"])
    a["semantic"].square().mean().backward()
    assert x.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="deterministic"):
        m(x, v, stochastic=True)


def test_mask_equal_logits_choose_semantic():
    m = small().eval()
    m.mask_net.weight.data.zero_()
    m.mask_net.bias.data.zero_()
    out = m(torch.randn(2, 8, 4, 4), torch.ones(2, 1, 4, 4))
    assert out["mask"].count_nonzero() == 0


def test_gmm_matches_integrated_normal_and_gradients():
    gm = GaussianMixture(1)
    gm.raw_scales.data.zero_()
    x = torch.tensor([[[[0.0, 1.0, -1.0]]]], requires_grad=True)
    bits, clipped = gm(x)
    scale = torch.nn.functional.softplus(torch.tensor(0.0)) + gm.min_scale
    expected = torch.distributions.Normal(0, scale).cdf(
        x + 0.5
    ) - torch.distributions.Normal(0, scale).cdf(x - 0.5)
    assert torch.allclose(bits, -expected.log2(), atol=1e-5)
    assert not clipped.any()
    bits.sum().backward()
    assert x.grad[0, 0, 0, 1].abs() > 0
    assert gm.raw_scales.grad.abs().sum() > 0
    far, clamped = gm(torch.full_like(x, 1e6))
    assert torch.isfinite(far).all() and clamped.all()


def test_rate_counts_masked_zeros_with_same_gmm_and_denominator():
    m = small()
    q = torch.ones(2, 4, 4, 4)
    mask = torch.zeros_like(q)
    v = torch.ones(2, 1, 4, 4)
    v[:, :, 0] = 0
    rate, _ = m.rate(dict(q=q, mask=mask, q_link=q * mask), v)
    expected = weighted_mean(m.entropy(q)[0], v) + weighted_mean(m.entropy(q * 0)[0], v)
    assert torch.allclose(rate, expected) and rate > 0
    with pytest.raises(ValueError, match="Empty"):
        weighted_mean(q, v * 0)


def test_frozen_supervisor_remains_deterministic_under_parent_train():
    model, heads = small(), ClinicalHeads(8)
    supervisor = FrozenSupervisor(model, heads, torch.zeros(8), torch.ones(8))
    wrapper = torch.nn.Sequential(supervisor)
    wrapper.train()
    assert not model.training and not heads.training
    raw = torch.randn(2, 8, 4, 4, requires_grad=True)
    v = torch.ones(2, 1, 4, 4)
    logits, _ = supervisor(raw, v, 1)
    logits.square().mean().backward()
    assert raw.grad.abs().sum() > 0
    assert all(p.grad is None for p in supervisor.parameters())


def test_full_reconstruction_not_semantic_reconstruction(cfg):
    m = small()
    head = freeze(ClinicalHeads(8))
    x = torch.randn(2, 8, 4, 4)
    v = torch.ones(2, 1, 4, 4)
    out = m(x, v)
    out["semantic"] = torch.zeros_like(out["semantic"])
    _, terms = core_objective(
        m,
        out,
        x,
        v,
        torch.tensor([0, 1]),
        head,
        0,
        torch.randn(2, 6),
        cfg["weights"],
        torch.ones(6),
    )
    assert torch.allclose(terms["all"], weighted_mean((out["full"] - x).square(), v))


def test_independent_compression_has_no_mask_or_link_decoder():
    m = small("compression")
    assert not hasattr(m, "mask_net") and not hasattr(m, "link_decoder")
    out = m(torch.randn(2, 8, 4, 4), torch.ones(2, 1, 4, 4))
    assert torch.equal(out["semantic"], out["full"])
    assert "q_link" not in out
