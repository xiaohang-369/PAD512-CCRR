"""Bounded numerical checks; opt into one visible CUDA device via CCRR_TEST_DEVICE."""

import math
import os
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from ccrr_v3.models import GaussianMixture, Separator, supported
from ccrr_v3.learning import initialize_attacker
from ccrr_v3.p3 import ncc_vector, source_match
from ccrr_v3.link_audit import embedding_report, run as link_audit


@pytest.fixture
def device():
    selected = torch.device(os.environ.get("CCRR_TEST_DEVICE", "cpu"))
    if selected.type == "cuda":
        assert torch.cuda.is_available(), "Requested CUDA is unavailable"
        assert torch.cuda.device_count() == 1, "Expose only the selected physical GPU"
        assert selected.index in (None, 0)
    elif selected.type != "cpu":
        raise ValueError("These regression checks support CPU or selected CUDA only")
    return selected


@pytest.mark.parametrize("variant", ["compression", "separation", "adversarial"])
def test_support_blocks_input_and_code_bypass(device, variant):
    m = Separator(8, 8, 4, 6, variant=variant).to(device).eval()
    x = torch.randn(2, 8, 4, 4, device=device, requires_grad=True)
    v = torch.ones(2, 1, 4, 4, device=device)
    v[:, :, 0] = 0
    v[:, :, 1] = 0.25
    changed = x.detach().clone()
    changed[:, :, 0] += 100
    a, b = m(x, v), m(changed, v)
    for key in a:
        assert torch.equal(a[key], b[key]), key
    invalid = (v == 0).expand_as(a["q"])
    for key in ("q", "q_sem", "q_link", "mask"):
        if key in a:
            assert a[key][invalid].count_nonzero() == 0
    assert torch.equal(a["q"], m.complete(x, v)["q"])
    assert torch.equal(a["full"], m.complete(x, v)["full"])
    poisoned = a["q"].detach().clone().masked_fill(invalid, 99)
    assert torch.equal(m.decode(poisoned, v), a["full"])
    (a["full"].square() * v).sum().backward()
    assert x.grad[(v == 0).expand_as(x)].count_nonzero() == 0
    assert x.grad.abs().sum() > 0
    # Fractional coverage weights losses, never attenuates surviving feature values.
    ones = supported(torch.ones_like(x), v)
    assert torch.equal(ones[:, :, 1], torch.ones_like(ones[:, :, 1]))


@pytest.mark.parametrize("q", [-5.0, 5.0])
def test_representable_tail_above_floor_keeps_gradient(device, q):
    gm = GaussianMixture(1).to(device)
    with torch.no_grad():
        gm.raw_scales.fill_(math.log(math.expm1(0.8 - gm.min_scale)))
    x = torch.tensor([[[[q]]]], device=device, requires_grad=True)
    bits, clamped = gm(x)
    a, b = (abs(q) - 0.5) / 0.8, (abs(q) + 0.5) / 0.8
    mass = 0.5 * (math.erfc(a / math.sqrt(2)) - math.erfc(b / math.sqrt(2)))
    density_a = math.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    density_b = math.exp(-b * b / 2) / math.sqrt(2 * math.pi)
    grad = math.copysign((density_a - density_b) / (0.8 * mass * math.log(2)), q)
    assert mass > gm.min_probability and not clamped.any()
    bits.sum().backward()
    assert bits.item() == pytest.approx(-math.log2(mass), rel=2e-6)
    assert x.grad.item() == pytest.approx(grad, rel=2e-5)


def test_mixture_probabilities_and_all_gradients_match_float64(device):
    gm = GaussianMixture(1).to(device)
    with torch.no_grad():
        gm.means.copy_(
            torch.tensor([-0.3, 0.1, 0.5], device=device).reshape_as(gm.means)
        )
        gm.logits.copy_(
            torch.tensor([0.2, -0.4, 0.7], device=device).reshape_as(gm.logits)
        )
    ref = deepcopy(gm).double()
    values = [-100.0, -7.0, -5.0, -1.0, 0.0, 1.0, 5.0, 7.0, 100.0]
    x = torch.tensor(values, device=device).reshape(1, 1, 1, -1).requires_grad_()
    y = x.detach().double().requires_grad_()
    bits, clipped = gm(x)
    expected, expected_clipped = ref(y)
    assert torch.equal(clipped, expected_clipped)
    assert torch.allclose(bits.double(), expected, atol=1e-5, rtol=1e-5)
    bits.sum().backward()
    expected.sum().backward()
    for left, right in [(x, y), *zip(gm.parameters(), ref.parameters())]:
        assert torch.isfinite(left.grad).all()
        assert torch.allclose(left.grad.double(), right.grad, atol=2e-5, rtol=2e-4)
    assert x.grad.flatten()[0] == 0 and x.grad.flatten()[-1] == 0


def test_attacker_construction_preserves_cpu_and_selected_device_rng(device):
    cpu_before = torch.get_rng_state().clone()
    gpu_before = (
        torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    )
    # Reject accidental reintroduction of global multi-device seeding.
    with (
        patch("torch.manual_seed", side_effect=AssertionError("global seed")),
        patch(
            "torch.cuda.manual_seed_all", side_effect=AssertionError("CUDA seed all")
        ),
    ):
        a = initialize_attacker(8, 1234, device)
        b = initialize_attacker(8, 1234, device)
    assert torch.equal(cpu_before, torch.get_rng_state())
    if gpu_before is not None:
        assert torch.equal(gpu_before, torch.cuda.get_rng_state(device))
    assert all(
        torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()
    )


def test_p3_public_and_assisted_information_conditions_are_distinct():
    image = torch.arange(16.0).reshape(1, 4, 4)
    v = torch.ones_like(image)
    v[:, 0] = 0
    public, assisted = ncc_vector(image), ncc_vector(image, v)
    assert torch.equal(public, (image - image.mean()).flatten())
    assert not torch.equal(public, assisted)
    for use_support, vector in ((False, public), (True, assisted)):
        report = source_match(
            torch.stack([vector, vector]),
            ["a", "b"],
            ["p", "q"],
            source_support_known=use_support,
        )
        assert report["source_support_used"] == use_support
        assert report["generated_query_evaluated"] is False
        assert report["rank1"] == pytest.approx(0.5)


def test_p3_run_keeps_public_report_independent_of_source_support(
    tmp_path, cfg, monkeypatch
):
    from ccrr_v3 import p3, runtime

    class Data:
        rows = [
            {"img_id": "a", "patient_id": "p"},
            {"img_id": "b", "patient_id": "q"},
        ]
        valid = torch.ones(1, 4, 4)

        def __len__(self):
            return 2

        def __getitem__(self, i):
            x = torch.arange(48.0).reshape(3, 4, 4) / 48
            return {"x": x if i == 0 else x.flip(-1), "v": self.valid}

    monkeypatch.setattr(p3, "PADData", lambda *args: Data())
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    p3.run({"data_root": "unused"}, cfg, first)
    Data.valid[:, 0] = 0
    p3.run({"data_root": "unused"}, cfg, second)
    assert runtime.read(first / "p3_baseline.json") == runtime.read(
        second / "p3_baseline.json"
    )
    assert runtime.read(second / "p3_source_support_auxiliary.json")[
        "source_support_used"
    ]


def test_link_audit_perfect_and_collapsed_embeddings(cache):
    data = cache["val_tune"]
    teacher = torch.eye(6).repeat_interleave(2, 0)[:, None].expand(-1, 2, -1)
    perfect = embedding_report(data, teacher, teacher, 0)
    collapsed = embedding_report(data, torch.ones_like(teacher), teacher, 0)
    zero = embedding_report(data, torch.zeros_like(teacher), teacher, 0)
    assert perfect["normalized_squared_error"]["rgb"] == 0
    for name in perfect["p2"]:
        assert perfect["p2"][name]["rank1"] == 1
        assert collapsed["p2"][name]["auroc"] == 0.5
        assert collapsed["p2"][name]["rank1"] == pytest.approx(
            collapsed["p2"][name]["chance_rank1"]
        )
        assert zero["p2"][name]["auroc"] == 0.5
    assert zero["near_zero_norm_count"]["rgb"] == len(data["rows"])


def test_link_audit_uses_fit_mean_and_never_updates_model(cache, cfg, tmp_path):
    model = Separator(8, 8, 4, 6).eval()
    before = deepcopy(model.state_dict())
    targets = {
        role: torch.eye(6).repeat_interleave(2, 0)[:, None].expand(-1, 2, -1).clone()
        for role in cache
    }
    # Make the fit mean e0; held-out means remain uniform.
    targets["fit"].zero_()
    targets["fit"][:, :, 0] = 1
    report = link_audit(cache, targets, model, cfg, tmp_path, "cpu")
    for role in ("val_tune", "val_select"):
        mean = report["roles"][role]["fit_mean"]
        assert mean["normalized_squared_error"]["rgb"] == pytest.approx(10 / 6)
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)
    assert all(p.grad is None for p in model.parameters())
    assert report["training_started"] is False
