"""Diagnostic isolation and protocol accounting; no medical capability claim."""

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ccrr.errors import CCRRError
from ccrr.training.b_diagnostics import (
    branch_statistics,
    compare_attack_scores,
    isolated_rng,
    post_adapted_attack,
)
from ccrr.training.feature_audit import FeatureAudit
from ccrr.training.optimization import optimizer
from test_stage_b_execution import context
from test_stage_a_training_loops import compare


def scores(values):
    return [
        {
            "domain": [a, b],
            "scores": list(values),
            "labels": [0, 0, 1, 1],
            "pairs": [["a", "b"]] * 4,
            "weights": [1.0] * 4,
        }
        for a in range(3)
        for b in range(3)
    ]


@pytest.mark.parametrize(
    "before,after,flag",
    [
        ([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], False),
        ([0.0, 2.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0], True),
        ([3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0], True),
        ([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], True),
    ],
)
def test_adaptation_flags_keep_auc_and_reverse_separate(before, after, flag):
    result = compare_attack_scores(scores(before), scores(after))
    assert all(r["alternating_attack_insufficient"] is flag for r in result)
    if before[0] == 3:
        assert result[0]["alternating_auroc"] == 0
        assert result[0]["alternating_a_dagger"] == 1


def test_adaptation_rejects_misalignment_and_nonfinite():
    original = scores([0.0, 1.0, 2.0, 3.0])
    bad = copy.deepcopy(original)
    bad[0]["weights"][0] = 2
    with pytest.raises(CCRRError, match="alignment"):
        compare_attack_scores(original, bad)
    bad = scores([float("nan"), 1.0, 2.0, 3.0])
    with pytest.raises(CCRRError, match="Nonfinite"):
        compare_attack_scores(original, bad)


def test_weighted_branch_statistics_ignore_padding():
    v = torch.tensor([[[[1.0, 0.5, 0.0]]]])
    raw = torch.tensor([[[[1.0, 3.0, 999.0]]]]).expand(1, 2, 1, 3)
    p = {
        "q": torch.tensor([[[[7.0, 0.0, 7.0]]]]),
        "standardized": raw,
        "semantic": raw,
        "all": raw + 2,
        "identity": torch.tensor([[1.0, 0.0]]),
    }
    r = branch_statistics(p, v, torch.tensor([[0.0, 1.0]]))
    assert r["code_saturation_fraction"] == pytest.approx(2 / 3)
    assert r["full_code_reconstruction_mse"] == 4
    assert r["identity_reconstruction_squared_l2"] == 2
    assert r["raw_spatial_variance_channel_mean"] == pytest.approx(8 / 9)


def test_rng_isolation_even_on_failure():
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    with pytest.raises(RuntimeError):
        with isolated_rng():
            random.random()
            np.random.random()
            torch.rand(3)
            raise RuntimeError("interrupted")
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_disposable_adaptation_exact_budget_and_no_feedback(tmp_path):
    # Real diagnostic orchestration, cheap attack stand-in to verify all 1000 calls.
    model = torch.nn.Linear(1, 1).requires_grad_(False).eval()
    saved = copy.deepcopy(model.state_dict())
    state = torch.get_rng_state().clone()
    e = SimpleNamespace(sem_attacker=model)
    calls = []

    def attack_scores(candidate, rep, scenario):
        assert candidate is not model and rep == "sem" and scenario in ("p1", "p2")
        return scores([0.0, 1.0, 2.0, 3.0])

    def attacker_update(candidate, opt, rep, scenario, family, step, **kwargs):
        calls.append(step)
        assert scenario == "mixed" and kwargs["total"] == 1000
        assert kwargs["rng_key"] == "B-post-adapt-mixed"
        opt.zero_grad()
        candidate(torch.ones(1, 1)).sum().backward()
        opt.step()
        torch.rand(1)

    audit = SimpleNamespace(
        e=e, root=tmp_path, attack_scores=attack_scores, attacker_update=attacker_update
    )
    result = post_adapted_attack(audit)
    assert calls == list(range(1000))
    assert result["updates"] == 1000 and not result["generator_updated"]
    assert not result["alternating_attack_insufficient"]
    compare(saved, model.state_dict())
    assert model.training is False and all(
        not p.requires_grad and p.grad is None for p in model.parameters()
    )
    assert torch.equal(state, torch.get_rng_state())
    artifact = torch.load(tmp_path / "post_adapted_attacker.pt", weights_only=True)
    assert artifact["stage"] == "B-diagnostic" and artifact["step"] == 1000
    assert not torch.equal(artifact["models"]["attacker"]["weight"], saved["weight"])


def test_adaptation_update_uses_mixed_sampler_and_frozen_cache(tmp_path):
    e = context(tmp_path)
    for i, row in enumerate(e.fit.rows):
        row["patient_id"] = str(i)
    e.purifier.eval().requires_grad_(False)
    audit = FeatureAudit(e)
    audit.cache()
    assert len(audit.branch_diagnostics) == len(e.data["val_select"])
    saved = copy.deepcopy(e.purifier.state_dict())
    scenarios = []
    sampler = e.sampler.sample

    def sample(rng, scenario=None):
        scenarios.append(scenario)
        return sampler(rng, scenario)

    e.sampler.sample = sample
    candidate = copy.deepcopy(e.sem_attacker).requires_grad_(True)
    opt = optimizer([candidate], [1e-4])
    audit.attacker_update(
        candidate,
        opt,
        "sem",
        "mixed",
        "spatial",
        0,
        total=1000,
        rng_key="B-post-adapt-mixed",
    )
    assert scenarios == ["mixed"]
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-6)
    assert len(opt.state) > 0
    assert all(p.grad is None for p in e.purifier.parameters())
    compare(saved, e.purifier.state_dict())
