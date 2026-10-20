from copy import deepcopy

import numpy as np
import pytest
import torch

from ccrr_v3 import learning, runtime
from ccrr_v3.models import ClinicalHeads
from ccrr_v3.selection import objectives, pareto_indices, summarize, write_tables
from ccrr_v3.dev2 import configs, STEPS


def test_dev2_preserves_old_hyperparameters_and_all_requested_endpoints(cfg):
    cfg.update(warmup_steps=500, joint_steps=2000)
    original = deepcopy(cfg)
    train, audit = configs(cfg, cfg)
    assert cfg == original
    assert len(STEPS) == 26 and 1250 in STEPS and 1750 in STEPS and 2500 in STEPS
    assert STEPS[-1] == 10000
    assert train["warmup_steps"] == original["warmup_steps"]
    for key in ("seed", "lr", "weights", "head_mode", "batch_size"):
        assert train[key] == audit[key] == original[key]


@pytest.mark.parametrize("steps", [[2, 1], [1, 1], [0], [4], [True], "1"])
def test_checkpoint_config_rejects_ambiguous_schedule(cfg, steps):
    cfg["checkpoint_steps"] = steps
    with pytest.raises(ValueError, match="checkpoint_steps"):
        runtime.validate_config(cfg)


def test_dev2_temperature_preserves_dev1_prefix(cfg):
    cfg.update(warmup_steps=500, joint_steps=2000)
    longer = {**cfg, "joint_steps": 9500, "temperature_decay_steps": 2000}
    assert [learning.temperature_at(cfg, s) for s in range(1, 2501)] == [
        learning.temperature_at(longer, s) for s in range(1, 2501)
    ]
    assert learning.temperature_at(longer, 10000) == .5


@pytest.mark.parametrize("variant", ["compression", "separation"])
def test_checkpoint_and_longer_budget_preserve_prefix(cfg, cache, tmp_path, variant):
    teacher = ClinicalHeads(8)
    initial = deepcopy(teacher.state_dict())
    targets = None if variant == "compression" else {
        role: torch.randn(len(x["rows"]), 2, 6) for role, x in cache.items()
    }
    statistics = dict(mean=torch.zeros(8), std=torch.ones(8))
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    learning.train(cache, targets, teacher, statistics, cfg, old, "cpu", variant)
    extended = {**cfg, "joint_steps": 4, "temperature_decay_steps": 2,
                "checkpoint_steps": [1, 2, 3, 4, 5]}
    learning.train(cache, targets, teacher, statistics, extended, new, "cpu", variant)
    before = torch.load(old / "separator.pt", weights_only=True)
    after = torch.load(learning.checkpoint_path(new, 3), weights_only=True)
    for k in before["model"]:
        assert torch.equal(before["model"][k], after["model"][k]), k
    assert torch.equal(before["torch_rng"], after["torch_rng"])
    assert before["medical_rng"] == after["medical_rng"]
    assert all(torch.equal(v, teacher.state_dict()[k]) for k, v in initial.items())
    assert len(runtime.read(new / "checkpoints.json")["checkpoints"]) == 5
    with pytest.raises(ValueError, match="not registered"):
        learning.checkpoint_path(new, 100)
    report = runtime.read(new / "checkpoints.json")["checkpoints"][2]["clinical"]
    assert set(report["semantic"]["gray"]) == {"macro_auroc", "macro_f1", "recall", "class_n"}


def test_operator_stop_finishes_at_saved_checkpoint(cfg, cache, tmp_path, monkeypatch):
    cfg["checkpoint_steps"] = [2, 3]
    original_log = learning.log

    def stop(output, event, **values):
        original_log(output, event, **values)
        if event == "separator_update" and values["step"] == 2:
            (output / "STOP_AFTER_CHECKPOINT").touch()

    monkeypatch.setattr(learning, "log", stop)
    learning.train(cache, None, ClinicalHeads(8),
                   dict(mean=torch.zeros(8), std=torch.ones(8)), cfg,
                   tmp_path, "cpu", "compression")
    result = runtime.read(tmp_path / "training_summary.json")
    assert result["steps"] == 2 and result["maximum_steps"] == 3
    assert result["endpoint"] == "operator_stop"
    assert not (tmp_path / "separator-step-000003.pt").exists()


def reports(medical=.85, attack=.8):
    clinical = dict(macro_auroc=medical, macro_f1=.6, recall=[.5] * 6)
    identity = dict(auroc=attack, rank1=.3)
    return {
        **{h: {d: deepcopy(clinical) for d in ("rgb", "gray")} for h in ("linear", "mlp")},
        **{h: {d: deepcopy(identity) for d in ("rgb_rgb", "gray_rgb", "gray_gray")}
           for h in ("global", "spatial")},
    }


def test_pareto_preserves_tradeoffs_and_all_class_recalls(tmp_path):
    a, b, c = reports(.9, .9), reports(.85, .7), reports(.8, .95)
    names, vector = objectives(a)
    assert len(names) == len(vector) == 44
    assert pareto_indices([objectives(r)[1] for r in (a, b, c)]) == [0, 1]
    # A rare-class benefit protects C from domination, despite worse averages.
    c["mlp"]["gray"]["recall"][2] = .9
    assert pareto_indices([objectives(r)[1] for r in (a, b, c)]) == [0, 1, 2]
    entries = [dict(variant=v, step=i+1, reports=r)
               for v in ("compression", "separation") for i, r in enumerate((a, b, c))]
    result = summarize(entries, limit=2)
    assert result["winner"] is None
    assert len(result["representative_indices"]) == 4
    assert all(len(x) == 3 for x in result["pareto_indices_by_variant"].values())
    write_tables(tmp_path, result)
    assert (tmp_path / "metrics.csv").read_text().count("\n") == 7
    assert "No automatic winner" in (tmp_path / "tradeoffs.html").read_text()
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([entries[0], entries[0]])
    broken = reports()
    broken["linear"]["rgb"]["macro_f1"] = np.nan
    with pytest.raises(ValueError, match="Undefined"):
        objectives(broken)
    inverted = reports(attack=.1)
    assert objectives(inverted)[1][32] == .9
