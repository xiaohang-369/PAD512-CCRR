import importlib.util
from pathlib import Path
import sys

import pytest
import torch

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "full_fit_probe", SCRIPTS / "check_stage_b_full_fit_reconstruction.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_template_uses_equal_image_coverage_weights_and_ignores_padding():
    # First image has one valid site, second has two. At site 0 the optimal
    # equal-image template is (2*1 + 8*.5)/(1+.5)=4, not unweighted mean 5.
    features = torch.tensor([[[[2.0, 999.0]]], [[[8.0, 10.0]]]])
    v = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 1.0]]]])
    bank = probe.Cache({"f0": features, "f1": features * 2, "v": v})
    templates = probe.fit_templates(
        bank, torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1)
    )
    torch.testing.assert_close(templates["f0"], torch.tensor([[[[4.0, 10.0]]]]))
    torch.testing.assert_close(templates["f1"], templates["f0"] * 2)
    target = features * (v > 0)
    loss = probe.per_image_mse(templates["f0"], target, v).mean()
    # The analytical minimizer must outperform nearby fixed templates.
    for change in (-1.0, 1.0):
        assert probe.per_image_mse(templates["f0"] + change, target, v).mean() > loss


def test_streamed_score_matches_native_loss_without_model_mutation():
    from ccrr.tensor_ops import masked_mean

    model = probe.ReconstructionControl(
        torch.zeros(1, 768, 1, 1), torch.ones(1, 768, 1, 1)
    )
    # Include an incomplete last batch: averaging batch means would be wrong.
    f = torch.randn(17, 768, 4, 4)
    v = torch.ones(17, 1, 4, 4)
    v[..., 0, :] = 0
    v[..., 1, :] = 0.25
    bank = probe.Cache({"f0": f, "f1": f * 0.7, "v": v})
    templates = probe.fit_templates(bank, model.mean, model.std)
    saved = {k: p.clone() for k, p in model.state_dict().items()}
    summary, errors = probe.score(model, bank, templates)
    for d in ("f0", "f1"):
        with torch.no_grad():
            p = model(bank.storage[d], v)
            expected = float(masked_mean((p["all"] - p["standardized"]).square(), v))
        assert summary[d]["model"] == pytest.approx(expected, rel=1e-6)
        assert len(errors[d]["model"]) == 17
        assert summary[d]["mse_over_zero"] == pytest.approx(
            summary[d]["model"] / summary[d]["zero"]
        )
    assert all(torch.equal(saved[k], p) for k, p in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_patient_bootstrap_keeps_image_weighted_point_and_patient_clusters():
    result = probe.bootstrap_difference(
        [0.0, 10.0, 10.0], [0.0, 0.0, 0.0], ["a", "b", "b"]
    )
    assert result["patients"] == 2
    assert result["mean_mse_difference"] == pytest.approx(20 / 3)
    assert result["patient_bootstrap_95_percentile_interval"] == [0.0, 10.0]
    same = probe.bootstrap_difference([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], ["a", "a", "b"])
    assert same["patient_bootstrap_95_percentile_interval"] == [-1.0, -1.0]


def test_full_budget_arms_share_batches_and_do_not_update_base(monkeypatch, tmp_path):
    import copy
    from ccrr.data.pad import rng_for

    base = torch.nn.Linear(1, 1)
    saved = copy.deepcopy(base.state_dict())
    batches, schedule_totals, snapshots = [], [], []
    monkeypatch.setattr(probe, "make_arm", lambda model, name: copy.deepcopy(model))
    monkeypatch.setattr(probe, "score", lambda *args: ({"f0": {}, "f1": {}}, {}))
    monkeypatch.setattr(probe, "measure", lambda *args: {})
    monkeypatch.setattr(probe, "write_json", lambda path, row: None)
    monkeypatch.setattr(
        probe, "log_json", lambda path, row: snapshots.append((row["arm"], row["step"]))
    )
    monkeypatch.setattr(torch, "save", lambda *args: None)

    def update(model, bank, ids, micro, opt, schedule):
        assert micro == len(ids) == 16
        batches.append(ids)
        schedule_totals.append(schedule.total)
        schedule.advance()
        return 1.0

    monkeypatch.setattr(probe, "control_update", update)
    t = torch.ones(5, 1, 1, 1)
    banks = {
        role: probe.Cache({"f0": t, "f1": t, "v": t}) for role in ("fit", "val_tune")
    }
    for name in probe.ARMS:
        probe.run_arm(base, name, banks, {r: [0, 1] for r in banks}, {}, tmp_path)
    assert len(batches) == 20000 and set(schedule_totals) == {10000}
    assert batches[:10000] == batches[10000:]
    assert snapshots == [(a, s) for a in probe.ARMS for s in probe.POINTS]
    assert (
        batches[0]
        == rng_for(probe.SEED, probe.PROTOCOL, "batch", 0).integers(5, size=16).tolist()
    )
    assert all(torch.equal(saved[k], v) for k, v in base.state_dict().items())


def test_prior_evidence_binding_rejects_changed_checkpoints(monkeypatch, tmp_path):
    import copy
    import json
    from ccrr.errors import CCRRError

    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "src/ccrr").mkdir(parents=True)
    old_probe = source / "scripts/probe_stage_b_reconstruction.py"
    old_probe.write_text("# unchanged pilot\n")
    production = source / "src/ccrr/example.py"
    production.write_text("# unchanged production\n")
    monkeypatch.setattr(
        probe,
        "__file__",
        str(source / "scripts/check_stage_b_full_fit_reconstruction.py"),
    )
    config = tmp_path / "config.yaml"
    config.write_text("fixture: true\n")
    baseline = tmp_path / "report.json"
    replicas = tmp_path / "replicas"
    arms = ("original", "encoder_x01", "decoder_x01", "both_x01")
    base = {
        "protocol": "B-R1-DIAG-v1",
        "seed": 20261008,
        "steps_per_arm": 1000,
        "disposable_optimizer_updates": 4000,
        "existing_checkpoints_unchanged": True,
        "formal_training_started": False,
        "scientific_admission_complete": False,
        "config_sha256": probe.sha256(config),
        "checker_sha256": probe.sha256(old_probe),
        "source_sha256": {"src/ccrr/example.py": probe.sha256(production)},
        "cohort": {
            "role": "fit",
            "train_patients": 64,
            "holdout_patients": 16,
            "one_image_per_patient": True,
        },
        "checkpoint_digests": {"A_initial": "unchanged"},
        "arms": {
            "original": [1.0, 1.0],
            "encoder_x01": [0.1, 1.0],
            "decoder_x01": [1.0, 0.1],
            "both_x01": [0.1, 0.1],
        },
        "trajectories": {
            a: [{"arm": a, "step": s} for s in (0, 100, 500, 1000)] for a in arms
        },
    }
    baseline.write_text(json.dumps(base))
    for seed in (20261009, 20261010):
        r = copy.deepcopy(base)
        r["seed"] = seed
        directory = replicas / f"seed-{seed}"
        directory.mkdir(parents=True)
        (directory / "report.json").write_text(json.dumps(r))
    actual, digests = probe.verify_evidence(config, baseline, replicas)
    assert actual == base and len(digests) == 3
    path = replicas / "seed-20261010/report.json"
    changed = json.loads(path.read_text())
    changed["checkpoint_digests"]["A_initial"] = "changed"
    path.write_text(json.dumps(changed))
    with pytest.raises(CCRRError, match="Prior binding mismatch"):
        probe.verify_evidence(config, baseline, replicas)
