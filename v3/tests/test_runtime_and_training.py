from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ccrr_v3 import runtime
from ccrr_v3.cli import (
    parser,
    validate_arguments,
    validate_reference_review,
    load_targets,
)
from ccrr_v3.models import ClinicalHeads, FrozenSupervisor
from ccrr_v3.learning import train, load_separator, represent
from ccrr_v3.evaluation import fit_heads, audit


def test_draft_is_readable_but_cannot_train(cfg):
    cfg["registration_status"] = "draft"
    runtime.validate_config(cfg)
    with pytest.raises(ValueError, match="Register"):
        runtime.validate_config(cfg, registered=True)


@pytest.mark.parametrize(
    "key,value",
    [
        ("lr", float("nan")),
        ("batch_size", 32),
        ("head_mode", "silent"),
        ("joint_steps", 0),
        ("min_probability", 1.0),
    ],
)
def test_config_rejects_invalid_numbers_and_contracts(cfg, key, value):
    cfg[key] = value
    with pytest.raises(ValueError):
        runtime.validate_config(cfg)


def test_explicit_reference_and_no_unexpected_flags():
    p = parser()
    common = ["--resources", "x", "--config", "y"]
    a = p.parse_args(
        ["train", *common, "--features", "f", "--heads", "h", "--variant", "separation"]
    )
    with pytest.raises(ValueError, match="targets"):
        validate_arguments(a)
    a = p.parse_args(["probe", *common, "--features", "f", "--variant", "compression"])
    with pytest.raises(ValueError, match="Unexpected"):
        validate_arguments(a)
    a = p.parse_args(["audit", *common, "--features", "f", "--view", "semantic"])
    with pytest.raises(ValueError, match="require"):
        validate_arguments(a)


def fake_artifact(directory, resources=None):
    directory.mkdir()
    runtime.write(
        directory / "binding.json",
        dict(
            schema="ccrr-v3-run-1",
            resources=resources or {},
            parents={"features_sha256": "abc"},
        ),
    )
    runtime.write(directory / "status.json", {"state": "completed"})
    runtime.write(directory / "payload.json", {"fixed": True})
    runtime.write(
        directory / "artifact.json",
        dict(
            schema="ccrr-v3-artifact-1",
            kind="audit",
            binding_sha256=runtime.digest(directory / "binding.json"),
            files={"payload.json": runtime.digest(directory / "payload.json")},
        ),
    )
    return directory


def test_artifact_hash_resources_and_feature_binding(tmp_path):
    p = fake_artifact(tmp_path / "parent")
    runtime.artifact(p, "audit", {}, "abc")
    with pytest.raises(ValueError, match="different PanDerm"):
        runtime.artifact(p, "audit", {}, "other")
    with pytest.raises(ValueError, match="resource"):
        runtime.artifact(p, resources={"changed": True})
    runtime.write(p / "payload.json", {"changed": True})
    with pytest.raises(ValueError, match="content"):
        runtime.artifact(p)


def test_artifact_rejects_path_escape(tmp_path):
    p = fake_artifact(tmp_path / "parent")
    other = tmp_path / "outside.json"
    runtime.write(other, {})
    m = runtime.read(p / "artifact.json")
    m["files"] = {"../outside.json": runtime.digest(other)}
    runtime.write(p / "artifact.json", m)
    with pytest.raises(ValueError, match="escape"):
        runtime.artifact(p)


def test_new_runs_are_isolated_and_no_overwrite(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(
        runtime.subprocess,
        "check_output",
        lambda args, **kw: "" if "status" in args else "a" * 40,
    )
    resources = {"workspace_root": str(tmp_path / "PAD512-CCRR-V3")}
    out = Path(resources["workspace_root"]) / "runs" / "fresh"
    runtime.new_run(resources, out, "train", cfg, {}, {})
    before = (out / "binding.json").read_bytes()
    with pytest.raises(FileExistsError):
        runtime.new_run(resources, out, "train", cfg, {}, {})
    assert (out / "binding.json").read_bytes() == before
    with pytest.raises(ValueError, match="V3 workspace"):
        runtime.new_run(
            resources, tmp_path / "PAD512-CCRR-V2/runs/x", "train", cfg, {}, {}
        )


def test_reference_review_is_explicit_and_bound(tmp_path):
    target = fake_artifact(tmp_path / "targets")
    review = tmp_path / "review.json"
    obj = dict(
        schema="ccrr-v3-reference-review-1",
        targets_artifact_sha256=runtime.digest(target / "artifact.json"),
        decision="allow_development_reference",
        directions=["rgb_rgb", "gray_rgb", "gray_gray"],
        reason="Synthetic test only, no medical conclusion",
    )
    runtime.write(review, obj)
    validate_reference_review(review, target)
    obj["targets_artifact_sha256"] = "wrong"
    runtime.write(review, obj)
    with pytest.raises(ValueError, match="bound"):
        validate_reference_review(review, target)


def test_targets_reject_wrong_row_order(tmp_path, cache):
    data = {
        role: dict(
            ids=[r["img_id"] for r in obj["rows"]],
            embeddings=torch.randn(len(obj["rows"]), 2, 6),
        )
        for role, obj in cache.items()
    }
    runtime.save(tmp_path / "targets.pt", data)
    load_targets(tmp_path, cache)
    data["fit"]["ids"].reverse()
    runtime.save(tmp_path / "targets.pt", data)
    with pytest.raises(ValueError, match="order"):
        load_targets(tmp_path, cache)


@pytest.mark.parametrize("variant", ["compression", "separation", "adversarial"])
def test_finite_training_preserves_teachers_and_inputs(cfg, cache, tmp_path, variant):
    teacher = ClinicalHeads(8)
    initial_teacher = deepcopy(teacher.state_dict())
    raw = cache["fit"]["features"].clone()
    targets = (
        None
        if variant == "compression"
        else {role: torch.randn(len(obj["rows"]), 2, 6) for role, obj in cache.items()}
    )
    stats = dict(mean=torch.zeros(8), std=torch.ones(8))
    train(
        {r: cache[r] for r in ("fit", "val_tune")},
        targets,
        teacher,
        stats,
        cfg,
        tmp_path,
        torch.device("cpu"),
        variant,
    )
    assert torch.equal(raw, cache["fit"]["features"])
    assert all(
        torch.equal(initial_teacher[k], teacher.state_dict()[k])
        for k in initial_teacher
    )
    model, heads, saved_stats = load_separator(tmp_path / "separator.pt", "cpu")
    assert saved_stats["mean"].shape == (8,)
    assert not model.training
    output = represent(cache, model, "semantic", "cpu")
    assert output["fit"]["features"].shape == raw.shape
    assert torch.equal(cache["fit"]["features"], raw)
    supervisor = FrozenSupervisor(model, heads, **saved_stats)
    x = torch.randn(2, 8, 4, 4, requires_grad=True)
    logits, _ = supervisor(x, torch.ones(2, 1, 4, 4), 1)
    logits.square().mean().backward()
    assert x.grad.abs().sum() > 0
    assert all(p.grad is None for p in supervisor.parameters())


def test_independent_probe_and_attack_pipeline(cfg, cache, tmp_path):
    raw = cache["fit"]["features"].clone()
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    heads = fit_heads(cache, cfg, probe_dir, "cpu")
    assert len(heads.heads) == 2
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    audit(cache, cfg, audit_dir, "cpu")
    assert torch.equal(raw, cache["fit"]["features"])
    report = runtime.read(audit_dir / "global.json")
    assert set(report) == {"rgb_rgb", "gray_rgb", "gray_gray"}
    assert "rank1" in report["gray_rgb"]
    assert not any("p1" in key for key in report)
    assert (audit_dir / "linear.json").exists()
