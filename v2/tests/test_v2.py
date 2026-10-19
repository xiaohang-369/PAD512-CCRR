import subprocess

import pytest
import torch

from ccrr_v2 import runtime
from ccrr_v2.audit import evaluate_attacker, fit_attacker, protocol_pairs
from ccrr_v2.cli import validate_config
from ccrr_v2.learning import fit_probe, train_s0
from ccrr_v2.models import PairAttacker, SemanticAdapter, confusion_loss, pool
from ccrr_v2.reid import embeddings, triplet_loss

torch.set_num_threads(1)


@pytest.fixture
def cfg():
    obj = runtime.read(runtime.REPO / "v2/configs/first_validation.json")
    obj.update(
        probe_steps=2,
        s0_warmup_steps=1,
        s0_joint_steps=1,
        attacker_warmup_steps=1,
        attacker_steps=2,
        evaluate_every=1,
        bootstrap_repetitions=10,
        identity_ramp_steps=1,
    )
    return obj


def synthetic():
    rows = []
    diagnoses = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
    for patient in range(6):
        for lesion in range(2):
            for view in range(2):
                rows.append(
                    {
                        "img_id": f"{patient}-{lesion}-{view}",
                        "patient_id": f"p{patient}",
                        "lesion_id": f"{patient}-{lesion}",
                        "diagnostic": diagnoses[patient],
                        "region": "arm",
                        "reid_observation_eligible": "1",
                    }
                )
    protocols = {}
    for scenario in ("p1", "p2"):
        queries = []
        for a in rows:
            pos = [
                b["img_id"]
                for b in rows
                if a["patient_id"] == b["patient_id"]
                and a["img_id"] != b["img_id"]
                and (
                    (a["lesion_id"] != b["lesion_id"])
                    if scenario == "p1"
                    else (a["lesion_id"] == b["lesion_id"])
                )
            ]
            neg = [b["img_id"] for b in rows if a["patient_id"] != b["patient_id"]]
            queries.append(
                {
                    "query": a["img_id"],
                    "patient_id": a["patient_id"],
                    "positive_gallery": pos,
                    "negative_gallery": neg,
                }
            )
        protocols[scenario] = {
            "queries": queries,
            "gallery": [r["img_id"] for r in rows],
        }
    return {
        "features": torch.randn(len(rows), 2, 768, 14, 14) * 0.1,
        "coverage": torch.ones(len(rows), 1, 14, 14),
        "labels": torch.tensor([i // 4 for i in range(len(rows))]),
        "rows": rows,
        "protocols": protocols,
    }


def test_s0_shape_and_frozen_attacker_gradient():
    s, a = SemanticAdapter(), PairAttacker(128)
    x, y = torch.randn(2, 768, 14, 14), torch.randn(2, 768, 14, 14)
    v = torch.ones(2, 1, 14, 14)
    for p in a.parameters():
        p.requires_grad_(False)
    out = s(x)
    assert out.shape == (2, 128, 14, 14)
    confusion_loss(a(out, v, s(y), v)).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in s.parameters())
    assert all(p.grad is None for p in a.parameters())


def test_confusion_minimum():
    assert abs(float(confusion_loss(torch.zeros(4)))) < 1e-6
    assert confusion_loss(torch.ones(4) * 5) > confusion_loss(torch.zeros(4))


def test_pool_ignores_padding():
    f = torch.tensor([[[[2.0, 100.0]]]])
    v = torch.tensor([[[[1.0, 0.0]]]])
    assert pool(f, v).item() == 2


def test_pair_protocol_excludes_self_and_weights_patients():
    data = synthetic()
    for scenario in ("p1", "p2"):
        pairs = protocol_pairs(data, scenario)
        assert all(a != b for a, b, *_ in pairs)
        for pid in {p[4] for p in pairs}:
            assert sum(
                p[3] for p in pairs if p[4] == pid and p[2] == 1
            ) == pytest.approx(1.0)
            assert sum(
                p[3] for p in pairs if p[4] == pid and p[2] == 0
            ) == pytest.approx(1.0)


@pytest.mark.parametrize("spatial", [False, True])
def test_audit_six_directions(spatial):
    data = synthetic()
    report = evaluate_attacker(PairAttacker(768, spatial), data)
    assert len(report) == 6
    assert all(0.5 <= r["a_dagger"] <= 1 for r in report.values())


@pytest.mark.parametrize("variant", ["medical", "joint"])
def test_synthetic_training_only(cfg, tmp_path, variant):
    cache = {role: synthetic() for role in ("fit", "val_tune", "val_select")}
    snapshot = cache["fit"]["features"].clone()
    runtime.write(tmp_path / "binding.json", {"stage": "synthetic_engineering_only"})
    train_s0(cache, cfg, tmp_path, variant)
    assert torch.equal(cache["fit"]["features"], snapshot)
    obj = torch.load(tmp_path / "s0.pt", weights_only=True)
    assert obj["step"] == 2
    assert (
        runtime.read(tmp_path / "status.json")["scientific_acceptance"]
        == "not_evaluated"
    )


def test_probe_heldout_report(cfg, tmp_path):
    cache = {role: synthetic() for role in ("fit", "val_tune", "val_select")}
    _, report = fit_probe(cache, cfg, tmp_path)
    assert len(report["domains"]) == 2
    obj = torch.load(tmp_path / "probe.pt", weights_only=True)
    assert obj["extra"]["selection_role"] == "val_tune"


def test_triplet_needs_real_positive_and_negative():
    x = torch.randn(4, 8, requires_grad=True)
    assert triplet_loss(x, torch.arange(4)).item() == 0
    value = triplet_loss(x, torch.tensor([0, 0, 1, 1]))
    assert torch.isfinite(value)
    value.backward()
    assert x.grad is not None


def test_independent_attacker_loop(cfg, tmp_path):
    cache = {role: synthetic() for role in ("fit", "val_tune", "val_select")}
    report = fit_attacker(cache, cfg, tmp_path, spatial=False)
    assert len(report) == 6
    assert all("auroc_ci" in value for value in report.values())
    saved = torch.load(tmp_path / "attacker.pt", weights_only=True)
    assert saved["extra"]["selection_role"] == "val_tune"


def test_reid_embedding_contract():
    class FakeModel(torch.nn.Module):
        def forward(self, x, rects):
            return {"embedding": x.mean((2, 3))}

    data = [
        {
            "x": torch.ones(3, 8, 8),
            "v": torch.ones(1, 8, 8),
            "legal": torch.ones(1),
            "label": 0,
            "patient_index": 0,
            "rect": (0, 0, 8, 8),
            "patient": "p0",
            "id": "i0",
            "metadata": {},
        }
    ]
    result = embeddings(FakeModel(), data, "cpu", 16)
    assert len(result) == 2
    assert result[0]["i0"]["embedding"].shape == (3,)
    assert "embedding" in result[1]["i0"]


def test_run_path_collision_and_source_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda cmd, **kw: "" if "status" in cmd else "a" * 40,
    )
    resources = {"workspace_root": str(tmp_path / "workspace")}
    out = tmp_path / "workspace/runs/one"
    runtime.new_run(resources, out, "s0", {})
    marker = out / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        runtime.new_run(resources, out, "s0", {})
    assert marker.read_text() == "preserve"
    with pytest.raises(ValueError):
        runtime.new_run(resources, tmp_path / "old/runs/one", "s0", {})


def test_old_checkpoint_resource_rejected(tmp_path):
    resources = runtime.read(runtime.REPO / "v2/configs/resources.example.json")
    resources["stage_a_source"] = {"output": "/old/run"}
    with pytest.raises(ValueError, match="Only explicit"):
        runtime.validate_resources(resources, hashes=False)


@pytest.mark.parametrize(
    "field,value", [("probe_steps", 0), ("batch_size", 32), ("probe_lr", float("nan"))]
)
def test_invalid_budget(cfg, field, value):
    cfg[field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_new_trainer_does_not_import_old_training_system():
    import ast

    for path in (runtime.REPO / "v2/ccrr_v2").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("ccrr.training", "ccrr.runtime.system", "ccrr.representation")
                )
