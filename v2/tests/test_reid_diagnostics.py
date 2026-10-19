"""Finite diagnostic checks with synthetic data; no real patients or weights."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ccrr.data.pad import PairSampler
from ccrr_v2 import reid_diagnostics as d
from ccrr_v2.runtime import read, save_model, write
from test_loading import image_dataset


def test_replay_matches_full_sampler_and_exposes_unsampled_classes(tmp_path):
    data = image_dataset(tmp_path)
    # Distinct regions force the legacy unused-negative fallback RNG branch.
    for r in data.rows:
        r["region"] = r["patient_id"]
    data.rows.append(
        {
            **data.rows[0],
            "patient_id": "single",
            "lesion_id": "single",
            "img_id": "single",
        }
    )
    data.rows.append(
        {
            **data.rows[0],
            "patient_id": "excluded",
            "lesion_id": "excluded",
            "img_id": "excluded",
            "reid_observation_eligible": "0",
        }
    )
    steps, seed = 19, 57
    expected, sampler, rng = Counter(), PairSampler(data), np.random.default_rng(seed)
    for _ in range(steps):
        pairs = sampler.sample(rng)
        expected.update(
            data.rows[i]["patient_id"] for a, b, _, _ in pairs[:8] for i in (a, b)
        )
    summary, private = d.coverage(data, seed, steps)
    assert summary["classifier_classes_never_positive_targets"] == 2
    assert summary["positive_pool_union_patients"] == 4
    assert summary["pair_sampler_fallbacks"] == sampler.fallbacks > 0
    assert summary["image_slot_draws_per_patient_including_zero"]["sum"] == 16 * steps
    assert {
        r["patient_id"]: r["image_slot_draws"] for r in private if r["image_slot_draws"]
    } == expected
    assert sum(r["rgb_slots"] for r in private) == 16 * 10
    assert sum(r["gray_slots"] for r in private) == 16 * 9
    assert summary["images_decoded"] == 0


def test_parent_rejects_wrong_source_resource_or_incomplete_run(tmp_path):
    cfg = {"seed": 20261019, "reid_steps": 10000, "batch_size": 16}
    binding = {
        "stage": "reid",
        "source_commit": d.TRAINING_COMMIT,
        "resources": {},
        "config": cfg,
    }
    write(tmp_path / "binding.json", binding)
    write(
        tmp_path / "status.json", {"state": "completed", "pair_sampler_fallbacks": 328}
    )
    assert d.validate_parent({}, tmp_path)[0] == cfg
    with pytest.raises(ValueError, match="resource"):
        d.validate_parent({"changed": True}, tmp_path)
    write(tmp_path / "status.json", {"state": "interrupted"})
    with pytest.raises(ValueError, match="completed"):
        d.validate_parent({}, tmp_path)
    binding["source_commit"] = "other"
    write(tmp_path / "binding.json", binding)
    with pytest.raises(ValueError, match="source"):
        d.validate_parent({}, tmp_path)


def test_checkpoint_checks_step_and_shapes(tmp_path):
    model = torch.nn.Linear(3, 2)
    save_model(tmp_path / "c.pt", model, 1000)
    d.load_checkpoint(model, tmp_path / "c.pt", 1000)
    with pytest.raises(ValueError, match="step mismatch"):
        d.load_checkpoint(model, tmp_path / "c.pt", 8000)
    with pytest.raises(RuntimeError):
        d.load_checkpoint(torch.nn.Linear(3, 4), tmp_path / "c.pt", 1000)


class TinyEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(3)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, rect):
        assert not self.training and not torch.is_grad_enabled()
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        z = self.dropout(self.bn(x.mean((2, 3))))
        return {"embedding": torch.nn.functional.normalize(z, dim=1)}


def test_frozen_inference_real_protocol_keeps_bn_and_parameters(tmp_path):
    data = image_dataset(tmp_path)
    data.role = "val_tune"
    protocols = tmp_path / "protocol_audit_20261009" / "pair_protocol"
    protocols.mkdir(parents=True)
    for scenario in ("p1", "p2"):
        queries = []
        for a in data.rows:
            positives = [
                b["img_id"]
                for b in data.rows
                if a["patient_id"] == b["patient_id"]
                and a["img_id"] != b["img_id"]
                and (
                    (a["lesion_id"] != b["lesion_id"])
                    if scenario == "p1"
                    else (a["lesion_id"] == b["lesion_id"])
                )
            ]
            queries.append(
                {
                    "patient_id": a["patient_id"],
                    "query": a["img_id"],
                    "positive_gallery": positives,
                    "negative_gallery": [
                        b["img_id"]
                        for b in data.rows
                        if a["patient_id"] != b["patient_id"]
                    ],
                }
            )
        write(protocols / f"val_tune_{scenario}.json", {"queries": queries})
    model = TinyEmbedding()
    before = d.state_digest(model)
    report, state = d.frozen_report(
        model, data, "cpu", {"batch_size": 4, "bootstrap_repetitions": 10}
    )
    assert state["before"] == state["after"] == before
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    summary = d.compact(report)
    assert summary["p1"]["rgb_rgb"]["queries"] == 16
    assert summary["p2"]["gray_gray"]["patients"] == 4


@pytest.mark.parametrize("role", ["fit", "val_select", "test"])
def test_non_tune_inference_is_rejected(role):
    with pytest.raises(ValueError, match="val_tune"):
        d.frozen_report(TinyEmbedding(), SimpleNamespace(role=role), "cpu", {})


def test_mutated_buffer_is_detected(monkeypatch):
    model = TinyEmbedding()

    def mutate(*args, **kwargs):
        model.bn.running_mean.add_(1)
        return {}

    monkeypatch.setattr(d, "evaluate", mutate)
    with pytest.raises(RuntimeError, match="changed"):
        d.frozen_report(model, SimpleNamespace(role="val_tune"), "cpu", {})


def test_main_mismatch_stops_before_model_and_preserves_parent(tmp_path, monkeypatch):
    parent, output = tmp_path / "runs" / "parent", tmp_path / "runs" / "new"
    parent.mkdir(parents=True)
    resources = {"workspace_root": str(tmp_path), "data_root": "synthetic"}
    write(tmp_path / "resources.json", resources)
    write(
        parent / "binding.json",
        {
            "stage": "reid",
            "source_commit": d.TRAINING_COMMIT,
            "resources": resources,
            "config": {"seed": 20261019, "reid_steps": 10000, "batch_size": 16},
        },
    )
    write(parent / "status.json", {"state": "completed", "pair_sampler_fallbacks": 328})
    for step in d.STEPS:
        (parent / f"reid-step-{step}.pt").write_bytes(b"not loaded")
    before = {p.name: p.read_bytes() for p in parent.iterdir()}
    monkeypatch.setattr(d, "validate_resources", lambda r: None)

    def create(r, path, *args):
        p = type(parent)(path)
        p.mkdir(exist_ok=False)
        write(p / "binding.json", {})
        return p

    monkeypatch.setattr(d, "new_run", create)

    def data(root, role):
        assert role == "fit"  # No inference set constructed after mismatch.
        return None

    monkeypatch.setattr(d, "PADData", data)
    monkeypatch.setattr(
        d, "coverage", lambda *a, **kw: ({"pair_sampler_fallbacks": 0}, [])
    )
    with pytest.raises(RuntimeError, match="fallback count"):
        d.main(
            [
                "--resources",
                str(tmp_path / "resources.json"),
                "--training-run",
                str(parent),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
    assert read(output / "status.json")["state"] == "failed"
    assert {p.name: p.read_bytes() for p in parent.iterdir()} == before
