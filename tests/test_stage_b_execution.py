"""Real Purifier/attacker, synthetic teachers: not H100/medical admission."""

import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from ccrr.codec.metadata import CLASSES
from ccrr.config import sha256
from ccrr.errors import CCRRError
from ccrr.representation.purifier import Purifier, SemanticAttacker
from ccrr.runtime.support import seed_everything
from ccrr.training.b_batches import (
    TeacherBank,
    image_backward,
    pair_backward,
    train_step,
)
from ccrr.training.b_resources import verify_teachers
from ccrr.training.b_state import BProgress, binding, output_lock
from ccrr.training.optimization import optimizer, TeacherSchedule
from ccrr.tensor_ops import pooled, masked_mean
from ccrr.training.losses import weighted_cross_entropy
from test_stage_a_training_loops import Data, TinyMFM, compare


class Identity(torch.nn.Module):
    def forward(self, x, rects):
        return {
            "embedding": F.normalize(x.mean((1, 2, 3))[:, None].expand(-1, 2048), dim=1)
        }


class Sampler:
    def sample(self, rng, scenario=None):
        return [
            (int(rng.integers(6)), int(rng.integers(6)), int(i < 8), "p1")
            for i in range(16)
        ]


def context(path, micro=4, resume=False):
    seed_everything(91)
    device = os.environ.get("CCRR_TEST_DEVICE", "cpu")
    path.mkdir(parents=True, exist_ok=True)
    ds = Data()
    e = SimpleNamespace(
        output=path,
        device=device,
        seed=91,
        cfg={
            "seed": 91,
            "device": device,
            "profile": {"size": 512},
            "weights": {},
            "budgets": {"purifier": 10000},
            "microbatch": micro,
            "resume_b": resume,
            "stage_a_source": {"freeze_sha256": "synthetic"},
        },
        data={"fit": ds, "val_select": ds},
        fit=ds,
        sampler=Sampler(),
        weights=torch.arange(1, 7, device=device).float(),
        purifier=Purifier().to(device),
        sem_attacker=SemanticAttacker().to(device),
        mfm=TinyMFM().to(device).eval().requires_grad_(False),
        reid=Identity().to(device).eval().requires_grad_(False),
    )
    e.purifier.stats_ready.fill_(True)
    return e


def setup(e):
    opt, adv = optimizer([e.purifier], [1e-4]), optimizer([e.sem_attacker], [1e-4])
    return opt, adv, TeacherSchedule(opt, 10000), TeacherSchedule(adv, 20000)


@pytest.mark.parametrize("micro", [1, 4, 16])
def test_resume_two_optimizers_rng_and_cold_bank(tmp_path, micro):
    def run(path, count, resume=False):
        e = context(path, micro, resume)
        bank = TeacherBank(e)
        opt, adv, sched, asched = setup(e)
        p = BProgress(e, opt, adv, sched, asched)
        if not resume:
            # Exercise the full adversarial branch, not just warmup.
            sched.next_step, asched.next_step = 1999, 3998
        for step in range(sched.next_step, sched.next_step + count):
            report = train_step(e, bank, step, opt, adv, sched, asched)
            assert np.isfinite(report["loss"])
            assert all(t.grad is None for t in e.sem_attacker.parameters())
            p.save()
        return torch.load(p.path, weights_only=True, map_location="cpu")

    uninterrupted = run(tmp_path / "full", 2)
    run(tmp_path / "split", 1)
    resumed = run(tmp_path / "split", 1, True)
    compare(uninterrupted, resumed)
    assert resumed["training_state"]["adversary_scheduler"]["next_step"] == 4002
    assert len(resumed["training_state"]["adversary_optimizer"]["state"]) > 0


def test_image_loss_matches_independent_single_image_reference(tmp_path):
    e = context(tmp_path)
    bank = TeacherBank(e)
    # Non-uniform coverage and repeated labels expose per-chunk normalization bugs.
    bank.storage["v"][0, :, :3] = 0
    bank.storage["v"][1, :, :2] = 0.5
    ids = [0, 1, 1, 3, 4, 2, 5, 0, 0, 2, 1, 4, 5, 0, 1, 1]
    initial = copy.deepcopy(e.purifier.state_dict())
    denom = sum(e.weights[i] for i in ids) * 3
    reference = 0.0
    for i in ids:
        t = bank.get([i])
        v = t["v"]
        px, py = e.purifier(t["f0"], v), e.purifier(t["f1"], v)
        cross = e.purifier(t["f1"], v, source_mask=px["mask"])
        loss = 0.0
        for out, target in ((px, t["e0"]), (py, t["e1"])):
            occupancy = pooled(out["mask"], v).mean(1)
            loss = (
                loss
                + (
                    masked_mean((out["all"] - out["standardized"]).square(), v)
                    + 0.1 * (out["identity"] - target).square().sum(1).mean()
                    + 10
                    * (
                        F.relu(0.1 - occupancy).square()
                        + F.relu(occupancy - 0.5).square()
                    ).mean()
                )
                / 32
            )
        for out in (px, py, cross):
            loss = loss + weighted_cross_entropy(
                out["logits"],
                torch.tensor([i], device=e.device),
                e.weights,
                weight_sum=denom,
            )
        loss.backward()
        reference += float(loss.detach())
    gradients = {k: p.grad.clone() for k, p in e.purifier.named_parameters()}
    e.purifier.load_state_dict(initial)
    e.purifier.zero_grad(set_to_none=True)
    actual = image_backward(e, bank, ids)
    assert float(actual) == pytest.approx(reference, rel=2e-5, abs=2e-5)
    for k, p in e.purifier.named_parameters():
        torch.testing.assert_close(p.grad, gradients[k], atol=3e-5, rtol=3e-4)


def test_pair_gradients_isolation_and_sampling(tmp_path):
    e = context(tmp_path)
    bank = TeacherBank(e)
    pairs = Sampler().sample(np.random.default_rng(7))
    rng = np.random.default_rng(9)
    pair_backward(e, bank, pairs, rng, 1.0, True)
    assert all(p.grad is None for p in e.purifier.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in e.sem_attacker.parameters()
    )
    expected = np.random.default_rng(9)
    expected.integers(3, size=32)
    assert rng.bit_generator.state == expected.bit_generator.state
    e.sem_attacker.zero_grad(set_to_none=True)
    e.sem_attacker.requires_grad_(False)
    pair_backward(e, bank, pairs, np.random.default_rng(9), -0.1, False)
    assert all(p.grad is None for p in e.sem_attacker.parameters())
    assert e.purifier.mask_head.weight.grad.abs().sum() > 0
    assert e.purifier.encoder.start.weight.grad.abs().sum() > 0


def resources(tmp_path):
    from pathlib import Path
    import ccrr

    a = tmp_path / "A"
    (a / "checkpoints").mkdir(parents=True)
    cfg = {
        "output": str(tmp_path / "B"),
        "seed": 91,
        "weights": {
            n: {"sha256": n} for n in ("efficientnet_b0", "resnet50", "panderm")
        },
    }
    accepted = {n: {"passed": True} for n in ("A-clinical", "A-mfm", "A-reid")}
    r = {
        "version": "PAD512-CCRR-v1.1",
        "seed": 91,
        "acceptance": accepted,
        "class_order": CLASSES,
        "base_resources": cfg["weights"],
        "checkpoints": {},
        "data_binding_sha256": sha256(
            Path(ccrr.__file__).parent / "data/frozen_evidence.json"
        ),
    }
    for n in ("clinical", "clinical_rgb", "clinical_gray", "mfm_probe", "reid"):
        p = a / "checkpoints" / (n + ".pt")
        p.write_bytes(n.encode())
        r["checkpoints"][n] = {
            "sha256": sha256(p),
            "selection": {
                "budget_completed": 2000
                if n == "mfm_probe"
                else 10000
                if n == "reid"
                else 5000
            },
        }
    (a / "acceptance.json").write_text(json.dumps(accepted))
    record = a / "teacher_freeze_record.json"
    record.write_text(json.dumps(r))
    cfg["stage_a_source"] = {"output": str(a), "freeze_sha256": sha256(record)}
    return cfg, a


@pytest.mark.parametrize("damage", ["hash", "overlap", "seed", "gate", "base"])
def test_teacher_tamper_or_wrong_run_rejected(tmp_path, damage):
    cfg, a = resources(tmp_path)
    verify_teachers(cfg)
    if damage == "hash":
        (a / "checkpoints/reid.pt").write_bytes(b"tampered")
    elif damage == "overlap":
        cfg["output"] = str(a / "nested")
    elif damage == "seed":
        cfg["seed"] += 1
    elif damage == "gate":
        (a / "acceptance.json").write_text("{}")
    else:
        cfg["weights"]["panderm"]["sha256"] = "wrong"
    with pytest.raises(CCRRError):
        verify_teachers(cfg)


def test_output_lock_duplicate_and_existing_progress(tmp_path):
    cfg, _ = resources(tmp_path)
    cfg.update(budgets={"purifier": 10000})
    with output_lock(cfg):
        with pytest.raises(ValueError, match="holds the output lock"):
            with output_lock(cfg):
                pass
    from pathlib import Path

    (Path(cfg["output"]) / "training.jsonl").touch()
    with pytest.raises(CCRRError, match="Existing B artifacts"):
        with output_lock(cfg):
            pass


def test_bank_bound_and_binding_change(tmp_path):
    e = context(tmp_path)
    digest = binding(e.cfg)
    e.cfg["microbatch"] = 16
    assert binding(e.cfg) != digest
    e.cfg["b_cache_bytes"] = 1
    with pytest.raises(CCRRError, match="exceeds"):
        TeacherBank(e)


def test_feature_audit_cache_and_batched_scores_preserve_rows(tmp_path):
    from ccrr.training.feature_audit import FeatureAudit
    from ccrr.tensor_ops import gray

    e = context(tmp_path)
    # Synthetic protocol only; no access to frozen private data.
    ds = e.fit
    ds.rows = [{"diagnostic": c, "patient_id": str(i)} for i, c in enumerate(CLASSES)]
    ds.root = tmp_path
    ds.index = {str(i): i for i in range(6)}
    protocol = tmp_path / "protocol_audit_20261009/pair_protocol"
    protocol.mkdir(parents=True)
    (protocol / "val_select_p1.json").write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query": "0",
                        "patient_id": "0",
                        "positive_gallery": ["1"],
                        "negative_gallery": ["2", "3"],
                    },
                    {
                        "query": "1",
                        "patient_id": "0",
                        "positive_gallery": ["0"],
                        "negative_gallery": ["4"],
                    },
                ]
            }
        )
    )
    e.purifier.eval().requires_grad_(False)
    audit = FeatureAudit(e)
    audit.cache()
    assert audit.cache_bytes <= 8 * 1024**3
    assert len(audit.diagnostics) == 6
    from ccrr.data.pad import collate

    b = collate([ds[1]], e.device)
    with torch.no_grad():
        f, v, _ = e.mfm(gray(b["x"]), b["rect"])
        expected = e.purifier(f, v)["standardized"]
    raw, _ = audit.features("fit", 1, "raw", 2)
    torch.testing.assert_close(raw, expected)
    before = audit.attack_scores(e.sem_attacker, "sem", "p1")
    e.cfg["microbatch"] = 1
    after = audit.attack_scores(e.sem_attacker, "sem", "p1")
    assert len(before) == 9
    for x, y in zip(before, after):
        for key in ("domain", "labels", "pairs", "weights"):
            assert x[key] == y[key]
        assert x["weights"] == [0.5, 0.25, 0.25, 0.5, 0.5]
        np.testing.assert_allclose(x["scores"], y["scores"], atol=2e-5, rtol=2e-4)
    # Train the actual batched audit update; frozen purifier is never updated.
    model = SemanticAttacker().to(e.device)
    opt = optimizer([model], [1e-4])
    audit.attacker_update(model, opt, "sem", "p1", "spatial", 0)
    assert opt.state
    assert all(p.grad is None for p in e.purifier.parameters())


def test_audit_progress_safe_serialization_and_changed_binding(tmp_path):
    e = context(tmp_path)
    opt, adv, sched, asched = setup(e)
    p = BProgress(e, opt, adv, sched, asched)
    p.phase = "audit"
    sched.next_step, asched.next_step = 10000, 20000
    p.audit_done = [1000]
    p.best = {"report": {"ci": np.array([0.7, 0.8]), "point": np.float64(0.75)}}
    p.save()
    e.cfg["resume_b"] = True
    loaded = BProgress(e, opt, adv, sched, asched)
    assert loaded.phase == "audit" and loaded.audit_done == [1000]
    assert loaded.best == {"report": {"ci": [0.7, 0.8], "point": 0.75}}
    e.cfg["b_cache_device"] = "cuda"
    with pytest.raises(CCRRError, match="binding"):
        BProgress(e, opt, adv, sched, asched)


def test_b_preflight_requires_new_bound_engineering_evidence(
    tmp_path, monkeypatch
):
    from ccrr.runtime.preflight import inspect
    import ccrr.runtime.preflight as preflight

    cfg, _ = resources(tmp_path)
    cfg.update(device="cpu", budgets={"purifier": 10000})
    monkeypatch.setattr(
        preflight, "verify_dataset", lambda *a, **kw: {"verified": True}
    )
    report = inspect(cfg, "B")
    assert report["b_frozen_teachers_verified"] is True
    assert not report["ready_for_stage"]
    assert any("B batched execution evidence" in item for item in report["missing"])


def test_preflight_refuses_A_output_before_any_write(tmp_path):
    from ccrr.runtime.commands import preflight
    import yaml

    cfg, a = resources(tmp_path)
    cfg["output"] = str(a)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(CCRRError, match="separate"):
        preflight(SimpleNamespace(config=config, stage="B"))
    assert not (a / "preflight.json").exists()


def test_diagnostic_checkpoint_cannot_resume_as_formal(tmp_path):
    e = context(tmp_path)
    e.cfg["b_diagnostic"] = True
    opt, adv, sched, asched = setup(e)
    p = BProgress(e, opt, adv, sched, asched)
    sched.next_step, asched.next_step = 1999, 3998
    p.save()
    e.cfg.update(b_diagnostic=False, resume_b=True)
    with pytest.raises(CCRRError, match="binding mismatch"):
        BProgress(e, opt, adv, sched, asched)
