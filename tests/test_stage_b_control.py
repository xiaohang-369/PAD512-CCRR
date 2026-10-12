import copy
import torch
import pytest

from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.training.b_batches import TeacherBank
from ccrr.training.b_control import (
    ReconstructionControl,
    control_update,
    prepare_control,
    CONTROL_PROTOCOL,
)
from ccrr.training.b_functional import check_functional
from ccrr.training.b_state import binding
from ccrr.training.optimization import optimizer, TeacherSchedule
from ccrr.training.feature_audit import FeatureAudit
from ccrr.errors import CCRRError
from test_stage_b_execution import context
from test_stage_a_training_loops import compare


def test_real_control_update_reload_and_no_branch(tmp_path):
    e = context(tmp_path)
    bank = TeacherBank(e)
    original = copy.deepcopy(e.purifier.state_dict())
    model = ReconstructionControl(e.purifier.mean, e.purifier.std).to(e.device)
    assert list(model._modules) == ["encoder", "decoder"]
    assert not torch.equal(model.encoder.start.weight, e.purifier.encoder.start.weight)
    opt = optimizer([model], [1e-4])
    scheduler = TeacherSchedule(opt, 10000)
    ids = [i % 6 for i in range(16)]
    control_update(model, bank, ids, 4, opt, scheduler)
    path = tmp_path / "control.pt"
    save_checkpoint(
        path,
        {"control": model},
        opt,
        1,
        "B-control",
        {},
        training_state={"scheduler": scheduler.state_dict()},
    )
    control_update(model, bank, ids, 4, opt, scheduler)
    expected = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    raw = load_checkpoint(path, {"control": model}, opt)
    scheduler.load_state_dict(raw["training_state"]["scheduler"])
    control_update(model, bank, ids, 4, opt, scheduler)
    compare(expected, (model.state_dict(), opt.state_dict()))
    compare(original, e.purifier.state_dict())


def test_control_complete_resume_and_selected_artifact_not_rewritten(
    tmp_path, monkeypatch
):
    import ccrr.training.b_control as module

    e = context(tmp_path)
    bank = TeacherBank(e)
    root = e.output / "no_branch_control"
    model = ReconstructionControl(e.purifier.mean, e.purifier.std).to(e.device)
    opt = optimizer([model], [1e-4])
    sched = TeacherSchedule(opt, 10000)
    sched.next_step = 9999
    for step in range(1000, 10000, 1000):
        save_checkpoint(
            root / f"step-{step}.pt", {"control": model}, None, step, "B-control", {}
        )
    save_checkpoint(
        root / "last.pt",
        {"control": model},
        opt,
        9999,
        "B-control",
        {},
        training_state={
            "binding": binding(e.cfg),
            "protocol": CONTROL_PROTOCOL,
            "scheduler": sched.state_dict(),
            "best": None,
        },
    )
    monkeypatch.setattr(module, "TeacherBank", lambda *a, **kw: bank)
    prepare_control(e, bank)
    assert not e.no_branch_control.training
    assert all(not p.requires_grad for p in e.no_branch_control.parameters())
    before = (root / "selected.pt").read_bytes()
    monkeypatch.setattr(
        module,
        "control_update",
        lambda *a, **kw: pytest.fail("retrained complete control"),
    )
    prepare_control(e, bank)
    assert before == (root / "selected.pt").read_bytes()


def test_control_cache_and_shared_readouts_bound_and_rng_isolated(tmp_path):
    e = context(tmp_path)
    for i, row in enumerate(e.fit.rows):
        row["patient_id"] = str(i)
    e.no_branch_control = (
        ReconstructionControl(e.purifier.mean, e.purifier.std)
        .to(e.device)
        .eval()
        .requires_grad_(False)
    )
    e.no_branch_control_sha256 = "test-control-digest"
    audit = FeatureAudit(e)
    audit.cache()
    a, _ = audit.feature_batch("fit", [0], "control", [1])
    b, _ = audit.feature_batch("fit", [0], "control", [2])
    assert torch.equal(a, b)
    assert audit.representations == ("raw", "control", "sem")
    state = torch.get_rng_state().clone()
    value = audit.readout("test-control", "control", lambda: [float(torch.rand(()))])
    assert torch.equal(state, torch.get_rng_state())
    e.feature_audit_tag = "another-endpoint"
    other = FeatureAudit(e)
    assert (
        other.readout(
            "test-control", "control", lambda: pytest.fail("duplicate baseline")
        )
        == value
    )
    e.no_branch_control_sha256 = "changed"
    with pytest.raises(CCRRError, match="binding mismatch"):
        other.readout("test-control", "control", lambda: None)


def test_functional_mask_geometry_with_real_purifier(tmp_path):
    e = context(tmp_path)
    bank = TeacherBank(e)
    before = copy.deepcopy(e.purifier.state_dict())
    report = check_functional(e, bank)
    assert report["passed"] and not report["heatmap_ranking_validated"]
    compare(before, e.purifier.state_dict())


def test_execution_evidence_rejects_old_source_and_missing_control(tmp_path):
    import json
    from pathlib import Path
    from ccrr.config import sha256
    from ccrr.training.b_resources import verify_execution_report

    e = context(tmp_path)
    cfg = e.cfg
    path = tmp_path / "report.json"
    r = {
        "version": "CCRR-B-ENGINEERING-2",
        "engineering_passed": True,
        "formal_training_started": False,
        "stage_a_freeze_sha256": "synthetic",
        "checker_sha256": sha256(
            Path(__file__).parents[1] / "scripts/check_stage_b_host.py"
        ),
        "environment": {"devices": ["NVIDIA H100 PCIe"]},
        "variants": [
            {
                "microbatch": 4,
                "binding": binding({**cfg, "b_diagnostic": True}),
                "complete_resume_state_match": True,
                "frozen_teachers_unchanged": True,
                "functional": {"passed": True},
                "control_replay": {"complete_resume_state_match": True},
                "audit_probe": {
                    "representations": ["raw", "control", "sem"],
                    "post_adaptation_probe": {"original_attacker_unchanged": True},
                },
            }
        ],
    }

    def write():
        path.write_text(json.dumps(r))
        cfg["b_execution_report"] = {"path": str(path), "sha256": sha256(path)}

    write()
    assert verify_execution_report(cfg)
    r["variants"][0]["control_replay"]["complete_resume_state_match"] = False
    write()
    with pytest.raises(CCRRError, match="coverage"):
        verify_execution_report(cfg)
    r["variants"][0]["control_replay"]["complete_resume_state_match"] = True
    r["variants"][0]["binding"] = "old-source"
    write()
    with pytest.raises(CCRRError, match="source/config/environment"):
        verify_execution_report(cfg)
