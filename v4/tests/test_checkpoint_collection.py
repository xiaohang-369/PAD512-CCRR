import copy
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

spec = importlib.util.spec_from_file_location(
    "checkpoint_collection",
    Path(__file__).parents[1] / "scripts/collect-checkpoint-selection.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root):
    cfg = json.loads(
        (Path(__file__).parents[1] / "configs/checkpoint-selection.json").read_text()
    )
    baseline = root / "baseline.pt"
    baseline.write_bytes(b"test baseline, never torch loaded")
    inspection = root / "inspection.json"
    module.write(inspection, {"results": {"baseline": {"checkpoint": str(baseline)}}})
    parents = {str(p): module.digest(p) for p in (baseline, inspection)}
    initial = root / "initial/report.json"
    previous_cfg = None
    for run_spec in cfg["runs"]:
        run = root / "runs" / run_spec["name"]
        run.mkdir(parents=True)
        train_cfg = dict(
            update_modules=["E1", "D1"],
            epochs=run_spec["last_epoch"],
            gpu_ids=[1, 2],
            inspection_report=str(inspection),
        )
        if previous_cfg is not None:
            train_cfg["resume"] = dict(
                run=str(root / "runs" / cfg["runs"][0]["name"]),
                sha256=cfg["runs"][0]["final_sha256"],
            )
        binding = {"config": train_cfg, "parents": parents}
        if previous_cfg is None:
            zero = dict(epoch=0, role="val_tune", images=169, recovery={})
            module.write(
                initial, dict(passed=True, binding=binding, initial_evaluation=zero)
            )
            module.write(initial.parent / "evaluation-000.json", zero)
        rows = []
        for epoch in range(run_spec["first_epoch"], run_spec["last_epoch"] + 1):
            ck = run / f"g1-epoch-{epoch:03d}.pt"
            ck.write_bytes(f"fake checkpoint {epoch}".encode())
            ev = dict(
                epoch=epoch,
                role="val_tune",
                images=169,
                recovery={"D0_direct_psnr": 38 - epoch / 5},
            )
            module.write(run / f"evaluation-{epoch:03d}.json", ev)
            rows.append(
                dict(
                    epoch=epoch,
                    step=epoch * 90,
                    validation=ev,
                    train={"total": 1.0},
                    checkpoint=dict(
                        file=ck.name,
                        sha256=module.digest(ck),
                        epoch=epoch,
                        step=epoch * 90,
                    ),
                )
            )
        run_spec["final_sha256"] = rows[-1]["checkpoint"]["sha256"]
        (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        module.write(run / "config.json", train_cfg)
        module.write(run / "binding.json", binding)
        module.write(run / "status.json", {"state": "completed"})
        module.write(
            run / "preflight-binding.json",
            {"path": str(initial), "sha256": module.digest(initial)},
        )
        module.write(
            run / "report.json",
            dict(
                binding=binding,
                training_completed=True,
                optimizer_updates=run_spec["last_epoch"] * 90,
                final_checkpoint=rows[-1]["checkpoint"],
                final_validation=rows[-1]["validation"],
                parents_unchanged=True,
                frozen_models_unchanged=True,
            ),
        )
        previous_cfg = train_cfg
    return cfg


def test_complete_export_and_no_model_copy(tmp_path):
    cfg = fixture(tmp_path)
    before = {str(p): module.digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    output = tmp_path / "export"
    result = module.collect(tmp_path, output, cfg)
    assert result["metadata_checks_passed"] and not result["issues"]
    assert result["evaluation_epochs"] == list(range(61))
    assert result["checkpoint_files_present"] == 60
    assert not result["tensor_states_loaded"] and not result["gpu_used"]
    assert all(module.digest(Path(p)) == sha for p, sha in before.items())
    with tarfile.open(tmp_path / "export.tar.gz") as tar:
        assert not any(n.endswith(".pt") for n in tar.getnames())
        assert any(n.endswith("epoch_metrics.csv") for n in tar.getnames())


@pytest.mark.parametrize(
    "fault",
    [
        "missing_checkpoint",
        "corrupt_checkpoint",
        "conflicting_evaluation",
        "missing_run",
    ],
)
def test_missing_or_conflicting_evidence_is_not_synthesized(tmp_path, fault):
    cfg = fixture(tmp_path)
    run = tmp_path / "runs" / cfg["runs"][1]["name"]
    if fault == "missing_checkpoint":
        (run / "g1-epoch-045.pt").unlink()
    elif fault == "corrupt_checkpoint":
        (run / "g1-epoch-045.pt").write_bytes(b"changed")
    elif fault == "conflicting_evaluation":
        p = run / "evaluation-045.json"
        d = json.loads(p.read_text())
        d["recovery"]["D0_direct_psnr"] = 0
        module.write(p, d)
    else:
        (run / "report.json").unlink()
    result = module.collect(tmp_path, tmp_path / "export", cfg)
    if fault == "missing_checkpoint":
        assert result["missing_checkpoint_epochs"] == [45]
        assert 45 in result["evaluation_epochs"]
    else:
        assert not result["metadata_checks_passed"]
    if fault == "missing_run":
        assert result["missing_checkpoint_epochs"] == list(range(31, 61))
        assert result["missing_evaluation_epochs"] == list(range(31, 61))


def test_parent_output_refused(tmp_path):
    cfg = fixture(tmp_path)
    output = tmp_path / "runs" / cfg["runs"][0]["name"] / "new-export"
    with pytest.raises(ValueError, match="separate"):
        module.collect(tmp_path, output, copy.deepcopy(cfg))
    assert not output.exists()
