import copy
import sys
from pathlib import Path

import pytest
from ccrr.errors import CCRRError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_frozen_teacher_evidence import validate_metrics


def fixture():
    r = {"macro_auroc": 0.8, "per_class_auroc": [0.8] * 6, "recall": [0.5] * 6}
    p = {
        "threshold_role": "val_tune",
        "cosine": {
            "valid": True,
            "auroc": 0.75,
            "auroc_ci": {"interval": [0.6, 0.9], "invalid_fraction": 0},
        },
    }
    return copy.deepcopy(
        {
            "A-clinical": {
                "passed": True,
                "report": {"joint": [r, r], "controls": [r, r]},
            },
            "A-mfm": {
                "passed": True,
                "report": {"domains": [r, r], "interface": {"passed": True}},
            },
            "A-reid": {
                "passed": True,
                "report": {"p1": {"reports": [p, p, p]}, "p2": {"reports": []}},
            },
        }
    )


def test_original_metrics_pass_and_p2_not_substituted_for_p1():
    r = validate_metrics(fixture())
    assert r["passed"] and not r["reid_p2_gate"]["passed"]
    assert len(r["reid_p1"]) == 3


@pytest.mark.parametrize("damage", ["ci", "missing_direction", "clinical", "nan"])
def test_passed_flag_cannot_hide_bad_raw_evidence(damage):
    a = fixture()
    p = a["A-reid"]["report"]["p1"]["reports"]
    if damage == "ci":
        p[0]["cosine"]["auroc_ci"]["interval"][0] = 0.5
    elif damage == "missing_direction":
        p.pop()
    elif damage == "clinical":
        a["A-clinical"]["report"]["joint"][0]["recall"][0] = 0
    else:
        a["A-clinical"]["report"]["joint"][0]["macro_auroc"] = float("nan")
    with pytest.raises(CCRRError):
        validate_metrics(a)


@pytest.mark.parametrize("recovery_ok", [True, False])
def test_prepare_uses_finished_micro16_even_if_remaining_sweep_was_stopped(
    tmp_path, monkeypatch, recovery_ok
):
    import json
    import yaml
    import prepare_formal_b as prepare

    root = tmp_path / "profile/result/purifier16"
    (root / "micro-16").mkdir(parents=True)
    report = {
        "complete": False,
        "effective_image_batch": 16,
        "effective_pair_batch": 16,
        "cases": {
            "16": {
                "microbatch": 16,
                "accumulation": 1,
                "sampling_equal": True,
                "one_step_numerical_screen": True,
                "seconds_per_step": 0.375,
                "measurement": {
                    "repeats": [{"restore_and_continuation_exact": recovery_ok}] * 3
                },
            }
        },
    }
    (root / "report.json").write_text(json.dumps(report))
    (root / "micro-16/profile.json").write_text(
        json.dumps({"activity": {"cuda_kernels_recorded": True}, "phases": {}})
    )
    aroot = tmp_path / "a"
    aroot.mkdir()
    (aroot / "teacher_freeze_record.json").write_text("{}")
    config = tmp_path / "base.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 20261008,
                "budgets": {"purifier": 10000},
                "custom_unchanged": "original",
            }
        )
    )
    original = config.read_bytes()
    monkeypatch.setattr(
        prepare, "verify_teachers", lambda cfg: (aroot, {"acceptance": fixture()})
    )
    out = tmp_path / "new"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--config",
            str(config),
            "--profile-run",
            str(tmp_path / "profile"),
            "--out",
            str(out),
        ],
    )
    if not recovery_ok:
        with pytest.raises(CCRRError, match="recovery checks"):
            prepare.main()
        assert not (out / "startup-input.yaml").exists()
        return
    prepare.main()
    cfg = yaml.safe_load((out / "startup-input.yaml").read_text())
    assert cfg["b_initialization"] == "PUR-INIT-v1.2-rc1"
    assert cfg["microbatch"] == cfg["batch_pairs"] == cfg["batch_teacher"] == 16
    assert cfg["budgets"]["purifier"] == 10000 and not cfg["resume_b"]
    assert cfg["custom_unchanged"] == "original" and cfg["output"] == str(aroot)
    assert config.read_bytes() == original
