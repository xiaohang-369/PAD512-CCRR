"""Synthetic fixtures test admission/refusal and real-loop save/reload mechanics."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from ccrr.config import sha256
from ccrr.data.resize import POLICY, teacher_input_sources
from ccrr.runtime import preflight
from ccrr.training import teachers
from ccrr.training.a_state import binding
from test_stage_a_training_loops import (
    context,
    IdentityData,
    TinyClinical,
    TinyReID,
    TinyMFM,
    Stats,
)

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location(
        "batch_admission", scripts / "check_stage_a_batch_admission.py"
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
finally:
    sys.path.remove(str(scripts))


@pytest.mark.parametrize(
    "kind,model",
    [("efficientnet_b0", TinyClinical), ("resnet50", TinyReID), ("panderm", TinyMFM)],
)
def test_three_production_loops_save_reload_full_state(
    tmp_path, monkeypatch, kind, model
):
    class Fit(IdentityData):
        def class_weights(self):
            return torch.arange(1, 7).float()

    e = context(tmp_path / "formal")
    e.cfg.update(
        microbatch=16, budgets={"clinical": 5000, "reid": 10000, "mfm_probe": 2000}
    )
    monkeypatch.setattr(teachers, "initialize_base", lambda *a: None)
    monkeypatch.setattr(teachers, "mfm_interface", lambda *a: {"passed": True})
    monkeypatch.setattr(probe, "Purifier", Stats)
    template = model()
    full, expected = probe.replay(e.cfg, Fit(), template, kind, tmp_path / "full", 2)
    probe.replay(e.cfg, Fit(), template, kind, tmp_path / "split", 1)
    resumed, actual = probe.replay(
        e.cfg, Fit(), template, kind, tmp_path / "split", 1, resume=True
    )
    assert probe.exact_equal(expected, actual)
    assert actual["step"] == 752
    assert full["optimizer_state_allocated"] and resumed["trainable_parameters_updated"]
    assert not list(e.output.iterdir())


def evidence(cfg):
    models = {}
    for name in preflight.BASE:
        interface = {
            "passed": True,
            "model": name,
            "sha256": "fixture",
            "batch_size": 16,
            "device": "cpu",
            "torch": "fixture",
            "adapter_source_sha256": sha256(
                Path(preflight.__file__).parents[1] / "teachers/adapters.py"
            ),
            "teacher_input_sources": teacher_input_sources(),
            "teacher_resize_policy": POLICY,
            "parameters_and_buffers_unchanged": True,
            "load": {
                "source_sha256": preflight.SOURCE,
                "source_commit": preflight.COMMIT,
            },
            "forward_backward": [
                {
                    "domain": domain,
                    "finite": True,
                    "input_gradient_l1": 16,
                    "frozen_parameter_gradients_absent": True,
                    "official_norm_patch_tokens_exact": True,
                    "input_shape": [16, 3, 512, 512],
                    "per_sample_input_gradient_l1": [1] * 16,
                    "valid_rects": [[0, 0, 512, 512], [0, 64, 512, 384]] * 8,
                }
                for domain in ("rgb", "gray") * 3
            ],
        }
        models[name] = {
            "interface": interface,
            "training": {
                "steps": 2,
                **{
                    k: True
                    for k in (
                        "resume_state_match",
                        "finite_nonzero_gradients",
                        "optimizer_state_allocated",
                        "buffers_unchanged",
                        "trainable_parameters_updated",
                        "fp32",
                        "frozen_backbone_unchanged",
                    )
                },
            },
        }
    return {
        "passed": True,
        "microbatch": 16,
        "configuration_binding": binding(cfg),
        "checker_sources": preflight.batch_checker_sources(),
        "formal_training_started": False,
        "models": models,
        "execution_flags": {
            "fp32": True,
            "strict_determinism": True,
            "matmul_tf32": False,
            "cudnn_tf32": False,
            "cudnn_benchmark": False,
        },
    }


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "batch1",
        "missing_reid",
        "no_resume",
        "zero_sample_grad",
        "stale_binding",
        "stale_checker",
        "tf32",
        "changed_pan",
    ],
)
def test_bound_batch_evidence_refuses_incomplete_or_stale_reports(tmp_path, defect):
    cfg = context(tmp_path / "formal").cfg
    cfg.update(
        device="cpu",
        microbatch=16,
        weights={n: {"sha256": "fixture"} for n in preflight.BASE},
    )
    report = evidence(cfg)
    if defect == "batch1":
        report["models"]["panderm"]["interface"]["batch_size"] = 1
    elif defect == "missing_reid":
        del report["models"]["resnet50"]
    elif defect == "no_resume":
        report["models"]["resnet50"]["training"]["resume_state_match"] = False
    elif defect == "zero_sample_grad":
        report["models"]["efficientnet_b0"]["interface"]["forward_backward"][0][
            "per_sample_input_gradient_l1"
        ][7] = 0
    elif defect == "stale_binding":
        report["configuration_binding"] = "stale"
    elif defect == "stale_checker":
        report["checker_sources"] = {}
    elif defect == "tf32":
        report["execution_flags"]["matmul_tf32"] = True
    elif defect == "changed_pan":
        report["models"]["panderm"]["training"]["frozen_backbone_unchanged"] = False
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    cfg["batch_execution_report"] = {"path": str(path), "sha256": sha256(path)}
    assert preflight.batch_evidence(cfg, {"packages": {"torch": "fixture"}}) == (
        defect is None
    )
    if defect is None:
        changed = copy.deepcopy(cfg)
        changed["microbatch"] = 8
        assert not preflight.batch_evidence(changed, {"packages": {"torch": "fixture"}})
        path.write_text("{}")
        assert not preflight.batch_evidence(cfg, {"packages": {"torch": "fixture"}})


def test_preflight_requires_batch_evidence_only_for_batched_a(monkeypatch):
    from test_stage_a_preflight import config

    monkeypatch.setattr(
        preflight, "verify_dataset", lambda *a, **kw: {"verified": True}
    )
    cfg = config()
    assert not any(
        "batch/resume evidence" in m for m in preflight.inspect(cfg, "A")["missing"]
    )
    cfg["microbatch"] = 16
    assert any(
        "batch/resume evidence" in m for m in preflight.inspect(cfg, "A")["missing"]
    )
