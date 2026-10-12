"""Stage-scoped engineering admission; never an assertion of learned capability."""

import json
import os
from pathlib import Path
from ccrr.runtime.support import environment
from ccrr.config import sha256
from ccrr.data.admission import verify_dataset
from ccrr.data.resize import POLICY, teacher_input_sources
from ccrr.errors import CCRRError

STAGES = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "all")
BASE = ("efficientnet_b0", "resnet50", "panderm")
REQUIRED = {s: ("resnet50", "panderm") for s in STAGES}
REQUIRED["A"] = BASE
REQUIRED["C"] = ()
REQUIRED["J"] = ("resnet50", "panderm", "convnext_tiny", "swin_t")
REQUIRED["all"] = BASE + ("convnext_tiny", "swin_t")
PINNED = {
    "efficientnet_b0": "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934",
    "resnet50": "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
    "panderm": "be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1",
}
SOURCE = "63fbb03283bb5d61ebd24e4790b66f3495996cc37f000f7003eec0a277bd1ab1"
COMMIT = "fd7a80748ba7fc3e203fed88f909f4689d0d6f24"
DEPENDENCIES = {
    "B": ("A-clinical", "A-mfm", "A-reid"),
    "C": (),
    "D": ("A-clinical", "A-mfm", "A-reid", "B", "C"),
    "E": ("D",),
    "F": ("E",),
    "G": ("E",),
    "H": ("E",),
    "I": ("H",),
    "J": ("H",),
}


def interface_evidence(item, name, config, env):
    entry = config.get("interface_reports", {}).get(name, {})
    path = entry.get("path")
    if not path or not Path(path).is_file() or sha256(path) != entry.get("sha256"):
        return False
    r = json.loads(Path(path).read_text())
    return interface_matches(item, name, config, env, r)


def interface_matches(item, name, config, env, r, batch_size=1):
    cases = r.get("forward_backward", [])
    return bool(
        r.get("passed")
        and r.get("model") == name
        and r.get("sha256") == item.get("sha256")
        and r.get("adapter_source_sha256")
        == sha256(Path(__file__).parents[1] / "teachers/adapters.py")
        and r.get("teacher_input_sources") == teacher_input_sources()
        and r.get("teacher_resize_policy") == POLICY
        and r.get("device")
        == ("cuda" if config.get("device", "cuda").startswith("cuda") else "cpu")
        and r.get("torch") == env["packages"]["torch"]
        and r.get("parameters_and_buffers_unchanged")
        and r.get("batch_size", 1) == batch_size
        and len(cases) == (6 if batch_size > 1 else 4)
        and {c.get("domain") for c in cases} == {"rgb", "gray"}
        and all(
            c.get("finite")
            and c.get("input_gradient_l1", 0) > 0
            and c.get("frozen_parameter_gradients_absent")
            for c in cases
        )
        and (
            name != "panderm"
            or (
                r.get("load", {}).get("source_sha256") == SOURCE
                and r.get("load", {}).get("source_commit") == COMMIT
                and all(c.get("official_norm_patch_tokens_exact") for c in cases)
            )
        )
    )


def batch_checker_sources():
    root = Path(__file__).parents[3]
    return {
        name: sha256(root / "scripts" / name)
        for name in (
            "check_stage_a_batch_admission.py",
            "check_teacher_interface.py",
            "probe_stage_a_speed.py",
        )
    }


def batch_evidence(config, env):
    """Bind real batched evidence to this execution, not a batch-1 report."""
    from ccrr.training.a_state import binding

    try:
        entry = config.get("batch_execution_report", {})
        path = Path(entry["path"])
        if sha256(path) != entry.get("sha256"):
            return False
        r = json.loads(path.read_text())
        n = config["microbatch"]
        if not (
            r.get("passed")
            and r.get("microbatch") == n
            and r.get("configuration_binding") == binding(config)
            and r.get("checker_sources") == batch_checker_sources()
            and r.get("execution_flags")
            == {
                "fp32": True,
                "strict_determinism": True,
                "matmul_tf32": False,
                "cudnn_tf32": False,
                "cudnn_benchmark": False,
            }
            and r.get("formal_training_started") is False
            and (not config["device"].startswith("cuda") or r.get("H100_validated"))
        ):
            return False
        for name in BASE:
            model = r["models"][name]
            interface, training = model["interface"], model["training"]
            if not interface_matches(
                config["weights"][name], name, config, env, interface, n
            ):
                return False
            cases = interface["forward_backward"]
            if not all(
                c["input_shape"][0] == n
                and len(c["per_sample_input_gradient_l1"]) == n
                and all(v > 0 for v in c["per_sample_input_gradient_l1"])
                for c in cases
            ):
                return False
            if not any(len({tuple(v) for v in c["valid_rects"]}) > 1 for c in cases):
                return False
            if (
                not all(
                    training.get(k) is True
                    for k in (
                        "resume_state_match",
                        "finite_nonzero_gradients",
                        "optimizer_state_allocated",
                        "buffers_unchanged",
                        "trainable_parameters_updated",
                        "fp32",
                    )
                )
                or training.get("steps") != 2
            ):
                return False
            if name == "panderm" and not training.get("frozen_backbone_unchanged"):
                return False
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def inspect(config, stage="all"):
    if stage not in STAGES:
        raise ValueError("Unknown preflight stage")
    env, missing, limitations, weights = environment(), [], [], {}
    batch_verified = None
    cuda = config.get("device", "cuda").startswith("cuda")
    if cuda:
        if not env["devices"] or not all("H100" in d for d in env["devices"]):
            missing.append("Requested H100 CUDA runtime unavailable")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            missing.append("CUBLAS_WORKSPACE_CONFIG must be set before Python starts")
    else:
        if config.get("device", "cuda") != "cpu":
            missing.append(
                "A replay is supported only on CPU or CUDA; MPS RNG replay unverified"
            )
        limitations.append(
            "CPU engineering checks only; H100 runtime/memory/throughput unverified"
        )
    required_packages = {
        "torch": "2.6.0",
        "torchvision": "0.21.0",
        "timm": "0.9.16",
        "numpy": "2.2.6",
        "Pillow": "11.3.0",
        "cryptography": "44.0.3",
        "scipy": "1.15.3",
        "scikit-learn": "1.6.1",
        "scikit-image": "0.25.2",
    }
    for n, version in required_packages.items():
        if (env["packages"].get(n) or "").split("+")[0] != version:
            missing.append(f"Dependency mismatch: {n}=={version}")
    for name in REQUIRED[stage]:
        item = config.get("weights", {}).get(name, {})
        path, expected = item.get("path"), item.get("sha256")
        exists = bool(path and Path(path).is_file())
        actual = sha256(path) if exists else None
        matches = bool(
            actual
            and actual == expected
            and (name not in PINNED or actual == PINNED[name])
        )
        weights[name] = {
            "configured": bool(path and expected),
            "exists": exists,
            "sha256_matches": matches,
        }
        if not matches:
            missing.append(name + " verified base weights")
        if name == "panderm":
            source = item.get("source")
            if not (
                source
                and Path(source).is_file()
                and item.get("source_sha256") == SOURCE
                and sha256(source) == SOURCE
            ):
                missing.append("PanDerm verified official source")
            overlap = item.get("overlap_audit")
            if overlap not in ("unknown", "verified_no_overlap"):
                missing.append("PanDerm overlap audit absent or known contaminated")
            elif overlap == "unknown":
                limitations.append(
                    "PanDerm pretraining overlap unknown; no leakage-free claim"
                )
        if name in BASE:
            weights[name]["interface_verified"] = interface_evidence(
                item, name, config, env
            )
            if not weights[name]["interface_verified"]:
                missing.append(name + " bound real-weight interface evidence")
    if stage in ("A", "all"):
        checks = [
            (
                config.get("profile", {}).get("size") == 512,
                "A requires 512 preprocessing",
            ),
            (
                config.get("seed") in (20261008, 20261009, 20261010),
                "Unregistered training seed",
            ),
            (
                config.get("batch_teacher", 16) == 16
                and config.get("batch_pairs", 16) == 16,
                "A effective batches must be 16",
            ),
            (
                type(config.get("microbatch", 1)) is int
                and config.get("microbatch", 1) in (1, 2, 4, 8, 16),
                "A microbatch must be 1, 2, 4, 8 or 16",
            ),
            (
                config.get("bootstrap_repetitions", 2000) == 2000,
                "Bootstrap must remain 2000",
            ),
            (
                all(
                    config.get("budgets", {}).get(n) == b
                    for n, b in [
                        ("clinical", 5000),
                        ("mfm_probe", 2000),
                        ("reid", 10000),
                    ]
                ),
                "A budgets must remain 5000/2000/10000",
            ),
            (
                isinstance(config.get("resume_interval", 100), int)
                and config.get("resume_interval", 100) > 0,
                "Invalid save interval",
            ),
            (
                type(config.get("pixel_cache_bytes", 0)) is int
                and config.get("pixel_cache_bytes", 0) >= 0,
                "pixel_cache_bytes must be a nonnegative integer",
            ),
        ]
        missing.extend(label for passed, label in checks if not passed)
        if type(config.get("microbatch", 1)) is int and config.get("microbatch", 1) > 1:
            batch_verified = batch_evidence(config, env)
            if not batch_verified:
                missing.append(
                    "A batched execution requires bound three-model batch/resume evidence"
                )
    data = None
    try:
        data = verify_dataset(config.get("data_root", ""), pixels=True)
    except (CCRRError, OSError, KeyError, ValueError) as exc:
        missing.append("Frozen data evidence: " + str(exc))
    b_teachers_verified = None
    if stage == "all":
        from ccrr.training.b_resources import SCIENTIFIC_GAPS

        missing.extend(
            "B scientific admission not implemented: " + gap for gap in SCIENTIFIC_GAPS
        )
    if stage == "B":
        from ccrr.training.b_resources import (
            verify_teachers,
            execution_options,
            SCIENTIFIC_GAPS,
            verify_execution_report,
        )

        try:
            execution_options(config)
            verify_teachers(config)
            b_teachers_verified = True
        except (CCRRError, OSError, KeyError, TypeError, ValueError) as exc:
            b_teachers_verified = False
            missing.append("B frozen teacher dependencies: " + str(exc))
        try:
            verify_execution_report(config)
        except (CCRRError, OSError, KeyError, TypeError, ValueError) as exc:
            missing.append("B batched execution evidence: " + str(exc))
        # Never allow the old subset of numeric gates to certify section 7.7.
        missing.extend(
            "B scientific admission not implemented: " + gap for gap in SCIENTIFIC_GAPS
        )
        limitations.append(
            "B execution admission does not establish purification efficacy; full endpoint audits remain required. Section 8.6 heatmap ranking requires separate downstream validation."
        )
    elif stage not in ("A", "all"):
        output = Path(config.get("output", "."))
        acceptance = output / "acceptance.json"
        accepted = json.loads(acceptance.read_text()) if acceptance.is_file() else {}
        for dep in DEPENDENCIES[stage]:
            if not accepted.get(dep, {}).get("passed"):
                missing.append("Missing accepted dependency " + dep)
        resources = (
            []
            if stage == "C"
            else [
                "clinical.pt",
                "reid.pt",
                "mfm_probe.pt" if stage == "B" else "purifier.pt",
            ]
        )
        previous = {
            "D": "C",
            "E": "D",
            "F": "E",
            "G": "F",
            "H": "F",
            "I": "H",
            "J": "H",
        }.get(stage)
        if previous:
            resources.append(previous + ".pt")
        for name in resources:
            if not (output / "checkpoints" / name).is_file():
                missing.append("Missing stage checkpoint " + name)
        limitations.append(
            "Post-A scientific gaps remain; stage resource checks are not full scientific admission"
        )
    return {
        "stage": stage,
        "environment": env,
        "weights": weights,
        "batch_execution_verified": batch_verified,
        "b_frozen_teachers_verified": b_teachers_verified,
        "data": data,
        "ready_for_stage": not missing,
        "ready_for_teacher_training": stage in ("A", "all") and not missing,
        "teacher_capability_validated": False,
        "formal_training_started": False,
        "missing": missing,
        "limitations": limitations,
    }
