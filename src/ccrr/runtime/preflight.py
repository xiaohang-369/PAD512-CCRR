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
        and len(cases) == 4
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


def inspect(config, stage="all"):
    if stage not in STAGES:
        raise ValueError("Unknown preflight stage")
    env, missing, limitations, weights = environment(), [], [], {}
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
                config.get("microbatch", 1) == 1,
                "A implementation currently executes microbatch=1",
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
    data = None
    try:
        data = verify_dataset(config.get("data_root", ""), pixels=True)
    except (CCRRError, OSError, KeyError, ValueError) as exc:
        missing.append("Frozen data evidence: " + str(exc))
    if stage not in ("A", "all"):
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
        "data": data,
        "ready_for_stage": not missing,
        "ready_for_teacher_training": stage in ("A", "all") and not missing,
        "teacher_capability_validated": False,
        "formal_training_started": False,
        "missing": missing,
        "limitations": limitations,
    }
