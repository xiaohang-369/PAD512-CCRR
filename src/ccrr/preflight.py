from pathlib import Path
from .runtime import environment
from .config import sha256


def inspect(config):
    env = environment()
    missing = []
    weights = {}
    if config.get("device", "cuda").startswith("cuda") and not env["devices"]:
        missing.append("CUDA/H100 unavailable on this host")
    for name, item in config["weights"].items():
        path = item.get("path")
        expected = item.get("sha256")
        exists = bool(path and Path(path).is_file())
        actual = sha256(path) if exists else None
        weights[name] = {
            "configured": bool(path and expected),
            "exists": exists,
            "sha256_matches": bool(actual and actual == expected),
        }
        if not weights[name]["sha256_matches"]:
            missing.append(name + " verified base weights")
        if name == "panderm":
            source = item.get("source")
            digest = item.get("source_sha256")
            if not (
                source
                and digest
                and Path(source).is_file()
                and sha256(source) == digest
            ):
                missing.append("PanDerm verified official source")
    return {
        "environment": env,
        "weights": weights,
        "ready_for_teacher_training": not missing,
        "missing": missing,
        "formal_training_started": False,
    }
