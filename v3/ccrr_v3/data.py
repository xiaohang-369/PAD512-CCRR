"""Read-only reuse of bound A0 features; no V2 models or P1 selection imported."""

from pathlib import Path

import torch

from ccrr.codec.metadata import CLASSES
from .runtime import read, digest, ROLES
from .pairs import validate_rows, validate_p2_protocol


def load_features(path, resources, roles=ROLES):
    if not roles or not set(roles) <= set(ROLES):
        raise ValueError("No test/calibration feature use in V3 development")
    path = Path(path).resolve()
    manifest = read(path / "features.json")
    binding = read(path / "binding.json")
    if manifest.get("schema") != "ccrr-v2-features-1" or binding["stage"] != "features":
        raise ValueError("Require a completed, original A0 PanDerm cache")
    if read(path / "status.json")["state"] != "completed" or manifest[
        "binding"
    ] != digest(path / "binding.json"):
        raise ValueError("Incomplete or changed feature binding")
    for key in ("data_root", "panderm", "resnet50"):
        if binding["resources"][key] != resources[key]:
            raise ValueError("A0/public resource mismatch")
    if set(manifest["roles"]) != set(ROLES):
        raise ValueError("Unexpected cache roles")

    def checked(entry):
        file = (path / entry["file"]).resolve()
        if file.parent != path or digest(file) != entry["sha256"]:
            raise ValueError("Cache bytes/path mismatch")
        return torch.load(file, map_location="cpu", weights_only=True)

    stats = checked(manifest["statistics"])
    mean, std = stats["mean"], stats["std"]
    if (
        mean.shape != (768,)
        or std.shape != (768,)
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("Invalid fit statistics")
    cache, seen = {}, set()
    # Inspect all role metadata from bound admission files without opening held-out feature tensors.
    data_root = Path(resources["data_root"])
    for role in (*ROLES, "calibration", "test"):
        rows = read(data_root / "protocol_audit_20261009/roles" / f"{role}.json")
        patients = {r["patient_id"] for r in rows}
        if seen & patients:
            raise ValueError("Patient overlap across roles")
        seen |= patients
    for role in roles:
        obj = checked(manifest["roles"][role])
        validate_rows(obj["rows"])
        if obj["rows"] != read(
            data_root / "protocol_audit_20261009/roles" / f"{role}.json"
        ):
            raise ValueError("Cached rows differ from bound data role")
        n = len(obj["rows"])
        f, v, y = obj["features"], obj["coverage"], obj["labels"]
        if (
            f.shape != (n, 2, 768, 14, 14)
            or v.shape != (n, 1, 14, 14)
            or y.shape != (n,)
        ):
            raise ValueError("PanDerm cache shape mismatch")
        if (
            not torch.isfinite(f).all()
            or not torch.isfinite(v).all()
            or (v < 0).any()
            or (v > 1).any()
            or (v.sum((1, 2, 3)) <= 0).any()
        ):
            raise ValueError("Nonfinite features or invalid coverage")
        if y.tolist() != [CLASSES.index(r["diagnostic"]) for r in obj["rows"]]:
            raise ValueError("Labels do not match rows")
        validate_p2_protocol(obj["protocols"]["p2"], obj["rows"])
        obj["features"] = (f - mean[None, None, :, None, None]) / std[
            None, None, :, None, None
        ]
        # Do not expose a P1 protocol accidentally to the new training path.
        obj["protocols"] = {"p2": obj["protocols"]["p2"]}
        cache[role] = obj
    return cache, stats


def batch(data, indices, domain, device):
    return (
        data["features"][indices, domain].to(device),
        data["coverage"][indices].to(device),
        data["labels"][indices].to(device),
    )


def pair_batch(data, sampler, rng, device):
    pairs = sampler.sample(rng)
    a, b, labels = map(list, zip(*[(p[0], p[1], p[2]) for p in pairs]))
    da, db = ((0, 0), (1, 0), (1, 1))[int(rng.integers(3))]
    xa, va, _ = batch(data, a, da, device)
    xb, vb, _ = batch(data, b, db, device)
    return xa, va, xb, vb, torch.tensor(labels, dtype=torch.float32, device=device)
