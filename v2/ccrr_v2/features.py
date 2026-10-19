"""Bound, private RGB/gray patch caches. No test images and no learned V1 weights."""

from pathlib import Path
import torch

from ccrr.data.pad import PADData, collate
from ccrr.teachers.adapters import PanDerm
from ccrr.tensor_ops import gray
from .runtime import completed, digest, read, write

ROLES = ("fit", "val_tune", "val_select")


def extract(resources, config, output, device):
    p = resources["panderm"]
    model = (
        PanDerm(p["source"], p["source_sha256"], p["path"], p["sha256"])
        .to(device)
        .eval()
    )
    manifest = {
        "schema": "ccrr-v2-features-1",
        "binding": digest(output / "binding.json"),
        "roles": {},
    }
    statistics = None
    for role in ROLES:
        data = PADData(resources["data_root"], role)
        features = torch.empty(len(data), 2, 768, 14, 14, dtype=torch.float32)
        coverage = torch.empty(len(data), 1, 14, 14)
        for start in range(0, len(data), config["batch_size"]):
            stop = min(start + config["batch_size"], len(data))
            b = collate([data[i] for i in range(start, stop)], device)
            with torch.no_grad():
                for domain, image in enumerate((b["x"], gray(b["x"]))):
                    f, v, _ = model(image, b["rect"])
                    if f.shape[1:] != (768, 14, 14) or not torch.isfinite(f).all():
                        raise RuntimeError("Invalid real PanDerm feature interface")
                    features[start:stop, domain].copy_(f.cpu())
                    coverage[start:stop].copy_(v.cpu())
            print(f"FEATURES {role} {stop}/{len(data)}", flush=True)
        if role == "fit":
            # Accumulate in FP64 without duplicating the full FP32 cache in FP64.
            total = torch.zeros(768, dtype=torch.float64)
            square = total.clone()
            count = 0.0
            for i in range(len(data)):
                f, v = features[i].double(), coverage[i].double()
                total += (f * v).sum((0, 2, 3))
                square += (f.square() * v).sum((0, 2, 3))
                count += float(v.sum()) * 2
            mean = total / count
            variance = (square / count - mean.square()).clamp_min(0)
            statistics = {"mean": mean.float(), "std": (variance + 1e-6).sqrt().float()}
            torch.save(statistics, output / "fit_statistics.pt")
        from ccrr.codec.metadata import CLASSES

        labels = torch.tensor([CLASSES.index(r["diagnostic"]) for r in data.rows])
        protocols = {
            s: read(
                Path(resources["data_root"])
                / "protocol_audit_20261009"
                / "pair_protocol"
                / f"{role}_{s}.json"
            )
            for s in ("p1", "p2")
        }
        path = output / f"{role}.pt"
        torch.save(
            {
                "features": features,
                "coverage": coverage,
                "labels": labels,
                "rows": data.rows,
                "protocols": protocols,
            },
            path,
        )
        manifest["roles"][role] = {
            "file": path.name,
            "sha256": digest(path),
            "images": len(data),
        }
        del features, coverage
    manifest["statistics"] = {
        "file": "fit_statistics.pt",
        "sha256": digest(output / "fit_statistics.pt"),
    }
    write(output / "features.json", manifest)
    completed(output, artifact="features.json", model_parameters_updated=False)


def load_cache(path, resources, device):
    path = Path(path).resolve()
    if (Path(resources["workspace_root"]).resolve() / "runs") not in path.parents:
        raise ValueError("Cache must be a V2 run")
    binding = read(path / "binding.json")
    if read(path / "status.json")["state"] != "completed":
        raise ValueError("Feature cache did not complete")
    if binding["stage"] != "features" or binding["resources"] != resources:
        raise ValueError("Cache resource or stage mismatch")
    manifest = read(path / "features.json")
    if manifest["binding"] != digest(path / "binding.json"):
        raise ValueError("Feature binding changed")
    for entry in [manifest["statistics"], *manifest["roles"].values()]:
        target = (path / entry["file"]).resolve()
        if target.parent != path or digest(target) != entry["sha256"]:
            raise ValueError("Cache content/path mismatch")
    stats = torch.load(path / manifest["statistics"]["file"], weights_only=True)
    result = {}
    for role in ROLES:
        obj = torch.load(path / manifest["roles"][role]["file"], weights_only=True)
        obj["features"] = obj["features"].to(device)
        mean = stats["mean"].to(device)[None, None, :, None, None]
        std = stats["std"].to(device)[None, None, :, None, None]
        obj["features"].sub_(mean).div_(std)
        obj["coverage"] = obj["coverage"].to(device)
        obj["labels"] = obj["labels"].to(device)
        result[role] = obj
    return result
