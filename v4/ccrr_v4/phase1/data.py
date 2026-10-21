"""Private phase-1 data cache. Read original PAD evidence; never change its files."""

import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ..verify import digest, read, source_check, REPO


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def load_config(path):
    cfg = read(path)
    reference = read(REPO / "v4/configs/phase1-train.json")
    # Strict source profile: only execution/logging knobs can change. No silent
    # "same method" label after modifying objectives, image size, batch or phases.
    adjustable = {
        "runtime",
        "log_every_steps",
        "validation_batch_size",
        "checkpoint_every_epochs",
    }
    if set(cfg) != set(reference):
        raise ValueError("Unexpected phase-1 configuration fields")
    for key in cfg.keys() - adjustable:
        if cfg[key] != reference[key]:
            raise ValueError(f"Strict author source profile changed: {key}")
    runtime = cfg["runtime"]
    if set(runtime) != set(reference["runtime"]):
        raise ValueError("Unexpected runtime settings")
    for key in ("precision", "tf32", "amp", "gradient_accumulation"):
        if runtime[key] != reference["runtime"][key]:
            raise ValueError(f"Source profile disallows changing {key}")
    for key in adjustable - {"runtime"}:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    if runtime["device"] not in ("cuda", "cpu") or runtime["cpu_threads"] < 1:
        raise ValueError("Invalid device/threads")
    ids = runtime["gpu_ids"]
    if not isinstance(ids, list) or len(set(map(str, ids))) != len(ids):
        raise ValueError("gpu_ids must be a list of unique physical IDs/UUIDs")
    if any(not isinstance(i, (int, str)) or "," in str(i) or not str(i) for i in ids):
        raise ValueError("Invalid GPU ID")
    return cfg


def pad256(x):
    """Named PAD adaptation, not an invented claim about VOC cropping."""
    a = torch.floor(x * 255 + 0.5).byte().permute(1, 2, 0).numpy()
    channels = [
        np.asarray(
            Image.fromarray(a[:, :, c].astype(np.float32)).resize(
                (256, 256), Image.Resampling.BILINEAR
            )
        )
        for c in range(3)
    ]
    return np.clip(np.floor(np.stack(channels, -1) + 0.5), 0, 255).astype(np.uint8)


def prepare(config_path, data_root, vgg_path, vgg_sha, output):
    from ccrr.data.admission import verify_dataset
    from ccrr.data.pad import PADData
    from .vgg import AuthorVGG

    cfg = load_config(config_path)
    source = source_check()
    # Fail before writing the cache if the actual source weights are absent/wrong.
    vgg = AuthorVGG.from_file(vgg_path, vgg_sha)
    del vgg
    admission = verify_dataset(data_root, pixels=False)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    roles = {}
    for role in (cfg["data"]["fit_role"], cfg["data"]["monitor_role"]):
        data = PADData(data_root, role)
        dest = output / f"{role}.npy"
        images = np.lib.format.open_memmap(
            dest, mode="w+", dtype=np.uint8, shape=(len(data), 256, 256, 3)
        )
        for i in range(len(data)):
            images[i] = pad256(data[i]["x"])  # PADData verifies every used RGB512 hash.
            if (i + 1) % 100 == 0:
                print(f"PREPARE role={role} images={i + 1}/{len(data)}", flush=True)
        images.flush()
        del images
        roles[role] = dict(
            file=dest.name,
            sha256=digest(dest),
            images=len(data),
            rows_sha256=hashlib.sha256(
                json.dumps(data.rows, sort_keys=True).encode()
            ).hexdigest(),
        )
    evidence_dir = Path(data_root) / "protocol_audit_20261009"
    provenance = dict(
        schema="ccrr-v4-phase1-cache-1",
        source=source,
        config=cfg,
        data_root=str(Path(data_root).resolve()),
        admission=admission,
        used_roles_pixels512_verified=True,
        test_used_for_performance=False,
        source_manifests={
            r: digest(evidence_dir / "roles" / f"{r}.json") for r in roles
        },
        roles=roles,
        vgg=dict(path=str(Path(vgg_path).resolve()), sha256=vgg_sha),
        view=cfg["data"]["view"],
    )
    write_json(output / "prepared.json", provenance)
    print(f"PHASE1_PREPARED={output / 'prepared.json'}", flush=True)


def load_prepared(path, cfg):
    path = Path(path).resolve()
    p = read(path)
    if (
        p["schema"] != "ccrr-v4-phase1-cache-1"
        or not p["used_roles_pixels512_verified"]
    ):
        raise ValueError("Require completed, pixel-verified phase-1 cache")
    if (
        p["config"]["data"] != cfg["data"]
        or p["config"]["image_size"] != cfg["image_size"]
    ):
        raise ValueError("Prepared data/configuration mismatch")
    if p["source"] != source_check() or not p["admission"]["verified"]:
        raise ValueError("Prepared source/admission binding mismatch")
    arrays = {}
    for role in ("fit", "val_tune"):
        entry = p["roles"][role]
        f = path.parent / entry["file"]
        if f.parent != path.parent or digest(f) != entry["sha256"]:
            raise ValueError("Prepared data checksum/path mismatch")
        a = np.load(f, mmap_mode="r", allow_pickle=False)
        if a.dtype != np.uint8 or a.shape != (entry["images"], 256, 256, 3):
            raise ValueError("Prepared array shape/dtype mismatch")
        arrays[role] = a
    if len(arrays["fit"]) < cfg["global_batch_size"]:
        raise ValueError("Insufficient source training images")
    n, b = len(arrays["fit"]), cfg["global_batch_size"]
    p["schedule_counts"] = dict(
        fit_images=n,
        global_batch=b,
        steps_per_epoch=math.ceil(n / b),
        decay_steps=(n // b) * sum(s["epochs"] for s in cfg["stages"]),
        total_steps=math.ceil(n / b) * sum(s["epochs"] for s in cfg["stages"]),
    )
    return p, arrays


def batch_tensor(array, indices, device):
    x = np.array(array[indices], copy=True)
    return (
        torch.from_numpy(x).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        / 127.5
        - 1
    )
