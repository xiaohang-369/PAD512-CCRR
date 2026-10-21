"""No-optimizer verification. All outputs are new, configuration and hashes recorded."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .config import load_verification
from .original import OriginalGrayscale, quantize8, scope_map
from .losses import source_terms

REPO = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def state_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_check():
    upstream = REPO / "v4" / "upstream"
    manifest = read(upstream / "provenance.json")
    for name, expected in manifest["files"].items():
        if digest(upstream / "InvertibleGrayscale" / name) != expected:
            raise ValueError(f"Author source changed: {name}")
    return {
        "author_commit": manifest["commit"],
        "files_verified": len(manifest["files"]),
        "original_tensorflow_runtime_executed": False,
    }


def finite_gradients(model):
    result = {}
    for name, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        gradients = [p.grad for p in module.parameters()]
        if not all(g is not None and torch.isfinite(g).all() for g in gradients):
            raise ValueError(f"Missing/nonfinite {name} gradient")
        total = sum(float(g.abs().sum()) for g in gradients)
        if total <= 0:
            raise ValueError(f"Zero {name} gradient")
        result[name] = total
    return result


def synthetic_check(cfg, device, output):
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = OriginalGrayscale().to(device)
    before = state_digest(model)
    x = (
        torch.rand(cfg["per_device_batch_size"], 3, *cfg["image_size"], device=device)
        * 2
        - 1
    )
    g, rgb = model(x)
    if g.shape != (len(x), 1, *x.shape[-2:]) or rgb.shape != x.shape:
        raise ValueError("Original architecture shape mismatch")
    terms = source_terms(x, g, rgb)
    # No perceptual-loss substitution: this ONLY checks the E->G->D graph.
    loss = 3 * terms["mse"] + terms["lightness"] + 0.5 * terms["variation"]
    loss.backward()
    gradients = finite_gradients(model)
    if state_digest(model) != before:
        raise ValueError("Verification mutated model state")
    with torch.no_grad():
        integers, received = quantize8(g[:1])
        path = output / f"synthetic-{str(device).replace(':', '-')}.png"
        Image.fromarray(integers[0, 0].cpu().numpy()).save(path)
        with Image.open(path) as im:
            if im.mode != "L":
                raise ValueError("Receive one-channel 8-bit PNG")
            reloaded = torch.from_numpy(np.array(im, copy=True)).to(device)[None, None]
        if not torch.equal(reloaded, integers):
            raise ValueError("PNG integers changed")
        a = model.decoder(received)
        b = model.decoder(reloaded.float() / 127.5 - 1)
        if not torch.equal(a, b):
            raise ValueError("Digital decoder input mismatch")
    return dict(
        device=str(device),
        batch_size=len(x),
        image_size=list(x.shape[-2:]),
        parameters=sum(p.numel() for p in model.parameters()),
        kernels=len(scope_map(model)),
        gradients=gradients,
        optimizer_updates=0,
        weights_unchanged=True,
        png_exact=True,
        quality_evaluated=False,
        vgg_loss_verified=False,
        peak_cuda_bytes=(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    )


def supervisor_check(cfg, device):
    # V3 is used only for its already-trained teacher and existing data adapter.
    from ccrr.data.pad import PADData, collate
    from ccrr.teachers.adapters import PanDerm
    from ccrr.tensor_ops import gray
    from ccrr_v3.runtime import validate_resources
    from ccrr_v3.learning import load_separator, checkpoint_path
    from ccrr_v3.grayscale import PixelSupervisor

    resource_path = Path(cfg["resources_file"])
    resource = read(resource_path)
    if resource["schema"] != "ccrr-v4-readonly-parent-resources-1":
        raise ValueError("Wrong parent resource schema")
    if cfg["image_size"] != [512, 512]:
        raise ValueError("Bound PAD/A0 resources require 512 pixels")
    if cfg["per_device_batch_size"] != resource["reference_cache_batch_size"]:
        raise ValueError(
            "A0 comparison must retain its reference batch, no silent microbatching"
        )
    parent_path = Path(resource["parent_resources"])
    parents = {
        str(resource_path): digest(resource_path),
        str(parent_path): digest(parent_path),
    }
    parent = read(parent_path)
    validate_resources(parent)
    run, feature_root = Path(resource["separator"]), Path(resource["features"])
    manifest, binding = read(run / "artifact.json"), read(run / "binding.json")
    if (
        read(run / "status.json")["state"] != "completed"
        or manifest["kind"] != "separator"
    ):
        raise ValueError("Require a completed separator parent")
    if (
        manifest["binding_sha256"] != digest(run / "binding.json")
        or binding["resources"] != parent
    ):
        raise ValueError("Separator parent binding mismatch")
    feature = read(feature_root / "features.json")
    if binding["parents"]["features_sha256"] != digest(feature_root / "features.json"):
        raise ValueError("Separator/A0 feature binding mismatch")
    step = resource["supervisor_step"]
    if step not in (8500, 7000):
        raise ValueError(
            "Current selected teacher steps are 8500 primary / 7000 control"
        )
    checkpoint = checkpoint_path(run, step)
    if digest(checkpoint) != manifest["files"][checkpoint.name]:
        raise ValueError("Selected checkpoint hash mismatch")
    for path in (
        run / "artifact.json",
        run / "binding.json",
        run / "status.json",
        checkpoint,
        feature_root / "features.json",
    ):
        parents[str(path)] = digest(path)
    for entry in (feature["statistics"], feature["roles"]["fit"]):
        path = feature_root / entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError("A0 payload hash mismatch")
        parents[str(path)] = entry["sha256"]
    p = parent["panderm"]
    for path, sha in ((p["source"], p["source_sha256"]), (p["path"], p["sha256"])):
        parents[path] = sha
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    backbone = PanDerm(p["source"], p["source_sha256"], p["path"], p["sha256"]).to(
        device
    )
    separator, heads, stats = load_separator(checkpoint, device)
    original_stats = torch.load(
        feature_root / feature["statistics"]["file"],
        map_location="cpu",
        weights_only=True,
    )
    if not all(torch.equal(stats[k].cpu(), original_stats[k]) for k in ("mean", "std")):
        raise ValueError("Teacher statistics differ from A0")
    teacher = PixelSupervisor(backbone, separator, heads, stats).to(device)
    teacher_before = state_digest(teacher)
    data = PADData(parent["data_root"], "fit")
    count = cfg["per_device_batch_size"]
    batch = collate([data[i] for i in range(count)], device)
    cache = torch.load(
        feature_root / feature["roles"]["fit"]["file"],
        map_location="cpu",
        weights_only=True,
    )
    if cache["rows"] != data.rows:
        raise ValueError("A0 image ordering changed")
    expected, expected_coverage = (
        cache["features"][:count].clone(),
        cache["coverage"][:count].clone(),
    )
    del cache
    cache_errors = []
    with torch.no_grad():
        for domain, image in enumerate((batch["x"], gray(batch["x"]))):
            features, coverage, _ = teacher.panderm(image, batch["rect"])
            if not torch.allclose(coverage.cpu(), expected_coverage, atol=1e-7, rtol=0):
                raise ValueError("A0 coverage changed")
            error = float((features.cpu() - expected[:, domain]).abs().max())
            if not torch.allclose(
                features.cpu(), expected[:, domain], atol=1e-4, rtol=1e-4
            ):
                raise ValueError(f"A0 feature mismatch, domain={domain}, error={error}")
            cache_errors.append(error)
    torch.manual_seed(cfg["seed"])
    model = OriginalGrayscale().to(device)
    model_before = state_digest(model)
    g, rgb = model(batch["x"] * 2 - 1)
    loss = F.cross_entropy(teacher((g + 1) / 2, batch["rect"], 1)[0], batch["label"])
    loss = loss + F.cross_entropy(
        teacher((rgb + 1) / 2, batch["rect"], 0)[0], batch["label"]
    )
    loss.backward()
    gradients = finite_gradients(model)
    if state_digest(teacher) != teacher_before or state_digest(model) != model_before:
        raise ValueError("Read-only gradient check mutated a model")
    if any(p.grad is not None or p.requires_grad for p in teacher.parameters()):
        raise ValueError("Supervisor parameters were not frozen")
    for path, sha in parents.items():
        if digest(path) != sha:
            raise ValueError(f"Parent changed during verification: {path}")
    return dict(
        device=str(device),
        supervisor_step=step,
        fit_images=count,
        cache_errors=cache_errors,
        gradients=gradients,
        parent_hashes=parents,
        parents_unchanged=True,
        teacher_frozen=True,
        optimizer_updates=0,
        generated_image_quality_evaluated=False,
        privacy_evaluated=False,
        numeric_policy="float32_cudnn_tf32_true_matmul_tf32_false",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--devices", help="comma-separated process-local devices, or all-cuda"
    )
    ap.add_argument("--resources", help="explicit read-only parent-resource JSON")
    args = ap.parse_args()
    devices = None if args.devices is None else args.devices.split(",")
    if args.devices == "all-cuda":
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    cfg = load_verification(args.config, devices, args.resources)
    output = Path(args.output).resolve()
    if REPO == output or REPO in output.parents:
        raise ValueError("Write verification evidence outside the source worktree")
    output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    report = dict(
        state="running",
        scope="engineering_verification_only",
        config=cfg,
        config_file_sha256=digest(args.config),
        training_started=False,
        optimizer_updates=0,
        scientific_acceptance="not_evaluated",
        time_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        devices=[],
    )
    try:
        git_root = REPO
        report["source_commit"] = subprocess.check_output(
            ["git", "-C", str(git_root), "rev-parse", "HEAD"], text=True
        ).strip()
        report["source_dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", str(git_root), "status", "--porcelain"], text=True
            ).strip()
        )
        if report["source_dirty"]:
            raise ValueError("Commit verification code first")
        report["upstream"] = source_check()
        torch.set_num_threads(cfg["cpu_threads"])
        for name in cfg["devices"]:
            device = torch.device(name)
            if device.type == "cuda":
                torch.cuda.set_device(device)
            result = synthetic_check(cfg, device, output)
            if cfg["verify_supervisor"]:
                result["supervisor"] = supervisor_check(cfg, device)
            report["devices"].append(result)
        report["state"] = "completed"
    except BaseException as exc:
        report.update(
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        (output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        print(
            "V4_REPORT:", output / "report.json", "state=" + report["state"], flush=True
        )


if __name__ == "__main__":
    main()
