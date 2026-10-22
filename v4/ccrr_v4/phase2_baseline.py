"""Frozen G0/ordinary-gray clinical and P2 reference evaluation; never trains."""

import argparse
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .original import OriginalGrayscale
from .phase1.data import load_prepared, pad512_bytes, write_json
from .phase1.runner import digital_roundtrip
from .verify import digest, read, state_digest


VIEWS = ("rgb", "ordinary_gray8", "G0_png")


def validate_config(cfg):
    if (
        cfg["schema"] != "ccrr-v4-phase2-baseline-1"
        or cfg["training_enabled"] is not False
    ):
        raise ValueError("Only frozen baseline evaluation is implemented")
    if cfg["role"] != "val_tune" or tuple(cfg["views"]) != VIEWS:
        raise ValueError("Use registered val_tune comparison only")
    if cfg["supervisor_steps"] != [8500, 7000] or cfg["evaluation_batch_size"] != 1:
        raise ValueError("Preserve teacher selection and single-image evaluation")
    if cfg["identity_reference"] != "public_ResNet50_initialization":
        raise ValueError("Different identity reference requires explicit binding")
    ids = cfg["gpu_ids"]
    if not ids or len(set(map(str, ids))) != len(ids):
        raise ValueError("Configure unique physical GPU IDs")
    if cfg["cpu_threads"] < 1 or cfg["log_every_images"] < 1:
        raise ValueError("Positive runtime limits required")


@contextmanager
def numeric_policy(teacher=False):
    old = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = teacher
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


def views_from_model(model, x):
    """x is verified RGB in [0,1]; both public comparison views are true 8bit."""
    with numeric_policy(False):
        g = digital_roundtrip(model.encoder(x * 2 - 1))
        reconstructed = (model.decoder(g) + 1) / 2
    y = (x * x.new_tensor([0.2989, 0.5870, 0.1140])[None, :, None, None]).sum(
        1, keepdim=True
    )
    y8 = torch.floor(y.clamp(0, 1) * 255 + 0.5) / 255
    psnr = -10 * (reconstructed - x).square().mean((1, 2, 3)).clamp_min(1e-10).log10()
    return {"rgb": x, "ordinary_gray8": y8, "G0_png": (g + 1) / 2}, psnr


def ordered_records(shards, size):
    rows = sorted((row for shard in shards for row in shard), key=lambda r: r["index"])
    if [r["index"] for r in rows] != list(range(size)):
        raise ValueError("Missing/duplicated evaluation image indices")
    return rows


def run(cfg, output):
    from ccrr.data.pad import PADData
    from ccrr.evaluation.metrics import clinical
    from ccrr.teachers.adapters import PanDerm
    from ccrr_v2.reid import initialize
    from ccrr_v3.learning import load_separator
    from ccrr_v3.models import FrozenSupervisor
    from ccrr_v3.evaluation import p2_report

    validate_config(cfg)
    world, rank = (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
    )
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world != len(cfg["gpu_ids"]):
        raise ValueError("Launch with config-selected device count")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        inspection = read(cfg["inspection_report"])
        if (
            inspection["schema"] != "ccrr-v4-phase2-inspection-report-1"
            or not inspection["resource_checks_passed"]
        ):
            raise ValueError("Require passed phase-2 resource inspection")
        if rank == 0:
            print("Checking bound parent hashes; no model fitting", flush=True)
        parents = inspection["checked_files"]
        for name, sha in parents.items():
            if digest(Path(name)) != sha:
                raise ValueError(f"Parent changed since inspection: {name}")
        baseline = inspection["results"]["baseline"]
        supervision = inspection["results"]["supervision"]
        identity = inspection["results"]["identity_reference"]
        if (
            identity["source"] != cfg["identity_reference"]
            or identity["reference"] is not None
        ):
            raise ValueError("Identity reference differs from public initialization")
        checkpoint = Path(baseline["checkpoint"])
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            ck["epoch"] != 120
            or ck["step"] != 10800
            or ck["binding"]["code_commit"] != baseline["source_commit"]
        ):
            raise ValueError("Baseline checkpoint internal endpoint/binding mismatch")
        if digest(Path(cfg["prepared"])) != ck["binding"]["prepared_sha256"]:
            raise ValueError("Prepared data differs from original training")
        prepared, arrays = load_prepared(cfg["prepared"], ck["binding"]["config"])
        data = PADData(supervision["data_root"], "val_tune")
        rows_sha = hashlib.sha256(
            json.dumps(data.rows, sort_keys=True).encode()
        ).hexdigest()
        if rows_sha != prepared["roles"]["val_tune"]["rows_sha256"]:
            raise ValueError("Dataset order differs from baseline cache")
        protocol_path = (
            data.root / "protocol_audit_20261009/pair_protocol/val_tune_p2.json"
        )
        protocol = read(protocol_path)
        protocol_sha = digest(protocol_path)
        model = OriginalGrayscale().to(device).eval().requires_grad_(False)
        model.load_state_dict(ck["model"], strict=True)
        del ck
        p = supervision["panderm"]
        panderm = (
            PanDerm(p["source"], p["source_sha256"], p["path"], p["sha256"])
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        teachers = {}
        for entry in supervision["steps"]:
            sep, heads, stats = load_separator(Path(entry["checkpoint"]), device)
            teachers[str(entry["step"])] = FrozenSupervisor(
                sep, heads, stats["mean"], stats["std"]
            ).to(device)
        if set(teachers) != set(map(str, cfg["supervisor_steps"])):
            raise ValueError("Unexpected supervisor steps")
        # Only pretrained embeddings are scored; random classifier/pair heads are unused.
        resource_paths = [
            name for name in parents if name.endswith("/private/resources.json")
        ]
        if len(resource_paths) != 1:
            raise ValueError("Expected one bound parent resource configuration")
        resource = read(resource_paths[0])
        patients = len(PADData(resource["data_root"], "fit").patients)
        reid = initialize(resource, patients).to(device).eval().requires_grad_(False)
        modules = {"baseline": model, "panderm": panderm, "reference": reid, **teachers}
        before = {k: state_digest(m) for k, m in modules.items()}
        results = []
        torch.cuda.reset_peak_memory_stats(device)
        for i in range(rank, len(data), world):
            item = data[i]
            if not np.array_equal(pad512_bytes(item["x"]), arrays["val_tune"][i]):
                raise ValueError("Image pixels differ from baseline cache")
            x = item["x"].unsqueeze(0).to(device)
            with torch.no_grad():
                images, psnr = views_from_model(model, x)
                row = dict(
                    index=i,
                    id=item["id"],
                    label=item["label"],
                    probabilities={},
                    embeddings={},
                    png_psnr=float(psnr[0]),
                )
                with numeric_policy(True):
                    for view, image in images.items():
                        raw, coverage, _ = panderm(image, [item["rect"]])
                        domain = 0 if view == "rgb" else 1
                        row["probabilities"][view] = {
                            step: teacher(raw, coverage, domain)[0]
                            .softmax(1)[0]
                            .cpu()
                            .tolist()
                            for step, teacher in teachers.items()
                        }
                        emb = reid(image, [item["rect"]])["embedding"][0]
                        if not torch.isfinite(emb).all() or emb.norm() <= 1e-8:
                            raise ValueError("Invalid reference embedding")
                        row["embeddings"][view] = emb.cpu().tolist()
            results.append(row)
            if len(results) % cfg["log_every_images"] == 0:
                print(
                    f"EVAL rank={rank} images={len(results)}/{len(range(rank, len(data), world))}",
                    flush=True,
                )
        for name, module in modules.items():
            if state_digest(module) != before[name] or any(
                p.grad is not None or p.requires_grad for p in module.parameters()
            ):
                raise ValueError(f"Frozen model changed: {name}")
        write_json(
            output / f"rank-{rank}.json",
            dict(rows=results, peak_cuda_bytes=torch.cuda.max_memory_allocated(device)),
        )
        if world > 1:
            dist.barrier(device_ids=[local])
        if rank == 0:
            shard_reports = [read(output / f"rank-{r}.json") for r in range(world)]
            rows = ordered_records([r["rows"] for r in shard_reports], len(data))
            labels = [r["label"] for r in rows]
            medical = {
                step: {
                    view: clinical(
                        labels, [r["probabilities"][view][step] for r in rows]
                    )
                    for view in VIEWS
                }
                for step in teachers
            }
            embeddings = {
                v: np.asarray([r["embeddings"][v] for r in rows], dtype=np.float64)
                for v in VIEWS
            }
            embeddings = {
                k: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
                for k, v in embeddings.items()
            }
            pairs = [
                ("rgb", "rgb"),
                ("ordinary_gray8", "rgb"),
                ("ordinary_gray8", "ordinary_gray8"),
                ("G0_png", "rgb"),
                ("G0_png", "G0_png"),
            ]
            linkage = {}
            for query, gallery in pairs:

                def score(index, indices):
                    return (
                        embeddings[gallery][indices] @ embeddings[query][index]
                    ).tolist()

                linkage[f"{query}_to_{gallery}"] = p2_report(data.rows, protocol, score)
            # Recheck parents after evaluation; all writes remain in this new run.
            for name, sha in parents.items():
                if digest(Path(name)) != sha:
                    raise ValueError(f"Parent changed during evaluation: {name}")
            if digest(protocol_path) != protocol_sha:
                raise ValueError("Pair protocol changed during evaluation")
            report = dict(
                mode="phase2_baseline_evaluation",
                state="completed",
                images=len(data),
                role="val_tune",
                views=list(VIEWS),
                medical_bound_heads=medical,
                p2_public_resnet_reference=linkage,
                png_psnr=float(np.mean([r["png_psnr"] for r in rows])),
                baseline_report_png_psnr=baseline["final_validation"]["png_psnr"],
                baseline_checkpoint=baseline,
                identity_reference=identity,
                numeric_policy=dict(
                    dtype="float32",
                    baseline_cudnn_tf32=False,
                    readout_cudnn_tf32=True,
                    matmul_tf32=False,
                    evaluation_batch_size=1,
                    a0_cache_bitwise_equality_claimed=False,
                ),
                world_size=world,
                peak_cuda_bytes_by_rank=[r["peak_cuda_bytes"] for r in shard_reports],
                frozen_models_unchanged=True,
                parent_files_unchanged=True,
                optimizer_updates=0,
                test_used_for_performance=False,
                phase2_training_started=False,
                medical_scope="original_bound_supervisor_heads_not_independent_clinical_acceptance",
                identity_scope="fixed_public_reference_P2_same_lesion_linkage_not_adaptive_attack_or_general_patient_identity",
            )
            write_json(
                output / "binding.json",
                dict(
                    config=cfg,
                    inspection_sha256=digest(Path(cfg["inspection_report"])),
                    checked_files=parents,
                    p2_protocol_sha256=protocol_sha,
                ),
            )
            write_json(output / "report.json", report)
            print(f"PHASE2_BASELINE_REPORT={output / 'report.json'}", flush=True)
    finally:
        if world > 1:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(read(args.config), args.output)


if __name__ == "__main__":
    main()
