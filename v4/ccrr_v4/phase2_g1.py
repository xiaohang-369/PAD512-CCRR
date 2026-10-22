"""Supervised continuation of the author's E/D; phase 3/4 are not implemented here."""

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .original import OriginalGrayscale, quantize8
from .losses import author_loss
from .phase1.data import load_prepared, batch_tensor, pad512_bytes, write_json
from .phase1.optim import ContinuousBatches, TF1Adam
from .phase1.runner import digital_roundtrip, mean_metrics
from .phase1.vgg import AuthorVGG
from .phase2_baseline import numeric_policy, ordered_records
from .phase2_objectives import (
    validate_config,
    digital_supervision,
    medical_objective,
    retrieval_confusion,
)
from .verify import REPO, read, digest, state_digest, finite_gradients, source_check


def barrier(device):
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])


def verify_files(files):
    for path, expected in files.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Bound parent changed: {path}")


def rows_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def load_context(cfg, device, rank):
    from ccrr.data.pad import PADData
    from ccrr.teachers.adapters import PanDerm
    from ccrr_v2.reid import initialize
    from ccrr_v3.learning import load_separator
    from ccrr_v3.models import FrozenSupervisor

    inspection = read(cfg["inspection_report"])
    if not inspection["resource_checks_passed"]:
        raise ValueError("Resource inspection must pass")
    files = dict(inspection["checked_files"])
    verify_files(files)
    baseline = inspection["results"]["baseline"]
    supervision = inspection["results"]["supervision"]
    ck = torch.load(baseline["checkpoint"], map_location="cpu", weights_only=True)
    if (
        ck["epoch"] != 120
        or ck["step"] != 10800
        or ck["binding"]["code_commit"] != baseline["source_commit"]
    ):
        raise ValueError("Initialize only from completed original baseline")
    if digest(Path(cfg["prepared"])) != ck["binding"]["prepared_sha256"]:
        raise ValueError("Prepared data mismatch")
    p, arrays = load_prepared(cfg["prepared"], ck["binding"]["config"])
    datasets = {
        role: PADData(supervision["data_root"], role) for role in ("fit", "val_tune")
    }
    for role, data in datasets.items():
        if rows_digest(data.rows) != p["roles"][role]["rows_sha256"]:
            raise ValueError(f"{role} row/order mismatch")
        path = data.root / "protocol_audit_20261009/roles" / f"{role}.json"
        files[str(path)] = digest(path)
        files[str(Path(cfg["prepared"]).parent / p["roles"][role]["file"])] = p[
            "roles"
        ][role]["sha256"]
    root = datasets["fit"].root
    for relative in (
        "raw/metadata.csv",
        "protocol_audit_20261009/preprocessing_inventory.json",
        "protocol_audit_20261009/pair_protocol/val_tune_p2.json",
    ):
        path = root / relative
        files[str(path)] = digest(path)
    eval_dir = Path(cfg["baseline_evaluation"])
    previous = read(eval_dir / "report.json")
    if (
        read(eval_dir / "status.json")["state"] != "completed"
        or previous["state"] != "completed"
        or previous["images"] != len(datasets["val_tune"])
        or previous["optimizer_updates"] != 0
        or not previous["frozen_models_unchanged"]
        or not previous["parent_files_unchanged"]
        or previous["baseline_checkpoint"] != baseline
    ):
        raise ValueError("Require the bound completed frozen baseline evaluation")
    eb = read(eval_dir / "binding.json")
    if eb["inspection_sha256"] != digest(Path(cfg["inspection_report"])):
        raise ValueError("Evaluation and G1 resource bindings differ")
    protocol_path = root / "protocol_audit_20261009/pair_protocol/val_tune_p2.json"
    if eb["p2_protocol_sha256"] != digest(protocol_path):
        raise ValueError("Evaluation protocol changed")
    shards = []
    for r in range(previous["world_size"]):
        path = eval_dir / f"rank-{r}.json"
        files[str(path)] = digest(path)
        shards.append(read(path)["rows"])
    previous_rows = ordered_records(shards, len(datasets["val_tune"]))
    for row, source in zip(previous_rows, datasets["val_tune"].rows):
        if row["id"] != source["img_id"]:
            raise ValueError("Baseline prediction order mismatch")
    for path in [
        Path(cfg["inspection_report"]),
        Path(cfg["prepared"]),
        eval_dir / "report.json",
        eval_dir / "binding.json",
        eval_dir / "status.json",
    ]:
        files[str(path)] = digest(path)
    model = OriginalGrayscale().to(device)
    model.load_state_dict(ck["model"], strict=True)
    baseline_model = OriginalGrayscale().to(device).eval().requires_grad_(False)
    baseline_model.load_state_dict(ck["model"], strict=True)
    author_cfg = ck["binding"]["config"]
    del ck
    vgg = AuthorVGG.from_file(p["vgg"]["path"], p["vgg"]["sha256"]).to(device)
    files[p["vgg"]["path"]] = p["vgg"]["sha256"]
    pp = supervision["panderm"]
    panderm = (
        PanDerm(pp["source"], pp["source_sha256"], pp["path"], pp["sha256"])
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
    if set(teachers) != {"8500", "7000"}:
        raise ValueError("Keep registered medical supervision checkpoints")
    paths = [path for path in files if path.endswith("/private/resources.json")]
    if (
        len(paths) != 1
        or inspection["results"]["identity_reference"]["source"]
        != "public_ResNet50_initialization"
    ):
        raise ValueError("Unexpected identity reference")
    reid = (
        initialize(read(paths[0]), len(datasets["fit"].patients))
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    frozen = {
        "E0_D0": baseline_model,
        "vgg": vgg,
        "panderm": panderm,
        "public_resnet": reid,
        **teachers,
    }
    if rank == 0:
        print(
            "Bound original E0/D0, fit/val_tune, VGG, 8500/7000 and public reference; no test fitting",
            flush=True,
        )
    return dict(
        model=model,
        baseline=baseline_model,
        vgg=vgg,
        panderm=panderm,
        teachers=teachers,
        reid=reid,
        frozen=frozen,
        frozen_hashes={k: state_digest(m) for k, m in frozen.items()},
        datasets=datasets,
        arrays=arrays,
        author_cfg=author_cfg,
        parents=files,
        previous=previous,
        previous_rows=previous_rows,
        protocol=read(protocol_path),
    )


def check_frozen(ctx):
    for name, module in ctx["frozen"].items():
        if state_digest(module) != ctx["frozen_hashes"][name] or any(
            p.requires_grad or p.grad is not None for p in module.parameters()
        ):
            raise ValueError(f"Frozen model mutated: {name}")


@torch.no_grad()
def build_fit_bank(ctx, cfg, output, device, rank, world, tag, initial=None):
    """Only fit samples; immutable initial RGB/G0 targets, epoch-refreshed G1 bank."""
    data = ctx["datasets"]["fit"]
    rows = []
    ctx["model"].eval()
    for i in range(rank, len(data), world):
        item = data[i]
        if not np.array_equal(pad512_bytes(item["x"]), ctx["arrays"]["fit"][i]):
            raise ValueError("Fit image differs from original cache")
        x = item["x"].unsqueeze(0).to(device)
        with numeric_policy(False):
            g = (digital_roundtrip(ctx["model"].encoder(x * 2 - 1)) + 1) / 2
        row = dict(index=i, rect=item["rect"], label=item["label"])
        with numeric_policy(True):
            row["g1"] = ctx["reid"](g, [item["rect"]])["embedding"][0].cpu()
            if initial is None:
                row["rgb"] = ctx["reid"](x, [item["rect"]])["embedding"][0].cpu()
                raw, cov, _ = ctx["panderm"](g, [item["rect"]])
                row["target"] = (
                    ctx["teachers"]["8500"](raw, cov, 1)[0].softmax(1)[0].cpu()
                )
        rows.append(row)
        if len(rows) % cfg["bank_log_every_images"] == 0:
            print(
                f"FIT_BANK {tag} rank={rank} images={len(rows)}/{len(range(rank, len(data), world))}",
                flush=True,
            )
    path = output / f"bank-{tag}-rank{rank}.pt"
    torch.save(rows, path)
    barrier(device)
    all_rows = ordered_records(
        [
            torch.load(
                output / f"bank-{tag}-rank{r}.pt", map_location="cpu", weights_only=True
            )
            for r in range(world)
        ],
        len(data),
    )
    bank = dict(initial or {})
    bank["g1"] = torch.stack([r["g1"] for r in all_rows]).to(device)
    if initial is None:
        for key in ("rgb", "target"):
            bank[key] = torch.stack([r[key] for r in all_rows]).to(device)
        bank["rects"] = [r["rect"] for r in all_rows]
        bank["labels"] = torch.tensor([r["label"] for r in all_rows], device=device)
    if not all(torch.isfinite(bank[key]).all() for key in ("rgb", "target", "g1")):
        raise ValueError("Nonfinite fit reference")
    ctx["model"].train()
    return bank


def supervision_gradient(ctx, cfg, g, ids, bank):
    """Two detached gray VJPs free teacher graphs before decoder/VGG forward.

    Both are exact chain-rule terms for the stated V4 STE objective, no splitting
    the batch or accumulating optimizer updates. Teachers stay frozen/eval.
    """
    rects = [bank["rects"][i] for i in ids]
    values, gradients = {}, {}
    with numeric_policy(True):
        leaf = g.detach().requires_grad_(True)
        raw, cov, _ = ctx["panderm"](digital_supervision(leaf), rects)
        logits = ctx["teachers"]["8500"](raw, cov, 1)[0]
        med = medical_objective(
            logits,
            bank["labels"][ids],
            bank["target"][ids],
            ctx["class_weights"],
            cfg["supervision"]["medical_kd"],
        )
        gradients["medical"] = torch.autograd.grad(med, leaf)[0].detach()
        values["medical"] = med.detach()
        del leaf, raw, cov, logits, med
        leaf = g.detach().requires_grad_(True)
        embedding = ctx["reid"](digital_supervision(leaf), rects)["embedding"]
        cross = retrieval_confusion(
            embedding, bank["rgb"], ids, cfg["retrieval_temperature"]
        )
        same = retrieval_confusion(
            embedding, bank["g1"], ids, cfg["retrieval_temperature"]
        )
        identity = (cross + same) / 2
        gradients["identity"] = torch.autograd.grad(identity, leaf)[0].detach()
        values.update(
            identity=identity.detach(),
            identity_rgb=cross.detach(),
            identity_g1=same.detach(),
        )
    for key, grad in gradients.items():
        if not torch.isfinite(grad).all() or grad.abs().sum() <= 0:
            raise ValueError(f"Missing/nonfinite gray input gradient: {key}")
    combined = sum(cfg["supervision"][key] * grad for key, grad in gradients.items())
    norms = {
        key + "_gray_gradient_l1": float(grad.abs().sum())
        for key, grad in gradients.items()
    }
    return combined, values, norms


def backward_step(ctx, cfg, enc, dec, x, ids, bank, world):
    with numeric_policy(False):
        g = enc(x)
    grad, new_terms, norms = supervision_gradient(ctx, cfg, g, ids, bank)
    # Teacher graphs have been freed before the large original decoder graph exists.
    with numeric_policy(False):
        restored = dec(g)
        with torch.no_grad():
            target = ctx["vgg"](x)
        total, terms = author_loss(
            x,
            g,
            restored,
            target,
            ctx["vgg"](g),
            2,
            ctx["author_cfg"]["stages"][1]["weights"],
            distributed=world > 1,
        )
        if not torch.isfinite(total) or not all(
            torch.isfinite(v) for v in new_terms.values()
        ):
            raise ValueError("Nonfinite training objective")
        torch.autograd.backward((total, g), (torch.ones_like(total), grad))
    displayed = total.detach() + sum(
        cfg["supervision"][k] * new_terms[k] for k in ("medical", "identity")
    )
    return displayed, {**terms, **new_terms}, norms


@torch.no_grad()
def save_preview(ctx, output, device, epoch):
    from PIL import Image

    rows = []
    with numeric_policy(False):
        for i in range(min(4, len(ctx["arrays"]["val_tune"]))):
            x = batch_tensor(ctx["arrays"]["val_tune"], [i], device)
            g0 = digital_roundtrip(ctx["baseline"].encoder(x))
            g1 = digital_roundtrip(ctx["model"].encoder(x))
            y = (
                (x + 1)
                / 2
                * x.new_tensor([0.2989, 0.5870, 0.1140])[None, :, None, None]
            ).sum(1, keepdim=True) * 2 - 1
            panels = (
                x,
                y,
                g0,
                g1,
                ctx["model"].decoder(g1),
                ctx["baseline"].decoder(g1),
            )
            rows.append(
                np.concatenate(
                    [
                        quantize8(p.expand(-1, 3, -1, -1))[0][0]
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                        for p in panels
                    ],
                    axis=1,
                )
            )
    Image.fromarray(np.concatenate(rows, axis=0)).save(
        output / f"preview-{epoch:03d}.png"
    )
    write_json(
        output / "preview-layout.json",
        dict(
            role="val_tune",
            indices=list(range(len(rows))),
            columns=["RGB", "ordinary_gray8", "G0", "G1", "D1_G1", "D0_G1"],
            private_images=True,
        ),
    )


@torch.no_grad()
def evaluate(ctx, output, device, rank, world, epoch):
    from ccrr.evaluation.metrics import clinical
    from ccrr_v3.evaluation import p2_report

    data = ctx["datasets"]["val_tune"]
    ctx["model"].eval()
    rows = []
    for i in range(rank, len(data), world):
        item = data[i]
        if not np.array_equal(pad512_bytes(item["x"]), ctx["arrays"]["val_tune"][i]):
            raise ValueError("Validation pixels changed")
        x = item["x"].unsqueeze(0).to(device)
        with numeric_policy(False):
            g1 = digital_roundtrip(ctx["model"].encoder(x * 2 - 1))
            r1 = (ctx["model"].decoder(g1) + 1) / 2
            r0 = (ctx["baseline"].decoder(g1) + 1) / 2
            g0 = digital_roundtrip(ctx["baseline"].encoder(x * 2 - 1))
        with numeric_policy(True):
            raw, cov, _ = ctx["panderm"]((g1 + 1) / 2, [item["rect"]])
            probs = {
                k: t(raw, cov, 1)[0].softmax(1)[0].cpu().tolist()
                for k, t in ctx["teachers"].items()
            }
            emb = (
                ctx["reid"]((g1 + 1) / 2, [item["rect"]])["embedding"][0].cpu().tolist()
            )
            recovered_embeddings = {
                name: ctx["reid"](image, [item["rect"]])["embedding"][0].cpu().tolist()
                for name, image in (("D0_G1", r0), ("D1_G1", r1))
            }

        def psnr(y):
            return float(-10 * (y - x).square().mean().clamp_min(1e-10).log10())

        rows.append(
            dict(
                index=i,
                id=item["id"],
                label=item["label"],
                probabilities=probs,
                embedding=emb,
                recovered_embeddings=recovered_embeddings,
                D1_psnr=psnr(r1),
                D0_direct_psnr=psnr(r0),
                G1_G0_gray_mae=float((g1 - g0).abs().mean() * 127.5),
            )
        )
    write_json(output / f"eval-{epoch:03d}-rank{rank}.json", rows)
    barrier(device)
    summary = None
    if rank == 0:
        rows = ordered_records(
            [read(output / f"eval-{epoch:03d}-rank{r}.json") for r in range(world)],
            len(data),
        )
        med = {
            k: clinical(
                [r["label"] for r in rows], [r["probabilities"][k] for r in rows]
            )
            for k in ctx["teachers"]
        }
        emb = np.asarray([r["embedding"] for r in rows], dtype=np.float64)
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        rgb = np.asarray(
            [r["embeddings"]["rgb"] for r in ctx["previous_rows"]], dtype=np.float64
        )
        rgb /= np.maximum(np.linalg.norm(rgb, axis=1, keepdims=True), 1e-8)
        linkage = {
            name: p2_report(
                data.rows,
                ctx["protocol"],
                lambda i, js, gallery=gal: (gallery[js] @ emb[i]).tolist(),
            )
            for name, gal in (("G1_to_rgb", rgb), ("G1_to_G1", emb))
        }
        for name in ("D0_G1", "D1_G1"):
            rec = np.asarray(
                [r["recovered_embeddings"][name] for r in rows], dtype=np.float64
            )
            rec /= np.maximum(np.linalg.norm(rec, axis=1, keepdims=True), 1e-8)
            linkage[name + "_to_rgb"] = p2_report(
                data.rows,
                ctx["protocol"],
                lambda i, js, q=rec: (rgb[js] @ q[i]).tolist(),
            )
        delta = {
            k: {
                ref: {
                    metric: med[k][metric]
                    - ctx["previous"]["medical_bound_heads"][k][ref][metric]
                    for metric in ("macro_auroc", "macro_f1", "balanced_accuracy")
                }
                for ref in ("ordinary_gray8", "G0_png")
            }
            for k in med
        }
        summary = dict(
            epoch=epoch,
            images=len(rows),
            role="val_tune",
            medical=med,
            medical_deltas=delta,
            p2_fixed_reference={
                k: {n: v for n, v in r.items() if n != "queries"}
                for k, r in linkage.items()
            },
            recovery={
                k: float(np.mean([r[k] for r in rows]))
                for k in ("D1_psnr", "D0_direct_psnr", "G1_G0_gray_mae")
            },
            medical_acceptance_threshold=None,
            scientific_acceptance="not_evaluated",
            fixed_training_reference_not_independent_privacy_acceptance=True,
        )
        write_json(
            output / f"evaluation-{epoch:03d}.json", {**summary, "p2_details": linkage}
        )
        save_preview(ctx, output, device, epoch)
        print("G1_EVAL " + json.dumps(summary), flush=True)
    ctx["model"].train()
    return summary


def run(cfg, output, mode, preflight=None):
    validate_config(cfg)
    world, rank, local = (
        int(os.environ.get(k, d))
        for k, d in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))
    )
    if world != len(cfg["gpu_ids"]) or torch.cuda.device_count() != world:
        raise ValueError("Explicit GPU count must match launcher")
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
        ctx = load_context(cfg, device, rank)
        ctx["class_weights"] = ctx["datasets"]["fit"].class_weights().to(device)
        binding = dict(
            config=cfg,
            code_commit=commit,
            parents=ctx["parents"],
            source=source_check(),
            original_loss_weights=ctx["author_cfg"]["stages"][1]["weights"],
            numeric_policy="E_D_VGG_FP32_no_TF32_teachers_cudnn_TF32_no_matmul_TF32",
            new_supervision_quantization="author_8bit_forward_STE_backward_V4_only",
        )
        if rank == 0:
            write_json(output / "binding.json", binding)
        if mode == "train":
            report = read(preflight)
            if (
                read(Path(preflight).parent / "status.json")["state"] != "completed"
                or report["mode"] != "preflight"
                or not report["passed"]
                or report["binding"] != binding
                or report["optimizer_updates"] != 0
            ):
                raise ValueError(
                    "Training requires a passed preflight for this exact config/release/parents"
                )
            bank_path = Path(preflight).parent / report["bank"]["file"]
            if digest(bank_path) != report["bank"]["sha256"]:
                raise ValueError("Preflight fit bank changed")
            bank = torch.load(bank_path, map_location=device, weights_only=True)
            if rank == 0:
                write_json(
                    output / "preflight-binding.json",
                    dict(path=str(preflight), sha256=digest(Path(preflight))),
                )
        else:
            bank = build_fit_bank(ctx, cfg, output, device, rank, world, "initial")
            if rank == 0:
                torch.save(
                    {
                        k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in bank.items()
                    },
                    output / "fit-bank.pt",
                )
            barrier(device)
        enc = (
            DDP(ctx["model"].encoder, device_ids=[local])
            if world > 1
            else ctx["model"].encoder
        )
        dec = (
            DDP(ctx["model"].decoder, device_ids=[local])
            if world > 1
            else ctx["model"].decoder
        )
        optimizer = TF1Adam(ctx["model"].parameters(), lr=cfg["learning_rate"])
        stream = ContinuousBatches(len(ctx["arrays"]["fit"]), 16, cfg["seed"])
        local_batch = 16 // world
        steps_per_epoch = math.ceil(len(ctx["arrays"]["fit"]) / 16)
        before = state_digest(ctx["model"])
        if mode == "preflight":
            ids = stream.next()[rank * local_batch : (rank + 1) * local_batch]
            x = batch_tensor(ctx["arrays"]["fit"], ids, device)
            torch.cuda.reset_peak_memory_stats(device)
            total, terms, norms = backward_step(ctx, cfg, enc, dec, x, ids, bank, world)
            grads = finite_gradients(ctx["model"])
            if before != state_digest(ctx["model"]) or before != state_digest(
                ctx["baseline"]
            ):
                raise ValueError("Preflight changed original initialization")
            metrics = mean_metrics(terms, total, world)
            check_frozen(ctx)
            write_json(
                output / f"preflight-rank{rank}.json",
                dict(
                    gradients=grads,
                    supervision_gradients=norms,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            del x, terms, total
            initial_eval = evaluate(ctx, output, device, rank, world, 0)
            verify_files(ctx["parents"])
            barrier(device)
            if rank == 0:
                write_json(
                    output / "report.json",
                    dict(
                        mode="preflight",
                        passed=True,
                        binding=binding,
                        optimizer_updates=0,
                        frozen_models_unchanged=True,
                        parents_unchanged=True,
                        initial_G1_equals_G0=True,
                        losses=metrics,
                        initial_evaluation=initial_eval,
                        ranks=[
                            read(output / f"preflight-rank{r}.json")
                            for r in range(world)
                        ],
                        bank=dict(
                            file="fit-bank.pt", sha256=digest(output / "fit-bank.pt")
                        ),
                    ),
                )
        else:
            step = 0
            final = None
            for epoch in range(1, cfg["epochs"] + 1):
                if epoch > 1:
                    bank = build_fit_bank(
                        ctx, cfg, output, device, rank, world, f"epoch{epoch:03d}", bank
                    )
                started, sums = time.monotonic(), {}
                for _ in range(steps_per_epoch):
                    ids = stream.next()[rank * local_batch : (rank + 1) * local_batch]
                    x = batch_tensor(ctx["arrays"]["fit"], ids, device)
                    optimizer.zero_grad(set_to_none=True)
                    total, terms, norms = backward_step(
                        ctx, cfg, enc, dec, x, ids, bank, world
                    )
                    finite_gradients(ctx["model"])
                    optimizer.step()
                    step += 1
                    metrics = mean_metrics(terms, total, world)
                    for k, v in metrics.items():
                        sums[k] = sums.get(k, 0.0) + v
                    if rank == 0 and step % cfg["log_every_steps"] == 0:
                        print(
                            "G1_STEP "
                            + json.dumps(dict(epoch=epoch, step=step, **metrics)),
                            flush=True,
                        )
                    del x, total, terms
                optimizer.zero_grad(set_to_none=True)
                validation = evaluate(ctx, output, device, rank, world, epoch)
                check_frozen(ctx)
                barrier(device)
                if rank == 0:
                    path = output / f"g1-epoch-{epoch:03d}.pt"
                    torch.save(
                        dict(
                            schema="ccrr-v4-g1-checkpoint-1",
                            model=ctx["model"].state_dict(),
                            optimizer=optimizer.state_dict(),
                            epoch=epoch,
                            step=step,
                            binding=binding,
                        ),
                        path,
                    )
                    final = dict(
                        file=path.name, sha256=digest(path), epoch=epoch, step=step
                    )
                    row = dict(
                        epoch=epoch,
                        step=step,
                        seconds=time.monotonic() - started,
                        train={k: v / steps_per_epoch for k, v in sums.items()},
                        validation=validation,
                        checkpoint=final,
                    )
                    with (output / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(row, allow_nan=False) + "\n")
                barrier(device)
            verify_files(ctx["parents"])
            if rank == 0:
                write_json(
                    output / "report.json",
                    dict(
                        mode="train",
                        training_completed=True,
                        optimizer_updates=step,
                        final_checkpoint=final,
                        final_validation=validation,
                        binding=binding,
                        frozen_models_unchanged=True,
                        parents_unchanged=True,
                        medical_acceptance_threshold=None,
                        scientific_acceptance="pending_review",
                        test_used_for_performance=False,
                        next_phase="stop",
                    ),
                )
        barrier(device)
        if rank == 0:
            print(f"G1_REPORT={output / 'report.json'}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    if args.mode == "train" and args.preflight is None:
        parser.error("train requires --preflight REPORT_JSON")
    run(read(args.config), args.output, args.mode, args.preflight)


if __name__ == "__main__":
    main()
