"""Fresh patient ReID from public ResNet50, never a previous learned teacher."""

from copy import deepcopy
import json
from time import perf_counter
import torch
from torch import nn

from ccrr.data.pad import PADData, PairSampler, collate
from ccrr.teachers.adapters import IdentityModel, freeze_bn
from ccrr.tensor_ops import gray
from ccrr.evaluation.teachers import teacher_identity_scores, reid_gate
from .runtime import completed, read, save_model, seed_all, write
from .loading import ReIDBatchIndices, cpu_batches, to_device


def initialize(resources, patients):
    model = IdentityModel(patients)
    state = torch.load(
        resources["resnet50"]["path"], map_location="cpu", weights_only=True
    )
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    model.model.load_state_dict(state, strict=True)
    # Only the new classifier and backbone participate. Untrained pair heads are not used for evaluation.
    for part in (model.project, model.pair):
        for p in part.parameters():
            p.requires_grad_(False)
    return model


def triplet_loss(embedding, labels, margin=0.3):
    distance = torch.cdist(embedding, embedding)
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    negative = ~same
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return embedding.sum() * 0
    hardest_pos = distance.masked_fill(~positive, -torch.inf).max(1).values
    hardest_neg = distance.masked_fill(~negative, torch.inf).min(1).values
    return (hardest_pos[valid] - hardest_neg[valid] + margin).clamp_min(0).mean()


def embeddings(model, data, device, batch_size):
    result = [{}, {}]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            b = collate(
                [data[i] for i in range(start, min(start + batch_size, len(data)))],
                device,
            )
            for domain, image in enumerate((b["x"], gray(b["x"]))):
                f = model(image, b["rect"])["embedding"].cpu()
                result[domain].update(
                    {name: {"embedding": f[i]} for i, name in enumerate(b["id"])}
                )
    return result


def evaluate(model, data, device, cfg, bootstrap=False):
    features = embeddings(model, data, device, cfg["batch_size"])
    reports = {}
    for scenario in ("p1", "p2"):
        protocol = read(
            data.root
            / "protocol_audit_20261009"
            / "pair_protocol"
            / f"{data.role}_{scenario}.json"
        )
        domains = []
        for da, db in ((0, 0), (1, 0), (1, 1)):
            domains.append(
                teacher_identity_scores(
                    protocol,
                    data.rows,
                    features[da],
                    features[db],
                    bootstrap=bootstrap,
                    repetitions=cfg["bootstrap_repetitions"],
                )
            )
        reports[scenario] = {"reports": domains}
        if bootstrap:
            reports[scenario].update(reid_gate(domains))
    return reports


def train(resources, cfg, output, device, *, loader_workers=0, prefetch_batches=2):
    seed_all(cfg["seed"])
    fit = PADData(resources["data_root"], "fit")
    tune = PADData(resources["data_root"], "val_tune")
    select = PADData(resources["data_root"], "val_select")
    model = initialize(resources, len(fit.patients)).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["reid_lr"]
    )
    sampler = PairSampler(fit)
    batches = iter(
        cpu_batches(
            fit,
            ReIDBatchIndices(sampler, cfg["seed"], cfg["reid_steps"]),
            workers=loader_workers,
            prefetch=prefetch_batches,
            seed=cfg["seed"],
            pin_memory=loader_workers > 0 and torch.device(device).type == "cuda",
        )
    )
    print(
        f"REID_LOADER workers={loader_workers} prefetch_per_worker={prefetch_batches} "
        "ordered=True sampler=legacy_parent_rng",
        flush=True,
    )
    best, selected, best_step = -float("inf"), None, None
    window_start, data_wait = perf_counter(), 0.0
    previous_logged_step = 0
    for step in range(1, cfg["reid_steps"] + 1):
        model.train()
        freeze_bn(model)
        wait_start = perf_counter()
        cpu_batch = next(batches)
        data_wait += perf_counter() - wait_start
        batch = to_device(cpu_batch, device, non_blocking=loader_workers > 0)
        image = batch["x"] if step % 2 else gray(batch["x"])
        optimizer.zero_grad(set_to_none=True)
        result = model(image, batch["rect"])
        ce = nn.functional.cross_entropy(result["logits"], batch["patient_index"])
        metric = triplet_loss(result["embedding"], batch["patient_index"])
        loss = ce + metric
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite ReID objective")
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == cfg["reid_steps"]:
            # Scalar reads wait for the GPU work before timing the window.
            ce_value, metric_value = float(ce.detach()), float(metric.detach())
            elapsed = perf_counter() - window_start
            count = step - previous_logged_step
            timing = {
                "step": step,
                "window_steps": count,
                "ce": ce_value,
                "triplet": metric_value,
                "wall_seconds": elapsed,
                "data_wait_seconds": data_wait,
                "steps_per_second": count / elapsed,
                "includes_worker_startup": previous_logged_step == 0,
            }
            with (output / "training.jsonl").open("a") as f:
                f.write(json.dumps(timing, allow_nan=False) + "\n")
            print(
                f"REID {step}/{cfg['reid_steps']} ce={ce_value:.6f} "
                f"triplet={metric_value:.6f} steps_per_s={count / elapsed:.3f} "
                f"data_wait_s={data_wait:.3f}",
                flush=True,
            )
            previous_logged_step = step
            window_start, data_wait = perf_counter(), 0.0
        if step % 1000 == 0 or step == cfg["reid_steps"]:
            report = evaluate(model, tune, device, cfg)
            values = [r["cosine"].get("auroc") for r in report["p1"]["reports"]]
            with (output / "validation.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "p1_auroc": values}) + "\n")
            print(f"REID_TUNE step={step} p1_auroc={values}", flush=True)
            if None not in values and min(values) > best:
                best, selected, best_step = (
                    min(values),
                    deepcopy(model.state_dict()),
                    step,
                )
            save_model(output / f"reid-step-{step}.pt", model, step)
            # Validation and checkpoint writes are not training throughput.
            window_start, data_wait = perf_counter(), 0.0
    # Exhaust the finite loader so worker processes exit before final evaluation.
    if next(batches, None) is not None:
        raise RuntimeError("Unexpected extra ReID batch")
    if selected is None:
        raise RuntimeError("No valid P1 endpoint; do not promote as patient teacher")
    model.load_state_dict(selected)
    save_model(output / "reid.pt", model, best_step, {"selection_role": "val_tune"})
    report = evaluate(model, select, device, cfg, True)
    write(output / "reid_capability.json", report)
    completed(
        output,
        p1_passed=report["p1"]["passed"],
        p2_passed=report["p2"]["passed"],
        scope="ReID_teacher_only_not_system_privacy",
        pair_sampler_fallbacks=sampler.fallbacks,
    )
