"""Optional fresh P2 reference adaptation; never runs legacy P1 selection."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from ccrr.data.pad import PADData, collate
from ccrr.teachers.adapters import freeze_bn
from ccrr.tensor_ops import gray
from ccrr_v2.reid import initialize
from ccrr_v2.loading import ReIDBatchIndices, cpu_batches, to_device
from .pairs import P2Sampler
from .evaluation import DIRECTIONS, p2_report
from .runtime import read, save, write, log, seed_all, ROLES


def extract(model, data, device, batch_size=16):
    model.eval()
    domains = [[], []]
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            b = collate(
                [data[i] for i in range(start, min(start + batch_size, len(data)))],
                device,
            )
            for d, image in enumerate((b["x"], gray(b["x"]))):
                emb = model(image, b["rect"])["embedding"]
                if not torch.isfinite(emb).all() or (emb.norm(dim=1) <= 1e-8).any():
                    raise ValueError("Invalid linkage embeddings")
                domains[d].append(F.normalize(emb, dim=1).cpu())
    return torch.stack([torch.cat(x) for x in domains], 1)


def reports(data, embeddings, repetitions=0):
    protocol = read(
        data.root / "protocol_audit_20261009/pair_protocol" / f"{data.role}_p2.json"
    )
    result = {}
    for name, (da, db) in DIRECTIONS.items():

        def score(a, bs):
            return (embeddings[bs, db] @ embeddings[a, da]).tolist()

        result[name] = p2_report(data.rows, protocol, score, repetitions=repetitions)
    return result


def p2_triplet(embeddings, patients, lesions, margin=0.3):
    dist = torch.cdist(embeddings, embeddings)
    same_patient = patients[:, None] == patients[None, :]
    positive = same_patient & (lesions[:, None] == lesions[None, :])
    positive &= ~torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    negative = ~same_patient
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return embeddings.sum() * 0
    return (
        (
            dist.masked_fill(~positive, -torch.inf).max(1).values[valid]
            - dist.masked_fill(~negative, torch.inf).min(1).values[valid]
            + margin
        )
        .clamp_min(0)
        .mean()
    )


def train(resources, cfg, output, device, workers=0):
    seed_all(cfg["seed"])
    fit, tune = (PADData(resources["data_root"], role) for role in ("fit", "val_tune"))
    model = initialize(resources, len(fit.patients)).to(device)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["reid_lr"]
    )
    sampler = P2Sampler(fit.rows)
    batches = iter(
        cpu_batches(
            fit,
            ReIDBatchIndices(sampler, cfg["seed"], cfg["reid_steps"]),
            workers=workers,
            prefetch=2,
            seed=cfg["seed"],
            pin_memory=workers > 0 and device.type == "cuda",
        )
    )
    image_lesions = {r["img_id"]: (r["patient_id"], r["lesion_id"]) for r in fit.rows}
    lesion_names = {key: i for i, key in enumerate(sorted(set(image_lesions.values())))}
    best, selected = -float("inf"), None
    write(
        output / "initial_reference.json", reports(tune, extract(model, tune, device))
    )
    for step in range(1, cfg["reid_steps"] + 1):
        b = to_device(next(batches), device, non_blocking=workers > 0)
        model.train()
        freeze_bn(model)
        opt.zero_grad(set_to_none=True)
        result = model(b["x"] if step % 2 else gray(b["x"]), b["rect"])
        lesions = torch.tensor(
            [lesion_names[image_lesions[name]] for name in b["id"]], device=device
        )
        ce = F.cross_entropy(result["logits"], b["patient_index"])
        metric = p2_triplet(result["embedding"], b["patient_index"], lesions)
        loss = ce + metric
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite P2 reference objective")
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
        )
        opt.step()
        if step % cfg["evaluate_every"] == 0 or step == cfg["reid_steps"]:
            report = reports(tune, extract(model, tune, device))
            value = min(r["auroc"] for r in report.values())
            if value > best:
                best, selected, best_step = value, deepcopy(model.state_dict()), step
            log(
                output,
                "p2_reference",
                step=step,
                ce=float(ce.detach()),
                triplet=float(metric.detach()),
                p2_min_auc=value,
                selection="val_tune_min_three_direction_P2_AUROC",
            )
    if next(batches, None) is not None or selected is None:
        raise RuntimeError("Loader/selection failure")
    model.load_state_dict(selected)
    save(
        output / "reference.pt",
        dict(
            model=selected,
            patients=len(fit.patients),
            selected_step=best_step,
            selection="val_tune_P2_only",
            resume_supported=False,
        ),
    )
    select = PADData(resources["data_root"], "val_select")
    write(
        output / "reference_report.json",
        reports(select, extract(model, select, device), cfg["bootstrap_repetitions"]),
    )
    write(
        output / "scope.json",
        dict(
            p2_only=True,
            p1_required=False,
            classification_positive_patients=len(sampler.eligible),
            fit_patients=len(fit.patients),
            note="CE uses P2-positive-eligible patients only; unobserved classes still receive negative softmax gradients",
            scientific_acceptance="not_evaluated",
        ),
    )


def targets(resources, cfg, output, device, reference=None):
    fit = PADData(resources["data_root"], "fit")
    seed_all(cfg["seed"])
    model = initialize(resources, len(fit.patients)).to(device)
    if reference is not None:
        obj = torch.load(
            reference / "reference.pt", map_location="cpu", weights_only=True
        )
        if obj["patients"] != len(fit.patients):
            raise ValueError("Reference cohort mismatch")
        model.load_state_dict(obj["model"], strict=True)
    model.eval().requires_grad_(False)
    payload, report = {}, {}
    for role in ROLES:
        data = fit if role == "fit" else PADData(resources["data_root"], role)
        emb = extract(model, data, device, cfg["batch_size"])
        payload[role] = dict(embeddings=emb, ids=[r["img_id"] for r in data.rows])
        if role != "fit":
            report[role] = reports(
                data, emb, cfg["bootstrap_repetitions"] if role == "val_select" else 0
            )
        log(output, "frozen_link_targets", role=role, images=len(data))
    save(output / "targets.pt", payload)
    write(
        output / "capability.json",
        dict(
            reports=report,
            source="fresh_V3_reference"
            if reference
            else "public_ResNet50_initialization",
            teacher_qualified=False,
            qualification="pending_protocol_review",
            p3_validated=False,
            model_parameters_updated=False,
        ),
    )
