"""Stage A: teacher adaptation, training and selection."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as tv
from ccrr.data.pad import collate, rng_for
from ccrr.codec.metadata import CLASSES
from ccrr.teachers.adapters import (
    ClinicalModel,
    PanDerm,
    freeze,
    freeze_bn,
    load_weights,
)
from ccrr.tensor_ops import gray, pooled
from ccrr.training.losses import (
    contrastive,
    weighted_cross_entropy,
)
from ccrr.runtime.support import write_json
from ccrr.training.optimization import optimizer, TeacherSchedule, update
from ccrr.training.a_state import AProgress
from ccrr.evaluation.teachers import (
    clinical_intervals,
    clinical_gate,
    PatchProbe,
    paired_gray_change,
)
from ccrr.teachers.capability import mfm_interface
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.evaluation.runners import (
    evaluate_classifier,
    evaluate_identity,
    dual_report,
    reid_capability,
)
from ccrr.errors import require


def initialize_base(experiment, wrapper, name):
    entry = experiment.cfg["weights"][name]
    require(
        entry.get("path") and entry.get("sha256"),
        f"Missing official {name} weights and SHA-256",
    )
    # Strictly load a base architecture before replacing its task-specific head.
    base = getattr(tv, name)(weights=None)
    load_weights(base, entry["path"], entry["sha256"])
    if name == "efficientnet_b0":
        base.classifier[-1] = wrapper.model.classifier[-1]
    elif name == "convnext_tiny":
        base.classifier[-1] = wrapper.model.classifier[-1]
    elif name == "resnet50":
        base.fc = nn.Identity()
    elif name == "swin_t":
        base.head = wrapper.model.head
    wrapper.model = base.to(experiment.device)


def train_classifier(experiment, model, name, budget, views, independent=False):
    initialize_base(experiment, model, model.architecture)
    # Split base and final classification layer without relying on a DEFAULT enum.
    head = (
        model.model.head if model.architecture == "swin_t" else model.model.classifier
    )
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = optimizer([model], [1e-4])
    groups = []
    for group in opt.param_groups:
        for ish, lr in ((False, 1e-4), (True, 1e-3)):
            ps = [p for p in group["params"] if (id(p) in head_ids) == ish]
            if ps:
                groups.append(
                    {
                        "params": ps,
                        "lr": lr,
                        "lr_max": lr,
                        "weight_decay": group["weight_decay"],
                        "head": ish,
                    }
                )
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    scheduler = TeacherSchedule(opt, budget, classifier=True)
    progress = (
        AProgress(experiment, name, {"model": model}, opt, scheduler)
        if not independent
        else None
    )
    best = (
        -float("inf")
        if progress is None or progress.selection["score"] is None
        else progress.selection["score"]
    )
    best_path = experiment.output / "checkpoints" / f"{name}.pt"
    last_report = None if progress is None else progress.selection["reports"]
    for step in range(0 if progress is None else progress.start, budget):
        rng = rng_for(experiment.seed, name, "images", step)
        model.train()
        freeze_bn(model)
        opt.zero_grad(set_to_none=True)
        for p in backbone:
            p.requires_grad_(step >= 500)
        selected = rng.integers(
            len(experiment.fit), size=experiment.cfg.get("batch_teacher", 16)
        )
        # Class weighted CE normalization spans the effective batch and domains.
        denom = sum(
            float(
                experiment.weights[
                    CLASSES.index(experiment.fit.rows[int(i)]["diagnostic"])
                ]
            )
            for i in selected
        ) * len(views)
        count = 0
        total = 0.0
        for i in selected:
            b = collate([experiment.fit[int(i)]], experiment.device)
            flip = rng.random() < 0.5
            for view in views:
                x = view(b)
                if x is None:
                    continue
                rects = b["rect"]
                if flip:
                    x = x.flip(-1)
                    rects = [
                        (x.shape[-1] - left - width, top, width, height)
                        for left, top, width, height in rects
                    ]
                loss = weighted_cross_entropy(
                    model(x, rects),
                    b["label"],
                    experiment.weights,
                    weight_sum=denom,
                )
                loss.backward()
                total += float(loss.detach())
                count += 1
        if not count:
            continue
        if progress is None:
            scheduler.next_step = step
        scheduler.apply(step)
        norm = update(opt)
        scheduler.advance()
        if (step + 1) % 100 == 0:
            experiment.log(name, step + 1, {"loss": total, "gradient_norm": norm})
        if (step + 1) % 500 == 0 or step + 1 == budget:
            reports = [
                evaluate_classifier(
                    model,
                    experiment.data["val_select"],
                    experiment.device,
                    v,
                    bootstrap=False,
                )
                for v in views
            ]
            score = min(
                r["macro_auroc"] if r["macro_auroc"] is not None else -float("inf")
                for r in reports
            )
            if progress is not None:
                progress.select(None if score == -float("inf") else score, reports)
            if score > best:
                best, last_report = score, reports
                if progress is None:
                    save_checkpoint(
                        best_path,
                        {"model": model},
                        opt,
                        step + 1,
                        name,
                        {"score": score, "independent": independent},
                    )
            write_json(
                experiment.output / "validation" / f"{name}-{step + 1}.json", reports
            )
        if progress is not None:
            endpoint = (step + 1) % 500 == 0 or step + 1 == budget
            if endpoint or (step + 1) % experiment.cfg.get("resume_interval", 100) == 0:
                progress.save(endpoint)
    if progress is not None:
        best, last_report = progress.finish(best_path)
        last_report = [clinical_intervals(r) for r in last_report]
        write_json(
            experiment.output / "validation" / f"{name}-selected.json", last_report
        )
    require(best_path.exists(), f"{name}: no evaluable checkpoint")
    load_checkpoint(best_path, {"model": model})
    freeze(model)
    return best, last_report


def stage_a(experiment):
    experiment.preflight("A")
    score, report = train_classifier(
        experiment,
        experiment.clinical,
        "clinical",
        experiment.cfg["budgets"]["clinical"],
        [lambda b: b["x"], lambda b: gray(b["x"])],
    )
    controls = []
    for mode, view in [("rgb", lambda b: b["x"]), ("gray", lambda b: gray(b["x"]))]:
        m = ClinicalModel().to(experiment.device)
        s, r = train_classifier(
            experiment,
            m,
            "clinical_" + mode,
            experiment.cfg["budgets"]["clinical"],
            [view],
        )
        controls.append(r[0])
        del m
    gate = clinical_gate(report, controls)
    experiment.mark(
        "A-clinical",
        gate["passed"],
        {
            "joint": report,
            "controls": controls,
            "gate": gate,
            "gray_change": paired_gray_change(report),
        },
    )
    e = experiment.cfg["weights"]["panderm"]
    experiment.mfm = PanDerm(
        e["source"], e["source_sha256"], e["path"], e["sha256"]
    ).to(experiment.device)
    train_mfm_probe(experiment)
    initialize_base(experiment, experiment.reid, "resnet50")
    train_reid_teacher(experiment)
    initialize_base(experiment, experiment.pub_attacker, "resnet50")
    from ccrr.teachers.freeze_record import write_freeze_record

    write_freeze_record(experiment)


def train_mfm_probe(experiment):
    interface = mfm_interface(experiment.mfm, experiment.fit, experiment.device)
    head = nn.Linear(768, 6).to(experiment.device)
    opt = optimizer([head], [1e-3])
    total = experiment.cfg["budgets"]["mfm_probe"]
    scheduler = TeacherSchedule(opt, total)
    progress = AProgress(
        experiment,
        "mfm_probe",
        {"probe": head, "purifier": experiment.purifier},
        opt,
        scheduler,
    )
    if progress.start == 0:
        sums = torch.zeros(768, device=experiment.device)
        squares = sums.clone()
        count = 0
        with torch.no_grad():
            for i in range(len(experiment.fit)):
                b = collate([experiment.fit[i]], experiment.device)
                for x in (b["x"], gray(b["x"])):
                    f, v, _ = experiment.mfm(x, b["rect"])
                    sums += pooled(f, v)[0]
                    squares += pooled(f.square(), v)[0]
                    count += 1
        mean = sums / count
        std = (squares / count - mean.square()).clamp_min(0).sqrt()
        experiment.purifier.mean.copy_(mean[None, :, None, None])
        experiment.purifier.std.copy_(std[None, :, None, None])
        experiment.purifier.stats_ready.fill_(True)
    for step in range(progress.start, total):
        head.train()
        opt.zero_grad(set_to_none=True)
        rng = rng_for(experiment.seed, "mfm_probe", "images", step)
        ids = rng.integers(len(experiment.fit), size=16)
        denom = (
            sum(
                float(
                    experiment.weights[
                        CLASSES.index(experiment.fit.rows[int(i)]["diagnostic"])
                    ]
                )
                for i in ids
            )
            * 2
        )
        for i in ids:
            b = collate([experiment.fit[int(i)]], experiment.device)
            for x in (b["x"], gray(b["x"])):
                with torch.no_grad():
                    f, v, _ = experiment.mfm(x, b["rect"])
                    h = pooled(f, v)
                weighted_cross_entropy(
                    head(h), b["label"], experiment.weights, weight_sum=denom
                ).backward()
        scheduler.apply(step)
        update(opt)
        scheduler.advance()
        endpoint = (step + 1) % 250 == 0 or step + 1 == total
        if endpoint:
            reports = dual_report(
                PatchProbe(experiment.mfm, head),
                experiment.data["val_select"],
                experiment.device,
                False,
            )
            scores = [r["macro_auroc"] for r in reports]
            progress.select(
                min(scores) if all(s is not None for s in scores) else None, reports
            )
            write_json(
                experiment.output / "validation" / f"mfm_probe-{step + 1}.json", reports
            )
        if endpoint or (step + 1) % experiment.cfg.get("resume_interval", 100) == 0:
            progress.save(endpoint)
    _, reports = progress.finish(
        experiment.output / "checkpoints/mfm_probe.pt", "A-mfm"
    )
    reports = [clinical_intervals(r) for r in reports]
    gate = clinical_gate(reports, enforce_class_defects=False)
    experiment.mark(
        "A-mfm",
        gate["passed"] and interface["passed"],
        {
            "domains": reports,
            "gate": gate,
            "interface": interface,
            "gray_change": paired_gray_change(reports),
            "selected_step": progress.selection["step"],
        },
    )


def train_reid_teacher(experiment):
    model = experiment.reid
    opt = optimizer([model.model, model.classifier], [1e-4, 1e-3])
    total = experiment.cfg["budgets"]["reid"]
    path = experiment.output / "checkpoints/reid.pt"
    scheduler = TeacherSchedule(opt, total)
    progress = AProgress(experiment, "reid", {"model": model}, opt, scheduler)
    experiment.sampler.fallbacks = progress.audit["fallbacks"]
    for step in range(progress.start, total):
        rng = rng_for(experiment.seed, "reid", "pairs", step)
        pairs = experiment.sampler.sample(rng)
        model.train()
        freeze_bn(model)
        opt.zero_grad(set_to_none=True)
        sampled = []
        for a, b, s, scenario in pairs:
            ba = collate([experiment.fit[a]], experiment.device)
            bb = collate([experiment.fit[b]], experiment.device)
            domain = int(rng.integers(3))
            sampled.append(
                {
                    "a": ba["id"][0],
                    "b": bb["id"][0],
                    "label": s,
                    "scenario": scenario,
                    "domain": domain,
                }
            )
            xa = gray(ba["x"]) if domain in (1, 2) else ba["x"]
            xb = gray(bb["x"]) if domain == 2 else bb["x"]
            if rng.random() < 0.5:
                xa = xa.flip(-1)
                ba["rect"] = [
                    (xa.shape[-1] - l - w, t, w, h) for l, t, w, h in ba["rect"]
                ]
            if rng.random() < 0.5:
                xb = xb.flip(-1)
                bb["rect"] = [
                    (xb.shape[-1] - l - w, t, w, h) for l, t, w, h in bb["rect"]
                ]
            fa = model(xa, ba["rect"])
            fb = model(xb, bb["rect"])
            label = xa.new_tensor([s])
            ce = 0.5 * (
                F.cross_entropy(fa["logits"], ba["patient_index"])
                + F.cross_entropy(fb["logits"], bb["patient_index"])
            )
            loss = (
                ce + 0.5 * contrastive(fa["embedding"], fb["embedding"], label)
            ) / 16
            loss.backward()
        scheduler.apply(step)
        update(opt)
        scheduler.advance()
        progress.audit["fallbacks"] = experiment.sampler.fallbacks
        write_json(
            experiment.output / "sampling" / f"reid-{step + 1}.json",
            {
                "pairs": sampled,
                "fallbacks": experiment.sampler.fallbacks,
                "unique_unordered_pairs": len(
                    {tuple(sorted((r["a"], r["b"]))) for r in sampled}
                ),
                "pairs_n": len(sampled),
            },
        )
        if (step + 1) % 1000 == 0 or step + 1 == total:
            reports = [
                evaluate_identity(
                    model,
                    experiment.data["val_select"],
                    experiment.device,
                    "p1",
                    qa,
                    ga,
                    bootstrap=False,
                    teacher=True,
                )
                for qa, ga in [
                    (None, None),
                    (lambda b: gray(b["x"]), None),
                    (lambda b: gray(b["x"]), lambda b: gray(b["x"])),
                ]
            ]
            values = [r["cosine"].get("auroc") for r in reports]
            progress.select(
                min(values) if all(v is not None for v in values) else None, reports
            )
            write_json(
                experiment.output / "validation" / f"reid-{step + 1}.json", reports
            )
        endpoint = (step + 1) % 1000 == 0 or step + 1 == total
        if endpoint or (step + 1) % experiment.cfg.get("resume_interval", 100) == 0:
            progress.save(endpoint)
    progress.finish(path, "A-reid")
    freeze(model)
    report = reid_capability(experiment, model)
    report["selected_step"] = progress.selection["step"]
    write_json(experiment.output / "validation/reid-capability.json", report)
    # P2 has its own status. Only P1 establishes cross-lesion patient supervision.
    experiment.mark("A-reid", report["p1"]["passed"], report)
