"""Stage A: teacher adaptation, training and selection."""

import copy
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
from ccrr.training.optimization import optimizer, schedule, update
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.evaluation.runners import (
    evaluate_classifier,
    evaluate_identity,
)
from ccrr.evaluation.metrics import clinical
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
    best = -float("inf")
    best_path = experiment.output / "checkpoints" / f"{name}.pt"
    last_report = None
    for step in range(budget):
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
        schedule(opt, step, budget)
        for group in opt.param_groups:
            if not group["head"]:
                if step < 500:
                    group["lr"] = 0
                elif step < 750:
                    group["lr"] = group["lr_max"] * (0.01 + 0.99 * (step - 500) / 249)
        norm = update(opt)
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
            if score > best:
                best = score
                last_report = reports
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
    require(best_path.exists(), f"{name}: no evaluable checkpoint")
    load_checkpoint(best_path, {"model": model})
    freeze(model)
    return best, last_report


def stage_a(experiment):
    experiment.preflight()
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
        controls.append(s)
    passed = score >= 0.75 and all(
        report[j]["macro_auroc"] >= controls[j] - 0.03
        and all(a is not None and a > 0.5 for a in report[j]["per_class_auroc"])
        and all(v is not None and v > 0 for v in report[j]["recall"])
        for j in range(2)
    )
    experiment.mark("A-clinical", passed, {"joint": report, "controls": controls})
    e = experiment.cfg["weights"]["panderm"]
    experiment.mfm = PanDerm(
        e["source"], e["source_sha256"], e["path"], e["sha256"]
    ).to(experiment.device)
    train_mfm_probe(experiment)
    initialize_base(experiment, experiment.reid, "resnet50")
    train_reid_teacher(experiment)
    initialize_base(experiment, experiment.pub_attacker, "resnet50")


def train_mfm_probe(experiment):
    head = nn.Linear(768, 6).to(experiment.device)
    opt = optimizer([head], [1e-3])
    total = experiment.cfg["budgets"]["mfm_probe"]
    best = -1
    best_state = None
    # Frozen standardization statistics: equal images and RGB/gray domains.
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
    for step in range(total):
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
                (
                    weighted_cross_entropy(
                        head(h), b["label"], experiment.weights, weight_sum=denom
                    )
                ).backward()
        schedule(opt, step, total)
        update(opt)
        if (step + 1) % 250 == 0 or step + 1 == total:
            reports = []
            for gray_mode in (False, True):
                ys = []
                ps = []
                with torch.no_grad():
                    for i in range(len(experiment.data["val_select"])):
                        b = collate(
                            [experiment.data["val_select"][i]], experiment.device
                        )
                        f, v, _ = experiment.mfm(
                            gray(b["x"]) if gray_mode else b["x"], b["rect"]
                        )
                        ys.append(int(b["label"][0]))
                        ps.append(head(pooled(f, v)).softmax(1)[0].cpu().numpy())
                reports.append(clinical(ys, ps))
            score = min(r["macro_auroc"] or 0 for r in reports)
            if score > best:
                best = score
                best_state = copy.deepcopy(head.state_dict())
                best_reports = reports
    head.load_state_dict(best_state)
    save_checkpoint(
        experiment.output / "checkpoints/mfm_probe.pt",
        {"probe": head, "purifier": experiment.purifier},
        None,
        total,
        "A-mfm",
        {"score": best},
    )
    experiment.mark("A-mfm", best >= 0.75, best_reports)


def train_reid_teacher(experiment):
    model = experiment.reid
    opt = optimizer([model.model, model.classifier], [1e-4, 1e-3])
    total = experiment.cfg["budgets"]["reid"]
    best = -1
    path = experiment.output / "checkpoints/reid.pt"
    best_report = None
    for step in range(total):
        rng = rng_for(experiment.seed, "reid", "pairs", step)
        pairs = experiment.sampler.sample(rng)
        model.train()
        freeze_bn(model)
        opt.zero_grad(set_to_none=True)
        for a, b, s, scenario in pairs:
            ba = collate([experiment.fit[a]], experiment.device)
            bb = collate([experiment.fit[b]], experiment.device)
            domain = int(rng.integers(3))
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
        schedule(opt, step, total)
        update(opt)
        if (step + 1) % 1000 == 0 or step + 1 == total:
            reports = [
                evaluate_identity(
                    model,
                    experiment.data["val_select"],
                    experiment.device,
                    "p1",
                    qa,
                    ga,
                    bootstrap=True,
                )
                for qa, ga in [
                    (None, None),
                    (lambda b: gray(b["x"]), None),
                    (lambda b: gray(b["x"]), lambda b: gray(b["x"])),
                ]
            ]
            score = min(r["cosine"].get("auroc", 0) for r in reports)
            if score > best:
                best = score
                best_report = reports
                save_checkpoint(
                    path,
                    {"model": model},
                    opt,
                    step + 1,
                    "A-reid",
                    {"score": score},
                )
    load_checkpoint(path, {"model": model})
    freeze(model)
    passed = best >= 0.70 and all(
        r["cosine"]["auroc_ci"]["interval"]
        and r["cosine"]["auroc_ci"]["interval"][0] > 0.5
        and r["cosine"]["auroc_ci"]["invalid_fraction"] <= 0.05
        for r in best_report
    )
    experiment.mark("A-reid", passed, best_report)
