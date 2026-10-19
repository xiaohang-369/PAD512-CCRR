"""Fresh patient ReID from public ResNet50, never a previous learned teacher."""

from copy import deepcopy
import numpy as np
import torch
from torch import nn

from ccrr.data.pad import PADData, PairSampler, collate
from ccrr.teachers.adapters import IdentityModel, freeze_bn
from ccrr.tensor_ops import gray
from ccrr.evaluation.teachers import teacher_identity_scores, reid_gate
from .runtime import completed, read, save_model, seed_all, write


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


def train(resources, cfg, output, device):
    seed_all(cfg["seed"])
    fit = PADData(resources["data_root"], "fit")
    tune = PADData(resources["data_root"], "val_tune")
    select = PADData(resources["data_root"], "val_select")
    model = initialize(resources, len(fit.patients)).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["reid_lr"]
    )
    sampler, rng = PairSampler(fit), np.random.default_rng(cfg["seed"])
    best, selected, best_step = -float("inf"), None, None
    for step in range(1, cfg["reid_steps"] + 1):
        model.train()
        freeze_bn(model)
        positives = sampler.sample(rng)[:8]
        indices = [i for a, b, _, _ in positives for i in (a, b)]
        batch = collate([fit[i] for i in indices], device)
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
        if step % 100 == 0:
            print(
                f"REID {step}/{cfg['reid_steps']} ce={float(ce):.6f} triplet={float(metric):.6f}",
                flush=True,
            )
        if step % 1000 == 0 or step == cfg["reid_steps"]:
            report = evaluate(model, tune, device, cfg)
            values = [r["cosine"].get("auroc") for r in report["p1"]["reports"]]
            if None not in values and min(values) > best:
                best, selected, best_step = (
                    min(values),
                    deepcopy(model.state_dict()),
                    step,
                )
            save_model(output / f"reid-step-{step}.pt", model, step)
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
