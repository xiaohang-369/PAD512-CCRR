"""New finite V2 loops. Every learned head and S0 starts from a fresh seed."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

from ccrr.data.pad import PairSampler
from ccrr.evaluation.metrics import clinical
from .models import ClinicalProbe, PairAttacker, SemanticAdapter, confusion_loss
from .runtime import completed, digest, read, save_model, seed_all, write


def class_weights(data):
    counts = torch.bincount(data["labels"], minlength=6).float()
    if (counts == 0).any():
        raise ValueError("Fit cohort missing classes")
    w = counts.rsqrt()
    return w / w.mean()


def classification(model, data, bootstrap=False, repetitions=2000):
    from ccrr.evaluation.teachers import clinical_intervals, paired_gray_change

    result = []
    model.eval()
    for domain in (0, 1):
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(data["rows"]), 64):
                logits = model(
                    data["features"][start : start + 64, domain],
                    data["coverage"][start : start + 64],
                )
                probabilities.extend(logits.softmax(1).cpu().tolist())
        labels = data["labels"].cpu().tolist()
        report = clinical(labels, probabilities)
        report.update(
            labels=labels,
            probabilities=probabilities,
            ids=[r["img_id"] for r in data["rows"]],
            patients=[r["patient_id"] for r in data["rows"]],
        )
        if bootstrap:
            report = clinical_intervals(report, repetitions)
        result.append(report)
    if bootstrap:
        return {
            "domains": result,
            "gray_change": paired_gray_change(result, repetitions),
        }
    return result


def fit_probe(cache, cfg, output, *, mlp=False):
    seed_all(cfg["seed"])
    fit = cache["fit"]
    model = ClinicalProbe(fit["features"].shape[2], mlp).to(fit["features"].device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["probe_lr"])
    rng = np.random.default_rng(cfg["seed"])
    weights = class_weights(fit)
    best, selected, best_step = -float("inf"), None, None
    for step in range(1, cfg["probe_steps"] + 1):
        model.train()
        indices = rng.integers(len(fit["rows"]), size=cfg["batch_size"])
        optimizer.zero_grad(set_to_none=True)
        loss = (
            sum(
                nn.functional.cross_entropy(
                    model(fit["features"][indices, d], fit["coverage"][indices]),
                    fit["labels"][indices],
                    weight=weights,
                )
                for d in (0, 1)
            )
            / 2
        )
        loss.backward()
        optimizer.step()
        if step % cfg["evaluate_every"] == 0 or step == cfg["probe_steps"]:
            reports = classification(model, cache["val_tune"])
            values = [r["macro_auroc"] for r in reports]
            if None not in values and min(values) > best:
                best, selected, best_step = (
                    min(values),
                    deepcopy(model.state_dict()),
                    step,
                )
            print(
                f"PROBE {step}/{cfg['probe_steps']} loss={float(loss):.6f} tune={values}",
                flush=True,
            )
    if selected is None:
        raise RuntimeError("No estimable development endpoint")
    model.load_state_dict(selected)
    save_model(
        output / "probe.pt",
        model,
        best_step,
        {"selection_role": "val_tune", "mlp": mlp},
    )
    report = classification(
        model, cache["val_select"], True, cfg["bootstrap_repetitions"]
    )
    write(output / "clinical.json", report)
    return model, report


def pair_batch(data, sampler, rng, domain_pair=None):
    pairs = sampler.sample(rng)
    a, b, labels = zip(*[(p[0], p[1], p[2]) for p in pairs])
    da, db = (
        domain_pair
        if domain_pair is not None
        else ((0, 0), (1, 0), (1, 1))[int(rng.integers(3))]
    )
    a, b = list(a), list(b)
    return (
        data["features"][a, da],
        data["coverage"][a],
        data["features"][b, db],
        data["coverage"][b],
        torch.tensor(labels, dtype=torch.float32, device=data["features"].device),
    )


def train_s0(cache, cfg, output, variant):
    seed_all(cfg["seed"])
    fit = cache["fit"]
    device = fit["features"].device
    adapter = SemanticAdapter().to(device)
    head = ClinicalProbe(128).to(device)
    adversary = PairAttacker(128).to(device)
    optimizer = torch.optim.Adam(
        [*adapter.parameters(), *head.parameters()], lr=cfg["s0_lr"]
    )
    attack_opt = torch.optim.Adam(adversary.parameters(), lr=cfg["attacker_lr"])
    sampler = PairSampler(SimpleNamespace(rows=fit["rows"]))
    medical_rng = np.random.default_rng(cfg["seed"])
    attack_rng = np.random.default_rng(cfg["seed"] + 1)
    weights = class_weights(fit)
    total = cfg["s0_warmup_steps"] + cfg["s0_joint_steps"]

    def attack_update():
        a, av, b, bv, labels = pair_batch(fit, sampler, attack_rng)
        for p in adversary.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            a, b = adapter(a), adapter(b)
        attack_opt.zero_grad(set_to_none=True)
        value = nn.functional.binary_cross_entropy_with_logits(
            adversary(a, av, b, bv), labels
        )
        value.backward()
        attack_opt.step()
        return float(value.detach())

    for step in range(1, total + 1):
        if variant == "joint" and step == cfg["s0_warmup_steps"] + 1:
            for _ in range(cfg["attacker_warmup_steps"]):
                attack_update()
            print("S0_ATTACKER_WARMUP_COMPLETED", flush=True)
        alpha = 0.0
        if variant == "joint" and step > cfg["s0_warmup_steps"]:
            for _ in range(2):
                attack_update()
            alpha = cfg["identity_weight"] * min(
                1.0, (step - cfg["s0_warmup_steps"]) / cfg["identity_ramp_steps"]
            )
        indices = medical_rng.integers(len(fit["rows"]), size=cfg["batch_size"])
        optimizer.zero_grad(set_to_none=True)
        medical = (
            sum(
                nn.functional.cross_entropy(
                    head(
                        adapter(fit["features"][indices, d]), fit["coverage"][indices]
                    ),
                    fit["labels"][indices],
                    weight=weights,
                )
                for d in (0, 1)
            )
            / 2
        )
        privacy = torch.zeros((), device=device)
        if alpha:
            for p in adversary.parameters():
                p.requires_grad_(False)
            a, av, b, bv, _ = pair_batch(fit, sampler, attack_rng)
            privacy = confusion_loss(adversary(adapter(a), av, adapter(b), bv))
        loss = medical + alpha * privacy
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite S0 objective")
        loss.backward()
        optimizer.step()
        if step % cfg["evaluate_every"] == 0 or step == total:
            with (output / "training.jsonl").open("a") as f:
                import json

                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "medical": float(medical.detach()),
                            "privacy": float(privacy.detach()),
                            "weight": alpha,
                        }
                    )
                    + "\n"
                )
            print(
                f"S0 {variant} {step}/{total} medical={float(medical):.6f}", flush=True
            )
    save_model(
        output / "s0.pt",
        adapter,
        total,
        {"variant": variant, "endpoint": "fixed_budget"},
    )
    write(
        output / "s0_artifact.json",
        {
            "sha256": digest(output / "s0.pt"),
            "binding_sha256": digest(output / "binding.json"),
        },
    )
    completed(
        output,
        variant=variant,
        clinical_identity_acceptance="requires_independent_audit",
        pair_sampler_fallbacks=sampler.fallbacks,
    )


def apply_s0(cache, path, resources, cache_digest):
    path = Path(path).resolve()
    runs = Path(resources["workspace_root"]).resolve() / "runs"
    if runs not in path.parents:
        raise ValueError("S0 must come from V2 run root")
    binding = read(path / "binding.json")
    artifact = read(path / "s0_artifact.json")
    if (
        artifact["sha256"] != digest(path / "s0.pt")
        or artifact["binding_sha256"] != digest(path / "binding.json")
        or read(path / "status.json")["state"] != "completed"
    ):
        raise ValueError("S0 artifact incomplete or changed")
    if (
        binding["stage"] != "s0"
        or binding["resources"] != resources
        or binding["parents"].get("features_sha256") != cache_digest
    ):
        raise ValueError("S0/cached-feature resource binding mismatch")
    device = cache["fit"]["features"].device
    model = SemanticAdapter().to(device)
    model.load_state_dict(
        torch.load(path / "s0.pt", weights_only=True, map_location=device)["model"]
    )
    model.eval()
    for data in cache.values():
        f = data["features"]
        output = torch.empty(len(f), 2, 128, 14, 14, device=device)
        with torch.no_grad():
            for start in range(0, len(f), 64):
                for domain in (0, 1):
                    output[start : start + 64, domain] = model(
                        f[start : start + 64, domain]
                    )
        data["features"] = output
    return digest(path / "s0.pt")
