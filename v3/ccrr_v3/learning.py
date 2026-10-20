"""Independent compression/separation/adversarial runs with fixed finite endpoints."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ccrr_v2.models import PairAttacker, confusion_loss
from .models import Separator, ClinicalHeads, freeze, weighted_mean
from .data import batch, pair_batch
from .pairs import P2Sampler
from .evaluation import weights_for, clinical_report
from .runtime import seed_all, log, save, write


def core_objective(
    model,
    result,
    source,
    coverage,
    labels,
    heads,
    domain,
    target_embedding,
    weights,
    clinical_weights,
):
    all_loss = weighted_mean((result["full"] - source.detach()).square(), coverage)
    rate, diagnostics = model.rate(result, coverage)
    medical = F.cross_entropy(
        heads(result["semantic"], coverage, domain), labels, weight=clinical_weights
    )
    link = all_loss.new_zeros(())
    if "embedding" in result:
        if (
            target_embedding is None
            or target_embedding.shape != result["embedding"].shape
        ):
            raise ValueError("Matching frozen linkage targets required")
        if (
            not torch.isfinite(target_embedding).all()
            or (target_embedding.norm(dim=1) < 1e-8).any()
        ):
            raise ValueError("Invalid or zero linkage target")
        link = (
            (
                F.normalize(result["embedding"], dim=1)
                - F.normalize(target_embedding.detach(), dim=1)
            )
            .square()
            .sum(1)
            .mean()
        )
    total = (
        weights["all"] * all_loss
        + weights["rate"] * rate
        + weights["medical"] * medical
        + weights["link"] * link
    )
    return total, dict(
        all=all_loss, rate=rate, medical=medical, link=link, **diagnostics
    )


def represent(cache, model, view, device):
    if view == "raw":
        return cache
    legal = {"full", "semantic", "q_sem", "q_link"}
    if view not in legal or (view == "q_link" and model.variant == "compression"):
        raise ValueError("Unavailable representation")
    model.eval()
    result = {}
    with torch.no_grad():
        for role, data in cache.items():
            domains = []
            for d in (0, 1):
                parts = []
                for start in range(0, len(data["rows"]), 32):
                    x, v, _ = batch(data, slice(start, start + 32), d, device)
                    parts.append(model(x, v, stochastic=False)[view].cpu())
                domains.append(torch.cat(parts))
            result[role] = {**data, "features": torch.stack(domains, 1)}
    return result


def train(cache, targets, teacher_heads, statistics, cfg, output, device, variant):
    seed_all(cfg["seed"])
    fit = cache["fit"]
    dim = targets["fit"].shape[2] if targets is not None else 2048
    model = Separator(
        input_channels=fit["features"].shape[2],
        width=cfg["width"],
        code_channels=cfg["code_channels"],
        embedding_dim=dim,
        min_scale=cfg["min_scale"],
        min_probability=cfg["min_probability"],
        variant=variant,
    ).to(device)
    heads = deepcopy(teacher_heads).to(device)
    if cfg["head_mode"] == "frozen":
        freeze(heads)
    else:
        heads.requires_grad_(True)
    opt = torch.optim.Adam(
        [*model.parameters(), *[p for p in heads.parameters() if p.requires_grad]],
        lr=cfg["lr"],
    )
    medical_rng = np.random.default_rng(cfg["seed"])
    medical_seen = np.zeros(len(fit["rows"]), dtype=np.int64)
    attack_rng = np.random.default_rng(cfg["seed"] + 1)
    clinical_weights = weights_for(fit, device)
    attack = attack_opt = sampler = None
    if variant == "adversarial":
        # Keep model/dropout/Gumbel RNG unchanged merely by constructing the extra head.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg["seed"] + 200)
            attack = PairAttacker(fit["features"].shape[2]).to(device)
        attack_opt = torch.optim.Adam(attack.parameters(), lr=cfg["attack_lr"])
        sampler = P2Sampler(fit["rows"])

    def attack_update():
        model.eval()
        attack.train().requires_grad_(True)
        a, av, b, bv, labels = pair_batch(fit, sampler, attack_rng, device)
        with torch.no_grad():
            a = model(a, av, stochastic=False)["semantic"]
            b = model(b, bv, stochastic=False)["semantic"]
        attack_opt.zero_grad(set_to_none=True)
        value = F.binary_cross_entropy_with_logits(attack(a, av, b, bv), labels)
        if not torch.isfinite(value):
            raise RuntimeError("Nonfinite training adversary")
        value.backward()
        attack_opt.step()

    total = cfg["warmup_steps"] + cfg["joint_steps"]
    for step in range(1, total + 1):
        joint = step > cfg["warmup_steps"]
        if variant == "adversarial" and step == cfg["warmup_steps"] + 1:
            for _ in range(cfg["attack_warmup_steps"]):
                attack_update()
            log(
                output, "adversary_warmup_completed", updates=cfg["attack_warmup_steps"]
            )
        if joint and attack is not None:
            for _ in range(2):
                attack_update()
        model.train()
        progress = max(
            0.0, (step - cfg["warmup_steps"] - 1) / max(1, cfg["joint_steps"] - 1)
        )
        tau = (
            cfg["temperature_start"]
            * (cfg["temperature_end"] / cfg["temperature_start"]) ** progress
        )
        idx = medical_rng.integers(len(fit["rows"]), size=cfg["batch_size"])
        np.add.at(medical_seen, idx, 1)
        opt.zero_grad(set_to_none=True)
        loss, terms = 0, {}
        for d in (0, 1):
            x, v, y = batch(fit, idx, d, device)
            if not joint:
                # Do not run the untrained mask/link branch during complete-code pretraining.
                from .models import quantize

                q = quantize(model.encoder(x))
                result = dict(q=q, full=model.decoder(q))
                reconstruction = weighted_mean((result["full"] - x).square(), v)
                rate, diagnostic = model.rate(result, v)
                value = (
                    cfg["weights"]["all"] * reconstruction
                    + cfg["weights"]["rate"] * rate
                )
                report = dict(all=reconstruction, rate=rate, **diagnostic)
            else:
                result = model(x, v, temperature=tau)
                embedding = (
                    targets["fit"][idx, d].to(device) if targets is not None else None
                )
                value, report = core_objective(
                    model,
                    result,
                    x,
                    v,
                    y,
                    heads,
                    d,
                    embedding,
                    cfg["weights"],
                    clinical_weights,
                )
            loss = loss + value / 2
            for key, val in report.items():
                terms[key] = terms.get(key, 0.0) + float(val.detach()) / 2
        privacy = torch.zeros((), device=device)
        if joint and attack is not None:
            freeze(attack)
            a, av, b, bv, _ = pair_batch(fit, sampler, attack_rng, device)
            # Match online attacks' deterministic representation; keep input gradients.
            privacy = confusion_loss(
                attack(
                    model(a, av, temperature=tau, stochastic=False)["semantic"],
                    av,
                    model(b, bv, temperature=tau, stochastic=False)["semantic"],
                    bv,
                )
            )
            loss = loss + cfg["weights"]["confusion"] * privacy
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite separator objective")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(heads.parameters()),
            cfg["gradient_clip"],
            error_if_nonfinite=True,
        )
        opt.step()
        if step % cfg["evaluate_every"] == 0 or step in (cfg["warmup_steps"], total):
            log(
                output,
                "separator_update",
                step=step,
                phase="joint" if joint else "warmup",
                temperature=tau,
                loss=float(loss.detach()),
                privacy=float(privacy.detach()),
                gradient_norm=float(norm),
                **terms,
            )
            if "mask" in result:
                occupancy = (result["mask"].detach() * v).sum(
                    (0, 2, 3)
                ) / v.sum().clamp_min(1e-8)
                log(
                    output,
                    "last_gray_batch_code_diagnostic",
                    step=step,
                    mask_channel_occupancy=occupancy.cpu().tolist(),
                    q_min=float(result["q"].detach().min()),
                    q_max=float(result["q"].detach().max()),
                    scope="last_sampled_gray_batch_not_full_dataset",
                )
            diagnostics = {}
            for view in ("full", "semantic"):
                projected = represent(
                    {"val_tune": cache["val_tune"]}, model, view, device
                )
                diagnostics[view] = {
                    d: r["macro_auroc"]
                    for d, r in clinical_report(
                        heads, projected["val_tune"], device
                    ).items()
                }
            log(
                output,
                "frozen_or_joint_head_diagnostic",
                step=step,
                head_mode=cfg["head_mode"],
                domains=diagnostics,
            )
    state = dict(
        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        spec=model.spec,
        heads={k: v.detach().cpu() for k, v in heads.state_dict().items()},
        statistics=statistics,
        optimizer=opt.state_dict(),
        step=total,
        config=cfg,
        torch_rng=torch.get_rng_state(),
        medical_rng=medical_rng.bit_generator.state,
        attack_rng=attack_rng.bit_generator.state,
        endpoint="fixed_budget_not_acceptance",
    )
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    if attack is not None:
        state.update(
            attack=attack.state_dict(), attack_optimizer=attack_opt.state_dict()
        )
    save(output / "separator.pt", state)
    write(
        output / "training_summary.json",
        dict(
            variant=variant,
            steps=total,
            head_mode=cfg["head_mode"],
            endpoint="fixed_budget",
            scientific_acceptance="requires_independent_audit",
            pair_fallbacks=sampler.fallbacks if sampler else 0,
            medical_images_seen=int((medical_seen > 0).sum()),
            medical_fit_images=len(fit["rows"]),
            medical_patients_seen=len(
                {r["patient_id"] for i, r in enumerate(fit["rows"]) if medical_seen[i]}
            ),
            masks_evaluated="deterministic_hard",
            resume_supported=False,
        ),
    )
    return model


def load_separator(path, device):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    model = Separator(**obj["spec"]).to(device)
    model.load_state_dict(obj["model"], strict=True)
    heads = ClinicalHeads(obj["spec"]["input_channels"]).to(device)
    heads.load_state_dict(obj["heads"], strict=True)
    return freeze(model), freeze(heads), obj["statistics"]
