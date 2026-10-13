"""R3 observational diagnostics; no extra optimizer updates or privacy claims."""

import torch
from torch.nn import functional as F

from ccrr.codec.metadata import CLASSES
from ccrr.tensor_ops import masked_mean, pooled
from ccrr.training.losses import weighted_cross_entropy
from ccrr.training.b_batches import pair_backward
from ccrr.training.optimization import update
from ccrr.data.pad import rng_for
from ccrr.errors import require
from ccrr.training.b_diagnostics import isolated_rng
from probe_stage_b_reconstruction import measure, cohort


def initialize(e):
    """Only on a new run, after the frozen A mfm_probe has been restored."""
    from ccrr.training.b_control import ReconstructionControl

    with torch.no_grad():
        e.purifier.encoder.end.weight.mul_(0.1)
        e.purifier.decoder.end.weight.mul_(0.1)
    with isolated_rng():
        control = ReconstructionControl(e.purifier.mean, e.purifier.std).to(e.device)
        control.encoder.load_state_dict(e.purifier.encoder.state_dict())
        control.decoder.load_state_dict(e.purifier.decoder.state_dict())
    return control


def image_backward(e, bank, ids):
    """Exact production loss arithmetic, plus detached component bookkeeping."""
    labels = [CLASSES.index(e.fit.rows[int(i)]["diagnostic"]) for i in ids]
    denom = e.weights[torch.tensor(labels, device=e.device)].sum() * 3
    total = torch.zeros((), device=e.device)
    terms = dict(all=0.0, identity=0.0, mask=0.0, clinical=0.0)
    n = e.cfg.get("microbatch", 1)
    for start in range(0, len(ids), n):
        group = ids[start : start + n]
        t = bank.get(group)
        v = t["v"]
        px, py = e.purifier(t["f0"], v), e.purifier(t["f1"], v)
        cross = e.purifier(t["f1"], v, source_mask=px["mask"])
        loss = torch.zeros((), device=e.device)
        for out, target in ((px, t["e0"]), (py, t["e1"])):
            all_loss = masked_mean((out["all"] - out["standardized"]).square(), v)
            id_loss = (out["identity"] - target).square().sum(1).mean()
            occ = pooled(out["mask"], v).mean(1)
            mask_loss = (F.relu(0.1 - occ).square() + F.relu(occ - 0.5).square()).mean()
            loss = loss + (all_loss + 0.1 * id_loss + 10 * mask_loss) * len(group) / 32
            for k, x in (("all", all_loss), ("identity", id_loss), ("mask", mask_loss)):
                terms[k] += float(x.detach()) * len(group) / 32
        y = torch.tensor(labels[start : start + n], device=e.device)
        for out in (px, py, cross):
            part = weighted_cross_entropy(out["logits"], y, e.weights, weight_sum=denom)
            loss = loss + part
            terms["clinical"] += float(part.detach())
        loss.backward()
        total += loss.detach()
    return total, terms


def train_step(e, bank, step, opt, adv, sched, advsched):
    e.purifier.train().requires_grad_(True)
    e.sem_attacker.train().requires_grad_(True)
    attacks = []
    for sub in range(2):
        rng = rng_for(e.seed, "B", "attacker" + str(sub), step)
        pairs = e.sampler.sample(rng)
        adv.zero_grad(set_to_none=True)
        attacks.append(float(pair_backward(e, bank, pairs, rng, 1.0, True)))
        advsched.apply(2 * step + sub)
        update(adv)
        advsched.advance()
    e.sem_attacker.eval().requires_grad_(False)
    adv.zero_grad(set_to_none=True)
    opt.zero_grad(set_to_none=True)
    rng = rng_for(e.seed, "B", "images", step)
    ids = rng.integers(len(e.fit), size=16)
    loss, terms = image_backward(e, bank, ids)
    ramp = max(0, min(1, (step - 999) / 1000))
    terms["adversarial_weighted"] = 0.0
    if ramp:
        pairs = e.sampler.sample(rng_for(e.seed, "B", "generator_pairs", step))
        part = pair_backward(e, bank, pairs, rng, -0.1 * ramp, False)
        loss += part
        terms["adversarial_weighted"] = float(part)
    require(torch.isfinite(loss), "Nonfinite diagnostic generator loss")
    gradients = {
        k: float(p.grad.norm())
        for k, p in e.purifier.named_parameters()
        if p.grad is not None
        and k in ("encoder.end.weight", "decoder.end.weight", "mask_head.weight")
    }
    sched.apply(step)
    norm = update(opt)
    sched.advance()
    return {
        "step": step + 1,
        "loss": float(loss),
        "raw_components": terms,
        "weighted_components": {
            "all": terms["all"],
            "clinical": terms["clinical"],
            "identity": 0.1 * terms["identity"],
            "mask": 10 * terms["mask"],
            "adversarial": terms["adversarial_weighted"],
        },
        "adversarial_bce": -terms["adversarial_weighted"] / (0.1 * ramp)
        if ramp
        else None,
        "gradient_norm_before_clip": norm,
        "clip_triggered": norm > 1,
        "layer_gradient_norms_before_clip": gradients,
        "adversarial_ramp": ramp,
        "learning_rates": [g["lr"] for g in opt.param_groups],
        "attacker_loss": attacks,
    }


@torch.no_grad()
def full_scores(e, model, bank, count, is_control=False):
    rows = {d: [] for d in ("f0", "f1")}
    for start in range(0, count, e.cfg["microbatch"]):
        ids = list(range(start, min(start + e.cfg["microbatch"], count)))
        t = bank.get(ids)
        for d in rows:
            p = model(t[d], t["v"])
            for j in range(len(ids)):
                v = t["v"][j : j + 1]
                item = {
                    "mse": float(
                        masked_mean(
                            (
                                p["all"][j : j + 1] - p["standardized"][j : j + 1]
                            ).square(),
                            v,
                        )
                    )
                }
                if not is_control:
                    gate, q = p["hard_mask"][j : j + 1], p["q"][j : j + 1]
                    item.update(
                        occupancy=float(pooled(gate, v).mean()),
                        identity_nonzero_fraction=float(
                            masked_mean(((gate * q) != 0).float(), v)
                        ),
                        semantic_nonzero_fraction=float(
                            masked_mean((((1 - gate) * q) != 0).float(), v)
                        ),
                        identity_code_energy=float(
                            masked_mean((gate * q / 7).square(), v)
                        ),
                        semantic_code_energy=float(
                            masked_mean(((1 - gate) * q / 7).square(), v)
                        ),
                    )
                rows[d].append(item)
    return rows


def observation(e, banks, step):
    with isolated_rng():
        rows = {}
        for role, bank in banks.items():
            ids = cohort(e.data[role].rows)[0]
            t = bank.get(ids)
            probes = {d: measure(e.purifier, t[d], t["v"]) for d in ("f0", "f1")}
            with torch.no_grad():
                for d in probes:
                    p = e.purifier(t[d], t["v"])
                    logits = e.purifier.mask_head(p["q"] / 7)
                    prob = logits.sigmoid()
                    v = t["v"]
                    probes[d]["mask_logit_abs_mean"] = float(
                        masked_mean(logits.abs(), v)
                    )
                    probes[d]["mask_sigmoid_derivative_mean"] = float(
                        masked_mean(prob * (1 - prob), v)
                    )
                    gate = p["hard_mask"]
                    mu = (gate * v).sum(0) / v.sum(0).clamp_min(1e-12)
                    probes[d]["mask_between_image_variance"] = float(
                        masked_mean((gate - mu).square(), v)
                    )
            rows[role] = {
                "probe_indices": ids,
                "probe_diagnostics": probes,
                "all_images": full_scores(e, e.purifier, bank, len(e.data[role])),
            }
        return {"step": step, "roles": rows}


def permute_mask(gate, generator):
    # A bijection at each position: exact occupancy, including fractional coverage.
    order = (
        torch.rand(gate.shape, device="cpu", generator=generator)
        .argsort(dim=1)
        .to(gate.device)
    )
    out = torch.gather(gate, 1, order)
    require(torch.equal(gate.sum(1), out.sum(1)), "Mask occupancy changed")
    return out


def mask_diagnostic(e, bank, audit):
    from b_r3_audit import classifier

    with isolated_rng(), torch.no_grad():
        heads = {}
        for family in ("linear", "mlp"):
            saved = torch.load(
                audit.heads / f"clinical-sem-{family}.pt",
                map_location="cpu",
                weights_only=True,
            )
            head = classifier(family).to(e.device)
            head.load_state_dict(saved["training_state"]["data"]["best_weights"])
            heads[family] = head.eval()
        rng = torch.Generator().manual_seed(20261008)
        rows = []
        for start in range(0, len(e.data["val_select"]), e.cfg["microbatch"]):
            ids = list(
                range(
                    start, min(start + e.cfg["microbatch"], len(e.data["val_select"]))
                )
            )
            t = bank.get(ids)
            v = t["v"]
            px = e.purifier(t["f0"], v)
            for d in range(3):
                f = t["f0" if d == 0 else "f1"]
                own = e.purifier(f, v, source_mask=px["hard_mask"] if d == 2 else None)
                gate = px["hard_mask"] if d == 2 else own["hard_mask"]
                altered = e.purifier(f, v, source_mask=permute_mask(gate, rng))
                target = t["e0" if d == 0 else "e1"]
                for j, i in enumerate(ids):
                    item = {
                        "index": i,
                        "domain": d,
                        "label": CLASSES.index(
                            e.data["val_select"].rows[i]["diagnostic"]
                        ),
                        "semantic_change_mse": float(
                            masked_mean(
                                (
                                    own["semantic"][j : j + 1]
                                    - altered["semantic"][j : j + 1]
                                ).square(),
                                v[j : j + 1],
                            )
                        ),
                        "identity_mse_before": float(
                            (own["identity"][j] - target[j]).square().sum()
                        ),
                        "identity_mse_after": float(
                            (altered["identity"][j] - target[j]).square().sum()
                        ),
                        "predictions": {},
                    }
                    for name, h in heads.items():
                        item["predictions"][name] = {
                            k: h(pooled(p["semantic"][j : j + 1], v[j : j + 1]))
                            .softmax(1)[0]
                            .cpu()
                            .tolist()
                            for k, p in (("original", own), ("permuted", altered))
                        }
                    rows.append(item)
        return {
            "rows": rows,
            "occupancy_preserved": True,
            "diagnostic_only": True,
            "identity_or_clinical_localization_proven": False,
            "privacy_evaluation": False,
        }
