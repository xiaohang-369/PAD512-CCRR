"""Fixed P2 queries/galleries, deterministic tie handling, independent readouts."""

from collections import Counter
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

from ccrr.evaluation.metrics import clinical, cluster_bootstrap
from ccrr_v2.models import PairAttacker
from .models import pool, ClinicalHeads
from .pairs import P2Sampler, validate_p2_protocol
from .data import batch, pair_batch
from .runtime import log, save, write, seed_all

DIRECTIONS = {"rgb_rgb": (0, 0), "gray_rgb": (1, 0), "gray_gray": (1, 1)}


def expected_retrieval(scores, labels):
    """Expected rank-1 and AP under uniform ordering within exactly tied scores."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    if not np.isfinite(scores).all() or not labels.sum():
        raise ValueError("Invalid retrieval input")
    ap, previous_n, previous_pos = 0.0, 0, 0
    for score in sorted(set(scores.tolist()), reverse=True):
        group = labels[scores == score]
        m, k = len(group), int(group.sum())
        for t in range(1, m + 1):
            inside = (t - 1) * (k - 1) / (m - 1) if m > 1 else 0
            ap += (k / m) * (previous_pos + 1 + inside) / (previous_n + t)
        previous_n += m
        previous_pos += k
    return float(labels[scores == scores.max()].mean()), float(ap / labels.sum())


def p2_report(rows, protocol, score_query, orientation=1, repetitions=0):
    validate_p2_protocol(protocol, rows)
    if orientation not in (-1, 1):
        raise ValueError("Orientation must be fixed on val_tune")
    index = {r["img_id"]: i for i, r in enumerate(rows)}
    counts = Counter(q["patient_id"] for q in protocol["queries"])
    labels, scores, weights, query_patients, queries = [], [], [], [], []
    for q in protocol["queries"]:
        names = q["positive_gallery"] + q["negative_gallery"]
        truth = np.array(
            [1] * len(q["positive_gallery"]) + [0] * len(q["negative_gallery"])
        )
        values = (
            np.asarray(score_query(index[q["query"]], [index[n] for n in names]))
            * orientation
        )
        if values.shape != truth.shape or not np.isfinite(values).all():
            raise ValueError("Invalid P2 score vector")
        w = [
            1 / counts[q["patient_id"]] / int((truth == label).sum()) for label in truth
        ]
        rank1, ap = expected_retrieval(values, truth)
        queries.append(
            dict(
                query=q["query"],
                patient=q["patient_id"],
                gallery=names,
                labels=truth.tolist(),
                scores=values.tolist(),
                rank1=rank1,
                ap=ap,
                chance=float(truth.mean()),
                weight=1 / counts[q["patient_id"]],
            )
        )
        labels.extend(truth.tolist())
        scores.extend(values.tolist())
        weights.extend(w)
        query_patients.extend([q["patient_id"]] * len(truth))
    qw = np.array([q["weight"] for q in queries])
    auc = float(roc_auc_score(labels, scores, sample_weight=weights))
    report = dict(
        auroc=auc,
        rank1=float(np.average([q["rank1"] for q in queries], weights=qw)),
        map=float(np.average([q["ap"] for q in queries], weights=qw)),
        chance_rank1=float(np.average([q["chance"] for q in queries], weights=qw)),
        orientation_from_val_tune=orientation,
        queries=queries,
        diagnostic_two_sided_auc=max(auc, 1 - auc),
        tie_policy="expected_uniform_within_equal_scores",
    )
    if repetitions:
        report["rank1_ci"] = cluster_bootstrap(
            [q["patient"] for q in queries],
            lambda w: np.average([q["rank1"] for q in queries], weights=qw * w),
            repetitions,
        )
        report["auroc_ci"] = cluster_bootstrap(
            query_patients,
            lambda w: roc_auc_score(
                labels, scores, sample_weight=np.array(weights) * w
            ),
            repetitions,
        )
        report["ci_scope"] = (
            "query_patient_cluster_given_fixed_gallery_not_retrained_attack_uncertainty"
        )
    return report


def clinical_report(heads, data, device, repetitions=0):
    heads.eval()
    result = {}
    for d, name in enumerate(("rgb", "gray")):
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(data["rows"]), 32):
                x, v, _ = batch(data, slice(start, start + 32), d, device)
                probabilities.extend(heads(x, v, d).softmax(1).cpu().tolist())
        labels = data["labels"].tolist()
        report = clinical(labels, probabilities)
        report.update(
            labels=labels,
            probabilities=probabilities,
            ids=[r["img_id"] for r in data["rows"]],
            patients=[r["patient_id"] for r in data["rows"]],
        )
        if repetitions:
            y, p = np.asarray(labels), np.asarray(probabilities)
            report["macro_auroc_ci"] = cluster_bootstrap(
                report["patients"],
                lambda w: np.mean(
                    [roc_auc_score(y == c, p[:, c], sample_weight=w) for c in range(6)]
                ),
                repetitions,
            )
        result[name] = report
    return result


def weights_for(data, device):
    counts = torch.bincount(data["labels"], minlength=6).float()
    if (counts <= 0).any():
        raise ValueError("Fit missing a clinical class")
    return (counts.rsqrt() / counts.rsqrt().mean()).to(device)


def fit_heads(cache, cfg, output, device, mlp=False):
    seed_all(cfg["seed"])
    channels = cache["fit"]["features"].shape[2]
    heads = ClinicalHeads(channels).to(device)
    if mlp:
        heads.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(channels, 256), nn.ReLU(), nn.Linear(256, 6))
                for _ in range(2)
            ]
        ).to(device)
    opt = torch.optim.Adam(heads.parameters(), lr=cfg["probe_lr"])
    fit, best, selected = cache["fit"], -float("inf"), None
    rng, weights = np.random.default_rng(cfg["seed"]), weights_for(fit, device)
    for step in range(1, cfg["probe_steps"] + 1):
        heads.train()
        idx = rng.integers(len(fit["rows"]), size=cfg["batch_size"])
        opt.zero_grad(set_to_none=True)
        loss = 0
        for d in (0, 1):
            x, v, y = batch(fit, idx, d, device)
            loss = (
                loss
                + nn.functional.cross_entropy(heads(x, v, d), y, weight=weights) / 2
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite clinical objective")
        loss.backward()
        opt.step()
        if step % cfg["evaluate_every"] == 0 or step == cfg["probe_steps"]:
            report = clinical_report(heads, cache["val_tune"], device)
            values = [r["macro_auroc"] for r in report.values()]
            if None not in values and min(values) > best:
                best, selected, best_step = (
                    min(values),
                    deepcopy(heads.state_dict()),
                    step,
                )
            log(
                output,
                "clinical_probe",
                step=step,
                loss=float(loss.detach()),
                tune_auc=values,
            )
    if selected is None:
        raise RuntimeError("No estimable clinical endpoint")
    heads.load_state_dict(selected)
    save(
        output / "heads.pt",
        dict(model=selected, channels=channels, mlp=mlp, selected_step=best_step),
    )
    return heads


def attack_report(model, data, device, orientations=None, repetitions=0):
    model.eval()
    embedded = []
    with torch.no_grad():
        for d in (0, 1):
            parts = []
            for start in range(0, len(data["rows"]), 32):
                x, v, _ = batch(data, slice(start, start + 32), d, device)
                parts.append(pool(model.encoder(x), v))
            embedded.append(torch.cat(parts))
        result = {}
        for name, (da, db) in DIRECTIONS.items():

            def score(a, bs):
                output = []
                for offset in range(0, len(bs), 256):
                    b = embedded[db][bs[offset : offset + 256]]
                    av = embedded[da][a].expand_as(b)
                    output.extend(
                        model.head(torch.cat(((av - b).abs(), av * b), 1))
                        .flatten()
                        .cpu()
                        .tolist()
                    )
                return output

            result[name] = p2_report(
                data["rows"],
                data["protocols"]["p2"],
                score,
                orientation=(orientations or {}).get(name, 1),
                repetitions=repetitions,
            )
    return result


def fit_attack(cache, cfg, output, device, spatial=False):
    seed_all(cfg["seed"] + 100)
    fit = cache["fit"]
    model = PairAttacker(fit["features"].shape[2], spatial).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["attack_lr"])
    sampler, rng = P2Sampler(fit["rows"]), np.random.default_rng(cfg["seed"] + 100)
    best, selected = -float("inf"), None
    for step in range(1, cfg["attack_steps"] + 1):
        model.train()
        a, av, b, bv, labels = pair_batch(fit, sampler, rng, device)
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(a, av, b, bv), labels
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite audit attack")
        loss.backward()
        opt.step()
        if step % cfg["evaluate_every"] == 0 or step == cfg["attack_steps"]:
            report = attack_report(model, cache["val_tune"], device)
            value = min(r["diagnostic_two_sided_auc"] for r in report.values())
            if value > best:
                best, selected, best_step = value, deepcopy(model.state_dict()), step
                orientations = {
                    name: (1 if r["auroc"] >= 0.5 else -1) for name, r in report.items()
                }
            log(
                output,
                "independent_attack",
                step=step,
                spatial=spatial,
                tune_worst_auc=value,
            )
    model.load_state_dict(selected)
    save(
        output / "attacker.pt",
        dict(
            model=selected,
            spatial=spatial,
            selected_step=best_step,
            orientations=orientations,
            selection_role="val_tune",
        ),
    )
    return attack_report(
        model, cache["val_select"], device, orientations, cfg["bootstrap_repetitions"]
    )


def audit(cache, cfg, output, device):
    for name, mlp in (("linear", False), ("mlp", True)):
        directory = output / name
        directory.mkdir()
        heads = fit_heads(cache, cfg, directory, device, mlp)
        write(
            output / f"{name}.json",
            clinical_report(
                heads, cache["val_select"], device, cfg["bootstrap_repetitions"]
            ),
        )
    for name, spatial in (("global", False), ("spatial", True)):
        directory = output / name
        directory.mkdir()
        write(
            output / f"{name}.json", fit_attack(cache, cfg, directory, device, spatial)
        )
