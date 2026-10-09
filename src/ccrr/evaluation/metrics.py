"""Metrics preserve failed-publication and failed-authentication denominators."""

from collections import Counter
import hashlib
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from skimage.color import rgb2lab, deltaE_ciede2000
from ccrr.tensor_ops import ssim_values


def quality(x, rec, v):
    with torch.no_grad():
        raw = rec
        rec = rec.clamp(0, 1)
        denom = (v.sum((1, 2, 3)) * x.shape[1]).clamp_min(1)
        mse = ((x - rec).square() * v).sum((1, 2, 3)) / denom
        psnr = -10 * mse.log10()
        ssim, valid = ssim_values(x, rec, v)
        out = []
        for i in range(len(x)):
            a = x[i].permute(1, 2, 0).cpu().numpy()
            b = rec[i].permute(1, 2, 0).cpu().numpy()
            mask = v[i, 0].bool().cpu().numpy()
            if a.shape[-1] == 1:
                a = np.repeat(a, 3, -1)
                b = np.repeat(b, 3, -1)
            de = deltaE_ciede2000(rgb2lab(a), rgb2lab(b))[mask]
            outside = ((raw[i] < 0) | (raw[i] > 1)) & v[i].bool()
            out.append(
                {
                    "mse": float(mse[i]),
                    "psnr": float(psnr[i]),
                    "ssim": float(ssim[i]) if valid[i] else None,
                    "delta_e00_mean": float(de.mean()),
                    "delta_e00_p95": float(np.quantile(de, 0.95, method="higher")),
                    "out_of_range_fraction": float(outside.sum() / denom[i]),
                    "exact_pixels": bool(mse[i] == 0),
                }
            )
        return out


def clinical(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = p.argmax(1)
    aucs = []
    for c in range(6):
        truth = y == c
        aucs.append(
            float(roc_auc_score(truth, p[:, c])) if len(np.unique(truth)) == 2 else None
        )
    cm = confusion_matrix(y, pred, labels=range(6))
    denom = cm.sum(1)
    return {
        "macro_auroc": float(np.mean(aucs))
        if all(a is not None for a in aucs)
        else None,
        "per_class_auroc": aucs,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, labels=range(6), average="macro", zero_division=0)
        ),
        "recall": [float(cm[i, i] / denom[i]) if denom[i] else None for i in range(6)],
        "class_n": denom.tolist(),
        "n": len(y),
    }


def cluster_bootstrap(patient_ids, statistic, repetitions=2000, seed=20261008):
    """statistic receives image multiplicities; paired methods share each draw."""
    ids = np.asarray(patient_ids)
    patients = np.unique(ids)
    rng = np.random.Generator(np.random.PCG64(seed))
    values = []
    for _ in range(repetitions):
        c = Counter(rng.choice(patients, len(patients), replace=True))
        weights = np.array([c[p] for p in ids])
        try:
            value = statistic(weights)
        except (ValueError, ZeroDivisionError):
            value = np.nan
        if value is not None and np.isfinite(value):
            values.append(float(value))
    invalid = 1 - len(values) / repetitions
    return {
        "lower": float(np.percentile(values, 2.5)) if values else None,
        "upper": float(np.percentile(values, 97.5)) if values else None,
        "invalid_fraction": invalid,
        "evidence_sufficient": invalid <= 0.05,
        "replicates": repetitions,
    }


def pair_bootstrap(pairs, statistic, repetitions=2000, seed=20261008):
    patients = sorted({p for a, b in pairs for p in (a, b)})
    rng = np.random.Generator(np.random.PCG64(seed))
    vals = []
    for _ in range(repetitions):
        c = Counter(rng.choice(patients, len(patients), replace=True))
        weights = np.array([c[a] if a == b else c[a] * c[b] for a, b in pairs])
        try:
            v = statistic(weights)
        except ValueError:
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    return {
        "interval": np.percentile(vals, [2.5, 97.5]).tolist() if vals else None,
        "invalid_fraction": 1 - len(vals) / repetitions,
    }


def verification(labels, scores, weights=None, threshold=None):
    a = float(roc_auc_score(labels, scores, sample_weight=weights))
    out = {"auroc": a, "a_dagger": max(a, 1 - a)}
    if threshold is not None:
        labels = np.asarray(labels)
        w = np.ones(len(labels)) if weights is None else np.asarray(weights)
        out["tpr_at_fixed_threshold"] = float(
            np.average(
                np.asarray(scores)[labels == 1] >= threshold, weights=w[labels == 1]
            )
        )
        out["fpr_at_fixed_threshold"] = float(
            np.average(
                np.asarray(scores)[labels == 0] >= threshold, weights=w[labels == 0]
            )
        )
    return out


def retrieval(scores, gallery_ids, positive_ids):
    positive = set(positive_ids)
    if not positive:
        return {"rank1": None, "ap": None, "reason": "No remaining legal positive"}
    order = sorted(
        range(len(scores)),
        key=lambda i: (
            -float(scores[i]),
            hashlib.sha256(gallery_ids[i].encode()).digest(),
        ),
    )
    truth = np.array([gallery_ids[i] in positive for i in order])
    hits = np.cumsum(truth)
    return {
        "rank1": float(truth[0]),
        "ap": float(((hits / np.arange(1, len(hits) + 1)) * truth).sum() / truth.sum()),
        "chance_rank1": float(truth.mean()),
        "ties": int(len(scores) - len(set(map(float, scores)))),
    }


def failure_funnel(rows):
    total = len(rows)
    published = [r for r in rows if r.get("published")]
    counts = {
        name: sum(r.get(name) == True for r in rows)
        for name in ("published", "c_success", "m_success", "both_success")
    }
    return {
        "all_inputs": total,
        "published_carriers": len(published),
        "counts": counts,
        "all_input_rates": {k: v / total if total else None for k, v in counts.items()},
        "published_rates": {
            k: sum(r.get(k) == True for r in published) / len(published)
            if published
            else None
            for k in counts
        },
        "public_utility_privacy_ids": [r["id"] for r in published],
        "authorized_quality_ids": [r["id"] for r in rows if r.get("c_success")],
        "failures": dict(Counter(r.get("failure") for r in rows if r.get("failure"))),
    }


def raw_ber(logits, bits, occ):
    active = occ.bool()
    n = int(active.sum())
    if not n:
        return {"data_ber": None, "header_ber": None, "records": 0}
    data = logits[..., 49:] > 0
    truth = bits[..., 43:].bool()
    small = []
    for start, end, width in ((1, 5, 2), (5, 7, 1), (7, 11, 2)):
        v = logits[..., start:end].argmax(-1)
        small.append(
            torch.stack([((v >> k) & 1).bool() for k in reversed(range(width))], -1)
        )
    header = torch.cat((*small, logits[..., 11:49] > 0), -1)
    return {
        "data_ber": float((data[active] != truth[active]).float().mean()),
        "header_ber": float(
            (header[active] != bits[..., :43][active].bool()).float().mean()
        ),
        "records": n,
    }
