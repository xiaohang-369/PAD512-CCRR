"""Experimental fit-only Rank-1 proxies. No optimizer or training entry point."""

import math
from collections import Counter
import torch
from torch.nn import functional as F


def fit_spec(rows, *, role, content_hashes):
    """Keep every fit row in bank order; exclude ineligible/duplicate source views.

    source_image_id may identify preprocessed views of one original acquisition.
    Exact RGB512 hashes additionally exclude identical content under different IDs.
    Metadata/hashes cannot establish that two acquisitions are independent.
    """
    if role != "fit" or not rows:
        raise ValueError("Training bank requires nonempty fit rows")
    required = ("img_id", "patient_id", "lesion_id")
    if any(any(not r.get(k) for k in required) for r in rows):
        raise ValueError("Missing source/patient/lesion identity")
    if len({r["img_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate row IDs")
    if any(not content_hashes.get(r["img_id"]) for r in rows):
        raise ValueError("Every row needs a bound content hash")
    eligible = [str(r.get("reid_observation_eligible", "1")) == "1" for r in rows]
    n = len(rows)
    positive = torch.zeros(n, n, dtype=torch.bool)
    negative = torch.zeros_like(positive)
    for i, q in enumerate(rows):
        for j, b in enumerate(rows):
            same_source = q.get("source_image_id", q["img_id"]) == b.get(
                "source_image_id", b["img_id"]
            )
            same_content = content_hashes[q["img_id"]] == content_hashes[b["img_id"]]
            if not eligible[i] or not eligible[j] or same_source or same_content:
                continue
            same_patient = q["patient_id"] == b["patient_id"]
            positive[i, j] = same_patient and q["lesion_id"] == b["lesion_id"]
            negative[i, j] = not same_patient
    valid = positive.any(1) & negative.any(1)
    counts = Counter(r["patient_id"] for r, v in zip(rows, valid) if v)
    # Uniform all-fit query sampling: E[mean_B(w_i L_i)] = patient-equal risk.
    # Fixed weights also make ordinary DDP averaging exact across empty shards.
    weights = torch.tensor([
        n / (len(counts) * counts[r["patient_id"]]) if bool(v) else 0.0
        for r, v in zip(rows, valid)
    ])
    return dict(positive=positive, negative=negative, valid=valid, weights=weights)


def score_loss(scores, positive, negative, *, tau=0.1, margin=0.0,
               mode="live", anchor=None, weights=None):
    """Raw softplus scale, matching the draft (no leading tau multiplier).

    live: draft max-positive minus max-negative; directly rewards raising negatives.
    stop_negative: same values, only positive derivative; moving q can still raise b.
    anchor: immutable per-query G0 negative maximum; not a current-rank surrogate.
    Hard maxima have selected-index subgradients at ties. Ties are failures in logs.
    weights=None is valid-query mean for diagnostics only. Fixed fit weights use
    full batch size (including invalid rows), required for uniform all-fit/DDP use.
    """
    if not math.isfinite(tau) or tau <= 0 or not math.isfinite(margin) or margin < 0:
        raise ValueError("Require finite tau>0 and margin>=0")
    if mode not in ("live", "stop_negative", "anchor"):
        raise ValueError("Unknown rank proxy")
    if scores.ndim != 2 or not len(scores) or not torch.isfinite(scores).all():
        raise ValueError("Nonempty finite similarity matrix required")
    for mask in (positive, negative):
        if mask.shape != scores.shape or mask.dtype != torch.bool:
            raise ValueError("Aligned boolean masks required")
    if (positive & negative).any():
        raise ValueError("Positive/negative overlap")
    if weights is not None and (weights.shape != (len(scores),) or
            not torch.isfinite(weights).all() or (weights < 0).any()):
        raise ValueError("Invalid fixed fit weights")
    if mode == "anchor" and (anchor is None or anchor.shape != (len(scores),) or
                              not torch.isfinite(anchor).all()):
        raise ValueError("Require aligned finite frozen G0 negative maxima")
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return scores.sum() * 0, dict(valid=valid, a=scores.new_empty(0),
                                     b=scores.new_empty(0), gap=scores.new_empty(0))
    s, p, n = scores[valid], positive[valid], negative[valid]
    a = s.masked_fill(~p, -torch.inf).max(1).values
    b = s.masked_fill(~n, -torch.inf).max(1).values
    ref = b if mode == "live" else b.detach()
    if mode == "anchor":
        ref = anchor[valid].detach()
    terms = F.softplus((a - ref + margin) / tau)
    loss = terms.mean() if weights is None else (terms * weights[valid]).sum() / len(scores)
    return loss, dict(valid=valid, a=a, b=b, gap=a-b,
                      strict_positive_top1=a > b, ties=a == b)


def rank_loss(query, gallery, positive, negative, **kwargs):
    if query.ndim != 2 or gallery.ndim != 2 or query.shape[1] != gallery.shape[1]:
        raise ValueError("Aligned embedding dimensions required")
    if not torch.isfinite(query).all() or not torch.isfinite(gallery).all():
        raise ValueError("Nonfinite embeddings")
    if (query.norm(dim=1) <= 1e-8).any() or (gallery.norm(dim=1) <= 1e-8).any():
        raise ValueError("Zero embeddings")
    scores = F.normalize(query, dim=1) @ F.normalize(gallery.detach(), dim=1).T
    return score_loss(scores, positive, negative, **kwargs)


def vector_summary(vectors):
    """Accumulate long VJP reductions in float64; zero cosine is undefined."""
    values = {k: v.detach().double() for k,v in vectors.items()}
    if any(not torch.isfinite(v).all() for v in values.values()):
        raise ValueError("Nonfinite diagnostic gradient")
    norms = {k: float(v.norm()) for k, v in values.items()}
    cosine = {}
    for k, a in values.items():
        for j, b in values.items():
            if k < j:
                denom = norms[k] * norms[j]
                cosine[k+"__"+j] = max(-1.,min(1.,float(a @ b) / denom)) if denom > 0 else None
    return dict(l2=norms, cosine=cosine)
