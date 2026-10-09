from collections import Counter, defaultdict
import json
import numpy as np
import torch
from .data import collate
from .models.common import gray
from .models.attacks import ncc
from .losses import ssim_values
from .metrics import verification, retrieval, pair_bootstrap
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def evaluate(
    data,
    device,
    scenario,
    query_view,
    gallery_view,
    method,
    allowed_queries=None,
    allowed_gallery=None,
):
    protocol = json.loads(
        (
            data.root
            / "protocol_audit_20261009/pair_protocol"
            / f"{data.role}_{scenario}.json"
        ).read_text()
    )
    query_images = {}
    gallery_images = {}
    for i in range(len(data)):
        b = collate([data[i]], device)
        name = b["id"][0]
        for dst, view, allowed in (
            (query_images, query_view, allowed_queries),
            (gallery_images, gallery_view, allowed_gallery),
        ):
            if allowed is not None and name not in allowed:
                continue
            x = view(b) if view else b["x"]
            if x is not None:
                dst[name] = gray(x) if x.shape[1] == 3 else x
    viable = []
    for q in protocol["queries"]:
        pos = [n for n in q["positive_gallery"] if n in gallery_images]
        neg = [n for n in q["negative_gallery"] if n in gallery_images]
        if q["query"] in query_images and pos and neg:
            viable.append((q, pos, neg))
    counts = Counter(q["patient_id"] for q, _, _ in viable)
    truth = []
    scores = []
    weights = []
    pairs = []
    ranks = []
    for q, pos, neg in viable:
        x = query_images[q["query"]]
        names = pos + neg
        s = []
        for name in names:
            y = gallery_images[name]
            value = (
                ncc(x, y)
                if method == "ncc"
                else ssim_values(x, y, torch.ones_like(x))[0]
            )
            s.append(float(value))
        ranks.append({"patient": q["patient_id"], **retrieval(s, names, pos)})
        for j, name in enumerate(names):
            label = int(j < len(pos))
            truth.append(label)
            scores.append(s[j])
            weights.append(
                1 / counts[q["patient_id"]] / (len(pos) if label else len(neg))
            )
            pairs.append((q["patient_id"], data.rows[data.index[name]]["patient_id"]))
    if not scores:
        return {"valid": False, "reason": "No legal query cohort"}
    result = verification(truth, scores, weights)
    by = defaultdict(list)
    for r in ranks:
        by[r["patient"]].append(r)
    result.update(
        {
            key: float(
                np.mean([np.mean([r[key] for r in group]) for group in by.values()])
            )
            for key in ("rank1", "ap", "chance_rank1")
        }
    )
    result.update(
        valid=True,
        queries=len(viable),
        dropped_queries=len(protocol["queries"]) - len(viable),
        method=method,
    )
    base = np.asarray(weights)
    result["auroc_ci"] = pair_bootstrap(
        pairs, lambda w: roc_auc_score(truth, scores, sample_weight=base * w)
    )
    return result
