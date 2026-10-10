"""A-stage capability reports and gates; no test-based selection (§6.6, §32)."""

from collections import defaultdict, Counter
import numpy as np
from torch import nn
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from ccrr.codec.metadata import CLASSES
from ccrr.tensor_ops import pooled
from ccrr.evaluation.metrics import (
    cluster_bootstrap,
    pair_bootstrap,
    verification,
    retrieval,
)
from ccrr.errors import require


class PatchProbe(nn.Module):
    """The very same masked patch grid passed to Purifier, never CLS."""

    def __init__(self, mfm, head):
        super().__init__()
        self.mfm, self.head = mfm, head

    def forward(self, x, rects):
        f, v, _ = self.mfm(x, rects)
        return self.head(pooled(f, v))


def clinical_intervals(report, repetitions=2000):
    y, p, patients = (
        np.asarray(report["labels"]),
        np.asarray(report["probabilities"]),
        report["patients"],
    )
    pred = p.argmax(1)

    def auc(c, w):
        return (
            roc_auc_score(y == c, p[:, c], sample_weight=w)
            if len(np.unique((y == c)[w > 0])) == 2
            else np.nan
        )

    def recall(c, w):
        return (
            np.average(pred[y == c] == c, weights=w[y == c])
            if w[y == c].sum()
            else np.nan
        )

    stats = {
        "macro_auroc": lambda w: np.mean([auc(c, w) for c in range(6)]),
        "balanced_accuracy": lambda w: np.mean([recall(c, w) for c in range(6)]),
        "macro_f1": lambda w: f1_score(
            y, pred, labels=range(6), average="macro", sample_weight=w, zero_division=0
        )
        if all(w[y == c].sum() for c in range(6))
        else np.nan,
    }
    for name, stat in stats.items():
        report[name + "_ci"] = cluster_bootstrap(patients, stat, repetitions)
    report["confusion_matrix"] = confusion_matrix(y, pred, labels=range(6)).tolist()
    report["patient_n"] = len(set(patients))
    report["per_class"] = [
        {
            "diagnosis": name,
            "images": int((y == c).sum()),
            "patients": len(set(np.asarray(patients)[y == c])),
            "negative_images": int((y != c).sum()),
            "auroc": report["per_class_auroc"][c],
            "recall": report["recall"][c],
            "auroc_ci": cluster_bootstrap(
                patients, lambda w, c=c: auc(c, w), repetitions
            ),
            "recall_ci": cluster_bootstrap(
                patients, lambda w, c=c: recall(c, w), repetitions
            ),
        }
        for c, name in enumerate(CLASSES)
    ]
    report["six_classes_estimable"] = all(
        a is not None for a in report["per_class_auroc"]
    )
    return report


def clinical_gate(reports, controls=None, *, enforce_class_defects=True):
    defects = []
    for domain, r in zip(("rgb", "gray"), reports):
        if r["macro_auroc"] is None or r["macro_auroc"] < 0.75:
            defects.append(domain + ": macro-AUROC below 0.75 or not estimable")
        for c, (auc, recall) in enumerate(zip(r["per_class_auroc"], r["recall"])):
            if enforce_class_defects and (
                auc is None or recall is None or auc <= 0.5 or recall == 0
            ):
                defects.append(domain + ": " + CLASSES[c] + " supervision defect")
    if controls is not None:
        for r, c in zip(reports, controls):
            if (
                c["macro_auroc"] is None
                or r["macro_auroc"] is None
                or r["macro_auroc"] < c["macro_auroc"] - 0.03
            ):
                defects.append("joint/control domain gap exceeds 0.03 or not estimable")
    return {
        "passed": not defects,
        "failures": defects,
        "interval_evidence_sufficient": all(
            r.get("macro_auroc_ci", {}).get("evidence_sufficient", False)
            for r in reports
        ),
    }


def paired_gray_change(reports, repetitions=2000):
    a, b = reports
    require(
        a["ids"] == b["ids"] and a["patients"] == b["patients"],
        "Unpaired RGB/gray capability report",
    )
    y = np.asarray(a["labels"])
    p, q = np.asarray(a["probabilities"]), np.asarray(b["probabilities"])

    def stat(w):
        if any(len(np.unique((y == c)[w > 0])) != 2 for c in range(6)):
            return np.nan
        return np.mean(
            [
                roc_auc_score(y == c, q[:, c], sample_weight=w)
                - roc_auc_score(y == c, p[:, c], sample_weight=w)
                for c in range(6)
            ]
        )

    return {
        "gray_minus_rgb": stat(np.ones(len(y))),
        "ci": cluster_bootstrap(a["patients"], stat, repetitions),
    }


def identity_summary(details, ranks, bootstrap=False, threshold=None, repetitions=2000):
    if not details:
        return {
            "valid": False,
            "reason": "No legal positive/negative queries",
            "auroc": None,
        }
    y = np.array([r["label"] for r in details])
    s = np.array([r["score"] for r in details])
    w = np.array([r["weight"] for r in details])
    pairs = [(r["patient"], r["gallery_patient"]) for r in details]
    report = verification(y, s, w, threshold)
    report.update(
        valid=True,
        score_details={
            "labels": y.tolist(),
            "scores": s.tolist(),
            "weights": w.tolist(),
            "pairs": pairs,
        },
        retrieval_rows=ranks,
    )
    groups = defaultdict(list)
    for r in ranks:
        groups[r["patient"]].append(r)
    report.update(queries=len(ranks), patients=len(groups))
    for k, out in [("rank1", "rank1"), ("ap", "map"), ("chance_rank1", "chance_rank1")]:
        vals = np.array([np.mean([r[k] for r in rows]) for rows in groups.values()])
        report[out] = float(vals.mean()) if len(vals) else None
        if bootstrap and len(vals):
            report[out + "_ci"] = cluster_bootstrap(
                list(groups), lambda wt: np.average(vals, weights=wt), repetitions
            )
    if bootstrap:
        report["auroc_ci"] = pair_bootstrap(
            pairs, lambda wt: roc_auc_score(y, s, sample_weight=wt * w), repetitions
        )
        if threshold is not None:
            report["tpr_ci"] = pair_bootstrap(
                pairs,
                lambda wt: np.average(s[y == 1] >= threshold, weights=(wt * w)[y == 1])
                if (wt * w)[y == 1].sum()
                else np.nan,
                repetitions,
            )
    report["retrieval_interval_scope"] = "query patients resampled; gallery fixed"
    return report


def teacher_identity_scores(
    protocol,
    rows,
    query_features,
    gallery_features,
    *,
    bootstrap=False,
    threshold=None,
    repetitions=2000,
):
    """Cosine only: the A ReID project/pair head is not trained by identity CE."""
    by_id = {r["img_id"]: r for r in rows}
    counts = Counter(q["patient_id"] for q in protocol["queries"])
    details, ranks, matched, unmatched = [], [], [], 0
    for q in protocol["queries"]:
        name, pid = q["query"], q["patient_id"]
        pos, neg = q["positive_gallery"], q["negative_gallery"]
        require(pos and neg, "Teacher protocol has an invalid query")
        scores = {
            n: float(
                (
                    query_features[name]["embedding"] * gallery_features[n]["embedding"]
                ).sum()
            )
            for n in pos + neg
        }
        rank = retrieval([scores[n] for n in pos + neg], pos + neg, pos)
        rank.update(patient=pid, query=name)
        ranks.append(rank)

        def row(n, label, weight):
            return {
                "patient": pid,
                "gallery_patient": by_id[n]["patient_id"],
                "label": label,
                "score": scores[n],
                "weight": weight,
                "diagnosis": by_id[name]["diagnostic"],
            }

        for label, names in [(1, pos), (0, neg)]:
            details.extend(row(n, label, 1 / counts[pid] / len(names)) for n in names)
        # Each positive member gets its own matched negatives; union is not a unit.
        units = [
            (
                p,
                [
                    n
                    for n in neg
                    if (by_id[n]["diagnostic"], by_id[n]["region"])
                    == (by_id[p]["diagnostic"], by_id[p]["region"])
                ],
            )
            for p in pos
        ]
        units = [(p, ns) for p, ns in units if ns]
        if not units:
            unmatched += 1
        for p, ns in units:
            matched.append(row(p, 1, 1 / len(units)))
            matched.extend(row(n, 0, 1 / len(units) / len(ns)) for n in ns)
            for r in matched[-(len(ns) + 1) :]:
                r["query"] = name
    # Patients with surviving matched queries have equal total weight.
    mq = defaultdict(set)
    for r in matched:
        mq[r["patient"]].add(r["query"])
    for r in matched:
        r["weight"] /= len(mq[r["patient"]])
    report = identity_summary(details, ranks, bootstrap, threshold, repetitions)
    aux = identity_summary(matched, [], bootstrap, threshold, repetitions)
    aux["queries_without_matched_negatives"] = unmatched
    aux["queries"] = sum(map(len, mq.values()))
    aux["patients"] = len(mq)
    aux["retrieval_applicable"] = False
    report["diagnosis_region_matched"] = aux
    report["per_diagnosis"] = {
        c: identity_summary(
            [r for r in details if r["diagnosis"] == c],
            [r for r in ranks if by_id[r["query"]]["diagnostic"] == c],
            bootstrap,
            threshold,
            repetitions,
        )
        for c in CLASSES
    }
    return {"cosine": report}


def reid_gate(reports):
    checks = []
    for r in reports:
        c = r["cosine"]
        ci = c.get("auroc_ci", {})
        checks.append(
            bool(
                c.get("valid")
                and c.get("auroc") is not None
                and c["auroc"] >= 0.70
                and ci.get("interval")
                and ci["interval"][0] > 0.5
                and ci.get("invalid_fraction", 1) <= 0.05
            )
        )
    return {"passed": len(checks) == 3 and all(checks), "directions_passed": checks}
