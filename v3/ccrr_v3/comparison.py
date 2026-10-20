"""Paired development intervals; no post-hoc acceptance thresholds."""

import numpy as np
from sklearn.metrics import roc_auc_score

from ccrr.evaluation.metrics import cluster_bootstrap
from .runtime import read, write


def clinical_delta(before, after, repetitions):
    for key in ("ids", "patients", "labels"):
        if before[key] != after[key]:
            raise ValueError("Clinical comparison cohort/order mismatch")
    y = np.asarray(before["labels"])
    a, b = np.asarray(before["probabilities"]), np.asarray(after["probabilities"])

    def statistic(w):
        return np.mean(
            [
                roc_auc_score(y == c, b[:, c], sample_weight=w)
                - roc_auc_score(y == c, a[:, c], sample_weight=w)
                for c in range(6)
            ]
        )

    return dict(
        after_minus_before=statistic(np.ones(len(y))),
        ci=cluster_bootstrap(before["patients"], statistic, repetitions),
    )


def identity_delta(before, after, repetitions):
    aq, bq = before["queries"], after["queries"]
    if len(aq) != len(bq):
        raise ValueError("Query count differs")
    patients, ys, sa, sb, pairweights, pairpatients = [], [], [], [], [], []
    for a, b in zip(aq, bq):
        for key in ("query", "patient", "gallery", "labels", "weight"):
            if a[key] != b[key]:
                raise ValueError(
                    "P2 comparison requires identical queries and galleries"
                )
        patients.append(a["patient"])
        ys.extend(a["labels"])
        sa.extend(a["scores"])
        sb.extend(b["scores"])
        for label in a["labels"]:
            pairweights.append(a["weight"] / a["labels"].count(label))
            pairpatients.append(a["patient"])
    query_weights = np.array([q["weight"] for q in aq])
    difference = np.array([a["rank1"] - b["rank1"] for a, b in zip(aq, bq)])
    weights = np.array(pairweights)

    def delta_auc(w):
        return roc_auc_score(ys, sa, sample_weight=weights * w) - roc_auc_score(
            ys, sb, sample_weight=weights * w
        )

    return dict(
        rank1_before_minus_after=float(np.average(difference, weights=query_weights)),
        rank1_ci=cluster_bootstrap(
            patients,
            lambda w: np.average(difference, weights=query_weights * w),
            repetitions,
        ),
        auc_before_minus_after=float(delta_auc(np.ones(len(ys)))),
        auc_ci=cluster_bootstrap(pairpatients, delta_auc, repetitions),
        scope="query_patient_uncertainty_given_fixed_gallery_and_fitted_attackers",
    )


def compare(before, after, output, repetitions):
    result = {
        "clinical": {},
        "p2": {},
        "scientific_acceptance": "not_evaluated",
        "limitations": [
            "not a multi-seed result",
            "no automatic clinical equivalence",
            "P3/lesions/concepts/final_public_image_not_evaluated",
        ],
    }
    for kind in ("linear", "mlp"):
        a, b = read(before / f"{kind}.json"), read(after / f"{kind}.json")
        result["clinical"][kind] = {
            d: clinical_delta(a[d], b[d], repetitions) for d in ("rgb", "gray")
        }
    for kind in ("global", "spatial"):
        a, b = read(before / f"{kind}.json"), read(after / f"{kind}.json")
        result["p2"][kind] = {d: identity_delta(a[d], b[d], repetitions) for d in a}
    write(output / "comparison.json", result)
