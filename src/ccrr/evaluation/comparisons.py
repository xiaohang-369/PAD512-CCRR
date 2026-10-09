import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from ccrr.evaluation.metrics import cluster_bootstrap


def paired_clinical(reference, public, auroc_margin=0.03, f1_margin=0.05):
    ref = {name: i for i, name in enumerate(reference["ids"])}
    ids = [n for n in public["ids"] if n in ref]
    pub = {name: i for i, name in enumerate(public["ids"])}
    y = np.array([public["labels"][pub[n]] for n in ids])
    a = np.array([reference["probabilities"][ref[n]] for n in ids])
    b = np.array([public["probabilities"][pub[n]] for n in ids])
    patients = [public["patients"][pub[n]] for n in ids]
    if not ids:
        return {
            "passed_point": False,
            "statistical_support": False,
            "reason": "No matched published cohort",
        }

    def auc(p, w):
        if any(len(np.unique((y == c)[w > 0])) < 2 for c in range(6)):
            return np.nan
        return np.mean(
            [roc_auc_score(y == c, p[:, c], sample_weight=w) for c in range(6)]
        )

    def f1(p, w):
        return f1_score(
            y,
            p.argmax(1),
            labels=range(6),
            sample_weight=w,
            average="macro",
            zero_division=0,
        )

    w = np.ones(len(y))
    da = auc(b, w) - auc(a, w)
    df = f1(b, w) - f1(a, w)
    aci = cluster_bootstrap(patients, lambda w: auc(b, w) - auc(a, w))
    fci = cluster_bootstrap(patients, lambda w: f1(b, w) - f1(a, w))
    point = bool(np.isfinite(da) and da >= -auroc_margin and df >= -f1_margin)
    supported = bool(
        point
        and aci["evidence_sufficient"]
        and fci["evidence_sufficient"]
        and aci["lower"] is not None
        and aci["lower"] >= -auroc_margin
        and fci["lower"] >= -f1_margin
    )
    return {
        "n": len(y),
        "auroc_difference": da,
        "f1_difference": df,
        "auroc_ci": aci,
        "f1_ci": fci,
        "passed_point": point,
        "statistical_support": supported,
    }


def privacy_points(rgb, gray, public):
    reports = []
    for mode in ("cosine", "pair"):
        r, y, p = rgb.get(mode, {}), gray.get(mode, {}), public.get(mode, {})
        if not all(x.get("valid") for x in (r, y, p)):
            reports.append(
                {"mode": mode, "passed": False, "reason": "Missing legal queries"}
            )
            continue
        ci = r.get("auroc_ci", {})
        lower = ci.get("interval", [None, None])
        effective = bool(
            lower
            and lower[0] is not None
            and lower[0] > 0.5
            and ci.get("invalid_fraction", 1) <= 0.05
            and y["a_dagger"] >= 0.60
        )
        auc_ok = p["a_dagger"] <= 0.60 and p["a_dagger"] - 0.5 <= 0.5 * (
            y["a_dagger"] - 0.5
        )
        c = p["chance_rank1"]
        rank_ok = (
            y["rank1"] - c >= 0.10
            and p["rank1"] <= c + 0.10
            and p["rank1"] - c <= 0.5 * (y["rank1"] - c)
        )
        reports.append(
            {
                "mode": mode,
                "effective_reference": effective,
                "passed_auc": auc_ok,
                "passed_rank1": rank_ok,
                "passed": effective and auc_ok and rank_ok,
            }
        )
    return {"readouts": reports, "passed": all(r["passed"] for r in reports)}


def capability_benefit(authorized, attacks):
    """Each attack uses the identical C-success cohort; paired patient resampling."""
    report = {}
    for name, rows in attacks.items():
        by = {r["id"]: r for r in rows}
        common = [r for r in authorized if r["id"] in by]
        if len(common) != len(authorized) or not common:
            report[name] = {
                "statistical_support": False,
                "reason": "Incomplete common C-success cohort",
            }
            continue
        patients = [r["patient"] for r in common]
        ps = np.array(
            [r["quality"]["psnr"] - by[r["id"]]["quality"]["psnr"] for r in common]
        )
        de = np.array(
            [
                by[r["id"]]["quality"]["delta_e00_mean"]
                - r["quality"]["delta_e00_mean"]
                for r in common
            ]
        )
        pci = cluster_bootstrap(patients, lambda w: np.average(ps, weights=w))
        dci = cluster_bootstrap(patients, lambda w: np.average(de, weights=w))
        report[name] = {
            "n": len(common),
            "psnr_gain": float(np.mean(ps)),
            "delta_e00_improvement": float(np.mean(de)),
            "psnr_ci": pci,
            "delta_e00_ci": dci,
            "statistical_support": bool(
                pci["evidence_sufficient"]
                and dci["evidence_sufficient"]
                and pci["lower"] is not None
                and dci["lower"] is not None
                and pci["lower"] > 0
                and dci["lower"] > 0
            ),
        }
    return report


def threshold_at_fpr(labels, scores, weights=None, fpr=0.01):
    y = np.asarray(labels)
    s = np.asarray(scores)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    neg = y == 0
    if not neg.any():
        return None
    # Lowest threshold with <=1% empirical FPR; ties are never split.
    candidates = sorted(set(s.tolist()) | {float(np.nextafter(s.max(), np.inf))})
    return next(t for t in candidates if np.average(s[neg] >= t, weights=w[neg]) <= fpr)
