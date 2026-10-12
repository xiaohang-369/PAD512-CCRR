"""Section 7.7 branch measurements and section 26.3 disposable adaptation.

These are diagnostics, not substitutes for independent feature acceptance.
Neither collapse nor numerical pass is inferred from training loss alone.
"""

import copy
from contextlib import contextmanager
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ccrr.errors import require
from ccrr.runtime.checkpoints import save_checkpoint
from ccrr.runtime.support import write_json
from ccrr.tensor_ops import masked_mean, pooled
from ccrr.training.optimization import optimizer


@contextmanager
def isolated_rng():
    """A disposable diagnostic must not advance any production random stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@torch.no_grad()
def branch_statistics(p, coverage, identity_target):
    """One image; channel means, fractional coverage, no padded-token evidence."""
    require(
        coverage.shape[0] == 1 and coverage.sum() > 0, "Invalid diagnostic coverage"
    )
    values = {
        "code_saturation_fraction": masked_mean((p["q"].abs() == 7).float(), coverage),
        "full_code_reconstruction_mse": masked_mean(
            (p["all"] - p["standardized"]).square(), coverage
        ),
        "identity_reconstruction_squared_l2": (p["identity"] - identity_target)
        .square()
        .sum(1)
        .mean(),
    }
    for name, key in (("raw", "standardized"), ("semantic", "semantic")):
        center = pooled(p[key], coverage)[:, :, None, None]
        values[name + "_spatial_variance_channel_mean"] = masked_mean(
            (p[key] - center).square(), coverage
        )
    require(
        all(torch.isfinite(x).all() for x in values.values()),
        "Nonfinite branch diagnostic",
    )
    return {key: float(value) for key, value in values.items()}


def compare_attack_scores(before, after):
    """Keep ordinary AUROC and reversible A-dagger distinct; no hidden cutoff."""
    require(len(before) == len(after) == 9, "Expected all nine attack directions")
    results = []
    for pre, post in zip(before, after):
        for key in ("domain", "labels", "pairs", "weights"):
            require(pre[key] == post[key], "Adapted attack pair alignment mismatch")
        y, weights = pre["labels"], pre["weights"]
        a, b = np.asarray(pre["scores"]), np.asarray(post["scores"])
        require(a.shape == b.shape == (len(y),), "Adapted attack score shape mismatch")
        require(
            np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite adapted scores"
        )
        pre_auc = float(roc_auc_score(y, a, sample_weight=weights))
        post_auc = float(roc_auc_score(y, b, sample_weight=weights))
        require(
            np.isfinite([pre_auc, post_auc]).all(), "Undefined adapted attack AUROC"
        )
        collapsed = bool(np.ptp(a) == 0 or np.ptp(b) == 0)
        reversed_score = pre_auc < 0.5 or post_auc < 0.5
        delta = post_auc - pre_auc
        results.append(
            {
                "domain": pre["domain"],
                "alternating_auroc": pre_auc,
                "adapted_auroc": post_auc,
                "alternating_a_dagger": max(pre_auc, 1 - pre_auc),
                "adapted_a_dagger": max(post_auc, 1 - post_auc),
                "auroc_increase": delta,
                "a_dagger_increase": max(post_auc, 1 - post_auc)
                - max(pre_auc, 1 - pre_auc),
                "alternating_logit_std": float(a.std()),
                "adapted_logit_std": float(b.std()),
                "exact_constant_output": collapsed,
                "reverse_scoring_observed": reversed_score,
                "alternating_attack_insufficient": bool(
                    delta >= 0.05 or collapsed or reversed_score
                ),
            }
        )
    return results


def post_adapted_attack(audit):
    """Copy the matching endpoint attacker; 1,000 fit updates, no val selection.

    A fresh AdamW uses the existing peak LR / 1,000-step schedule. The mixed
    P1/P2 sampler and three independent endpoint domains match B training.
    Current-endpoint interruption replays this diagnostic with the audit;
    its checkpoint is an audit artifact, never a generator-resume checkpoint.
    """
    e = audit.e
    # Keep diagnostic fallback counters off the production sampler as well.
    audit = copy.copy(audit)
    audit.e = copy.copy(e)
    if hasattr(e, "sampler"):
        audit.e.sampler = copy.copy(e.sampler)
    with isolated_rng():
        model = copy.deepcopy(e.sem_attacker).requires_grad_(True).eval()
        opt = optimizer([model], [1e-4])
        print(
            "B_POST_ADAPT_START: matching endpoint attacker, 1000 fit updates",
            flush=True,
        )
        before = {s: audit.attack_scores(model, "sem", s) for s in ("p1", "p2")}
        for step in range(1000):
            audit.attacker_update(
                model,
                opt,
                "sem",
                "mixed",
                "spatial",
                step,
                total=1000,
                rng_key="B-post-adapt-mixed",
            )
            if (step + 1) % 100 == 0:
                print(f"B_POST_ADAPT_STEP: {step + 1}/1000", flush=True)
        after = {s: audit.attack_scores(model, "sem", s) for s in ("p1", "p2")}
        checks = {s: compare_attack_scores(before[s], after[s]) for s in before}
        result = {
            "updates": 1000,
            "fit_only": True,
            "generator_updated": False,
            "optimizer": "fresh AdamW, lr_max=1e-4, existing 1000-step schedule",
            "selection": "final step, no validation selection",
            "collapse_definition": "exact constant logits; standard deviations also reported",
            "checks": checks,
            "alternating_attack_insufficient": any(
                r["alternating_attack_insufficient"]
                for rows in checks.values()
                for r in rows
            ),
            "replaces_independent_attack_acceptance": False,
        }
        save_checkpoint(
            audit.root / "post_adapted_attacker.pt",
            {"attacker": model},
            opt,
            1000,
            "B-diagnostic",
            result,
        )
        write_json(
            audit.root / "post_adapted_scores.json", {"before": before, "after": after}
        )
        write_json(audit.root / "post_adapted_attack.json", result)
        return result
