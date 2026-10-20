"""Direct held-out linkage reconstruction diagnostics; no attack fitting."""

import torch
from torch.nn import functional as F

from .data import batch
from .evaluation import DIRECTIONS, p2_report
from .runtime import write


def embedding_report(data, embeddings, teacher, repetitions):
    norms = embeddings.norm(dim=2)
    if not torch.isfinite(embeddings).all():
        raise ValueError("Nonfinite reconstructed embedding")
    unit = F.normalize(embeddings, dim=2, eps=1e-8)
    reference = F.normalize(teacher, dim=2, eps=1e-8)
    result = {
        "normalized_squared_error": {
            name: float((unit[:, d] - reference[:, d]).square().sum(1).mean())
            for d, name in enumerate(("rgb", "gray"))
        },
        "near_zero_norm_count": {
            name: int((norms[:, d] < 1e-8).sum())
            for d, name in enumerate(("rgb", "gray"))
        },
        "p2": {},
    }
    for name, (da, db) in DIRECTIONS.items():
        result["p2"][name] = p2_report(
            data["rows"],
            data["protocols"]["p2"],
            lambda a, bs: (unit[bs, db] @ unit[a, da]).tolist(),
            repetitions=repetitions,
        )
    return result


def run(cache, targets, model, cfg, output, device):
    if model.variant == "compression":
        raise ValueError("Compression has no linkage decoder")
    if set(cache) != {"fit", "val_tune", "val_select"} or set(targets) != set(cache):
        raise ValueError("Link audit requires fit and both development roles")
    model.eval().requires_grad_(False)
    # Domain-specific means of unit teacher embeddings, computed on fit ONLY.
    means = F.normalize(targets["fit"], dim=2, eps=1e-8).mean(0, keepdim=True)
    report = dict(
        schema="ccrr-v3-link-audit-1",
        training_started=False,
        model_parameters_updated=False,
        scientific_acceptance="not_evaluated",
        scope="held_out_patients_development_not_independent_final_test",
        score="cosine_fixed_positive_orientation_no_eval_selection",
        normalization_epsilon=1e-8,
        mean_baseline="domain_specific_fit_mean_of_unit_teacher_then_normalize",
        near_zero_policy="report_count_normalize_with_epsilon_ties_are_not_success",
        error_scope="image_mean_normalized_squared_distance_to_teacher",
        roles={},
    )
    with torch.no_grad():
        for role in ("val_tune", "val_select"):
            data = cache[role]
            domains = []
            for d in (0, 1):
                parts = []
                for start in range(0, len(data["rows"]), cfg["batch_size"]):
                    x, v, _ = batch(
                        data, slice(start, start + cfg["batch_size"]), d, device
                    )
                    parts.append(model(x, v, stochastic=False)["embedding"].cpu())
                domains.append(torch.cat(parts))
            reconstructed = torch.stack(domains, dim=1)
            teacher = targets[role]
            if reconstructed.shape != teacher.shape:
                raise ValueError("Reconstructed and reference embedding shapes differ")
            reps = cfg["bootstrap_repetitions"] if role == "val_select" else 0
            report["roles"][role] = {
                name: embedding_report(data, values, teacher, reps)
                for name, values in (
                    ("teacher", teacher),
                    ("reconstructed", reconstructed),
                    ("fit_mean", means.expand(len(data["rows"]), -1, -1)),
                )
            }
    write(output / "link_audit.json", report)
    return report
