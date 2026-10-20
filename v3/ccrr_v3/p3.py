"""Ordinary-gray exact-source baseline. No model fitting or privacy certificate."""

import numpy as np
import torch

from ccrr.data.pad import PADData
from ccrr.tensor_ops import gray
from ccrr.evaluation.metrics import cluster_bootstrap
from .evaluation import expected_retrieval
from .runtime import write, log


def ncc_vector(image, valid=None):
    """Public canvas by default; source support is an explicit auxiliary input."""
    if valid is None:
        return (image - image.mean()).flatten()
    if valid.shape != image.shape or not torch.isfinite(valid).all():
        raise ValueError("Invalid source support")
    if (valid < 0).any() or (valid > 1).any() or valid.sum() <= 0:
        raise ValueError("Invalid source support")
    return ((image - (image * valid).sum() / valid.sum()) * valid).flatten()


def source_match(vectors, ids, patients, repetitions=0, source_support_known=False):
    """Self-match of one deterministic ordinary-gray transform, NOT generated P3."""
    if len(ids) != len(set(ids)) or len(vectors) != len(ids):
        raise ValueError("Invalid source gallery")
    if not torch.isfinite(vectors).all() or (vectors.norm(dim=1) <= 1e-8).any():
        raise ValueError("NCC undefined for constant/empty grayscale")
    normalized = torch.nn.functional.normalize(vectors, dim=1)
    scores = (normalized @ normalized.T).cpu().numpy()
    queries = []
    # Same public deterministic grayscale operation on query and original gallery.
    # Self-score is exactly 1 mathematically; avoid roundoff placing another image above self.
    scores = np.clip(scores, -1, 1)
    scores[np.isclose(scores, 1.0, rtol=0, atol=1e-6)] = 1.0
    np.fill_diagonal(scores, 1.0)
    for i, name in enumerate(ids):
        labels = np.arange(len(ids)) == i
        rank1, ap = expected_retrieval(scores[i], labels)
        queries.append(
            dict(
                query=name,
                patient=patients[i],
                rank1=rank1,
                ap=ap,
                chance_rank1=1 / len(ids),
            )
        )
    from collections import Counter

    counts = Counter(patients)
    weights = np.array([1 / counts[p] for p in patients])
    ranks = [q["rank1"] for q in queries]
    result = dict(
        rank1=float(np.average(ranks, weights=weights)),
        chance_rank1=1 / len(ids),
        queries=queries,
        gallery=ids,
        role="val_tune",
        training_started=False,
        query="ordinary_gray",
        gallery_transform="same_ordinary_gray",
        comparison=(
            "normalized_centered_source_valid_pixels_on_fixed_512_canvas"
            if source_support_known
            else "normalized_centered_full_public_512_canvas"
        ),
        information_condition=(
            "source_support_known_auxiliary"
            if source_support_known
            else "public_pixels_only"
        ),
        source_support_used=source_support_known,
        generated_query_evaluated=False,
        exact_source_in_gallery=True,
        scope="source_image_not_cross_lesion_patient_identity",
        near_unit_ncc_tie_tolerance=1e-6,
        limitation="high baseline matching is not an impossibility proof for learned grayscale",
    )
    if repetitions:
        result["rank1_ci"] = cluster_bootstrap(
            patients, lambda w: np.average(ranks, weights=weights * w), repetitions
        )
        result["ci_scope"] = "query_patients_given_fixed_gallery"
    return result


def run(resources, cfg, output):
    data = PADData(resources["data_root"], "val_tune")
    vectors, assisted = [], []
    for i in range(len(data)):
        row = data[i]
        g, v = gray(row["x"][None])[0], row["v"]
        vectors.append(ncc_vector(g))
        assisted.append(ncc_vector(g, v))
    report = source_match(
        torch.stack(vectors),
        [r["img_id"] for r in data.rows],
        [r["patient_id"] for r in data.rows],
        cfg["bootstrap_repetitions"],
    )
    write(output / "p3_baseline.json", report)
    auxiliary = source_match(
        torch.stack(assisted),
        [r["img_id"] for r in data.rows],
        [r["patient_id"] for r in data.rows],
        cfg["bootstrap_repetitions"],
        source_support_known=True,
    )
    write(output / "p3_source_support_auxiliary.json", auxiliary)
    log(
        output,
        "ordinary_gray_p3",
        images=len(data),
        rank1=report["rank1"],
        auxiliary_rank1=auxiliary["rank1"],
        model_parameters_updated=False,
    )
