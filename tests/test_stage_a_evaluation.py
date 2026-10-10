import copy
import numpy as np
import pytest
import torch
from ccrr.evaluation.metrics import clinical
from ccrr.evaluation.teachers import (
    clinical_intervals,
    clinical_gate,
    paired_gray_change,
    teacher_identity_scores,
    reid_gate,
    PatchProbe,
)
from ccrr.evaluation.comparisons import threshold_at_fpr


def report():
    y = np.tile(np.arange(6), 3)
    p = np.eye(6)[y] * 0.9 + 0.1 / 6
    return {
        **clinical(y, p),
        "labels": y.tolist(),
        "probabilities": p.tolist(),
        "patients": [str(i) for i in range(len(y))],
        "ids": list(range(len(y))),
    }


def test_clinical_complete_report_patient_denominators_and_identical_gray():
    r = clinical_intervals(report(), 40)
    assert len(r["confusion_matrix"]) == 6 and r["patient_n"] == 18
    assert all(
        c["images"] == 3 and c["patients"] == 3 and c["negative_images"] == 15
        for c in r["per_class"]
    )
    assert r["macro_auroc_ci"]["replicates"] == 40
    assert r["macro_auroc_ci"]["lower"] == 1
    delta = paired_gray_change([r, r], 40)
    assert delta["gray_minus_rgb"] == delta["ci"]["lower"] == delta["ci"]["upper"] == 0
    assert clinical_gate([r, r], [r, r])["passed"]


def test_missing_classes_defects_domain_control_and_thresholds_not_relaxed():
    r = report()
    r["macro_auroc"] = 0.75
    assert clinical_gate([r, r])["passed"]
    bad = copy.deepcopy(r)
    bad["per_class_auroc"][2] = 0.5
    assert not clinical_gate([r, bad])["passed"]
    bad = copy.deepcopy(r)
    bad["recall"][2] = 0
    assert not clinical_gate([r, bad])["passed"]
    bad = copy.deepcopy(r)
    bad["macro_auroc"] = None
    assert not clinical_gate([r, bad])["passed"]
    control = copy.deepcopy(r)
    control["macro_auroc"] = 0.781
    assert not clinical_gate([r, r], [r, control])["passed"]


def test_patient_bootstrap_reports_non_estimable_classes():
    r = report()
    r["patients"] = ["single"] * 18
    a = clinical_intervals(r, 30)
    assert a["patient_n"] == 1 and a["macro_auroc_ci"]["lower"] == 1
    r = report()
    r["labels"] = [0] * 18
    r.update(clinical(r["labels"], r["probabilities"]))
    a = clinical_intervals(r, 30)
    assert a["macro_auroc"] is None and not a["six_classes_estimable"]
    assert a["macro_auroc_ci"]["invalid_fraction"] == 1
    assert not a["macro_auroc_ci"]["evidence_sufficient"]


def identity_fixture():
    rows = [
        {
            "img_id": str(i),
            "patient_id": str(i // 2),
            "lesion_id": str(i % 2),
            "diagnostic": "ACK" if i < 4 else "BCC",
            "region": "ARM",
        }
        for i in range(6)
    ]
    features = {str(i): {"embedding": torch.eye(3)[i // 2][None]} for i in range(6)}
    protocol = {
        "gallery": list(features),
        "queries": [
            {
                "query": str(i),
                "patient_id": str(i // 2),
                "positive_gallery": [str(i ^ 1)],
                "negative_gallery": [str(j) for j in range(6) if j // 2 != i // 2],
            }
            for i in range(6)
        ],
    }
    return rows, features, protocol


def test_identity_patient_weights_matched_subset_and_gallery_fixed_intervals():
    rows, features, protocol = identity_fixture()
    r = teacher_identity_scores(
        protocol,
        rows,
        features,
        features,
        bootstrap=True,
        threshold=0.5,
        repetitions=40,
    )["cosine"]
    assert r["auroc"] == r["rank1"] == r["map"] == r["tpr_at_fixed_threshold"] == 1
    assert r["fpr_at_fixed_threshold"] == 0
    assert (
        r["queries"] == 6
        and r["patients"] == 3
        and r["chance_rank1"] == pytest.approx(0.2)
    )
    d = r["score_details"]
    w = np.array(d["weights"])
    y = np.array(d["labels"])
    assert w[y == 1].sum() == w[y == 0].sum() == 3
    assert r["diagnosis_region_matched"]["queries_without_matched_negatives"] == 2
    assert r["diagnosis_region_matched"]["auroc"] == 1
    assert r["per_diagnosis"]["MEL"]["valid"] is False
    assert r["rank1_ci"]["replicates"] == 40
    assert threshold_at_fpr(d["labels"], d["scores"], d["weights"]) == 1


def test_matched_negatives_are_per_positive_not_union():
    rows, features, p = identity_fixture()
    # Query has two positive members with distinct diagnoses; each gets its own negative set.
    rows[2]["patient_id"] = "0"
    rows[2]["diagnostic"] = "BCC"
    p["queries"] = [
        {
            "query": "0",
            "patient_id": "0",
            "positive_gallery": ["1", "2"],
            "negative_gallery": ["3", "4", "5"],
        }
    ]
    r = teacher_identity_scores(p, rows, features, features)["cosine"][
        "diagnosis_region_matched"
    ]["score_details"]
    assert sorted(w for w, y in zip(r["weights"], r["labels"]) if y == 0) == [
        0.25,
        0.25,
        0.5,
    ]
    assert sum(w for w, y in zip(r["weights"], r["labels"]) if y == 1) == 1


def test_p2_success_cannot_establish_p1_and_ci_boundary():
    r = {
        "cosine": {
            "valid": True,
            "auroc": 0.7,
            "auroc_ci": {"interval": [0.51, 0.9], "invalid_fraction": 0.05},
        }
    }
    assert reid_gate([r] * 3)["passed"]
    bad = copy.deepcopy(r)
    bad["cosine"]["auroc_ci"]["interval"][0] = 0.5
    assert not reid_gate([r, r, bad])["passed"]
    assert not reid_gate([r, r])["passed"]
    bad = copy.deepcopy(r)
    bad["cosine"]["auroc"] = 0.69999
    assert not reid_gate([bad] * 3)["passed"]


def test_patch_probe_uses_coverage_weighted_grid_and_backprop():
    class Grid(torch.nn.Module):
        def forward(self, x, rects):
            return x, x.new_tensor([[[[1.0, 0.0]]]]), None

    head = torch.nn.Linear(768, 6)
    probe = PatchProbe(Grid(), head)
    x = torch.randn(1, 768, 1, 2, requires_grad=True)
    out = probe(x, None)
    assert torch.equal(out, head(x[:, :, :, 0].squeeze(-1)))
    out.sum().backward()
    assert (
        torch.count_nonzero(x.grad[:, :, :, 1]) == 0
        and x.grad[:, :, :, 0].abs().sum() > 0
    )
