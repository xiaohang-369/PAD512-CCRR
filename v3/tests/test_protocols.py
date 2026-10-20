from copy import deepcopy

import numpy as np
import pytest
import torch

from ccrr_v3.pairs import P2Sampler, validate_p2_protocol
from ccrr_v3.evaluation import p2_report, expected_retrieval
from ccrr_v3.comparison import identity_delta, clinical_delta
from ccrr_v3.p3 import source_match
from ccrr_v3.reid import p2_triplet


def test_sampler_works_without_any_p1_and_balances_pairs(cache):
    rows = cache["fit"]["rows"]
    sampler = P2Sampler(rows)
    for _ in range(20):
        pairs = sampler.sample(np.random.default_rng(_))
        assert sum(p[2] for p in pairs) == 8 and len(pairs) == 16
        for a, b, label, _ in pairs:
            assert a != b
            if label:
                assert rows[a]["patient_id"] == rows[b]["patient_id"]
                assert rows[a]["lesion_id"] == rows[b]["lesion_id"]
            else:
                assert rows[a]["patient_id"] != rows[b]["patient_id"]


def test_sampler_does_not_make_same_patient_other_lesion_negative(cache):
    rows = deepcopy(cache["fit"]["rows"])
    rows.append({**rows[0], "img_id": "other", "lesion_id": "other-lesion"})
    sampler = P2Sampler(rows)
    for i in range(25):
        for a, b, label, _ in sampler.sample(np.random.default_rng(i)):
            if not label:
                assert rows[a]["patient_id"] != rows[b]["patient_id"]


@pytest.mark.parametrize(
    "mutation", ["self", "other_lesion", "same_patient_negative", "duplicate"]
)
def test_protocol_rejects_p3_or_invalid_p2(cache, mutation):
    data = deepcopy(cache["fit"])
    q = data["protocols"]["p2"]["queries"][0]
    if mutation == "self":
        q["positive_gallery"] = [q["query"]]
    elif mutation == "other_lesion":
        data["rows"][1]["lesion_id"] = "new"
    elif mutation == "same_patient_negative":
        q["negative_gallery"].append(q["query"])
    else:
        q["negative_gallery"].append(q["negative_gallery"][0])
    with pytest.raises(ValueError):
        validate_p2_protocol(data["protocols"]["p2"], data["rows"])


def test_ties_are_not_resolved_by_image_ids(cache):
    data = cache["val_tune"]
    report = p2_report(
        data["rows"], data["protocols"]["p2"], lambda a, bs: [0.0] * len(bs)
    )
    assert report["auroc"] == 0.5
    assert report["rank1"] == pytest.approx(report["chance_rank1"])
    assert expected_retrieval([1, 1], [1, 0]) == pytest.approx((0.5, 0.75))


def test_fixed_orientation_is_not_flipped_on_eval(cache):
    data = cache["val_tune"]

    def score(a, bs):
        return [
            float(data["rows"][b]["patient_id"] != data["rows"][a]["patient_id"])
            for b in bs
        ]

    wrong = p2_report(data["rows"], data["protocols"]["p2"], score, orientation=1)
    right = p2_report(data["rows"], data["protocols"]["p2"], score, orientation=-1)
    assert wrong["auroc"] == 0 and wrong["diagnostic_two_sided_auc"] == 1
    assert right["auroc"] == 1


def test_matched_p2_comparison_rejects_changed_gallery(cache):
    data = cache["val_tune"]
    a = p2_report(data["rows"], data["protocols"]["p2"], lambda a, bs: [0.0] * len(bs))
    b = deepcopy(a)
    same = identity_delta(a, b, 8)
    assert same["rank1_before_minus_after"] == 0 and same["auc_before_minus_after"] == 0
    b["queries"][0]["gallery"].reverse()
    with pytest.raises(ValueError, match="identical"):
        identity_delta(a, b, 8)


def test_clinical_comparison_rejects_reordering():
    a = dict(ids=["a", "b"], patients=["p1", "p2"], labels=[0, 1])
    b = {**a, "ids": ["b", "a"]}
    with pytest.raises(ValueError, match="cohort"):
        clinical_delta(a, b, 8)


def test_p3_is_exact_source_and_constant_rejected():
    report = source_match(torch.eye(4), list("abcd"), list("pqrs"))
    assert report["rank1"] == 1 and report["exact_source_in_gallery"]
    assert report["training_started"] is False
    with pytest.raises(ValueError, match="NCC"):
        source_match(torch.zeros(2, 3), ["a", "b"], ["p", "q"])


def test_p3_identical_sources_are_ties_not_perfect_identification():
    report = source_match(torch.ones(4, 3), list("abcd"), list("pqrs"))
    assert report["rank1"] == pytest.approx(0.25)


def test_triplet_excludes_same_patient_other_lesion():
    e = torch.tensor(
        [[0.0, 0.0], [0.0, 0.1], [100.0, 100.0], [5.0, 5.0]], requires_grad=True
    )
    value = p2_triplet(e, torch.tensor([0, 0, 0, 1]), torch.tensor([0, 0, 1, 2]))
    assert (
        value.item() == 0
    )  # distant same-patient other lesion is neither positive nor negative
    value.backward()
    assert torch.isfinite(e.grad).all()
