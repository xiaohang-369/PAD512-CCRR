import copy
import json
from pathlib import Path
import pytest
from ccrr.data.admission import verify_pairs, verify_source_row, verify_dataset
from ccrr.codec.metadata import FIELDS
from ccrr.errors import CCRRError


def fixture(scenario):
    rows = [
        {
            "img_id": str(i),
            "patient_id": str(i // 4),
            "lesion_id": str(i % 4 // 2),
            "diagnostic": "ACK",
            "region": "ARM",
            "reid_observation_eligible": "1",
        }
        for i in range(8)
    ]
    p = {"gallery": [r["img_id"] for r in rows], "queries": []}
    for r in rows:
        pos = [
            s["img_id"]
            for s in rows
            if s["img_id"] != r["img_id"]
            and s["patient_id"] == r["patient_id"]
            and (
                (s["lesion_id"] != r["lesion_id"])
                if scenario == "p1"
                else s["lesion_id"] == r["lesion_id"]
            )
        ]
        neg = [s["img_id"] for s in rows if s["patient_id"] != r["patient_id"]]
        p["queries"].append(
            {
                "query": r["img_id"],
                "patient_id": r["patient_id"],
                "positive_gallery": pos,
                "negative_gallery": neg,
                "diagnosis_region_matched_negative_union": neg,
            }
        )
    return rows, p


@pytest.mark.parametrize("scenario", ["p1", "p2"])
def test_pair_membership_recomputed_not_trusted(scenario):
    rows, p = fixture(scenario)
    assert verify_pairs(rows, p, scenario) == {
        "gallery": 8,
        "queries": 8,
        "patients": 2,
    }
    for field in [
        "positive_gallery",
        "negative_gallery",
        "diagnosis_region_matched_negative_union",
    ]:
        wrong = copy.deepcopy(p)
        wrong["queries"][0][field].append("0")
        with pytest.raises(CCRRError):
            verify_pairs(rows, wrong, scenario)
    wrong = copy.deepcopy(p)
    wrong["gallery"] = wrong["gallery"][:-1]
    with pytest.raises(CCRRError):
        verify_pairs(rows, wrong, scenario)
    wrong = copy.deepcopy(p)
    wrong["queries"].append(wrong["queries"][0])
    with pytest.raises(CCRRError):
        verify_pairs(rows, wrong, scenario)


def test_source_26_fields_and_role_patient_must_agree():
    source = {k: "" for k in FIELDS}
    source.update(
        img_id="synthetic.png", patient_id="p0", lesion_id="l0", diagnostic="ACK"
    )
    cleaned = {**source, "source_relpath": "synthetic.png"}
    row = {
        k: cleaned[k]
        for k in ("img_id", "patient_id", "lesion_id", "diagnostic", "source_relpath")
    }
    verify_source_row(row, source, cleaned)
    for changed in ("age", "diagnostic", "patient_id"):
        wrong = dict(cleaned)
        wrong[changed] = "changed"
        with pytest.raises(CCRRError):
            verify_source_row(row, source, wrong)
    wrong = dict(row)
    wrong["patient_id"] = "other"
    with pytest.raises(CCRRError):
        verify_source_row(wrong, source, cleaned)


def test_missing_or_replaced_frozen_evidence_fails(tmp_path):
    with pytest.raises(CCRRError, match="Frozen evidence"):
        verify_dataset(tmp_path)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/metadata.csv").write_text("replacement")
    with pytest.raises(CCRRError, match="Frozen evidence"):
        verify_dataset(tmp_path)


def test_shipped_binding_has_hashes_not_patient_records():
    import ccrr.data.admission as module

    m = json.loads(Path(module.__file__).with_name("frozen_evidence.json").read_text())
    assert len(m["files"]) == 29 and all(len(v) == 64 for v in m["files"].values())
    assert sum(r["images"] for r in m["counts"].values()) == 2264
    assert "patient_id" not in json.dumps(m)
