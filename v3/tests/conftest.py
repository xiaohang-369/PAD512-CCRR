from copy import deepcopy

import pytest
import torch

from ccrr_v3.runtime import REPO, read

torch.set_num_threads(1)


@pytest.fixture
def cfg():
    obj = read(REPO / "v3/configs/development.example.json")
    obj.update(
        registration_status="registered",
        registration_note="synthetic engineering test only",
        warmup_steps=1,
        joint_steps=2,
        probe_steps=2,
        attack_steps=2,
        attack_warmup_steps=1,
        reid_steps=2,
        evaluate_every=1,
        width=8,
        code_channels=4,
        bootstrap_repetitions=8,
    )
    return obj


def cohort(prefix, channels=8):
    classes = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
    rows = [
        dict(
            img_id=f"{prefix}-{c}-{v}",
            patient_id=f"{prefix}-p{c}",
            lesion_id=f"l{c}",
            diagnostic=classes[c],
            region="arm",
            reid_observation_eligible="1",
        )
        for c in range(6)
        for v in range(2)
    ]
    protocol = {"queries": []}
    for row in rows:
        protocol["queries"].append(
            dict(
                query=row["img_id"],
                patient_id=row["patient_id"],
                positive_gallery=[
                    r["img_id"]
                    for r in rows
                    if r["patient_id"] == row["patient_id"] and r != row
                ],
                negative_gallery=[
                    r["img_id"] for r in rows if r["patient_id"] != row["patient_id"]
                ],
            )
        )
    return dict(
        rows=rows,
        features=torch.randn(len(rows), 2, channels, 4, 4),
        coverage=torch.ones(len(rows), 1, 4, 4),
        labels=torch.arange(6).repeat_interleave(2),
        protocols={"p2": protocol},
    )


@pytest.fixture
def cache():
    return {role: deepcopy(cohort(role)) for role in ("fit", "val_tune", "val_select")}
