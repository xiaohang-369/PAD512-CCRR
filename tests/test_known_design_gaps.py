"""D01/D03/D04 regressions, promoted from strict xfail after correctness repair."""

import torch
from ccrr.communication.records import split_stream, logits_to_records
from ccrr.errors import CCRRError
from ccrr.training.losses import clinical_loss, js_divergence


def test_receiver_rejects_occupancy_hole():
    record = split_stream(bytes(16), 0, 1)[0]
    bits = torch.tensor(record.bits())
    row = torch.full((177,), -20.0)
    row[0] = 20
    row[1 + record.version] = 20
    row[5 + record.alpha] = 20
    row[7 + record.level] = 20
    row[11:] = (bits[5:] * 2 - 1) * 20
    logits = torch.full((1, 4, 177), -20.0)
    logits[0, 0] = row
    logits[0, 2] = row  # 1,0,1,0: cannot compress the hole.
    try:
        records, errors = logits_to_records(logits, 0.5)
    except CCRRError:
        return
    assert errors[0] or errors[1] or not records


def test_weighted_clinical_loss_is_invariant_to_microbatch_partition():
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0]])
    labels = torch.tensor([0, 1])
    weights = torch.tensor([1.0, 4.0])
    whole = clinical_loss(logits, logits, labels, weights)
    accumulated = (
        sum(
            clinical_loss(
                logits[i : i + 1],
                logits[i : i + 1],
                labels[i : i + 1],
                weights,
                effective_weight_mean=weights[labels].mean(),
            )
            for i in range(2)
        )
        / 2
    )
    assert torch.allclose(accumulated, whole)


def test_clinical_risk_js_uses_design_unit_interval():
    a = torch.tensor([[100.0, -100.0]])
    b = torch.tensor([[-100.0, 100.0]])
    assert torch.allclose(js_divergence(a, b), torch.ones(1), atol=1e-6)
