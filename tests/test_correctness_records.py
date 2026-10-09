"""§22.3/23.5 parser tests; supplied logits are digital fixtures, not a trained Decoder."""

import itertools
import pytest
import torch
from ccrr.communication.records import logits_to_records, split_stream
from ccrr.errors import Status
from test_integration import fixture_system


def row_logits(record):
    bits = torch.tensor(record.bits())
    row = torch.full((177,), -20.0)
    row[0] = 20
    row[1 + record.version] = 20
    row[5 + record.alpha] = 20
    row[7 + record.level] = 20
    row[11:] = (bits[5:] * 2 - 1) * 20
    return row


@pytest.mark.parametrize("pattern", list(itertools.product((0, 1), repeat=4)))
@pytest.mark.parametrize("alpha", (0, 1))
def test_all_occupancy_patterns(pattern, alpha):
    source = split_stream(bytes(range(64)), alpha, 1 - alpha)
    logits = torch.stack([row_logits(r) for r in source])[None]
    logits[0, :, 0] = torch.tensor(pattern) * 40.0 - 20
    records, errors = logits_to_records(logits, 0.5)
    prefix = next((i for i, value in enumerate(pattern) if not value), 4)
    hole = any(pattern[prefix:])
    assert bool(errors[alpha]) == hole
    assert not errors[1 - alpha]
    assert (
        records == source[:prefix]
    )  # Never compress holes or infer an empty row's type.


@pytest.mark.parametrize("affected", (0, 1))
def test_hole_failure_preserves_other_authenticated_capability(tmp_path, affected):
    system, allocator, kc, km, records, metadata = fixture_system(tmp_path)
    try:
        # Every complete stream record is present. The extra post-hole row must
        # invalidate its own stream even though it would otherwise authenticate.
        logits = torch.full((len(records) + 1, 4, 177), -20.0)
        for i, record in enumerate(records):
            logits[i, 0] = row_logits(record)
        logits[-1, 1] = row_logits(next(r for r in records if r.alpha == affected))
        parsed, errors = logits_to_records(logits, 0.5)
        assert parsed == records
        with torch.no_grad():
            result = system.receive_records(
                torch.full((1, 1, 256, 256), 128 / 255), parsed, errors, {0: kc, 1: km}
            )
        assert result[("C", "M")[affected]].status == Status.EXTRACTION
        assert result[("M", "C")[affected]].status == Status.SUCCESS
        if affected == 0:
            assert result["M"].value == metadata
    finally:
        allocator.close()


def test_mixed_post_hole_types_fail_both_without_reading_empty_fields():
    c = split_stream(bytes(16), 0, 1)[0]
    m = split_stream(bytes(16), 1, 0)[0]
    logits = torch.full((1, 4, 177), -20.0)
    logits[0, 1], logits[0, 2] = row_logits(c), row_logits(m)
    records, errors = logits_to_records(logits, 0.5)
    assert not records and errors[0] and errors[1]
