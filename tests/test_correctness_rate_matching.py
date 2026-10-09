import dataclasses
import numpy as np
import pytest
import torch
from ccrr.codec.crypto import open_stream
from ccrr.errors import CCRRError, Status
from test_integration import fixture_system


@pytest.mark.parametrize(
    "value,selected,rejected", [(32768.0, 2, [1]), (65536.0, 3, [1, 2])]
)
def test_symbol_overflow_rejects_only_affected_levels(
    tmp_path, value, selected, rejected
):
    s, allocator, kc, km, _, row = fixture_system(tmp_path)
    try:
        z = torch.zeros(1, 2, 256, 256)
        z[0, 0, 0, 0] = value
        z[0, 1, 0, 0] = -value
        original = z.clone()
        out = s.payload(z, row, (kc, km), allocator, [4] * 64)
        assert out["level"] == selected
        assert [f["level"] for f in out["rejected"]] == rejected
        assert all(f["reason"] == "symbol_range" for f in out["rejected"])
        assert torch.equal(z, original)
        plain, _ = open_stream(out["c"], kc, 0, selected, s.digest)
        n, latent = s.codec.decode(plain, selected)
        assert torch.equal(n, out["n"]) and torch.equal(latent, out["zq"])
        assert n[0, 0, 0, 0] == value / s.profile.step(selected)
    finally:
        allocator.close()


def test_all_levels_out_of_range_returns_capacity_failure(tmp_path):
    s, allocator, kc, km, _, row = fixture_system(tmp_path)
    try:
        z = torch.zeros(1, 2, 256, 256)
        z[0, 0, 0, 0] = 131072
        with pytest.raises(CCRRError) as error:
            s.payload(z, row, (kc, km), allocator, [4] * 64)
        assert error.value.status == Status.CAPACITY
        for level in (1, 2, 3):
            assert f"'level': {level}" in error.value.detail
    finally:
        allocator.close()


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("sign", [-1, 1])
def test_quantization_boundaries_do_not_clip_or_wrap(tmp_path, level, sign):
    s, allocator, *_ = fixture_system(tmp_path)
    try:
        z = torch.zeros(1, 2, 256, 256)
        bound = s.profile.symbol_bound
        step = s.profile.step(level)
        z[0, 0, 0, 0] = sign * (bound + 0.4375) * step
        assert s.codec.quantize(z, level)[0, 0, 0, 0] == sign * bound
        z[0, 0, 0, 0] = sign * (bound + 0.5) * step
        with pytest.raises(CCRRError, match="range"):
            s.codec.quantize(z, level)
    finally:
        allocator.close()


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), -float("inf"), torch.finfo(torch.float32).max]
)
def test_nonfinite_input_or_scaled_symbols_are_not_candidate_rejections(tmp_path, bad):
    s, allocator, kc, km, _, row = fixture_system(tmp_path)
    try:
        z = torch.zeros(1, 2, 256, 256)
        z[0, 0, 0, 0] = bad
        with pytest.raises(CCRRError) as error:
            s.payload(z, row, (kc, km), allocator, [4] * 64)
        assert error.value.status == Status.NUMERICAL
    finally:
        allocator.close()


def test_quality_order_and_exact_complete_stream_capacity(tmp_path):
    s, allocator, kc, km, _, row = fixture_system(tmp_path)
    try:
        z = torch.zeros(1, 2, 256, 256)
        s.profile = dataclasses.replace(s.profile, quality_order=(3, 1, 2))
        s.codec.profile = s.profile
        out = s.payload(z, row, (kc, km), allocator, [4] * 64)
        assert out["level"] == 3  # validated order, not r or shortest stream
        required = (len(out["c"]) + 15) // 16 + (len(out["m"]) + 15) // 16
        assert (
            len(out["m"]) == 408
            and len(out["c"]) == len(s.codec.encode(out["n"], 3)) + 88
        )
        budgets = np.zeros(64, dtype=int)
        for i in range(required):
            budgets[i // 4] += 1
        assert s.payload(z, row, (kc, km), allocator, budgets)["level"] == 3
        budgets[np.flatnonzero(budgets)[-1]] -= 1
        with pytest.raises(CCRRError) as error:
            s.payload(z, row, (kc, km), allocator, budgets)
        assert error.value.status == Status.CAPACITY
        # Exact encoding can also reject a legal level: uniform CDF exceeds the
        # 16384-byte stream limit; the next quality-ranked concentrated CDF fits.
        s.codec.cdfs[2] = np.tile(np.arange(513, dtype=np.int32) * 128, (128, 1))
        out = s.payload(z, row, (kc, km), allocator, [4] * 64)
        assert out["level"] == 1 and out["rejected"][0]["level"] == 3
        assert out["rejected"][0]["source_bytes"] + 88 > 16384
    finally:
        allocator.close()


@pytest.mark.parametrize("level", [0, 4, 2])
def test_source_encode_rejects_invalid_or_unregistered_level_before_cdf_access(
    tmp_path, level
):
    s, allocator, *_ = fixture_system(tmp_path)
    try:
        s.codec.profile = dataclasses.replace(
            s.profile, legal_levels=(1, 3), quality_order=(1, 3)
        )
        with pytest.raises(CCRRError) as error:
            s.codec.encode(torch.zeros(1, 2, 256, 256), level)
        assert error.value.status == Status.INPUT
    finally:
        allocator.close()


@pytest.mark.parametrize(
    "failure",
    [
        CCRRError(Status.INPUT, "protocol"),
        CCRRError(Status.DECODE, "CDF"),
        ValueError("unexpected"),
    ],
)
def test_unrelated_quantizer_exceptions_propagate(tmp_path, monkeypatch, failure):
    s, allocator, kc, km, _, row = fixture_system(tmp_path)
    calls = []

    def broken(z, level):
        calls.append(level)
        raise failure

    monkeypatch.setattr(s.codec, "quantize", broken)
    try:
        with pytest.raises(type(failure)) as error:
            s.payload(torch.zeros(1, 2, 256, 256), row, (kc, km), allocator, [4] * 64)
        assert error.value is failure and calls == [1]
    finally:
        allocator.close()


def test_actual_shape_level_and_cdf_errors_do_not_fall_through(tmp_path):
    s, allocator, kc, km, _, row = fixture_system(tmp_path)
    try:
        bad_shape = torch.full((1, 2, 8, 8), 131072.0)
        with pytest.raises(CCRRError, match="shape") as error:
            s.payload(bad_shape, row, (kc, km), allocator, [4] * 64)
        assert error.value.status == Status.INPUT
        z = torch.zeros(1, 2, 256, 256)
        with pytest.raises(CCRRError, match="level") as error:
            s.payload(z, row, (kc, km), allocator, [4] * 64, level_order=[0, 1])
        assert error.value.status == Status.INPUT
        s.codec.cdfs[0, 0, 1] = 0
        with pytest.raises(CCRRError, match="CDF") as error:
            s.payload(z, row, (kc, km), allocator, [4] * 64)
        assert error.value.status == Status.DECODE
    finally:
        allocator.close()
