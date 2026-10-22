"""Image-domain, digital transport and shard integrity checks; no medical claims."""

import copy
import json
from pathlib import Path

import pytest
import torch

from ccrr_v4.original import OriginalGrayscale, quantize8
from ccrr_v4.phase2_baseline import (
    validate_config,
    views_from_model,
    ordered_records,
    numeric_policy,
)
from ccrr_v4.verify import state_digest


def test_digital_views_use_original_gray_and_keep_frozen_parameters():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = OriginalGrayscale().eval().requires_grad_(False)
        before = state_digest(model)
        x = torch.arange(3 * 8 * 12).reshape(1, 3, 8, 12).float() / (3 * 8 * 12 - 1)
        with torch.no_grad():
            views, psnr = views_from_model(model, x)
            expected_g = quantize8(model.encoder(x * 2 - 1))[0].float() / 255
        assert views["rgb"] is x
        torch.testing.assert_close(views["G0_png"], expected_g, atol=1e-7, rtol=0)
        expected_y = (
            x * x.new_tensor([0.2989, 0.5870, 0.1140])[None, :, None, None]
        ).sum(1, keepdim=True)
        expected_y = torch.floor(expected_y * 255 + 0.5) / 255
        assert torch.equal(views["ordinary_gray8"], expected_y)
        assert not torch.equal(views["G0_png"], expected_y)
        assert psnr.shape == (1,) and torch.isfinite(psnr).all()
        assert before == state_digest(model)
        assert all(p.grad is None for p in model.parameters())
    finally:
        torch.set_num_threads(old)


def test_shards_merge_exactly_once_in_original_order():
    shards = [[dict(index=0), dict(index=2)], [dict(index=1)]]
    assert [r["index"] for r in ordered_records(shards, 3)] == [0, 1, 2]
    with pytest.raises(ValueError):
        ordered_records(shards, 4)
    with pytest.raises(ValueError):
        ordered_records([*shards, [dict(index=1)]], 3)


def test_registered_scope_rejects_training_and_test_split():
    cfg = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/phase2-baseline.json"
        ).read_text()
    )
    validate_config(cfg)
    for key, value in (
        ("training_enabled", True),
        ("role", "test"),
        ("gpu_ids", [0, 0]),
        ("evaluation_batch_size", 16),
    ):
        bad = copy.deepcopy(cfg)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_config(bad)


def test_numeric_policy_restores_flags_on_failure():
    before = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    with pytest.raises(RuntimeError):
        with numeric_policy(False):
            assert not torch.backends.cudnn.allow_tf32
            assert not torch.backends.cuda.matmul.allow_tf32
            raise RuntimeError("fixture")
    assert (
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    ) == before
