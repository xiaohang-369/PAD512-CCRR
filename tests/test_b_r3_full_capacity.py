"""Batch loss-contract checks; not server/H100 performance evidence."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from b_r3_full_capacity import (
    purifier_update,
    control_update,
    head_update,
    search_batches,
    evaluate,
)
from b_r3_h100_scaling import Head
from b_r3_perf_common import exact
from profile_b_r3_h100 import Workload
import b_r3_gpu_state as state
import check_stage_b_short_multi_gpu as runner
from ccrr.training.b_batches import TeacherBank
from test_b_r3_gpu import score_context
from test_b_r3_cached_execution import audit as audit  # noqa: F401
from test_b_r3_h100_scaling import audit as base_audit  # noqa: F401


def test_capacity_search_brackets_to_16_without_throughput_plateau_stop():
    seen = []

    def trial(batch):
        seen.append(batch)
        return batch <= 176

    result = search_batches(trial, cap=1024)
    assert result["largest_tested_stable_batch"] == 176
    assert result["first_failing_batch_bound"] == 192
    assert 256 in seen
    assert not result["search_cap_reached"]
    capped = search_batches(lambda batch: True, cap=128)
    assert capped["search_cap_reached"] and capped["first_failing_batch_bound"] is None
    assert search_batches(lambda batch: False)["largest_tested_stable_batch"] == 0


@pytest.mark.parametrize("kind", ["purifier", "control"])
def test_generalized_batch16_preserves_original_model_optimizer_and_losses(
    tmp_path, kind
):
    e = score_context(tmp_path / "resources")
    e.cfg["microbatch"] = 16
    banks = {role: TeacherBank(e, role) for role in ("fit", "val_select")}
    st = runner.make_state(e, {"source": "fixture"}, False)
    st.schedules["purifier"].next_step = 2000
    st.schedules["attacker"].next_step = 4000
    st.data["next"] = 2000
    source = tmp_path / "initial.pt"
    st.save(source)
    work = Workload(
        e,
        banks,
        None,
        {"state": state, "runner": runner},
        tmp_path / kind,
        kind,
        source,
    )
    work.update()
    expected = work.capture()
    expected_loss = (
        work.state.data["losses"][-1]["loss"]
        if kind == "purifier"
        else work.state.data["control_losses"][-1]
    )
    work.reset()
    fn = purifier_update if kind == "purifier" else control_update
    got = fn(work, 16)
    actual = work.capture()
    for field in ("models", "optimizers", "schedulers", "gradients_after_clip", "rng"):
        assert exact(expected[field], actual[field]), field
    assert float(got["loss"]) == expected_loss
    work.reset()
    sizes = []
    model = e.purifier if kind == "purifier" else e.no_branch_control
    hook = model.register_forward_pre_hook(
        lambda _, args: sizes.append(args[0].shape[0])
    )
    values = fn(work, 32)
    hook.remove()
    assert 32 in sizes and all(torch.isfinite(v) for v in values.values())
    metrics = evaluate(work, banks)
    assert metrics["rgb_mse"] >= 0 and metrics["gray_mse"] >= 0
    if kind == "purifier":
        assert len(metrics["clinical_classifier_macro_auc"]) == 3
        assert (
            len(metrics["coupled_adversary_auc_proxy_not_independent_audit"]["p2"]) == 9
        )


@pytest.mark.parametrize("family", ["spatial", "pooled"])
def test_full_capacity_head_batch16_matches_and_32_is_actual_forward(audit, family):  # noqa: F811
    h = Head(audit, ("p2", "control", family))
    h.update()
    expected = h.capture(gradients=True)
    h.restore()
    head_update(h, 16)
    assert exact(expected, h.capture(gradients=True))
    h.restore()
    sizes = []
    hook = h.model.register_forward_pre_hook(
        lambda _, args: sizes.append(args[0].shape[0])
    )
    assert torch.isfinite(head_update(h, 32)["loss"])
    hook.remove()
    assert sizes == [32]


def test_all_workloads_write_table_and_validation_budgets(audit, tmp_path, monkeypatch):  # noqa: F811
    # Exercise all 14 workload transitions with real CPU updates. Fake GPU clocks
    # and metric summaries test orchestration only, never H100 validity/speed.
    import time
    from types import SimpleNamespace
    import b_r3_full_capacity as capacity
    import profile_b_r3_h100 as probe

    banks = {role: TeacherBank(audit.e, role) for role in ("fit", "val_select")}
    st = runner.make_state(audit.e, {"source": "fixture"}, False)
    st.schedules["purifier"].next_step = 2000
    st.schedules["attacker"].next_step = 4000
    st.data["next"] = 2000
    source = tmp_path / "initial.pt"
    st.save(source)
    search = capacity.search_batches
    monkeypatch.setattr(capacity, "search_batches", lambda trial: search(trial, cap=16))
    monkeypatch.setattr(capacity, "evaluate", lambda *args: {"fixture_metric": 0})

    def timing(fn):
        start = time.perf_counter()
        value = fn()
        return value, {"wall_seconds": time.perf_counter() - start}

    monkeypatch.setattr(probe, "timed", timing)
    for name in (
        "synchronize",
        "empty_cache",
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    result = capacity.run_full_capacity(
        audit.e,
        banks,
        audit,
        {"state": state, "runner": runner},
        source,
        SimpleNamespace(
            out=tmp_path, warmup=1, steps=1, repeats=2, capacity_min_seconds=0
        ),
        SimpleNamespace(check=lambda: None, rows=[]),
    )
    assert result["complete"] and len(result["results"]) == 14
    for row in result["results"].values():
        assert row["search_cap_reached"] and row["best_throughput_batch"] == 16
        assert row["validation"]["variants"]["best"]["sample_exposures"] == 320
    assert (
        len((tmp_path / "full-capacity/comparison.csv").read_text().splitlines()) == 15
    )
