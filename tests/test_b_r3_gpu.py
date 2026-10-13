"""Synthetic equivalence and crash recovery; not medical or H100 speed evidence."""

import copy
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import b_r3_gpu_state as fast
import check_stage_b_short_multi_gpu as gpu_runner
import check_stage_b_short_multi as old_runner
from b_r3_state import State as OldState
from b_r3_gpu_features import GPUFeatureAudit
from b_r3_diagnostics import train_step
from ccrr.errors import CCRRError
from ccrr.evaluation.attacks import PooledAttacker
from ccrr.representation.purifier import SemanticAttacker
from ccrr.runtime.support import seed_everything
from ccrr.training.b_batches import TeacherBank
from ccrr.training.b_control import control_update
from ccrr.training.b_diagnostics import isolated_rng
from ccrr.training.feature_audit import FeatureAudit
from test_stage_a_training_loops import compare
from test_stage_b_execution import context


@pytest.fixture(autouse=True)
def reset():
    fast.State.reset_registry()
    fast.Timings.path = None
    fast.Timings.values = {}
    yield
    fast.State.reset_registry()


def execute(path, steps, resume=False, force=False):
    seed_everything(31)
    device = os.environ.get("CCRR_TEST_DEVICE", "cpu")
    model = torch.nn.Linear(3, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    st = fast.State(path, {"test": "interval"}, {"head": model}, {"head": opt})
    if not st.loaded:
        st.data = {"next": 0, "evaluated": 0, "complete": False}
        st.save()
    ran = []
    for n in range(st.data["next"], steps):
        opt.zero_grad()
        model(
            torch.rand(2, 3, device=device) + np.random.rand() + random.random()
        ).square().mean().backward()
        opt.step()
        st.data["next"] = n + 1
        st.save()
        ran.append(n)
    if force:
        fast.State.flush_pending()
    return st, ran


def test_hard_restart_replays_only_uncommitted_interval(tmp_path):
    full, _ = execute(tmp_path / "full.pt", 237, force=True)
    split, first = execute(tmp_path / "split.pt", 237)
    assert torch.load(split.path, weights_only=True)["step"] == 200
    fast.State.reset_registry()  # Simulates loss of all process memory.
    split, replay = execute(tmp_path / "split.pt", 237, resume=True, force=True)
    assert replay == list(range(200, 237))
    assert len(first) == 237
    compare(
        torch.load(full.path, weights_only=True, map_location="cpu"),
        torch.load(split.path, weights_only=True, map_location="cpu"),
    )


def test_safe_flush_preserves_rng_inside_isolated_head(tmp_path):
    seed_everything(73)
    caller = torch.get_rng_state().clone()
    with isolated_rng():
        st, _ = execute(tmp_path / "head.pt", 3)
        expected = torch.get_rng_state().clone()
    assert torch.equal(caller, torch.get_rng_state())
    fast.State.flush_pending()
    raw = torch.load(st.path, weights_only=True, map_location="cpu")
    assert raw["step"] == 3
    assert torch.equal(raw["torch_rng"], expected)
    assert torch.equal(torch.get_rng_state(), caller)


def test_failed_replace_keeps_old_full_checkpoint(tmp_path, monkeypatch):
    st, _ = execute(tmp_path / "head.pt", 99)
    before = st.path.read_bytes()

    def fail(*args):
        raise OSError("injected failure before rename")

    monkeypatch.setattr(fast.os, "replace", fail)
    with pytest.raises(OSError):
        st._commit()
    assert st.path.read_bytes() == before


def test_old_checkpoint_rejected(tmp_path):
    m = torch.nn.Linear(2, 1)
    o = torch.optim.AdamW(m.parameters())
    old = OldState(tmp_path / "old.pt", {}, {"head": m}, {"head": o})
    old.save()
    with pytest.raises(CCRRError, match="protocol/binding"):
        fast.State(old.path, {}, {"head": m}, {"head": o})


def test_real_update_math_and_complete_optimizer_state_unchanged(tmp_path):
    results = []
    for name, runner in (("old", old_runner), ("gpu", gpu_runner)):
        e = context(tmp_path / name, micro=16)
        if name == "gpu":
            e.cfg["b_cache_device"] = e.device
        bank = TeacherBank(e)
        st = runner.make_state(e, {"test": "update"}, False)
        st.schedules["purifier"].next_step = 1999
        st.schedules["attacker"].next_step = 3998
        for _ in range(2):
            n = st.schedules["purifier"].next_step
            train_step(
                e,
                bank,
                n,
                st.opts["purifier"],
                st.opts["attacker"],
                st.schedules["purifier"],
                st.schedules["attacker"],
            )
            control_update(
                e.no_branch_control,
                bank,
                [0, 1, 2, 3, 4, 5] * 2 + [0, 1, 2, 3],
                16,
                st.opts["control"],
                st.schedules["control"],
            )
            st.data["next"] = n + 1
            st.save()
        if name == "gpu":
            fast.State.flush_pending()
        raw = torch.load(st.path, weights_only=True, map_location="cpu")
        raw["stage"] = "compared separately"
        results.append(raw)
    compare(*results)


def score_context(tmp_path):
    e = context(tmp_path, micro=4)
    e.data = {k: copy.copy(v) for k, v in e.data.items()}
    for ds in e.data.values():
        ds.rows = [
            {**r, "img_id": str(i), "patient_id": str(i // 2)}
            for i, r in enumerate(ds.rows)
        ]
        ds.index = {str(i): i for i in range(6)}
        ds.root = tmp_path
    e.fit = e.data["fit"]
    root = tmp_path / "protocol_audit_20261009/pair_protocol"
    root.mkdir(parents=True)
    queries = [
        {
            "query": str(i),
            "patient_id": str(i // 2),
            "positive_gallery": [str(i ^ 1)],
            "negative_gallery": [str((i + 2) % 6), str((i + 3) % 6)],
        }
        for i in range(6)
    ]
    for scenario in ("p1", "p2"):
        (root / f"val_select_{scenario}.json").write_text(
            json.dumps({"queries": queries})
        )
    return e


@pytest.mark.parametrize("family", ["spatial", "pooled"])
@pytest.mark.parametrize("rep", ["raw", "control", "sem"])
def test_scores_identical_all_directions_with_exact_batch_reuse(tmp_path, family, rep):
    e = score_context(tmp_path)
    old, new = FeatureAudit(e), GPUFeatureAudit(e)
    seed_everything(12)
    v = torch.ones(6, 1, 14, 14)
    v[0, :, :2] = 0
    store = {"v": v, rep: torch.randn(6, 3 if rep == "sem" else 2, 768, 14, 14)}
    old.store = {"val_select": store}
    new.store = {"val_select": {k: v.to(e.device) for k, v in store.items()}}
    model = (SemanticAttacker() if family == "spatial" else PooledAttacker()).to(
        e.device
    )
    for scenario in ("p1", "p2"):
        rng = torch.get_rng_state().clone()
        expected = old.attack_scores(model, rep, scenario)
        actual = new.attack_scores(model, rep, scenario)
        compare(expected, actual)
        assert torch.equal(rng, torch.get_rng_state())
        # A new scoring call must not reuse encodings from before this parameter update.
        with torch.no_grad():
            next(model.parameters()).add_(0.05)
        compare(
            old.attack_scores(model, rep, scenario),
            new.attack_scores(model, rep, scenario),
        )


def test_cache_reuses_teacher_banks_and_matches_original(tmp_path, monkeypatch):
    e = score_context(tmp_path)
    e.no_branch_control = old_runner.initialize(e)
    e.purifier.eval()
    e.no_branch_control.eval()
    banks = {r: TeacherBank(e, r) for r in e.data}
    e.b_teacher_bank = banks["fit"]
    e.b_teacher_banks = banks
    old = FeatureAudit(e)
    old.cache()
    import ccrr.training.b_batches as batches

    def unexpected(*args, **kwargs):
        raise AssertionError("Should reuse both existing banks")

    monkeypatch.setattr(batches, "TeacherBank", unexpected)
    new = GPUFeatureAudit(e)
    new.cache()
    compare(old.means, new.means)
    for role, store in old.store.items():
        compare(store, {k: v.cpu() for k, v in new.store[role].items()})
        assert all(
            v.device.type == torch.device(e.device).type
            for v in new.store[role].values()
        )


def test_pending_score_best_and_final_commit(tmp_path):
    st, _ = execute(tmp_path / "head.pt", 0)

    class Unlimited:
        def check(self):
            pass

        def record(self):
            pass

    def update(n):
        opt = next(iter(st.opts.values()))
        model = next(iter(st.modules.values()))
        opt.zero_grad()
        model(torch.ones(1, 3, device=next(model.parameters()).device)).sum().backward()
        opt.step()

    def score():
        return 1.0, {"step": st.data["next"]}

    st.loaded = False
    fast.run_job(st, 4, update, score, 2, Unlimited())
    raw = torch.load(st.path, weights_only=True)
    assert raw["training_state"]["data"]["best"]["step"] == 2
    assert raw["training_state"]["data"]["evaluated"] == 4
    assert raw["training_state"]["data"]["complete"]
    assert str(st.path) not in fast.State.registry


@pytest.mark.parametrize("family", ["linear", "mlp"])
def test_resident_clinical_means_preserve_updates_selection_and_rng(
    tmp_path, monkeypatch, family
):
    import b_r3_audit as old
    import b_r3_gpu_audit as new
    from b_r3_state import run_job as old_job

    class Unlimited:
        def check(self):
            pass

        def record(self):
            pass

    for module, job in ((old, old_job), (new, fast.run_job)):
        monkeypatch.setattr(
            module, "run_job", lambda s, t, u, v, i, b, job=job: job(s, 4, u, v, 2, b)
        )
    states = []
    for name, module in (("old", old), ("new", new)):
        e = context(tmp_path / name)
        audit = module.Audit(e, {"test": "means"}, Unlimited())
        seed_everything(25)
        rows = [
            {"sem": torch.randn(3, 768), "label": i, "patient": str(i)}
            for i in range(6)
        ]
        audit.means = {"fit": rows, "val_select": rows}
        audit.readout(
            "clinical-sem-" + family, "sem", lambda: audit._classifier("sem", family)
        )
        raw = torch.load(
            audit.heads / f"clinical-sem-{family}.pt",
            weights_only=True,
            map_location="cpu",
        )
        raw["stage"] = "compared separately"
        states.append(raw)
    compare(*states)


def test_regular_save_requests_reduce_to_interval_commits(tmp_path):
    st, _ = execute(tmp_path / "head.pt", 235)
    assert fast.Timings.values["checkpoint.total"]["calls"] == 3  # initial, 100, 200
    fast.State.flush_pending()
    assert fast.Timings.values["checkpoint.total"]["calls"] == 4
    assert torch.load(st.path, weights_only=True)["step"] == 235


def test_stop_command_rejects_other_workers_and_missing_out(tmp_path):
    from stop_b_r3_for_restart import matches_worker

    result = tmp_path / "result"
    assert matches_worker(
        ["python", "/old/scripts/check_stage_b_short_multi.py", "--out", str(result)],
        result,
    )
    assert not matches_worker(["python", "other.py", "--out", str(result)], result)
    assert not matches_worker(
        ["python", "check_stage_b_short_multi.py", "--out", str(tmp_path / "other")],
        result,
    )
    assert not matches_worker(
        ["python", "check_stage_b_short_multi.py", "--out"], result
    )
    assert not matches_worker(
        ["python", "check_stage_b_short_multi_gpu.py", "--out", str(result)], result
    )


def test_new_runner_keeps_full_registered_counts_and_observations(
    tmp_path, monkeypatch
):
    import test_b_r3_short_multi as original_tests

    monkeypatch.setattr(original_tests, "runner", gpu_runner)
    original_tests.test_registered_budget_and_observations_without_expensive_training(
        tmp_path, monkeypatch
    )


def test_new_runner_rejects_incomplete_audit(tmp_path, monkeypatch):
    import test_b_r3_short_multi as original_tests

    monkeypatch.setattr(original_tests, "runner", gpu_runner)
    original_tests.test_incomplete_attack_cannot_be_reported_as_completed(tmp_path)
