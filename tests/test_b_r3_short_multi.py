"""Real model/optimizer replay on synthetic features; NOT clinical/privacy evidence."""

import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from b_r3_state import State, Budget, Paused, run_job
from b_r3_diagnostics import initialize, permute_mask
import b_r3_diagnostics as diag
import b_r3_audit as readers
import check_stage_b_short_multi as runner
from ccrr.training.b_batches import TeacherBank, train_step as production_step
from ccrr.training.b_control import control_update
from ccrr.runtime.support import seed_everything
from ccrr.errors import CCRRError
from test_stage_b_execution import context
from test_stage_a_training_loops import compare


class Unlimited:
    def check(self):
        pass

    def record(self):
        pass


def test_scale_is_once_and_matched_control_is_independent(tmp_path):
    e = context(tmp_path / "e")
    before = copy.deepcopy(e.purifier.state_dict())
    c = initialize(e)
    for k, v in e.purifier.state_dict().items():
        expected = (
            before[k] * 0.1
            if k in ("encoder.end.weight", "decoder.end.weight")
            else before[k]
        )
        assert torch.equal(v, expected)
    compare(c.encoder.state_dict(), e.purifier.encoder.state_dict())
    compare(c.decoder.state_dict(), e.purifier.decoder.state_dict())
    assert c.encoder.end.weight.data_ptr() != e.purifier.encoder.end.weight.data_ptr()


@pytest.mark.parametrize("start", [999, 1999])
def test_diagnostic_step_exactly_matches_original_full_losses(tmp_path, start):
    snapshots = []
    for name, fn in [("original", production_step), ("diagnostic", diag.train_step)]:
        e = context(tmp_path / name)
        bank = TeacherBank(e)
        st = runner.make_state(e, {"test": "math"}, False)
        s = st.schedules
        s["purifier"].next_step = start
        s["attacker"].next_step = 2 * start
        for n in range(start, start + 2):
            result = fn(
                e,
                bank,
                n,
                st.opts["purifier"],
                st.opts["attacker"],
                s["purifier"],
                s["attacker"],
            )
            assert np.isfinite(result["loss"])
        st.data["next"] = start + 2
        st.save()
        snapshots.append(torch.load(st.path, weights_only=True, map_location="cpu"))
    compare(snapshots[0], snapshots[1])


def test_full_three_optimizer_restore_with_control_and_rng(tmp_path):
    def execute(path, count, resume=False):
        e = context(path)
        bank = TeacherBank(e)
        st = runner.make_state(e, {"test": "restore"}, resume)
        if not resume:
            st.schedules["purifier"].next_step = 1999
            st.schedules["attacker"].next_step = 3998
        for _ in range(count):
            n = st.schedules["purifier"].next_step
            diag.train_step(
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
                list(range(6)) * 2 + [0, 1, 2, 3],
                4,
                st.opts["control"],
                st.schedules["control"],
            )
            torch.rand(1)
            np.random.rand()
            random.random()
            st.data["next"] = n + 1
            st.save()
        return torch.load(st.path, weights_only=True, map_location="cpu")

    full = execute(tmp_path / "full", 2)
    execute(tmp_path / "split", 1)
    split = execute(tmp_path / "split", 1, True)
    compare(full, split)
    assert split["training_state"]["schedulers"]["purifier"]["total"] == 10000
    assert split["training_state"]["schedulers"]["attacker"]["total"] == 20000
    with pytest.raises(CCRRError, match="binding"):
        runner.make_state(context(tmp_path / "split"), {"test": "wrong"}, True)


@pytest.mark.parametrize("interrupt_at", [1, 2, 3, 4])
def test_head_recovery_pending_selection_and_best_weights(tmp_path, interrupt_at):
    def execute(path, crash=None):
        seed_everything(52)
        model = torch.nn.Linear(3, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        state = State(path, {"head": "test"}, {"head": model}, {"opt": opt})
        calls = []

        class Stop:
            def record(self):
                pass

            def check(self):
                if crash is not None and state.data.get("next") == crash:
                    raise Paused("test")

        def step(n):
            calls.append(n)
            opt.zero_grad()
            x = torch.rand(2, 3) + np.random.rand() + random.random()
            model(x).square().mean().backward()
            opt.step()

        def score():
            # Earlier endpoint remains selected; best state must not become final.
            n = state.data["next"]
            return -n, {
                "prediction": np.array([float(model.weight.sum().detach())]),
                "numpy_scalar": np.float64(n),
            }

        # Production score is >=0; use a positive descending score for this test.
        def positive():
            v, r = score()
            return 10 + v, r

        try:
            run_job(state, 4, step, positive, 2, Stop())
        except Paused:
            pass
        return calls, torch.load(path, weights_only=True, map_location="cpu")

    _, full = execute(tmp_path / "full.pt")
    before, _ = execute(tmp_path / "split.pt", interrupt_at)
    after, split = execute(tmp_path / "split.pt")
    assert before + after == [0, 1, 2, 3]
    compare(full, split)
    assert split["training_state"]["data"]["best_score"] == 8


@pytest.mark.parametrize("family", ["linear", "mlp"])
def test_actual_classifier_callbacks_resume(tmp_path, monkeypatch, family):
    original = readers.run_job

    def small(s, total, train, score, interval, budget):
        return original(s, 4, train, score, 2, budget)

    monkeypatch.setattr(readers, "run_job", small)

    def run(path, interrupt=False):
        e = context(path)
        a = readers.Audit(e, {"test": "classifier"}, Unlimited())
        torch.manual_seed(8)
        rows = [
            {"sem": torch.randn(3, 768), "label": i, "patient": str(i)}
            for i in range(6)
        ]
        a.means = {"fit": rows, "val_select": rows}
        if interrupt:

            class Stop(Unlimited):
                def check(self):
                    p = a.heads / f"clinical-sem-{family}.pt"
                    if (
                        p.exists()
                        and torch.load(p, weights_only=True)["training_state"]["data"][
                            "next"
                        ]
                        == 3
                    ):
                        raise Paused("test")

            a.budget = Stop()
        try:
            a.readout(
                "clinical-sem-" + family, "sem", lambda: a._classifier("sem", family)
            )
        except Paused:
            pass
        return torch.load(
            a.heads / f"clinical-sem-{family}.pt", weights_only=True, map_location="cpu"
        )

    full = run(tmp_path / "full")
    run(tmp_path / "split", True)
    split = run(tmp_path / "split")
    compare(full, split)


def test_mask_permutation_preserves_fractional_occupancy_and_padding():
    g = torch.randint(0, 2, (3, 64, 5, 5)).float()
    g[:, :, :1] = 0
    v = torch.ones(3, 1, 5, 5)
    v[:, :, 1] = 0.25
    v[:, :, :1] = 0
    a = permute_mask(g, torch.Generator().manual_seed(7))
    assert torch.equal(g.sum(1), a.sum(1))
    assert torch.equal((g * v).sum(), (a * v).sum())
    assert not torch.equal(g, a)
    assert not a[:, :, :1].any()


def test_budget_is_cumulative_and_pause_is_not_success(tmp_path):
    clock = [0.0]
    b = Budget(tmp_path, limit=10, clock=lambda: clock[0])
    clock[0] = 6
    b.record()
    clock[0] = 1000
    c = Budget(tmp_path, limit=10, clock=lambda: clock[0])
    c.check()
    clock[0] += 4
    with pytest.raises(Paused):
        c.check()
    with pytest.raises(CCRRError, match="Budget changed"):
        Budget(tmp_path, limit=20)


@pytest.mark.parametrize("family", ["spatial", "pooled"])
def test_actual_independent_attack_update_resume(tmp_path, monkeypatch, family):
    original = readers.run_job
    monkeypatch.setattr(
        readers,
        "run_job",
        lambda s, t, train, score, i, b: original(s, 4, train, score, 2, b),
    )

    def execute(path, interrupt=False):
        e = context(path)
        a = readers.Audit(e, {"test": "attacker"}, Unlimited())
        bank = TeacherBank(e)
        a.store = {
            r: {
                "v": bank.storage["v"].cpu(),
                "sem": torch.stack(
                    [bank.storage["f0"], bank.storage["f1"], bank.storage["f0"]], 1
                ).cpu(),
            }
            for r in ("fit", "val_select")
        }

        # Score on fixed synthetic pairs; real attacker forward and real update.
        def scores(model, rep, scenario):
            model.eval()
            with torch.no_grad():
                fa, v = a.feature_batch("val_select", [0, 1, 2, 3, 4, 5], rep, [0] * 6)
                fb, vb = a.feature_batch("val_select", [1, 2, 3, 4, 5, 0], rep, [1] * 6)
                values = model(fa, v, fb, vb).cpu().tolist()
            return [
                {
                    "domain": [i, j],
                    "scores": values,
                    "labels": [0, 1, 0, 1, 0, 1],
                    "weights": [1.0] * 6,
                    "pairs": [(str(k), str((k + 1) % 6)) for k in range(6)],
                }
                for i in range(3)
                for j in range(3)
            ]

        a.attack_scores = scores
        if interrupt:

            class Stop(Unlimited):
                def check(self):
                    p = a.heads / f"identity-p1-sem-{family}.pt"
                    if (
                        p.exists()
                        and torch.load(p, weights_only=True)["training_state"]["data"][
                            "next"
                        ]
                        == 3
                    ):
                        raise Paused("test")

            a.budget = Stop()
        try:
            a.readout(
                "identity-p1-sem-" + family,
                "sem",
                lambda: a._attacker("p1", "sem", family),
            )
        except Paused:
            pass
        return torch.load(
            a.heads / f"identity-p1-sem-{family}.pt",
            weights_only=True,
            map_location="cpu",
        )

    full = execute(tmp_path / "full")
    execute(tmp_path / "split", True)
    split = execute(tmp_path / "split")
    compare(full, split)


def test_adaptation_resume_never_changes_original_attacker(tmp_path, monkeypatch):
    # Actual optimizer recovery, 1000 fixed budget, tiny model to bound test cost.
    from types import SimpleNamespace

    counts = []

    def execute(path, interrupt=False):
        seed_everything(42)
        e = SimpleNamespace(
            output=path,
            device="cpu",
            sampler=SimpleNamespace(fallbacks=0),
            sem_attacker=torch.nn.Linear(2, 1),
            seed=42,
        )
        a = readers.Audit(e, {"test": "adapt"}, Unlimited())
        initial = copy.deepcopy(e.sem_attacker.state_dict())

        def update(self, model, opt, rep, scenario, family, step, **kwargs):
            counts.append(step)
            opt.zero_grad()
            model(torch.rand(2, 2)).square().mean().backward()
            opt.step()

        def score(self, model, rep, scenario):
            v = float(model.weight.sum().detach())
            return [
                {
                    "domain": [i, j],
                    "scores": [v, v + 1],
                    "labels": [0, 1],
                    "weights": [1.0, 1.0],
                    "pairs": [("a", "b"), ("a", "a")],
                }
                for i in range(3)
                for j in range(3)
            ]

        monkeypatch.setattr(readers.Audit, "attacker_update", update)
        monkeypatch.setattr(readers.Audit, "attack_scores", score)
        if interrupt:

            class Stop(Unlimited):
                def check(self):
                    p = a.heads / "post-adapt.pt"
                    if (
                        p.exists()
                        and torch.load(p, weights_only=True)["training_state"]["data"][
                            "next"
                        ]
                        == 777
                    ):
                        raise Paused("test")

            a.budget = Stop()
        try:
            a.adapted()
        except Paused:
            pass
        compare(initial, e.sem_attacker.state_dict())
        return torch.load(
            a.heads / "post-adapt.pt", weights_only=True, map_location="cpu"
        )

    full = execute(tmp_path / "full")
    counts.clear()
    execute(tmp_path / "split", True)
    split = execute(tmp_path / "split")
    compare(full, split)
    assert counts == list(range(1000))


def test_registered_budget_and_observations_without_expensive_training(
    tmp_path, monkeypatch
):
    # Exercise real orchestrator for all 3000+3000 steps; updates are explicitly mocked.
    from types import SimpleNamespace

    e = context(tmp_path)
    sched = {
        k: SimpleNamespace(next_step=0) for k in ("purifier", "attacker", "control")
    }
    saves = []
    seen = {"purifier": [], "control": []}

    class MockState:
        bound = {}
        schedules = sched
        opts = {k: None for k in sched}
        data = {
            "next": 0,
            "phase": "purifier",
            "observations": {},
            "control_observations": {},
            "losses": [],
            "control_losses": [],
        }

        def save(self, destination=None):
            if destination:
                saves.append(destination.name)

    def step(e, b, n, o, a, s, ads):
        assert n == s.next_step
        seen["purifier"].append(n)
        s.next_step += 1
        ads.next_step += 2
        return {"loss": 0.0}

    def ctrl(m, b, ids, micro, o, s):
        seen["control"].append(s.next_step)
        expected = (
            runner.rng_for(e.seed, "B", "images", s.next_step)
            .integers(len(e.fit), size=16)
            .tolist()
        )
        assert ids == expected
        s.next_step += 1
        return 0.0

    monkeypatch.setattr(runner, "train_step", step)
    monkeypatch.setattr(runner, "control_update", ctrl)
    monkeypatch.setattr(runner, "observation", lambda e, b, n: {"step": n})
    monkeypatch.setattr(runner, "full_scores", lambda *args: {})
    monkeypatch.setattr(runner, "verify_unchanged", lambda *args: None)
    e.no_branch_control = None
    st = MockState()
    runner.training(e, {"fit": None, "val_select": None}, st, Unlimited(), {})
    assert seen["purifier"] == seen["control"] == list(range(3000))
    assert sched["attacker"].next_step == 6000
    assert list(map(int, st.data["observations"])) == list(runner.POINTS)
    assert st.data["phase"] == "audit"
    assert len(saves) == 12


def test_observation_and_mask_diagnostic_do_not_update_models(tmp_path, monkeypatch):
    e = context(tmp_path)
    bank = TeacherBank(e)
    initialize(e)
    monkeypatch.setattr(diag, "cohort", lambda rows: ([0, 1, 2, 3, 4, 5], []))
    model_before = copy.deepcopy(e.purifier.state_dict())
    rng = torch.get_rng_state().clone()
    result = diag.observation(e, {"fit": bank, "val_select": bank}, 1000)
    assert len(result["roles"]["fit"]["all_images"]["f0"]) == 6
    assert (
        "mask_between_image_variance"
        in result["roles"]["fit"]["probe_diagnostics"]["f0"]
    )
    assert torch.equal(rng, torch.get_rng_state())
    audit = readers.Audit(e, {"test": "mask"}, Unlimited())
    for family in ("linear", "mlp"):
        h = readers.classifier(family)
        torch.save(
            {"training_state": {"data": {"best_weights": h.state_dict()}}},
            audit.heads / f"clinical-sem-{family}.pt",
        )
    rng = torch.get_rng_state().clone()
    result = diag.mask_diagnostic(e, bank, audit)
    assert result["occupancy_preserved"] and len(result["rows"]) == 18
    assert not result["privacy_evaluation"]
    compare(model_before, e.purifier.state_dict())
    assert torch.equal(rng, torch.get_rng_state())


def test_incomplete_attack_cannot_be_reported_as_completed(tmp_path):
    from types import SimpleNamespace

    a = SimpleNamespace(heads=tmp_path, bound={"test": "budget"})
    jobs = [
        (f"clinical-{r}-{f}", 2000)
        for r in ("raw", "control", "sem")
        for f in ("linear", "mlp")
    ]
    jobs += [
        (f"identity-{s}-{r}-{f}", 5000)
        for s in ("p1", "p2")
        for r in ("raw", "control", "sem")
        for f in ("spatial", "pooled")
    ]
    jobs += [("post-adapt", 1000)]
    for n, total in jobs:
        torch.save(
            {
                "stage": runner.PROTOCOL,
                "training_state": {
                    "binding": {**a.bound, "head": n},
                    "data": {"next": total, "complete": True, "after": {}},
                },
            },
            tmp_path / (n + ".pt"),
        )
    assert runner.audit_update_counts(a) == {
        "clinical_readouts": 12000,
        "independent_attackers": 60000,
        "post_adaptation": 1000,
    }
    p = tmp_path / "identity-p2-sem-spatial.pt"
    raw = torch.load(p, weights_only=True)
    raw["training_state"]["data"]["next"] = 4999
    torch.save(raw, p)
    with pytest.raises(CCRRError, match="Incomplete evaluation budget"):
        runner.audit_update_counts(a)
