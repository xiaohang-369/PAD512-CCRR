"""R5 real loss/update path on synthetic images; no server experiment."""

from copy import deepcopy
import json

import pytest
import torch

from test_grayscale import SyntheticData, teacher, snapshot, same
from ccrr_v3 import grayscale_r5 as r5, grayscale_r5_cli as cli
from ccrr_v3.grayscale import canonical_gray
from ccrr_v3.runtime import read


def config():
    cfg = deepcopy(r5.CONFIG)
    cfg.update(baseline_updates=1, c1_cycles=2, checkpoint_every=1)
    return cfg


def engine(step=8500, variant="learned", t=None):
    return r5.Engine(
        teacher() if t is None else t,
        SyntheticData("", "fit"),
        step,
        torch.device("cpu"),
        variant,
        config(),
    )


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def test_direct16_required_config_and_snapshot_schedule():
    cfg = r5.CONFIG
    assert (cfg["batch_size"], cfg["microbatch_size"], cfg["accumulation_steps"]) == (
        16,
        16,
        1,
    )
    assert cfg["baseline_updates"] // cfg["checkpoint_every"] == 8
    assert cfg["c1_cycles"] // cfg["checkpoint_every"] == 40
    for changes in (
        {"accumulation_steps": 8, "microbatch_size": 2},
        {"batch_size": 8, "microbatch_size": 8},
    ):
        bad = {**config(), **changes}
        with pytest.raises(ValueError, match="direct full batch"):
            r5.Engine(teacher(), SyntheticData("", "fit"), 8500, "cpu", cfg=bad)
    assert r5.lr_factor(1, 2000, cfg) == 0.1
    assert r5.lr_factor(100, 2000, cfg) == pytest.approx(1)
    assert r5.lr_factor(2000, 2000, cfg) == 0.1
    assert r5.lr_factor(10000, 10000, cfg) == 0.1


def test_padding_gradient_and_fixed_body_receiver_contract():
    x = torch.rand(2, 1, 32, 32, requires_grad=True)
    v = torch.ones_like(x)
    v[:, :, :8] = 0
    g = canonical_gray(x, v)
    assert torch.all(g[v == 0] == 128 / 255)
    g.sum().backward()
    assert torch.equal(x.grad, v)
    for bits in (4, 8):
        q = (torch.arange(8192) % 2**bits).byte().reshape(1, 8, 32, 32)
        body = r5.pack_symbols(q, bits)
        assert len(body) == 8192 * bits // 8
        assert torch.equal(q, r5.unpack_symbols(body, bits))
        with pytest.raises(ValueError):
            r5.unpack_symbols(body[:-1], bits)
    with pytest.raises(ValueError):
        r5.pack_symbols(torch.full((8192,), 16, dtype=torch.uint8), 4)


def test_real_direct_batch_updates_adam_groups_pairing_and_resume():
    e = engine()
    original_p = snapshot(e.model.public)
    old_teacher = snapshot(e.teacher)
    first = e.cycle()
    assert len(first["indices"]) == 16 and e.seen.sum() == 16
    assert same(e.model.public, original_p) and e.ed_updates == 1 and e.p_updates == 0
    assert e.ed.param_groups[1]["weight_decay"] == 0
    assert e.ed.param_groups[0]["weight_decay"] == 1e-4
    assert not e.p.state
    state = deepcopy(e.state())
    second = e.cycle(inspect_gradients=True)
    expected = deepcopy(e.state())
    assert second["public"]["grad_p"] > 0
    assert second["public"]["grad_e"] == second["public"]["grad_d"] == 0
    assert e.ed_updates == 2 and e.p_updates == 1 and e.seen.sum() == 32
    assert same(e.teacher, old_teacher)
    restored = engine(t=deepcopy(e.teacher))
    restored.restore(state)
    repeated = restored.cycle(inspect_gradients=True)
    assert second["indices"] == repeated["indices"]
    actual = restored.state()
    for k in expected:
        if k != "elapsed_seconds":
            equal_tree(expected[k], actual[k])
    other = engine(step=7000)
    assert other.initial_model_hash == e.initial_model_hash
    assert other.cycle()["indices"] == first["indices"]
    with pytest.raises(ValueError, match="mismatch"):
        other.restore(state)
    assert not torch.equal(
        e.model.decoder.precision.weight[:, 0], e.model.decoder.precision.weight[:, 1]
    )
    r5.copy_precision_condition(e.model.decoder)
    assert torch.equal(
        e.model.decoder.precision.weight[:, 0], e.model.decoder.precision.weight[:, 1]
    )


def test_incomplete_cycle_cannot_be_saved_or_continued(monkeypatch):
    e = engine()
    e.cycle()
    original = r5.update

    def fail_public(*args, **kwargs):
        if args[6] == "p":
            raise RuntimeError("injected failure after ED")
        return original(*args, **kwargs)

    monkeypatch.setattr(r5, "update", fail_public)
    with pytest.raises(RuntimeError):
        e.cycle()
    assert e.ed_updates == 2 and e.p_updates == 0 and e.attempted_updates == 3
    assert not e.boundary
    with pytest.raises(ValueError, match="incomplete"):
        e.state()
    with pytest.raises(ValueError, match="partial"):
        e.cycle()


def test_snapshot_atomic_roundtrip_tamper_stale_and_storage_limit(tmp_path):
    e = engine()
    e.cycle()
    reg = {"source_commit": "local-test", "resources": {}}
    budget = {"test": "budget"}
    binding = {
        **reg,
        "config": r5.CONFIG,
        "parents": {},
        "options": {"budget_sha256": cli.argsafe_budget_hash(budget)},
    }
    r5.atomic_json(tmp_path / "binding.json", binding)
    path, manifest = cli.save_checkpoint(e, tmp_path, [], 0)
    r5.atomic_json(
        tmp_path / "cost.json", cli.cost_record(e, manifest["checkpoint_bytes"], 16)
    )
    state, cost = cli.read_checkpoint(path, budget, reg)
    restored = engine(t=deepcopy(e.teacher))
    restored.restore(state)
    assert restored.position == 1 and cost["checkpoint_bytes"] == path.stat().st_size
    with pytest.raises(ValueError, match="overwrite"):
        cli.save_checkpoint(e, tmp_path, [], 0)
    with cli.run_lock(tmp_path):
        with pytest.raises(ValueError, match="active"):
            cli.read_checkpoint(path, budget, reg)
    e.cycle()
    with pytest.raises(cli.StorageCeiling):
        cli.save_checkpoint(e, tmp_path, [], 0, 1)
    assert not (tmp_path / "generator-c1-000001.pt").exists()
    newpath, _ = cli.save_checkpoint(e, tmp_path, [], cost["checkpoint_bytes"])
    with pytest.raises(ValueError, match="latest"):
        cli.read_checkpoint(path, budget, reg)
    with newpath.open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        cli.read_checkpoint(newpath, budget, reg)


@pytest.mark.parametrize("variant,updates", [("learned", 5), ("fixed_gray", 3)])
def test_full_b_c1_diagnostic_loop_real_models_direct16(
    monkeypatch, tmp_path, variant, updates
):
    cfg = config()
    monkeypatch.setattr(cli, "CONFIG", cfg)
    e = engine(variant=variant)
    monkeypatch.setattr(cli, "make_engine", lambda *a: e)
    monkeypatch.setattr(cli, "PADData", SyntheticData)
    r5.atomic_json(tmp_path / "binding.json", {"test": "synthetic"})
    budget = dict(
        wall_seconds=3600,
        checkpoint_bytes=2**30,
        max_optimizer_attempts=updates,
        max_sampled_positions=48,
    )
    before = snapshot(e.model.public)
    files, state = cli.run_training(
        tmp_path, {"data_root": "synthetic"}, 8500, torch.device("cpu"), variant, budget
    )
    summary = read(tmp_path / "training_summary.json")
    assert state == "completed" and summary["attempted_updates"] == updates
    assert (
        summary["successful_ed_updates"] == 3
        and summary["sampled_positions_attempted"] == 48
    )
    assert summary["successful_p_updates"] == (2 if variant == "learned" else 0)
    assert same(e.model.public, before) == (variant == "fixed_gray")
    assert len(list(tmp_path.glob("generator-*.pt"))) == 3
    assert all((tmp_path / f).exists() for f in files)
    report = read(tmp_path / "c1-000002-diagnostic.json")
    assert report["images"] == 6 and report["role"] == "val_tune"
    assert not report["privacy_evaluated"] and report["selected_checkpoint"] is None
    assert len(report["medical"]["public_gray"]["recall"]) == 6


def test_wall_budget_stops_before_any_update(monkeypatch, tmp_path):
    e = engine()
    monkeypatch.setattr(cli, "make_engine", lambda *a: e)
    monkeypatch.setattr(cli, "PADData", SyntheticData)
    files, state = cli.run_training(
        tmp_path,
        {"data_root": "synthetic"},
        8500,
        torch.device("cpu"),
        "learned",
        dict(
            wall_seconds=1e-9,
            checkpoint_bytes=2**30,
            max_optimizer_attempts=5,
            max_sampled_positions=48,
        ),
    )
    assert state == "stopped" and e.ed_updates == e.p_updates == 0
    assert read(tmp_path / "training_summary.json")["stop_reason"] == "wall_ceiling"

    # Reaching the last update without being able to commit its checkpoint is stopped.
    final = engine()
    final.cfg.update(baseline_updates=0, c1_cycles=1)
    monkeypatch.setattr(cli, "make_engine", lambda *a: final)
    limited = tmp_path / "storage-limited"
    limited.mkdir()
    _, storage_state = cli.run_training(
        limited,
        {"data_root": "synthetic"},
        8500,
        torch.device("cpu"),
        "learned",
        dict(
            wall_seconds=3600,
            checkpoint_bytes=1,
            max_optimizer_attempts=2,
            max_sampled_positions=16,
        ),
    )
    assert final.position == final.total and storage_state == "stopped"
    assert not list(limited.glob("generator-*.pt"))


def test_full16_verify_backward_no_updates_and_initial_diagnostics(
    monkeypatch, tmp_path
):
    # Keep actual network and pixel-loss path. Enlarge only this fixture so q is 8x32x32.
    # Full batch uses32px; the one receiver serialization sample is tested separately at512.
    e = engine()
    before = snapshot(e.model)
    monkeypatch.setattr(cli.old, "verify", lambda *a: [])
    monkeypatch.setattr(cli, "make_engine", lambda *a: e)
    monkeypatch.setattr(cli, "PADData", SyntheticData)
    original = cli.public_forward

    def receiver512(model, x, v):
        x = torch.nn.functional.interpolate(x, size=(512, 512))
        v = torch.ones_like(x[:, :1])
        return original(model, x, v)

    monkeypatch.setattr(cli, "public_forward", receiver512)
    files = cli.verify(tmp_path, {"data_root": "synthetic"}, 8500, torch.device("cpu"))
    report = read(tmp_path / "r5-verification.json")
    assert report["batch_size"] == 16 and report["accumulation_steps"] == 1
    assert (
        report["verified"]
        and not report["model_parameters_updated"]
        and same(e.model, before)
    )
    assert (
        not e.ed.state and not e.p.state and all((tmp_path / f).exists() for f in files)
    )


def test_published_config_matches_engine():
    from pathlib import Path

    path = Path(__file__).parents[1] / "configs/grayscale-r5-direct16.json"
    assert json.loads(path.read_text()) == r5.CONFIG


def test_checkpoint_resume_full_runner_and_unclean_cost_rejection(
    monkeypatch, tmp_path
):
    cfg = config()
    monkeypatch.setattr(cli, "CONFIG", cfg)
    monkeypatch.setattr(cli, "PADData", SyntheticData)
    original = engine()
    t = deepcopy(original.teacher)
    monkeypatch.setattr(cli, "make_engine", lambda *a: original)
    budget = dict(
        wall_seconds=3600,
        checkpoint_bytes=2**30,
        max_optimizer_attempts=5,
        max_sampled_positions=48,
    )
    reg = {"source_commit": "local-test", "resources": {}}
    binding = {
        **reg,
        "config": cfg,
        "parents": {},
        "options": {"budget_sha256": cli.argsafe_budget_hash(budget)},
    }
    first = tmp_path / "first"
    first.mkdir()
    r5.atomic_json(first / "binding.json", binding)
    (first / "STOP_AFTER_CHECKPOINT").touch()
    _, state = cli.run_training(
        first, {"data_root": "synthetic"}, 8500, torch.device("cpu"), "learned", budget
    )
    assert state == "stopped" and original.position == 1
    checkpoint = first / read(first / "latest.json")["checkpoint"]
    saved, costs = cli.read_checkpoint(checkpoint, budget, reg)
    second = tmp_path / "second"
    second.mkdir()
    r5.atomic_json(second / "binding.json", binding)
    resumed = engine(t=t)
    monkeypatch.setattr(cli, "make_engine", lambda *a: resumed)
    _, state = cli.run_training(
        second,
        {"data_root": "synthetic"},
        8500,
        torch.device("cpu"),
        "learned",
        budget,
        saved,
        costs,
    )
    assert (
        state == "completed"
        and resumed.attempted_updates == 5
        and resumed.seen.sum() == 48
    )
    assert read(second / "cost.json")["checkpoint_bytes"] > costs["checkpoint_bytes"]
    assert read(second / "cost.json")["elapsed_seconds"] > costs["elapsed_seconds"]
    cost = read(first / "cost.json")
    cost["operation_in_progress"] = True
    r5.atomic_json(first / "cost.json", cost)
    with pytest.raises(ValueError, match="Unclean interruption"):
        cli.read_checkpoint(checkpoint, budget, reg)


def test_no_implicit_train_or_unbudgeted_launch(monkeypatch):
    monkeypatch.setattr(cli, "registration", lambda: {"parents": {}})
    monkeypatch.setattr(cli, "base_parents", lambda reg: {})
    with pytest.raises(SystemExit):
        cli.main(["train"])
    with pytest.raises(SystemExit):
        cli.main(["start"])
