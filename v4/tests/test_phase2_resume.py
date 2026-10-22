"""Continuation preserves Adam, next batch, original targets and provenance."""

from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch

from ccrr_v4.phase1.optim import TF1Adam, ContinuousBatches
from ccrr_v4.phase2_resume import load_resume, restore_training_state
from ccrr_v4.phase2_objectives import validate_config
from ccrr_v4.verify import digest, state_digest


def write(path, obj):
    path.write_text(json.dumps(obj))


class TinyED(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 2)
        self.decoder = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def fixture(tmp_path):
    cfg = json.loads((Path(__file__).parents[1] / "configs/phase2-g1.json").read_text())
    cfg["epochs"] = 2
    cfg["gpu_ids"] = [0]
    model = TinyED()
    opt = TF1Adam(model.parameters(), lr=cfg["learning_rate"])
    stream = ContinuousBatches(17, 16, cfg["seed"])
    for _ in range(4):
        stream.next()
        opt.zero_grad()
        model(torch.ones(2, 3)).square().mean().backward()
        opt.step()
    binding = {"config": cfg, "parents": {}}
    ck = dict(
        schema="ccrr-v4-g1-checkpoint-1",
        model=model.state_dict(),
        optimizer=opt.state_dict(),
        epoch=2,
        step=4,
        binding=binding,
    )
    cp = tmp_path / "g1-epoch-002.pt"
    torch.save(ck, cp)
    bank_path = tmp_path / "fit-bank.pt"
    torch.save(dict(target=torch.tensor([[0.1, 0.9]]), rgb=torch.ones(1, 3)), bank_path)
    preflight = tmp_path / "preflight.json"
    write(
        preflight,
        dict(
            passed=True,
            binding=binding,
            bank=dict(file=bank_path.name, sha256=digest(bank_path)),
        ),
    )
    write(
        tmp_path / "preflight-binding.json",
        dict(path=str(preflight), sha256=digest(preflight)),
    )
    write(tmp_path / "config.json", cfg)
    write(tmp_path / "binding.json", binding)
    write(tmp_path / "status.json", dict(state="completed"))
    write(
        tmp_path / "report.json",
        dict(
            training_completed=True,
            frozen_models_unchanged=True,
            parents_unchanged=True,
            binding=binding,
            final_checkpoint=dict(file=cp.name, sha256=digest(cp), epoch=2, step=4),
        ),
    )
    new_cfg = {**cfg, "epochs": 4, "resume": dict(run=str(tmp_path), sha256=digest(cp))}
    ctx = dict(model=TinyED(), parents={}, arrays={"fit": list(range(17))})
    return new_cfg, ctx, model, opt, stream


def test_resume_matches_uninterrupted_next_update_and_stream(tmp_path):
    cfg, ctx, original, opt, stream = fixture(tmp_path)
    resume = load_resume(cfg, ctx)
    assert state_digest(ctx["model"]) == state_digest(original)
    resumed_opt = TF1Adam(ctx["model"].parameters(), lr=cfg["learning_rate"])
    epoch, step, resumed_stream = restore_training_state(resumed_opt, cfg, 17, resume)
    assert (epoch, step, resumed_stream.position) == (2, 4, 64)
    for _ in range(6):
        assert resumed_stream.next() == stream.next()
        for model, optimizer in ((original, opt), (ctx["model"], resumed_opt)):
            optimizer.zero_grad()
            model(torch.ones(2, 3)).square().mean().backward()
            optimizer.step()
        assert state_digest(original) == state_digest(ctx["model"])
    for a, b in zip(opt.state.values(), resumed_opt.state.values()):
        for key in ("m", "v"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    bank = torch.load(resume["bank_path"], weights_only=True)
    torch.testing.assert_close(bank["target"], torch.tensor([[0.1, 0.9]]))
    assert len(ctx["parents"]) == 8
    assert not resume["evidence"]["bitwise_uninterrupted_equivalence_claimed"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("learning_rate", 1e-5),
        ("seed", 1),
        ("epochs", 2),
        ("update_modules", ["E1"]),
        ("supervision", {"medical": 0.2, "identity": 0.1, "medical_kd": 0.5}),
    ],
)
def test_reject_changed_experiment_or_non_extension(tmp_path, key, value):
    cfg, ctx, *_ = fixture(tmp_path)
    cfg[key] = value
    with pytest.raises(ValueError):
        load_resume(cfg, ctx)


@pytest.mark.parametrize(
    "target", ["checkpoint", "bank", "preflight", "parents", "unfinished"]
)
def test_reject_changed_bound_files(tmp_path, target):
    cfg, ctx, *_ = fixture(tmp_path)
    if target == "checkpoint":
        (tmp_path / "g1-epoch-002.pt").write_bytes(b"changed")
    elif target == "bank":
        (tmp_path / "fit-bank.pt").write_bytes(b"changed")
    elif target == "preflight":
        (tmp_path / "preflight.json").write_text("{}")
    elif target == "parents":
        ctx["parents"]["unexpected"] = "00"
    else:
        write(tmp_path / "status.json", dict(state="running"))
    with pytest.raises(ValueError):
        load_resume(cfg, ctx)


def test_reject_reset_optimizer_even_with_matching_file_hash(tmp_path):
    cfg, ctx, *_ = fixture(tmp_path)
    path = tmp_path / "g1-epoch-002.pt"
    ck = torch.load(path, weights_only=True)
    ck["optimizer"]["param_groups"][0]["updates"] = 0
    torch.save(ck, path)
    cfg["resume"]["sha256"] = digest(path)
    report = json.loads((tmp_path / "report.json").read_text())
    report["final_checkpoint"]["sha256"] = digest(path)
    write(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="optimizer"):
        load_resume(cfg, ctx)


def test_new_config_only_extends_joint_run():
    root = Path(__file__).parents[1] / "configs"
    old = json.loads((root / "phase2-g1.json").read_text())
    new = json.loads((root / "phase2-g1-resume60.json").read_text())
    validate_config(new)
    assert {k for k in old if old[k] != new[k]} == {"epochs", "gpu_ids"}
    assert new["epochs"] == 60 and new["update_modules"] == ["E1", "D1"]
    assert new["gpu_ids"] == [1, 2]


def test_resume_allows_physical_gpu_remap_but_not_world_size_change(tmp_path):
    cfg, ctx, original, *_ = fixture(tmp_path)
    cfg["gpu_ids"] = [2]
    validate_config(cfg)
    resume = load_resume(cfg, ctx)
    assert resume["evidence"]["start_step"] == 4
    assert state_digest(ctx["model"]) == state_digest(original)
    cfg["gpu_ids"] = [1, 2]
    with pytest.raises(ValueError, match="GPU count"):
        load_resume(cfg, ctx)


def test_no_resume_keeps_initial_stream_and_empty_adam(tmp_path):
    cfg, ctx, *_ = fixture(tmp_path)
    cfg.pop("resume")
    assert load_resume(cfg, ctx) is None
    opt = TF1Adam(ctx["model"].parameters())
    epoch, step, stream = restore_training_state(opt, cfg, 17, None)
    assert (epoch, step, stream.position) == (0, 0, 0)
    assert opt.param_groups[0]["updates"] == 0 and not opt.state


def test_bank_refresh_keeps_g0_target_not_resumed_g1(tmp_path, monkeypatch):
    import ccrr_v4.phase2_g1 as g1

    class Data:
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return dict(x=torch.ones(3, 2, 2) * i, rect=(0, 0, 2, 2), label=i)

    class Model(torch.nn.Module):
        def encoder(self, x):
            return x[:, :1]

    class Reid:
        def __call__(self, x, rects):
            return {"embedding": x.mean((2, 3)).repeat(1, 3)}

    original = dict(
        target=torch.tensor([[0.1, 0.9], [0.2, 0.8]]),
        rgb=torch.randn(2, 3),
        labels=torch.tensor([0, 1]),
        rects=[(0, 0, 2, 2)] * 2,
        g1=torch.ones(2, 3) * 99,
    )
    ctx = dict(
        datasets={"fit": Data()}, model=Model(), reid=Reid(), arrays={"fit": [0, 1]}
    )
    monkeypatch.setattr(g1, "pad512_bytes", lambda x: int(x[0, 0, 0]))
    monkeypatch.setattr(g1, "digital_roundtrip", lambda x: x)
    from contextlib import nullcontext

    monkeypatch.setattr(g1, "numeric_policy", lambda _: nullcontext())
    refreshed = g1.build_fit_bank(
        ctx,
        {"bank_log_every_images": 100},
        tmp_path,
        torch.device("cpu"),
        0,
        1,
        "resume",
        original,
    )
    for key in ("target", "rgb", "labels", "rects"):
        assert refreshed[key] is original[key]
    assert not torch.equal(refreshed["g1"], original["g1"])


def test_preflight_then_continuation_loop_does_not_restart_or_overwrite(
    tmp_path, monkeypatch
):
    import ccrr_v4.phase2_g1 as g1

    parent = tmp_path / "parent"
    parent.mkdir()
    cfg, ctx, _, _, _ = fixture(parent)
    old_files = {p.name: digest(p) for p in parent.iterdir()}
    ctx.update(
        baseline=deepcopy(ctx["model"]),
        frozen={},
        frozen_hashes={},
        author_cfg={"stages": [{}, {"weights": {"reconstruction": 3}}]},
    )

    class Data:
        def class_weights(self):
            return torch.ones(6)

    ctx["datasets"] = {"fit": Data()}
    class CPUFacade:
        def device(self, *args):
            return torch.device("cpu")

        def __getattr__(self, name):
            return getattr(torch, name)

    monkeypatch.setattr(g1, "torch", CPUFacade())
    monkeypatch.setattr(g1.torch.cuda, "device_count", lambda: 1)
    for name in ("set_device", "reset_peak_memory_stats"):
        monkeypatch.setattr(g1.torch.cuda, name, lambda *a: None)
    monkeypatch.setattr(g1.torch.cuda, "max_memory_allocated", lambda *a: 0)
    monkeypatch.setattr(g1, "load_context", lambda *a: deepcopy(ctx))
    monkeypatch.setattr(g1, "source_check", lambda: {})
    monkeypatch.setattr(
        g1, "batch_tensor", lambda arrays, ids, device: torch.ones(len(ids), 3)
    )
    banks = []

    def bank(context, config, output, device, rank, world, tag, initial=None):
        assert initial is not None
        torch.testing.assert_close(initial["target"], torch.tensor([[0.1, 0.9]]))
        banks.append(tag)
        return initial

    monkeypatch.setattr(g1, "build_fit_bank", bank)

    def backward(context, config, enc, dec, x, ids, bank, world):
        total = dec(enc(x)).square().mean()
        total.backward()
        return total, {"mse": total.detach()}, {"medical": 1, "identity": 1}

    monkeypatch.setattr(g1, "backward_step", backward)
    evaluated = []

    def evaluate(context, output, device, rank, world, epoch):
        evaluated.append(epoch)
        return {"epoch": epoch}

    monkeypatch.setattr(g1, "evaluate", evaluate)
    pre = tmp_path / "pre"
    pre.mkdir()
    g1.run(cfg, pre, "preflight")
    write(pre / "status.json", {"state": "completed"})
    report = json.loads((pre / "report.json").read_text())
    assert report["optimizer_updates"] == 0 and report["optimizer_state_updates"] == 4
    assert report["start_epoch"] == 2 and not report["initial_G1_equals_G0"]
    train = tmp_path / "train"
    train.mkdir()
    g1.run(cfg, train, "train", pre / "report.json")
    final = json.loads((train / "report.json").read_text())
    assert final["optimizer_updates"] == 8 and final["optimizer_updates_this_run"] == 4
    assert final["final_checkpoint"]["epoch"] == 4
    rows = [json.loads(s) for s in (train / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [3, 4]
    assert [r["step"] for r in rows] == [6, 8]
    assert evaluated == [2, 3, 4] and banks == ["initial", "epoch004"]
    ck = torch.load(train / "g1-epoch-004.pt", weights_only=True)
    assert (
        ck["stream_position"] == 128
        and ck["optimizer"]["param_groups"][0]["updates"] == 8
    )
    assert old_files == {p.name: digest(p) for p in parent.iterdir()}
