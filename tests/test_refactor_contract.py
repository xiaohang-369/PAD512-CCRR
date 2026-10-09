"""Structural regression checks. None of these prove trained medical capability."""

import importlib
import json
import pkgutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import ccrr
from ccrr.cli import main
from ccrr.codec.entropy import EntropyModel, SourceCodec
from ccrr.codec.quantization import quantize_latent, dequantize_latent
from ccrr.config import Profile
from ccrr.errors import CCRRError
from ccrr.representation.inn import INN
from ccrr.runtime.checkpoints import load_checkpoint, save_checkpoint
from ccrr.runtime.smoke import run as smoke
from ccrr.runtime.system import CCRR, save_public
from ccrr.tensor_ops import public_adapter, q8
from ccrr.training import runner
from ccrr.training.optimization import optimizer, schedule, update


def test_every_python_package_imports():
    modules = list(pkgutil.walk_packages(ccrr.__path__, ccrr.__name__ + "."))
    assert len(modules) >= 40
    for module in modules:
        assert importlib.import_module(module.name)


def test_cli_help_inspect_and_missing_resource_preflight(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "preflight" in capsys.readouterr().out
    assert main(["inspect"]) is None  # Preserve the original success return.
    assert "python" in json.loads(capsys.readouterr().out)
    cfg = {
        "device": "cpu",
        "output": str(tmp_path),
        "weights": {
            "panderm": {
                "path": None,
                "sha256": None,
                "source": None,
                "source_sha256": None,
            }
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main(["preflight", "--config", str(path)]) == 2
    report = json.loads((tmp_path / "preflight.json").read_text())
    assert not report["ready_for_teacher_training"]
    assert not report["formal_training_started"]
    assert "PanDerm verified official source" in report["missing"]


def test_512_nontrivial_inn_inverse_and_parameter_gradients():
    torch.manual_seed(73)
    model = INN()
    for blocks in model.levels:
        for block in blocks:
            torch.nn.init.normal_(block.B, std=0.01)
            torch.nn.init.normal_(block.net[-1].weight, std=0.002)
    x = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        u, z = model(x)
        exact = model.inverse(u, z)
    assert u.shape == (1, 1, 512, 512) and z.shape == (1, 2, 512, 512)
    assert (exact - x).abs().max() < 1e-4
    assert (exact - x).square().mean().sqrt() < 1e-5
    # A perturbed inverse exercises all inverse blocks, not a zero-loss identity.
    model.inverse(u + 0.001, z + 0.001).square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert model.levels[0][0].net[-1].weight.grad.abs().sum() > 0


def test_512_frozen_inverse_preserves_public_input_gradient():
    torch.manual_seed(74)
    model = INN().requires_grad_(False)
    x = torch.rand(1, 3, 512, 512)
    u, z = model(x)
    g = q8(u.sigmoid()).detach().requires_grad_(True)
    n = quantize_latent(z, 1 / 32)
    rec = model.inverse(public_adapter(g), dequantize_latent(n, 1 / 32))
    (rec - x).square().mean().backward()
    assert (
        g.grad is not None and torch.isfinite(g.grad).all() and g.grad.abs().sum() > 0
    )
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("level", [1, 2, 3])
def test_shared_quantizer_ties_gradient_and_exact_source_roundtrip(level):
    profile = Profile(size=256)
    step = profile.step(level)
    values = torch.tensor([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5])
    z = (values * step).requires_grad_(True)
    n = quantize_latent(z, step)
    assert torch.equal(n, torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
    dequantize_latent(n, step).sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))
    codec = SourceCodec(profile, bytes(32), EntropyModel().export())
    full = torch.tensor([-256.0, -255.0, -1.0, 0.0, 1.0, 255.0, 256.0, 2**20 - 1.0])
    full = full.repeat(profile.symbols // len(full)).reshape(1, 2, 256, 256)
    encoded = codec.encode(full, level)
    decoded, latent = codec.decode(encoded, level)
    assert np.issubdtype(decoded.numpy().dtype, np.integer)
    assert torch.equal(decoded, full.to(decoded.dtype))
    assert torch.equal(latent, dequantize_latent(full, step))


def test_checkpoint_resume_reproduces_next_optimizer_update(tmp_path):
    # Real AdamW moments and Torch RNG; not a test of full-stage resume.
    torch.manual_seed(15)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 6), torch.nn.Dropout(0.2), torch.nn.Linear(6, 1)
    )
    opt = optimizer([model], [1e-3])
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])

    def step(m, o, index):
        o.zero_grad()
        loss = m(x).square().mean()
        loss.backward()
        schedule(o, index, 10)
        update(o)
        return loss.detach()

    step(model, opt, 0)
    path = tmp_path / "state.pt"
    save_checkpoint(path, {"model": model}, opt, 1, "engineering", {})
    expected_loss = step(model, opt, 1)
    restored = torch.nn.Sequential(
        torch.nn.Linear(3, 6), torch.nn.Dropout(0.2), torch.nn.Linear(6, 1)
    )
    opt2 = optimizer([restored], [1e-3])
    state = load_checkpoint(path, {"model": restored}, opt2)
    actual_loss = step(restored, opt2, state["step"])
    assert torch.equal(actual_loss, expected_loss)
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters())
    )
    assert opt2.param_groups[0]["lr"] == opt.param_groups[0]["lr"]


@pytest.mark.parametrize(
    "fixed_gray,joint_b,expected",
    [
        (False, False, list("ABCDEFGHJ")),
        (False, True, list("ABCDEFGHIJ")),
        (True, False, list("ABEFGHJ")),
    ],
)
def test_dispatch_order_only(monkeypatch, tmp_path, fixed_gray, joint_b, expected):
    """Call recording checks dispatch only; deliberately no scientific gate results."""
    events = []
    context = SimpleNamespace(
        cfg={"joint_b": joint_b},
        system=SimpleNamespace(fixed_gray=fixed_gray),
        output=tmp_path,
        allocator=SimpleNamespace(close=lambda: events.append("closed")),
    )
    for module, name, label in [
        (runner.teachers, "stage_a", "A"),
        (runner.purifier, "stage_b", "B"),
        (runner.communication, "stage_f", "F"),
        (runner.calibration, "stage_g", "G"),
        (runner, "final_evaluation", "J"),
    ]:
        monkeypatch.setattr(module, name, lambda e, label=label: events.append(label))
    monkeypatch.setattr(
        runner.representation,
        "train_representation",
        lambda e, stage: events.append(stage),
    )
    monkeypatch.setattr(runner.joint, "joint", lambda e, stage: events.append(stage))
    runner.run(context)
    assert events == expected + ["closed"]
    assert not (tmp_path / "acceptance.json").exists()


def test_missing_dependency_stops_before_training_and_closes_allocator(tmp_path):
    (tmp_path / "acceptance.json").write_text("{}")
    closed = []
    context = SimpleNamespace(
        output=tmp_path, allocator=SimpleNamespace(close=lambda: closed.append(True))
    )
    with pytest.raises(CCRRError, match="Missing accepted dependency"):
        runner.run(context, "B")
    report = json.loads((tmp_path / "stopped.json").read_text())
    assert (
        report["stage"] == "B" and not report["training_complete"] and closed == [True]
    )


def test_real_512_smoke_keeps_untrained_extraction_failure(tmp_path, capsys):
    smoke(tmp_path, 512)
    capsys.readouterr()
    report = json.loads((tmp_path / "smoke.json").read_text())
    assert report["size"] == 512 and report["writer_gradient_norm"] > 0
    assert report["max_integer_modification"] <= 4.0001
    assert report["digital_oracle_records"] == {"C": "Success", "M": "Success"}
    assert report["actual_blind_decoder"] == {
        "C": "Extraction Failure",
        "M": "Extraction Failure",
    }
    assert not report["training_started"] and not report["formal_experiment_passed"]


def test_cli_export_and_receive_real_untrained_bundle(tmp_path, capsys):
    """Exercise serialization/CLI plumbing; blind extraction must still fail."""
    torch.manual_seed(20261008)
    system = CCRR(Profile(size=256))
    checkpoint = tmp_path / "system.pt"
    torch.save(system.state_dict(), checkpoint)
    cdf = tmp_path / "cdf.npy"
    np.save(cdf, system.entropy.export())
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"profile": {"size": 256, "research": True}}))
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"scope":"untrained engineering fixture"}')
    bundle = tmp_path / "bundle"
    assert (
        main(
            [
                "export",
                "--checkpoint",
                str(checkpoint),
                "--cdf",
                str(cdf),
                "--config",
                str(config),
                "--evidence",
                str(evidence),
                "--out",
                str(bundle),
            ]
        )
        == 0
    )
    public = tmp_path / "public.png"
    save_public(torch.full((1, 1, 256, 256), 128 / 255), public)
    recovered = tmp_path / "received"
    assert (
        main(
            [
                "receive",
                "--manifest",
                str(bundle / "manifest.json"),
                "--image",
                str(public),
                "--out",
                str(recovered),
            ]
        )
        == 0
    )
    report = json.loads((recovered / "status.json").read_text())
    assert report["C"]["status"] == report["M"]["status"] == "Extraction Failure"
    assert not (recovered / "recovered.png").exists()
    assert not (recovered / "metadata.json").exists()
    capsys.readouterr()
