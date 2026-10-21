"""Source math, global-batch reduction and phase-1 control-flow regression tests."""

import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from ccrr_v4.phase1.optim import TF1Adam, polynomial_lr, ContinuousBatches
from ccrr_v4.phase1.vgg import AuthorVGG, LAYERS, read_author_npy
from ccrr_v4.phase1.data import load_config, pad256
from ccrr_v4.phase1 import runner
from ccrr_v4.losses import source_terms
from ccrr_v4.verify import digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_tf_adam_matches_uncorrected_epsilon_and_resumes(tmp_path):
    a = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    optimizer = TF1Adam([a])
    expected = a.detach().numpy().copy()
    m, v = np.zeros(2), np.zeros(2)
    for step, grad in enumerate(([1e-8, -2e-8], [3e-7, -1e-6], [2.0, -4.0]), 1):
        a.grad = torch.tensor(grad, dtype=torch.float64)
        optimizer.step()
        m, v = 0.9 * m + 0.1 * np.array(grad), 0.999 * v + 0.001 * np.square(grad)
        rate = 0.0002 * np.sqrt(1 - 0.999**step) / (1 - 0.9**step)
        expected -= rate * m / (np.sqrt(v) + 1e-8)
        np.testing.assert_allclose(a.detach().numpy(), expected, rtol=0, atol=1e-14)
    clone = a.detach().clone().requires_grad_()
    opt2 = TF1Adam([clone])
    opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    a.grad = torch.ones_like(a) * 1e-8
    clone.grad = a.grad.clone()
    optimizer.step()
    opt2.step()
    assert torch.equal(a, clone)
    # Stage 2 must be a genuinely new optimizer, not just a changed loss weight.
    fresh = TF1Adam([clone])
    assert fresh.param_groups[0]["updates"] == 0 and not fresh.state


def test_source_ceil_updates_floor_decay_and_full_batches():
    stream = ContinuousBatches(1433, 8, 20261021)
    first = [i for _ in range(180) for i in stream.next()]
    assert len(first) == 1440 and len(set(first[:1433])) == 1433
    assert stream.position == 1440
    resumed = ContinuousBatches(1433, 8, 20261021, position=1440)
    assert resumed.next() == stream.next()
    decay = (1433 // 8) * 120
    assert decay == 21480
    assert polynomial_lr(0, decay) == pytest.approx(2e-4)
    assert polynomial_lr(decay, decay) == 2e-6
    assert polynomial_lr(21600, decay) == 2e-6


def test_float32_quant_grid_uses_author_numpy_construction():
    levels = torch.from_numpy(
        (np.arange(256, dtype=np.float64) / 127.5 - 1).astype(np.float32)
    )
    g = levels.reshape(1, 1, 16, 16).clone().requires_grad_()
    x = torch.zeros(1, 3, 16, 16)
    penalty = source_terms(x, g, x)["quantization"]
    assert penalty == 0
    dense = (g[..., None] - levels).abs().amin(-1).mean()
    assert torch.equal(
        torch.autograd.grad(penalty, g, retain_graph=True)[0],
        torch.autograd.grad(dense, g)[0],
    )


def test_vgg_bgr_means_gray_replication_endpoint_and_input_gradient():
    # Center-only filters propagate the first B channel; all other weights zero.
    data = {}
    for name, cin, cout in LAYERS:
        w = np.zeros((3, 3, cin, cout), dtype=np.float32)
        w[1, 1, 0, :] = 1
        data[name] = (w, np.zeros(cout, dtype=np.float32))
    net = AuthorVGG(data)
    x = (
        torch.tensor([0.2, 0.4, 0.8])
        .view(1, 3, 1, 1)
        .expand(1, 3, 256, 256)
        .clone()
        .requires_grad_()
    )
    out = net(x)
    assert out.shape == (1, 512, 28, 28)
    assert torch.allclose(out, torch.full_like(out, (0.8 + 1) / 2 * 255 - 103.939))
    out.mean().backward()
    assert x.grad[:, 2].abs().sum() > 0
    assert x.grad[:, :2].abs().sum() == 0
    gray = torch.full((1, 1, 256, 256), 0.3, requires_grad=True)
    gout = net(gray)
    assert torch.allclose(gout, torch.full_like(gout, 1.3 / 2 * 255 - 103.939))
    gout.mean().backward()
    assert gray.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in net.parameters())


def test_npy_restricted_reader_hash_and_unregistered_weights(tmp_path):
    p = tmp_path / "weights.npy"
    np.save(p, {"example": (np.zeros((1,), np.float32), np.ones((1,), np.float32))})
    data = read_author_npy(p, digest(p))
    assert data["example"][1][0] == 1
    with pytest.raises(ValueError, match="SHA256"):
        read_author_npy(p, "0" * 64)
    with pytest.raises(ValueError, match="Unregistered"):
        AuthorVGG.from_file(p, digest(p))


def test_config_prevents_silent_loss_and_schedule_changes(tmp_path):
    c = load_config(ROOT / "configs/phase1-train.json")
    c["runtime"]["gpu_ids"] = [0, 2]
    p = tmp_path / "config.json"
    p.write_text(json.dumps(c))
    assert load_config(p)["runtime"]["gpu_ids"] == [0, 2]
    c["stages"][1]["weights"]["variation"] = 0.25
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="stages"):
        load_config(p)


def test_data_adapter_and_png_boundary():
    x = torch.full((3, 512, 512), 128 / 255)
    a = pad256(x)
    assert a.shape == (256, 256, 3) and a.dtype == np.uint8 and np.all(a == 128)
    g = torch.tensor([-1.0, -0.123, 0.0, 0.123, 1.0]).view(1, 1, 1, 5)
    received = runner.digital_roundtrip(g)
    expected = torch.floor((g + 1) * 127.5 + 0.5) / 127.5 - 1
    assert torch.equal(received, expected)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 1, 1), nn.Tanh())
        self.decoder = nn.Sequential(nn.Conv2d(1, 3, 1), nn.Tanh())

    def forward(self, x):
        g = self.encoder(x)
        return g, self.decoder(g)


class ToyVGG(nn.Module):
    def forward(self, x):
        return x.mean(1, keepdim=True)


def test_training_controller_90_30_reset_checkpoint_resume_and_stop(
    tmp_path, monkeypatch
):
    cfg = load_config(ROOT / "configs/phase1-train.json")
    cfg["runtime"]["device"] = "cpu"
    cfg["runtime"]["cpu_threads"] = 2
    cfg["log_every_steps"] = 1000
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}")
    array = np.random.default_rng(6).integers(0, 256, (8, 8, 8, 3), dtype=np.uint8)
    p = dict(
        vgg=dict(path="test-only", sha256="test-only"),
        schedule_counts=dict(
            fit_images=8,
            global_batch=8,
            steps_per_epoch=1,
            decay_steps=120,
            total_steps=120,
        ),
    )
    monkeypatch.setattr(
        runner,
        "load_prepared",
        lambda *args: (p, {"fit": array, "val_tune": array[:1]}),
    )
    monkeypatch.setattr(runner, "OriginalGrayscale", ToyModel)
    monkeypatch.setattr(runner.AuthorVGG, "from_file", lambda *args: ToyVGG())
    first = tmp_path / "first"
    runner.run(config, prepared, first, "train")
    ck90 = torch.load(first / "epoch-090.pt", weights_only=True)
    ck120 = torch.load(first / "epoch-120.pt", weights_only=True)
    assert ck90["optimizer"]["param_groups"][0]["updates"] == 90
    assert ck120["optimizer"]["param_groups"][0]["updates"] == 30
    assert ck120["step"] == 120 and ck120["stage"] == 2
    assert json.loads((first / "report.json").read_text())["next_phase"] == "stop"
    resumed = tmp_path / "resumed"
    runner.run(config, prepared, resumed, "train", first / "epoch-090.pt")
    restored = torch.load(resumed / "epoch-120.pt", weights_only=True)
    assert all(torch.equal(restored["model"][k], v) for k, v in ck120["model"].items())
    with pytest.raises(FileExistsError):
        runner.run(config, prepared, first, "train")


class ScalarGray(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.6))

    def forward(self, pattern):
        return pattern * self.scale


def ddp_worker(rank, init_path, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + init_path, rank=rank, world_size=2
    )
    model = DDP(ScalarGray())
    pattern = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    # Rank-local TV deltas have opposite signs. Abs-before-reduce is wrong.
    x = pattern.expand(-1, 3, -1, -1) * (0.1 if rank == 0 else 0.9)
    g = model(pattern)
    loss = source_terms(x, g, x, distributed=True)["variation"]
    loss.backward()
    torch.save(
        {"loss": loss.detach(), "grad": model.module.scale.grad},
        Path(output) / f"{rank}.pt",
    )
    dist.destroy_process_group()


def test_two_process_global_tv_matches_single_global_batch(tmp_path):
    mp.spawn(
        ddp_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True
    )
    model = ScalarGray()
    patterns = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]]).repeat(2, 1, 1, 1)
    x = patterns.expand(-1, 3, -1, -1) * torch.tensor([0.1, 0.9])[:, None, None, None]
    loss = source_terms(x, model(patterns), x)["variation"]
    loss.backward()
    for rank in range(2):
        observed = torch.load(tmp_path / f"{rank}.pt", weights_only=True)
        assert torch.allclose(observed["loss"], loss, atol=1e-7)
        assert torch.allclose(observed["grad"], model.scale.grad, atol=1e-7)
