"""Check digital forward, input-gradient freezing and exact staged chain rule."""

from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ccrr_v4.original import OriginalGrayscale, quantize8
from ccrr_v4.phase2_objectives import (
    digital_supervision,
    medical_objective,
    retrieval_confusion,
    validate_config,
    encoder_only,
    gray_objective,
)
from ccrr_v4.phase2_g1 import (
    backward_step,
    configure_training,
    training_gradients,
    check_frozen,
)
from ccrr_v4.phase1.optim import TF1Adam
from ccrr_v4.losses import author_loss
from ccrr_v4.verify import state_digest


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 1, 3, padding=1), nn.Tanh())
        self.decoder = nn.Sequential(nn.Conv2d(1, 3, 3, padding=1), nn.Tanh())


class TinyPan(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3, padding=1)

    def forward(self, x, rects):
        return self.conv(x), torch.ones_like(x), None


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 6)

    def forward(self, raw, cov, domain):
        return self.linear(raw.mean((2, 3))), raw


class TinyReid(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3, padding=1)

    def forward(self, x, rects):
        return {"embedding": F.normalize(self.conv(x).mean((2, 3)), dim=1)}


class TinyVGG(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x.expand(-1, 3, -1, -1), 2)


def fixture(e1only=False):
    torch.manual_seed(171)
    cfg = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/phase2-g1.json").read_text()
    )
    ctx = dict(
        model=TinyModel(),
        panderm=TinyPan().requires_grad_(False),
        teachers={"8500": TinyHead().requires_grad_(False)},
        reid=TinyReid().requires_grad_(False),
        vgg=TinyVGG(),
        class_weights=torch.tensor([1.0, 2.0, 0.5, 1.0, 1.0, 1.0]),
        author_cfg={
            "stages": [
                {},
                {
                    "weights": dict(
                        reconstruction=3.0,
                        vgg=1e-7,
                        variation=0.1,
                        lightness=1.0,
                        quantization=10.0,
                    )
                },
            ]
        },
    )
    bank = dict(
        rgb=F.normalize(torch.randn(8, 6), dim=1),
        g1=F.normalize(torch.randn(8, 6), dim=1),
        labels=torch.tensor([0, 1, 2, 3, 4, 5, 0, 1]),
        target=torch.randn(8, 6).softmax(1),
        rects=[(0, 0, 8, 8)] * 8,
    )
    if e1only:
        cfg.update(
            schema="ccrr-v4-phase2-g1-e1only-1", update_modules=["E1"], gpu_ids=[1, 3]
        )
    ctx["frozen"] = {
        "panderm": ctx["panderm"],
        "reid": ctx["reid"],
        "teacher": ctx["teachers"]["8500"],
    }
    ctx["frozen_hashes"] = {k: state_digest(v) for k, v in ctx["frozen"].items()}
    configure_training(ctx, cfg)
    return cfg, ctx, bank, torch.rand(4, 3, 8, 8) * 2 - 1


def direct_backward(ctx, cfg, x, bank, ids):
    g = ctx["model"].encoder(x)
    weights = ctx["author_cfg"]["stages"][1]["weights"]
    if encoder_only(cfg):
        original, _ = gray_objective(x, g, ctx["vgg"](x), ctx["vgg"](g), weights)
    else:
        original, _ = author_loss(
            x, g, ctx["model"].decoder(g), ctx["vgg"](x), ctx["vgg"](g), 2, weights
        )
    image = digital_supervision(g)
    raw, cov, _ = ctx["panderm"](image, bank["rects"])
    med = medical_objective(
        ctx["teachers"]["8500"](raw, cov, 1)[0],
        bank["labels"][ids],
        bank["target"][ids],
        ctx["class_weights"],
        cfg["supervision"]["medical_kd"],
    )
    emb = ctx["reid"](image, bank["rects"])["embedding"]
    identity = (
        sum(
            retrieval_confusion(emb, bank[k], ids, cfg["retrieval_temperature"])
            for k in ("rgb", "g1")
        )
        / 2
    )
    loss = (
        original
        + cfg["supervision"]["medical"] * med
        + cfg["supervision"]["identity"] * identity
    )
    loss.backward()
    return loss


@pytest.mark.parametrize("e1only", [False, True])
def test_sequential_vjp_equals_full_graph_all_parameters_and_no_teacher_grads(e1only):
    cfg, ctx, bank, x = fixture(e1only)
    direct = deepcopy(ctx)
    snapshots = {k: state_digest(ctx[k]) for k in ("panderm", "reid")}
    actual, _, norms = backward_step(
        ctx, cfg, ctx["model"].encoder, ctx["model"].decoder, x, [0, 1, 2, 3], bank, 1
    )
    expected = direct_backward(direct, cfg, x, bank, [0, 1, 2, 3])
    torch.testing.assert_close(actual, expected.detach())
    for a, b in zip(ctx["model"].parameters(), direct["model"].parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=1e-6)
        else:
            assert a.grad is None and b.grad is None
    assert all(v > 0 for v in norms.values())
    for key in ("panderm", "reid"):
        assert state_digest(ctx[key]) == snapshots[key]
        assert all(p.grad is None for p in ctx[key].parameters())
    assert all(p.grad is None for p in ctx["teachers"]["8500"].parameters())


def test_ste_has_exact_author_digital_forward_and_only_surrogate_gradient():
    g = torch.linspace(-1, 1, 101).reshape(1, 1, 1, -1).requires_grad_()
    out = digital_supervision(g)
    assert torch.equal(out, (quantize8(g)[1] + 1) / 2)
    out.sum().backward()
    torch.testing.assert_close(g.grad, torch.full_like(g, 0.5), rtol=0, atol=0)


def test_retrieval_excludes_exact_source_and_does_not_backprop_into_gallery():
    torch.manual_seed(10)
    q = torch.randn(1, 6, requires_grad=True)
    bank = torch.randn(5, 6, requires_grad=True)
    loss = retrieval_confusion(q, bank, [2], 0.1)
    changed = bank.detach().clone()
    changed[2] = q.detach()[0] * 1000
    torch.testing.assert_close(
        loss, retrieval_confusion(q, changed, [2], 0.1), rtol=0, atol=0
    )
    loss.backward()
    assert q.grad.abs().sum() > 0 and bank.grad is None
    equal = torch.ones(5, 6)
    assert abs(float(retrieval_confusion(q.detach(), equal, [2], 0.1))) < 1e-6


def test_no_new_generator_parameters_and_reference_unchanged_on_copy_update():
    ref = OriginalGrayscale().eval().requires_grad_(False)
    original_hash = state_digest(ref)
    copy = deepcopy(ref).train().requires_grad_(True)
    assert list(copy.state_dict()) == list(ref.state_dict())
    assert sum(p.numel() for p in copy.parameters()) == 7413312
    with torch.no_grad():
        next(copy.encoder.parameters()).add_(0.01)
    assert state_digest(ref) == original_hash and state_digest(copy) != original_hash


def test_config_keeps_user_scope_but_epochs_and_devices_are_configurable():
    cfg, _, _, _ = fixture()
    validate_config(cfg)
    assert cfg["epochs"] == 30 and cfg["medical_acceptance_threshold"] is None
    alternate = {**cfg, "epochs": 60, "gpu_ids": [2]}
    validate_config(alternate)
    for key, value in (
        ("global_batch_size", 8),
        ("image_size", 256),
        ("medical_acceptance_threshold", 0.02),
        ("gpu_ids", [0, 0]),
        ("supervisor_step", 7000),
        ("next_phase", "phase3"),
    ):
        with pytest.raises(ValueError):
            validate_config({**cfg, key: value})


def ddp_worker(rank, init_file, result, e1only):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + init_file, rank=rank, world_size=2
    )
    try:
        cfg, ctx, bank, x = fixture(e1only)
        encoder = DistributedDataParallel(ctx["model"].encoder)
        decoder = (
            ctx["model"].decoder
            if e1only
            else DistributedDataParallel(ctx["model"].decoder)
        )
        ids = [rank * 2, rank * 2 + 1]
        backward_step(ctx, cfg, encoder, decoder, x[ids], ids, bank, 2)
        torch.save([p.grad for p in ctx["model"].parameters()], result + str(rank))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("e1only", [False, True])
def test_two_rank_gradients_match_full_batch_without_accumulation(tmp_path, e1only):
    import torch.multiprocessing as mp

    mp.spawn(
        ddp_worker,
        args=(str(tmp_path / "init"), str(tmp_path / "grad"), e1only),
        nprocs=2,
        join=True,
    )
    cfg, ctx, bank, x = fixture(e1only)
    direct_backward(ctx, cfg, x, bank, list(range(4)))
    for rank in (0, 1):
        grads = torch.load(str(tmp_path / "grad") + str(rank), weights_only=True)
        for expected, actual in zip(ctx["model"].parameters(), grads):
            if expected.requires_grad:
                torch.testing.assert_close(actual, expected.grad, rtol=2e-5, atol=1e-6)
            else:
                assert actual is None and expected.grad is None


def test_encoder_only_updates_encoder_without_decoder_forward_or_teacher_updates():
    cfg, ctx, bank, x = fixture(True)
    parameters = configure_training(ctx, cfg)
    assert {id(p) for p in parameters} == {
        id(p) for p in ctx["model"].encoder.parameters()
    }
    optimizer = TF1Adam(parameters, lr=cfg["learning_rate"])
    before = state_digest(ctx["model"].encoder)

    def forbidden(*args):
        raise AssertionError("E1-only training must not invoke decoder")

    _, terms, norms = backward_step(
        ctx, cfg, ctx["model"].encoder, forbidden, x, [0, 1, 2, 3], bank, 1
    )
    assert "mse" not in terms
    assert all(v > 0 for v in norms.values())
    assert training_gradients(ctx, cfg)["decoder"] == 0
    optimizer.step()
    assert before != state_digest(ctx["model"].encoder)
    check_frozen(ctx)


def test_gray_objective_preserves_original_gray_values_and_gradients():
    cfg, ctx, _, x = fixture()
    g = ctx["model"].encoder(x)
    weights = ctx["author_cfg"]["stages"][1]["weights"]
    full, old = author_loss(
        x, g, ctx["model"].decoder(g), ctx["vgg"](x), ctx["vgg"](g), 2, weights
    )
    actual, terms = gray_objective(x, g, ctx["vgg"](x), ctx["vgg"](g), weights)
    expected = full - weights["reconstruction"] * old["mse"]
    torch.testing.assert_close(actual, expected)
    for key, value in terms.items():
        torch.testing.assert_close(value, old[key], rtol=0, atol=0)
    grad = torch.autograd.grad(actual, g, retain_graph=True)[0]
    torch.testing.assert_close(
        grad, torch.autograd.grad(expected, g)[0], rtol=2e-5, atol=1e-6
    )


def test_comparison_config_changes_only_training_profile_and_devices():
    root = Path(__file__).resolve().parents[1] / "configs"
    joint = json.loads((root / "phase2-g1.json").read_text())
    solo = json.loads((root / "phase2-g1-e1only.json").read_text())
    validate_config(solo)
    assert {k for k in joint if joint[k] != solo[k]} == {
        "schema",
        "update_modules",
        "gpu_ids",
    }
    assert solo["gpu_ids"] == [1, 3] and solo["update_modules"] == ["E1"]
    with pytest.raises(ValueError):
        validate_config({**solo, "update_modules": ["E1", "D1"]})
