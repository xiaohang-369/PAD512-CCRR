from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ccrr.tensor_ops import gray, q8
from ccrr_v3.models import Separator, ClinicalHeads
from ccrr_v3.grayscale import (
    GrayscaleSystem,
    PixelSupervisor,
    quantize_private,
    local_variation,
    appearance,
    clinical_loss,
    update,
)
from ccrr_v3.grayscale_dev import png_roundtrip
from ccrr_v3 import grayscale_dev


class TinyPanDerm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x, rects):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        f = F.adaptive_avg_pool2d(self.bn(self.conv(x)), (4, 4))
        return f, torch.ones_like(f[:, :1]), rects


def teacher():
    return PixelSupervisor(
        TinyPanDerm(),
        Separator(input_channels=8, width=8, code_channels=4, embedding_dim=6),
        ClinicalHeads(8),
        dict(mean=torch.zeros(8), std=torch.ones(8)),
    )


def snapshot(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def same(module, old):
    return all(torch.equal(v, old[k]) for k, v in module.state_dict().items())


def test_exact_architecture_canvas_quantization_and_receiver_only(tmp_path):
    torch.manual_seed(3)
    model = GrayscaleSystem().eval()
    assert not any(
        isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.Dropout))
        for m in model.modules()
    )
    x = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x, 8)
        assert out["q"].shape == (1, 8, 32, 32)
        assert out["q"].dtype == torch.uint8
        assert torch.equal(out["gray"], gray(x))
        g = png_roundtrip(out["gray"], tmp_path / "gray.png")
        # A new decoder with identical weights has no sender activations or X.
        receiver = deepcopy(model.decoder)
        _, rgb = receiver(g, out["q"].float() / 255, 8)
        assert torch.allclose(rgb, out["rgb"], atol=1e-6, rtol=1e-6)
        q4, _ = quantize_private(out["z"], 4)
        assert q4.max() <= 15 and q4.numel() == 8192
        assert receiver(g, q4.float() / 15, 4)[0].shape == x.shape
        with pytest.raises(ValueError):
            receiver(g, q4.float() / 15, 6)


def test_quantizers_half_up_hard_values_unscaled_gradients():
    for bits in (4, 8):
        scale = 2**bits - 1
        z = torch.tensor([0.0, 0.5 / scale, 1.0], requires_grad=True)
        q, y = quantize_private(z, bits)
        assert q.tolist() == [0, 1, scale]
        y.sum().backward()
        assert torch.equal(z.grad, torch.ones_like(z))
    x = torch.tensor([-1.0, 0.5 / 255, 2.0], requires_grad=True)
    q8(x).sum().backward()
    assert x.grad.tolist() == [0.0, 1.0, 0.0]


def test_pixel_teacher_frozen_under_train_and_domain_gradients():
    t = teacher()
    before = snapshot(t)
    nn.ModuleList([t]).train()
    for d, channels in ((0, 3), (1, 1)):
        x = torch.rand(2, channels, 32, 32, requires_grad=True)
        logits, _ = t(x, [(0, 0, 32, 32)] * 2, d)
        logits.square().sum().backward()
        assert x.grad.abs().sum() > 0 and torch.isfinite(x.grad).all()
    assert all(not m.training for m in t.modules())
    assert all(p.grad is None for p in t.parameters()) and same(t, before)
    with pytest.raises(ValueError, match="RGB head"):
        t(torch.rand(1, 3, 32, 32), [], 1)


def test_real_substep_updates_are_isolated_and_frozen_ed_keep_input_gradient():
    torch.manual_seed(5)
    m, t = GrayscaleSystem(), teacher()
    x = torch.rand(2, 3, 32, 32)
    b = dict(
        x=x,
        v=torch.ones(2, 1, 32, 32),
        label=torch.tensor([0, 1]),
        rect=[(0, 0, 32, 32)] * 2,
    )
    with torch.no_grad():
        targets = (t(x, b["rect"], 0)[0], t(gray(x), b["rect"], 1)[0])
    weights = torch.ones(6)
    ed = torch.optim.Adam([*m.private.parameters(), *m.decoder.parameters()], lr=1e-4)
    p = torch.optim.Adam(m.public.parameters(), lr=1e-4)
    frozen = snapshot(t)
    before = [snapshot(n) for n in (m.public, m.private, m.decoder)]
    update(m, t, b, targets, weights, ed, "b")
    assert same(m.public, before[0])
    assert not same(m.private, before[1]) and not same(m.decoder, before[2])
    before = [snapshot(n) for n in (m.public, m.private, m.decoder)]
    stats = update(m, t, b, targets, weights, p, "p", inspect_gradients=True)
    assert not same(m.public, before[0])
    assert same(m.private, before[1]) and same(m.decoder, before[2])
    assert stats["grad_p"] > 0 and stats["grad_e"] == stats["grad_d"] == 0
    assert stats["weighted_public_gradient_norms"]["medical"] > 0
    before = snapshot(m.public)
    update(m, t, b, targets, weights, ed, "ed")
    assert same(m.public, before) and same(t, frozen)
    # Isolate the G->E->D route from the direct G->D route.
    g = gray(x).detach().requires_grad_()
    z = m.private(x, g)
    _, z = quantize_private(z, 8)
    raw, _ = m.decoder(g.detach(), z, 8)
    (gradient,) = torch.autograd.grad(raw.square().mean(), g)
    assert gradient.abs().sum() > 0


def test_appearance_valid_support_and_twelve_edge_definition():
    ramp = torch.arange(9).reshape(1, 1, 3, 3).float()
    assert local_variation(ramp).item() == 2.0  # six edges of3, six edges of1
    x = torch.rand(1, 1, 32, 32)
    v = torch.ones_like(x)
    loss, _ = appearance(x, x, v)
    assert loss == 0
    v[..., :8, :] = 0
    y = x.clone()
    y[..., :8, :] += 10
    loss, _ = appearance(y, x, v)
    assert loss == 0  # no invalid pixels or windows crossing the border


def test_clinical_ce_weight_normalization_and_target_detach():
    logits = torch.randn(3, 6, requires_grad=True)
    target = torch.randn(3, 6, requires_grad=True)
    labels, weights = torch.tensor([0, 0, 2]), torch.arange(1, 7).float()
    expected = (
        F.cross_entropy(logits, labels, reduction="none") * weights[labels]
    ).sum() / weights[labels].sum()
    assert torch.allclose(
        clinical_loss(logits, labels, target, weights, kd=0), expected
    )
    clinical_loss(logits, labels, target, weights).backward()
    assert target.grad is None and logits.grad.abs().sum() > 0


class SyntheticData:
    def __init__(self, root, role, size=32):
        self.rows = [dict(img_id=f"{role}-{i}") for i in range(6)]
        self.size = size
        gen = torch.Generator().manual_seed(98)
        self.x = torch.rand(6, 3, size, size, generator=gen)

    def __len__(self):
        return 6

    def __getitem__(self, i):
        return dict(
            x=self.x[i],
            v=torch.ones(1, self.size, self.size),
            legal=torch.ones(1),
            rect=(0, 0, self.size, self.size),
            label=i,
            patient=f"p{i}",
            patient_index=i,
            id=f"image{i}",
            metadata={},
        )

    def class_weights(self):
        return torch.ones(6)


def test_pixel_verification_full_path_without_parameter_updates(monkeypatch, tmp_path):
    t = teacher()
    data = SyntheticData("", "fit", size=512)
    monkeypatch.setattr(grayscale_dev, "load_teacher", lambda *args: t)
    monkeypatch.setattr(grayscale_dev, "PADData", lambda *args: data)
    monkeypatch.setattr(grayscale_dev, "FEATURES", tmp_path)
    (tmp_path / "features.json").write_text('{"roles":{"fit":{"file":"fit.pt"}}}')
    with torch.no_grad():
        x = data.x[:1]
        fs = torch.stack(
            [t.panderm(im, [data[0]["rect"]])[0][0] for im in (x, gray(x))]
        )[None]
        cov = torch.ones(1, 1, 4, 4)
    torch.save(dict(rows=data.rows, features=fs, coverage=cov), tmp_path / "fit.pt")
    files = grayscale_dev.verify(
        tmp_path, {"data_root": "synthetic"}, 8500, torch.device("cpu")
    )
    report = grayscale_dev.read(tmp_path / "verification.json")
    assert (
        report["real_pixel_supervision_verified"]
        and not report["model_parameters_updated"]
    )
    assert all((tmp_path / f).exists() for f in files)


def test_pilot_b_then_c1_full_loop_and_reports(monkeypatch, tmp_path):
    monkeypatch.setattr(grayscale_dev, "load_teacher", lambda *args: teacher())
    monkeypatch.setattr(grayscale_dev, "PADData", SyntheticData)
    monkeypatch.setitem(grayscale_dev.CONFIG, "baseline_updates", 1)
    monkeypatch.setitem(grayscale_dev.CONFIG, "c1_cycles", 1)
    files = grayscale_dev.pilot(
        tmp_path, {"data_root": "synthetic"}, 7000, torch.device("cpu")
    )
    report = grayscale_dev.read(tmp_path / "training_summary.json")
    assert report["optimizer_updates"] == 3 and report["frozen_teacher_unchanged"]
    assert not report["direct_pixel_p2_p3_evaluated"]
    assert grayscale_dev.read(tmp_path / "b-diagnostic.json")["mean_gray_l1"] == 0
    assert all((tmp_path / f).exists() for f in files)
