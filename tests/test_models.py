import torch
import numpy as np
from ccrr.models import INN, Communication, Purifier
from ccrr.models.common import pack, unpack, haar, ihaar
from ccrr.records import split_stream, route
from ccrr.losses import message_loss
from ccrr.models.teachers import ClinicalModel, freeze
from ccrr.supervision import align

torch.set_num_threads(2)


def test_inn_nontrivial_inverse_and_quantized_gradient():
    torch.manual_seed(17)
    m = INN()
    for level in m.levels:
        for block in level:
            torch.nn.init.normal_(block.B, std=0.01)
            torch.nn.init.normal_(block.net[-1].weight, std=0.002)
    x = torch.rand(1, 3, 32, 32)
    u, z = m(x)
    rec = m.inverse(u, z)
    assert (rec - x).abs().max() < 1e-4
    assert torch.equal(unpack(pack(x)), x)
    ((m.inverse(u + 0.001, z + 0.001) - x) ** 2).mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.levels[0][0].net[-1].weight.grad.abs().sum() > 0


def test_haar_and_alignment():
    x = torch.rand(2, 1, 32, 32)
    assert torch.allclose(ihaar(haar(x)), x, atol=2e-7)
    f = torch.ones(1, 5, 14, 14)
    a, cov = align(f, [(0, 0, 512, 512, 0, 0, 224, 224, 224)], 512)
    assert torch.allclose(a, torch.ones_like(a)) and torch.allclose(
        cov.sum(), torch.tensor(512.0**2)
    )


def test_full_communication_shapes_bounds_and_gradients():
    torch.manual_seed(5)
    m = Communication(4)
    # 128 is only a unit-test canvas for local/global grouping, not a system profile.
    g = torch.floor(torch.rand(1, 1, 128, 128) * 255) / 255
    c = split_stream(bytes(range(144)), 0, 1)
    mm = split_stream(bytes(408), 1, 0)
    bits, occ = route(c, mm, [4] * 16, np.ones(16))
    bt = torch.tensor(bits)[None]
    ot = torch.tensor(occ)[None]
    legal = torch.ones(1, 16)
    legal[0, -1] = 0
    bt[0, -1] = 0
    ot[0, -1] = 0
    p, q = m.write(g, bt, ot, legal)
    logits = m.read(p)
    assert logits.shape == (1, 16, 4, 177) and q.shape == (1, 16, 2, 4)
    assert ((p - g) * 255).abs().max() <= 4.00001
    assert torch.equal(p[:, :, -32:, -32:], g[:, :, -32:, -32:])
    loss, _ = message_loss(logits, bt, ot)
    loss.backward()
    assert m.writer.out.weight.grad.abs().sum() > 0
    assert m.carrier.patch.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_purifier_mask_and_input_gradient():
    p = Purifier()
    f = torch.randn(2, 768, 14, 14, requires_grad=True)
    v = torch.ones(2, 1, 14, 14)
    v[:, :, :, 0] = 0
    out = p(f, v)
    assert out["q"].abs().max() <= 7
    assert torch.equal(out["hard_mask"].unique(), torch.tensor([0.0, 1.0]))
    out["semantic"].square().mean().backward()
    assert f.grad.abs().sum() > 0
    assert torch.equal(
        out["semantic"][:, :, :, 0], torch.zeros_like(out["semantic"][:, :, :, 0])
    )


def test_frozen_teacher_input_gradient():
    # Random weights test the adapter's graph only, never teacher capability.
    m = freeze(ClinicalModel())
    x = torch.rand(1, 1, 128, 128, requires_grad=True)
    logits = m(x, [(0, 0, 128, 128)])
    logits.sum().backward()
    assert x.grad.abs().sum() > 0 and all(p.grad is None for p in m.parameters())
