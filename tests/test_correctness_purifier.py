import copy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from ccrr.representation.purifier import Purifier, SemanticAttacker
from ccrr.tensor_ops import ResBlock, pooled
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint


def test_exact_identity_and_semantic_attacker_structure():
    p, a = Purifier(), SemanticAttacker()
    for net, channels in ((p.identity, 64), (a.net, 768)):
        assert list(net._modules) == ["start", "blocks"]
        assert isinstance(net.start, nn.Conv2d)
        assert (
            net.start.in_channels,
            net.start.out_channels,
            net.start.kernel_size,
        ) == (channels, 128, (1, 1))
        assert len(net.blocks) == 2 and all(isinstance(b, ResBlock) for b in net.blocks)
        assert not any(
            isinstance(m, (nn.BatchNorm2d, nn.Dropout)) for m in net.modules()
        )
    # Es and Ds retain their required terminal projections.
    assert p.encoder.end.weight.shape == (64, 128, 1, 1)
    assert p.decoder.end.weight.shape == (768, 128, 1, 1)
    assert p.id_head.weight.shape == (2048, 128)
    assert p.clinical.weight.shape == (6, 768)
    assert a.embed[0].weight.shape == (256, 640)
    assert all("identity.end" not in k for k in p.state_dict())
    assert all("net.end" not in k for k in a.state_dict())


def test_forward_matches_design_and_all_purifier_branch_gradients():
    torch.manual_seed(55)
    p = Purifier()
    features = torch.randn(2, 768, 14, 14, requires_grad=True)
    v = torch.ones(2, 1, 14, 14)
    v[..., :2, :] = 0
    v[..., 2, :] = 0.5
    out = p(features, v)
    assert out["q"].shape == (2, 64, 14, 14)
    assert out["semantic"].shape == (2, 768, 14, 14)
    assert out["identity"].shape == (2, 2048)
    assert out["logits"].shape == (2, 6)
    mask = (v > 0).float()
    h = p.identity.start(out["mask"] * out["q"] / 7) * mask
    for block in p.identity.blocks:
        h = block(h, v)
    expected = F.normalize(p.id_head(pooled(h, v)), dim=1, eps=1e-6)
    torch.testing.assert_close(out["identity"], expected)
    torch.testing.assert_close(out["identity"].norm(dim=1), torch.ones(2))
    for name in ("q", "semantic", "all", "mask"):
        assert not out[name][..., :2, :].count_nonzero()
    loss = (
        out["all"].square().mean()
        + out["semantic"].square().mean()
        + out["logits"].square().mean()
    )
    loss = loss + (out["identity"] * torch.randn_like(out["identity"])).sum()
    loss.backward()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    for parameter in p.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    # Source mask controls only the feature split; no INN private latent enters.
    cross = p(features.detach(), v, source_mask=out["hard_mask"].detach())
    torch.testing.assert_close(cross["mask"], out["hard_mask"].detach())


def test_attacker_update_detaches_generator_and_generator_update_freezes_attacker():
    torch.manual_seed(5)
    p, a = Purifier(), SemanticAttacker()
    v = torch.ones(1, 1, 14, 14)
    pa = p(torch.randn(1, 768, 14, 14), v)["semantic"]
    pb = p(torch.randn(1, 768, 14, 14), v)["semantic"]
    loss = F.binary_cross_entropy_with_logits(
        a(pa.detach(), v, pb.detach(), v), torch.ones(1)
    )
    loss.backward()
    assert all(w.grad is None for w in p.parameters())
    assert all(
        w.grad is not None and torch.isfinite(w.grad).all() for w in a.parameters()
    )
    a.zero_grad(set_to_none=True)
    for w in a.parameters():
        w.requires_grad_(False)
    (-F.binary_cross_entropy_with_logits(a(pa, v, pb, v), torch.ones(1))).backward()
    assert p.encoder.start.weight.grad.abs().sum() > 0
    assert all(w.grad is None for w in a.parameters())


def test_corrected_checkpoint_roundtrip_and_wrong_old_structure_rejected(tmp_path):
    p, a = Purifier(), SemanticAttacker()
    path = tmp_path / "correct.pt"
    save_checkpoint(path, {"purifier": p, "attacker": a}, None, 0, "B", {})
    p2, a2 = Purifier(), SemanticAttacker()
    load_checkpoint(path, {"purifier": p2, "attacker": a2})
    for before, after in ((p, p2), (a, a2)):
        for key, value in before.state_dict().items():
            assert torch.equal(value, after.state_dict()[key])
    for model, prefix in ((p2, "identity"), (a2, "net")):
        wrong = copy.deepcopy(model.state_dict())
        wrong[prefix + ".end.weight"] = torch.empty(128, 128, 1, 1)
        wrong[prefix + ".end.bias"] = torch.empty(128)
        with pytest.raises(RuntimeError, match="Unexpected key"):
            model.load_state_dict(wrong, strict=True)
