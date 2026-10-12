"""Batch execution preserves the effective loss, samples and within-mode resume."""

import copy
import os
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from ccrr.data.pad import collate, rng_for
from ccrr.tensor_ops import gray
from ccrr.training.losses import contrastive, weighted_cross_entropy
from ccrr.training.teacher_batches import (
    clinical_backward,
    identity_loss,
    microbatch,
    reid_backward,
    flipped,
)
from ccrr.training.a_state import binding
from ccrr.errors import CCRRError
from test_stage_a_training_loops import context, TinyClinical

DEVICE = os.environ.get("CCRR_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("n", [4, 8, 16])
def test_clinical_effective_loss_and_gradients_match_scalar_objective(tmp_path, n):
    e = context(tmp_path)
    a = (
        TinyClinical().to(DEVICE).eval()
    )  # Disable Dropout only for objective equivalence.
    b = copy.deepcopy(a)
    ids = np.array([0, 0, 0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 0, 5])
    r1, r2 = rng_for(42, "c", "i", 7), rng_for(42, "c", "i", 7)
    views = [lambda v: v["x"], lambda v: gray(v["x"])]
    denom = sum(float(e.weights[i]) for i in ids) * 2
    expected = 0
    for i in ids:
        item = collate([e.fit[int(i)]], DEVICE)
        flip = r1.random() < 0.5
        for view in views:
            x = view(item)
            if flip:
                x = x.flip(-1)
            loss = weighted_cross_entropy(
                a(x, item["rect"]), item["label"], e.weights, weight_sum=denom
            )
            loss.backward()
            expected += float(loss.detach())
    e.cfg["microbatch"] = n
    seen = []
    handle = b.register_forward_pre_hook(lambda _, args: seen.append(len(args[0])))
    actual = clinical_backward(e, b, ids, r2, views)
    handle.remove()
    assert seen == [n] * (32 // n)
    assert r1.bit_generator.state == r2.bit_generator.state
    assert float(actual) == pytest.approx(expected, rel=2e-6)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-5, atol=1e-7)


def test_reid_uneven_pair_labels_do_not_change_contrastive_weighting():
    torch.manual_seed(31)
    ea = F.normalize(torch.randn(4, 8, device=DEVICE), dim=1).requires_grad_()
    eb = F.normalize(torch.randn(4, 8, device=DEVICE), dim=1).requires_grad_()
    la = torch.randn(4, 6, device=DEVICE, requires_grad=True)
    lb = torch.randn(4, 6, device=DEVICE, requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE)
    ya = torch.tensor([0, 1, 2, 3], device=DEVICE)
    yb = torch.tensor([0, 4, 5, 1], device=DEVICE)
    expected = sum(
        (
            0.5
            * (
                F.cross_entropy(la[i : i + 1], ya[i : i + 1])
                + F.cross_entropy(lb[i : i + 1], yb[i : i + 1])
            )
            + 0.5 * contrastive(ea[i : i + 1], eb[i : i + 1], labels[i : i + 1])
        )
        / 16
        for i in range(4)
    )
    actual = identity_loss(
        {"embedding": ea, "logits": la}, {"embedding": eb, "logits": lb}, ya, yb, labels
    )
    torch.testing.assert_close(actual, expected)
    for x, y in zip(
        torch.autograd.grad(actual, (ea, eb, la, lb), retain_graph=True),
        torch.autograd.grad(expected, (ea, eb, la, lb)),
    ):
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize("n", [0, 3, True, 4.0, 32])
def test_invalid_microbatch(n):
    with pytest.raises(CCRRError, match="microbatch"):
        microbatch({"microbatch": n})


def test_resume_binds_microbatch(tmp_path):
    cfg = context(tmp_path).cfg
    assert binding({**cfg, "microbatch": 4}) != binding({**cfg, "microbatch": 8})


def test_reid_keeps_pair_domains_flips_and_gradient_objective(tmp_path):
    from test_stage_a_training_loops import IdentityData, TinyReID

    e = context(tmp_path)
    e.fit = IdentityData()
    pairs = [(i, (i + 1) % 24, int(i % 4 == 0), "p1") for i in range(16)]
    base = TinyReID().to(DEVICE).eval()
    results = []
    for n in (1, 4, 8):
        model = copy.deepcopy(base)
        e.cfg["microbatch"] = n
        rng = rng_for(42, "r", "p", 7)
        sampled = reid_backward(e, model, pairs, rng)
        results.append(
            (sampled, rng.bit_generator.state, [p.grad for p in model.parameters()])
        )
    for actual in results[1:]:
        assert actual[:2] == results[0][:2]
        for a, b in zip(actual[2], results[0][2]):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_individual_flip_keeps_offset_rectangles():
    x = torch.arange(2 * 3 * 8 * 12).reshape(2, 3, 8, 12)
    value, rects = flipped(x, [(1, 2, 5, 4), (2, 0, 7, 8)], [True, False])
    assert rects == [(6, 2, 5, 4), (2, 0, 7, 8)]
    assert torch.equal(value[0], x[0].flip(-1)) and torch.equal(value[1], x[1])
