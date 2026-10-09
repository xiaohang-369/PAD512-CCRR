"""Analytical effective-batch checks, not teacher capability evidence."""

import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from ccrr.training.losses import clinical_loss, weighted_cross_entropy
from ccrr.training.experiment import Experiment
from ccrr.codec.metadata import CLASSES
from ccrr.representation.supervision import TeacherSupervision
from ccrr.training import joint, representation


@pytest.mark.parametrize("sizes", [(1, 1, 1, 1, 1), (2, 3), (4, 1), (5,)])
@pytest.mark.parametrize("views", [1, 2, 3])
def test_weighted_ce_and_kd_loss_and_parameter_gradients(sizes, views):
    torch.manual_seed(91)
    x = torch.randn(5, views, 4, dtype=torch.float64)
    targets = torch.randn(5, views, 6, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 5, 1, 5, 2])
    weights = torch.tensor([1, 2, 3, 4, 5, 9], dtype=torch.float64)
    model = nn.Linear(4, 6).double()
    split = copy.deepcopy(model)
    logits = model(x).flatten(0, 1)
    labels = y.repeat_interleave(views)
    denominator = weights[labels].sum()
    expected = (
        F.cross_entropy(logits, labels, reduction="none") * weights[labels]
    ).sum() / denominator
    expected = expected + 0.5 * F.kl_div(
        logits.log_softmax(-1),
        targets.detach().flatten(0, 1).softmax(-1),
        reduction="batchmean",
    )
    expected.backward()
    total = 0
    start = 0
    for size in sizes:
        for view in range(views):
            loss = (
                clinical_loss(
                    split(x[start : start + size, view]),
                    targets[start : start + size, view],
                    y[start : start + size],
                    weights,
                    effective_weight_mean=weights[labels].mean(),
                )
                * size
                / (len(y) * views)
            )
            loss.backward()
            total += loss.detach()
        start += size
    torch.testing.assert_close(total, expected.detach(), rtol=1e-12, atol=1e-12)
    for a, b in zip(model.parameters(), split.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
    assert targets.grad is None


@pytest.mark.parametrize("views", [1, 2, 3])
def test_teacher_and_purifier_summed_ce_global_denominator(views):
    torch.manual_seed(22)
    x = torch.randn(5, views, 6, dtype=torch.float64, requires_grad=True)
    split = x.detach().clone().requires_grad_()
    y = torch.tensor([0, 1, 4, 5, 1])
    weights = torch.tensor([1, 7, 3, 4, 2, 5], dtype=torch.float64)
    whole = F.cross_entropy(x.flatten(0, 1), y.repeat_interleave(views), weight=weights)
    whole.backward()
    total = sum(
        weighted_cross_entropy(
            split[i : i + 1, view],
            y[i : i + 1],
            weights,
            weight_sum=weights[y].sum() * views,
        )
        for i in range(len(y))
        for view in range(views)
    )
    total.backward()
    torch.testing.assert_close(total, whole)
    torch.testing.assert_close(split.grad, x.grad)


def test_effective_cohort_counts_repeated_draws_and_only_feasible_images():
    e = object.__new__(Experiment)
    e.fit = SimpleNamespace(rows=[{"diagnostic": CLASSES[i]} for i in range(6)])
    e.weights = torch.arange(1, 7).float()
    assert e.class_weight_mean([0, 1, 5, 5]) == 3.75
    assert e.class_weight_mean([0, 1]) == 1.5
    for seed in (0, 1, 20261008):
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        assert np.array_equal(
            a.integers(137, size=8), [b.integers(137, size=1)[0] for _ in range(8)]
        )


class TinyClinical(nn.Module):
    """Analytical logits for gradient routing, not a mock passing teacher."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 6)

    def forward(self, x, rects):
        return self.head(x.flatten(1).mean(1, keepdim=True))


def test_teacher_supervision_preserves_global_ce_and_frozen_input_gradient():
    torch.manual_seed(9)
    teacher = TinyClinical()
    weights = torch.arange(1, 7).float()
    supervisor = TeacherSupervision(
        teacher,
        nn.Identity(),
        nn.Identity(),
        nn.Identity(),
        weights,
        local_enabled=False,
    )
    y = torch.tensor([0, 4, 5])
    x = torch.randn(3, 1, 4, 4, requires_grad=True)
    split = x.detach().clone().requires_grad_()
    targets = {"logits": torch.randn(3, 6)}
    loss = supervisor.loss(x, [], y, targets)[0]
    loss.backward()
    accumulated = sum(
        supervisor.loss(
            split[i : i + 1],
            [],
            y[i : i + 1],
            {"logits": targets["logits"][i : i + 1]},
            effective_weight_mean=weights[y].mean(),
        )[0]
        / len(y)
        for i in range(len(y))
    )
    accumulated.backward()
    torch.testing.assert_close(accumulated, loss)
    torch.testing.assert_close(split.grad, x.grad)
    assert x.grad.abs().sum() > 0 and all(p.grad is None for p in teacher.parameters())


@pytest.mark.parametrize("path", ["D", "E", "H", "I-message", "I-representation"])
def test_training_paths_supply_the_actual_cohort_denominator(monkeypatch, path):
    """Exercise real stage loops with analytical components, no acceptance reports.

    Only class supervision is nonzero. Stop before validation/selection; compare
    the gradient produced by the real accumulation loop with the batch formula.
    """

    class Fit:
        rows = [{"diagnostic": CLASSES[i]} for i in (0, 5, 1, 5)]

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return {
                "x": torch.tensor([[[float(i + 1)]]]),
                "v": torch.ones(1, 1, 1),
                "legal": torch.ones(1),
                "label": (0, 5, 1, 5)[i],
                "patient_index": i,
                "patient": str(i),
                "id": str(i),
                "metadata": {},
                "rect": (0, 0, 1, 1),
            }

    class System(nn.Module):
        def __init__(self):
            super().__init__()
            self.inn = nn.Linear(1, 6, bias=False)
            self.entropy = nn.Linear(1, 1)
            self.entropy.rate = lambda z, *args: z.flatten(1).sum(1) * 0
            self.fixed_gray = False
            self.comm = SimpleNamespace(
                write=lambda g, *args: (g, None), read=lambda g: g
            )

        def quantized(self, x, level):
            g = self.inn(x.flatten(1))
            return g, g, g, torch.zeros_like(g), g

        def recover(self, g, z):
            return g

    class Supervision:
        def targets(self, *args):
            return None

        def loss(self, g, rects, y, target, *, effective_weight_mean):
            calls.append((y.item(), float(effective_weight_mean)))
            return (
                clinical_loss(
                    g,
                    g.detach(),
                    y,
                    e.weights,
                    effective_weight_mean=effective_weight_mean,
                ),
                g.sum() * 0,
                {},
            )

    e = object.__new__(Experiment)
    e.fit, e.device, e.seed = Fit(), "cpu", 20261008
    e.weights = torch.arange(1, 7).float()
    e.cfg = {
        "batch_images": 4,
        "budgets": {"representation": 1, "multirate": 1},
        "lambda_privacy": 0.1,
        "lambda_rate": 0.01,
    }
    e.system, e.supervision = System(), Supervision()
    e.profile = SimpleNamespace(step=lambda level: 1 / 32)
    e.pub_attacker = nn.Linear(1, 1)
    e.bind_snapshot = lambda *args: None
    e.log = lambda *args: None
    calls = []
    module = representation if path in ("D", "E") else joint
    for name in ("reconstruction", "gray_loss", "saturation", "masked_mean"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, lambda x, *args: x.sum() * 0)
    if module is joint:
        monkeypatch.setattr(joint, "message_loss", lambda x, *args: (x.sum() * 0, {}))
    chosen = [0, 1, 3] if path in ("H", "I-message") else [0, 1, 2, 3]
    if module is representation:
        from ccrr.data.pad import rng_for

        rng = rng_for(e.seed, path, "images", 0)
        if path == "E":
            rng.integers(1, 4)
        chosen = rng.integers(len(e.fit), size=4).tolist()
        monkeypatch.setattr(
            representation.adversarial, "pair_objective", lambda *args: None
        )
        monkeypatch.setattr(
            representation.adversarial, "adversarial_updates", lambda *args: None
        )
        monkeypatch.setattr(
            representation, "update", lambda opt: 0
        )  # retain raw gradient

        class EndOfUpdate(Exception):
            pass

        def stop(*args):
            raise EndOfUpdate()

        monkeypatch.setattr(representation.validation, "validate_recovery", stop)
        with pytest.raises(EndOfUpdate):
            representation.train_representation(e, path)
    elif path == "I-representation":
        joint._accumulate_representation_gradients(e, chosen, np.random.default_rng(0))
    else:
        feasible = [
            (i, {"level": 1, "n": torch.zeros(1, 6), "bits": None, "occupancy": None})
            for i in chosen
        ]
        joint._accumulate_message_gradients(e, feasible)
    y = torch.tensor([(0, 5, 1, 5)[i] for i in chosen])
    assert calls == [(int(label), float(e.weights[y].mean())) for label in y]
    expected_model = copy.deepcopy(e.system.inn)
    expected_model.zero_grad()
    logits = expected_model(torch.tensor([[float(i + 1)] for i in chosen]))
    expected = F.cross_entropy(logits, y, weight=e.weights)
    if path == "D":
        expected = expected / 2000  # prescribed ramp on the first generator step
    expected.backward()
    torch.testing.assert_close(
        e.system.inn.weight.grad, expected_model.weight.grad, atol=1e-7, rtol=1e-5
    )
