"""Synthetic CPU/selected-CUDA regressions; no optimizer, PAD data or checkpoint."""

import argparse
import json
import math

import torch

from .learning import initialize_attacker
from .models import ClinicalHeads, FrozenSupervisor, GaussianMixture, Separator


def check(device):
    device = torch.device(device)
    gmm = GaussianMixture(1).to(device)
    with torch.no_grad():
        gmm.raw_scales.fill_(math.log(math.expm1(0.75)))
    q = torch.tensor([[[[-5.0, 5.0]]]], device=device, requires_grad=True)
    bits, clipped = gmm(q)
    bits.sum().backward()
    assert not clipped.any()
    assert torch.allclose(bits, torch.full_like(bits, 26.6844257273), atol=1e-4)
    assert torch.allclose(
        q.grad, q.new_tensor([[[[-10.44616755, 10.44616755]]]]), atol=1e-4
    )
    cpu_state = torch.get_rng_state().clone()
    cuda_state = (
        torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    )
    initialize_attacker(8, 20261220, device)
    assert torch.equal(cpu_state, torch.get_rng_state())
    if cuda_state is not None:
        assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))

    shapes = {}
    for variant in ("compression", "separation", "adversarial"):
        model = Separator(8, 8, 4, 6, variant=variant).to(device).eval()
        x = torch.randn(2, 8, 4, 4, device=device)
        v = torch.ones(2, 1, 4, 4, device=device)
        v[:, :, 0] = 0
        v[:, :, 1] = 0.5
        changed = x.clone()
        changed[:, :, 0] += 100
        with torch.no_grad():
            a, b = model(x, v), model(changed, v)
            assert all(torch.equal(a[k], b[k]) for k in a)
            assert a["q"][:, :, 0].count_nonzero() == 0
            assert torch.equal(a["full"], model.complete(x, v)["full"])
            poison = a["q"].clone()
            poison[:, :, 0] = 99
            assert torch.equal(a["full"], model.decode(poison, v))
        heads = ClinicalHeads(8).to(device)
        supervisor = FrozenSupervisor(
            model, heads, torch.zeros(8, device=device), torch.ones(8, device=device)
        )
        before = {k: t.detach().clone() for k, t in supervisor.state_dict().items()}
        supervisor.train()
        raw = x.clone().requires_grad_()
        logits, _ = supervisor(raw, v, 0)
        again, _ = supervisor(raw, v, 0)
        assert torch.equal(logits, again)
        logits.square().sum().backward()
        assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
        assert raw.grad[:, :, 0].count_nonzero() == 0
        assert all(p.grad is None for p in supervisor.parameters())
        assert all(
            torch.equal(before[k], t) for k, t in supervisor.state_dict().items()
        )
        shapes[variant] = list(a["q"].shape)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return dict(
        device=str(device),
        name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        tail_bits=bits.detach().cpu().flatten().tolist(),
        tail_gradients=q.grad.detach().cpu().flatten().tolist(),
        support_variants=shapes,
        attacker_rng_unchanged=True,
        feature_supervisor_input_gradient=True,
        frozen_parameters_and_buffers_unchanged=True,
        passed=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device != "cpu":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose only the selected physical GPU; CUDA required")
    devices = ("cpu",) if args.device == "cpu" else ("cpu", args.device)
    print(
        json.dumps(
            dict(
                torch=str(torch.__version__),
                checks=[check(device) for device in devices],
                training_started=False,
                model_parameters_updated=False,
                real_pixel_supervision_verified=False,
                scientific_acceptance="not_evaluated",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
