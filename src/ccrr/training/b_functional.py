"""B feature/mask interface checks; no heat-map or clinical localization claim.

Section 8.6 heat-map ranking/perturbation experiments belong to subsequent
local-supervision deployment and are not replaced by these numerical checks.
"""

import copy

import torch

from ccrr.errors import require
from ccrr.representation.supervision import align, geometry_weights, heat_weights
from ccrr.training.b_diagnostics import isolated_rng


def geometry_check(device):
    cases = (
        (0, 0, 512, 512, 0, 0, 224, 224, 224),
        (0, 64, 512, 384, 0, 28, 224, 168, 224),
        (64, 0, 384, 512, 28, 0, 168, 224, 224),
    )
    for t in cases:
        for grid in (14, 16):
            area = geometry_weights(t, (grid, grid), device=device)
            torch.testing.assert_close(
                area.sum(), area.new_tensor(t[2] * t[3]), atol=0.1, rtol=1e-6
            )
            f = torch.ones(1, 1, grid, grid, device=device, requires_grad=True)
            values, coverage = align(f, [t], 512)
            torch.testing.assert_close(
                values[0, 0, coverage[0] > 0],
                torch.ones_like(values[0, 0, coverage[0] > 0]),
            )
            weights = heat_weights(torch.zeros_like(coverage), coverage)
            torch.testing.assert_close(
                weights, coverage / coverage.sum(1, keepdim=True)
            )
            torch.testing.assert_close(weights.sum(1), torch.ones(1, device=device))
            require(
                not weights[coverage == 0].count_nonzero(),
                "Padding acquired local weight",
            )
            (values * weights[:, None]).sum().backward()
            require(torch.isfinite(f.grad).all(), "Nonfinite alignment gradient")
    # Coordinate labels on the untransformed 14x14 grid must not be permuted.
    f = torch.arange(196, device=device).float().reshape(1, 1, 14, 14)
    values, _ = align(f, [cases[0]], 512)
    torch.testing.assert_close(values, f.flatten(2), atol=1e-4, rtol=1e-5)
    return {"rectangles": 3, "native_grids": [14, 16], "passed": True}


def check_functional(e, bank):
    """Disposable Purifier copy, real bank inputs, gradient and hard-forward checks."""
    with isolated_rng():
        p = copy.deepcopy(e.purifier).eval().requires_grad_(False)
        p.zero_grad(set_to_none=True)
        rows = []
        with torch.enable_grad():
            t = bank.get(list(range(min(2, len(e.fit)))))
            v = t["v"].clone()
            # Explicit padding and fractional coverage ensure the contract is exercised
            # even if the chosen source images happen to have square content.
            v[..., :2, :] = 0
            v[..., 2, :] = 0.5
            for key in ("f0", "f1"):
                x = t[key].detach().clone().requires_grad_(True)
                out = p(x, v)
                hard = p(x, v, source_mask=out["hard_mask"].detach())
                torch.testing.assert_close(
                    out["semantic"], hard["semantic"], atol=0, rtol=0
                )
                require(
                    torch.equal(out["q"], out["q"].round())
                    and out["q"].abs().max() <= 7,
                    "Invalid 15-level code",
                )
                require(
                    ((out["hard_mask"] == 0) | (out["hard_mask"] == 1)).all(),
                    "Invalid hard mask",
                )
                for name in ("q", "mask", "semantic", "all"):
                    require(
                        not out[name][..., :2, :].count_nonzero(),
                        "Nonzero padded output",
                    )
                changed = x.detach().clone()
                changed[..., :2, :] += 100
                with torch.no_grad():
                    ignored = p(changed, v)
                torch.testing.assert_close(
                    out["semantic"], ignored["semantic"], atol=0, rtol=0
                )
                # Trainable source-mask path must remain differentiable during B.
                cross = p(x, v, source_mask=out["mask"])
                loss = cross["semantic"].square().mean() + out["all"].square().mean()
                grad = torch.autograd.grad(loss, x)[0]
                require(
                    torch.isfinite(grad).all() and grad.abs().sum() > 0,
                    "Invalid feature gradient",
                )
                require(
                    not grad[..., :2, :].count_nonzero(), "Padding acquired gradient"
                )
                rows.append(
                    {
                        "domain": key,
                        "input_gradient_l1": float(grad.abs().sum()),
                        "passed": True,
                    }
                )
        require(
            all(w.grad is None for w in p.parameters()),
            "Frozen Purifier acquired parameter gradients",
        )
        return {
            "geometry": geometry_check(e.device),
            "domains": rows,
            "passed": True,
            "heatmap_ranking_validated": False,
            "limitation": "Numerical feature/mask interface only; section 8.6 heatmaps need separate downstream validation",
        }
