"""TF 1 Adam epsilon placement and the fixed author's shared LR schedule."""

import math
import torch


class TF1Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, updates=0))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("No closure in the source training loop")
        for group in self.param_groups:
            group["updates"] += 1
            t = group["updates"]
            b1, b2 = group["betas"]
            # TF1: epsilon is added to sqrt(uncorrected v), not sqrt(v_hat).
            rate = group["lr"] * math.sqrt(1 - b2**t) / (1 - b1**t)
            for p in group["params"]:
                if p.grad is None:
                    raise ValueError("All original E/D parameters require gradients")
                if p.grad.is_sparse:
                    raise ValueError("Dense source kernels only")
                state = self.state[p]
                if not state:
                    state["m"], state["v"] = torch.zeros_like(p), torch.zeros_like(p)
                m, v = state["m"], state["v"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                p.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-rate)


def polynomial_lr(step, decay_steps, initial=2e-4, final=2e-6, power=0.9):
    if decay_steps <= 0:
        raise ValueError("Source schedule needs at least one full batch")
    return (initial - final) * (
        1 - min(step, decay_steps) / decay_steps
    ) ** power + final


class ContinuousBatches:
    """Infinite shuffled source cycles; full batches cross logical epoch boundaries.

    TF queue seed/order is not reproduced. Each deterministic cycle is a new
    permutation. Position alone resumes the stream without saving patient IDs.
    """

    def __init__(self, count, batch, seed, position=0):
        self.count, self.batch, self.seed, self.position = count, batch, seed, position
        self.cycle = None
        self.order = None

    def next(self):
        result = []
        while len(result) < self.batch:
            cycle, offset = divmod(self.position, self.count)
            if cycle != self.cycle:
                generator = torch.Generator().manual_seed(self.seed + cycle)
                self.order = torch.randperm(self.count, generator=generator).tolist()
                self.cycle = cycle
            take = min(self.batch - len(result), self.count - offset)
            result.extend(self.order[offset : offset + take])
            self.position += take
        return result
