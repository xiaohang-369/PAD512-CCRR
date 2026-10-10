"""H100-only synthetic A resource check. No dataset access or optimizer updates.

Run one teacher per process. This measures a full effective batch's forward and
input/parameter backward workload; it is not a formal A training result.
"""

import argparse
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from check_teacher_interface import load_real_model
from ccrr.config import load_config
from ccrr.errors import require
from ccrr.runtime.support import seed_everything, environment, write_json
from ccrr.runtime.preflight import PINNED
from ccrr.teachers.adapters import freeze_bn
from ccrr.training.optimization import optimizer
from ccrr.training.losses import contrastive
from ccrr.tensor_ops import gray


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model", choices=("efficientnet_b0", "resnet50"), required=True
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    require(
        torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0),
        "H100 required",
    )
    entry = cfg["weights"][args.model]
    require(entry["sha256"] == PINNED[args.model], "Official weight version mismatch")
    args.weights, args.sha256, args.source = entry["path"], entry["sha256"], None
    report = {
        "model": args.model,
        "environment": environment(),
        "dataset_read": False,
        "optimizer_updates": 0,
        "formal_training_started": False,
        "effective_batch": 16,
    }
    try:
        model = load_real_model(args, report).cuda()
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        freeze_bn(model)
        opt = optimizer([model], [1e-4])
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        for i in range(16):
            a = torch.rand(1, 3, 512, 512, device="cuda")
            rects = [(0, 0, 512, 512)]
            if args.model == "efficientnet_b0":
                for view in (a, gray(a)):
                    loss = (
                        F.cross_entropy(
                            model(view, rects), torch.tensor([i % 6], device="cuda")
                        )
                        / 32
                    )
                    loss.backward()
            else:
                b = torch.rand_like(a)
                domain = i % 3
                fa = model(gray(a) if domain in (1, 2) else a, rects)
                fb = model(gray(b) if domain == 2 else b, rects)
                label = torch.tensor([float(i < 8)], device="cuda")
                ce = 0.5 * (
                    F.cross_entropy(fa["logits"], torch.tensor([i], device="cuda"))
                    + F.cross_entropy(
                        fb["logits"],
                        torch.tensor([i if i < 8 else i + 16], device="cuda"),
                    )
                )
                (
                    (ce + 0.5 * contrastive(fa["embedding"], fb["embedding"], label))
                    / 16
                ).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        require(
            grads and all(torch.isfinite(g).all() for g in grads),
            "Missing/nonfinite H100 gradients",
        )
        torch.cuda.synchronize()
        report.update(
            passed=True,
            seconds=time.monotonic() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            free_total_bytes=torch.cuda.mem_get_info(),
            adamw_state_additional_bytes_estimate=sum(
                p.numel() * p.element_size() * 2
                for p in model.parameters()
                if p.grad is not None
            ),
            limitations="No optimizer step; AdamW persistent states estimated, not allocated; formal throughput and long-run determinism pending",
        )
    except Exception as exc:
        report.update(passed=False, error=str(exc))
        raise
    finally:
        write_json(Path(args.out), report)


if __name__ == "__main__":
    main()
