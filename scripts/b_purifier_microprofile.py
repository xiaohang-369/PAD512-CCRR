"""Bounded profiling of the existing R3 Purifier step at effective batch 16.

No new update/loss implementation, batch search, training-config promotion or
resume of the source experiment. All cases restore the same step-2000 snapshot.
"""

from contextlib import ExitStack
import csv
import statistics
from unittest.mock import patch

import torch

from b_r3_perf_common import cpu_copy, digest, exact

MICROBATCHES = (16, 1, 2, 4, 8)


def differences(reference, candidate, *, atol=1e-6, rtol=1e-4):
    """Numerical screen only; tolerances do not establish scientific equivalence."""
    result = {
        "atol": atol,
        "rtol": rtol,
        "tensors": 0,
        "outside_tolerance": 0,
        "max_absolute_difference": 0.0,
        "max_relative_l2_difference": 0.0,
        "nonfinite": False,
    }

    def visit(a, b):
        if isinstance(a, torch.Tensor):
            if (
                not isinstance(b, torch.Tensor)
                or a.shape != b.shape
                or a.dtype != b.dtype
            ):
                raise ValueError("Tensor schema changed")
            result["tensors"] += 1
            if a.is_floating_point():
                a, b = a.double(), b.double()
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                result["nonfinite"] |= not finite
                if not finite:
                    result["outside_tolerance"] += 1
                    return
                delta = a - b
                result["max_absolute_difference"] = max(
                    result["max_absolute_difference"],
                    float(delta.abs().max()) if delta.numel() else 0,
                )
                result["max_relative_l2_difference"] = max(
                    result["max_relative_l2_difference"],
                    float(delta.norm() / a.norm().clamp_min(1e-12)),
                )
                result["outside_tolerance"] += int(
                    not torch.allclose(a, b, atol=atol, rtol=rtol)
                )
            else:
                result["outside_tolerance"] += int(not torch.equal(a, b))
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise ValueError("State dictionary changed")
            for key in a:
                visit(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise ValueError("State sequence changed")
            for x, y in zip(a, b):
                visit(x, y)
        elif a != b:
            raise ValueError("Non-tensor training state changed")

    visit(reference, candidate)
    result["within_tolerance"] = result["outside_tolerance"] == 0
    return result


def one_step_evidence(work):
    """Capture actual sample identities, all three preclip gradients and states.

    Shapes of microbatch gathers deliberately do not enter the sampling digest.
    Capture happens outside timing. Never log patient data in the brief report.
    """
    import b_r3_diagnostics as diagnostics

    work.reset()
    gradients, samples = [], []
    original_clip = torch.nn.utils.clip_grad_norm_

    def clip(params, *args, **kwargs):
        params = list(params)
        gradients.append(cpu_copy([p.grad for p in params]))
        return original_clip(params, *args, **kwargs)

    def image_backward(e, bank, ids):
        samples.append(("images", list(map(int, ids))))
        return original_image(e, bank, ids)

    def pair_backward(e, bank, pairs, rng, coefficient, detach):
        samples.append(
            ("pairs", pairs, cpu_copy(rng.bit_generator.state), coefficient, detach)
        )
        return original_pair(e, bank, pairs, rng, coefficient, detach)

    original_image, original_pair = (
        diagnostics.image_backward,
        diagnostics.pair_backward,
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.nn.utils, "clip_grad_norm_", clip))
        stack.enter_context(patch.object(diagnostics, "image_backward", image_backward))
        stack.enter_context(patch.object(diagnostics, "pair_backward", pair_backward))
        work.update()
    capture = work.capture()
    if len(gradients) != 3:
        raise RuntimeError(
            "Expected two complete attacker updates plus one generator update"
        )
    image_draws = sum(len(row[1]) for row in samples if row[0] == "images")
    pair_draws = [len(row[1]) for row in samples if row[0] == "pairs"]
    if image_draws != 16 or pair_draws != [16, 16, 16]:
        raise RuntimeError(
            "Effective image/pair batch changed or adversarial branch missing"
        )
    return {
        "sampling_sha256": digest(samples),
        "image_draws": image_draws,
        "pair_draws_per_backward_group": pair_draws,
        "unclipped_gradients": gradients,
        "models": capture["models"],
        "optimizers": capture["optimizers"],
        "schedulers": capture["schedulers"],
        "rng": capture["rng"],
        "loss": work.state.data["losses"][-1]["loss"],
    }


def run_microprofile(e, banks, engine, snapshot, args, guard):
    from profile_b_r3_h100 import Workload, benchmark, write

    root = args.out / "purifier16"
    root.mkdir()
    report = {
        "complete": False,
        "effective_image_batch": 16,
        "effective_pair_batch": 16,
        "microbatches": list(MICROBATCHES),
        "cases": {},
        "scope": "Original archived b_r3_diagnostics.train_step, fixed step-2000 snapshot; each transaction is two attacker optimizer steps plus one Purifier optimizer step. Full adversarial ramp is active.",
        "unchanged": "FP32, models, 64-channel code, quantization, masks, all loss weights, AdamW, learning-rate schedules, sampling and checkpoint policy.",
        "timing_scope": "Original update transactions, save requests and budget heartbeat. Teacher bank build reported separately. Endpoint audit, observations and startup excluded; not complete Stage B time.",
        "numerical_scope": "One-step preclip gradients, model and AdamW states are compared with microbatch16. FP32 reduction order can differ. This is not long-run equivalence or scientific admission.",
        "scientific_admission_complete": False,
        "training_configuration_changed": False,
        "production_microbatch_changed": False,
        "formal_training_started": False,
    }
    original_micro = e.cfg["microbatch"]
    reference = None

    def persist():
        write(root / "report.json", report)
        columns = [
            "microbatch",
            "accumulation",
            "seconds_per_step",
            "speedup_vs_micro16",
            "images_per_second",
            "peak_allocated_GiB",
            "peak_reserved_GiB",
            "sampled_gpu_util_percent",
            "full_checkpoint_seconds",
            "sampling_equal",
            "one_step_numerical_screen",
        ]
        with (root / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(
                {k: row.get(k) for k in columns} for row in report["cases"].values()
            )

    persist()
    try:
        for micro in MICROBATCHES:
            guard.check()
            e.cfg["microbatch"] = micro
            work = Workload(
                e, banks, None, engine, root / f"micro-{micro}", "purifier", snapshot
            )
            evidence = one_step_evidence(work)
            if reference is None:
                reference = evidence
            checks = {
                key: differences(reference[key], evidence[key])
                for key in ("unclipped_gradients", "models", "optimizers")
            }
            sampling_equal = reference["sampling_sha256"] == evidence["sampling_sha256"]
            state_equal = exact(
                reference["schedulers"], evidence["schedulers"]
            ) and exact(reference["rng"], evidence["rng"])
            if not sampling_equal or not state_equal:
                raise RuntimeError("Sampling, schedule or RNG changed with microbatch")
            write(
                work.root / "one-step-check.json",
                {
                    "sampling_equal": sampling_equal,
                    "schedulers_rng_equal": state_equal,
                    "image_draws": evidence["image_draws"],
                    "pair_draws_per_backward_group": evidence[
                        "pair_draws_per_backward_group"
                    ],
                    "baseline_loss": reference["loss"],
                    "candidate_loss": evidence["loss"],
                    "comparisons": checks,
                },
            )
            del evidence
            print(
                f"PURIFIER16_START microbatch={micro} accumulation={16 // micro}",
                flush=True,
            )
            sample_start = len(guard.rows)
            result = benchmark(work, args, guard)
            if not result["profile"]["activity"]["cuda_kernels_recorded"]:
                raise RuntimeError("Missing CUDA kernel trace; incomplete measurement")
            sampled = [
                float(x["nvidia_smi"].split(",")[2])
                for x in guard.rows[sample_start:]
                if x["phase"].startswith("purifier/throughput/")
            ]
            row = {
                "microbatch": micro,
                "accumulation": 16 // micro,
                "seconds_per_step": result["median_step_seconds"],
                "images_per_second": 16 / result["median_step_seconds"],
                "peak_allocated_GiB": max(
                    r["peak_allocated_bytes"] for r in result["repeats"]
                )
                / 1024**3,
                "peak_reserved_GiB": max(
                    r["peak_reserved_bytes"] for r in result["repeats"]
                )
                / 1024**3,
                "sampled_gpu_util_percent": statistics.mean(sampled)
                if sampled
                else None,
                "full_checkpoint_seconds": statistics.median(
                    r["full_checkpoint"]["wall_seconds"] for r in result["repeats"]
                ),
                "sampling_equal": sampling_equal,
                "one_step_numerical_screen": all(
                    c["within_tolerance"] for c in checks.values()
                ),
                "measurement": result,
            }
            baseline = report["cases"].get("16", row)["seconds_per_step"]
            row["speedup_vs_micro16"] = baseline / row["seconds_per_step"]
            report["cases"][str(micro)] = row
            persist()
            print(
                "PURIFIER16_RESULT",
                {k: v for k, v in row.items() if k != "measurement"},
                flush=True,
            )
        eligible = [
            r for r in report["cases"].values() if r["one_step_numerical_screen"]
        ]
        report["timing_candidate_microbatch"] = (
            min(eligible, key=lambda r: r["seconds_per_step"])["microbatch"]
            if eligible
            else None
        )
        report["candidate_is_not_automatic_promotion"] = True
        report["complete"] = True
        return report
    finally:
        e.cfg["microbatch"] = original_micro
        persist()
