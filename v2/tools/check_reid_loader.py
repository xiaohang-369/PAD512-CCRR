"""Compare real fit batches on CPU. No model, optimizer or training is started."""

import argparse
import hashlib
import json
from time import perf_counter

import torch

from ccrr.data.pad import PADData, PairSampler
from ccrr_v2.loading import ReIDBatchIndices, cpu_batches
from ccrr_v2.runtime import completed, new_run, read, validate_resources, write


def batch_digest(batch):
    h = hashlib.sha256()
    for key, value in sorted(batch.items()):
        h.update(key.encode())
        if isinstance(value, torch.Tensor):
            h.update(str((value.dtype, tuple(value.shape))).encode())
            h.update(value.contiguous().numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resources", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch-batches", type=int, default=2)
    p.add_argument("--batches", type=int, default=16)
    args = p.parse_args()
    if args.workers < 1 or args.prefetch_batches < 1 or args.batches < 3:
        p.error("Positive workers/prefetch and at least 3 batches required")
    torch.set_num_threads(4)
    resources = read(args.resources)
    validate_resources(resources)
    config = {
        "seed": 20261019,
        "batch_size": 16,
        "workers": args.workers,
        "prefetch_batches": args.prefetch_batches,
        "batches": args.batches,
        "scope": "CPU_batch_equivalence_only_not_training_throughput",
    }
    output = new_run(resources, args.output, "reid_loader_check", config)
    try:
        data = PADData(resources["data_root"], "fit")
        reference, summaries = [], []
        for workers in (0, args.workers):
            sampler = PairSampler(data)
            batches = iter(
                cpu_batches(
                    data,
                    ReIDBatchIndices(sampler, config["seed"], args.batches),
                    workers=workers,
                    prefetch=args.prefetch_batches,
                    seed=config["seed"],
                )
            )
            start, waits = perf_counter(), []
            for i in range(args.batches):
                before = perf_counter()
                batch = next(batches)
                waits.append(perf_counter() - before)
                digest = batch_digest(batch)
                if workers == 0:
                    reference.append(digest)
                elif digest != reference[i]:
                    raise RuntimeError(
                        f"Serial/parallel batch mismatch at step {i + 1}"
                    )
                print(
                    f"LOADER_CHECK workers={workers} batch={i + 1}/{args.batches}",
                    flush=True,
                )
            if next(batches, None) is not None:
                raise RuntimeError("Unexpected extra batch")
            summaries.append(
                {
                    "workers": workers,
                    "wall_seconds_including_hash_checks_and_shutdown": perf_counter()
                    - start,
                    "first_batch_wait_seconds": waits[0],
                    "mean_wait_seconds_after_two_batches": sum(waits[2:])
                    / len(waits[2:]),
                    "sampler_fallbacks": sampler.fallbacks,
                }
            )
        if summaries[0]["sampler_fallbacks"] != summaries[1]["sampler_fallbacks"]:
            raise RuntimeError("Sampler fallback accounting changed")
        report = {
            "batches_equal": True,
            "batches": args.batches,
            "summaries": summaries,
            "training_started": False,
            "test_images_used": False,
            "limitation": "CPU checks only; serial runs first; cache/startup/consumer hashing affect timings. Not a GPU training speedup measurement.",
        }
        write(output / "loader_check.json", report)
        completed(output, batches_equal=True, training_started=False)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        write(
            output / "status.json",
            {
                "state": "interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "failed",
                "error_type": type(exc).__name__,
                "scientific_acceptance": "not_evaluated",
            },
        )
        raise


if __name__ == "__main__":
    main()
