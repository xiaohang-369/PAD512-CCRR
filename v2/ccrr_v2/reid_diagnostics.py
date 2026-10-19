"""Finite ReID diagnostics: deterministic sampler replay and frozen inference."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ccrr.data.pad import PADData, PairSampler
from .loading import ReIDBatchIndices
from .reid import evaluate, initialize
from .runtime import (
    completed,
    device_for,
    digest,
    new_run,
    read,
    seed_all,
    validate_resources,
    write,
)

TRAINING_COMMIT = "d585c50612282b01786faaef9bb573b266bb144c"
STEPS = (1000, 8000, 10000)
DOMAINS = ("rgb_rgb", "gray_rgb", "gray_gray")


def distribution(values):
    a = np.asarray(list(values), dtype=np.int64)
    return {
        "count": len(a),
        "zero": int((a == 0).sum()),
        "sum": int(a.sum()),
        "min": int(a.min()) if len(a) else None,
        "median": float(np.median(a)) if len(a) else None,
        "max": int(a.max()) if len(a) else None,
    }


def coverage(data, seed, steps, progress=False):
    """Read row metadata only; replay FULL RNG calls, including unused negatives."""
    sampler = PairSampler(data)
    patients, images, lesions = Counter(), Counter(), Counter()
    by_domain = {d: Counter() for d in ("rgb", "gray")}
    by_scenario = {s: Counter() for s in ("p1", "p2")}
    diagnoses = Counter()
    duplicate_batches = duplicate_slots = single_patient_batches = 0
    distinct_patient_counts = []
    for step, indices in enumerate(ReIDBatchIndices(sampler, seed, steps), 1):
        rows = [data.rows[i] for i in indices]
        domain = "rgb" if step % 2 else "gray"
        pids = [r["patient_id"] for r in rows]
        unique_images = len({r["img_id"] for r in rows})
        duplicate_batches += unique_images < len(rows)
        duplicate_slots += len(rows) - unique_images
        distinct_patient_counts.append(len(set(pids)))
        single_patient_batches += len(set(pids)) == 1
        for j, r in enumerate(rows):
            pid = r["patient_id"]
            patients[pid] += 1
            images[r["img_id"]] += 1
            lesions[(pid, r["lesion_id"])] += 1
            by_domain[domain][pid] += 1
            by_scenario["p1" if j < 8 else "p2"][pid] += 1
            diagnoses[r["diagnostic"]] += 1
        if progress and step % 1000 == 0:
            print(f"COVERAGE_REPLAY {step}/{steps}", flush=True)
    all_patients = {r["patient_id"] for r in data.rows}
    all_lesions = {(r["patient_id"], r["lesion_id"]) for r in data.rows}
    p1, p2 = set(sampler.p1), set(sampler.p2)
    result = {
        "role": "fit",
        "seed": seed,
        "steps": steps,
        "method": "deterministic_replay_not_runtime_trace",
        "fit_images": len(data.rows),
        "fit_patients": len(all_patients),
        "fit_lesions": len(all_lesions),
        "eligible_observation_patients": len(sampler.all),
        "p1_eligible_patients": len(p1),
        "p2_eligible_patients": len(p2),
        "positive_pool_union_patients": len(p1 | p2),
        "positive_pool_intersection_patients": len(p1 & p2),
        "patients_outside_positive_pools": len(all_patients - (p1 | p2)),
        "sampled_patients": len(patients),
        "sampled_images": len(images),
        "sampled_lesions": len(lesions),
        "classifier_classes_never_positive_targets": len(
            all_patients - patients.keys()
        ),
        "image_slot_draws_per_patient_including_zero": distribution(
            patients[p] for p in sorted(all_patients)
        ),
        "image_slot_draws_per_sampled_patient": distribution(patients.values()),
        "image_slot_draws_per_fit_image_including_zero": distribution(
            images[r["img_id"]] for r in data.rows
        ),
        "sampled_patients_by_domain": {d: len(c) for d, c in by_domain.items()},
        "sampled_patients_by_positive_scenario": {
            s: len(c) for s, c in by_scenario.items()
        },
        "sampled_image_slots_by_diagnosis": dict(sorted(diagnoses.items())),
        "batches_with_repeated_image": duplicate_batches,
        "repeated_image_slots": duplicate_slots,
        "distinct_patients_per_batch": distribution(distinct_patient_counts),
        "batches_without_triplet_negative": single_patient_batches,
        "pair_sampler_fallbacks": sampler.fallbacks,
        "images_decoded": 0,
        "model_parameters_updated": False,
        "interpretation": "Never-positive classes still receive negative softmax gradients; coverage does not by itself establish the cause of poor generalization.",
    }
    # Keep this table on the private server; stdout contains aggregates only.
    private = [
        {
            "patient_id": p,
            "image_slot_draws": patients[p],
            "rgb_slots": by_domain["rgb"][p],
            "gray_slots": by_domain["gray"][p],
            "p1_slots": by_scenario["p1"][p],
            "p2_slots": by_scenario["p2"][p],
        }
        for p in sorted(all_patients)
    ]
    return result, private


def validate_parent(resources, parent):
    binding, status = read(parent / "binding.json"), read(parent / "status.json")
    if (
        binding.get("stage") != "reid"
        or binding.get("source_commit") != TRAINING_COMMIT
    ):
        raise ValueError("Expected the completed ordered-prefetch V2 ReID source")
    if binding.get("resources") != resources:
        raise ValueError("Parent resource binding differs")
    cfg = binding["config"]
    if (cfg.get("seed"), cfg.get("reid_steps"), cfg.get("batch_size")) != (
        20261019,
        10000,
        16,
    ):
        raise ValueError(
            "Unexpected parent budget/seed/batch; do not silently change diagnostic"
        )
    if (
        status.get("state") != "completed"
        or type(status.get("pair_sampler_fallbacks")) is not int
    ):
        raise ValueError("Parent must be completed with a recorded fallback count")
    return cfg, status


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        v = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((v.dtype, tuple(v.shape))).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def load_checkpoint(model, path, step):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("step") != step:
        raise ValueError(f"Checkpoint step mismatch: {path.name}")
    model.load_state_dict(checkpoint["model"], strict=True)


def frozen_report(model, data, device, cfg):
    if data.role != "val_tune":
        raise ValueError("Diagnostic inference is restricted to val_tune")
    model.eval().requires_grad_(False)
    before = state_digest(model)
    with torch.inference_mode():
        report = evaluate(model, data, device, cfg, bootstrap=False)
    after = state_digest(model)
    if before != after:
        raise RuntimeError("Frozen inference changed parameters or buffers")
    return report, {"before": before, "after": after, "unchanged": True}


def compact(report):
    keys = ("valid", "auroc", "rank1", "map", "chance_rank1", "queries", "patients")
    output = {}
    for scenario in ("p1", "p2"):
        output[scenario] = {}
        for domain, item in zip(DOMAINS, report[scenario]["reports"], strict=True):
            r = item["cosine"]
            output[scenario][domain] = {k: r.get(k) for k in keys}
            matched = r.get("diagnosis_region_matched", {})
            output[scenario][domain]["diagnosis_region_matched"] = {
                k: matched.get(k) for k in keys
            }
    return output


def compare(resources, cfg, fit, tune, parent, output, device):
    results = []
    for step in (0, *STEPS):
        name = "public_init" if step == 0 else f"step_{step}"
        print(f"FROZEN_EVAL_BEGIN {name} role=val_tune images={len(tune)}", flush=True)
        seed_all(cfg["seed"])
        model = initialize(resources, len(fit.patients))
        if step:
            load_checkpoint(model, parent / f"reid-step-{step}.pt", step)
        model.to(device)
        report, unchanged = frozen_report(model, tune, device, cfg)
        write(output / f"private-report-{name}.json", report)
        metrics = compact(report)
        result = {
            "candidate": name,
            "step": step,
            "metrics": metrics,
            "state_sha256": unchanged,
        }
        results.append(result)
        write(output / f"summary-{name}.json", result)
        print(f"FROZEN_EVAL_DONE {name} " + json.dumps(metrics), flush=True)
        del model
    for result in results:
        result["auroc_delta_vs_public_init"] = {
            scenario: {
                domain: (
                    result["metrics"][scenario][domain]["auroc"]
                    - results[0]["metrics"][scenario][domain]["auroc"]
                )
                if result["metrics"][scenario][domain]["auroc"] is not None
                and results[0]["metrics"][scenario][domain]["auroc"] is not None
                else None
                for domain in DOMAINS
            }
            for scenario in ("p1", "p2")
        }
    return {
        "role": "val_tune",
        "images": len(tune),
        "patients": len(tune.patients),
        "candidates": results,
        "bootstrap": False,
        "new_checkpoint_selection": False,
        "training_started": False,
        "model_parameters_updated": False,
        "test_images_used": False,
        "val_select_images_used": False,
        "scope": "exploratory_development_comparison_not_acceptance_or_causal_proof",
        "limitations": [
            "Step 8000 was previously selected on this val_tune set; this is not independent validation.",
            "Point differences have no uncertainty interval and do not establish significance.",
            "Matched-negative reports can have different query cohorts; their differences are not causal shortcut estimates.",
        ],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resources", required=True)
    p.add_argument("--training-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    resources = read(args.resources)
    validate_resources(resources)
    parent = Path(args.training_run).resolve()
    runs = Path(resources["workspace_root"]).resolve() / "runs"
    if runs not in parent.parents or not Path(args.output).is_absolute():
        raise ValueError("Parent must be in V2 runs; output must be absolute")
    target = Path(args.output).resolve()
    if target == parent or parent in target.parents or target in parent.parents:
        raise ValueError("Diagnostic output must be separate from the parent run")
    cfg, status = validate_parent(resources, parent)
    files = [parent / "binding.json", parent / "status.json"] + [
        parent / f"reid-step-{s}.pt" for s in STEPS
    ]
    source_hashes = {str(f): digest(f) for f in files}
    output = new_run(
        resources,
        args.output,
        "reid-diagnostics",
        {
            "seed": cfg["seed"],
            "replay_steps": cfg["reid_steps"],
            "candidates": ["public_init", *STEPS],
            "evaluation_role": "val_tune",
            "device": args.device,
            "training_started": False,
        },
        {"training_run": str(parent), "input_sha256": source_hashes},
    )
    try:
        # The generic training flag is ambiguous here. Record explicit input
        # intent; a successful comparison will verify frozen V2 inference.
        binding = read(output / "binding.json")
        binding.pop("old_trained_checkpoints_loaded", None)
        binding.update(
            v1_trained_checkpoints_loaded=False,
            existing_v2_checkpoints_requested=list(STEPS),
        )
        write(output / "binding.json", binding)
        fit = PADData(resources["data_root"], "fit")
        replay, private = coverage(fit, cfg["seed"], cfg["reid_steps"], progress=True)
        replay["recorded_pair_sampler_fallbacks"] = status["pair_sampler_fallbacks"]
        replay["fallback_count_matches"] = (
            replay["pair_sampler_fallbacks"] == status["pair_sampler_fallbacks"]
        )
        write(output / "coverage_summary.json", replay)
        write(output / "private-patient-draws.json", private)
        print("COVERAGE_SUMMARY " + json.dumps(replay), flush=True)
        if not replay["fallback_count_matches"]:
            raise RuntimeError(
                "Replay fallback count disagrees with parent; stop before inference"
            )
        tune = PADData(resources["data_root"], "val_tune")
        summary = compare(
            resources, cfg, fit, tune, parent, output, device_for(args.device)
        )
        write(output / "comparison_summary.json", summary)
        table = [
            "candidate\tscenario\tdomain\tauroc\trank1\tmap\tchance_rank1\tqueries\tpatients\tdelta_auroc_vs_init"
        ]
        for candidate in summary["candidates"]:
            for scenario in ("p1", "p2"):
                for domain in DOMAINS:
                    m = candidate["metrics"][scenario][domain]
                    values = (
                        [candidate["candidate"], scenario, domain]
                        + [
                            m[k]
                            for k in (
                                "auroc",
                                "rank1",
                                "map",
                                "chance_rank1",
                                "queries",
                                "patients",
                            )
                        ]
                        + [candidate["auroc_delta_vs_public_init"][scenario][domain]]
                    )
                    table.append(
                        "\t".join(
                            str(v) if not isinstance(v, float) else f"{v:.6f}"
                            for v in values
                        )
                    )
        (output / "comparison_table.tsv").write_text("\n".join(table) + "\n")
        if source_hashes != {str(f): digest(f) for f in files}:
            raise RuntimeError("Parent input files changed during diagnostics")
        completed(
            output,
            training_started=False,
            model_parameters_updated=False,
            test_images_used=False,
            val_select_images_used=False,
            parent_inputs_unchanged=True,
            scope="two_development_diagnostics_only",
        )
        print(f"DIAGNOSTICS_COMPLETED output={output}", flush=True)
    except BaseException as error:
        write(
            output / "status.json",
            {
                "state": "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "training_started": False,
                "scientific_acceptance": "not_evaluated",
            },
        )
        raise


if __name__ == "__main__":
    main()
