"""Real Stage B capacity experiment only; production training sources stay frozen."""

import copy
import csv
import gc
import itertools
import statistics
import time

import torch
from torch.nn import functional as F

from b_r3_cached_execution import CachedPairSampler
from b_r3_h100_scaling import Head
from ccrr.codec.metadata import CLASSES
from ccrr.data.pad import rng_for
from ccrr.errors import require
from ccrr.tensor_ops import masked_mean, pooled
from ccrr.training.b_batches import semantic_batch
from ccrr.training.losses import weighted_cross_entropy
from ccrr.training.optimization import schedule, update


def sample_pairs(e, rng, batch, scenario="mixed"):
    return [p for _ in range(batch // 16) for p in e.sampler.sample(rng, scenario)]


def pair_backward(e, bank, pairs, rng, coefficient, detach):
    domains = [(int(rng.integers(3)), int(rng.integers(3))) for _ in pairs]
    ta = bank.get([p[0] for p in pairs])
    tb = bank.get([p[1] for p in pairs])
    with torch.set_grad_enabled(not detach):
        fa, va = semantic_batch(e, ta, [d[0] for d in domains])
        fb, vb = semantic_batch(e, tb, [d[1] for d in domains])
    logits = e.sem_attacker(fa, va, fb, vb)
    loss = (
        coefficient
        * F.binary_cross_entropy_with_logits(
            logits, logits.new_tensor([p[2] for p in pairs]), reduction="sum"
        )
        / len(pairs)
    )
    loss.backward()
    return loss.detach()


def image_backward(e, bank, ids):
    t = bank.get(ids)
    v = t["v"]
    px, py = e.purifier(t["f0"], v), e.purifier(t["f1"], v)
    cross = e.purifier(t["f1"], v, source_mask=px["mask"])
    loss = torch.zeros((), device=e.device)
    for out, target in ((px, t["e0"]), (py, t["e1"])):
        all_loss = masked_mean((out["all"] - out["standardized"]).square(), v)
        identity = (out["identity"] - target).square().sum(1).mean()
        occupancy = pooled(out["mask"], v).mean(1)
        mask = (
            F.relu(0.1 - occupancy).square() + F.relu(occupancy - 0.5).square()
        ).mean()
        loss = loss + (all_loss + 0.1 * identity + 10 * mask) / 2
    labels = torch.tensor(
        [CLASSES.index(e.fit.rows[int(i)]["diagnostic"]) for i in ids], device=e.device
    )
    denominator = e.weights[labels].sum() * 3
    for out in (px, py, cross):
        loss = loss + weighted_cross_entropy(
            out["logits"], labels, e.weights, weight_sum=denominator
        )
    loss.backward()
    return loss.detach()


def purifier_update(work, batch):
    e, bank = work.e, work.banks["fit"]
    step = work.schedules["purifier"].next_step
    opt, adv = work.opts["purifier"], work.opts["attacker"]
    e.purifier.train().requires_grad_(True)
    e.sem_attacker.train().requires_grad_(True)
    attacks = []
    for sub in range(2):
        rng = rng_for(e.seed, "B", "attacker" + str(sub), step)
        pairs = sample_pairs(e, rng, batch)
        adv.zero_grad(set_to_none=True)
        attacks.append(pair_backward(e, bank, pairs, rng, 1.0, True))
        work.schedules["attacker"].apply(2 * step + sub)
        update(adv)
        work.schedules["attacker"].advance()
    e.sem_attacker.eval().requires_grad_(False)
    adv.zero_grad(set_to_none=True)
    opt.zero_grad(set_to_none=True)
    rng = rng_for(e.seed, "B", "images", step)
    loss = image_backward(e, bank, rng.integers(len(e.fit), size=batch))
    ramp = max(0, min(1, (step - 999) / 1000))
    if ramp:
        pairs = sample_pairs(e, rng_for(e.seed, "B", "generator_pairs", step), batch)
        loss += pair_backward(e, bank, pairs, rng, -0.1 * ramp, False)
    work.schedules["purifier"].apply(step)
    update(opt)
    work.schedules["purifier"].advance()
    return {"loss": loss, "attacker_loss": torch.stack(attacks).mean()}


def control_update(work, batch):
    e, bank = work.e, work.banks["fit"]
    scheduler, opt = work.schedules["control"], work.opts["control"]
    ids = (
        rng_for(e.seed, "B", "images", scheduler.next_step)
        .integers(len(e.fit), size=batch)
        .tolist()
    )
    model = e.no_branch_control
    model.train().requires_grad_(True)
    opt.zero_grad(set_to_none=True)
    targets = bank.get(ids)
    values = []
    for key in ("f0", "f1"):
        out = model(targets[key], targets["v"])
        loss = (
            masked_mean((out["all"] - out["standardized"]).square(), targets["v"]) / 2
        )
        loss.backward()
        values.append(loss.detach())
    scheduler.apply(scheduler.next_step)
    update(opt)
    scheduler.advance()
    return {"loss": sum(values)}


def head_update(head, batch):
    # One actual batch, no accumulation of batch-16 forward/backward graphs.
    e, model, opt = head.e, head.model, head.opt
    model.train()
    opt.zero_grad(set_to_none=True)
    rng = rng_for(
        e.seed, "feature-attack", head.scenario + head.rep + head.family, head.next
    )
    pairs, domains = [], []
    for _ in range(batch // 16):
        block = e.sampler.sample(rng, head.scenario)
        pairs.extend(block)
        domains.extend((int(rng.integers(3)), int(rng.integers(3))) for _ in block)
    fa, va = head.audit.feature_batch(
        "fit", [p[0] for p in pairs], head.rep, [d[0] for d in domains]
    )
    fb, vb = head.audit.feature_batch(
        "fit", [p[1] for p in pairs], head.rep, [d[1] for d in domains]
    )
    logits = model(fa, va, fb, vb)
    loss = (
        F.binary_cross_entropy_with_logits(
            logits, logits.new_tensor([p[2] for p in pairs]), reduction="sum"
        )
        / batch
    )
    loss.backward()
    schedule(opt, head.next, 5000)
    update(opt)
    head.next += 1
    return {"loss": loss.detach()}


@torch.no_grad()
def evaluate(work, banks, head=None):
    from sklearn.metrics import roc_auc_score

    if head is not None:
        rows = head.audit.attack_scores(head.model, head.rep, head.scenario)
        auc = [
            float(roc_auc_score(r["labels"], r["scores"], sample_weight=r["weights"]))
            for r in rows
        ]
        return {
            "nine_direction_auc": auc,
            "nine_direction_a_dagger": [max(a, 1 - a) for a in auc],
        }
    e = work.e
    model = e.purifier if work.kind == "purifier" else e.no_branch_control
    model.eval()
    ds = e.data["val_select"]
    scores = {k: [] for k in ("rgb_mse", "gray_mse")}
    logits, labels, masks = [[], [], []], [], [[], [], []]
    semantic_features, coverages = [], []
    for start in range(0, len(ds), 16):
        ids = list(range(start, min(start + 16, len(ds))))
        t = banks["val_select"].get(ids)
        px, py = model(t["f0"], t["v"]), model(t["f1"], t["v"])
        for name, out in (("rgb_mse", px), ("gray_mse", py)):
            scores[name].append(
                float(masked_mean((out["all"] - out["standardized"]).square(), t["v"]))
                * len(ids)
            )
        if work.kind == "purifier":
            cross = model(t["f1"], t["v"], source_mask=px["mask"])
            semantic_features.append(
                torch.stack([p["semantic"] for p in (px, py, cross)], 1)
            )
            coverages.append(t["v"])
            for d, out in enumerate((px, py, cross)):
                logits[d].append(out["logits"].softmax(1).cpu())
                masks[d].append(pooled(out["mask"], t["v"]).mean(1).cpu())
            labels.extend(CLASSES.index(ds.rows[i]["diagnostic"]) for i in ids)
    result = {name: sum(values) / len(ds) for name, values in scores.items()}
    if work.kind == "purifier":
        result["clinical_classifier_macro_auc"] = []
        result["clinical_accuracy"] = []
        result["mask_occupancy_in_range_fraction"] = []
        target = torch.tensor(labels)
        for d in range(3):
            prob = torch.cat(logits[d])
            result["clinical_accuracy"].append(
                float((prob.argmax(1) == target).float().mean())
            )
            per_class = [
                float(roc_auc_score((target == c).numpy(), prob[:, c].numpy()))
                for c in range(len(CLASSES))
                if 0 < int((target == c).sum()) < len(target)
            ]
            result["clinical_classifier_macro_auc"].append(
                statistics.mean(per_class) if per_class else None
            )
            occupancy = torch.cat(masks[d])
            result["mask_occupancy_in_range_fraction"].append(
                float(((occupancy >= 0.1) & (occupancy <= 0.5)).float().mean())
            )
        result["metric_scope"] = (
            "Existing Purifier classifier/mask development checks; not independent clinical readout or privacy admission. Full independent audit remains required."
        )
        from b_r3_gpu_features import GPUFeatureAudit

        diagnostic = GPUFeatureAudit(e)
        diagnostic.store = {
            "val_select": {
                "sem": torch.cat(semantic_features),
                "v": torch.cat(coverages),
            }
        }
        result["coupled_adversary_auc_proxy_not_independent_audit"] = {}
        for scenario in ("p1", "p2"):
            rows = diagnostic.attack_scores(e.sem_attacker, "sem", scenario)
            result["coupled_adversary_auc_proxy_not_independent_audit"][scenario] = [
                float(
                    roc_auc_score(r["labels"], r["scores"], sample_weight=r["weights"])
                )
                for r in rows
            ]
    return result


def search_batches(trial, *, cap=65536):
    """Exponential search then a 16-pair/image grid bracket; no plateau stop."""
    good, bad, batch = 0, None, 16
    while batch <= cap:
        if trial(batch):
            good, batch = batch, batch * 2
        else:
            bad = batch
            break
    if bad is not None and good:
        while bad - good > 16:
            middle = ((good + bad) // 32) * 16
            if trial(middle):
                good = middle
            else:
                bad = middle
    return {
        "largest_tested_stable_batch": good,
        "first_failing_batch_bound": bad,
        "search_cap_reached": bad is None,
        "batch_grid": 16,
        "meaning": "Largest successful short-run batch tested; binary refinement assumes local memory feasibility is monotone. Not a guarantee for arbitrary data/long training.",
    }


def gpu_stats(guard, phase):
    values = [r["nvidia_smi"].split(",") for r in guard.rows if r["phase"] == phase]
    return {
        "gpu_util_samples": len(values),
        "sampled_gpu_util_percent": statistics.mean(float(v[2]) for v in values)
        if values
        else None,
    }


def run_full_capacity(e, banks, audit, engine, snapshot, args, guard):
    from profile_b_r3_h100 import Workload, timed, write

    root = args.out / "full-capacity"
    root.mkdir()
    report = {
        "complete": False,
        "results": {},
        "formal_training_started": False,
        "scope": "Purifier, Control and all 12 identity heads. Real frozen teacher features built from real server images; trainable networks always recomputed with full forward/backward/optimizer updates.",
        "loss_contract": "Original coefficients, quantization and models. Sample means normalized by actual batch. Purifier keeps two adversary updates and one generator update. LR schedules remain update-based.",
        "stability_scope": f"Each accepted batch: {args.repeats} repeats, {args.warmup} warmup updates then at least {args.steps} updates and 2 seconds per timed window; reset from same start. Actual steps recorded. Not long-run convergence.",
        "validation_scope": "20 updates each at original and best batch from identical initial state. Sample exposures differ and are explicitly counted; not an equal-sample convergence comparison. Scientific thresholds unchanged, full admission not performed.",
        "measurement_scope": "Throughput ratios compare the same generalized runner at B=16 against larger batches, not a complete R3 wall-clock speedup. Null GPU utilization means no monitor sample landed in that short timing window. Memory includes all resident frozen caches and existing runtime models.",
    }

    def persist():
        write(root / "summary.json", report)
        columns = [
            "workload",
            "original_batch",
            "sample_unit",
            "largest_tested_stable_batch",
            "search_cap_reached",
            "best_throughput_batch",
            "pairs_or_images_per_second",
            "speedup_vs_batch16",
            "best_seconds_per_step",
            "fixed_update_count_speedup",
            "best_peak_allocated_GiB",
            "largest_batch_peak_allocated_GiB",
            "best_gpu_util_percent",
            "validation_status",
        ]
        with (root / "comparison.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for name, row in report["results"].items():
                writer.writerow(
                    {"workload": name, **{k: row.get(k) for k in columns[1:]}}
                )

    source = copy.copy(audit)
    source.e = copy.copy(e)
    source.e.sampler = CachedPairSampler(e.sampler)
    specs = ["purifier", "control"] + list(
        itertools.product(
            ("p1", "p2"), ("raw", "control", "sem"), ("spatial", "pooled")
        )
    )
    persist()
    for spec in specs:
        name = spec if isinstance(spec, str) else "identity-" + "-".join(spec)
        head = work = None
        if isinstance(spec, str):
            work = Workload(e, banks, audit, engine, root / name, name, snapshot)
            e.sampler = CachedPairSampler(e.sampler)
            work.e.sampler = e.sampler
            reset = work.reset

            def step(batch, work=work):
                return (purifier_update if work.kind == "purifier" else control_update)(
                    work, batch
                )
        else:
            head = Head(source, spec)
            reset = head.restore

            def step(batch, head=head):
                return head_update(head, batch)

        row = report["results"][name] = {
            "original_batch": 16,
            "sample_unit": "images" if isinstance(spec, str) else "pairs",
            "cases": {},
            "validation_status": "pending",
        }
        persist()

        def trial(batch):
            case = row["cases"][str(batch)] = {"repeats": [], "status": "running"}
            print(f"FULL_CAPACITY_START {name} batch={batch}", flush=True)
            failed = False
            try:
                for repeat in range(args.repeats):
                    guard.check()
                    reset()
                    gc.collect()
                    torch.cuda.empty_cache()
                    guard.phase = f"full/{name}/{batch}/warmup/{repeat}"
                    for _ in range(args.warmup):
                        guard.check()
                        step(batch)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    phase = f"full/{name}/{batch}/updates/{repeat}"
                    guard.phase = phase

                    def loop():
                        values = []
                        started = time.perf_counter()
                        minimum_seconds = getattr(args, "capacity_min_seconds", 2.0)
                        while (
                            len(values) < args.steps
                            or time.perf_counter() - started < minimum_seconds
                        ):
                            guard.check()
                            metrics = step(batch)
                            values.append(torch.stack(list(metrics.values())))
                        return torch.stack(values), list(metrics)

                    (losses, keys), timing = timed(loop)
                    if not bool(torch.isfinite(losses).all()):
                        raise FloatingPointError("Nonfinite training loss")
                    values = losses.cpu().tolist()
                    measured_steps = len(values)
                    r = {
                        "repeat": repeat,
                        "batch": batch,
                        "steps": measured_steps,
                        "sample_exposures": batch * measured_steps,
                        "warmup_sample_exposures": batch * args.warmup,
                        "seconds_per_step": timing["wall_seconds"] / measured_steps,
                        "samples_per_second": batch
                        * measured_steps
                        / timing["wall_seconds"],
                        "loss_first": dict(zip(keys, values[0])),
                        "loss_last": dict(zip(keys, values[-1])),
                        "loss_range": {
                            key: [min(v[i] for v in values), max(v[i] for v in values)]
                            for i, key in enumerate(keys)
                        },
                        "peak_allocated_GiB": torch.cuda.max_memory_allocated()
                        / 1024**3,
                        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
                        **timing,
                        **gpu_stats(guard, phase),
                    }
                    case["repeats"].append(r)
                    print("FULL_CAPACITY_REPEAT", name, batch, r, flush=True)
                    persist()
            except torch.cuda.OutOfMemoryError:
                case["status"] = "oom"
                failed = True
            except FloatingPointError as exc:
                case["status"] = "numerically_unstable"
                case["error"] = str(exc)
                failed = True
            if failed:
                reset()
                gc.collect()
                torch.cuda.empty_cache()
                persist()
                return False
            case["status"] = "stable_short_run"
            case["throughput"] = statistics.median(
                r["samples_per_second"] for r in case["repeats"]
            )
            persist()
            return True

        row.update(search_batches(trial))
        require(
            row["largest_tested_stable_batch"] >= 16,
            "Original batch failed; stop this probe",
        )
        successes = [
            (int(b), c)
            for b, c in row["cases"].items()
            if c["status"] == "stable_short_run"
        ]
        best_batch, best_case = max(successes, key=lambda item: item[1]["throughput"])
        row["largest_batch_peak_allocated_GiB"] = max(
            r["peak_allocated_GiB"]
            for r in row["cases"][str(row["largest_tested_stable_batch"])]["repeats"]
        )
        utils = [
            r["sampled_gpu_util_percent"]
            for r in best_case["repeats"]
            if r["sampled_gpu_util_percent"] is not None
        ]
        row.update(
            best_throughput_batch=best_batch,
            pairs_or_images_per_second=best_case["throughput"],
            speedup_vs_batch16=best_case["throughput"]
            / row["cases"]["16"]["throughput"],
            best_seconds_per_step=statistics.median(
                r["seconds_per_step"] for r in best_case["repeats"]
            ),
            best_peak_allocated_GiB=max(
                r["peak_allocated_GiB"] for r in best_case["repeats"]
            ),
            best_gpu_util_percent=statistics.mean(utils) if utils else None,
        )
        row["fixed_update_count_speedup"] = (
            statistics.median(
                r["seconds_per_step"] for r in row["cases"]["16"]["repeats"]
            )
            / row["best_seconds_per_step"]
        )
        persist()
        validation = row["validation"] = {
            "variants": {},
            "scope": "Same 20 updates, different sample budgets; do not interpret as equal-budget efficacy comparison.",
        }
        for label, batch in (("original", 16), ("best", best_batch)):
            guard.phase = f"full/{name}/validation/{label}"
            reset()
            before = evaluate(work, banks, head)
            count = 20

            def validate():
                values = []
                for _ in range(count):
                    guard.check()
                    values.append(step(batch)["loss"])
                return torch.stack(values)

            losses, timing = timed(validate)
            require(bool(torch.isfinite(losses).all()), "Nonfinite validation loss")
            after = evaluate(work, banks, head)
            validation["variants"][label] = {
                "batch": batch,
                "updates": count,
                "sample_exposures": count * batch,
                "additional_adversary_pair_exposures": count * batch * 2
                if name == "purifier"
                else 0,
                "additional_generator_pair_exposures": count * batch
                if name == "purifier"
                else 0,
                "loss_first": float(losses[0]),
                "loss_last": float(losses[-1]),
                "before": before,
                "after": after,
                **timing,
            }
            persist()
        row["validation_status"] = (
            "finite_training_and_heldout_metrics_recorded_not_scientific_admission"
        )
        print(
            "FULL_CAPACITY_RESULT",
            name,
            {k: v for k, v in row.items() if k not in ("cases", "validation")},
            flush=True,
        )
        persist()
        # Drop callable references before freeing this workload's models/optimizer.
        trial = step = reset = head = work = None
        gc.collect()
        torch.cuda.empty_cache()
    report["complete"] = True
    report["brief"] = {
        "complete": True,
        "comparison_csv": "full-capacity/comparison.csv",
        "full_results": "full-capacity/summary.json",
        "scientific_admission_complete": False,
    }
    persist()
    return report
