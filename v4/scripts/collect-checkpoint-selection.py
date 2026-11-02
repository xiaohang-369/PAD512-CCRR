#!/usr/bin/env python3
"""CPU-only collection of the two bound joint runs; never import torch."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def collect(root, output, cfg):
    if cfg.get("schema") != "ccrr-v4-checkpoint-selection-1":
        raise ValueError("Unexpected selection config")
    root, output = root.resolve(), output.resolve()
    sources = [root / "runs" / item["name"] for item in cfg["runs"]]
    if any(output == p or p in output.parents for p in sources):
        raise ValueError("Export must be separate from both training runs")
    output.mkdir(parents=True, exist_ok=False)
    provenance, issues, checks, checkpoints, evaluations = {}, [], [], [], {}
    metadata = {}

    def problem(message):
        issues.append(message)

    def capture(path, relative):
        if not path.is_file():
            problem("Missing evidence: " + str(path))
            return None
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        dest = output / "received" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        provenance[str(path)] = {"sha256": sha, "export": str(dest.relative_to(output))}
        return raw

    def read(path, relative):
        raw = capture(path, relative)
        return json.loads(raw) if raw is not None else None

    def check(condition, name):
        checks.append({"check": name, "passed": bool(condition)})
        if not condition:
            problem(name)

    for spec, run in zip(cfg["runs"], sources):
        label = spec["label"]
        print("Inspecting " + str(run), flush=True)
        docs = {
            name: read(run / (name + ".json"), Path(label) / (name + ".json"))
            for name in ("status", "report", "binding", "config", "preflight-binding")
        }
        metadata[label] = docs
        status, report, binding, train_cfg = [
            docs[k] for k in ("status", "report", "binding", "config")
        ]
        if any(d is None for d in (status, report, binding, train_cfg)):
            continue
        check(
            status.get("state") == "completed"
            and report.get("training_completed") is True,
            label + ": completed",
        )
        check(
            report.get("parents_unchanged") is True
            and report.get("frozen_models_unchanged") is True,
            label + ": reported parent/frozen integrity",
        )
        check(
            report.get("binding") == binding and binding.get("config") == train_cfg,
            label + ": report/config binding",
        )
        check(
            train_cfg.get("update_modules") == ["E1", "D1"]
            and train_cfg.get("epochs") == spec["last_epoch"],
            label + ": joint endpoint",
        )
        check(
            report.get("optimizer_updates") == spec["last_epoch"] * 90,
            label + ": final step",
        )
        final = report.get("final_checkpoint", {})
        check(
            final.get("sha256") == spec["final_sha256"], label + ": pinned final hash"
        )
        raw = capture(run / "metrics.jsonl", Path(label) / "metrics.jsonl")
        records = (
            [json.loads(line) for line in raw.splitlines() if line.strip()]
            if raw
            else []
        )
        expected = list(range(spec["first_epoch"], spec["last_epoch"] + 1))
        check(
            [r.get("epoch") for r in records] == expected,
            label + ": complete ordered metric epochs",
        )
        by_epoch = {r["epoch"]: r for r in records}
        actual_names = {p.name for p in run.glob("g1-epoch-*.pt") if p.is_file()}
        expected_names = {f"g1-epoch-{e:03d}.pt" for e in expected}
        check(
            not (actual_names - expected_names),
            label + ": no unexpected checkpoint filenames",
        )
        for epoch in expected:
            row = by_epoch.get(epoch)
            ev = read(
                run / f"evaluation-{epoch:03d}.json",
                Path(label) / f"evaluation-{epoch:03d}.json",
            )
            if ev is not None:
                summary = {k: v for k, v in ev.items() if k != "p2_details"}
                check(
                    ev.get("epoch") == epoch
                    and ev.get("role") == "val_tune"
                    and ev.get("images") == 169,
                    f"epoch {epoch}: evaluation scope",
                )
                if row:
                    check(
                        row.get("validation") == summary,
                        f"epoch {epoch}: metric/evaluation agreement",
                    )
                evaluations[epoch] = {
                    "summary": summary,
                    "train": row.get("train", {}) if row else {},
                    "source": "evaluation_file",
                }
            elif row:
                evaluations[epoch] = {
                    "summary": row["validation"],
                    "train": row.get("train", {}),
                    "source": "metrics_fallback_evaluation_file_missing",
                }
            path = run / f"g1-epoch-{epoch:03d}.pt"
            entry = dict(
                epoch=epoch,
                step=epoch * 90,
                path=str(path),
                exists=path.is_file(),
                internal_tensor_binding_checked=False,
            )
            if path.is_file():
                before = path.stat()
                actual = digest(path)
                after = path.stat()
                logged = row.get("checkpoint", {}) if row else {}
                matched = (
                    logged.get("sha256") == actual
                    and logged.get("epoch") == epoch
                    and logged.get("step") == epoch * 90
                    and logged.get("file") == path.name
                )
                stable = (before.st_size, before.st_mtime_ns) == (
                    after.st_size,
                    after.st_mtime_ns,
                )
                entry.update(
                    sha256=actual,
                    bytes=after.st_size,
                    matches_metrics_hash=matched,
                    stable_during_hash=stable,
                )
                check(
                    matched and stable, f"epoch {epoch}: checkpoint file hash/position"
                )
                if epoch == spec["last_epoch"]:
                    check(
                        actual == spec["final_sha256"] and final == logged,
                        label + ": actual final checkpoint binding",
                    )
                entry["eligible_for_later_tensor_verification"] = matched and stable
            else:
                entry["eligible_for_later_tensor_verification"] = False
            checkpoints.append(entry)
            if epoch % 10 == 0:
                print(f"Hashed inventory through epoch={epoch}", flush=True)
        if records:
            check(
                records[-1].get("validation") == report.get("final_validation"),
                label + ": final evaluation agreement",
            )
        pre = docs["preflight-binding"]
        if label == "joint" and pre:
            path = Path(pre["path"])
            initial = read(path, Path("initial") / "report.json")
            if initial:
                check(
                    provenance[str(path)]["sha256"] == pre["sha256"]
                    and initial.get("binding") == binding
                    and initial.get("passed") is True,
                    "epoch 0: original preflight binding",
                )
                ev = read(
                    path.parent / "evaluation-000.json",
                    Path("initial") / "evaluation-000.json",
                )
                if ev:
                    summary = {k: v for k, v in ev.items() if k != "p2_details"}
                    check(
                        summary == initial.get("initial_evaluation")
                        and ev.get("epoch") == 0,
                        "epoch 0: preflight evaluation agreement",
                    )
                    evaluations[0] = {
                        "summary": summary,
                        "train": {},
                        "source": "original_preflight",
                    }
                # Epoch 0 is reconstructed from the original E0; no fictitious E1 epoch-000 file.
                inspection_path = Path(train_cfg["inspection_report"])
                inspection = read(
                    inspection_path, Path("initial") / "resource-inspection.json"
                )
                if inspection:
                    check(
                        provenance[str(inspection_path)]["sha256"]
                        == binding.get("parents", {}).get(str(inspection_path)),
                        "epoch 0: resource binding",
                    )
                    baseline = inspection["results"]["baseline"]
                    base_path = Path(baseline["checkpoint"])
                    base_sha = digest(base_path) if base_path.is_file() else None
                    expected_sha = binding.get("parents", {}).get(str(base_path))
                    check(
                        base_sha is not None and base_sha == expected_sha,
                        "epoch 0: original E0 checkpoint hash",
                    )
                    write(
                        output / "epoch0_source.json",
                        dict(
                            kind="original_E0_copy",
                            path=str(base_path),
                            sha256=base_sha,
                            expected_sha256=expected_sha,
                            internal_tensor_binding_checked=False,
                        ),
                    )

    joint, resume = metadata.get("joint", {}), metadata.get("resume", {})
    if joint.get("config") and resume.get("config"):
        old, new = joint["config"], resume["config"]
        check(
            {k: v for k, v in old.items() if k not in ("epochs", "gpu_ids")}
            == {
                k: v for k, v in new.items() if k not in ("epochs", "gpu_ids", "resume")
            },
            "continuation: unchanged training mechanism",
        )
        spec = new.get("resume", {})
        check(
            spec.get("run") == str(sources[0])
            and spec.get("sha256") == cfg["runs"][0]["final_sha256"],
            "continuation: exact parent binding",
        )
    for path, record in provenance.items():
        check(digest(Path(path)) == record["sha256"], "metadata unchanged: " + path)
    flat = []
    for epoch in range(61):
        item = evaluations.get(epoch)
        zero_file = output / "epoch0_source.json"
        zero = json.loads(zero_file.read_text()) if zero_file.exists() else {}
        row = dict(
            epoch=epoch,
            step=epoch * 90,
            metrics_available=item is not None,
            checkpoint_available=bool(zero.get("sha256"))
            and zero["sha256"] == zero.get("expected_sha256")
            if epoch == 0
            else any(c["epoch"] == epoch and c["exists"] for c in checkpoints),
            residual_bpp_s1="",
            residual_entropy="",
            compensated_psnr="",
        )
        if item:
            row["evaluation_source"] = item["source"]
            ev = item["summary"]
            row.update(ev.get("recovery", {}))
            for teacher, metrics in ev.get("medical", {}).items():
                for metric in ("macro_auroc", "macro_f1", "balanced_accuracy"):
                    row[f"medical_{teacher}_{metric}"] = metrics[metric]
                for i, value in enumerate(metrics.get("recall", [])):
                    row[f"medical_{teacher}_recall_{i}"] = value
            for route, metrics in ev.get("p2_fixed_reference", {}).items():
                for metric in ("rank1", "auroc", "map", "orientation_from_val_tune"):
                    row[route + "_" + metric] = metrics.get(metric)
            row.update({"train_" + k: v for k, v in item["train"].items()})
        flat.append(row)
    keys = list(dict.fromkeys(k for r in flat for k in r))
    with (output / "epoch_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)
    write(
        output / "evaluations.json", {str(k): v for k, v in sorted(evaluations.items())}
    )
    write(output / "checkpoints.json", checkpoints)
    write(output / "source_files.json", provenance)
    write(output / "collection_config.json", cfg)
    summary = dict(
        schema="ccrr-v4-checkpoint-inventory-1",
        metadata_checks_passed=not issues,
        checks=checks,
        issues=issues,
        evaluation_epochs=sorted(evaluations),
        missing_evaluation_epochs=sorted(set(range(61)) - set(evaluations)),
        missing_checkpoint_epochs=sorted(
            set(range(1, 61)) - {c["epoch"] for c in checkpoints if c["exists"]}
        ),
        checkpoint_files_present=sum(c["exists"] for c in checkpoints),
        tensor_states_loaded=False,
        gpu_used=False,
        residual_inference_executed=False,
        training_started=False,
        selected_epoch=None,
        scientific_acceptance="not_evaluated",
    )
    write(output / "inventory.json", summary)
    archive = output.with_name(output.name + ".tar.gz")
    with tarfile.open(archive, "x:gz") as tar:
        tar.add(output, arcname=output.name)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "metadata_checks_passed",
                    "evaluation_epochs",
                    "missing_evaluation_epochs",
                    "missing_checkpoint_epochs",
                    "checkpoint_files_present",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print("ISSUES=" + json.dumps(issues, ensure_ascii=False), flush=True)
    print("RETURN_ARCHIVE=" + str(archive), flush=True)
    print("ARCHIVE_SHA256=" + digest(archive), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    collect(args.root, args.output, json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
