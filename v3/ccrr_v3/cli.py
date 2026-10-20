"""One explicit V3 development stage at a time; no automatic downstream training."""

import argparse
import json
from pathlib import Path

import torch

from .runtime import (
    read,
    write,
    digest,
    validate_resources,
    validate_config,
    new_run,
    status,
    finish,
    artifact,
    device_for,
    log,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=(
            "preflight",
            "probe",
            "reid",
            "targets",
            "p3-baseline",
            "train",
            "audit",
            "compare",
        ),
    )
    p.add_argument("--resources", required=True)
    p.add_argument(
        "--config",
        required=True,
        help="Explicit registered finite development configuration",
    )
    p.add_argument("--output", help="New absolute directory below V3 workspace/runs")
    p.add_argument("--features", help="Read-only completed V2 A0 feature cache")
    p.add_argument("--heads", help="Completed V3 probe run")
    p.add_argument("--targets", help="Completed frozen V3 linkage target run")
    p.add_argument(
        "--reference",
        help="Optional fresh V3 P2 ReID run; otherwise public initialization",
    )
    p.add_argument(
        "--reference-review",
        help="Explicit development reference decision, bound to targets artifact",
    )
    p.add_argument("--separator", help="Completed V3 separator run")
    p.add_argument("--variant", choices=("compression", "separation", "adversarial"))
    p.add_argument("--view", choices=("raw", "full", "semantic", "q_sem", "q_link"))
    p.add_argument("--before", help="Completed baseline audit")
    p.add_argument("--after", help="Completed candidate audit")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Only P2 ReID loading; ordered CPU prefetch",
    )
    return p


def validate_arguments(args):
    required = {
        "probe": ("features",),
        "reid": (),
        "targets": ("features",),
        "train": ("features", "heads", "variant"),
        "audit": ("features", "view"),
        "compare": ("before", "after"),
        "preflight": (),
        "p3-baseline": (),
    }
    allowed = {
        "probe": {"features"},
        "reid": set(),
        "targets": {"features", "reference"},
        "train": {"features", "heads", "variant", "targets", "reference_review"},
        "audit": {"features", "view", "separator"},
        "compare": {"before", "after"},
        "preflight": {"features"},
        "p3-baseline": set(),
    }
    for key in required[args.stage]:
        if not getattr(args, key):
            raise ValueError(f"{args.stage} requires --{key.replace('_', '-')}")
    for key in (
        "features",
        "heads",
        "targets",
        "reference",
        "reference_review",
        "separator",
        "variant",
        "view",
        "before",
        "after",
    ):
        if getattr(args, key) is not None and key not in allowed[args.stage]:
            raise ValueError(f"Unexpected --{key} for {args.stage}")
    if args.workers < 0 or (args.stage != "reid" and args.workers):
        raise ValueError("Workers are nonnegative and only supported for reid")
    if args.stage == "train":
        if args.variant != "compression" and (
            not args.targets or not args.reference_review
        ):
            raise ValueError(
                "Separation requires targets and an explicit bound reference review"
            )
        if args.variant == "compression" and (args.targets or args.reference_review):
            raise ValueError(
                "Independent compression control must not consume linkage targets"
            )
    if args.stage == "audit" and bool(args.separator) != (args.view != "raw"):
        raise ValueError("Raw audit has no separator; all other views require one")
    if args.stage == "preflight" and args.output:
        raise ValueError("Preflight is read-only")


def load_targets(path, cache):
    obj = torch.load(path / "targets.pt", weights_only=True, map_location="cpu")
    result = {}
    for role, data in cache.items():
        if obj[role]["ids"] != [r["img_id"] for r in data["rows"]]:
            raise ValueError("Target image order/cohort mismatch")
        x = obj[role]["embeddings"]
        if (
            x.ndim != 3
            or x.shape[:2] != (len(data["rows"]), 2)
            or not torch.isfinite(x).all()
            or (x.norm(dim=2) < 1e-8).any()
        ):
            raise ValueError("Invalid frozen target tensor")
        result[role] = x
    if len({v.shape[2] for v in result.values()}) != 1:
        raise ValueError("Target dimensions differ")
    return result


def validate_reference_review(path, targets):
    review = read(path)
    if set(review) != {
        "schema",
        "targets_artifact_sha256",
        "decision",
        "directions",
        "reason",
    }:
        raise ValueError("Unknown reference review format")
    if review["schema"] != "ccrr-v3-reference-review-1" or review[
        "targets_artifact_sha256"
    ] != digest(targets / "artifact.json"):
        raise ValueError("Reference review is not bound to these targets")
    if (
        review["decision"] != "allow_development_reference"
        or set(review["directions"]) != {"rgb_rgb", "gray_rgb", "gray_gray"}
        or not review["reason"].strip()
    ):
        raise ValueError("Review must explicitly cover each required P2 direction")
    return review


def main(argv=None):
    args = parser().parse_args(argv)
    validate_arguments(args)
    resources, cfg = read(args.resources), read(args.config)
    validate_config(cfg, registered=args.stage != "preflight")
    validate_resources(resources)
    if args.stage == "preflight":
        from ccrr.data.admission import verify_dataset

        report = dict(
            resources_verified=True,
            data=verify_dataset(resources["data_root"], pixels=False),
            registration_status=cfg["registration_status"],
            training_started=False,
            p1_required=False,
            scientific_acceptance="not_evaluated",
        )
        if args.features:
            from .data import load_features

            cache, _ = load_features(args.features, resources, roles=("fit",))
            report["fit_features_verified"] = len(cache["fit"]["rows"])
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if not args.output:
        raise ValueError("Fresh --output required")
    parents = {}
    if args.features:
        parents.update(
            features=str(Path(args.features).resolve()),
            features_sha256=digest(Path(args.features) / "features.json"),
        )
    for flag, kind in (
        ("heads", "clinical_heads"),
        ("targets", "link_targets"),
        ("reference", "p2_reference"),
        ("separator", "separator"),
        ("before", "audit"),
        ("after", "audit"),
    ):
        if getattr(args, flag):
            path = Path(getattr(args, flag)).resolve()
            artifact(
                path,
                kind,
                resources,
                parents.get("features_sha256") if flag != "reference" else None,
            )
            parents.update(
                {flag: str(path), f"{flag}_sha256": digest(path / "artifact.json")}
            )
    if args.reference_review:
        validate_reference_review(args.reference_review, Path(args.targets))
        parents.update(
            reference_review=str(Path(args.reference_review).resolve()),
            reference_review_sha256=digest(args.reference_review),
        )
    if args.stage == "compare":
        _, ba = artifact(args.before)
        _, bb = artifact(args.after)
        if ba["parents"]["features_sha256"] != bb["parents"]["features_sha256"]:
            raise ValueError("Audit feature datasets differ")
        for key in (
            "probe_steps",
            "probe_lr",
            "attack_steps",
            "attack_lr",
            "evaluate_every",
            "batch_size",
            "seed",
        ):
            if ba["config"][key] != bb["config"][key]:
                raise ValueError(f"Unmatched audit budget/config: {key}")
    device = device_for(
        "cpu" if args.stage in ("compare", "p3-baseline") else args.device
    )
    # Parent validation precedes creation; no failed run mutates parent directories.
    output = new_run(
        resources,
        args.output,
        args.stage,
        cfg,
        parents,
        dict(
            variant=args.variant,
            view=args.view,
            workers=args.workers,
            device=str(device),
        ),
    )
    try:
        from .evaluation import fit_heads, clinical_report
        from .data import load_features
        from .models import ClinicalHeads
        from .learning import train, load_separator, represent

        if args.stage == "p3-baseline":
            from .p3 import run

            run(resources, cfg, output)
            finish(output, "p3_baseline", ["p3_baseline.json"])
        elif args.stage == "reid":
            from .reid import train as train_reid

            train_reid(resources, cfg, output, device, args.workers)
            finish(
                output,
                "p2_reference",
                [
                    "reference.pt",
                    "reference_report.json",
                    "initial_reference.json",
                    "scope.json",
                ],
            )
        elif args.stage == "targets":
            from .reid import targets

            # Verify feature/data alignment before producing a linkage reference.
            load_features(args.features, resources, roles=("fit",))
            targets(
                resources,
                cfg,
                output,
                device,
                Path(args.reference) if args.reference else None,
            )
            finish(output, "link_targets", ["targets.pt", "capability.json"])
        elif args.stage == "compare":
            from .comparison import compare

            compare(
                Path(args.before),
                Path(args.after),
                output,
                cfg["bootstrap_repetitions"],
            )
            finish(output, "comparison", ["comparison.json"])
        else:
            roles = (
                ("fit", "val_tune")
                if args.stage == "train"
                else ("fit", "val_tune", "val_select")
            )
            cache, stats = load_features(args.features, resources, roles)
            if args.stage == "probe":
                heads = fit_heads(cache, cfg, output, device)
                write(
                    output / "clinical.json",
                    clinical_report(
                        heads, cache["val_select"], device, cfg["bootstrap_repetitions"]
                    ),
                )
                finish(output, "clinical_heads", ["heads.pt", "clinical.json"])
            elif args.stage == "train":
                obj = torch.load(
                    Path(args.heads) / "heads.pt", weights_only=True, map_location="cpu"
                )
                if obj["mlp"] or obj["channels"] != 768:
                    raise ValueError(
                        "First V3 supervisor requires bound linear domain heads"
                    )
                heads = ClinicalHeads().to(device)
                heads.load_state_dict(obj["model"], strict=True)
                target = (
                    load_targets(Path(args.targets), cache) if args.targets else None
                )
                train(cache, target, heads, stats, cfg, output, device, args.variant)
                finish(
                    output,
                    "separator",
                    ["separator.pt", "training_summary.json"],
                    variant=args.variant,
                    supervision_admitted=False,
                )
            elif args.stage == "audit":
                from .evaluation import audit

                if args.separator:
                    model, _, _ = load_separator(
                        Path(args.separator) / "separator.pt", device
                    )
                    cache = represent(cache, model, args.view, device)
                audit(cache, cfg, output, device)
                finish(
                    output,
                    "audit",
                    ["linear.json", "mlp.json", "global.json", "spatial.json"],
                    view=args.view,
                    p3_evaluated=False,
                    lesions_evaluated=False,
                )
        # Bind immutable parent inputs again after execution, including explicit review.
        for key, value in parents.items():
            if key.endswith("_sha256"):
                name = key[:-7]
                path = Path(parents[name])
                file = (
                    path
                    if name == "reference_review"
                    else path
                    / ("features.json" if name == "features" else "artifact.json")
                )
                if digest(file) != value:
                    raise ValueError("Parent manifest/review changed during execution")
                if name not in ("features", "reference_review"):
                    artifact(path, resources=resources)
                elif name == "features":
                    manifest = read(file)
                    for entry in [manifest["statistics"], *manifest["roles"].values()]:
                        target = (path / entry["file"]).resolve()
                        if (
                            target.parent != path.resolve()
                            or digest(target) != entry["sha256"]
                        ):
                            raise ValueError("A0 source cache changed during execution")
        log(output, "parents_unchanged")
    except BaseException as exc:
        status(
            output,
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
        )
        raise


if __name__ == "__main__":
    main()
