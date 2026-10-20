"""Val-tune Pareto candidates, with no clinical threshold or automatic winner."""

import argparse
import csv
import html
from pathlib import Path

import numpy as np

from .runtime import artifact, digest, finish, new_run, read, write


def objectives(reports):
    """All 44 observed metrics participate; lower loss-vector components are better."""
    names, values = [], []
    for head in ("linear", "mlp"):
        for domain in ("rgb", "gray"):
            r = reports[head][domain]
            for key in ("macro_auroc", "macro_f1"):
                names.append(f"{head}.{domain}.{key}")
                values.append(-r[key])
            if len(r["recall"]) != 6:
                raise ValueError("Six disease recalls required")
            for c, value in zip(("ACK", "BCC", "MEL", "NEV", "SCC", "SEK"), r["recall"]):
                names.append(f"{head}.{domain}.recall.{c}")
                values.append(-value)
    for head in ("global", "spatial"):
        for domain in ("rgb_rgb", "gray_rgb", "gray_gray"):
            r = reports[head][domain]
            # Direction reversal must never be treated as successful de-identification.
            names.extend((f"{head}.{domain}.attack_auc", f"{head}.{domain}.rank1"))
            values.extend((max(r["auroc"], 1 - r["auroc"]), r["rank1"]))
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).all() or np.any(np.abs(v) > 1):
        raise ValueError("Undefined or out-of-range candidate metric")
    return names, v


def pareto_indices(values):
    values = np.asarray(values)
    return [
        i for i, row in enumerate(values)
        if not any(np.all(other <= row) and np.any(other < row)
                   for j, other in enumerate(values) if j != i)
    ]


def representatives(values, frontier, limit):
    """Deterministic diversity preview; retain the entire frontier separately."""
    if len(frontier) <= limit:
        return list(frontier)
    x = np.asarray(values)[frontier]
    span = np.ptp(x, axis=0)
    z = (x - x.min(0)) / np.where(span > 0, span, 1)
    selected = [0]  # Caller sorts by step; no claim that this anchor is best.
    while len(selected) < limit:
        distance = ((z[:, None] - z[selected]) ** 2).sum(2).min(1)
        distance[selected] = -1
        selected.append(int(distance.argmax()))
    return [frontier[i] for i in selected]


def summarize(entries, limit=2):
    if not entries or type(limit) is not int or limit < 1:
        raise ValueError("Nonempty audits and positive representative limit required")
    entries = sorted(entries, key=lambda r: (r["variant"], r["step"]))
    keys = [(e["variant"], e["step"]) for e in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate variant/checkpoint; do not choose favorable audit repeats")
    names = None
    values = []
    for entry in entries:
        n, v = objectives(entry["reports"])
        if names is not None and n != names:
            raise ValueError("Objective schema mismatch")
        names = n
        values.append(v)
    frontiers, preview = {}, []
    # Group-wise frontiers preserve the compression control, even if globally dominated.
    for group in sorted({e["variant"] for e in entries}):
        positions = [i for i, e in enumerate(entries) if e["variant"] == group]
        local = [values[i] for i in positions]
        front = pareto_indices(local)
        frontiers[group] = [positions[i] for i in front]
        preview.extend(positions[i] for i in representatives(local, front, limit))
    return dict(
        candidates=entries, objective_names=names,
        minimization_vectors=[v.tolist() for v in values],
        pareto_indices_by_variant=frontiers,
        joint_pareto_indices=pareto_indices(values),
        representative_indices=preview,
        winner=None, clinical_minimum=None, selection_role="val_tune",
        scientific_acceptance="not_evaluated",
        limitations=[
            "Observed 44-metric dominance, not confidence-supported dominance.",
            "High dimensionality can leave many nondominated checkpoints.",
            "Up to two diversity representatives per variant by default; no quality ranking.",
            "All frontier members are retained; no clinical minimum has been established.",
            "Both evaluator selection and this screening use val_tune; these are development results.",
            "val_select has already been examined in DEV1; it is not an untouched holdout.",
            "No test use. No automatic val_select evaluation, checkpoint adoption or generator training.",
        ],
    )


def write_tables(output, result):
    entries = result["candidates"]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "step", *result["objective_names"]])
        for e, v in zip(entries, result["minimization_vectors"]):
            writer.writerow([e["variant"], e["step"], *[-x if i < 32 else x for i, x in enumerate(v)]])
    points = []
    for e in entries:
        medical = min(e["reports"][h][d]["macro_auroc"]
                      for h in ("linear", "mlp") for d in ("rgb", "gray"))
        risk = max(max(e["reports"][h][d]["auroc"], 1 - e["reports"][h][d]["auroc"])
                   for h in ("global", "spatial") for d in ("rgb_rgb", "gray_rgb", "gray_gray"))
        points.append((risk, medical))
    # Fixed axes avoid exaggerating small changes. Lines connect steps, not a fitted frontier.
    parts = ['<!doctype html><meta charset="utf-8"><title>Checkpoint trade-offs</title>',
             '<h1>Val-tune checkpoint trade-offs</h1>',
             '<p>Lower attack AUROC and higher medical AUROC are preferable. '
             'This projection does not display F1 or recall; Pareto screening uses all 44 metrics. '
             'Lines follow training steps. No automatic winner or clinical acceptance.</p>',
             '<svg viewBox="0 0 740 540" width="740" role="img" aria-label="Medical and identity trade-offs">',
             '<path d="M60 30V470H700" fill="none" stroke="black"/>',
             '<text x="200" y="525">Worst attack AUROC (0.5 to 1.0)</text>',
             '<text x="65" y="20">Minimum medical AUROC across readers/domains (0 to 1)</text>']
    for tick in range(6):
        x = 60 + tick * 128
        parts.append(f'<text x="{x-12}" y="492">{.5+.1*tick:.1f}</text>')
    for tick in range(6):
        y = 470 - tick * 88
        parts.append(f'<text x="25" y="{y}">{.2*tick:.1f}</text>')
    for group, color in (("compression", "#1768ac"), ("separation", "#b33b26")):
        ids = [i for i, e in enumerate(entries) if e["variant"] == group]
        coords = [(60 + (points[i][0]-.5)*1280, 470-points[i][1]*440) for i in ids]
        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" opacity=".5"/>')
        for i, (x, y) in zip(ids, coords):
            label = html.escape(f"{group}, step {entries[i]['step']}, attack={points[i][0]:.6f}, medical={points[i][1]:.6f}")
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"><title>{label}</title></circle>'
                         f'<text x="{x+6}" y="{y-6}" font-size="10">{entries[i]["step"]}</text>')
    parts.append('</svg><p>Blue: compression. Red: separation. Full metrics: metrics.csv; '
                 'all candidates and Pareto membership: candidates.json.</p>')
    (output / "tradeoffs.html").write_text("\n".join(parts))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audits", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--representatives-per-variant", type=int, default=2)
    args = p.parse_args(argv)
    entries, parents = [], {}
    reference = None
    cohorts = None
    groups = {}
    for name in args.audits:
        path = Path(name).resolve()
        _, binding = artifact(path, "audit")
        options = binding["options"]
        if options.get("evaluation_role") != "val_tune" or options.get("separator_step") is None:
            raise ValueError("Only explicit val_tune checkpoint audits may enter screening")
        separator = Path(binding["parents"]["separator"])
        manifest, train_binding = artifact(separator, "separator")
        if digest(separator / "artifact.json") != binding["parents"]["separator_sha256"]:
            raise ValueError("Separator parent changed")
        checkpoint = separator / f"separator-step-{options['separator_step']:06d}.pt"
        if (
            manifest["files"].get(checkpoint.name) != binding["parents"].get("separator_checkpoint_sha256")
            or str(checkpoint.resolve()) != binding["parents"].get("separator_checkpoint")
        ):
            raise ValueError("Audit checkpoint identity differs from its claimed step")
        variant = manifest["variant"]
        if variant not in ("compression", "separation") or options["view"] != (
            "full" if variant == "compression" else "semantic"
        ):
            raise ValueError("Require independent compression/full and separation/semantic")
        signature = (binding["resources"], binding["config"],
                     binding["parents"]["features_sha256"], binding["source_commit"],
                     train_binding["config"], train_binding["source_commit"])
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("Unequal training/evaluator budgets, code or datasets")
        groups.setdefault(variant, set()).add(str(separator))
        if len(groups[variant]) != 1:
            raise ValueError("One training run per variant; no favorable-seed mixing")
        reports = {h: read(path / f"{h}.json") for h in ("linear", "mlp", "global", "spatial")}
        # Prevent accidentally pooling different cohorts or galleries.
        cohort = [
            [[reports[h][d][k] for k in ("ids", "patients", "labels")] for d in ("rgb", "gray")]
            for h in ("linear", "mlp")
        ] + [
            [[[q[k] for k in ("query", "patient", "gallery", "labels", "weight")]
              for q in reports[h][d]["queries"]]
             for d in ("rgb_rgb", "gray_rgb", "gray_gray")]
            for h in ("global", "spatial")
        ]
        if cohorts is not None and cohort != cohorts:
            raise ValueError("Candidate cohorts or query/gallery protocols differ")
        cohorts = cohort
        # Do not propagate individual predictions, queries or patient identifiers.
        compact = {h: {d: {k: v for k, v in r.items()
                          if k not in ("ids", "patients", "labels", "probabilities", "queries")}
                       for d, r in report.items()} for h, report in reports.items()}
        entries.append(dict(variant=variant, step=options["separator_step"],
                            audit=str(path), reports=compact))
        parents[str(path)] = digest(path / "artifact.json")
    if set(groups) != {"compression", "separation"}:
        raise ValueError("Both control and separation are required")
    step_sets = [{e["step"] for e in entries if e["variant"] == v}
                 for v in ("compression", "separation")]
    if step_sets[0] != step_sets[1]:
        raise ValueError("Evaluate matching checkpoint steps in both variants")
    result = summarize(entries, args.representatives_per_variant)
    output = new_run(reference[0], args.output, "checkpoint-selection", reference[1], parents,
                     dict(representatives_per_variant=args.representatives_per_variant))
    write(output / "candidates.json", result)
    write_tables(output, result)
    finish(output, "checkpoint_candidates", ["candidates.json", "metrics.csv", "tradeoffs.html"])
    print("Candidate report:", output, "No winner selected.")


if __name__ == "__main__":
    main()
