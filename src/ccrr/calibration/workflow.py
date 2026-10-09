"""Stage G: frozen-system capacity and reliability calibration workflow."""

from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from ccrr.data.pad import collate, rng_for
from ccrr.runtime.support import write_json
from ccrr.training.optimization import optimizer, schedule, update
from ccrr.errors import require, CCRRError, Status
from ccrr.calibration.probes import (
    ProbeGenerator,
    clinical_risk,
    probe_region,
    select_regions,
    labels,
    weighted_quantile,
    choose_threshold,
)
from collections import Counter


def collect_risks(experiment, roles):
    raw = {}
    weights = []
    values = []
    for role in roles:
        ds = experiment.data[role]
        patient_images = Counter([r["patient_id"] for r in ds.rows])
        for i in range(len(ds)):
            b = collate([ds[i]], experiment.device)
            with torch.no_grad():
                g = experiment.system.decompose(b["x"])[2]
            risk = clinical_risk(
                g,
                b["legal"][0].cpu().numpy(),
                b["id"][0],
                experiment.clinical,
                b["rect"],
            )
            raw[b["id"][0]] = risk
            if role == "calibration":
                legal = b["legal"][0].cpu().numpy().astype(bool)
                values.extend(risk[legal])
                weights.extend(
                    [1 / patient_images[b["patient"][0]] / int(legal.sum())]
                    * int(legal.sum())
                )
    q95 = weighted_quantile(values, weights)
    for name, r in raw.items():
        experiment.risks[name] = torch.tensor(
            np.minimum(r / max(q95, 1e-6), 1),
            device=experiment.device,
            dtype=torch.float32,
        )[None]
    return q95


def stage_g(experiment, final=False):
    experiment.system.eval()
    experiment.snapshot_index += 1
    q95 = collect_risks(experiment, ("calibration", "val_tune", "fit", "val_select"))
    generator = ProbeGenerator(
        experiment.profile, experiment.system.digest, experiment.allocator
    )
    threshold_samples = []
    for i in range(len(experiment.data["val_tune"])):
        b = collate([experiment.data["val_tune"][i]], experiment.device)
        legal = b["legal"][0].cpu().numpy()
        with torch.no_grad():
            g = experiment.system.decompose(b["x"])[2]
        for region in select_regions(b["id"][0], legal, experiment.snapshot_index):
            for k in range(5):
                rng = rng_for(
                    experiment.seed, "G-threshold", b["id"][0] + str(region), k
                )
                p = probe_region(
                    experiment.system.comm,
                    g,
                    legal,
                    b["metadata"][0],
                    b["rect"],
                    experiment.clinical,
                    generator,
                    region,
                    k,
                    32,
                    rng,
                    0.5,
                )
                threshold_samples.extend(zip(p["logits"], p["truth"], p["occupancy"]))
    threshold, accepted, details = choose_threshold(threshold_samples)
    require(accepted, "No receiver threshold passed empty-table criterion")
    from dataclasses import replace

    experiment.profile = replace(
        experiment.profile,
        occupancy_threshold=threshold["threshold"],
        q95=max(q95, 1e-6),
    )
    experiment.system.profile = experiment.profile
    training = []
    external = []
    for role in ("calibration", "val_tune"):
        ds = experiment.data[role]
        for i in range(len(ds)):
            b = collate([ds[i]], experiment.device)
            legal = b["legal"][0].cpu().numpy()
            with torch.no_grad():
                g = experiment.system.decompose(b["x"])[2]
                z = experiment.system.comm.carrier(g)
            for region in select_regions(b["id"][0], legal, experiment.snapshot_index):
                probes = {}
                for k in range(5):
                    rng = rng_for(
                        experiment.seed,
                        "G-final" if final else "G",
                        b["id"][0] + str(region),
                        k + experiment.snapshot_index * 5,
                    )
                    probes[k] = probe_region(
                        experiment.system.comm,
                        g,
                        legal,
                        b["metadata"][0],
                        b["rect"],
                        experiment.clinical,
                        generator,
                        region,
                        k,
                        256 if final else 64,
                        rng,
                        experiment.profile.occupancy_threshold,
                        deployment=experiment.policies.get(b["id"][0], (None, None))[0],
                    )
                label = labels(probes, final)
                entry = {
                    "id": b["id"][0],
                    "patient": b["patient"][0],
                    "region": region,
                    "label": label,
                    "z": z[0, region].detach().cpu(),
                    "risk": float(experiment.risks[b["id"][0]][0, region]),
                }
                (training if role == "calibration" else external).append(entry)
    head = experiment.system.comm.heads
    for p in head.parameters():
        p.requires_grad_(True)
    opt = optimizer([head], [1e-3])
    total = experiment.cfg["budgets"]["heads"]
    groups = defaultdict(lambda: defaultdict(list))
    for row in training:
        groups[row["patient"]][row["id"]].append(row)
    for step in range(total):
        rng = rng_for(
            experiment.seed,
            "G-head",
            "sample",
            step + experiment.snapshot_index * total,
        )
        p = str(rng.choice(sorted(groups)))
        name = str(rng.choice(sorted(groups[p])))
        row = groups[p][name][int(rng.integers(len(groups[p][name])))]
        z = row["z"][None, None].to(experiment.device)
        risk = z.new_tensor([[row["risk"]]])
        k = int(rng.integers(5))
        cap, rel = head(z, risk, z.new_tensor([[k]]))
        loss = F.cross_entropy(
            cap.reshape(1, 5),
            torch.tensor([row["label"]["kstar"]], device=experiment.device),
        ) + F.binary_cross_entropy(
            rel.clamp(1e-6, 1 - 1e-6),
            rel.new_tensor([[row["label"]["reliability"][k]]]),
        )
        opt.zero_grad()
        loss.backward()
        schedule(opt, step, total)
        update(opt)
    d_report = choose_conservative_budget(experiment, external)
    experiment.profile = replace(
        experiment.profile, conservative_budget=d_report["d"], reliability_sort=False
    )
    experiment.system.profile = experiment.profile
    experiment.policies = {}
    with torch.no_grad():
        for role in ("fit", "val_tune", "val_select", "calibration"):
            for i in range(len(experiment.data[role])):
                b = collate([experiment.data[role][i]], experiment.device)
                g = experiment.system.decompose(b["x"])[2]
                z = experiment.system.comm.carrier(g)
                risk = experiment.risks[b["id"][0]]
                cap, _ = head(z, risk)
                k = (cap.argmax(-1) - experiment.profile.conservative_budget).clamp_min(
                    0
                ) * b["legal"].long()
                _, rel = head(z, risk, k)
                experiment.policies[b["id"][0]] = (
                    k[0].cpu().numpy(),
                    rel[0].cpu().numpy(),
                )
    from ccrr.calibration.routing import choose_routing

    sorting = choose_routing(experiment)
    experiment.profile = replace(
        experiment.profile, reliability_sort=sorting["enabled"]
    )
    experiment.system.profile = experiment.profile
    write_json(
        experiment.output / "calibration" / f"{experiment.snapshot_index}.json",
        {
            "final": final,
            "threshold": threshold,
            "q95": q95,
            "budget": d_report,
            "sorting": sorting,
            "training": training,
            "external": external,
        },
    )
    experiment.bind_snapshot(f"G-{experiment.snapshot_index}")


def choose_conservative_budget(experiment, entries):
    from ccrr.calibration.probes import wilson

    for d in (0, 1, 2):
        by_patient = defaultdict(lambda: defaultdict(list))
        with torch.no_grad():
            for r in entries:
                z = r["z"][None, None].to(experiment.device)
                cap, _ = experiment.system.comm.heads(z, z.new_tensor([[r["risk"]]]))
                k = max(0, int(cap.argmax(-1)) - d)
                by_patient[r["patient"]][r["id"]].append(int(k > r["label"]["kstar"]))
        vals = np.array(
            [np.mean([np.mean(v) for v in ims.values()]) for ims in by_patient.values()]
        )
        pids = list(by_patient)
        rng = np.random.Generator(np.random.PCG64(20261008))
        upper = float(
            np.percentile(
                [
                    np.mean(rng.choice(vals, len(vals), replace=True))
                    for _ in range(2000)
                ],
                95,
            )
        )
        any_over = sum(any(any(v) for v in ims.values()) for ims in by_patient.values())
        if not any_over:
            upper = max(upper, wilson(0, len(pids), upper=True))
        if vals.mean() <= 0.05 and upper <= 0.10:
            return {
                "d": d,
                "overestimate_rate": float(vals.mean()),
                "bootstrap_upper_95": upper,
                "patients": len(pids),
            }
    raise CCRRError(
        Status.INPUT, "Budget head not admitted: all conservative budgets fail"
    )
