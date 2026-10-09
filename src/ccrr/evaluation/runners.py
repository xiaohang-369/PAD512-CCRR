from pathlib import Path
from collections import Counter, defaultdict
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from ccrr.data.pad import collate
from ccrr.tensor_ops import gray
from ccrr.evaluation.metrics import (
    clinical,
    cluster_bootstrap,
    verification,
    retrieval,
    pair_bootstrap,
    quality,
    failure_funnel,
    raw_ber,
)
from ccrr.runtime.support import write_json
from ccrr.runtime.system import save_public
from ccrr.errors import CCRRError, Status, require


@torch.no_grad()
def evaluate_classifier(model, data, device, representation=None, bootstrap=False):
    model.eval()
    ys = []
    ps = []
    ids = []
    patients = []
    for i in range(len(data)):
        b = collate([data[i]], device)
        x = b["x"] if representation is None else representation(b)
        if x is None:
            continue
        ps.append(model(x, b["rect"]).softmax(1)[0].cpu().numpy())
        ys.append(int(b["label"][0]))
        ids.append(b["id"][0])
        patients.append(b["patient"][0])
    require(len(ys) > 0, "No published classification images")
    result = clinical(ys, ps)
    y = np.asarray(ys)
    p = np.asarray(ps)
    if bootstrap:

        def stat(w):
            if any(len(np.unique((y == c)[w > 0])) < 2 for c in range(6)):
                return np.nan
            return np.mean(
                [roc_auc_score(y == c, p[:, c], sample_weight=w) for c in range(6)]
            )

        result["macro_auroc_ci"] = cluster_bootstrap(patients, stat)
    result.update(
        ids=ids, patients=patients, labels=ys, probabilities=np.asarray(ps).tolist()
    )
    return result


@torch.no_grad()
def evaluate_identity(
    model,
    data,
    device,
    scenario="p1",
    query_view=None,
    gallery_view=None,
    bootstrap=False,
    allowed_queries=None,
    allowed_gallery=None,
    thresholds=None,
):
    model.eval()
    features = {}
    gallery_features = {}
    by_id = {r["img_id"]: r for r in data.rows}
    protocol = json.loads(
        (
            data.root
            / "protocol_audit_20261009"
            / "pair_protocol"
            / f"{data.role}_{scenario}.json"
        ).read_text()
    )
    needed = set(protocol["gallery"]) | {r["query"] for r in protocol["queries"]}
    for image_id in sorted(needed):
        b = collate([data[data.index[image_id]]], device)
        for dest, view in ((features, query_view), (gallery_features, gallery_view)):
            x = b["x"] if view is None else view(b)
            if (
                x is not None
                and (
                    dest is not features
                    or allowed_queries is None
                    or image_id in allowed_queries
                )
                and (
                    dest is not gallery_features
                    or allowed_gallery is None
                    or image_id in allowed_gallery
                )
            ):
                f = model(x)  # attacks use the public full canvas, no sender V
                dest[image_id] = {k: f[k].detach() for k in ("embedding", "h")}
    patient_query_count = Counter(
        q["patient_id"] for q in protocol["queries"] if q["query"] in features
    )
    outputs = {}
    for mode in ("cosine", "pair"):
        scores = []
        truth = []
        weights = []
        pairs = []
        ranks = []
        dropped = []
        for q in protocol["queries"]:
            name = q["query"]
            pid = q["patient_id"]
            if name not in features:
                dropped.append(name)
                continue
            pos = [n for n in q["positive_gallery"] if n in gallery_features]
            neg = [n for n in q["negative_gallery"] if n in gallery_features]
            if not pos or not neg:
                dropped.append(name)
                continue
            gids = pos + neg
            f = features[name]
            if mode == "cosine":
                s = [
                    float((f["embedding"] * gallery_features[n]["embedding"]).sum())
                    for n in gids
                ]
            else:
                s = [float(model.pair(f["h"], gallery_features[n]["h"])) for n in gids]
            ret = retrieval(s, gids, pos)
            ret.update(patient=pid, query=name)
            ranks.append(ret)
            # Patients, then query observations, then each class are equally weighted.
            for j, n in enumerate(gids):
                label = int(j < len(pos))
                truth.append(label)
                scores.append(s[j])
                pairs.append((pid, by_id[n]["patient_id"]))
                weights.append(
                    1 / patient_query_count[pid] / (len(pos) if label else len(neg))
                )
        if not scores:
            outputs[mode] = {"valid": False, "dropped_queries": dropped}
            continue
        base = np.asarray(weights)
        y = np.asarray(truth)
        s = np.asarray(scores)
        report = verification(
            y, s, base, threshold=None if thresholds is None else thresholds.get(mode)
        )
        report["score_details"] = {
            "labels": y.tolist(),
            "scores": s.tolist(),
            "weights": base.tolist(),
            "pairs": pairs,
        }
        report["retrieval_rows"] = ranks
        grouped = defaultdict(list)
        for r in ranks:
            grouped[r["patient"]].append(r)
        report.update(
            rank1=float(
                np.mean([np.mean([r["rank1"] for r in v]) for v in grouped.values()])
            ),
            map=float(
                np.mean([np.mean([r["ap"] for r in v]) for v in grouped.values()])
            ),
            chance_rank1=float(
                np.mean(
                    [np.mean([r["chance_rank1"] for r in v]) for v in grouped.values()]
                )
            ),
            queries=len(ranks),
            patients=len(grouped),
            dropped_queries=dropped,
            valid=True,
        )
        if bootstrap:
            report["auroc_ci"] = pair_bootstrap(
                pairs, lambda w: roc_auc_score(y, s, sample_weight=w * base)
            )
        outputs[mode] = report
    return outputs


@torch.no_grad()
def evaluate_system(
    system,
    data,
    device,
    allocator,
    outdir,
    risk_provider,
    policy_provider=None,
    level_order=None,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    system.eval()
    rows = []
    for i in range(len(data)):
        b = collate([data[i]], device)
        name = b["id"][0]
        row = {
            "id": name,
            "patient": b["patient"][0],
            "diagnostic": b["metadata"][0]["diagnostic"],
            "published": False,
            "c_success": False,
            "m_success": False,
            "both_success": False,
        }
        credentials = allocator.new_object(system.digest)
        try:
            risk = risk_provider(b)
            policy = policy_provider(b) if policy_provider else None
            sent = system.send(
                b["x"],
                b["metadata"][0],
                b["legal"],
                risk,
                credentials,
                allocator,
                policy,
                level_order,
            )
            # Keys are written only in a private output subdirectory, never PNG or reports.
            private = outdir / "credentials"
            private.mkdir(mode=0o700, exist_ok=True)
            for a, c in enumerate(credentials):
                c.save(private / f"{name}.{a}.key")
            g = save_public(sent["public"], outdir / name).to(device)
            row["published"] = True
            result = system.receive(g, {0: credentials[0], 1: credentials[1]})
            c, m = result["C"], result["M"]
            row.update(
                c_status=str(c.status),
                m_status=str(m.status),
                level=sent["level"],
                protected_bpp=8
                * (len(sent["c"]) + len(sent["m"]))
                / system.profile.size**2,
                c_bytes=len(sent["c"]),
                m_bytes=len(sent["m"]),
                source_bytes=len(sent["c"]) - 88,
                ber=raw_ber(system.comm.read(g), sent["bits"], sent["occupancy"]),
            )
            if c.status == Status.SUCCESS:
                require(
                    torch.equal(c.value["integers"].float(), sent["n"]),
                    "Authenticated latent differs from source",
                )
                row["c_success"] = True
                row["quality"] = quality(b["x"], c.value["rgb_unclipped"], b["v"])[0]
                oracle = system.recover(g, sent["zq"])
                require(
                    torch.allclose(
                        oracle, c.value["rgb_unclipped"], atol=1e-6, rtol=1e-6
                    ),
                    "T3/T4 mismatch",
                )
                rgbdir = outdir / "rgb"
                rgbdir.mkdir(exist_ok=True)
                np.save(
                    rgbdir / (name + ".npy"),
                    c.value["rgb"][0].cpu().numpy(),
                    allow_pickle=False,
                )
                # These are evaluation artifacts, not part of the public carrier.
                from PIL import Image

                Image.fromarray(
                    torch.floor(c.value["rgb"][0].permute(1, 2, 0).cpu() * 255 + 0.5)
                    .byte()
                    .numpy()
                ).save(rgbdir / name)
            if m.status == Status.SUCCESS:
                require(
                    m.value == b["metadata"][0],
                    "Authenticated metadata differs from source",
                )
                row["m_success"] = True
            row["both_success"] = row["c_success"] and row["m_success"]
            row["public_quality"] = quality(gray(b["x"]), g, b["v"])[0]
        except CCRRError as exc:
            row.update(failure=str(exc.status), detail=exc.detail)
        rows.append(row)
        write_json(
            outdir / "progress.json", {"processed": len(rows), "total": len(data)}
        )
    result = {
        "rows": rows,
        "funnel": failure_funnel(rows),
        "profile": system.digest.hex(),
        "role": data.role,
    }
    write_json(outdir / "system_results.json", result)
    return result
