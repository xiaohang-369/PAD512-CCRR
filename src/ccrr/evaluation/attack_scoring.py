"""Frozen attack scoring, source matching and public-only side features."""

from collections import defaultdict
import hashlib
import numpy as np
import torch
from ccrr.data.pad import collate
from ccrr.evaluation.attacks import (
    ncc,
    metadata_tensor,
)
from ccrr.tensor_ops import gray
from ccrr.tensor_ops import ssim_values
from ccrr.evaluation.metrics import quality
from ccrr.communication.records import logits_to_records, reassemble
from ccrr.codec.metadata import FIELDS


@torch.no_grad()
def source_matching(study, model, role, representation="public", structure=None):
    e = study.e
    ds = e.data[role]
    gallery = []
    queries = []
    for i in range(len(ds)):
        b = collate([ds[i]], e.device)
        gallery.append(
            (
                b["id"][0],
                b["patient"][0],
                gray(b["x"]) if structure else model(b["x"]),
            )
        )
        q = study.representation(role, representation)(b)
        if q is not None:
            if structure and q.shape[1] == 3:
                q = gray(q)
            queries.append((b["id"][0], b["patient"][0], q if structure else model(q)))
    outputs = {}
    for mode in (structure,) if structure else ("cosine", "pair"):
        rows = []
        for name, pid, q in queries:
            scores = []
            for gname, gpid, g in gallery:
                if mode == "ncc":
                    score = float(ncc(q, g))
                elif mode == "ssim":
                    score = float(ssim_values(q, g, torch.ones_like(q))[0])
                elif mode == "cosine":
                    score = float((q["embedding"] * g["embedding"]).sum())
                else:
                    score = float(model.pair(q["h"], g["h"]))
                scores.append(score)
            order = sorted(
                range(len(gallery)),
                key=lambda j: (
                    -scores[j],
                    hashlib.sha256(gallery[j][0].encode()).digest(),
                ),
            )
            rows.append(
                {
                    "id": name,
                    "patient": pid,
                    "source_rank1": float(gallery[order[0]][0] == name),
                    "patient_rank1": float(gallery[order[0]][1] == pid),
                    "source_rank5": float(
                        any(gallery[j][0] == name for j in order[:5])
                    ),
                    "source_chance": 1 / len(gallery),
                    "patient_chance": sum(gp == pid for _, gp, _ in gallery)
                    / len(gallery),
                }
            )
        groups = defaultdict(list)
        for row in rows:
            groups[row["patient"]].append(row)
        summary = (
            {
                key: float(
                    np.mean(
                        [np.mean([r[key] for r in group]) for group in groups.values()]
                    )
                )
                for key in (
                    "source_rank1",
                    "patient_rank1",
                    "source_rank5",
                    "source_chance",
                    "patient_chance",
                )
            }
            if groups
            else {}
        )
        summary.update(rows=rows, gallery_size=len(gallery), query_size=len(rows))
        outputs[mode] = summary
    return outputs


@torch.no_grad()
def reconstruction_scores(study, model, role, latent=False, with_metadata=False):
    e = study.e
    model.eval()
    rows = []
    for i in range(len(e.data[role])):
        b = collate([e.data[role][i]], e.device)
        g = study.public(role, b)
        if g is None:
            continue
        if (
            with_metadata
            and not next(
                r for r in study.results[role]["rows"] if r["id"] == b["id"][0]
            )["m_success"]
        ):
            continue
        mb = metadata_tensor(b["metadata"], FIELDS, e.device) if with_metadata else None
        pred = model(g, mb)
        rec = study.system.recover(g, pred) if latent else pred
        rows.append(
            {
                "id": b["id"][0],
                "patient": b["patient"][0],
                "quality": quality(b["x"], rec, b["v"])[0],
            }
        )
    return rows


@torch.no_grad()
def public_side_features(study, g):
    records, errors = logits_to_records(
        study.system.comm.read(g)[0], study.e.profile.occupancy_threshold
    )
    result = g.new_zeros((1, 8))
    for alpha in (0, 1):
        try:
            if errors[alpha]:
                continue
            stream, r = reassemble(records, alpha)
            if len(stream) < 88 or stream[:4] != b"MCR1":
                continue
            result[0, 6 + alpha] = 1
            if alpha == 0:
                result[0, 0] = len(stream) * 8 / 131072
                result[0, 1] = len([r for r in records if r.alpha == 0]) / 1024
                result[0, 2 + r] = 1
        except Exception:
            continue
    return result
