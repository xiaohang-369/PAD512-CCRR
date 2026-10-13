"""Resident frozen features and exact original-batch scoring reuse, diagnostic only."""

from collections import Counter
import json
import time

import torch

from ccrr.codec.metadata import CLASSES
from ccrr.errors import require
from ccrr.tensor_ops import pooled, masked_mean
from ccrr.training.feature_audit import FeatureAudit
from ccrr.representation.purifier import SemanticAttacker
from ccrr.evaluation.attacks import PooledAttacker
from b_r3_gpu_state import Timings


class GPUFeatureAudit(FeatureAudit):
    def cache(self):
        from ccrr.training.b_batches import TeacherBank
        from ccrr.training.teacher_batches import chunks
        from ccrr.errors import require

        e = self.e
        self.store = {}
        self.diagnostics = []
        self.branch_diagnostics = []
        budget = e.cfg.get("b_audit_cache_bytes", 8 * 1024**3)
        total_bytes = 0
        for role in ("fit", "val_select"):
            ds = e.data[role]
            # raw has two unique domains; source-mask gray shares raw gray.
            maps = 7 if "control" in self.representations else 5
            required = len(ds) * (maps * 768 * 196 + 196) * 4
            require(
                total_bytes + required <= budget,
                "B audit cache exceeds b_audit_cache_bytes",
            )
            bank = getattr(e, "b_teacher_banks", {}).get(role)
            if bank is None and role == "fit":
                bank = getattr(e, "b_teacher_bank", None)
            if bank is None:
                bank = TeacherBank(e, role, identity=True)
            rows, storage = [], {}
            with torch.no_grad():
                for ids in chunks(list(range(len(ds))), e.cfg.get("microbatch", 1)):
                    t = bank.get(ids)
                    v = t["v"]
                    px = e.purifier(t["f0"], v)
                    py = e.purifier(t["f1"], v)
                    cross = e.purifier(t["f1"], v, source_mask=px["hard_mask"])
                    content = {
                        "raw": torch.stack(
                            (px["standardized"], py["standardized"]), 1
                        ).cpu(),
                        "sem": torch.stack(
                            (px["semantic"], py["semantic"], cross["semantic"]), 1
                        ).cpu(),
                        "v": v.cpu(),
                    }
                    if "control" in self.representations:
                        content["control"] = torch.stack(
                            [
                                e.no_branch_control(t[key], v)["all"]
                                for key in ("f0", "f1")
                            ],
                            1,
                        ).cpu()
                    for key, value in content.items():
                        require(
                            torch.isfinite(value).all(), "Nonfinite B audit features"
                        )
                        if key not in storage:
                            storage[key] = torch.empty(
                                (len(ds), *value.shape[1:]),
                                dtype=value.dtype,
                                device=e.device,
                            )
                        storage[key][ids[0] : ids[-1] + 1].copy_(value)
                    for j, i in enumerate(ids):
                        raw = content["raw"][j][[0, 1, 1]]
                        sem = content["sem"][j]
                        vc = content["v"][j : j + 1].expand(3, -1, -1, -1)
                        rows.append(
                            {
                                "raw": pooled(raw, vc),
                                "sem": pooled(sem, vc),
                                "label": CLASSES.index(ds.rows[i]["diagnostic"]),
                                "patient": ds.rows[i]["patient_id"],
                            }
                        )
                        if "control" in content:
                            rows[-1]["control"] = pooled(
                                content["control"][j][[0, 1, 1]], vc
                            )
                        if role == "val_select":
                            from ccrr.training.b_diagnostics import branch_statistics

                            self.branch_diagnostics.append(
                                {
                                    "image": ds.rows[i].get("img_id", str(i)),
                                    "domains": [
                                        branch_statistics(
                                            {
                                                k: value[j : j + 1]
                                                for k, value in p.items()
                                            },
                                            v[j : j + 1],
                                            t[target][j : j + 1],
                                        )
                                        for p, target in ((px, "e0"), (py, "e1"))
                                    ],
                                }
                            )
                            self.occupancy.append(
                                [
                                    float(
                                        pooled(
                                            p["hard_mask"][j : j + 1], v[j : j + 1]
                                        ).mean()
                                    )
                                    for p in (px, py)
                                ]
                            )
                            self.diagnostics.append(
                                [
                                    float(
                                        masked_mean(
                                            (
                                                p["all"][j : j + 1]
                                                - p["standardized"][j : j + 1]
                                            ).square(),
                                            v[j : j + 1],
                                        )
                                    )
                                    for p in (px, py)
                                ]
                            )
            self.means[role] = rows
            self.store[role] = storage
            total_bytes += sum(v.numel() * v.element_size() for v in storage.values())
        self.cache_bytes = total_bytes
        print(f"B_AUDIT_CACHE_READY: bytes={total_bytes}", flush=True)

    def pair_table(self, scenario):
        if not hasattr(self, "pair_tables"):
            self.pair_tables = {}
        if scenario not in self.pair_tables:
            start = time.perf_counter()
            ds = self.e.data["val_select"]
            protocol = json.loads(
                (
                    ds.root
                    / "protocol_audit_20261009/pair_protocol"
                    / f"val_select_{scenario}.json"
                ).read_text()
            )
            count = Counter(q["patient_id"] for q in protocol["queries"])
            indices, labels, pairs, weights = [], [], [], []
            for q in protocol["queries"]:
                for label, field in ((1, "positive_gallery"), (0, "negative_gallery")):
                    for name in q[field]:
                        bi = ds.index[name]
                        indices.append((ds.index[q["query"]], bi))
                        labels.append(label)
                        pairs.append((q["patient_id"], ds.rows[bi]["patient_id"]))
                        weights.append(1 / count[q["patient_id"]] / len(q[field]))
            self.pair_tables[scenario] = (indices, labels, pairs, weights)
            Timings.record("score.pair_prepare", time.perf_counter() - start)
        return self.pair_tables[scenario]

    @torch.no_grad()
    def attack_scores(self, model, rep, scenario):
        require(
            isinstance(model, (SemanticAttacker, PooledAttacker)),
            "Unsupported cached scoring model",
        )
        endpoints, labels, pairs, weights = self.pair_table(scenario)
        model.eval()
        n = self.e.cfg.get("microbatch", 1)
        # Local to one scoring call: never survives an optimizer update or head switch.
        encodings = {}

        def encoded(indices, domain):
            domain = min(domain, 1) if rep in ("raw", "control") else domain
            key = (domain, tuple(indices))
            if key not in encodings:
                f, v = Timings.gpu_call(
                    "score.gather",
                    self.feature_batch,
                    "val_select",
                    indices,
                    rep,
                    [domain] * len(indices),
                )
                fn = model.encode if isinstance(model, SemanticAttacker) else pooled
                encodings[key] = Timings.gpu_call("score.encode", fn, f, v)
            return encodings[key]

        rows = []
        for da in range(3):
            for db in range(3):
                start = time.perf_counter()
                print(f"B_AUDIT_SCORE: {scenario}/{rep} {da}->{db}", flush=True)
                chunks = []
                for begin in range(0, len(endpoints), n):
                    group = endpoints[begin : begin + n]
                    a = encoded([p[0] for p in group], da)
                    b = encoded([p[1] for p in group], db)
                    if isinstance(model, SemanticAttacker):
                        chunks.append(Timings.gpu_call("score.pair", model.pair, a, b))
                    else:

                        def pair():
                            return model.net(
                                torch.cat(((a - b).abs(), a * b), 1)
                            ).squeeze(1)

                        chunks.append(Timings.gpu_call("score.pair", pair))
                scores = (
                    Timings.call("score.d2h", lambda: torch.cat(chunks).cpu().tolist())
                    if chunks
                    else []
                )
                Timings.collect_events()
                rows.append(
                    {
                        "domain": [da, db],
                        "scores": scores,
                        "labels": list(labels),
                        "pairs": list(pairs),
                        "weights": list(weights),
                    }
                )
                Timings.record("score.direction", time.perf_counter() - start)
        return rows
