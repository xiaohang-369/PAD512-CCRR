"""Independent readout acceptance for the frozen Purifier (design section 7.7)."""

from collections import Counter
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from .data import collate, rng_for
from .models.common import gray, pooled
from .models.purifier import SemanticAttacker
from .runtime import optimizer, schedule, update, write_json
from .metrics import clinical, cluster_bootstrap


class PooledAttacker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1536, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, a, va, b, vb):
        a = pooled(a, va)
        b = pooled(b, vb)
        return self.net(torch.cat(((a - b).abs(), a * b), 1)).squeeze(1)


class FeatureAudit:
    def __init__(self, experiment):
        self.e = experiment
        self.root = (
            experiment.output
            / "feature_audit"
            / getattr(experiment, "feature_audit_tag", "current")
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.means = {}
        self.occupancy = []

    def cache(self):
        e = self.e
        for role in ("fit", "val_select"):
            ds = e.data[role]
            rows = []
            folder = self.root / role
            folder.mkdir(exist_ok=True)
            with torch.no_grad():
                for i in range(len(ds)):
                    b = collate([ds[i]], e.device)
                    fx, v, _ = e.mfm(b["x"], b["rect"])
                    fy, _, _ = e.mfm(gray(b["x"]), b["rect"])
                    px = e.purifier(fx, v)
                    py = e.purifier(fy, v)
                    cross = e.purifier(fy, v, source_mask=px["hard_mask"])
                    raw = torch.stack(
                        (
                            px["standardized"][0],
                            py["standardized"][0],
                            py["standardized"][0],
                        )
                    )
                    sem = torch.stack(
                        (px["semantic"][0], py["semantic"][0], cross["semantic"][0])
                    )
                    content = {"raw": raw.cpu(), "sem": sem.cpu(), "v": v[0].cpu()}
                    torch.save(content, folder / f"{i}.pt")
                    rows.append(
                        {
                            "raw": pooled(raw, v.expand(3, -1, -1, -1)).cpu(),
                            "sem": pooled(sem, v.expand(3, -1, -1, -1)).cpu(),
                            "label": int(b["label"][0]),
                            "patient": b["patient"][0],
                        }
                    )
                    if role == "val_select":
                        self.occupancy.append(
                            [float(pooled(p["hard_mask"], v).mean()) for p in (px, py)]
                        )
            self.means[role] = rows

    def features(self, role, i, representation, domain):
        p = torch.load(
            self.root / role / f"{i}.pt", weights_only=True, map_location=self.e.device
        )
        return p[representation][domain : domain + 1], p["v"][None]

    def classifiers(self):
        e = self.e
        out = {}
        predictions = {}
        for rep in ("raw", "sem"):
            for family in ("linear", "mlp"):
                head = (
                    nn.Linear(768, 6)
                    if family == "linear"
                    else nn.Sequential(
                        nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 6)
                    )
                ).to(e.device)
                opt = optimizer([head], [1e-3])
                best = -1
                best_predictions = None
                for step in range(2000):
                    rng = rng_for(e.seed, "feature-clinical", rep + family, step)
                    ids = rng.integers(len(e.fit), size=16)
                    x = (
                        torch.stack([self.means["fit"][i][rep] for i in ids])
                        .to(e.device)
                        .flatten(0, 1)
                    )
                    y = torch.tensor(
                        [self.means["fit"][i]["label"] for i in ids], device=e.device
                    ).repeat_interleave(3)
                    opt.zero_grad()
                    F.cross_entropy(head(x), y, weight=e.weights).backward()
                    schedule(opt, step, 2000)
                    update(opt)
                    if (step + 1) % 500 == 0:
                        with torch.no_grad():
                            x = torch.stack(
                                [r[rep] for r in self.means["val_select"]]
                            ).to(e.device)
                            probs = [
                                head(x[:, d]).softmax(1).cpu().numpy() for d in range(3)
                            ]
                        y = np.array([r["label"] for r in self.means["val_select"]])
                        reports = [clinical(y, p) for p in probs]
                        score = min(r["macro_auroc"] or 0 for r in reports)
                        if score > best:
                            best = score
                            best_predictions = probs
                            best_reports = reports
                predictions[rep, family] = best_predictions
                out[rep + "_" + family] = best_reports
        ids = [r["patient"] for r in self.means["val_select"]]
        y = np.array([r["label"] for r in self.means["val_select"]])
        checks = []

        def macro(p, w):
            if any(len(np.unique((y == c)[w > 0])) < 2 for c in range(6)):
                return np.nan
            return np.mean(
                [roc_auc_score(y == c, p[:, c], sample_weight=w) for c in range(6)]
            )

        for d in range(3):
            raw = predictions["raw", "linear"][d]
            sem = predictions["sem", "linear"][d]
            delta = macro(sem, np.ones(len(y))) - macro(raw, np.ones(len(y)))
            ci = cluster_bootstrap(ids, lambda w: macro(sem, w) - macro(raw, w))
            checks.append(
                {
                    "domain": d,
                    "delta": delta,
                    "ci": ci,
                    "passed": out["sem_linear"][d]["macro_auroc"] >= 0.75
                    and delta >= -0.03
                    and ci["evidence_sufficient"]
                    and ci["lower"] is not None
                    and ci["lower"] >= -0.03,
                }
            )
        return {
            "readouts": out,
            "checks": checks,
            "passed": all(c["passed"] for c in checks),
        }

    @torch.no_grad()
    def attack_scores(self, model, rep, scenario):
        e = self.e
        ds = e.data["val_select"]
        protocol = json.loads(
            (
                ds.root
                / "protocol_audit_20261009/pair_protocol"
                / f"val_select_{scenario}.json"
            ).read_text()
        )
        rows = []
        query_count = Counter(q["patient_id"] for q in protocol["queries"])
        model.eval()
        for da in range(3):
            for db in range(3):
                scores = []
                truth = []
                pairs = []
                weights = []
                for q in protocol["queries"]:
                    ai = ds.index[q["query"]]
                    a, va = self.features("val_select", ai, rep, da)
                    for label, field in (
                        (1, "positive_gallery"),
                        (0, "negative_gallery"),
                    ):
                        names = q[field]
                        for name in names:
                            bi = ds.index[name]
                            b, vb = self.features("val_select", bi, rep, db)
                            scores.append(float(model(a, va, b, vb)))
                            truth.append(label)
                            pairs.append((q["patient_id"], ds.rows[bi]["patient_id"]))
                            weights.append(
                                1 / query_count[q["patient_id"]] / len(names)
                            )
                rows.append(
                    {
                        "domain": [da, db],
                        "scores": scores,
                        "labels": truth,
                        "pairs": pairs,
                        "weights": weights,
                    }
                )
        return rows

    def attackers(self):
        e = self.e
        all_scores = {}
        for scenario in ("p1", "p2"):
            for rep in ("raw", "sem"):
                for family in ("spatial", "pooled"):
                    model = (
                        SemanticAttacker() if family == "spatial" else PooledAttacker()
                    ).to(e.device)
                    opt = optimizer([model], [1e-4])
                    best = -1
                    best_scores = None
                    for step in range(5000):
                        model.train()
                        opt.zero_grad()
                        rng = rng_for(
                            e.seed, "feature-attack", scenario + rep + family, step
                        )
                        for a, b, s, _ in e.sampler.sample(rng, scenario):
                            fa, va = self.features("fit", a, rep, int(rng.integers(3)))
                            fb, vb = self.features("fit", b, rep, int(rng.integers(3)))
                            logit = model(fa, va, fb, vb)
                            (
                                F.binary_cross_entropy_with_logits(
                                    logit, logit.new_tensor([s])
                                )
                                / 16
                            ).backward()
                        schedule(opt, step, 5000)
                        update(opt)
                        if (step + 1) % 500 == 0:
                            scores = self.attack_scores(model, rep, scenario)
                            auc = [
                                roc_auc_score(
                                    r["labels"], r["scores"], sample_weight=r["weights"]
                                )
                                for r in scores
                            ]
                            key = float(np.mean([max(a, 1 - a) for a in auc]))
                            if key > best:
                                best = key
                                best_scores = scores
                    all_scores[scenario, rep, family] = best_scores
        checks = []
        for scenario in ("p1", "p2"):
            for d in range(9):
                reference = all_scores[scenario, "raw", "spatial"][d]
                pairs = reference["pairs"]
                y = np.asarray(reference["labels"])
                base = np.asarray(reference["weights"])
                patients = sorted({p for pair in pairs for p in pair})
                rng = rng_for(e.seed, "feature-bootstrap", scenario, d)
                arrays = {
                    (r, f): np.asarray(all_scores[scenario, r, f][d]["scores"])
                    for r in ("raw", "sem")
                    for f in ("spatial", "pooled")
                }

                def statistic(w):
                    aucs = {
                        (r, f): roc_auc_score(y, s, sample_weight=w * base)
                        for (r, f), s in arrays.items()
                    }
                    strongest = {
                        r: max(
                            max(aucs[r, f], 1 - aucs[r, f])
                            for f in ("spatial", "pooled")
                        )
                        for r in ("raw", "sem")
                    }
                    original = max(aucs["raw", f] for f in ("spatial", "pooled"))
                    return (
                        strongest["raw"],
                        strongest["sem"],
                        strongest["raw"] - strongest["sem"],
                        original,
                    )

                point = statistic(np.ones(len(y)))
                boot = []
                for _ in range(2000):
                    c = Counter(rng.choice(patients, len(patients), replace=True))
                    w = np.array([c[a] if a == b else c[a] * c[b] for a, b in pairs])
                    try:
                        v = statistic(w)
                    except ValueError:
                        continue
                    if np.isfinite(v).all():
                        boot.append(v)
                low = np.percentile(boot, 2.5, axis=0) if boot else [np.nan] * 4
                invalid = 1 - len(boot) / 2000
                effective = point[0] >= 0.6 and low[3] > 0.5 and invalid <= 0.05
                reduction = point[1] - 0.5 <= 0.75 * (point[0] - 0.5)
                checks.append(
                    {
                        "scenario": scenario,
                        "domain": reference["domain"],
                        "raw_strongest": point[0],
                        "sem_strongest": point[1],
                        "difference_lower": low[2],
                        "invalid_fraction": invalid,
                        "passed": bool(effective and reduction and low[2] > 0),
                    }
                )
        return {"checks": checks, "passed": all(r["passed"] for r in checks)}

    def run(self):
        self.cache()
        c = self.classifiers()
        a = self.attackers()
        occ = np.asarray(self.occupancy)
        fraction = ((occ >= 0.1) & (occ <= 0.5)).mean(0)
        result = {
            "clinical": c,
            "identity": a,
            "occupancy_in_range": fraction,
            "passed": bool(c["passed"] and a["passed"] and (fraction >= 0.95).all()),
        }
        write_json(self.root / "acceptance.json", result)
        return result
