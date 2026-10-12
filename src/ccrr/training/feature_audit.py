"""Independent readout acceptance for the frozen Purifier (design section 7.7)."""

from ccrr.evaluation.attacks import PooledAttacker

from collections import Counter
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from ccrr.data.pad import rng_for
from ccrr.codec.metadata import CLASSES
from ccrr.tensor_ops import pooled, masked_mean
from ccrr.representation.purifier import SemanticAttacker
from ccrr.training.optimization import optimizer, schedule, update
from ccrr.runtime.support import write_json
from ccrr.evaluation.metrics import clinical, cluster_bootstrap


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
        self.representations = (
            ("raw", "control", "sem")
            if hasattr(experiment, "no_branch_control")
            else ("raw", "sem")
        )

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
            bank = getattr(e, "b_teacher_bank", None) if role == "fit" else None
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
                                (len(ds), *value.shape[1:]), dtype=value.dtype
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

    def feature_batch(self, role, indices, representation, domains):
        source = self.store[role]
        if representation in ("raw", "control"):
            domains = [min(d, 1) for d in domains]
        return (
            source[representation][indices, domains].to(self.e.device),
            source["v"][indices].to(self.e.device),
        )

    def features(self, role, i, representation, domain):
        return self.feature_batch(role, [i], representation, [domain])

    def readout(self, name, representation, train):
        """Completed immutable baseline readouts are shared across B endpoints.

        All heads have named, isolated initialization seeds. Cache hits and misses
        consume no caller RNG; source/config/teachers/control changes reject reuse.
        Unfinished heads restart; we never reduce a readout's update budget.
        """
        from ccrr.training.b_diagnostics import isolated_rng
        from ccrr.training.b_state import binding
        from ccrr.runtime.support import seed_everything
        from ccrr.errors import require

        shared = representation in ("raw", "control")
        path = (self.root.parent / "shared_readouts" if shared else self.root) / (
            name + ".json"
        )
        bound = {
            "execution": binding(self.e.cfg),
            "protocol": "B-readout-seeds-v1",
            "control": getattr(self.e, "no_branch_control_sha256", None),
        }
        if shared and path.exists():
            saved = json.loads(path.read_text())
            require(saved["binding"] == bound, "B shared readout binding mismatch")
            print(f"B_READOUT_REUSE: {name}", flush=True)
            return saved["result"]
        with isolated_rng():
            seed_everything(
                int(rng_for(self.e.seed, "B-readout", name, 0).integers(2**31))
            )
            result = train()
        write_json(path, {"binding": bound, "result": result})
        return result

    def _classifier(self, rep, family):
        e = self.e
        head = (
            nn.Linear(768, 6)
            if family == "linear"
            else nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 6))
        ).to(e.device)
        opt = optimizer([head], [1e-3])
        best = -1
        best_predictions = None
        print(f"B_AUDIT_CLASSIFIER: {rep}/{family}", flush=True)
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
                    x = torch.stack([r[rep] for r in self.means["val_select"]]).to(
                        e.device
                    )
                    probs = [head(x[:, d]).softmax(1).cpu().numpy() for d in range(3)]
                y = np.array([r["label"] for r in self.means["val_select"]])
                reports = [clinical(y, p) for p in probs]
                score = min(r["macro_auroc"] or 0 for r in reports)
                if score > best:
                    best = score
                    best_predictions = probs
                    best_reports = reports
        return best_predictions, best_reports

    def classifiers(self):
        out = {}
        predictions = {}
        for rep in self.representations:
            for family in ("linear", "mlp"):
                best_predictions, best_reports = self.readout(
                    "clinical-" + rep + "-" + family,
                    rep,
                    lambda: self._classifier(rep, family),
                )
                best_predictions = [np.asarray(p) for p in best_predictions]
                predictions[rep, family] = best_predictions
                out[rep + "_" + family] = best_reports
        ids = [r["patient"] for r in self.means["val_select"]]
        y = np.array([r["label"] for r in self.means["val_select"]])
        checks = []
        control_checks = []

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
            if "control" in self.representations:
                ctrl = predictions["control", "linear"][d]
                control_checks.append(
                    {
                        "domain": d,
                        "delta_vs_raw": macro(ctrl, np.ones(len(y)))
                        - macro(raw, np.ones(len(y))),
                        "ci_vs_raw": cluster_bootstrap(
                            ids, lambda w: macro(ctrl, w) - macro(raw, w)
                        ),
                    }
                )
        return {
            "readouts": out,
            "checks": checks,
            "control_comparisons": control_checks,
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
                print(f"B_AUDIT_SCORE: {scenario}/{rep} {da}->{db}", flush=True)
                scores = []
                truth = []
                pairs = []
                weights = []
                endpoints = []
                for q in protocol["queries"]:
                    ai = ds.index[q["query"]]
                    for label, field in (
                        (1, "positive_gallery"),
                        (0, "negative_gallery"),
                    ):
                        names = q[field]
                        for name in names:
                            bi = ds.index[name]
                            endpoints.append((ai, bi))
                            truth.append(label)
                            pairs.append((q["patient_id"], ds.rows[bi]["patient_id"]))
                            weights.append(
                                1 / query_count[q["patient_id"]] / len(names)
                            )
                n = e.cfg.get("microbatch", 1)
                for start in range(0, len(endpoints), n):
                    group = endpoints[start : start + n]
                    a, va = self.feature_batch(
                        "val_select", [p[0] for p in group], rep, [da] * len(group)
                    )
                    b, vb = self.feature_batch(
                        "val_select", [p[1] for p in group], rep, [db] * len(group)
                    )
                    scores.extend(model(a, va, b, vb).cpu().tolist())
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

    def attacker_update(
        self, model, opt, rep, scenario, family, step, *, total=5000, rng_key=None
    ):
        e = self.e
        model.train()
        opt.zero_grad()
        rng = rng_for(
            e.seed, "feature-attack", rng_key or (scenario + rep + family), step
        )
        pairs = e.sampler.sample(rng, scenario)
        domains = [(int(rng.integers(3)), int(rng.integers(3))) for _ in pairs]
        n = e.cfg.get("microbatch", 1)
        for begin in range(0, len(pairs), n):
            group = pairs[begin : begin + n]
            view = domains[begin : begin + n]
            fa, va = self.feature_batch(
                "fit", [p[0] for p in group], rep, [d[0] for d in view]
            )
            fb, vb = self.feature_batch(
                "fit", [p[1] for p in group], rep, [d[1] for d in view]
            )
            logit = model(fa, va, fb, vb)
            (
                F.binary_cross_entropy_with_logits(
                    logit,
                    logit.new_tensor([p[2] for p in group]),
                    reduction="sum",
                )
                / 16
            ).backward()
        schedule(opt, step, total)
        update(opt)

    def _attacker(self, scenario, rep, family):
        e = self.e
        model = (SemanticAttacker() if family == "spatial" else PooledAttacker()).to(
            e.device
        )
        opt = optimizer([model], [1e-4])
        best = -1
        best_scores = None
        print(f"B_AUDIT_ATTACKER: {scenario}/{rep}/{family}", flush=True)
        for step in range(5000):
            self.attacker_update(model, opt, rep, scenario, family, step)
            if (step + 1) % 500 == 0:
                print(
                    f"B_AUDIT_ATTACKER_STEP: {scenario}/{rep}/{family} {step + 1}/5000",
                    flush=True,
                )
                scores = self.attack_scores(model, rep, scenario)
                auc = [
                    roc_auc_score(r["labels"], r["scores"], sample_weight=r["weights"])
                    for r in scores
                ]
                key = float(np.mean([max(a, 1 - a) for a in auc]))
                if key > best:
                    best = key
                    best_scores = scores
        return best_scores

    def attackers(self):
        e = self.e
        all_scores = {}
        for scenario in ("p1", "p2"):
            for rep in self.representations:
                for family in ("spatial", "pooled"):
                    best_scores = self.readout(
                        "identity-" + scenario + "-" + rep + "-" + family,
                        rep,
                        lambda: self._attacker(scenario, rep, family),
                    )
                    all_scores[scenario, rep, family] = best_scores
        checks = []
        for scenario in ("p1", "p2"):
            for d in range(9):
                print(f"B_AUDIT_BOOTSTRAP: {scenario} direction={d}", flush=True)
                reference = all_scores[scenario, "raw", "spatial"][d]
                pairs = reference["pairs"]
                y = np.asarray(reference["labels"])
                base = np.asarray(reference["weights"])
                patients = sorted({p for pair in pairs for p in pair})
                rng = rng_for(e.seed, "feature-bootstrap", scenario, d)
                arrays = {
                    (r, f): np.asarray(all_scores[scenario, r, f][d]["scores"])
                    for r in self.representations
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
                        for r in self.representations
                    }
                    original = max(aucs["raw", f] for f in ("spatial", "pooled"))
                    return (
                        strongest["raw"],
                        strongest["sem"],
                        strongest["raw"] - strongest["sem"],
                        original,
                    ) + (
                        (strongest["control"], strongest["raw"] - strongest["control"])
                        if "control" in self.representations
                        else ()
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
                low = (
                    np.percentile(boot, 2.5, axis=0) if boot else [np.nan] * len(point)
                )
                high = (
                    np.percentile(boot, 97.5, axis=0) if boot else [np.nan] * len(point)
                )
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
                if "control" in self.representations:
                    checks[-1]["control_diagnostic"] = {
                        "strongest_a_dagger": point[4],
                        "ci": [low[4], high[4]],
                        "raw_minus_control": point[5],
                        "difference_ci": [low[5], high[5]],
                        "invalid_fraction": invalid,
                    }
        return {"checks": checks, "passed": all(r["passed"] for r in checks)}

    def run(self):
        from ccrr.training.b_diagnostics import post_adapted_attack
        from ccrr.training.b_functional import check_functional
        from ccrr.errors import require

        require(
            "control" in self.representations, "Independent no-branch control required"
        )
        self.cache()
        functional = check_functional(self.e, self.e.b_teacher_bank)
        diagnostic = post_adapted_attack(self)
        c = self.classifiers()
        a = self.attackers()
        occ = np.asarray(self.occupancy)
        fraction = ((occ >= 0.1) & (occ <= 0.5)).mean(0)
        result = {
            "clinical": c,
            "identity": a,
            "occupancy_in_range": fraction,
            "l_all": float(np.mean(self.diagnostics)),
            "cache_tensor_bytes": self.cache_bytes,
            "branch_diagnostics": self.branch_diagnostics,
            "post_adapted_attack": diagnostic,
            "functional": functional,
            "passed": bool(c["passed"] and a["passed"] and (fraction >= 0.95).all()),
        }
        write_json(self.root / "acceptance.json", result)
        return result
