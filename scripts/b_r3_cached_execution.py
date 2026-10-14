"""Unpromoted exact-arithmetic candidates for the bounded H100 A/B probe only."""

from collections import defaultdict
import copy
import time

import torch
from torch.nn import functional as F

from b_r3_gpu_features import GPUFeatureAudit
from ccrr.data.pad import PairSampler, rng_for
from ccrr.errors import require
from ccrr.evaluation.attacks import PooledAttacker
from ccrr.representation.purifier import SemanticAttacker
from ccrr.tensor_ops import pooled
from ccrr.training.optimization import schedule, update


class CachedPairSampler(PairSampler):
    """Cache metadata enumerations; preserve every RNG call and candidate order."""

    def __init__(self, source):
        self.__dict__.update(source.__dict__)
        self.images = {
            p: [i for ids in ls.values() for i in ids]
            for p, ls in self.patients.items()
        }
        self.lesions = {p: sorted(ls) for p, ls in self.patients.items()}
        self.repeated = {
            p: sorted(k for k, ids in ls.items() if len(ids) >= 2)
            for p, ls in self.patients.items()
        }
        self.others = {p: [q for q in self.all if q != p] for p in self.all}
        by_group = defaultdict(lambda: defaultdict(list))
        for p, ids in self.images.items():
            for i in ids:
                r = self.data.rows[i]
                by_group[r["diagnostic"], r["region"]][p].append(i)
        self.groups = {k: dict(v) for k, v in by_group.items()}
        self.hard_patients = {
            (p, *key): sorted(q for q in candidates if q != p)
            for key, candidates in self.groups.items()
            for p in self.all
        }

    def image(self, p, rng):
        return int(rng.choice(self.images[p]))

    def sample(self, rng, scenario="mixed"):
        require(bool(self.p1) and bool(self.p2), "Missing real P1/P2 positive pools")
        result = []
        for j in range(8):
            s = ("p1" if j < 4 else "p2") if scenario == "mixed" else scenario
            p = str(rng.choice(self.p1 if s == "p1" else self.p2))
            lesions = self.patients[p]
            if s == "p1":
                la, lb = rng.choice(self.lesions[p], 2, replace=False)
                a, b = int(rng.choice(lesions[la])), int(rng.choice(lesions[lb]))
            else:
                lesion = str(rng.choice(self.repeated[p]))
                a, b = map(int, rng.choice(lesions[lesion], 2, replace=False))
            result.append((a, b, 1, s))
        for j in range(8):
            if j < 4:
                pa, pb = rng.choice(self.all, 2, replace=False)
                a, b = self.image(pa, rng), self.image(pb, rng)
            else:
                a = result[j - 4][int(rng.integers(2))]
                row = self.data.rows[a]
                key = row["diagnostic"], row["region"]
                patients = self.hard_patients[(row["patient_id"], *key)]
                if patients:
                    p = str(rng.choice(patients))
                    b = int(rng.choice(self.groups[key][p]))
                else:
                    self.fallbacks += 1
                    p = str(rng.choice(self.others[row["patient_id"]]))
                    b = self.image(p, rng)
            result.append((a, b, 0, "negative"))
        return result


class CachedAudit(GPUFeatureAudit):
    def __init__(self, source):
        self.__dict__.update(source.__dict__)
        self.e = copy.copy(source.e)
        self.e.cfg = dict(source.e.cfg)
        self.e.sampler = CachedPairSampler(source.e.sampler)
        self.pair_tables = dict(getattr(source, "pair_tables", {}))
        self.pooled_store = {}
        start = time.perf_counter()
        with torch.no_grad():
            for role, storage in self.store.items():
                self.pooled_store[role] = {}
                for rep in ("raw", "control", "sem"):
                    x = storage[rep]
                    domains = []
                    for domain in range(x.shape[1]):
                        pieces = []
                        for first in range(0, len(x), 16):
                            ids = list(range(first, min(first + 16, len(x))))
                            # Same contiguous gather layout and FP32 reduction.
                            f, coverage = source.feature_batch(
                                role, ids, rep, [domain] * len(ids)
                            )
                            pieces.append(pooled(f, coverage))
                        domains.append(torch.cat(pieces))
                    self.pooled_store[role][rep] = torch.stack(domains, 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.cache_seconds = time.perf_counter() - start

    def pooled_batch(self, role, indices, rep, domains):
        if rep in ("raw", "control"):
            domains = [min(d, 1) for d in domains]
        return self.pooled_store[role][rep][indices, domains]

    def attacker_update(
        self, model, opt, rep, scenario, family, step, *, total=5000, rng_key=None
    ):
        if family != "pooled":
            return super().attacker_update(
                model, opt, rep, scenario, family, step, total=total, rng_key=rng_key
            )
        model.train()
        opt.zero_grad()
        rng = rng_for(
            self.e.seed, "feature-attack", rng_key or (scenario + rep + family), step
        )
        pairs = self.e.sampler.sample(rng, scenario)
        domains = [(int(rng.integers(3)), int(rng.integers(3))) for _ in pairs]
        n = self.e.cfg.get("microbatch", 1)
        for begin in range(0, len(pairs), n):
            group, view = pairs[begin : begin + n], domains[begin : begin + n]
            a = self.pooled_batch(
                "fit", [p[0] for p in group], rep, [d[0] for d in view]
            )
            b = self.pooled_batch(
                "fit", [p[1] for p in group], rep, [d[1] for d in view]
            )
            logit = model.net(torch.cat(((a - b).abs(), a * b), 1)).squeeze(1)
            (
                F.binary_cross_entropy_with_logits(
                    logit, logit.new_tensor([p[2] for p in group]), reduction="sum"
                )
                / 16
            ).backward()
        schedule(opt, step, total)
        update(opt)

    @torch.no_grad()
    def attack_scores(self, model, rep, scenario):
        require(
            isinstance(model, (SemanticAttacker, PooledAttacker)),
            "Unsupported cached scoring model",
        )
        endpoints, labels, pairs, weights = self.pair_table(scenario)
        model.eval()
        n = self.e.cfg.get("microbatch", 1)
        encodings, directions = {}, {}

        def encoded(indices, domain):
            key = domain, tuple(indices)
            if key not in encodings:
                if isinstance(model, PooledAttacker):
                    encodings[key] = self.pooled_batch(
                        "val_select", indices, rep, [domain] * len(indices)
                    )
                else:
                    f, v = self.feature_batch(
                        "val_select", indices, rep, [domain] * len(indices)
                    )
                    encodings[key] = model.encode(f, v)
            return encodings[key]

        # Queue all unique directions, retaining every original microbatch shape.
        # No per-operation timing events or per-direction device-wide barriers.
        for da in range(3 if rep == "sem" else 2):
            for db in range(3 if rep == "sem" else 2):
                parts = []
                for begin in range(0, len(endpoints), n):
                    group = endpoints[begin : begin + n]
                    a = encoded([p[0] for p in group], da)
                    b = encoded([p[1] for p in group], db)
                    parts.append(
                        model.pair(a, b)
                        if isinstance(model, SemanticAttacker)
                        else model.net(torch.cat(((a - b).abs(), a * b), 1)).squeeze(1)
                    )
                directions[da, db] = (
                    torch.cat(parts) if parts else torch.empty(0, device=self.e.device)
                )
        keys = list(directions)
        host = torch.stack([directions[key] for key in keys]).cpu().tolist()
        scores = dict(zip(keys, host))
        rows = []
        for da in range(3):
            for db in range(3):
                key = (da, db) if rep == "sem" else (min(da, 1), min(db, 1))
                rows.append(
                    {
                        "domain": [da, db],
                        "scores": list(scores[key]),
                        "labels": list(labels),
                        "pairs": list(pairs),
                        "weights": list(weights),
                    }
                )
        return rows
