"""Independent R3 readers: existing math and acceptance, resumable head updates."""

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from ccrr.data.pad import rng_for
from ccrr.training.feature_audit import FeatureAudit
from ccrr.training.optimization import optimizer, schedule, update
from ccrr.training.b_diagnostics import isolated_rng, compare_attack_scores
from ccrr.runtime.support import seed_everything, write_json
from ccrr.representation.purifier import SemanticAttacker
from ccrr.evaluation.attacks import PooledAttacker
from ccrr.evaluation.metrics import clinical
from b_r3_state import State, run_job


class Audit(FeatureAudit):
    def __init__(self, e, bound, budget):
        super().__init__(e)
        self.bound, self.budget = bound, budget
        self.heads = self.root / "heads"
        self.heads.mkdir(parents=True, exist_ok=True)

    def readout(self, name, representation, train):
        self.budget.check()
        with isolated_rng():
            seed_everything(
                int(rng_for(self.e.seed, "B-readout", name, 0).integers(2**31))
            )
            return train()

    def job(self, name, model, opt, total, train, score):
        state = State(
            self.heads / (name + ".pt"),
            {**self.bound, "head": name},
            {"head": model},
            {"head": opt},
            sampler=self.e.sampler,
        )
        result = run_job(state, total, train, score, 500, self.budget)
        print(f"R3_HEAD_COMPLETE: {name} updates={total}", flush=True)
        return result

    def _classifier(self, rep, family):
        e = self.e
        head = classifier(family).to(e.device)
        opt = optimizer([head], [1e-3])

        def train(step):
            ids = rng_for(e.seed, "feature-clinical", rep + family, step).integers(
                len(e.fit), size=16
            )
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

        def score():
            with torch.no_grad():
                x = torch.stack([r[rep] for r in self.means["val_select"]]).to(e.device)
                probs = [head(x[:, d]).softmax(1).cpu().numpy() for d in range(3)]
            y = np.array([r["label"] for r in self.means["val_select"]])
            reports = [clinical(y, p) for p in probs]
            return min(r["macro_auroc"] or 0 for r in reports), (
                [p.tolist() for p in probs],
                reports,
            )

        return self.job("clinical-" + rep + "-" + family, head, opt, 2000, train, score)

    def _attacker(self, scenario, rep, family):
        model = (SemanticAttacker() if family == "spatial" else PooledAttacker()).to(
            self.e.device
        )
        opt = optimizer([model], [1e-4])

        def score():
            scores = self.attack_scores(model, rep, scenario)
            auc = [
                roc_auc_score(r["labels"], r["scores"], sample_weight=r["weights"])
                for r in scores
            ]
            return float(np.mean([max(a, 1 - a) for a in auc])), scores

        return self.job(
            "identity-" + scenario + "-" + rep + "-" + family,
            model,
            opt,
            5000,
            lambda step: self.attacker_update(model, opt, rep, scenario, family, step),
            score,
        )

    def attack_scores(self, model, rep, scenario):
        self.budget.check()
        result = super().attack_scores(model, rep, scenario)
        self.budget.record()
        return result

    def adapted(self):
        # Isolate both RNG and sampler accounting from all other audits.
        with isolated_rng():
            audit = copy.copy(self)
            audit.e = copy.copy(self.e)
            audit.e.sampler = copy.copy(self.e.sampler)
            model = copy.deepcopy(self.e.sem_attacker).requires_grad_(True).eval()
            opt = optimizer([model], [1e-4])
            state = State(
                self.heads / "post-adapt.pt",
                {**self.bound, "head": "post-adapt"},
                {"head": model},
                {"head": opt},
                sampler=audit.e.sampler,
            )
            if not state.loaded:
                state.data = {
                    "next": 0,
                    "before": {
                        s: audit.attack_scores(model, "sem", s) for s in ("p1", "p2")
                    },
                }
                state.save()
            while state.data["next"] < 1000:
                self.budget.check()
                step = state.data["next"]
                audit.attacker_update(
                    model,
                    opt,
                    "sem",
                    "mixed",
                    "spatial",
                    step,
                    total=1000,
                    rng_key="B-post-adapt-mixed",
                )
                state.data["next"] += 1
                state.save()
                self.budget.record()
                if state.data["next"] % 100 == 0:
                    print(f"R3_ADAPT: {state.data['next']}/1000", flush=True)
            if "after" not in state.data:
                state.data["after"] = {
                    s: audit.attack_scores(model, "sem", s) for s in ("p1", "p2")
                }
                state.save()
            checks = {
                s: compare_attack_scores(
                    state.data["before"][s], state.data["after"][s]
                )
                for s in ("p1", "p2")
            }
            result = {
                "updates": 1000,
                "generator_updated": False,
                "fit_only": True,
                "checks": checks,
                "alternating_attack_insufficient": any(
                    r["alternating_attack_insufficient"]
                    for v in checks.values()
                    for r in v
                ),
                "replaces_independent_attack_acceptance": False,
            }
            write_json(self.root / "post_adapted_attack.json", result)
            return result


def classifier(family):
    return (
        nn.Linear(768, 6)
        if family == "linear"
        else nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 6))
    )
