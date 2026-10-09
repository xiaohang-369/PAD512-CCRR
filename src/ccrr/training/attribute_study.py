from ccrr.evaluation import attack_scoring
from ccrr.evaluation.attacks import AttributeModel
from ccrr.training.teachers import initialize_base
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score
from ccrr.teachers.adapters import freeze_bn
from ccrr.data.pad import collate, rng_for
from ccrr.training.optimization import optimizer, schedule, update
from ccrr.runtime.support import write_json
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.errors import require


class AttributeStudy:
    def __init__(self, study):
        self.study = study
        self.e = study.e
        self.gender = sorted(
            {self.e.fit.metadata[r["img_id"]]["gender"] for r in self.e.fit.rows}
            - {"", "UNK"}
        )
        ages = [
            self.age(self.e.fit.metadata[r["img_id"]]["age"]) for r in self.e.fit.rows
        ]
        ages = [a for a in ages if a is not None]
        require(len(ages) > 0, "No fit age labels")
        self.mean = float(np.mean(ages))
        self.std = max(float(np.std(ages)), 1e-6)
        self.conflicts = set()
        for ds in self.e.data.values():
            by = defaultdict(set)
            for r in ds.rows:
                value = ds.metadata[r["img_id"]]["age"]
                if self.age(value) is not None:
                    by[r["patient_id"]].add(value)
            self.conflicts.update(p for p, a in by.items() if len(a) > 1)

    @staticmethod
    def age(value):
        try:
            v = float(value)
        except (ValueError, TypeError):
            return None
        return v if np.isfinite(v) else None

    def label(self, row, task):
        if task == "age":
            a = self.age(row["age"])
            return None if a is None else (a - self.mean) / self.std
        return (
            self.gender.index(row["gender"]) if row["gender"] in self.gender else None
        )

    def inputs(self, role, b, view):
        x = self.study.representation(role, view)(b)
        if x is None:
            return None, None
        g = self.study.public(role, b)
        with torch.no_grad():
            side = (
                attack_scoring.public_side_features(self.study, g)
                if g is not None
                else x.new_zeros((1, 8))
            )
        return x, side

    @torch.no_grad()
    def evaluate(self, model, role, task, view):
        model.eval()
        rows = []
        ds = self.e.data[role]
        for i in range(len(ds)):
            b = collate([ds[i]], self.e.device)
            label = self.label(b["metadata"][0], task)
            if label is None:
                continue
            x, side = self.inputs(role, b, view)
            if x is None:
                continue
            pred = model(x, side)
            if task == "age":
                target = label * self.std + self.mean
                value = float(pred[0, 0]) * self.std + self.mean
            else:
                target = label
                value = int(pred.argmax(1)[0])
            rows.append(
                {
                    "id": b["id"][0],
                    "patient": b["patient"][0],
                    "target": target,
                    "prediction": value,
                    "age_conflict": b["patient"][0] in self.conflicts,
                }
            )

        def score(subset):
            if not subset:
                return None
            if task == "age":
                return float(
                    np.mean([abs(r["prediction"] - r["target"]) for r in subset])
                )
            return float(
                balanced_accuracy_score(
                    [r["target"] for r in subset], [r["prediction"] for r in subset]
                )
            )

        return {
            "metric": "MAE-years" if task == "age" else "balanced_accuracy",
            "score": score(rows),
            "without_age_conflicts": score([r for r in rows if not r["age_conflict"]])
            if task == "age"
            else None,
            "rows": rows,
            "n": len(rows),
        }

    def run(self):
        e = self.e
        reports = {}
        for task in ("age", "gender"):
            if task == "gender" and len(self.gender) < 2:
                reports["gender"] = {"status": "Insufficient observed fit classes"}
                continue
            for kind in ("image", "side", "combined"):
                for view in ("public", "recovered") if kind == "image" else ("public",):
                    name = f"{task}-{kind}-{view}"
                    m = AttributeModel(
                        kind, 1 if task == "age" else len(self.gender)
                    ).to(e.device)
                    if kind != "side":
                        initialize_base(e, m.image, "resnet50")
                    opt = optimizer([m], [1e-4])
                    best = float("inf") if task == "age" else -1
                    path = self.study.root / "checkpoints" / f"attribute-{name}.pt"
                    available = [
                        i
                        for i, r in enumerate(e.fit.rows)
                        if self.label(e.fit.metadata[r["img_id"]], task) is not None
                        and (
                            self.study.folders["fit"]
                            / ("rgb" if view == "recovered" else "")
                            / r["img_id"]
                        ).exists()
                    ]
                    if not available:
                        reports[name] = {"status": "No available labeled fit samples"}
                        continue
                    for step in range(5000):
                        rng = rng_for(e.seed, "attribute", name, step)
                        opt.zero_grad()
                        m.train()
                        freeze_bn(m)
                        for i in rng.choice(available, 16):
                            b = collate([e.fit[int(i)]], e.device)
                            x, side = self.inputs("fit", b, view)
                            label = self.label(b["metadata"][0], task)
                            pred = m(x, side)
                            loss = (
                                F.mse_loss(pred.flatten(), pred.new_tensor([label]))
                                if task == "age"
                                else F.cross_entropy(
                                    pred, torch.tensor([label], device=e.device)
                                )
                            )
                            (loss / 16).backward()
                        schedule(opt, step, 5000)
                        update(opt)
                        if (step + 1) % 500 == 0:
                            report = self.evaluate(m, "val_select", task, view)
                            s = report["score"]
                            if s is not None and (
                                (s < best) if task == "age" else (s > best)
                            ):
                                best = s
                                save_checkpoint(
                                    path,
                                    {"model": m},
                                    opt,
                                    step + 1,
                                    name,
                                    {"score": s},
                                )
                    if not path.exists():
                        reports[name] = {"status": "No evaluable validation checkpoint"}
                        continue
                    load_checkpoint(path, {"model": m})
                    reports[name] = self.evaluate(m, "test", task, view)
        reports["label_profile"] = {
            "gender_mapping": self.gender,
            "fit_age_mean": self.mean,
            "fit_age_std": self.std,
            "conflict_patients": len(self.conflicts),
        }
        write_json(self.study.root / "attacks/attributes.json", reports)
        return reports
