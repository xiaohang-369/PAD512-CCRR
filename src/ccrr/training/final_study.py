"""Independent evaluation on a frozen system, including adaptive source attacks."""

from ccrr.codec.quantization import quantize_latent, dequantize_latent


from ccrr.evaluation import attack_scoring
from ccrr.training.teachers import initialize_base


from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from ccrr.data.pad import collate, rng_for
from ccrr.teachers.adapters import ClinicalModel, IdentityModel, freeze_bn, freeze
from ccrr.evaluation.attacks import (
    ReconstructionAttack,
    constant_latent_attack,
    metadata_tensor,
)
from ccrr.tensor_ops import gray
from ccrr.training.losses import reconstruction, contrastive
from ccrr.training.optimization import optimizer, schedule, update
from ccrr.runtime.support import write_json
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.evaluation.runners import (
    evaluate_classifier,
    evaluate_identity,
    evaluate_system,
)
from ccrr.evaluation.metrics import quality
from ccrr.runtime.system import load_public
from ccrr.errors import require
from ccrr.calibration.probes import clinical_risk
from ccrr.codec.metadata import FIELDS


class FinalStudy:
    def __init__(self, experiment):
        self.e = experiment
        self.root = experiment.output / "final"
        self.root.mkdir(exist_ok=True)
        self.system = freeze(experiment.system)
        self.folders = {}
        self.results = {}

    def public(self, role, b):
        p = self.folders[role] / b["id"][0]
        return (
            load_public(p, self.e.profile.size).to(self.e.device)
            if p.exists()
            else None
        )

    def representation(self, role, name):
        def view(b):
            if name == "rgb":
                return b["x"]
            if name == "gray":
                return gray(b["x"])
            if name == "g0":
                with torch.no_grad():
                    return self.system.decompose(b["x"])[2]
            if name == "public":
                return self.public(role, b)
            if name == "recovered":
                p = self.folders[role] / "rgb" / (b["id"][0] + ".npy")
                if not p.exists():
                    return None
                return torch.from_numpy(np.load(p, allow_pickle=False))[None].to(
                    self.e.device
                )
            raise ValueError(name)

        return view

    def publish(self):
        e = self.e
        for role in ("fit", "val_tune", "val_select", "test"):
            folder = self.root / "representations" / role
            self.folders[role] = folder

            def risk(b):
                with torch.no_grad():
                    g = self.system.decompose(b["x"])[2]
                r = clinical_risk(
                    g, b["legal"][0].cpu().numpy(), b["id"][0], e.clinical, b["rect"]
                )
                return torch.tensor(
                    np.minimum(r / e.profile.q95, 1),
                    device=e.device,
                    dtype=torch.float32,
                )[None]

            self.results[role] = evaluate_system(
                self.system, e.data[role], e.device, e.allocator, folder, risk
            )

    def classifiers(self):
        e = self.e
        out = {}
        # The final classifiers never feed their results back into the generator.
        for family in ("convnext_tiny", "swin_t"):
            for rep in ("rgb", "gray", "g0", "public", "recovered"):
                model = ClinicalModel(family).to(e.device)
                # Dataset role is explicit; a fit representation cannot read test outputs.
                score, val = self.train_classifier(model, f"{family}-{rep}", rep)
                test = evaluate_classifier(
                    model,
                    e.data["test"],
                    e.device,
                    self.representation("test", rep),
                    True,
                )
                out[family, rep] = {
                    "val": val,
                    "test": test,
                    "baseline_valid": score >= 0.75 if rep == "rgb" else None,
                }
                if rep == "rgb":
                    for target in ("gray", "g0", "public", "recovered"):
                        out[family, "rgb-zero-adaptation-" + target] = (
                            evaluate_classifier(
                                model,
                                e.data["test"],
                                e.device,
                                self.representation("test", target),
                                True,
                            )
                        )
        write_json(self.root / "clinical.json", out)
        return out

    def train_classifier(self, model, name, rep):
        e = self.e
        initialize_base(e, model, model.architecture)
        head = (
            model.model.head
            if model.architecture == "swin_t"
            else model.model.classifier
        )
        headids = {id(p) for p in head.parameters()}
        opt = optimizer([model], [1e-4])
        groups = []
        for group in opt.param_groups:
            for ish, lr in ((False, 1e-4), (True, 1e-3)):
                ps = [p for p in group["params"] if (id(p) in headids) == ish]
                if ps:
                    groups.append(
                        {
                            "params": ps,
                            "lr": lr,
                            "lr_max": lr,
                            "weight_decay": group["weight_decay"],
                            "head": ish,
                        }
                    )
        opt = torch.optim.AdamW(groups)
        total = e.cfg["budgets"]["evaluator"]
        best = -1
        report = None
        path = self.root / "checkpoints" / f"{name}.pt"
        # Failed publications remain in the system funnel; task models train on their available cohort.
        available = [
            i
            for i, r in enumerate(e.fit.rows)
            if rep not in ("public", "recovered")
            or (
                self.folders["fit"]
                / ("rgb" if rep == "recovered" else "")
                / r["img_id"]
            ).exists()
        ]
        require(available, f"No fit images for {rep}")
        for step in range(total):
            rng = rng_for(e.seed, name, "fit", step)
            model.train()
            freeze_bn(model)
            opt.zero_grad()
            for p in model.parameters():
                p.requires_grad_(id(p) in headids or step >= 500)
            selected = rng.choice(available, 16)
            denom = sum(float(e.weights[e.fit[int(i)]["label"]]) for i in selected)
            for i in selected:
                b = collate([e.fit[int(i)]], e.device)
                x = self.representation("fit", rep)(b)
                if rng.random() < 0.5:
                    x = x.flip(-1)
                    b["rect"] = [
                        (x.shape[-1] - l - w, t, w, h) for l, t, w, h in b["rect"]
                    ]
                (
                    F.cross_entropy(
                        model(x, b["rect"]),
                        b["label"],
                        weight=e.weights,
                        reduction="sum",
                    )
                    / denom
                ).backward()
            schedule(opt, step, total)
            for group in opt.param_groups:
                if not group["head"] and step < 750:
                    group["lr"] = (
                        0
                        if step < 500
                        else group["lr_max"] * (0.01 + 0.99 * (step - 500) / 249)
                    )
            update(opt)
            if (step + 1) % 500 == 0 or step + 1 == total:
                report = evaluate_classifier(
                    model,
                    e.data["val_select"],
                    e.device,
                    self.representation("val_select", rep),
                    True,
                )
                score = report["macro_auroc"] or 0
                if score > best:
                    best = score
                    best_report = report
                    save_checkpoint(
                        path, {"model": model}, opt, step + 1, name, {"score": score}
                    )
        load_checkpoint(path, {"model": model})
        freeze(model)
        return best, best_report

    def identity_attack(self, family, scenario, direction, representation="public"):
        e = self.e
        model = IdentityModel(len(e.fit.patients), family).to(e.device)
        initialize_base(e, model, family)
        opt = optimizer(
            [model.model, model.classifier, model.project, model.pair],
            [1e-4, 1e-3, 1e-3, 1e-3],
        )
        total = e.cfg["budgets"]["attacker"]
        name = f"{family}-{scenario}-{direction}-{representation}"
        best = -1
        path = self.root / "checkpoints" / f"{name}.pt"
        qrep = representation
        grep = representation if direction == "same" else "rgb"
        for step in range(total):
            rng = rng_for(e.seed, name, "pairs", step)
            model.train()
            freeze_bn(model)
            opt.zero_grad()
            pairs = (
                e.sampler.sample(rng, scenario)
                if scenario != "p3"
                else self.source_pairs(rng)
            )
            prepared = []
            for a, b, s, _ in pairs:
                ba = collate([e.fit[a]], e.device)
                bb = collate([e.fit[b]], e.device)
                xa = self.representation("fit", qrep)(ba)
                xb = self.representation("fit", grep)(bb)
                if xa is not None and xb is not None:
                    prepared.append((ba, bb, xa, xb, s))
            counts = Counter(p[-1] for p in prepared)
            if not counts[0] or not counts[1]:
                continue
            for ba, bb, xa, xb, s in prepared:
                fa = model(xa)
                fb = model(xb)
                label = xa.new_tensor([s])
                score = model.score(fa, fb)
                loss = 0.5 * contrastive(
                    fa["embedding"], fb["embedding"], label
                ) + F.binary_cross_entropy_with_logits(score, label)
                if scenario != "p3":
                    loss = loss + 0.5 * (
                        F.cross_entropy(fa["logits"], ba["patient_index"])
                        + F.cross_entropy(fb["logits"], bb["patient_index"])
                    )
                (loss / (2 * counts[s])).backward()
            schedule(opt, step, total)
            update(opt)
            if (step + 1) % 1000 == 0 or step + 1 == total:
                if scenario == "p3":
                    report = attack_scoring.source_matching(
                        self, model, "val_select", qrep
                    )
                    score = report["pair"]["source_rank1"]
                else:
                    report = evaluate_identity(
                        model,
                        e.data["val_select"],
                        e.device,
                        scenario,
                        self.representation("val_select", qrep),
                        self.representation("val_select", grep),
                        True,
                    )
                    score = max(v.get("a_dagger", 0) for v in report.values())
                if score > best:
                    best = score
                    save_checkpoint(
                        path, {"model": model}, opt, step + 1, name, {"score": score}
                    )
        require(path.exists(), f"No valid attack checkpoint: {name}")
        load_checkpoint(path, {"model": model})
        freeze(model)
        result = (
            attack_scoring.source_matching(self, model, "test", qrep)
            if scenario == "p3"
            else evaluate_identity(
                model,
                e.data["test"],
                e.device,
                scenario,
                self.representation("test", qrep),
                self.representation("test", grep),
                True,
            )
        )
        if scenario != "p3":
            from ccrr.evaluation.comparisons import threshold_at_fpr

            calibration = evaluate_identity(
                model,
                e.data["val_tune"],
                e.device,
                scenario,
                self.representation("val_tune", qrep),
                self.representation("val_tune", grep),
                False,
            )
            thresholds = {}
            for mode, r in calibration.items():
                if r.get("valid"):
                    d = r["score_details"]
                    thresholds[mode] = threshold_at_fpr(
                        d["labels"], d["scores"], d["weights"]
                    )
            public_ids = {
                r["id"] for r in self.results["test"]["rows"] if r["published"]
            }
            result = evaluate_identity(
                model,
                e.data["test"],
                e.device,
                scenario,
                self.representation("test", qrep),
                self.representation("test", grep),
                True,
                allowed_queries=public_ids,
                allowed_gallery=public_ids if direction == "same" else None,
                thresholds=thresholds,
            )
            result["frozen_val_tune_thresholds"] = thresholds
        write_json(self.root / "attacks" / f"{name}.json", result)
        return result

    def source_pairs(self, rng):
        n = len(self.e.fit)
        pairs = []
        for _ in range(8):
            i = int(rng.integers(n))
            pairs.append((i, i, 1, "p3"))
        for _ in range(8):
            a, b = map(int, rng.choice(n, 2, replace=False))
            pairs.append((a, b, 0, "p3-negative"))
        return pairs

    def reconstruction_attack(self, latent=False, with_metadata=False):
        e = self.e
        model = ReconstructionAttack(latent, with_metadata).to(e.device)
        name = "latent" if latent else "metadata-rgb" if with_metadata else "rgb"
        opt = optimizer([model], [1e-4])
        total = e.cfg["budgets"]["reconstruction_attack"]
        best = -1
        path = self.root / "checkpoints" / f"attack-{name}.pt"
        available = [
            i
            for i, r in enumerate(e.fit.rows)
            if (self.folders["fit"] / r["img_id"]).exists()
        ]
        require(available, "No public fit samples")
        rows_by_id = {r["id"]: r for r in self.results["fit"]["rows"]}
        for step in range(total):
            opt.zero_grad()
            rng = rng_for(e.seed, "reconstruction-attack", name, step)
            for i in rng.choice(available, 8):
                b = collate([e.fit[int(i)]], e.device)
                g = self.public("fit", b)
                mb = (
                    metadata_tensor(b["metadata"], FIELDS, e.device)
                    if with_metadata
                    else None
                )
                pred = model(g, mb)
                if latent:
                    with torch.no_grad():
                        z = self.system.decompose(b["x"])[1]
                        r = rows_by_id[b["id"][0]]["level"]
                        zq = dequantize_latent(
                            quantize_latent(z, e.profile.step(r)), e.profile.step(r)
                        )
                    rec = self.system.recover(g, pred)
                    loss = (
                        reconstruction(b["x"], rec, b["v"])
                        + 0.1 * (pred - zq).abs().mean()
                    )
                else:
                    loss = reconstruction(b["x"], pred, b["v"])
                (loss / 8).backward()
            schedule(opt, step, total)
            update(opt)
            if (step + 1) % 2000 == 0 or step + 1 == total:
                metrics = attack_scoring.reconstruction_scores(
                    self, model, "val_select", latent, with_metadata
                )
                score = (
                    np.mean([r["quality"]["psnr"] for r in metrics]) if metrics else -1
                )
                if score > best:
                    best = score
                    save_checkpoint(
                        path,
                        {"model": model},
                        opt,
                        step + 1,
                        name,
                        {"psnr": float(score)},
                    )
        require(path.exists(), "No reconstruction attack validation examples")
        load_checkpoint(path, {"model": model})
        result = attack_scoring.reconstruction_scores(
            self, model, "test", latent, with_metadata
        )
        write_json(self.root / "attacks" / f"reconstruction-{name}.json", result)
        return result

    def attributes(self):
        """Auxiliary age/gender models: image, public fields, and their combination."""
        from ccrr.training.attribute_study import AttributeStudy

        return AttributeStudy(self).run()

    def run(self):
        self.publish()
        clinical_results = self.classifiers()
        attacks = {}
        for rep in ("rgb", "gray", "public"):
            for family in ("resnet50", "swin_t"):
                for scenario in ("p1", "p2", "p3"):
                    for direction in (
                        ("original",) if scenario == "p3" else ("original", "same")
                    ):
                        attacks[f"{rep}-{family}-{scenario}-{direction}"] = (
                            self.identity_attack(family, scenario, direction, rep)
                        )
            for structure in ("ncc", "ssim"):
                attacks[f"{rep}-p3-{structure}"] = attack_scoring.source_matching(
                    self, None, "test", rep, structure
                )
        from ccrr.evaluation.structural_attacks import evaluate as evaluate_structure

        public_ids = {r["id"] for r in self.results["test"]["rows"] if r["published"]}
        for rep in ("rgb", "gray", "public"):
            for scenario in ("p1", "p2"):
                for direction in ("original", "same"):
                    for method in ("ncc", "ssim"):
                        attacks[f"{rep}-{scenario}-{direction}-{method}"] = (
                            evaluate_structure(
                                self.e.data["test"],
                                self.e.device,
                                scenario,
                                self.representation("test", rep),
                                self.representation(
                                    "test", rep if direction == "same" else "rgb"
                                ),
                                method,
                                allowed_queries=public_ids,
                                allowed_gallery=public_ids
                                if direction == "same"
                                else None,
                            )
                        )
        recon = {}
        for latent, meta in ((False, False), (True, False), (False, True)):
            recon[f"{latent}-{meta}"] = self.reconstruction_attack(latent, meta)
        constant = []
        with torch.no_grad():
            for i in range(len(self.e.data["test"])):
                b = collate([self.e.data["test"][i]], self.e.device)
                g = self.public("test", b)
                if g is None:
                    continue
                for r in self.e.profile.legal_levels:
                    for use_map in (False, True):
                        rec = constant_latent_attack(self.system, g, r, use_map)
                        constant.append(
                            {
                                "id": b["id"][0],
                                "patient": b["patient"][0],
                                "level": r,
                                "map": use_map,
                                "quality": quality(b["x"], rec, b["v"])[0],
                            }
                        )
        write_json(self.root / "attacks/constant-latent.json", constant)
        from ccrr.evaluation.comparisons import capability_benefit

        authorized = [r for r in self.results["test"]["rows"] if r["c_success"]]
        comparison_attacks = {k: v for k, v in recon.items() if k != "False-True"}
        for r in self.e.profile.legal_levels:
            for use_map in (False, True):
                comparison_attacks[f"constant-{r}-{use_map}"] = [
                    v for v in constant if v["level"] == r and v["map"] == use_map
                ]
        benefit = capability_benefit(authorized, comparison_attacks)
        attributes = self.attributes()
        report = {
            "clinical": clinical_results,
            "identity": attacks,
            "reconstruction": recon,
            "constant_latent": constant,
            "attributes": attributes,
            "capability_benefit": benefit,
            "system": {k: v["funnel"] for k, v in self.results.items()},
            "formal_results_computed": True,
        }
        write_json(self.root / "report.json", report)
        return report
