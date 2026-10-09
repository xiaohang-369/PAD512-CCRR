"""A single experiment runner owns teachers, generator, communication and attacks.

Stage boundaries are scientific dependencies inside this complete system, not
separate products. No phase is allowed to fabricate a passed prerequisite.
"""

from pathlib import Path
from collections import defaultdict
import copy
import hashlib
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as tv
from .config import Profile, canonical_json, sha256
from .data import PADData, PairSampler, collate, rng_for
from .metadata import CLASSES
from .models.teachers import (
    ClinicalModel,
    IdentityModel,
    PanDerm,
    freeze,
    freeze_bn,
    load_weights,
)
from .models.purifier import Purifier, SemanticAttacker
from .models.common import gray, pooled, ste_round
from .system import CCRR
from .crypto import NonceAllocator
from .supervision import TeacherSupervision, align
from .losses import (
    reconstruction,
    gray_loss,
    saturation,
    contrastive,
    message_loss,
    masked_mean,
)
from .runtime import (
    seed_everything,
    environment,
    write_json,
    log_json,
    optimizer,
    schedule,
    update,
    save_checkpoint,
    load_checkpoint,
)
from .evaluation import evaluate_classifier, evaluate_identity, evaluate_system
from .metrics import quality, clinical
from .errors import require, CCRRError, Status
from .calibration import (
    ProbeGenerator,
    clinical_risk,
    probe_region,
    select_regions,
    labels,
    weighted_quantile,
    choose_threshold,
)


class Experiment:
    def __init__(self, config):
        self.cfg = config
        self.seed = int(config["seed"])
        seed_everything(self.seed)
        self.device = torch.device(config["device"])
        self.output = Path(config["output"])
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(
            self.output.resolve() != Path(__file__).parents[2].resolve(),
            "Outputs belong outside source",
        )
        write_json(self.output / "environment.json", environment())
        require(
            self.device.type != "cuda" or torch.cuda.is_available(),
            "CUDA unavailable on this host; use CPU for engineering checks and the H100 for the formal run",
        )
        self.profile = Profile.from_dict(config.get("profile", {}))
        self.data = {}
        for role in ("fit", "calibration", "val_tune", "val_select"):
            self.data[role] = PADData(config["data_root"], role, self.profile.size)
        self.fit = self.data["fit"]
        self.sampler = PairSampler(self.fit)
        self.weights = self.fit.class_weights().to(self.device)
        self.system = CCRR(
            self.profile, config.get("fixed_gray", False), config.get("patch_batch", 16)
        ).to(self.device)
        self.clinical = ClinicalModel().to(self.device)
        self.reid = IdentityModel(len(self.fit.patients)).to(self.device)
        self.purifier = Purifier().to(self.device)
        self.sem_attacker = SemanticAttacker().to(self.device)
        self.pub_attacker = IdentityModel(len(self.fit.patients)).to(self.device)
        self.mfm = None
        self.supervision = None
        self.allocator = NonceAllocator(self.output / "private" / "nonce.sqlite")
        self.acceptance = {}
        self.policies = {}
        self.risks = {}
        self.snapshot_index = 0
        write_json(self.output / "config.json", config)
        write_json(self.output / "environment.json", environment())
        write_json(
            self.output / "data_hashes.json",
            {
                role: sha256(
                    Path(config["data_root"])
                    / "protocol_audit_20261009"
                    / "roles"
                    / f"{role}.json"
                )
                for role in self.data
            },
        )

    def initialize_base(self, wrapper, name):
        entry = self.cfg["weights"][name]
        require(
            entry.get("path") and entry.get("sha256"),
            f"Missing official {name} weights and SHA-256",
        )
        # Strictly load a base architecture before replacing its task-specific head.
        base = getattr(tv, name)(weights=None)
        load_weights(base, entry["path"], entry["sha256"])
        if name == "efficientnet_b0":
            base.classifier[-1] = wrapper.model.classifier[-1]
        elif name == "convnext_tiny":
            base.classifier[-1] = wrapper.model.classifier[-1]
        elif name == "resnet50":
            base.fc = nn.Identity()
        elif name == "swin_t":
            base.head = wrapper.model.head
        wrapper.model = base.to(self.device)

    def preflight(self):
        report = {
            "environment": environment(),
            "missing": [],
            "training_started": False,
        }
        if self.device.type == "cuda" and not torch.cuda.is_available():
            report["missing"].append("CUDA device unavailable")
        for name, e in self.cfg["weights"].items():
            for key in ("path", "sha256"):
                if not e.get(key):
                    report["missing"].append(f"{name}.{key}")
            if e.get("path") and e.get("sha256"):
                if not Path(e["path"]).is_file() or sha256(e["path"]) != e["sha256"]:
                    report["missing"].append(f"{name} weight integrity")
        p = self.cfg["weights"]["panderm"]
        for key in ("source", "source_sha256"):
            if not p.get(key):
                report["missing"].append("panderm." + key)
        report["ready"] = not report["missing"]
        write_json(self.output / "preflight.json", report)
        require(report["ready"], f"Preflight incomplete: {report['missing']}")

    def batch(self, rng, n=1):
        return collate(
            [self.fit[int(i)] for i in rng.integers(len(self.fit), size=n)], self.device
        )

    def log(self, stage, step, values):
        log_json(
            self.output / "training.jsonl", {"stage": stage, "step": step, **values}
        )

    def mark(self, stage, passed, report):
        self.acceptance[stage] = {"passed": bool(passed), "report": report}
        write_json(self.output / "acceptance.json", self.acceptance)
        require(
            passed, f"{stage} did not pass its preregistered gate; see acceptance.json"
        )

    def train_classifier(self, model, name, budget, views, independent=False):
        self.initialize_base(model, model.architecture)
        # Split base and final classification layer without relying on a DEFAULT enum.
        head = (
            model.model.head
            if model.architecture == "swin_t"
            else model.model.classifier
        )
        head_ids = {id(p) for p in head.parameters()}
        backbone = [p for p in model.parameters() if id(p) not in head_ids]
        opt = optimizer([model], [1e-4])
        groups = []
        for group in opt.param_groups:
            for ish, lr in ((False, 1e-4), (True, 1e-3)):
                ps = [p for p in group["params"] if (id(p) in head_ids) == ish]
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
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        best = -float("inf")
        best_path = self.output / "checkpoints" / f"{name}.pt"
        last_report = None
        for step in range(budget):
            rng = rng_for(self.seed, name, "images", step)
            model.train()
            freeze_bn(model)
            opt.zero_grad(set_to_none=True)
            for p in backbone:
                p.requires_grad_(step >= 500)
            selected = rng.integers(
                len(self.fit), size=self.cfg.get("batch_teacher", 16)
            )
            # Class weighted CE normalization spans the effective batch and domains.
            denom = sum(
                float(self.weights[CLASSES.index(self.fit.rows[int(i)]["diagnostic"])])
                for i in selected
            ) * len(views)
            count = 0
            total = 0.0
            for i in selected:
                b = collate([self.fit[int(i)]], self.device)
                flip = rng.random() < 0.5
                for view in views:
                    x = view(b)
                    if x is None:
                        continue
                    rects = b["rect"]
                    if flip:
                        x = x.flip(-1)
                        rects = [
                            (x.shape[-1] - left - width, top, width, height)
                            for left, top, width, height in rects
                        ]
                    loss = (
                        F.cross_entropy(
                            model(x, rects),
                            b["label"],
                            weight=self.weights,
                            reduction="sum",
                        )
                        / denom
                    )
                    loss.backward()
                    total += float(loss.detach())
                    count += 1
            if not count:
                continue
            schedule(opt, step, budget)
            for group in opt.param_groups:
                if not group["head"]:
                    if step < 500:
                        group["lr"] = 0
                    elif step < 750:
                        group["lr"] = group["lr_max"] * (
                            0.01 + 0.99 * (step - 500) / 249
                        )
            norm = update(opt)
            if (step + 1) % 100 == 0:
                self.log(name, step + 1, {"loss": total, "gradient_norm": norm})
            if (step + 1) % 500 == 0 or step + 1 == budget:
                reports = [
                    evaluate_classifier(
                        model, self.data["val_select"], self.device, v, bootstrap=False
                    )
                    for v in views
                ]
                score = min(
                    r["macro_auroc"] if r["macro_auroc"] is not None else -float("inf")
                    for r in reports
                )
                if score > best:
                    best = score
                    last_report = reports
                    save_checkpoint(
                        best_path,
                        {"model": model},
                        opt,
                        step + 1,
                        name,
                        {"score": score, "independent": independent},
                    )
                write_json(
                    self.output / "validation" / f"{name}-{step + 1}.json", reports
                )
        require(best_path.exists(), f"{name}: no evaluable checkpoint")
        load_checkpoint(best_path, {"model": model})
        freeze(model)
        return best, last_report

    def stage_a(self):
        self.preflight()
        score, report = self.train_classifier(
            self.clinical,
            "clinical",
            self.cfg["budgets"]["clinical"],
            [lambda b: b["x"], lambda b: gray(b["x"])],
        )
        controls = []
        for mode, view in [("rgb", lambda b: b["x"]), ("gray", lambda b: gray(b["x"]))]:
            m = ClinicalModel().to(self.device)
            s, r = self.train_classifier(
                m, "clinical_" + mode, self.cfg["budgets"]["clinical"], [view]
            )
            controls.append(s)
        passed = score >= 0.75 and all(
            report[j]["macro_auroc"] >= controls[j] - 0.03
            and all(a is not None and a > 0.5 for a in report[j]["per_class_auroc"])
            and all(v is not None and v > 0 for v in report[j]["recall"])
            for j in range(2)
        )
        self.mark("A-clinical", passed, {"joint": report, "controls": controls})
        e = self.cfg["weights"]["panderm"]
        self.mfm = PanDerm(e["source"], e["source_sha256"], e["path"], e["sha256"]).to(
            self.device
        )
        self.train_mfm_probe()
        self.initialize_base(self.reid, "resnet50")
        self.train_reid_teacher()
        self.initialize_base(self.pub_attacker, "resnet50")

    def train_mfm_probe(self):
        head = nn.Linear(768, 6).to(self.device)
        opt = optimizer([head], [1e-3])
        total = self.cfg["budgets"]["mfm_probe"]
        best = -1
        best_state = None
        # Frozen standardization statistics: equal images and RGB/gray domains.
        sums = torch.zeros(768, device=self.device)
        squares = sums.clone()
        count = 0
        with torch.no_grad():
            for i in range(len(self.fit)):
                b = collate([self.fit[i]], self.device)
                for x in (b["x"], gray(b["x"])):
                    f, v, _ = self.mfm(x, b["rect"])
                    sums += pooled(f, v)[0]
                    squares += pooled(f.square(), v)[0]
                    count += 1
        mean = sums / count
        std = (squares / count - mean.square()).clamp_min(0).sqrt()
        self.purifier.mean.copy_(mean[None, :, None, None])
        self.purifier.std.copy_(std[None, :, None, None])
        self.purifier.stats_ready.fill_(True)
        for step in range(total):
            opt.zero_grad(set_to_none=True)
            rng = rng_for(self.seed, "mfm_probe", "images", step)
            ids = rng.integers(len(self.fit), size=16)
            denom = (
                sum(
                    float(
                        self.weights[CLASSES.index(self.fit.rows[int(i)]["diagnostic"])]
                    )
                    for i in ids
                )
                * 2
            )
            for i in ids:
                b = collate([self.fit[int(i)]], self.device)
                for x in (b["x"], gray(b["x"])):
                    with torch.no_grad():
                        f, v, _ = self.mfm(x, b["rect"])
                        h = pooled(f, v)
                    (
                        F.cross_entropy(
                            head(h), b["label"], weight=self.weights, reduction="sum"
                        )
                        / denom
                    ).backward()
            schedule(opt, step, total)
            update(opt)
            if (step + 1) % 250 == 0 or step + 1 == total:
                reports = []
                for gray_mode in (False, True):
                    ys = []
                    ps = []
                    with torch.no_grad():
                        for i in range(len(self.data["val_select"])):
                            b = collate([self.data["val_select"][i]], self.device)
                            f, v, _ = self.mfm(
                                gray(b["x"]) if gray_mode else b["x"], b["rect"]
                            )
                            ys.append(int(b["label"][0]))
                            ps.append(head(pooled(f, v)).softmax(1)[0].cpu().numpy())
                    reports.append(clinical(ys, ps))
                score = min(r["macro_auroc"] or 0 for r in reports)
                if score > best:
                    best = score
                    best_state = copy.deepcopy(head.state_dict())
                    best_reports = reports
        head.load_state_dict(best_state)
        save_checkpoint(
            self.output / "checkpoints/mfm_probe.pt",
            {"probe": head, "purifier": self.purifier},
            None,
            total,
            "A-mfm",
            {"score": best},
        )
        self.mark("A-mfm", best >= 0.75, best_reports)

    def train_reid_teacher(self):
        model = self.reid
        opt = optimizer([model.model, model.classifier], [1e-4, 1e-3])
        total = self.cfg["budgets"]["reid"]
        best = -1
        path = self.output / "checkpoints/reid.pt"
        best_report = None
        for step in range(total):
            rng = rng_for(self.seed, "reid", "pairs", step)
            pairs = self.sampler.sample(rng)
            model.train()
            freeze_bn(model)
            opt.zero_grad(set_to_none=True)
            for a, b, s, scenario in pairs:
                ba = collate([self.fit[a]], self.device)
                bb = collate([self.fit[b]], self.device)
                domain = int(rng.integers(3))
                xa = gray(ba["x"]) if domain in (1, 2) else ba["x"]
                xb = gray(bb["x"]) if domain == 2 else bb["x"]
                if rng.random() < 0.5:
                    xa = xa.flip(-1)
                    ba["rect"] = [
                        (xa.shape[-1] - l - w, t, w, h) for l, t, w, h in ba["rect"]
                    ]
                if rng.random() < 0.5:
                    xb = xb.flip(-1)
                    bb["rect"] = [
                        (xb.shape[-1] - l - w, t, w, h) for l, t, w, h in bb["rect"]
                    ]
                fa = model(xa, ba["rect"])
                fb = model(xb, bb["rect"])
                label = xa.new_tensor([s])
                ce = 0.5 * (
                    F.cross_entropy(fa["logits"], ba["patient_index"])
                    + F.cross_entropy(fb["logits"], bb["patient_index"])
                )
                loss = (
                    ce + 0.5 * contrastive(fa["embedding"], fb["embedding"], label)
                ) / 16
                loss.backward()
            schedule(opt, step, total)
            update(opt)
            if (step + 1) % 1000 == 0 or step + 1 == total:
                reports = [
                    evaluate_identity(
                        model,
                        self.data["val_select"],
                        self.device,
                        "p1",
                        qa,
                        ga,
                        bootstrap=True,
                    )
                    for qa, ga in [
                        (None, None),
                        (lambda b: gray(b["x"]), None),
                        (lambda b: gray(b["x"]), lambda b: gray(b["x"])),
                    ]
                ]
                score = min(r["cosine"].get("auroc", 0) for r in reports)
                if score > best:
                    best = score
                    best_report = reports
                    save_checkpoint(
                        path,
                        {"model": model},
                        opt,
                        step + 1,
                        "A-reid",
                        {"score": score},
                    )
        load_checkpoint(path, {"model": model})
        freeze(model)
        passed = best >= 0.70 and all(
            r["cosine"]["auroc_ci"]["interval"]
            and r["cosine"]["auroc_ci"]["interval"][0] > 0.5
            and r["cosine"]["auroc_ci"]["invalid_fraction"] <= 0.05
            for r in best_report
        )
        self.mark("A-reid", passed, best_report)

    def semantic_view(self, b, domain):
        x = b["x"]
        y = gray(x)
        with torch.no_grad():
            fx, v, _ = self.mfm(x, b["rect"])
            fy, _, _ = self.mfm(y, b["rect"])
        px = self.purifier(fx, v)
        if domain == 0:
            return px, v
        return self.purifier(fy, v, source_mask=px["mask"] if domain == 2 else None), v

    def semantic_adversary_step(self, step, substep, opt):
        rng = rng_for(self.seed, "B", "attacker" + str(substep), step)
        pairs = self.sampler.sample(rng)
        opt.zero_grad(set_to_none=True)
        for a, b, s, _ in pairs:
            ba = collate([self.fit[a]], self.device)
            bb = collate([self.fit[b]], self.device)
            with torch.no_grad():
                pa, va = self.semantic_view(ba, int(rng.integers(3)))
                pb, vb = self.semantic_view(bb, int(rng.integers(3)))
            logit = self.sem_attacker(
                pa["semantic"].detach(), va, pb["semantic"].detach(), vb
            )
            (
                F.binary_cross_entropy_with_logits(logit, logit.new_tensor([s])) / 16
            ).backward()
        update(opt)

    def stage_b(self):
        require(
            self.mfm is not None and bool(self.purifier.stats_ready),
            "Teacher stages required",
        )
        self.purifier.train()
        self.sem_attacker.train()
        opt = optimizer([self.purifier], [1e-4])
        advopt = optimizer([self.sem_attacker], [1e-4])
        total = self.cfg["budgets"]["purifier"]
        for step in range(total):
            for p in self.sem_attacker.parameters():
                p.requires_grad_(True)
            for sub in range(2):
                self.semantic_adversary_step(step, sub, advopt)
            freeze(self.sem_attacker)
            opt.zero_grad(set_to_none=True)
            rng = rng_for(self.seed, "B", "images", step)
            ids = rng.integers(len(self.fit), size=16)
            denom = (
                sum(
                    float(
                        self.weights[CLASSES.index(self.fit.rows[int(i)]["diagnostic"])]
                    )
                    for i in ids
                )
                * 3
            )
            for i in ids:
                b = collate([self.fit[int(i)]], self.device)
                x = b["x"]
                y = gray(x)
                with torch.no_grad():
                    fx, v, _ = self.mfm(x, b["rect"])
                    fy, _, _ = self.mfm(y, b["rect"])
                    ex = self.reid(x, b["rect"])["embedding"]
                    ey = self.reid(y, b["rect"])["embedding"]
                px = self.purifier(fx, v)
                py = self.purifier(fy, v)
                cross = self.purifier(fy, v, source_mask=px["mask"])
                loss = x.sum() * 0
                for out, e in ((px, ex), (py, ey)):
                    all_loss = masked_mean(
                        (out["all"] - out["standardized"]).square(), v
                    )
                    id_loss = (out["identity"] - e).square().sum(1).mean()
                    occupancy = pooled(out["mask"], v).mean(1)
                    mask_loss = (
                        F.relu(0.1 - occupancy).square()
                        + F.relu(occupancy - 0.5).square()
                    ).mean()
                    loss = loss + (all_loss + 0.1 * id_loss + 10 * mask_loss) / 32
                for out in (px, py, cross):
                    loss = (
                        loss
                        + F.cross_entropy(
                            out["logits"],
                            b["label"],
                            weight=self.weights,
                            reduction="sum",
                        )
                        / denom
                    )
                loss.backward()
            ramp = max(0, min(1, (step - 999) / 1000))
            if ramp:
                pairs = self.sampler.sample(
                    rng_for(self.seed, "B", "generator_pairs", step)
                )
                for a, b, s, _ in pairs:
                    ba = collate([self.fit[a]], self.device)
                    bb = collate([self.fit[b]], self.device)
                    pa, va = self.semantic_view(ba, int(rng.integers(3)))
                    pb, vb = self.semantic_view(bb, int(rng.integers(3)))
                    logit = self.sem_attacker(pa["semantic"], va, pb["semantic"], vb)
                    (
                        -0.1
                        * ramp
                        * F.binary_cross_entropy_with_logits(
                            logit, logit.new_tensor([s])
                        )
                        / 16
                    ).backward()
            schedule(opt, step, total)
            norm = update(opt)
            if (step + 1) % 100 == 0:
                self.log(
                    "B", step + 1, {"gradient_norm": norm, "adversarial_ramp": ramp}
                )
            if (step + 1) % 1000 == 0 or step + 1 == total:
                save_checkpoint(
                    self.output / "checkpoints" / f"purifier-{step + 1}.pt",
                    {"purifier": self.purifier, "attacker": self.sem_attacker},
                    opt,
                    step + 1,
                    "B",
                    {},
                )
        # Select among all registered endpoints using independently trained readouts.
        best = None
        for path in sorted(
            (self.output / "checkpoints").glob("purifier-*.pt"),
            key=lambda p: int(p.stem.split("-")[-1]),
        ):
            checkpoint = load_checkpoint(path, {"purifier": self.purifier})
            freeze(self.purifier)
            self.feature_audit_tag = "step-" + str(checkpoint["step"])
            report = self.validate_purifier()
            if report["passed"]:
                clinical_min = min(
                    r["macro_auroc"]
                    for r in report["clinical"]["readouts"]["sem_linear"]
                )
                strongest = max(
                    r["sem_strongest"] for r in report["identity"]["checks"]
                )
                key = (clinical_min, -strongest, -checkpoint["step"])
                if best is None or key > best[0]:
                    best = (key, report)
                    save_checkpoint(
                        self.output / "checkpoints/purifier.pt",
                        {"purifier": self.purifier},
                        None,
                        checkpoint["step"],
                        "B",
                        report,
                    )
        self.mark(
            "B",
            best is not None,
            best[1]
            if best
            else {"reason": "No endpoint passed frozen-feature acceptance"},
        )
        load_checkpoint(
            self.output / "checkpoints/purifier.pt", {"purifier": self.purifier}
        )
        freeze(self.purifier)
        tau = self.identity_threshold()
        self.supervision = TeacherSupervision(
            self.clinical, self.mfm, self.reid, self.purifier, self.weights, tau
        )

    def validate_purifier(self):
        from .feature_audit import FeatureAudit

        return FeatureAudit(self).run()

    def identity_threshold(self):
        path = (
            self.fit.root
            / "protocol_audit_20261009/pair_protocol/fit_local_identity_negative_pairs.json"
        )
        obj = json.loads(path.read_text())
        pairs = obj.get("pairs", obj) if isinstance(obj, dict) else obj
        values = []
        weights = []
        with torch.no_grad():
            for pair in pairs:
                # Schema is checked explicitly; no guessed pairing or fabricated threshold.
                a = pair.get("rgb_img")
                b = pair.get("gray_img")
                require(
                    a in self.fit.index and b in self.fit.index,
                    "Local negative-pair schema requires rgb_img/gray_img",
                )
                ba = collate([self.fit[self.fit.index[a]]], self.device)
                bb = collate([self.fit[self.fit.index[b]]], self.device)
                fa = self.reid(ba["x"], ba["rect"])
                fb = self.reid(gray(bb["x"]), bb["rect"])
                aa, av = align(fa["spatial"], fa["transform"], self.profile.size)
                bf, bv = align(fb["spatial"], fb["transform"], self.profile.size)
                s = F.cosine_similarity(aa, bf, dim=1, eps=1e-6)[0]
                w = torch.minimum(av, bv)[0]
                w = w / w.sum()
                values.extend(s.cpu().tolist())
                weights.extend((w * float(pair["pair_weight"])).cpu().tolist())
        tau = weighted_quantile(values, weights)
        write_json(
            self.output / "identity_threshold.json",
            {"tau": tau, "pair_manifest": sha256(path)},
        )
        return tau

    def bind_snapshot(self, tag):
        cdfs = self.system.entropy.export()
        path = self.output / "snapshots" / tag
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.system.state_dict(), path / "system.pt")
        np.save(path / "cdfs.npy", cdfs)
        manifest = {
            "research_snapshot": tag,
            "fixed_gray": self.system.fixed_gray,
            "system": sha256(path / "system.pt"),
            "cdf": sha256(path / "cdfs.npy"),
            "profile": self.profile.__dict__,
        }
        raw = canonical_json(manifest)
        (path / "manifest.json").write_bytes(raw)
        self.system.bind(hashlib.sha256(raw).digest(), cdfs)

    def pair_public(self, b, stage, rng):
        if stage in ("D", "E"):
            return self.system.decompose(b["x"])[2]
        require(b["id"][0] in self.policies, "Missing snapshot-bound sender policy")
        credentials = self.allocator.new_object(self.system.digest)
        sent = self.system.send(
            b["x"],
            b["metadata"][0],
            b["legal"],
            self.risks[b["id"][0]],
            credentials,
            self.allocator,
            self.policies[b["id"][0]],
            rng.permutation(self.profile.legal_levels).tolist(),
        )
        return sent["public"]

    def pair_objective(self, stage, step, attacker_update=False, opt=None, sub=0):
        rng = rng_for(
            self.seed,
            stage,
            "attack" if attacker_update else "generator-pairs",
            step * 3 + sub,
        )
        pairs = self.sampler.sample(rng)
        feasible = []
        # Determine feasibility from complete source rate before forming the two means.
        for a, b, s, scenario in pairs:
            domain = int(rng.integers(2))
            ba = collate([self.fit[a]], self.device)
            bb = collate([self.fit[b]], self.device)
            try:
                with torch.no_grad():
                    xa = self.pair_public(ba, stage, rng)
                    if domain:
                        xb = self.pair_public(bb, stage, rng)
                feasible.append((a, b, s, domain, scenario))
            except CCRRError as exc:
                if exc.status != Status.CAPACITY:
                    raise
        counts = CounterLocal([s for _, _, s, _, _ in feasible])
        if not counts.get(0) or not counts.get(1):
            return False
        if attacker_update:
            opt.zero_grad(set_to_none=True)
        for a, b, s, domain, scenario in feasible:
            ba = collate([self.fit[a]], self.device)
            bb = collate([self.fit[b]], self.device)
            if attacker_update:
                with torch.no_grad():
                    xa = self.pair_public(ba, stage, rng)
                    xb = self.pair_public(bb, stage, rng) if domain else bb["x"]
            else:
                xa = self.pair_public(ba, stage, rng)
                xb = self.pair_public(bb, stage, rng) if domain else bb["x"]
            fa = self.pub_attacker(xa)
            fb = self.pub_attacker(xb)
            logit = self.pub_attacker.score(fa, fb)
            loss = F.binary_cross_entropy_with_logits(logit, logit.new_tensor([s])) / (
                2 * counts[s]
            )
            if not attacker_update:
                loss = (
                    -self.cfg.get("lambda_privacy", 0.1)
                    * (min(1, (step + 1) / 2000) if stage == "D" else 1)
                    * loss
                )
            loss.backward()
        if attacker_update:
            update(opt)
        return True

    def adversarial_updates(self, stage, step, opt):
        self.pub_attacker.train()
        freeze_bn(self.pub_attacker)
        for p in self.pub_attacker.parameters():
            p.requires_grad_(True)
        for sub in range(2):
            valid = False
            for attempt in range(100):
                if self.pair_objective(stage, step * 100 + attempt, True, opt, sub):
                    valid = True
                    break
            require(valid, "100 invalid attacker pair batches; stop stage")
        freeze(self.pub_attacker)

    def validate_recovery(self, stage, step, levels):
        results = []
        self.system.eval()
        with torch.no_grad():
            for i in range(len(self.data["val_select"])):
                b = collate([self.data["val_select"][i]], self.device)
                u, z, g = self.system.decompose(b["x"])
                continuous = self.system.inn.inverse(u, z)
                err = (continuous - b["x"]).abs()
                require(
                    float(err.max()) <= 1e-4
                    and float(err.square().mean().sqrt()) <= 1e-5,
                    "INN numeric gate",
                )
                for level in levels:
                    n = ste_round(z / self.profile.step(level))
                    rec = self.system.recover(g, n * self.profile.step(level))
                    q = quality(b["x"], rec, b["v"])[0]
                    q.update(id=b["id"][0], level=level)
                    if stage == "E":
                        source = self.system.codec.encode(n, level)
                        decoded, zq = self.system.codec.decode(
                            source, level, self.device
                        )
                        require(
                            torch.equal(decoded.float(), n),
                            "Actual entropy integer mismatch",
                        )
                        q["source_bytes"] = len(source)
                        q["ideal_fits"] = (len(source) + 88 + 15) // 16 + 26 <= 4 * int(
                            b["legal"].sum()
                        )
                    results.append(q)
        report = {"stage": stage, "step": step, "rows": results}
        report["mean_psnr"] = float(np.mean([r["psnr"] for r in results]))
        report["mean_ssim"] = float(
            np.mean([r["ssim"] for r in results if r["ssim"] is not None])
        )
        report["fraction_psnr28_ssim90"] = float(
            np.mean(
                [
                    r["psnr"] >= 28 and r["ssim"] is not None and r["ssim"] >= 0.90
                    for r in results
                ]
            )
        )
        report["passed_quality"] = (
            report["mean_psnr"] >= 30
            and report["mean_ssim"] >= 0.95
            and report["fraction_psnr28_ssim90"] >= 0.90
        )
        if stage == "E":
            good = {
                r["id"]
                for r in results
                if r["ideal_fits"]
                and r["psnr"] >= 30
                and r["ssim"] is not None
                and r["ssim"] >= 0.95
            }
            report["ideal_coverage"] = len(good) / len(self.data["val_select"])
            report["mean_protected_bpp"] = float(
                np.mean(
                    [
                        8 * (r["source_bytes"] + 88 + 408) / self.profile.size**2
                        for r in results
                    ]
                )
            )
        write_json(self.output / "validation" / f"{stage}-{step}.json", report)
        return report

    def validate_representation_semantics(self):
        from .comparisons import paired_clinical, privacy_points
        from .source_privacy import evaluate_source_privacy

        ds = self.data["val_select"]

        def generated(b):
            with torch.no_grad():
                return self.system.decompose(b["x"])[2]

        standard = lambda b: gray(b["x"])
        rgb = evaluate_classifier(self.clinical, ds, self.device, None, False)
        generated_clinical = evaluate_classifier(
            self.clinical, ds, self.device, generated, False
        )
        clinical_check = paired_clinical(rgb, generated_clinical)
        checks = {}
        for scenario in ("p1", "p2"):
            for direction in ("original", "same"):
                rgb_report = evaluate_identity(
                    self.pub_attacker, ds, self.device, scenario, None, None, True
                )
                gray_report = evaluate_identity(
                    self.pub_attacker,
                    ds,
                    self.device,
                    scenario,
                    standard,
                    standard if direction == "same" else None,
                    True,
                )
                public_report = evaluate_identity(
                    self.pub_attacker,
                    ds,
                    self.device,
                    scenario,
                    generated,
                    generated if direction == "same" else None,
                    True,
                )
                checks[scenario + "-" + direction] = privacy_points(
                    rgb_report, gray_report, public_report
                )
        source = evaluate_source_privacy(
            ds,
            self.device,
            generated,
            {"reid": self.reid, "adaptive": self.pub_attacker},
        )
        return {
            "clinical": clinical_check,
            "identity": checks,
            "source": source,
            "passed": clinical_check["passed_point"]
            and all(v["passed"] for v in checks.values())
            and source["passed"],
        }

    def train_representation(self, stage):
        require(stage in ("C", "D", "E"), "Representation stage")
        if stage != "C":
            require(
                self.supervision is not None,
                "Validated teacher/Purifier bundle required",
            )
        for p in self.system.inn.parameters():
            p.requires_grad_(True)
        for p in self.system.entropy.parameters():
            p.requires_grad_(stage != "C")
        name = {"C": "recovery", "D": "representation", "E": "multirate"}[stage]
        total = self.cfg["budgets"][name]
        opt = optimizer(
            [self.system.inn, self.system.entropy],
            [5e-5 if stage == "E" else 1e-4, 5e-4 if stage == "E" else 1e-3],
        )

        for p in self.pub_attacker.parameters():
            p.requires_grad_(True)
        advopt = optimizer([self.pub_attacker], [1e-4])
        best = None
        selected = None
        interval = 1000 if stage == "C" else 2000
        if stage == "D":
            for p in self.pub_attacker.parameters():
                p.requires_grad_(True)
            for j in range(1000):
                self.pair_objective("D", j, True, advopt)
        for step in range(total):
            self.system.train()
            rng = rng_for(self.seed, stage, "images", step)
            if stage != "C":
                self.adversarial_updates(stage, step, advopt)
            opt.zero_grad(set_to_none=True)
            loss_values = []
            level = int(rng.integers(1, 4)) if stage == "E" else 1
            for _ in range(self.cfg.get("batch_images", 8)):
                b = self.batch(rng)
                u, z, g, n, zq = self.system.quantized(b["x"], level)
                rec = self.system.recover(g, zq)
                loss = (
                    reconstruction(b["x"], rec, b["v"])
                    + 0.1 * gray_loss(g, b["x"], b["v"])
                    + 1e-3 * saturation(u, b["v"])
                )
                if stage != "C":
                    target = self.supervision.targets(b["x"], b["rect"], b["label"])
                    clin, ident, _ = self.supervision.loss(
                        g, b["rect"], b["label"], target
                    )
                    rate = self.system.entropy.rate(
                        z, level, self.profile.step(level)
                    ).mean()
                    ramp = min(1, (step + 1) / 2000) if stage == "D" else 1
                    loss = loss + ramp * (
                        clin
                        + 0.1 * self.cfg["lambda_privacy"] * ident
                        + self.cfg["lambda_rate"] * rate
                    )
                (loss / self.cfg.get("batch_images", 8)).backward()
                loss_values.append(float(loss.detach()))
            if stage != "C" and not self.system.fixed_gray:
                self.pair_objective(stage, step)
            schedule(opt, step, total)
            norm = update(opt)
            if (step + 1) % 100 == 0:
                self.log(
                    stage,
                    step + 1,
                    {
                        "loss": np.mean(loss_values),
                        "gradient_norm": norm,
                        "level": level,
                    },
                )
            if (step + 1) % interval == 0 or step + 1 == total:
                if stage == "E":
                    self.bind_snapshot(f"E-{step + 1}")
                report = self.validate_recovery(
                    stage, step + 1, (1, 2, 3) if stage == "E" else (1,)
                )
                if stage in ("D", "E") and not self.system.fixed_gray:
                    report["semantic_screen"] = self.validate_representation_semantics()
                    report["passed_quality"] = (
                        report["passed_quality"] and report["semantic_screen"]["passed"]
                    )
                value = (
                    (
                        -report.get("ideal_coverage", 0),
                        report.get("mean_protected_bpp", 0),
                        -report["mean_psnr"],
                    )
                    if stage == "E"
                    else (-report["mean_psnr"],)
                )
                if report["passed_quality"] and (best is None or value < best):
                    best = value
                    selected = report
                    save_checkpoint(
                        self.output / "checkpoints" / f"{stage}.pt",
                        {"system": self.system},
                        opt,
                        step + 1,
                        stage,
                        report,
                    )
        require(selected is not None, f"{stage}: no quality-qualified checkpoint")
        load_checkpoint(
            self.output / "checkpoints" / f"{stage}.pt", {"system": self.system}
        )
        self.mark(
            stage,
            selected["passed_quality"]
            and (stage != "E" or selected["ideal_coverage"] >= 0.95),
            selected,
        )
        if stage == "E":
            self.bind_snapshot("E-selected")

    def stage_f(self):
        freeze(self.system.inn)
        freeze(self.system.entropy)
        self.system.comm.train()
        total = self.cfg["budgets"]["communication"]
        opt = optimizer(
            [
                self.system.comm.carrier,
                self.system.comm.writer,
                self.system.comm.decoder,
            ],
            [1e-4] * 3,
        )
        generator = ProbeGenerator(self.profile, self.system.digest, self.allocator)
        best = None
        for step in range(total):
            rng = rng_for(self.seed, "F", "records", step)
            opt.zero_grad(set_to_none=True)
            values = []
            maxload = 1 if step < 5000 else 2 if step < 10000 else 4
            for _ in range(self.cfg.get("batch_images", 8)):
                b = self.batch(rng)
                with torch.no_grad():
                    g = self.system.decompose(b["x"])[2]
                loads = (
                    np.zeros(self.profile.regions, int)
                    if rng.random() < 0.1
                    else rng.integers(0, maxload + 1, self.profile.regions)
                    * b["legal"][0].cpu().numpy().astype(int)
                )
                bits, occ = generator.table(b["metadata"][0], loads, rng)
                bt = torch.tensor(bits, device=self.device)[None]
                ot = torch.tensor(occ, device=self.device)[None]
                p, _ = self.system.comm.write(g, bt, ot, b["legal"])
                logits = self.system.comm.read(p)
                dec, _ = message_loss(logits, bt, ot)
                loss = dec + masked_mean((p - g).abs() / (4 / 255), b["v"])
                (loss / self.cfg.get("batch_images", 8)).backward()
                values.append(float(loss.detach()))
            schedule(opt, step, total)
            norm = update(opt)
            if (step + 1) % 100 == 0:
                self.log(
                    "F",
                    step + 1,
                    {
                        "loss": np.mean(values),
                        "gradient_norm": norm,
                        "maximum_load": maxload,
                    },
                )
            if (step + 1) % 2000 == 0 or step + 1 == total:
                save_checkpoint(
                    self.output / "checkpoints" / f"F-{step + 1}.pt",
                    {"system": self.system},
                    opt,
                    step + 1,
                    "F",
                    {},
                )
        candidates = sorted(
            (self.output / "checkpoints").glob("F-*.pt"),
            key=lambda p: int(p.stem.split("-")[-1]),
        )
        for path in candidates:
            checkpoint = load_checkpoint(path, {"system": self.system})
            if checkpoint["step"] < 10000:
                continue
            report = self.validate_communication_warmup(checkpoint["step"])
            score = (
                report["full_table_success"],
                -report["empty_false_rate"],
                -report["mean_modification"],
                -checkpoint["step"],
            )
            if best is None or score > best:
                best = score
                save_checkpoint(
                    self.output / "checkpoints/F.pt",
                    {"system": self.system},
                    None,
                    checkpoint["step"],
                    "F",
                    report,
                )
        require(best is not None, "No last-curriculum communication checkpoint")
        load_checkpoint(self.output / "checkpoints/F.pt", {"system": self.system})
        self.bind_snapshot("F-selected")

    @torch.no_grad()
    def validate_communication_warmup(self, step):
        self.system.eval()
        generator = ProbeGenerator(self.profile, self.system.digest, self.allocator)
        successes = []
        empty_errors = []
        modifications = []
        for i in range(len(self.data["val_tune"])):
            b = collate([self.data["val_tune"][i]], self.device)
            legal = b["legal"][0].cpu().numpy()
            g = self.system.decompose(b["x"])[2]
            for region in select_regions(b["id"][0], legal, 0):
                for k in range(5):
                    rng = rng_for(
                        self.seed, "F-validation", b["id"][0] + str(region), k
                    )
                    result = probe_region(
                        self.system.comm,
                        g,
                        legal,
                        b["metadata"][0],
                        b["rect"],
                        self.clinical,
                        generator,
                        region,
                        k,
                        32,
                        rng,
                        self.profile.occupancy_threshold,
                    )
                    if k:
                        successes.extend(result["success"])
                    else:
                        empty_errors.extend(not v for v in result["success"])
                    modifications.extend(result["rmse"])
        report = {
            "full_table_success": float(np.mean(successes)),
            "empty_false_rate": float(np.mean(empty_errors)),
            "mean_modification": float(np.mean(modifications)),
            "step": step,
        }
        write_json(self.output / "validation" / f"F-{step}.json", report)
        return report

    def collect_risks(self, roles):
        raw = {}
        weights = []
        values = []
        for role in roles:
            ds = self.data[role]
            patient_images = CounterLocal([r["patient_id"] for r in ds.rows])
            for i in range(len(ds)):
                b = collate([ds[i]], self.device)
                with torch.no_grad():
                    g = self.system.decompose(b["x"])[2]
                risk = clinical_risk(
                    g, b["legal"][0].cpu().numpy(), b["id"][0], self.clinical, b["rect"]
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
            self.risks[name] = torch.tensor(
                np.minimum(r / max(q95, 1e-6), 1),
                device=self.device,
                dtype=torch.float32,
            )[None]
        return q95

    def stage_g(self, final=False):
        self.system.eval()
        self.snapshot_index += 1
        q95 = self.collect_risks(("calibration", "val_tune", "fit", "val_select"))
        generator = ProbeGenerator(self.profile, self.system.digest, self.allocator)
        threshold_samples = []
        for i in range(len(self.data["val_tune"])):
            b = collate([self.data["val_tune"][i]], self.device)
            legal = b["legal"][0].cpu().numpy()
            with torch.no_grad():
                g = self.system.decompose(b["x"])[2]
            for region in select_regions(b["id"][0], legal, self.snapshot_index):
                for k in range(5):
                    rng = rng_for(self.seed, "G-threshold", b["id"][0] + str(region), k)
                    p = probe_region(
                        self.system.comm,
                        g,
                        legal,
                        b["metadata"][0],
                        b["rect"],
                        self.clinical,
                        generator,
                        region,
                        k,
                        32,
                        rng,
                        0.5,
                    )
                    threshold_samples.extend(
                        zip(p["logits"], p["truth"], p["occupancy"])
                    )
        threshold, accepted, details = choose_threshold(threshold_samples)
        require(accepted, "No receiver threshold passed empty-table criterion")
        from dataclasses import replace

        self.profile = replace(
            self.profile, occupancy_threshold=threshold["threshold"], q95=max(q95, 1e-6)
        )
        self.system.profile = self.profile
        training = []
        external = []
        for role in ("calibration", "val_tune"):
            ds = self.data[role]
            for i in range(len(ds)):
                b = collate([ds[i]], self.device)
                legal = b["legal"][0].cpu().numpy()
                with torch.no_grad():
                    g = self.system.decompose(b["x"])[2]
                    z = self.system.comm.carrier(g)
                for region in select_regions(b["id"][0], legal, self.snapshot_index):
                    probes = {}
                    for k in range(5):
                        rng = rng_for(
                            self.seed,
                            "G-final" if final else "G",
                            b["id"][0] + str(region),
                            k + self.snapshot_index * 5,
                        )
                        probes[k] = probe_region(
                            self.system.comm,
                            g,
                            legal,
                            b["metadata"][0],
                            b["rect"],
                            self.clinical,
                            generator,
                            region,
                            k,
                            256 if final else 64,
                            rng,
                            self.profile.occupancy_threshold,
                            deployment=self.policies.get(b["id"][0], (None, None))[0],
                        )
                    label = labels(probes, final)
                    entry = {
                        "id": b["id"][0],
                        "patient": b["patient"][0],
                        "region": region,
                        "label": label,
                        "z": z[0, region].detach().cpu(),
                        "risk": float(self.risks[b["id"][0]][0, region]),
                    }
                    (training if role == "calibration" else external).append(entry)
        head = self.system.comm.heads
        for p in head.parameters():
            p.requires_grad_(True)
        opt = optimizer([head], [1e-3])
        total = self.cfg["budgets"]["heads"]
        groups = defaultdict(lambda: defaultdict(list))
        for row in training:
            groups[row["patient"]][row["id"]].append(row)
        for step in range(total):
            rng = rng_for(
                self.seed, "G-head", "sample", step + self.snapshot_index * total
            )
            p = str(rng.choice(sorted(groups)))
            name = str(rng.choice(sorted(groups[p])))
            row = groups[p][name][int(rng.integers(len(groups[p][name])))]
            z = row["z"][None, None].to(self.device)
            risk = z.new_tensor([[row["risk"]]])
            k = int(rng.integers(5))
            cap, rel = head(z, risk, z.new_tensor([[k]]))
            loss = F.cross_entropy(
                cap.reshape(1, 5),
                torch.tensor([row["label"]["kstar"]], device=self.device),
            ) + F.binary_cross_entropy(
                rel.clamp(1e-6, 1 - 1e-6),
                rel.new_tensor([[row["label"]["reliability"][k]]]),
            )
            opt.zero_grad()
            loss.backward()
            schedule(opt, step, total)
            update(opt)
        d_report = self.choose_conservative_budget(external)
        self.profile = replace(
            self.profile, conservative_budget=d_report["d"], reliability_sort=False
        )
        self.system.profile = self.profile
        self.policies = {}
        with torch.no_grad():
            for role in ("fit", "val_tune", "val_select", "calibration"):
                for i in range(len(self.data[role])):
                    b = collate([self.data[role][i]], self.device)
                    g = self.system.decompose(b["x"])[2]
                    z = self.system.comm.carrier(g)
                    risk = self.risks[b["id"][0]]
                    cap, _ = head(z, risk)
                    k = (cap.argmax(-1) - self.profile.conservative_budget).clamp_min(
                        0
                    ) * b["legal"].long()
                    _, rel = head(z, risk, k)
                    self.policies[b["id"][0]] = (
                        k[0].cpu().numpy(),
                        rel[0].cpu().numpy(),
                    )
        from .routing_validation import choose_routing

        sorting = choose_routing(self)
        self.profile = replace(self.profile, reliability_sort=sorting["enabled"])
        self.system.profile = self.profile
        write_json(
            self.output / "calibration" / f"{self.snapshot_index}.json",
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
        self.bind_snapshot(f"G-{self.snapshot_index}")

    def choose_conservative_budget(self, entries):
        from .calibration import wilson

        for d in (0, 1, 2):
            by_patient = defaultdict(lambda: defaultdict(list))
            with torch.no_grad():
                for r in entries:
                    z = r["z"][None, None].to(self.device)
                    cap, _ = self.system.comm.heads(z, z.new_tensor([[r["risk"]]]))
                    k = max(0, int(cap.argmax(-1)) - d)
                    by_patient[r["patient"]][r["id"]].append(
                        int(k > r["label"]["kstar"])
                    )
            vals = np.array(
                [
                    np.mean([np.mean(v) for v in ims.values()])
                    for ims in by_patient.values()
                ]
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
            any_over = sum(
                any(any(v) for v in ims.values()) for ims in by_patient.values()
            )
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

    def joint(self, stage="H"):
        require(stage in ("H", "I"), "Joint stage")
        total = self.cfg["budgets"]["joint_a" if stage == "H" else "joint_b"]
        interval = 2000 if stage == "H" else 500
        if stage == "I":
            require(
                self.acceptance.get("H", {}).get("passed"),
                "Joint-B requires accepted Joint-A communication",
            )
        for p in self.system.inn.parameters():
            p.requires_grad_(stage == "I")
        for p in self.system.entropy.parameters():
            p.requires_grad_(stage == "I")
        modules = [
            self.system.comm.carrier,
            self.system.comm.writer,
            self.system.comm.decoder,
        ]
        rates = [5e-5 if stage == "H" else 1e-5] * 3
        if stage == "I":
            modules.extend([self.system.inn, self.system.entropy])
            rates.extend([1e-6, 1e-4])
        opt = optimizer(modules, rates)
        for p in self.pub_attacker.parameters():
            p.requires_grad_(True)
        advopt = optimizer([self.pub_attacker], [1e-4])
        for j in range(1000):
            self.pair_objective(stage, j, True, advopt)
        step = 0
        attempts = 0
        empty_streak = 0
        best = None
        while step < total and attempts < 100000:
            attempts += 1
            rng = rng_for(self.seed, stage, "images", attempts)
            chosen = rng.integers(len(self.fit), size=self.cfg.get("batch_images", 8))
            feasible = []
            for i in chosen:
                b = collate([self.fit[int(i)]], self.device)
                try:
                    with torch.no_grad():
                        credentials = self.allocator.new_object(self.system.digest)
                        sent = self.system.send(
                            b["x"],
                            b["metadata"][0],
                            b["legal"],
                            self.risks[b["id"][0]],
                            credentials,
                            self.allocator,
                            self.policies[b["id"][0]],
                            rng.permutation(self.profile.legal_levels).tolist(),
                        )
                    # Reuse this exact protected table for the gradient pass; no success filtering.
                    feasible.append((int(i), sent))
                except CCRRError as exc:
                    if exc.status != Status.CAPACITY:
                        raise
            if not feasible and stage == "H":
                empty_streak += 1
                require(
                    empty_streak < 100,
                    "100 capacity-rejected batches; joint stage stopped",
                )
                continue
            empty_streak = 0
            self.adversarial_updates(stage, step, advopt)
            opt.zero_grad(set_to_none=True)
            values = []
            for i, cached in feasible:
                b = collate([self.fit[i]], self.device)
                u, z, g, n, zq = self.system.quantized(b["x"], cached["level"])
                require(
                    torch.equal(n.detach(), cached["n"]),
                    "Latent changed within one optimizer update",
                )
                public, _ = self.system.comm.write(
                    g, cached["bits"], cached["occupancy"], b["legal"]
                )
                logits = self.system.comm.read(public)
                dec, _ = message_loss(logits, cached["bits"], cached["occupancy"])
                rec = self.system.recover(public, zq)
                target = self.supervision.targets(b["x"], b["rect"], b["label"])
                clin, ident, _ = self.supervision.loss(
                    public, b["rect"], b["label"], target
                )
                loss = (
                    dec
                    + masked_mean((public - g).abs() / (4 / 255), b["v"])
                    + reconstruction(b["x"], rec, b["v"])
                    + clin
                    + 0.1 * self.cfg["lambda_privacy"] * ident
                )
                (loss / len(feasible)).backward()
                values.append(float(loss.detach()))
            if stage == "I":
                for i in chosen:
                    b = collate([self.fit[int(i)]], self.device)
                    r = int(rng.integers(1, 4))
                    u, z, g, n, zq = self.system.quantized(b["x"], r)
                    target = self.supervision.targets(b["x"], b["rect"], b["label"])
                    clin, ident, _ = self.supervision.loss(
                        g, b["rect"], b["label"], target
                    )
                    loss = (
                        reconstruction(b["x"], self.system.recover(g, zq), b["v"])
                        + 0.1 * gray_loss(g, b["x"], b["v"])
                        + 1e-3 * saturation(u, b["v"])
                        + clin
                        + 0.1 * self.cfg["lambda_privacy"] * ident
                        + self.cfg["lambda_rate"]
                        * self.system.entropy.rate(z, r, self.profile.step(r)).mean()
                    )
                    (loss / len(chosen)).backward()
            self.pair_objective(stage, step)
            schedule(opt, step, total)
            norm = update(opt)
            step += 1
            if step % 100 == 0:
                self.log(
                    stage,
                    step,
                    {
                        "loss": np.mean(values) if values else None,
                        "feasible": len(feasible),
                        "sampled": len(chosen),
                        "gradient_norm": norm,
                        "attempts": attempts,
                    },
                )
            if step % interval == 0 or step == total:
                self.bind_snapshot(f"{stage}-{step}")
                self.stage_g()
                report = self.validate_deployment(stage, step)
                score = (report["both_rate"], -report["mean_bpp"], report["psnr"])
                if report["passed"] and (best is None or score > best[0]):
                    best = (score, report, copy.deepcopy(self.profile))
                    save_checkpoint(
                        self.output / "checkpoints" / f"{stage}.pt",
                        {"system": self.system},
                        opt,
                        step,
                        stage,
                        report,
                    )
        require(step == total, "Joint stage did not reach prescribed updates")
        if stage == "I":
            from .joint_b import accepts

            baseline = self.acceptance["H"]["report"]
            decision = (
                accepts(baseline, best[1])
                if best
                else {
                    "accepted": False,
                    "reason": "No Joint-B candidate passed base targets",
                }
            )
            write_json(self.output / "joint_b_comparison.json", decision)
            if not decision["accepted"]:
                load_checkpoint(
                    self.output / "checkpoints/H.pt", {"system": self.system}
                )
                self.profile = Profile.from_dict(baseline["profile"])
                self.system.profile = self.profile
                self.bind_snapshot("H-restored-after-I")
                self.acceptance["I"] = {
                    "passed": False,
                    "report": decision,
                    "action": "Restored accepted Joint-A",
                }
                write_json(self.output / "acceptance.json", self.acceptance)
                return
        self.mark(
            stage,
            best is not None,
            best[1] if best else {"reason": "No admissible deployment checkpoint"},
        )
        load_checkpoint(
            self.output / "checkpoints" / f"{stage}.pt", {"system": self.system}
        )
        self.profile = best[2]
        self.system.profile = self.profile
        self.bind_snapshot(f"{stage}-selected")
        self.stage_g(final=True)
        final_report = self.validate_deployment(stage + "-final", step)
        if stage == "I":
            from .joint_b import accepts

            decision = accepts(self.acceptance["H"]["report"], final_report)
            if not decision["accepted"]:
                baseline = self.acceptance["H"]["report"]
                load_checkpoint(
                    self.output / "checkpoints/H.pt", {"system": self.system}
                )
                self.profile = Profile.from_dict(baseline["profile"])
                self.system.profile = self.profile
                self.bind_snapshot("H-restored-after-final-I")
                self.acceptance["I"] = {
                    "passed": False,
                    "report": decision,
                    "action": "Restored Joint-A after final calibration",
                }
                write_json(self.output / "acceptance.json", self.acceptance)
                return
        self.mark(stage, final_report["passed"], final_report)
        save_checkpoint(
            self.output / "checkpoints" / f"{stage}.pt",
            {"system": self.system},
            None,
            step,
            stage,
            final_report,
        )
        self.bind_snapshot(f"{stage}-final-accepted")

    def validate_deployment(self, stage, step):
        from dataclasses import replace

        self.profile = replace(
            self.profile, legal_levels=(1, 2, 3), quality_order=(1, 2, 3)
        )
        self.system.profile = self.profile
        self.bind_snapshot(f"{stage}-{step}-levels")
        reports = {
            r: self._validate_deployment_once(f"{stage}-r{r}", step, (r,))
            for r in (1, 2, 3)
        }
        legal = tuple(r for r, report in reports.items() if report["passed"])
        if not legal:
            diagnostic = max(
                reports.values(), key=lambda r: (r["both_rate"], r["psnr"])
            )
            return {
                **diagnostic,
                "passed": False,
                "level_reports": reports,
                "reason": "No admissible deployment level",
            }
        actual = {}
        for r in legal:
            path = (
                self.output
                / "validation"
                / f"{stage}-r{r}-{step}-actual"
                / "system_results.json"
            )
            rows = json.loads(path.read_text())["rows"]
            actual[r] = {v["id"]: v for v in rows if v["c_success"]}
        common = set.intersection(*(set(v) for v in actual.values()))
        require(bool(common), "No common successful cohort for quality ordering")
        order = tuple(
            sorted(
                legal,
                key=lambda r: (
                    -np.mean([actual[r][i]["quality"]["psnr"] for i in common]),
                    -np.mean([actual[r][i]["quality"]["ssim"] for i in common]),
                    np.mean([actual[r][i]["c_bytes"] for i in common]),
                    r,
                ),
            )
        )
        self.profile = replace(self.profile, legal_levels=legal, quality_order=order)
        self.system.profile = self.profile
        self.bind_snapshot(f"{stage}-{step}-policy")
        report = self._validate_deployment_once(stage, step, order)
        report["level_reports"] = reports
        report["legal_levels"] = list(legal)
        report["quality_order"] = list(order)
        report["ordering_common_n"] = len(common)
        return report

    def _validate_deployment_once(self, stage, step, level_order=None):
        out = self.output / "validation" / f"{stage}-{step}-actual"
        result = evaluate_system(
            self.system,
            self.data["val_select"],
            self.device,
            self.allocator,
            out,
            lambda b: self.risks[b["id"][0]],
            lambda b: self.policies[b["id"][0]],
            level_order=level_order,
        )
        rows = result["rows"]
        pub = [r for r in rows if r["published"]]
        cs = [r for r in rows if r["c_success"]]
        both = sum(r["both_success"] for r in rows) / len(rows)
        coverage = len(pub) / len(rows)
        c_cond = len(cs) / len(pub) if pub else 0
        m_cond = sum(r["m_success"] for r in pub) / len(pub) if pub else 0
        psnr = float(np.mean([r["quality"]["psnr"] for r in cs])) if cs else 0
        ssim = float(np.mean([r["quality"]["ssim"] for r in cs])) if cs else 0
        # Generator selection uses only the development clinical teacher / attacks.
        from .system import load_public

        def public_view(b):
            p = out / b["id"][0]
            return (
                load_public(p, self.profile.size).to(self.device)
                if p.exists()
                else None
            )

        clin = (
            evaluate_classifier(
                self.clinical, self.data["val_select"], self.device, public_view, True
            )
            if pub
            else None
        )
        identity = {}
        privacy_checks = {}
        from .comparisons import paired_clinical, privacy_points

        public_ids = {r["id"] for r in pub}
        reference = evaluate_classifier(
            self.clinical, self.data["val_select"], self.device, None, False
        )
        clinical_check = (
            paired_clinical(reference, clin) if clin else {"passed_point": False}
        )
        if pub:
            for scenario in ("p1", "p2"):
                for direction in ("original", "same"):
                    q = public_view
                    gallery = public_view if direction == "same" else None
                    common_gallery = public_ids if direction == "same" else None
                    current = evaluate_identity(
                        self.pub_attacker,
                        self.data["val_select"],
                        self.device,
                        scenario,
                        q,
                        gallery,
                        True,
                        public_ids,
                        common_gallery,
                    )
                    rgb = evaluate_identity(
                        self.pub_attacker,
                        self.data["val_select"],
                        self.device,
                        scenario,
                        None,
                        None,
                        True,
                        public_ids,
                        common_gallery,
                    )
                    gy = lambda b: gray(b["x"])
                    standard = evaluate_identity(
                        self.pub_attacker,
                        self.data["val_select"],
                        self.device,
                        scenario,
                        gy,
                        gy if direction == "same" else None,
                        True,
                        public_ids,
                        common_gallery,
                    )
                    key = scenario + "-" + direction
                    identity[key] = current
                    privacy_checks[key] = privacy_points(rgb, standard, current)
        report = {
            "stage": stage,
            "step": step,
            "actual_results_path": str(out / "system_results.json"),
            "profile": self.profile.__dict__,
            "both_rate": both,
            "coverage": coverage,
            "c_conditional": c_cond,
            "m_conditional": m_cond,
            "psnr": psnr,
            "ssim": ssim,
            "mean_bpp": float(np.mean([r["protected_bpp"] for r in pub]))
            if pub
            else float("inf"),
            "clinical": clin,
            "clinical_check": clinical_check,
            "identity": identity,
            "privacy_checks": privacy_checks,
            "system": result["funnel"],
            "class_coverage": {
                label: {
                    "publication": np.mean(
                        [r["published"] for r in rows if r["diagnostic"] == label]
                    ),
                    "both": np.mean(
                        [r["both_success"] for r in rows if r["diagnostic"] == label]
                    ),
                }
                for label in CLASSES
            },
            "both_conditional": sum(r["both_success"] for r in pub) / len(pub)
            if pub
            else 0,
            "c_all": len(cs) / len(rows),
            "m_all": sum(r["m_success"] for r in rows) / len(rows),
            "fraction_psnr28_ssim90": np.mean(
                [
                    r["quality"]["psnr"] >= 28 and r["quality"]["ssim"] >= 0.90
                    for r in cs
                ]
            )
            if cs
            else 0,
        }
        from .source_privacy import evaluate_source_privacy

        report["source_privacy"] = evaluate_source_privacy(
            self.data["val_select"],
            self.device,
            public_view,
            {"reid": self.reid, "adaptive": self.pub_attacker},
        )
        # Scientific utility/privacy acceptance is computed by evidence rules, not training loss.
        from .gates import deployment_gate

        report.update(deployment_gate(report, self.acceptance))
        write_json(out / "selection.json", report)
        return report

    def final_evaluation(self):
        require(
            self.acceptance.get("H", {}).get("passed"),
            "Final evaluation requires accepted frozen generator",
        )
        self.data["test"] = PADData(self.cfg["data_root"], "test", self.profile.size)
        from .final_study import FinalStudy

        study = FinalStudy(self)
        return study.run()

    def run(self, stage="all"):
        try:
            if stage == "all":
                self.stage_a()
                self.stage_b()
                if not self.system.fixed_gray:
                    self.train_representation("C")
                    self.train_representation("D")
                self.train_representation("E")
                self.stage_f()
                self.stage_g()
                self.joint("H")
                if self.cfg.get("joint_b"):
                    self.joint("I")
                return self.final_evaluation()
            require(
                stage in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"),
                "Unknown stage",
            )
            if stage != "A":
                self.restore_dependencies(stage)
            return {
                "A": self.stage_a,
                "B": self.stage_b,
                "C": lambda: self.train_representation("C"),
                "D": lambda: self.train_representation("D"),
                "E": lambda: self.train_representation("E"),
                "F": self.stage_f,
                "G": self.stage_g,
                "H": lambda: self.joint("H"),
                "I": lambda: self.joint("I"),
                "J": self.final_evaluation,
            }[stage]()
        except Exception as exc:
            write_json(
                self.output / "stopped.json",
                {
                    "stage": stage,
                    "type": type(exc).__name__,
                    "reason": str(exc),
                    "training_complete": False,
                },
            )
            raise
        finally:
            self.allocator.close()

    def restore_dependencies(self, stage):
        self.acceptance = json.loads((self.output / "acceptance.json").read_text())
        required = {
            "B": ["A-clinical", "A-mfm", "A-reid"],
            "C": [],
            "D": ["A-clinical", "A-mfm", "A-reid", "B", "C"],
            "E": ["D"],
            "F": ["E"],
            "G": ["E"],
            "H": ["E"],
            "I": ["H"],
            "J": ["H"],
        }[stage]
        for dep in required:
            require(
                self.acceptance.get(dep, {}).get("passed"),
                f"Missing accepted dependency {dep}",
            )
        if stage != "C":
            load_checkpoint(
                self.output / "checkpoints/clinical.pt", {"model": self.clinical}
            )
            freeze(self.clinical)
            load_checkpoint(self.output / "checkpoints/reid.pt", {"model": self.reid})
            freeze(self.reid)
            e = self.cfg["weights"]["panderm"]
            self.mfm = PanDerm(
                e["source"], e["source_sha256"], e["path"], e["sha256"]
            ).to(self.device)
            if stage == "B":
                load_checkpoint(
                    self.output / "checkpoints/mfm_probe.pt",
                    {"purifier": self.purifier},
                )
            else:
                load_checkpoint(
                    self.output / "checkpoints/purifier.pt", {"purifier": self.purifier}
                )
                freeze(self.purifier)
                tau = json.loads((self.output / "identity_threshold.json").read_text())[
                    "tau"
                ]
                self.supervision = TeacherSupervision(
                    self.clinical, self.mfm, self.reid, self.purifier, self.weights, tau
                )
            self.initialize_base(self.pub_attacker, "resnet50")
        previous = {
            "D": "C",
            "E": "D",
            "F": "E",
            "G": "F",
            "H": "F",
            "I": "H",
            "J": "H",
        }.get(stage)
        if previous:
            load_checkpoint(
                self.output / "checkpoints" / f"{previous}.pt", {"system": self.system}
            )
            self.bind_snapshot("resume-" + stage)
        if stage in ("H", "I", "J"):
            self.stage_g(final=stage == "J")


def CounterLocal(values):
    from collections import Counter

    return Counter(values)
