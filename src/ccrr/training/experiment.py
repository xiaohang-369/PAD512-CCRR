"""Shared experiment resources, evidence and snapshot bookkeeping."""

from pathlib import Path
import hashlib
import numpy as np
import torch
from ccrr.config import Profile, canonical_json, sha256
from ccrr.data.pad import PADData, PairSampler, collate
from ccrr.data.cache import PixelCache, cache_budget
from ccrr.teachers.adapters import (
    ClinicalModel,
    IdentityModel,
)
from ccrr.representation.purifier import Purifier, SemanticAttacker
from ccrr.runtime.system import CCRR
from ccrr.codec.crypto import NonceAllocator
from ccrr.codec.metadata import CLASSES
from ccrr.runtime.support import seed_everything, environment, write_json, log_json
from ccrr.errors import require


class Experiment:
    def __init__(self, config):
        self.cfg = config
        self.seed = int(config["seed"])
        seed_everything(self.seed)
        self.device = torch.device(config["device"])
        self.output = Path(config["output"])
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(
            self.output.resolve() != Path(__file__).parents[3].resolve(),
            "Outputs belong outside source",
        )
        if not config.get("resume_a"):
            write_json(self.output / "environment.json", environment())
        require(
            self.device.type != "cuda" or torch.cuda.is_available(),
            "CUDA unavailable on this host; use CPU for engineering checks and the H100 for the formal run",
        )
        self.profile = Profile.from_dict(config.get("profile", {}))
        self.data = {}
        self.pixel_cache = PixelCache(cache_budget(config))
        for role in ("fit", "calibration", "val_tune", "val_select"):
            self.data[role] = self.pixel_cache.wrap(
                PADData(config["data_root"], role, self.profile.size)
            )
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
        if not config.get("resume_a"):
            write_json(self.output / "config.json", config)
        if not config.get("resume_a"):
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

    def preflight(self, stage="all"):
        from ccrr.runtime.preflight import inspect

        if getattr(self, "_preflight_stage", None) in (stage, "all"):
            return
        report = inspect(self.cfg, stage)
        report["ready"] = report["ready_for_stage"]
        report["training_started"] = False
        write_json(self.output / "preflight.json", report)
        require(report["ready"], f"Preflight incomplete: {report['missing']}")

    def batch(self, rng, n=1):
        return collate(
            [self.fit[int(i)] for i in rng.integers(len(self.fit), size=n)], self.device
        )

    def class_weight_mean(self, indices):
        """Full effective cohort, including repeated draws; no image decoding."""
        labels = torch.tensor(
            [CLASSES.index(self.fit.rows[int(i)]["diagnostic"]) for i in indices],
            device=self.weights.device,
        )
        require(labels.numel() > 0, "Empty clinical effective batch")
        return self.weights[labels].mean()

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

    def run(self, stage="all"):
        """Execute the registered stages using this experiment context."""
        from ccrr.training.runner import run

        return run(self, stage)
