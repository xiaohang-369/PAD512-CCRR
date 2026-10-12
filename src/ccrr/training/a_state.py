"""Transactional A-substage progress, separate from the selected frozen model."""

import hashlib
import math
import os
from pathlib import Path
import torch
from ccrr.config import canonical_json, sha256
from ccrr.errors import require
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.runtime.support import environment


def binding(config):
    # Paths may move between hosts; content, protocol and execution environment may not.
    files = Path(__file__).parents[1]
    return hashlib.sha256(
        canonical_json(
            {
                "seed": config["seed"],
                "budgets": config["budgets"],
                "profile": config["profile"],
                "batch_teacher": config.get("batch_teacher", 16),
                "pixel_cache_bytes": config.get("pixel_cache_bytes", 0),
                "device": config["device"],
                "environment": environment(),
                "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "weights": {
                    n: {k: v for k, v in e.items() if k not in ("path", "source")}
                    for n, e in config["weights"].items()
                    if n in ("efficientnet_b0", "resnet50", "panderm")
                },
                "sources": {
                    str(p.relative_to(files)): sha256(p)
                    for p in sorted(files.rglob("*.py"))
                },
                "data": sha256(files / "data/frozen_evidence.json"),
            }
        )
    ).hexdigest()


class AProgress:
    def __init__(self, experiment, name, modules, opt, scheduler):
        self.e, self.name, self.modules, self.opt, self.scheduler = (
            experiment,
            name,
            modules,
            opt,
            scheduler,
        )
        self.path = experiment.output / "checkpoints" / (name + ".last.pt")
        self.binding = binding(experiment.cfg)
        self.selection = {"score": None, "step": None, "models": None, "reports": None}
        self.audit = {"fallbacks": 0}
        if self.path.exists():
            require(
                experiment.cfg.get("resume_a", False),
                "Existing A progress: use --resume or a new output directory",
            )
            # Validate before mutating models/optimizer/RNG; v1 cannot resume a whole A loop.
            raw = torch.load(self.path, map_location="cpu", weights_only=True)
            state = raw.get("training_state")
            require(
                raw["format"] == "CCRR-CHECKPOINT-2"
                and raw["stage"] == name
                and state
                and state["binding"] == self.binding,
                "A resume protocol/content/environment mismatch",
            )
            require(
                len(raw["cuda_rng"])
                == (torch.cuda.device_count() if torch.cuda.is_available() else 0),
                "A resume CUDA topology mismatch",
            )
            scheduler.load_state_dict(state["scheduler"])
            require(
                raw["step"] == scheduler.next_step, "A resume step/scheduler mismatch"
            )
            load_checkpoint(self.path, modules, opt)
            self.selection, self.audit = state["selection"], state["audit"]

    @property
    def start(self):
        return self.scheduler.next_step

    def select(self, score, reports):
        require(score is None or math.isfinite(score), "Nonfinite A selection metric")
        if score is not None and (
            self.selection["score"] is None or score > self.selection["score"]
        ):
            self.selection = {
                "score": float(score),
                "step": self.start,
                "models": None,
                "reports": reports,
            }
            # CPU copies without moving the active model or its optimizer tensors.
            self.selection["models"] = {
                k: {n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
                for k, m in self.modules.items()
            }

    def save(self, endpoint=False):
        state = {
            "binding": self.binding,
            "scheduler": self.scheduler.state_dict(),
            "selection": self.selection,
            "audit": self.audit,
        }
        if endpoint:
            path = self.path.parent / "endpoints" / f"{self.name}-{self.start}.pt"
            save_checkpoint(
                path,
                self.modules,
                self.opt,
                self.start,
                self.name,
                {},
                training_state=state,
            )
        # Commit progress only after the mandatory endpoint is durable. If the
        # process dies between the writes, replay the previous durable interval.
        save_checkpoint(
            self.path,
            self.modules,
            self.opt,
            self.start,
            self.name,
            {},
            training_state=state,
        )

    def finish(self, path, stage=None):
        require(
            self.selection["models"] is not None,
            f"{self.name}: no evaluable checkpoint",
        )
        for k, m in self.modules.items():
            m.load_state_dict(self.selection["models"][k], strict=True)
        save_checkpoint(
            path,
            self.modules,
            None,
            self.selection["step"],
            stage or self.name,
            {
                "score": self.selection["score"],
                "budget_completed": self.start,
                "binding": self.binding,
                "selected_only": True,
            },
        )
        return self.selection["score"], self.selection["reports"]


def validate_a_output(config):
    """Refuse destructive restarts/mismatched resumes before Experiment writes evidence."""
    folder = Path(config["output"]) / "checkpoints"
    last = [
        folder / (n + ".last.pt")
        for n in ("clinical", "clinical_rgb", "clinical_gray", "mfm_probe", "reid")
    ]
    existing = [p for p in last if p.exists()]
    selected = [
        folder / (n + ".pt")
        for n in ("clinical", "clinical_rgb", "clinical_gray", "mfm_probe", "reid")
    ]
    if not config.get("resume_a"):
        require(
            not existing and not any(p.exists() for p in selected),
            "Existing A artifacts: use --resume or a new output directory",
        )
        return
    require(
        bool(existing),
        "No durable A progress available; legacy selected checkpoints cannot resume A",
    )
    expected = binding(config)
    furthest = max(i for i, p in enumerate(last) if p.exists())
    require(
        all(p.exists() for p in last[: furthest + 1]),
        "Missing preceding A progress; cannot reconstruct stage order",
    )
    budgets = [config["budgets"]["clinical"]] * 3 + [
        config["budgets"]["mfm_probe"],
        config["budgets"]["reid"],
    ]
    for i, p in enumerate(existing):
        obj = torch.load(p, map_location="cpu", weights_only=True)
        require(
            obj["stage"] == p.name.removesuffix(".last.pt"),
            "A checkpoint stage/path mismatch",
        )
        require(
            0 <= obj["step"] <= budgets[i]
            and (i == furthest or obj["step"] == budgets[i]),
            "A substage order/budget mismatch",
        )
        require(
            obj.get("format") == "CCRR-CHECKPOINT-2"
            and obj.get("training_state", {}).get("binding") == expected,
            "A resume binding mismatch before experiment initialization",
        )
