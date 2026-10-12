"""Atomic B optimizer-boundary and endpoint-audit recovery."""

from contextlib import contextmanager
import fcntl
import hashlib
from pathlib import Path
import torch

from ccrr.config import canonical_json
from ccrr.errors import require
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.runtime.support import json_safe
from ccrr.training.a_state import binding as common_binding
from ccrr.training.b_resources import verify_teachers, execution_options


def binding(cfg):
    return hashlib.sha256(
        canonical_json(
            {
                "common": common_binding(cfg),
                "freeze_sha256": cfg["stage_a_source"]["freeze_sha256"],
                "batch_pairs": cfg.get("batch_pairs", 16),
                "b_cache_device": cfg.get("b_cache_device", "cpu"),
                "b_cache_bytes": cfg.get("b_cache_bytes", 4 * 1024**3),
                "b_audit_cache_bytes": cfg.get("b_audit_cache_bytes", 8 * 1024**3),
                "cpu_threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "diagnostic_only": cfg.get("b_diagnostic", False),
            }
        )
    ).hexdigest()


def validate_output(cfg):
    require(
        not cfg.get("b_diagnostic"), "Diagnostic B config cannot launch formal training"
    )
    execution_options(cfg)
    verify_teachers(cfg)
    out = Path(cfg["output"])
    path = out / "checkpoints/b.last.pt"
    if cfg.get("resume_b"):
        require(path.is_file(), "B resume requires checkpoints/b.last.pt")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        require(
            raw.get("format") == "CCRR-CHECKPOINT-2"
            and raw.get("stage") == "B"
            and (raw.get("training_state") or {}).get("binding") == binding(cfg),
            "B resume source/config/environment/teacher binding mismatch",
        )
    else:
        require(
            not (out / "checkpoints").exists()
            and not (out / "training.jsonl").exists()
            and not (out / "config.json").exists()
            and not (out / "acceptance.json").exists(),
            "Existing B artifacts: use --resume or a new output directory",
        )


@contextmanager
def output_lock(cfg):
    # Validate the A/B separation before creating anything inside the output.
    verify_teachers(cfg)
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (out / ".stage-b.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another Stage B process holds the output lock") from None
        try:
            validate_output(cfg)
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class BProgress:
    def __init__(self, e, opt, advopt, scheduler, advscheduler):
        self.e, self.opt, self.advopt = e, opt, advopt
        self.scheduler, self.advscheduler = scheduler, advscheduler
        self.modules = {"purifier": e.purifier, "attacker": e.sem_attacker}
        self.path = e.output / "checkpoints/b.last.pt"
        self.binding = binding(e.cfg)
        self.phase = "train"
        self.audit_done = []
        self.best = None
        if self.path.exists():
            require(e.cfg.get("resume_b"), "Existing B progress requires --resume")
            raw = torch.load(self.path, map_location="cpu", weights_only=True)
            state = raw.get("training_state") or {}
            require(
                raw["stage"] == "B" and state.get("binding") == self.binding,
                "B resume binding mismatch",
            )
            require(
                len(raw["cuda_rng"])
                == (torch.cuda.device_count() if torch.cuda.is_available() else 0),
                "B resume CUDA topology mismatch",
            )
            scheduler.load_state_dict(state["scheduler"])
            advscheduler.load_state_dict(state["adversary_scheduler"])
            require(
                raw["step"] == scheduler.next_step
                and advscheduler.next_step == 2 * scheduler.next_step,
                "B resume optimizer counters mismatch",
            )
            load_checkpoint(self.path, self.modules, opt)
            advopt.load_state_dict(state["adversary_optimizer"])
            self.phase, self.audit_done, self.best = (
                state["phase"],
                state["audit_done"],
                state["best"],
            )
            require(self.phase in ("train", "audit"), "Unknown B recovery phase")
            require(
                self.audit_done == sorted(set(self.audit_done))
                and all(
                    0 < step <= scheduler.next_step and step % 1000 == 0
                    for step in self.audit_done
                ),
                "Invalid B audit progress",
            )
            require(
                self.phase != "audit" or scheduler.next_step == scheduler.total,
                "B audit cannot precede completed training budget",
            )

    def save(self, endpoint=False):
        state = {
            "binding": self.binding,
            "scheduler": self.scheduler.state_dict(),
            "adversary_scheduler": self.advscheduler.state_dict(),
            "adversary_optimizer": self.advopt.state_dict(),
            "phase": self.phase,
            "audit_done": self.audit_done,
            "best": json_safe(self.best),
        }
        if endpoint:
            save_checkpoint(
                self.path.parent / f"purifier-{self.scheduler.next_step}.pt",
                self.modules,
                self.opt,
                self.scheduler.next_step,
                "B",
                {},
                training_state=state,
            )
        save_checkpoint(
            self.path,
            self.modules,
            self.opt,
            self.scheduler.next_step,
            "B",
            {},
            training_state=state,
        )

    def check_endpoints(self):
        # Committed endpoints must survive; never silently select a subset.
        for step in range(1000, self.scheduler.next_step + 1, 1000):
            path = self.path.parent / f"purifier-{step}.pt"
            require(path.is_file(), "Missing mandatory B endpoint " + str(step))
            raw = torch.load(path, map_location="cpu", weights_only=True)
            require(
                raw["step"] == step
                and raw["training_state"]["binding"] == self.binding,
                "B endpoint binding mismatch",
            )
