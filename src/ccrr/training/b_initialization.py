"""Explicit opt-in to the registered PUR-INIT candidate; never scale a resume."""

import copy
import json

import torch

from ccrr.config import sha256
from ccrr.errors import require
from ccrr.runtime.checkpoints import save_checkpoint
from ccrr.runtime.support import write_json

LEGACY = "original"
CANDIDATE = "PUR-INIT-v1.2-rc1"


def version(cfg):
    value = cfg.get("b_initialization", LEGACY)
    require(value in (LEGACY, CANDIDATE), "Unknown Stage B initialization")
    return value


def after_frozen_a_restore(e):
    """Called immediately after loading A's mfm_probe, before creating B AdamW."""
    if version(e.cfg) == LEGACY or e.cfg.get("resume_b", False):
        return
    require(
        not getattr(e, "b_initialization_applied", False),
        "Refuse repeated initialization scaling",
    )
    with torch.no_grad():
        e.purifier.encoder.end.weight.mul_(0.1)
        e.purifier.decoder.end.weight.mul_(0.1)
    e.b_initialization_applied = True
    e.b_control_initial = {
        part: {
            k: v.detach().cpu().clone()
            for k, v in getattr(e.purifier, part).state_dict().items()
        }
        for part in ("encoder", "decoder")
    }


def record_start(e):
    """Preserve the untrained candidate and its content binding in the new run."""
    if version(e.cfg) == LEGACY:
        return
    from ccrr.training.b_state import binding

    path = e.output / "b_initialization.pt"
    record = e.output / "b_initialization.json"
    contract = {
        "version": CANDIDATE,
        "binding": binding(e.cfg),
        "stage_a_freeze_sha256": e.cfg["stage_a_source"]["freeze_sha256"],
        "fresh_optimizer_step": 0,
        "scaled_parameters": ["encoder.end.weight", "decoder.end.weight"],
        "scale": 0.1,
        "r3_warm_start": False,
    }
    if e.cfg.get("resume_b", False):
        saved = json.loads(record.read_text())
        require(
            saved["contract"] == contract and sha256(path) == saved["sha256"],
            "Initialization record changed on resume",
        )
        return
    require(
        getattr(e, "b_initialization_applied", False),
        "Candidate was not initialized from frozen A",
    )
    require(
        not path.exists() and not record.exists(),
        "Existing initialization must not be overwritten",
    )
    save_checkpoint(
        path,
        {"purifier": e.purifier, "attacker": e.sem_attacker},
        None,
        0,
        "B-initialization",
        contract,
    )
    write_json(record, {"contract": contract, "sha256": sha256(path)})


def initialize_control(e, model):
    """Use exactly the candidate's untrained Es/Ds, never trained Purifier weights."""
    if version(e.cfg) == LEGACY:
        return
    initial = getattr(e, "b_control_initial", None)
    if initial is None:
        # Resume: recover immutable step-zero weights, not the current B weights.
        record_start(e)
        obj = torch.load(
            e.output / "b_initialization.pt", map_location="cpu", weights_only=True
        )
        weights = obj["models"]["purifier"]
        initial = {
            part: {
                k[len(part) + 1 :]: v
                for k, v in weights.items()
                if k.startswith(part + ".")
            }
            for part in ("encoder", "decoder")
        }
    for part in ("encoder", "decoder"):
        getattr(model, part).load_state_dict(copy.deepcopy(initial[part]))


def image_rng_stage(cfg):
    # Candidate control matches the Purifier image stream; legacy runs unchanged.
    return "B" if version(cfg) == CANDIDATE else "B-control"
