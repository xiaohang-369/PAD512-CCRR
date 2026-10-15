"""Registered no-branch reconstruction control, approved 2026-10-13.

Independent Es -> 15-level code -> Ds; only full reconstruction MSE,
10,000 updates, LR 1e-4, batch 16, equal RGB/Gray, val_select every 1,000.
This is not a trained Purifier with its mask forced to zero.
"""

import math
import torch
from torch import nn

from ccrr.config import sha256
from ccrr.errors import require
from ccrr.representation.purifier import MaskedNet
from ccrr.runtime.checkpoints import save_checkpoint, load_checkpoint
from ccrr.runtime.support import seed_everything, write_json
from ccrr.tensor_ops import init_comm, ste_round, masked_mean
from ccrr.data.pad import rng_for
from ccrr.training.b_batches import TeacherBank
from ccrr.training.b_diagnostics import isolated_rng
from ccrr.training.b_state import binding
from ccrr.training.optimization import optimizer, TeacherSchedule, update
from ccrr.training.teacher_batches import chunks


CONTROL_PROTOCOL = "B-no-branch-reconstruction-v1"


class ReconstructionControl(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.encoder = MaskedNet(768, 64)
        self.decoder = MaskedNet(64, 768)
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())
        self.apply(init_comm)

    def forward(self, features, v):
        mask = (v > 0).to(features)
        target = (features - self.mean) / self.std.clamp_min(1e-3) * mask
        q = ste_round(7 * self.encoder(target, v).tanh()) * mask
        return {"standardized": target, "q": q, "all": self.decoder(q / 7, v)}


def control_update(model, bank, ids, microbatch, opt, scheduler):
    require(len(ids) == 16, "Control effective batch must remain 16")
    model.train()
    opt.zero_grad(set_to_none=True)
    loss_value = 0.0
    for group in chunks(list(ids), microbatch):
        t = bank.get(group)
        for key in ("f0", "f1"):
            out = model(t[key], t["v"])
            loss = (
                masked_mean((out["all"] - out["standardized"]).square(), t["v"])
                * len(group)
                / 32
            )
            loss.backward()
            loss_value += float(loss.detach())
    scheduler.apply(scheduler.next_step)
    update(opt)
    scheduler.advance()
    return loss_value


@torch.no_grad()
def reconstruction_score(model, bank, count, microbatch):
    model.eval()
    loss = 0.0
    for ids in chunks(list(range(count)), microbatch):
        t = bank.get(ids)
        for key in ("f0", "f1"):
            p = model(t[key], t["v"])
            loss += float(
                masked_mean((p["all"] - p["standardized"]).square(), t["v"])
            ) * len(ids)
    return loss / (2 * count)


def prepare_control(e, bank):
    """Own atomic checkpoints; a restart does not redo a completed control."""
    root = e.output / "no_branch_control"
    root.mkdir(parents=True, exist_ok=True)
    last, selected = root / "last.pt", root / "selected.pt"
    bound = binding(e.cfg)
    with isolated_rng():
        seed_everything(int(rng_for(e.seed, "B", CONTROL_PROTOCOL, 0).integers(2**31)))
        model = ReconstructionControl(e.purifier.mean, e.purifier.std).to(e.device)
        from ccrr.training.b_initialization import initialize_control, image_rng_stage

        initialize_control(e, model)
        opt = optimizer([model], [1e-4])
        scheduler = TeacherSchedule(opt, 10000)
        best = None
        if last.exists():
            saved = torch.load(last, map_location="cpu", weights_only=True)
            state = saved.get("training_state") or {}
            require(
                saved["stage"] == "B-control"
                and state.get("binding") == bound
                and state.get("protocol") == CONTROL_PROTOCOL,
                "Control resume binding mismatch",
            )
            scheduler.load_state_dict(state["scheduler"])
            require(saved["step"] == scheduler.next_step, "Control step mismatch")
            load_checkpoint(last, {"control": model}, opt)
            best = state["best"]
            for committed in range(1000, scheduler.next_step + 1, 1000):
                require(
                    (root / f"step-{committed}.pt").is_file(),
                    "Missing mandatory control endpoint",
                )
        val_bank = None
        if scheduler.next_step < 10000:
            val_bank = TeacherBank(e, "val_select", identity=False)
        print(f"B_CONTROL_START: {scheduler.next_step}/10000", flush=True)
        for step in range(scheduler.next_step, 10000):
            rng = rng_for(e.seed, image_rng_stage(e.cfg), "images", step)
            ids = rng.integers(len(e.fit), size=16).tolist()
            loss = control_update(
                model, bank, ids, e.cfg.get("microbatch", 1), opt, scheduler
            )
            if (step + 1) % 100 == 0:
                print(f"B_CONTROL_STEP: {step + 1}/10000 loss={loss:.6f}", flush=True)
            endpoint = (step + 1) % 1000 == 0
            if endpoint:
                score = reconstruction_score(
                    model,
                    val_bank,
                    len(e.data["val_select"]),
                    e.cfg.get("microbatch", 1),
                )
                require(math.isfinite(score), "Nonfinite control validation loss")
                if best is None or score < best["loss"]:
                    best = {"step": step + 1, "loss": score}
                # Preserve every eligible endpoint. Select by metadata after budget completion.
                save_checkpoint(
                    root / f"step-{step + 1}.pt",
                    {"control": model},
                    None,
                    step + 1,
                    "B-control",
                    {"validation_loss": score, "binding": bound},
                )
                write_json(
                    root / f"validation-{step + 1}.json",
                    {"step": step + 1, "loss": score},
                )
            if endpoint or (step + 1) % e.cfg.get("resume_interval", 100) == 0:
                save_checkpoint(
                    last,
                    {"control": model},
                    opt,
                    step + 1,
                    "B-control",
                    {},
                    training_state={
                        "binding": bound,
                        "protocol": CONTROL_PROTOCOL,
                        "scheduler": scheduler.state_dict(),
                        "best": best,
                    },
                )
        require(best is not None, "Control selection missing")
        load_checkpoint(root / f"step-{best['step']}.pt", {"control": model})
        if not selected.exists():
            save_checkpoint(
                selected,
                {"control": model},
                None,
                best["step"],
                "B-control",
                {
                    "protocol": CONTROL_PROTOCOL,
                    "budget_completed": 10000,
                    "selection": best,
                    "binding": bound,
                },
            )
        else:
            existing = torch.load(selected, map_location="cpu", weights_only=True)
            require(
                existing["extra"].get("binding") == bound
                and existing["step"] == best["step"]
                and all(
                    torch.equal(v.cpu(), existing["models"]["control"][k])
                    for k, v in model.state_dict().items()
                ),
                "Selected control mismatch",
            )
        write_json(
            root / "record.json",
            {
                "protocol": CONTROL_PROTOCOL,
                "budget_completed": 10000,
                "selection": best,
                "sha256": sha256(selected),
                "initialization": e.cfg.get("b_initialization", "original"),
                "binding": bound,
            },
        )
        e.no_branch_control = model.eval().requires_grad_(False)
        e.no_branch_control_sha256 = sha256(selected)
