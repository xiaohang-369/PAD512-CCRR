"""R5 B/C1 engine: direct batch16, exact cycle snapshots, no automatic launch."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from PIL import Image

from ccrr.data.pad import collate
from ccrr.evaluation.metrics import clinical
from ccrr.tensor_ops import gray, masked_mean, ssim_values
from .grayscale import GrayscaleSystem, public_forward, update
from .grayscale_dev import CONFIG as PILOT_CONFIG, state_digest, teacher_targets
from .runtime import seed_all

CONFIG = {
    "schema": "ccrr-grayscale-r5-direct16-a0-2",
    "scope": "development_only",
    "seed": 20261020,
    "batch_size": 16,
    "microbatch_size": 16,
    "accumulation_steps": 1,
    "baseline_updates": 2000,
    "c1_cycles": 10000,
    "checkpoint_every": 250,
    "diagnostic_batch_size": 16,
    "optimizer": "AdamW",
    "ed_lr": 1e-4,
    "p_lr": 1e-5,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 1e-4,
    "decay_bias": False,
    "warmup_updates": 100,
    "lr_floor_fraction": 0.1,
    "schedule": "phase_warmup_cosine_preserve_adam_moments",
    "precision_bits": 8,
    "fp32": True,
    "cudnn_allow_tf32": True,
    "matmul_allow_tf32": False,
    "numeric_policy": "a0_compatible_float32_tensors_cudnn_tf32_allowed_matmul_tf32_disabled",
    "padding": "sender_invalid_constant_128_over_255",
    "precision_initialization": "copy_8bit_condition_to_4bit",
    "weights": deepcopy(PILOT_CONFIG["weights"]),
    "primary_supervisor": 8500,
    "comparison_supervisor": 7000,
    "sampling": "uniform_images_with_replacement_same_batch_for_ed_and_p",
    "variants": ["learned", "fixed_gray"],
    "no_automatic_winner": True,
    "test_images_used": False,
    "independent_pixel_evaluation": "deferred_separate_protocol",
}


def configure_numerics():
    """A0-compatible process policy for verification, targets, training and diagnostics.

    Float32 tensor storage does not imply that every cuDNN kernel disables TF32.
    The returned server reproduction has exact A0 equality at batch16 with this
    policy. No autocast or low precision parameter/optimizer storage is used.
    """
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rng_state():
    ns = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=[ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("CUDA RNG topology changed")
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])


def lr_factor(cycle, total, cfg):
    if not 1 <= cycle <= total:
        raise ValueError("Schedule position outside phase")
    warm = min(cfg["warmup_updates"], total)
    floor = cfg["lr_floor_fraction"]
    if cycle <= warm:
        return floor + (1 - floor) * (cycle - 1) / max(warm - 1, 1)
    return (
        floor
        + (1 - floor) * (1 + math.cos(math.pi * (cycle - warm) / (total - warm))) / 2
    )


def optimizer(modules, lr, cfg):
    decay, bias = [], []
    for module in modules:
        for name, p in module.named_parameters():
            (bias if name.endswith("bias") else decay).append(p)
    return torch.optim.AdamW(
        [
            dict(params=decay, weight_decay=cfg["weight_decay"]),
            dict(params=bias, weight_decay=0.0),
        ],
        lr=lr,
        betas=tuple(cfg["betas"]),
        eps=cfg["eps"],
    )


def copy_precision_condition(decoder):
    """At an explicitly registered D1 boundary, clone learned 8bit into 4bit."""
    with torch.no_grad():
        decoder.precision.weight[:, 0].copy_(decoder.precision.weight[:, 1])


def pack_symbols(q, bits):
    if bits not in (4, 8) or q.dtype != torch.uint8 or q.numel() != 8192:
        raise ValueError("Require 8192 uint8 symbols in a registered precision")
    flat = q.detach().cpu().contiguous().view(-1)
    if int(flat.max()) >= 2**bits:
        raise ValueError("Symbol outside alphabet")
    if bits == 4:
        flat = (flat[::2] << 4) | flat[1::2]
    return bytes(flat.tolist())


def unpack_symbols(body, bits, device="cpu"):
    if bits not in (4, 8) or len(body) != 8192 * bits // 8:
        raise ValueError("Noncanonical private body length")
    b = torch.tensor(list(body), dtype=torch.uint8, device=device)
    if bits == 4:
        b = torch.stack((b >> 4, b & 15), 1).flatten()
    return b.reshape(1, 8, 32, 32)


class Engine:
    """Only complete cycles are serializable. Dataset and frozen teacher are bound."""

    def __init__(
        self, teacher, fit, supervisor_step, device, variant="learned", cfg=None
    ):
        self.cfg = deepcopy(CONFIG if cfg is None else cfg)
        if (
            self.cfg["batch_size"] != 16
            or self.cfg["accumulation_steps"] != 1
            or self.cfg["microbatch_size"] != 16
        ):
            raise ValueError("R5 uses a direct full batch, never accumulation")
        if self.cfg["weights"] != PILOT_CONFIG["weights"]:
            raise ValueError(
                "Loss coefficients require an explicit new implementation/version"
            )
        if variant not in self.cfg["variants"] or supervisor_step not in (8500, 7000):
            raise ValueError("Unregistered variant or supervisor")
        self.teacher, self.fit, self.step = teacher, fit, supervisor_step
        self.device, self.variant = torch.device(device), variant
        self.teacher_hash = state_digest(teacher)
        self.rows_hash = json_hash(fit.rows)
        # Teacher loading must not perturb the paired generators' initialization.
        seed_all(self.cfg["seed"])
        configure_numerics()
        self.model = GrayscaleSystem().to(self.device)
        copy_precision_condition(self.model.decoder)
        self.initial_model_hash = state_digest(self.model)
        self.ed = optimizer(
            [self.model.private, self.model.decoder], self.cfg["ed_lr"], self.cfg
        )
        self.p = optimizer([self.model.public], self.cfg["p_lr"], self.cfg)
        self.weights = fit.class_weights().to(self.device)
        self.rng = np.random.default_rng(self.cfg["seed"])
        self.seen = np.zeros(len(fit), dtype=np.int64)
        self.position = self.ed_updates = self.p_updates = 0
        self.attempted_updates = 0
        self.elapsed_seconds = 0.0
        self.boundary = True

    @property
    def total(self):
        return self.cfg["baseline_updates"] + self.cfg["c1_cycles"]

    def phase_at(self, position):
        if position <= self.cfg["baseline_updates"]:
            return "b", position, self.cfg["baseline_updates"]
        return "c1", position - self.cfg["baseline_updates"], self.cfg["c1_cycles"]

    def cycle(self, inspect_gradients=False):
        if self.position >= self.total or not self.boundary:
            raise ValueError("No cycle available; partial cycles cannot continue")
        phase, cycle, total = self.phase_at(self.position + 1)
        indices = self.rng.integers(len(self.fit), size=self.cfg["batch_size"])
        b = collate([self.fit[int(i)] for i in indices], self.device)
        sync(self.device)
        start = time.perf_counter()
        self.boundary = False
        factor = lr_factor(cycle, total, self.cfg)
        for opt, base in ((self.ed, self.cfg["ed_lr"]), (self.p, self.cfg["p_lr"])):
            for group in opt.param_groups:
                group["lr"] = base * factor
        try:
            targets = teacher_targets(self.teacher, b)
            self.attempted_updates += 1
            ed = update(
                self.model,
                self.teacher,
                b,
                targets,
                self.weights,
                self.ed,
                "b" if phase == "b" or self.variant == "fixed_gray" else "ed",
                canonical_padding=True,
                step_optimizer=True,
            )
            self.ed_updates += 1
            public = None
            if phase == "c1" and self.variant == "learned":
                self.attempted_updates += 1
                public = update(
                    self.model,
                    self.teacher,
                    b,
                    targets,
                    self.weights,
                    self.p,
                    "p",
                    inspect_gradients=inspect_gradients,
                    canonical_padding=True,
                    step_optimizer=True,
                )
                self.p_updates += 1
            self.position += 1
            np.add.at(self.seen, indices, 1)
            self.boundary = True
        finally:
            sync(self.device)
            self.elapsed_seconds += time.perf_counter() - start
        return dict(
            phase=phase,
            cycle=cycle,
            position=self.position,
            indices=indices.tolist(),
            optimizer_updates=self.ed_updates + self.p_updates,
            attempted_updates=self.attempted_updates,
            ed_lr=self.ed.param_groups[0]["lr"],
            p_lr=self.p.param_groups[0]["lr"] if public is not None else None,
            ed=ed,
            public=public,
            full_cycle_seconds=time.perf_counter() - start,
        )

    def state(self):
        if not self.boundary:
            raise ValueError("Cannot save an incomplete ED/P cycle")
        if state_digest(self.teacher) != self.teacher_hash:
            raise ValueError("Frozen teacher changed")
        return dict(
            schema="r5-cycle-state-1",
            config=self.cfg,
            supervisor_step=self.step,
            variant=self.variant,
            teacher_hash=self.teacher_hash,
            rows_hash=self.rows_hash,
            initial_model_hash=self.initial_model_hash,
            model=self.model.state_dict(),
            ed=self.ed.state_dict(),
            public=self.p.state_dict(),
            position=self.position,
            ed_updates=self.ed_updates,
            p_updates=self.p_updates,
            attempted_updates=self.attempted_updates,
            elapsed_seconds=self.elapsed_seconds,
            seen=self.seen.tolist(),
            sampling_rng=self.rng.bit_generator.state,
            rng=rng_state(),
            schedule=dict(
                phase=self.phase_at(self.position)[0], completed_position=self.position
            ),
        )

    def restore(self, state):
        expected = dict(
            schema="r5-cycle-state-1",
            config=self.cfg,
            supervisor_step=self.step,
            variant=self.variant,
            teacher_hash=self.teacher_hash,
            rows_hash=self.rows_hash,
            initial_model_hash=self.initial_model_hash,
        )
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "Checkpoint/config/teacher/dataset/initialization mismatch"
            )
        pos = state["position"]
        expected_p = (
            max(0, pos - self.cfg["baseline_updates"])
            if self.variant == "learned"
            else 0
        )
        if (
            not 0 <= pos <= self.total
            or state["ed_updates"] != pos
            or state["p_updates"] != expected_p
        ):
            raise ValueError("Invalid cycle/update counters")
        if (
            len(state["seen"]) != len(self.fit)
            or any(n < 0 for n in state["seen"])
            or sum(state["seen"]) != pos * self.cfg["batch_size"]
        ):
            raise ValueError("Invalid sample accounting")
        if state["schedule"] != dict(
            phase=self.phase_at(pos)[0], completed_position=pos
        ):
            raise ValueError("Schedule counter mismatch")
        self.model.load_state_dict(state["model"], strict=True)
        self.ed.load_state_dict(state["ed"])
        self.p.load_state_dict(state["public"])
        self.position, self.ed_updates, self.p_updates = (
            pos,
            state["ed_updates"],
            state["p_updates"],
        )
        self.attempted_updates, self.elapsed_seconds = (
            state["attempted_updates"],
            state["elapsed_seconds"],
        )
        self.seen = np.asarray(state["seen"], dtype=np.int64)
        self.rng.bit_generator.state = state["sampling_rng"]
        restore_rng(state["rng"])
        self.boundary = True


def diagnose(engine, data, output, label):
    """Full val_tune frozen-head diagnostics, with predictions for later paired analysis."""
    original_rng = rng_state()
    was_training = engine.model.training
    sync(engine.device)
    start = time.perf_counter()
    rows, truth, per_image = [], [], []
    probabilities = {
        k: [] for k in ("rgb", "ordinary_gray", "public_gray", "restored_rgb")
    }
    engine.model.eval()
    try:
        with torch.no_grad():
            for offset in range(0, len(data), engine.cfg["diagnostic_batch_size"]):
                items = [
                    data[i]
                    for i in range(
                        offset,
                        min(offset + engine.cfg["diagnostic_batch_size"], len(data)),
                    )
                ]
                b = collate(items, engine.device)
                out = public_forward(engine.model, b["x"], b["v"])
                for name, image, domain in (
                    ("rgb", b["x"], 0),
                    ("ordinary_gray", gray(b["x"]), 1),
                    ("public_gray", out["gray"], 1),
                    ("restored_rgb", out["rgb"], 0),
                ):
                    probabilities[name].extend(
                        engine.teacher(image, b["rect"], domain)[0]
                        .softmax(1)
                        .cpu()
                        .tolist()
                    )
                truth.extend(b["label"].cpu().tolist())
                rows.extend(
                    dict(id=i, patient=p) for i, p in zip(b["id"], b["patient"])
                )
                for i in range(len(items)):
                    v, x = b["v"][i : i + 1], b["x"][i : i + 1]
                    g, rec = out["gray"][i : i + 1], out["rgb"][i : i + 1]
                    ss, ok = ssim_values(rec, x, v)
                    gs, gok = ssim_values(g, gray(x), v)
                    mse = masked_mean((rec - x).square(), v)
                    per_image.append(
                        dict(
                            psnr_db=float(-10 * torch.log10(mse.clamp_min(1e-12))),
                            ssim=float(ss[0]) if ok[0] else None,
                            gray_ssim=float(gs[0]) if gok[0] else None,
                            gray_l1=float(masked_mean((g - gray(x)).abs(), v)),
                            gray_bias=float(masked_mean(g - gray(x), v)),
                        )
                    )
                if offset == 0:
                    panels = [
                        b["x"][:1],
                        gray(b["x"][:1]).expand(-1, 3, -1, -1),
                        out["gray"][:1].expand(-1, 3, -1, -1),
                        out["rgb"][:1],
                    ]
                    a = torch.cat(panels, -1)[0].permute(1, 2, 0).cpu()
                    Image.fromarray(
                        torch.floor(a.clamp(0, 1) * 255 + 0.5).byte().numpy()
                    ).save(Path(output) / f"{label}-panel.png")
        report = dict(
            role="val_tune",
            images=len(truth),
            position=engine.position,
            supervisor_step=engine.step,
            variant=engine.variant,
            teacher_diagnostic_only=True,
            independent_evaluation=False,
            privacy_evaluated=False,
            medical={k: clinical(truth, v) for k, v in probabilities.items()},
            per_image=per_image,
            selected_checkpoint=None,
            scientific_acceptance="not_evaluated",
        )
        for key in ("psnr_db", "ssim", "gray_ssim", "gray_l1", "gray_bias"):
            values = [r[key] for r in per_image if r[key] is not None]
            report["mean_" + key] = float(np.mean(values)) if values else None
            report[key + "_valid_images"] = len(values)
        atomic_json(Path(output) / f"{label}-diagnostic.json", report)
        atomic_json(
            Path(output) / f"{label}-predictions.json",
            dict(rows=rows, labels=truth, probabilities=probabilities),
        )
        return [
            f"{label}-{suffix}"
            for suffix in ("diagnostic.json", "predictions.json", "panel.png")
        ]
    finally:
        engine.model.train(was_training)
        restore_rng(original_rng)
        sync(engine.device)
        engine.elapsed_seconds += time.perf_counter() - start
