"""Explicit registration, real pixel gradient verification, and bounded B/C1 pilot."""

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import os

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ccrr.data.pad import PADData, collate
from ccrr.teachers.adapters import PanDerm
from ccrr.tensor_ops import gray, masked_mean, ssim_values
from ccrr.evaluation.metrics import clinical
from .dev2 import ROOT, FEATURES, current_commit, check_hashes
from .runtime import (
    artifact,
    digest,
    read,
    write,
    new_run,
    finish,
    status,
    log,
    seed_all,
    save,
    validate_resources,
)
from .learning import checkpoint_path, load_separator
from .grayscale import GrayscaleSystem, PixelSupervisor, update

PLAN = ROOT / "plans/grayscale-dev1-20261020"
SEPARATOR = ROOT / "runs/train/dev2-separation.20261020-184228.3079814"
OLD_COMMIT = "0c38a58f1c31f9b9bc9844ef9f0ad58ab2bda275"
STEPS = (8500, 7000)
CONFIG = dict(
    schema="ccrr-v3-grayscale-pilot-1",
    scope="development_only",
    primary_supervisor=8500,
    comparison_supervisor=7000,
    seed=20261020,
    batch_size=2,
    baseline_updates=500,
    c1_cycles=500,
    lr=1e-4,
    checkpoint_every=100,
    precision_bits=8,
    architecture="design2-r4-pecd-v1",
    weights=dict(
        rgb_l1=1.0,
        rgb_ssim=0.1,
        rgb_clinical=0.1,
        kd=0.5,
        appearance=1.0,
        public_clinical=0.1,
        range=0.001,
        privacy=0.0,
        feature_alignment=0.0,
        rate=0.0,
        region=0.0,
        concept=0.0,
    ),
    no_early_selection=True,
    no_resume=True,
    test_images_used=False,
    note="B fixed ordinary gray then C1 alternating ED/P. Engineering/development pilot, "
    "not privacy training or independent clinical/identity acceptance. "
    "Original bound checkpoint heads, not independent audit readout heads. "
    "8500 primary;7000 comparison uses identical initialization/sampling/budgets. "
    "4-bit architecture supported but not trained in this 8-bit pilot. "
    "No Writer, key authorization, or embedding transport in this stage.",
)


def state_digest(module):
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def register():
    resources = read(ROOT / "private/resources.json")
    validate_resources(resources)
    print(
        "Validating existing DEV2 separator and selected audits; no model fitting.",
        flush=True,
    )
    manifest, binding = artifact(SEPARATOR, "separator", resources)
    if manifest["variant"] != "separation" or binding["source_commit"] != OLD_COMMIT:
        raise ValueError("Wrong selected separator")
    feature = read(FEATURES / "features.json")
    assert binding["parents"]["features_sha256"] == digest(FEATURES / "features.json")
    parents = {
        str(SEPARATOR / "artifact.json"): digest(SEPARATOR / "artifact.json"),
        str(FEATURES / "features.json"): digest(FEATURES / "features.json"),
        str(FEATURES / feature["statistics"]["file"]): feature["statistics"]["sha256"],
    }
    parents.update(
        {
            str(FEATURES / feature["roles"][role]["file"]): feature["roles"][role][
                "sha256"
            ]
            for role in ("fit",)
        }
    )
    for step in STEPS:
        checkpoint = checkpoint_path(SEPARATOR, step)
        parents[str(checkpoint)] = manifest["files"][checkpoint.name]
        candidates = sorted(
            (ROOT / "runs/audit").glob(f"dev2-separation-step{step}-tune.*")
        )
        if len(candidates) != 1:
            raise ValueError(f"Require unique completed independent audit for {step}")
        _, audit_binding = artifact(candidates[0], "audit", resources)
        assert audit_binding["parents"]["separator"] == str(SEPARATOR)
        assert (
            audit_binding["parents"]["separator_checkpoint_sha256"]
            == parents[str(checkpoint)]
        )
        assert audit_binding["options"]["evaluation_role"] == "val_tune"
        parents[str(candidates[0] / "artifact.json")] = digest(
            candidates[0] / "artifact.json"
        )
    check_hashes(parents)
    PLAN.mkdir(parents=True, exist_ok=False)
    write(PLAN / "config.json", CONFIG)
    write(
        PLAN / "registration.json",
        dict(
            state="registered",
            source_commit=current_commit(),
            parents=parents,
            config_sha256=digest(PLAN / "config.json"),
            resources=resources,
            decision="user_selected_8500_primary_7000_control_for_experimental_integration",
            training_started=False,
            scientific_acceptance="not_evaluated",
        ),
    )
    print("GRAYSCALE_REGISTERED:", PLAN, "training_started=False", flush=True)


def load_teacher(resources, step, device):
    p = resources["panderm"]
    backbone = PanDerm(p["source"], p["source_sha256"], p["path"], p["sha256"]).to(
        device
    )
    model, heads, stats = load_separator(checkpoint_path(SEPARATOR, step), device)
    fit_stats = read(FEATURES / "features.json")["statistics"]
    original = torch.load(
        FEATURES / fit_stats["file"], map_location="cpu", weights_only=True
    )
    assert all(torch.equal(stats[k].cpu(), original[k]) for k in ("mean", "std"))
    return PixelSupervisor(backbone, model, heads, stats).to(device)


def png_roundtrip(g, path):
    if g.shape[0:2] != (1, 1):
        raise ValueError("PNG check takes one one-channel image")
    integers = (
        torch.floor(g.detach().cpu()[0, 0].clamp(0, 1) * 255 + 0.5).byte().numpy()
    )
    Image.fromarray(integers).save(path)
    with Image.open(path) as im:
        assert im.mode == "L"
        loaded = (
            torch.from_numpy(np.array(im, copy=True)).to(g.device).float()[None, None]
            / 255
        )
    assert torch.allclose(loaded, g.detach(), atol=1e-7, rtol=0)
    return loaded


def verify(output, resources, step, device):
    seed_all(CONFIG["seed"])
    teacher = load_teacher(resources, step, device)
    before = state_digest(teacher)
    data = PADData(resources["data_root"], "fit")
    # Fixed fit image only; no held-out images used for this engineering check.
    b = collate([data[0]], device)
    manifest = read(FEATURES / "features.json")
    cached = torch.load(
        FEATURES / manifest["roles"]["fit"]["file"],
        map_location="cpu",
        weights_only=True,
    )
    assert cached["rows"] == data.rows
    expected = cached["features"][0].clone()
    coverage = cached["coverage"][0].clone()
    del cached
    result = dict(supervisor_step=step, fit_image_index=0, domains={})
    for domain, image in enumerate((b["x"], gray(b["x"]))):
        generated = image.detach().clone().requires_grad_(True)
        with torch.no_grad():
            raw, v, _ = teacher.panderm(image, b["rect"])
            assert torch.allclose(v.cpu()[0], coverage, atol=1e-7, rtol=0)
            error = float((raw.cpu()[0] - expected[domain]).abs().max())
            if not torch.allclose(raw.cpu()[0], expected[domain], atol=1e-4, rtol=1e-4):
                raise ValueError(
                    f"Live PanDerm differs from bound A0 features: {error}"
                )
            reference, _ = teacher(image, b["rect"], domain)
        logits, _ = teacher(generated, b["rect"], domain)
        (gradient,) = torch.autograd.grad(
            F.cross_entropy(logits, b["label"]), generated
        )
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        assert torch.equal(logits.detach(), reference)
        result["domains"][str(domain)] = dict(
            cache_max_abs_error=error,
            input_gradient_norm=float(gradient.norm()),
            deterministic=True,
            gradient_finite_nonzero=True,
        )
    model = GrayscaleSystem().to(device)
    model.train()
    teacher.train()
    assert not any(m.training for m in teacher.modules())
    assert not any(p.requires_grad for p in teacher.parameters())
    before_model = state_digest(model)
    out = model(b["x"])
    assert out["q"].shape == (1, 8, 32, 32)
    assert torch.equal(out["gray"], gray(b["x"]))
    disk = png_roundtrip(out["gray"], output / "initial_gray.png")
    with torch.no_grad():
        _, restored = model.decoder(disk, out["q"].float() / 255, 8)
    assert torch.allclose(restored, out["rgb"].detach(), atol=1e-6, rtol=1e-6)
    logits, _ = teacher(out["gray"], b["rect"], 1)
    grads = torch.autograd.grad(
        F.cross_entropy(logits, b["label"]), tuple(model.public.parameters())
    )
    norm = torch.sqrt(sum(g.square().sum() for g in grads))
    assert torch.isfinite(norm) and norm > 0
    assert state_digest(teacher) == before and state_digest(model) == before_model
    result.update(
        public_medical_parameter_gradient_norm=float(norm),
        frozen_teacher_parameters_and_buffers_unchanged=True,
        checkpoint_original_heads=True,
        png_and_integer_code_forward_match=True,
        private_symbols=8192,
        private_body_bytes_8bit=8192,
        training_started=False,
        model_parameters_updated=False,
        real_pixel_supervision_verified=True,
        scientific_acceptance="not_evaluated",
    )
    write(output / "verification.json", result)
    return ["verification.json", "initial_gray.png"]


def teacher_targets(teacher, b):
    with torch.no_grad():
        return (
            teacher(b["x"], b["rect"], 0)[0],
            teacher(gray(b["x"]), b["rect"], 1)[0],
        )


def diagnose(model, teacher, data, output, label, device):
    """Full val_tune fixed-teacher diagnostics, never independent acceptance."""
    truth = []
    probabilities = {
        k: [] for k in ("rgb", "ordinary_gray", "public_gray", "restored_rgb")
    }
    metrics = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), CONFIG["batch_size"]):
            b = collate(
                [
                    data[i]
                    for i in range(start, min(start + CONFIG["batch_size"], len(data)))
                ],
                device,
            )
            out = model(b["x"])
            for key, image, domain in (
                ("rgb", b["x"], 0),
                ("ordinary_gray", gray(b["x"]), 1),
                ("public_gray", out["gray"], 1),
                ("restored_rgb", out["rgb"], 0),
            ):
                probabilities[key].extend(
                    teacher(image, b["rect"], domain)[0].softmax(1).cpu().tolist()
                )
            truth.extend(b["label"].cpu().tolist())
            # Report per-image statistics on valid support.
            for i in range(len(b["x"])):
                v, x = b["v"][i : i + 1], b["x"][i : i + 1]
                g, rgb = out["gray"][i : i + 1], out["rgb"][i : i + 1]
                ssim, valid = ssim_values(rgb, x, v)
                mse = masked_mean((rgb - x).square(), v)
                metrics.append(
                    dict(
                        psnr_db=float(-10 * torch.log10(mse.clamp_min(1e-12))),
                        ssim=float(ssim[0]) if valid[0] else None,
                        gray_l1=float(masked_mean((g - gray(x)).abs(), v)),
                    )
                )
            if start == 0:
                # Panel stored privately on server: source / Y / G0 / restored.
                panels = [
                    b["x"][:1],
                    gray(b["x"][:1]).expand(-1, 3, -1, -1),
                    out["gray"][:1].expand(-1, 3, -1, -1),
                    out["rgb"][:1],
                ]
                a = torch.cat(panels, -1)[0].permute(1, 2, 0).cpu()
                Image.fromarray(
                    torch.floor(a.clamp(0, 1) * 255 + 0.5).byte().numpy()
                ).save(output / f"{label}-panel.png")
    report = dict(
        role="val_tune",
        images=len(truth),
        teacher_diagnostic_only=True,
        medical={k: clinical(truth, v) for k, v in probabilities.items()},
        mean_psnr_db=float(np.mean([m["psnr_db"] for m in metrics])),
        mean_ssim=float(np.mean([m["ssim"] for m in metrics if m["ssim"] is not None])),
        ssim_valid_images=sum(m["ssim"] is not None for m in metrics),
        mean_gray_l1=float(np.mean([m["gray_l1"] for m in metrics])),
        privacy_evaluated=False,
        scientific_acceptance="not_evaluated",
    )
    write(output / f"{label}-diagnostic.json", report)
    model.train()
    return [f"{label}-diagnostic.json", f"{label}-panel.png"]


def pilot(output, resources, step, device):
    seed_all(CONFIG["seed"])
    teacher = load_teacher(resources, step, device)
    teacher_hash = state_digest(teacher)
    model = GrayscaleSystem().to(device)
    fit, tune = (
        PADData(resources["data_root"], "fit"),
        PADData(resources["data_root"], "val_tune"),
    )
    weights = fit.class_weights().to(device)
    rng = np.random.default_rng(CONFIG["seed"])
    seen = np.zeros(len(fit), dtype=np.int64)
    ed = torch.optim.Adam(
        [*model.private.parameters(), *model.decoder.parameters()], lr=CONFIG["lr"]
    )
    public = torch.optim.Adam(model.public.parameters(), lr=CONFIG["lr"])
    files, updates = [], 0
    for phase, limit in (
        ("b", CONFIG["baseline_updates"]),
        ("c1", CONFIG["c1_cycles"]),
    ):
        for cycle in range(1, limit + 1):
            indices = rng.integers(len(fit), size=CONFIG["batch_size"])
            np.add.at(seen, indices, 1)
            b = collate([fit[int(i)] for i in indices], device)
            targets = teacher_targets(teacher, b)
            terms = update(
                model, teacher, b, targets, weights, ed, "b" if phase == "b" else "ed"
            )
            updates += 1
            if phase == "c1":
                pterms = update(
                    model,
                    teacher,
                    b,
                    targets,
                    weights,
                    public,
                    "p",
                    inspect_gradients=(cycle == 1),
                )
                updates += 1
            else:
                pterms = None
            if cycle == 1 or cycle % 25 == 0:
                log(
                    output,
                    "grayscale_update",
                    phase=phase,
                    cycle=cycle,
                    optimizer_updates=updates,
                    ed=terms,
                    public=pterms,
                )
            if cycle % CONFIG["checkpoint_every"] == 0 or cycle == limit:
                name = f"generator-{phase}-{cycle:06d}.pt"
                save(
                    output / name,
                    dict(
                        model=model.state_dict(),
                        config=CONFIG,
                        supervisor_step=step,
                        phase=phase,
                        cycle=cycle,
                        optimizer_updates=updates,
                        ed_optimizer=ed.state_dict(),
                        p_optimizer=public.state_dict(),
                        torch_rng=torch.get_rng_state(),
                        sampling_rng=rng.bit_generator.state,
                        cuda_rng=torch.cuda.get_rng_state_all()
                        if device.type == "cuda"
                        else [],
                        resume_supported=False,
                    ),
                )
                files.append(name)
                log(
                    output,
                    "generator_checkpoint_saved",
                    phase=phase,
                    cycle=cycle,
                    file=name,
                    selected=False,
                )
        files.extend(diagnose(model, teacher, tune, output, phase, device))
    assert state_digest(teacher) == teacher_hash
    assert all(p.grad is None for p in teacher.parameters())
    summary = dict(
        supervisor_step=step,
        baseline_updates=CONFIG["baseline_updates"],
        c1_cycles=CONFIG["c1_cycles"],
        optimizer_updates=updates,
        sampled_positions=int(seen.sum()),
        unique_fit_images=int((seen > 0).sum()),
        frozen_teacher_unchanged=True,
        precision_bits_trained=8,
        private_symbols=8192,
        private_body_bytes=8192,
        selected_checkpoint=None,
        privacy_training=False,
        direct_pixel_p2_p3_evaluated=False,
        independent_medical_evaluation=False,
        authorization_and_writer_implemented=False,
        scientific_acceptance="not_evaluated",
    )
    write(output / "training_summary.json", summary)
    return [*files, "training_summary.json"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "verify", "pilot", "start"))
    parser.add_argument("--supervisor-step", type=int, choices=STEPS, default=8500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--verification", help="Completed matching real-pixel verification run"
    )
    args = parser.parse_args(argv)
    if args.action == "start":
        if args.verification:
            raise ValueError(
                "start creates its own verification; use pilot to supply one"
            )
        verification = main(
            [
                "verify",
                "--supervisor-step",
                str(args.supervisor_step),
                "--device",
                args.device,
            ]
        )
        return main(
            [
                "pilot",
                "--supervisor-step",
                str(args.supervisor_step),
                "--device",
                args.device,
                "--verification",
                str(verification),
            ]
        )
    if args.action == "register":
        register()
        return
    reg = read(PLAN / "registration.json")
    assert reg["state"] == "registered" and current_commit() == reg["source_commit"]
    assert digest(PLAN / "config.json") == reg["config_sha256"]
    assert read(PLAN / "config.json") == CONFIG
    check_hashes(reg["parents"])
    resources = reg["resources"]
    validate_resources(resources)
    parents = dict(reg["parents"])
    parents[str(PLAN / "registration.json")] = digest(PLAN / "registration.json")
    if args.action == "pilot":
        if not args.verification:
            raise ValueError(
                "A matching real-pixel verification is required before pilot updates"
            )
        _, binding = artifact(
            args.verification, "pixel_supervisor_verification", resources
        )
        report = read(Path(args.verification) / "verification.json")
        assert binding["source_commit"] == current_commit()
        assert binding["parents"] == parents
        assert report["supervisor_step"] == args.supervisor_step
        assert report["real_pixel_supervision_verified"]
        parents[str(Path(args.verification) / "artifact.json")] = digest(
            Path(args.verification) / "artifact.json"
        )
    output = (
        ROOT
        / "runs/grayscale"
        / (
            f"{args.action}-s{args.supervisor_step}.{datetime.now():%Y%m%d-%H%M%S}.{os.getpid()}"
        )
    )
    device = torch.device(args.device)
    out = new_run(
        resources,
        output,
        "grayscale-" + args.action,
        CONFIG,
        parents,
        dict(supervisor_step=args.supervisor_step, device=str(device)),
    )
    print("RUN:", out, flush=True)
    try:
        payload = (verify if args.action == "verify" else pilot)(
            out, resources, args.supervisor_step, device
        )
        check_hashes(parents)
        log(out, "parents_unchanged")
        finish(
            out,
            "pixel_supervisor_verification"
            if args.action == "verify"
            else "grayscale_pilot",
            payload,
            supervisor_step=args.supervisor_step,
        )
    except Exception as exc:
        status(out, "failed", error=str(exc))
        raise
    print("GRAYSCALE_COMPLETED:", out, flush=True)
    return out


if __name__ == "__main__":
    main()
