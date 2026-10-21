"""Original-method baseline only. Every successful run terminates at epoch 120."""

from datetime import timedelta
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ..original import OriginalGrayscale, quantize8
from ..losses import author_loss
from ..verify import REPO, digest, state_digest, finite_gradients, source_check
from .data import load_config, load_prepared, batch_tensor, write_json
from .optim import TF1Adam, polynomial_lr, ContinuousBatches
from .vgg import AuthorVGG


def implementation_hashes():
    files = [
        REPO / "v4/ccrr_v4/original.py",
        REPO / "v4/ccrr_v4/losses.py",
        *sorted((REPO / "v4/ccrr_v4/phase1").glob("*.py")),
    ]
    return {str(p.relative_to(REPO)): digest(p) for p in files}


def setup(cfg):
    world, rank, local = (
        int(os.environ.get(k, d))
        for k, d in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))
    )
    runtime = cfg["runtime"]
    if cfg["global_batch_size"] % world:
        raise ValueError("Global batch 8 must be divisible by GPU count")
    torch.set_num_threads(runtime["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if runtime["device"] == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != world:
            raise ValueError("Explicit GPU selection must match world size")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=10)
        )
    return world, rank, device


def mean_metrics(terms, total, world):
    keys = ["total", *sorted(terms)]
    values = torch.stack([total.detach(), *(terms[k].detach() for k in sorted(terms))])
    if world > 1:
        dist.all_reduce(values)
        values /= world
    return dict(zip(keys, values.cpu().tolist()))


def loss_step(model, vgg, x, cfg, stage, world):
    g, restored = model(x)
    with torch.no_grad():
        target_features = vgg(x)
    gray_features = vgg(g)  # Frozen weights, retain gray -> E input gradient.
    return author_loss(
        x,
        g,
        restored,
        target_features,
        gray_features,
        stage,
        cfg["stages"][stage - 1]["weights"],
        distributed=world > 1,
    )


def digital_roundtrip(g):
    integers, expected = quantize8(g)
    copies = []
    for row in integers:
        stream = BytesIO()
        Image.fromarray(row[0].cpu().numpy()).save(stream, format="PNG")
        stream.seek(0)
        with Image.open(stream) as im:
            if im.mode != "L":
                raise ValueError("Require a single-channel 8bit PNG")
            copies.append(torch.from_numpy(np.array(im, copy=True)))
    received = torch.stack(copies)[:, None].to(g.device)
    if not torch.equal(integers, received):
        raise ValueError("PNG altered gray pixels")
    decoded = received.to(g.dtype) / 127.5 - 1
    if not torch.equal(decoded, expected):
        raise ValueError("8bit normalization mismatch")
    return decoded


@torch.no_grad()
def evaluate(model, array, batch, device):
    totals = dict(
        continuous_mse=0.0,
        png_mse=0.0,
        continuous_psnr=0.0,
        png_psnr=0.0,
        png_mean_channel_psnr=0.0,
        quantization_gray_mae=0.0,
    )
    model.eval()
    for start in range(0, len(array), batch):
        x = batch_tensor(
            array, list(range(start, min(start + batch, len(array)))), device
        )
        g, continuous = model(x)
        received = digital_roundtrip(g)
        rgb = model.decoder(received)
        for name, output in [("continuous", continuous), ("png", rgb)]:
            mse = ((output - x) / 2).square().mean((1, 2, 3))
            totals[name + "_mse"] += float(mse.sum())
            totals[name + "_psnr"] += float(
                (-10 * torch.log10(mse.clamp_min(1e-10))).sum()
            )
        cmse = ((rgb - x) / 2).square().mean((2, 3))
        totals["png_mean_channel_psnr"] += float(
            (-10 * cmse.clamp_min(1e-10).log10()).mean(1).sum()
        )
        totals["quantization_gray_mae"] += (
            float((g - received).abs().mean((1, 2, 3)).sum()) * 127.5
        )
    model.train()
    return {
        **{k: v / len(array) for k, v in totals.items()},
        "images": len(array),
        "role": "val_tune",
        "png_exact": True,
        "used_for_selection": False,
    }


def optimizer_for(model, cfg):
    o = cfg["optimizer"]
    return TF1Adam(
        model.parameters(),
        lr=o["learning_rate"],
        betas=tuple(o["betas"]),
        eps=o["epsilon"],
    )


@torch.no_grad()
def save_preview(model, array, device, path):
    """Fixed first four val_tune examples, kept only in the private server run."""
    rows = []
    for i in range(min(4, len(array))):
        x = batch_tensor(array, [i], device)
        received = digital_roundtrip(model.encoder(x))
        rgb = model.decoder(received)
        ordinary = (
            x * x.new_tensor([0.2989, 0.5870, 0.1140])[None, :, None, None]
        ).sum(1, keepdim=True)
        panels = []
        for t in (x, ordinary.expand_as(x), received.expand_as(x), rgb):
            panels.append(quantize8(t)[0][0].permute(1, 2, 0).cpu().numpy())
        error = ((rgb - x).abs() * 127.5 * 4).clamp(0, 255).round().byte()
        panels.append(error[0].permute(1, 2, 0).cpu().numpy())
        rows.append(np.concatenate(panels, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def checkpoint(path, model, optimizer, epoch, step, stage, binding):
    # Immutable epoch files; keep previous evidence instead of source max_to_keep=1.
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(".partial")
    torch.save(
        dict(
            schema="ccrr-v4-phase1-checkpoint-1",
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            step=step,
            stage=stage,
            binding=binding,
        ),
        tmp,
    )
    os.replace(tmp, path)
    return dict(file=path.name, sha256=digest(path), epoch=epoch, step=step)


def run(config, prepared_path, output, mode, resume=None):
    if mode not in ("preflight", "train"):
        raise ValueError("Only phase-1 preflight or training")
    cfg = load_config(config)
    total_epochs = sum(s["epochs"] for s in cfg["stages"])
    p, arrays = load_prepared(prepared_path, cfg)
    world, rank, device = setup(cfg)
    output = Path(output).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    binding = dict(
        config=cfg,
        prepared_sha256=digest(prepared_path),
        source=source_check(),
        implementation=implementation_hashes(),
        code_commit=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        vgg=p["vgg"],
        counts=p["schedule_counts"],
        runtime=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            world_size=world,
            per_device_batch=cfg["global_batch_size"] // world,
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            device_name=torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
        ),
    )
    if rank == 0:
        write_json(output / "binding.json", binding)
        write_json(output / "status.json", dict(state="running", mode=mode, phase=1))
    try:
        model = OriginalGrayscale().to(device)
        vgg = AuthorVGG.from_file(p["vgg"]["path"], p["vgg"]["sha256"]).to(device)
        start_epoch, step, previous_stage = 0, 0, 1
        optimizer = optimizer_for(model, cfg)
        if resume:
            if mode != "train":
                raise ValueError("No resume in zero-update preflight")
            ck = torch.load(resume, map_location=device, weights_only=True)
            if ck["schema"] != "ccrr-v4-phase1-checkpoint-1":
                raise ValueError("Not a phase-1 checkpoint")
            for key in ("prepared_sha256", "implementation", "source", "counts", "vgg"):
                if ck["binding"][key] != binding[key]:
                    raise ValueError(f"Resume binding mismatch: {key}")
            for key in cfg.keys() - {
                "runtime",
                "log_every_steps",
                "validation_batch_size",
                "checkpoint_every_epochs",
            }:
                if ck["binding"]["config"][key] != cfg[key]:
                    raise ValueError(f"Resume configuration mismatch: {key}")
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch, step, previous_stage = ck["epoch"], ck["step"], ck["stage"]
            if (
                step != start_epoch * p["schedule_counts"]["steps_per_epoch"]
                or start_epoch >= total_epochs
            ):
                raise ValueError("Invalid/completed epoch checkpoint")
            if rank == 0:
                write_json(
                    output / "resume.json",
                    dict(path=str(Path(resume).resolve()), sha256=digest(resume)),
                )
            del ck
        wrapped = (
            DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None
            )
            if world > 1
            else model
        )
        stream = ContinuousBatches(
            len(arrays["fit"]),
            cfg["global_batch_size"],
            cfg["seed"],
            step * cfg["global_batch_size"],
        )
        local_batch = cfg["global_batch_size"] // world

        if mode == "preflight":
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            before, vbefore = state_digest(model), state_digest(vgg)
            ids = stream.next()[rank * local_batch : (rank + 1) * local_batch]
            x = batch_tensor(arrays["fit"], ids, device)
            report = {}
            for stage in (1, 2):
                model.zero_grad(set_to_none=True)
                total, terms = loss_step(wrapped, vgg, x, cfg, stage, world)
                if not torch.isfinite(total):
                    raise ValueError("Nonfinite preflight loss")
                total.backward()
                grads = finite_gradients(model)
                report[str(stage)] = dict(
                    losses=mean_metrics(terms, total, world), gradients=grads
                )
            if before != state_digest(model) or vbefore != state_digest(vgg):
                raise ValueError("Zero-update preflight changed parameters")
            if any(t.grad is not None for t in vgg.parameters()):
                raise ValueError("VGG must remain frozen")
            with torch.no_grad():
                digital_roundtrip(model.encoder(x[:1]))
            if rank == 0:
                write_json(
                    output / "report.json",
                    dict(
                        mode=mode,
                        phase=1,
                        passed=True,
                        stages=report,
                        counts=p["schedule_counts"],
                        optimizer_updates=0,
                        png_exact=True,
                        vgg_frozen=True,
                        weights_unchanged=True,
                        scientific_acceptance="not_evaluated",
                        world_size=world,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else None,
                    ),
                )
        else:
            steps_per_epoch = p["schedule_counts"]["steps_per_epoch"]
            checkpoints = []
            for epoch in range(start_epoch, total_epochs):
                stage = 1 if epoch < cfg["stages"][0]["epochs"] else 2
                if stage != previous_stage:
                    optimizer = optimizer_for(
                        model, cfg
                    )  # Original second Adam starts empty.
                    previous_stage = stage
                started = time.monotonic()
                sums = {}
                for _ in range(steps_per_epoch):
                    ids = stream.next()[rank * local_batch : (rank + 1) * local_batch]
                    x = batch_tensor(arrays["fit"], ids, device)
                    lr = polynomial_lr(
                        step,
                        p["schedule_counts"]["decay_steps"],
                        cfg["optimizer"]["learning_rate"],
                        cfg["schedule"]["final_learning_rate"],
                        cfg["schedule"]["power"],
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    total, terms = loss_step(wrapped, vgg, x, cfg, stage, world)
                    if not torch.isfinite(total):
                        raise ValueError(f"Nonfinite loss at update {step}")
                    total.backward()
                    if not all(
                        t.grad is not None and torch.isfinite(t.grad).all()
                        for t in model.parameters()
                    ):
                        raise ValueError(f"Nonfinite/missing gradient at update {step}")
                    optimizer.step()
                    step += 1
                    metrics = mean_metrics(terms, total, world)
                    for name, value in metrics.items():
                        sums[name] = sums.get(name, 0.0) + value
                    if rank == 0 and step % cfg["log_every_steps"] == 0:
                        print(
                            json.dumps(
                                dict(
                                    epoch=epoch + 1,
                                    author_stage=stage,
                                    step=step,
                                    lr=lr,
                                    **metrics,
                                )
                            ),
                            flush=True,
                        )
                # Only rank 0 monitors all val_tune images; other ranks wait.
                if rank == 0:
                    validation = evaluate(
                        model, arrays["val_tune"], cfg["validation_batch_size"], device
                    )
                    row = dict(
                        epoch=epoch + 1,
                        author_stage=stage,
                        step=step,
                        lr=lr,
                        seconds=time.monotonic() - started,
                        train={k: v / steps_per_epoch for k, v in sums.items()},
                        validation=validation,
                    )
                    with (output / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(row, allow_nan=False) + "\n")
                    print("EPOCH " + json.dumps(row, allow_nan=False), flush=True)
                    if (epoch + 1) % cfg[
                        "checkpoint_every_epochs"
                    ] == 0 or epoch + 1 in (cfg["stages"][0]["epochs"], total_epochs):
                        record = checkpoint(
                            output / f"epoch-{epoch + 1:03d}.pt",
                            model,
                            optimizer,
                            epoch + 1,
                            step,
                            stage,
                            binding,
                        )
                        checkpoints.append(record)
                        write_json(output / "checkpoints.json", checkpoints)
                        save_preview(
                            model,
                            arrays["val_tune"],
                            device,
                            output / f"preview-epoch-{epoch + 1:03d}.png",
                        )
                        write_json(
                            output / "preview-layout.json",
                            dict(
                                role="val_tune",
                                indices=list(range(min(4, len(arrays["val_tune"])))),
                                columns=[
                                    "input_RGB",
                                    "ordinary_gray",
                                    "G0_8bit",
                                    "D0_from_PNG",
                                    "absolute_RGB_error_x4",
                                ],
                                private_patient_images=True,
                            ),
                        )
                if world > 1:
                    dist.barrier()
            if rank == 0:
                write_json(
                    output / "report.json",
                    dict(
                        phase=1,
                        training_completed=True,
                        final_epoch=total_epochs,
                        final_step=step,
                        final_validation=validation,
                        baseline_checkpoint=checkpoints[-1],
                        scientific_acceptance="pending_review",
                        next_phase="stop",
                        test_used_for_performance=False,
                    ),
                )
        if world > 1:
            dist.barrier()
        if rank == 0:
            write_json(
                output / "status.json",
                dict(state="completed", mode=mode, phase=1, next_phase="stop"),
            )
            print(f"PHASE1_REPORT={output / 'report.json'}", flush=True)
    except BaseException as exc:
        if rank == 0:
            write_json(
                output / "status.json",
                dict(
                    state="failed",
                    mode=mode,
                    phase=1,
                    error=type(exc).__name__ + ": " + str(exc),
                ),
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
