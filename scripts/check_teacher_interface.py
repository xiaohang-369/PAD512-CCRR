"""Offline, one-model-at-a-time real weight checks; never downloads or trains.

Uses synthetic 512 inputs to check adapters, not to measure medical capability.
The supplied weight digest must come from the user's recorded acquisition.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import torch
from torchvision import models
from ccrr.config import sha256
from ccrr.runtime.support import seed_everything, environment
from ccrr.data.pad import adapt
from ccrr.data.resize import POLICY, teacher_input_sources
from ccrr.teachers.adapters import (
    ClinicalModel,
    IdentityModel,
    PanDerm,
    freeze,
    load_weights,
)
from ccrr.training.teachers import initialize_base
from ccrr.tensor_ops import gray

PANDERM_COMMIT = "fd7a80748ba7fc3e203fed88f909f4689d0d6f24"
PANDERM_SOURCE_SHA256 = (
    "63fbb03283bb5d61ebd24e4790b66f3495996cc37f000f7003eec0a277bd1ab1"
)


def fingerprints(model):
    return {
        k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
        for k, v in model.state_dict().items()
    }


def load_real_model(args, report):
    if args.model == "panderm":
        if not args.source or sha256(args.source) != PANDERM_SOURCE_SHA256:
            raise ValueError(
                "Expected audited PanDerm modeling_finetune.py at fixed commit"
            )
        model = PanDerm(
            args.source, PANDERM_SOURCE_SHA256, args.weights, args.sha256
        ).eval()
        state = torch.load(
            args.weights, map_location="cpu", weights_only=True, mmap=True
        )
        actual = model.model.state_dict()
        assert set(state) == set(actual)
        assert all(torch.equal(v, actual[k]) for k, v in state.items())
        report["load"] = {
            "container": type(state).__name__,
            "backbone_entries": len(state),
            "missing_before_head_removal": ["head.weight", "head.bias"],
            "unexpected": [],
            "all_backbone_tensors_exact": True,
            "source_commit": PANDERM_COMMIT,
            "source_sha256": PANDERM_SOURCE_SHA256,
            "source_path": str(Path(args.source).resolve()),
            "official_full_weight_hash_available": False,
        }
        return model
    base = getattr(models, args.model)(weights=None)
    report["official_1000_class_load"] = load_weights(base, args.weights, args.sha256)
    state = torch.load(args.weights, map_location="cpu", weights_only=True, mmap=True)
    actual = base.state_dict()
    assert set(actual) == set(state)
    assert all(torch.equal(v, actual[k]) for k, v in state.items())
    report["official_state_entries"] = len(state)
    del base, actual
    gc.collect()
    model = (
        ClinicalModel()
        if args.model == "efficientnet_b0"
        else IdentityModel(patients=863)
    )
    context = SimpleNamespace(
        cfg={"weights": {args.model: {"path": args.weights, "sha256": args.sha256}}},
        device=torch.device("cpu"),
    )
    initialize_base(context, model, args.model)
    prefix = "classifier.1." if args.model == "efficientnet_b0" else "fc."
    keys = [k for k in state if not k.startswith(prefix)]
    assert all(torch.equal(state[k], model.model.state_dict()[k]) for k in keys)
    report["production_load"] = {
        "backbone_entries_exact": len(keys),
        "task_heads": "New untrained task heads; no capability claim",
    }
    return freeze(model)


def check_inputs(model, kind):
    cases = []
    rgb = torch.rand(1, 3, 512, 512, device=next(model.parameters()).device)
    for rect in [(0, 0, 512, 512), (0, 64, 512, 384)]:
        source = rgb.clone()
        if rect[1]:
            source[:, :, :64] = 0
            source[:, :, 448:] = 0
        for domain, value in (("rgb", source), ("gray", gray(source))):
            x = value.detach().clone().requires_grad_(True)
            case = {"domain": domain, "valid_rect": rect, "input_shape": list(x.shape)}
            if kind == "panderm":
                feature, coverage, transform = model(x, [rect])
                assert feature.shape == (1, 768, 14, 14)
                objective = feature[:, :16].square().mean()
                captured = []
                hook = model.model.norm.register_forward_hook(
                    lambda module, inp, result: captured.append(result.detach())
                )
                try:
                    with torch.no_grad():
                        a, _, _ = adapt(x.detach(), [rect], 224, panderm=True)
                        cls = model.model.forward_features(a, is_train=False)
                finally:
                    hook.remove()
                patches = captured[0][:, 1:].transpose(1, 2).reshape(1, 768, 14, 14)
                assert torch.equal(patches, feature.detach())
                assert torch.equal(cls, captured[0][:, 0])
                case["official_norm_patch_tokens_exact"] = True
                case["feature_variance"] = float(feature.detach().var())
                case["effective_tokens"] = int((coverage > 0).sum())
            elif kind == "efficientnet_b0":
                logits, feature, coverage, transform = model(x, [rect], spatial=True)
                assert logits.shape == (1, 6) and feature.shape == (1, 1280, 16, 16)
                assert torch.isfinite(logits).all()
                objective = logits.square().mean()
                case["logits_shape"] = list(logits.shape)
            else:
                result = model(x, [rect])
                feature, coverage, transform = (
                    result["spatial"],
                    result["coverage"],
                    result["transform"],
                )
                assert feature.shape == (1, 2048, 16, 16)
                assert result["embedding"].shape == (1, 2048)
                assert result["logits"].shape == (1, 863)
                assert all(
                    torch.isfinite(v).all()
                    for v in result.values()
                    if torch.is_tensor(v)
                )
                norm = result["embedding"].norm(dim=1)
                torch.testing.assert_close(norm, torch.ones(1), atol=1e-6, rtol=1e-6)
                case["embedding_l2_norm"] = float(norm.detach()[0])
                objective = result["embedding"][:, :16].sum()
            assert torch.isfinite(feature).all() and torch.isfinite(coverage).all()
            assert coverage.shape == (1, 1, *feature.shape[-2:])
            assert ((coverage >= 0) & (coverage <= 1)).all()
            grad = torch.autograd.grad(objective, x)[0]
            assert torch.isfinite(grad).all() and grad.abs().sum() > 0
            assert all(
                not p.requires_grad and p.grad is None for p in model.parameters()
            )
            case.update(
                features_shape=list(feature.shape),
                coverage_shape=list(coverage.shape),
                coverage_min=float(coverage.min()),
                coverage_max=float(coverage.max()),
                input_gradient_l1=float(grad.abs().sum()),
                input_gradient_max=float(grad.abs().max()),
                transform=transform,
                finite=True,
                frozen_parameter_gradients_absent=True,
            )
            cases.append(case)
            print("INTERFACE_PASS:", kind, domain, rect, flush=True)
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", choices=["efficientnet_b0", "resnet50", "panderm"], required=True
    )
    p.add_argument("--weights", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument(
        "--source", help="Only PanDerm: audited official modeling_finetune.py"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    seed_everything(20261008)
    start = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    report = {
        "scope": "Real weights; synthetic inputs; no medical capability acceptance",
        "model": args.model,
        "device": args.device,
        "environment": environment(),
        "batch_size": 1,
        "weights": str(Path(args.weights).resolve()),
        "sha256": args.sha256,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "adapter_source_sha256": sha256(root / "src/ccrr/teachers/adapters.py"),
        "teacher_input_sources": teacher_input_sources(),
        "teacher_resize_policy": POLICY,
        "checker_source_sha256": sha256(__file__),
        "torch": torch.__version__,
        "formal_training_started": False,
        "teacher_capability_validated": False,
        "H100_validated": False,
    }
    try:
        assert sha256(args.weights) == args.sha256, "Weight hash mismatch"
        report["weight_bytes"] = Path(args.weights).stat().st_size
        if args.device == "cuda":
            assert torch.cuda.is_available(), "CUDA unavailable"
            torch.cuda.reset_peak_memory_stats()
        model = load_real_model(args, report).to(args.device)
        before = fingerprints(model)
        report["forward_backward"] = check_inputs(model, args.model)
        assert fingerprints(model) == before and not model.training
        report.update(passed=True, parameters_and_buffers_unchanged=True)
        if args.device == "cuda":
            torch.cuda.synchronize()
            report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["H100_validated"] = "H100" in torch.cuda.get_device_name(0)
            report["H100_validation_scope"] = (
                "batch-1 interface only; full training memory/throughput pending"
            )
    except Exception as exc:
        report.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("REPORT:", target, flush=True)


if __name__ == "__main__":
    main()
