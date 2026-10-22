"""V4 additions around the unchanged author E/D. No new generator network."""

import math
import torch
from torch.nn import functional as F
from .original import quantize8


def digital_supervision(g):
    """Hard author 8bit values forward, identity derivative only for new losses.

    This STE is a V4 surrogate, NOT the author's phase-1 training method.
    Author reconstruction/VGG/TV/grid losses still use continuous g.
    """
    digital = quantize8(g)[1]
    return (g + (digital - g).detach() + 1) / 2


def medical_objective(logits, labels, target, class_weights, kd):
    # Fixed fit-derived weights with an image mean: exact local/global DDP scaling.
    ce = (
        F.cross_entropy(logits, labels, reduction="none") * class_weights[labels]
    ).mean()
    consistency = F.kl_div(
        logits.log_softmax(1), target.detach(), reduction="batchmean"
    )
    return ce + kd * consistency


def retrieval_confusion(query, gallery, indices, temperature):
    """KL(uniform || retrieval softmax) excluding each exact source image.

    Gallery contains fit images only, ordered like the fit data. The current-G1
    gallery is detached and refreshed each epoch; gradients flow through query.
    Flattening this fixed reference is a training proxy, not a privacy guarantee.
    """
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Positive finite retrieval temperature required")
    if len(gallery) < 3 or len(indices) != len(query):
        raise ValueError("Require a nontrivial fit gallery and aligned indices")
    q = F.normalize(query, dim=1)
    bank = F.normalize(gallery.detach(), dim=1)
    scores = q @ bank.T / temperature
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask.scatter_(1, torch.as_tensor(indices, device=q.device)[:, None], False)
    values = scores[mask].reshape(len(q), len(bank) - 1)
    return (values.logsumexp(1) - values.mean(1) - math.log(values.shape[1])).mean()


def encoder_only(cfg):
    return cfg["schema"] == "ccrr-v4-phase2-g1-e1only-1"


def validate_config(cfg):
    if cfg["schema"] not in ("ccrr-v4-phase2-g1-1", "ccrr-v4-phase2-g1-e1only-1"):
        raise ValueError("Unknown G1 profile")
    for key, value in dict(
        image_size=512,
        global_batch_size=16,
        update_modules=["E1"] if encoder_only(cfg) else ["E1", "D1"],
        supervisor_step=8500,
        comparison_step=7000,
        precision="float32",
        medical_acceptance_threshold=None,
        next_phase="stop",
    ).items():
        if cfg[key] != value:
            raise ValueError(f"Registered G1 scope mismatch: {key}")
    ids = cfg["gpu_ids"]
    if not ids or len(set(map(str, ids))) != len(ids) or 16 % len(ids):
        raise ValueError("Unique GPUs must divide global batch 16")
    if any(not isinstance(i, (int, str)) or not str(i) or "," in str(i) for i in ids):
        raise ValueError("Invalid physical GPU ID")
    for key in ("epochs", "cpu_threads", "log_every_steps", "bank_log_every_images"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"Positive integer required: {key}")
    for key in ("medical", "identity", "medical_kd"):
        if not math.isfinite(cfg["supervision"][key]) or cfg["supervision"][key] < 0:
            raise ValueError(f"Nonnegative finite weight required: {key}")
    if not 0 < cfg["retrieval_temperature"] < float("inf"):
        raise ValueError("Positive retrieval temperature required")
    if not 0 < cfg["learning_rate"] < float("inf"):
        raise ValueError("Positive learning rate required")
