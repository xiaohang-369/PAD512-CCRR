"""Fresh independent probes/attackers, selected on val_tune, reported on val_select."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

from ccrr.data.pad import PairSampler
from ccrr.evaluation.metrics import pair_bootstrap, verification
from .models import PairAttacker, pool
from .learning import fit_probe, pair_batch
from .runtime import completed, save_model, seed_all, write

DIRECTIONS = {"rgb_rgb": (0, 0), "gray_rgb": (1, 0), "gray_gray": (1, 1)}


def protocol_pairs(data, scenario):
    protocol = data["protocols"][scenario]
    index = {r["img_id"]: i for i, r in enumerate(data["rows"])}
    counts = Counter(q["patient_id"] for q in protocol["queries"])
    result = []
    for q in protocol["queries"]:
        a = index[q["query"]]
        for label, key in ((1, "positive_gallery"), (0, "negative_gallery")):
            names = q[key]
            if not names:
                raise ValueError("Empty frozen positive/negative group")
            for name in names:
                b = index[name]
                result.append(
                    (
                        a,
                        b,
                        label,
                        1 / counts[q["patient_id"]] / len(names),
                        q["patient_id"],
                        data["rows"][b]["patient_id"],
                    )
                )
    if not result:
        raise ValueError(f"No legal {scenario} pairs")
    return result


def evaluate_attacker(model, data, repetitions=0):
    model.eval()
    encoded = []
    with torch.no_grad():
        for d in (0, 1):
            parts = []
            for start in range(0, len(data["rows"]), 64):
                parts.append(
                    pool(
                        model.encoder(data["features"][start : start + 64, d]),
                        data["coverage"][start : start + 64],
                    )
                )
            encoded.append(torch.cat(parts))
        result = {}
        for scenario in ("p1", "p2"):
            pairs = protocol_pairs(data, scenario)
            labels = np.array([p[2] for p in pairs])
            weights = np.array([p[3] for p in pairs])
            patients = [(p[4], p[5]) for p in pairs]
            for direction, (da, db) in DIRECTIONS.items():
                scores = []
                for start in range(0, len(pairs), 512):
                    batch = pairs[start : start + 512]
                    a = encoded[da][[p[0] for p in batch]]
                    b = encoded[db][[p[1] for p in batch]]
                    scores.extend(
                        model.head(torch.cat(((a - b).abs(), a * b), 1))
                        .flatten()
                        .cpu()
                        .tolist()
                    )
                report = verification(labels, scores, weights)
                if repetitions:
                    from sklearn.metrics import roc_auc_score

                    report["auroc_ci"] = pair_bootstrap(
                        patients,
                        lambda w: roc_auc_score(
                            labels, scores, sample_weight=weights * w
                        ),
                        repetitions,
                    )
                report["score_details"] = {
                    "labels": labels.tolist(),
                    "scores": scores,
                    "weights": weights.tolist(),
                    "pairs": patients,
                }
                report["pair_order"] = "frozen_protocol_positive_then_negative"
                result[f"{scenario}/{direction}"] = report
        return result


def fit_attacker(cache, cfg, output, spatial):
    seed_all(cfg["seed"] + 100)
    fit = cache["fit"]
    model = PairAttacker(fit["features"].shape[2], spatial).to(fit["features"].device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["attacker_lr"])
    sampler = PairSampler(SimpleNamespace(rows=fit["rows"]))
    rng = np.random.default_rng(cfg["seed"] + 100)
    best, selected, best_step = -float("inf"), None, None
    for step in range(1, cfg["attacker_steps"] + 1):
        model.train()
        a, av, b, bv, labels = pair_batch(fit, sampler, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(a, av, b, bv), labels
        )
        loss.backward()
        optimizer.step()
        if (
            step % max(1000, cfg["evaluate_every"]) == 0
            or step == cfg["attacker_steps"]
        ):
            reports = evaluate_attacker(model, cache["val_tune"])
            score = float(np.mean([r["a_dagger"] for r in reports.values()]))
            if score > best:
                best, selected, best_step = score, deepcopy(model.state_dict()), step
            print(
                f"AUDIT_ATTACK spatial={spatial} {step}/{cfg['attacker_steps']} tune={score:.6f}",
                flush=True,
            )
    model.load_state_dict(selected)
    save_model(
        output / "attacker.pt",
        model,
        best_step,
        {"spatial": spatial, "selection_role": "val_tune"},
    )
    report = evaluate_attacker(model, cache["val_select"], cfg["bootstrap_repetitions"])
    write(output / "identity.json", report)
    return report


def audit(cache, cfg, output):
    # Distinct directories and fresh seeds: these heads never update S0.
    for name, is_mlp in (("linear", False), ("mlp", True)):
        target = output / name
        target.mkdir()
        fit_probe(cache, cfg, target, mlp=is_mlp)
    for name, spatial in (("global", False), ("spatial", True)):
        target = output / name
        target.mkdir()
        fit_attacker(cache, cfg, target, spatial)
    completed(
        output,
        audit_scope="independent_linear_mlp_global_spatial_P1_P2",
        incomplete_claims=[
            "lesion",
            "concept",
            "ReID_cross_representation_attack",
            "final_image_privacy",
        ],
        note="No automatic acceptance; compare matched raw/medical/joint reports and paired intervals",
    )
