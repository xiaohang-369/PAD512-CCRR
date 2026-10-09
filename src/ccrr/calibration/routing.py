"""Paired reliability-sort selection with identical ciphertexts and no retries."""

from collections import defaultdict
import numpy as np
import torch
from ccrr.communication.records import route
from ccrr.evaluation.metrics import quality, cluster_bootstrap
from ccrr.errors import CCRRError, Status
from ccrr.evaluation.comparisons import paired_clinical
from ccrr.data.pad import collate


def choose_routing(experiment):
    e = experiment
    system = e.system
    ds = e.data["val_tune"]
    pairs = []
    clin = {False: [], True: []}
    successful = {False: {}, True: {}}
    with torch.no_grad():
        for i in range(len(ds)):
            b = collate([ds[i]], e.device)
            name = b["id"][0]
            budgets, rel = e.policies[name]
            u, z, g = system.decompose(b["x"])
            per = {False: [], True: []}
            for repeat in range(3):
                creds = e.allocator.new_object(system.digest)
                try:
                    payload = system.payload(
                        z, b["metadata"][0], creds, e.allocator, budgets
                    )
                except CCRRError as exc:
                    if exc.status != Status.CAPACITY:
                        raise
                    per[False].append(0)
                    per[True].append(0)
                    continue
                for sorted_policy in (False, True):
                    bits, occ = route(
                        payload["c_records"],
                        payload["m_records"],
                        budgets,
                        rel,
                        sorted_policy,
                    )
                    public, _ = system.comm.write(
                        g,
                        torch.from_numpy(bits)[None].to(g),
                        torch.from_numpy(occ)[None].to(g),
                        b["legal"],
                    )
                    result = system.receive(public, {0: creds[0], 1: creds[1]})
                    cs = result["C"].status == Status.SUCCESS
                    ms = result["M"].status == Status.SUCCESS
                    per[sorted_policy].append(int(cs and ms))
                    probs = e.clinical(public, b["rect"]).softmax(1)[0].cpu().numpy()
                    clin[sorted_policy].append(
                        {
                            "id": f"{name}/{repeat}",
                            "patient": b["patient"][0],
                            "label": int(b["label"][0]),
                            "p": probs,
                        }
                    )
                    if cs:
                        successful[sorted_policy][name, repeat] = quality(
                            b["x"], result["C"].value["rgb_unclipped"], b["v"]
                        )[0]["psnr"]
            pairs.append(
                {
                    "patient": b["patient"][0],
                    "delta": np.mean(per[True]) - np.mean(per[False]),
                }
            )
    groups = defaultdict(list)
    for r in pairs:
        groups[r["patient"]].append(r["delta"])
    pids = sorted(groups)
    deltas = np.array([np.mean(groups[p]) for p in pids])
    delta = float(deltas.mean())
    ci = cluster_bootstrap(pids, lambda w: np.average(deltas, weights=w))

    def summary(rows):
        return {
            "ids": [r["id"] for r in rows],
            "patients": [r["patient"] for r in rows],
            "labels": [r["label"] for r in rows],
            "probabilities": [r["p"].tolist() for r in rows],
        }

    clinical_comparison = paired_clinical(
        summary(clin[False]), summary(clin[True]), auroc_margin=0.01, f1_margin=1
    )
    shared = set(successful[False]) & set(successful[True])
    psnr = (
        float(np.mean([successful[True][k] - successful[False][k] for k in shared]))
        if shared
        else None
    )
    enabled = bool(
        delta >= 0.005
        and ci["evidence_sufficient"]
        and ci["lower"] is not None
        and ci["lower"] > 0
        and clinical_comparison.get("auroc_difference", -1) >= -0.01
        and psnr is not None
        and psnr >= -0.2
    )
    return {
        "enabled": enabled,
        "success_improvement": delta,
        "success_ci": ci,
        "clinical_comparison": clinical_comparison,
        "common_success_psnr_difference": psnr,
        "repeats_per_image": 3,
        "same_ciphertext_pairing": True,
    }
