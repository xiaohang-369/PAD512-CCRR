"""Lossy-channel failures and capability matrix, with no ECC or hidden retries."""

import io
import torch
import numpy as np
from PIL import Image


def channels(g):
    a = torch.floor(g[0, 0].detach().cpu() * 255 + 0.5).byte().numpy()
    result = {"lossless": g}
    for quality in (95, 85, 70):
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as im:
            b = np.asarray(im.convert("L")).copy()
        result[f"jpeg-{quality}"] = torch.from_numpy(b)[None, None].to(g) / 255
    for ratio in (0.75, 0.5):
        im = Image.fromarray(a)
        small = im.resize(
            (round(a.shape[1] * ratio), round(a.shape[0] * ratio)),
            Image.Resampling.BILINEAR,
        )
        b = np.asarray(small.resize(im.size, Image.Resampling.BILINEAR)).copy()
        result[f"resize-{ratio}"] = torch.from_numpy(b)[None, None].to(g) / 255
    return result


@torch.no_grad()
def capability_matrix(system, g, kc, km):
    from dataclasses import replace
    import secrets

    variants = {
        "none": {},
        "C": {0: kc},
        "M": {1: km},
        "both": {0: kc, 1: km},
        "wrong_C": {0: replace(kc, key=secrets.token_bytes(32)), 1: km},
        "wrong_M": {0: kc, 1: replace(km, key=secrets.token_bytes(32))},
    }
    results = {}
    for name, creds in variants.items():
        r = system.receive(g, creds)
        results[name] = {a: str(r[a].status) for a in ("C", "M")}
    return results


@torch.no_grad()
def evaluate_channels(system, g, credentials, clinical=None, rects=None):
    out = {}
    for name, image in channels(g).items():
        result = system.receive(image, credentials)
        out[name] = {
            "C": str(result["C"].status),
            "M": str(result["M"].status),
            "retries": 0,
            "ecc": False,
        }
        if clinical is not None:
            out[name]["public_probabilities"] = (
                clinical(image, rects).softmax(1).cpu().tolist()
            )
    return out
