"""Untrained engineering smoke: no claimed clinical or communication success."""

from pathlib import Path
import time
import os
import numpy as np
import torch
from .config import Profile
from .system import CCRR, save_public
from .crypto import NonceAllocator, seal
from .records import split_stream, route
from .metadata import FIELDS, serialize
from .losses import message_loss
from .runtime import write_json, environment


def run(out, size=512):
    torch.set_num_threads(2)
    torch.manual_seed(20261008)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    system = CCRR(Profile(size=size), patch_batch=8)
    cdfs = system.entropy.export()
    digest = os.urandom(32)
    system.bind(digest, cdfs)
    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        u, z, g = system.decompose(x)
        rec = system.inn.inverse(u, z)
    rows = dict.fromkeys(FIELDS, "")
    rows.update(
        patient_id="SYNTHETIC", lesion_id="1", img_id="synthetic.png", diagnostic="ACK"
    )
    allocator = NonceAllocator(out / "private/nonce.sqlite")
    credentials = allocator.new_object(digest)
    # Deliberately highly compressible engineering latent, NOT an image-derived latent.
    n = torch.zeros(1, 2, size, size)
    # A concentrated valid integer CDF tests complete digital transport feasibility.
    cdf = np.tile(np.arange(513, dtype=np.int32), (3, 128, 1))
    cdf[:, :, 256:] += 65024
    system.bind(digest, cdf)
    source = system.codec.encode(n, 1)
    c = seal(source, credentials[0], 1, allocator)
    m = seal(serialize(rows), credentials[1], 0, allocator)
    cr, mr = split_stream(c, 0, 1), split_stream(m, 1, 0)
    bits, occ = route(
        cr, mr, [4] * system.profile.regions, np.ones(system.profile.regions)
    )
    bt = torch.tensor(bits)[None]
    ot = torch.tensor(occ)[None]
    legal = torch.ones(1, system.profile.regions)
    # Neural forward + gradient on the same message table. The neural decoder is untrained.
    p, gates = system.comm.write(g.detach(), bt, ot, legal)
    logits = system.comm.read(p)
    loss, _ = message_loss(logits, bt, ot)
    loss.backward()
    reloaded = save_public(p, out / "synthetic-public.png")
    # Oracle records prove digital protocol only, kept separate from actual blind decoding.
    digital = system.receive_records(
        reloaded, cr + mr, {0: [], 1: []}, {0: credentials[0], 1: credentials[1]}
    )
    actual = system.receive(reloaded, {0: credentials[0], 1: credentials[1]})
    report = {
        "scope": "Untrained engineering smoke; synthetic zero latent; no medical claims",
        "size": size,
        "environment": environment(),
        "continuous_max_error": float((rec - x).abs().max()),
        "continuous_rmse": float((rec - x).square().mean().sqrt()),
        "ciphertext_bytes": {"C": len(c), "M": len(m)},
        "records": len(cr) + len(mr),
        "message_loss": float(loss.detach()),
        "writer_gradient_norm": float(system.comm.writer.out.weight.grad.norm()),
        "max_integer_modification": float(((p - g) * 255).abs().max().detach()),
        "digital_oracle_records": {a: str(digital[a].status) for a in ("C", "M")},
        "actual_blind_decoder": {a: str(actual[a].status) for a in ("C", "M")},
        "elapsed_seconds": time.time() - t,
        "training_started": False,
        "formal_experiment_passed": False,
    }
    allocator.close()
    write_json(out / "smoke.json", report)
    print(report)
