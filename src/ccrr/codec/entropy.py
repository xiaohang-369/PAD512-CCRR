from ccrr.codec.quantization import quantize_latent, dequantize_latent
import struct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ccrr.tensor_ops import pack, unpack
from ccrr.errors import require, Status
from ccrr.codec import rans

SOURCE = struct.Struct("<4sBBHII32s")


class EntropyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(3, 128, 3))
        self.d = nn.Parameter(torch.zeros(3, 128, 3))
        self.b = nn.Parameter(
            torch.log(torch.expm1(torch.tensor([0.25, 1.0, 4.0]) - 0.05))
            .expand(3, 128, 3)
            .clone()
        )

    def log_pmf(self, n, level):
        k = n.unsqueeze(-1)
        s = 0.05 + F.softplus(self.b[level - 1])[None, :, None, None, :]
        mu = 64 * self.d[level - 1].tanh()[None, :, None, None, :]
        logs = (
            F.logsigmoid((k + 0.5 - mu) / s)
            + F.logsigmoid(-(k - 0.5 - mu) / s)
            + torch.log(-torch.expm1(-1 / s))
        )
        return torch.logsumexp(
            F.log_softmax(self.a[level - 1], -1)[None, :, None, None, :] + logs, -1
        )

    def rate(self, z, level, step):
        n = pack(quantize_latent(z, step))
        return (
            -self.log_pmf(n, level).sum((1, 2, 3))
            / np.log(2)
            / (z.shape[-1] * z.shape[-2])
        )

    @torch.no_grad()
    def export(self):
        # Use a detached CPU FP64 copy, never mutate the live optimizer's model.
        m = EntropyModel().double()
        m.load_state_dict(
            {k: v.detach().cpu().double() for k, v in self.state_dict().items()}
        )
        tables = []
        for r in (1, 2, 3):
            k = torch.arange(-255, 256, dtype=torch.float64)[
                None, None, :, None
            ].expand(1, 128, -1, 1)
            regular = m.log_pmf(k, r).exp()[0, :, :, 0]
            s = 0.05 + F.softplus(m.b[r - 1])
            mu = 64 * m.d[r - 1].tanh()
            w = m.a[r - 1].softmax(-1)
            tail = (
                w * (torch.sigmoid((-255.5 - mu) / s) + torch.sigmoid((mu - 255.5) / s))
            ).sum(-1)
            p = torch.cat((regular, tail[:, None]), 1).numpy()
            require(
                np.isfinite(p).all() and (p >= 0).all(), "Invalid entropy probability"
            )
            p /= p.sum(1, keepdims=True)
            scaled = p * (65536 - 512)
            f = 1 + np.floor(scaled).astype(np.int64)
            for i in range(128):
                rem = int(65536 - f[i].sum())
                require(0 <= rem <= 511, "CDF remainder")
                order = np.lexsort((np.arange(512), -(scaled[i] - np.floor(scaled[i]))))
                f[i, order[:rem]] += 1
            cdf = np.concatenate(
                (np.zeros((128, 1), dtype=np.int64), f.cumsum(1)), 1
            ).astype(np.int32)
            rans.validate_cdfs(cdf)
            tables.append(cdf)
        return np.stack(tables)


class SourceCodec:
    def __init__(self, profile, digest, cdfs):
        self.profile, self.digest = profile, digest
        self.cdfs = np.asarray(cdfs)
        require(self.cdfs.shape == (3, 128, 513), "CDF resource shape")
        for table in self.cdfs:
            rans.validate_cdfs(table)
        self.indexes = np.repeat(np.arange(128), (profile.size // 8) ** 2).tolist()

    def quantize(self, z, level):
        n = quantize_latent(z, self.profile.step(level))
        require(
            torch.isfinite(n).all() and n.abs().max() <= self.profile.symbol_bound,
            "Latent out of range",
        )
        return n

    def encode(self, n, level):
        require(
            tuple(n.shape) == (1, 2, self.profile.size, self.profile.size),
            "Source latent shape",
        )
        flat = pack(n).detach().cpu().numpy().reshape(-1)
        require(
            np.isfinite(flat).all() and np.equal(flat, np.round(flat)).all(),
            "Noninteger latent",
        )
        body = rans.encode(flat.tolist(), self.indexes, self.cdfs[level - 1])
        return (
            SOURCE.pack(b"ZC01", 1, level, 0, len(flat), len(body), self.digest) + body
        )

    def decode(self, plain, level, device="cpu"):
        require(len(plain) >= 56, "Short source", Status.DECODE)
        magic, v, r, flags, count, length, dg = SOURCE.unpack(plain[:48])
        require(
            (magic, v, r, flags, count, dg)
            == (b"ZC01", 1, level, 0, self.profile.symbols, self.digest),
            "Source header",
            Status.DECODE,
        )
        require(
            level in self.profile.legal_levels and length == len(plain) - 48,
            "Source length/level",
            Status.DECODE,
        )
        n = rans.decode(plain[48:], self.indexes, self.cdfs[level - 1])
        packed = torch.from_numpy(n.reshape((1, *self.profile.packed_shape))).to(
            device=device
        )
        return unpack(packed), dequantize_latent(
            unpack(packed.float()), self.profile.step(level)
        )
