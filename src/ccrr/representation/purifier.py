import torch
from torch import nn
from torch.nn import functional as F
from ccrr.tensor_ops import ResBlock, ste_round, pooled, init_comm


class MaskedNet(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.start = nn.Conv2d(cin, 128, 1)
        self.blocks = nn.ModuleList([ResBlock(128), ResBlock(128)])
        self.end = nn.Conv2d(128, cout, 1)

    def forward(self, x, v):
        m = (v > 0).to(x)
        x = self.start(x) * m
        for block in self.blocks:
            x = block(x, v)
        return self.end(x) * m


class Purifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MaskedNet(768, 64)
        self.decoder = MaskedNet(64, 768)
        self.mask_head = nn.Conv2d(64, 64, 3, padding=1)
        self.identity = MaskedNet(64, 128)
        self.id_head = nn.Linear(128, 2048)
        self.clinical = nn.Linear(768, 6)
        self.register_buffer("mean", torch.zeros(1, 768, 1, 1))
        self.register_buffer("std", torch.ones(1, 768, 1, 1))
        self.register_buffer("stats_ready", torch.tensor(False))
        self.apply(init_comm)

    def forward(self, features, v, source_mask=None):
        m = (v > 0).to(features)
        f = (features - self.mean) / self.std.clamp_min(1e-3) * m
        a = 7 * self.encoder(f, v).tanh()
        q = ste_round(a) * m
        logits = self.mask_head(q / 7)
        p = logits.sigmoid()
        hard = (logits > 0).to(q) * m
        gate = (p + (hard - p).detach()) * m if source_mask is None else source_mask * m
        semantic = self.decoder((1 - gate) * q / 7, v)
        return {
            "standardized": f,
            "q": q,
            "a": a,
            "mask": gate,
            "hard_mask": hard,
            "all": self.decoder(q / 7, v),
            "semantic": semantic,
            "identity": F.normalize(
                self.id_head(pooled(self.identity(gate * q / 7, v), v)), dim=1, eps=1e-6
            ),
            "logits": self.clinical(pooled(semantic, v)),
        }


class SymmetricHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 2, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, a, b):
        return self.net(torch.cat(((a - b).abs(), a * b), 1)).squeeze(1)


class SemanticAttacker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MaskedNet(768, 128)
        self.embed = nn.Sequential(nn.Linear(640, 256), nn.GELU())
        self.pair = SymmetricHead()

    def encode(self, f, v):
        x = self.net(f, v)
        pieces = [pooled(x, v)]
        for a in range(2):
            for b in range(2):
                pieces.append(
                    pooled(
                        x[:, :, a * 7 : (a + 1) * 7, b * 7 : (b + 1) * 7],
                        v[:, :, a * 7 : (a + 1) * 7, b * 7 : (b + 1) * 7],
                    )
                )
        return self.embed(torch.cat(pieces, 1))

    def forward(self, fa, va, fb, vb):
        return self.pair(self.encode(fa, va), self.encode(fb, vb))
