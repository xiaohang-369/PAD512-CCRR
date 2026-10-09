import torch
from torch import nn
from torch.nn import functional as F
from ccrr.tensor_ops import ResBlock, pos1, pos2, haar, ihaar, q8, init_comm


def patches(x):
    b, c, h, w = x.shape
    return (
        x.unfold(2, 32, 32)
        .unfold(3, 32, 32)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(-1, c, 32, 32)
    )


def assemble(x, b, size):
    n = size // 32
    return (
        x.reshape(b, n, n, 1, 32, 32)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(b, 1, size, size)
    )


class SpatialUnit(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(d, 4, dropout=0, batch_first=True) for _ in range(2)]
        )
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                for _ in range(2)
            ]
        )

    def grouped(self, x, attention, grid):
        b, h, w, d = x.shape
        a = h // 4
        c = w // 4
        x = x.reshape(b, a, 4, c, 4, d)
        if grid:
            x = x.permute(0, 2, 4, 1, 3, 5).reshape(b * 16, a * c, d)
            x = attention(x, x, x, need_weights=False)[0]
            x = x.reshape(b, 4, 4, a, c, d).permute(0, 3, 1, 4, 2, 5)
        else:
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(b * a * c, 16, d)
            x = attention(x, x, x, need_weights=False)[0]
            x = x.reshape(b, a, c, 4, 4, d).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(b, h, w, d)

    def forward(self, x):
        for j in range(2):
            x = x + self.grouped(self.norm[2 * j](x), self.attn[j], bool(j))
            x = x + self.mlp[j](self.norm[2 * j + 1](x))
        return x


class CarrierEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(1, 128, 32, stride=32)
        self.units = nn.Sequential(SpatialUnit(), SpatialUnit())
        self.norm = nn.LayerNorm(128)
        self.apply(init_comm)

    def forward(self, g):
        x = self.patch(g).permute(0, 2, 3, 1)
        b, h, w, d = x.shape
        x = x + pos2(h, w, d, x.device).reshape(1, h, w, d)
        return self.norm(self.units(x)).reshape(b, h * w, d)


class BudgetHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.clin = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.capacity = nn.Sequential(nn.Linear(144, 64), nn.GELU(), nn.Linear(64, 5))
        self.reliability = nn.Sequential(
            nn.Linear(129, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.apply(init_comm)

    def forward(self, z, risk, k=None):
        z = self.norm(z)
        cap = self.capacity(torch.cat((z, self.clin(risk[..., None])), -1))
        if k is None:
            k = cap.argmax(-1)
        rel = (
            self.reliability(torch.cat((z, k[..., None].to(z) / 4), -1))
            .squeeze(-1)
            .sigmoid()
        )
        return cap, rel


class LocalWriter(nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.GELU(), ResBlock(32))
        self.w1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(), ResBlock(64)
        )
        self.w2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(), ResBlock(128)
        )
        self.record = nn.Sequential(nn.Linear(172, 128), nn.GELU(), nn.Linear(128, 128))
        self.z = nn.Linear(128, 128)
        self.qnorm = nn.LayerNorm(128)
        self.knorm = nn.LayerNorm(128)
        self.attn = nn.MultiheadAttention(128, 4, batch_first=True, dropout=0)
        self.mnorm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
        self.fuse = nn.Sequential(nn.Conv2d(256, 128, 1), nn.GELU(), ResBlock(128))
        self.up = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1), nn.GELU(), ResBlock(64)
        )
        self.out = nn.Conv2d(64, 4, 3, padding=1)
        self.znorm = nn.LayerNorm(128)
        self.gate = nn.Sequential(nn.Linear(259, 128), nn.GELU(), nn.Linear(128, 4))
        self.apply(init_comm)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, g, z, bits, occ, uniform_gate=False):
        a = self.w1(self.w0(g))
        b = self.w2(a)
        h = (
            b.flatten(2).transpose(1, 2)
            + self.z(z)[:, None, :]
            + pos2(8, 8, 128, g.device)[None]
        )
        inp = torch.cat((occ[..., None], occ[..., None] * (2 * bits - 1)), -1)
        tokens = (self.record(inp) + pos1(4, 128, g.device)[None]) * occ[..., None]
        total = torch.zeros((len(g), 4, 16, 16), device=g.device, dtype=g.dtype)
        active_count = torch.zeros(len(g), device=g.device)
        all_gates = []
        for alpha in (0, 1):
            valid = (occ > 0) & (bits[..., 2] == alpha)
            active = valid.any(1)
            coeff = total.new_zeros(total.shape)
            gates = z.new_zeros((len(g), 4))
            if active.any():
                idx = active.nonzero().flatten()
                hh = h[idx]
                kv = self.knorm(tokens[idx])
                att = self.attn(
                    self.qnorm(hh),
                    kv,
                    kv,
                    key_padding_mask=~valid[idx],
                    need_weights=False,
                )[0]
                hh = hh + att
                hh = hh + self.mlp(self.mnorm(hh))
                fused = self.fuse(
                    torch.cat((hh.transpose(1, 2).reshape(-1, 128, 8, 8), b[idx]), 1)
                )
                u = self.out(
                    self.up(
                        torch.cat(
                            (
                                F.interpolate(fused, scale_factor=2, mode="nearest"),
                                a[idx],
                            ),
                            1,
                        )
                    )
                ).tanh()
                typecode = z.new_zeros((len(idx), 2))
                typecode[:, alpha] = 1
                gatein = torch.cat(
                    (
                        self.znorm(z[idx]),
                        b[idx].mean((2, 3)),
                        typecode,
                        valid[idx].sum(1, keepdim=True).to(z) / 4,
                    ),
                    1,
                )
                q = self.gate(gatein).softmax(-1)
                if uniform_gate:
                    q = q * 0 + 0.25
                coeff[idx] = q[:, :, None, None] * u
                gates[idx] = q
            total = total + coeff
            active_count = active_count + active.to(z)
            all_gates.append(gates)
        total = total * (2 / active_count.clamp_min(1))[:, None, None, None]
        return ihaar(total), torch.stack(all_gates, 1)


class LocalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d0 = nn.Sequential(nn.Conv2d(4, 64, 3, padding=1), nn.GELU(), ResBlock(64))
        self.d1 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU())
        self.z = nn.Linear(128, 128)
        self.blocks = nn.Sequential(ResBlock(128), ResBlock(128))
        self.d2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GELU(), ResBlock(128)
        )
        self.d3 = nn.Sequential(nn.Flatten(), nn.Linear(2048, 512), nn.GELU())
        self.heads = nn.ModuleList(
            [nn.Linear(512, 4 * n) for n in (1, 4, 2, 4, 10, 10, 18, 128)]
        )
        self.apply(init_comm)

    def forward(self, g, z):
        x = self.d1(self.d0(haar(g))) + self.z(z)[:, :, None, None]
        h = self.d3(self.d2(self.blocks(x)))
        return torch.cat(
            [
                head(h).reshape(-1, 4, n)
                for head, n in zip(self.heads, (1, 4, 2, 4, 10, 10, 18, 128))
            ],
            -1,
        )


class Communication(nn.Module):
    def __init__(self, patch_batch=16):
        super().__init__()
        self.carrier = CarrierEncoder()
        self.writer = LocalWriter()
        self.decoder = LocalDecoder()
        self.heads = BudgetHeads()
        self.patch_batch = patch_batch

    def write(self, g, bits, occ, legal, delta=4 / 255, uniform_gate=False):
        b, _, s, _ = g.shape
        z = self.carrier(g).reshape(-1, 128)
        p = patches(g)
        bits = bits.reshape(-1, 4, 171)
        occ = occ.reshape(-1, 4)
        legal = legal.reshape(-1)
        outputs = []
        gates = []
        for start in range(0, len(p), self.patch_batch):
            sl = slice(start, start + self.patch_batch)
            r, q = self.writer(p[sl], z[sl], bits[sl], occ[sl], uniform_gate)
            outputs.append(r * legal[sl, None, None, None] * delta)
            gates.append(q)
        residual = assemble(torch.cat(outputs), b, s)
        return q8(g + residual), torch.cat(gates).reshape(b, -1, 2, 4)

    def read(self, g):
        b, _, _, _ = g.shape
        z = self.carrier(g).reshape(-1, 128)
        p = patches(g)
        out = [
            self.decoder(p[i : i + self.patch_batch], z[i : i + self.patch_batch])
            for i in range(0, len(p), self.patch_batch)
        ]
        return torch.cat(out).reshape(b, -1, 4, 177)
