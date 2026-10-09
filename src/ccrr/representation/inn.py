import math
import torch
from torch import nn
from ccrr.tensor_ops import squeeze, unsqueeze, pack, unpack, finite


class ReversibleBlock(nn.Module):
    def __init__(self, c, width, swap):
        super().__init__()
        self.B = nn.Parameter(torch.zeros(c, c))
        self.swap = swap
        self.net = nn.Sequential(
            nn.Conv2d(c // 2, width, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width, width, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width, c, 3, padding=1),
        )
        for m in (self.net[0], self.net[2]):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def matrix(self):
        d = self.B - self.B.T
        a = d / (2 * torch.linalg.vector_norm(d).clamp_min(1))
        eye = torch.eye(len(d), device=d.device, dtype=d.dtype)
        return torch.linalg.solve(eye - a, eye + a)

    def forward(self, x, inverse=False):
        w = self.matrix()
        if not inverse:
            x = torch.einsum("ij,bjhw->bihw", w, x)
        a, b = x.chunk(2, 1)
        if self.swap:
            a, b = b, a
        s, t = self.net(a).chunk(2, 1)
        s = 0.5 * s.tanh()
        t = 2 * t.tanh()
        b = (b - t) * (-s).exp() if inverse else b * s.exp() + t
        out = torch.cat((b, a) if self.swap else (a, b), 1)
        return torch.einsum("ji,bjhw->bihw", w, out) if inverse else out


class INN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        q = torch.tensor(
            [
                [1 / math.sqrt(3)] * 3,
                [1 / math.sqrt(2), 0, -1 / math.sqrt(2)],
                [1 / math.sqrt(6), -2 / math.sqrt(6), 1 / math.sqrt(6)],
            ]
        )
        self.register_buffer("q0", q)
        self.levels = nn.ModuleList(
            [
                nn.ModuleList([ReversibleBlock(c, w, j % 2 == 1) for j in range(4)])
                for c, w in ((12, 64), (48, 96), (192, 128))
            ]
        )

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            finite(x, "INN input")
            x = torch.einsum("ij,bjhw->bihw", self.q0, 2 * x - 1)
            for blocks in self.levels:
                x = squeeze(x)
                for block in blocks:
                    x = block(x)
            finite(x, "INN output")
            return unpack(x[:, :64]), unpack(x[:, 64:])

    def inverse(self, u, z):
        with torch.autocast(device_type=u.device.type, enabled=False):
            x = torch.cat((pack(u.float()), pack(z.float())), 1)
            for blocks in reversed(self.levels):
                for block in reversed(blocks):
                    x = block(x, inverse=True)
                x = unsqueeze(x)
            x = (torch.einsum("ji,bjhw->bihw", self.q0, x) + 1) / 2
            finite(x, "INN inverse")
            return x


class FixedGrayDecomposition(nn.Module):
    def forward(self, x):
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return y, torch.cat((x[:, 0:1] - y, x[:, 2:3] - y), 1)

    def inverse(self, g, z):
        return torch.cat(
            (
                g + z[:, 0:1],
                g - (0.299 * z[:, 0:1] + 0.114 * z[:, 1:2]) / 0.587,
                g + z[:, 1:2],
            ),
            1,
        )
