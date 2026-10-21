"""PyTorch port of Xia et al., model.py at 5dc2a0f.

Derived from InvertibleGrayscale, Copyright (c) 2018 CUHK and the authors.
The complete original notice/license is in ../upstream/InvertibleGrayscale/LICENSE.txt.
Names match TensorFlow scopes; parameters are independent E/D CNNs, not an INN.
No public/private head, residual-code branch, encryption or Writer is added here.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


def same_pad(x, kernel, stride):
    h, w = x.shape[-2:]
    ph = max((math.ceil(h / stride) - 1) * stride + kernel - h, 0)
    pw = max((math.ceil(w / stride) - 1) * stride + kernel - w, 0)
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


class Conv(nn.Module):
    def __init__(self, cin, cout, scope, kernel=3, stride=1, relu=False):
        super().__init__()
        self.tf_scope, self.stride, self.kernel, self.relu = scope, stride, kernel, relu
        self.weight = nn.Parameter(torch.empty(cout, cin, kernel, kernel))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        y = F.conv2d(
            same_pad(x, self.kernel, self.stride), self.weight, stride=self.stride
        )
        return F.relu(y) if self.relu else y


class Residual(nn.Module):
    def __init__(self, channels, scope, index):
        super().__init__()
        self.c1 = Conv(channels, channels, f"{scope}/c1/{index}", relu=True)
        self.c2 = Conv(channels, channels, f"{scope}/c2/{index}")

    def forward(self, x):
        return x + self.c2(self.c1(x))


def blocks(channels, count, scope):
    return nn.Sequential(*(Residual(channels, scope, i) for i in range(count)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = Conv(3, 64, "encode/in/k3n64s1")
        self.start = blocks(64, 2, "encode/sen64s1")
        self.down1 = Conv(64, 128, "encode/down1-1/k3n128s2", stride=2)
        self.down1b = Conv(128, 128, "encode/down1-2/k3n128s1", relu=True)
        self.down2 = Conv(128, 256, "encode/down2-1/k3n256s2", stride=2)
        self.down2b = Conv(256, 256, "encode/down2-2/k3n256s1", relu=True)
        self.middle = blocks(256, 4, "encode/en64s1")
        self.up1 = Conv(256, 128, "encode/up1-1/k3n128s1")
        self.up1b = Conv(128, 128, "encode/up1-2/k3n128s1", relu=True)
        self.up2 = Conv(128, 64, "encode/up2-1/k3n64s1")
        self.up2b = Conv(64, 64, "encode/up2-2/k3n64s1", relu=True)
        self.end = blocks(64, 2, "encode/een64s1")
        self.latent = Conv(64, 1, "encode/latent")

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or any(n % 4 for n in x.shape[-2:]):
            raise ValueError(
                "Encoder requires NCHW RGB, height/width multiples of four"
            )
        n1 = self.start(self.input(x))
        n2 = self.down1b(self.down1(n1))
        n = self.middle(self.down2b(self.down2(n2)))
        # TensorFlow resize_images(method=1), exact integer factor 2.
        n = self.up1b(self.up1(F.interpolate(n, scale_factor=2, mode="nearest"))) + n2
        n = self.up2b(self.up2(F.interpolate(n, scale_factor=2, mode="nearest"))) + n1
        return torch.tanh(self.latent(self.end(n)))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = Conv(1, 64, "decode/in/k3n64s1")
        self.blocks = blocks(64, 8, "decode/dn64s1")
        self.expand = Conv(64, 256, "decode/n256s1/2")
        self.output = Conv(256, 3, "decode/out", kernel=1)

    def forward(self, g):
        if g.ndim != 4 or g.shape[1] != 1:
            raise ValueError("Original decoder accepts only one grayscale channel")
        return torch.tanh(self.output(self.expand(self.blocks(self.input(g)))))


def quantize8(g):
    """Original PNG rounding, [-1,1] -> uint8 -> [-1,1]; no STE by default."""
    integers = torch.floor((g.clamp(-1, 1) + 1) * 127.5 + 0.5).to(torch.uint8)
    return integers, integers.to(g.dtype) / 127.5 - 1


class OriginalGrayscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = Encoder(), Decoder()

    def forward(self, x):
        # Original training is continuous; digital reception is checked separately.
        g = self.encoder(x)
        return g, self.decoder(g)


def scope_map(model):
    return {
        module.tf_scope + "/filter": name + ".weight"
        for name, module in model.named_modules()
        if isinstance(module, Conv)
    }


def import_tf_kernels(model, kernels):
    """Explicit in-memory HWIO -> OIHW import. Does not unpickle or download weights."""
    mapping = scope_map(model)
    if set(kernels) != set(mapping):
        raise ValueError("Require every E/D kernel, no missing or extra variables")
    params = dict(model.named_parameters())
    converted = {}
    for scope, name in mapping.items():
        t = torch.as_tensor(kernels[scope]).permute(3, 2, 0, 1)
        if t.shape != params[name].shape or not torch.isfinite(t).all():
            raise ValueError(f"Invalid kernel: {scope}")
        converted[name] = t
    with torch.no_grad():
        for name, tensor in converted.items():
            params[name].copy_(tensor)
