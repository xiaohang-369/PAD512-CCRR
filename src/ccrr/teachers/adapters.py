import importlib.util
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from ccrr.data.pad import adapt
from ccrr.config import sha256
from ccrr.errors import require
from ccrr.tensor_ops import SymmetricHead


def freeze(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def freeze_bn(module):
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def load_weights(model, path, expected_hash, allowed_missing=(), allowed_unexpected=()):
    require(expected_hash and sha256(path) == expected_hash, "Weight hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    require(
        set(missing) <= set(allowed_missing)
        and set(unexpected) <= set(allowed_unexpected),
        f"Weight key mismatch: {missing}, {unexpected}",
    )
    return {"sha256": expected_hash, "missing": missing, "unexpected": unexpected}


class ClinicalModel(nn.Module):
    def __init__(self, architecture="efficientnet_b0", classes=6):
        super().__init__()
        self.architecture = architecture
        if architecture == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights=None)
            self.model.classifier[-1] = nn.Linear(1280, classes)
        elif architecture == "convnext_tiny":
            self.model = models.convnext_tiny(weights=None)
            self.model.classifier[-1] = nn.Linear(768, classes)
        elif architecture == "swin_t":
            self.model = models.swin_t(weights=None)
            self.model.head = nn.Linear(768, classes)
        else:
            raise ValueError(architecture)

    def forward(self, x, rects=None, spatial=False):
        if rects is None:
            rects = [(0, 0, x.shape[-1], x.shape[-2])] * len(x)
        a, v, trans = adapt(x, rects, 512)
        if self.architecture == "swin_t":
            f = self.model.norm(self.model.features(a)).permute(0, 3, 1, 2)
            logits = self.model.head(self.model.flatten(self.model.avgpool(f)))
        else:
            f = self.model.features(a)
            if self.architecture == "efficientnet_b0":
                logits = self.model.classifier(self.model.avgpool(f).flatten(1))
            else:
                logits = self.model.classifier(self.model.avgpool(f))
        return (
            (logits, f, F.adaptive_avg_pool2d(v, f.shape[-2:]), trans)
            if spatial
            else logits
        )


class IdentityModel(nn.Module):
    def __init__(self, patients=863, architecture="resnet50"):
        super().__init__()
        self.architecture = architecture
        if architecture == "resnet50":
            self.model = models.resnet50(weights=None)
            self.model.fc = nn.Identity()
            dim = 2048
        elif architecture == "swin_t":
            self.model = models.swin_t(weights=None)
            self.model.head = nn.Identity()
            dim = 768
        else:
            raise ValueError(architecture)
        self.classifier = nn.Linear(dim, patients)
        self.project = nn.Sequential(nn.Linear(dim, 256), nn.GELU())
        self.pair = SymmetricHead()

    def features(self, a):
        m = self.model
        if self.architecture == "swin_t":
            return m.norm(m.features(a)).permute(0, 3, 1, 2)
        a = m.maxpool(m.relu(m.bn1(m.conv1(a))))
        return m.layer4(m.layer3(m.layer2(m.layer1(a))))

    def forward(self, x, rects=None):
        if rects is None:
            rects = [(0, 0, x.shape[-1], x.shape[-2])] * len(x)
        a, v, t = adapt(x, rects, 512)
        f = self.features(a)
        h = f.mean((2, 3))
        return {
            "embedding": F.normalize(h, dim=1, eps=1e-6),
            "h": self.project(h),
            "logits": self.classifier(h),
            "spatial": f,
            "coverage": F.adaptive_avg_pool2d(v, f.shape[-2:]),
            "transform": t,
        }

    def score(self, a, b):
        return self.pair(a["h"], b["h"])


class PanDerm(nn.Module):
    """Loads the audited official source explicitly; never substitutes random features.

    The source module is user-configured and SHA-256 pinned. We do not silently
    substitute timm ViT: PanDerm uses a different attention/checkpoint layout.
    """

    def __init__(self, source, source_hash, weights, weights_hash):
        super().__init__()
        require(sha256(source) == source_hash, "PanDerm source hash mismatch")
        spec = importlib.util.spec_from_file_location("ccrr_official_panderm", source)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.model = mod.panderm_base_patch16_224()
        load_weights(
            self.model,
            weights,
            weights_hash,
            allowed_missing=("head.weight", "head.bias"),
        )
        self.model.head = nn.Identity()
        freeze(self.model)

    def forward(self, x, rects):
        a, v, t = adapt(x, rects, 224, panderm=True)
        m = self.model
        f = m.patch_embed(a)
        f = torch.cat((m.cls_token.expand(len(a), -1, -1), f), 1)
        f = m.pos_drop(f + m.pos_embed.to(f))
        bias = m.rel_pos_bias() if m.rel_pos_bias is not None else None
        for block in m.blocks:
            f = block(f, rel_pos_bias=bias)
        f = m.norm(f)[:, 1:]
        require(f.shape[1:] == (196, 768), "PanDerm patch interface changed")
        return (
            f.transpose(1, 2).reshape(-1, 768, 14, 14),
            F.adaptive_avg_pool2d(v, (14, 14)),
            t,
        )
