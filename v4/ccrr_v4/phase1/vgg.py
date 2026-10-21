"""Frozen author's VGG19 up to conv4_1. No torchvision weight substitution."""

import codecs
import importlib
import pickle
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..losses import legacy_resize_bilinear
from ..verify import digest, read, REPO

LAYERS = (
    ("conv1_1", 3, 64),
    ("conv1_2", 64, 64),
    ("conv2_1", 64, 128),
    ("conv2_2", 128, 128),
    ("conv3_1", 128, 256),
    ("conv3_2", 256, 256),
    ("conv3_3", 256, 256),
    ("conv3_4", 256, 256),
    ("conv4_1", 256, 512),
)


class ArrayUnpickler(pickle.Unpickler):
    """Load old NumPy arrays without allowing arbitrary pickle globals."""

    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("_codecs", "encode"): codecs.encode,
        }
        module_path = (
            "numpy._core.multiarray"
            if int(np.__version__.split(".")[0]) >= 2
            else "numpy.core.multiarray"
        )
        multiarray = importlib.import_module(module_path)
        for prefix in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed[prefix, "_reconstruct"] = multiarray._reconstruct
            allowed[prefix, "scalar"] = multiarray.scalar
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(
                f"Unsupported weight pickle global: {module}.{name}"
            )
        return allowed[module, name]


def read_author_npy(path, expected_sha):
    if not expected_sha or digest(path) != expected_sha:
        raise ValueError("VGG weight SHA256 mismatch or missing binding")
    with Path(path).open("rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError("Unsupported author NPY version")
        if shape != () or not dtype.hasobject:
            raise ValueError("Expected author's scalar NumPy dictionary")
        data = ArrayUnpickler(f, encoding="latin1").load().item()
    if not isinstance(data, dict):
        raise ValueError("Author weight dictionary missing")
    return data


class AuthorVGG(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.convs = nn.ModuleDict()
        for name, cin, cout in LAYERS:
            w, b = data[name]
            if w.shape != (3, 3, cin, cout) or b.shape != (cout,):
                raise ValueError(f"VGG source shape mismatch: {name}")
            if w.dtype != np.float32 or b.dtype != np.float32:
                raise ValueError("VGG source must be float32")
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError("Nonfinite VGG weights")
            conv = nn.Conv2d(cin, cout, 3, padding=1)
            conv.weight.data.copy_(torch.from_numpy(w.transpose(3, 2, 0, 1).copy()))
            conv.bias.data.copy_(torch.from_numpy(b.copy()))
            self.convs[name] = conv
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_file(cls, path, expected_sha):
        registry = read(REPO / "v4/upstream/vgg-provenance.json")
        if expected_sha not in (registry["npy_sha256"], registry["feature_sha256"]):
            raise ValueError(
                "Unregistered VGG weights; do not substitute pretrained models"
            )
        if digest(path) != expected_sha:
            raise ValueError("VGG weight SHA256 mismatch")
        if expected_sha == registry["feature_sha256"]:
            with np.load(path, allow_pickle=False) as archive:
                data = {
                    name: (archive[name + ".weight"], archive[name + ".bias"])
                    for name, _, _ in LAYERS
                }
            return cls(data)
        return cls(read_author_npy(path, expected_sha))

    def forward(self, normalized_image):
        x = legacy_resize_bilinear(normalized_image, (224, 224))
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        elif x.shape[1] != 3:
            raise ValueError("VGG input requires RGB or single-channel grayscale")
        x = (x + 1) / 2 * 255
        x = (
            x[:, [2, 1, 0]]
            - x.new_tensor([103.939, 116.779, 123.68])[None, :, None, None]
        )
        for name, _, _ in LAYERS:
            x = F.relu(self.convs[name](x))
            if name in ("conv1_2", "conv2_2", "conv3_4"):
                # Sizes 224/112/56 are even: SAME == no padding here.
                x = F.max_pool2d(x, 2, 2)
        return x
