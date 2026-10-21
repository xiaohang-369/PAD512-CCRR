"""Original model.py executes with an independent NumPy operator oracle.

This is source/operation verification, NOT a TensorFlow-runtime equivalence claim.
"""

from contextlib import contextmanager
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from ccrr_v4.original import (
    OriginalGrayscale,
    Conv,
    scope_map,
    import_tf_kernels,
    quantize8,
)
from ccrr_v4.losses import legacy_resize_bilinear, source_terms, author_loss
from ccrr_v4.config import load_verification
from ccrr_v4.verify import source_check, synthetic_check

ROOT = Path(__file__).resolve().parents[1]


class Tensor(np.ndarray):
    def get_shape(self):
        return self.shape


def tensor(x):
    return np.asarray(x, dtype=np.float32).view(Tensor)


def numpy_tf(kernels):
    scopes = []
    tf = types.ModuleType("tensorflow")

    @contextmanager
    def scope(name, **kwargs):
        scopes.append(name)
        try:
            yield
        finally:
            scopes.pop()

    def variable(name, shape, **kwargs):
        array = kernels["/".join(scopes + [name])]
        assert list(array.shape) == list(shape)
        return tensor(array)

    def conv(x, w, strides, padding):
        assert padding == "SAME"
        sy, sx = strides[1:3]
        kh, kw, _, _ = w.shape
        h, wi = x.shape[1:3]
        oh, ow = (h + sy - 1) // sy, (wi + sx - 1) // sx
        ph, pw = max((oh - 1) * sy + kh - h, 0), max((ow - 1) * sx + kw - wi, 0)
        padded = np.pad(
            x, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0))
        )
        out = np.zeros((len(x), oh, ow, w.shape[-1]), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                out += np.einsum(
                    "nhwc,co->nhwo",
                    padded[:, i : i + oh * sy : sy, j : j + ow * sx : sx, :],
                    w[i, j],
                    optimize=False,
                )
        return tensor(out)

    def resize(x, size, method):
        assert method == 1 and list(size) == [x.shape[1] * 2, x.shape[2] * 2]
        return tensor(np.repeat(np.repeat(x, 2, axis=1), 2, axis=2))

    tf.variable_scope = scope
    tf.get_variable = variable
    tf.float32 = np.float32
    tf.random_normal_initializer = lambda *args: None
    tf.shape = lambda x: np.array(x.shape)
    tf.add = lambda x, y: tensor(x + y)
    tf.nn = types.SimpleNamespace(
        conv2d=conv,
        relu=lambda x: tensor(np.maximum(x, 0)),
        tanh=lambda x: tensor(np.tanh(x)),
    )
    tf.image = types.SimpleNamespace(resize_images=resize)
    return tf


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_entire_original_forward_with_independent_numpy_oracle(monkeypatch):
    torch.manual_seed(52)
    model = OriginalGrayscale().eval()
    parameters = dict(model.named_parameters())
    kernels = {
        scope: parameters[name].detach().numpy().transpose(2, 3, 1, 0).copy()
        for scope, name in scope_map(model).items()
    }
    monkeypatch.setitem(sys.modules, "tensorflow", numpy_tf(kernels))
    monkeypatch.setitem(sys.modules, "util", types.ModuleType("util"))
    namespace = {"__name__": "author_model_numpy_oracle"}
    source = ROOT / "upstream/InvertibleGrayscale/model.py"
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    x = tensor(np.random.default_rng(5).uniform(-1, 1, (1, 8, 12, 3)))
    expected_gray = namespace["encode"](x, 1)
    expected_rgb = namespace["decode"](expected_gray, 3)
    with torch.no_grad():
        actual_gray, actual_rgb = model(
            torch.from_numpy(np.asarray(x)).permute(0, 3, 1, 2)
        )
    np.testing.assert_allclose(
        actual_gray.permute(0, 2, 3, 1).numpy(), expected_gray, atol=2e-6, rtol=2e-5
    )
    np.testing.assert_allclose(
        actual_rgb.permute(0, 2, 3, 1).numpy(), expected_rgb, atol=2e-6, rtol=2e-5
    )


def test_source_fidelity_and_scope_import():
    assert source_check()["files_verified"] == 7
    model = OriginalGrayscale()
    assert len(scope_map(model)) == 45
    kernels = {
        s: dict(model.named_parameters())[n].detach().permute(2, 3, 1, 0).clone()
        for s, n in scope_map(model).items()
    }
    clone = OriginalGrayscale()
    import_tf_kernels(clone, kernels)
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), clone.parameters())
    )
    kernels.pop(next(iter(kernels)))
    with pytest.raises(ValueError):
        import_tf_kernels(clone, kernels)


@pytest.mark.parametrize("shape", [(8, 12), (9, 7)])
def test_same_padding_even_and_odd(shape):
    x = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(1, 1, *shape)
    conv = Conv(1, 1, "example", stride=2)
    with torch.no_grad():
        conv.weight.fill_(1)
    expected_pad = (0, 1, 0, 1) if shape == (8, 12) else (1, 1, 1, 1)
    assert torch.equal(conv(x), F.conv2d(F.pad(x, expected_pad), conv.weight, stride=2))
    if shape == (8, 12):
        assert not torch.equal(conv(x), F.conv2d(x, conv.weight, stride=2, padding=1))


def test_legacy_resize_coordinates_and_gradient():
    x = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], requires_grad=True)
    out = legacy_resize_bilinear(x, (4, 4))
    expected = torch.tensor(
        [
            [
                [
                    [0.0, 0.5, 1.0, 1.0],
                    [1.0, 1.5, 2.0, 2.0],
                    [2.0, 2.5, 3.0, 3.0],
                    [2.0, 2.5, 3.0, 3.0],
                ]
            ]
        ]
    )
    assert torch.equal(out, expected)
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.sum() == 16


def test_quantizer_all_levels_and_half_steps():
    integers = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
    encoded, decoded = quantize8(integers.float() / 127.5 - 1)
    assert torch.equal(encoded, integers)
    assert torch.equal(quantize8(decoded)[0], integers)
    g = torch.tensor([-1.0, 0.0, 1.0]).reshape(1, 1, 1, 3)
    assert quantize8(g)[0].flatten().tolist() == [0, 128, 255]


def test_quantization_penalty_matches_256_level_definition():
    g = (
        torch.linspace(-0.99, 0.99, 51, dtype=torch.float64)
        .reshape(1, 1, 3, 17)
        .requires_grad_()
    )
    x = g.detach().expand(-1, 3, -1, -1).clone()
    actual = source_terms(x, g, x)["quantization"]
    levels = torch.arange(256, dtype=g.dtype) / 127.5 - 1
    reference = (g[..., None] - levels).abs().amin(-1).mean()
    assert torch.allclose(actual, reference, atol=1e-15, rtol=0)
    ga = torch.autograd.grad(actual, g, retain_graph=True)[0]
    gr = torch.autograd.grad(reference, g)[0]
    assert torch.equal(ga, gr)


def test_missing_vgg_is_not_silently_replaced():
    x = torch.zeros(1, 3, 4, 4)
    with pytest.raises(ValueError, match="mandatory"):
        author_loss(x, x[:, :1], x, None, None, 1, {})


def test_config_separates_runtime_and_training(tmp_path):
    cfg = load_verification(ROOT / "configs/verify-cpu.json", ["cuda:0", "cuda:1"])
    assert cfg["devices"] == ["cuda:0", "cuda:1"]
    with pytest.raises(ValueError):
        load_verification(ROOT / "configs/train-author-reference.json")
    with pytest.raises(ValueError):
        load_verification(ROOT / "configs/verify-cpu.json", [])


def test_no_optimizer_digital_check(tmp_path):
    cfg = load_verification(ROOT / "configs/verify-cpu.json")
    result = synthetic_check(cfg, torch.device("cpu"), tmp_path)
    assert (
        result["optimizer_updates"] == 0
        and result["weights_unchanged"]
        and result["png_exact"]
    )


def test_no_private_decoder_input_or_odd_encoder_shape():
    model = OriginalGrayscale()
    with pytest.raises(ValueError):
        model.encoder(torch.zeros(1, 3, 7, 8))
    with pytest.raises(ValueError):
        model.decoder(torch.zeros(1, 2, 8, 8))
