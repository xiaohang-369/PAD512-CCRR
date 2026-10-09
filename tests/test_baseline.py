import torch
from ccrr.representation.inn import FixedGrayDecomposition


def test_fixed_gray_baseline_continuous_inverse():
    m = FixedGrayDecomposition()
    x = torch.rand(2, 3, 32, 32)
    g, z = m(x)
    assert (m.inverse(g, z) - x).abs().max() < 2e-7
