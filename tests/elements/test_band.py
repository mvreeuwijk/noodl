"""The shared power-law band equals the expression it replaced, bit for bit."""

from __future__ import annotations

import pytest
import torch

from noodl.elements._band import power_law_band, power_law_band_coeffs

GAMMA = 1.5


def _old_regularised(C, dp, m, dpt):
    """`RegularizedPowerLaw._regularised` as it was written inline."""
    b = 1 / 8 * m**2 - 3 * GAMMA - 3 / 2 * m + 35.0 / 8
    c = -1 / 4 * m**2 + 3 * GAMMA + 5 / 2 * m - 21.0 / 4
    d = 1 / 8 * m**2 - GAMMA - m + 15.0 / 8
    C_dpt = C * dpt**m
    mask = dp.abs() < dpt
    dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
    sharp = C * torch.sign(dp) * dp_safe**m
    pi = dp / dpt
    pi2 = pi * pi
    inner = C_dpt * pi * (GAMMA + pi2 * (b + pi2 * (c + pi2 * d)))
    return torch.where(mask, inner, sharp)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("m", [0.5, 0.65, 1.0])
def test_band_is_bitwise_the_old_inline_expression(dtype, m):
    dpt = 0.1
    C = torch.tensor(0.01, dtype=dtype)
    mt = torch.tensor(m, dtype=dtype)
    # inside the band, at its edges, outside, both signs, and zero
    dp = torch.tensor([-5.0, -0.1, -0.05, -1e-6, 0.0, 1e-6, 0.03, 0.1, 0.5, 20.0], dtype=dtype)
    new = power_law_band(C, dp, mt, dpt, power_law_band_coeffs(C, mt, dpt))
    assert torch.equal(new, _old_regularised(C, dp, mt, dpt))
