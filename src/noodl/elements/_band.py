"""The MBL power-law band, shared by every element that evaluates it.

``C sign(dp) |dp|^m`` for ``|dp| >= dp_turbulent``, replaced inside the band by the 7th-order odd
polynomial ``C dp_turbulent^m pi (a + pi^2 (b + pi^2 (c + pi^2 d)))``, ``pi = dp/dp_turbulent``
(``BaseClasses/powerLawFixedM.mo:21-30``, coefficients ``PowerLawResistanceParameters.mo:9-15``).
Used by the doors and by `RegularizedPowerLaw`; the other laminar blends are different
curves and stay with their elements.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor

_GAMMA = 1.5  # DoorOpen.mo:16-17, DoorOperable.mo:35-36, PowerLawResistanceParameters.mo:6


def power_law_band_coeffs(C: Tensor, m: Tensor, dp_turbulent: float) -> tuple:
    """The dp-independent part of `power_law_band`: the band polynomial's `a..d` and
    `C dp_turbulent^m` (a law's memo keeps them across the calls of a solve)."""
    a = _GAMMA
    b = 1 / 8 * m**2 - 3 * _GAMMA - 3 / 2 * m + 35.0 / 8
    c = -1 / 4 * m**2 + 3 * _GAMMA + 5 / 2 * m - 21.0 / 4
    d = 1 / 8 * m**2 - _GAMMA - m + 15.0 / 8
    return a, b, c, d, C * dp_turbulent**m


def power_law_band(C: Tensor, dp: Tensor, m: Tensor, dp_turbulent: float,
                  coeffs: tuple) -> Tensor:
    """Volume flow of the sharp-or-band law, with the coefficients
    `power_law_band_coeffs(C, m, dp_turbulent)` given."""
    a, b, c, d, C_dpt = coeffs
    mask = dp.abs() < dp_turbulent
    dp_safe = torch.where(mask, torch.full_like(dp, dp_turbulent), dp.abs())
    sharp = C * torch.sign(dp) * dp_safe**m
    pi = dp / dp_turbulent
    pi2 = pi * pi
    inner = C_dpt * pi * (a + pi2 * (b + pi2 * (c + pi2 * d)))
    return torch.where(mask, inner, sharp)


def power_law_band_slope(C: Tensor, dp: Tensor, m: Tensor, dp_turbulent: float,
                        coeffs: tuple, grad: Tensor) -> Tensor:
    """`grad * dpower_law_band/d dp`, by the chain rule in the order of operations reverse-
    mode autograd applies to `power_law_band` (`grad` the flow's own adjoint), so that it
    equals `torch.autograd.grad` of it bit for bit (checked in the tests) at a fraction of the
    cost. For the Newton Jacobian, under `no_grad` (the elements' `dflow`)."""
    a, b, c, d, C_dpt = coeffs
    mask = dp.abs() < dp_turbulent
    band, sharp = bool(mask.any()), not bool(mask.all())
    out = None
    if band:
        g_in = torch.where(mask, grad, torch.zeros_like(grad))
        pi = dp / dp_turbulent
        pi2 = pi * pi
        s1 = c + pi2 * d
        s2 = b + pi2 * s1
        t1 = C_dpt * pi
        g_t1 = g_in * (a + pi2 * s2)
        g_P = g_in * t1
        g_s2 = g_P * pi2
        g_s1 = g_s2 * pi2
        g_pi2 = (g_P * s2 + g_s2 * s1) + g_s1 * d
        g_pi2_pi = g_pi2 * pi
        out = ((g_t1 * C_dpt + g_pi2_pi) + g_pi2_pi) / dp_turbulent
    if sharp:
        g_sh = torch.where(mask, torch.zeros_like(grad), grad)
        dp_safe = torch.where(mask, torch.full_like(dp, dp_turbulent), dp.abs())
        g_safe = torch.where(m == 0.0, torch.zeros((), dtype=dp.dtype),
                             g_sh * (C * torch.sign(dp)) * (m * dp_safe.pow(m - 1)))
        g_sharp = torch.where(mask, torch.zeros_like(g_safe), g_safe) * dp.sgn()
        out = g_sharp if out is None else out + g_sharp
    return torch.zeros_like(dp) if out is None else out
