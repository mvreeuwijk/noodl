"""The `shape_constant` closure option: the Bessel shape parameter as an exact root or as
MUNICH's argmin on a 0.01 grid (ATM `ComputeSiraneC`, `MeteorologyStreet.cxx:114-152`)."""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.street_aq import build_model, twelve_street_grid
from noodl.apps.street_aq.canyon import (
    EULER_GAMMA_TRUNCATED,
    bessel_shape_parameter,
    grid_shape_parameter,
    roof_wind,
    shape_parameter,
)

DT = torch.float64


def _munich_c(ratio: float) -> float:
    """`ComputeSiraneC` transcribed line by line, with scipy's Bessel functions."""
    from scipy.special import j1, y1

    temp, list_c, c = [], [], 0.0
    for _ in range(100):
        c += 0.01
        list_c.append(c)
        temp.append(abs(2.0 / c * math.exp(math.pi / 2.0 * y1(c) / j1(c)
                                           - EULER_GAMMA_TRUNCATED) - ratio))
    index, best = 0, temp[0]
    for i in range(1, 100):
        if best > temp[i]:
            best, index = temp[i], i
    return list_c[index]


RATIOS = [1e-4, 0.001, 0.002, 0.0026666666666666666, 0.005, 0.0075, 0.01, 0.03]


def test_the_grid_search_is_munichs_compute_sirane_c():
    got = grid_shape_parameter(torch.tensor(RATIOS, dtype=DT))
    want = torch.tensor([_munich_c(r) for r in RATIOS], dtype=DT)
    assert torch.equal(got, want)
    exact = bessel_shape_parameter(torch.tensor(RATIOS, dtype=DT))
    assert float((got - exact).abs().max()) <= 0.005 + 1e-12


def test_the_grid_search_carries_the_exact_roots_gradient():
    ratio = torch.tensor([0.0026666666666666666, 0.015], dtype=DT, requires_grad=True)
    (grid_grad,) = torch.autograd.grad(grid_shape_parameter(ratio).sum(), ratio)
    (exact_grad,) = torch.autograd.grad(bessel_shape_parameter(ratio).sum(), ratio)
    torch.testing.assert_close(grid_grad, exact_grad, rtol=1e-14, atol=0)
    assert bool((grid_grad > 0).all())


def test_the_grid_search_refuses_a_ratio_it_cannot_solve():
    with pytest.raises(ValueError, match=r"grid_shape_parameter.*exact_root"):
        grid_shape_parameter(torch.tensor([0.01, 0.5], dtype=DT))


def test_shape_parameter_dispatches_on_the_option():
    ratio = torch.tensor([0.0026666666666666666], dtype=DT)
    assert torch.equal(shape_parameter(ratio, "exact_root"), bessel_shape_parameter(ratio))
    assert torch.equal(shape_parameter(ratio, "grid_search"), grid_shape_parameter(ratio))
    assert float(shape_parameter(ratio, "grid_search")) == pytest.approx(0.62, abs=1e-12)
    with pytest.raises(ValueError, match="shape_constant"):
        shape_parameter(ratio, "nearest")


def test_the_munich_preset_takes_the_grid_and_the_roof_wind_moves_by_under_2_percent():
    net, _ = twelve_street_grid()
    sirane, _, _ = build_model(net)
    munich, _, _ = build_model(net, preset="munich")
    assert sirane.closures[0].shape_constant == "exact_root"
    assert munich.closures[0].shape_constant == "grid_search"
    u_star, h, w = torch.tensor(0.3, dtype=DT), torch.tensor(7.5, dtype=DT), 15.0
    exact = roof_wind(u_star, h, torch.tensor(w, dtype=DT), kappa=0.41)
    grid = roof_wind(u_star, h, torch.tensor(w, dtype=DT), kappa=0.41,
                     shape_constant="grid_search")
    assert 1e-3 < abs(float(grid / exact) - 1.0) < 2e-2
