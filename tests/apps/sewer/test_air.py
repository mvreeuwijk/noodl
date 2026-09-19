"""The headspace element, the interfacial-drag drive and the air-density helper."""

import pytest
import torch

from noodl.apps.sewer import geometry as g
from noodl.apps.sewer.air import (
    F_AIR_DEFAULT,
    F_I_DEFAULT,
    RHO_AIR_REF,
    Drag,
    Headspace,
    air_density,
)
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network

F64 = torch.float64


def _drivers(diameter, h_over_d, v_w, n_edges=1):
    d = torch.full((n_edges,), diameter, dtype=F64)
    h = torch.full((n_edges,), h_over_d * diameter, dtype=F64)
    a_air, _, d_h = g.air_geometry(h, d)
    return {
        "sewer.A_air": a_air,
        "sewer.D_h": d_h,
        "sewer.T": g.top_width(h, d),
        "sewer.v": torch.full((n_edges,), v_w, dtype=F64),
    }


def test_the_calibrated_drag_factor_is_pinned():
    assert F_I_DEFAULT == 7.49e-4
    assert F_AIR_DEFAULT == 0.02


def test_air_density_is_the_ideal_gas_law():
    assert float(air_density(torch.tensor(293.15, dtype=F64))) == pytest.approx(
        1.2040972472143983, rel=1e-12
    )
    assert float(air_density(torch.tensor(283.15, dtype=F64))) == pytest.approx(
        1.2466223133353378, rel=1e-12
    )


def test_headspace_resistance_is_the_darcy_weisbach_form():
    drv = _drivers(0.30, 0.6, 0.0)
    el = Headspace(torch.tensor([15.0], dtype=F64), torch.tensor([0]))
    expected = (
        F_AIR_DEFAULT * 15.0 * RHO_AIR_REF
        / (2.0 * float(drv["sewer.D_h"]) * float(drv["sewer.A_air"]) ** 2)
    )
    assert float(el.resistance(drv)) == pytest.approx(expected, rel=1e-14)


def test_headspace_inverts_its_own_law():
    drv = _drivers(0.30, 0.6, 0.0)
    el = Headspace(torch.tensor([15.0], dtype=F64), torch.tensor([0]))
    dp = torch.tensor([0.5], dtype=F64)
    q = el.flow(dp, drv)
    r = el.resistance(drv)
    assert float(r * q.abs() * q) == pytest.approx(0.5, rel=1e-12)
    assert float(el.flow(-dp, drv)) == pytest.approx(-float(q), rel=1e-12)


def test_headspace_is_smooth_and_finite_at_dp_zero():
    drv = _drivers(0.30, 0.6, 0.0)
    el = Headspace(torch.tensor([15.0], dtype=F64), torch.tensor([0]))
    dp = torch.zeros(1, dtype=F64, requires_grad=True)
    el.flow(dp, drv).sum().backward()
    assert torch.isfinite(dp.grad).all()
    assert float(el.flow(torch.zeros(1, dtype=F64), drv)) == 0.0
    analytic = el.dflow(torch.tensor([0.3], dtype=F64), drv)
    leaf = torch.tensor([0.3], dtype=F64, requires_grad=True)
    el.flow(leaf, drv).sum().backward()
    assert float(analytic) == pytest.approx(float(leaf.grad), rel=1e-10)


def test_headspace_refuses_a_full_pipe():
    el = Headspace(torch.tensor([15.0], dtype=F64), torch.tensor([0]))
    drv = _drivers(0.30, 1.0, 0.0)
    with pytest.raises(ValueError, match="headspace area is not positive"):
        el.flow(torch.tensor([0.5], dtype=F64), drv)


def test_headspace_refuses_a_missing_driver():
    el = Headspace(torch.tensor([15.0], dtype=F64), torch.tensor([0]))
    with pytest.raises(KeyError, match="'sewer.A_air'"):
        el.flow(torch.tensor([0.5], dtype=F64), {})


def test_headspace_resistance_names_only_the_missing_key():
    """FR-11: only the ACTUALLY missing key is named, not always both -- a caller who gave
    `sewer.D_h` but forgot `sewer.A_air` must not be told `sewer.D_h` is missing too."""
    el = Headspace(torch.tensor([15.0], dtype=F64), torch.tensor([0]))
    with pytest.raises(KeyError, match=r"\['sewer\.A_air'\]") as excinfo:
        el.resistance({"sewer.D_h": torch.tensor([0.1], dtype=F64)})
    assert "sewer.D_h" not in str(excinfo.value)
    with pytest.raises(KeyError, match=r"\['sewer\.D_h'\]") as excinfo:
        el.resistance({"sewer.A_air": torch.tensor([0.1], dtype=F64)})
    assert "sewer.A_air" not in str(excinfo.value)


def test_drag_returns_the_positive_shear_term():
    drv = _drivers(0.30, 0.6, 0.8)
    drive = Drag(torch.tensor([15.0], dtype=F64), torch.tensor([0]))
    expected = (
        0.5 * F_I_DEFAULT * RHO_AIR_REF * 0.8 * 0.8
        * float(drv["sewer.T"]) * 15.0 / float(drv["sewer.A_air"])
    )
    assert float(drive(drv)) == pytest.approx(expected, rel=1e-14)
    assert float(drive(drv)) > 0.0


def test_drag_follows_the_sign_of_the_water_velocity():
    drive = Drag(torch.tensor([15.0], dtype=F64), torch.tensor([0]))
    forward = float(drive(_drivers(0.30, 0.6, 0.8)))
    backward = float(drive(_drivers(0.30, 0.6, -0.8)))
    assert backward == pytest.approx(-forward, rel=1e-14)


def test_drag_is_masked_on_the_outfall_edge():
    drv = _drivers(0.30, 0.6, 0.8, n_edges=2)
    drive = Drag(torch.tensor([15.0, 15.0], dtype=F64), torch.tensor([0, 1]),
                 zero_positions=(1,))
    value = drive(drv)
    assert float(value[0]) > 0.0
    assert float(value[1]) == 0.0


def test_drag_refuses_a_missing_driver():
    drive = Drag(torch.tensor([15.0], dtype=F64), torch.tensor([0]))
    with pytest.raises(KeyError, match="'sewer.v'"):
        drive({"sewer.T": torch.ones(1, dtype=F64),
               "sewer.A_air": torch.ones(1, dtype=F64)})


@pytest.mark.parametrize(
    "h_over_d, v_w, measured", [(0.50, 0.2, 0.35), (0.60, 0.8, 0.25), (0.62, 0.4, 0.275)]
)
def test_a1_pescod_and_price_air_to_water_velocity_ratio(h_over_d, v_w, measured):
    """Row A1. A 300 mm UPVC pipe, 15 m, open at BOTH ends: two half-length headspace edges
    in series with both outer nodes prescribed at ambient pressure and one interior manhole
    between them. Measured by the plan writer: 24.139 %, 24.995 %, 25.149 % -- all inside
    the 20-40 % band, and equal to the closed form to 1.12e-8 relative."""
    net = Network(dtype=F64)
    for name in ("A1", "M", "A2"):
        net.add_node(name)
    net.add_edge("A1", "M", kind="headspace")
    net.add_edge("M", "A2", kind="headspace")
    half = torch.tensor([7.5, 7.5], dtype=F64)
    layer = PotentialFlowLayer(
        net, "air", [Headspace(half, torch.tensor([0, 1]))],
        drives=[Drag(half, torch.tensor([0, 1]))], boundary=["A1", "A2"],
        linear_solver="direct",
    )
    drv = _drivers(0.30, h_over_d, v_w, n_edges=2)
    phi, q = layer.solve(torch.zeros(2, dtype=F64), drv, None, differentiable=False,
                         atol=1e-12, rtol=1e-12)
    ratio = float(q[0]) / float(drv["sewer.A_air"][0]) / v_w
    closed = (
        F_I_DEFAULT / F_AIR_DEFAULT
        * float(drv["sewer.T"][0]) * float(drv["sewer.D_h"][0])
        / float(drv["sewer.A_air"][0])
    ) ** 0.5
    assert ratio == pytest.approx(closed, rel=1e-6)
    assert 0.20 <= ratio <= 0.40, (ratio, measured)
