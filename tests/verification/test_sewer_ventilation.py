"""Published ventilation data and the equilibrium fixed point (rows A2, A3, H3)."""

import pytest
import torch

from tellegen.apps.sewer import geometry as g
from tellegen.apps.sewer.air import RHO_AIR_REF, Drag, Headspace
from tellegen.elements.fixed import FixedFlow
from tellegen.elements.powerlaw import Orifice
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.topology import Network

F64 = torch.float64


def _open_reach(diameter, length, h_over_d, v_w):
    net = Network(dtype=F64)
    for name in ("A1", "M", "A2"):
        net.add_node(name)
    net.add_edge("A1", "M", kind="headspace")
    net.add_edge("M", "A2", kind="headspace")
    half = torch.tensor([length / 2.0] * 2, dtype=F64)
    layer = PotentialFlowLayer(
        net, "air", [Headspace(half, torch.tensor([0, 1]))],
        drives=[Drag(half, torch.tensor([0, 1]))], boundary=["A1", "A2"],
        linear_solver="direct",
    )
    d = torch.full((2,), diameter, dtype=F64)
    h = torch.full((2,), h_over_d * diameter, dtype=F64)
    a_air, _, d_h = g.air_geometry(h, d)
    drv = {"sewer.A_air": a_air, "sewer.D_h": d_h, "sewer.T": g.top_width(h, d),
           "sewer.v": torch.full((2,), v_w, dtype=F64)}
    _, q = layer.solve(torch.zeros(2, dtype=F64), drv, None, differentiable=False,
                       atol=1e-12, rtol=1e-12)
    return float(q[0]), float(a_air[0])


def _vented_reach(diameter, length, h_over_d, v_w, leak_area, fan=None):
    net = Network(dtype=F64)
    for name in ("M1", "M2", "ambient"):
        net.add_node(name)
    net.add_edge("M1", "M2", kind="headspace")
    net.add_edge("M1", "ambient", kind="leak")
    net.add_edge("M2", "ambient", kind="leak")
    elements = [
        Headspace(torch.tensor([length], dtype=F64), torch.tensor([0])),
        Orifice(torch.full((2,), 0.6, dtype=F64), torch.full((2,), leak_area, dtype=F64),
                rho=RHO_AIR_REF, kind="leak", dp_transition=1e-8),
    ]
    if fan is not None:
        net.add_edge("M2", "ambient", kind="fan")
        elements.append(FixedFlow(torch.tensor([fan], dtype=F64), kind="fan"))
    layer = PotentialFlowLayer(
        net, "air", elements,
        drives=[Drag(torch.tensor([length], dtype=F64), torch.tensor([0]))],
        boundary=["ambient"], linear_solver="direct",
    )
    d = torch.tensor([diameter], dtype=F64)
    h = torch.tensor([h_over_d * diameter], dtype=F64)
    a_air, _, d_h = g.air_geometry(h, d)
    drv = {"sewer.A_air": a_air, "sewer.D_h": d_h, "sewer.T": g.top_width(h, d),
           "sewer.v": torch.tensor([v_w], dtype=F64)}
    phi, q = layer.solve(torch.zeros(1, dtype=F64), drv, None, differentiable=False,
                         atol=1e-12, rtol=1e-12)
    return layer, phi, q, float(a_air), drv


def test_a2_the_tyneside_band_is_bracketed():
    """Row A2, as amended (spec amendment A5). A 1650 mm, 100 m reach at 175 mm depth and
    1 m/s surface velocity. MEASURED: open at both ends 1253.78 m3/h (a velocity ratio of
    17.27 %, inside Pescod and Price's own 5-30 % envelope); vented through one 8 cm2
    pick-hole cover 0.24 m3/h. The published 105-315 m3/h band lies strictly between them.
    Neither the reach length nor the venting at Tyneside is reported, so the row asserts the
    BRACKETING and records the vent areas that reproduce the band."""
    q_open, a_air = _open_reach(1.65, 100.0, 0.175 / 1.65, 1.0)
    assert q_open * 3600.0 == pytest.approx(1253.78, rel=1e-4)
    assert 0.05 <= q_open / a_air / 1.0 <= 0.30
    _, _, q_shut, _, _ = _vented_reach(1.65, 100.0, 0.175 / 1.65, 1.0, 8e-4)
    assert float(q_shut[0]) * 3600.0 < 105.0
    assert q_open * 3600.0 > 315.0
    _, _, q_low, _, _ = _vented_reach(1.65, 100.0, 0.175 / 1.65, 1.0, 0.355)
    _, _, q_high, _, _ = _vented_reach(1.65, 100.0, 0.175 / 1.65, 1.0, 1.097)
    assert float(q_low[0]) * 3600.0 == pytest.approx(105.0, rel=5e-3)
    assert float(q_high[0]) * 3600.0 == pytest.approx(315.0, rel=5e-3)


def test_a3_a_fan_draws_exactly_through_the_leaks():
    """Row A3, 1e-12 on the flow balance. Measured 1.234e-14; the nodal residual is
    6.3e-15 and the power residual 8.1e-13 (checked at 1e-11, spec amendment A6)."""
    layer, phi, q, _, drv = _vented_reach(0.30, 15.0, 0.6, 0.0, 8e-4, fan=0.01)
    leaks = -(float(q[1]) + float(q[2]))
    assert leaks == pytest.approx(float(q[3]), abs=1e-12)
    residual = layer.residual(phi[..., layer.interior], torch.zeros(1, dtype=F64), drv, None)
    assert float(residual.abs().max()) < 1e-12
    assert float(layer.power_residual(phi, q, drv).abs().max()) < 1e-11


def test_h3_transfer_dominated_steady_state_is_henry_equilibrium():
    """Row H3, 1e-10. With a large K_L a and no gas-phase sink, the two phases reach the
    Henry partition: C_G = H f C_S (M_H2S / M_S)."""
    from tellegen.apps.sewer.quality import M_H2S, M_S, free_fraction, henry_h2s

    sulfide = torch.tensor([2.0e-3], dtype=F64)
    free = free_fraction(torch.tensor([7.0], dtype=F64))
    henry = henry_h2s(torch.tensor([293.15], dtype=F64))
    gas = henry * free * sulfide * (M_H2S / M_S)
    from tellegen.apps.sewer.quality import two_film_flux

    flux = two_film_flux(
        sulfide, gas, torch.tensor([7.25], dtype=F64), torch.tensor([1.0], dtype=F64),
        free, henry,
    )
    assert float(flux.abs()) < 1e-10
