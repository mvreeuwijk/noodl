"""Published ventilation data and the equilibrium fixed point (rows A2, A3, H3)."""

import pytest
import torch

from noodl.apps.sewer import geometry as g
from noodl.apps.sewer.air import RHO_AIR_REF, Drag, Headspace
from noodl.elements.fixed import FixedFlow
from noodl.elements.powerlaw import Orifice
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network

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
    """Row H3, 1e-10 (spec). N7: the earlier version of this test asserted an ALGEBRAIC
    identity of `two_film_flux` directly -- it never ran the model at all. This version
    builds a genuine `Model` (one manhole `M`, one boundary node `B`, a `water_quality` and
    an `air_quality` `TransportLayer` each on their own zero-flow edge kind, and
    `H2STransfer` as the model's only closure) and steps it with `Model.step`/no reaction,
    starting the water sulfide away from equilibrium and the headspace at zero, until the
    fixed point.

    There is no gas-phase sink in this application at all (spec's `k_gas` is a documented,
    always-zero, unverified parameter with no reaction registered for it -- confirmed by
    inspection of `build_model` and `quality.py`, neither of which builds one), so
    "no gas-phase sink" needs no extra step to arrange.

    The ASSEMBLED tree fixture (`build_model(tree_steady())`) is NOT used here: its
    leaks vent H2S to ambient (a zero-concentration boundary) and its air layer always has
    a net flow path to the outfall's own headspace edge, so mass continuously leaves the
    system and true Henry equilibrium is never reached there, only approached asymptotically
    as leak/outfall losses shrink relative to the two-film transfer rate -- exactly the
    caveat this row's brief anticipates. The single CLOSED manhole model here has zero flow
    on both `water`/`air` kinds (no boundary exchange at all, `"...q"` supplied as zero
    drivers rather than solved by a potential layer), so the only thing that can happen is
    mass moving between the two phases until the two-film flux itself is zero -- which is
    exactly the Henry partition `C_G = H f C_S (M_H2S / M_S)` this row is about.

    MEASURED: 600 steps of dt = 60 s from sulfide = 1e-3 kg/m3 (S), gas = 0, reach a
    relative difference between the headspace concentration and its Henry-equilibrium value
    of 1.4e-16 -- far inside the row's 1e-10, recorded here as the measured figure the
    assertion actually uses (1e-10, unchanged from the spec). A fixed 600-step `model.step`
    loop is used rather than `sewer_steady`'s own tolerance-driven early exit, which (on
    this system's absolute per-step-change test, applied to both quality layers) stops
    noticeably before the relative gas/equilibrium gap has fully settled."""
    from noodl.apps.sewer.quality import M_H2S, M_S, H2STransfer, free_fraction, henry_h2s
    from noodl.layers.transport import TransportLayer
    from noodl.model import Model
    from noodl.topology import Network

    net = Network(dtype=F64)
    net.add_node("M")
    net.add_node("B")
    net.add_edge("M", "B", kind="water")
    net.add_edge("M", "B", kind="air")
    v_water, v_air = 10.0, 1.0
    water = TransportLayer(
        net, "water_quality", capacity=torch.tensor([v_water], dtype=F64),
        flow_kind="water", boundary=["B"], n_species=1, scheme="implicit",
        quantity="concentration", unit="kg/m3",
    )
    air = TransportLayer(
        net, "air_quality", capacity=torch.tensor([v_air], dtype=F64),
        flow_kind="air", boundary=["B"], n_species=1, scheme="implicit",
        quantity="concentration", unit="kg/m3",
    )
    manhole_idx = torch.tensor([net.nodes.index("M")], dtype=torch.long)
    closure = H2STransfer(net.n, manhole_idx, sulfide=0, out_pipe=None)
    model = Model(net, {"water_quality": water, "air_quality": air}, closures=[closure])
    state = {
        "water_quality.x": torch.tensor([1.0e-3], dtype=F64),
        "air_quality.x": torch.tensor([0.0], dtype=F64),
    }
    drivers = {
        "water_quality.x_boundary": torch.zeros(1, dtype=F64),
        "air_quality.x_boundary": torch.zeros(1, dtype=F64),
        "water_quality.q": torch.zeros(1, dtype=F64),
        "air_quality.q": torch.zeros(1, dtype=F64),
        "sewer.V_wet": torch.tensor([v_water], dtype=F64),
        "sewer.q_slope": torch.tensor([0.01], dtype=F64),
        "sewer.v": torch.tensor([0.5], dtype=F64),
        "sewer.d_m": torch.tensor([0.1], dtype=F64),
        "T_head": torch.tensor(293.15, dtype=F64),
        "pH": torch.tensor(7.0, dtype=F64),
    }
    for _ in range(600):
        state = model.step(state, drivers, 60.0)
    sulfide = state["water_quality.x"][0]
    gas = state["air_quality.x"][0]
    free = free_fraction(torch.tensor(7.0, dtype=F64))
    henry = henry_h2s(torch.tensor(293.15, dtype=F64))
    expected_gas = henry * free * sulfide * (M_H2S / M_S)
    assert float((gas - expected_gas).abs() / expected_gas) < 1e-10
