"""WaterNetwork validation, the builder, the tank closure and the report helpers."""

import csv
import dataclasses
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import (
    Control,
    Junction,
    Pump,
    Reservoir,
    Tank,
    Valve,
    WaterNetwork,
    WaterOptions,
    WaterPipe,
    build_model,
    initial_state,
    twoloop,
    water_steady,
)
from noodl.apps.water.report import link_table, pressure_head, to_kilopascal

F64 = torch.float64


def _minimal(**kwargs) -> WaterNetwork:
    base = {
        "junctions": (Junction("J1", 0.0, 0.005),),
        "reservoirs": (Reservoir("R1", 50.0),),
        "pipes": (WaterPipe("P1", "R1", "J1", 500.0, 0.3, 130.0),),
    }
    base.update(kwargs)
    return WaterNetwork(**base)


# ------------------------------------------------------------------- validation
def test_the_fixture_is_the_committed_two_loop():
    net = twoloop()
    assert net.nodes() == ["J1", "J2", "J3", "J4", "J5", "J6", "R1"]
    assert [p.name for p in net.pipes] == ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    assert [j.demand for j in net.junctions] == pytest.approx(
        [0.005, 0.008, 0.006, 0.010, 0.007, 0.009], rel=1e-15
    )
    assert all(p.roughness == 130.0 for p in net.pipes)
    net.validate()


def test_a_duplicate_node_name_is_refused():
    with pytest.raises(ValueError, match="duplicate node name 'J1'"):
        _minimal(junctions=(Junction("J1", 0.0), Junction("J1", 1.0))).validate()


def test_a_duplicate_link_name_is_refused():
    with pytest.raises(ValueError, match="duplicate link name 'P1'"):
        _minimal(
            pipes=(
                WaterPipe("P1", "R1", "J1", 500.0, 0.3, 130.0),
                WaterPipe("P1", "J1", "R1", 500.0, 0.3, 130.0),
            )
        ).validate()


def test_a_link_naming_an_unknown_node_is_refused():
    with pytest.raises(ValueError, match=r"link 'P1' names node 'Nowhere'"):
        _minimal(
            pipes=(WaterPipe("P1", "R1", "Nowhere", 500.0, 0.3, 130.0),)
        ).validate()


def test_a_network_with_no_fixed_head_node_is_refused():
    with pytest.raises(ValueError, match="no reservoir and no tank"):
        WaterNetwork(
            junctions=(Junction("J1", 0.0), Junction("J2", 0.0)),
            pipes=(WaterPipe("P1", "J1", "J2", 500.0, 0.3, 130.0),),
        ).validate()


@pytest.mark.parametrize(
    "length, diameter, roughness", [(0.0, 0.3, 130.0), (500.0, 0.0, 130.0),
                                    (500.0, 0.3, 0.0)]
)
def test_a_non_positive_pipe_property_is_refused(length, diameter, roughness):
    with pytest.raises(ValueError, match="must have positive length"):
        _minimal(
            pipes=(WaterPipe("P1", "R1", "J1", length, diameter, roughness),)
        ).validate()


@pytest.mark.parametrize("status", ["CLOSED", "CV"])
def test_a_closed_pipe_or_check_valve_is_refused(status):
    with pytest.raises(ValueError, match="status-switching"):
        _minimal(
            pipes=(WaterPipe("P1", "R1", "J1", 500.0, 0.3, 130.0, status=status),)
        ).validate()


def test_an_unknown_pipe_status_is_refused():
    with pytest.raises(ValueError, match="only OPEN, CLOSED and CV"):
        _minimal(
            pipes=(WaterPipe("P1", "R1", "J1", 500.0, 0.3, 130.0, status="AJAR"),)
        ).validate()


def test_a_constant_power_pump_is_refused():
    with pytest.raises(ValueError, match="constant-POWER pump"):
        _minimal(
            pumps=(Pump("PU1", "R1", "J1", None, power=50.0),)
        ).validate()


def test_a_pump_with_no_curve_and_no_power_is_refused():
    with pytest.raises(ValueError, match="neither a HEAD curve nor a POWER"):
        _minimal(pumps=(Pump("PU1", "R1", "J1", None),)).validate()


def test_a_pump_naming_an_unknown_curve_is_refused():
    with pytest.raises(ValueError, match=r"names curve 'C9'"):
        _minimal(pumps=(Pump("PU1", "R1", "J1", "C9"),)).validate()


def test_a_multi_point_pump_curve_is_refused():
    with pytest.raises(ValueError, match="which is out of scope"):
        _minimal(
            pumps=(Pump("PU1", "R1", "J1", "C1"),),
            curves={"C1": ((0.0, 100.0), (0.05, 80.0), (0.1, 40.0), (0.15, 0.0))},
        ).validate()


@pytest.mark.parametrize("kind", ["PRV", "PSV", "PBV", "GPV"])
def test_the_status_switching_valves_are_refused(kind):
    with pytest.raises(ValueError, match="out of scope"):
        _minimal(
            valves=(Valve("V1", "R1", "J1", 0.2, kind, 30.0),)
        ).validate()


def test_an_unknown_valve_type_is_refused():
    with pytest.raises(ValueError, match=r"has type 'XYZ'"):
        _minimal(valves=(Valve("V1", "R1", "J1", 0.2, "XYZ", 1.0),)).validate()


def test_an_unmodelled_headloss_formula_is_refused():
    with pytest.raises(ValueError, match="only H-W and D-W"):
        _minimal(headloss="C-M").validate()


def test_a_control_on_a_non_tank_is_refused():
    with pytest.raises(ValueError, match="which is not a tank"):
        _minimal(controls=(Control("PU1", "OPEN", "J1", "BELOW", 1.0),)).validate()


def test_a_control_on_a_non_pump_is_refused():
    with pytest.raises(ValueError, match="which is not a pump"):
        _minimal(
            tanks=(Tank("T1", 10.0, 5.0, 1.0, 9.0, 3.0),),
            controls=(Control("P1", "OPEN", "T1", "BELOW", 1.0),),
        ).validate()


# ---------------------------------------------------------------------- builder
def test_the_builder_returns_a_model_a_state_and_drivers():
    model, state, drivers = build_model(twoloop())
    assert set(model.potential) == {"water"}
    assert model.transport == {}
    assert model.node_names == ["J1", "J2", "J3", "J4", "J5", "J6", "R1"]
    assert model.link_names == ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    assert [model.net.nodes[i] for i in model.potential["water"].bound] == ["R1"]
    assert set(state) == {"water.phi", "water.q"}
    assert set(drivers) == {"water.phi_boundary", "water.sources"}
    assert float(drivers["water.phi_boundary"][0]) == 50.0
    assert model.head_scale == 1.0


def test_the_demand_enters_as_a_negative_nodal_source():
    model, _, drivers = build_model(twoloop())
    assert drivers["water.sources"].tolist() == pytest.approx(
        [-0.005, -0.008, -0.006, -0.010, -0.007, -0.009, 0.0], rel=1e-15
    )


def test_the_two_loop_solves_to_the_measured_heads():
    """The same numbers spec row D1 asserts against EPANET, pinned here without wntr."""
    model, state, drivers = build_model(twoloop())
    final = water_steady(model, state, drivers)
    assert final["water.phi"].tolist() == pytest.approx(
        [49.267724991, 48.877301673, 48.430865920, 48.213849910, 48.117097853,
         48.122916699, 50.0],
        abs=1e-9,
    )


def test_the_solved_residual_is_at_the_newton_tolerance():
    model, state, drivers = build_model(twoloop())
    final = water_steady(model, state, drivers)
    layer = model.potential["water"]
    residual = layer.residual(
        final["water.phi"][..., layer.interior], drivers["water.phi_boundary"], {},
        drivers["water.sources"],
    )
    assert float(residual.abs().max()) < 1e-11


def test_darcy_weisbach_switches_the_layer_to_pressure():
    model, state, drivers = build_model(twoloop(), headloss="D-W")
    assert model.potential["water"].quantity == "pressure"
    assert model.head_scale == pytest.approx(998.2 * 9.80665, rel=1e-15)
    assert float(drivers["water.phi_boundary"][0]) == pytest.approx(
        50.0 * 998.2 * 9.80665, rel=1e-15
    )
    assert "headloss" in model.notes


def test_pda_replaces_the_nodal_sources_with_a_node_source():
    model, _, drivers = build_model(
        twoloop(), pda=True, p_min=0.0, p_req=60.0
    )
    assert float(drivers["water.sources"].abs().max()) == 0.0
    sources = model.potential["water"]._node_sources
    assert len(sources) == 1
    assert sources[0].nodes.tolist() == [0, 1, 2, 3, 4, 5]
    assert "demand_model" in model.notes


def test_a_quality_layer_carries_the_demand_as_a_removal_rate():
    model, _, _ = build_model(twoloop(), quality=0.5)
    layer = model.transport["quality"]
    demand = torch.tensor([j.demand for j in twoloop().junctions], dtype=F64)
    expected = demand / layer.capacity + 0.5 / 86400.0
    assert layer.removal.squeeze(-1).tolist() == pytest.approx(
        expected.tolist(), rel=1e-14
    )
    assert "quality_removal" in model.notes
    assert "quality_capacity" in model.notes


def test_a_junction_capacity_is_half_of_every_incident_pipe():
    model, _, _ = build_model(twoloop(), quality=0.0)
    net = twoloop()
    volume = {p.name: torch.pi * p.diameter**2 / 4.0 * p.length for p in net.pipes}
    expected = 0.5 * (volume["P1"] + volume["P2"] + volume["P4"])
    assert float(model.transport["quality"].capacity[0]) == pytest.approx(
        expected, rel=1e-14
    )


def test_a_junction_touching_no_pipe_is_refused_for_quality():
    net = WaterNetwork(
        junctions=(Junction("J1", 0.0, 0.005), Junction("J2", 0.0, 0.0)),
        reservoirs=(Reservoir("R1", 50.0),),
        tanks=(),
        pipes=(WaterPipe("P1", "R1", "J1", 500.0, 0.3, 130.0),),
        valves=(Valve("V1", "J1", "J2", 0.2, "TCV", 5.0),),
    )
    with pytest.raises(ValueError, match="touch no pipe"):
        build_model(net, quality=0.0)


def test_an_unmodelled_headloss_argument_is_refused():
    with pytest.raises(ValueError, match="must be 'H-W' or 'D-W'"):
        build_model(twoloop(), headloss="C-M")


# --------------------------------------------------------------------- [OPTIONS] (N5)
def test_pda_defaults_from_the_networks_own_options():
    """`build_model(net)` with NO `pda=`/`p_min=`/... reads them from `net.options`,
    exactly as a `DEMAND MODEL PDA` `.inp` would set them (row D5)."""
    net = dataclasses.replace(
        twoloop(),
        options=WaterOptions(
            demand_model="PDA", minimum_pressure=0.0, required_pressure=60.0,
            pressure_exponent=0.5,
        ),
    )
    model, _, drivers = build_model(net)
    assert float(drivers["water.sources"].abs().max()) == 0.0
    sources = model.potential["water"]._node_sources
    assert len(sources) == 1
    assert sources[0].p_min == 0.0
    assert sources[0].p_req == 60.0
    assert "demand_model" in model.notes


def test_a_non_default_viscosity_on_hazen_williams_is_refused():
    net = dataclasses.replace(twoloop(), options=WaterOptions(viscosity=1.5))
    with pytest.raises(ValueError, match="VISCOSITY"):
        build_model(net)


def test_a_non_default_specific_gravity_on_hazen_williams_is_refused():
    net = dataclasses.replace(twoloop(), options=WaterOptions(specific_gravity=1.1))
    with pytest.raises(ValueError, match="SPECIFIC GRAVITY"):
        build_model(net)


def _hand_head_losses(q, length, diameter, eps, nu):
    """Darcy-Weisbach head loss (m) for a VOLUMETRIC flow q (m3/s), by hand.

    EPANET 2.2 (Manual section 13.1; hydraul.c): Re = 4 q / (pi D nu) with nu the
    KINEMATIC viscosity, nu = VISCOSITY x nu_water(20 C), and h = f L/D V^2 / (2 g) in
    metres of the flowing fluid -- SPECIFIC GRAVITY does not enter the head loss. Returns
    (Re, h with the Colebrook form `Duct` uses [CONTAM TN 1887r1 eq. 50], h with the
    Swamee-Jain f EPANET uses above Re = 4000).
    """
    area = math.pi * diameter**2 / 4.0
    velocity = q / area
    reynolds = velocity * diameter / nu
    rel = eps / diameter
    g = 8.0
    for _ in range(200):
        g = 1.14 - 2.0 * math.log10(rel) - 2.0 * math.log10(1.0 + 9.3 / (reynolds * rel / g))
    swamee = 0.25 / math.log10(rel / 3.7 + 5.74 / reynolds**0.9) ** 2
    scale = length / diameter * velocity**2 / (2.0 * 9.80665)
    return reynolds, scale / g**2, scale * swamee


def _single_dw_pipe(options: WaterOptions, minor_loss: float = 0.0) -> WaterNetwork:
    return WaterNetwork(
        junctions=(Junction("J1", 0.0, 0.05),),
        reservoirs=(Reservoir("R1", 50.0),),
        pipes=(WaterPipe("P1", "R1", "J1", 500.0, 0.3, 0.26e-3, minor_loss),),
        headloss="D-W",
        options=options,
    )


@pytest.mark.parametrize(
    "gravity, viscosity", [(1.0, 1.0), (1.1, 1.0), (1.0, 1.5), (1.1, 1.5)]
)
def test_darcy_weisbach_head_loss_matches_the_hand_computed_epanet_value(
    gravity, viscosity
):
    """Flows are m3/s and heads metres; nu = VISCOSITY x nu_w, and SPECIFIC GRAVITY
    changes only the pressure scale, never the head loss (EPANET 2.2's definitions)."""
    net = _single_dw_pipe(WaterOptions(specific_gravity=gravity, viscosity=viscosity))
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers, atol=1e-12, rtol=1e-12)
    head = final["water.phi"] / model.head_scale
    names = net.nodes()
    loss = float(head[names.index("R1")] - head[names.index("J1")])
    nu = 1.002e-3 / 998.2 * viscosity
    reynolds, colebrook, swamee_jain = _hand_head_losses(0.05, 500.0, 0.3, 0.26e-3, nu)
    assert reynolds > 4000.0  # turbulent, so EPANET is on its Swamee-Jain branch
    assert float(final["water.q"][0]) == pytest.approx(0.05, rel=1e-9)
    assert loss == pytest.approx(colebrook, rel=1e-6)
    # EPANET's own value: Swamee-Jain is an explicit fit to Colebrook, within ~1 %
    assert loss == pytest.approx(swamee_jain, rel=2e-2)


@pytest.mark.parametrize("gravity", [1.0, 1.1])
def test_a_darcy_weisbach_pipe_carries_its_minor_loss(gravity):
    """A [PIPES] minor-loss coefficient K adds K V^2 / (2 g) to the D-W head loss, as in
    EPANET (Manual section 13.1), rather than being dropped."""
    net = _single_dw_pipe(WaterOptions(specific_gravity=gravity), minor_loss=5.0)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers, atol=1e-12, rtol=1e-12)
    head = final["water.phi"] / model.head_scale
    names = net.nodes()
    loss = float(head[names.index("R1")] - head[names.index("J1")])
    _, colebrook, _ = _hand_head_losses(0.05, 500.0, 0.3, 0.26e-3, 1.002e-3 / 998.2)
    velocity = 0.05 / (math.pi * 0.3**2 / 4.0)
    expected = colebrook + 5.0 * velocity**2 / (2.0 * 9.80665)
    assert expected == pytest.approx(0.996, abs=1e-3)
    assert loss == pytest.approx(expected, rel=1e-6)


def test_a_non_default_viscosity_and_gravity_reach_the_darcy_weisbach_duct():
    """The Duct works in mass-flow form; fed rho' = 1/rho and mu' = nu it returns the
    VOLUMETRIC flow the water layer balances, with Re = V D / nu."""
    net = dataclasses.replace(
        twoloop(), options=WaterOptions(specific_gravity=1.1, viscosity=1.5)
    )
    model, _, _ = build_model(net, headloss="D-W")
    duct = model.potential["water"]._elements[0]
    assert duct.rho == pytest.approx(1.0 / (998.2 * 1.1), rel=1e-15)
    assert duct.mu == pytest.approx(1.002e-3 / 998.2 * 1.5, rel=1e-15)
    assert model.head_scale == pytest.approx(998.2 * 1.1 * 9.80665, rel=1e-15)


def test_initial_state_refuses_an_unknown_quantity():
    model, _, _ = build_model(twoloop())
    model.potential["water"].quantity = "nonsense"
    with pytest.raises(ValueError, match="nonsense"):
        initial_state(model)


def test_initial_state_matches_the_builders_own_state():
    model, state, _ = build_model(twoloop())
    fresh = initial_state(model)
    assert set(fresh) == set(state)
    for key in fresh:
        assert torch.equal(fresh[key], state[key])


# ------------------------------------------------------------------- report
def test_pressure_head_and_kilopascal():
    model, state, drivers = build_model(twoloop())
    final = water_steady(model, state, drivers)
    head = pressure_head(model, final)
    # every junction sits at elevation 0, so the pressure head IS the solved head
    assert float(head[0]) == pytest.approx(49.267724991, abs=1e-9)
    assert float(to_kilopascal(torch.tensor(10.0, dtype=F64))) == pytest.approx(
        10.0 * 998.2 * 9.80665 / 1000.0, rel=1e-15
    )


def test_link_table_writes_every_link(tmp_path):
    model, state, drivers = build_model(twoloop())
    final = water_steady(model, state, drivers)
    path = tmp_path / "links.csv"
    rows = link_table(model, final, path=path)
    assert [row["link"] for row in rows] == list(model.link_names)
    assert rows[0]["flow"] == pytest.approx(0.045, abs=1e-12)
    assert rows[0]["velocity"] == pytest.approx(
        0.045 / (torch.pi * 0.3**2 / 4.0), rel=1e-12
    )
    with path.open(newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 8
    assert set(written[0]) == {"link", "flow", "headloss", "velocity"}


def test_tank_inflow_is_the_net_inflow_at_the_tank():
    """A tank-free network has no tank rows, so this is exercised on Net1 in D3; here the
    identity itself is checked on the reservoir, which is the same accumulation."""
    model, state, drivers = build_model(twoloop())
    final = water_steady(model, state, drivers)
    layer = model.potential["water"]
    inflow = -layer._accumulate(final["water.q"])[..., layer.bound]
    # the reservoir supplies the whole demand
    assert float(inflow[0]) == pytest.approx(-0.045, abs=1e-12)
    assert model.tank_closure is None


# --------------------------------------------------------- M4-R13: q_max is refused by name
def test_water_steady_refuses_a_converged_pump_flow_beyond_q_max():
    """A single pump between a reservoir and a junction whose demand forces q > q_max.

    The pump's curve (a single point at q_d=1.0, h_d=30.0) fits h0=40, r=10, so
    q_max = (h0/r)**0.5 = 2.0. A junction demand of 5.0 has no other outlet, so the
    steady solve is FORCED to deliver exactly that flow through the pump (the nodal
    balance at the junction is q = demand, regardless of dp) -- comfortably beyond
    q_max, and comfortably within reach of the closed-form extrapolation this element
    keeps for Newton's own sake (ruling M4-R13).
    """
    net = WaterNetwork(
        junctions=(Junction("J1", 0.0, 5.0),),
        reservoirs=(Reservoir("R1", 250.0),),
        pumps=(Pump("PU1", "R1", "J1", "C1"),),
        curves={"C1": ((1.0, 30.0),)},
    )
    model, state, drivers = build_model(net)
    with pytest.raises(RuntimeError, match=r"pump 'PU1'.*instance 0") as excinfo:
        water_steady(model, state, drivers)
    assert "q_max" in str(excinfo.value)


def test_net1_pump_stays_below_q_max_and_is_not_refused():
    """The converse of the refusal above: Net1's own duty point never approaches q_max
    (spec row D2), so `water_steady` must NOT raise on it."""
    data = Path(__file__).resolve().parents[2] / "data" / "water"
    net = read_epanet_inp(data / "Net1.inp")
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    assert torch.isfinite(final["water.phi"]).all()
