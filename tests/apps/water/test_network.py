"""WaterNetwork validation, the builder, the tank closure and the report helpers."""

import csv
import dataclasses
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.water.controls import CLOSED, OPEN, Control
from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import (
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
    link_states,
    power_curve,
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
def test_a_closed_pipe_or_check_valve_is_valid(status):
    _minimal(
        pipes=(WaterPipe("P1", "R1", "J1", 500.0, 0.3, 130.0, status=status),)
    ).validate()


@pytest.mark.parametrize("link, value, match", [
    ("P1", "AJAR", "only OPEN, CLOSED, ACTIVE"),
    ("P1", "ACTIVE", "only a valve can take"),
    ("PU1", 0.8, "pump speed settings are out of scope"),
])
def test_an_unmodelled_status_line_is_refused(link, value, match):
    with pytest.raises(ValueError, match=match):
        _minimal(pumps=(Pump("PU1", "R1", "J1", "C1"),), curves={"C1": ((0.004, 10.0),)},
                 initial_status={link: value}).validate()


def test_an_unknown_pipe_status_is_refused():
    with pytest.raises(ValueError, match="only OPEN, CLOSED and CV"):
        _minimal(
            pipes=(WaterPipe("P1", "R1", "J1", 500.0, 0.3, 130.0, status="AJAR"),)
        ).validate()


def test_a_constant_power_pump_needs_a_positive_power():
    _minimal(pumps=(Pump("PU1", "R1", "J1", None, power=50.0),)).validate()
    with pytest.raises(ValueError, match="needs a positive power"):
        _minimal(pumps=(Pump("PU1", "R1", "J1", None, power=0.0),)).validate()


def test_a_pump_with_no_curve_and_no_power_is_refused():
    with pytest.raises(ValueError, match="neither a HEAD curve nor a POWER"):
        _minimal(pumps=(Pump("PU1", "R1", "J1", None),)).validate()


def test_a_pump_naming_an_unknown_curve_is_refused():
    with pytest.raises(ValueError, match=r"names curve 'C9'"):
        _minimal(pumps=(Pump("PU1", "R1", "J1", "C9"),)).validate()


def test_a_multipoint_pump_curve_needs_falling_heads_and_rising_flows():
    """EPANET refuses a segmented pump curve whose heads do not fall (error 227) or whose
    flows do not rise (error 230)."""
    pumps = (Pump("PU1", "R1", "J1", "C1"),)
    _minimal(pumps=pumps, curves={"C1": ((0.0, 100.0), (0.05, 80.0), (0.1, 40.0),
                                         (0.15, 0.0))}).validate()
    for points in (((0.0, 100.0), (0.05, 80.0), (0.1, 80.0), (0.15, 0.0)),
                   ((0.0, 100.0), (0.05, 80.0), (0.05, 40.0), (0.15, 0.0))):
        with pytest.raises(ValueError, match="flows rising and its heads falling"):
            _minimal(pumps=pumps, curves={"C1": points}).validate()


@pytest.mark.parametrize("points, match", [
    (((0.0, 30.0), (0.1, 31.0), (0.2, 10.0)), "EPANET error 227"),     # heads rise
    (((0.0, 30.0), (0.1, 10.0), (0.1000001, 9.0)), "EPANET error 227"),  # flows too close
    (((0.0, 30.0), (0.1, 29.99999), (0.2, 0.0)), r"exponent .* outside \(0, 20\]"),
])
def test_a_three_point_curve_without_a_power_fit_is_refused(points, match):
    """EPANET's `powercurve` (`input3.c`) fails, and EPANET stops with error 227."""
    with pytest.raises(ValueError, match=match):
        _minimal(pumps=(Pump("PU1", "R1", "J1", "C1"),), curves={"C1": points}).validate()


def test_a_three_point_curve_takes_its_fitted_exponent():
    """``n = ln((h0 - h2) / (h0 - h1)) / ln(q2 / q1)``, ``r = (h0 - h1) / q1^n``
    (`input3.c`, `powercurve`)."""
    h0, r, n = power_curve(((0.0, 104.0), (2000.0, 92.0), (4000.0, 63.0)))
    assert n == pytest.approx(math.log(41 / 12) / math.log(2), rel=1e-15)
    assert (h0, r) == pytest.approx((104.0, 12.0 / 2000.0**n), rel=1e-15)


@pytest.mark.parametrize("kind, u, v", [("PRV", "J1", "R1"), ("PSV", "R1", "J1"),
                                        ("PRV", "J1", "T1"), ("PSV", "T1", "J1")])
def test_a_pressure_valve_controlling_a_reservoir_or_tank_is_refused(kind, u, v):
    """A PRV holds its end node's head, a PSV its start node's: a reservoir's or tank's
    head is fixed already (EPANET refuses either, error 219)."""
    tanks = (Tank("T1", 10.0, 2.0, 0.0, 5.0, 10.0),)
    held = v if kind == "PRV" else u
    with pytest.raises(ValueError, match=rf"{kind} 'V1' controls the head of '{held}'"):
        _minimal(tanks=tanks, valves=(Valve("V1", u, v, 0.2, kind, 30.0),)).validate()


@pytest.mark.parametrize("kind", ["PRV", "PSV"])
def test_a_pressure_valve_needs_a_non_negative_setting(kind):
    junctions = (Junction("J1", 0.0, 0.005), Junction("J2", 0.0, 0.0))
    with pytest.raises(ValueError, match=rf"{kind} 'V1' needs a pressure setting"):
        _minimal(junctions=junctions,
                 valves=(Valve("V1", "J1", "J2", 0.2, kind, -1.0),)).validate()


@pytest.mark.parametrize("first, second", [
    (("PRV", "J1", "J2"), ("PRV", "J3", "J2")),   # two PRVs sharing their end node
    (("PRV", "J1", "J2"), ("PRV", "J2", "J3")),   # two PRVs in series
    (("PSV", "J1", "J2"), ("PSV", "J1", "J3")),   # two PSVs sharing their start node
    (("PSV", "J1", "J2"), ("PSV", "J3", "J1")),   # two PSVs in series
    (("PRV", "J1", "J2"), ("PSV", "J2", "J3")),   # a PSV starting at a PRV's end
    (("FCV", "J1", "J2"), ("PSV", "J2", "J3")),   # a PSV starting at an FCV's end
    (("FCV", "J2", "J3"), ("PRV", "J1", "J2")),   # a PRV ending at an FCV's start
])
def test_pressure_valves_sharing_a_controlled_node_are_refused(first, second):
    """EPANET 2.2's own refusals (`valvecheck`, error 220): each pair holds one node's head
    twice, or a head and a flow through one node. The error names both valves."""
    junctions = tuple(Junction(n, 0.0, 0.0) for n in ("J1", "J2", "J3"))
    valves = (Valve("V1", *first[1:], 0.2, first[0], 1.0),
              Valve("V2", *second[1:], 0.2, second[0], 1.0))
    with pytest.raises(ValueError, match=r"'V2'.*'V1'"):
        _minimal(junctions=junctions, valves=valves).validate()


def test_a_pbv_needs_a_non_negative_setting():
    with pytest.raises(ValueError, match="PBV 'V1' needs a head drop"):
        _minimal(valves=(Valve("V1", "R1", "J1", 0.2, "PBV", -1.0),)).validate()


@pytest.mark.parametrize("curves, match", [
    ({}, r"has no \[CURVES\] entry"),
    ({"HL": ((0.0, 0.0),)}, "at least two points"),
    ({"HL": ((0.0, 0.0), (0.0, 1.0))}, "increasing flows"),
])
def test_a_gpv_needs_a_head_loss_curve(curves, match):
    with pytest.raises(ValueError, match=match):
        _minimal(valves=(Valve("V1", "R1", "J1", 0.2, "GPV", 0.0, curve="HL"),),
                 curves=curves).validate()


def test_an_unknown_valve_type_is_refused():
    with pytest.raises(ValueError, match=r"has type 'XYZ'"):
        _minimal(valves=(Valve("V1", "R1", "J1", 0.2, "XYZ", 1.0),)).validate()


def test_an_unmodelled_headloss_formula_is_refused():
    with pytest.raises(ValueError, match="Z-Q"):
        _minimal(headloss="Z-Q").validate()


def test_every_control_form_is_accepted():
    """Time, clock-time, tank, junction and (with a pipe or a pump as its link) every
    control the reader reads is modelled."""
    _minimal(controls=(Control("P1", OPEN, None, "node", "J1", False, 1.0),
                       Control("P1", 0, None, "time", value=3600.0),
                       Control("P1", 0, None, "clocktime", value=21600.0))).validate()
    _minimal(
        tanks=(Tank("T1", 10.0, 5.0, 1.0, 9.0, 3.0),),
        controls=(Control("P1", OPEN, None, "node", "T1", False, 1.0),),
    ).validate()


def test_a_numeric_setting_on_a_pipe_is_refused():
    with pytest.raises(ValueError, match="not a pipe"):
        _minimal(controls=(Control("P1", None, 0.5, "time", value=0.0),)).validate()


# ---------------------------------------------------------------------- builder
def test_the_builder_returns_a_model_a_state_and_drivers():
    model, state, drivers = build_model(twoloop())
    assert set(model.kirchhoff) == {"water"}
    assert model.transport == {}
    assert model.node_names == ["J1", "J2", "J3", "J4", "J5", "J6", "R1"]
    assert model.link_names == ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    assert [model.net.nodes[i] for i in model.kirchhoff["water"].bound] == ["R1"]
    assert set(state) == {"water.phi", "water.q"}
    assert set(drivers) == {"water.phi_boundary", "water.sources", "water.status",
                            "water.setting", "water.one_way"}
    assert float(drivers["water.phi_boundary"][0]) == 50.0
    assert model.head_scale == 1.0


def test_the_demand_enters_as_a_negative_nodal_source():
    model, _, drivers = build_model(twoloop())
    assert drivers["water.sources"].tolist() == pytest.approx(
        [-0.005, -0.008, -0.006, -0.010, -0.007, -0.009, 0.0], rel=1e-15
    )


def test_the_two_loop_solves_to_the_measured_heads():
    """The same numbers
    `tests/verification/test_water_parity.py::test_two_loop_heads_and_flows` asserts
    against EPANET, pinned here without wntr."""
    model, state, drivers = build_model(twoloop())
    final = water_steady(model, state, drivers)
    assert final["water.phi"].tolist() == pytest.approx(
        [49.267724991, 48.877301673, 48.430865920, 48.213849910, 48.117097853,
         48.122916699, 50.0],
        abs=1e-9,
    )


def test_the_solved_residual_is_at_the_newton_tolerance():
    model, state, drivers = build_model(twoloop(), pipe_form="potential")
    final = water_steady(model, state, drivers)
    layer = model.kirchhoff["water"]
    residual = layer.residual(
        final["water.phi"][..., layer.interior], drivers["water.phi_boundary"], {},
        drivers["water.sources"],
    )
    assert float(residual.abs().max()) < 1e-11


def test_the_two_pipe_forms_give_the_same_solution():
    potential = build_model(twoloop(), pipe_form="potential")
    flow = build_model(twoloop())
    assert flow[0].kirchhoff["water"].implicit_edges.numel() > 0
    assert potential[0].kirchhoff["water"].implicit_edges.numel() == 0
    a, b = water_steady(*potential), water_steady(*flow)
    torch.testing.assert_close(a["water.phi"], b["water.phi"], rtol=0, atol=1e-12)
    torch.testing.assert_close(a["water.q"], b["water.q"], rtol=0, atol=1e-14)
    layer = flow[0].kirchhoff["water"]
    kcl = layer.net_outflow(b["water.q"])[layer.interior] - flow[2]["water.sources"][
        layer.interior]
    assert float(kcl.abs().max()) < 1e-11


def test_the_flow_form_starts_from_epanets_one_foot_per_second():
    model, state, _ = build_model(twoloop())
    pipes = state["water.q"][model.kirchhoff["water"].kind_slice("pipe")]
    d = torch.tensor([p.diameter for p in twoloop().pipes], dtype=torch.float64)
    torch.testing.assert_close(pipes, torch.pi * d**2 / 4.0 * 0.3048, rtol=1e-15, atol=0)
    assert torch.equal(initial_state(model)["water.q"], state["water.q"])


def test_darcy_weisbach_switches_the_layer_to_pressure():
    model, state, drivers = build_model(twoloop(), headloss="D-W")
    assert model.kirchhoff["water"].quantity == "pressure"
    assert model.head_scale == pytest.approx(998.2 * 9.80665, rel=1e-15)
    assert float(drivers["water.phi_boundary"][0]) == pytest.approx(
        50.0 * 998.2 * 9.80665, rel=1e-15
    )
    assert "headloss" in model.notes


def test_pda_replaces_the_nodal_sources_with_a_withdrawal():
    model, _, drivers = build_model(
        twoloop(), pda=True, p_min=0.0, p_req=60.0
    )
    assert float(drivers["water.sources"].abs().max()) == 0.0
    sources = model.kirchhoff["water"]._withdrawals
    assert len(sources) == 1
    assert sources[0].nodes.tolist() == [0, 1, 2, 3, 4, 5]
    assert "demand_model" in model.notes


def test_a_quality_layer_carries_the_withdrawal_as_dilution_and_only_decay_as_removal():
    model, _, _ = build_model(twoloop(), quality=0.5)
    layer = model.transport["quality"]
    # only the bulk decay is a build-time removal; the withdrawal enters through dilution = 1,
    # withdrawals only (an injection is no negative removal)
    assert layer.removal.squeeze(-1).tolist() == pytest.approx([0.5 / 86400.0] * 6, rel=1e-14)
    assert layer.dilution == 1.0
    assert layer.dilution_mode == "withdrawal"
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
    with pytest.raises(ValueError, match="must be 'H-W', 'D-W' or 'C-M'"):
        build_model(twoloop(), headloss="Z-Q")


def test_chezy_manning_refuses_a_non_default_viscosity_by_name():
    net = dataclasses.replace(twoloop(), headloss="C-M", options=WaterOptions(viscosity=0.9))
    with pytest.raises(ValueError, match="Chezy-Manning does not take the fluid's viscosity"):
        build_model(net)
    # a non-default specific gravity is accepted: EPANET uses it for pressures only
    build_model(dataclasses.replace(net, options=WaterOptions(specific_gravity=0.9)))


# --------------------------------------------------------------------- [OPTIONS]
def test_pda_defaults_from_the_networks_own_options():
    """`build_model(net)` with NO `pda=`/`p_min=`/... reads them from `net.options`,
    exactly as a `DEMAND MODEL PDA` `.inp` would set them (checked in
    `tests/verification/test_water_parity.py::test_pressure_driven_demand`)."""
    net = dataclasses.replace(
        twoloop(),
        options=WaterOptions(
            demand_model="PDA", minimum_pressure=0.0, required_pressure=60.0,
            pressure_exponent=0.5,
        ),
    )
    model, _, drivers = build_model(net)
    assert float(drivers["water.sources"].abs().max()) == 0.0
    sources = model.kirchhoff["water"]._withdrawals
    assert len(sources) == 1
    assert sources[0].p_min == 0.0
    assert sources[0].p_req == 60.0
    assert "demand_model" in model.notes


def test_a_non_default_viscosity_on_hazen_williams_is_refused():
    net = dataclasses.replace(twoloop(), options=WaterOptions(viscosity=1.5))
    with pytest.raises(ValueError, match="VISCOSITY"):
        build_model(net)


def test_a_non_default_specific_gravity_on_hazen_williams_moves_no_head():
    """EPANET uses SPECIFIC GRAVITY on H-W only to turn a pressure into a head, which the
    reader does; a network built in SI heads solves the same whatever it says."""
    plain = water_steady(*build_model(twoloop()))
    net = dataclasses.replace(twoloop(), options=WaterOptions(specific_gravity=1.1))
    heavy = water_steady(*build_model(net))
    torch.testing.assert_close(heavy["water.phi"], plain["water.phi"], rtol=0, atol=0)


def _hand_head_losses(q, length, diameter, eps, nu, g=9.80665):
    """Darcy-Weisbach head loss (m) for a VOLUMETRIC flow q (m3/s), by hand.

    Re = 4 q / (pi D nu) with nu the KINEMATIC viscosity, and h = f L/D V^2 / (2 g) in
    metres of the flowing fluid -- SPECIFIC GRAVITY does not enter the head loss. Returns
    (Re, h with the Colebrook form `Duct` uses [CONTAM TN 1887r1 eq. 50], h with the
    Swamee-Jain f EPANET 2.2 uses above Re = 4000).
    """
    area = math.pi * diameter**2 / 4.0
    velocity = q / area
    reynolds = velocity * diameter / nu
    rel = eps / diameter
    root = 8.0
    for _ in range(200):
        root = 1.14 - 2.0 * math.log10(rel) - 2.0 * math.log10(
            1.0 + 9.3 / (reynolds * rel / root)
        )
    swamee = 0.25 / math.log10(rel / 3.7 + 5.74 / reynolds**0.9) ** 2
    scale = length / diameter * velocity**2 / (2.0 * g)
    return reynolds, scale / root**2, scale * swamee


#: EPANET 2.2 computes in feet: water at VISCOS = 1.1e-5 ft2/s and, in the D-W resistance,
#: g = 32.2 ft/s2 (`hydcoeffs.c`, `resistcoeff`) -- not standard gravity.
EPANET_NU_SI = 1.1e-5 * 0.3048**2
EPANET_G_SI = 32.2 * 0.3048


def _single_dw_pipe(options: WaterOptions, minor_loss: float = 0.0) -> WaterNetwork:
    return WaterNetwork(
        junctions=(Junction("J1", 0.0, 0.05),),
        reservoirs=(Reservoir("R1", 50.0),),
        pipes=(WaterPipe("P1", "R1", "J1", 500.0, 0.3, 0.26e-3, minor_loss),),
        headloss="D-W",
        options=options,
    )


def _single_pipe_loss(net, **kw):
    model, state, drivers = build_model(net, **kw)
    final = water_steady(model, state, drivers, atol=1e-12, rtol=1e-12)
    head = final["water.phi"] / model.head_scale
    names = net.nodes()
    assert float(final["water.q"][0]) == pytest.approx(0.05, rel=1e-9)
    return float(head[names.index("R1")] - head[names.index("J1")])


@pytest.mark.parametrize(
    "gravity, viscosity", [(1.0, 1.0), (1.1, 1.0), (1.0, 1.5), (1.1, 1.5)]
)
def test_darcy_weisbach_head_loss_matches_the_hand_computed_epanet_value(
    gravity, viscosity
):
    """The default friction law is EPANET's: Swamee-Jain at this Re, with EPANET's own
    water (1.1e-5 ft2/s x VISCOSITY) and 32.2 ft/s2. Flows are m3/s and heads metres;
    SPECIFIC GRAVITY changes only the pressure scale, never the head loss."""
    net = _single_dw_pipe(WaterOptions(specific_gravity=gravity, viscosity=viscosity))
    loss = _single_pipe_loss(net)
    reynolds, _, swamee_jain = _hand_head_losses(
        0.05, 500.0, 0.3, 0.26e-3, EPANET_NU_SI * viscosity, g=EPANET_G_SI
    )
    assert reynolds > 4000.0  # turbulent, so EPANET is on its Swamee-Jain branch
    assert loss == pytest.approx(swamee_jain, rel=1e-9)


@pytest.mark.parametrize(
    "gravity, viscosity", [(1.0, 1.0), (1.1, 1.0), (1.0, 1.5), (1.1, 1.5)]
)
def test_the_colebrook_option_matches_the_hand_computed_colebrook_value(
    gravity, viscosity
):
    """`friction="colebrook"` keeps the generic `Duct`: Colebrook, water at
    1.002e-3 / 998.2 m2/s x VISCOSITY, standard gravity."""
    net = _single_dw_pipe(WaterOptions(specific_gravity=gravity, viscosity=viscosity))
    loss = _single_pipe_loss(net, friction="colebrook")
    nu = 1.002e-3 / 998.2 * viscosity
    _, colebrook, swamee_jain = _hand_head_losses(0.05, 500.0, 0.3, 0.26e-3, nu)
    assert loss == pytest.approx(colebrook, rel=1e-6)
    # Swamee-Jain is an explicit fit to Colebrook: the two laws differ, by ~1 % here
    assert loss == pytest.approx(swamee_jain, rel=2e-2)


@pytest.mark.parametrize("gravity", [1.0, 1.1])
def test_a_darcy_weisbach_pipe_carries_its_minor_loss(gravity):
    """A [PIPES] minor-loss coefficient K adds K V^2 / (2 g) to the D-W head loss, as in
    EPANET, rather than being dropped -- with EPANET's own ``0.02517 K / d^4`` (feet,
    cfs), which is ``8 / (32.2 pi^2)`` rounded to four figures (`input1.c`)."""
    net = _single_dw_pipe(WaterOptions(specific_gravity=gravity), minor_loss=5.0)
    loss = _single_pipe_loss(net)
    _, _, friction = _hand_head_losses(
        0.05, 500.0, 0.3, 0.26e-3, EPANET_NU_SI, g=EPANET_G_SI
    )
    q_cfs = 0.05 / 0.3048**3
    minor = 0.3048 * 0.02517 * 5.0 / (0.3 / 0.3048) ** 4 * q_cfs**2
    velocity = 0.05 / (math.pi * 0.3**2 / 4.0)
    exact = 5.0 * velocity**2 / (2.0 * EPANET_G_SI)
    # 0.02517 is 1.2e-4 below 8 / (32.2 pi^2) = 0.025173, and EPANET uses 0.02517
    assert minor / exact == pytest.approx(0.02517 * 32.2 * math.pi**2 / 8.0, rel=1e-12)
    assert loss == pytest.approx(friction + minor, rel=1e-9)


@pytest.mark.parametrize("gravity", [1.0, 1.1])
def test_a_colebrook_pipe_carries_its_minor_loss(gravity):
    net = _single_dw_pipe(WaterOptions(specific_gravity=gravity), minor_loss=5.0)
    loss = _single_pipe_loss(net, friction="colebrook")
    _, colebrook, _ = _hand_head_losses(0.05, 500.0, 0.3, 0.26e-3, 1.002e-3 / 998.2)
    velocity = 0.05 / (math.pi * 0.3**2 / 4.0)
    expected = colebrook + 5.0 * velocity**2 / (2.0 * 9.80665)
    assert expected == pytest.approx(0.996, abs=1e-3)
    assert loss == pytest.approx(expected, rel=1e-6)


def test_a_non_default_viscosity_and_gravity_reach_the_darcy_weisbach_elements():
    """EPANET's law takes nu = 1.1e-5 ft2/s x VISCOSITY and works in head x rho g; the
    Colebrook Duct works in mass-flow form, fed rho' = 1/rho and mu' = nu, so it returns
    the VOLUMETRIC flow the water layer balances, with Re = V D / nu."""
    net = dataclasses.replace(
        twoloop(), options=WaterOptions(specific_gravity=1.1, viscosity=1.5)
    )
    model, _, _ = build_model(net, headloss="D-W")
    pipe = model.kirchhoff["water"]._laws[0].pipe   # the resistance form wraps it
    assert pipe.nu == pytest.approx(EPANET_NU_SI * 1.5, rel=1e-15)
    assert pipe.scale == pytest.approx(998.2 * 1.1 * 9.80665, rel=1e-15)
    assert pipe.cfs_per_m3s == pytest.approx(1.0 / 0.3048**3, rel=1e-15)
    model, _, _ = build_model(net, headloss="D-W", friction="colebrook")
    duct = model.kirchhoff["water"]._laws[0]
    assert duct.rho == pytest.approx(1.0 / (998.2 * 1.1), rel=1e-15)
    assert duct.mu == pytest.approx(1.002e-3 / 998.2 * 1.5, rel=1e-15)
    assert model.head_scale == pytest.approx(998.2 * 1.1 * 9.80665, rel=1e-15)


@pytest.mark.parametrize("friction", ["swamee"])
def test_an_unknown_friction_law_is_refused(friction):
    with pytest.raises(ValueError, match=rf"friction must be 'composite' or 'colebrook', got "
                                         rf"'{friction}'"):
        build_model(twoloop(), headloss="D-W", friction=friction)


def test_a_files_flow_units_select_epanets_rounded_cfs_factor(tmp_path):
    """An LPS file reaches EPANET as q / 28.317 cfs, not q / 28.3168466: both the D-W and the
    H-W pipe laws take that factor; a network without a flow unit keeps the exact one."""
    net = dataclasses.replace(
        twoloop(), headloss="D-W", options=WaterOptions(flow_units="LPS")
    )
    model, _, _ = build_model(net)
    pipe = model.kirchhoff["water"]._laws[0].pipe   # the resistance form wraps it
    assert pipe.cfs_per_m3s == pytest.approx(1000.0 / 28.317, rel=1e-15)
    model, _, _ = build_model(dataclasses.replace(net, headloss="H-W"))
    pipe = model.kirchhoff["water"]._laws[0].pipe
    assert pipe.cfs_per_m3s == pytest.approx(1000.0 / 28.317, rel=1e-15)
    model, _, _ = build_model(twoloop())         # no file flow unit: the exact conversion
    assert model.kirchhoff["water"]._laws[0].pipe.cfs_per_m3s is None


def test_initial_state_refuses_an_unknown_quantity():
    model, _, _ = build_model(twoloop())
    model.kirchhoff["water"].quantity = "nonsense"
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
    """A tank-free network has no tank rows, so this is exercised on Net1 in
    `tests/verification/test_water_parity.py::test_net1_extended_period_tank_level`; here
    the identity itself is checked on the reservoir, which is the same accumulation."""
    model, state, drivers = build_model(twoloop())
    final = water_steady(model, state, drivers)
    layer = model.kirchhoff["water"]
    inflow = -layer.net_outflow(final["water.q"])[..., layer.bound]
    # the reservoir supplies the whole demand
    assert float(inflow[0]) == pytest.approx(-0.045, abs=1e-12)
    assert model.control_closure is None


# --------------------------------------------------------- q_max is refused by name
def test_water_steady_refuses_a_converged_pump_flow_beyond_q_max():
    """A single pump between a reservoir and a junction whose demand forces q > q_max.

    The pump's curve (a single point at q_d=1.0, h_d=30.0) fits h0=40, r=10, so
    q_max = (h0/r)**0.5 = 2.0. A junction demand of 5.0 has no other outlet, so the
    steady solve is FORCED to deliver exactly that flow through the pump (the nodal
    balance at the junction is q = demand, regardless of dp) -- comfortably beyond
    q_max, and comfortably within reach of the closed-form extrapolation this element
    keeps for Newton's own sake.
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
    (checked in `tests/verification/test_water_parity.py::test_net1_single_period`), so
    `water_steady` must NOT raise on it."""
    data = Path(__file__).resolve().parents[2] / "data" / "water"
    net = read_epanet_inp(data / "Net1.inp")
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    assert torch.isfinite(final["water.phi"]).all()


# ------------------------------------------------------------- pumps in the head-gain form
def _pump_feeding_pda():
    """R - pipe - A - PUMP - B - pipe - C: the smallest network on which a pump with a
    pressure-driven demand downstream went singular when the pump entered as its inverted
    curve `q = g(dp)` (an iterate beyond shut-off left B and C tied to nothing but C's
    demand slope)."""
    return _device_network("pump")


@pytest.mark.parametrize("pipe_form", ["flow", "potential"])
def test_a_pump_feeding_pressure_driven_demand_converges(pipe_form):
    model, state, drivers = build_model(_pump_feeding_pda(), pda=True, p_req=60.0,
                                        pipe_form=pipe_form)
    out = water_steady(model, state, drivers)
    phi, q = out["water.phi"], out["water.q"]
    # the pump lifts B above A by its own curve's gain at its own flow
    pump = model.kirchhoff["water"].law_for("pump")[0].pump
    gain = float(pump.head_gain(q[2:3]))
    assert float(phi[1] - phi[0]) == pytest.approx(gain, rel=1e-10)
    # C (p ~ 59.96 m) is short of its 60 m required pressure: the demand really is limited
    source = model.kirchhoff["water"].withdrawals[0]
    delivered = source.flow(phi[:3])
    assert 0.99 * 0.004 < float(delivered[2]) < 0.004
    assert float(q[2]) == pytest.approx(float(delivered[2]), rel=1e-10)


@pytest.mark.parametrize("pipe_form", ["flow", "potential"])
def test_a_pump_held_beyond_shut_off_is_shut_with_no_flow(pipe_form):
    """R1 (50 m) - PUMP (shut-off 13.3334 m) - J - pipe - R2 (100 m): the pump cannot lift
    to R2, so it is shut, as EPANET closes it (`pumpstatus`, ``XHEAD``): exactly zero flow
    (EPANET's closed link leaks `excess / CBIG`, 3e-8 m3/s here), J at R2's head, and the
    pump's state CLOSED."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.0),),
        reservoirs=(Reservoir("R1", 50.0), Reservoir("R2", 100.0)),
        pipes=(WaterPipe("P", "J", "R2", 500.0, 0.3, 130.0),),
        pumps=(Pump("PU", "R1", "J", "C1"),), curves={"C1": ((0.004, 10.0),)},
    )
    model, state, drivers = build_model(net, pipe_form=pipe_form)
    out = water_steady(model, state, drivers)
    assert float(out["water.q"][-1]) == 0.0
    assert float(out["water.phi"][0]) == pytest.approx(100.0, abs=1e-9)
    assert link_states(model, out, drivers).tolist() == [OPEN, CLOSED]


def test_a_pump_at_speed_zero_is_closed():
    """A speed of 0 in `"water.setting"` closes the pump, as EPANET's `pumpcoeff` does: no
    flow, state CLOSED, and the junction it alone fed, without demand, held at the limit
    head of its closed link (the suction side's)."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.0),),
        reservoirs=(Reservoir("R1", 50.0),),
        pumps=(Pump("PU", "R1", "J", "C1"),), curves={"C1": ((0.004, 10.0),)},
    )
    model, state, drivers = build_model(net)
    drivers = {**drivers, "water.setting": torch.zeros(1, dtype=F64)}
    out = water_steady(model, state, drivers)
    assert float(out["water.q"][0]) == 0.0
    assert float(out["water.phi"][0]) == pytest.approx(50.0, abs=1e-12)
    assert link_states(model, out, drivers).tolist() == [CLOSED]


def test_a_zero_demand_junction_fed_only_by_a_pump_sits_at_its_shut_off_head():
    """With no demand behind it the pump runs at zero flow on its curve (a tie of the
    no-flow and open rows takes the open one): the junction sits at the suction head plus
    the shut-off head, and the pump is OPEN, as EPANET reports it (its `pumpstatus` closes
    a pump only above the shut-off head)."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.0),),
        reservoirs=(Reservoir("R1", 50.0),),
        pumps=(Pump("PU", "R1", "J", "C1"),), curves={"C1": ((0.004, 10.0),)},
    )
    model, state, drivers = build_model(net)
    out = water_steady(model, state, drivers)
    assert abs(float(out["water.q"][0])) < 1e-12
    assert float(out["water.phi"][0]) == pytest.approx(50.0 + 1.33334 * 10.0, abs=1e-9)
    assert link_states(model, out, drivers).tolist() == [OPEN]


# ---------------------------------------------------------- quality: constant-tracer invariants
def _device_network(device):
    """R - pipe - A - DEVICE - B - pipe - C, demands at A and C."""
    junctions = (Junction("A", 0.0, 0.005), Junction("B", 0.0, 0.0),
                 Junction("C", 0.0, 0.004))
    pipes = (WaterPipe("P1", "R", "A", 500.0, 0.3, 130.0),
             WaterPipe("P2", "B", "C", 500.0, 0.3, 130.0))
    kwargs = {}
    if device == "tcv":
        kwargs["valves"] = (Valve("V", "A", "B", 0.2, "TCV", 5.0),)
    elif device == "fcv":
        # a flow-control valve is dp-independent: the nodes behind it need another
        # connection to the reservoir for the head solve, and carry its flow regardless
        kwargs["valves"] = (Valve("V", "A", "B", 0.2, "FCV", 0.002),)
        pipes += (WaterPipe("P3", "R", "C", 500.0, 0.3, 130.0),)
    else:
        kwargs["pumps"] = (Pump("PU", "A", "B", "C1"),)
        kwargs["curves"] = {"C1": ((0.004, 10.0),)}
    return WaterNetwork(junctions=junctions, reservoirs=(Reservoir("R", 50.0),),
                        pipes=pipes, **kwargs)


@pytest.mark.parametrize("pda", [False, True])
@pytest.mark.parametrize("device", ["tcv", "fcv", "pump"])
def test_a_constant_tracer_is_preserved_through_every_device(device, pda):
    kwargs = {"pda": True, "p_req": 60.0} if pda else {}
    model, state, drivers = build_model(_device_network(device), quality=0.0, **kwargs)
    drivers["quality.x_boundary"] = torch.tensor([100.0], dtype=F64)
    out = water_steady(model, state, drivers)
    assert out["quality.x"].tolist() == pytest.approx([100.0] * 3, rel=1e-8)


def test_a_constant_tracer_is_preserved_under_pressure_driven_demand():
    model, state, drivers = build_model(twoloop(), quality=0.0, pda=True, p_req=100.0)
    drivers["quality.x_boundary"] = torch.tensor([100.0], dtype=F64)
    out = water_steady(model, state, drivers)
    # the hydraulics really are demand-limited, so the test discriminates
    source = model.kirchhoff["water"].withdrawals[0]
    flow = source.flow(out["water.phi"][:6])
    assert bool((flow < source.q_required.detach() * 0.999).any())
    assert out["quality.x"].tolist() == pytest.approx([100.0] * 6, rel=1e-8)


# The interplay of quality with the pipe form, the PBV/GPV valves, tanks and PDA. A reservoir
# and an explicit tank (a quality boundary) both hold the tracer at 100; every junction must
# stay at 100 in steady state, however the hydraulics are written.
def _tank_device_network(device):
    """R - pipe - A - DEVICE - B - pipe - C - pipe - T (tank), demands at A and C."""
    junctions = (Junction("A", 0.0, 0.003), Junction("B", 0.0, 0.0),
                 Junction("C", 0.0, 0.002))
    pipes = (WaterPipe("P1", "R", "A", 500.0, 0.3, 130.0),
             WaterPipe("P2", "B", "C", 500.0, 0.3, 130.0),
             WaterPipe("P3", "C", "T", 300.0, 0.3, 130.0))
    kwargs = {}
    if device == "pbv":
        kwargs["valves"] = (Valve("V", "A", "B", 0.2, "PBV", 3.0),)
    elif device == "gpv":
        kwargs["valves"] = (Valve("V", "A", "B", 0.2, "GPV", 0.0, curve="G"),)
        kwargs["curves"] = {"G": ((0.001, 1.0), (0.01, 5.0))}
    else:
        kwargs["valves"] = (Valve("V", "A", "B", 0.2, "TCV", 5.0),)
    tank = Tank("T", elevation=40.0, init_level=5.0, min_level=0.0, max_level=10.0,
                diameter=10.0)
    return WaterNetwork(junctions=junctions, reservoirs=(Reservoir("R", 60.0),),
                        tanks=(tank,), pipes=pipes, **kwargs)


@pytest.mark.parametrize("pda", [False, True])
@pytest.mark.parametrize("pipe_form", [None, "potential", "flow"])
@pytest.mark.parametrize("device", ["tcv", "pbv", "gpv"])
def test_a_constant_tracer_is_preserved_with_a_tank_every_pipe_form_and_valve(
    device, pipe_form, pda
):
    kwargs = {"pda": True, "p_req": 15.0} if pda else {}
    model, state, drivers = build_model(
        _tank_device_network(device), quality=0.0, pipe_form=pipe_form, **kwargs
    )
    drivers["quality.x_boundary"] = torch.full_like(drivers["quality.x_boundary"], 100.0)
    out = water_steady(model, state, drivers)
    assert out["quality.x"].tolist() == pytest.approx([100.0] * 3, rel=1e-8)


@pytest.mark.parametrize("pda", [False, True])
def test_storage_tanks_with_quality_are_refused_explicitly(pda):
    kwargs = {"pda": True, "p_req": 15.0} if pda else {}
    with pytest.raises(ValueError, match="tank_scheme='backward_euler' with quality"):
        build_model(_tank_device_network("tcv"), quality=0.0, tank_scheme="backward_euler",
                    **kwargs)


@pytest.mark.parametrize("scheme, interior", [("forward_euler", False),
                                              ("backward_euler", True)])
def test_a_pressure_valve_leaves_the_tank_scheme_alone(scheme, interior):
    """A PRV's controlled elevations once took the name of the tank-scheme flag in
    `build_model`, so a network with a PRV and a tank stepped inside the solve held its tank
    as a boundary node and the layer refused it (MEASURED at 8c2fa66)."""
    net = _minimal(
        junctions=(Junction("J1", 0.0, 0.001), Junction("J2", 5.0, 0.0)),
        tanks=(Tank("T1", 10.0, 2.0, 0.0, 5.0, 10.0),),
        pipes=(WaterPipe("P1", "R1", "J1", 500.0, 0.3, 130.0),
               WaterPipe("P2", "J2", "T1", 500.0, 0.3, 130.0)),
        valves=(Valve("V1", "J1", "J2", 0.2, "PRV", 30.0),),
    )
    model, _, _ = build_model(net, tank_scheme=scheme)
    tank = model.node_names.index("T1")
    assert (tank in model.kirchhoff["water"].interior.tolist()) is interior


@pytest.mark.parametrize("r2", [None, 100.0])
def test_a_pump_iterate_on_its_floor_tangent_with_reverse_flow_still_converges(r2):
    """An adversarial first iterate: the pump's flow reversed (-1 L/s, on the RQtol
    tangent below its floor flow) with ``0 < r < -kappa q``, so THE SELECTION keeps the
    curve's row and its Newton step is of order ``r / RQtol``. The solve still ends in the
    state it reaches from its own first iterate: on the curve feeding J's demand (no R2),
    or shut beyond its shut-off head under R2 at 100 m."""
    pipes = () if r2 is None else (WaterPipe("P", "J", "R2", 500.0, 0.3, 130.0),)
    reservoirs = (Reservoir("R1", 50.0),) + (() if r2 is None else (Reservoir("R2", r2),))
    net = WaterNetwork(junctions=(Junction("J", 0.0, 0.002),), reservoirs=reservoirs,
                       pipes=pipes, pumps=(Pump("PU", "R1", "J", "C1"),),
                       curves={"C1": ((0.004, 10.0),)})
    model, state, drivers = build_model(net)
    reference = water_steady(model, state, drivers)
    law, cols = model.kirchhoff["water"].law_for("pump")
    q0 = torch.tensor([-1e-3], dtype=F64)
    shut_off = float(law.head_gain(q0, torch.ones(1, dtype=F64)))
    adversarial = {k: v.clone() for k, v in state.items()}
    adversarial["water.q"][cols] = q0
    adversarial["water.phi"][model.node_names.index("J")] = 50.0 + shut_off - 5e-4
    adversarial["water.phi"][model.node_names.index("R1")] = 50.0
    r = float(law.residual(torch.tensor([-(shut_off - 5e-4)], dtype=F64), q0, drivers, None))
    assert 0.0 < r < -float(law.kappa[0]) * float(q0)       # the iterate is the one asked
    final = water_steady(model, adversarial, drivers)
    torch.testing.assert_close(final["water.phi"], reference["water.phi"], rtol=0, atol=1e-9)
    torch.testing.assert_close(final["water.q"], reference["water.q"], rtol=0, atol=1e-12)
    assert link_states(model, final, drivers).tolist() == link_states(
        model, reference, drivers).tolist()


def test_the_shut_check_valves_reach_only_the_pump_evaluation_they_were_made_for():
    """`ShutLinks` hands the check valves the pipe law shut to a pump law evaluated with the
    same drivers mapping (one layer evaluation), and nothing to one evaluated with drivers
    of its own, which tests its pumps alone."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.0005), Junction("K", 0.0, 0.0)),
        reservoirs=(Reservoir("R1", 50.0), Reservoir("R2", 100.0)),
        pipes=(WaterPipe("P1", "J", "K", 100.0, 0.3, 130.0),
               WaterPipe("P2", "K", "R2", 100.0, 0.3, 130.0, status="CV")),
        pumps=(Pump("PU", "R1", "J", "C1"),), curves={"C1": ((0.004, 10.0),)},
    )
    model, state, drivers = build_model(net)
    layer = model.kirchhoff["water"]
    pipes, pcols = layer.law_for("pipe")
    pumps, ucols = layer.law_for("pump")
    # an iterate with P2 flowing backwards and PU beyond its shut-off head: both shut
    q = torch.tensor([0.0, -0.01, -0.001], dtype=F64)
    dp_pipes = torch.tensor([0.0, -5.0], dtype=F64)
    dp_pump = torch.tensor([-60.0], dtype=F64)
    pipes(dp_pipes, q[pcols], drivers, None)
    shut = pumps.shut_links.links_for(drivers)
    assert shut is not None and shut.tolist() == [False, True, False]
    assert pumps.shut_links.links_for(dict(drivers)) is None
    # the pump law with its own drivers: its pumps alone, so PU is shut (J looks fed
    # through P2); with the layer's, P2 is shut too and PU is kept on its curve
    alone = pumps.states(dp_pump, q[ucols], dict(drivers))
    joint = pumps.states(dp_pump, q[ucols], drivers)
    assert alone.tolist() == [CLOSED] and joint.tolist() == [OPEN]
