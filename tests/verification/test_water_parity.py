"""EPANET 2.2 parity for `noodl.apps.water`.

The reference implementation is the REAL EPANET 2.2 engine: `wntr` 1.5.0 bundles `epanet22.dll` and
`wntr.sim.EpanetSimulator` drives it. `EpanetSimulator` reads EPANET's binary output, whose
on-disk reals are FLOAT32 (`wntr/epanet/io.py`, `ftype = '=f4'`, matching the manual's
Appendix C.4), so roughly 1e-7 relative is the reference's own floor and tightening
`ACCURACY`/`HEADERROR`/`FLOWCHANGE` is bit-identical -- measured on both committed
fixtures. Every test copies its fixture into `tmp_path` and passes `file_prefix`, so
EPANET's `.rpt`/`.bin` side files never land in the repository.

Newton runs at `water_steady`'s own defaults, `atol=rtol=1e-11` and `max_iter=200`. Both
are measured, and both matter here: 1e-14 does not converge at all (the residual floors at
1.3e-14 against a flow scale of 0.1 m3/s), 1e-12 fails at Net1's demand multiplier 1.2, and
50 Newton iterations fail outright at the step where a control closes the pump and leaves
pipe 10 dead-ended (136 are needed there). See `apps.water.network.water_steady`.
"""

import math
import warnings
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from noodl.apps.water.demand import EmitterOutflow, PressureDrivenDemand
from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import (
    build_model,
    initial_state,
    link_states,
    tank_inflow,
    tank_levels,
    tank_overflow,
    twoloop,
    water_event_step,
    water_steady,
)

wntr = pytest.importorskip("wntr")

F64 = torch.float64
DATA = Path(__file__).resolve().parents[1] / "data" / "water"


def _epanet(tmp_path, name, *, edits=(), duration=None):
    """Copy a committed fixture into `tmp_path`, optionally edit it, and run EPANET.

    Returns `(path, results)`. `edits` is a sequence of `(old, new)` literal replacements
    applied to the file's text, which is how the D-W and PDA variants are produced without
    committing a second fixture for each.
    """
    path = tmp_path / name
    text = (DATA / name).read_text()
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new)
    path.write_text(text)
    model = wntr.network.WaterNetworkModel(str(path))
    if duration is not None:
        model.options.time.duration = duration
    results = wntr.sim.EpanetSimulator(model).run_sim(file_prefix=str(tmp_path / "epanet"))
    return path, results


def _worst_heads(model, state, heads, skip=()):
    """The worst relative head difference; a node at head 0 (a reservoir at the datum, held
    exactly) is compared absolutely."""
    return max(
        abs(float(state["water.phi"][i]) / model.head_scale - float(heads[name]))
        / (abs(float(heads[name])) or 1.0)
        for i, name in enumerate(model.node_names)
        if name not in skip
    )


def _worst_flows(model, state, flows):
    return max(
        abs(float(state["water.q"][i]) - float(flows[name])) / abs(float(flows[name]))
        for i, name in enumerate(model.link_names)
    )


# --------------------------------------------------------------------------- two-loop
def test_two_loop_heads_and_flows(tmp_path):
    """1e-6 relative on heads and flows.

    MEASURED: heads 4.361e-7, flows 8.090e-8 -- the same pair an independent
    from-scratch Newton/GGA reference (iterated to a 1e-16 residual) reaches, so the
    residual is the float32 output path, not either solver.
    """
    path, results = _epanet(tmp_path, "twoloop_si.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    heads = results.node["head"].loc[0]
    flows = results.link["flowrate"].loc[0]
    worst_h = _worst_heads(model, final, heads, skip=("R1",))
    worst_q = _worst_flows(model, final, flows)
    assert worst_h < 1e-6, worst_h
    assert worst_q < 1e-6, worst_q


def test_reservoir_head_is_held_exactly(tmp_path):
    path, _ = _epanet(tmp_path, "twoloop_si.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    assert float(final["water.phi"][model.node_names.index("R1")]) == 50.0


def test_nodal_continuity_is_exact(tmp_path):
    """This model's own flows satisfy continuity to machine precision, which is why the
    two-loop and Net1 residuals above are attributable to the reference: EPANET's reported
    flows miss continuity at Net1's node 13 by 4.5e-10 m3/s (measured)."""
    path, _ = _epanet(tmp_path, "twoloop_si.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    layer = model.kirchhoff["water"]
    imbalance = (layer.net_outflow(final["water.q"]) - drivers["water.sources"])[layer.interior]
    # MEASURED 2.093e-14 on the potential form, at J1 (the 0.045 m3/s pipe next to the
    # reservoir); Newton's own atol here is 1e-11, so this is as converged as the solve was
    # asked to be.
    assert float(imbalance.abs().max()) < 1e-13


# --------------------------------------------------------------------------- Net1 single period
def test_net1_single_period(tmp_path):
    """Heads 1e-6, flows 1e-6 relative, pump head gain 1e-6, pump flow 1e-8 m3/s.

    MEASURED: heads 7.224e-8, flows 2.647e-7, pump head gain 1.942e-7, pump flow 3.2e-9
    m3/s. With the pump curve's exponent fixed at 2 instead of EPANET's 1.99998
    (`three_point_curve`) the flows were 2.868e-6 apart (pipe 113, 0.00185 m3/s, the
    smallest in the network) and the pump's 1.15e-7 m3/s. EPANET's own reported flows miss
    continuity at node 13 by 4.5e-10 m3/s while this model's satisfy it to 1e-17, and
    tightening ACCURACY/HEADERROR/FLOWCHANGE to 1e-8/1e-9/1e-9 with 500 trials leaves
    EPANET's output BIT-IDENTICAL.
    """
    path, results = _epanet(tmp_path, "Net1.inp", duration=0)
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    heads = results.node["head"].loc[0]
    flows = results.link["flowrate"].loc[0]
    headloss = results.link["headloss"].loc[0]
    worst_h = _worst_heads(model, final, heads, skip=("9", "2"))
    worst_q = _worst_flows(model, final, flows)
    assert worst_h < 1e-6, worst_h
    assert worst_q < 1e-6, worst_q
    pump = model.link_names.index("9")
    assert abs(float(final["water.q"][pump]) - float(flows["9"])) < 1e-8
    gain = float(
        final["water.phi"][model.node_names.index("10")]
        - final["water.phi"][model.node_names.index("9")]
    )
    expected_gain = -float(headloss["9"])
    assert abs(gain - expected_gain) / abs(expected_gain) < 1e-6


def test_the_reader_converts_net1_gpm_and_feet(tmp_path):
    """The unit conversion the Net1 single-period check rests on, pinned field by field."""
    path, _ = _epanet(tmp_path, "Net1.inp", duration=0)
    net = read_epanet_inp(path)
    assert net.notes["flow_units"] == "GPM"
    assert float(net.reservoirs[0].head) == pytest.approx(800 * 0.3048, rel=0.0)
    pipe = next(p for p in net.pipes if p.name == "10")
    assert pipe.length == pytest.approx(10530 * 0.3048, rel=1e-15)
    assert pipe.diameter == pytest.approx(18 * 0.0254, rel=1e-15)
    assert pipe.roughness == 100.0
    tank = net.tanks[0]
    assert tank.elevation == pytest.approx(850 * 0.3048, rel=1e-15)
    assert tank.init_level == pytest.approx(120 * 0.3048, rel=1e-15)
    assert tank.diameter == pytest.approx(50.5 * 0.3048, rel=1e-15)
    assert net.curves["1"] == ((1500 * 3.785411784e-3 / 60.0, 250 * 0.3048),)
    assert [(c.link, c.status, c.trigger, c.node, c.above) for c in net.controls] == [
        ("9", 1, "node", "2", False), ("9", 0, "node", "2", True)
    ]
    assert [c.value for c in net.controls] == pytest.approx(
        [110 * 0.3048, 140 * 0.3048], rel=1e-15
    )
    assert net.patterns["1"] == (1.0, 1.2, 1.4, 1.6, 1.4, 1.2,
                                 1.0, 0.8, 0.6, 0.4, 0.6, 0.8)
    assert (net.duration, net.hydraulic_timestep, net.pattern_timestep) == (
        86400.0, 3600.0, 7200.0
    )


# --------------------------------------------------------------------------- Net1 extended period
def _pattern_factor(net, seconds):
    """EPANET's demand pattern at `seconds`: `floor(t / pattern step) mod len`."""
    if net.demand_pattern is None or net.demand_pattern not in net.patterns:
        return 1.0
    values = net.patterns[net.demand_pattern]
    return values[int(seconds // net.pattern_timestep) % len(values)]


def _extended_period(net, model, state, drivers):
    """Net1's 24 h duty cycle with EPANET's EVENT-SHORTENED hydraulic step.

    Manual section 13.1 item 17, p.113: the next step is the minimum of the nominal step
    and the time until a tank level "reaches a point that triggers a change in status for
    some link", the level "assumed to change in a linear fashion based on the current flow
    solution". `water_event_step` takes one such step (one solve, the tank held at its
    step-start head, then moved by forward Euler); `TankLevels.__call__` re-applies the
    controls at the start of every sub-step.

    Returns the tank level at every REPORT time, starting at t = 0.
    """
    base = drivers["water.sources"].clone()
    levels = [float(tank_levels(model, state)[0])]
    time = 0.0
    n_sub = 0
    while time < net.duration - 1e-9:
        report_end = time + net.report_timestep
        while time < report_end - 1e-9:
            step_drivers = dict(drivers)
            step_drivers["water.sources"] = base * _pattern_factor(net, time)
            state, dt = water_event_step(model, state, step_drivers, report_end - time)
            time += dt
            n_sub += 1
        levels.append(float(tank_levels(model, state)[0]))
    return levels, n_sub


def test_net1_extended_period_tank_level(tmp_path):
    """2e-4 m ABSOLUTE per reported step.

    MEASURED: 3.826e-5 m worst over the 25 reported steps, with 26 hydraulic sub-steps for
    24 report steps. A FIXED 1 h step instead diverges by 2.07 m, because EPANET shortens
    its own step to the instant the level reaches the 140 ft trigger and a fixed step
    switches the pump a whole hour late. The 3.8e-5 m floor is the reference's float32 head
    output, whose resolution at 296 m is 3.5e-5 m (8.2e-5 m with the pump curve's exponent
    fixed at 2 instead of EPANET's 1.99998).
    """
    path, results = _epanet(tmp_path, "Net1.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    levels, n_sub = _extended_period(net, model, state, drivers)
    reference = results.node["head"]["2"] - net.tanks[0].elevation
    assert len(levels) == len(reference)
    worst = max(
        abs(mine - float(theirs))
        for mine, theirs in zip(levels, reference.tolist(), strict=True)
    )
    assert worst < 2e-4, worst
    # The event shortening costs a handful of extra solves, not a finer grid throughout.
    assert n_sub <= 2 * len(reference)


def test_the_pump_switches_within_one_reported_step(tmp_path):
    """The switch TIME must agree, which is what the event shortening buys."""
    path, results = _epanet(tmp_path, "Net1.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    levels, _ = _extended_period(net, model, state, drivers)
    reference = (results.node["head"]["2"] - net.tanks[0].elevation).tolist()
    mine_peak = levels.index(max(levels))
    their_peak = reference.index(max(reference))
    assert abs(mine_peak - their_peak) <= 1


def test_event_step_returns_the_nominal_step_when_nothing_is_crossed(tmp_path):
    path, _ = _epanet(tmp_path, "Net1.inp")
    net = read_epanet_inp(path)
    model, _, _ = build_model(net)
    closure = model.tank_closure
    assert closure.event_step(
        torch.tensor([36.576], dtype=F64), torch.tensor([0.0], dtype=F64), 3600.0
    ) == 3600.0
    # rising towards the 42.672 m upper trigger at 1e-3 m/s crosses it in 6096 s, which is
    # longer than the 3600 s step, so the step is not shortened
    assert closure.event_step(
        torch.tensor([36.576], dtype=F64), torch.tensor([1e-3], dtype=F64), 3600.0
    ) == 3600.0
    # from 42.0 m the same rate crosses in 672 s, so the step IS shortened to it
    assert closure.event_step(
        torch.tensor([42.0], dtype=F64), torch.tensor([1e-3], dtype=F64), 3600.0
    ) == pytest.approx(672.0, abs=1e-6)


def test_a_tank_outside_its_limits_is_refused_by_the_implicit_schemes(tmp_path):
    """Stepped inside the solve a tank has no full or empty rule, so a level outside its
    limits is refused; EPANET's forward Euler closes links instead (`TankLevels`)."""
    path, _ = _epanet(tmp_path, "Net1.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net, tank_scheme="backward_euler")
    state = initial_state(model, values={"water.tank_level": {"2": 100.0}})
    with pytest.raises(ValueError, match=r"tank\(s\) \['2'\] are outside"):
        water_steady(model, state, drivers)


# --------------------------------------------------------------------------- Darcy-Weisbach
# EPANET 2.2's D-W is a COMPOSITE friction law (`hydcoeffs.c`, `DWpipecoeff` and
# `frictionFactor`): Hagen-Poiseuille below Re = 2000, Dunlop's cubic to 4000, Swamee-Jain
# above, evaluated in feet and cfs with g = 32.2 ft/s^2 and water at 1.1e-5 ft^2/s.
# `build_model` reproduces it (`CompositeDarcyWeisbach`, `friction="composite"`, the default), so
# these tests hold D-W to the SAME tolerances as the Hazen-Williams tests above, which are
# the float32 output floor (`2**-24` = 6e-8 relative per reported value) with an order of
# magnitude to spare -- NOT a band around a measurement. The generic Colebrook `Duct`
# (`friction="colebrook"`) is a different law and misses by 1e-4..1e-1 (last test).

# ONE literal replacement covers all eight two-loop pipes: the "roughness / minor loss /
# status" run is identical on every [PIPES] row (it occurs exactly 8 times) and
# `str.replace` rewrites every occurrence. Replacing per (length, diameter) pair does NOT
# work -- P2 and P5 are both 400 m by 250 mm, so the first pass would consume both.
_DW_TWO_LOOP = [
    (" Headloss           \tH-W", " Headloss           \tD-W"),
    ("130         \t0           \tOpen", "0.26        \t0           \tOpen"),
]
# Net1's twelve pipes all read roughness 100 (H-W C); 0.5 millifeet is 0.152 mm.
_DW_NET1 = [
    (" Headloss           \tH-W", " Headloss           \tD-W"),
    ("100         \t0           \tOpen", "0.5         \t0           \tOpen"),
]


def _dw_epanet(tmp_path, name, edits=(), duration=None):
    # The edits' roughness is already in D-W units, so wntr's warning that switching
    # HEADLOSS to D-W leaves the roughness units unchanged is expected and benign.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _epanet(tmp_path, name, edits=edits, duration=duration)


def test_darcy_weisbach_two_loop(tmp_path):
    """1e-6 relative on heads and flows, as for Hazen-Williams.

    MEASURED: heads 5.819e-8, flows 6.482e-8 (the float32 floor). With the Colebrook
    `Duct` this was 3.566e-4 / 2.731e-3, and with EPANET's law but the EXACT cfs
    conversion instead of EPANET's rounded ``LPSperCFS = 28.317`` it is 4.292e-7 heads:
    the rounding is part of what EPANET computes, and reproducing it is what reaches the
    floor.
    """
    path, results = _dw_epanet(tmp_path, "twoloop_si.inp", _DW_TWO_LOOP)
    net = read_epanet_inp(path)
    assert net.headloss == "D-W"
    assert net.options.flow_units == "LPS"
    assert net.pipes[0].roughness == pytest.approx(0.26e-3, rel=1e-15)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    worst_h = _worst_heads(model, final, results.node["head"].loc[0], skip=("R1",))
    worst_q = _worst_flows(model, final, results.link["flowrate"].loc[0])
    assert worst_h < 1e-6, worst_h
    assert worst_q < 1e-6, worst_q


def test_darcy_weisbach_net1_single_period(tmp_path):
    """Heads 1e-6, flows 1e-5: the Hazen-Williams Net1 tolerances, for the same reason.

    MEASURED: heads 7.009e-8, flows 3.986e-6. The flow residual is again the smallest
    flow in the network, where EPANET's own reported flows miss continuity (see
    `test_net1_single_period`); Colebrook gave 1.344e-4 / 1.466e-3.
    """
    path, results = _dw_epanet(tmp_path, "Net1.inp", _DW_NET1, duration=0)
    net = read_epanet_inp(path)
    assert net.headloss == "D-W"
    assert net.pipes[0].roughness == pytest.approx(0.5e-3 * 0.3048, rel=1e-15)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    worst_h = _worst_heads(model, final, results.node["head"].loc[0], skip=("9", "2"))
    worst_q = _worst_flows(model, final, results.link["flowrate"].loc[0])
    assert worst_h < 1e-6, worst_h
    assert worst_q < 1e-5, worst_q


def test_darcy_weisbach_laminar_transitional_and_turbulent_pipes(tmp_path):
    """All three branches of EPANET's composite law, against EPANET.

    `lowflow_dw_si.inp` is built so that EPANET's own flows put pipe P6 laminar
    (Re ~ 1000), P2, P3, P4, P5 in Dunlop's transition (2017..3513) and P1 turbulent
    (Re ~ 5731); P5 also carries a minor loss K = 2. Its `Accuracy` is 1e-8, because at
    EPANET's default 1e-3 the REFERENCE stops short of its float32 floor on the loop
    flows (5.0e-7 measured on P4; 9.4e-8 at 1e-8).

    Heads and flows: 1e-6 relative, as above. MEASURED 3.5e-8 / 9.4e-8.
    Per-pipe head LOSS is the sharp check here, because losses are only a few per cent
    of the heads. Its bound is DERIVED from how EPANET reports a head: the hydraulic
    solution is written to its scratch hydraulics file as FLOAT32 FEET (`output.c`,
    `savehyd`), read back for the reporting step and written again as FLOAT32 METRES
    (`nodeoutput`), so each reported head carries at most half a float32 spacing in feet
    (2**-22 ft for heads in [4, 8) ft) plus half a spacing in metres (2**-24 m for heads
    in [1, 2] m), and a loss twice that: 2 (0.3048 * 2**-22 + 2**-24) = 2.65e-7 m.
    MEASURED 9.6e-8 m worst (P4). The Colebrook `Duct` misses these losses by 2 % (P1,
    turbulent) to 81 % (P3, Re just above 2000).
    """
    path, results = _dw_epanet(tmp_path, "lowflow_dw_si.inp")
    net = read_epanet_inp(path)
    assert net.headloss == "D-W"
    flows = results.link["flowrate"].loc[0]
    heads = results.node["head"].loc[0]
    nu = 1.1e-5 * 0.3048**2
    reynolds = {
        p.name: 4.0 * abs(float(flows[p.name])) / (math.pi * p.diameter * nu)
        for p in net.pipes
    }
    assert reynolds["P6"] < 2000.0
    assert all(2000.0 < reynolds[n] < 4000.0 for n in ("P2", "P3", "P4", "P5"))
    assert reynolds["P1"] > 4000.0
    # every reported head is in [1, 2] m (so in [4, 8) ft), where the spacings above hold
    assert all(1.0 <= float(h) <= 2.0 for h in heads)

    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    assert _worst_heads(model, final, heads, skip=("R1",)) < 1e-6
    assert _worst_flows(model, final, flows) < 1e-6
    phi = final["water.phi"] / model.head_scale
    index = {name: i for i, name in enumerate(model.node_names)}
    for pipe in net.pipes:
        mine = float(phi[index[pipe.u]] - phi[index[pipe.v]])
        theirs = float(heads[pipe.u]) - float(heads[pipe.v])
        assert abs(mine - theirs) <= 2.0 * (0.3048 * 2.0**-22 + 2.0**-24), (
            pipe.name, mine - theirs
        )


@pytest.mark.parametrize(
    "name, edits", [("twoloop_si.inp", _DW_TWO_LOOP), ("lowflow_dw_si.inp", [])]
)
def test_the_colebrook_duct_remains_available_and_is_not_epanet(tmp_path, name, edits):
    """`friction="colebrook"` keeps the generic `Duct`. It is a DIFFERENT law (implicit
    Colebrook throughout, a straight laminar line to its Re = 2000 value), so it does not
    reproduce EPANET. MEASURED heads 3.566e-4 (two-loop) and 3.574e-2 (low-flow)."""
    path, results = _dw_epanet(tmp_path, name, edits)
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net, friction="colebrook")
    final = water_steady(model, state, drivers)
    worst_h = _worst_heads(model, final, results.node["head"].loc[0], skip=("R1",))
    assert worst_h > 1e-4, worst_h


# --------------------------------------------------------------------------- PDA
_PDA_EDIT = (
    " Tolerance          \t0.01",
    " Tolerance          \t0.01\n DEMAND MODEL       \tPDA\n"
    " MINIMUM PRESSURE   \t0\n REQUIRED PRESSURE  \t60\n PRESSURE EXPONENT  \t0.5",
)


def test_pressure_driven_demand(tmp_path):
    """1e-5 relative on heads and delivered demands.

    MEASURED against EPANET's own `DEMAND MODEL PDA` at `MINIMUM PRESSURE 0`,
    `REQUIRED PRESSURE 60`, `PRESSURE EXPONENT 0.5`: heads 3.521e-7, demands 2.184e-7. This
    test verifies the core's potential-dependent nodal sources -- EPANET itself formulates PDA
    as "a virtual pipe from the junction to a fictitious reservoir" (Manual section 13.1, p.110),
    i.e. as exactly this potential-dependent nodal source.

    `build_model` is called with NO `pda=`/`p_min=`/`p_req=`/`exponent=`: the
    file's own `[OPTIONS]` edit below is what turns PDA on, exactly as `read_epanet_inp`
    -> `net.options` -> the builder's own defaults are meant to be exercised together,
    rather than the test re-supplying by hand what the file already says.
    """
    path, results = _epanet(tmp_path, "twoloop_si.inp", edits=[_PDA_EDIT])
    net = read_epanet_inp(path)
    assert net.options.demand_model == "PDA"
    assert (net.options.minimum_pressure, net.options.required_pressure,
            net.options.pressure_exponent) == (0.0, 60.0, 0.5)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    heads = results.node["head"].loc[0]
    demands = results.node["demand"].loc[0]
    worst_h = _worst_heads(model, final, heads, skip=("R1",))
    assert worst_h < 1e-5, worst_h
    source = model.kirchhoff["water"]._withdrawals[0]
    delivered = source.flow(final["water.phi"][..., source.nodes]).detach()
    worst_d = max(
        abs(float(delivered[i]) - float(demands[j.name])) / abs(float(demands[j.name]))
        for i, j in enumerate(net.junctions)
    )
    assert worst_d < 1e-5, worst_d


def test_pda_delivers_less_than_the_required_demand(tmp_path):
    """The whole point of PDA: below the required pressure the demand is curtailed."""
    path, _ = _epanet(tmp_path, "twoloop_si.inp", edits=[_PDA_EDIT])
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    source = model.kirchhoff["water"]._withdrawals[0]
    delivered = source.flow(final["water.phi"][..., source.nodes]).detach()
    required = torch.tensor([j.demand for j in net.junctions], dtype=F64)
    assert bool((delivered < required).all())
    assert bool((delivered > 0.8 * required).all())


# The two-loop with a pump between the reservoir and a new demand-free junction J0, which
# P1 now leaves: every junction downstream of the pump is pressure-limited (56-58 m against
# the required 60 m). EPANET is run to ACCURACY 1e-8: at its default 1e-3 its own flows stop
# up to 1.9e-4 relative short of converged on this network (P7, the smallest flow).
_PUMP_EDITS = (
    (" J1              \t0           \t5 ",
     " J0              \t0           \t0           \t                \t;\n"
     " J1              \t0           \t5 "),
    (" P1              \tR1              \tJ1 ", " P1              \tJ0              \tJ1 "),
    (";ID              \tNode1           \tNode2           \tParameters",
     ";ID\tNode1\tNode2\tParameters\n PU1\tR1\tJ0\tHEAD C1"),
    (";ID              \tX-Value     \tY-Value", ";ID\tX-Value\tY-Value\n C1\t45\t8"),
    (" Accuracy           \t0.001",
     " Accuracy           \t1e-8\n HEADERROR \t1e-9\n FLOWCHANGE \t1e-9"),
    (" Trials             \t40", " Trials             \t500"),
)


@pytest.mark.parametrize("pipe_form", ["flow", "potential"])
def test_a_pump_feeding_pressure_driven_demand(tmp_path, pipe_form):
    """1e-5 relative on heads, flows and delivered demands, as for the PDA row above.

    MEASURED (both pipe forms): heads 3.526e-7, flows 2.354e-7, demands 2.355e-7, the
    reference's float32 floor. Before pumps entered in their head-gain form
    (`FlowControlledPump`) this network did not solve at all: the system went singular.
    """
    path, results = _epanet(tmp_path, "twoloop_si.inp", edits=[*_PUMP_EDITS, _PDA_EDIT])
    net = read_epanet_inp(path)
    assert [p.name for p in net.pumps] == ["PU1"] and net.options.demand_model == "PDA"
    model, state, drivers = build_model(net, pipe_form=pipe_form)
    final = water_steady(model, state, drivers)
    heads = results.node["head"].loc[0]
    flows = results.link["flowrate"].loc[0]
    demands = results.node["demand"].loc[0]
    worst_h = _worst_heads(model, final, heads, skip=("R1",))
    worst_q = _worst_flows(model, final, flows)
    source = model.kirchhoff["water"]._withdrawals[0]
    delivered = source.flow(final["water.phi"][..., source.nodes]).detach()
    required = torch.tensor([j.demand for j in net.junctions], dtype=F64)
    worst_d = max(
        abs(float(delivered[i]) - float(demands[j.name])) / abs(float(demands[j.name]))
        for i, j in enumerate(net.junctions)
        if j.demand > 0
    )
    assert worst_h < 1e-5, worst_h
    assert worst_q < 1e-5, worst_q
    assert worst_d < 1e-5, worst_d
    loaded = required > 0
    assert bool((delivered[loaded] < 0.99 * required[loaded]).all())   # really limited


def test_a_zero_pressure_span_is_refused(tmp_path):
    path, _ = _epanet(tmp_path, "twoloop_si.inp")
    net = read_epanet_inp(path)
    with pytest.raises(ValueError, match="must exceed MINIMUM PRESSURE"):
        build_model(net, pda=True, p_min=0.0, p_req=0.0)


# --------------------------------------------------------------------------- loop head loss
def test_head_loss_sums_to_zero_around_every_loop(tmp_path):
    """1e-12. MEASURED: exactly 0.0 on both independent loops.

    This is an IDENTITY of the formulation, not a convergence result: the pipe kind carries
    no head, so `dp = A^T phi` and every cycle-basis row annihilates it by construction.
    This test therefore pins that `Network.cycle_basis` and the layer's own `difference`
    sign convention agree -- which is a real thing to get wrong -- rather than that the
    Newton solve converged.
    """
    path, _ = _epanet(tmp_path, "twoloop_si.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    layer = model.kirchhoff["water"]
    basis = model.net.cycle_basis("pipe")
    assert basis.shape[0] == 2
    loops = basis.to(F64) @ layer.dp(final["water.phi"], {})
    assert float(loops.abs().max()) < 1e-12


# --------------------------------------------------------------------------- gradients
def test_gradients_against_central_differences(tmp_path):
    """`max|analytic - fd| <= 1e-6 * max|fd|`, for BOTH the
    nodal demand `water.sources` and the Hazen-Williams roughness (both channels are
    differenced for real below, so `solve_sum`'s `roughness` branch is exercised). The pump `h0` and
    tank AREA channels are checked separately, on Net1
    (`test_gradient_reaches_the_pump_head_gain_h0` and
    `test_gradient_reaches_the_tank_area_through_advance` below), since
    `twoloop_si.inp` has neither a pump nor a tank.

    MEASURED: sources 3.212e-6 against an allowance of 4.444e-4; roughness 2.505e-10
    against an allowance of 6.259e-8. Both are SCALED criteria rather than entrywise
    relative ones because the smallest-flow pipe's own gradient is a tiny fraction of the
    largest, where the central difference is the noisy party.
    """
    path, _ = _epanet(tmp_path, "twoloop_si.inp")
    net = read_epanet_inp(path)

    def solve_sum(sources=None, roughness=None):
        local = net
        if roughness is not None:
            local = type(net)(
                **{
                    **net.__dict__,
                    "pipes": tuple(
                        type(p)(p.name, p.u, p.v, p.length, p.diameter, float(r),
                                p.minor_loss, p.status)
                        for p, r in zip(net.pipes, roughness, strict=True)
                    ),
                }
            )
        model, state, drivers = build_model(local)
        if sources is not None:
            drivers = dict(drivers)
            drivers["water.sources"] = sources
        final = water_steady(model, state, drivers)
        return model, final

    model, _, drivers = build_model(net)
    element = model.kirchhoff["water"]._laws[0].pipe   # the resistance form wraps it
    sources = drivers["water.sources"].clone().requires_grad_(True)
    element.roughness.requires_grad_(True)
    final = water_steady(model, {"water.phi": torch.zeros(model.net.n, dtype=F64),
                                 "water.q": torch.zeros(
                                     len(model.kirchhoff["water"].cols), dtype=F64)},
                         {**drivers, "water.sources": sources},
                         differentiable=True)
    layer = model.kirchhoff["water"]
    final["water.phi"][..., layer.interior].sum().backward()
    analytic_sources = sources.grad.clone()
    analytic_roughness = element.roughness.grad.clone()

    eps = 1e-8
    fd_sources = []
    for i in range(model.net.n):
        up = drivers["water.sources"].clone()
        up[i] += eps
        down = drivers["water.sources"].clone()
        down[i] -= eps
        m_up, s_up = solve_sum(sources=up)
        m_down, s_down = solve_sum(sources=down)
        fd_sources.append(
            (
                float(s_up["water.phi"][..., m_up.kirchhoff["water"].interior].sum())
                - float(s_down["water.phi"][..., m_down.kirchhoff["water"].interior].sum())
            )
            / (2 * eps)
        )
    scale = max(abs(v) for v in fd_sources)
    worst = max(
        abs(float(a) - b)
        for a, b in zip(analytic_sources.tolist(), fd_sources, strict=True)
    )
    assert worst <= 1e-6 * scale, (worst, 1e-6 * scale)

    eps_r = 1e-4
    base_roughness = [p.roughness for p in net.pipes]
    fd_roughness = []
    for i in range(len(net.pipes)):
        up_r = list(base_roughness)
        up_r[i] += eps_r
        down_r = list(base_roughness)
        down_r[i] -= eps_r
        m_up, s_up = solve_sum(roughness=up_r)
        m_down, s_down = solve_sum(roughness=down_r)
        fd_roughness.append(
            (
                float(s_up["water.phi"][..., m_up.kirchhoff["water"].interior].sum())
                - float(s_down["water.phi"][..., m_down.kirchhoff["water"].interior].sum())
            )
            / (2 * eps_r)
        )
    scale_r = max(abs(v) for v in fd_roughness)
    worst_r = max(
        abs(float(a) - b)
        for a, b in zip(analytic_roughness.tolist(), fd_roughness, strict=True)
    )
    assert worst_r <= 1e-6 * scale_r, (worst_r, 1e-6 * scale_r)


def test_gradient_reaches_the_pump_head_gain_h0(tmp_path):
    """The pump `h0` channel, on Net1 (`twoloop_si.inp` has no pump).

    MEASURED: 9.2e-10 relative (analytic 0.4306365, central difference 0.4306365).
    """
    path, _ = _epanet(tmp_path, "Net1.inp", duration=0)
    net = read_epanet_inp(path)

    def solve_with_h0(delta: float):
        model, state, drivers = build_model(net)
        pump = next(el for el in model.kirchhoff["water"]._laws if el.kind == "pump")
        with torch.no_grad():
            pump.h0.add_(delta)
        final = water_steady(model, state, drivers)
        return model, final

    model, state, drivers = build_model(net)
    pump = next(el for el in model.kirchhoff["water"]._laws if el.kind == "pump")
    pump.h0.requires_grad_(True)
    final = water_steady(model, state, drivers, differentiable=True)
    layer = model.kirchhoff["water"]
    final["water.phi"][..., layer.interior].sum().backward()
    analytic = float(pump.h0.grad.detach())

    eps = 1e-4
    m_up, up = solve_with_h0(eps)
    m_down, down = solve_with_h0(-eps)
    fd = (
        float(up["water.phi"][..., m_up.kirchhoff["water"].interior].sum())
        - float(down["water.phi"][..., m_down.kirchhoff["water"].interior].sum())
    ) / (2 * eps)
    assert abs(analytic - fd) <= 1e-5 * abs(fd), (analytic, fd)


def test_tank_area_is_not_reached_by_a_single_steady_solve(tmp_path):
    """The solve holds every tank at its head, so a single `water_steady` call's output is
    STRUCTURALLY independent of the tank area (the storage capacity) -- not a small
    gradient, but no path in the graph at all. This is the "structurally not
    differentiable" case: a blanket claim that roughness, demands, pump h0 AND
    tank area all reach the steady solve would overstate the last one. The next test shows
    where the framework DOES differentiate w.r.t. area: the step of `water_event_step`.
    """
    path, _ = _epanet(tmp_path, "Net1.inp", duration=0)
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    (storage,) = model.kirchhoff["water"].storage
    storage.C.requires_grad_(True)
    final = water_steady(model, state, drivers, differentiable=True)
    assert not final["water.phi"].requires_grad


def test_gradient_reaches_the_tank_area_through_the_event_step(tmp_path):
    """The tank-area channel is the forward-Euler move of `water_event_step` (the previous
    test), not the steady solve. Net1 is Hazen-Williams, so the storage capacity IS the
    area. MEASURED at the old explicit update: 2.753e-9 relative (analytic -5.0256097e-3,
    central difference -5.0256097e-3).
    """
    path, _ = _epanet(tmp_path, "Net1.inp", duration=0)
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    (storage,) = model.kirchhoff["water"].storage
    assert model.head_scale == 1.0

    def level_sum():
        stepped, dt = water_event_step(model, state, drivers, 3600.0)
        assert dt == 3600.0
        return tank_levels(model, stepped).sum()

    storage.C.requires_grad_(True)
    level_sum().backward()
    analytic = float(storage.C.grad.detach().sum())

    eps = 1e-4
    base_area = storage.C.detach().clone()
    with torch.no_grad():
        storage.C.copy_(base_area + eps)
        up = float(level_sum())
        storage.C.copy_(base_area - eps)
        down = float(level_sum())
        storage.C.copy_(base_area)
    fd = (up - down) / (2 * eps)
    assert abs(analytic - fd) <= 1e-5 * abs(fd), (analytic, fd)


def test_gradient_reaches_a_learnable_roughness(tmp_path):
    path, _ = _epanet(tmp_path, "twoloop_si.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    element = model.kirchhoff["water"]._laws[0].pipe   # the resistance form wraps it
    element.roughness.requires_grad_(True)
    final = water_steady(model, state, drivers, differentiable=True)
    layer = model.kirchhoff["water"]
    final["water.phi"][..., layer.interior].sum().backward()
    assert element.roughness.grad is not None
    assert torch.isfinite(element.roughness.grad).all()
    assert float(element.roughness.grad.abs().max()) > 0.0


# --------------------------------------------------------------------------- trace quality
def test_trace_quality_on_the_two_loop(tmp_path):
    """1e-3 on the steady trace fraction. MEASURED: 6.438e-12 percentage points
    (noise many orders below the 1e-3 tolerance, floating on the reference's own float32 output and
    this solve's Newton tolerance rather than on any physics this test could regress).

    A SMOKE test: the fixture has a single source, so mass balance
    alone forces 100 % everywhere once transients clear. A genuinely discriminating
    two-source trace (Net3's own "percent of Lake water" scenario) is not implemented.

    What it does pin is the FORMULATION: the layer is well posed only because the nodal
    demand enters as a first-order removal rate. Without it the
    generator's row sums are the demands and the steady system is singular.
    """
    path, results = _epanet(tmp_path, "twoloop_trace.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net, quality=0.0)
    drivers = dict(drivers)
    drivers["quality.x_boundary"] = torch.tensor([100.0], dtype=F64)
    final = water_steady(model, state, drivers)
    trace = results.node["quality"].iloc[-1]
    worst = max(
        abs(float(final["quality.x"][i]) - float(trace[j.name]))
        for i, j in enumerate(net.junctions)
    )
    assert worst < 1e-3, worst


def test_the_withdrawal_sink_is_the_layers_dilution_not_a_build_time_removal(tmp_path):
    """The measurement behind the demand-as-removal formulation (`build_model`'s docstring),
    pinned so a future edit cannot undo it."""
    path, _ = _epanet(tmp_path, "twoloop_trace.inp")
    net = read_epanet_inp(path)
    model, _, _ = build_model(net, quality=0.0)
    layer = model.transport["quality"]
    assert layer.removal is not None
    # the withdrawal is the layer's dilution = 1 (net inflow / capacity from the current
    # flows); `removal` is only the bulk decay (zero here)
    assert layer.dilution == 1.0
    assert layer.dilution_mode == "withdrawal"
    assert float(layer.removal.abs().max()) == 0.0


def test_a_junction_capacity_is_half_of_every_incident_pipe(tmp_path):
    path, _ = _epanet(tmp_path, "twoloop_trace.inp")
    net = read_epanet_inp(path)
    model, _, _ = build_model(net, quality=0.0)
    volume = {
        p.name: torch.pi * p.diameter**2 / 4.0 * p.length for p in net.pipes
    }
    # J1 touches P1, P2 and P4
    expected = 0.5 * (volume["P1"] + volume["P2"] + volume["P4"])
    assert float(model.transport["quality"].capacity[0]) == pytest.approx(
        expected, rel=1e-14
    )


# --------------------------------------------------------------------------- golden
def test_the_water_golden_case_is_reproduced(tmp_path):
    """1e-10, against `tests/golden/water_twoloop.json`."""
    from tests.golden import load_golden

    golden = load_golden("water_twoloop")
    model, state, drivers = build_model(twoloop())
    final = water_steady(model, state, drivers)
    for key, expected in golden.items():
        assert key in final, key
        assert final[key].flatten().tolist() == pytest.approx(
            expected, rel=1e-10, abs=1e-18
        )


# --------------------------------------------------------------------------- PBV and GPV
# The two-loop network with pipe P2 ending at a new demand-free junction J7, and a valve
# V1 from J7 to J2 closing the loop again.
_J7 = (" J6              \t0           \t9           \t                \t;\n",
       " J6              \t0           \t9           \t                \t;\n"
       " J7              \t0           \t0           \t                \t;\n")
_P2_TO_J7 = (" P2              \tJ1              \tJ2  ",
             " P2              \tJ1              \tJ7  ")
_VALVES = ("[VALVES]\n;ID              \tNode1           \tNode2           \tDiameter    "
           "\tType\tSetting     \tMinorLoss\n")
_CURVES = "[CURVES]\n;ID              \tX-Value     \tY-Value\n"


def _valve_edits(valve, curve=""):
    return (_J7, _P2_TO_J7, (_VALVES, _VALVES + valve), (_CURVES, _CURVES + curve))


_PBV_ACTIVE = _valve_edits(" V1  J7  J2  250  PBV  5  0\n")
# A 0.05 m setting below the open valve's minor loss at this flow: EPANET treats it as an
# open valve (pbvcoeff's `Km q^2 > setting` branch).
_PBV_OPEN = _valve_edits(" V1  J7  J2  80  PBV  0.05  10\n")
_GPV = _valve_edits(" V1  J7  J2  250  GPV  HL1  0\n",
                    " HL1 0 0\n HL1 10 1\n HL1 30 6\n HL1 60 20\n")


@pytest.mark.parametrize("edits", [_PBV_ACTIVE, _PBV_OPEN, _GPV],
                         ids=["pbv-active", "pbv-open", "gpv"])
def test_pressure_breaker_and_general_purpose_valves(tmp_path, edits):
    """Heads 1e-6 relative; flows 1e-6 m3/s absolute; the valve's head loss 1e-5 relative.

    Flows are compared absolutely because EPANET's own are not continuous through an
    active PBV (its `pbvcoeff` holds the head drop by a 1e8 penalty, `CBIG`): P2 and V1 are
    in series through the demand-free J7, yet EPANET reports them 1.6e-7 m3/s apart, and
    its head loss across the active valve is 4.999997 m for a 5 m setting (MEASURED); this
    model's flows agree to round-off and its head drop is the setting. MEASURED: PBV active
    heads 3.2e-7, flows 2.6e-7 m3/s, loss 5.7e-7; PBV open loss 6.8e-6 (an open valve's
    `m q^2` doubles the flows' relative difference); GPV heads 5.5e-7, loss 3.9e-6.
    """
    path, results = _epanet(tmp_path, "twoloop_si.inp", edits=edits)
    model, state, drivers = build_model(read_epanet_inp(path))
    assert model.kirchhoff["water"].implicit_edges.numel() > 0
    final = water_steady(model, state, drivers)
    heads = results.node["head"].loc[0]
    flows = results.link["flowrate"].loc[0]
    assert _worst_heads(model, final, heads, skip=("R1",)) < 1e-6
    worst_q = max(abs(float(final["water.q"][i]) - float(flows[name]))
                  for i, name in enumerate(model.link_names))
    assert worst_q < 1e-6, worst_q
    names = model.node_names
    phi = final["water.phi"]
    loss = float(phi[names.index("J7")] - phi[names.index("J2")])
    reference = float(results.link["headloss"].loc[0]["V1"])
    assert abs(loss - reference) <= 1e-5 * abs(reference), (loss, reference)
    i, k = model.link_names.index("P2"), model.link_names.index("V1")
    assert abs(float(final["water.q"][i] - final["water.q"][k])) < 1e-15   # continuity at J7


def test_gradient_through_a_valve_network_against_central_differences(tmp_path):
    """d(sum of junction heads)/d(demands) on the GPV network: the MNA layer's implicit
    adjoint against central differences, `max|analytic - fd| <= 1e-6 max|fd|`."""
    path, _ = _epanet(tmp_path, "twoloop_si.inp", edits=_GPV)
    model, state, drivers = build_model(read_epanet_inp(path))
    junctions = [model.node_names.index(f"J{k}") for k in range(1, 8)]

    def total(sources):
        final = water_steady(model, state, {**drivers, "water.sources": sources})
        return final["water.phi"][junctions].sum()

    sources = drivers["water.sources"].clone().requires_grad_(True)
    analytic, = torch.autograd.grad(total(sources), sources)
    h = 1e-6
    fd = torch.zeros_like(sources)
    for i in junctions:
        e = torch.zeros_like(sources)
        e[i] = h
        with torch.no_grad():
            fd[i] = (total(sources + e) - total(sources - e)) / (2 * h)
    scale = float(fd.abs().max())
    assert float((analytic - fd)[junctions].abs().max()) <= 1e-6 * scale


# --------------------------------------------------------------------------- link status
def _pipe_status_edit(pipe, status):
    """Replace the `Open` status column of `pipe`'s `[PIPES]` line in the two-loop fixture."""
    line = next(row for row in (DATA / "twoloop_si.inp").read_text().splitlines()
                if row.split()[:1] == [pipe])
    return (line, line.replace("Open", status))


# P4 (J3 -> J1) carries its water from J1 to J3, against its own orientation, so a check
# valve there closes; P5 (J3 -> J4) carries it forward, so one there stays open; P7 is
# closed by `[STATUS]`. What is left is a tree fed from R1.
_STATUS_EDITS = (
    _pipe_status_edit("P4", "CV"),
    _pipe_status_edit("P5", "CV"),
    ("[STATUS]\n;ID              \tStatus/Setting",
     "[STATUS]\n;ID              \tStatus/Setting\n P7              \tClosed"),
)


def test_check_valve_and_closed_pipe_match_epanet(tmp_path):
    """Heads 1e-6 relative, flows 1e-6 m3/s absolute, and every link's state (closed or
    open) as EPANET reports it.

    EPANET does not hold a closed link at exactly zero flow: it gives it the conductance
    `1 / CBIG` (`hydcoeffs.c`, `pipecoeff`), so its closed P4 and P7 carry a leak of order
    `dh / CBIG`, far below the flow bound (it reports them as 0); this model's closed links
    carry exactly zero. MEASURED: heads 9.58e-7 relative (J6, 4.4e-5 m), flows 2.2e-9 m3/s.
    The head residual is not the check valves (closing P4 by `[STATUS]` instead gives the same
    9.58e-7, node for node) but EPANET's rounded LPS factor (`LPSperCFS = 28.317` against
    28.3168466: EPANET's internal flows are 5.4e-6 relative smaller, and a Hazen-Williams
    loss 1.852 times that): scaling this model's demands by 28.3168466 / 28.317 brings the
    heads to 6.6e-8 (MEASURED; the plain two-loop goes from 4.4e-7 to 5.3e-8).
    """
    path, results = _epanet(tmp_path, "twoloop_si.inp", edits=_STATUS_EDITS)
    model, state, drivers = build_model(read_epanet_inp(path))
    final = water_steady(model, state, drivers)
    heads = results.node["head"].loc[0]
    flows = results.link["flowrate"].loc[0]
    worst_h = _worst_heads(model, final, heads, skip=("R1",))
    assert worst_h < 1e-6, worst_h
    worst_q = max(abs(float(final["water.q"][i]) - float(flows[name]))
                  for i, name in enumerate(model.link_names))
    assert worst_q < 1e-6, worst_q
    states = link_states(model, final, drivers).tolist()
    reference = results.link["status"].loc[0]
    assert {name: int(s) for name, s in zip(model.link_names, states, strict=True)} == {
        name: int(reference[name]) for name in model.link_names}
    assert states[model.link_names.index("P4")] == 0
    assert states[model.link_names.index("P7")] == 0
    q = dict(zip(model.link_names, final["water.q"].tolist(), strict=True))
    assert q["P4"] == 0.0 and q["P7"] == 0.0 and q["P5"] > 0.0


def _junction_demand_edit(junction, demand):
    """Replace `junction`'s demand on its `[JUNCTIONS]` line of the two-loop fixture."""
    line = next(row for row in (DATA / "twoloop_si.inp").read_text().splitlines()
                if row.split()[:1] == [junction])
    fields = line.split("\t")
    fields[2] = f"{demand:<12g}"
    return (line, "\t".join(fields))


def _compare_with_epanet(model, final, drivers, results, *, heads=1e-6, flows=1e-6):
    """Heads `heads` relative (every junction), flows `flows` m3/s absolute, link states
    equal; returns the measured worst head and flow."""
    worst_h = _worst_heads(model, final, results.node["head"].loc[0], skip=("R1",))
    reference = results.link["flowrate"].loc[0]
    worst_q = max(abs(float(final["water.q"][i]) - float(reference[name]))
                  for i, name in enumerate(model.link_names))
    assert worst_h < heads, worst_h
    assert worst_q < flows, worst_q
    states = link_states(model, final, drivers).tolist()
    status = results.link["status"].loc[0]
    assert dict(zip(model.link_names, states, strict=True)) == {
        name: int(status[name]) for name in model.link_names}
    return worst_h, worst_q


# J6 with no demand, both its pipes closed: EPANET still gives it a head, through the
# `1 / CBIG` conductance of the closed links.
_J6_CUT_OFF = (
    _junction_demand_edit("J6", 0),
    ("[STATUS]\n;ID              \tStatus/Setting",
     "[STATUS]\n;ID              \tStatus/Setting\n P7              \tClosed\n"
     " P8              \tClosed"),
)


def test_a_junction_cut_off_without_demand_matches_epanet(tmp_path):
    """J6 is held at EPANET's limit head, the average of J5 and J3 (its closed links' far
    ends), with every flow into it exactly zero. MEASURED: heads 2.9e-7 relative (J6 too),
    flows 2.2e-9 m3/s."""
    path, results = _epanet(tmp_path, "twoloop_si.inp", edits=_J6_CUT_OFF)
    model, state, drivers = build_model(read_epanet_inp(path))
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results)
    phi = dict(zip(model.node_names, final["water.phi"].tolist(), strict=True))
    assert phi["J6"] == pytest.approx(0.5 * (phi["J5"] + phi["J3"]), abs=1e-12)
    q = dict(zip(model.link_names, final["water.q"].tolist(), strict=True))
    assert q["P7"] == 0.0 and q["P8"] == 0.0


_ALL_CV_FIVE_FOLD = (
    *(_pipe_status_edit(f"P{k}", "CV") for k in range(1, 9)),
    *(_junction_demand_edit(name, 5 * demand) for name, demand in
      (("J1", 5), ("J2", 8), ("J3", 6), ("J4", 10), ("J5", 7), ("J6", 9))),
)


def test_every_pipe_a_check_valve_at_five_times_the_demand_matches_epanet(tmp_path):
    """An iterate of this network closes both check valves that feed J6 (P7 and P8), which
    cuts J6 off; the pipe takes P7's open branch again for that step (`FlowControlledPipe`,
    A CUT-OFF ITERATE) and the solve converges to EPANET's tree, P4 and P8 closed.

    Heads are compared at 4e-5 relative, not 1e-6. MEASURED 3.1e-5, and the cause is
    EPANET's rounded LPS factor (`LPSperCFS = 28.317` against 28.3168466: its internal flows
    are 5.4e-6 smaller, its Hazen-Williams losses 1.0e-5 smaller), which on this network's
    110 m of losses moves J6 (-59 m) by 1.2e-3 m. With this model's demands scaled by that
    factor the heads agree to 1.4e-6, asserted below at 2e-6. Flows: 5.7e-8 m3/s."""
    path, results = _epanet(tmp_path, "twoloop_si.inp", edits=_ALL_CV_FIVE_FOLD)
    model, state, drivers = build_model(read_epanet_inp(path))
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results, heads=4e-5)
    rounded = {**drivers, "water.sources": drivers["water.sources"] * (28.3168466 / 28.317)}
    final = water_steady(model, state, rounded)
    # (its flows are EPANET's internal ones, 5.4e-6 below the reported: 1.2e-6 m3/s here)
    _compare_with_epanet(model, final, rounded, results, heads=2e-6, flows=2e-6)


_TCV = _valve_edits(" V1  J7  J2  250  TCV  5  0\n")


def test_a_throttle_control_valve_matches_epanet(tmp_path):
    """Heads, flows and the valve's head loss. The TCV's coefficient is EPANET's rounded
    `0.02517 K / d^4` (ft, cfs): `8 / (32.2 pi^2) = 0.025173` rounded to four figures,
    9.2e-4 below the exact `8 / (g pi^2) = 0.0251933` a `MinorLoss` would use.

    The head loss is compared at 1e-5 m ABSOLUTE: EPANET reports a valve's loss as the
    difference of its two FLOAT32 heads (spacing 3.8e-6 m at 48 m), which on this 0.0516 m
    loss is 7e-5 relative. MEASURED 3.2e-6 m; this model's loss is `Km q^2` at EPANET's own
    flow to 3e-7 relative."""
    path, results = _epanet(tmp_path, "twoloop_si.inp", edits=_TCV)
    model, state, drivers = build_model(read_epanet_inp(path))
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results)
    phi = dict(zip(model.node_names, final["water.phi"].tolist(), strict=True))
    loss, reference = phi["J7"] - phi["J2"], float(results.link["headloss"].loc[0]["V1"])
    assert abs(loss - reference) <= 1e-5, (loss, reference)


def test_gradient_through_a_closed_check_valve_against_central_differences(tmp_path):
    """d(sum of junction heads)/d(demands) on the check-valve network (P4 closed by its
    check valve, P7 by `[STATUS]`): the implicit adjoint on the converged branch against
    central differences, `max|analytic - fd| <= 1e-6 max|fd|`."""
    path, _ = _epanet(tmp_path, "twoloop_si.inp", edits=_STATUS_EDITS)
    model, state, drivers = build_model(read_epanet_inp(path))
    junctions = [model.node_names.index(f"J{k}") for k in range(1, 7)]

    def total(sources):
        final = water_steady(model, state, {**drivers, "water.sources": sources})
        return final["water.phi"][junctions].sum()

    sources = drivers["water.sources"].clone().requires_grad_(True)
    analytic, = torch.autograd.grad(total(sources), sources)
    h = 1e-6
    fd = torch.zeros_like(sources)
    for i in junctions:
        e = torch.zeros_like(sources)
        e[i] = h
        with torch.no_grad():
            fd[i] = (total(sources + e) - total(sources - e)) / (2 * h)
    scale = float(fd.abs().max())
    assert float((analytic - fd)[junctions].abs().max()) <= 1e-6 * scale


# --------------------------------------------------------------------------- fenced by CVs
def _hand_built(tmp_path, junctions, reservoirs, pipes):
    """`(model, final, drivers, results)` for a hand-built network: this model's steady
    state and EPANET's, the latter from the same network written through wntr."""
    from noodl.apps.water.network import WaterNetwork

    net = WaterNetwork(junctions=junctions, reservoirs=reservoirs, pipes=pipes)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    w = wntr.network.WaterNetworkModel()
    for j in junctions:
        w.add_junction(j.name, base_demand=j.demand, elevation=j.elevation)
    for r in reservoirs:
        w.add_reservoir(r.name, base_head=r.head)
    for p in pipes:
        w.add_pipe(p.name, p.u, p.v, length=p.length, diameter=p.diameter,
                   roughness=p.roughness, check_valve=p.status == "CV",
                   initial_status="CLOSED" if p.status == "CLOSED" else "OPEN")
    w.options.time.duration = 0
    results = wntr.sim.EpanetSimulator(w).run_sim(file_prefix=str(tmp_path / "epanet"))
    return model, final, drivers, results


def _fenced(*middle):
    """R1 (100 m) and R2 (50 m) joined through K (4 L/s), and a zero-demand group (`middle`,
    a chain of junctions) fenced off by two check valves that both point out of it towards
    the reservoirs' far sides: P1 from the group to R1, P2 from R2 into the group."""
    from noodl.apps.water.network import Junction, Reservoir, WaterPipe

    def pipe(name, u, v, status="OPEN"):
        return WaterPipe(name, u, v, 500.0, 0.3, 130.0, status=status)

    junctions = (*(Junction(n, 0.0, 0.0) for n in middle), Junction("K", 0.0, 0.004))
    pairs = zip(middle, middle[1:], strict=False)
    chain = tuple(pipe(f"P{5 + i}", a, b) for i, (a, b) in enumerate(pairs))
    pipes = (pipe("P1", middle[0], "R1", "CV"), *chain, pipe("P2", "R2", middle[-1], "CV"),
             pipe("P3", "R1", "K"), pipe("P4", "K", "R2"))
    return junctions, (Reservoir("R1", 100.0), Reservoir("R2", 50.0)), pipes


def test_a_junction_fenced_off_by_check_valves_matches_epanet(tmp_path):
    """J's two check valves both hold no flow (J above R2, below R1). Any head between 50 and
    100 m keeps both shut; EPANET's `1 / CBIG` through both gives the average, 75 m, and so
    does this model (`_limit`). MEASURED: J exactly 75 against EPANET's 75.000000, heads
    3e-9 relative, both valves CLOSED as EPANET."""
    model, final, drivers, results = _hand_built(tmp_path, *_fenced("J"))
    _compare_with_epanet(model, final, drivers, results)
    assert float(final["water.phi"][model.node_names.index("J")]) == 75.0


def test_a_group_fenced_off_by_check_valves_is_held_at_the_limit_head(tmp_path):
    """J - P5 - L, fenced off by the two check valves. This model holds both at 75 m, the
    `1 / CBIG` limit. EPANET reports 75.021263, which is not a limit: MEASURED with P5
    5 m, 500 m and 5000 m long it gives 73.2097, 75.0213 and 74.9940 (not monotonic),
    unchanged with ACCURACY 1e-8. Its 2.3e-8 m3/s in P5 is the `1 / CBIG` flow through the
    closed check valves (25 m = 82.0 ft over 1e8). The mechanism is a HYPOTHESIS: EPANET may
    stop on an unconverged iterate for a group whose flows are all below its relative-flow
    test. So J and L are compared at 3e-4 relative (MEASURED 2.8e-4); every
    other head at 1e-6, every flow at 1e-6 m3/s, every state exactly. Documented in
    docs/applications/water.md, Limitations."""
    model, final, drivers, results = _hand_built(tmp_path, *_fenced("J", "L"))
    heads = results.node["head"].loc[0]
    assert _worst_heads(model, final, heads, skip=("J", "L")) < 1e-6
    for name in ("J", "L"):
        mine = float(final["water.phi"][model.node_names.index(name)])
        assert mine == pytest.approx(75.0, abs=1e-12)
        assert abs(mine - float(heads[name])) < 3e-4 * float(heads[name])
    _compare_with_epanet(model, final, drivers, results, heads=3e-4)


def _fence_cases():
    """The reviewer's adversarial networks: R1 (100 m) and R2 (50 m) joined through K
    (4 L/s), and zero-demand junctions behind check valves only."""
    from noodl.apps.water.network import Junction, Reservoir, WaterPipe

    def pipe(name, u, v, status="OPEN"):
        return WaterPipe(name, u, v, 500.0, 0.3, 130.0, status=status)

    k = Junction("K", 0.0, 0.004)
    feed = (pipe("P3", "R1", "K"), pipe("P4", "K", "R2"))

    def reservoirs(third=None):
        base = (Reservoir("R1", 100.0), Reservoir("R2", 50.0))
        return base if third is None else (*base, Reservoir("R3", third))

    j, j1, j2 = Junction("J", 0.0), Junction("J1", 0.0), Junction("J2", 0.0)
    return {
        # both valves point into J: P1 opens (J = 100, R1's head), P2 stays shut
        "both-into-J": ((j, k), reservoirs(), (pipe("P1", "R1", "J", "CV"),
                                               pipe("P2", "R2", "J", "CV"), *feed)),
        # three valves shut, J at the average 70
        "three-shut": ((j, k), reservoirs(60.0), (
            pipe("P1", "J", "R1", "CV"), pipe("P2", "R2", "J", "CV"),
            pipe("P6", "R3", "J", "CV"), *feed)),
        # the average 80 would open P6 from R3 (90 m): it opens and J = 90
        "one-opens": ((j, k), reservoirs(90.0), (
            pipe("P1", "J", "R1", "CV"), pipe("P2", "R2", "J", "CV"),
            pipe("P6", "R3", "J", "CV"), *feed)),
        # J1 at 64 above J2 at 42 opens the internal valve P7: both at 170 / 3
        "internal-opens": ((j1, j2, k), reservoirs(20.0), (
            pipe("P1", "J1", "R1", "CV"), pipe("P2", "R2", "J1", "CV"),
            pipe("P7", "J1", "J2", "CV"), pipe("P8", "R3", "J2", "CV"), *feed)),
        # the same valve the other way stays shut: J1 = 64, J2 = 42
        "internal-shut": ((j1, j2, k), reservoirs(20.0), (
            pipe("P1", "J1", "R1", "CV"), pipe("P2", "R2", "J1", "CV"),
            pipe("P7", "J2", "J1", "CV"), pipe("P8", "R3", "J2", "CV"), *feed)),
        # ten reservoirs at 0 behind closed links pull J2 down, so with every valve shut Rb
        # (60 m) is the first to feed it; once Ra (90 m) feeds J1 and P7 ties J2 to J1, Pb
        # must shut again: J1 = J2 = 90, Pa and P7 open, Pb closed
        "reshut": ((j1, j2, k), (*reservoirs(), Reservoir("Ra", 90.0), Reservoir("Rb", 60.0),
                                 Reservoir("Rc", 65.0),
                                 *(Reservoir(f"Z{i}", 0.0) for i in range(10))), (
            pipe("Pa", "Ra", "J1", "CV"), pipe("P7", "J1", "J2", "CV"),
            pipe("Pb", "Rb", "J2", "CV"), pipe("Pc", "J1", "Rc", "CLOSED"),
            *(pipe(f"PZ{i}", "J2", f"Z{i}", "CLOSED") for i in range(10)), *feed)),
        # two parallel valves from Ra (90 m) into J share one constraint row: both open,
        # J = 90 (EPANET: 1.43e-8 m3/s through each), B and C shut
        "parallel": ((j, k), reservoirs(90.0), (
            pipe("A1", "R3", "J", "CV"), pipe("A2", "R3", "J", "CV"),
            pipe("B", "J", "R1", "CV"), pipe("C", "R2", "J", "CV"), *feed)),
    }


@pytest.mark.parametrize("case", sorted(_fence_cases()))
def test_check_valves_fencing_a_junction_match_epanet(tmp_path, case):
    """EPANET's `1 / CBIG` limit with check valves around zero-demand junctions
    (`_limit`, the resistor-and-diode program): the shut ones are equal conductances, an
    open one at zero flow ties its ends. Heads 1e-6 relative, flows 1e-6 m3/s, every state.
    MEASURED heads 2.5e-9; "one-opens", "reshut" and "parallel" 8.5e-8 (90 against EPANET's
    float32 90.0000076); flows 1.1e-7 (K's pipes), and 8.9e-7 in "reshut", EPANET's own leak
    through its OPEN Pa and P7 at `1 / RQtol` (this model's zero-flow valves carry exactly
    0), just under the bound.

    "internal-opens" is compared at 5e-5 (MEASURED 4.1e-5: J1 and J2 at 56.666667, the
    limit, against EPANET's 56.668983), and the cause is EPANET's, not this model's: its
    head there is not a limit at all. MEASURED with P7 5 m, 500 m and 5000 m long, EPANET
    gives 55.3140, 56.6690 and 56.6655 m, unchanged with `ACCURACY 1e-8`; the same family
    as the two-junction group of the test above (docs/applications/water.md,
    Limitations)."""
    model, final, drivers, results = _hand_built(tmp_path, *_fence_cases()[case])
    _compare_with_epanet(model, final, drivers, results,
                         heads=5e-5 if case == "internal-opens" else 1e-6)


@pytest.mark.parametrize("case, reservoir, expected", [
    ("three-shut", "R3", 1 / 3),      # J the average of three shut valves' far ends
    ("one-opens", "R3", 1.0),         # J tied to R3 through the open P6
    ("one-opens", "R1", 0.0),         # ... and no longer a function of R1
    ("internal-opens", "R3", 1 / 3),  # J1 = J2 = (R1 + R2 + R3) / 3 through the open P7
    ("reshut", "Rb", 0.0),            # Pb shut again: J2 does not see Rb
    ("reshut", "Ra", 1.0),            # J2 tied to J1 tied to Ra
])
def test_gradient_of_a_held_head_with_respect_to_a_reservoir_head(case, reservoir, expected):
    """A held head is a linear map of the heads around it for the active set found
    (`_limit`): d(held head)/d(reservoir head) by autograd through `water_steady` against
    central differences (1e-3 m, within the active set) and the analytic slope, at 1e-9."""
    from noodl.apps.water.network import WaterNetwork

    junctions, reservoirs, pipes = _fence_cases()[case]
    net = WaterNetwork(junctions=junctions, reservoirs=reservoirs, pipes=pipes)
    model, state, drivers = build_model(net)
    held = model.node_names.index(junctions[0].name)   # J, or J2 (J1 in reshut: either)
    if case == "reshut":
        held = model.node_names.index("J2")
    r = [x.name for x in reservoirs].index(reservoir)

    def head(boundary):
        return water_steady(model, state, {**drivers, "water.phi_boundary": boundary})[
            "water.phi"][held]

    boundary = drivers["water.phi_boundary"].clone().requires_grad_(True)
    (analytic,) = torch.autograd.grad(head(boundary), boundary)
    e = torch.zeros_like(boundary)
    e[r] = 1e-3
    with torch.no_grad():
        fd = (head(boundary + e) - head(boundary - e)) / 2e-3
    assert float(analytic[r]) == pytest.approx(expected, abs=1e-9)
    assert float(fd) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- PRV and PSV
#: EPANET's `HTOL` (`input1.c`, 0.0005 ft): the margin of its PRV/PSV status rules.
HTOL = 0.0005 * 0.3048


def _series_inp(junctions, reservoirs, pipes, valves, status=""):
    """A small LPS / H-W network as `.inp` text: every pipe 1000 m, 200 mm, C = 100, every
    valve 200 mm. These are the series networks of the design note's section 4
    (`docs/plans/water-prv-psv-design.md`), whose measured numbers they reproduce: 0.206 m
    across the open PRV with `Km = 10`, the PRV's switch at an upstream reservoir of
    53.8214 m."""
    def rows(items, fmt):
        return "".join(fmt.format(*item) for item in items)
    return ("[JUNCTIONS]\n" + rows(junctions, " {} 0 {}\n")
            + "[RESERVOIRS]\n" + rows(reservoirs, " {} {}\n")
            + "[PIPES]\n" + rows(pipes, " {} {} {} 1000 200 100 0 Open\n")
            + "[VALVES]\n" + rows(valves, " {} {} {} 200 {} {} {}\n")
            + (f"[STATUS]\n{status}" if status else "")
            + "[TIMES]\n Duration 0:00\n[OPTIONS]\n Units LPS\n Headloss H-W\n[END]\n")


def _prv_line(r1, setting, minor=0, demand=20):
    """R1 -> P1 -> J1 -> PRV V1 -> J2 -> P2 -> J3 (`demand` L/s)."""
    return _series_inp((("J1", 0), ("J2", 0), ("J3", demand)), (("R1", r1),),
                       (("P1", "R1", "J1"), ("P2", "J2", "J3")),
                       (("V1", "J1", "J2", "PRV", setting, minor),))


def _psv_line(r1, setting, r2):
    """R1 -> P1 -> J1 -> PSV V1 -> J2 -> P2 -> R2."""
    return _series_inp((("J1", 0), ("J2", 0)), (("R1", r1), ("R2", r2)),
                       (("P1", "R1", "J1"), ("P2", "J2", "R2")),
                       (("V1", "J1", "J2", "PSV", setting, 0),))


def _twoloop_inp(valve):
    """The two-loop fixture with J7 and the valve line `valve` (`_valve_edits`), and EPANET's
    `ACCURACY` at 1e-8 instead of the fixture's 0.001. At 0.001 EPANET stops with an ACTIVE
    PRV's flow 5.8e-6 m3/s short of its own node balance (its `CBIG` penalty on J2 absorbs
    the rest): heads 5.8e-6 relative from this model's, J7 by 2.8e-4 m; at 1e-8 they agree
    to 6e-8 (MEASURED, case "prv-active")."""
    text = (DATA / "twoloop_si.inp").read_text()
    for old, new in (*_valve_edits(valve),
                     (" Accuracy           \t0.001", " Accuracy           \t1e-8")):
        assert old in text, old
        text = text.replace(old, new)
    return text


#: The upstream reservoir heads of the near-switch sweep, all at least 2 HTOL from the
#: switch at 53.8214 m: OPEN below it, ACTIVE above.
_NEAR_SWITCH = (53.80, 53.81, 53.82, 53.822, 53.83, 53.85)

#: The cases of the design note's section 4: `(networks, {valve: EPANET state})`, the
#: states as EPANET reports them (MEASURED, wntr 1.5.0).
_PRV_PSV_CASES = {
    "prv-active": ([_twoloop_inp(" V1  J7  J2  250  PRV  48  0\n")], {"V1": 2}),
    "prv-open": ([_twoloop_inp(" V1  J7  J2  250  PRV  49.9  0\n")], {"V1": 1}),
    "prv-open-minor-loss": ([_prv_line(40, 50, minor=10)], {"V1": 1}),
    "prv-closed-reverse": ([_twoloop_inp(" V1  J2  J7  250  PRV  49.9  0\n")], {"V1": 0}),
    "prv-closed-downstream-above": ([_twoloop_inp(" V1  J7  J2  250  PRV  45  0\n")],
                                    {"V1": 0}),
    "psv-active": ([_psv_line(100, 70, 20)], {"V1": 2}),
    "psv-open": ([_psv_line(100, 30, 20)], {"V1": 1}),
    "psv-closed-upstream-below": ([_psv_line(60, 70, 20)], {"V1": 0}),
    "psv-closed-reverse": ([_psv_line(60, 30, 80)], {"V1": 0}),
    "prv-psv-series": ([_series_inp(
        (("J1", 0), ("J2", 0), ("J3", 10), ("J4", 0)), (("R1", 100), ("R2", 10)),
        (("P1", "R1", "J1"), ("P2", "J2", "J3"), ("P3", "J4", "R2")),
        (("V1", "J1", "J2", "PRV", 60, 0), ("V2", "J3", "J4", "PSV", 55, 0)))],
        {"V1": 2, "V2": 2}),
    "near-switch": ([_prv_line(r1, 50) for r1 in _NEAR_SWITCH], None),
}


def _run_inp(tmp_path, text, tag="case"):
    """`(path, results)`: `text` written to `tmp_path` and run by EPANET."""
    path = tmp_path / f"{tag}.inp"
    path.write_text(text)
    model = wntr.network.WaterNetworkModel(str(path))
    results = wntr.sim.EpanetSimulator(model).run_sim(file_prefix=str(tmp_path / tag))
    return path, results


def _controlled(model, name):
    """`(node, Hset)` of the valve `name`: its controlled node (a PRV's end, a PSV's start)
    and the head it holds there (the node's elevation plus the setting), in metres."""
    net = model.water_network
    valve = next(v for v in net.valves if v.name == name)
    node = valve.v if valve.kind.upper() == "PRV" else valve.u
    elevation = next(j.elevation for j in net.junctions if j.name == node)
    return node, elevation + valve.setting


def _as_epanet(drivers):
    """`drivers` with the demands EPANET computes with: its LPS factor is rounded
    (`LPSperCFS = 28.317` against 28.3168466), so its internal flows are 5.4e-6 relative
    below the file's (`test_every_pipe_a_check_valve_at_five_times_the_demand_matches_epanet`)."""
    return {**drivers, "water.sources": drivers["water.sources"] * (28.3168466 / 28.317)}


@pytest.mark.parametrize("case", list(_PRV_PSV_CASES))
def test_prv_psv_match_epanet(tmp_path, case):
    """The design note's eleven cases (`docs/plans/water-prv-psv-design.md`, section 4):
    heads 1e-6 relative, flows 1e-6 m3/s absolute, every link's state as EPANET reports it,
    and an ACTIVE valve's controlled head equal to its setting to Newton's tolerance and
    within 1e-6 m of EPANET's. The near-switch case sweeps the PRV's upstream reservoir
    across its switch (53.8214 m) at points at least `2 HTOL` from it: OPEN below, ACTIVE
    above, as EPANET.

    The heads are compared with the demands EPANET computes with (`_as_epanet`). With the
    file's demands they differ by up to 2.4e-6 relative (MEASURED, J3 of
    "prv-open-minor-loss", 7.9e-5 m): the C = 100 pipes lose 3.8 m per km, and EPANET's
    rounded LPS factor makes those losses 1.0e-5 smaller. With EPANET's demands the worst
    is 9.2e-8, flows 3.9e-7 m3/s, every state equal, every controlled head exactly its
    setting and EPANET's (the table in docs/applications/water.md). The file's demands are
    held to 1e-5."""
    texts, expected = _PRV_PSV_CASES[case]
    for k, text in enumerate(texts):
        path, results = _run_inp(tmp_path, text, f"{case}{k}")
        model, state, drivers = build_model(read_epanet_inp(path))
        final = water_steady(model, state, drivers)
        _compare_with_epanet(model, final, drivers, results, heads=1e-5)
        drivers = _as_epanet(drivers)
        final = water_steady(model, state, drivers)
        _compare_with_epanet(model, final, drivers, results)
        states = dict(zip(model.link_names, link_states(model, final, drivers).tolist(),
                          strict=True))
        if expected is not None:
            assert {name: states[name] for name in expected} == expected
        heads = dict(zip(model.node_names, final["water.phi"].tolist(), strict=True))
        for name in (n for n in model.link_names if n.startswith("V")):
            if states[name] == 2:
                node, hset = _controlled(model, name)
                assert abs(heads[node] - hset) < 1e-9, (name, heads[node], hset)
                assert abs(heads[node] - float(results.node["head"].loc[0][node])) < 1e-6


def test_prv_inside_htol_band(tmp_path):
    """Inside EPANET's `HTOL` band below the switch EPANET holds the PRV ACTIVE with the
    downstream head at the setting while the upstream head is below it (a negative head drop
    through an active valve, MEASURED 8.8e-5 m at 53.8213); this model gives the ideal
    valve, OPEN with no drop. The heads differ there by less than `HTOL`, and every flow
    agrees to 1e-6 m3/s; with EPANET's own demands (`_as_epanet`), which on their own move
    J3 by 7.9e-5 m."""
    for k, r1 in enumerate((53.8213, 53.82135, 53.8214 - 0.5 * HTOL)):
        path, results = _run_inp(tmp_path, _prv_line(r1, 50), f"band{k}")
        model, state, drivers = build_model(read_epanet_inp(path))
        drivers = _as_epanet(drivers)
        final = water_steady(model, state, drivers)
        heads = results.node["head"].loc[0]
        worst = max(abs(float(final["water.phi"][i]) - float(heads[name]))
                    for i, name in enumerate(model.node_names))
        assert worst < HTOL, (r1, worst)
        flows = results.link["flowrate"].loc[0]
        assert max(abs(float(final["water.q"][i]) - float(flows[name]))
                   for i, name in enumerate(model.link_names)) < 1e-6
        v1 = model.link_names.index("V1")
        assert int(results.link["status"].loc[0]["V1"]) == 2
        assert int(link_states(model, final, drivers)[v1]) == 1


def _setting_gradient(model, state, drivers, link, node):
    """`(analytic, fd)`: d(head at `node`)/d(setting of `link`), by the implicit adjoint and
    by central differences of 1e-4 m."""
    i, k = model.node_names.index(node), model.link_names.index(link)

    def head(setting):
        return water_steady(model, state, {**drivers, "water.setting": setting})[
            "water.phi"][i]

    setting = drivers["water.setting"].clone().requires_grad_(True)
    (analytic,) = torch.autograd.grad(head(setting), setting)
    e = torch.zeros_like(setting)
    e[k] = 1e-4
    with torch.no_grad():
        fd = (head(setting + e) - head(setting - e)) / 2e-4
    return float(analytic[k]), float(fd)


def _demand_and_roughness_gradients(model, state, drivers):
    """`[(analytic, fd)]` of d(sum of junction heads) by the demands (`water.sources`, steps
    of 1e-6 m3/s) and by the pipe roughnesses (steps of 1e-4)."""
    layer = model.kirchhoff["water"]
    junctions = [model.node_names.index(j.name) for j in model.water_network.junctions]
    pipe = layer._laws[0].pipe   # the resistance form wraps it

    def total(sources):
        final = water_steady(model, state, {**drivers, "water.sources": sources})
        return final["water.phi"][junctions].sum()

    sources = drivers["water.sources"].clone().requires_grad_(True)
    pipe.roughness.requires_grad_(True)
    try:
        a_sources, a_rough = torch.autograd.grad(total(sources), (sources, pipe.roughness))
    finally:
        pipe.roughness.requires_grad_(False)
    fd_sources = torch.zeros(len(junctions), dtype=F64)
    for n, i in enumerate(junctions):
        e = torch.zeros_like(sources)
        e[i] = 1e-6
        with torch.no_grad():
            fd_sources[n] = (total(sources + e) - total(sources - e)) / 2e-6
    fd_rough = torch.zeros_like(a_rough)
    base = drivers["water.sources"]
    for n in range(len(fd_rough)):
        with torch.no_grad():
            pipe.roughness[n] += 1e-4
            up = total(base)
            pipe.roughness[n] -= 2e-4
            down = total(base)
            pipe.roughness[n] += 1e-4
        fd_rough[n] = (up - down) / 2e-4
    return [(a_sources[junctions], fd_sources), (a_rough, fd_rough)]


def _assert_scaled(analytic, fd):
    """`max|analytic - fd| <= 1e-6 max|fd|`, as the other gradient checks here."""
    scale = float(fd.abs().max())
    assert float((analytic - fd).abs().max()) <= 1e-6 * max(scale, 1e-12), (analytic, fd)


def test_prv_gradients_against_central_differences(tmp_path):
    """The implicit adjoint through an ACTIVE PRV, against central differences: d(sum of
    junction heads)/d(demands, roughness) on the two-loop PRV-active case, and
    d(head)/d(setting). In the tree (`_prv_line`) the downstream head is the setting:
    dH_J2/dsetting = 1 exactly, and the upstream head does not move with it. An OPEN PRV
    (two-loop, setting 49.9) is a minor loss whatever its setting: d/dsetting = 0."""
    path, _ = _run_inp(tmp_path, _PRV_PSV_CASES["prv-active"][0][0], "active")
    model, state, drivers = build_model(read_epanet_inp(path))
    for analytic, fd in _demand_and_roughness_gradients(model, state, drivers):
        _assert_scaled(analytic, fd)
    analytic, fd = _setting_gradient(model, state, drivers, "V1", "J5")
    assert analytic == pytest.approx(fd, rel=1e-6)

    path, _ = _run_inp(tmp_path, _prv_line(60, 50), "tree")
    model, state, drivers = build_model(read_epanet_inp(path))
    analytic, fd = _setting_gradient(model, state, drivers, "V1", "J2")
    assert analytic == 1.0 and fd == pytest.approx(1.0, abs=1e-8)
    analytic, fd = _setting_gradient(model, state, drivers, "V1", "J1")
    assert analytic == 0.0 and fd == pytest.approx(0.0, abs=1e-8)

    path, _ = _run_inp(tmp_path, _PRV_PSV_CASES["prv-open"][0][0], "open")
    model, state, drivers = build_model(read_epanet_inp(path))
    for node in ("J2", "J5"):
        analytic, fd = _setting_gradient(model, state, drivers, "V1", node)
        assert analytic == 0.0 and fd == pytest.approx(0.0, abs=1e-8)


def test_psv_gradients_against_central_differences(tmp_path):
    """The same through an ACTIVE PSV (`psv-active`): d(sum of junction heads)/d(demands,
    roughness), and d(head)/d(setting): the upstream head is the setting
    (dH_J1/dsetting = 1)."""
    path, _ = _run_inp(tmp_path, _PRV_PSV_CASES["psv-active"][0][0], "psv")
    model, state, drivers = build_model(read_epanet_inp(path))
    for analytic, fd in _demand_and_roughness_gradients(model, state, drivers):
        _assert_scaled(analytic, fd)
    analytic, fd = _setting_gradient(model, state, drivers, "V1", "J1")
    assert analytic == pytest.approx(1.0, abs=1e-12) and fd == pytest.approx(1.0, abs=1e-8)
    analytic, fd = _setting_gradient(model, state, drivers, "V1", "J2")
    assert analytic == pytest.approx(fd, rel=1e-6, abs=1e-9)


def test_prv_one_sided_gradients_jump(tmp_path):
    """dH_J2/dsetting of the series PRV either side of its switch (R1 at 55 m: the switch is
    the setting equal to the upstream head J1): ACTIVE below it (1), OPEN above it (0). At
    `±delta` the adjoint equals the central difference on its own side, so it jumps from 1
    to 0 across the switch; a difference straddling the switch gives neither."""
    path, _ = _run_inp(tmp_path, _prv_line(55, 50), "jump")
    model, state, drivers = build_model(read_epanet_inp(path))
    k = model.link_names.index("V1")
    upstream = float(water_steady(model, state, drivers)["water.phi"][
        model.node_names.index("J1")])   # the upstream head, the same in both states
    sides = {}
    for side, delta in (("below", -1e-2), ("above", 1e-2)):
        setting = drivers["water.setting"].clone()
        setting[k] = upstream + delta
        sides[side] = _setting_gradient(model, state, {**drivers, "water.setting": setting},
                                        "V1", "J2")
    assert sides["below"][0] == 1.0 and sides["below"][1] == pytest.approx(1.0, abs=1e-8)
    assert sides["above"][0] == 0.0 and sides["above"][1] == pytest.approx(0.0, abs=1e-8)
    setting = drivers["water.setting"].clone()
    setting[k] = upstream
    i = model.node_names.index("J2")
    e = torch.zeros_like(setting)
    e[k] = 1e-3
    with torch.no_grad():
        up = water_steady(model, state, {**drivers, "water.setting": setting + e})
        down = water_steady(model, state, {**drivers, "water.setting": setting - e})
    straddle = float(up["water.phi"][i] - down["water.phi"][i]) / 2e-3
    assert 0.3 < straddle < 0.7


def test_parallel_prvs_with_equal_settings_are_refused(tmp_path):
    """Two PRVs in parallel with one setting both hold J2 at 48 m and leave the split of the
    flow between them undetermined; EPANET refuses the file (error 220, "illegal valve
    connection to another valve"). The error names both valves."""
    path = tmp_path / "parallel.inp"
    path.write_text(_twoloop_inp(" V1  J7  J2  250  PRV  48  0\n V2  J7  J2  250  PRV  48  0\n"))
    with pytest.raises(ValueError, match=r"'V1'.*'V2'|'V2'.*'V1'"):
        build_model(read_epanet_inp(path))


def test_smoothed_prv_converges_to_exact(tmp_path):
    """`build_model(valve_smoothing=1e-3)` (metres): the heads within 2e-3 m of the exact
    valve's, on the two-loop ACTIVE and OPEN cases, across the near-switch sweep, and on
    the PSV line ACTIVE, OPEN and CLOSED."""
    texts = [_PRV_PSV_CASES["prv-active"][0][0], _PRV_PSV_CASES["prv-open"][0][0],
             *(_prv_line(r1, 50) for r1 in (53.80, 53.8214, 53.85)),
             _psv_line(100, 70, 20), _psv_line(100, 30, 20), _psv_line(60, 70, 20)]
    for k, text in enumerate(texts):
        path = tmp_path / f"smooth{k}.inp"
        path.write_text(text)
        net = read_epanet_inp(path)
        model, state, drivers = build_model(net)
        exact = water_steady(model, state, drivers)["water.phi"]
        model, state, drivers = build_model(net, valve_smoothing=1e-3)
        smooth = water_steady(model, state, drivers)["water.phi"]
        assert float((smooth - exact).abs().max()) < 2e-3


def test_prv_directly_downstream_of_a_reservoir(tmp_path):
    """A PRV whose start node is the reservoir: the law reads the head of a boundary node,
    which has no column of its own. EPANET refuses a PRV touching a reservoir (error 219),
    so its reference has a 1 mm pipe of 1 m diameter between them (a head loss of order
    1e-13 m at 20 L/s): heads 1e-6 relative, flows 1e-6 m3/s, the state ACTIVE; and the
    Jacobian the layer assembles equals autograd of its residual, at a point that is not
    the solution."""
    text = _series_inp((("J2", 0), ("J3", 20)), (("R1", 60),), (("P2", "J2", "J3"),),
                       (("V1", "R1", "J2", "PRV", 50, 0),))
    path = tmp_path / "direct.inp"
    path.write_text(text)
    model, state, drivers = build_model(read_epanet_inp(path))
    final = water_steady(model, state, drivers)
    reference = (text.replace("[JUNCTIONS]\n", "[JUNCTIONS]\n J1 0 0\n")
                 .replace(" V1 R1 J2", " V1 J1 J2")
                 .replace("[PIPES]\n", "[PIPES]\n P0 R1 J1 0.001 1000 100 0 Open\n"))
    _, results = _run_inp(tmp_path, reference, "reference")
    heads, flows = results.node["head"].loc[0], results.link["flowrate"].loc[0]
    for i, name in enumerate(model.node_names):
        assert abs(float(final["water.phi"][i]) - float(heads[name])) < 1e-6 * float(
            heads[name])
    for i, name in enumerate(model.link_names):
        assert abs(float(final["water.q"][i]) - float(flows[name])) < 1e-6
    assert link_states(model, final, drivers).tolist()[model.link_names.index("V1")] == 2
    assert int(results.link["status"].loc[0]["V1"]) == 2
    assert float(final["water.phi"][model.node_names.index("J2")]) == pytest.approx(
        50.0, abs=1e-9)

    layer = model.kirchhoff["water"]
    phi_b = drivers["water.phi_boundary"]
    # J2, J3, then P2's and V1's flows: a point that is not the solution
    x = torch.tensor([47.0, 41.5, 0.013, 0.017], dtype=F64)
    assert x.numel() == layer.interior.numel() + layer.implicit_edges.numel()

    def residual(x):
        return layer._residual(x, phi_b, drivers, drivers["water.sources"])[0]

    assert float(residual(x).abs().max()) > 1e-3
    expected = torch.autograd.functional.jacobian(residual, x)
    got = layer._operator(x, phi_b, drivers).assemble()
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-12)


def test_a_closed_prv_cutting_off_a_junction_is_held_at_the_limit_head(tmp_path):
    """A PRV CLOSED by `[STATUS]` leaves J2 and J3 (no demand) with no head of their own:
    they are held at the `1 / CBIG` limit, J1's 60 m (`_limit`), with no flow. EPANET
    reports 53.687092 for both, and the same with a closed TCV in the PRV's place: its head
    of a group fed only through closed links is not a limit (docs/applications/water.md,
    Limitations). Every other head, every flow and every state match."""
    text = _series_inp((("J1", 0), ("J2", 0), ("J3", 0)), (("R1", 60),),
                       (("P1", "R1", "J1"), ("P2", "J2", "J3")),
                       (("V1", "J1", "J2", "PRV", 50, 0),), status=" V1 Closed\n")
    path, results = _run_inp(tmp_path, text, "closed")
    model, state, drivers = build_model(read_epanet_inp(path))
    final = water_steady(model, state, drivers)
    heads = dict(zip(model.node_names, final["water.phi"].tolist(), strict=True))
    assert heads["J2"] == pytest.approx(60.0, abs=1e-9)
    assert heads["J3"] == pytest.approx(60.0, abs=1e-9)
    assert float(results.node["head"].loc[0]["J2"]) == pytest.approx(53.687092, abs=1e-5)
    assert _worst_heads(model, final, results.node["head"].loc[0], skip=("J2", "J3")) < 1e-6
    _compare_with_epanet(model, final, drivers, results, heads=0.2)


#: `(valve, R1, which end is the dead end, EPANET's state, EPANET's dead-end head or None)`:
#: the dead end is J2 - P2 - J3 with no demand, on the valve's end (`"end"`: J1 -> J2) or its
#: start (`"start"`: J2 -> J1). `None`: EPANET's head there is its own, compared exactly.
_DEAD_ENDS = [
    ("PRV", 60, "end", 2, None),          # ACTIVE at the setting
    ("PRV", 40, "end", 1, None),          # OPEN at the upstream head
    ("PSV", 60, "end", 1, None),          # OPEN at the upstream head
    ("PSV", 40, "end", 0, 35.791393),     # CLOSED: the upstream head below the setting
    ("PRV", 40, "start", 0, 39.996788),   # CLOSED: no water flows out of a dead end
    ("PRV", 60, "start", 0, 60.000637),
    ("PSV", 40, "start", 0, 39.996788),
    ("PSV", 60, "start", 0, 53.687092),
]


@pytest.mark.parametrize("valve, r1, side, expected, epanet_head", _DEAD_ENDS)
def test_a_valve_feeding_a_dead_end_without_demand_matches_epanet(
        tmp_path, valve, r1, side, expected, epanet_head):
    """A PRV or PSV with a dead end on one side (J2 - P2 - J3, no demand, joined to the rest
    only through the valve) carries no flow, and its state is EPANET's: on its end, a PRV
    ACTIVE at its setting (upstream above it) or OPEN at the upstream head (below it), a PSV
    OPEN, or CLOSED where the upstream head is below its setting; on its start, CLOSED
    (`_PressureValve`, A DEAD END). Every flow matches to 1e-6 m3/s.

    Where the valve is CLOSED the dead end has no head of its own: this model holds it at
    the `1 / CBIG` limit, the valve's other end (`dp = 0`, as `_limit` does for any closed
    link). EPANET's head there is not that limit (MEASURED, `epanet_head`: 3.2e-3 m and
    6.4e-4 m off it; 53.687 m and 35.791 m, the same values as a closed TCV or PRV by
    `[STATUS]` in its place, `test_a_closed_prv_cutting_off_a_junction_is_held_at_the_limit_
    head`): compared as recorded, and every other head at 1e-6 relative."""
    u, v = ("J1", "J2") if side == "end" else ("J2", "J1")
    text = _series_inp((("J1", 0), ("J2", 0), ("J3", 0)), (("R1", r1),),
                       (("P1", "R1", "J1"), ("P2", "J2", "J3")),
                       (("V1", u, v, valve, 50, 0),))
    path, results = _run_inp(tmp_path, text, "dead")
    model, state, drivers = build_model(read_epanet_inp(path))
    final = water_steady(model, state, drivers)
    states = link_states(model, final, drivers).tolist()
    assert states[model.link_names.index("V1")] == expected
    if epanet_head is None:
        _compare_with_epanet(model, final, drivers, results)
        return
    heads = dict(zip(model.node_names, final["water.phi"].tolist(), strict=True))
    reference = results.node["head"].loc[0]
    for name in ("J2", "J3"):
        assert heads[name] == pytest.approx(heads["J1"], abs=1e-12)   # the limit
        assert float(reference[name]) == pytest.approx(epanet_head, abs=1e-5)
    assert _worst_heads(model, final, reference, skip=("J2", "J3")) < 1e-6
    _compare_with_epanet(model, final, drivers, results, heads=0.2)


def test_a_valve_between_two_dead_ends_is_refused(tmp_path):
    """A valve whose both sides lie in a part without demand that a CLOSED link cuts off
    from every reservoir: its state and those heads are undetermined. Refused by name."""
    text = _series_inp((("J1", 5), ("J2", 0), ("J3", 0), ("J4", 0)), (("R1", 60),),
                       (("P1", "R1", "J1"), ("P2", "J1", "J2"), ("P3", "J3", "J4")),
                       (("V1", "J2", "J3", "PRV", 50, 0),), status=" P2 Closed\n")
    path = tmp_path / "both.inp"
    path.write_text(text)
    model, state, drivers = build_model(read_epanet_inp(path))
    with pytest.raises(ValueError, match=r"valve 'V1' joins two parts"):
        water_steady(model, state, drivers)


def test_prv_and_psv_on_a_darcy_weisbach_pressure_layer(tmp_path):
    """The PRV-and-PSV series case with `HEADLOSS D-W` (roughness 0.1 mm): the layer works in
    pressure (`rho g` times head), and so do the valves' elevation and setting. Both valves
    ACTIVE as EPANET, each controlled head its setting, and the heads within 1e-6 relative
    of EPANET's with its demands (MEASURED 9.0e-7)."""
    text = _PRV_PSV_CASES["prv-psv-series"][0][0].replace("H-W", "D-W").replace(
        " 1000 200 100 0 Open", " 1000 200 0.1 0 Open")
    path, results = _run_inp(tmp_path, text, "dw")
    model, state, drivers = build_model(read_epanet_inp(path))
    assert model.head_scale != 1.0
    drivers = _as_epanet(drivers)
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results)
    heads = dict(zip(model.node_names, (final["water.phi"] / model.head_scale).tolist(),
                     strict=True))
    assert heads["J2"] == pytest.approx(60.0, abs=1e-9)
    assert heads["J3"] == pytest.approx(55.0, abs=1e-9)


def test_a_batch_of_prv_settings_takes_each_instance_to_its_own_state(tmp_path):
    """One solve over three settings of the two-loop PRV (48, 49.9 and 45 m): ACTIVE, OPEN
    and CLOSED in the one batch, each instance equal to its own single solve."""
    path, _ = _run_inp(tmp_path, _PRV_PSV_CASES["prv-active"][0][0], "batch")
    model, state, drivers = build_model(read_epanet_inp(path))
    k = model.link_names.index("V1")
    settings = drivers["water.setting"].expand(3, -1).clone()
    settings[:, k] = torch.tensor([48.0, 49.9, 45.0], dtype=F64)
    batch = {key: value.expand(3, *value.shape).clone() for key, value in state.items()}
    final = water_steady(model, batch, {**drivers, "water.setting": settings})
    states = link_states(model, final, {**drivers, "water.setting": settings})
    assert states[:, k].tolist() == [2, 1, 0]
    for b in range(3):
        single = water_steady(model, state, {**drivers, "water.setting": settings[b]})
        torch.testing.assert_close(final["water.phi"][b], single["water.phi"], rtol=0,
                                   atol=1e-9)


# --------------------------------------------------------------------------- pumps
#: US units -> the two-loop's LPS and metres, so a reference network's pump keeps its own
#: curve: the gallon is exact (3.785411784 L), the foot 0.3048 m.
_LPS_PER_GPM = 3.785411784 / 60.0
_M_PER_FT = 0.3048


def _curve_lines(name, points_gpm_ft):
    """`[CURVES]` lines of `name` for points given in GPM and feet, in LPS and metres."""
    return "".join(f" {name}\t{q * _LPS_PER_GPM!r}\t{h * _M_PER_FT!r}\n"
                   for q, h in points_gpm_ft)


#: Net3's pump 10 (`wntr/library/networks/Net3.inp`, curve 1): three points with the first
#: at zero flow, whose fit has the exponent 1.77, not 2.
_NET3_CURVE = _curve_lines("C1", ((0, 104), (2000, 92), (4000, 63)))
#: Anytown's pump curve (`wntr/tests/networks_for_testing/Anytown.inp`, curve 2): five
#: points, joined by straight segments.
_ANYTOWN_CURVE = _curve_lines("C1", ((0, 300), (2000, 292), (4000, 270), (6000, 230),
                                     (8000, 181)))
#: Net6's constant-power pump (`wntr/library/networks/Net6.inp`, PUMP-3889, POWER 15 hp),
#: in the kilowatts of an SI file.
_NET6_POWER = f"POWER {15 * 0.745699872!r}"
#: The single-point curve of `_PUMP_EDITS`: 45 L/s at 8 m, shut-off 10.667 m.
_ONE_POINT = " C1\t45\t8\n"


def _pump_edits(parameters, curve="", extra=()):
    """The two-loop fixture with R1 feeding J1 through the pump `PU1` (R1 -> J0) and pipe P1
    (J0 -> J1), the pump on `[PUMPS]` `parameters` with `[CURVES]` lines `curve`, at EPANET
    `ACCURACY` 1e-8 (as `_PUMP_EDITS`)."""
    return (
        (" J1              \t0           \t5 ",
         " J0              \t0           \t0           \t                \t;\n"
         " J1              \t0           \t5 "),
        (" P1              \tR1              \tJ1 ", " P1              \tJ0              \tJ1 "),
        (";ID              \tNode1           \tNode2           \tParameters",
         ";ID\tNode1\tNode2\tParameters\n PU1\tR1\tJ0\t" + parameters),
        (";ID              \tX-Value     \tY-Value", ";ID\tX-Value\tY-Value\n" + curve),
        (" Accuracy           \t0.001",
         " Accuracy           \t1e-8\n HEADERROR \t1e-9\n FLOWCHANGE \t1e-9"),
        (" Trials             \t40", " Trials             \t500"),
        *extra,
    )


def _demand_multiplier(factor):
    return (" Demand Multiplier  \t1.0", f" Demand Multiplier  \t{factor}")


_P8_LINE = (" P8              \tJ6              \tJ3              \t500         \t200         "
            "\t130         \t0           \tOpen  \t;")

#: A second reservoir R2 at 70 m on J3 (pipe P9, 200 m of 300 mm): above the pump's reach of
#: 50 + 10.667 m, so the pump is shut beyond its shut-off head.
_R2_ABOVE_SHUT_OFF = (
    (" R1              \t50          \t                \t;",
     " R1              \t50          \t                \t;\n R2\t70\t\t;"),
    (_P8_LINE, _P8_LINE + "\n P9\tR2\tJ3\t200\t300\t130\t0\tOpen\t;"),
)


def _pump_case(tmp_path, edits):
    """`(model, state, drivers, results)` of the two-loop pump network `edits`."""
    path, results = _epanet(tmp_path, "twoloop_si.inp", edits=edits)
    model, state, drivers = build_model(read_epanet_inp(path))
    return model, state, drivers, results


def test_three_point_pump_with_general_exponent(tmp_path):
    """Net3's pump curve (three points, the first at zero flow): EPANET fits
    `h = A - B q^C` through them (`input2.c`, `updatepumpparams`; `input3.c`, `powercurve`),
    here with `C = ln(41/12) / ln 2 = 1.77`. Heads 1e-6 relative, flows 1e-6 m3/s, states
    equal. MEASURED: heads 2.4e-7, flows 1.8e-9 m3/s."""
    model, state, drivers, results = _pump_case(tmp_path, _pump_edits("HEAD C1", _NET3_CURVE))
    pump = model.kirchhoff["water"].law_for("pump")[0]
    assert float(pump.curves[0].n) == pytest.approx(math.log(41 / 12) / math.log(2),
                                                    rel=1e-12)
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results)


@pytest.mark.parametrize("multiplier", [1, 4])
def test_multipoint_pump_curve(tmp_path, multiplier):
    """Anytown's five-point pump curve, joined by straight segments (`hydcoeffs.c`,
    `curvecoeff`): at the fixture's demands the pump works on the first segment, at four
    times them (180 L/s) on the second. Heads 1e-6 relative, flows 1e-6 m3/s, states
    equal.

    At four times the demands the heads are compared at 3e-6 relative. MEASURED 2.2e-6, and
    the cause is EPANET's rounded LPS factor (`LPSperCFS = 28.317` against 28.3168466; see
    `test_every_pipe_a_check_valve_at_five_times_the_demand_matches_epanet`) on this
    network's 25 m of losses; with this model's demands scaled by that factor the heads
    agree to 5.2e-7, asserted at 1e-6 (flows then 9.8e-7 m3/s: EPANET's internal flows,
    below its reported ones, asserted at 2e-6). MEASURED at the fixture's demands: heads
    1.3e-7, flows 1.8e-9 m3/s."""
    model, state, drivers, results = _pump_case(tmp_path, _pump_edits(
        "HEAD C1", _ANYTOWN_CURVE, (_demand_multiplier(multiplier),)))
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results,
                         heads=1e-6 if multiplier == 1 else 3e-6)
    q = float(final["water.q"][model.link_names.index("PU1")])
    assert q == pytest.approx(0.045 * multiplier, rel=1e-5)
    if multiplier != 1:
        rounded = _as_epanet(drivers)
        final = water_steady(model, state, rounded)
        _compare_with_epanet(model, final, rounded, results, heads=1e-6, flows=2e-6)


def test_constant_power_pump(tmp_path):
    """Net6's constant-power pump, `h = 8.814 P / q` in feet, horsepower and cfs
    (`hydcoeffs.c`, `pumpcoeff`; `input2.c`, `updatepumpparams`). Heads 1e-6 relative,
    flows 1e-6 m3/s, states equal; no `q_max` refusal (EPANET has no flow limit for it).
    MEASURED: heads 2.8e-7, flows 1.8e-9 m3/s."""
    model, state, drivers, results = _pump_case(tmp_path, _pump_edits(_NET6_POWER))
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results)


def test_pump_closes_beyond_shut_off(tmp_path):
    """R2 holds J0 above the pump's shut-off head (50 + 10.667 m): the pump is shut, with
    exactly zero flow (EPANET: its closed-link leak, reported as 0) and state CLOSED, as
    EPANET reports it. Heads 1e-6 relative, flows 1e-6 m3/s. MEASURED: heads 1.1e-7, flows
    1.8e-8 m3/s (EPANET's leak through the shut pump)."""
    model, state, drivers, results = _pump_case(
        tmp_path, _pump_edits("HEAD C1", _ONE_POINT, _R2_ABOVE_SHUT_OFF))
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results)
    k = model.link_names.index("PU1")
    assert float(final["water.q"][k]) == 0.0
    assert int(link_states(model, final, drivers)[k]) == 0


#: Every kind of pump curve, as `[PUMPS]` parameters and `[CURVES]` lines.
_PUMP_KINDS = {
    "one-point": ("HEAD C1", _ONE_POINT),
    "three-point": ("HEAD C1", _NET3_CURVE),
    "multipoint": ("HEAD C1", _ANYTOWN_CURVE),
    "power": (_NET6_POWER, ""),
}


def _at_unit_speed(path):
    """The network of `path` with every pump at speed 1."""
    net = read_epanet_inp(path)
    return type(net)(**{**net.__dict__, "pumps": tuple(
        type(p)(**{**p.__dict__, "speed": 1.0}) for p in net.pumps)})


@pytest.mark.parametrize("kind", list(_PUMP_KINDS))
def test_pump_speed_setting(tmp_path, kind):
    """A relative speed of 0.8 given through the driver `"water.setting"` on a model built
    at speed 1 matches EPANET with `SPEED 0.8` in the file (`hydcoeffs.c`, `pumpcoeff`: the
    affinity laws). Heads 1e-6 relative, flows 1e-6 m3/s, states equal. MEASURED heads
    (flows 1.8e-9 m3/s each): one point 3.8e-7, three points 2.9e-7, segments 2.2e-7,
    constant power 3.6e-7."""
    parameters, curve = _PUMP_KINDS[kind]
    path, results = _epanet(tmp_path, "twoloop_si.inp",
                            edits=_pump_edits(parameters + " SPEED 0.8", curve))
    model, state, drivers = build_model(_at_unit_speed(path))
    k = model.link_names.index("PU1")
    assert float(drivers["water.setting"][k]) == 1.0
    setting = drivers["water.setting"].clone()
    setting[k] = 0.8
    drivers = {**drivers, "water.setting": setting}
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results)


@pytest.mark.parametrize("kind", list(_PUMP_KINDS))
def test_pump_speed_gradient_against_central_differences(tmp_path, kind):
    """d(sum of junction heads)/d(pump speed) by the implicit adjoint against central
    differences of 1e-6, `|analytic - fd| <= 1e-6 |fd|`, at speed 0.9. MEASURED: one point
    134.4 (3.1e-10 relative), three points 398.39 (7.4e-12), segments 1146.06 (2.7e-11),
    constant power 431.34 (3.8e-11)."""
    parameters, curve = _PUMP_KINDS[kind]
    model, state, drivers, _ = _pump_case(tmp_path, _pump_edits(parameters, curve))
    k = model.link_names.index("PU1")
    junctions = [model.node_names.index(j.name) for j in model.water_network.junctions]

    def total(setting):
        final = water_steady(model, state, {**drivers, "water.setting": setting})
        return final["water.phi"][junctions].sum()

    setting = drivers["water.setting"].clone()
    setting[k] = 0.9
    setting.requires_grad_(True)
    (analytic,) = torch.autograd.grad(total(setting), setting)
    e = torch.zeros_like(setting)
    e[k] = 1e-6
    with torch.no_grad():
        fd = float((total(setting + e) - total(setting - e)) / 2e-6)
    assert abs(float(analytic[k]) - fd) <= 1e-6 * abs(fd), (float(analytic[k]), fd)


def test_ky10_constant_power_pumps_at_time_zero(tmp_path):
    """ky10 (`wntr/library/networks/ky10.inp`): thirteen constant-power pumps, PRVs and
    tanks, solved at time zero with EPANET's own demands there (the demand patterns are
    applied over time by `water_run`). Heads 1e-6 relative, flows 1e-6 m3/s, every link's
    state as EPANET's. MEASURED: heads 6.3e-7, flows 9.1e-8 m3/s."""
    path = Path(wntr.__file__).parent / "library" / "networks" / "ky10.inp"
    reference = wntr.network.WaterNetworkModel(str(path))
    reference.options.time.duration = 0
    results = wntr.sim.EpanetSimulator(reference).run_sim(file_prefix=str(tmp_path / "ky10"))
    net = read_epanet_inp(path)
    assert all(p.power is not None for p in net.pumps)
    model, state, drivers = build_model(net)
    sources = drivers["water.sources"].clone()
    demand = results.node["demand"].loc[0]
    for j in net.junctions:
        sources[model.node_names.index(j.name)] = -float(demand[j.name])
    drivers = {**drivers, "water.sources": sources}
    final = water_steady(model, state, drivers)
    skip = tuple(r.name for r in net.reservoirs)
    assert _worst_heads(model, final, results.node["head"].loc[0], skip=skip) < 1e-6
    flows = results.link["flowrate"].loc[0]
    worst_q = max(abs(float(final["water.q"][i]) - float(flows[name]))
                  for i, name in enumerate(model.link_names))
    assert worst_q < 1e-6, worst_q
    states = link_states(model, final, drivers).tolist()
    status = results.link["status"].loc[0]
    assert dict(zip(model.link_names, states, strict=True)) == {
        name: int(status[name]) for name in model.link_names}


#: ky4's two pairs of pipes joining the same two junctions, `(a, b)` with `a` from J-930 to
#: J-929 (from J-702 to J-703) and `b` the other way, so their head losses cancel exactly.
_KY4_PARALLEL = (("P-952", "P-969"), ("P-625", "P-696"))


def test_ky4_at_time_zero(tmp_path):
    """ky4 (`wntr/library/networks/ky4.inp`, `DURATION 0`): two constant-power pumps, one
    closed by `[STATUS]`, T-2 at its minimum (empty), 959 junctions. One steady solve at
    t = 0 with the t = 0 demands: every junction's demand times pattern "1" at t = 0
    (0.33), applied here because `water_run` (which applies patterns) is not in this test.
    The `[CONTROLS]` (pump ~@Pump-1 on T-3 below 90.75 / above 105.75 ft) do not act at
    t = 0: T-3 starts at 100.751 ft. Without the pattern the demands are 1/0.33 times
    EPANET's and P-539's flow is 1.4e-2 m3/s off (MEASURED; the reviewer's report).

    Heads 1e-6 relative, link states equal, flows 1e-6 m3/s on every link but the four
    below. MEASURED: heads 8.2e-8, flows 2.0e-7 on the others.

    EPANET runs with `ACCURACY 1e-8`, not the file's 1e-4: EPANET stops when the flow
    change is below that fraction of the total flow (`hydsolver.c`, `newflows`), and at
    1e-4 its flows are off by up to 1.1e-5 m3/s (P-965, 2.8e-5 m3/s); at 1e-6 and 1e-8 its
    answer no longer moves (MEASURED).

    The two pairs of parallel pipes in `_KY4_PARALLEL` carry near-zero flows (1e-6 to
    2e-5 m3/s) on a head-loss slope of 1e-4 to 1e-2 m per m3/s, so a head difference of
    1e-10 m moves their split by 1e-6 m3/s. EPANET's split does not satisfy its own
    equations: the two pipes' Hazen-Williams losses at its flows differ by 1.0e-7 m
    (P-952/P-969) and 5.9e-9 m (P-625/P-696), though they join the same nodes (MEASURED),
    and its flows there differ from noodl's by up to 1.9e-6 m3/s. Those four are held to
    3e-6 m3/s and, instead, to energy balance: noodl's losses around each pair cancel to
    1e-9 m (MEASURED 3.5e-10 m).
    """
    path = Path(wntr.__file__).parent / "library" / "networks" / "ky4.inp"
    reference = wntr.network.WaterNetworkModel(str(path))
    reference.options.hydraulic.accuracy = 1e-8
    results = wntr.sim.EpanetSimulator(reference).run_sim(file_prefix=str(tmp_path / "ky4"))
    net = read_epanet_inp(path)
    assert net.duration == 0 and net.demand_pattern == "1"
    assert [p.power is not None for p in net.pumps] == [True, True]
    model, state, drivers = build_model(net)
    drivers = {**drivers,
               "water.sources": drivers["water.sources"] * _pattern_factor(net, 0.0)}
    final = water_steady(model, state, drivers)
    skip = tuple(r.name for r in net.reservoirs)
    assert _worst_heads(model, final, results.node["head"].loc[0], skip=skip) < 1e-6
    flows = results.link["flowrate"].loc[0]
    parallel = {name for pair in _KY4_PARALLEL for name in pair}
    q = dict(zip(model.link_names, final["water.q"].tolist(), strict=True))
    worst = max(abs(q[name] - float(flows[name])) for name in q if name not in parallel)
    assert worst < 1e-6, worst
    assert max(abs(q[name] - float(flows[name])) for name in parallel) < 3e-6
    pipes = {p.name: p for p in net.pipes}

    def loss(name):   # Hazen-Williams, m, signed along the pipe
        p = pipes[name]
        k = 10.666829500036 * p.roughness ** -1.852 * p.diameter ** -4.871 * p.length
        return math.copysign(k * abs(q[name]) ** 1.852, q[name])

    for a, b in _KY4_PARALLEL:
        assert abs(loss(a) + loss(b)) < 1e-9, (a, b, loss(a) + loss(b))
    states = link_states(model, final, drivers).tolist()
    status = results.link["status"].loc[0]
    assert dict(zip(model.link_names, states, strict=True)) == {
        name: int(status[name]) for name in model.link_names}


#: A single point, 4 L/s at 10 m.
_ONE_POINT_4_10 = " C1\t4\t10\n"

_BEHIND_A_PUMP = """[JUNCTIONS]
 J 0 0
 K 0 0
[RESERVOIRS]
 R1 50
 R2 100
[PIPES]
 P1 J K 100 300 130 0 Open
 P2 K R2 100 300 130 0 {status}
[PUMPS]
 PU R1 J HEAD C1
[CURVES]
{curve}[OPTIONS]
 Units LPS
 Headloss H-W
 Accuracy 1e-8
 Trials 500
[TIMES]
 Duration 0
[END]
"""


@pytest.mark.parametrize("curve, shut_off", [(_NET3_CURVE, 104 * _M_PER_FT),
                                             (_ONE_POINT_4_10, 13.3334)],
                         ids=["three-point", "one-point"])
@pytest.mark.parametrize("status, pump_state", [("Closed", 1), ("Open", 0)])
def test_a_zero_demand_group_behind_a_pump_matches_epanet(tmp_path, status, pump_state,
                                                          curve, shut_off):
    """R1 (50 m) - PU - J - P1 - K - P2 - R2 (100 m), no demand; Net3's pump curve
    (shut-off 31.7 m) or a single point (4 L/s at 10 m: EPANET's shut-off 1.33334 x 10 m).
    P2 CLOSED: J and K are fed by the pump alone, which runs at zero flow on its curve, J
    and K at R1 plus the shut-off head, and the pump OPEN, as EPANET has it (its closed P2
    leaks 3e-8 m3/s backwards through the pump, too little for `pumpstatus` to close it).
    P2 OPEN: R2 holds them above the shut-off head and the pump is CLOSED. Heads 1e-6
    relative, flows 1e-6 m3/s, states equal. (With the exact 4/3 the single point's J was
    6.7e-5 m low, 1.05e-6 relative, MEASURED.)"""
    path, results = _run_inp(tmp_path, _BEHIND_A_PUMP.format(status=status, curve=curve))
    model, state, drivers = build_model(read_epanet_inp(path))
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results)
    assert link_states(model, final, drivers)[model.link_names.index("PU")] == pump_state
    phi = dict(zip(model.node_names, final["water.phi"].tolist(), strict=True))
    expected = 50.0 + shut_off if status == "Closed" else 100.0
    assert phi["J"] == pytest.approx(expected, abs=1e-9)
    assert final["water.q"].abs().max() < 1e-12


_ABOVE_ZERO_FLOW = """[JUNCTIONS]
 J 0 0
[RESERVOIRS]
 R1 50
 R2 {r2}
[PIPES]
 P1 J R2 100 300 130 0 Open
[PUMPS]
 PU R1 J HEAD C1
[CURVES]
 C1 10 30
 C1 20 25
 C1 30 15
 C1 40 0
[OPTIONS]
 Units LPS
 Headloss H-W
 Accuracy 1e-8
 Trials 500
[TIMES]
 Duration 0
[END]
"""


@pytest.mark.parametrize("r2, pump_state", [(78.0, 1), (82.0, 0), (86.0, 0)])
def test_a_segmented_curve_starting_above_zero_flow_shuts_at_its_first_head(
        tmp_path, r2, pump_state):
    """A four-point curve from (10 L/s, 30 m): its first segment extended reaches 35 m at
    zero flow, but EPANET shuts the pump once more than the first point's 30 m is asked of
    it (`pumpstatus`, ``Hmax``): at R2 = 82 m the extended segment would still deliver
    6 L/s. Heads 1e-6 relative, flows 1e-6 m3/s, states equal."""
    path, results = _run_inp(tmp_path, _ABOVE_ZERO_FLOW.format(r2=r2))
    model, state, drivers = build_model(read_epanet_inp(path))
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results)
    assert link_states(model, final, drivers)[model.link_names.index("PU")] == pump_state


_PUMP_AND_CHECK_VALVE = """[JUNCTIONS]
 J 0 {demand}
 K 0 0
[RESERVOIRS]
 R1 50
 R2 {r2}
[PIPES]
 P1 J K 100 300 130 0 Open
 P2 K R2 100 300 130 0 CV
[PUMPS]
 PU R1 J HEAD C1
[CURVES]
 C1 4 10
[OPTIONS]
 Units LPS
 Headloss H-W
 Accuracy 1e-8
 Trials 500
[TIMES]
 Duration 0
[END]
"""


@pytest.mark.parametrize("demand, r2", [(0.5, 100), (0.1, 150), (0.0, 70)])
def test_a_pump_and_a_check_valve_shutting_together_match_epanet(tmp_path, demand, r2):
    """R1 (50 m) - PU (shut-off 13.3334 m) - J - P1 - K - P2 (a check valve) - R2 above:
    an iterate shuts both PU and P2 and cuts J and K off. Each law alone saw the other as
    open, and the step was singular (MEASURED before `ShutLinks`); the pumps' cut-off test
    now counts the check valves shut on the same iterate and keeps PU on its curve. At the
    solution PU feeds J's demand (or runs at zero flow, J at R1 plus the shut-off head) and
    P2 is shut, as EPANET has it. Heads 1e-6 relative, flows 1e-6 m3/s, states equal.

    With no demand and R2 above R1 plus the shut-off head (100 m), J's head is not unique:
    every value from 63.3334 m (PU at zero flow at its shut-off head) up to R2 satisfies
    every link. EPANET's value is set by the ratio of its closed-link leaks and its status
    path (74.48 m, PU shut); noodl returns the lower end, which EPANET also returns when a
    CLOSED pipe replaces the check valve (`test_a_zero_demand_group_behind_a_pump_matches_
    epanet`). That case is therefore not compared here."""
    path, results = _run_inp(tmp_path, _PUMP_AND_CHECK_VALVE.format(demand=demand, r2=r2))
    model, state, drivers = build_model(read_epanet_inp(path))
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results)
    states = dict(zip(model.link_names, link_states(model, final, drivers).tolist(),
                      strict=True))
    assert states["PU"] == 1 and states["P2"] == 0


_STOPPED_PUMP_AND_CHECK_VALVE = _PUMP_AND_CHECK_VALVE.replace("[OPTIONS]", "{status}[OPTIONS]")


@pytest.mark.parametrize("r2", [100, 55])
def test_a_stopped_pump_and_a_check_valve_fencing_a_group(tmp_path, r2):
    """R1 (50 m) - PU - J - P1 - K - P2 (a check valve) - R2, no demand, the pump at speed 0
    through `"water.setting"` (EPANET: `[STATUS] PU 0`, which it reports CLOSED). The step was
    singular: a pump at speed 0 counted as open in the cut-off test (`ShutLinks.closed`).
    Now it is solved as with the pump CLOSED by its status, to the same heads, every flow
    exactly zero, PU CLOSED as EPANET reports it, flows within 1e-6 m3/s of EPANET's leaks.

    J's head is not unique (every value up to R2 satisfies every link): noodl gives the
    `1 / CBIG` limit with P2 shut, the average of R1 and R2; EPANET keeps P2 OPEN from its
    first iterate (`cvstatus` holds a status inside `HTOL`) and reports R2's head. MEASURED:
    EPANET J = R2 with P2 OPEN at both R2."""
    text = _STOPPED_PUMP_AND_CHECK_VALVE.format(demand=0, r2=r2, status="{status}")
    _, results = _run_inp(tmp_path, text.format(status="[STATUS]\n PU 0\n"), "epanet")
    path = tmp_path / "noodl.inp"
    path.write_text(text.format(status=""))
    model, state, drivers = build_model(read_epanet_inp(path))
    k = model.link_names.index("PU")
    stopped = drivers["water.setting"].clone()
    stopped[k] = 0.0
    by_speed = {**drivers, "water.setting": stopped}
    final = water_steady(model, state, by_speed)
    closed = drivers["water.status"].clone()
    closed[k] = 0.0
    by_status = {**drivers, "water.status": closed}
    reference = water_steady(model, state, by_status)
    assert torch.equal(final["water.phi"], reference["water.phi"])
    phi = dict(zip(model.node_names, final["water.phi"].tolist(), strict=True))
    assert phi["J"] == phi["K"] == pytest.approx(0.5 * (50.0 + r2), abs=1e-9)
    assert bool((final["water.q"] == 0).all())
    flows = results.link["flowrate"].loc[0]
    assert max(abs(float(flows[name])) for name in model.link_names) < 1e-6
    states = dict(zip(model.link_names, link_states(model, final, by_speed).tolist(),
                      strict=True))
    assert states["PU"] == int(results.link["status"].loc[0]["PU"]) == 0
    assert states["P2"] == 0 and int(results.link["status"].loc[0]["P2"]) == 1
    assert float(results.node["head"].loc[0]["J"]) == pytest.approx(r2, abs=1e-4)


# --------------------------------------------------------------------------- emitters, C-M
_EMITTERS = (
    (";Junction        \tCoefficient", ";Junction\tCoefficient\n J4\t2.0\n J6\t1.5"),
)


def _emitter_flows(model, final):
    layer = model.kirchhoff["water"]
    source = next(w for w in layer.withdrawals if isinstance(w, EmitterOutflow))
    flow = source.flow(final["water.phi"][..., source.nodes]).detach()
    return {model.node_names[int(n)]: float(f) for n, f in zip(source.nodes, flow, strict=True)}


def _emitters_as_epanet(net):
    """`net` with its emitter coefficients in EPANET's internal flow unit, the demands'
    correction (`_as_epanet`): EPANET's `Ke` is built with its rounded LPS factor."""
    return replace(net, emitters=tuple(
        replace(e, coefficient=e.coefficient * (28.3168466 / 28.317)) for e in net.emitters))


@pytest.mark.parametrize("elevation_j4", [0, 60], ids=["positive-pressure", "negative-pressure"])
def test_emitters_match_epanet(tmp_path, elevation_j4):
    """Heads 1e-6 relative, flows and emitter outflows 1e-6 m3/s, with emitters at J4 and J6
    (`[EMITTERS]` 2.0 and 1.5 LPS per m^0.5); and, raising J4 to 60 m (above the reservoir),
    an emitter at NEGATIVE pressure, which EPANET (`hydcoeffs.c`, `emitterheadloss`: the loss
    is odd in the flow) turns into an inflow (J4: -6.93e-3 m3/s, the sign checked below).

    The comparison is with EPANET's own flow unit (`_as_epanet` for the demands,
    `_emitters_as_epanet` for the coefficients), exactly as the other Hazen-Williams tests:
    with the file's values the heads differ by 1.03e-6 relative (MEASURED, the rounded LPS
    factor; held to 1e-5). MEASURED with EPANET's unit: heads 8.64e-8 (positive pressure) and
    6.99e-8 (negative), emitter outflows 7.4e-8 and 5.7e-8 m3/s, flows 3.7e-7 and 2.6e-7 m3/s
    (EPANET REPORTS flows in LPS with the same rounded factor, 5.4e-6 relative of 0.04 m3/s).
    The reference's emitter flow is its own `demand` minus the junction's base demand."""
    text = (DATA / "twoloop_si.inp").read_text().replace(*_EMITTERS[0])
    text = text.replace(" J4              	0           	10 ",
                        f" J4              	{elevation_j4}           	10 ")
    path, results = _run_inp(tmp_path, text, "emitters")
    net = read_epanet_inp(path)
    assert [e.node for e in net.emitters] == ["J4", "J6"]
    model, state, drivers = build_model(net)
    heads = results.node["head"].loc[0]
    demands = results.node["demand"].loc[0]
    final = water_steady(model, state, drivers)
    worst_h = _worst_heads(model, final, heads, skip=("R1",))
    assert worst_h < 1e-5, worst_h
    model, state, drivers = build_model(_emitters_as_epanet(net))
    final = water_steady(model, state, _as_epanet(drivers))
    worst_h = _worst_heads(model, final, heads, skip=("R1",))
    assert worst_h < 1e-6, worst_h
    base = {j.name: j.demand for j in net.junctions}
    emitted = _emitter_flows(model, final)
    for name, flow in emitted.items():
        assert abs(flow - (float(demands[name]) - base[name])) < 1e-6, (name, flow)
    assert (emitted["J4"] < 0) == (elevation_j4 == 60)
    flows = results.link["flowrate"].loc[0]
    worst_q = max(abs(float(final["water.q"][i]) - float(flows[name]))
                  for i, name in enumerate(model.link_names))
    assert worst_q < 1e-6, worst_q


def test_chezy_manning_matches_epanet(tmp_path):
    """Heads 1e-6 relative, flows 1e-6 m3/s with `HEADLOSS C-M` and roughness 0.011 on
    every pipe (the file's `HEADLOSS H-W` replaced, the roughness 130 replaced).

    MEASURED with the file's own demands: heads 7.6e-8 relative, flows 8.3e-9 m3/s. The law
    carries EPANET's own LPS factor (`ChezyManning(cfs_per_m3s=...)`, as Darcy-Weisbach does),
    so unlike Hazen-Williams it needs no `_as_epanet` correction: with it the heads are
    4.8e-7, WORSE, which is the correction applied twice."""
    text = (DATA / "twoloop_si.inp").read_text().replace("130         	0", "0.011       	0")
    assert text.count("0.011") == 8 and text.count("	H-W") == 1
    path, results = _run_inp(tmp_path, text.replace("	H-W", "	C-M"), "cm")
    net = read_epanet_inp(path)
    assert net.headloss == "C-M"
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    heads = results.node["head"].loc[0]
    flows = results.link["flowrate"].loc[0]
    worst_h = _worst_heads(model, final, heads, skip=("R1",))
    assert worst_h < 1e-6, worst_h
    worst_q = max(abs(float(final["water.q"][i]) - float(flows[name]))
                  for i, name in enumerate(model.link_names))
    assert worst_q < 1e-6, worst_q


def test_emitter_coefficient_gradient_against_central_differences(tmp_path):
    """d(head at J6)/d(coefficient at J4) through the solve equals the central difference
    of the solve (step 1e-4 relative) to 1e-8 relative; MEASURED 4.8e-11 (4.4e-9 at 1e-3,
    4.4e-7 at 1e-2: the difference's own O(eps^2) error)."""
    path, _ = _epanet(tmp_path, "twoloop_si.inp", edits=_EMITTERS)
    net = read_epanet_inp(path)

    def head_at(scale, grad=False):
        emitters = tuple(type(e)(e.node, e.coefficient * (scale if e.node == "J4" else 1.0))
                         for e in net.emitters)
        model, state, drivers = build_model(replace(net, emitters=emitters))
        layer = model.kirchhoff["water"]
        source = next(w for w in layer.withdrawals if isinstance(w, EmitterOutflow))
        if grad:
            source.coefficient.requires_grad_(True)
        final = water_steady(model, state, drivers)
        return final["water.phi"][model.node_names.index("J6")], source

    head, source = head_at(1.0, grad=True)
    head.backward()
    j4 = int((source.nodes == build_model(net)[0].node_names.index("J4")).nonzero())
    analytic = float(source.coefficient.grad[j4])
    c4 = net.emitters[0].coefficient
    eps = 1e-4
    numeric = (float(head_at(1 + eps)[0]) - float(head_at(1 - eps)[0])) / (2 * eps * c4)
    assert abs(analytic - numeric) < 1e-8 * abs(numeric), (analytic, numeric)


def _series_text(junctions, pipes, emitters, *, head="H-W", options=""):
    """R1 (50 m) and the junctions `(name, elevation, demand)`, pipes
    `(name, from, to, status)` (1000 m, 200 mm, C = 100) and emitters `(junction, coefficient)`,
    as LPS `.inp` text with EPANET run to ACCURACY 1e-8."""
    return ("[JUNCTIONS]\n" + "".join(f" {n} {z} {d}\n" for n, z, d in junctions)
            + "[RESERVOIRS]\n R1 50\n[PIPES]\n"
            + "".join(f" {n} {a} {b} 1000 200 100 0 {s}\n" for n, a, b, s in pipes)
            + "[EMITTERS]\n" + "".join(f" {n} {c}\n" for n, c in emitters)
            + "[TIMES]\n Duration 0:00\n[OPTIONS]\n Units LPS\n Headloss " + head
            + "\n Accuracy 1e-8\n HEADERROR 1e-9\n FLOWCHANGE 1e-9\n Trials 500\n"
            + options + "[END]\n")


_BEHIND_A_CLOSED_PIPE = {
    # the reviewer's repro: the emitter drains J2 toward its elevation, J2 = 0 m
    "emitter-only-junction": (
        (("J1", 0, 0), ("J2", 0, 0)),
        (("P1", "R1", "J1", "Open"), ("P2", "J1", "J2", "Closed")), (("J2", 1.0),)),
    # an emitter junction and a zero-demand junction joined by an open pipe
    "emitter-and-zero-demand-junction": (
        (("J1", 0, 0), ("J2", 0, 0), ("J3", 0, 0)),
        (("P1", "R1", "J1", "Open"), ("P2", "J1", "J2", "Closed"), ("P3", "J2", "J3", "Open")),
        (("J2", 1.0),)),
    # a demand supplied only by the emitter's INFLOW at negative pressure (J2 = -25 m)
    "demand-fed-by-an-emitter-inflow": (
        (("J1", 0, 0), ("J2", 0, 5)),
        (("P1", "R1", "J1", "Open"), ("P2", "J1", "J2", "Closed")), (("J2", 1.0),)),
}


@pytest.mark.parametrize("case", list(_BEHIND_A_CLOSED_PIPE))
def test_emitters_behind_a_closed_pipe_match_epanet(tmp_path, case):
    """An emitter ties its junction's head to the junction's elevation as a reservoir there
    would, so the part of the network behind a closed pipe that holds one is neither "without
    a head" (the limit head EPANET gives a zero-demand part, tied to the other side of the
    closed link) nor "cut off" (refused when it has a demand): the solve finds its heads.
    Heads 1e-6 relative to the 50 m reservoir, flows 1e-6 m3/s, every link's state as
    EPANET's. Before `WaterNetwork.fixed_head_nodes` the first case gave J2 = 49.976 m,
    EPANET's 0."""
    junctions, pipes, emitters = _BEHIND_A_CLOSED_PIPE[case]
    path, results = _run_inp(tmp_path, _series_text(junctions, pipes, emitters), case)
    net = _emitters_as_epanet(read_epanet_inp(path))
    model, state, drivers = build_model(net)
    drivers = _as_epanet(drivers)
    final = water_steady(model, state, drivers)
    heads = results.node["head"].loc[0]
    worst_h = max(abs(float(final["water.phi"][i]) - float(heads[name]))
                  for i, name in enumerate(model.node_names))
    # The third case is 6.92e-4 m (2.8e-5 relative to its 25 m) apart, with the cause
    # measured: EPANET gives the closed P2 the conductance `1 / CBIG` (`hydcoeffs.c`,
    # `pipecoeff`), so the 75 m across it leaks 7e-5 LPS, 1.4e-5 of the 5 LPS demand the
    # emitter supplies, which shifts the emitter's head by twice that; this model's closed
    # link carries exactly zero. The first two are 1e-6 relative to the 50 m reservoir.
    tolerance = 1e-3 if case == "demand-fed-by-an-emitter-inflow" else 1e-6 * 50
    assert worst_h < tolerance, worst_h
    reference = results.link["flowrate"].loc[0]
    for i, name in enumerate(model.link_names):
        assert abs(float(final["water.q"][i]) - float(reference[name])) < 1e-6, name
    states = link_states(model, final, drivers).tolist()
    status = results.link["status"].loc[0]
    assert states == [int(status[name]) for name in model.link_names]


def test_a_check_valve_on_a_chezy_manning_network_matches_epanet(tmp_path):
    """P4 (J3 -> J1) as a check valve closes (its water runs the other way) and P5 stays
    open, on the two-loop with `HEADLOSS C-M`, roughness 0.011; P7 closed by `[STATUS]`.
    Heads and flows 1e-6, every link state as EPANET's."""
    text = (DATA / "twoloop_si.inp").read_text()
    for old, new in (_pipe_status_edit("P4", "CV"), _pipe_status_edit("P5", "CV"),
                     ("[STATUS]\n;ID              \tStatus/Setting",
                      "[STATUS]\n;ID              \tStatus/Setting\n P7              \tClosed")):
        assert old in text, old
        text = text.replace(old, new)
    text = text.replace("130         \t0", "0.011       \t0").replace("\tH-W", "\tC-M")
    assert text.count("0.011") == 8 and text.count("CV") >= 2
    path, results = _run_inp(tmp_path, text, "cm-cv")
    net = read_epanet_inp(path)
    assert net.headloss == "C-M"
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    _compare_with_epanet(model, final, drivers, results)


def test_an_emitter_with_pressure_driven_demand_on_one_junction_matches_epanet(tmp_path):
    """`DEMAND MODEL PDA` (REQUIRED PRESSURE 60 m) and emitters at J4 and J6 together on the
    two-loop. EPANET's reported demand is the pressure-driven demand plus the emitter flow."""
    text = (DATA / "twoloop_si.inp").read_text().replace(*_EMITTERS[0])
    text = text.replace(*_PDA_EDIT)
    path, results = _run_inp(tmp_path, text, "pda-emitter")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(_emitters_as_epanet(net))
    final = water_steady(model, state, _as_epanet(drivers))
    _compare_with_epanet(model, final, drivers, results, heads=1e-5)
    layer = model.kirchhoff["water"]
    demand = next(w for w in layer.withdrawals if isinstance(w, PressureDrivenDemand))
    pda = demand.flow(final["water.phi"][..., demand.nodes]).detach()
    emitted = _emitter_flows(model, final)
    reported = results.node["demand"].loc[0]
    for k, j in enumerate(net.junctions):
        total = float(pda[k]) + emitted.get(j.name, 0.0)
        assert abs(total - float(reported[j.name])) < 1e-5, (j.name, total)


def test_emitters_on_a_darcy_weisbach_network_match_epanet(tmp_path):
    """The pressure layer (Pa): emitters at J4 and J6 on the D-W two-loop, with EPANET's flow
    unit (`_emitters_as_epanet`)."""
    text = (DATA / "twoloop_si.inp").read_text().replace(*_EMITTERS[0])
    for old, new in _DW_TWO_LOOP:
        assert old in text, old
        text = text.replace(old, new)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        path, results = _run_inp(tmp_path, text, "dw-emitters")
    net = read_epanet_inp(path)
    assert net.headloss == "D-W"
    model, state, drivers = build_model(_emitters_as_epanet(net))
    final = water_steady(model, state, drivers)
    heads = results.node["head"].loc[0]
    scale = model.head_scale
    worst_h = max(abs(float(final["water.phi"][i]) / scale - float(heads[name]))
                  / (abs(float(heads[name])) or 1.0)
                  for i, name in enumerate(model.node_names) if name != "R1")
    assert worst_h < 1e-5, worst_h
    emitted = _emitter_flows(model, final)
    demands = results.node["demand"].loc[0]
    base = {j.name: j.demand for j in net.junctions}
    for name, flow in emitted.items():
        assert abs(flow - (float(demands[name]) - base[name])) < 1e-5, (name, flow)


# --------------------------------------------------------------------------- tanks over time
# Volume curves, full and empty tanks and overflow, each over an extended period against
# EPANET. Until `water_run` exists these tests step with a local loop (`_tank_run`): one
# `water_event_step` per hydraulic step, which solves with every tank held, shortens the
# step to the next control crossing and to the time a tank fills or empties
# (`hydraul.c`, `tanktimestep`), and moves the tanks.
_NET1_TANK = (" 2               \t850         \t120         \t100         \t150         "
              "\t50.5        \t0           \t                \t;")
_NET1_CONTROLS = (" LINK 9 OPEN IF NODE 2 BELOW 110\n LINK 9 CLOSED IF NODE 2 ABOVE 140", "")


def _tank_run(net, model, state, drivers):
    """`(levels, flows, states)` at every report time from 0 to the duration: every tank's
    level, the link flows of the solve at that time (EPANET reports the solution of the
    step that starts there) and the solved states."""
    base = drivers["water.sources"].clone()

    def at(time):
        return {**drivers, "water.sources": base * _pattern_factor(net, time)}

    levels, flows, states = [], [], []
    time = 0.0
    while True:
        levels.append(tank_levels(model, state).clone())
        if time >= net.duration - 1e-9:
            solved = water_steady(model, state, at(time))
            flows.append(solved["water.q"])
            states.append(solved)
            break
        report_end = time + net.report_timestep
        first = None
        while time < report_end - 1e-9:
            state, dt = water_event_step(model, state, at(time), report_end - time)
            if first is None:
                first = state
            time += dt
        flows.append(first["water.q"])
        states.append(first)
    return torch.stack(levels), torch.stack(flows), states


def _compare_tank_run(tmp_path, text, tag, *, tanks, links=()):
    """Run `text` through EPANET and `_tank_run`; return the worst level difference (m)
    over every reported time and tank, and the worst flow difference (m3/s) of `links`.
    The file is read from its own name: EPANET's run rewrites `<tag>.inp` in wntr's words."""
    path = tmp_path / f"{tag}-input.inp"
    path.write_text(text)
    results = wntr.sim.EpanetSimulator(wntr.network.WaterNetworkModel(str(path))).run_sim(
        file_prefix=str(tmp_path / tag))
    net = read_epanet_inp(path)
    model, start, drivers = build_model(net)
    levels, flows, states = _tank_run(net, model, start, drivers)
    heads = results.node["head"]
    assert levels.shape[0] == len(heads.index)
    worst_level = 0.0
    for k, name in enumerate(t.name for t in net.tanks):
        if name not in tanks:
            continue
        theirs = torch.tensor(heads[name].to_numpy(), dtype=F64) - net.tanks[k].elevation
        worst_level = max(worst_level, float((levels[:, k] - theirs).abs().max()))
    worst_flow = 0.0
    for name in links:
        j = model.link_names.index(name)
        theirs = torch.tensor(results.link["flowrate"][name].to_numpy(), dtype=F64)
        worst_flow = max(worst_flow, float((flows[:, j] - theirs).abs().max()))
    return worst_level, worst_flow, (net, model, levels, flows, start, drivers)


def test_volume_curve_tank_matches_epanet(tmp_path):
    """Net1 with tank 2 on a volume curve whose area changes at every point the level
    crosses (the pump's controls at 110 and 140 ft fall on different segments), 24 h.
    MEASURED: 6.9e-5 m levels, 5.5e-8 m3/s in pipe 110."""
    curve = "\n".join(f" VC {x} {y}" for x, y in ((0, 0), (60, 90000), (110, 200000),
                                                   (130, 250000), (160, 360000)))
    text = (DATA / "Net1.inp").read_text()
    assert _NET1_TANK in text
    text = text.replace(_NET1_TANK, _NET1_TANK.replace("\t                \t;",
                                                       "\tVC              \t;"))
    text = text.replace("[CURVES]", "[CURVES]\n" + curve)
    worst, worst_q, (net, *_rest) = _compare_tank_run(tmp_path, text, "vc", tanks={"2"},
                                                      links=("110",))
    assert net.tanks[0].volume_curve == "VC"
    assert worst < 2e-4, worst
    assert worst_q < 1e-6, worst_q


def test_tank_fills_and_closes_its_inlet_like_epanet(tmp_path):
    """Net1 without its pump controls: the tank fills at about 15.5 h and stays full, with
    pipe 110 closed against inflow (`tankstatus`) and the pump supplying only the demand.
    MEASURED: 2.5e-5 m levels, 3.7e-8 m3/s in pipe 110 and the pump."""
    text = (DATA / "Net1.inp").read_text().replace(*_NET1_CONTROLS)
    worst, worst_q, (net, model, levels, flows, _s, _d) = _compare_tank_run(
        tmp_path, text, "fill", tanks={"2"}, links=("110", "9"))
    full = levels[:, 0] >= net.tanks[0].max_level - 1e-6
    assert int(full.sum()) == 9                               # 16 h to 24 h, at the max
    assert bool((flows[full, model.link_names.index("110")].abs() < 1e-12).all())
    assert worst < 2e-4, worst
    assert worst_q < 1e-6, worst_q


def test_tank_empties_like_epanet(tmp_path):
    """Net1 without its controls at 1.5 times the demand and a minimum level of 115 ft: the
    tank empties at about 6.5 h, pipe 110 then refuses to drain it while the pump alone
    supplies the network, and from 12 h the network refills it. MEASURED: 2.3e-5 m levels,
    3.2e-8 m3/s in pipe 110 and the pump."""
    text = (DATA / "Net1.inp").read_text().replace(*_NET1_CONTROLS)
    text = text.replace(" Demand Multiplier  \t1.0", " Demand Multiplier  \t1.5")
    text = text.replace(_NET1_TANK, _NET1_TANK.replace("\t120         \t100   ",
                                                       "\t120         \t115   "))
    worst, worst_q, (net, model, levels, flows, _s, _d) = _compare_tank_run(
        tmp_path, text, "empty", tanks={"2"}, links=("110", "9"))
    empty = levels[:, 0] <= net.tanks[0].min_level + 1e-6
    assert int(empty.sum()) >= 5
    pipe = flows[empty, model.link_names.index("110")]
    # every report time the tank is empty, pipe 110 is closed against draining it, except
    # the last (12 h), where the network head has risen past the tank's and the pipe is
    # refilling it (-16 L/s, EPANET -16.41 L/s) from an empty tank, which the rule allows
    assert bool((pipe[:-1].abs() < 1e-12).all())
    assert float(pipe[-1]) < -0.01
    assert worst < 2e-4, worst
    assert worst_q < 1e-6, worst_q


_FULL_AT_START = """[TITLE]
A pump filling a tank that starts full
[JUNCTIONS]
 J0  44  0
 J1  40  5
[RESERVOIRS]
 R1  45
[TANKS]
 T1  50  10  1  10  8  0  *  ;
[PIPES]
 P1  T1  J1  500  200  100  0  Open
 P0  R1  J0  50  300  100  0  Open
[PUMPS]
 PU1  J0  T1  HEAD C1
[CURVES]
 C1  12  30
[PATTERNS]
 1  1.0  1.6  0.6  1.2
[OPTIONS]
 Units  LPS
 Headloss  H-W
 Pattern  1
[TIMES]
 Duration  12:00
 Hydraulic Timestep  1:00
 Pattern Timestep  3:00
 Report Timestep  1:00
[END]
"""


def test_full_tank_at_start_closes_the_pump(tmp_path):
    """A pump discharging into a tank that starts full is closed (`tankstatus`, case 1) --
    a CLOSED status, so its own shut-off row is not reached -- and reopens once the tank
    has drained below the band; from then on it fills the tank to full and closes again
    many times an hour, each fill ended by the step to the instant the tank is full
    (`tanktimestep`, rounded to whole seconds as EPANET does). MEASURED: 5.3e-6 m levels,
    8.9e-9 m3/s in the pump and the pipe."""
    worst, worst_q, (net, model, levels, flows, start, drivers) = _compare_tank_run(
        tmp_path, _FULL_AT_START, "full", tanks={"T1"}, links=("PU1", "P1"))
    pump = model.link_names.index("PU1")
    assert float(flows[0, pump]) == 0.0
    solved = water_steady(model, start, drivers)
    assert int(link_states(model, solved, drivers)[pump]) == 0
    assert float(flows[1, pump]) > 0.01
    assert worst < 2e-4, worst
    assert worst_q < 1e-6, worst_q


def test_tank_overflow_like_epanet(tmp_path):
    """Net1 without its pump controls and tank 2 allowed to overflow: once full the tank
    keeps its inlet open, its level is held at the maximum, and what flows in is the
    overflow (`tank_overflow`). MEASURED: 2.5e-5 m levels, 2.5e-8 m3/s in pipe 110 and the
    pump."""
    text = (DATA / "Net1.inp").read_text().replace(*_NET1_CONTROLS)
    text = text.replace(_NET1_TANK, _NET1_TANK.replace("\t                \t;",
                                                       "\t*               \tYES"))
    worst, worst_q, (net, model, levels, flows, start, drivers) = _compare_tank_run(
        tmp_path, text, "overflow", tanks={"2"}, links=("110", "9"))
    assert net.tanks[0].overflow
    full = levels[:, 0] >= net.tanks[0].max_level - 1e-6
    assert int(full.sum()) == 9
    pipe = model.link_names.index("110")
    assert bool((flows[full, pipe] < -0.03).all())     # still filling the full tank
    # over a step that starts full, everything that flows in overflows
    full_state = initial_state(model, values={"water.tank_level": {"2": 150 * 0.3048}})
    stepped, dt = water_event_step(model, full_state, drivers, 600.0)
    assert float(tank_levels(model, stepped)[0]) == pytest.approx(150 * 0.3048, abs=1e-9)
    overflow = tank_overflow(model, full_state, stepped, dt)
    assert float(overflow[0]) > 0.03
    torch.testing.assert_close(overflow, tank_inflow(model, stepped), rtol=1e-12, atol=0)
    assert worst < 2e-4, worst
    assert worst_q < 1e-6, worst_q
