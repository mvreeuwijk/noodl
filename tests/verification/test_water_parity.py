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
from pathlib import Path

import pytest
import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import (
    build_model,
    initial_state,
    link_states,
    tank_levels,
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
    """Heads 1e-6, FLOWS 1e-5, pump head gain 1e-6.

    MEASURED: heads 7.058e-8, flows 2.868e-6 on pipe 113 (0.00185 m3/s, the smallest in the
    network), pump head gain 1.189e-7. The flow residual is the reference's: EPANET's own
    reported flows miss continuity at node 13 by 4.5e-10 m3/s while this model's satisfy it
    to 1e-17, and tightening ACCURACY/HEADERROR/FLOWCHANGE to 1e-8/1e-9/1e-9 with 500
    trials leaves EPANET's output BIT-IDENTICAL.
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
    assert worst_q < 1e-5, worst_q
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

    MEASURED: 8.181e-5 m worst over the 25 reported steps, with 26 hydraulic sub-steps for
    24 report steps. A FIXED 1 h step instead diverges by 2.07 m, because EPANET shortens
    its own step to the instant the level reaches the 140 ft trigger and a fixed step
    switches the pump a whole hour late. The 8.2e-5 m floor is the reference's float32 head
    output, whose resolution at 296 m is 3.5e-5 m.
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


def test_a_tank_outside_its_limits_is_refused(tmp_path):
    path, _ = _epanet(tmp_path, "Net1.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
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

    MEASURED: 1.007e-6 relative (analytic 0.430638, central difference 0.430638).
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
