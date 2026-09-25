"""EPANET 2.2 parity for `noodl.apps.water` (verification rows D1-D8 and the golden case G2).

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

import warnings
from pathlib import Path

import pytest
import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import (
    build_model,
    tank_inflow,
    twoloop,
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
    return max(
        abs(float(state["water.phi"][i]) / model.head_scale - float(heads[name]))
        / abs(float(heads[name]))
        for i, name in enumerate(model.node_names)
        if name not in skip
    )


def _worst_flows(model, state, flows):
    return max(
        abs(float(state["water.q"][i]) - float(flows[name])) / abs(float(flows[name]))
        for i, name in enumerate(model.link_names)
    )


# --------------------------------------------------------------------------- D1
def test_d1_two_loop_heads_and_flows(tmp_path):
    """Row D1, 1e-6 relative on heads and flows.

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


def test_d1_reservoir_head_is_held_exactly(tmp_path):
    path, _ = _epanet(tmp_path, "twoloop_si.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    assert float(final["water.phi"][model.node_names.index("R1")]) == 50.0


def test_d1_nodal_continuity_is_exact(tmp_path):
    """This model's own flows satisfy continuity to machine precision, which is why the
    D1/D2 residuals above are attributable to the reference: EPANET's reported flows miss
    continuity at Net1's node 13 by 4.5e-10 m3/s (measured)."""
    path, _ = _epanet(tmp_path, "twoloop_si.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    layer = model.potential["water"]
    residual = layer.residual(
        final["water.phi"][..., layer.interior], drivers["water.phi_boundary"],
        {}, drivers["water.sources"],
    )
    # MEASURED 2.093e-14, at J1 (the 0.045 m3/s pipe next to the reservoir); Newton's own
    # atol here is 1e-11, so this is as converged as the solve was asked to be.
    assert float(residual.abs().max()) < 1e-13


# --------------------------------------------------------------------------- D2
def test_d2_net1_single_period(tmp_path):
    """Row D2: heads 1e-6, FLOWS 1e-5, pump head gain 1e-6.

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


def test_d2_the_reader_converts_net1_gpm_and_feet(tmp_path):
    """The unit conversion D2 rests on, pinned field by field."""
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
    assert [(c.link, c.status, c.node, c.test) for c in net.controls] == [
        ("9", "OPEN", "2", "BELOW"), ("9", "CLOSED", "2", "ABOVE")
    ]
    assert [c.level for c in net.controls] == pytest.approx(
        [110 * 0.3048, 140 * 0.3048], rel=1e-15
    )
    assert net.patterns["1"] == (1.0, 1.2, 1.4, 1.6, 1.4, 1.2,
                                 1.0, 0.8, 0.6, 0.4, 0.6, 0.8)
    assert (net.duration, net.hydraulic_timestep, net.pattern_timestep) == (
        86400.0, 3600.0, 7200.0
    )


# --------------------------------------------------------------------------- D3
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
    solution". `TankLevels.event_step` is that rule; `TankLevels.__call__` re-applies the
    controls at the start of every sub-step.

    Returns the tank level at every REPORT time, starting at t = 0.
    """
    closure = model.tank_closure
    base = drivers["water.sources"].clone()
    levels = [float(state["water.tank_level"][0])]
    time = 0.0
    n_sub = 0
    while time < net.duration - 1e-9:
        report_end = time + net.report_timestep
        while time < report_end - 1e-9:
            step_drivers = dict(drivers)
            step_drivers["water.sources"] = base * _pattern_factor(net, time)
            solved = water_steady(model, state, step_drivers)
            inflow = tank_inflow(model, solved)
            rate = inflow / closure.area
            dt = closure.event_step(state["water.tank_level"], rate, report_end - time)
            state = dict(solved)
            state["water.tank_level"] = closure.advance(
                state["water.tank_level"], inflow, dt
            )
            time += dt
            n_sub += 1
        levels.append(float(state["water.tank_level"][0]))
    return levels, n_sub


def test_d3_net1_extended_period_tank_level(tmp_path):
    """Row D3, 2e-4 m ABSOLUTE per reported step.

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


def test_d3_the_pump_switches_within_one_reported_step(tmp_path):
    """The switch TIME must agree, which is what the event shortening buys."""
    path, results = _epanet(tmp_path, "Net1.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    levels, _ = _extended_period(net, model, state, drivers)
    reference = (results.node["head"]["2"] - net.tanks[0].elevation).tolist()
    mine_peak = levels.index(max(levels))
    their_peak = reference.index(max(reference))
    assert abs(mine_peak - their_peak) <= 1


def test_d3_event_step_returns_the_nominal_step_when_nothing_is_crossed(tmp_path):
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


def test_d3_a_tank_outside_its_limits_is_refused(tmp_path):
    path, _ = _epanet(tmp_path, "Net1.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    state = dict(state)
    state["water.tank_level"] = torch.tensor([100.0], dtype=F64)
    with pytest.raises(ValueError, match=r"tank\(s\) \['2'\] are outside"):
        water_steady(model, state, drivers)


# --------------------------------------------------------------------------- D4
def test_d4_darcy_weisbach_residual_is_recorded_not_bounded(tmp_path):
    """Row D4 RECORDS its residual rather than bounding it.

    MEASURED on the two-loop fixture at roughness 0.26 mm: heads 3.566e-4 relative, flows
    2.731e-3 relative (worst link). EPANET uses Swamee-Jain above Re = 4000,
    Hagen-Poiseuille below 2000 and Dunlop's cubic interpolation between (Manual section
    13.1 item 3, p.111), while the existing
    `Duct` element uses Colebrook throughout; the two turbulent friction factors differ by
    up to about 1 %. (EPANET's water viscosity, 1.1e-5 ft2/s = 1.022e-6 m2/s, against this
    reader's 1.002e-3 / 998.2 = 1.004e-6 m2/s moves the heads residual only to 3.0e-4.)
    Before the Duct was fed rho' = 1/rho and mu' = nu -- i.e. while it returned MASS flow
    into a layer that balances m3/s -- this row recorded 3.822e-2 / 4.446e-1. Hazen-Williams
    is the formula compared against EPANET; D-W is offered, and this row states what it costs.
    """
    # ONE literal replacement covers all eight pipes: the "roughness / minor loss / status"
    # run is identical on every [PIPES] row of the fixture (it occurs exactly 8 times) and
    # `str.replace` rewrites every occurrence. Replacing per (length, diameter) pair does
    # NOT work -- P2 and P5 are both 400 m by 250 mm, so the first pass would consume both
    # and the second would find nothing.
    edits = [
        (" Headloss           \tH-W", " Headloss           \tD-W"),
        ("130         \t0           \tOpen", "0.26        \t0           \tOpen"),
    ]
    # The edit's roughness (0.26 mm) is already in D-W units, so wntr's own warning that
    # switching HEADLOSS to D-W leaves the roughness units unchanged is expected and benign.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        path, results = _epanet(tmp_path, "twoloop_si.inp", edits=edits)
    net = read_epanet_inp(path)
    assert net.headloss == "D-W"
    assert net.pipes[0].roughness == pytest.approx(0.26e-3, rel=1e-15)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers, atol=1e-10, rtol=1e-10)
    heads = results.node["head"].loc[0]
    flows = results.link["flowrate"].loc[0]
    worst_h = _worst_heads(model, final, heads, skip=("R1",))
    worst_q = _worst_flows(model, final, flows)
    print(f"\nD4 Darcy-Weisbach: heads {worst_h:.3e} relative, flows {worst_q:.3e}")
    # Recorded, not bounded: the band is one order of magnitude either side of the measured
    # pair, so a CHANGE in the friction-factor treatment is caught while the known
    # difference is not asserted away.
    assert 3.5e-5 < worst_h < 3.5e-3, worst_h
    assert 2.7e-4 < worst_q < 2.7e-2, worst_q


# --------------------------------------------------------------------------- D5
_PDA_EDIT = (
    " Tolerance          \t0.01",
    " Tolerance          \t0.01\n DEMAND MODEL       \tPDA\n"
    " MINIMUM PRESSURE   \t0\n REQUIRED PRESSURE  \t60\n PRESSURE EXPONENT  \t0.5",
)


def test_d5_pressure_driven_demand(tmp_path):
    """Row D5, 1e-5 relative on heads and delivered demands.

    MEASURED against EPANET's own `DEMAND MODEL PDA` at `MINIMUM PRESSURE 0`,
    `REQUIRED PRESSURE 60`, `PRESSURE EXPONENT 0.5`: heads 3.521e-7, demands 2.184e-7. The
    row validates the core's potential-dependent nodal sources -- EPANET itself formulates PDA as "a
    virtual pipe from the junction to a fictitious reservoir" (Manual section 13.1, p.110),
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
    source = model.potential["water"]._node_sources[0]
    delivered = source.flow(final["water.phi"][..., source.nodes]).detach()
    worst_d = max(
        abs(float(delivered[i]) - float(demands[j.name])) / abs(float(demands[j.name]))
        for i, j in enumerate(net.junctions)
    )
    assert worst_d < 1e-5, worst_d


def test_d5_pda_delivers_less_than_the_required_demand(tmp_path):
    """The whole point of PDA: below the required pressure the demand is curtailed."""
    path, _ = _epanet(tmp_path, "twoloop_si.inp", edits=[_PDA_EDIT])
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    source = model.potential["water"]._node_sources[0]
    delivered = source.flow(final["water.phi"][..., source.nodes]).detach()
    required = torch.tensor([j.demand for j in net.junctions], dtype=F64)
    assert bool((delivered < required).all())
    assert bool((delivered > 0.8 * required).all())


def test_d5_a_zero_pressure_span_is_refused(tmp_path):
    path, _ = _epanet(tmp_path, "twoloop_si.inp")
    net = read_epanet_inp(path)
    with pytest.raises(ValueError, match="must exceed MINIMUM PRESSURE"):
        build_model(net, pda=True, p_min=0.0, p_req=0.0)


# --------------------------------------------------------------------------- D6
def test_d6_head_loss_sums_to_zero_around_every_loop(tmp_path):
    """Row D6, 1e-12. MEASURED: exactly 0.0 on both independent loops.

    This is an IDENTITY of the formulation, not a convergence result: the pipe kind carries
    no drive, so `dp = A^T phi` and every cycle-basis row annihilates it by construction.
    The row therefore pins that `Network.cycle_basis` and the layer's own `difference`
    sign convention agree -- which is a real thing to get wrong -- rather than that the
    Newton solve converged.
    """
    path, _ = _epanet(tmp_path, "twoloop_si.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    final = water_steady(model, state, drivers)
    layer = model.potential["water"]
    basis = model.net.cycle_basis("pipe")
    assert basis.shape[0] == 2
    loops = basis.to(F64) @ layer.dp(final["water.phi"], {})
    assert float(loops.abs().max()) < 1e-12


# --------------------------------------------------------------------------- D7
def test_d7_gradients_against_central_differences(tmp_path):
    """Row D7, `max|analytic - fd| <= 1e-6 * max|fd|`, for BOTH the
    nodal demand `water.sources` and the Hazen-Williams roughness (both channels are
    differenced for real below, so `solve_sum`'s `roughness` branch is exercised). The pump `h0` and
    tank AREA channels are checked separately, on Net1
    (`test_d7_gradient_reaches_the_pump_head_gain_h0` and `test_d7_...tank_area...` below), since
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
    element = model.potential["water"]._elements[0]
    sources = drivers["water.sources"].clone().requires_grad_(True)
    element.roughness.requires_grad_(True)
    final = water_steady(model, {"water.phi": torch.zeros(model.net.n, dtype=F64),
                                 "water.q": torch.zeros(
                                     len(model.potential["water"].cols), dtype=F64)},
                         {**drivers, "water.sources": sources},
                         differentiable=True)
    layer = model.potential["water"]
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
                float(s_up["water.phi"][..., m_up.potential["water"].interior].sum())
                - float(s_down["water.phi"][..., m_down.potential["water"].interior].sum())
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
                float(s_up["water.phi"][..., m_up.potential["water"].interior].sum())
                - float(s_down["water.phi"][..., m_down.potential["water"].interior].sum())
            )
            / (2 * eps_r)
        )
    scale_r = max(abs(v) for v in fd_roughness)
    worst_r = max(
        abs(float(a) - b)
        for a, b in zip(analytic_roughness.tolist(), fd_roughness, strict=True)
    )
    assert worst_r <= 1e-6 * scale_r, (worst_r, 1e-6 * scale_r)


def test_d7_gradient_reaches_the_pump_head_gain_h0(tmp_path):
    """Row D7's pump `h0` channel, on Net1 (`twoloop_si.inp` has no pump).

    MEASURED: 1.007e-6 relative (analytic 0.430638, central difference 0.430638).
    """
    path, _ = _epanet(tmp_path, "Net1.inp", duration=0)
    net = read_epanet_inp(path)

    def solve_with_h0(delta: float):
        model, state, drivers = build_model(net)
        pump = next(el for el in model.potential["water"]._elements if el.kind == "pump")
        with torch.no_grad():
            pump.h0.add_(delta)
        final = water_steady(model, state, drivers)
        return model, final

    model, state, drivers = build_model(net)
    pump = next(el for el in model.potential["water"]._elements if el.kind == "pump")
    pump.h0.requires_grad_(True)
    final = water_steady(model, state, drivers, differentiable=True)
    layer = model.potential["water"]
    final["water.phi"][..., layer.interior].sum().backward()
    analytic = float(pump.h0.grad.detach())

    eps = 1e-4
    m_up, up = solve_with_h0(eps)
    m_down, down = solve_with_h0(-eps)
    fd = (
        float(up["water.phi"][..., m_up.potential["water"].interior].sum())
        - float(down["water.phi"][..., m_down.potential["water"].interior].sum())
    ) / (2 * eps)
    assert abs(analytic - fd) <= 1e-5 * abs(fd), (analytic, fd)


def test_d7_tank_area_is_not_reached_by_a_single_steady_solve(tmp_path):
    """Tank AREA never enters `TankLevels.__call__` (only `bottom + level` does), so a
    single `water_steady` call's output is STRUCTURALLY independent of it -- not a small
    gradient, but no path in the graph at all. This is the "structurally not
    differentiable" case: a blanket claim that roughness, demands, pump h0 AND
    tank area all reach the steady solve would overstate the last one. The next test shows
    where the framework DOES differentiate w.r.t. area: `TankLevels.advance()`.
    """
    path, _ = _epanet(tmp_path, "Net1.inp", duration=0)
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    closure = model.tank_closure
    closure.area.requires_grad_(True)
    final = water_steady(model, state, drivers, differentiable=True)
    assert not final["water.phi"].requires_grad


def test_d7_gradient_reaches_the_tank_area_through_advance(tmp_path):
    """Row D7's tank-area channel is `TankLevels.advance`'s explicit-Euler update (the
    previous test), not the steady solve. MEASURED: 2.753e-9 relative (analytic
    -5.0256097e-3, central difference -5.0256097e-3).
    """
    path, _ = _epanet(tmp_path, "Net1.inp", duration=0)
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    closure = model.tank_closure
    final = water_steady(model, state, drivers)
    inflow = tank_inflow(model, final).detach()

    closure.area.requires_grad_(True)
    closure.advance(state["water.tank_level"], inflow, 3600.0).sum().backward()
    analytic = float(closure.area.grad.detach().sum())

    eps = 1e-4
    base_area = closure.area.detach().clone()
    with torch.no_grad():
        closure.area.copy_(base_area + eps)
        up = float(closure.advance(state["water.tank_level"], inflow, 3600.0).sum())
        closure.area.copy_(base_area - eps)
        down = float(closure.advance(state["water.tank_level"], inflow, 3600.0).sum())
        closure.area.copy_(base_area)
    fd = (up - down) / (2 * eps)
    assert abs(analytic - fd) <= 1e-5 * abs(fd), (analytic, fd)


def test_d7_gradient_reaches_a_learnable_roughness(tmp_path):
    path, _ = _epanet(tmp_path, "twoloop_si.inp")
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    element = model.potential["water"]._elements[0]
    element.roughness.requires_grad_(True)
    final = water_steady(model, state, drivers, differentiable=True)
    layer = model.potential["water"]
    final["water.phi"][..., layer.interior].sum().backward()
    assert element.roughness.grad is not None
    assert torch.isfinite(element.roughness.grad).all()
    assert float(element.roughness.grad.abs().max()) > 0.0


# --------------------------------------------------------------------------- D8
def test_d8_trace_quality_on_the_two_loop(tmp_path):
    """Row D8, 1e-3 on the steady trace fraction. MEASURED: 6.438e-12 percentage points
    (noise many orders below the 1e-3 tolerance, floating on the reference's own float32 output and
    this solve's Newton tolerance rather than on any physics this row could regress).

    A SMOKE row: the fixture has a single source, so mass balance
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


def test_d8_without_the_demand_removal_the_system_is_singular(tmp_path):
    """The measurement behind the demand-as-removal formulation (`build_model`'s docstring),
    pinned so a future edit cannot undo it."""
    path, _ = _epanet(tmp_path, "twoloop_trace.inp")
    net = read_epanet_inp(path)
    model, _, _ = build_model(net, quality=0.0)
    layer = model.transport["quality"]
    assert layer.removal is not None
    assert float(layer.removal.abs().min()) > 0.0
    # the removal rate IS the demand divided by the junction's own capacity
    expected = torch.tensor([j.demand for j in net.junctions], dtype=F64) / layer.capacity
    assert torch.allclose(layer.removal.squeeze(-1), expected, rtol=1e-14)


def test_d8_a_junction_capacity_is_half_of_every_incident_pipe(tmp_path):
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
    """Row G2, 1e-10, against `tests/golden/water_twoloop.json`."""
    from tests.golden import load_golden

    golden = load_golden("water_twoloop")
    model, state, drivers = build_model(twoloop())
    final = water_steady(model, state, drivers)
    for key, expected in golden.items():
        assert key in final, key
        assert final[key].flatten().tolist() == pytest.approx(
            expected, rel=1e-10, abs=1e-18
        )
