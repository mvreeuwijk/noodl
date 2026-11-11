"""`water_run`: EPANET 2.2's extended-period run (`hydraul.c`, `runhyd`, `nexthyd`,
`timestep`) -- report times, the step rule, patterns and batches."""

import dataclasses
from pathlib import Path

import pytest
import torch

from noodl.apps.water import simulate
from noodl.apps.water.controls import CLOSED, OPEN, Action, Control, Premise, Rule
from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import (
    Junction,
    Pump,
    Reservoir,
    Tank,
    WaterNetwork,
    WaterPipe,
    build_model,
    initial_drivers,
)
from noodl.apps.water.simulate import WaterResults, pattern_factor, water_run
from tests.apps.water import ZERO_FLOW

F64 = torch.float64
NET1 = Path(__file__).resolve().parents[2] / "data" / "water" / "Net1.inp"


def _net1(**changes):
    return dataclasses.replace(read_epanet_inp(NET1), **changes)


def test_pattern_factor_follows_pattern_start_and_step():
    """EPANET's pattern period `p = (t + Pstart) / Pstep` (integer seconds, `hydraul.c`,
    `demands`), the multiplier `F[p mod length]`; no pattern is 1."""
    net = WaterNetwork(junctions=(Junction("J", 0.0),), reservoirs=(Reservoir("R", 1.0),),
                       pipes=(WaterPipe("P", "R", "J", 1.0, 0.1, 100.0),),
                       patterns={"A": (1.0, 2.0, 3.0)}, pattern_timestep=3600.0,
                       pattern_start=1800.0)
    assert [pattern_factor(net, "A", t) for t in (0, 1799, 1800, 5399, 5400, 9000, 12600)] \
        == [1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 2.0]
    assert pattern_factor(net, None, 5400.0) == 1.0


def test_report_times_follow_report_start_and_step():
    """EPANET reports at REPORT START + k REPORT TIMESTEP up to the duration (`quality.c`,
    `runqual`: `Htime >= Rtime`, `Rtime += Rstep`). A two-hour report step of Net1 keeps its
    one-hour hydraulic steps, but the hydraulic report grid (`Rtime`, every REPORT TIMESTEP
    from 0) no longer ends the step after a control at an odd hour, so the values are the
    hourly run's at the even hours to the explicit step's own error. MEASURED: equal to
    22 h, 3.6e-5 m at 24 h (the pump's 16.3 h switch). EPANET itself is compared in
    `test_water_parity.py::test_report_start_and_step_match_epanet`."""
    hourly = water_run(*build_model(_net1())[:1])
    assert isinstance(hourly, WaterResults)
    torch.testing.assert_close(hourly.time, torch.arange(0, 86401, 3600, dtype=F64))
    model, _, _ = build_model(_net1(report_start=7200.0, report_timestep=7200.0))
    sparse = water_run(model)
    torch.testing.assert_close(sparse.time, torch.arange(7200, 86401, 7200, dtype=F64))
    rows = [int(t // 3600) for t in sparse.time.tolist()]
    torch.testing.assert_close(sparse.tank_level[:-1], hourly.tank_level[rows[:-1]], rtol=0,
                               atol=0)
    torch.testing.assert_close(sparse.tank_level, hourly.tank_level[rows], rtol=0, atol=1e-4)
    torch.testing.assert_close(sparse.head, hourly.head[rows], rtol=0, atol=1e-4)
    assert sparse.head.shape == (12, len(model.node_names))
    assert sparse.flow.shape == sparse.status.shape == (12, len(model.link_names))
    assert sparse.node_names == model.node_names and sparse.link_names == model.link_names
    assert sparse.energy.pump_names == ["9"]
    # a duration given to the run overrides the network's
    short = water_run(build_model(_net1())[0], duration=7200.0)
    torch.testing.assert_close(short.time, torch.tensor([0.0, 3600.0, 7200.0], dtype=F64))
    torch.testing.assert_close(short.tank_level, hourly.tank_level[:3], rtol=0, atol=0)


def _patterned_network(pattern_start=0.0, control=1800.0, report=7200.0):
    """A reservoir feeding a tank and a junction whose demand pattern changes hourly; an
    `AT TIME` control at `control` s shortens the first step."""
    return WaterNetwork(
        junctions=(Junction("J", 0.0, 0.01, "D"),),
        reservoirs=(Reservoir("R", 50.0),),
        tanks=(Tank("T", 30.0, 5.0, 0.0, 20.0, 10.0),),
        pipes=(WaterPipe("P1", "R", "J", 500.0, 0.3, 100.0),
               WaterPipe("P2", "J", "T", 500.0, 0.3, 100.0),
               WaterPipe("P3", "R", "T", 500.0, 0.2, 100.0)),
        patterns={"D": (1.0, 3.0, 0.5, 2.0)},
        controls=(Control("P3", CLOSED, None, "time", value=control),),
        hydraulic_timestep=3600.0, pattern_timestep=3600.0, report_timestep=report,
        pattern_start=pattern_start, duration=4 * 3600.0)


def _demand_factors(out):
    return [round(float(d) / 0.01, 12) for d in out.demand[:, 0]]


def test_a_pattern_boundary_inside_a_hydraulic_step_ends_the_step():
    """EPANET's `timestep` (`hydraul.c`): the step ends at the next pattern period
    `n Pstep - Htime`, `n = (Htime + Pstart) / Pstep + 1`. After the control's 0:30 step
    that is 1:00: without the rule the step would run to 1:30 (the hydraulic step) and the
    demand change at 1:00 would be taken half an hour late."""
    model, _, _ = build_model(_patterned_network())
    diagnostics: dict = {}
    out = water_run(model, diagnostics=diagnostics)
    assert diagnostics["times"] == [0.0, 1800.0, 3600.0, 7200.0, 10800.0, 14400.0]
    assert out.time.tolist() == [0.0, 7200.0, 14400.0]
    assert _demand_factors(out) == [1.0, 0.5, 1.0]


def test_pattern_start_shifts_the_pattern_as_epanet_does():
    """PATTERN START 0:30 makes the period at time t `(t + 1800) / 3600`, so it changes at
    0:30, 1:30, ... (`demands`). EPANET's step rule looks for `n Pstep - Htime`, not
    `n Pstep - Pstart - Htime` (`timestep`), so from a 0:15 control it steps to 1:00, not to
    the 0:30 period boundary, and takes the new period on only there. Transcribed as is."""
    model, _, _ = build_model(_patterned_network(pattern_start=1800.0, control=900.0))
    diagnostics: dict = {}
    water_run(model, diagnostics=diagnostics)
    assert diagnostics["times"] == [0.0, 900.0, 3600.0, 7200.0, 10800.0, 14400.0]
    model, _, _ = build_model(_patterned_network(pattern_start=1800.0, report=1800.0))
    out = water_run(model)
    assert _demand_factors(out) == [1.0, 3.0, 3.0, 0.5, 0.5, 2.0, 2.0, 1.0, 1.0]


def test_a_speed_pattern_drives_the_pump_and_zero_stops_it():
    """A pump's speed pattern sets its speed at every step (`hydraul.c`, `demands`,
    `setlinksetting`): 0 closes it, a positive speed opens it again."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.005),),
        reservoirs=(Reservoir("R", 0.0),),
        tanks=(Tank("T", 20.0, 5.0, 0.0, 20.0, 15.0),),
        pipes=(WaterPipe("P1", "J", "T", 300.0, 0.3, 100.0),),
        pumps=(Pump("PU", "R", "J", "C", speed_pattern="S"),),
        curves={"C": ((0.02, 40.0),)},
        patterns={"S": (1.0, 0.0, 0.8, 1.2)},
        duration=3 * 3600.0)
    model, _, _ = build_model(net)
    out = water_run(model)
    pump = model.link_names.index("PU")
    assert out.status[:, pump].tolist() == [OPEN, CLOSED, OPEN, OPEN]
    torch.testing.assert_close(out.setting[:, pump], torch.tensor([1.0, 0.0, 0.8, 1.2],
                                                                  dtype=F64))
    assert abs(float(out.flow[1, pump])) <= ZERO_FLOW
    assert float(out.flow[3, pump]) > float(out.flow[2, pump]) > 0.0


def test_a_reservoir_head_pattern_moves_the_head():
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.005),),
        reservoirs=(Reservoir("R", 50.0, "H"),),
        pipes=(WaterPipe("P1", "R", "J", 300.0, 0.3, 100.0),),
        patterns={"H": (1.0, 0.9, 1.1)}, duration=2 * 3600.0)
    model, _, _ = build_model(net)
    out = water_run(model)
    r = model.node_names.index("R")
    torch.testing.assert_close(out.head[:, r], torch.tensor([50.0, 45.0, 55.0], dtype=F64))
    j = model.node_names.index("J")
    torch.testing.assert_close(out.head[:, j] - out.head[:, r],
                               out.head[0, j] - out.head[0, r] + torch.zeros(3, dtype=F64))
    torch.testing.assert_close(out.pressure[:, r], torch.tensor([0.0, -5.0, 5.0], dtype=F64))


def _net1_controls_by_multiplier(multipliers):
    """Net1 and drivers whose demands are `multipliers` times the file's (one row each)."""
    model, _, drivers = build_model(read_epanet_inp(NET1))
    base = drivers["water.sources"]
    sources = torch.stack([base * m for m in multipliers])
    return model, {**drivers, "water.sources": sources.squeeze(0)}


def _forced(monkeypatch, times):
    """`water_run` with its steps also ended at every one of `times` (s)."""
    step = simulate._Clock.step

    def forced(self, t, report_time):
        out = step(self, t, report_time)
        ahead = [round(b) - t for b in times if round(b) > t]
        return min([out, *ahead])

    monkeypatch.setattr(simulate._Clock, "step", forced)


def test_batched_run_equals_single_runs(monkeypatch):
    """Review Focus 5: two instances whose pump switches at different times share one step,
    the shortest either needs (spec, "Batches"). Each instance equals its own single run
    except for the other's extra step boundaries: run alone with the batch's step
    boundaries forced on it, it is the batch's instance to rounding, so no shared step
    crosses its own event (a crossing ends the step for the batch; the rounding to whole
    seconds is the instance's own).

    Against the PLAIN single runs the levels differ by more than the 2e-4 m the plan asked:
    an extra boundary is an extra solve, and EPANET's explicit step then carries the tank
    on that solve's inflow. MEASURED: instance 1.25 re-solved at 26069 s (instance 0.8's
    pump switch) is 1.7e-3 m off its single run an hour later, and the two runs' switch
    times then move apart by seconds (63970 s single, 63973 s batched), 6.5e-3 m at worst.
    That is the spec's "differs ... only by that extra step boundary", which the forced runs
    measure exactly."""
    multipliers = (0.8, 1.25)
    singles, times = [], []
    for m in multipliers:
        model, drivers = _net1_controls_by_multiplier([m])
        diagnostics: dict = {}
        singles.append(water_run(model, drivers=drivers, diagnostics=diagnostics))
        times.append(diagnostics["times"])
    assert times[0] != times[1]                      # the pumps switch at different times
    model, drivers = _net1_controls_by_multiplier(multipliers)
    diagnostics = {}
    batch = water_run(model, drivers=drivers, diagnostics=diagnostics)
    shared = diagnostics["times"]
    assert batch.tank_level.shape == (2, 25, 1) and batch.head.shape[:2] == (2, 25)
    assert len(shared) > max(len(t) for t in times)
    for k, single in enumerate(singles):
        assert batch.status[k].tolist() == single.status.tolist()
        worst = float((batch.tank_level[k] - single.tank_level).abs().max())
        assert worst < 1e-2, (k, worst)
    _forced(monkeypatch, shared)
    for k, m in enumerate(multipliers):
        model, drivers = _net1_controls_by_multiplier([m])
        diagnostics = {}
        alone = water_run(model, drivers=drivers, diagnostics=diagnostics)
        assert diagnostics["times"] == shared
        torch.testing.assert_close(alone.tank_level, batch.tank_level[k], rtol=0, atol=1e-9)
        torch.testing.assert_close(alone.flow, batch.flow[k], rtol=0, atol=1e-9)


def _ruled_tank():
    """A pump filling a tank under two rules: STOP above 6 m, FILL below 3 m."""
    return WaterNetwork(
        junctions=(Junction("J", 10.0, 0.004),),
        reservoirs=(Reservoir("R", 5.0),),
        tanks=(Tank("T", 40.0, 4.0, 1.0, 9.0, 8.0),),
        pipes=(WaterPipe("P1", "J", "T", 400.0, 0.25, 100.0),),
        pumps=(Pump("PU", "R", "J", "C"),),
        curves={"C": ((0.025, 65.0),)},
        rules=(Rule("STOP", (Premise("IF", "NODE", "T", "LEVEL", ">", 6.0),),
                    (Action("PU", "STATUS", CLOSED),), ()),
               Rule("FILL", (Premise("IF", "NODE", "T", "LEVEL", "<", 3.0),),
                    (Action("PU", "STATUS", OPEN),), ())),
        duration=12 * 3600.0)


def test_a_rule_acting_in_one_instance_ends_the_batchs_step(monkeypatch):
    """`ruletimestep` with a batch: a rule that acts in one instance ends the shared step
    for all (the other's rules then find nothing to do at that time), and each instance is
    its single run forced onto the batch's step boundaries, to rounding. Run alone, STOP
    acts at 3240 s at 0.8 times the demand (inside the first hydraulic step), and at 1.6 at
    3600 s and again at 33840 s after FILL. In the batch instance 1.6 takes the 3240 s
    boundary too, an extra solve of EPANET's explicit step, so its later rules act at other
    rule steps (MEASURED: 28440 s and 33480 s), which its forced single run reproduces."""
    model, _, drivers = build_model(_ruled_tank())
    base = drivers["water.sources"]
    multipliers = (0.8, 1.6)
    times = []
    for m in multipliers:
        diagnostics: dict = {}
        water_run(model, drivers={**drivers, "water.sources": base * m},
                  diagnostics=diagnostics)
        times.append(diagnostics["times"])
    assert 3240.0 in times[0] and 3240.0 not in times[1]
    assert 33840.0 in times[1] and 33840.0 not in times[0]
    diagnostics = {}
    batch = water_run(model, drivers={**drivers, "water.sources": torch.stack(
        [base * m for m in multipliers])}, diagnostics=diagnostics)
    shared = diagnostics["times"]
    # instance 0.8's STOP at 3240 s ends the step for both; instance 1.6's FILL and STOP,
    # moved by that extra boundary, end it at 28440 s and 33480 s
    assert {3240.0, 28440.0, 33480.0} <= set(shared)
    _forced(monkeypatch, shared)
    for k, m in enumerate(multipliers):
        diagnostics = {}
        alone = water_run(model, drivers={**drivers, "water.sources": base * m},
                          diagnostics=diagnostics)
        assert diagnostics["times"] == shared
        torch.testing.assert_close(alone.tank_level, batch.tank_level[k], rtol=0, atol=1e-9)
        assert alone.status.tolist() == batch.status[k].tolist()


def test_the_run_owns_the_time_driver():
    """`water_run` sets `"water.time"` itself; a caller's value is not read."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.002),),
        reservoirs=(Reservoir("R", 50.0),),
        tanks=(Tank("T", 40.0, 5.0, 0.0, 20.0, 10.0),),
        pipes=(WaterPipe("P1", "R", "J", 500.0, 0.3, 130.0),
               WaterPipe("P2", "J", "T", 100.0, 0.3, 130.0)),
        controls=(Control("P2", CLOSED, None, "time", value=600.0),),
        hydraulic_timestep=3600.0, report_timestep=600.0, pattern_timestep=3600.0,
        duration=1800.0)
    model, _, drivers = build_model(net)
    out = water_run(model, drivers={**drivers, "water.time": torch.tensor(5000.0)})
    p2 = model.link_names.index("P2")
    assert out.time.tolist() == [0.0, 600.0, 1200.0, 1800.0]
    assert float(out.flow[0, p2]) > 0.0
    assert out.status[1:, p2].tolist() == [CLOSED] * 3


def test_the_implicit_tank_schemes_are_refused_by_name():
    model, _, _ = build_model(read_epanet_inp(NET1), tank_scheme="backward_euler")
    with pytest.raises(ValueError, match=r"water_run.*tank_scheme='forward_euler'"):
        water_run(model)


def test_pressure_driven_demand_follows_its_pattern():
    """On a pressure-driven network the pattern scales the required demand
    (`"water.demand_factor"`), and the delivered demand follows."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0, 0.01, "D"),),
        reservoirs=(Reservoir("R", 30.0),),
        pipes=(WaterPipe("P1", "R", "J", 300.0, 0.3, 100.0),),
        patterns={"D": (1.0, 2.0)}, duration=3600.0)
    model, _, _ = build_model(net, pda=True, p_min=0.0, p_req=10.0)
    out = water_run(model)
    torch.testing.assert_close(out.demand[:, 0], torch.tensor([0.01, 0.02], dtype=F64))
    drivers = initial_drivers(model)
    assert "water.sources" in drivers


def _closure_net():
    return WaterNetwork(
        junctions=(Junction("J", 0.0, 0.002),),
        reservoirs=(Reservoir("R", 50.0),),
        tanks=(Tank("T", 40.0, 5.0, 0.0, 20.0, 10.0),),
        pipes=(WaterPipe("P1", "R", "J", 500.0, 0.3, 130.0),
               WaterPipe("P2", "J", "T", 100.0, 0.3, 130.0)),
        controls=(Control("P2", CLOSED, None, "time", value=600.0),))


def test_the_control_closure_follows_every_driver_it_reads():
    """The closure remembers its last answer while its inputs are the same objects at the same
    versions; the reservoir head (`"water.reservoir_head"`) is one of the inputs, a number
    for the time is read by value, and the answer keeps no graph of an input alive."""
    import gc
    import weakref

    model, state, drivers = build_model(_closure_net())
    closure = model.control_closure
    heads = drivers.get("water.reservoir_head", closure.reservoir_heads)
    first = closure(state, {**drivers, "water.reservoir_head": heads})
    second = closure(state, {**drivers, "water.reservoir_head": heads * 1.5})
    assert float(second["water.phi_boundary"][0]) == pytest.approx(
        1.5 * float(first["water.phi_boundary"][0]), rel=1e-12)
    # the time as a plain number: before 600 s the control has not fired, after it has
    early = closure(state, {**drivers, "water.time": 5.0})
    late = closure(state, {**drivers, "water.time": 600.0})
    assert float(early["water.link_status"][1]) == OPEN
    assert float(late["water.link_status"][1]) == CLOSED
    assert closure(state, {**drivers, "water.time": 600.0})["water.link_status"].tolist() \
        == late["water.link_status"].tolist()
    # an input that requires grad is not kept alive by the remembered answer
    leaf = heads.clone().requires_grad_(True)
    scaled = leaf * 2.0
    ref = weakref.ref(scaled)
    closure(state, {**drivers, "water.reservoir_head": scaled})
    del scaled
    gc.collect()
    assert ref() is None


TWOLOOP = NET1.with_name("twoloop_si.inp")
TWOLOOP_TRACE = NET1.with_name("twoloop_trace.inp")


def test_quality_runs_by_default_when_the_network_defines_it():
    """`quality=None` runs EPANET's quality when `[OPTIONS] QUALITY` names a mode and the
    duration is positive (`adjustdata`: `DURATION 0` turns it off)."""
    model, _, _ = build_model(read_epanet_inp(NET1))
    run = water_run(model)
    assert run.node_quality is not None and run.quality_units == "mg/L"
    assert run.node_quality.shape == (len(run.time), len(run.node_names))
    assert run.mass_balance is not None
    model, _, _ = build_model(read_epanet_inp(TWOLOOP))
    run = water_run(model)
    assert run.node_quality is None and run.mass_balance is None
    model, _, _ = build_model(read_epanet_inp(NET1))
    run = water_run(model, duration=0.0)
    assert run.node_quality is None and run.quality_units is None


def test_quality_false_turns_it_off_and_true_demands_it():
    model, _, _ = build_model(read_epanet_inp(NET1))
    assert water_run(model, quality=False).node_quality is None
    assert water_run(model, quality=True).node_quality is not None
    with pytest.raises(ValueError, match="DURATION"):
        water_run(model, duration=0.0, quality=True)
    plain, _, _ = build_model(read_epanet_inp(TWOLOOP))
    with pytest.raises(ValueError, match="QUALITY"):
        water_run(plain, quality=True)


def test_the_steady_tracer_is_refused_alongside_epanet_quality():
    """The steady layer "quality" of `build_model(quality=float)` is not EPANET's quality: a
    network whose quality resolves on is refused by name; `quality=False` runs."""
    model, _, _ = build_model(read_epanet_inp(TWOLOOP_TRACE), quality=0.0)
    with pytest.raises(ValueError, match="steady tracer"):
        water_run(model)
    assert water_run(model, quality=False).node_quality is None


def test_quality_leaves_the_hydraulics_unchanged():
    """The quality never feeds back (Net1, 24 h): heads, flows, tank levels, statuses and the
    energy are bit-identical with it on and off."""
    model, _, _ = build_model(read_epanet_inp(NET1))
    on, off = water_run(model, quality=True), water_run(model, quality=False)
    assert on.node_quality is not None and off.node_quality is None
    for name in ("head", "flow", "tank_level", "status", "setting", "demand", "pressure"):
        assert torch.equal(getattr(on, name), getattr(off, name)), name
    for field in dataclasses.fields(on.energy):
        mine, theirs = getattr(on.energy, field.name), getattr(off.energy, field.name)
        if isinstance(mine, torch.Tensor):
            assert torch.equal(mine, theirs), field.name
        else:
            assert mine == theirs, field.name
