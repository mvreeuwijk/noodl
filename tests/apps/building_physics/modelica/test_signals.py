"""`signals.evaluate` against closed forms written here from the MSL v4.1.0 source.

Each expected value below is transcribed directly from `Modelica/Blocks/Sources.mo`
(Constant :164, Step :204, Ramp :244, Sine :291, Pulse :773, TimeTable :1378,
CombiTimeTable :1588) and, for `CombiTimeTable`, the C evaluation in
`Modelica/Resources/C-Sources/ModelicaStandardTables.c`
(`ModelicaStandardTables_CombiTimeTable_getValue`, :921-1190), evaluated by hand at chosen
times -- never by calling the implementation under test. Times always include one before
`startTime` and one after the end of the signal's own definition range.

Event-time convention (controller ruling, Task 9): a time ON an event (a start time, a
table knot, a period boundary) takes the value BEFORE the event, the left limit, because
that is what OpenModelica 1.27.1 records at an output time that coincides with a time event:
the solver writes the output point before it handles the event. The reference CSVs show it
(`Validation/DoorOpenClosed`: the `Step` with `startTime = 0.5` is still 0 in the `t = 0.5`
row; `Validation/OpenDoorPressure`: the `TimeTable` knot at 3600 s keeps the pre-knot
pressure at `t = 3600`), even for `Step`, whose MSL equation alone would give `height` at the
instant. The initial instant has no left limit: at `t = 0` (and at `evaluate`'s `t_start`)
the relations are evaluated as written.
"""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.building_physics.modelica import signals
from noodl.apps.building_physics.modelica.schema import ModelicaImportError, Signal

F64 = torch.float64


def _sig(cls: str, **parameters) -> Signal:
    return Signal(name="s", cls=f"Modelica.Blocks.Sources.{cls}", parameters=parameters,
                  drives=("x.y",))


def _eval(sig: Signal, times) -> list[float]:
    t = torch.tensor(times, dtype=F64)
    y = signals.evaluate(sig, t)
    assert y.dtype == F64
    assert y.shape == t.shape
    return y.tolist()


def test_constant():
    assert _eval(_sig("Constant", k=3.5), [-1.0, 0.0, 10.0]) == [3.5, 3.5, 3.5]


def test_step_defaults_and_offset():
    # y = offset + (if time < startTime then 0 else height); at t = startTime the left
    # limit (module docstring), except at the initial instant t = 0.
    assert _eval(_sig("Step"), [-1.0, 0.0, 5.0]) == [0.0, 1.0, 1.0]
    got = _eval(_sig("Step", height=2.0, offset=-1.0, startTime=10.0),
                [0.0, 9.999, 10.0, 10.001, 99.0])
    assert got == [-1.0, -1.0, -1.0, 1.0, 1.0]


def test_ramp():
    # y = offset + (0 if t < startTime else (t - startTime) height/duration if t <
    #     startTime + duration else height)
    sig = _sig("Ramp", height=100.0, duration=500.0, offset=-50.0, startTime=100.0)
    times = [0.0, 100.0, 350.0, 599.0, 600.0, 1000.0]
    expected = [-50.0, -50.0, -50.0 + 250.0 * 100.0 / 500.0, -50.0 + 499.0 * 100.0 / 500.0,
                50.0, 50.0]
    assert _eval(sig, times) == pytest.approx(expected, rel=1e-15, abs=1e-12)


def test_ramp_zero_duration_is_a_step():
    sig = _sig("Ramp", height=2.0, duration=0.0, startTime=1.0)
    assert _eval(sig, [0.5, 1.0, 1.5, 2.0]) == [0.0, 0.0, 2.0, 2.0]  # left limit at 1.0


def test_sine_discontinuous_and_continuous():
    amp, f, ph, off, t0 = 2.0, 0.01, 0.3, 1.0, 10.0
    times = [0.0, 10.0, 35.0, 1000.0]
    # Discontinuous: at t = startTime the left limit, offset alone.
    expected = [off, off, off + amp * math.sin(2 * math.pi * f * 25 + ph),
                off + amp * math.sin(2 * math.pi * f * 990 + ph)]
    sig = _sig("Sine", amplitude=amp, f=f, phase=ph, offset=off, startTime=t0)
    assert _eval(sig, times) == pytest.approx(expected, rel=1e-14)
    cont = _sig("Sine", amplitude=amp, f=f, phase=ph, offset=off, startTime=t0, continuous=True)
    expected[0] = expected[1] = off + amp * math.sin(ph)
    assert _eval(cont, times) == pytest.approx(expected, rel=1e-14)


def test_pulse():
    # period 10, width 30 % -> high for 3 s from each period start; startTime 5, 2 periods.
    sig = _sig("Pulse", amplitude=4.0, width=30.0, period=10.0, nperiod=2, offset=1.0,
               startTime=5.0)
    # On each switching instant (5, 8, 15, 18) the left limit.
    times = [0.0, 5.0, 5.1, 7.9, 8.0, 8.1, 14.9, 15.0, 15.1, 17.5, 18.0, 18.1, 25.0, 26.0]
    expected = [1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0]
    assert _eval(sig, times) == expected
    infinite = _sig("Pulse", amplitude=1.0, width=50.0, period=2.0)
    assert _eval(infinite, [0.0, 0.5, 1.0, 1.5, 1000.0, 1000.2, 1001.0, 1001.5]) == [
        1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    none = _sig("Pulse", amplitude=1.0, period=2.0, nperiod=0)
    assert _eval(none, [0.0, 0.5]) == [0.0, 0.0]


def test_timetable_linear_with_discontinuity_and_extrapolation():
    table = [[0.0, 0.0], [1.0, 1.0], [1.0, 3.0], [3.0, 5.0]]
    sig = _sig("TimeTable", table=table, offset=0.5, startTime=0.0)
    # before 1: 0 -> 1; AT the duplicated knot 1 the left limit (the left segment's end);
    # after it the right segment 3 -> 5; beyond 3 the last two points extrapolate (slope 1).
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    expected = [0.5, 1.0, 1.5, 4.0, 4.5, 5.5, 6.5]
    assert _eval(sig, times) == pytest.approx(expected, rel=1e-15)


def test_timetable_start_and_shift_time():
    table = [[0.0, 10.0], [10.0, 20.0]]
    sig = _sig("TimeTable", table=table, startTime=5.0, shiftTime=5.0, timeScale=2.0)
    # t < startTime: offset (0), and at t = startTime the left limit, 0 too. Else y at
    # (t - shift)/timeScale on the table.
    times = [0.0, 5.0, 7.0, 15.0, 45.0]
    expected = [0.0, 0.0, 11.0, 15.0, 30.0]
    assert _eval(sig, times) == pytest.approx(expected, rel=1e-15)


def test_timetable_single_row():
    sig = _sig("TimeTable", table=[[0.0, 7.0]], offset=1.0)
    assert _eval(sig, [0.0, 50.0]) == [8.0, 8.0]


def test_combitimetable_linear_last_two_points_default():
    table = [[0.0, 1.0], [10.0, 3.0], [20.0, 2.0]]
    sig = _sig("CombiTimeTable", table=table, offset=[0.5])
    times = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    # startTime defaults to 0, so t < 0 gives offset only (the C code returns 0 there).
    expected = [0.5, 1.5, 2.5, 3.5, 3.0, 2.5, 1.5]
    assert _eval(sig, times) == pytest.approx(expected, rel=1e-15)


def test_combitimetable_hold_and_constant_segments():
    table = [[0.0, 1.0], [10.0, 3.0], [20.0, 2.0]]
    hold = _sig("CombiTimeTable", table=table,
                extrapolation="Modelica.Blocks.Types.Extrapolation.HoldLastPoint",
                startTime=-100.0, shiftTime=0.0)
    assert _eval(hold, [-10.0, 5.0, 25.0]) == pytest.approx([1.0, 2.0, 2.0], rel=1e-15)
    const = _sig("CombiTimeTable", table=table, smoothness="ConstantSegments",
                 extrapolation="HoldLastPoint")
    # At the knots 10 and 20 the left limit (the previous segment's value).
    assert _eval(const, [0.0, 9.9, 10.0, 10.1, 19.0, 20.0, 20.1, 50.0]) == [
        1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 2.0, 2.0]
    # ConstantSegments with LastTwoPoints still extrapolates LINEARLY on the last two rows
    # (ModelicaStandardTables.c, the LAST_TWO_POINTS case shared with LINEAR_SEGMENTS).
    const_l2 = _sig("CombiTimeTable", table=table, smoothness="ConstantSegments")
    assert _eval(const_l2, [30.0]) == pytest.approx([1.0], rel=1e-15)


def test_combitimetable_periodic():
    table = [[0.0, 0.0], [2.0, 2.0], [4.0, 0.0]]
    sig = _sig("CombiTimeTable", table=table, extrapolation="Periodic")
    assert _eval(sig, [1.0, 5.0, 7.0, 9.5]) == pytest.approx([1.0, 1.0, 1.0, 1.5], rel=1e-15)


def test_combitimetable_selects_one_column():
    table = [[0.0, 1.0, 10.0], [10.0, 3.0, 30.0]]
    sig = _sig("CombiTimeTable", table=table, columns=[3])
    assert _eval(sig, [5.0]) == pytest.approx([20.0], rel=1e-15)
    both = _sig("CombiTimeTable", table=table)
    with pytest.raises(ModelicaImportError, match="exactly one output column"):
        signals.evaluate(both, torch.zeros(1, dtype=F64))


def test_combitimetable_no_extrapolation_raises_outside():
    table = [[0.0, 1.0], [10.0, 3.0]]
    sig = _sig("CombiTimeTable", table=table, extrapolation="NoExtrapolation")
    assert _eval(sig, [0.0, 10.0]) == pytest.approx([1.0, 3.0])
    with pytest.raises(ModelicaImportError, match="NoExtrapolation"):
        signals.evaluate(sig, torch.tensor([11.0], dtype=F64))


@pytest.mark.parametrize(
    ("cls", "params", "match"),
    [
        ("CombiTimeTable",
         {"table": [[0.0, 1.0], [1.0, 2.0]], "smoothness": "ContinuousDerivative"},
         "smoothness"),
        ("CombiTimeTable", {"tableOnFile": True, "table": [[0.0, 1.0]]}, "tableOnFile"),
        ("Ramp", {"height": 1.0}, "duration"),
        ("Constant", {}, "k"),
        ("TimeTable", {"table": [[1.0, 0.0], [2.0, 1.0]]}, "first point"),
    ],
)
def test_refusals_name_the_signal(cls, params, match):
    with pytest.raises(ModelicaImportError, match=match) as exc:
        signals.evaluate(_sig(cls, **params), torch.zeros(1, dtype=F64))
    assert "s (Modelica.Blocks.Sources." in str(exc.value)


def test_unknown_block_refused():
    sig = Signal(name="pid", cls="Modelica.Blocks.Continuous.LimPID", parameters={},
                 drives=("a.b",))
    with pytest.raises(ModelicaImportError, match="pid"):
        signals.evaluate(sig, torch.zeros(1, dtype=F64))


def test_a_grid_time_one_ulp_after_an_event_takes_the_pre_event_value():
    # An event is judged "on" the grid time with MSL's relative epsilon (TimeEps = 100*eps,
    # Sources.mo:1473-1474): a grid time computed as start + k*interval that lands a rounding
    # error AFTER the event is the event instant, and so takes the left limit (module
    # docstring) as OpenModelica records it.
    after = math.nextafter(2.1, 3.0)
    assert _eval(_sig("Step", startTime=2.1), [after]) == [0.0]
    assert _eval(_sig("Pulse", period=2.1, width=10.0, startTime=0.0), [after]) == [0.0]
    tt = _sig("TimeTable", table=[[0.0, 0.0], [2.1, 0.0], [2.1, 5.0], [3.0, 5.0]])
    assert _eval(tt, [after]) == pytest.approx([0.0], abs=1e-12)
    ct = _sig("CombiTimeTable", table=[[0.0, 0.0], [2.1, 1.0], [3.0, 1.0]],
              smoothness="ConstantSegments")
    assert _eval(ct, [after]) == [0.0]
    # Clearly after the event, the post-event value.
    assert _eval(_sig("Step", startTime=2.1), [2.1001]) == [1.0]


def test_the_initial_instant_takes_the_value_as_written():
    # There is no left limit at the experiment start: Modelica's initialization evaluates
    # `time < startTime` as written, so a Step whose startTime is the start time is already
    # `height` there; the same grid time is a left limit when it is not the start.
    sig = _sig("Step", startTime=2.0)
    t = torch.tensor([2.0, 3.0], dtype=F64)
    assert signals.evaluate(sig, t, t_start=2.0).tolist() == [1.0, 1.0]
    assert signals.evaluate(sig, t, t_start=0.0).tolist() == [0.0, 1.0]
    assert signals.evaluate(sig, t).tolist() == [0.0, 1.0]


def test_no_extrapolation_accepts_the_first_knot_exactly():
    # The left limit at the first knot lies outside the table, but the value AT the knot is
    # defined: the range check uses the time itself, not the nudged event time.
    sig = _sig("CombiTimeTable", table=[[5.0, 1.0], [10.0, 3.0]],
               extrapolation="NoExtrapolation", startTime=0.0)
    assert _eval(sig, [5.0, 10.0]) == pytest.approx([1.0, 3.0])


# ------------------------------------------------------------------ Math blocks
# Closed forms from MSL v4.1.0 `Modelica/Blocks/Math.mo`: Gain :552 `y = k*u`; MultiSum :624
# `y = k*u` (0 for nu = 0); Sum :791 `y = k*u` with `k = ones(nin)` by default; Feedback :832
# `y = u1 - u2`; Add :880 `y = k1*u1 + k2*u2`; Add3 :934; Product :976 `y = u1*u2`; Division
# :1004 `y = u1/u2`.
def _math(cls: str, **parameters) -> Signal:
    return Signal(name="m", cls=f"Modelica.Blocks.Math.{cls}", parameters=parameters,
                  drives=("x.y",))


U1 = torch.tensor([1.0, -2.0, 4.0], dtype=F64)
U2 = torch.tensor([0.5, 3.0, -1.0], dtype=F64)
U3 = torch.tensor([10.0, 20.0, 30.0], dtype=F64)


@pytest.mark.parametrize("cls, params, inputs, expected", [
    ("Gain", {"k": 2.5}, {"u": U1}, [2.5, -5.0, 10.0]),
    ("Add", {}, {"u1": U1, "u2": U2}, [1.5, 1.0, 3.0]),
    ("Add", {"k1": 2.0, "k2": -1.0}, {"u1": U1, "u2": U2}, [1.5, -7.0, 9.0]),
    ("Add3", {"k3": 0.5}, {"u1": U1, "u2": U2, "u3": U3}, [6.5, 11.0, 18.0]),
    ("Sum", {"nin": 2}, {"u[1]": U1, "u[2]": U2}, [1.5, 1.0, 3.0]),
    ("Sum", {"nin": 3, "k": [1.0, -1.0, 0.1]}, {"u[1]": U1, "u[2]": U2, "u[3]": U3},
     [1.5, -3.0, 8.0]),
    ("MultiSum", {"nu": 2, "k": [3.0, 1.0]}, {"u[1]": U1, "u[2]": U2}, [3.5, -3.0, 11.0]),
    ("Product", {}, {"u1": U1, "u2": U2}, [0.5, -6.0, -4.0]),
    ("Feedback", {}, {"u1": U1, "u2": U2}, [0.5, -5.0, 5.0]),
    ("Division", {}, {"u1": U1, "u2": U2}, [2.0, -2.0 / 3.0, -4.0]),
])
def test_math_blocks_match_msl(cls, params, inputs, expected):
    sig = _math(cls, **params)
    assert signals.is_math(sig)
    assert set(signals.math_inputs(sig)) == set(inputs)
    y = signals.combine(sig, inputs)
    assert y.dtype == F64
    assert torch.allclose(y, torch.tensor(expected, dtype=F64), rtol=1e-15, atol=0.0)


def test_math_block_input_names_follow_msl_connectors():
    assert signals.math_inputs(_math("Gain", k=1.0)) == ("u",)
    assert signals.math_inputs(_math("Sum")) == ("u[1]",)  # MISO nin = 1 by default
    assert signals.math_inputs(_math("MultiSum")) == ()  # PartialRealMISO nu = 0
    assert signals.combine(_math("MultiSum"), {}).tolist() == 0.0


def test_math_refusals_name_the_block():
    with pytest.raises(ModelicaImportError, match=r"m \(Modelica.Blocks.Math.Gain\).*'k'"):
        signals.combine(_math("Gain"), {"u": U1})  # MSL declares k with no default
    with pytest.raises(ModelicaImportError, match=r"m \(Modelica.Blocks.Math.Sum\).*k"):
        signals.combine(_math("Sum", nin=2, k=[1.0]), {"u[1]": U1, "u[2]": U2})
    assert not signals.is_math(_math("Abs"))
    with pytest.raises(ModelicaImportError, match=r"m \(Modelica.Blocks.Math.Abs\)"):
        signals.math_inputs(_math("Abs"))
