"""Source waveforms: unit values (Modelica Standard Library semantics) and breakpoints."""

import math

import pytest

from noodl.apps.electrical import Constant, Pulse, Ramp, Signal, Sine, Step


def test_constant_is_one_and_time_independent():
    w = Constant()
    assert w.unit(-5.0) == 1.0 and w.unit(3.0) == 1.0
    assert not w.time_dependent
    assert w.breakpoints(0.0, 10.0) == []


def test_step_switches_on_at_start_time():
    w = Step(start_time=1.0)
    assert w.time_dependent
    assert w.unit(0.999) == 0.0
    assert w.unit(1.0) == 1.0          # MSL: `if time < startTime then 0 else height`
    assert w.unit(5.0) == 1.0
    assert w.breakpoints(0.0, 2.0) == [1.0]
    assert w.breakpoints(0.0, 1.0) == [1.0]     # (t0, t1] includes t1
    assert w.breakpoints(1.0, 2.0) == []        # ... and excludes t0


def test_ramp_rises_linearly_over_its_duration():
    w = Ramp(2.0, start_time=1.0)
    assert w.unit(1.0) == 0.0
    assert w.unit(0.5) == 0.0
    assert w.unit(2.0) == 0.5
    assert w.unit(3.0) == 1.0
    assert w.unit(10.0) == 1.0
    assert w.breakpoints(0.0, 10.0) == [1.0, 3.0]
    assert w.breakpoints(1.5, 10.0) == [3.0]


def test_ramp_of_zero_duration_is_a_step():
    w = Ramp(0.0, start_time=1.0)
    assert w.unit(0.99) == 0.0 and w.unit(1.0) == 1.0
    assert w.breakpoints(0.0, 2.0) == [1.0]


def test_sine_is_zero_before_start_then_shifted():
    w = Sine(50.0, phase=0.3, start_time=0.01)
    assert w.unit(0.005) == 0.0
    assert w.unit(0.01) == pytest.approx(math.sin(0.3), abs=1e-15)
    t = 0.0137
    assert w.unit(t) == pytest.approx(math.sin(2 * math.pi * 50.0 * (t - 0.01) + 0.3),
                                      abs=1e-15)
    assert w.breakpoints(0.0, 1.0) == [0.01]
    assert w.breakpoints(0.01, 1.0) == []


def test_pulse_on_for_width_percent_of_each_period():
    w = Pulse(25.0, 2.0, start_time=1.0)
    assert w.unit(0.5) == 0.0
    assert w.unit(1.0) == 1.0           # rising edge belongs to the pulse
    assert w.unit(1.49) == 1.0
    assert w.unit(1.5) == 0.0           # falling edge: `time < T_start + T_width`
    assert w.unit(2.9) == 0.0
    assert w.unit(3.0) == 1.0
    assert w.unit(3.2) == 1.0
    assert w.breakpoints(0.0, 5.0) == [1.0, 1.5, 3.0, 3.5, 5.0]
    assert w.breakpoints(1.0, 3.0) == [1.5, 3.0]


def test_pulse_stops_after_n_periods():
    w = Pulse(50.0, 1.0, n_periods=2)
    assert w.unit(0.2) == 1.0 and w.unit(1.2) == 1.0
    assert w.unit(2.2) == 0.0 and w.unit(7.2) == 0.0
    assert w.breakpoints(-1.0, 10.0) == [0.0, 0.5, 1.0, 1.5]
    assert Pulse(50.0, 1.0, n_periods=0).unit(0.2) == 0.0
    assert Pulse(50.0, 1.0, n_periods=0).breakpoints(-1.0, 10.0) == []


def test_pulse_switches_where_the_float_period_grid_lands():
    # 0.1 * 3 rounds above 0.3: MSL fires the period event at startTime + 3*period, so
    # the pulse is still off at t = 0.3 and comes on at the event time itself.
    w = Pulse(50.0, 0.1)
    assert 3 * 0.1 > 0.3
    assert w.unit(0.3) == 0.0
    assert w.unit(3 * 0.1) == 1.0
    assert 3 * 0.1 in w.breakpoints(0.25, 0.35)


def test_full_width_pulse_has_no_inner_edges():
    w = Pulse(100.0, 1.0, n_periods=3)
    assert w.unit(0.0) == 1.0 and w.unit(1.0) == 1.0 and w.unit(2.99) == 1.0
    assert w.unit(3.0) == 0.0
    assert w.breakpoints(-1.0, 10.0) == [0.0, 3.0]


def test_signal_is_read_from_drivers():
    w = Signal("u", index=2)
    assert (w.key, w.index) == ("u", 2)
    assert not w.time_dependent
    assert w.breakpoints(0.0, 1.0) == []
    with pytest.raises(TypeError, match="drivers"):
        w.unit(0.0)


@pytest.mark.parametrize("make, match", [
    (lambda: Ramp(-1.0), "duration"),
    (lambda: Sine(-1.0), "frequency"),
    (lambda: Pulse(0.0, 1.0), "width"),
    (lambda: Pulse(120.0, 1.0), "width"),
    (lambda: Pulse(50.0, 0.0), "period"),
])
def test_invalid_parameters_are_refused(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_waveforms_are_frozen():
    with pytest.raises(AttributeError):
        Step().start_time = 2.0


def test_pulse_period_count_follows_the_event_times_not_the_quotient():
    # 1.7 / 0.1 floors to 17 although 17 * 0.1 > 1.7 (period 17 has not begun: off);
    # 4.3 / 0.1 floors to 42 although 43 * 0.1 == 4.3 (period 43 has begun: on)
    w = Pulse(50.0, 0.1)
    assert math.floor(1.7 / 0.1) == 17 and 17 * 0.1 > 1.7
    assert w.unit(1.7) == 0.0
    assert math.floor(4.3 / 0.1) == 42 and 43 * 0.1 == 4.3
    assert w.unit(4.3) == 1.0
