"""Calendar, CONTAM day/week schedules and time series, as measured on ContamX 3.4.1.7."""

from __future__ import annotations

import pytest
import torch

from noodl.apps.building_physics.schedules import (
    DAY,
    Calendar,
    DaySchedule,
    Signal,
    TimeSeries,
    WeekSchedule,
)

F64 = torch.float64
H = 3600.0


def _tzone():
    """`test_OneZoneWthCtmStack.prj`'s day schedule 1 (rectangular)."""
    return DaySchedule(
        [0.0, 6 * H, 12 * H + 50 * 60, 12 * H + 55 * 60, 13 * H, 24 * H],
        [293.15, 293.15, 293.15, 293.15, 303.15, 303.15],
    )


def test_rectangular_day_schedule_takes_the_left_limit_at_a_jump():
    d = _tzone()
    assert d.at(13 * H, left=True).item() == 293.15  # still old AT the jump
    assert d.at(13 * H, left=False).item() == 303.15
    assert d.at(13 * H + 300.0, left=True).item() == 303.15
    assert d.at(0.0, left=False).item() == 293.15
    assert d.at(DAY, left=True).item() == 303.15


def test_trapezoidal_day_schedule_interpolates_linearly():
    d = DaySchedule(
        [0.0, 12 * H + 55 * 60, 13 * H + 150.0, DAY],
        [293.15, 293.15, 303.15, 303.15],
        shape="trapezoidal",
    )
    # 12:55 -> 13:02:30 ramp; at 13:00 two thirds of the way.
    assert d.at(13 * H, left=True).item() == pytest.approx(293.15 + 10.0 * 2.0 / 3.0, rel=1e-15)
    assert d.knots() == [12 * H + 55 * 60, 13 * H + 150.0]


@pytest.mark.parametrize(
    "times, values, shape, match",
    [
        ([60.0, DAY], [1.0, 1.0], "rectangular", "00:00:00"),
        ([0.0, 3600.0], [1.0, 1.0], "rectangular", "24:00:00"),
        ([0.0, 7200.0, 3600.0, DAY], [1.0] * 4, "rectangular", "increasing"),
        ([0.0, DAY], [1.0], "rectangular", "values"),
        ([0.0, DAY], [1.0, 1.0], "smooth", "shape"),
    ],
)
def test_day_schedule_refuses_bad_definitions(times, values, shape, match):
    with pytest.raises(ValueError, match=match):
        DaySchedule(times, values, shape=shape)


@pytest.mark.parametrize(
    "start, expected",
    [
        (1, [1, 2, 3, 4, 5, 6, 7, 1, 2]),
        (6, [6, 7, 1, 2, 3, 4, 5, 6, 7]),
        (9, [9, 3, 4, 5, 6, 7, 1, 2, 3]),  # measured: a holiday type, then 3
    ],
)
def test_calendar_cycles_day_types_1_to_7_from_the_start_type(start, expected):
    cal = Calendar(start_day=1, start_day_type=start)
    assert [cal.day_type(d) for d in range(1, 10)] == expected


def test_calendar_uses_weather_day_types_and_names_a_missing_day():
    cal = Calendar(start_day=1, day_types={1: 4, 2: 5})
    assert (cal.day_type(1), cal.day_type(2)) == (4, 5)
    with pytest.raises(KeyError, match=r"day 3"):
        cal.day_type(3)


def test_start_time_offsets_the_time_of_day():
    cal = Calendar(start_day=32, start_time=6 * H)
    assert cal.seconds_from_new_year(0.0) == 31 * DAY + 6 * H
    assert cal.day_and_time(0.0, left=False) == (32, 6 * H)
    assert cal.day_and_time(18 * H, left=True) == (32, DAY)  # midnight, left limit
    assert cal.day_and_time(18 * H, left=False) == (33, 0.0)


def test_week_schedule_switches_day_at_midnight_with_the_left_limit():
    a = DaySchedule([0.0, 6 * H, DAY], [303.15, 293.15, 293.15])
    b = DaySchedule([0.0, DAY], [283.15, 283.15])
    days = [1] * 12
    days[4] = 2  # day type 5 -> b
    week = WeekSchedule([a, b], days)
    cal = Calendar(start_day=1, day_types={1: 4, 2: 5})
    assert week.value(0.0, cal).item() == 303.15  # right value at t = 0
    assert week.value(6 * H, cal).item() == 303.15  # left limit at a jump
    assert week.value(6 * H + 300.0, cal).item() == 293.15
    assert week.value(DAY, cal).item() == 293.15  # midnight: the previous day's value
    assert week.value(DAY + 300.0, cal).item() == 283.15
    assert week.breakpoints(0.0, 2 * DAY, cal) == [6 * H, DAY]


def test_a_week_schedule_needs_twelve_day_types_naming_its_day_schedules():
    d = DaySchedule([0.0, DAY], [1.0, 1.0])
    with pytest.raises(ValueError, match="12"):
        WeekSchedule([d], [1] * 7)
    with pytest.raises(ValueError, match="day type 3"):
        WeekSchedule([d], [1, 1, 2] + [1] * 9)


def test_a_run_past_31_december_is_refused():
    cal = Calendar(start_day=365, start_time=23 * H)
    with pytest.raises(ValueError, match="31 December"):
        cal.day_and_time(2 * H, left=True)


def test_time_series_linear_and_step_never_extrapolate():
    lin = TimeSeries([0.0, 100.0, 200.0], [0.0, 1.0, 3.0])
    step = TimeSeries([0.0, 100.0, 200.0], [0.0, 1.0, 3.0], interpolation="step")
    cal = Calendar()
    assert lin.value(150.0, cal).item() == 2.0
    assert step.value(100.0, cal).item() == 0.0  # left limit at a jump
    assert step.value(0.0, cal).item() == 0.0
    assert step.value(150.0, cal).item() == 1.0
    assert lin.breakpoints(0.0, 200.0, cal) == [100.0]
    with pytest.raises(ValueError, match="outside"):
        lin.value(250.0, cal)


def test_time_series_values_carry_gradients_and_trailing_shape():
    v = torch.tensor([[1.0, 2.0], [3.0, 6.0]], dtype=F64, requires_grad=True)
    s = TimeSeries([0.0, 10.0], v)
    out = s.value(5.0, Calendar())
    assert out.shape == (2,)
    out.sum().backward()
    assert torch.equal(v.grad, torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=F64))


def test_every_schedule_is_a_signal():
    d = DaySchedule([0.0, DAY], [1.0, 1.0])
    assert isinstance(WeekSchedule([d]), Signal)
    assert isinstance(TimeSeries([0.0, 1.0], [0.0, 1.0]), Signal)
