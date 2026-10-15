"""Occupants: occupancy schedules and the presence signal (CONTAM sections 21, 22)."""

from __future__ import annotations

import pytest
import torch

from noodl.apps.building_physics.occupants import Occupancy, OccupancyDay, Presence
from noodl.apps.building_physics.schedules import DAY, Calendar
from noodl.topology import Network

F64 = torch.float64


def _net():
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("one", volume=300.0)
    net.add_node("two", volume=600.0)
    net.add_edge("ambient", "one", kind="airpath")
    net.add_edge("one", "two", kind="airpath")
    net.add_edge("two", "ambient", kind="airpath")
    return net


def _moves():
    """One until 01:02:30, two until 02:00, out until 03:00, then one."""
    return OccupancyDay([0.0, 3750.0, 7200.0, 10800.0, DAY], ["one", "two", None, "one", "one"],
                        name="moves")


def test_an_occupancy_day_must_run_from_midnight_to_midnight():
    with pytest.raises(ValueError, match="first time must be 00:00:00"):
        OccupancyDay([60.0, DAY], ["one", "one"])
    with pytest.raises(ValueError, match="last time must be 24:00:00"):
        OccupancyDay([0.0, 3600.0], ["one", "two"])        # ContamX stops on this
    with pytest.raises(ValueError, match="strictly increasing"):
        OccupancyDay([0.0, 600.0, 600.0, DAY], ["one", "two", "one", "one"])
    with pytest.raises(ValueError, match="3 times but 2 zones"):
        OccupancyDay([0.0, 600.0, DAY], ["one", "two"])


def test_the_location_changes_at_the_listed_time():
    day = _moves()
    assert day.zone_at(3750.0, left=False) == "two"
    assert day.zone_at(3750.0, left=True) == "one"
    assert day.zone_at(7300.0, left=False) is None
    assert day.zone_at(0.0, left=True) == "one"
    assert day.zone_at(DAY, left=True) == "one"
    assert day.knots() == [3750.0, 7200.0, 10800.0]
    with pytest.raises(ValueError, match="not a time of day"):
        day.zone_at(DAY + 1.0, left=False)


def test_the_day_type_selects_the_occupancy_day():
    """Measured on ContamX: day 2 of a run starting on day type 1 is type 2, and uses the
    occupancy schedule its second entry names."""
    home = OccupancyDay([0.0, DAY], ["one", "one"])
    away = OccupancyDay([0.0, 43200.0, DAY], [None, "two", "two"])
    occ = Occupancy([home, away], [1, 2] + [1] * 10)
    cal = Calendar(start_day_type=1)
    assert occ.zone_at(1000.0, cal, left=False) == "one"
    assert occ.zone_at(DAY + 1000.0, cal, left=False) is None
    assert occ.zone_at(DAY + 50000.0, cal, left=False) == "two"
    assert occ.zone_at(DAY, cal, left=True) == "one"         # midnight: the previous day
    assert occ.zones == {"one", "two"}
    with pytest.raises(ValueError, match="12 day types"):
        Occupancy([home], [1, 1])
    with pytest.raises(ValueError, match="day type 2 names occupancy day 3"):
        Occupancy([home, away], [1, 3] + [1] * 10)


def test_segments_split_a_step_at_moves_and_at_midnight():
    occ = Occupancy([_moves()])
    cal = Calendar()
    assert occ.segments(3600.0, 3900.0, cal) == [(3600.0, 3750.0, "one"),
                                                 (3750.0, 3900.0, "two")]
    assert occ.segments(7200.0, 7500.0, cal) == [(7200.0, 7500.0, None)]
    across = occ.segments(DAY - 150.0, DAY + 150.0, cal)
    assert across == [(DAY - 150.0, DAY, "one"), (DAY, DAY + 150.0, "one")]
    assert occ.breakpoints(0.0, 2 * DAY, cal)[:4] == [3750.0, 7200.0, 10800.0, DAY]
    with pytest.raises(ValueError, match="t1 > t0"):
        occ.segments(300.0, 300.0, cal)


@pytest.mark.parametrize("calendar", [
    Calendar(start_day=40),
    Calendar(start_time=1234.567),
    Calendar(start_day=200, start_time=1234.567),
], ids=["start_day_40", "start_time_fractional", "both"])
def test_a_move_is_placed_at_its_breakpoint_despite_round_off(calendar):
    """A breakpoint is simulation time = midnight + knot, which need not round-trip exactly
    through the seconds-from-new-year sum: the zone FROM it must still be the new one, and
    the zone up to it the old one."""
    occ = Occupancy([OccupancyDay([0.0, 46800.1, DAY], ["one", "two", "two"])])
    moves = [p for p in occ.breakpoints(0.0, 2 * DAY, calendar)
             if occ.zone_at(p - 1e-3, calendar, left=False) == "one"
             and occ.zone_at(p + 1e-3, calendar, left=False) == "two"]
    assert moves, "no move from one to two found"
    for p in moves:
        assert occ.zone_at(p, calendar, left=False) == "two"
        assert occ.zone_at(p, calendar, left=True) == "one"
        before, after = occ.segments(p - 300.0, p + 300.0, calendar)
        assert (before[2], after[2]) == ("one", "two")


def test_presence_is_one_hot_and_its_step_mean_is_the_time_fraction():
    net = _net()
    p = Presence(Occupancy([_moves()]), net, name="visitor")
    cal = Calendar()
    one, two = net.node_index("one"), net.node_index("two")
    v0 = p.value(0.0, cal)
    assert v0.dtype == F64 and v0.shape == (net.n,) and v0[one].item() == 1.0
    assert p.value(3750.0, cal)[one].item() == 1.0           # left limit at the move
    assert p.value(3900.0, cal)[two].item() == 1.0
    m = p.step_mean(3600.0, 3900.0, cal)
    assert (m[one].item(), m[two].item(), m.sum().item()) == (0.5, 0.5, 1.0)
    assert p.breakpoints(3600.0, 3900.0, cal) == [3750.0]


def test_presence_is_zero_out_of_the_building():
    net = _net()
    p = Presence(Occupancy([_moves()]), net)
    cal = Calendar()
    assert p.value(7500.0, cal).abs().sum().item() == 0.0
    m = p.step_mean(7050.0, 7350.0, cal)                     # half in, half out
    assert m[net.node_index("two")].item() == 0.5 and m.sum().item() == 0.5


def test_a_presence_on_an_unknown_node_is_refused():
    day = OccupancyDay([0.0, DAY], ["kitchen", "kitchen"])
    with pytest.raises(KeyError, match=r"'visitor'.*'kitchen'"):
        Presence(Occupancy([day]), _net(), name="visitor")
