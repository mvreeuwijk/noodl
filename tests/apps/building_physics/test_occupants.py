"""Occupants: occupancy schedules and the presence signal (CONTAM sections 21, 22)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.inputs import Input
from noodl.apps.building_physics.occupants import (
    Occupancy,
    OccupancyDay,
    Occupant,
    OccupantGeneration,
    Presence,
    occupant_inputs,
)
from noodl.apps.building_physics.prj import project_to_model, read_prj
from noodl.apps.building_physics.schedules import DAY, Calendar, TimeSeries
from noodl.apps.building_physics.simulation import simulate
from noodl.apps.building_physics.sources import ConstantSource, SpeciesSources
from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64
DATA = Path(__file__).resolve().parents[2] / "data" / "contam"
THREE = DATA / "valThreeZonesWthCtm-UseApi.prj"
TIGHT = dict(atol=1e-14, rtol=1e-14)


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


@pytest.mark.parametrize(("calendar", "raw_side"), [
    (Calendar(start_day=40), "above"),
    (Calendar(start_time=1234.567), "above"),
    (Calendar(start_day=200, start_time=1234.567), "above"),
    (Calendar(start_day=100), "below"),
], ids=["start_day_40", "start_time_fractional", "both", "below_the_knot"])
def test_a_move_is_placed_at_its_breakpoint_despite_round_off(calendar, raw_side):
    """A breakpoint is simulation time = midnight + knot, which need not round-trip exactly
    through the seconds-from-new-year sum: the zone FROM it must still be the new one, and
    the zone up to it the old one. `raw_side` is where the breakpoint's computed time of day
    lands against the knot before the snap, a few ulps above or below it (or on it)."""
    knot = 46800.1
    occ = Occupancy([OccupancyDay([0.0, knot, DAY], ["one", "two", "two"])])
    moves = [p for p in occ.breakpoints(0.0, 2 * DAY, calendar)
             if occ.zone_at(p - 1e-3, calendar, left=False) == "one"
             and occ.zone_at(p + 1e-3, calendar, left=False) == "two"]
    assert moves, "no move from one to two found"
    sides = set()
    for p in moves:
        tod = calendar.day_and_time(p, left=False)[1]
        sides.add("exact" if tod == knot else "below" if tod < knot else "above")
        assert occ.zone_at(p, calendar, left=False) == "two"
        assert occ.zone_at(p, calendar, left=True) == "one"
        before, after = occ.segments(p - 300.0, p + 300.0, calendar)
        assert (before[2], after[2]) == ("one", "two")
    assert raw_side in sides and sides <= {raw_side, "exact"}


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


def _layer(species=("sarin", "CO2"), *, names=True):
    """A species layer over `_net()`: zones one and two, the ambient its boundary."""
    return TransportLayer(_net(), "species", capacity=torch.tensor([360.0, 720.0], dtype=F64),
                          flow_kind="airpath", boundary=["ambient"], n_species=len(species),
                          species_names=species if names else None)


def _visitor(**kw):
    day = OccupancyDay([0.0, 450.0, DAY], ["one", "two", "two"])
    return Occupant(Occupancy([day]), generation={"sarin": 1e-6}, multiplier=2.0,
                    name="visitor", **kw)


def _staying(zone, name="visitor"):
    day = OccupancyDay([0.0, DAY], [zone, zone])
    return Occupant(Occupancy([day]), generation={"sarin": 1e-6}, name=name)


def test_an_occupant_carries_its_generation_and_keys():
    v = _visitor()
    assert v.generation["sarin"].dtype == F64 and v.multiplier.item() == 2.0
    assert (v.presence_key, v.multiplier_key) == ("presence.visitor",
                                                  "generation_multiplier.visitor")
    with pytest.raises(ValueError, match=r"'visitor'.*schedules for \['CO2'\]"):
        _visitor(schedules={"CO2": TimeSeries([0.0, 1.0], [1.0, 1.0])})


@pytest.mark.parametrize(("kw", "match"), [
    (dict(generation={"sarin": [1e-6, 2e-6]}), r"'visitor'.*generation of 'sarin'.*scalar"),
    (dict(generation={"sarin": float("nan")}), r"'visitor'.*generation of 'sarin'.*finite"),
    (dict(multiplier=[1.0, 2.0]), r"'visitor'.*multiplier.*scalar"),
    (dict(multiplier=float("inf")), r"'visitor'.*multiplier.*finite"),
])
def test_an_occupant_refuses_a_rate_that_is_not_one_finite_number(kw, match):
    day = OccupancyDay([0.0, DAY], ["one", "one"])
    with pytest.raises(ValueError, match=match):
        Occupant(Occupancy([day]), name="visitor", **{"generation": {"sarin": 1e-6}, **kw})


def test_the_generation_lands_on_the_presence_times_rate_and_schedule():
    layer = _layer()
    net = layer.net
    gen = OccupantGeneration([_visitor()], layer=layer)
    assert gen.species == ["sarin", "CO2"]
    one, two = net.node_index("one"), net.node_index("two")
    presence = torch.zeros(net.n, dtype=F64)
    presence[one], presence[two] = 0.25, 0.75
    out = gen({}, {"presence.visitor": presence,
                   "generation_multiplier.visitor": torch.tensor([0.5], dtype=F64)})
    s = out["species.sources"]
    assert s.shape == (net.n, 2)
    assert s[one, 0].item() == pytest.approx(0.25 * 2e-6 * 0.5, rel=1e-15)
    assert s[two, 0].item() == pytest.approx(0.75 * 2e-6 * 0.5, rel=1e-15)
    assert s[:, 1].abs().sum().item() == 0.0
    base = torch.ones(net.n, 2, dtype=F64)
    added = gen({}, {"presence.visitor": presence, "species.sources": base})
    assert added["species.sources"][one, 0].item() == pytest.approx(1.0 + 0.25 * 2e-6)


def test_generation_into_a_single_species_without_an_axis_and_batched():
    layer = _layer(("sarin",))
    net = layer.net
    gen = OccupantGeneration([_visitor()], layer=layer)
    presence = torch.zeros(2, net.n, dtype=F64)
    presence[0, net.node_index("one")] = 1.0
    out = gen({}, {"presence.visitor": presence,
                   "species.sources": torch.zeros(net.n, dtype=F64)})["species.sources"]
    assert out.shape == (2, net.n)
    assert out[0, net.node_index("one")].item() == 2e-6 and out[1].abs().sum().item() == 0.0


def test_generation_needs_the_presence_and_known_species():
    with pytest.raises(KeyError, match=r"'visitor'.*\['sarin'\].*\['CO2'\]"):
        OccupantGeneration([_visitor()], layer=_layer(("CO2",)))
    with pytest.raises(ValueError, match="repeat"):
        OccupantGeneration([_visitor(), _visitor()], layer=_layer(("sarin",)))
    gen = OccupantGeneration([_visitor()], layer=_layer(("sarin",)))
    with pytest.raises(KeyError, match="'presence.visitor'"):
        gen({}, {})
    quiet = Occupant(Occupancy([OccupancyDay([0.0, DAY], ["one", "one"])]), name="quiet")
    assert OccupantGeneration([quiet], layer=_layer(("sarin",)))({}, {}) == {}


def test_the_species_names_come_from_the_layer_or_are_given_for_an_unnamed_one():
    unnamed = _layer(("sarin", "CO2"), names=False)
    assert OccupantGeneration([_visitor()], layer=unnamed,
                              species=["CO2", "sarin"]).species == ["CO2", "sarin"]
    with pytest.raises(ValueError, match=r"names no species.*species="):
        OccupantGeneration([_visitor()], layer=unnamed)
    with pytest.raises(ValueError, match=r"2 species.*\['sarin'\]"):
        OccupantGeneration([_visitor()], layer=unnamed, species=["sarin"])
    with pytest.raises(ValueError, match=r"\['CO2', 'sarin'\].*\['sarin', 'CO2'\]"):
        OccupantGeneration([_visitor()], layer=_layer(), species=["CO2", "sarin"])


def _plenum_layer():
    """Zones one and two, a storage-free plenum between them, and a loft no path reaches."""
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("one", volume=300.0)
    net.add_node("two", volume=600.0)
    net.add_node("plenum", volume=0.0)
    net.add_node("loft", volume=50.0)
    net.add_edge("ambient", "one", kind="airpath")
    net.add_edge("one", "plenum", kind="airpath")
    net.add_edge("plenum", "two", kind="airpath")
    net.add_edge("two", "ambient", kind="airpath")
    return TransportLayer(net, "species", capacity=torch.tensor([360.0, 720.0, 1.0], dtype=F64),
                          flow_kind="airpath", boundary=["ambient"], storage_free=["plenum"],
                          scheme="implicit", species_names=("sarin",))


@pytest.mark.parametrize(("zone", "error", "match"), [
    ("ambient", ValueError, r"'visitor'.*boundary node 'ambient'"),
    ("loft", ValueError, r"'visitor'.*node 'loft'.*no flow edge"),
    ("plenum", ValueError, r"'visitor'.*storage-free node 'plenum'"),
    ("kitchen", KeyError, r"'visitor'.*zone 'kitchen' is not a node"),
])
def test_an_occupant_where_the_layer_holds_no_air_is_refused_by_name(zone, error, match):
    """Its generation would land on a row the layer discards or refuses."""
    layer = _plenum_layer()
    with pytest.raises(error, match=match):
        OccupantGeneration([_staying("one", "host"), _staying(zone)], layer=layer)
    quiet = Occupant(Occupancy([OccupancyDay([0.0, DAY], [zone, zone])]), name="quiet")
    assert OccupantGeneration([quiet], layer=layer).occupants == []   # generates nothing


def test_the_occupants_and_the_sources_add_up_in_either_order():
    """`SpeciesSources` and `OccupantGeneration` both add to what `"species.sources"` holds,
    so neither drops the other's generation, whichever runs first."""
    layer = _layer(("sarin",))
    net = layer.net
    one = net.node_index("one")
    srcs = SpeciesSources([ConstantSource(one, 3e-6, 0.0)], species=[0], layer=layer)
    gen = OccupantGeneration([_staying("one")], layer=layer)
    state = {"species.x": torch.zeros(2, 1, dtype=F64)}
    presence = torch.zeros(net.n, dtype=F64)
    presence[one] = 1.0
    drivers = {"time": torch.tensor(0.0, dtype=F64),
               "species.x_boundary": torch.zeros(1, 1, dtype=F64),
               "species.source_multiplier": torch.ones(1, dtype=F64),
               "presence.visitor": presence}

    def run(closures):
        d = dict(drivers)
        for c in closures:
            d.update(c(state, d))
        return d["species.sources"]

    first, second = run([srcs, gen]), run([gen, srcs])
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first[one, 0].item() == pytest.approx(4e-6, rel=1e-15)


def test_occupant_inputs_write_the_presence_and_the_schedules():
    net = _net()
    sched = TimeSeries([0.0, 450.0, 900.0], [0.0, 1.0, 1.0], interpolation="step")
    v = _visitor(schedules={"sarin": sched})
    quiet = Occupant(Occupancy([OccupancyDay([0.0, DAY], ["one", "one"])]), name="quiet")
    inputs = occupant_inputs([v, quiet], net)
    assert [i.key for i in inputs] == ["presence.visitor", "generation_multiplier.visitor"]
    assert all(isinstance(i, Input) for i in inputs)
    cal = Calendar()
    assert inputs[1].signal.value(600.0, cal).tolist() == [1.0]
    assert inputs[1].signal.breakpoints(0.0, 900.0, cal) == [450.0]
    with pytest.raises(ValueError, match=r"occupant_inputs.*\['visitor'\].*repeat"):
        occupant_inputs([v, _visitor()], net)


def test_a_mid_step_move_splits_the_generation_by_time():
    """ContamX spreads an occupant's generation over the zones by the time spent in each
    during the step (measured: 2.1e-8 against ContamX; the step-end zone is off by 1e-1)."""
    model, state, drivers = project_to_model(read_prj(THREE))
    model.closures.append(OccupantGeneration([_visitor()], layer=model.transport["species"]))
    times = [0.0, 300.0, 600.0, 900.0]
    out = simulate(model, state, drivers, times, inputs=occupant_inputs([_visitor()], model.net),
                   land_on_breakpoints=False, **TIGHT)
    plain, s, d = project_to_model(read_prj(THREE))
    one, two = plain.net.node_index("one"), plain.net.node_index("two")
    manual = dict(s)
    for t0, t1, (f1, f2) in ((0.0, 300.0, (1.0, 0.0)), (300.0, 600.0, (0.5, 0.5)),
                             (600.0, 900.0, (0.0, 1.0))):
        S = torch.zeros(plain.net.n, 1, dtype=F64)
        S[one, 0], S[two, 0] = 2e-6 * f1, 2e-6 * f2
        manual = plain.step(manual, {**d, "species.sources": S,
                                     "time": torch.tensor(t1, dtype=F64)},
                            t1 - t0, t=t0, **TIGHT)
    torch.testing.assert_close(out["species.x"][-1], manual["species.x"], rtol=1e-10,
                               atol=1e-18)
