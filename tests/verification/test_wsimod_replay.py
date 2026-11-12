"""Each noodl node replayed on the calls WSIMOD 0.8.1's node received (spec, Verification 1).

WSIMOD runs each purpose-built case (``_wsimod_cases.py``) under ``capture_node_calls``; the
replay builds the noodl node alone on scripted arcs and compares every arc call it makes,
every reply and every store exactly (the node class's ``replay_bound``, 0 for Node, Waste and
Catchment), in both modes: with the case's pollutants,
and quantity-only on the volumes.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import io
import math
from pathlib import Path

import pytest

pytest.importorskip("wsimod")

import re  # noqa: E402

import torch  # noqa: E402

from noodl.apps.wsimod.config import read_wsimod_config  # noqa: E402
from noodl.apps.wsimod.day import WsimodDay  # noqa: E402
from noodl.apps.wsimod.engine import Ctx, Engine, make_node  # noqa: E402
from noodl.apps.wsimod.vqip import QUANTITY_ONLY, empty, sum_vqip  # noqa: E402
from tests.apps.wsimod._replay import (  # noqa: E402
    Mismatch,
    precondition_holds,
    replay,
    replay_arc,
    to_vqip,
)
from tests.verification._wsimod_cases import _NUTRIENT_POLLUTANTS, CASES  # noqa: E402
from tests.verification._wsimod_reference import (  # noqa: E402
    ArcDay,
    capture_arc_calls,
    capture_node_calls,
    record_wsimod_run,
    wsimod_model,
    wsimod_reference,
)

_RIVER_POLLUTANTS = _NUTRIENT_POLLUTANTS["pollutants"]


def _days(network, days: int):
    """Step ``network`` with its own pollutants through the day driver
    (``WsimodDay.run_days``): ``(day, ctx, engine)`` after each day's orchestration."""
    return WsimodDay(network, pollutants=True).run_days(days=days)


PAIRS = [(case, name) for case, spec in CASES.items() if spec.wsimod for name in spec.replay]
# W15: a case longer than this is replayed on its sampled days (``sample_days``); the others, up
# to 40 days, on every day. Only ``river_nutrients`` (400 days) is longer.
REPLAY_EVERY_DAY_UP_TO = 120


@functools.cache
def captured(case: str):
    spec = CASES[case]
    model = wsimod_model(spec.build())
    with capture_node_calls(model, spec.replay) as calls:
        record_wsimod_run(model, days=spec.days)
    return calls


@pytest.mark.parametrize(("case", "name"), PAIRS, ids=[f"{c}-{n}" for c, n in PAIRS])
def test_replay_matches_wsimod(case, name):
    """Every captured call of the node replayed bitwise, in both modes; a long case on its
    sampled days (W15). A River with biochemistry (``river_nutrients``) has its two lag
    registers loaded from WSIMOD's lists before each call and compared after it."""
    spec = CASES[case]
    calls = captured(case)[name]
    assert calls, f"WSIMOD made no call into {name!r}"
    if spec.days > REPLAY_EVERY_DAY_UP_TO:
        keep = set(sample_days(calls))
        calls = [c for c in calls if c.day in keep]
        assert 60 < len(keep) < spec.days
    network = read_wsimod_config(spec.build())
    for mode in spec.modes:
        pollutants = network.pollutants if mode == "pollutants" else QUANTITY_ONLY
        worst = replay(network, name, calls, scale=spec.scale, pollutants=pollutants)
        node_spec = next(n for n in network.nodes if n.name == name)
        bound = type(make_node(node_spec, network, pollutants)).replay_bound
        assert worst <= bound, (mode, worst)
        if bound == 0.0:  # Node, Waste, Catchment: bitwise WSIMOD
            assert worst == 0.0, (mode, worst)


def test_the_cases_reach_the_branches_they_are_built_for():
    """three_out_arcs over-subscribes the junction on some days and not others, and polls
    again after a round; node_chain's pushes ask the checks down the chain."""
    calls = captured("three_out_arcs")["junction"]
    pushes = [c for c in calls if c.method == "push_set"]
    rounds = [sum(a.method == "send_push_request" for a in c.arc_calls) for c in pushes]
    assert max(rounds) > 3 and min(rounds) == 3
    oversubscribed = [
        c.vqip["volume"] > sum(a.reply["volume"] for a in c.arc_calls[:3]) for c in pushes
    ]
    assert any(oversubscribed) and not all(oversubscribed)
    chain = captured("node_chain")
    checks = [c for c in chain["reach_2"] if c.method == "push_check"]
    assert checks and all([a.method for a in c.arc_calls] == ["send_push_check"] for c in checks)
    routes = [c for c in chain["headwater"] if c.method == "route"]
    unrouted = [c.post["unrouted_water"]["volume"] > 0 for c in routes]
    assert len(routes) == CASES["node_chain"].days and any(unrouted) and not all(unrouted)


def test_decay_storage_reaches_the_cap_and_a_factor_of_exactly_one():
    """Review Focus 4: ``res`` (constant 0.9) is capped on some warm days and not others, and
    ``riv`` (constant 1) at exactly 20 degC has the raw factor ``1 * exponent ** 0 == 1``, which
    the cap leaves alone (``1.0 > 1`` is false); its stored factor is 1 and its decayed phosphate
    is its whole storage. ``pond`` decays the non-additive ``do``; ``ammonia`` is undecayed."""
    spec = CASES["decay_storage"]
    day = WsimodDay(read_wsimod_config(spec.build()), pollutants=True)
    day.record_decisions = True
    capped: dict[str, list[bool]] = {}
    for k, ctx, engine in day.run_days(days=spec.days):
        for d in ctx.decisions:
            if "make_decay/cap" in d.site:
                capped.setdefault(d.site, []).append(bool(d.outcome))
        if k == 3:
            factor = engine.nodes["riv"].tank.decay.factor
            assert float(factor[1]) == 1.0  # phosphate: exactly 1, uncapped
            assert capped["node:riv/tank/make_decay/cap:phosphate"][3] is False
            decayed = engine.nodes["pond"].tank.decay.factor
            assert float(decayed[3]) > 0 and float(decayed[2]) == 0.0  # do decays, ammonia not
    assert any(capped["node:res/tank/make_decay/cap:phosphate"])
    assert not all(capped["node:res/tank/make_decay/cap:phosphate"])
    assert "node:pond/tank/make_decay/cap:ammonia" not in capped


@pytest.mark.parametrize("case", ["decay_queue_groundwater", "decay_queue_groundwater_n0"])
def test_decay_queue_groundwater_water_waits_in_slot_zero_for_distribute(case):
    """Plan Review Focus 1, X13 (from WSIMOD's captured queue): on the days nothing is pushed
    into ``gw`` the water the last end of day moved to slot 0 is still there when
    ``distribute`` starts, and its ``update_queue`` (``storage.py:352``) moves it into
    ``active_storage``; ``DecayQueueTank``'s end of day does not flush it. The internal arc's
    cap binds on some days (``ammonia``) and the queue keys show ``number_of_timesteps``
    (X12: 1, X28: 0)."""
    fa = 1e-11  # WSIMOD's FLOAT_ACCURACY: a smaller push returns before the queue
    calls = captured(case)["gw"]
    flushed = {c.day for c in calls if c.method == "push_set" and c.vqip["volume"] >= fa}
    waiting = []
    for c in calls:
        if c.method != "distribute":
            continue
        slot0 = c.pre["tank.internal_arc.register"][0]["volume"]
        if slot0 > 0:
            waiting.append(c.day)
            assert c.post["tank.internal_arc.register"][0]["volume"] == 0.0  # update_queue
    assert waiting and set(waiting).isdisjoint(flushed)
    keys = {k for c in calls if c.method == "push_set" for k in c.post["tank.internal_arc.present"]}
    assert keys == ({0, 1, 2, 3} if case == "decay_queue_groundwater" else {0, 1, 2})
    day = WsimodDay(read_wsimod_config(CASES[case].build()), pollutants=True)
    day.record_decisions = True
    capped = [
        bool(d.outcome)
        for _, ctx, _ in day.run_days(days=CASES[case].days)
        for d in ctx.decisions
        if d.site == "node:gw/tank/internal_arc/make_decay/cap:ammonia"
    ]
    assert any(capped) and not all(capped)


def test_the_river_nutrients_capture_holds_wsimods_lag_lists_and_forcing():
    """The capture of a River with biochemistry records both lag lists as WSIMOD's lists, which
    fill to ``max_temp_lag + 1`` and ``max_phosphorus_lag + 1`` entries (21 and 366 for ``r_up``,
    6 and 301 for the overridden ``r_low``) and then turn over, ``bio_in``/``bio_out``, and the
    day's temperature forcing. The replay loads and compares all of them: a captured register
    slot moved by one ulp, or a changed forcing, fails the replay naming the call."""
    calls = captured("river_nutrients")
    mapping = CASES["river_nutrients"].build()
    for river, sizes in (("r_up", (21, 366)), ("r_low", (6, 301))):
        configured = next(n for n in mapping["nodes"] if n["name"] == river)["data_input_dict"]
        discharges = [c for c in calls[river] if c.method == "calculate_discharge"]
        assert len(discharges) == CASES["river_nutrients"].days
        for c in discharges:
            assert len(c.pre["lagged_temperatures"]) == min(c.day, sizes[0]), c.day
            assert len(c.pre["lagged_total_phosphorus"]) == min(c.day, sizes[1]), c.day
            assert len(c.post["lagged_temperatures"]) == min(c.day + 1, sizes[0])
            assert len(c.post["lagged_total_phosphorus"]) == min(c.day + 1, sizes[1])
            assert (
                c.pre["forcing.temperature"] == configured[("temperature", mapping["dates"][c.day])]
            )
            assert set(c.pre["bio_out"]) == {"volume", *_RIVER_POLLUTANTS}
        assert any(c.post["bio_out"]["nitrate"] > 0 for c in discharges)
        ends = [c for c in calls[river] if c.method == "end_timestep"]
        assert all(c.post["bio_in"]["org-nitrogen"] == 0 for c in ends)
    network = read_wsimod_config(CASES["river_nutrients"].build())
    scale = CASES["river_nutrients"].scale
    early = [c for c in calls["r_up"] if c.day < 30]
    assert replay(network, "r_up", copy.deepcopy(early), scale=scale) == 0.0
    changed = copy.deepcopy(early)
    call = next(c for c in changed if c.method == "calculate_discharge" and c.day == 20)
    slot = call.pre["lagged_total_phosphorus"]
    slot[-1] = math.nextafter(slot[-1], math.inf)
    with pytest.raises(Mismatch, match=r"r_up call \d+ \(day 20, calculate_discharge"):
        replay(network, "r_up", changed, scale=scale)
    changed = copy.deepcopy(early)
    call = next(c for c in changed if c.method == "calculate_discharge" and c.day == 20)
    call.pre["forcing.temperature"] += 1.0
    with pytest.raises(Mismatch, match=r"day 20, calculate_discharge.*forcing temperature"):
        replay(network, "r_up", changed, scale=scale)


def test_the_replay_notices_a_difference():
    """A reply moved by 1e-11 x scale, or a reordered arc script, fails the replay."""
    spec = CASES["three_out_arcs"]
    network = read_wsimod_config(spec.build())
    calls = copy.deepcopy(captured("three_out_arcs")["junction"])
    call = next(c for c in calls if c.method == "push_set" and len(c.arc_calls) > 6)
    push = next(a for a in call.arc_calls if a.method == "send_push_request")
    push.reply["volume"] += 1e-11 * spec.scale
    with pytest.raises(Mismatch):
        replay(network, "junction", calls, scale=spec.scale)
    calls = copy.deepcopy(captured("three_out_arcs")["junction"])
    call = next(c for c in calls if c.method == "push_set")
    call.arc_calls[3], call.arc_calls[4] = call.arc_calls[4], call.arc_calls[3]
    with pytest.raises(Mismatch):
        replay(network, "junction", calls, scale=spec.scale)


def test_maxiter_pull_reaches_maxiter_and_noodl_says_so_on_the_same_days():
    """W38: the pull path and the ``MAXITER`` exit, live. WSIMOD's ``pull_distributed`` into the
    reservoir ends at ``MAXITER`` every day (it prints "Maxiter reached"); noodl's
    ``"maxiter"`` diagnostic fires on exactly those days."""
    spec = CASES["maxiter_pull"]
    printed = wsimod_reference(spec.build(), days=spec.days).printed
    network = read_wsimod_config(spec.build())
    index = {str(d.date()): i for i, d in enumerate(network.dates)}
    days = {
        index[m.group(1)]
        for line in printed
        if (m := re.fullmatch(r"Maxiter reached in res at (.*)", line))
    }
    assert "Maxiter reached in res at 2000-01-01" in printed
    assert days == set(range(spec.days))  # every day of the case
    engine = Engine(network, pollutants=network.pollutants)
    ours = set()
    for day in range(spec.days):
        ctx = Ctx(
            network.pollutants,
            network.float_accuracy,
            torch.Size(()),
            day=day,
            node_names=list(engine.nodes),
        )
        engine.begin_day(ctx)
        for node in engine.nodes.values():
            node.bind(node.configured_parameters(network.dates[day]))
        res = engine.nodes["res"]
        res.make_abstractions(active=torch.tensor(True))
        if bool(ctx.maxiter[("res", "pull_distributed", 0)]):
            ours.add(day)
        res.distribute(active=torch.tensor(True))
        engine.end_day()
    assert ours == days


def test_the_storage_cases_reach_the_branches_they_are_built_for():
    # forced_negative_excess: the reservoir starts above capacity, so its push check is empty
    res = captured("forced_negative_excess")["res"]
    assert res[0].pre["tank.storage"]["volume"] > 12.0
    checks = [c for c in res if c.method == "push_check" and c.day == 0]
    assert checks and all(c.reply["volume"] == 0.0 for c in checks)
    # queue_groundwater: pro rata pulls from a queue that holds water, and its frozen check
    gw = captured("queue_groundwater")["gw"]
    pulls = [c for c in gw if c.method == "pull_set" and c.reply["volume"] > 0]
    assert pulls and any(
        any(row["volume"] > 0 for time, row in c.pre["tank.internal_arc.register"].items() if time)
        for c in pulls
    )
    assert any(c.method == "push_check" for c in gw) and any(c.method == "pull_check" for c in gw)
    # river_reservoir_environmental: some water counts as environmental flow before the
    # satisfy step, and the step runs short of the flow on some days and not others
    rr = captured("river_reservoir_environmental")["rr"]
    pushes = [c for c in rr if c.method == "push_set"]
    assert any(c.post["total_environmental_satisfied"] > 0 for c in pushes)
    satisfy = [c for c in rr if c.method == "satisfy_environmental"]
    assert len(satisfy) == CASES["river_reservoir_environmental"].days
    # catchment_abstraction: the pull path of a Catchment
    src = captured("catchment_abstraction")["src"]
    assert any(c.method == "pull_set" and c.reply["volume"] > 0 for c in src)
    assert any(c.method == "pull_check" for c in src)
    # pull_and_push_arcs: the deny handlers answered both ways
    a = captured("pull_and_push_arcs")["a"]
    denied_push = [
        x for c in a for x in c.arc_calls if x.arc == "pull_ab" and x.method == "send_push_request"
    ]
    assert denied_push and all(x.reply["volume"] == x.vqip["volume"] for x in denied_push)
    # user_alt_queue_arc: the storage pushed through the queue arc
    pushes = [
        x for c in captured("user_alt_queue_arc")["a"] for x in c.arc_calls if x.arc == "a_to_b"
    ]
    assert any(x.method == "send_push_request" for x in pushes)
    # reservoir_over_capacity: day 0 asks for no water, later days do
    over = [
        c for c in captured("reservoir_over_capacity")["res"] if c.method == "make_abstractions"
    ]
    asked = [x.vqip["volume"] for c in over for x in c.arc_calls if x.method == "send_pull_request"]
    assert asked[0] == 0.0 and max(asked) > 0.0


def _noodl_event_days(case: str, condition: str, node: str) -> set[int]:
    spec = CASES[case]
    network = read_wsimod_config(spec.build())
    days = set()
    for day, ctx, _ in _days(network, spec.days):
        if bool(ctx.events.get(condition, {}).get(node, torch.tensor(False))):
            days.add(day)
    return days


def _printed_days(case: str, pattern: str) -> set[int]:
    """The days of the lines WSIMOD printed that match ``pattern`` (group 1 is the date)."""
    spec = CASES[case]
    printed = wsimod_reference(spec.build(), days=spec.days).printed
    network = read_wsimod_config(spec.build())
    index = {str(d.date()): i for i, d in enumerate(network.dates)}
    return {index[m.group(1)] for line in printed if (m := re.fullmatch(pattern, line))}


def test_demand_deficits_and_unsuccessful_leaks_fall_on_the_days_wsimod_printed_them():
    """D13: ``demand deficit of x at <node> on <date>`` and ``warning, distribution leakage not
    going to GW in <node> at <date>`` against noodl's ``"demand_deficit"`` and
    ``"leakage_not_going_to_gw"`` events, day by day; every case below has days on both sides."""
    for case, node in (
        ("distribution_leakage", "town"),
        ("unlimited_distribution", "town"),
        ("demand_constant_and_loads", "town"),
        ("demand_constant_and_loads", "village"),
        ("reservoir_spill", "town"),
    ):
        ours = _noodl_event_days(case, "demand_deficit", node)
        theirs = _printed_days(case, rf"demand deficit of \S+ at {node} on (.*)")
        assert ours == theirs, (case, node)
        assert 0 < len(theirs) < CASES[case].days, (case, node)
    for case, node in (("distribution_leakage", "dist_gw"), ("distribution_leakage", "dist_dry")):
        ours = _noodl_event_days(case, "leakage_not_going_to_gw", node)
        theirs = _printed_days(
            case, rf"warning, distribution leakage not going to GW in {node} at (.*)"
        )
        assert ours == theirs, (case, node)
    # dist_gw's groundwater takes the leak on some days and is full on others; dist_dry has none
    assert 0 < len(_noodl_event_days("distribution_leakage", "leakage_not_going_to_gw", "dist_gw"))
    assert (
        len(_noodl_event_days("distribution_leakage", "leakage_not_going_to_gw", "dist_gw"))
        < CASES["distribution_leakage"].days
    )
    # the unpushed foul water of a ResidentialDemand with no Sewer, and a village whose out-arcs
    # are too small, are "Demand not able to push" on every day
    assert _noodl_event_days("residential_demand", "demand_not_able_to_push", "homes") == set(
        range(CASES["residential_demand"].days)
    )
    unpushed = _noodl_event_days("demand_constant_and_loads", "demand_not_able_to_push", "town")
    assert unpushed == set()


def test_reservoir_spill_days_are_the_days_wsimod_printed_spill_at_reservoir():
    """D14, live: a reservoir pulling through a leaking ``Distribution`` with no groundwater gets
    more than its excess (``distribution.py:40-50``) and spills (``storage.py:983-989``). noodl's
    ``"reservoir_spill"`` event falls on exactly the days WSIMOD prints "Spill at reservoir",
    and the tank ends those days above its capacity on both sides."""
    spec = CASES["reservoir_spill"]
    model = wsimod_model(spec.build())
    network = read_wsimod_config(spec.build())
    index = {d: i for i, d in enumerate(model.dates)}
    res = model.nodes["res"]
    printed_on: set[int] = set()
    original = res.make_abstractions

    def watched():
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            original()
        if "Spill at reservoir" in buffer.getvalue():
            printed_on.add(index[res.t])
        print(buffer.getvalue(), end="")

    res.make_abstractions = watched
    reference = record_wsimod_run(model, days=spec.days)
    slot = reference.tank_names.index(("res", "tank"))
    capacity = 10.0
    theirs_end = reference.tank_storage[:, slot]
    ours = _noodl_event_days("reservoir_spill", "reservoir_spill", "res")
    assert ours == printed_on
    assert 0 < len(ours) < spec.days  # the supply runs dry part-way
    ours_end = {}
    for day, _, engine in _days(network, spec.days):
        ours_end[day] = float(engine.nodes["res"].tank.storage[0])
    for day in ours:
        assert ours_end[day] > capacity and float(theirs_end[day]) > capacity
        assert ours_end[day] == float(theirs_end[day])


def _printed_count(case: str, line: str) -> int:
    spec = CASES[case]
    return wsimod_reference(spec.build(), days=spec.days).printed.count(line)


def test_the_sewer_and_works_cases_reach_the_branches_they_are_built_for():
    # sewer_timearea: the sewer is reached through the Demand tag and the default tag, and its
    # push check limits a sender on some days
    sewer = captured("sewer_timearea")["sewer"]
    tags = {c.tag for c in sewer if c.method == "push_set"}
    assert tags == {"Demand", "default"}
    checks = [c for c in sewer if c.method == "push_check" and c.vqip is not None]
    assert any(c.reply["volume"] < c.vqip["volume"] for c in checks)
    # foul_sewers: foul_main runs in the "Foul" slot, the trunk in the "Sewer" slot, and the
    # trunk's queue holds water at its pipe time of 2
    foul = captured("foul_sewers")
    assert sum(c.method == "make_discharge" for c in foul["foul_main"]) == CASES["foul_sewers"].days
    assert {c.tag for c in foul["trunk"] if c.method == "push_set"} == {"default"}
    assert any(
        c.post["sewer_tank.internal_arc.register"].get(2, {}).get("volume", 0) > 0
        for c in foul["trunk"]
    )
    # groundwater_infiltrate: infiltration above and below the threshold, part sent back
    gw = [c for c in captured("groundwater_infiltrate")["gw"] if c.method == "infiltrate"]
    for_sewer = []
    for c in gw:
        infiltration = (max(c.pre["tank.storage"]["volume"] - 50.0, 0) * 0.05) ** 0.5
        sent = sum(
            a.vqip["volume"] - a.reply["volume"]
            for a in c.arc_calls
            if a.arc == "gw_to_sewer" and a.method == "send_push_request"
        )
        for_sewer.append((infiltration, sent))
    assert any(0 < sent < infiltration for infiltration, sent in for_sewer)
    assert any(infiltration == 0 for infiltration, _ in for_sewer)
    # wwtw_storm_tank: the tank fills, drains into the spare throughput, and takes the forced
    # rejection; the reuse demand pulls from the treated water
    works = captured("wwtw_storm_tank")["works"]
    storm = [c.post["stormwater_tank.storage"]["volume"] for c in works]
    assert max(storm) > 30.0 and min(storm) == 0.0
    drains = [
        c
        for c in works
        if c.method == "calculate_discharge"
        and c.post["stormwater_tank.storage"]["volume"] < c.pre["stormwater_tank.storage"]["volume"]
    ]
    assert drains
    assert any(c.method == "pull_set" and c.reply["volume"] > 0 for c in works)
    couldnt = _printed_count("wwtw_storm_tank", "WWTW couldnt push")
    assert 0 < couldnt < CASES["wwtw_storm_tank"].days
    # fwtw_deficit_sludge: deficits once the supply is dry, unpushed sludge once the sewer fills
    plant = [c for c in captured("fwtw_deficit_sludge")["plant"] if c.method == "treat_water"]
    deficits = [c.post["total_deficit"]["volume"] > 0 for c in plant]
    assert any(deficits) and not all(deficits)
    assert any(c.post["unpushed_sludge"]["volume"] > 0 for c in plant)


def test_service_reservoir_shortfalls_and_unpushed_sludge_fall_on_the_days_wsimod_printed():
    """D13: ``Service reservoirs not filled at <node> on <date>`` day by day against noodl's
    ``"service_reservoirs_not_filled"``; the undated "nowhere for sludge to go" and "WWTW
    couldnt push" by count."""
    case = "fwtw_deficit_sludge"
    ours = _noodl_event_days(case, "service_reservoirs_not_filled", "plant")
    theirs = _printed_days(case, r"Service reservoirs not filled at plant on (.*)")
    assert ours == theirs and 0 < len(theirs) < CASES[case].days
    sludge = _noodl_event_days(case, "sludge_nowhere", "plant")
    assert len(sludge) == _printed_count(case, "nowhere for sludge to go") > 0
    couldnt = _noodl_event_days("wwtw_storm_tank", "wwtw_couldnt_push", "works")
    assert len(couldnt) == _printed_count("wwtw_storm_tank", "WWTW couldnt push")


def test_pipe_time_zero_water_leaves_the_sewer_the_day_it_arrives():
    """Spec, Flushing a ``QueueTank`` on push: with ``pipe_time = 0`` each day's inflow is
    discharged by the same day's ``make_discharge``, in WSIMOD and in noodl."""
    spec = CASES["queue_tank_time_zero"]
    reference = record_wsimod_run(wsimod_model(spec.build()), days=spec.days)
    inflow = reference.arc_flow[:, reference.arc_names.index("basin_to_sewer")]
    outflow = reference.arc_flow[:, reference.arc_names.index("sewer_to_sea")]
    assert bool((inflow > 0).all()) and torch.equal(inflow, outflow)
    network = read_wsimod_config(spec.build())
    for day, _, engine in _days(network, spec.days):
        ours_in = engine.arcs["basin_to_sewer"].flow_in
        assert float(engine.arcs["sewer_to_sea"].flow_in) == float(ours_in) == float(inflow[day])


def _arc_flows(network, days: int) -> dict[str, list[float]]:
    flows: dict[str, list[float]] = {}
    for _, _, engine in _days(network, days):
        for name, arc in engine.arcs.items():
            flows.setdefault(name, []).append(float(arc.flow_in))
    return flows


def test_foul_sewers_run_in_their_own_slot_and_match_wsimod_day_by_day():
    """``model.py:658-660``: ``foul_main`` runs in the ``"Foul"`` slot, after the ``"Sewer"`` slot
    in this orchestration, so the trunk discharges before ``foul_main`` pushes into it. Every
    arc's daily flow equals WSIMOD's; filing ``foul_main`` under ``"Sewer"`` changes them."""
    spec = CASES["foul_sewers"]
    reference = record_wsimod_run(wsimod_model(spec.build()), days=spec.days)
    network = read_wsimod_config(spec.build())
    ours = _arc_flows(network, spec.days)
    for name, flows in ours.items():
        theirs = reference.arc_flow[:, reference.arc_names.index(name)].tolist()
        assert flows == theirs, name
    foul = next(n for n in network.nodes if n.name == "foul_main")
    assert foul.slot_key == "Foul"
    foul.slot_key = "Sewer"  # the wrong slot: foul_main would run before the trunk
    assert _arc_flows(network, spec.days) != ours


# --- Land (Task 12) -----------------------------------------------------------------------------
LAND_CASES = (
    "impervious_only",
    "pervious_only",
    "evaporation_clamp",
    "temperature_blending",
    "land_percolation_threshold",
    "residential_with_land",
)


@pytest.mark.parametrize("case", LAND_CASES)
def test_the_land_cases_match_wsimod_day_by_day(case):
    """The day driver on each Land case against WSIMOD's own run: every arc's flow
    and quality, every surface's storage and quality (``record_all``, ``model.py:1135-1170``)
    and the Land residence tanks, bitwise on every day."""
    spec = CASES[case]
    reference = record_wsimod_run(wsimod_model(spec.build()), days=spec.days)
    network = read_wsimod_config(spec.build())
    tanks = [
        (k, node, prop)
        for k, (node, prop) in enumerate(reference.tank_names)
        if prop in ("percolation", "subsurface_runoff", "surface_runoff")
    ]
    assert reference.surface_names and tanks
    for day, _, engine in _days(network, spec.days):
        for i, name in enumerate(reference.arc_names):
            ours = engine.arcs[name].vqip_out
            assert float(ours[0]) == float(reference.arc_flow[day, i]), (day, name)
            assert ours[1:].tolist() == reference.arc_quality[day, i].tolist(), (day, name)
        for j, (node, surface) in enumerate(reference.surface_names):
            ours = engine.nodes[node]._surface(surface).storage
            assert float(ours[0]) == float(reference.surface["storage"][day, j]), (day, surface)
            assert ours[1:].tolist() == reference.surface_quality[day, j].tolist(), (day, surface)
        for k, node, prop in tanks:
            ours = getattr(engine.nodes[node], prop).storage
            assert float(ours[0]) == float(reference.tank_storage[day, k]), (day, node, prop)


def test_the_land_cases_reach_the_branches_they_are_built_for():
    # impervious_only: the roofs (capacity 0) keep water only when the sewer's check refuses
    # part of their runoff, which it does on some days
    town = [c for c in captured("impervious_only")["town"] if c.method == "run"]
    kept = [c.post["surfaces.roofs.storage"]["volume"] > 0 for c in town]
    assert any(kept) and not all(kept)
    dry = [c.post["surfaces.roads.evaporation"]["volume"] for c in town]
    assert len(set(dry)) > 1
    # pervious_only: each soil passes water through its store on some days and evaporates on
    # others (land.py:985-1001)
    farm = [c for c in captured("pervious_only")["farm"] if c.method == "run"]
    for name in ("grass", "wood"):
        through = [
            c.post[f"surfaces.{name}.tank_recharge"]
            + c.post[f"surfaces.{name}.subsurface_flow"]["volume"]
            + c.post[f"surfaces.{name}.percolation"]["volume"]
            > 0
            for c in farm
        ]
        assert any(through) and not all(through), name
    # land_percolation_threshold: rejections above FLOAT_ACCURACY, in (0, FLOAT_ACCURACY] and 0
    plot = [c for c in captured("land_percolation_threshold")["plot"] if c.method == "run"]
    rejected = [
        a.reply["volume"]
        for c in plot
        for a in c.arc_calls
        if a.arc == "plot_to_aquifer" and a.method == "send_push_request"
    ]
    assert len(rejected) == CASES["land_percolation_threshold"].days
    assert any(r > 1e-11 for r in rejected) and any(0 < r <= 1e-11 for r in rejected)
    assert any(r == 0 for r in rejected)
    # residential_with_land: the land answers the garden check 0 twice a day (the garden
    # demand's get_connected, demand.py:262, then push_distributed's, nodes.py:444), so with two
    # out-arcs the push loop never runs and the land receives no garden push
    gardens = captured("residential_with_land")["gardens"]
    garden = [c for c in gardens if c.tag == ("Demand", "Garden")]
    assert [c.method for c in garden] == ["push_check"] * 2 * CASES["residential_with_land"].days
    assert all(c.vqip is None and c.reply["volume"] == 0 for c in garden)
    # temperature_blending: the surfaces' temperature changes from day to day
    suburb = [c for c in captured("temperature_blending")["suburb"] if c.method == "run"]
    assert len({c.post["surfaces.roofs.storage"]["temperature"] for c in suburb}) > 5


def _printed_on(case: str, node: str, method: str, text: str) -> set[int]:
    """The days on which WSIMOD's ``node.method`` printed a line containing ``text``."""
    spec = CASES[case]
    model = wsimod_model(spec.build())
    index = {d: i for i, d in enumerate(model.dates)}
    target = model.nodes[node]
    original = getattr(target, method)
    days: set[int] = set()

    def watched(*args, **kwargs):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            out = original(*args, **kwargs)
        if text in buffer.getvalue():
            days.add(index[target.t])
        print(buffer.getvalue(), end="")
        return out

    setattr(target, method, watched)
    record_wsimod_run(model, days=spec.days)
    return days


def test_inaccurate_evaporation_falls_on_the_days_wsimod_printed_it():
    """D13: ``"inaccurate_evaporation"`` (``land.py:1003-1021``) on exactly the days WSIMOD's
    ``Land.run`` printed "inaccurate evaporation calculation"."""
    case = "evaporation_clamp"
    theirs = _printed_on(case, "heath", "run", "inaccurate evaporation calculation")
    ours = _noodl_event_days(case, "inaccurate_evaporation", "heath")
    assert ours == theirs and 0 < len(theirs) < CASES[case].days


def test_the_dropped_percolation_falls_on_the_days_wsimod_reports_a_land_imbalance():
    """``land.py:213-215``: on the days noodl records the vanished percolation rejection as
    dropped water, WSIMOD's own node balance of the land is out by it ("mass balance error")."""
    case = "land_percolation_threshold"
    spec = CASES[case]
    network = read_wsimod_config(spec.build())
    ours = {day for day, ctx, _ in _days(network, spec.days) if float(ctx.dropped[0]) != 0}
    theirs = _printed_on(case, "plot", "node_mass_balance", "mass balance error for volume")
    assert ours == theirs and ours


# --- the demos (Task 15) -------------------------------------------------------------------------
DEMOS = Path(__file__).resolve().parents[1] / "data" / "wsimod"
DEMO_DAYS = {"quickstart": 1456, "oxford": 365}  # the days the sample can reach (D15, Task 16)
FULL_DAYS = {"quickstart": 1456, "oxford": 1456}  # the slow run: every day of the demo
# The classes the demos hold, in the names of ``NodeSpec.cls`` (``ResidentialDemand`` is
# WSIMOD's ``Demand`` under a different name); the brief's list is the union.
DEMO_CLASSES = {
    "quickstart": ("Land", "Sewer", "Groundwater", "Node", "Waste"),
    "oxford": (
        "Land",
        "Sewer",
        "WWTW",
        "FWTW",
        "Reservoir",
        "Groundwater",
        "ResidentialDemand",
        "Catchment",
        "Node",
        "Waste",
    ),
}
# (id, demo, config, mode): quantity-only on the demo as saved; every pollutant on the no-deposition
# configuration, which neutralises only the surfaces' deposition, 3c (README there)
RUNS = (
    ("", "quickstart", "config.yml", "quantity"),
    ("", "oxford", "config.yml", "quantity"),
    ("pollutants-", "quickstart", "config_no_deposition.yml", "pollutants"),
    ("pollutants-", "oxford", "config_no_deposition.yml", "pollutants"),
)
DEMO_PAIRS = [
    pytest.param(demo, config, mode, cls)
    for _, demo, config, mode in RUNS
    for cls in DEMO_CLASSES[demo]
]
DEMO_IDS = [f"{prefix}{demo}-{cls}" for prefix, demo, _, _ in RUNS for cls in DEMO_CLASSES[demo]]


# nodes of no arc, which WSIMOD never calls (oxford's cherwell_thames), by class
UNCALLED = {"quickstart": {}, "oxford": {"Node": {"cherwell_thames"}}}


@functools.cache
def demo_network(demo: str, config: str):
    return read_wsimod_config(DEMOS / demo, config_name=config)


@functools.cache
def demo_capture(demo: str, config: str, days: int):
    """One live WSIMOD run of the demo with every node wrapped (once per process, so once per
    xdist worker)."""
    model = wsimod_model(DEMOS / demo, config_name=config)
    names = [spec.name for spec in demo_network(demo, config).nodes]
    with capture_node_calls(model, names) as calls:
        record_wsimod_run(model, days=days)
    return calls


def _outcome(day_calls) -> tuple:
    """D15: the ``(arc, method, tag)`` of the node's captured arc calls that day."""
    return tuple((a.arc, a.method, repr(a.tag)) for c in day_calls for a in c.arc_calls)


def sample_days(calls) -> list[int]:
    """D15: days 0-59 and every 20th, plus each captured day whose branch outcome is new."""
    by_day: dict[int, list] = {}
    for call in calls:
        by_day.setdefault(call.day, []).append(call)
    last = max(by_day)
    sample = {d for d in range(last + 1) if d < 60 or d % 20 == 0}
    seen = {_outcome(by_day.get(d, [])) for d in sample}
    for day in sorted(by_day):
        outcome = _outcome(by_day[day])
        if day not in sample and outcome not in seen:
            sample.add(day)
            seen.add(outcome)
    return sorted(sample)


def _magnitudes(value, volume_only: bool):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield abs(float(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            if volume_only and isinstance(item, (int, float)) and key != "volume":
                continue
            yield from _magnitudes(item, volume_only)


def node_scale(calls, volume_only: bool) -> float:
    """The largest magnitude over the node's captured records (volumes when quantity-only)."""
    top = 0.0
    for call in calls:
        records = [call.vqip, call.reply, call.pre, call.post]
        for arc in call.arc_calls:
            records += [arc.vqip, arc.reply]
        top = max(top, *(m for r in records for m in _magnitudes(r, volume_only)), 0.0)
    return top or 1.0


def _replay_class(demo, config, mode, cls, days, *, every_day, decisions=None):
    network = demo_network(demo, config)
    pollutants = network.pollutants if mode == "pollutants" else QUANTITY_ONLY
    captured_calls = demo_capture(demo, config, days)
    sizes, worst, uncalled = {}, 0.0, set()
    for spec in network.nodes:
        if spec.cls != cls:
            continue
        calls = captured_calls[spec.name]
        # all() over an empty call list is True: no calls at all still means "WSIMOD called
        # nothing"
        if all(c.method == "end_timestep" for c in calls):  # WSIMOD called nothing but the end
            uncalled.add(spec.name)
            continue
        keep = None if every_day else set(sample_days(calls))
        used = [c for c in calls if keep is None or c.day in keep]
        sizes[spec.name] = len({c.day for c in used})
        scale = node_scale(calls, volume_only=mode == "quantity")
        found = replay(
            network, spec.name, used, scale=scale, pollutants=pollutants, decisions=decisions
        )
        bound = type(make_node(spec, network, pollutants)).replay_bound
        assert found <= bound * scale, (spec.name, found)
        worst = max(worst, found)
    assert uncalled == UNCALLED[demo].get(cls, set()), (demo, cls, uncalled)
    assert sizes, f"{demo} has no {cls} that WSIMOD called"
    return sizes, worst


def test_the_demos_hold_the_classes_the_brief_names():
    named = {
        "Land", "Sewer", "WWTW", "FWTW", "Reservoir", "Groundwater",
        "ResidentialDemand", "Catchment", "Node", "Waste",
    }  # fmt: skip
    for demo, classes in DEMO_CLASSES.items():
        held = {spec.cls for spec in demo_network(demo, "config.yml").nodes}
        assert held == set(classes) and held <= named, demo
    assert set().union(*map(set, DEMO_CLASSES.values())) == named


def _through_soil_outcomes(decisions) -> dict[str, set[bool]]:
    outcomes: dict[str, set[bool]] = {}
    for _, d in decisions:
        if d.site.endswith("/through_soil") and bool(d.active):
            outcomes.setdefault(d.site.split("/")[0], set()).add(bool(d.outcome))
    return outcomes


@pytest.mark.parametrize(("demo", "config", "mode", "cls"), DEMO_PAIRS, ids=DEMO_IDS)
def test_demo_nodes_replay_wsimod(demo, config, mode, cls):
    """Every node of ``cls`` replayed on the sampled days of WSIMOD's captured calls, bitwise."""
    decisions: list = []
    _, worst = _replay_class(
        demo, config, mode, cls, DEMO_DAYS[demo], every_day=False, decisions=decisions
    )
    assert worst == 0.0
    if cls == "Land":
        outcomes = _through_soil_outcomes(decisions)
        assert outcomes, "no pervious surface decided through_soil"
        assert all(found == {True, False} for found in outcomes.values()), outcomes


def test_a_demo_replay_notices_a_difference():
    """A captured Land reply moved by 1e-11 x scale fails the demo replay (it is not vacuous)."""
    network = demo_network("oxford", "config.yml")
    calls = copy.deepcopy(
        [c for c in demo_capture("oxford", "config.yml", 365)["oxford_land"] if c.day < 3]
    )
    scale = node_scale(calls, volume_only=True)
    arc = next(a for c in calls for a in c.arc_calls if a.reply["volume"] > 0)
    arc.reply["volume"] += 1e-11 * scale
    with pytest.raises(Mismatch):
        replay(network, "oxford_land", calls, scale=scale, pollutants=QUANTITY_ONLY)


def test_a_pollutant_mode_demo_replay_notices_a_difference():
    """A pollutant column of a captured WWTW reply, moved by 1e-11 x scale, fails the replay."""
    network = demo_network("oxford", "config_no_deposition.yml")
    calls = copy.deepcopy(
        [
            c
            for c in demo_capture("oxford", "config_no_deposition.yml", 365)["oxford_wwtw"]
            if c.day < 20
        ]
    )
    scale = node_scale(calls, volume_only=False)
    reply, key = next(
        (c.reply, k)
        for c in calls
        if c.reply is not None
        for k, v in c.reply.items()
        if k != "volume" and v > 0
    )
    reply[key] += 1e-11 * scale
    with pytest.raises(Mismatch):
        replay(network, "oxford_wwtw", calls, scale=scale)


@pytest.mark.slow
@pytest.mark.parametrize(("demo", "config", "mode", "cls"), DEMO_PAIRS, ids=DEMO_IDS)
def test_demo_nodes_replay_wsimod_on_every_day(demo, config, mode, cls):
    """The same on every captured day (oxford's whole 1456-day run)."""
    _, worst = _replay_class(demo, config, mode, cls, FULL_DAYS[demo], every_day=True)
    assert worst == 0.0


# --- the user Decay arcs (Task 11; spec, Verification 1, the replay harness) ------------------
ARC_PAIRS = [(case, arc) for case, spec in CASES.items() if spec.wsimod for arc in spec.replay_arcs]
ARC_IDS = [f"{case}-{arc}" for case, arc in ARC_PAIRS]
# Task 10 / R1: decay_arc's four arcs each carry one direction, so X41's precondition holds
# trivially there; decay_arc_mixed carries both and stays inside it.
PRECONDITION_PAIRS = [(c, a) for c, a in ARC_PAIRS if c in ("decay_arc", "decay_arc_mixed")]
PRECONDITION_IDS = [f"{c}-{a}" for c, a in PRECONDITION_PAIRS]
# W19. If a case's captured days broke X41's precondition (D21), the term of those days could
# differ from WSIMOD's in its last bits, and only there: the order of its summands is lost, while
# the values are exact. The bound is the rounding of summing the same ``m`` terms in another
# order, ``m * eps`` of their absolute sum. No current case breaks the precondition.
ORDER_UNKNOWN_EPS = torch.finfo(torch.float64).eps


@functools.cache
def captured_arcs(case: str):
    spec = CASES[case]
    model = wsimod_model(spec.build())
    with capture_arc_calls(model, spec.replay_arcs) as days:
        record_wsimod_run(model, days=spec.days)
    return days


def arc_spec(network, name):
    return next(a for a in network.arcs if a.name == name)


@functools.cache
def replayed(case: str, arc: str):
    """The case's pollutants replay of one arc, with its day contexts."""
    network = read_wsimod_config(CASES[case].build())
    contexts: list = []
    replay_arc(network, arc, captured_arcs(case)[arc], scale=CASES[case].scale, contexts=contexts)
    return network, contexts


@pytest.mark.parametrize(("case", "arc"), ARC_PAIRS, ids=ARC_IDS)
def test_arc_replay_matches_wsimod(case, arc):
    """The noodl arc, alone between scripted ports, makes WSIMOD's port calls and returns its
    replies; every row, reply, accumulator and per-request ``diff`` is bitwise WSIMOD's (bound 0),
    in both modes. A DecayArc's ``in_port`` is the node whose temperature it reads: a catchment
    among them (``src2_to_j``)."""
    spec = CASES[case]
    days = captured_arcs(case)[arc]
    assert len(days) == spec.days
    assert any(call.method.startswith("send_") for day in days for call in day.calls)
    network = read_wsimod_config(spec.build())
    for mode in spec.modes:
        pollutants = network.pollutants if mode == "pollutants" else QUANTITY_ONLY
        replay_arc(network, arc, days, scale=spec.scale, pollutants=pollutants)


def test_a_decay_arc_replay_reads_its_in_ports_temperature():
    """The arc decays at the temperature of its ``in_port`` on that day, a catchment's included
    (``src2_to_j``'s ``in_port`` is ``src2``, which lists no temperature in its
    ``configured_parameters``): the replay sets it on the scripted port from the configuration's
    ``("temperature", date)`` row, so a changed row fails that day's first call (``n = 0``: the
    decayed request reaches ``j`` at once)."""
    case, arc = "decay_arc", "src2_to_j"
    days = captured_arcs(case)[arc]
    network = read_wsimod_config(CASES[case].build())
    source = next(n for n in network.nodes if n.name == arc_spec(network, arc).in_port)
    assert source.cls == "Catchment"
    day = next(d for d in days if d.entry_diffs and d.entry_diffs[0][1]["phosphate"] > 0)
    source.data[("temperature", network.dates[day.day])] += 1.0
    with pytest.raises(Mismatch, match=rf"arc 'src2_to_j' day {day.day} call 0"):
        replay_arc(network, arc, days, scale=CASES[case].scale)


def event_on(ctx, arc: str) -> bool:
    return bool(ctx.events.get("decay_arc_order_unknown", {}).get(arc, False))


@pytest.mark.parametrize(("case", "arc"), PRECONDITION_PAIRS, ids=PRECONDITION_IDS)
def test_the_decay_arcs_cases_hold_the_order_precondition(case, arc):
    """R1 (X41, D21): on every captured day no due request waits in both directions, by
    WSIMOD's own queue, and noodl's ``"decay_arc_order_unknown"`` is false on every day. The
    mixed arc holds requests in both directions, none of them due at the end of a day."""
    days = captured_arcs(case)[arc]
    network, contexts = replayed(case, arc)
    n = arc_spec(network, arc).number_of_timesteps
    assert all(precondition_holds(day, n) for day in days)
    assert not any(event_on(ctx, arc) for ctx in contexts)
    if case == "decay_arc_mixed":
        for day in days:
            assert {r["direction"] for r in day.queue_at_end} == {"push", "pull"}, day.day


def test_the_precondition_check_sees_a_due_request_in_both_directions():
    """``precondition_holds`` is not vacuously true: a due request (``time`` 0) waiting in each
    direction breaks it, one direction does not, and an arc with ``n = 0`` queues nothing."""

    def day(*queue):
        return ArcDay(day=0, queue_at_end=[{"direction": d, "time": t} for d, t in queue])

    assert not precondition_holds(day(("push", 0), ("pull", 0)), 2)
    assert precondition_holds(day(("push", 0), ("push", 0), ("pull", 1)), 2)
    assert precondition_holds(day(("push", 1), ("pull", 1)), 2)
    assert precondition_holds(day(("push", 0), ("pull", 0)), 0)


def rewindowed(network, day: ArcDay) -> torch.Tensor:
    """Day d's term as WSIMOD's captured ``diff``s give it (X24): the entries of day d in call
    order, then the end-of-day ``diff``s of day d in list order, summed from empty."""
    p = network.pollutants
    total = empty(p, ())
    for _, diff in [*day.entry_diffs, *day.end_diffs]:
        total = sum_vqip(p, total, to_vqip(p, diff))
    return total


@pytest.mark.parametrize(("case", "arc"), ARC_PAIRS, ids=ARC_IDS)
def test_the_decay_arc_term_equals_wsimods_rewindowed_diffs(case, arc):
    """X24, R6: noodl's ``"DecayArc.decayed"``/``"DecayArcAlt.decayed"`` for day d equals,
    exactly, the day's entry ``diff``s in call order then its end-of-day ``diff``s in list
    order. A day that broke the precondition would fall back to W19's named bound
    (``ORDER_UNKNOWN_EPS``); none does."""
    network, contexts = replayed(case, arc)
    spec = arc_spec(network, arc)
    term = f"{spec.cls}.decayed"
    days = captured_arcs(case)[arc]
    assert len(contexts) == len(days)
    assert any(float(rewindowed(network, day)[1]) > 0 for day in days)  # the arc decays
    for day, ctx in zip(days, contexts, strict=True):
        got = ctx.terms.get(term, empty(network.pollutants, ()))
        expected = rewindowed(network, day)
        if precondition_holds(day, spec.number_of_timesteps):
            assert torch.equal(got, expected), (day.day, got, expected)
        else:  # W19: the summation order is unknown, so the last bits may differ
            terms = [to_vqip(network.pollutants, d) for _, d in [*day.entry_diffs, *day.end_diffs]]
            magnitude = sum(t.abs() for t in terms)
            assert bool(
                ((got - expected).abs() <= len(terms) * ORDER_UNKNOWN_EPS * magnitude).all()
            )


@pytest.mark.parametrize(("case", "arc"), ARC_PAIRS, ids=ARC_IDS)
def test_wsimods_total_decayed_is_its_own_window(case, arc):
    """The harness check (X24): WSIMOD's ``total_decayed`` before each ``end_timestep`` equals the
    end-of-day ``diff``s of day d - 1 then the entry ``diff``s of day d, summed from empty in
    that order, bitwise; after it, the end-of-day ``diff``s of day d. That window differs from
    the term's (the entries of d, then the end of d), which is why the term is rebuilt."""
    p = read_wsimod_config(CASES[case].build()).pollutants

    def total(*parts) -> torch.Tensor:
        out = empty(p, ())
        for _, diff in [d for part in parts for d in part]:
            out = sum_vqip(p, out, to_vqip(p, diff))
        return out

    days = captured_arcs(case)[arc]
    previous: list = []
    for day in days:
        before = total(previous, day.entry_diffs)
        assert torch.equal(before, to_vqip(p, day.total_decayed_before_reset)), day.day
        assert torch.equal(total(day.end_diffs), to_vqip(p, day.total_decayed_after)), day.day
        previous = day.end_diffs
    assert any(float(to_vqip(p, d.total_decayed_before_reset)[1]) > 0 for d in days)


@pytest.mark.parametrize("where", ["entry", "end"])
def test_the_arc_replay_notices_a_difference(where):
    """A captured ``diff`` moved by one ulp is reported, naming the arc, the day and the call or
    row."""
    case, arc = "decay_arc", "q"
    days = copy.deepcopy(captured_arcs(case)[arc])
    network = read_wsimod_config(CASES[case].build())
    scale = CASES[case].scale
    if where == "entry":
        day = next(d for d in days if d.entry_diffs and d.entry_diffs[0][1]["phosphate"] > 0)
        index, diff = day.entry_diffs[0]
        expected = rf"arc 'q' day {day.day} entry call {index}"
    else:
        day = next(d for d in days if d.end_diffs and d.end_diffs[0][1]["phosphate"] > 0)
        row, diff = day.end_diffs[0]
        expected = rf"arc 'q' day {day.day} end row {re.escape(str(row))}"
    diff["phosphate"] = math.nextafter(diff["phosphate"], math.inf)
    replay_arc(network, arc, captured_arcs(case)[arc], scale=scale)  # the capture itself passes
    with pytest.raises(Mismatch, match=expected):
        replay_arc(network, arc, days, scale=scale)


def test_capturing_does_not_perturb_wsimods_run():
    """The same run with and without the capture gives the same ``record_wsimod_run`` output,
    for an arc capture (``decay_arc``) and a node capture (``three_out_arcs``)."""

    def run(case, capture=None, names=()):
        spec = CASES[case]
        model = wsimod_model(spec.build())
        if capture is None:
            return record_wsimod_run(model, days=spec.days)
        with capture(model, names):
            return record_wsimod_run(model, days=spec.days)

    for case, capture, names in (
        ("decay_arc", capture_arc_calls, CASES["decay_arc"].replay_arcs),
        ("three_out_arcs", capture_node_calls, CASES["three_out_arcs"].replay),
    ):
        plain, captured_run = run(case), run(case, capture, names)
        assert plain.printed == captured_run.printed
        for name in ("arc_flow", "arc_quality", "tank_storage", "tank_quality", "surface_quality"):
            assert torch.allclose(
                getattr(plain, name), getattr(captured_run, name), rtol=0, atol=0, equal_nan=True
            )
