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

import pytest

pytest.importorskip("wsimod")

import re  # noqa: E402

import torch  # noqa: E402

from noodl.apps.wsimod.config import read_wsimod_config  # noqa: E402
from noodl.apps.wsimod.engine import Ctx, Engine, make_node  # noqa: E402
from noodl.apps.wsimod.vqip import QUANTITY_ONLY  # noqa: E402
from tests.apps.wsimod._replay import Mismatch, replay  # noqa: E402
from tests.apps.wsimod._stepping import step_days  # noqa: E402
from tests.verification._wsimod_cases import CASES  # noqa: E402
from tests.verification._wsimod_reference import (  # noqa: E402
    capture_node_calls,
    record_wsimod_run,
    wsimod_model,
    wsimod_reference,
)

PAIRS = [(case, name) for case, spec in CASES.items() if spec.wsimod for name in spec.replay]


@functools.cache
def captured(case: str):
    spec = CASES[case]
    model = wsimod_model(spec.build())
    with capture_node_calls(model, spec.replay) as calls:
        record_wsimod_run(model, days=spec.days)
    return calls


@pytest.mark.parametrize(("case", "name"), PAIRS, ids=[f"{c}-{n}" for c, n in PAIRS])
def test_replay_matches_wsimod(case, name):
    spec = CASES[case]
    calls = captured(case)[name]
    assert calls, f"WSIMOD made no call into {name!r}"
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
        c
        for c in captured("reservoir_over_capacity")["res"]
        if c.method == "make_abstractions"
    ]
    asked = [x.vqip["volume"] for c in over for x in c.arc_calls if x.method == "send_pull_request"]
    assert asked[0] == 0.0 and max(asked) > 0.0


def _noodl_event_days(case: str, condition: str, node: str) -> set[int]:
    spec = CASES[case]
    network = read_wsimod_config(spec.build())
    days = set()
    for day, ctx, _ in step_days(network, network.pollutants, spec.days):
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
    for day, _, engine in step_days(network, network.pollutants, spec.days):
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
    for day, _, engine in step_days(network, network.pollutants, spec.days):
        ours_in = engine.arcs["basin_to_sewer"].flow_in
        assert float(engine.arcs["sewer_to_sea"].flow_in) == float(ours_in) == float(inflow[day])


def _arc_flows(network, days: int) -> dict[str, list[float]]:
    flows: dict[str, list[float]] = {}
    for _, _, engine in step_days(network, network.pollutants, days):
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
