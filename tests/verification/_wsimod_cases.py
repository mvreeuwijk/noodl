"""Purpose-built WSIMOD cases (D8): small models that exercise what the demos skip.

Each case is a mapping in ``Model.load``'s post-read form, with inline ``data_input_dict``
keyed ``(variable, date string)``. WSIMOD builds it by ``Model.load``'s own steps
(``_wsimod_reference.wsimod_model``); noodl by ``read_wsimod_config(mapping)``. A case is added
in the first task where every class it needs exists.

``scale`` is the largest volume in the case's forcing: the replay and parity bounds are
``1e-12 * scale`` (spec, Verification).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

__all__ = ["CASES", "Case"]


@dataclass(frozen=True)
class Case:
    """A purpose-built case (D8). A case with a River declares pollutant lists without nitrate."""

    build: Callable[[], dict]  # the mapping
    days: int  # 30-120
    scale: float  # the largest volume in the forcing
    replay: tuple[str, ...] = ()  # nodes the replay test captures
    modes: tuple[str, ...] = ("quantity", "pollutants")
    wsimod: bool = True  # False: noodl-only (a WSIMOD crash), Task 17


# A short pollutant list: one additive, one more additive, one non-additive.
_POLLUTANTS = {
    "pollutants": ["phosphate", "ammonia", "temperature"],
    "additive_pollutants": ["phosphate", "ammonia"],
    "non_additive_pollutants": ["temperature"],
}
_FLOW_MAX = 50.0


def _dates(n: int) -> list[str]:
    start = datetime(2000, 1, 1)
    return [(start + timedelta(days=k)).strftime("%Y-%m-%d") for k in range(n)]


def _catchment(name: str, days: int, seed: int) -> dict:
    """A catchment with flows in [5, 50] and concentrations that vary by day."""
    rng = random.Random(seed)
    data = {}
    for date in _dates(days):
        data[("flow", date)] = round(rng.uniform(5.0, _FLOW_MAX), 6)
        data[("phosphate", date)] = round(rng.uniform(1e-3, 1e-2), 8)
        data[("ammonia", date)] = round(rng.uniform(1e-3, 5e-2), 8)
        data[("temperature", date)] = round(rng.uniform(4.0, 20.0), 4)
    return {"name": name, "type_": "Catchment", "data_input_dict": data}


def _arc(name: str, in_port: str, out_port: str, **values) -> dict:
    return {"name": name, "type_": "Arc", "in_port": in_port, "out_port": out_port, **values}


_THREE_OUT_ARCS_DAYS = 40


def _three_out_arcs() -> dict:
    """Two catchments push into one junction that shares over three outlets by capacity and
    preference. The second catchment's push finds the junction partly used, so on days when
    the two flows sum to more than 60 the push is over-subscribed (the capacity switch,
    ``nodes.py:446-450``), and the preference shares overfill an arc, so the loop polls again."""
    days = _THREE_OUT_ARCS_DAYS
    return {
        **_POLLUTANTS,
        "dates": _dates(days),
        "nodes": [
            _catchment("upland", days, 1),
            _catchment("lowland", days, 2),
            {"name": "junction", "type_": "Node"},
            {"name": "outlet_a", "type_": "Waste"},
            {"name": "outlet_b", "type_": "Waste"},
            {"name": "outlet_c", "type_": "Waste"},
        ],
        "arcs": [
            _arc("upland_to_junction", "upland", "junction"),
            _arc("lowland_to_junction", "lowland", "junction"),
            _arc("junction_to_a", "junction", "outlet_a", capacity=20.0, preference=1.0),
            _arc("junction_to_b", "junction", "outlet_b", capacity=30.0, preference=2.0),
            _arc("junction_to_c", "junction", "outlet_c", capacity=10.0, preference=3.0),
        ],
        "orchestration": [{"Catchment": "route"}],
    }


_NODE_CHAIN_DAYS = 40


def _node_chain() -> dict:
    """A catchment through three pass-through nodes to an outlet, with capacities narrowing
    downstream: every push asks the chain's checks recursively (``check_basic``,
    ``nodes.py:480-508``), and flows above 25 come back as unrouted water."""
    days = _NODE_CHAIN_DAYS
    return {
        **_POLLUTANTS,
        "dates": _dates(days),
        "nodes": [
            _catchment("headwater", days, 3),
            {"name": "reach_1", "type_": "Node"},
            {"name": "reach_2", "type_": "Node"},
            {"name": "reach_3", "type_": "Node"},
            {"name": "sea", "type_": "Waste"},
        ],
        "arcs": [
            _arc("headwater_to_1", "headwater", "reach_1", capacity=40.0),
            _arc("reach_1_to_2", "reach_1", "reach_2", capacity=35.0),
            _arc("reach_2_to_3", "reach_2", "reach_3", capacity=30.0),
            _arc("reach_3_to_sea", "reach_3", "sea", capacity=25.0),
        ],
        "orchestration": [{"Catchment": "route"}],
    }


def _largest_volume(mapping: dict) -> float:
    """The largest flow in a case's forcing: its ``scale``."""
    return max(
        value
        for node in mapping["nodes"]
        for (variable, _), value in node.get("data_input_dict", {}).items()
        if variable == "flow"
    )


_STORAGE_DAYS = 40


def _store(name: str, type_: str, **values) -> dict:
    return {"name": name, "type_": type_, **values}


def _stocked(volume: float) -> dict:
    """An ``initial_storage`` VQIP, so pollutants move with the water."""
    return {
        "volume": volume,
        "phosphate": 0.002 * volume,
        "ammonia": 0.01 * volume,
        "temperature": 11.0,
    }


def _storage_case(nodes: list[dict], arcs: list[dict], orchestration: list[dict]) -> dict:
    return {
        **_POLLUTANTS,
        "dates": _dates(_STORAGE_DAYS),
        "nodes": nodes,
        "arcs": arcs,
        "orchestration": orchestration,
    }


def _forced_negative_excess() -> dict:
    """A reservoir that starts above its capacity: its excess is negative before the clamp, a
    push to it is refused, and ``Storage.distribute`` forces the unpushed rest back in."""
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _catchment("src", days, 11),
            _store("head", "Node"),
            _store("res", "Reservoir", capacity=12.0, initial_storage=_stocked(15.0)),
            _store("sink", "Waste"),
        ],
        [
            _arc("src_to_head", "src", "head"),
            _arc("head_to_res", "head", "res", capacity=30.0),
            _arc("res_to_sink", "res", "sink", capacity=4.0),
        ],
        [{"Catchment": "route"}, {"Reservoir": "distribute"}],
    )


# Five sources, each behind an arc of capacity ``avail``; the reservoir asks for its whole
# capacity (188) and the arcs with the largest preference x avail run dry one round at a time,
# so the fifth round still has a deficit (found by simulating ``nodes.py:345-407``).
_MAXITER_AVAIL = (1.0, 4.0, 71.0, 60.0, 52.0)
_MAXITER_PREFERENCE = (4.0, 8.0, 11.0, 53.0, 13.0)


def _maxiter_pull() -> dict:
    wells = [f"well_{i}" for i in range(len(_MAXITER_AVAIL))]
    return _storage_case(
        [
            *(_store(n, "Storage", capacity=1e4, initial_storage=_stocked(5000.0)) for n in wells),
            _store("res", "Reservoir", capacity=sum(_MAXITER_AVAIL)),
            _store("sink", "Waste"),
        ],
        [
            *(
                _arc(f"{n}_to_res", n, "res", capacity=a, preference=p)
                for n, a, p in zip(wells, _MAXITER_AVAIL, _MAXITER_PREFERENCE, strict=True)
            ),
            _arc("res_to_sink", "res", "sink"),
        ],
        [{"Reservoir": "make_abstractions"}, {"Reservoir": "distribute"}],
    )


def _pull_and_push_arcs() -> dict:
    """``a`` is joined to ``b`` and to the reservoir ``r`` by a ``PullArc`` and a ``PushArc``
    each: pushes go through the push arcs only, pulls through the pull arcs only, and the
    refused direction answers as WSIMOD's deny handlers do."""
    return _storage_case(
        [
            _store("a", "Storage", capacity=100.0, initial_storage=_stocked(100.0)),
            _store("b", "Storage", capacity=50.0),
            _store("r", "Reservoir", capacity=30.0),
            _store("sink", "Waste"),
        ],
        [
            _arc("push_ab", "a", "b", type_="PushArc", capacity=7.0),
            _arc("pull_ab", "a", "b", type_="PullArc", capacity=5.0),
            _arc("pull_ar", "a", "r", type_="PullArc", capacity=4.0),
            _arc("push_ar", "a", "r", type_="PushArc", capacity=6.0),
            _arc("b_to_sink", "b", "sink", capacity=3.0),
            _arc("r_to_b", "r", "b", capacity=2.0),
        ],
        [
            {"Reservoir": "make_abstractions"},
            {"Storage": "distribute"},
            {"Reservoir": "distribute"},
        ],
    )


def _user_alt_queue_arc() -> dict:
    """A storage distributes through an ``AltQueueArc`` of two days' travel into another."""
    return _storage_case(
        [
            _store("a", "Storage", capacity=2500.0, initial_storage=_stocked(2000.0)),
            _store("b", "Storage", capacity=200.0),
            _store("sink", "Waste"),
        ],
        [
            _arc("a_to_b", "a", "b", type_="AltQueueArc", number_of_timesteps=2, capacity=15.0),
            _arc("b_to_sink", "b", "sink", capacity=9.0),
        ],
        [{"Storage": "distribute"}],
    )


def _storage_distribute() -> dict:
    """A catchment fills a reservoir; the reservoir, a groundwater (with a residence time and
    an infiltration that finds no sewer) and a storage each distribute downstream, the
    unpushed rest forced back."""
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _catchment("src", days, 12),
            _store("head", "Node"),
            _store("res", "Reservoir", capacity=50.0, initial_storage=_stocked(20.0)),
            _store("buf", "Storage", capacity=30.0),
            _store(
                "aquifer",
                "Groundwater",
                capacity=200.0,
                initial_storage=_stocked(60.0),
                residence_time=5.0,
                infiltration_threshold=0.2,
                infiltration_pct=0.04,
            ),
            _store("sink", "Waste"),
        ],
        [
            _arc("src_to_head", "src", "head"),
            _arc("head_to_res", "head", "res", capacity=40.0),
            _arc("res_to_buf", "res", "buf", capacity=20.0),
            _arc("res_to_aquifer", "res", "aquifer", capacity=10.0),
            _arc("buf_to_sink", "buf", "sink", capacity=12.0),
            _arc("aquifer_to_sink", "aquifer", "sink", capacity=6.0),
        ],
        [
            {"Catchment": "route"},
            {"Reservoir": "distribute"},
            {"Groundwater": "infiltrate"},
            {"Groundwater": "distribute"},
            {"Storage": "distribute"},
        ],
    )


def _queue_groundwater() -> dict:
    """A reservoir feeds a ``QueueGroundwater`` with time-area ``{0: 0.5, 2: 0.5}``; a second
    reservoir pulls from it through ``pull_set_active`` (pro rata over the queue), and the
    groundwater distributes its active storage back into that reservoir."""
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _catchment("src", days, 13),
            _store("head", "Node"),
            _store("feed", "Reservoir", capacity=60.0),
            _store(
                "gw",
                "QueueGroundwater",
                capacity=500.0,
                initial_storage=_stocked(30.0),
                timearea={0: 0.5, 2: 0.5},
            ),
            _store("res", "Reservoir", capacity=25.0),
            _store("sink", "Waste"),
        ],
        [
            _arc("src_to_head", "src", "head"),
            _arc("head_to_feed", "head", "feed", capacity=45.0),
            _arc("feed_to_gw", "feed", "gw", capacity=30.0),
            _arc("gw_to_res", "gw", "res", capacity=12.0),
            _arc("res_to_sink", "res", "sink", capacity=8.0),
        ],
        [
            {"Catchment": "route"},
            {"Reservoir": "make_abstractions"},
            {"QueueGroundwater": "distribute"},
            {"Reservoir": "distribute"},
        ],
    )


def _reservoir_over_capacity() -> dict:
    """D14: a reservoir started at 12 in a capacity of 10 has no excess, pulls nothing, is
    never clamped, and only ``distribute`` brings it down."""
    return _storage_case(
        [
            _store("supply", "Storage", capacity=1e4, initial_storage=_stocked(5000.0)),
            _store("res", "Reservoir", capacity=10.0, initial_storage=_stocked(12.0)),
            _store("sink", "Waste"),
        ],
        [
            _arc("supply_to_res", "supply", "res", capacity=3.0),
            _arc("res_to_sink", "res", "sink", capacity=5.0),
        ],
        [{"Reservoir": "make_abstractions"}, {"Reservoir": "distribute"}],
    )


def _river_reservoir_environmental() -> dict:
    """A river reservoir receives water from a catchment through a node, spills downstream
    (counting towards its environmental flow) and satisfies the rest of the flow
    (``satisfy_environmental``); no ``River``."""
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _catchment("src", days, 14),
            _store("head", "Node"),
            _store(
                "rr",
                "RiverReservoir",
                capacity=60.0,
                initial_storage=_stocked(30.0),
                environmental_flow=20.0,
            ),
            _store("stream", "Node"),
            _store("sea", "Waste"),
        ],
        [
            _arc("src_to_head", "src", "head"),
            _arc("head_to_rr", "head", "rr", capacity=35.0),
            _arc("rr_to_stream", "rr", "stream", capacity=14.0),
            _arc("stream_to_sea", "stream", "sea"),
        ],
        [
            {"Catchment": "route"},
            {"RiverReservoir": "make_abstractions"},
            {"RiverReservoir": "satisfy_environmental"},
        ],
    )


def _river(name: str, *, damp: float = 0.5, mrf: float = 0.0, length=20000.0, **values) -> dict:
    """A ``River`` whose routing coefficient is well below 1 (``kt = damp * length / 17280``)."""
    return _store(name, "River", length=length, damp=damp, mrf=mrf, **values)


def _river_chain() -> dict:
    """Rivers in three levels with ties (``river_discharge_order``, ``model.py:697-807``).

    Two independent chains reach the sea: ``r_head`` (level 3, ``damp = 0`` so it discharges
    everything) -> ``r_mid`` (2, ``mrf > 0``) -> ``r_low`` (1), and ``r_side`` (2) -> ``r_end``
    (1), so each of levels 1 and 2 holds a tie that changes no value (W13). ``r_side`` is fed
    through a ``Node``. A ``Reservoir``, which empties 25 a day into the sea,
    pulls from ``r_low`` (which pulls ``r_mid``, which pulls ``r_head``) and from ``r_end``
    (which pulls ``r_side`` and the node), so the river pull check recurses three deep, with
    ``mrf`` and with the pull split over two arcs. ``r_lost`` links to no ``Waste`` and never
    distributes. Pollutants without nitrate (River biochemistry is 3b).
    """
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _catchment("src_head", days, 21),
            _catchment("src_side", days, 23),
            _catchment("src_lost", days, 24),
            _store("feed", "Node"),
            _river("r_head", damp=0.0, initial_storage=_stocked(8.0)),
            _river("r_mid", damp=0.4, mrf=6.0, initial_storage=_stocked(3.0)),
            _river("r_low", damp=0.8, mrf=3.0, length=12000.0, initial_storage=_stocked(10.0)),
            _river("r_side", damp=0.3, initial_storage=_stocked(12.0)),
            _river("r_end", damp=0.6, mrf=4.0, initial_storage=_stocked(6.0)),
            _river("r_lost", initial_storage=_stocked(5.0)),
            _store("pond", "Node"),
            _store("res", "Reservoir", capacity=60.0, initial_storage=_stocked(10.0)),
            _store("sea", "Waste"),
        ],
        [
            _arc("head_in", "src_head", "r_head"),
            _arc("side_in", "src_side", "feed"),
            _arc("feed_to_side", "feed", "r_side"),
            _arc("lost_in", "src_lost", "r_lost"),
            _arc("head_to_mid", "r_head", "r_mid"),
            _arc("mid_to_low", "r_mid", "r_low"),
            _arc("low_to_sea", "r_low", "sea"),
            _arc("side_to_end", "r_side", "r_end"),
            _arc("end_to_sea", "r_end", "sea"),
            _arc("lost_to_pond", "r_lost", "pond"),
            _arc("low_to_res", "r_low", "res"),
            _arc("end_to_res", "r_end", "res"),
            _arc("res_to_sea", "res", "sea", capacity=25.0),
        ],
        [
            {"Catchment": "route"},
            {"Reservoir": "make_abstractions"},
            {"River": "calculate_discharge"},
            {"Reservoir": "distribute"},
        ],
    )


def _catchment_abstraction() -> dict:
    """A reservoir abstracts from a catchment through the pull path (``pull_check`` and
    ``pull_set``); the catchment then routes what is left."""
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _catchment("src", days, 15),
            _store("res", "Reservoir", capacity=40.0, initial_storage=_stocked(5.0)),
            _store("sink", "Waste"),
            _store("sea", "Waste"),
        ],
        [
            _arc("src_to_res", "src", "res"),
            _arc("src_to_sea", "src", "sea"),
            _arc("res_to_sink", "res", "sink", capacity=20.0),
        ],
        [
            {"Reservoir": "make_abstractions"},
            {"Catchment": "route"},
            {"Reservoir": "distribute"},
        ],
    )


_QUEUE_ARC_DAYS = 30
_QUEUE_ARC_POLLUTANTS = {  # one additive, one non-additive (W21)
    "pollutants": ["phosphate", "temperature"],
    "additive_pollutants": ["phosphate"],
    "non_additive_pollutants": ["temperature"],
}
# The flows of the two catchments by day (0 on every other day). ``src`` pushes through node
# ``a``, whose only out-arc is the QueueArc ``q`` (n = 2): day 0 is the clip below
# FLOAT_ACCURACY (below), days 1 and 5 a gap longer than n + 1, day 9 a push of 1e-12 on a day
# when day 7's request is due, day 10 the push that delivers it. ``src2`` fills the reservoir
# ``feed`` on days 1 and 7, whose ``distribute`` pushes through node ``b`` (the arc to ``a``
# and an arc to the sink): on day 7 the due day-5 request meets a full ``res``, and its
# backflow makes ``b``'s ``push_distributed`` run a second round.
_QUEUE_ARC_SRC = {0: 80.0, 1: 3.0, 5: 10.0, 9: 1e-12, 10: 2.5, 14: 3.5, 15: 1.25, 20: 4.5, 26: 2.0}
_QUEUE_ARC_SRC2 = {1: 2.0, 7: 5.0, 20: 3.0, 21: 6.0}
# res starts 9.998e-12 below its capacity of 6. On day 0 the arc src -> a clips the push of 80
# to that spare, and rounding at 80 makes it 1.00044e-11 (above FLOAT_ACCURACY); q clips it
# again to the spare, 9.998e-12, which enters the queue and is removed by update_queue
# (arcs.py:483-485): the update-time drop. Found by simulating arcs.py:382-390 in floats.
_QUEUE_ARC_RES_START = 5.999999999990002


def _queue_catchment(name: str, flows: dict[int, float], seed: int) -> dict:
    rng = random.Random(seed)
    data = {}
    for k, date in enumerate(_dates(_QUEUE_ARC_DAYS)):
        data[("flow", date)] = flows.get(k, 0.0)
        data[("phosphate", date)] = round(rng.uniform(1e-3, 1e-2), 8)
        data[("temperature", date)] = round(rng.uniform(4.0, 20.0), 4)
    return {"name": name, "type_": "Catchment", "data_input_dict": data}


def _queue_stocked(volume: float) -> dict:
    return {"volume": volume, "phosphate": 0.002 * volume, "temperature": 11.0}


def _queue_arc() -> dict:
    """W21: the push path of a ``QueueArc`` (``q``, two days) with two senders upstream of
    its in-port ``a`` (see the flows above), into a reservoir of capacity 6 that rejects; and
    the pull path of a second ``QueueArc`` (``q_pull``, one day) through which the reservoir
    ``basin`` abstracts from the storage ``well`` (slot ``Abstraction``), so each pull
    hands over the previous day's water."""
    return {
        **_QUEUE_ARC_POLLUTANTS,
        "dates": _dates(_QUEUE_ARC_DAYS),
        "nodes": [
            _queue_catchment("src", _QUEUE_ARC_SRC, 21),
            _queue_catchment("src2", _QUEUE_ARC_SRC2, 22),
            {"name": "a", "type_": "Node"},
            {"name": "b", "type_": "Node"},
            {"name": "j", "type_": "Node"},
            _store("feed", "Reservoir", capacity=100.0),
            _store(
                "res",
                "Reservoir",
                capacity=6.0,
                initial_storage=_queue_stocked(_QUEUE_ARC_RES_START),
            ),
            _store("well", "Storage", capacity=1000.0, initial_storage=_queue_stocked(500.0)),
            {
                "name": "basin",
                "type_": "Abstraction",
                "node_type_override": "Reservoir",
                "capacity": 10.0,
                "initial_storage": _queue_stocked(2.0),
            },
            _store("sink", "Waste"),
        ],
        "arcs": [
            _arc("src_to_a", "src", "a"),
            _arc("src2_to_j", "src2", "j"),
            _arc("j_to_feed", "j", "feed"),
            _arc("feed_to_b", "feed", "b"),
            _arc("b_to_a", "b", "a", preference=10.0),
            _arc("b_to_sink", "b", "sink", capacity=1.0),
            _arc("q", "a", "res", type_="QueueArc", number_of_timesteps=2, capacity=6.0),
            _arc("res_to_sink", "res", "sink", capacity=1.5),
            _arc("q_pull", "well", "basin", type_="QueueArc", number_of_timesteps=1, capacity=1.0),
            _arc("basin_to_sink", "basin", "sink", capacity=2.0),
        ],
        "orchestration": [
            {"Catchment": "route"},
            {"Reservoir": "distribute"},
            {"Abstraction": "make_abstractions"},
            {"Abstraction": "distribute"},
        ],
    }


def _temperatures(days: int, seed: int) -> dict:
    """Daily air temperatures in [4, 20] for a ``ResidentialDemand``'s ``data_input_dict``."""
    rng = random.Random(seed)
    return {("temperature", date): round(rng.uniform(4.0, 20.0), 4) for date in _dates(days)}


def _distribution_leakage() -> dict:
    """A supply feeds a town through two distributions: ``dist_gw`` (leakage 0.2) leaks into a
    small groundwater that fills, so on some days the leak cannot go anywhere and returns to
    the caller (``distribution.py:40-50``); ``dist_dry`` (leakage 0.1) has no groundwater at
    all, so its whole leak returns. The town pulls through both (a check with no request, then
    the decorated pull sets); the supply runs dry after about two weeks, so the demand falls
    short on later days."""
    return _storage_case(
        [
            _store("supply", "Storage", capacity=1e4, initial_storage=_stocked(150.0)),
            _store("dist_gw", "Distribution", leakage=0.2),
            _store("dist_dry", "Distribution", leakage=0.1),
            _store("aq", "Groundwater", capacity=3.0, residence_time=4.0),
            _store(
                "town",
                "Demand",
                constant_demand=9.0,
                pollutant_load={"phosphate": 0.02, "ammonia": 0.1, "temperature": 15.0},
            ),
            _store("sea", "Waste"),
        ],
        [
            _arc("supply_to_gw", "supply", "dist_gw", capacity=8.0),
            _arc("supply_to_dry", "supply", "dist_dry", capacity=6.0),
            _arc("gw_to_town", "dist_gw", "town", preference=2.0),
            _arc("dry_to_town", "dist_dry", "town"),
            _arc("leak_to_aq", "dist_gw", "aq"),
            _arc("aq_to_sea", "aq", "sea"),
            _arc("town_to_sea", "town", "sea"),
        ],
        [{"Demand": "create_demand"}, {"Groundwater": "distribute"}],
    )


def _unlimited_distribution() -> dict:
    """A town pulls from an ``UnlimitedDistribution`` and a well through arcs of capacity 5 and
    3; the well runs dry, and the demand of 7 then falls short."""
    return _storage_case(
        [
            _store("mains", "UnlimitedDistribution"),
            _store("well", "Storage", capacity=100.0, initial_storage=_stocked(40.0)),
            _store("town", "Demand", constant_demand=7.0),
            _store("sea", "Waste"),
        ],
        [
            _arc("mains_to_town", "mains", "town", capacity=5.0),
            _arc("well_to_town", "well", "town", capacity=3.0, preference=2.0),
            _arc("town_to_sea", "town", "sea"),
        ],
        [{"Demand": "create_demand"}],
    )


def _demand_constant_and_loads() -> dict:
    """Two demands with a constant demand and a ``pollutant_load`` push through the ``default``
    component to a ``Waste``: ``town`` through one arc, ``village`` over two arcs whose
    capacities (2.5) fall short of the demand, so water is left unpushed every day."""
    load = {"phosphate": 0.02, "ammonia": 0.1, "temperature": 15.0}
    return _storage_case(
        [
            _store("supply", "Storage", capacity=1e4, initial_storage=_stocked(60.0)),
            _store("town", "Demand", constant_demand=4.0, pollutant_load=load),
            _store("village", "Demand", constant_demand=4.0, pollutant_load=load),
            _store("sea", "Waste"),
            _store("sea_a", "Waste"),
            _store("sea_b", "Waste"),
        ],
        [
            _arc("supply_to_town", "supply", "town"),
            _arc("supply_to_village", "supply", "village"),
            _arc("town_to_sea", "town", "sea"),
            _arc("village_to_a", "village", "sea_a", capacity=1.5, preference=1.0),
            _arc("village_to_b", "village", "sea_b", capacity=1.0, preference=2.0),
        ],
        [{"Demand": "create_demand"}],
    )


def _residential_demand() -> dict:
    """A ``ResidentialDemand`` (it needs its own orchestration key) pulls its demand and builds
    the foul water of its population with the mixed temperature; with no ``Land`` or ``Sewer``
    attached, the garden demand is 0 and the foul water has nowhere to go."""
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _store("supply", "Storage", capacity=1e4, initial_storage=_stocked(500.0)),
            _store(
                "homes",
                "ResidentialDemand",
                population=20.0,
                per_capita=0.15,
                pollutant_load={"phosphate": 0.01, "ammonia": 0.05},
                constant_temp=14.0,
                constant_weighting=0.2,
                data_input_dict=_temperatures(days, 21),
            ),
            _store("sea", "Waste"),
        ],
        [_arc("supply_to_homes", "supply", "homes"), _arc("homes_to_sea", "homes", "sea")],
        [{"ResidentialDemand": "create_demand"}],
    )


def _reservoir_spill() -> dict:
    """D14, live: a reservoir pulls its excess through a leaking ``Distribution`` with no
    groundwater, which returns the leak to the caller, so the pull exceeds the excess and the
    surplus spills (``storage.py:983-989``). A town drains the reservoir each day; the tank
    ends each day above capacity until the supply runs dry."""
    return _storage_case(
        [
            _store("supply", "Storage", capacity=1e4, initial_storage=_stocked(60.0)),
            _store("dist", "Distribution", leakage=0.2),
            _store("res", "Reservoir", capacity=10.0, initial_storage=_stocked(5.0)),
            _store("town", "Demand", constant_demand=3.0),
            _store("sea", "Waste"),
        ],
        [
            _arc("supply_to_dist", "supply", "dist"),
            _arc("dist_to_res", "dist", "res", capacity=30.0),
            _arc("res_to_town", "res", "town"),
            _arc("town_to_sea", "town", "sea"),
        ],
        [{"Demand": "create_demand"}, {"Reservoir": "make_abstractions"}],
    )


# --- sewers and treatment works (Task 11) ---------------------------------------------------
def _neutral_wtw(*, percent_solids: float = 0.0002, liquor: float = 0.03, **values) -> dict:
    """A treatment works' neutral process (spec, Pollutants in 3a): every additive pollutant
    treated by ``c_v = 1 - percent_solids - liquor`` with exponent 1 and liquored at ``liquor``."""
    c_v = 1 - percent_solids - liquor
    additive = _POLLUTANTS["additive_pollutants"]
    return {
        "percent_solids": percent_solids,
        "process_parameters": {p: {"constant": c_v, "exponent": 1} for p in additive},
        "liquor_multiplier": {**{p: liquor for p in additive}, "volume": liquor},
        **values,
    }


def _homes(name: str, days: int, seed: int) -> dict:
    """A ``ResidentialDemand`` of 40 people at 0.15 a day (6 a day of foul water)."""
    return _store(
        name,
        "ResidentialDemand",
        population=40.0,
        per_capita=0.15,
        pollutant_load={"phosphate": 0.01, "ammonia": 0.05},
        constant_temp=14.0,
        constant_weighting=0.2,
        data_input_dict=_temperatures(days, seed),
    )


def _sewer_timearea() -> dict:
    """A sewer reached through the ``"Demand"`` tag (a ``ResidentialDemand``'s foul water, spread
    over ``pipe_timearea``, ``sewer.py:178-199``) and the default tag (a catchment's water through
    a node and a reservoir, entering at ``pipe_time``, ``:163-176``). Its outlet is narrower
    than the inflow on wet days, so the sewer fills and its push check limits both senders."""
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _store("supply", "Storage", capacity=1e4, initial_storage=_stocked(800.0)),
            _homes("homes", days, 31),
            _catchment("inflow", days, 32),
            {"name": "junction", "type_": "Node"},
            _store("basin", "Reservoir", capacity=30.0),
            _store(
                "sewer",
                "Sewer",
                capacity=25.0,
                pipe_time=1,
                pipe_timearea={0: 0.5, 1: 0.3, 2: 0.2},
            ),
            _store("sea", "Waste"),
        ],
        [
            _arc("supply_to_homes", "supply", "homes"),
            _arc("homes_to_sewer", "homes", "sewer"),
            _arc("inflow_to_junction", "inflow", "junction"),
            _arc("junction_to_basin", "junction", "basin"),
            _arc("basin_to_sewer", "basin", "sewer", capacity=20.0),
            _arc("sewer_to_sea", "sewer", "sea", capacity=15.0),
        ],
        [
            {"ResidentialDemand": "create_demand"},
            {"Catchment": "route"},
            {"Reservoir": "distribute"},
            {"Sewer": "make_discharge"},
        ],
    )


def _foul_sewers() -> dict:
    """``foul_main`` (a name with "foul", so the ``"Foul"`` slot, ``model.py:658-660``) takes a
    demand's foul water over its time-area diagram and passes it to ``trunk`` (the ``"Sewer"``
    slot) through the default tag at ``trunk``'s pipe time of 2 days. The trunk's outlet takes
    3 a day of the 6, so the trunk fills, then ``foul_main``, then the demand backs up."""
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _store("supply", "Storage", capacity=1e4, initial_storage=_stocked(800.0)),
            _homes("homes", days, 33),
            _store("foul_main", "Sewer", capacity=10.0, pipe_timearea={0: 0.6, 1: 0.4}),
            _store("trunk", "Sewer", capacity=8.0, pipe_time=2),
            _store("sea", "Waste"),
        ],
        [
            _arc("supply_to_homes", "supply", "homes"),
            _arc("homes_to_foul", "homes", "foul_main"),
            _arc("foul_to_trunk", "foul_main", "trunk"),
            _arc("trunk_to_sea", "trunk", "sea", capacity=3.0),
        ],
        [
            {"ResidentialDemand": "create_demand"},
            {"Foul": "make_discharge"},
            {"Sewer": "make_discharge"},
        ],
    )


def _groundwater_infiltrate() -> dict:
    """A groundwater above its threshold infiltrates ``(excess x pct) ** 0.5`` into a sewer
    (``storage.py:231-245``); the sewer's outlet is narrow, so it fills and its push check
    sends part of the infiltration back."""
    return _storage_case(
        [
            _store(
                "gw",
                "Groundwater",
                capacity=100.0,
                initial_storage=_stocked(90.0),
                infiltration_threshold=0.5,
                infiltration_pct=0.05,
                residence_time=20.0,
            ),
            _store("sewer", "Sewer", capacity=1.5, pipe_time=1),
            _store("sea", "Waste"),
        ],
        [
            _arc("gw_to_sewer", "gw", "sewer"),
            _arc("gw_to_sea", "gw", "sea"),
            _arc("sewer_to_sea", "sewer", "sea", capacity=1.0),
        ],
        [
            {"Groundwater": "infiltrate"},
            {"Sewer": "make_discharge"},
            {"Groundwater": "distribute"},
        ],
    )


def _wwtw_storm_tank() -> dict:
    """A catchment's water reaches a ``WWTW`` through a reservoir and a sewer. The works treat
    30 a day; the rest fills the stormwater tank on wet days and drains from it into the spare
    throughput on dry ones (``wtw.py:323-341``, ``:374-407``). The works' outlet takes 26 a
    day, so on full days the rejection is forced into the tank (``"wwtw_couldnt_push"``,
    ``:343-349``), and a reuse demand pulls from the treated water between the two steps
    (``:409-428``)."""
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _catchment("inflow", days, 34),
            {"name": "junction", "type_": "Node"},
            _store("basin", "Reservoir", capacity=40.0),
            _store("combined", "Sewer", capacity=60.0),
            _store(
                "works",
                "WWTW",
                **_neutral_wtw(
                    # 1 - 0.0007 - 0.035 is not 1 - (0.0007 + 0.035): the order shows
                    percent_solids=0.0007,
                    liquor=0.035,
                    treatment_throughput_capacity=30.0,
                    stormwater_storage_capacity=30.0,
                ),
            ),
            _store("reuse", "Demand", constant_demand=2.0),
            _store("river", "Waste"),
            _store("sea", "Waste"),
        ],
        [
            _arc("inflow_to_junction", "inflow", "junction"),
            _arc("junction_to_basin", "junction", "basin"),
            _arc("basin_to_combined", "basin", "combined"),
            _arc("combined_to_works", "combined", "works"),
            _arc("works_to_river", "works", "river", capacity=26.0),
            _arc("works_to_reuse", "works", "reuse"),
            _arc("reuse_to_sea", "reuse", "sea"),
        ],
        [
            {"Catchment": "route"},
            {"Reservoir": "distribute"},
            {"Sewer": "make_discharge"},
            {"WWTW": "calculate_discharge"},
            {"Demand": "create_demand"},
            {"WWTW": "make_discharge"},
        ],
    )


def _fwtw_deficit_sludge() -> dict:
    """An ``FWTW`` fills its service reservoir from a supply that runs dry after about two
    weeks; then the shortfall is created at the previous day's pulled quality
    (``wtw.py:588-596``). A town draws from the reservoir. Liquor and solids go to a small sewer
    whose outlet takes 0.1 a day, so it fills and the rest is unpushed sludge (``:606-611``)."""
    return _storage_case(
        [
            _store("supply", "Storage", capacity=1e4, initial_storage=_stocked(150.0)),
            _store(
                "plant",
                "FWTW",
                **_neutral_wtw(
                    treatment_throughput_capacity=10.0,
                    service_reservoir_storage_capacity=30.0,
                    service_reservoir_initial_storage=10.0,
                ),
            ),
            _store(
                "town",
                "Demand",
                constant_demand=6.0,
                pollutant_load={"phosphate": 0.02, "ammonia": 0.1, "temperature": 15.0},
            ),
            _store("sludge_sewer", "Sewer", capacity=0.5),
            _store("sea", "Waste"),
        ],
        [
            _arc("supply_to_plant", "supply", "plant"),
            _arc("plant_to_town", "plant", "town"),
            _arc("plant_to_sewer", "plant", "sludge_sewer"),
            _arc("town_to_sea", "town", "sea"),
            _arc("sewer_to_sea", "sludge_sewer", "sea", capacity=0.1),
        ],
        [{"FWTW": "treat_water"}, {"Demand": "create_demand"}, {"Sewer": "make_discharge"}],
    )


def _queue_tank_time_zero() -> dict:
    """A sewer with ``pipe_time = 0``: its push enters the queue at time 0 and the same push's
    ``update_queue`` makes it active at once (spec, Flushing a ``QueueTank`` on push), so the
    same day's ``make_discharge`` sends it on."""
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _catchment("inflow", days, 35),
            {"name": "junction", "type_": "Node"},
            _store("basin", "Reservoir", capacity=1e3),
            _store("sewer", "Sewer", capacity=100.0, pipe_time=0),
            _store("sea", "Waste"),
        ],
        [
            _arc("inflow_to_junction", "inflow", "junction"),
            _arc("junction_to_basin", "junction", "basin"),
            _arc("basin_to_sewer", "basin", "sewer"),
            _arc("sewer_to_sea", "sewer", "sea"),
        ],
        [{"Catchment": "route"}, {"Reservoir": "distribute"}, {"Sewer": "make_discharge"}],
    )


def _wtw_zero_input() -> dict:
    """noodl only: WSIMOD raises ``ZeroDivisionError`` in the liquor blend (``wtw.py:176-182``).
    ``plant`` starts with a full service reservoir, so it treats nothing on day 1 and has no
    liquor yet; ``works`` has ``liquor_multiplier["volume"] = 0`` and no input, so its
    denominator is 0 every day (spec, defect table)."""
    return _storage_case(
        [
            _store("supply", "Storage", capacity=100.0),
            _store(
                "plant",
                "FWTW",
                **_neutral_wtw(
                    service_reservoir_storage_capacity=30.0,
                    service_reservoir_initial_storage=30.0,
                ),
            ),
            _store("town", "Demand", constant_demand=5.0),
            _store("works", "WWTW", **_neutral_wtw(liquor=0.0)),
            _store("sea", "Waste"),
        ],
        [
            _arc("supply_to_plant", "supply", "plant"),
            _arc("plant_to_town", "plant", "town"),
            _arc("town_to_sea", "town", "sea"),
            _arc("works_to_sea", "works", "sea"),
        ],
        [
            {"FWTW": "treat_water"},
            {"Demand": "create_demand"},
            {"WWTW": "calculate_discharge"},
            {"WWTW": "make_discharge"},
        ],
    )


CASES: dict[str, Case] = {
    "three_out_arcs": Case(
        build=_three_out_arcs,
        days=_THREE_OUT_ARCS_DAYS,
        scale=_largest_volume(_three_out_arcs()),
        replay=("upland", "lowland", "junction"),
    ),
    "node_chain": Case(
        build=_node_chain,
        days=_NODE_CHAIN_DAYS,
        scale=_largest_volume(_node_chain()),
        replay=("headwater", "reach_1", "reach_2", "reach_3"),
    ),
    "forced_negative_excess": Case(
        build=_forced_negative_excess,
        days=_STORAGE_DAYS,
        scale=_largest_volume(_forced_negative_excess()),
        replay=("res", "head"),
    ),
    "maxiter_pull": Case(
        build=_maxiter_pull,
        days=_STORAGE_DAYS,
        scale=sum(_MAXITER_AVAIL),
        replay=("res",),
    ),
    "pull_and_push_arcs": Case(
        build=_pull_and_push_arcs,
        days=_STORAGE_DAYS,
        scale=100.0,
        replay=("a", "b", "r"),
    ),
    "user_alt_queue_arc": Case(
        build=_user_alt_queue_arc,
        days=_STORAGE_DAYS,
        scale=2000.0,
        replay=("a", "b"),
    ),
    "storage_distribute": Case(
        build=_storage_distribute,
        days=_STORAGE_DAYS,
        scale=_largest_volume(_storage_distribute()),
        replay=("res", "buf", "aquifer"),
    ),
    "queue_groundwater": Case(
        build=_queue_groundwater,
        days=_STORAGE_DAYS,
        scale=_largest_volume(_queue_groundwater()),
        replay=("feed", "gw", "res"),
    ),
    "reservoir_over_capacity": Case(
        build=_reservoir_over_capacity,
        days=_STORAGE_DAYS,
        scale=12.0,
        replay=("res", "supply"),
    ),
    "river_reservoir_environmental": Case(
        build=_river_reservoir_environmental,
        days=_STORAGE_DAYS,
        scale=_largest_volume(_river_reservoir_environmental()),
        replay=("rr", "head"),
    ),
    "catchment_abstraction": Case(
        build=_catchment_abstraction,
        days=_STORAGE_DAYS,
        scale=_largest_volume(_catchment_abstraction()),
        replay=("res", "src"),
    ),
    "river_chain": Case(
        build=_river_chain,
        days=_STORAGE_DAYS,
        scale=_largest_volume(_river_chain()),
        replay=("r_head", "r_mid", "r_low", "r_side", "r_end", "r_lost", "res"),
    ),
    "queue_arc": Case(
        build=_queue_arc,
        days=_QUEUE_ARC_DAYS,
        scale=_largest_volume(_queue_arc()),
        replay=("a", "b", "res", "basin"),
    ),
    "distribution_leakage": Case(
        build=_distribution_leakage,
        days=_STORAGE_DAYS,
        scale=150.0,
        replay=("dist_gw", "dist_dry", "town"),
    ),
    "unlimited_distribution": Case(
        build=_unlimited_distribution,
        days=_STORAGE_DAYS,
        scale=40.0,
        replay=("mains", "town"),
    ),
    "demand_constant_and_loads": Case(
        build=_demand_constant_and_loads,
        days=_STORAGE_DAYS,
        scale=60.0,
        replay=("town", "village"),
    ),
    "residential_demand": Case(
        build=_residential_demand,
        days=_STORAGE_DAYS,
        scale=500.0,
        replay=("homes",),
    ),
    "reservoir_spill": Case(
        build=_reservoir_spill,
        days=_STORAGE_DAYS,
        scale=60.0,
        replay=("dist", "res", "town"),
    ),
    "sewer_timearea": Case(
        build=_sewer_timearea,
        days=_STORAGE_DAYS,
        scale=800.0,
        replay=("sewer",),
    ),
    "foul_sewers": Case(
        build=_foul_sewers,
        days=_STORAGE_DAYS,
        scale=800.0,
        replay=("foul_main", "trunk"),
    ),
    "groundwater_infiltrate": Case(
        build=_groundwater_infiltrate,
        days=_STORAGE_DAYS,
        scale=100.0,
        replay=("gw", "sewer"),
    ),
    "wwtw_storm_tank": Case(
        build=_wwtw_storm_tank,
        days=_STORAGE_DAYS,
        scale=_largest_volume(_wwtw_storm_tank()),
        replay=("works", "combined"),
    ),
    "fwtw_deficit_sludge": Case(
        build=_fwtw_deficit_sludge,
        days=_STORAGE_DAYS,
        scale=150.0,
        replay=("plant", "sludge_sewer"),
    ),
    "queue_tank_time_zero": Case(
        build=_queue_tank_time_zero,
        days=_STORAGE_DAYS,
        scale=_largest_volume(_queue_tank_time_zero()),
        replay=("sewer",),
    ),
    "wtw_zero_input": Case(
        build=_wtw_zero_input,
        days=_STORAGE_DAYS,
        scale=30.0,
        wsimod=False,
    ),
}
