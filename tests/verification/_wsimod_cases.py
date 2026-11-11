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
}
