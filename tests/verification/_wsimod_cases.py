"""Purpose-built WSIMOD cases (D8): small models that exercise what the demos skip.

Each case is a mapping in ``Model.load``'s post-read form, with inline ``data_input_dict``
keyed ``(variable, date string)``. WSIMOD builds it by ``Model.load``'s own steps
(``_wsimod_reference.wsimod_model``); noodl by ``read_wsimod_config(mapping)``. A case is added
in the first task where every class it needs exists.

``scale`` is the largest volume in the case's forcing: the replay and parity bounds are
``1e-12 * scale`` (spec, Verification).
"""

from __future__ import annotations

import csv
import functools
import random
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

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


@functools.cache
def _forcing_dir() -> Path:
    """One temporary directory per process (each xdist worker writes its own, D11)."""
    return Path(tempfile.mkdtemp(prefix="noodl-wsimod-forcing-"))


def _forcing_file(case: str, rows: Mapping[tuple[str, str], float]) -> str:
    """The CSV of a ``filename`` override (D11): columns ``variable``, ``time``, ``value``, as
    ``read_csv`` reads them (``model.py:1252-1269``). Returns its **absolute** path: both
    engines open an override's file relative to the working directory, so a relative path would
    depend on where pytest runs (X27)."""
    path = _forcing_dir() / f"{case}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variable", "time", "value"])
        for (variable, date), value in rows.items():
            writer.writerow([variable, date, repr(float(value))])
    return str(path.resolve())


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


def _nonresidential_demand() -> dict:
    """A ``NonResidentialDemand`` drawing a constant demand from a storage and pushing it on.
    Noodl-only: WSIMOD's ``NonResidentialDemand.get_demand`` calls itself (``demand.py:143-150``,
    ``self.get_demand()`` instead of the parent's), so ``create_demand`` recurses without end."""
    load = {"phosphate": 0.02, "ammonia": 0.1, "temperature": 15.0}
    return _storage_case(
        [
            _store("supply", "Storage", capacity=1e4, initial_storage=_stocked(60.0)),
            _store("factory", "NonResidentialDemand", constant_demand=3.0, pollutant_load=load),
            _store("sea", "Waste"),
        ],
        [_arc("supply_to_factory", "supply", "factory"), _arc("factory_to_sea", "factory", "sea")],
        [{"NonResidentialDemand": "create_demand"}],
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
    than the inflow on wet days, so the sewer fills and its push check limits both senders.

    The default-tag sender is the reservoir (``Storage.distribute`` pushes with no ``of_type``,
    ``storage.py:120-127``), not the node: a ``Node`` pushes and checks only towards Node, River,
    Waste and Reservoir (``nodes.py:62-71``), and its single-arc push drops any other type
    (``:430-438``). A node wired straight to a sewer moves nothing in WSIMOD; run live, every
    arc of that model carries 0 on every day."""
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
    3 a day of the 6, so the trunk fills, then ``foul_main``, then the demand backs up. The
    ``"Sewer"`` slot runs before the ``"Foul"`` slot, so the trunk discharges before
    ``foul_main`` pushes into it: a foul sewer filed under the wrong slot changes the flows."""
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
            {"Sewer": "make_discharge"},
            {"Foul": "make_discharge"},
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


def _wtw_catchment(name: str, days: int, seed: int, *, flow: tuple[float, float]) -> dict:
    """A catchment whose temperature crosses 20 degC (daily values in [-5, 35]), for the
    treatment's temperature factor ``exponent ** (20 - T)``."""
    rng = random.Random(seed)
    data = {}
    for date in _dates(days):
        data[("flow", date)] = round(rng.uniform(*flow), 6)
        data[("phosphate", date)] = round(rng.uniform(1e-3, 1e-2), 8)
        data[("ammonia", date)] = round(rng.uniform(1e-3, 5e-2), 8)
        data[("temperature", date)] = round(rng.uniform(-5.0, 35.0), 4)
    return {"name": name, "type_": "Catchment", "data_input_dict": data}


def _wtw_treatment() -> dict:
    """The treatment processes (``wtw.py:163-214``) on the topologies of ``wwtw_storm_tank`` and
    ``fwtw_deficit_sludge``, side by side. The ``WWTW`` has per-pollutant constants and
    exponents (one above 1, one below), liquor multipliers other than the volume's and a
    catchment whose temperature crosses 20 degC; cold days make its phosphate solids negative
    (``0.55 x 1.07 ** (20 - T) + 0.35 > 1`` below about 15 degC, ``:207-208``). The ``FWTW``'s
    supply is topped up by a small, varying catchment, so its water's temperature moves too."""
    works = _wwtw_storm_tank()
    plant = _fwtw_deficit_sludge()
    days = _STORAGE_DAYS
    nodes = [dict(n) for n in works["nodes"]]
    for node in nodes:
        if node["name"] == "inflow":
            node.update(_wtw_catchment("inflow", days, 41, flow=(5.0, _FLOW_MAX)))
        if node["name"] == "works":
            node["process_parameters"] = {
                "phosphate": {"constant": 0.55, "exponent": 1.07},
                "ammonia": {"constant": 0.3, "exponent": 0.98},
            }
            node["liquor_multiplier"] = {"phosphate": 0.35, "ammonia": 0.5, "volume": 0.035}
    for node in plant["nodes"]:
        node = dict(node)
        if node["name"] == "sea":
            node["name"] = "sea_f"
        if node["name"] == "plant":
            node["process_parameters"] = {
                "phosphate": {"constant": 0.3, "exponent": 1.02},
                "ammonia": {"constant": 0.45, "exponent": 1.0},
            }
            node["liquor_multiplier"] = {"phosphate": 0.645, "ammonia": 0.2, "volume": 0.03}
        nodes.append(node)
    nodes.append(_wtw_catchment("source", days, 42, flow=(0.5, 3.0)))
    arcs = [dict(a) for a in works["arcs"]]
    for arc in plant["arcs"]:
        arc = dict(arc)
        for end in ("in_port", "out_port"):
            if arc[end] == "sea":
                arc[end] = "sea_f"
        arcs.append(arc)
    arcs.append(_arc("source_to_supply", "source", "supply"))
    return _storage_case(
        nodes,
        arcs,
        [
            {"Catchment": "route"},
            {"Reservoir": "distribute"},
            {"Sewer": "make_discharge"},
            {"WWTW": "calculate_discharge"},
            {"FWTW": "treat_water"},
            {"Demand": "create_demand"},
            {"WWTW": "make_discharge"},
        ],
    )


# --- land and its surfaces (Task 12) -----------------------------------------------------------
_LAND_DAYS = 40


def _land_forcing(
    days: int,
    seed: int,
    *,
    rain: tuple[float, ...] = (0.0, 0.0, 2e-4, 1e-3, 4e-3, 8e-3, 1.5e-2),
    et0: tuple[float, float] = (5e-4, 4e-3),
) -> dict:
    """A ``Land``'s daily ``precipitation`` (drawn from ``rain``, jittered), ``et0`` and
    ``temperature`` (``land.py:451-460``; all three are read every day, spec Forcing checks)."""
    rng = random.Random(seed)
    data = {}
    for date in _dates(days):
        depth = rng.choice(rain)
        data[("precipitation", date)] = round(depth * rng.uniform(0.8, 1.2), 7)
        data[("et0", date)] = round(rng.uniform(*et0), 7)
        data[("temperature", date)] = round(rng.uniform(4.0, 20.0), 4)
    return data


def _land(name: str, surfaces: list[dict], forcing: dict, **values) -> dict:
    return {
        "name": name,
        "type_": "Land",
        "surfaces": surfaces,
        "data_input_dict": forcing,
        **values,
    }


def _pervious(name: str, **values) -> dict:
    return {"surface": name, "type_": "PerviousSurface", **values}


def _impervious(name: str, **values) -> dict:
    return {"surface": name, "type_": "ImperviousSurface", **values}


def _neutral_soil(land: str, *surfaces: str) -> dict:
    """The neutral soil temperature (spec, Pollutants in 3a: 3b) as WSIMOD's overrides, the only
    way to set it (``land.py:753-756``, ``:843-879``). Each surface's physical depth is a power
    of two so that the override's ``depth / total_porosity * total_porosity`` round trip
    (``:858``, ``:883``) is exact and the derived values equal construction's bitwise."""
    neutral = {"soil_temp_w_prev": 1, "soil_temp_w_air": 0, "soil_temp_w_deep": 0}
    return {
        "nodes": {
            land: {"type_": "Land", "name": land, "surfaces": {s: dict(neutral) for s in surfaces}}
        }
    }


def _impervious_only() -> dict:
    """Two impervious surfaces drain into a sewer whose outlet takes 10 a day: ``roads`` has
    pores (2 mm) that evaporate on dry days, ``roofs`` none, so every wet day ponds at once
    (``land.py:645-721``). On wet days the sewer fills and its push check sends part of the
    runoff back into the surfaces."""
    days = _LAND_DAYS
    return {
        **_POLLUTANTS,
        "dates": _dates(days),
        "nodes": [
            _land(
                "town",
                [
                    _impervious(
                        "roads", area=2000.0, pore_depth=0.002, et0_to_e=0.9, initial_storage=1.0
                    ),
                    _impervious("roofs", area=1000.0),
                ],
                _land_forcing(days, 41),
            ),
            _store("drain", "Sewer", capacity=25.0),
            _store("sea", "Waste"),
        ],
        "arcs": [
            _arc("town_to_drain", "town", "drain"),
            _arc("drain_to_sea", "drain", "sea", capacity=10.0),
        ],
        "orchestration": [{"Land": "run"}, {"Sewer": "make_discharge"}],
    }


def _pervious_only() -> dict:
    """Two IHACRES soils near field capacity: wet days pass water through the soil store and
    dry days evaporate from it (both branches of ``land.py:985-1001``). Percolation goes to a
    groundwater, runoff through a node to the sea."""
    days = _LAND_DAYS
    return {
        **_POLLUTANTS,
        "dates": _dates(days),
        "nodes": [
            _land(
                "farm",
                [
                    _pervious(
                        "grass",
                        area=1e4,
                        depth=0.5,
                        infiltration_capacity=0.01,
                        initial_storage=1700.0,
                    ),
                    _pervious(
                        "wood",
                        area=5000.0,
                        depth=1.0,
                        total_porosity=0.45,
                        field_capacity=0.32,
                        wilting_point=0.15,
                        infiltration_capacity=0.008,
                        et0_coefficient=0.7,
                        ihacres_p=5,
                        initial_storage=1900.0,
                    ),
                ],
                _land_forcing(days, 42),
                subsurface_residence_time=3,
                percolation_residence_time=10,
            ),
            _store("aquifer", "Groundwater", capacity=1e5, residence_time=20.0),
            {"name": "brook", "type_": "Node"},
            _store("sea", "Waste"),
        ],
        "arcs": [
            _arc("farm_to_aquifer", "farm", "aquifer"),
            _arc("farm_to_brook", "farm", "brook"),
            _arc("aquifer_to_sea", "aquifer", "sea"),
            _arc("brook_to_sea", "brook", "sea"),
        ],
        "overrides": _neutral_soil("farm", "grass", "wood"),
        "orchestration": [{"Land": "run"}, {"Groundwater": "distribute"}],
    }


def _evaporation_clamp() -> dict:
    """A dry moor whose rain often exceeds its infiltration capacity of 0.5 mm under a high
    evaporative demand: the soil store evaporates (``land.py:997-1001``) and WSIMOD prints
    "inaccurate evaporation calculation" (``:1003-1021``), on some days and not others."""
    days = _LAND_DAYS
    return {
        **_POLLUTANTS,
        "dates": _dates(days),
        "nodes": [
            _land(
                "heath",
                [
                    _pervious(
                        "moor",
                        area=1000.0,
                        depth=0.5,
                        infiltration_capacity=0.0005,
                        initial_storage=5.0,
                    )
                ],
                _land_forcing(days, 43, rain=(0.0, 3e-4, 2e-3, 4e-3), et0=(0.01, 0.03)),
            ),
            {"name": "ditch", "type_": "Node"},
            _store("sea", "Waste"),
        ],
        "arcs": [_arc("heath_to_ditch", "heath", "ditch"), _arc("ditch_to_sea", "ditch", "sea")],
        "overrides": _neutral_soil("heath", "moor"),
        "orchestration": [{"Land": "run"}],
    }


def _temperature_blending() -> dict:
    """Rain enters roofs and lawns at the day's air temperature (``land.py:685``, ``:991``,
    ``:1028``); the non-additive temperature blends in the surfaces, the residence tanks, the
    sewer's queue and a node, so every ``sum_vqip`` blend is on the replayed path."""
    days = _LAND_DAYS
    return {
        **_POLLUTANTS,
        "dates": _dates(days),
        "nodes": [
            _land(
                "suburb",
                [
                    _impervious("roofs", area=1500.0, pore_depth=0.001),
                    _pervious(
                        "lawns",
                        area=3000.0,
                        depth=0.25,
                        wilting_point=0.1,
                        infiltration_capacity=0.006,
                        initial_storage=250.0,
                    ),
                ],
                _land_forcing(days, 44),
                surface_residence_time=2,
            ),
            _store("storm", "Sewer", capacity=20.0, pipe_time=1),
            {"name": "brook", "type_": "Node"},
            _store("river", "Waste"),
        ],
        "arcs": [
            _arc("suburb_to_storm", "suburb", "storm"),
            _arc("suburb_to_brook", "suburb", "brook"),
            _arc("storm_to_river", "storm", "river", capacity=15.0),
            _arc("brook_to_river", "brook", "river"),
        ],
        "overrides": _neutral_soil("suburb", "lawns"),
        "orchestration": [{"Land": "run"}, {"Sewer": "make_discharge"}],
    }


def _land_percolation_threshold() -> dict:
    """``Land.run`` forces the percolation the groundwater rejects back only above
    ``FLOAT_ACCURACY`` (``land.py:213-215``). A plot of 1e-8 m2 percolates about 1e-10 a day into
    an aquifer that takes 5e-11 and empties daily, so the rejection is above the threshold on
    wet days, in (0, 1e-11] on some days (where it vanishes) and 0 on dry ones."""
    days = _LAND_DAYS
    return {
        **_POLLUTANTS,
        "dates": _dates(days),
        "nodes": [
            _land(
                "plot",
                [
                    _pervious(
                        "patch",
                        area=1e-8,
                        depth=0.5,
                        infiltration_capacity=0.02,
                        initial_storage=1.7e-9,
                    )
                ],
                _land_forcing(days, 45),
                percolation_residence_time=1,
            ),
            _store("aquifer", "Groundwater", capacity=5e-11, residence_time=1.0),
            _store("sea", "Waste"),
        ],
        "arcs": [
            _arc("plot_to_aquifer", "plot", "aquifer"),
            _arc("aquifer_to_sea", "aquifer", "sea"),
        ],
        "overrides": _neutral_soil("plot", "patch"),
        "orchestration": [{"Land": "run"}, {"Groundwater": "distribute"}],
    }


def _residential_with_land() -> dict:
    """A ``ResidentialDemand`` with a garden ``Land`` (one pervious surface) and a sewer: the
    garden check asks the land (``demand.py:250-301``), which answers 0 without a
    ``GardenSurface`` (``land.py:93-94``, W16). With two out-arcs the garden push of 0 runs no
    round (``nodes.py:453``), so the land is asked twice a day and pushed nothing."""
    days = _LAND_DAYS
    return {
        **_POLLUTANTS,
        "dates": _dates(days),
        "nodes": [
            _store("supply", "Storage", capacity=1e4, initial_storage=_stocked(800.0)),
            _homes("homes", days, 46),
            _land(
                "gardens",
                [_pervious("lawn", area=2000.0, depth=0.5, initial_storage=340.0)],
                _land_forcing(days, 47),
            ),
            _store("drain", "Sewer", capacity=50.0),
            {"name": "brook", "type_": "Node"},
            _store("sea", "Waste"),
        ],
        "arcs": [
            _arc("supply_to_homes", "supply", "homes"),
            _arc("homes_to_gardens", "homes", "gardens"),
            _arc("homes_to_drain", "homes", "drain"),
            _arc("gardens_to_brook", "gardens", "brook"),
            _arc("brook_to_sea", "brook", "sea"),
            _arc("drain_to_sea", "drain", "sea"),
        ],
        "overrides": _neutral_soil("gardens", "lawn"),
        "orchestration": [
            {"ResidentialDemand": "create_demand"},
            {"Land": "run"},
            {"Sewer": "make_discharge"},
        ],
    }


def _land_sewer_ponding() -> dict:
    """noodl only: a city's streets drain into a combined sewer whose outlet takes 5 a day, and
    the sewer has a ponding arc back to the city: the ``Land -> Sewer`` and ``Sewer -> Land``
    arcs are a legal topological cycle. Ponding with the tag ``"Sewer"`` (``sewer.py:233-237``)
    crashes WSIMOD (``land.py:262`` reads ``self.surface``; W4). With these fixed parameters the
    sewer never ponds (every push into it is clipped by its check), so WSIMOD runs this mapping;
    ``test_land_sewer_ponding_runs_the_cycle_and_ponds_into_the_streets`` lowers the sewer's
    capacity for one day to reach the ponding."""
    days = _LAND_DAYS
    return {
        **_POLLUTANTS,
        "dates": _dates(days),
        "nodes": [
            _land("city", [_impervious("streets", area=5000.0)], _land_forcing(days, 48)),
            _store("combined", "Sewer", capacity=10.0),
            _store("sea", "Waste"),
        ],
        "arcs": [
            _arc("city_to_combined", "city", "combined"),
            _arc("combined_to_sea", "combined", "sea", capacity=5.0),
            _arc("combined_to_city", "combined", "city"),
        ],
        "orchestration": [{"Land": "run"}, {"Sewer": "make_discharge"}],
    }


def _batch_divergence() -> dict:
    """``three_out_arcs``' split ``Node`` fed by a ``Land`` instead of the catchments (Task 14,
    Review Focus 2): the Land's runoff and soil-through flows follow the weather, so a batch
    of capacities and soil moistures sends its instances down different branches (the
    over-subscription switch at the junction, IHACRES's moisture and through-soil branches) on
    the same day."""
    days = _LAND_DAYS
    return {
        **_POLLUTANTS,
        "dates": _dates(days),
        "nodes": [
            _land(
                "farm",
                [
                    _pervious(
                        "grass",
                        area=1e4,
                        depth=0.5,
                        infiltration_capacity=0.01,
                        initial_storage=1700.0,
                    )
                ],
                _land_forcing(days, 44, rain=(0.0, 2e-3, 5e-3, 8e-3, 1.5e-2)),
                subsurface_residence_time=2,
                surface_residence_time=1,
            ),
            {"name": "junction", "type_": "Node"},
            _store("outlet_a", "Waste"),
            _store("outlet_b", "Waste"),
            _store("outlet_c", "Waste"),
        ],
        "arcs": [
            _arc("farm_to_junction", "farm", "junction"),
            _arc("junction_to_a", "junction", "outlet_a", capacity=20.0, preference=1.0),
            _arc("junction_to_b", "junction", "outlet_b", capacity=30.0, preference=2.0),
            _arc("junction_to_c", "junction", "outlet_c", capacity=10.0, preference=3.0),
        ],
        "overrides": _neutral_soil("farm", "grass"),
        "orchestration": [{"Land": "run"}],
    }


def _soil_temperature() -> dict:
    """Two soils with weights other than the neutral ones (``land.py:1060-1079``): ``fen`` forgets
    its storage temperature (``w_prev = 0``) and ``heath`` blends all three terms with a deep
    temperature of 8; an impervious roof shares the land. The air temperature crosses 0 and 20.
    Runoff goes through a ``Node`` to a ``Waste`` and the percolation to a ``Groundwater``. The
    depths are powers of two, so the override's round trip is exact (see ``_neutral_soil``)."""
    days = _LAND_DAYS
    forcing = _land_forcing(days, 46)
    for number, date in enumerate(_dates(days)):
        forcing[("temperature", date)] = round(-4.0 + 26.0 * (number % 10) / 9, 4)
    weights = {
        "fen": {"soil_temp_w_prev": 0, "soil_temp_w_air": 0.6, "soil_temp_w_deep": 0.4},
        "heath": {
            "soil_temp_w_prev": 0.3,
            "soil_temp_w_air": 0.5,
            "soil_temp_w_deep": 0.2,
            "soil_temp_deep": 8,
        },
    }
    return {
        "pollutants": ["phosphate", "temperature"],
        "additive_pollutants": ["phosphate"],
        "non_additive_pollutants": ["temperature"],
        "dates": _dates(days),
        "nodes": [
            _land(
                "moor",
                [
                    _pervious(
                        "fen",
                        area=8000.0,
                        depth=0.5,
                        infiltration_capacity=0.01,
                        initial_storage=1500.0,
                    ),
                    _pervious(
                        "heath",
                        area=6000.0,
                        depth=1.0,
                        total_porosity=0.45,
                        field_capacity=0.32,
                        wilting_point=0.15,
                        infiltration_capacity=0.008,
                        initial_storage=2500.0,
                    ),
                    _impervious("roof", area=500.0, pore_depth=0.001),
                ],
                forcing,
                subsurface_residence_time=3,
                percolation_residence_time=10,
            ),
            _store("aquifer", "Groundwater", capacity=1e5, residence_time=20.0),
            {"name": "brook", "type_": "Node"},
            _store("sea", "Waste"),
        ],
        "arcs": [
            _arc("moor_to_aquifer", "moor", "aquifer"),
            _arc("moor_to_brook", "moor", "brook"),
            _arc("aquifer_to_sea", "aquifer", "sea"),
            _arc("brook_to_sea", "brook", "sea"),
        ],
        "overrides": {"nodes": {"moor": {"type_": "Land", "name": "moor", "surfaces": weights}}},
        "orchestration": [{"Land": "run"}, {"Groundwater": "distribute"}],
    }


# --- the cases no node task owns (Task 17) ---------------------------------------------------
_SIMPLE = {  # ``set_simple_pollutants`` (constants.py:93-97)
    "pollutants": ["phosphate", "temperature"],
    "additive_pollutants": ["phosphate"],
    "non_additive_pollutants": ["temperature"],
}


def _simple_catchment(name: str, days: int, seed: int, *, tiny: float | None = None) -> dict:
    """A catchment over the simple pollutant lists; with ``tiny``, every third day's flow is
    that small volume (below 1e-6, above 1e-11), the others in [5, 50]."""
    rng = random.Random(seed)
    data = {}
    for k, date in enumerate(_dates(days)):
        flow = round(rng.uniform(5.0, _FLOW_MAX), 6)
        data[("flow", date)] = tiny if tiny is not None and k % 3 == 2 else flow
        data[("phosphate", date)] = round(rng.uniform(1e-3, 1e-2), 8)
        data[("temperature", date)] = round(rng.uniform(4.0, 20.0), 4)
    return {"name": name, "type_": "Catchment", "data_input_dict": data}


def _capacity_first_come() -> dict:
    """Two orchestration slots push into one ``Node`` whose only outlet has capacity 30: the
    catchment's water (slot 1, flows 5-50) takes the capacity first, the reservoir's
    distribution (slot 2) gets what is left, and the rest is forced back. The arc's
    ``flow_in`` accumulates over the day (``arcs.py:130-160``), so the order is the slots'."""
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _catchment("src", days, 61),
            _store("res", "Reservoir", capacity=80.0, initial_storage=_stocked(60.0)),
            _store("hub", "Node"),
            _store("sea", "Waste"),
        ],
        [
            _arc("src_to_hub", "src", "hub"),
            _arc("res_to_hub", "res", "hub", capacity=25.0),
            _arc("hub_to_sea", "hub", "sea", capacity=30.0),
        ],
        [{"Catchment": "route"}, {"Reservoir": "distribute"}],
    )


def _simple_pollutants() -> dict:
    """``set_simple_pollutants``' lists (phosphate and temperature): a catchment through a
    ``Node`` and a reservoir to a sink."""
    days = _STORAGE_DAYS
    return {
        **_SIMPLE,
        "dates": _dates(days),
        "nodes": [
            _simple_catchment("src", days, 62),
            _store("head", "Node"),
            _store(
                "res",
                "Reservoir",
                capacity=60.0,
                initial_storage={"volume": 10.0, "phosphate": 0.02, "temperature": 11.0},
            ),
            _store("sink", "Waste"),
        ],
        "arcs": [
            _arc("src_to_head", "src", "head"),
            _arc("head_to_res", "head", "res"),
            _arc("res_to_sink", "res", "sink", capacity=20.0),
        ],
        "orchestration": [{"Catchment": "route"}, {"Reservoir": "distribute"}],
    }


def _float_accuracy_override() -> dict:
    """``float_accuracy: 1e-6`` (``model.py:238-241``): every third day's flow is 5e-7, a push
    that WSIMOD's default accuracy (1e-11) would move and this one drops. It reaches the
    ``QueueArc`` ``q`` (``arcs.py:378-379``: a push below ``FLOAT_ACCURACY`` returns an empty
    VQIP, the water vanishing), which would queue it at the default accuracy."""
    days = _STORAGE_DAYS
    return {
        **_SIMPLE,
        "float_accuracy": 1e-6,
        "dates": _dates(days),
        "nodes": [
            _simple_catchment("src", days, 63, tiny=5e-7),
            _store("head", "Node"),
            _store("buf", "Reservoir", capacity=100.0),
            _store("sink", "Waste"),
        ],
        "arcs": [
            _arc("src_to_head", "src", "head"),
            _arc("q", "head", "buf", type_="QueueArc", number_of_timesteps=1, capacity=30.0),
            _arc("buf_to_sink", "buf", "sink", capacity=12.0),
        ],
        "orchestration": [{"Catchment": "route"}, {"Reservoir": "distribute"}],
    }


def _node_type_override() -> dict:
    """A ``ResidentialDemand`` under ``type_: Demand`` with ``node_type_override`` (``model.py:
    653-660``): the class is the override's, the orchestration key the ``type_``'s."""
    days = _STORAGE_DAYS
    return _storage_case(
        [
            _store("supply", "Storage", capacity=1e4, initial_storage=_stocked(500.0)),
            {
                **_store(
                    "homes",
                    "Demand",
                    population=20.0,
                    per_capita=0.15,
                    pollutant_load={"phosphate": 0.01, "ammonia": 0.05},
                    constant_temp=14.0,
                    constant_weighting=0.2,
                    data_input_dict=_temperatures(days, 21),
                ),
                "node_type_override": "ResidentialDemand",
            },
            _store("sea", "Waste"),
        ],
        [_arc("supply_to_homes", "supply", "homes"), _arc("homes_to_sea", "homes", "sea")],
        [{"Demand": "create_demand"}],
    )


_DECAY_POLLUTANTS = {
    "pollutants": ["phosphate", "ammonia", "do", "temperature"],
    "additive_pollutants": ["phosphate", "ammonia"],
    "non_additive_pollutants": ["do", "temperature"],
}
_DECAY_DAYS = 40


def _decay_temperatures(days: int, seed: int, *, exactly_20: int | None = None) -> dict:
    """Air temperatures in [12, 28], so the decay factor crosses its reference at 20 degC;
    day ``exactly_20`` (if any) is 20.0 itself, where ``exponent ** 0 == 1``."""
    rng = random.Random(seed)
    data = {("temperature", date): round(rng.uniform(12.0, 28.0), 4) for date in _dates(days)}
    if exactly_20 is not None:
        data[("temperature", _dates(days)[exactly_20])] = 20.0
    return data


def _decay_catchment(name: str, days: int, seed: int) -> dict:
    """A catchment that also carries dissolved oxygen (``do``, non-additive)."""
    rng = random.Random(seed)
    data = {}
    for date in _dates(days):
        data[("flow", date)] = round(rng.uniform(5.0, _FLOW_MAX), 6)
        data[("phosphate", date)] = round(rng.uniform(1e-3, 1e-2), 8)
        data[("ammonia", date)] = round(rng.uniform(1e-3, 5e-2), 8)
        data[("do", date)] = round(rng.uniform(6.0, 10.0), 4)
        data[("temperature", date)] = round(rng.uniform(4.0, 20.0), 4)
    return {"name": name, "type_": "Catchment", "data_input_dict": data}


def _stocked_do(volume: float) -> dict:
    return {**_stocked(volume), "do": 8.0}


def _decay_storage() -> dict:
    """A ``Storage``, a ``Reservoir``, a ``RiverReservoir`` and a ``River`` (no nitrate), each
    with a ``decays`` dict and its own temperature forcing, which crosses 20 degC
    (``DecayTank.end_timestep_decay``, ``tanks.py:442-447``).

    - ``pond`` decays ``phosphate`` and the non-additive ``do`` and leaves ``ammonia`` and
      ``temperature`` alone;
    - ``res`` has a constant of 0.9, so ``constant * exponent ** (T - 20)`` exceeds 1 on warm days
      and the cap binds; an override merges ``ammonia`` into its ``decays``
      (``storage.py:101``);
    - ``rr`` decays two pollutants with a ``decays`` entry whose factor is small;
    - ``riv`` has a constant of 1 and a day at exactly 20 degC: its factor is exactly 1, which
      the cap leaves alone (Python's ``min(1.0, 1)`` is ``1.0``).
    """
    days = _DECAY_DAYS
    return {
        **_DECAY_POLLUTANTS,
        "dates": _dates(days),
        "nodes": [
            _decay_catchment("src", days, 31),
            _store("head", "Node"),
            _store(
                "pond",
                "Storage",
                capacity=80.0,
                initial_storage=_stocked_do(30.0),
                decays={
                    "phosphate": {"constant": 0.05, "exponent": 1.07},
                    "do": {"constant": 0.04, "exponent": 1.05},
                },
                data_input_dict=_decay_temperatures(days, 32),
            ),
            _store(
                "res",
                "Reservoir",
                capacity=60.0,
                initial_storage=_stocked_do(20.0),
                decays={"phosphate": {"constant": 0.9, "exponent": 1.08}},
                data_input_dict=_decay_temperatures(days, 33),
            ),
            _store(
                "rr",
                "RiverReservoir",
                capacity=50.0,
                initial_storage=_stocked_do(15.0),
                environmental_flow=5.0,
                decays={
                    "phosphate": {"constant": 0.02, "exponent": 1.03},
                    "ammonia": {"constant": 0.03, "exponent": 1.1},
                },
                data_input_dict=_decay_temperatures(days, 34),
            ),
            _river(
                "riv",
                damp=0.3,
                initial_storage=_stocked_do(10.0),
                decays={"phosphate": {"constant": 1.0, "exponent": 1.05}},
                data_input_dict=_decay_temperatures(days, 35, exactly_20=3),
            ),
            _store("sea", "Waste"),
        ],
        "arcs": [
            _arc("src_to_head", "src", "head"),
            _arc("head_to_pond", "head", "pond", capacity=40.0),
            _arc("pond_to_res", "pond", "res", capacity=15.0),
            _arc("res_to_rr", "res", "rr", capacity=12.0),
            _arc("rr_to_riv", "rr", "riv", capacity=10.0),
            _arc("riv_to_sea", "riv", "sea"),
        ],
        "overrides": {
            "nodes": {
                "res": {
                    "type_": "Reservoir",
                    "name": "res",
                    "decays": {"ammonia": {"constant": 0.02, "exponent": 1.03}},
                }
            }
        },
        "orchestration": [
            {"Catchment": "route"},
            {"Storage": "distribute"},
            {"Reservoir": "distribute"},
            {"RiverReservoir": "satisfy_environmental"},
        ],
    }


def _decay_groundwater(*, override: bool = True) -> dict:
    """A ``Groundwater`` with ``decays`` between a catchment and a waste. WSIMOD loses the
    groundwater's configured ``data_input_dict`` (``storage.py:202-203``: a decoy here, with
    temperatures that would decay it differently), so its temperature comes from a ``filename``
    override (``nodes.py:93-99``, X11). ``override=False`` leaves the override out: WSIMOD then
    raises ``TypeError`` at the first decay (``core.py:485``) and noodl refuses by name."""
    days = _DECAY_DAYS
    mapping = {
        **_DECAY_POLLUTANTS,
        "dates": _dates(days),
        "nodes": [
            _decay_catchment("src", days, 61),
            _store(
                "gw",
                "Groundwater",
                capacity=80.0,
                initial_storage=_stocked_do(30.0),
                residence_time=15.0,
                decays={
                    "phosphate": {"constant": 0.05, "exponent": 1.07},
                    "do": {"constant": 0.04, "exponent": 1.05},
                },
                data_input_dict=_decay_temperatures(days, 62),
            ),
            _store("sea", "Waste"),
        ],
        "arcs": [_arc("src_to_gw", "src", "gw"), _arc("gw_to_sea", "gw", "sea")],
        "orchestration": [{"Catchment": "route"}, {"Groundwater": "distribute"}],
    }
    if override:
        temperatures = _decay_temperatures(days, 63, exactly_20=4)
        mapping["overrides"] = {
            "nodes": {
                "gw": {
                    "type_": "Groundwater",
                    "name": "gw",
                    "filename": _forcing_file("decay_groundwater", temperatures),
                }
            }
        }
    return mapping


def _nitrifying(volume: float) -> dict:
    """An ``initial_storage`` VQIP over WSIMOD's 12 default pollutants, with ammonia, nitrite and
    nitrate to decay (every key is needed: WSIMOD sums over its pollutant list)."""
    return {
        "volume": volume,
        "do": 8.0,
        "org-phosphorus": 0.0004 * volume,
        "phosphate": 0.002 * volume,
        "ammonia": 0.02 * volume,
        "solids": 0.5 * volume,
        "bod": 0.01 * volume,
        "cod": 0.02 * volume,
        "ph": 7.0,
        "temperature": 11.0,
        "nitrate": 0.01 * volume,
        "nitrite": 0.005 * volume,
        "org-nitrogen": 0.003 * volume,
    }


def _nitrifying_land(*, twice: bool) -> dict:
    """An impervious and a pervious surface that decay ammonia, nitrite and phosphate, each with
    its own ``decays`` (D13) over WSIMOD's 12 default pollutants and the live soil temperature.
    The air temperature crosses 20 degC, so ``roads`` nitrite hits the cap of 1 on warm days
    (``core.py:513-515``). There is no ``River`` (nitrate would start its biochemistry): the
    percolation goes to a ``Groundwater`` without decays, the runoff through a ``Node`` to a
    ``Waste``. ``twice`` runs the ``Land`` twice a day (X16: the carry-over, ``land.py:427-439``,
    is converted at every ``Surface.run``)."""
    days = _LAND_DAYS
    forcing = _land_forcing(days, 71 if twice else 70)
    for number, date in enumerate(_dates(days)):
        forcing[("temperature", date)] = round(8.0 + 16.0 * (number % 10) / 9, 4)
    orchestration = [{"Land": "run"}] * (2 if twice else 1) + [{"Groundwater": "distribute"}]
    return {
        "dates": _dates(days),
        "nodes": [
            _land(
                "plot",
                [
                    _impervious(
                        "roads",
                        area=2000.0,
                        pore_depth=0.002,
                        et0_to_e=0.9,
                        initial_storage=_nitrifying(3.0),
                        decays={
                            "ammonia": {"constant": 0.05, "exponent": 1.08},
                            "nitrite": {"constant": 0.9, "exponent": 1.1},
                            "phosphate": {"constant": 0.01, "exponent": 1.03},
                        },
                    ),
                    _pervious(
                        "turf",
                        area=1e4,
                        depth=0.5,
                        infiltration_capacity=0.01,
                        initial_storage=_nitrifying(1700.0),
                        decays={
                            "ammonia": {"constant": 0.08, "exponent": 1.06},
                            "nitrite": {"constant": 0.12, "exponent": 1.04},
                            "phosphate": {"constant": 0.02, "exponent": 1.02},
                        },
                    ),
                ],
                forcing,
                subsurface_residence_time=3,
                percolation_residence_time=10,
            ),
            _store("aquifer", "Groundwater", capacity=1e5, residence_time=20.0),
            {"name": "brook", "type_": "Node"},
            _store("sea", "Waste"),
        ],
        "arcs": [
            _arc("plot_to_aquifer", "plot", "aquifer"),
            _arc("plot_to_brook", "plot", "brook"),
            _arc("aquifer_to_sea", "aquifer", "sea"),
            _arc("brook_to_sea", "brook", "sea"),
        ],
        "orchestration": orchestration,
    }


def _surface_decay_nitrification() -> dict:
    return _nitrifying_land(twice=False)


def _land_run_twice() -> dict:
    return _nitrifying_land(twice=True)


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
    "wtw_treatment": Case(
        build=_wtw_treatment,
        days=_STORAGE_DAYS,
        scale=_largest_volume(_wtw_treatment()),
        replay=("works", "combined", "plant", "sludge_sewer"),
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
    "impervious_only": Case(
        build=_impervious_only,
        days=_LAND_DAYS,
        scale=50.0,
        replay=("town", "drain"),
    ),
    "pervious_only": Case(
        build=_pervious_only,
        days=_LAND_DAYS,
        scale=2000.0,
        replay=("farm",),
    ),
    "soil_temperature": Case(
        build=_soil_temperature,
        days=_LAND_DAYS,
        scale=2000.0,
        replay=("moor",),
    ),
    "evaporation_clamp": Case(
        build=_evaporation_clamp,
        days=_LAND_DAYS,
        scale=5.0,
        replay=("heath",),
    ),
    "temperature_blending": Case(
        build=_temperature_blending,
        days=_LAND_DAYS,
        scale=300.0,
        replay=("suburb", "storm"),
    ),
    "land_percolation_threshold": Case(
        build=_land_percolation_threshold,
        days=_LAND_DAYS,
        scale=2e-9,
        replay=("plot",),
    ),
    "residential_with_land": Case(
        build=_residential_with_land,
        days=_LAND_DAYS,
        scale=800.0,
        replay=("homes", "gardens"),
    ),
    "land_sewer_ponding": Case(
        build=_land_sewer_ponding,
        days=_LAND_DAYS,
        scale=100.0,
    ),
    "nonresidential_demand": Case(
        build=_nonresidential_demand,
        days=_STORAGE_DAYS,
        scale=60.0,
        wsimod=False,
    ),
    "batch_divergence": Case(
        build=_batch_divergence,
        days=_LAND_DAYS,
        scale=100.0,
    ),
    "capacity_first_come": Case(
        build=_capacity_first_come,
        days=_STORAGE_DAYS,
        scale=_largest_volume(_capacity_first_come()),
        replay=("res", "hub"),
    ),
    "simple_pollutants": Case(
        build=_simple_pollutants,
        days=_STORAGE_DAYS,
        scale=_largest_volume(_simple_pollutants()),
        replay=("res",),
    ),
    "float_accuracy_override": Case(
        build=_float_accuracy_override,
        days=_STORAGE_DAYS,
        scale=_largest_volume(_float_accuracy_override()),
        replay=("buf",),
    ),
    "decay_storage": Case(
        build=_decay_storage,
        days=_DECAY_DAYS,
        scale=_largest_volume(_decay_storage()),
        replay=("pond", "res", "rr", "riv"),
    ),
    "decay_groundwater": Case(
        build=_decay_groundwater,
        days=_DECAY_DAYS,
        scale=_largest_volume(_decay_groundwater()),
        replay=("gw",),
    ),
    "surface_decay_nitrification": Case(
        build=_surface_decay_nitrification,
        days=_LAND_DAYS,
        scale=2000.0,
        replay=("plot",),
    ),
    "land_run_twice": Case(
        build=_land_run_twice,
        days=_LAND_DAYS,
        scale=2000.0,
        replay=("plot",),
    ),
    "node_type_override": Case(
        build=_node_type_override,
        days=_STORAGE_DAYS,
        scale=500.0,
        replay=("homes",),
    ),
}
