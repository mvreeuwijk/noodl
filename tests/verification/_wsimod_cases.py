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
}
