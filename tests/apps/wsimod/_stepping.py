"""Step an :class:`Engine` through a configuration's days, WSIMOD's way (``model.py:999-1176``).

A stand-in for the whole-model driver (``build_model``/``wsimod_run``, Tasks 13-14), which
replaces it. Every day binds the configured forcing and runs the orchestration slot by slot,
each slot's nodes in configuration order, as ``Model.nodes_type`` holds them
(``model.py:999-1003``). It then yields, so the caller can record the day (and add the nodes'
mass-balance terms), and runs ``end_timestep`` on every node, then on every arc
(``:1172-1176``). The river order is :func:`river_order`, a minimal transcription of
``add_arcs``/``assign_upstream`` (``model.py:697-735``, ``:780-807``); Task 13's driver owns the
real one.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from noodl.apps.wsimod.classes import NODE_SCHEMAS
from noodl.apps.wsimod.config import WsimodNetwork
from noodl.apps.wsimod.engine import Ctx, Engine
from noodl.apps.wsimod.vqip import Pollutants

__all__ = ["river_order", "step_days"]

_RIVER_TYPES = ("River", "Node", "Waste", "Reservoir")


def river_order(network: WsimodNetwork) -> list[str]:
    """The rivers that ``distribute`` after the orchestration, most upstream first
    (``model.py:697-735``, ``:780-807``).

    The arcs with both ends of WSIMOD type River, Node, Waste or Reservoir give every ``Waste``
    node upstreamness 0 and each layer upstream of it one more; a node keeps the first (lowest)
    value it gets. Only nodes under the type key ``"River"`` enter, and a river no river path
    links to a ``Waste`` is left out. WSIMOD breaks ties within a level by the set-hash order of
    the node names (``model.py:780-807``), which varies with ``PYTHONHASHSEED``, so there is no
    fixed order to copy; this breaks them by configuration order (spec, W13). ``river_chain``'s
    ties sit on independent chains, so no value depends on that order.
    River arcs without any ``Waste`` raise ``ValueError`` (``max([])``, ``model.py:799``).
    """
    types = {n.name: NODE_SCHEMAS[n.cls].renamed_type for n in network.nodes}
    arcs = [
        a
        for a in network.arcs
        if types[a.in_port] in _RIVER_TYPES and types[a.out_port] in _RIVER_TYPES
    ]
    if not arcs:
        return []
    level = {n.name: 0 for n in network.nodes if n.slot_key == "Waste"}
    if not level:
        raise ValueError("river arcs but no Waste node (model.py:799)")
    while True:
        upstream = {a.in_port for a in arcs if a.out_port in level} - level.keys()
        if not upstream:
            break
        layer = max(level.values()) + 1
        for n in network.nodes:
            if n.name in upstream:
                level[n.name] = layer
    rivers = [n.name for n in network.nodes if n.slot_key == "River" and n.name in level]
    return sorted(rivers, key=lambda name: -level[name])


def step_days(
    network: WsimodNetwork,
    pollutants: Pollutants,
    days: int,
    *,
    record_decisions: bool = False,
    engine: Engine | None = None,
) -> Iterator[tuple[int, Ctx, Engine]]:
    """Yield ``(day, ctx, engine)`` after each day's orchestration, before ``end_timestep``."""
    if engine is None:
        engine = Engine(network, pollutants=pollutants)
    active = torch.tensor(True)
    slots = {spec.name: spec.slot_key for spec in network.nodes}
    rivers = river_order(network)
    for day in range(days):
        ctx = Ctx(
            pollutants,
            network.float_accuracy,
            torch.Size(()),
            day=day,
            node_names=list(engine.nodes),
            record_decisions=record_decisions,
        )
        engine.begin_day(ctx)
        for node in engine.nodes.values():
            node.bind(node.configured_parameters(network.dates[day]))
        for slot, method in network.orchestration:
            for name, node in engine.nodes.items():
                if slots[name] == slot:
                    getattr(node, method)(active=active)
        for name in rivers:  # model.py:1005-1007
            engine.nodes[name].distribute(active=active)
        yield day, ctx, engine
        engine.end_day()
