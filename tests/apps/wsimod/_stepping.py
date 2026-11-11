"""Step an :class:`Engine` through a configuration's days, WSIMOD's way (``model.py:999-1176``).

A stand-in for the whole-model driver (``build_model``/``wsimod_run``, Tasks 13-14), which
replaces it. Every day binds the configured forcing and runs the orchestration slot by slot,
each slot's nodes in configuration order, as ``Model.nodes_type`` holds them
(``model.py:999-1003``). It then yields, so the caller can record the day (and add the nodes'
mass-balance terms), and runs ``end_timestep`` on every node, then on every arc
(``:1172-1176``). There is no river order: no case stepped through here has a ``River``.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from noodl.apps.wsimod.config import WsimodNetwork
from noodl.apps.wsimod.engine import Ctx, Engine
from noodl.apps.wsimod.vqip import Pollutants

__all__ = ["step_days"]


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
        yield day, ctx, engine
        engine.end_day()
