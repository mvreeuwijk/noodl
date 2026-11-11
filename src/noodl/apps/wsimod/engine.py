"""One WSIMOD day as objects: the day context, and the engine that builds the nodes and arcs.

The port runs WSIMOD 0.8.1's calls in WSIMOD's order (spec, Approach). Every node and arc holds
the day's :class:`Ctx` as ``self.ctx``; :meth:`Engine.begin_day` sets it. The context carries
the batch shape, the pollutant columns and ``FLOAT_ACCURACY``, and it collects what a day
reports: the branch decisions (D12), the warning events (D13), the ``MAXITER`` exits of the
distribute loops, the sub-``FLOAT_ACCURACY`` water WSIMOD drops, and the mass-balance terms
(D11).

Every branch on data goes through :meth:`Ctx.decide`, which compares detached values, so the
outcome is a bool tensor over the batch and the values stay in the autograd graph.

Decision records (spec, Diagnostics). Each is a :class:`Decision`
``(site, call, round, active, outcome, margin)``:

- ``site`` is ``"<node|arc>:<name>/<WSIMOD method>/<branch>"``, with the direction in the
  branch for methods that serve both directions. The sites so far:
  ``node:N/get_connected/<push|pull>:<arc>/below_fa``,
  ``node:N/push_distributed/{oversubscribed,not_pushed,avail,amount}``,
  ``node:N/pull_distributed/{deficit,avail}``, ``node:N/check_basic/<push|pull>/clip``,
  ``arc:A/get_excess/<push|pull>/min``, ``arc:A/send_push_request/clip``,
  ``arc:A/send_pull_request/clip``, ``node:N/pull_check_abstraction/clip``,
  ``node:N/pull_set_abstraction/clip``; an ``AltQueueArc``'s ``arc:A/send_push_request/below_fa``.
  A tank is not a node: its sites are its owner's with the tank's attribute path,
  ``node:N/<tank>/get_excess/{headroom,clip}``, ``node:N/<tank>/get_avail/clip``,
  ``node:N/<tank>/push_storage/clip``, ``node:N/<tank>/pull_storage/{empty,clip}``,
  ``node:N/<tank>/pull_ponded/excess``, ``node:N/<tank>/evaporate/clip``, and for a
  ``QueueTank`` ``node:N/<tank>/push_check/clip`` and
  ``node:N/<tank>/pull_storage_exact/clip:<volume|additive pollutant>``; its internal arc's
  sites are ``node:N/<tank>/internal_arc/...`` (``get_excess/push/min``,
  ``send_push_request/{below_fa,clip}``). Later classes extend the same pattern.
- ``call`` is noodl's static per-day index of the enclosing ``(element, method)`` call
  (:meth:`Ctx.call`, which :meth:`Ctx.loop_site` uses too). It counts the calls noodl's
  structure makes, masked ones included, so it is the same on every day and for every
  instance; it is not WSIMOD's dynamic count of the calls it actually made.
- ``round`` is the distribute loop's round, 1 to ``MAXITER``, or ``None`` outside a loop.
- ``active`` is the instance mask the branch was evaluated under. Only its true entries are
  WSIMOD's decisions: a masked call or round is computed and discarded. A loop's condition at
  round ``r`` is evaluated under the mask round ``r - 1`` ran with (WSIMOD tests it after
  every round it runs, and before the first under the caller's mask); the branches inside the
  round under round ``r``'s own mask.
- ``outcome`` is the comparison; ``margin`` is ``lhs - rhs``.

Tests key the records by ``(site, call, round)``.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from noodl.apps.wsimod.vqip import Pollutants, empty, sum_vqip

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.arcs import Arc
    from noodl.apps.wsimod.config import NodeSpec, WsimodNetwork
    from noodl.apps.wsimod.node import Node

__all__ = ["Ctx", "Decision", "Engine", "dmax", "dmin", "make_node"]

_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def _f64(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


@dataclass(frozen=True)
class Decision:
    """One recorded branch (see the module text for the vocabulary)."""

    site: str
    call: int
    round: int | None
    active: Tensor
    outcome: Tensor
    margin: Tensor


class Ctx:
    """One day of one batch: shape, pollutants, ``FLOAT_ACCURACY`` and the day's records.

    ``decisions`` is the list of :class:`Decision` records in call order, kept only when
    ``record_decisions``. ``events[condition][node]`` ORs the masks of WSIMOD's warning
    conditions. ``maxiter[(node, method, call index)]`` says whether that distribute loop ran
    all ``MAXITER`` rounds (WSIMOD prints "Maxiter reached", ``nodes.py:405-406``, ``:475-476``).
    ``dropped`` sums the water WSIMOD drops below ``FLOAT_ACCURACY``; ``terms`` the mass-balance
    terms ``"<Class>.<term>"``.
    """

    def __init__(
        self,
        p: Pollutants,
        float_accuracy: float,
        shape: torch.Size,
        *,
        day: int,
        node_names: Sequence[str],
        record_decisions: bool = False,
    ) -> None:
        self.p = p
        self.fa = float(float_accuracy)
        self.shape = torch.Size(shape)
        self.day = int(day)
        self.node_names = tuple(node_names)
        self.record_decisions = record_decisions
        self.decisions: list[Decision] = []
        self.events: dict[str, dict[str, Tensor]] = {}
        self.maxiter: dict[tuple[str, str, int], Tensor] = {}
        self.dropped: Tensor = self.empty()
        self.terms: dict[str, Tensor] = {}
        self._calls: dict[tuple[str, str], int] = {}

    def empty(self) -> Tensor:
        """An empty VQIP of the batch shape, ``shape + (1 + K,)``."""
        return empty(self.p, self.shape)

    def zeros(self) -> Tensor:
        """Zeros of the batch shape."""
        return torch.zeros(self.shape, dtype=torch.float64)

    def decide(
        self, site: str, lhs, op: str, rhs, *, active: Tensor, call: int, round: int | None = None
    ) -> Tensor:
        """``lhs op rhs`` on detached values: a bool tensor of the batch shape. When recording,
        appends a :class:`Decision` with the ``active`` mask it was evaluated under (D12)."""
        lhs, rhs = _f64(lhs), _f64(rhs)
        outcome = torch.broadcast_to(_OPS[op](lhs.detach(), rhs.detach()), self.shape)
        if self.record_decisions:
            mask = torch.broadcast_to(torch.as_tensor(active, dtype=torch.bool), self.shape)
            margin = torch.broadcast_to((lhs - rhs).detach(), self.shape)
            self.decisions.append(Decision(site, call, round, mask, outcome, margin))
        return outcome

    def event(self, condition: str, node: str, mask: Tensor) -> None:
        """Record that WSIMOD's warning ``condition`` holds at ``node`` where ``mask`` (D13)."""
        mask = torch.broadcast_to(torch.as_tensor(mask, dtype=torch.bool), self.shape)
        table = self.events.setdefault(condition, {})
        table[node] = table[node] | mask if node in table else mask.clone()

    def drop(self, vqip: Tensor, mask: Tensor) -> None:
        """Water WSIMOD drops below ``FLOAT_ACCURACY`` (the mass balance's ``"dropped"``)."""
        self.dropped = self.dropped + _masked(vqip, mask)

    def term(self, name: str, vqip: Tensor, mask: Tensor) -> None:
        """Add ``vqip`` where ``mask`` to the mass-balance term ``name`` (D11)."""
        old = self.terms.get(name, self.empty())
        self.terms[name] = sum_vqip(self.p, old, _masked(vqip, mask))

    def call(self, element: str, method: str) -> int:
        """The static index of this call of ``method`` on ``element`` (``"node:N"``,
        ``"arc:A"``) this day: 0, 1, ... in noodl's call order, masked calls included."""
        index = self._calls.get((element, method), 0)
        self._calls[(element, method)] = index + 1
        return index

    def loop_site(self, node: str, method: str) -> tuple[str, str, int]:
        """The call site ``(node, method, call index this day)`` of a distribute loop."""
        return (node, method, self.call(f"node:{node}", method))

    def loop_exit(self, site: tuple[str, str, int], hit: Tensor) -> None:
        """Record whether the loop at ``site`` ran all ``MAXITER`` rounds."""
        self.maxiter[site] = torch.broadcast_to(hit, self.shape).clone()


def _masked(vqip: Tensor, mask: Tensor) -> Tensor:
    mask = torch.as_tensor(mask, dtype=torch.bool)
    return torch.where(mask[..., None], vqip, torch.zeros_like(vqip))


def dmin(ctx: Ctx, site: str, a, b, *, active: Tensor, call: int, round: int | None = None):
    """Python's ``min(a, b)`` as a recorded decision: ``a`` unless ``b < a``."""
    a, b = _f64(a), _f64(b)
    return torch.where(ctx.decide(site, b, "<", a, active=active, call=call, round=round), b, a)


def dmax(ctx: Ctx, site: str, a, b, *, active: Tensor, call: int, round: int | None = None):
    """Python's ``max(a, b)`` as a recorded decision: ``a`` unless ``b > a``."""
    a, b = _f64(a), _f64(b)
    return torch.where(ctx.decide(site, b, ">", a, active=active, call=call, round=round), b, a)


def make_node(spec: NodeSpec, network: WsimodNetwork, pollutants: Pollutants) -> Node:
    """The transcribed node of ``spec``'s class (``NODE_CLASSES[spec.cls]``)."""
    from noodl.apps.wsimod.node import NODE_CLASSES

    _register_all()
    if spec.cls not in NODE_CLASSES:
        raise ValueError(f"node {spec.name!r}: class {spec.cls!r} is not ported yet")
    return NODE_CLASSES[spec.cls](spec, pollutants)


def _register_all() -> None:
    """Import every module that registers node classes."""
    import noodl.apps.wsimod.catchment  # noqa: F401


class Engine:
    """The nodes and arcs of a :class:`WsimodNetwork`, connected in configuration order (W30).

    WSIMOD fills a node's ``in_arcs`` and ``out_arcs`` in the arcs' construction order
    (``arcs.py:70-80``), so every sum over a node's arcs follows the configuration's arc order.
    """

    def __init__(self, network: WsimodNetwork, *, pollutants: Pollutants) -> None:
        from noodl.apps.wsimod.arcs import ARC_CLASSES

        self.network = network
        self.p = pollutants
        self.nodes: dict[str, Node] = {
            spec.name: make_node(spec, network, pollutants) for spec in network.nodes
        }
        self.arcs: dict[str, Arc] = {}
        for spec in network.arcs:
            if spec.cls not in ARC_CLASSES:
                raise ValueError(f"arc {spec.name!r}: class {spec.cls!r} is not ported yet")
            self.arcs[spec.name] = ARC_CLASSES[spec.cls](
                spec, self.nodes[spec.in_port], self.nodes[spec.out_port], pollutants
            )

    def begin_day(self, ctx: Ctx) -> None:
        """Give every node and arc the day's context."""
        for node in self.nodes.values():
            node.ctx = ctx
        for arc in self.arcs.values():
            arc.ctx = ctx

    def end_day(self) -> None:
        """``end_timestep`` on every node in configuration order, then on every arc
        (``model.py:1172-1176``)."""
        for node in self.nodes.values():
            node.end_timestep()
        for arc in self.arcs.values():
            arc.end_timestep()
