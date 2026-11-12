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
  ``node:N/pull_set_abstraction/clip``; an ``AltQueueArc``'s ``arc:A/send_push_request/below_fa``;
  a ``QueueArc``'s ``arc:A/send_push_request/below_fa`` and, in ``update_queue``, one
  ``arc:A/update_queue/<push|pull>:<i>,<j>/below_fa`` per waiting row and
  ``arc:A/update_queue/<push|pull>:new/below_fa`` for the call's own request (each evaluated
  where the row is present).
  A tank is not a node: its sites are its owner's with the tank's attribute path,
  ``node:N/<tank>/get_excess/{headroom,clip}``, ``node:N/<tank>/get_avail/clip``,
  ``node:N/<tank>/push_storage/clip``, ``node:N/<tank>/pull_storage/{empty,clip}``,
  ``node:N/<tank>/make_decay/cap:<pollutant>`` (a ``Decay``'s cap, evaluated at bind, D5),
  ``node:N/<tank>/pull_ponded/excess``, ``node:N/<tank>/evaporate/clip``, and for a
  ``QueueTank`` ``node:N/<tank>/push_check/clip`` and
  ``node:N/<tank>/pull_storage_exact/clip:<volume|additive pollutant>``; its internal arc's
  sites are ``node:N/<tank>/internal_arc/...`` (``get_excess/push/min``,
  ``send_push_request/{below_fa,clip}``). The storage family adds
  ``node:N/infiltrate/excess``, ``node:N/pull_check_active/clip``,
  ``node:N/pull_set_active/{clip,below_fa}``, ``node:N/push_check_river_reservoir/clip`` and
  ``node:N/satisfy_environmental/owed``; a ``Demand``'s ``node:N/create_demand/deficit`` and
  ``node:N/create_demand/not_able_to_push:<garden|house|default>`` (``demand.py:90``, ``:114``;
  events ``"demand_deficit"``, ``"demand_not_able_to_push"``); a leaking ``Distribution``'s
  ``node:N/pull_set_leakage/unsuccessful`` (``distribution.py:46``; event
  ``"leakage_not_going_to_gw"``); a ``River``'s ``node:N/pull_check_river/{mrf,clip}``
  (the ``max(.., 0)`` against the minimum residual flow and the clip to the request); a
  ``QueueGroundwater``'s frozen push-check tank (``storage.py:71``) is
  ``node:N/check_tank/get_excess/{headroom,clip}``. A ``Sewer`` adds
  ``node:N/push_check_sewer/clip`` and ``node:N/make_discharge/ponded`` (``sewer.py:160``,
  ``:235``; its tank is ``sewer_tank``). The treatment works add
  ``node:N/get_excess_throughput/headroom``,
  ``node:N/treat_current_input/liquor_zero_denominator`` (``wtw.py:155-161``, ``:174-182``);
  a ``WWTW``'s ``node:N/calculate_discharge/{stormwater,spare,clip}``,
  ``node:N/make_discharge/rejected``, ``node:N/push_check_sewer/clip``,
  ``node:N/push_set_sewer/{direct,all_direct,below_fa}`` and ``node:N/pull_set_reuse/clip``
  (``:331-334``, ``:347``, ``:370``, ``:388-403``, ``:421``; its tank is ``stormwater_tank``);
  an ``FWTW``'s ``node:N/treat_water/{target,deficit}`` (``:578-588``; its tank is
  ``service_reservoir_tank``). Their events are ``"miscalculated_tank_storage"`` and
  ``"ponded_water_cant_reenter"`` (``sewer.py:230-239``), ``"more_treated_than_input"``,
  ``"wwtw_couldnt_push"``, ``"service_reservoirs_not_filled"``, ``"sludge_nowhere"``,
  ``"excess_treated_water"`` (``wtw.py:213``, ``:349``, ``:598``, ``:610``, ``:616``) and noodl's
  ``"wtw_liquor_zero_denominator"``. A ``Land`` adds
  ``node:N/run/{percolation_reply,runoff,rejected,rejected_surface,rejected_subsurface}``
  (``land.py:213``, ``:223``, ``:228``, ``:241``, ``:243``); its surfaces are tanks under the
  path ``surfaces.<name>`` and add ``node:N/surfaces.<name>/precipitation_evaporation/dry``
  (``:660``) and ``node:N/surfaces.<name>/ihacres/{infiltration,excess,evaporation_kink,
  outflow_kink,moisture,through_soil}`` (``:927``, ``:928``, ``:935``, ``:948``, ``:957``,
  ``:985``); its event is ``"inaccurate_evaporation"`` (``:1003-1021``). Later classes extend
  the same pattern.
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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
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
        self.dropped_arcs: Tensor = self.empty()  # the part of ``dropped`` lost inside an arc
        self.terms: dict[str, Tensor] = {}
        # each WWTW's current_input as of its last treat_current_input (the water that was
        # treated); what is pushed in later vanishes at end_timestep (W69)
        self.treated_input: dict[str, Tensor] = {}
        self.frames: list[str] = []
        # every (element, method, index) of :meth:`call`, in call order: the day's static call
        # sequence, which D2 fixes on the tracing day
        self.sequence: list[tuple[str, str, int]] = []
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

    def drop(self, vqip: Tensor, mask: Tensor, *, arc: bool = False) -> None:
        """Water WSIMOD drops below ``FLOAT_ACCURACY`` (the mass balance's ``"dropped"``).
        ``arc=True`` marks a drop inside a ``QueueArc``, between two nodes: it is also summed in
        ``dropped_arcs``, which no node's own balance sees."""
        lost = _masked(vqip, mask)
        self.dropped = self.dropped + lost
        if arc:
            self.dropped_arcs = self.dropped_arcs + lost

    def term(self, name: str, vqip: Tensor, mask: Tensor) -> None:
        """Add ``vqip`` where ``mask`` to the mass-balance term ``name`` (D11)."""
        old = self.terms.get(name, self.empty())
        self.terms[name] = sum_vqip(self.p, old, _masked(vqip, mask))

    def call(self, element: str, method: str) -> int:
        """The static index of this call of ``method`` on ``element`` (``"node:N"``,
        ``"arc:A"``) this day: 0, 1, ... in noodl's call order, masked calls included."""
        index = self._calls.get((element, method), 0)
        self._calls[(element, method)] = index + 1
        self.sequence.append((element, method, index))
        return index

    @contextmanager
    def frame(self, label: str) -> Iterator[None]:
        """Name the enclosing call site while the block runs: a distribute loop's round
        (``"node:N/push_distributed#<call>/round <r>"``). ``frames`` is the stack; a
        ``QueueArc`` labels its request calls with it (D2)."""
        self.frames.append(label)
        try:
            yield
        finally:
            self.frames.pop()

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
    import noodl.apps.wsimod.demand  # noqa: F401
    import noodl.apps.wsimod.land  # noqa: F401
    import noodl.apps.wsimod.sewer  # noqa: F401
    import noodl.apps.wsimod.storage  # noqa: F401
    import noodl.apps.wsimod.wtw  # noqa: F401


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
        (``model.py:1172-1176``).

        The end-of-day rule: an ``end_timestep`` must not call ``ctx.decide`` or ``ctx.call``.
        Such a call would be absent from the D2 sequence check, and the D14 re-snapshot
        (``WsimodDay._finish_record``) would miss it, since it re-snapshots no decision and no
        ``maxiter``. It may record terms, drops and events. Decay caps are evaluated at bind
        (D5), not here."""
        for node in self.nodes.values():
            node.end_timestep()
        for arc in self.arcs.values():
            arc.end_timestep()
