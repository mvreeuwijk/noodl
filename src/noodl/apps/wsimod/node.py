"""WSIMOD 0.8.1's ``Node`` and ``Waste`` (``wsimod/nodes/nodes.py:15-646``, ``nodes/waste.py``).

A node answers four kinds of call, each through a dict of handlers keyed by tag
(``nodes.py:61-72``): ``pull_set`` returns the VQIP obtained, ``push_set`` the VQIP **not**
accepted, and the two checks change nothing. A plain ``Node`` is a junction: it distributes a
push over its out-arcs and a pull over its in-arcs (``push_distributed``, ``pull_distributed``)
and its checks sum its neighbours' (``check_basic``).

Batching (spec, Approach). Every method takes the caller's ``active`` mask. The two distribute
loops run exactly ``MAXITER`` masked rounds; a round runs where ``active`` and WSIMOD's loop
condition both still hold, so an instance WSIMOD would have left the loop for is unchanged by
every later round. The structure (which arcs, which tags, the single-arc shortcut) is Python
control flow; every data-dependent branch is a :meth:`Ctx.decide`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor

from noodl.apps.wsimod.constants import MAXITER, UNBOUNDED_CAPACITY
from noodl.apps.wsimod.engine import Ctx, dmin
from noodl.apps.wsimod.vqip import (
    Pollutants,
    extract_vqip,
    safe_den,
    sum_vqip,
    v_change_vqip,
    write,
)

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.arcs import Arc
    from noodl.apps.wsimod.config import NodeSpec, WsimodNetwork

__all__ = ["NODE_CLASSES", "Connected", "Node", "Tag", "Waste"]

Tag = str | tuple[str, str]
SetHandler = Callable[..., Tensor]  # handler(vqip, *, active)
CheckHandler = Callable[..., Tensor]  # handler(vqip=None, *, active)

# nodes.py:63-71: where a plain Node pushes and what its push check asks
DEFAULT_PUSH_TYPES = ("Node", "River", "Waste", "Reservoir")


@dataclass
class Connected:
    """``get_connected``'s reply (``nodes.py:300-317``): total ``avail``, ``priority`` (the sum
    of avail x preference), and per arc name its ``allocation`` and ``capacity`` (its avail),
    in the order the arcs were polled."""

    avail: Tensor
    priority: Tensor
    allocation: dict[str, Tensor]
    capacity: dict[str, Tensor]


def _types(of_type: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """A string ``of_type`` is a one-element list (``nodes.py:253-254``).

    WSIMOD passes strings at ``storage.py:240`` and ``wtw.py:608`` (``of_type="Sewer"``).
    ``get_direction_arcs`` wraps them in a list, but the single-arc test (``nodes.py:367-368``,
    ``:430-431``) does ``x in of_type`` on the string itself, a substring test. No WSIMOD class
    name is a substring of another one an in-scope call site names, so the one-element tuple
    is equivalent at every call site in scope.
    """
    if of_type is None:
        return None
    return (of_type,) if isinstance(of_type, str) else tuple(of_type)


class Node:
    """``Node`` (``nodes.py:15-646``): a junction with WSIMOD's handler dictionaries.

    ``wsimod_type`` is the class name WSIMOD gives the instance after its renames; ``of_type``
    filters and the river order use it. ``store_paths`` names the node's stores by WSIMOD
    attribute path (the keys of :meth:`stores`), ``carried_paths`` the carried ones among them
    (the rest are per-day values); ``step_methods`` the zero-argument methods an
    orchestration slot may name; ``mass_balance`` the terms :meth:`record_terms` adds, with
    their side (``"in"``, ``"out"``, ``"ds"``; D11).
    """

    wsimod_type: ClassVar[str] = "Node"
    store_paths: ClassVar[tuple[str, ...]] = ()
    # the store paths carried from one day to the next (spec, State; W68, W75): a subset of
    # ``store_paths``, which also names the per-day values the replay loads at every call.
    # ``WsimodDay`` loads and dumps only these; the others reset every day, as WSIMOD's
    # ``end_timestep`` resets them (or overwrites them before they are read).
    carried_paths: ClassVar[tuple[str, ...]] = ()
    step_methods: ClassVar[tuple[str, ...]] = ()
    # nodes.py:76-78: in and out are the arcs' flows, which cancel; ds is empty
    mass_balance: ClassVar[dict[str, str]] = {}
    # the directions the class distributes over (its handlers call pull/push_distributed):
    # the zero-preference refusal applies to these only
    distributes: ClassVar[tuple[str, ...]] = ("out", "in")
    # the node replay's bound relative to the case scale (spec, Verification 1): 0, bitwise,
    # for a class whose arithmetic is IEEE + - * / and comparisons only
    replay_bound: ClassVar[float] = 0.0

    ctx: Ctx

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        self.spec = spec
        self.name = spec.name
        self.p = pollutants
        self.params: dict[str, Tensor] = {}
        self.in_arcs: dict[str, Arc] = {}
        self.out_arcs: dict[str, Arc] = {}
        self.in_arcs_type: dict[str, dict[str, Arc]] = {}
        self.out_arcs_type: dict[str, dict[str, Arc]] = {}
        # nodes.py:61-72
        self.pull_set_handler: dict[Tag, SetHandler] = {"default": self._pull_set_default}
        self.push_set_handler: dict[Tag, SetHandler] = {"default": self._push_set_default}
        self.pull_check_handler: dict[Tag, CheckHandler] = {"default": self.pull_check_basic}
        self.push_check_handler: dict[Tag, CheckHandler] = {"default": self._push_check_default}

    # --- the default handlers (nodes.py:61-72) ----------------------------------------------
    def _pull_set_default(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        return self.pull_distributed(vqip, active=active)

    def _push_set_default(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        return self.push_distributed(vqip, of_type=DEFAULT_PUSH_TYPES, active=active)

    def _push_check_default(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        return self.push_check_basic(vqip, of_type=DEFAULT_PUSH_TYPES, active=active)

    # --- structure --------------------------------------------------------------------------
    def connect(self, arc: Arc, end: str) -> None:
        """File ``arc`` as one of this node's in-arcs (``end="in"``) or out-arcs (``"out"``),
        under the type of its other end (``arcs.py:70-80``)."""
        if end == "in":
            self.in_arcs[arc.name] = arc
            self.in_arcs_type.setdefault(arc.in_type, {})[arc.name] = arc
        elif end == "out":
            self.out_arcs[arc.name] = arc
            self.out_arcs_type.setdefault(arc.out_type, {})[arc.name] = arc
        else:
            raise ValueError(f"node {self.name!r}: arc end {end!r} is not 'in' or 'out'")

    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """This day's parameter and forcing slices, by name."""
        self.params = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in parameters.items()}

    def configured_parameters(self, date: datetime) -> dict[str, Tensor]:
        """The configuration's values for ``date`` (unbatched), in :meth:`bind`'s form; the
        replay harness uses it to load a call's day forcing."""
        return {}

    def stores(self) -> dict[str, Tensor]:
        """The node's stores by WSIMOD attribute path (``store_paths``)."""
        return {}

    def load_stores(self, stores: Mapping[str, Tensor]) -> None:
        unknown = set(stores) - set(self.store_paths)
        if unknown:
            raise KeyError(f"node {self.name!r} has no store {sorted(unknown)[0]!r}")
        for path, value in stores.items():
            *parents, last = path.split(".")
            owner = self
            for part in parents:
                owner = getattr(owner, part)
            setattr(owner, last, torch.as_tensor(value, dtype=torch.float64))

    def end_timestep(self) -> None:
        """``nodes.py:637-642``: nothing for a plain node."""

    def record_terms(self) -> None:
        """Add this day's mass-balance terms to ``ctx.terms`` (D11); a plain node has none."""

    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """Refuse by name what this node cannot run; return the parameters a quantity-only run
        ignores. Preferences (spec, Protocol): negative ones are refused, and so are two or more
        arcs of one direction whose preferences are all zero (WSIMOD divides by a zero
        ``priority``, ``nodes.py:393``, ``:462``)."""
        for direction, port in (("out", "in_port"), ("in", "out_port")):
            arcs = [a for a in network.arcs if getattr(a, port) == spec.name]
            for arc in arcs:
                if arc.preference < 0:
                    raise ValueError(f"arc {arc.name!r}: preference {arc.preference} is negative")
            if direction not in cls.distributes:
                continue
            if len(arcs) >= 2 and all(arc.preference == 0 for arc in arcs):
                raise ValueError(
                    f"node {spec.name!r}: every {direction}-arc has preference 0 "
                    f"({', '.join(a.name for a in arcs)})"
                )
        return []

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        """Whether this node's ``direction`` check for ``tag`` calls its neighbours' checks
        (spec, Approach: recursion edges): ``pull_check_basic`` and ``push_check_basic``."""
        return tag == "default"

    # --- the four entry points (nodes.py:148-216, query_handler :319-343) -------------------
    def _handler(self, table: dict, kind: str, tag: Tag) -> Callable[..., Tensor]:
        try:
            return table[tag]
        except KeyError:
            raise KeyError(f"node {self.name!r} has no {kind} handler for tag {tag!r}") from None

    def pull_set(self, vqip: Tensor, tag: Tag = "default", *, active: Tensor) -> Tensor:
        return self._handler(self.pull_set_handler, "pull_set", tag)(vqip, active=active)

    def push_set(self, vqip: Tensor, tag: Tag = "default", *, active: Tensor) -> Tensor:
        return self._handler(self.push_set_handler, "push_set", tag)(vqip, active=active)

    def pull_check(
        self, vqip: Tensor | None = None, tag: Tag = "default", *, active: Tensor
    ) -> Tensor:
        return self._handler(self.pull_check_handler, "pull_check", tag)(vqip, active=active)

    def push_check(
        self, vqip: Tensor | None = None, tag: Tag = "default", *, active: Tensor
    ) -> Tensor:
        return self._handler(self.push_check_handler, "push_check", tag)(vqip, active=active)

    # --- the protocol -----------------------------------------------------------------------
    def get_direction_arcs(
        self, direction: str, of_type: str | Sequence[str] | None = None
    ) -> tuple[str, list[Arc]]:
        """``nodes.py:218-269``: the check method's name and the arcs, all of them in arc order
        or, with ``of_type``, type by type in ``of_type``'s order."""
        if direction not in ("pull", "push"):
            raise ValueError(f"node {self.name!r}: direction {direction!r} is not pull or push")
        f = "send_pull_check" if direction == "pull" else "send_push_check"
        types = _types(of_type)
        if types is None:
            arcs = self.in_arcs if direction == "pull" else self.out_arcs
            return f, list(arcs.values())
        table = self.in_arcs_type if direction == "pull" else self.out_arcs_type
        out: list[Arc] = []
        for type_ in types:
            out += list(table.get(type_, {}).values())
        return f, out

    def get_connected(
        self,
        direction: str = "pull",
        of_type: str | Sequence[str] | None = None,
        tag: Tag = "default",
        *,
        active: Tensor,
    ) -> Connected:
        """``nodes.py:271-317``: each arc's unclipped check, zeroed below ``FLOAT_ACCURACY``,
        summed in polling order."""
        ctx = self.ctx
        call = ctx.call(f"node:{self.name}", f"get_connected/{direction}")
        f, arcs = self.get_direction_arcs(direction, of_type)
        avail = ctx.zeros()
        priority = ctx.zeros()
        allocation: dict[str, Tensor] = {}
        capacity: dict[str, Tensor] = {}
        for arc in arcs:
            a = getattr(arc, f)(tag=tag, active=active)[..., 0]
            low = ctx.decide(
                f"node:{self.name}/get_connected/{direction}:{arc.name}/below_fa",
                a,
                "<",
                ctx.fa,
                active=active,
                call=call,
            )
            a = torch.where(low, torch.zeros_like(a), a)
            avail = avail + a
            preference = arc.preference
            priority = priority + a * preference
            allocation[arc.name] = a * preference
            capacity[arc.name] = a
        return Connected(avail, priority, allocation, capacity)

    def _single_arc_viable(self, table: dict[str, dict], types: tuple[str, ...] | None) -> bool:
        """``nodes.py:363-372``, ``:426-435``: no ``of_type``, or some connected type in it."""
        return types is None or any(x in types for x, y in table.items() if len(y) > 0)

    def _volume_vqip(self, volume: Tensor) -> Tensor:
        """``{"volume": volume}``: a request carrying only a volume."""
        volume = torch.as_tensor(volume, dtype=torch.float64)
        rest = torch.zeros(volume.shape + (self.p.K,), dtype=torch.float64)
        return torch.cat([volume[..., None], rest], dim=-1)

    def _zero_priority(self, connected: Connected, running: Tensor) -> None:
        """WSIMOD divides by ``priority`` (``nodes.py:393``, ``:462``) and raises
        ``ZeroDivisionError`` when it is 0 while ``avail > FLOAT_ACCURACY``: some arcs have
        preference 0 and the others no room. noodl distributes nothing that round (each share
        is ``allocation = 0`` over a safe denominator) and records ``"zero_priority"`` (spec,
        defect table)."""
        ctx = self.ctx
        zero = (connected.priority == 0).detach() & (connected.avail > ctx.fa).detach()
        ctx.event("zero_priority", self.name, running & zero)

    def pull_distributed(
        self,
        vqip: Tensor,
        of_type: str | Sequence[str] | None = None,
        tag: Tag = "default",
        *,
        active: Tensor,
    ) -> Tensor:
        """``nodes.py:345-407``: pull ``vqip``'s volume from the in-arcs by preference x avail.

        One in-arc: a straight pull, or nothing when its type is not in ``of_type``. Otherwise
        ``MAXITER`` rounds while ``deficit > FLOAT_ACCURACY`` and ``avail > FLOAT_ACCURACY``,
        each asking every arc for ``deficit x allocation / priority`` with the round's deficit.
        """
        types = _types(of_type)
        ctx, p = self.ctx, self.p
        if len(self.in_arcs) == 1:
            arc = next(iter(self.in_arcs.values()))
            if self._single_arc_viable(self.in_arcs_type, types):
                return arc.send_pull_request(vqip, active=active, tag=tag)
            return ctx.empty()
        pulled = ctx.empty()
        request = vqip[..., 0]
        deficit = request
        connected = self.get_connected("pull", types, tag, active=active)
        site = ctx.loop_site(self.name, "pull_distributed")
        where, call = f"node:{self.name}/pull_distributed", site[2]
        running = torch.as_tensor(active, dtype=torch.bool)
        for r in range(1, MAXITER + 1):
            entering = running
            running = (
                entering
                & ctx.decide(
                    f"{where}/deficit", deficit, ">", ctx.fa, active=entering, call=call, round=r
                )
                & ctx.decide(
                    f"{where}/avail",
                    connected.avail,
                    ">",
                    ctx.fa,
                    active=entering,
                    call=call,
                    round=r,
                )
            )
            self._zero_priority(connected, running)
            den = safe_den(connected.priority)
            with ctx.frame(f"{where}#{call}/round {r}"):
                for key, allocation in connected.allocation.items():
                    received = self.in_arcs[key].send_pull_request(
                        self._volume_vqip(deficit * allocation / den), active=running, tag=tag
                    )
                    pulled = write(running, sum_vqip(p, pulled, received), pulled)
            deficit = write(running, request - pulled[..., 0], deficit)
            connected = self.get_connected("pull", types, tag, active=running)
        ctx.loop_exit(site, running)
        return pulled

    def push_distributed(
        self,
        vqip: Tensor,
        of_type: str | Sequence[str] | None = None,
        tag: Tag = "default",
        *,
        active: Tensor,
    ) -> Tensor:
        """``nodes.py:409-478``: push ``vqip`` over the out-arcs; returns what was not pushed.

        One out-arc: a straight push, or ``vqip`` back when its type is not in ``of_type``.
        Otherwise, when the volume exceeds ``avail``, the first round shares by capacity
        (``:446-450``; every re-poll restores preference shares), then ``MAXITER`` rounds while
        ``not_pushed > FLOAT_ACCURACY`` and ``avail > FLOAT_ACCURACY``.
        """
        types = _types(of_type)
        ctx, p = self.ctx, self.p
        if len(self.out_arcs) == 1:
            arc = next(iter(self.out_arcs.values()))
            if self._single_arc_viable(self.out_arcs_type, types):
                return arc.send_push_request(vqip, active=active, tag=tag)
            return vqip
        not_pushed_ = vqip
        not_pushed = vqip[..., 0]
        connected = self.get_connected("push", types, tag, active=active)
        site = ctx.loop_site(self.name, "push_distributed")
        where, call = f"node:{self.name}/push_distributed", site[2]
        switch = ctx.decide(
            f"{where}/oversubscribed", not_pushed, ">", connected.avail, active=active, call=call
        )
        connected = Connected(
            connected.avail,
            torch.where(switch, connected.avail, connected.priority),
            {
                key: torch.where(switch, connected.capacity[key], allocation)
                for key, allocation in connected.allocation.items()
            },
            connected.capacity,
        )
        running = torch.as_tensor(active, dtype=torch.bool)
        for r in range(1, MAXITER + 1):
            entering = running
            running = (
                entering
                & ctx.decide(
                    f"{where}/not_pushed",
                    not_pushed,
                    ">",
                    ctx.fa,
                    active=entering,
                    call=call,
                    round=r,
                )
                & ctx.decide(
                    f"{where}/avail",
                    connected.avail,
                    ">",
                    ctx.fa,
                    active=entering,
                    call=call,
                    round=r,
                )
            )
            amount = dmin(
                ctx,
                f"{where}/amount",
                connected.avail,
                not_pushed,
                active=running,
                call=call,
                round=r,
            )
            self._zero_priority(connected, running)
            den = safe_den(connected.priority)
            with ctx.frame(f"{where}#{call}/round {r}"):
                for key, allocation in connected.allocation.items():
                    to_send = v_change_vqip(p, not_pushed_, amount * allocation / den)
                    reply = self.out_arcs[key].send_push_request(to_send, active=running, tag=tag)
                    sent = extract_vqip(p, to_send, reply)
                    not_pushed_ = write(running, extract_vqip(p, not_pushed_, sent), not_pushed_)
            not_pushed = not_pushed_[..., 0]
            connected = self.get_connected("push", types, tag, active=running)
        ctx.loop_exit(site, running)
        return not_pushed_

    def check_basic(
        self,
        direction: str,
        vqip: Tensor | None = None,
        of_type: str | Sequence[str] | None = None,
        tag: Tag = "default",
        *,
        active: Tensor,
    ) -> Tensor:
        """``nodes.py:480-508``: the ordered sum of the neighbours' unclipped checks, clipped to
        the request only when one is given (``:505-506``). A negative pipe excess after a
        forced push is summed as it is."""
        call = self.ctx.call(f"node:{self.name}", f"check_basic/{direction}")
        f, arcs = self.get_direction_arcs(direction, of_type)
        avail = self.ctx.empty()
        for arc in arcs:
            avail = sum_vqip(self.p, avail, getattr(arc, f)(tag=tag, active=active))
        if vqip is not None:
            volume = dmin(
                self.ctx,
                f"node:{self.name}/check_basic/{direction}/clip",
                avail[..., 0],
                vqip[..., 0],
                active=active,
                call=call,
            )
            avail = v_change_vqip(self.p, avail, volume)
        return avail

    def pull_check_basic(
        self, vqip: Tensor | None = None, of_type=None, tag: Tag = "default", *, active: Tensor
    ) -> Tensor:
        """``nodes.py:510-527``."""
        return self.check_basic("pull", vqip, of_type, tag, active=active)

    def push_check_basic(
        self, vqip: Tensor | None = None, of_type=None, tag: Tag = "default", *, active: Tensor
    ) -> Tensor:
        """``nodes.py:529-545``."""
        return self.check_basic("push", vqip, of_type, tag, active=active)

    # --- the deny and accept handlers (nodes.py:547-623) ------------------------------------
    def pull_set_deny(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``nodes.py:547-561``: nothing is obtained."""
        return self.ctx.empty()

    def pull_check_deny(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``nodes.py:563-577``: nothing is available."""
        return self.ctx.empty()

    def push_set_deny(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``nodes.py:579-593``: everything is refused."""
        return vqip

    def push_check_deny(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``nodes.py:595-609``: no capacity."""
        return self.ctx.empty()

    def push_check_accept(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``nodes.py:611-623``: the request, or ``UNBOUNDED_CAPACITY`` without one."""
        if vqip is None:
            return self._volume_vqip(
                torch.full(self.ctx.shape, UNBOUNDED_CAPACITY, dtype=torch.float64)
            )
        return vqip


class Waste(Node):
    """``Waste`` (``waste.py:10-49``): accepts every push, refuses every pull; the water leaves
    the model (its ``total_in`` is a mass-balance outflow, ``waste.py:38``)."""

    wsimod_type: ClassVar[str] = "Waste"
    distributes: ClassVar[tuple[str, ...]] = ()
    mass_balance: ClassVar[dict[str, str]] = {"Waste.total_in": "out"}

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        super().__init__(spec, pollutants)
        # waste.py:32-35
        self.pull_set_handler["default"] = self.pull_set_deny
        self.pull_check_handler["default"] = self.pull_check_deny
        self.push_set_handler["default"] = self.push_set_accept
        self.push_check_handler["default"] = self.push_check_accept

    def push_set_accept(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``waste.py:40-49``: everything is accepted."""
        return self.ctx.empty()

    def total_in(self) -> Tensor:
        """``nodes.py:104-117``: the in-arcs' ``vqip_out``, summed in arc order."""
        total = self.ctx.empty()
        for arc in self.in_arcs.values():
            total = sum_vqip(self.p, total, arc.vqip_out)
        return total

    def record_terms(self) -> None:
        ctx = self.ctx
        ctx.term("Waste.total_in", self.total_in(), torch.ones(ctx.shape, dtype=torch.bool))

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        return False


NODE_CLASSES: dict[str, type[Node]] = {"Node": Node, "Waste": Waste}
