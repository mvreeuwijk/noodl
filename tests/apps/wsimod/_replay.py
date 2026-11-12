"""Replay one noodl node against the calls WSIMOD's own node received (spec, Verification 1).

The noodl node is built alone (``make_node``). Its arcs are :class:`ScriptedArc`\\ s that
return WSIMOD's captured replies in order and assert that noodl made the same calls, in the
same order, with the same arguments to the class's ``replay_bound * scale``. The replay loads
the node's stores from each call's ``pre``, makes the call, and compares the reply and the
``post`` stores.

The comparison is exact for a class whose ``replay_bound`` is 0 (``Node``, ``Waste``,
``Catchment``): every value must be bitwise WSIMOD's. A class with ``pow``/``exp``/``sqrt`` may
declare ``1e-12`` (relative to the case's scale) with its cause written next to it.

:class:`StoreNode` is a test double for the unit tests: a VQIP store of a given capacity whose
WSIMOD type is settable.

:func:`replay_arc` does the same for a user ``DecayArc`` or ``DecayArcAlt`` (spec, Verification
1, the replay harness): the arc is built alone between two :class:`ScriptedPort`\\ s that return
the replies WSIMOD's ports gave, and every request call, reply, accumulator, per-request ``diff``
and queued row is compared bitwise with the capture (:class:`ArcDay`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from noodl.apps.wsimod.arcs import ARC_CLASSES, DecayArcAlt
from noodl.apps.wsimod.classes import NODE_SCHEMAS
from noodl.apps.wsimod.config import ArcSpec, NodeSpec, WsimodNetwork
from noodl.apps.wsimod.constants import UNBOUNDED_CAPACITY
from noodl.apps.wsimod.engine import Ctx, make_node
from noodl.apps.wsimod.node import Node
from noodl.apps.wsimod.vqip import (
    Pollutants,
    empty,
    extract_vqip,
    sum_vqip,
    tmax,
    tmin,
    v_change_vqip,
    write,
)

if TYPE_CHECKING:  # pragma: no cover
    from tests.verification._wsimod_reference import ArcCall, ArcDay

__all__ = [
    "Mismatch",
    "ScriptedArc",
    "ScriptedPort",
    "StoreNode",
    "precondition_holds",
    "replay",
    "replay_arc",
]


def _keys(p: Pollutants) -> tuple[str, ...]:
    return ("volume", *p.names)


def to_vqip(p: Pollutants, values: Mapping[str, float]) -> Tensor:
    """A captured WSIMOD VQIP dict as a noodl VQIP; keys outside ``p`` are left out (a
    quantity-only replay keeps the volume)."""
    return p.to_vqip({k: float(v) for k, v in values.items() if k in _keys(p)})


class Mismatch(AssertionError):
    pass


class _Comparer:
    """Compares noodl tensors with WSIMOD values to ``bound * scale``; keeps the worst."""

    def __init__(self, p: Pollutants, scale: float, bound: float) -> None:
        self.p, self.scale, self.bound, self.worst = p, scale, bound, 0.0

    def value(self, label: str, ours, theirs: float) -> None:
        diff = abs(float(ours) - float(theirs))
        self.worst = max(self.worst, diff / self.scale)
        if not diff <= self.bound * self.scale:
            raise Mismatch(f"{label}: noodl {float(ours)!r}, WSIMOD {float(theirs)!r}")

    def vqip(self, label: str, ours: Tensor, theirs: Mapping, *, volume_only=False) -> None:
        for i, key in enumerate(_keys(self.p)):
            if volume_only and key != "volume":
                break
            if key not in theirs:
                raise Mismatch(f"{label}: WSIMOD's VQIP has no {key!r}")
            self.value(f"{label}[{key}]", ours[..., i], theirs[key])


class ScriptedArc:
    """``Arc``'s call interface over a script of WSIMOD's arc calls (D3, D4).

    ``calls`` is consumed in order by the active calls; several arcs may share one list, so
    the order across arcs is checked too. An inactive call is skipped and returns zeros. Each
    call asserts the arc, method, tag and force equal, and the request within the node class's
    bound: a pull request by its volume (D4), a push request in full. The accumulators are
    WSIMOD's: after each call they are set to the capture's values at return, and at entry they
    are compared with the capture where they are known (from the node call's ``pre`` or an
    earlier call), and taken from it otherwise. ``flow_out``/``vqip_out`` alias them, as an
    ``Arc``'s do.
    """

    def __init__(
        self,
        spec: ArcSpec,
        calls: list,
        pollutants: Pollutants,
        *,
        scale: float,
        in_type: str,
        out_type: str,
    ) -> None:
        self.spec = spec
        self.name = spec.name
        self.p = pollutants
        self.calls = calls
        self.in_type, self.out_type = in_type, out_type
        self.capacity = torch.tensor(spec.capacity, dtype=torch.float64)
        self.preference = torch.tensor(spec.preference, dtype=torch.float64)
        self.flow_in = torch.zeros((), dtype=torch.float64)
        self.vqip_in = empty(pollutants, ())
        self.compare = _Comparer(pollutants, scale, 0.0)
        self.label = spec.name
        self.known = False  # whether flow_in/vqip_in hold WSIMOD's values

    @property
    def flow_out(self) -> Tensor:
        return self.flow_in

    @property
    def vqip_out(self) -> Tensor:
        return self.vqip_in

    def assert_exhausted(self) -> None:
        if self.calls:
            raise Mismatch(
                f"{self.label}: noodl made no call for WSIMOD's {self.calls[0].method} on "
                f"{self.calls[0].arc!r} ({len(self.calls)} left)"
            )

    def _next(self, method: str, vqip: Tensor | None, tag, force: bool):
        label = f"{self.label}: {method} on {self.name!r}"
        if not self.calls:
            raise Mismatch(f"{label}: WSIMOD made no such call")
        call = self.calls.pop(0)
        if (call.arc, call.method, call.tag, call.force) != (self.name, method, tag, force):
            raise Mismatch(
                f"{label} (tag {tag!r}, force {force}): WSIMOD called {call.method} on "
                f"{call.arc!r} (tag {call.tag!r}, force {call.force})"
            )
        if (call.vqip is None) != (vqip is None):
            raise Mismatch(f"{label}: request {vqip!r}, WSIMOD's {call.vqip!r}")
        if vqip is not None:
            pull = method.startswith("send_pull")
            self.compare.vqip(f"{label} request", vqip, call.vqip, volume_only=pull)
        if self.known:
            self.compare.value(f"{label} flow_in at entry", self.flow_in, call.flow_in)
            self.compare.vqip(f"{label} vqip_in at entry", self.vqip_in, call.vqip_in)
        self.flow_in = torch.tensor(float(call.flow_in_after), dtype=torch.float64)
        self.vqip_in = to_vqip(self.p, call.vqip_in_after)
        self.known = True
        return to_vqip(self.p, call.reply)

    def send_push_request(self, vqip, *, active, tag="default", force=False) -> Tensor:
        if not bool(active):
            return torch.zeros_like(vqip)
        return self._next("send_push_request", vqip, tag, force)

    def send_pull_request(self, vqip, *, active, tag="default") -> Tensor:
        if not bool(active):
            return torch.zeros_like(vqip)
        return self._next("send_pull_request", vqip, tag, False)

    def send_push_check(self, vqip=None, *, active, tag="default") -> Tensor:
        if not bool(active):
            return empty(self.p, ())
        return self._next("send_push_check", vqip, tag, False)

    def send_pull_check(self, vqip=None, *, active, tag="default") -> Tensor:
        if not bool(active):
            return empty(self.p, ())
        return self._next("send_pull_check", vqip, tag, False)


class StoreNode(Node):
    """A VQIP store of ``capacity`` (a test double, not a WSIMOD class): ``push_set`` accepts up
    to the spare capacity, ``pull_set`` gives up to its volume. Both checks give the unclipped
    value without a request and clip to the request's volume with one."""

    store_paths = ("storage",)

    def __init__(
        self,
        name: str,
        pollutants: Pollutants,
        *,
        storage: Tensor | float = 0.0,
        capacity: float = UNBOUNDED_CAPACITY,
        wsimod_type: str = "Node",
    ) -> None:
        super().__init__(NodeSpec(name, "StoreNode", "StoreNode", {}), pollutants)
        self.wsimod_type = wsimod_type  # shadows the class's, as WSIMOD's renames do
        if not isinstance(storage, Tensor):
            storage = pollutants.to_vqip({"volume": storage})
        self.storage = storage
        self.capacity = torch.tensor(capacity, dtype=torch.float64)
        self.pull_set_handler["default"] = self._pull
        self.push_set_handler["default"] = self._push
        self.pull_check_handler["default"] = self._pull_check
        self.push_check_handler["default"] = self._push_check

    def stores(self) -> dict[str, Tensor]:
        return {"storage": self.storage}

    def _spare(self) -> Tensor:
        return v_change_vqip(self.p, self.storage, tmax(self.capacity - self.storage[..., 0], 0.0))

    def _push(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        accepted = tmin(vqip[..., 0], self._spare()[..., 0])
        reply = v_change_vqip(self.p, vqip, vqip[..., 0] - accepted)
        taken = extract_vqip(self.p, vqip, reply)
        self.storage = write(active, sum_vqip(self.p, self.storage, taken), self.storage)
        return reply

    def _pull(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        reply = v_change_vqip(self.p, self.storage, tmin(vqip[..., 0], self.storage[..., 0]))
        self.storage = write(active, extract_vqip(self.p, self.storage, reply), self.storage)
        return reply

    def _clip(self, avail: Tensor, vqip: Tensor | None) -> Tensor:
        if vqip is None:
            return avail
        return v_change_vqip(self.p, avail, tmin(avail[..., 0], vqip[..., 0]))

    def _pull_check(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        return self._clip(self.storage, vqip)

    def _push_check(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        return self._clip(self._spare(), vqip)


def _is_register(path: str) -> bool:
    return path.endswith("internal_arc.register")


def _is_present(path: str) -> bool:
    return path.endswith("internal_arc.present")


def _internal_arc(node: Node, path: str):
    """The internal arc a ``<tank>.internal_arc.<store>`` path names (``tank``,
    ``sewer_tank``, ...)."""
    owner = node
    for part in path.split(".")[:-1]:
        owner = getattr(owner, part)
    return owner


def load_value(p: Pollutants, node: Node, path: str, value) -> Tensor:
    """A captured store as the noodl tensor ``Node.load_stores`` takes. A ``QueueTank``'s queue
    is WSIMOD's ``{time: vqip}`` dict, captured under the ``register`` path, and its key list
    under the ``present`` path; a store that is a number is a scalar."""
    if _is_register(path) or _is_present(path):
        length = _internal_arc(node, path).length
    if _is_register(path):
        rows = torch.zeros(length + 1, 1 + p.K, dtype=torch.float64)
        for time, vqip in value.items():
            rows[time] = to_vqip(p, vqip)
        return rows
    if _is_present(path):
        mask = torch.zeros(length + 1, dtype=torch.bool)
        mask[list(value)] = True
        return mask
    if isinstance(value, Mapping):
        return to_vqip(p, value)
    return torch.tensor(float(value), dtype=torch.float64)


def compare_store(compare: _Comparer, label: str, path: str, ours: Tensor, theirs) -> None:
    """``ours`` against the captured store ``theirs`` (see :func:`load_value`); an absent queue
    key is an empty row."""
    if _is_register(path):
        for time in range(ours.shape[-2]):
            vqip = theirs.get(time)
            row = {} if vqip is None else vqip
            expected = {k: row.get(k, 0.0) for k in _keys(compare.p)}
            compare.vqip(f"{label}[{time}]", ours[..., time, :], expected)
    elif _is_present(path):
        keys = sorted(theirs)
        got = [t for t in range(ours.shape[-1]) if bool(ours[..., t])]
        if got != keys:
            raise Mismatch(f"{label}: noodl's key set {got}, WSIMOD's {keys}")
    elif isinstance(theirs, Mapping):
        compare.vqip(label, ours, theirs)
    else:
        compare.value(label, ours, theirs)


def _spec(network: WsimodNetwork, name: str) -> NodeSpec:
    for spec in network.nodes:
        if spec.name == name:
            return spec
    raise KeyError(f"no node {name!r}")


def replay(
    network: WsimodNetwork,
    name: str,
    calls: list,
    *,
    scale: float,
    pollutants: Pollutants | None = None,
    decisions: list | None = None,
) -> float:
    """Replay ``calls`` (``NodeCall``\\ s captured from WSIMOD) into noodl's node ``name``.

    ``pollutants`` defaults to the network's; ``QUANTITY_ONLY`` replays the volumes. Raises
    :class:`Mismatch` at the first difference beyond the node class's ``replay_bound`` x
    ``scale``; returns the worst difference relative to ``scale``. A list passed as ``decisions``
    receives ``(day, Decision)`` for every branch noodl's node took (``Ctx.decisions``).
    """
    p = network.pollutants if pollutants is None else pollutants
    spec = _spec(network, name)
    node = make_node(spec, network, p)
    types = {n.name: NODE_SCHEMAS[n.cls].renamed_type for n in network.nodes}
    script: list = []
    arcs: dict[str, ScriptedArc] = {}
    for arc_spec in network.arcs:
        if name not in (arc_spec.in_port, arc_spec.out_port):
            continue
        arc = ScriptedArc(
            arc_spec,
            script,
            p,
            scale=scale,
            in_type=types[arc_spec.in_port],
            out_type=types[arc_spec.out_port],
        )
        arcs[arc.name] = arc
        if arc_spec.in_port == name:
            node.connect(arc, "out")
        if arc_spec.out_port == name:
            node.connect(arc, "in")
    compare = _Comparer(p, scale, type(node).replay_bound)
    active = torch.tensor(True)
    for index, call in enumerate(calls):
        label = f"{name} call {index} (day {call.day}, {call.method}, tag {call.tag!r})"
        node.ctx = Ctx(
            p,
            network.float_accuracy,
            (),
            day=call.day,
            node_names=[name],
            record_decisions=decisions is not None,
        )
        script[:] = list(call.arc_calls)
        for arc in arcs.values():
            arc.label, arc.compare = label, compare
            arc.known = f"arcs.{arc.name}.flow_in" in call.pre
            flow_in = call.pre.get(f"arcs.{arc.name}.flow_in", 0.0)
            arc.flow_in = torch.tensor(float(flow_in), dtype=torch.float64)
            arc.vqip_in = to_vqip(p, call.pre.get(f"arcs.{arc.name}.vqip_in", {}))
        node.bind(node.configured_parameters(network.dates[call.day]))
        node.load_stores(
            {path: load_value(p, node, path, call.pre[path]) for path in node.store_paths}
        )
        if call.method == "end_timestep":
            node.end_timestep()
            reply = None
        elif call.method in type(node).step_methods:
            reply = getattr(node, call.method)(active=active)
        else:
            vqip = None if call.vqip is None else to_vqip(p, call.vqip)
            reply = getattr(node, call.method)(vqip, call.tag, active=active)
        if decisions is not None:
            decisions.extend((call.day, d) for d in node.ctx.decisions)
        for arc in arcs.values():
            arc.assert_exhausted()
        if call.reply is not None:
            compare.vqip(f"{label} reply", reply, call.reply)
        stores = node.stores()
        for path in node.store_paths:
            compare_store(compare, f"{label} post {path}", path, stores[path], call.post[path])
        for arc in arcs.values():
            key = f"arcs.{arc.name}"
            if f"{key}.flow_in" in call.post:
                compare.value(
                    f"{label} post {key}.flow_in", arc.flow_in, call.post[f"{key}.flow_in"]
                )
                compare.vqip(
                    f"{label} post {key}.vqip_in", arc.vqip_in, call.post[f"{key}.vqip_in"]
                )
    return compare.worst


# --- user Decay arcs (spec, Verification 1, the replay harness) -------------------------------
ARC_REPLAY_BOUND = 0.0  # bitwise: IEEE + - * / and comparisons, and ``pow_python`` (X18)
_DIRECTIONS = ("push", "pull")
_REQUESTS = {"send_push_request": "push", "send_pull_request": "pull"}
_OUT_PORT_CALLS = ("push_check", "push_set")  # the others go into the in_port (arcs.py:112-258)


class ScriptedPort(Node):
    """One end of an arc replayed alone (``ScriptedArc``'s counterpart for an arc): it answers
    the calls the arc makes into it with the replies WSIMOD's port gave, in order, and asserts
    that the arc made the same calls (method, tag and request, bitwise: ``ARC_REPLAY_BOUND``).

    A pull request is compared by its volume only (D4: its additive entries are not rescaled).
    An inactive call is skipped and returns zeros. ``script`` holds the port calls of the request
    under way, ``label`` names it in a mismatch, and ``params`` is the node's day forcing: the
    ``temperature`` a ``Decay*`` arc reads when this is its ``in_port`` (``core.py:464-470``).
    """

    def __init__(self, spec: NodeSpec, pollutants: Pollutants, *, wsimod_type: str, scale: float):
        super().__init__(spec, pollutants)
        self.wsimod_type = wsimod_type
        self.script: list[ArcCall] = []
        self.label = spec.name
        self.compare = _Comparer(pollutants, scale, ARC_REPLAY_BOUND)

    def assert_exhausted(self) -> None:
        if self.script:
            raise Mismatch(
                f"{self.label}: noodl made no {self.script[0].method} into {self.name!r} "
                f"({len(self.script)} left)"
            )

    def _next(self, method: str, vqip: Tensor | None, tag, active: Tensor) -> Tensor:
        if not bool(active):
            return empty(self.p, ()) if vqip is None else torch.zeros_like(vqip)
        label = f"{self.label}: {method} into {self.name!r}"
        if not self.script:
            raise Mismatch(f"{label}: WSIMOD made no such call")
        call = self.script.pop(0)
        if (call.method, call.tag) != (method, tag):
            raise Mismatch(f"{label} (tag {tag!r}): WSIMOD called {call.method} (tag {call.tag!r})")
        if (call.vqip is None) != (vqip is None):
            raise Mismatch(f"{label}: request {vqip!r}, WSIMOD's {call.vqip!r}")
        if vqip is not None:
            self.compare.vqip(f"{label} request", vqip, call.vqip, volume_only=method == "pull_set")
        return to_vqip(self.p, call.reply)

    def push_set(self, vqip: Tensor, tag="default", *, active: Tensor) -> Tensor:
        return self._next("push_set", vqip, tag, active)

    def pull_set(self, vqip: Tensor, tag="default", *, active: Tensor) -> Tensor:
        return self._next("pull_set", vqip, tag, active)

    def push_check(self, vqip: Tensor | None = None, tag="default", *, active: Tensor) -> Tensor:
        return self._next("push_check", vqip, tag, active)

    def pull_check(self, vqip: Tensor | None = None, tag="default", *, active: Tensor) -> Tensor:
        return self._next("pull_check", vqip, tag, active)


def precondition_holds(day: ArcDay, n: int) -> bool:
    """X41's precondition on one captured day, from WSIMOD's own queue (D21): the end of the
    day does not find a due request waiting in the push direction and another in the pull
    direction (``time`` 0 before the end of the day). ``n`` is the arc's
    ``number_of_timesteps``; with ``n = 0`` nothing is queued. A ``DecayArcAlt`` has one
    direction, so it always holds."""
    if n == 0:
        return True
    waiting = {r["direction"] for r in day.queue_at_end if r.get("time") == 0}
    return waiting != set(_DIRECTIONS)


@dataclass
class _Group:
    """A request call WSIMOD made, the port calls it caused and whether it reached
    ``update_queue``."""

    request: ArcCall
    ports: list[ArcCall] = field(default_factory=list)
    queued: bool = False

    @property
    def direction(self) -> str:
        return _REQUESTS[self.request.method]


def _groups(day: ArcDay) -> list[_Group]:
    groups: list[_Group] = []
    for call in day.calls:
        if call.method in _REQUESTS:
            groups.append(_Group(call))
        elif call.method == "update_queue":
            groups[-1].queued = True
        else:
            groups[-1].ports.append(call)
    return groups


def _template(days: Sequence[Sequence[_Group]]) -> list[tuple[str, int, object]]:
    """The static request calls of one day, ``(direction, index, tag)`` in order: as many per
    direction as its busiest day makes, the directions in the order they meet within a day.
    noodl's call structure is the same every day (D2); a call WSIMOD did not make is inactive,
    and WSIMOD's calls fill the first slots of their direction. Days whose directions interleave
    cannot be laid on one template and are refused."""
    counts = dict.fromkeys(_DIRECTIONS, 0)
    tags: dict[str, list] = {d: [] for d in _DIRECTIONS}
    order: list[str] = []
    for groups in days:
        seen: list[str] = []
        per = dict.fromkeys(_DIRECTIONS, 0)
        for group in groups:
            d = group.direction
            if not seen or seen[-1] != d:
                seen.append(d)
            if per[d] == len(tags[d]):
                tags[d].append(group.request.tag)
            per[d] += 1
        if len(set(seen)) != len(seen):
            raise ValueError(f"a day's requests interleave the directions: {seen}")
        counts = {d: max(counts[d], per[d]) for d in _DIRECTIONS}
        if len(seen) == 2 and order and order != seen:
            raise ValueError(f"the directions meet as {seen} on one day and {order} on another")
        if len(seen) == 2:
            order = seen
    order += [d for d in _DIRECTIONS if d not in order]
    return [(d, k, tags[d][k]) for d in order for k in range(counts[d])]


class _ArcReplay:
    """:func:`replay_arc`'s state: the arc between its two ports, and what it decayed."""

    def __init__(self, network: WsimodNetwork, name: str, p: Pollutants, scale: float) -> None:
        self.network, self.name, self.p = network, name, p
        self.spec = next(a for a in network.arcs if a.name == name)
        self.nodes = {n.name: n for n in network.nodes}
        ports = {
            end: ScriptedPort(
                self.nodes[end],
                p,
                wsimod_type=NODE_SCHEMAS[self.nodes[end].cls].renamed_type,
                scale=scale,
            )
            for end in (self.spec.in_port, self.spec.out_port)
        }
        self.in_port, self.out_port = ports[self.spec.in_port], ports[self.spec.out_port]
        self.arc = ARC_CLASSES[self.spec.cls](self.spec, self.in_port, self.out_port, p)
        self.alt = isinstance(self.arc, DecayArcAlt)
        self.n = self.arc.number_of_timesteps
        self.compare = _Comparer(p, scale, ARC_REPLAY_BOUND)
        self.decays: list[tuple[str, int, Tensor]] = []  # (phase, template place, diff)
        self.phase, self.place = "entry", 0
        original = self.arc.make_decay

        def recording(vqip: Tensor):
            result = original(vqip)
            self.decays.append((self.phase, self.place, result[1]))
            return result

        self.arc.make_decay = recording
        self.day = 0
        self.last: dict[str, int | None] = dict.fromkeys(_DIRECTIONS)  # last call to update_queue

    def new_day(self, k: int) -> Ctx:
        """The day's context, and the arc's decay bound at its ``in_port``'s temperature: the
        configuration's ``("temperature", date)`` row, set on the port (no node lists it in its
        ``configured_parameters``)."""
        ctx = Ctx(self.p, self.network.float_accuracy, (), day=k, node_names=[self.name])
        self.arc.ctx = ctx
        row = self.nodes[self.spec.in_port].data[("temperature", self.network.dates[k])]
        self.in_port.params = {"temperature": torch.tensor(float(row), dtype=torch.float64)}
        self.arc.bind_decay(
            self.arc.defaults["decays.constant"], self.arc.defaults["decays.exponent"]
        )
        return ctx

    def trace(self, template: list[tuple[str, int, object]], day: int) -> None:
        """``WsimodDay._trace`` (D2): one day of every request call, inactive, fixes the arc's
        call labels; its values are discarded."""
        arc = self.arc
        arc.begin_tracing()
        self.new_day(day)
        for direction, _, tag in template:
            handle = arc.send_push_request if direction == "push" else arc.send_pull_request
            handle(empty(self.p, ()), active=torch.tensor(False), tag=tag)
        arc.end_timestep()
        arc.end_tracing()
        self.decays.clear()

    def call(self, label: str, group: _Group | None, direction: str, tag) -> None:
        """One request call of the template: WSIMOD's ``group``, or inactive (``None``)."""
        arc, compare, p = self.arc, self.compare, self.p
        for port in (self.in_port, self.out_port):
            port.label = label
            port.script = (
                []
                if group is None
                else [
                    c
                    for c in group.ports
                    if (c.method in _OUT_PORT_CALLS) == (port is self.out_port)
                ]
            )
        handle = arc.send_push_request if direction == "push" else arc.send_pull_request
        if group is None:
            handle(empty(p, ()), active=torch.tensor(False), tag=tag)
            return
        request = group.request
        compare.value(f"{label}: flow_in at entry", arc.flow_in, request.flow_in)
        compare.vqip(f"{label}: vqip_in at entry", arc.vqip_in, request.vqip_in)
        kwargs = {"force": request.force} if direction == "push" else {}
        reply = handle(to_vqip(p, request.vqip), active=torch.tensor(True), tag=tag, **kwargs)
        for port in (self.in_port, self.out_port):
            port.assert_exhausted()
        compare.vqip(f"{label}: reply", reply, request.reply)
        compare.value(f"{label}: flow_in", arc.flow_in, request.flow_in_after)
        compare.vqip(f"{label}: vqip_in", arc.vqip_in, request.vqip_in_after)
        compare.value(f"{label}: flow_out", arc.flow_out, request.flow_out_after)
        compare.vqip(f"{label}: vqip_out", arc.vqip_out, request.vqip_out_after)
        if group.queued:
            self.last[group.direction] = self.day

    def check_queue(self, label: str, items: list[dict], shift: int) -> None:
        """The arc's queue against WSIMOD's: ``items`` are WSIMOD's queued requests (``row``,
        ``vqip``) or, for a ``DecayArcAlt``, its ``{key: vqip}`` dict. noodl's row ``(d, i, j)``
        holds the request entered ``i`` days before the last call to reach ``update_queue`` in
        direction ``d`` (``QueueArc``), and ``since`` counts the days since it, ``shift`` more
        after the end of the day."""
        arc, compare, p = self.arc, self.compare, self.p
        if self.alt:
            for path in ("register", "present"):
                compare_store(
                    compare, f"{label} {path}", f"internal_arc.{path}", getattr(arc, path), items
                )
            return
        if not self.n:
            return
        expected: dict[tuple[int, int, int], dict] = {}
        for item in items:
            direction, entry_day, j = item["row"]
            d = _DIRECTIONS.index(direction)
            i = self.last[direction] - entry_day
            if not 0 <= i < self.n:
                raise Mismatch(
                    f"{label}: WSIMOD's request {item['row']} waits {i} days (n={self.n})"
                )
            expected[d, i, j] = item["vqip"]
        zero = dict.fromkeys(_keys(p), 0.0)
        for d, direction in enumerate(_DIRECTIONS):
            for i in range(self.n):
                for j in range(arc.rows.shape[-2]):
                    compare.vqip(
                        f"{label} rows[{direction}, {i}, {j}]",
                        arc.rows[d, i, j],
                        expected.get((d, i, j), zero),
                    )
            last = self.last[direction]
            since = self.n if last is None else min(self.day - last + shift, self.n)
            compare.value(f"{label} since[{direction}]", arc.since[d], since)

    def end_day(self, label: str, day: ArcDay, place_of: dict[int, int]) -> None:
        arc, compare = self.arc, self.compare
        entries = {place: diff for phase, place, diff in self.decays if phase == "entry"}
        for index, diff in day.entry_diffs:
            compare.vqip(f"{label} entry call {index}", entries[place_of[index]], diff)
        before = {r["slot"]: r["vqip"] for r in day.queue_at_end} if self.alt else day.queue_at_end
        self.check_queue(f"{label} before end_timestep", before, 0)
        self.decays.clear()
        self.phase = "end"
        arc.end_timestep()
        self.phase = "entry"
        ends = [diff for _, _, diff in self.decays]
        width = 0 if self.alt else arc.rows.shape[-2]
        for row, diff in day.end_diffs:
            if self.alt:
                index = row[1] - 1 if row[1] else arc.length
            else:
                direction, entry_day, j = row
                d = _DIRECTIONS.index(direction)
                index = (d * self.n + self.last[direction] - entry_day) * width + j
            compare.vqip(f"{label} end row {row}", ends[index], diff)
        self.check_queue(f"{label} after end_timestep", day.post["queue"], 1)
        if self.alt:
            compare.vqip(f"{label} total_decayed", arc.total_decayed, day.total_decayed_after)


def replay_arc(
    network: WsimodNetwork,
    name: str,
    days: Sequence[ArcDay],
    *,
    scale: float,
    pollutants: Pollutants | None = None,
    contexts: list | None = None,
) -> None:
    """Replay the captured ``days`` (:class:`ArcDay`\\ s of WSIMOD's arc ``name``) into noodl's
    arc, built alone between :class:`ScriptedPort`\\ s. Raises :class:`Mismatch` at the first
    difference, naming the arc, the day and the call or row (every bound is 0, bitwise):

    - each request call's arguments into the ports, its reply and the arc's four accumulators;
    - each entry ``diff`` (by its call) and each end-of-day ``diff`` (by its row);
    - the queued requests before and after ``end_timestep``: a ``DecayArc``'s rows and ``since``,
      a ``DecayArcAlt``'s register and key set and its ``total_decayed``.

    A ``DecayArc``'s calls are laid on a static template (:func:`_template`), and a tracing day
    over it fixes the call labels first, as ``WsimodDay._trace`` does (D2). The arc reads its
    ``in_port``'s temperature of the day (:meth:`_ArcReplay.new_day`). A list passed as
    ``contexts`` receives each day's ``Ctx`` after its ``end_timestep``, with the day's terms and
    events.
    """
    p = network.pollutants if pollutants is None else pollutants
    replay = _ArcReplay(network, name, p, scale)
    all_groups = [_groups(day) for day in days]
    template = [] if replay.alt else _template(all_groups)
    if template:
        replay.trace(template, days[0].day)
    for day, groups in zip(days, all_groups, strict=True):
        label = f"arc {name!r} day {day.day}"
        replay.day = day.day
        ctx = replay.new_day(day.day)
        replay.decays.clear()
        place_of: dict[int, int] = {}
        if replay.alt:
            slots = [(g.direction, g, g.request.tag) for g in groups]
        else:
            by = {d: [g for g in groups if g.direction == d] for d in _DIRECTIONS}
            slots = [(d, by[d][k] if k < len(by[d]) else None, tag) for d, k, tag in template]
        for place, (direction, group, tag) in enumerate(slots):
            replay.place = place
            if group is not None:
                place_of[groups.index(group)] = place
            replay.call(f"{label} call {place}", group, direction, tag)
        replay.end_day(label, day, place_of)
        if contexts is not None:
            contexts.append(ctx)
