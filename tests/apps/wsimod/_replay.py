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
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

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

__all__ = ["Mismatch", "ScriptedArc", "StoreNode", "replay"]


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
) -> float:
    """Replay ``calls`` (``NodeCall``\\ s captured from WSIMOD) into noodl's node ``name``.

    ``pollutants`` defaults to the network's; ``QUANTITY_ONLY`` replays the volumes. Raises
    :class:`Mismatch` at the first difference beyond the node class's ``replay_bound`` x
    ``scale``; returns the worst difference relative to ``scale``.
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
        node.ctx = Ctx(p, network.float_accuracy, (), day=call.day, node_names=[name])
        script[:] = list(call.arc_calls)
        for arc in arcs.values():
            arc.label, arc.compare = label, compare
            arc.known = f"arcs.{arc.name}.flow_in" in call.pre
            flow_in = call.pre.get(f"arcs.{arc.name}.flow_in", 0.0)
            arc.flow_in = torch.tensor(float(flow_in), dtype=torch.float64)
            arc.vqip_in = to_vqip(p, call.pre.get(f"arcs.{arc.name}.vqip_in", {}))
        node.bind(node.configured_parameters(network.dates[call.day]))
        node.load_stores({path: to_vqip(p, call.pre[path]) for path in node.store_paths})
        if call.method in type(node).step_methods:
            reply = getattr(node, call.method)(active=active)
        else:
            vqip = None if call.vqip is None else to_vqip(p, call.vqip)
            reply = getattr(node, call.method)(vqip, call.tag, active=active)
        for arc in arcs.values():
            arc.assert_exhausted()
        if call.reply is not None:
            compare.vqip(f"{label} reply", reply, call.reply)
        stores = node.stores()
        for path in node.store_paths:
            compare.vqip(f"{label} post {path}", stores[path], call.post[path])
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
