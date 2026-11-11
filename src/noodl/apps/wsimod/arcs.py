"""WSIMOD 0.8.1's plain arcs (``wsimod/arcs/arcs.py:18-270``, ``:808-905``), batched.

An arc clips a push or a pull to ``capacity - flow_in`` and to the far node's check, passes the
rest on, and accumulates what it moved in ``flow_in`` and ``vqip_in`` until ``end_timestep``,
so capacity is served first come, first served (spec, Protocol). Every call takes the caller's
``active`` mask: the accumulators are written only where it holds (D3).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from noodl.apps.wsimod.engine import Ctx, dmax, dmin
from noodl.apps.wsimod.vqip import (
    Pollutants,
    empty,
    extract_vqip,
    safe_den,
    sum_vqip,
    v_change_vqip,
    write,
)

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.config import ArcSpec, WsimodNetwork
    from noodl.apps.wsimod.node import Node, Tag

__all__ = [
    "ARC_CLASSES",
    "AltQueueArc",
    "Arc",
    "PullArc",
    "PushArc",
    "QueueArc",
    "SewerArc",
    "WeirArc",
    "check_travel_time",
]


class Arc:
    """``Arc`` (``arcs.py:18-270``): one capacity-limited link from ``in_port`` to ``out_port``.

    ``capacity`` and ``preference`` start at the configured values; :meth:`bind` replaces them
    with the day's (batched) values. ``in_type`` and ``out_type`` are the ends' WSIMOD class
    names after the renames, the keys of the nodes' ``in_arcs_type``/``out_arcs_type``.
    """

    ctx: Ctx
    # the stores carried from one day to the next (the keys of :meth:`stores`)
    carried_paths: tuple[str, ...] = ()

    def __init__(self, spec: ArcSpec, in_port: Node, out_port: Node, pollutants: Pollutants):
        self.spec = spec
        self.name = spec.name
        # the element of the decision sites and call counts (engine.py, Decision records)
        self.site = f"arc:{spec.name}"
        self.p = pollutants
        self.in_port = in_port
        self.out_port = out_port
        self.in_type = in_port.wsimod_type
        self.out_type = out_port.wsimod_type
        self.capacity = torch.tensor(float(spec.capacity), dtype=torch.float64)
        self.preference = torch.tensor(float(spec.preference), dtype=torch.float64)
        Arc.end_timestep(self)
        # arcs.py:70-80: the ports learn of the arc in construction order
        in_port.connect(self, "out")
        out_port.connect(self, "in")

    def bind(self, capacity: Tensor, preference: Tensor) -> None:
        """This day's ``capacity`` and ``preference`` (batched)."""
        self.capacity = torch.as_tensor(capacity, dtype=torch.float64)
        self.preference = torch.as_tensor(preference, dtype=torch.float64)

    @classmethod
    def validate(cls, spec: ArcSpec, network: WsimodNetwork) -> None:
        """Refuse by name what this arc class cannot run (D7); a plain arc refuses nothing."""

    def stores(self) -> dict[str, Tensor]:
        """Carried state: none for a plain arc (its accumulators reset every day)."""
        return {}

    def load_stores(self, stores: Mapping[str, Tensor]) -> None:
        if stores:
            raise KeyError(f"arc {self.name!r} carries no state, got {sorted(stores)}")

    def end_timestep(self) -> None:
        """``arcs.py:259-265``: reset the day's accumulators."""
        self.flow_in = torch.zeros((), dtype=torch.float64)
        self.flow_out = self.flow_in
        self.vqip_in = empty(self.p, ())
        self.vqip_out = self.vqip_in

    def _accumulate(self, moved: Tensor, active: Tensor) -> None:
        """``arcs.py:152-156``, ``:191-195``."""
        self.flow_in = write(active, self.flow_in + moved[..., 0], self.flow_in)
        self.flow_out = self.flow_in
        self.vqip_in = write(active, sum_vqip(self.p, self.vqip_in, moved), self.vqip_in)
        self.vqip_out = self.vqip_in

    def send_push_request(
        self, vqip: Tensor, *, active: Tensor, tag: Tag = "default", force: bool = False
    ) -> Tensor:
        """``arcs.py:112-158``: push ``vqip`` to ``out_port``; returns what was not pushed.

        ``force`` skips the clip but still adds to ``flow_in`` (``:131-152``), so the pipe
        excess can go negative afterwards; that is transcribed, not clamped.
        """
        p = self.p
        call = self.ctx.call(self.site, "send_push_request")
        if force:
            not_pushed = torch.zeros_like(vqip)
        else:
            excess_in = self.get_excess("push", vqip, active=active, tag=tag)
            clip = dmax(
                self.ctx,
                f"{self.site}/send_push_request/clip",
                vqip[..., 0] - excess_in[..., 0],
                0.0,
                active=active,
                call=call,
            )
            not_pushed = v_change_vqip(p, vqip, clip)
        vqip = extract_vqip(p, vqip, not_pushed)
        reply = self.out_port.push_set(vqip, tag, active=active)
        vqip = extract_vqip(p, vqip, reply)
        reply = sum_vqip(p, reply, not_pushed)
        self._accumulate(vqip, active)
        return reply

    def send_pull_request(self, vqip: Tensor, *, active: Tensor, tag: Tag = "default") -> Tensor:
        """``arcs.py:160-197``: pull ``vqip``'s volume from ``in_port``; returns what was obtained.

        The request's additive entries are not rescaled (D4): no in-scope ``pull_set`` reads
        them (spec, Protocol, Request mutation); the request passed on carries the clipped
        volume and the caller's other entries.
        """
        call = self.ctx.call(self.site, "send_pull_request")
        volume = vqip[..., 0]
        excess_in = self.get_excess("pull", vqip, active=active, tag=tag)[..., 0]
        not_pulled = dmax(
            self.ctx,
            f"{self.site}/send_pull_request/clip",
            volume - excess_in,
            0.0,
            active=active,
            call=call,
        )
        volume = volume - not_pulled
        request = torch.cat(
            [volume[..., None], vqip[..., 1:].expand(volume.shape + (self.p.K,))], -1
        )
        reply = self.in_port.pull_set(request, tag, active=active)
        self._accumulate(reply, active)
        return reply

    def send_push_check(
        self, vqip: Tensor | None = None, *, active: Tensor, tag: Tag = "default"
    ) -> Tensor:
        """``arcs.py:199-212``."""
        return self.get_excess("push", vqip, active=active, tag=tag)

    def send_pull_check(
        self, vqip: Tensor | None = None, *, active: Tensor, tag: Tag = "default"
    ) -> Tensor:
        """``arcs.py:214-228``."""
        return self.get_excess("pull", vqip, active=active, tag=tag)

    def get_excess(
        self, direction: str, vqip: Tensor | None = None, *, active: Tensor, tag: Tag = "default"
    ) -> Tensor:
        """``arcs.py:230-257``: ``min(capacity - flow_in, node check volume)`` at the node
        check's concentration. ``vqip=None`` asks the node for its unclipped check."""
        if direction not in ("push", "pull"):
            raise ValueError(f"arc {self.name!r}: direction {direction!r} is not push or pull")
        call = self.ctx.call(self.site, f"get_excess/{direction}")
        pipe_excess = self.capacity - self.flow_in
        if direction == "push":
            node_excess = self.out_port.push_check(vqip, tag, active=active)
        else:
            node_excess = self.in_port.pull_check(vqip, tag, active=active)
        excess = dmin(
            self.ctx,
            f"{self.site}/get_excess/{direction}/min",
            pipe_excess,
            node_excess[..., 0],
            active=active,
            call=call,
        )
        return v_change_vqip(self.p, node_excess, excess)


def check_travel_time(where: str, key: str, value: object) -> int:
    """A travel time or time-area key as an ``int``, or a ``ValueError`` naming ``where`` and
    ``key``. An ``AltQueueArc`` shifts its integer keys only (``arcs.py:653-660``), so any
    other key would never reach the out-port: the spec refuses it by name (Nodes,
    AltQueueArc)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: {key} {value!r} is not a whole number of days >= 0")
    if int(value) != value or value < 0:
        raise ValueError(f"{where}: {key} {value!r} is not a whole number of days >= 0")
    return int(value)


def _put_row(register: Tensor, t: int, row: Tensor, mask: Tensor) -> Tensor:
    """``register`` (..., L+1, 1+K) with row ``t`` set to ``row`` (..., 1+K) where ``mask``."""
    one = torch.zeros(register.shape[-2], dtype=torch.bool)
    one[t] = True
    sel = torch.as_tensor(mask, dtype=torch.bool)[..., None, None] & one[:, None]
    return torch.where(sel, row[..., None, :], register)


class AltQueueArc(Arc):
    """``AltQueueArc`` (``arcs.py:555-665``, ``send_push_request`` from ``QueueArc``,
    ``:360-407``): a push enters the queue under its travel time and every push delivers the
    queue's slot 0 at once (``update_queue``).

    The ``{time: vqip}`` queue is a dense shift register ``register`` (..., L+1, 1+K) (spec,
    Representation): ``L`` is the largest travel time the structure can enter, at least 1 (the
    keys 0 and 1 exist from the start, ``:567``). An absent key is an empty row. ``present``
    (..., L+1) is WSIMOD's key set: a first entry under an absent key is assigned
    (``:599-601``), a later one summed with ``sum_vqip`` (``:595-598``), and the two differ by
    rounding in the non-additive columns (``(c*V + 0)/V`` is not always ``c``). Keys are never
    removed; the shift creates key ``i`` when key ``i+1`` exists (``:655-658``).

    A user ``AltQueueArc`` shifts at ``end_timestep`` and delivers only on its next push,
    because nothing calls its ``update_queue`` at the end of the day; a ``QueueTank``'s
    internal arc is also flushed by the tank's ``end_timestep`` (``tanks.py:631-635``).
    ``site`` replaces ``"arc:<name>"`` in the decision sites (a tank's internal arc has no
    name of its own, ``tanks.py:484-486``); ``owner`` is the node an event is recorded at.
    """

    carried_paths: tuple[str, ...] = ("register", "present")

    def __init__(
        self,
        spec: ArcSpec,
        in_port,
        out_port,
        pollutants: Pollutants,
        *,
        length: int | None = None,
        site: str | None = None,
        owner: str | None = None,
    ) -> None:
        self.number_of_timesteps = check_travel_time(
            f"arc {spec.name!r}", "number_of_timesteps", spec.number_of_timesteps
        )
        self.length = max(1, self.number_of_timesteps if length is None else int(length))
        if self.length < self.number_of_timesteps:
            raise ValueError(
                f"arc {spec.name!r}: register length {self.length} is shorter than "
                f"number_of_timesteps {self.number_of_timesteps}"
            )
        super().__init__(spec, in_port, out_port, pollutants)
        if site is not None:
            self.site = site
        self.owner = in_port.name if owner is None else owner
        # arcs.py:567-568: keys 0 and 1, both empty
        self.register = torch.zeros(self.length + 1, 1 + pollutants.K, dtype=torch.float64)
        self.present = torch.arange(self.length + 1) < 2

    @property
    def present(self) -> Tensor:
        """WSIMOD's key set, (..., L+1) bool."""
        return self._present

    @present.setter
    def present(self, value: Tensor) -> None:
        # a store loaded through Node.load_stores arrives as float64
        value = torch.as_tensor(value)
        self._present = value if value.dtype == torch.bool else value != 0

    @classmethod
    def validate(cls, spec: ArcSpec, network: WsimodNetwork) -> None:
        """``number_of_timesteps`` must be a whole number of days >= 0 (spec, Nodes,
        AltQueueArc); the error names the arc and the key."""
        check_travel_time(f"arc {spec.name!r}", "number_of_timesteps", spec.number_of_timesteps)

    def stores(self) -> dict[str, Tensor]:
        """``register`` (the queue's values) and ``present`` (its key set)."""
        return {"register": self.register, "present": self.present}

    def load_stores(self, stores: Mapping[str, Tensor]) -> None:
        unknown = set(stores) - {"register", "present"}
        if unknown:
            raise KeyError(f"arc {self.name!r} has no store {sorted(unknown)[0]!r}")
        if "register" in stores:
            self.register = torch.as_tensor(stores["register"], dtype=torch.float64)
        if "present" in stores:
            self.present = stores["present"]

    def send_push_request(
        self,
        vqip: Tensor,
        *,
        active: Tensor,
        tag: Tag = "default",
        force: bool = False,
        time: int = 0,
        record_drop: bool = True,
    ) -> Tensor:
        """``QueueArc.send_push_request`` (``arcs.py:360-407``); returns what was not pushed.

        A push below ``FLOAT_ACCURACY`` returns empty before the queue (``:378-379``): it is
        neither enqueued nor does it deliver slot 0, and its water is recorded as dropped
        (spec, defect table). A ``QueueTank`` passes ``record_drop=False``: its ``storage``
        keeps the water, so the mass balance counts it there and not as dropped (W49). Otherwise
        the clip (``:382-390``), the entry under ``time + number_of_timesteps`` (``:393-396``,
        ``enter_arc`` ``:409-430``, ``enter_queue`` ``:581-601``), then ``update_queue``
        (``:399``), whose backflow joins the reply and leaves ``vqip_in`` (``:400-405``).
        """
        p, ctx = self.p, self.ctx
        active = torch.as_tensor(active, dtype=torch.bool)
        t = check_travel_time(f"arc {self.name!r}", "time", time) + self.number_of_timesteps
        if t > self.length:
            raise ValueError(
                f"{self.site}: travel time {t} exceeds the register length {self.length} "
                "the structure declared"
            )
        call = ctx.call(self.site, "send_push_request")
        tiny = ctx.decide(
            f"{self.site}/send_push_request/below_fa",
            vqip[..., 0],
            "<",
            ctx.fa,
            active=active,
            call=call,
        )
        if record_drop:
            ctx.drop(vqip, active & tiny)
        run = active & ~tiny
        if force:
            not_pushed = torch.zeros_like(vqip)
        else:
            excess_in = self.get_excess("push", vqip, active=run, tag=tag)
            clip = dmax(
                ctx,
                f"{self.site}/send_push_request/clip",
                vqip[..., 0] - excess_in[..., 0],
                0.0,
                active=run,
                call=call,
            )
            not_pushed = v_change_vqip(p, vqip, clip)
        entered = extract_vqip(p, vqip, not_pushed)
        # enter_arc (arcs.py:423-428): average_flow = V / (time + 1)
        self.flow_in = write(run, self.flow_in + entered[..., 0] / (t + 1), self.flow_in)
        self.vqip_in = write(run, sum_vqip(p, self.vqip_in, entered), self.vqip_in)
        # enter_queue (arcs.py:595-601): sum into an existing key, assign a new one
        row = self.register[..., t, :]
        row = torch.where(self.present[..., t, None], sum_vqip(p, row, entered), entered)
        self.register = _put_row(self.register, t, row, run)
        key = torch.arange(self.length + 1) == t
        self.present = self.present | (run[..., None] & key)
        backflow = self.update_queue(active=run)
        not_pushed = sum_vqip(p, not_pushed, backflow)
        # arcs.py:402-403: a print
        more = (backflow[..., 0] > vqip[..., 0]).detach()
        ctx.event("more_backflow_than_vqip", self.owner, run & more)
        self.vqip_in = write(run, extract_vqip(p, self.vqip_in, backflow), self.vqip_in)
        return torch.where(tiny[..., None], ctx.empty(), not_pushed)

    def update_queue(self, *, active: Tensor) -> Tensor:
        """``arcs.py:603-638``: push slot 0 to the out-port without a tag; the rejected part is
        the backflow (``backflow_enabled`` is always true here) and slot 0 is emptied."""
        p = self.p
        active = torch.as_tensor(active, dtype=torch.bool)
        total_removed = self.register[..., 0, :]
        backflow = self.out_port.push_set(total_removed, active=active)
        self.register = _put_row(self.register, 0, torch.zeros_like(total_removed), active)
        total_removed = v_change_vqip(p, total_removed, total_removed[..., 0] - backflow[..., 0])
        self.flow_out = write(active, self.flow_out + total_removed[..., 0], self.flow_out)
        self.vqip_out = write(active, sum_vqip(p, self.vqip_out, total_removed), self.vqip_out)
        return backflow

    def end_timestep(self) -> None:
        """``arcs.py:640-660``: reset the day's accumulators, then shift: ``new[i] = old[i+1]``
        for ``1 <= i < L``, ``new[L]`` empty, ``new[0] = sum_vqip(old[0], old[1])``. Key ``i``
        exists afterwards if key ``i`` or key ``i+1`` did."""
        Arc.end_timestep(self)
        old = self.register
        first = sum_vqip(self.p, old[..., 0, :], old[..., 1, :])
        tail = torch.zeros_like(old[..., :1, :])
        self.register = torch.cat([first[..., None, :], old[..., 2:, :], tail], dim=-2)
        here = self.present
        self.present = torch.cat([here[..., :-1] | here[..., 1:], here[..., -1:]], dim=-1)


_DIRECTIONS = ("push", "pull")


class QueueArc(Arc):
    """``QueueArc`` (``arcs.py:272-552``): every request waits ``number_of_timesteps`` days in
    the arc and is handed over by the arc's next call in that direction once it is due.

    WSIMOD keeps a list of requests, each with a countdown (``:432-446``), and ``update_queue``
    (``:448-528``) walks it in list order at the end of every request call that gets past the
    early return. noodl keeps dense rows (D1; spec, Representation, ``QueueArc`` slots):
    ``rows`` (..., 2, n, J, 1+K), direction ``push``/``pull`` first. Row ``(i, j)`` holds the
    request entered ``i`` days before the arc's last call in that direction that reached
    ``update_queue``, by the ``j``-th request call of its day in the arc's static call order.
    ``since`` (..., 2) counts the days since that call, saturating at ``n``, and starts at
    ``n``. A row is due iff ``since + i >= n``; due rows go out in the order ``i = n-1 ... 0``,
    then ``j``, which is WSIMOD's list order, and the waiting ones move to ``i + since``. A
    row's ``average_flow`` is ``V/(n+1)`` (``:423``): a waiting row's volume never changes,
    because ``backflow_enabled`` is always true here (``:399``, ``:510-512``). A zero row is
    absent.

    ``calls`` (D2) holds each direction's static per-day call labels, indexed by ``j``: the
    enclosing distribute-loop frames (:meth:`Ctx.frame`, which name the site and the round)
    and the tag. The first day that calls the arc records them, or :meth:`begin_tracing` and
    :meth:`end_tracing` bracket a tracing day; every later day must make the same calls in the
    same order, or the call raises (an internal error: noodl's call structure is static).
    ``J`` grows while the labels are recorded and is fixed afterwards. It is the larger of
    the two directions' counts, so the other direction's surplus rows stay empty.
    """

    carried_paths: tuple[str, ...] = ("rows", "since")

    def __init__(self, spec: ArcSpec, in_port: Node, out_port: Node, pollutants: Pollutants):
        self.number_of_timesteps = check_travel_time(
            f"arc {spec.name!r}", "number_of_timesteps", spec.number_of_timesteps
        )
        super().__init__(spec, in_port, out_port, pollutants)
        n = self.number_of_timesteps
        self.rows = torch.zeros(2, n, 0, 1 + pollutants.K, dtype=torch.float64)
        self.since = torch.full((2,), float(n), dtype=torch.float64)
        self.calls: dict[str, list[str]] = {d: [] for d in _DIRECTIONS}
        self._tags: dict[str, list[Tag]] = {d: [] for d in _DIRECTIONS}
        self._frozen = False
        self._seen = dict.fromkeys(_DIRECTIONS, 0)

    @classmethod
    def validate(cls, spec: ArcSpec, network: WsimodNetwork) -> None:
        """``number_of_timesteps`` must be a whole number of days >= 0 (spec, Nodes); the error
        names the arc and the key."""
        check_travel_time(f"arc {spec.name!r}", "number_of_timesteps", spec.number_of_timesteps)

    # --- state ------------------------------------------------------------------------------
    def stores(self) -> dict[str, Tensor]:
        """``rows`` (the waiting requests) and ``since`` (the anchors)."""
        return {"rows": self.rows, "since": self.since}

    def load_stores(self, stores: Mapping[str, Tensor]) -> None:
        unknown = set(stores) - {"rows", "since"}
        if unknown:
            raise KeyError(f"arc {self.name!r} has no store {sorted(unknown)[0]!r}")
        if "rows" in stores:
            rows = torch.as_tensor(stores["rows"], dtype=torch.float64)
            n = self.number_of_timesteps
            if (
                rows.ndim < 4
                or tuple(rows.shape[-4:-2]) != (2, n)
                or rows.shape[-1] != 1 + self.p.K
            ):
                raise ValueError(
                    f"arc {self.name!r}: rows of shape {tuple(rows.shape)}, expected "
                    f"(..., 2, {n}, J, {1 + self.p.K})"
                )
            if rows.shape[-2] < self.rows.shape[-2]:
                raise ValueError(
                    f"arc {self.name!r}: rows for {rows.shape[-2]} calls, the arc makes "
                    f"{self.rows.shape[-2]}"
                )
            self.rows = rows
        if "since" in stores:
            since = torch.as_tensor(stores["since"], dtype=torch.float64)
            n = self.number_of_timesteps
            if since.ndim < 1 or since.shape[-1] != 2:
                raise ValueError(
                    f"arc {self.name!r}: since of shape {tuple(since.shape)}, expected (..., 2)"
                )
            if bool(((since < 0) | (since > n) | (since != since.round())).any()):
                raise ValueError(
                    f"arc {self.name!r}: since must hold whole days in [0, {n}], got "
                    f"{since.tolist()}"
                )
            self.since = since

    def end_timestep(self) -> None:
        """``arcs.py:530-545``: reset the day's accumulators and age every request by a day
        (``since = min(since + 1, n)``; the rows keep their places). The day's calls are
        checked against the recorded labels, or recorded if this was the first day that
        called the arc (D2)."""
        Arc.end_timestep(self)
        self.since = torch.clamp(self.since + 1, max=float(self.number_of_timesteps))
        if self._frozen:
            for d in _DIRECTIONS:
                if self._seen[d] and self._seen[d] != len(self.calls[d]):
                    raise RuntimeError(
                        f"{self.site}: {self._seen[d]} {d} request calls today, "
                        f"{len(self.calls[d])} on the traced day (D2)"
                    )
        elif any(self._seen.values()):
            self._frozen = True
        self._seen = dict.fromkeys(_DIRECTIONS, 0)

    # --- the static call labels (D2) --------------------------------------------------------
    def begin_tracing(self) -> None:
        """Forget the recorded labels: the next calls record them afresh (D2's tracing day,
        whose values the caller discards)."""
        self.calls = {d: [] for d in _DIRECTIONS}
        self._tags = {d: [] for d in _DIRECTIONS}
        self._frozen = False
        self._seen = dict.fromkeys(_DIRECTIONS, 0)

    def end_tracing(self) -> dict[str, list[str]]:
        """Fix the labels recorded since :meth:`begin_tracing` (and so ``J``); returns them."""
        self._frozen = True
        return {d: list(labels) for d, labels in self.calls.items()}

    def _enter(self, direction: str, tag: Tag) -> int:
        """This request call's index ``j`` in the day's static call order, recorded or checked
        (D2)."""
        j = self._seen[direction]
        self._seen[direction] = j + 1
        frames = " > ".join(self.ctx.frames) or "-"
        label = f"{frames} | tag {tag!r}"
        labels = self.calls[direction]
        if self._frozen:
            if j >= len(labels) or labels[j] != label:
                expected = labels[j] if j < len(labels) else "no such call"
                raise RuntimeError(
                    f"{self.site}: {direction} request call {j} is {label!r}; the traced day "
                    f"made {expected!r} (D2: the call structure must not change)"
                )
            return j
        labels.append(label)
        self._tags[direction].append(tag)
        width = max(len(v) for v in self.calls.values())
        if width > self.rows.shape[-2]:
            pad = list(self.rows.shape)
            pad[-2] = width - self.rows.shape[-2]
            self.rows = torch.cat([self.rows, self.rows.new_zeros(pad)], dim=-2)
        return j

    # --- the rows ---------------------------------------------------------------------------
    def _put(self, d: int, i: int, j: int, row: Tensor, mask: Tensor) -> None:
        """Row ``(d, i, j)`` set to ``row`` where ``mask``."""
        one = torch.zeros(self.rows.shape[-4:-1], dtype=torch.bool)
        one[d, i, j] = True
        sel = torch.as_tensor(mask, dtype=torch.bool)[..., None, None, None] & one
        self.rows = torch.where(sel[..., None], row[..., None, None, None, :], self.rows)

    def _shift(self, d: int, mask: Tensor) -> None:
        """Move direction ``d``'s rows from ``i`` to ``i + since`` where ``mask`` (D1); the due
        rows are already empty. A selection, not a sum: at most one ``i`` lands on a slot."""
        old = self.rows[..., d, :, :, :]
        since = self.since[..., d]
        slots = []
        for t in range(self.number_of_timesteps):
            slot = torch.zeros_like(old[..., t, :, :])
            for i in range(t + 1):
                here = (since == float(t - i))[..., None, None]
                slot = torch.where(here, old[..., i, :, :], slot)
            slots.append(slot)
        moved = torch.stack(slots, dim=-3)
        moved = torch.where(torch.as_tensor(mask)[..., None, None, None], moved, old)
        other = self.rows[..., 1 - d, :, :, :]
        pair = (moved, other) if d == 0 else (other, moved)
        self.rows = torch.stack(torch.broadcast_tensors(*pair), dim=-4)

    def _update_queue(
        self, direction: str, new: Tensor, j: int, tag: Tag, *, active: Tensor, call: int
    ) -> Tensor:
        """``update_queue`` (``arcs.py:448-528``) after the call's own request ``new`` entered
        (``:393-396``): walk the list in order, the waiting rows ``i = n-1 ... 0`` then ``j``,
        then ``new``. A request below ``FLOAT_ACCURACY`` is removed and its water dropped
        (``:483-485``). A due one is handed over: a push to ``out_port.push_set`` under its own
        tag, the rejected part becoming backflow (``:486-512``), a pull as it is (``:492-494``).
        Returns the backflow (push) or the water handed over (pull) (``:523-526``).

        The waiting rows are handed over before ``new`` is stored as row ``(0, j)``: for
        ``n >= 1`` that is WSIMOD's order, because ``new`` is not due; for ``n = 0`` ``new`` is
        handed over last, where WSIMOD's list has it, and never stored. WSIMOD removes the
        done requests by equality (``:519-520``); two equal requests meet the same fate, so
        that is removal by identity."""
        p, ctx = self.p, self.ctx
        n = self.number_of_timesteps
        d = _DIRECTIONS.index(direction)
        where = f"{self.site}/update_queue/{direction}"
        total = {"removed": ctx.empty(), "backflow": ctx.empty()}

        def visit(row: Tensor, due: Tensor, label: str, row_tag: Tag) -> Tensor:
            """One list entry; returns where it leaves the list."""
            volume = row[..., 0]
            present = active & (volume != 0).detach()
            below = ctx.decide(
                f"{where}:{label}/below_fa", volume, "<", ctx.fa, active=present, call=call
            )
            ctx.drop(row, present & below)
            deliver = present & ~below & due
            if direction == "push":
                reply = self.out_port.push_set(row, row_tag, active=deliver)
                removed = volume - reply[..., 0]
            else:
                removed = volume  # :492-494: already pulled
            # :500: flow_out += average_flow * removed / V, average_flow = V/(n+1) (:423)
            flow = self.flow_out + volume / (n + 1) * removed / safe_den(volume)
            self.flow_out = write(deliver, flow, self.flow_out)
            handed = v_change_vqip(p, row, removed)
            total["removed"] = write(
                deliver, sum_vqip(p, total["removed"], handed), total["removed"]
            )
            if direction == "push":
                rejected = v_change_vqip(p, row, volume - removed)
                # :511: sum_vqip(rejected, total_backflow), in that order
                total["backflow"] = write(
                    deliver, sum_vqip(p, rejected, total["backflow"]), total["backflow"]
                )
            return present & (below | due)

        tags = self._tags[direction]
        for i in reversed(range(n)):
            due = (self.since[..., d] + i >= n).detach()
            for jj in range(self.rows.shape[-2]):
                row = self.rows[..., d, i, jj, :]
                row_tag = tags[jj] if jj < len(tags) else "default"
                left = visit(row, due, f"{i},{jj}", row_tag)
                self._put(d, i, jj, torch.zeros_like(row), left)
        left = visit(new, torch.full(ctx.shape, n == 0), "new", tag)
        if n > 0:
            self._shift(d, active)
            # a zero request is absent: storing it would leave its non-additive entries behind
            stored = active & ~left & (new[..., 0] != 0).detach()
            self._put(d, 0, j, new, stored)
            hit = torch.as_tensor(active)[..., None] & (torch.arange(2) == d)
            self.since = torch.where(hit, torch.zeros_like(self.since), self.since)
        # :516: vqip_out = sum_vqip(vqip_out, total_removed), on every call that got here
        self.vqip_out = write(active, sum_vqip(p, self.vqip_out, total["removed"]), self.vqip_out)
        return total["backflow"] if direction == "push" else total["removed"]

    # --- the requests -----------------------------------------------------------------------
    def send_push_request(
        self, vqip: Tensor, *, active: Tensor, tag: Tag = "default", force: bool = False
    ) -> Tensor:
        """``arcs.py:360-407``; returns what was not pushed.

        A push below ``FLOAT_ACCURACY`` returns empty at once (``:378-379``): it is neither
        enqueued nor does it hand the due rows over, and ``since`` keeps counting; its water is
        dropped (spec, defect table). Otherwise the clip (``:382-390``), the entry with
        ``average_flow = V/(n+1)`` (``enter_arc``, ``:409-430``) and ``update_queue``
        (``:399``), whose backflow joins the reply and leaves ``vqip_in`` (``:400-405``).
        """
        p, ctx = self.p, self.ctx
        active = torch.as_tensor(active, dtype=torch.bool)
        j = self._enter("push", tag)
        call = ctx.call(self.site, "send_push_request")
        tiny = ctx.decide(
            f"{self.site}/send_push_request/below_fa",
            vqip[..., 0],
            "<",
            ctx.fa,
            active=active,
            call=call,
        )
        ctx.drop(vqip, active & tiny)
        run = active & ~tiny
        if force:
            not_pushed = torch.zeros_like(vqip)
        else:
            excess_in = self.get_excess("push", vqip, active=run, tag=tag)
            clip = dmax(
                ctx,
                f"{self.site}/send_push_request/clip",
                vqip[..., 0] - excess_in[..., 0],
                0.0,
                active=run,
                call=call,
            )
            not_pushed = v_change_vqip(p, vqip, clip)
        entered = extract_vqip(p, vqip, not_pushed)
        self._enter_arc(entered, run)
        backflow = self._update_queue("push", entered, j, tag, active=run, call=call)
        not_pushed = sum_vqip(p, not_pushed, backflow)
        # arcs.py:402-403: a print
        more = (backflow[..., 0] > vqip[..., 0]).detach()
        ctx.event("more_backflow_than_vqip", self.in_port.name, run & more)
        self.vqip_in = write(run, extract_vqip(p, self.vqip_in, backflow), self.vqip_in)
        return torch.where(tiny[..., None], ctx.empty(), not_pushed)

    def send_pull_request(self, vqip: Tensor, *, active: Tensor, tag: Tag = "default") -> Tensor:
        """``arcs.py:318-358``: pull now, hand over what is due; returns the due pulls.

        The clip and ``pull_set`` take no tag (``:337``, ``:348``), so ``tag`` only labels the
        call (D2). The request's additive entries are not rescaled (D4): WSIMOD's rescale
        (``:341-343``) divides by zero for a zero-volume request with additive keys, a defect,
        and nothing reads the rescaled entries. There is no early return, so every active pull
        reaches ``update_queue`` and resets ``since``.
        """
        p, ctx = self.p, self.ctx
        active = torch.as_tensor(active, dtype=torch.bool)
        j = self._enter("pull", tag)
        call = ctx.call(self.site, "send_pull_request")
        volume = vqip[..., 0]
        excess_in = self.get_excess("pull", vqip, active=active)[..., 0]
        not_pulled = dmax(
            ctx,
            f"{self.site}/send_pull_request/clip",
            volume - excess_in,
            0.0,
            active=active,
            call=call,
        )
        volume = volume - not_pulled
        request = torch.cat([volume[..., None], vqip[..., 1:].expand(volume.shape + (p.K,))], -1)
        pulled = self.in_port.pull_set(request, active=active)
        self._enter_arc(pulled, active)
        return self._update_queue("pull", pulled, j, "default", active=active, call=call)

    def _enter_arc(self, vqip: Tensor, active: Tensor) -> None:
        """``enter_arc`` (``arcs.py:423-428``): ``flow_in += V/(n+1)``, ``vqip_in += vqip``."""
        n = self.number_of_timesteps
        self.flow_in = write(active, self.flow_in + vqip[..., 0] / (n + 1), self.flow_in)
        self.vqip_in = write(active, sum_vqip(self.p, self.vqip_in, vqip), self.vqip_in)


class PullArc(Arc):
    """``PullArc`` (``arcs.py:808-849``): pushes are refused."""

    def send_push_request(
        self, vqip: Tensor, *, active: Tensor, tag: Tag = "default", force: bool = False
    ) -> Tensor:
        """``send_push_deny`` (``:821-835``): returns the VQIP."""
        return vqip

    def send_push_check(
        self, vqip: Tensor | None = None, *, active: Tensor, tag: Tag = "default"
    ) -> Tensor:
        """``send_push_check_deny`` (``:837-849``): empty."""
        return self.ctx.empty()


class PushArc(Arc):
    """``PushArc`` (``arcs.py:852-893``): pulls are refused."""

    def send_pull_request(self, vqip: Tensor, *, active: Tensor, tag: Tag = "default") -> Tensor:
        """``send_pull_deny`` (``:865-879``): empty."""
        return self.ctx.empty()

    def send_pull_check(
        self, vqip: Tensor | None = None, *, active: Tensor, tag: Tag = "default"
    ) -> Tensor:
        """``send_pull_check_deny`` (``:881-893``): empty."""
        return self.ctx.empty()


class SewerArc(Arc):
    """``SewerArc`` (``arcs.py:896-899``): a plain arc."""


class WeirArc(SewerArc):
    """``WeirArc`` (``arcs.py:902-905``): a plain arc."""


ARC_CLASSES: dict[str, type[Arc]] = {
    "Arc": Arc,
    "AltQueueArc": AltQueueArc,
    "PullArc": PullArc,
    "QueueArc": QueueArc,
    "PushArc": PushArc,
    "SewerArc": SewerArc,
    "WeirArc": WeirArc,
}
