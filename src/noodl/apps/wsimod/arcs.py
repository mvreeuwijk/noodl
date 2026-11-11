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
    ) -> Tensor:
        """``QueueArc.send_push_request`` (``arcs.py:360-407``); returns what was not pushed.

        A push below ``FLOAT_ACCURACY`` returns empty before the queue (``:378-379``): it is
        neither enqueued nor does it deliver slot 0, and its water is recorded as dropped
        (spec, defect table; for a ``QueueTank`` the tank's ``storage`` keeps it). Otherwise
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
    "PushArc": PushArc,
    "SewerArc": SewerArc,
    "WeirArc": WeirArc,
}
