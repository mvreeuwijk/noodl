"""WSIMOD 0.8.1's plain arcs (``wsimod/arcs/arcs.py:18-270``, ``:808-905``), batched.

An arc clips a push or a pull to ``capacity - flow_in`` and to the far node's check, passes the
rest on, and accumulates what it moved in ``flow_in`` and ``vqip_in`` until ``end_timestep``,
so capacity is served first come, first served (spec, Protocol). Every call takes the caller's
``active`` mask: the accumulators are written only where it holds (D3).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor

from noodl.apps.wsimod.decay import Decay, DecaySpec, check_inert
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
    "ARC_EVENTS",
    "AltQueueArc",
    "Arc",
    "DECAY_ARCS",
    "DecayArc",
    "DecayArcAlt",
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
    # the mass-balance terms the arc records (``Ctx.term``) with their side, ``"in"`` or
    # ``"out"``, as ``Node.mass_balance`` (D11); ``run._balance`` collects both
    mass_balance: ClassVar[dict[str, str]] = {}

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

    def bind_decay(self, constant: Tensor, exponent: Tensor) -> None:
        """This day's ``decays.constant`` and ``decays.exponent`` ``(..., K)`` of a ``Decay*``
        arc (D15): a non-inert entry at a key its ``decays`` does not list is refused, naming the
        arc (:func:`~noodl.apps.wsimod.decay.check_inert`), then its
        :class:`~noodl.apps.wsimod.decay.Decay` is bound at the temperature of the node it reads
        (``core.py:464-470``): its ``parent``, else its ``in_port``. ``WsimodDay._bind`` calls it
        after every node is bound, so that temperature is the same day's. A quantity-only run
        decays nothing (``K = 0``); an arc without a ``decay`` attribute is refused."""
        if not hasattr(self, "decay"):
            raise TypeError(f"arc {self.name!r}: a {type(self).__name__} has no decay to bind")
        if self.p.K == 0:
            return
        constant = torch.as_tensor(constant, dtype=torch.float64)
        exponent = torch.as_tensor(exponent, dtype=torch.float64)
        if self.decay is None:
            check_inert(
                f"arc {self.name!r}",
                "decays.constant/exponent",
                torch.zeros(self.p.K, dtype=torch.bool),
                constant,
                exponent,
                self.p.names,
            )
            return
        source = _temperature_source(self)
        if "temperature" not in getattr(source, "params", {}):
            raise KeyError(
                f"arc {self.name!r}: node {source.name!r} has no ('temperature', t) forcing "
                "(core.py:484-487)"
            )
        self.decay.bind(self.ctx, constant, exponent, source.params["temperature"])

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


def _temperature_source(arc: Arc):
    """The node a decaying arc reads its temperature from (``DecayObj.__init__``,
    ``core.py:464-470``): its ``parent`` where it has one (a ``DecayQueueTank``'s internal arc,
    ``tanks.py:663-669``), else its ``in_port`` (a user ``Decay*`` arc, for either direction)."""
    parent = getattr(arc, "parent", None)
    return arc.in_port if parent is None else parent


def _arc_decay(arc: Arc, spec: ArcSpec, pollutants: Pollutants) -> Decay | None:
    """A user ``Decay*`` arc's :class:`Decay` and its ``decays.*`` driver defaults, dense over the
    ``K`` pollutants (inert ``(0, 1)`` where ``decays`` lists none, X9, X34). A quantity-only run
    decays nothing: the keys outside its pollutants are dropped, as ``Storage._decay_spec`` does.
    ``None`` when nothing decays; ``make_decay`` still reads the temperature then."""
    configured = spec.decays or DecaySpec((), {}, {})
    arc.defaults = {
        "decays.constant": torch.tensor(
            [configured.constant.get(n, 0.0) for n in pollutants.names], dtype=torch.float64
        ).reshape(pollutants.K),
        "decays.exponent": torch.tensor(
            [configured.exponent.get(n, 1.0) for n in pollutants.names], dtype=torch.float64
        ).reshape(pollutants.K),
    }
    keys = tuple(k for k in configured.keys if k in pollutants.names)
    if not keys:
        return None
    known = DecaySpec(
        keys, {k: configured.constant[k] for k in keys}, {k: configured.exponent[k] for k in keys}
    )
    return Decay(pollutants, known, site=arc.site)


def _make_decay(arc: Arc, vqip: Tensor) -> tuple[Tensor, Tensor]:
    """``DecayObj.make_decay`` (``core.py:474-492``) without the accumulation: the decayed VQIP
    and its ``diff``. With nothing to decay the water is unchanged, but the temperature is still
    read (``core.py:484-487``), so a missing one fails as WSIMOD's ``KeyError`` does."""
    if arc.decay is None:
        source = _temperature_source(arc)
        if "temperature" not in getattr(source, "params", {}):
            raise KeyError(f"{arc.site}: node {source.name!r} has no ('temperature', t) forcing")
        return vqip, torch.zeros_like(vqip)
    return arc.decay.make_decay(vqip)


def _validate_decay_arc(spec: ArcSpec, network: WsimodNetwork) -> None:
    """A user ``Decay*`` arc reads its ``in_port``'s ``("temperature", t)`` at every decay
    (``core.py:484-487``), whatever the pollutants: refused by name where a date lacks it."""
    port = next(n for n in network.nodes if n.name == spec.in_port)
    for date, label in zip(network.dates, network.date_labels, strict=True):
        if ("temperature", date) not in port.data:
            raise ValueError(
                f"arc {spec.name!r}: its in_port, node {port.name!r}, has no 'temperature' "
                f"forcing on {label} (a {spec.cls} decays at the in_port's temperature, "
                "core.py:464-470, :484-487)"
            )


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
            ctx.drop(vqip, active & tiny, arc=True)
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
        entered = self._on_entry(entered, run)
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

    def _on_entry(self, entered: Tensor, run: Tensor) -> Tensor:
        """The request as it enters the queue, after ``enter_arc``: unchanged here."""
        return entered

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


class DecayArcAlt(AltQueueArc):
    """``DecayArcAlt`` (``arcs.py:731-805``): an :class:`AltQueueArc` whose queued water decays
    on entry and at every end of day. It is a ``DecayQueueTank``'s internal arc
    (``tanks.py:663-669``, built with ``parent`` the owning node) and a user arc class.

    - **Entry** (``enter_queue``, ``:752-778``): ``enter_arc`` counts the request **before**
      decay in ``vqip_in`` and ``flow_in`` (``:766``), then ``make_decay`` (``:769``), then the
      decayed request goes into the register as in :class:`AltQueueArc`.
    - **End of day** (``_end_timestep``, ``:780-805``): the accumulators and ``total_decayed``
      are reset (``:787-791``); for ``i = 0 ... L - 1`` where key ``i + 1`` exists,
      ``queue[i] = make_decay(old[i + 1])`` and ``queue[i + 1]`` is emptied (``:798-803``); then
      ``queue[0] = sum_vqip(queue[0], make_decay(old[0]))`` (``:805``). The merge is
      therefore ``sum_vqip(decay(old[1]), decay(old[0]))``, the other order from
      :meth:`AltQueueArc.end_timestep`'s ``sum_vqip(old[0], old[1])`` (``:660``; X13). The blend
      is symmetric, so the order shows only where both slots hold no water: ``sum_vqip`` then
      keeps its first argument's non-additive values (``core.py:109``).

    Every ``make_decay`` adds its ``diff`` to ``total_decayed`` (``core.py:491``), so after an
    end of day it holds ``diff(slot 1) + ... + diff(slot L) + diff(slot 0)``, then each entry's
    in push order (X30). An absent key adds nothing (WSIMOD skips it; its row is empty, so its
    ``diff`` would be an exact zero). An internal arc's ``total_decayed`` is carried: a
    ``DecayQueueTank`` extracts it from its ``storage`` at the next end of day
    (``tanks.py:694``).

    A **user** ``DecayArcAlt`` (no ``parent``) reads its ``in_port``'s temperature
    (``core.py:464-470``) and builds its own :class:`~noodl.apps.wsimod.decay.Decay` from its
    ``decays`` (``ArcSpec.decays``; the drivers ``"wsimod.DecayArcAlt.decays.*"``, bound by
    :meth:`Arc.bind_decay`). Its ``total_decayed`` feeds only its own balance (``:750``), so it is
    not carried: ``WsimodDay`` resets it when a day starts, and its value after an end of day is
    WSIMOD's. Every ``diff`` is also the ``"out"`` term ``"DecayArcAlt.decayed"``, in the order
    above (entries, then the end of day); an internal arc records none, as its tank's
    ``"DecayQueueTank.decayed"`` holds the decay.

    ``decay`` is ``None`` when nothing decays (an empty ``decays``, or a quantity-only run): the
    water is unchanged, but ``make_decay`` still reads the temperature (``core.py:484-487``), so
    a missing one fails as WSIMOD's ``KeyError`` does. No method here takes a decision: the cap
    was evaluated at bind (the end-of-day rule, ``Engine.end_day``).
    """

    carried_paths: tuple[str, ...] = ("register", "present", "total_decayed")
    mass_balance: ClassVar[dict[str, str]] = {"DecayArcAlt.decayed": "out"}

    def __init__(
        self,
        spec: ArcSpec,
        in_port,
        out_port,
        pollutants: Pollutants,
        *,
        decay=None,
        parent=None,
        **kwargs,
    ) -> None:
        self.decay = decay
        self.parent = parent
        super().__init__(spec, in_port, out_port, pollutants, **kwargs)
        if parent is None and decay is None:  # a user arc decays as its own decays say
            self.decay = _arc_decay(self, spec, pollutants)
        self.total_decayed = empty(pollutants, ())  # core.py:472

    @property
    def user(self) -> bool:
        """A user arc (no ``parent``): its decay is its own mass-balance term."""
        return self.parent is None

    @classmethod
    def validate(cls, spec: ArcSpec, network: WsimodNetwork) -> None:
        """:meth:`AltQueueArc.validate`, and the ``in_port``'s temperature on every date."""
        super().validate(spec, network)
        _validate_decay_arc(spec, network)

    def stores(self) -> dict[str, Tensor]:
        """The register, its key set and ``total_decayed``."""
        return {**super().stores(), "total_decayed": self.total_decayed}

    def load_stores(self, stores: Mapping[str, Tensor]) -> None:
        stores = dict(stores)
        if "total_decayed" in stores:
            self.total_decayed = torch.as_tensor(stores.pop("total_decayed"), dtype=torch.float64)
        super().load_stores(stores)

    def make_decay(self, vqip: Tensor) -> tuple[Tensor, Tensor]:
        """``DecayObj.make_decay`` (``core.py:474-492``) without the accumulation: the decayed
        VQIP and its ``diff``."""
        return _make_decay(self, vqip)

    def _on_entry(self, entered: Tensor, run: Tensor) -> Tensor:
        """``arcs.py:769``: the request decays after ``enter_arc`` counted it."""
        decayed, diff = self.make_decay(entered)
        self.total_decayed = write(
            run, sum_vqip(self.p, self.total_decayed, diff), self.total_decayed
        )
        if self.user:
            self.ctx.term("DecayArcAlt.decayed", diff, run)
        return decayed

    def end_timestep(self) -> None:
        """``arcs.py:780-805`` (see the class text): reset, decay and shift, swapped merge."""
        Arc.end_timestep(self)  # :787-791 (vqip_in, vqip_out, flow_in, flow_out)
        p = self.p
        total = empty(p, ())  # :789
        old, here = self.register, self.present
        rows = list(old.unbind(-2))
        for i in range(self.length):  # :798-803; WSIMOD's max_travel covers only existing keys
            key = here[..., i + 1]
            if i == 0:  # arcs.py:567-568, :803: WSIMOD's key 1 always exists, as the merge below
                key = key | True  # assumes (a loaded row with slot 1 absent cannot be WSIMOD's)
            decayed, diff = self.make_decay(old[..., i + 1, :])
            rows[i] = torch.where(key[..., None], decayed, rows[i])
            rows[i + 1] = torch.where(key[..., None], torch.zeros_like(rows[i + 1]), rows[i + 1])
            total = write(key, sum_vqip(p, total, diff), total)
            if self.user:
                self.ctx.term("DecayArcAlt.decayed", diff, key)
        decayed, diff = self.make_decay(old[..., 0, :])  # :805; key 0 always exists
        rows[0] = sum_vqip(p, rows[0], decayed)
        self.total_decayed = sum_vqip(p, total, diff)
        if self.user:
            self.ctx.term("DecayArcAlt.decayed", diff, torch.ones(self.ctx.shape, dtype=torch.bool))
        self.register = torch.stack(rows, dim=-2)
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
            ctx.drop(row, present & below, arc=True)
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
        ctx.drop(vqip, active & tiny, arc=True)
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
        entered = self._on_entry(entered, run)
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
        pulled = self._on_entry(pulled, active)
        return self._update_queue("pull", pulled, j, "default", active=active, call=call)

    def _enter_arc(self, vqip: Tensor, active: Tensor) -> None:
        """``enter_arc`` (``arcs.py:423-428``): ``flow_in += V/(n+1)``, ``vqip_in += vqip``."""
        n = self.number_of_timesteps
        self.flow_in = write(active, self.flow_in + vqip[..., 0] / (n + 1), self.flow_in)
        self.vqip_in = write(active, sum_vqip(self.p, self.vqip_in, vqip), self.vqip_in)

    def _on_entry(self, entered: Tensor, run: Tensor) -> Tensor:
        """The request as it joins the list, after ``enter_arc``: unchanged here."""
        return entered


class DecayArc(QueueArc):
    """``DecayArc`` (``arcs.py:668-728``): a :class:`QueueArc` whose requests decay on entry and
    at every end of day, at the ``in_port``'s temperature (``core.py:464-470``; a WSIMOD TODO,
    ``:700-702``, notes it is not the water's own).

    - **Entry** (``enter_queue``, ``:686-708``), for pushes (``:396``) and pulls (``:354``):
      ``enter_arc`` counts the request undecayed in ``vqip_in`` and ``flow_in`` (``:698``), then
      ``make_decay`` (``:705``), then it joins the list (``:708``). A pulled request decays after
      ``in_port.pull_set`` removed it from the sender. A push's backflow is the decayed water,
      returned to the sender and taken off ``vqip_in`` (``:399-405``). With
      ``number_of_timesteps = 0`` a request decays once, on entry, and is delivered at once.
    - **End of the day** (``end_timestep``, ``:710-728``): every waiting request, in list order
      and both directions, decays in place (``:726-727``), then ``since`` advances as in
      :class:`QueueArc` (``time = max(time - 1, 0)``, ``:728``).

    Every ``diff`` is the ``"out"`` term ``"DecayArc.decayed"`` (``total_decayed``, ``:684``),
    added one at a time from empty: the day's entries in call order, then the end of the day in
    WSIMOD's list order (X24, X33). List order is entry order, because ``update_queue`` removes
    requests without reordering the rest (``:518-520``): by entry age ``i + since[d]``
    descending, then by ``entry_rank``, the position of the row's request call in the day's
    merged static call sequence of both directions (fixed with the call labels, D2 of 3a). A
    masked add is an exact ``+0.0``.

    **The order precondition (D21, X41).** ``since`` saturates at ``n``, so a due row that is
    still waiting (``i + since[d] >= n``, no call in its direction since it fell due) has lost
    its true age. The term's order is exact while no end of day finds due-and-waiting rows in
    both directions; where one does, the detached event ``"decay_arc_order_unknown"`` is
    recorded at the arc's name, and only the term's last bit may differ. The delivered values
    and the rows are exact in every case.

    ``decay`` is ``None`` when nothing decays (an empty ``decays``, or a quantity-only run): the
    water is unchanged, but the temperature is still read (``core.py:484-487``). No method here
    takes a decision: the cap was evaluated at bind (D5).
    """

    mass_balance: ClassVar[dict[str, str]] = {"DecayArc.decayed": "out"}

    def __init__(self, spec: ArcSpec, in_port: Node, out_port: Node, pollutants: Pollutants):
        super().__init__(spec, in_port, out_port, pollutants)
        self.decay = _arc_decay(self, spec, pollutants)
        # (direction, j) -> position in the day's merged static call sequence (D2)
        self.entry_rank: dict[tuple[str, int], int] = {}
        self._merged = 0  # today's request calls so far, both directions

    @classmethod
    def validate(cls, spec: ArcSpec, network: WsimodNetwork) -> None:
        """:meth:`QueueArc.validate`, and the ``in_port``'s temperature on every date."""
        super().validate(spec, network)
        _validate_decay_arc(spec, network)

    def begin_tracing(self) -> None:
        super().begin_tracing()
        self.entry_rank = {}
        self._merged = 0

    def _enter(self, direction: str, tag: Tag) -> int:
        """:meth:`QueueArc._enter`; while the labels are recorded, the call's place in the
        merged sequence of both directions is recorded too. Once they are fixed, every later
        day must interleave the two directions as the traced day did, or the call raises (D2:
        the term's order of addition relies on ``entry_rank``)."""
        recording = not self._frozen
        j = super()._enter(direction, tag)
        position, self._merged = self._merged, self._merged + 1
        if recording:
            self.entry_rank[(direction, j)] = len(self.entry_rank)
        elif (traced := self.entry_rank.get((direction, j))) != position:
            raise RuntimeError(
                f"{self.site}: {direction} request call {j} is call {position} of the day in "
                f"both directions; the traced day made it call {traced}"
                " (D2: the call structure must not change)"
            )
        return j

    def make_decay(self, vqip: Tensor) -> tuple[Tensor, Tensor]:
        """``DecayObj.make_decay`` (``core.py:474-492``) without the accumulation."""
        return _make_decay(self, vqip)

    def _on_entry(self, entered: Tensor, run: Tensor) -> Tensor:
        """``arcs.py:705``: the request decays after ``enter_arc`` counted it."""
        decayed, diff = self.make_decay(entered)
        self.ctx.term("DecayArc.decayed", diff, run)
        return decayed

    def end_timestep(self) -> None:
        """``arcs.py:710-728``: decay every present row in place, add the ``diff``s to the term
        in WSIMOD's list order, record the event of D21, then :meth:`QueueArc.end_timestep`
        (the reset and ``since``, which is read here before it advances)."""
        n, rows = self.number_of_timesteps, self.rows
        width = rows.shape[-2]
        if n and width:
            ctx, since = self.ctx, self.since
            present = (rows[..., 0] != 0).detach()  # (..., 2, n, J): a zero row is absent
            # a present row outside the traced calls (a hand-built or loaded state) would be
            # decayed but left out of the term: refused before anything changes
            for d, direction in enumerate(_DIRECTIONS):
                for j in range(width):
                    if (direction, j) not in self.entry_rank and bool(present[..., d, :, j].any()):
                        raise RuntimeError(
                            f"{self.site}: a {direction} row at call {j} holds water, but no "
                            "traced request call has that index (D2); its decay would be left "
                            "out of the term"
                        )
            decayed, diffs = {}, {}
            for d in range(2):
                for i in range(n):
                    for j in range(width):
                        decayed[d, i, j], diffs[d, i, j] = self.make_decay(rows[..., d, i, j, :])
            ranked = sorted(self.entry_rank, key=self.entry_rank.__getitem__)
            for age in range(2 * n - 1, -1, -1):  # oldest first; age = i + since in [0, 2n-1]
                for direction, j in ranked:
                    d = _DIRECTIONS.index(direction)
                    for i in range(max(0, age - n), min(n - 1, age) + 1):
                        here = present[..., d, i, j] & (since[..., d] + i == age)
                        ctx.term("DecayArc.decayed", diffs[d, i, j], here)
            due = (since[..., :, None] + torch.arange(n, dtype=since.dtype) >= n).detach()
            waiting = (present & due[..., None]).flatten(-2).any(-1)  # (..., 2)
            ctx.event("decay_arc_order_unknown", self.name, waiting[..., 0] & waiting[..., 1])
            shape = torch.broadcast_shapes(
                rows.shape, *(x.shape[:-1] + (1, 1, 1, x.shape[-1]) for x in decayed.values())
            )
            new = torch.stack(
                [
                    torch.stack(
                        [
                            torch.stack(
                                [
                                    decayed[d, i, j].expand(*shape[:-4], shape[-1])
                                    for j in range(width)
                                ],
                                dim=-2,
                            )
                            for i in range(n)
                        ],
                        dim=-3,
                    )
                    for d in range(2)
                ],
                dim=-4,
            )
            self.rows = torch.where(present[..., None], new, rows)
        self._merged = 0
        super().end_timestep()


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
    "DecayArc": DecayArc,
    "DecayArcAlt": DecayArcAlt,
    "PullArc": PullArc,
    "QueueArc": QueueArc,
    "PushArc": PushArc,
    "SewerArc": SewerArc,
    "WeirArc": WeirArc,
}

# the user arc classes that decay, read their in_port's temperature and bind their decays after
# the nodes (D15)
DECAY_ARCS: tuple[type[Arc], ...] = (DecayArc, DecayArcAlt)
# the events recorded at an arc's name rather than a node's (D21): DayRecord.events gives them
# one column per arc
ARC_EVENTS: tuple[str, ...] = ("decay_arc_order_unknown",)
