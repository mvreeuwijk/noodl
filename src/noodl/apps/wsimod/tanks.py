"""WSIMOD 0.8.1's tanks (``wsimod/nodes/tanks.py``), batched.

A tank is not a node: a node owns it under an attribute (``tank``, ``sewer_tank``, ...) and
gives it the day's :class:`~noodl.apps.wsimod.engine.Ctx`. Its decision sites are the owner's,
``"node:<owner>/<path>/<WSIMOD method>/<branch>"`` (engine.py, Decision records), and its call
counts are kept under the owner's element ``"node:<owner>"`` with method ``"<path>/<method>"``.

Every method takes the caller's ``active`` mask and writes its state only where it holds (D3).
The arithmetic is WSIMOD's ``+ - * /`` and ``min``/``max`` in WSIMOD's order, so every value is
bitwise WSIMOD's.

``storage`` is a VQIP ``(..., 1 + K)``; a :class:`QueueTank` adds ``active_storage`` and the
shift register of its internal :class:`~noodl.apps.wsimod.arcs.AltQueueArc` (a
:class:`DecayQueueTank`'s is a :class:`~noodl.apps.wsimod.arcs.DecayArcAlt`, which adds
``total_decayed``). :meth:`Tank.stores` names them by WSIMOD attribute path, relative to the
tank (``storage``, ``active_storage``, ``internal_arc.register``, ``internal_arc.present``).
``storage_`` and the other lagged copies serve only WSIMOD's mass balance and are not ported
(spec, State).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import ClassVar

import torch
from torch import Tensor

from noodl.apps.wsimod.arcs import AltQueueArc, DecayArcAlt, check_travel_time
from noodl.apps.wsimod.config import ArcSpec
from noodl.apps.wsimod.constants import UNBOUNDED_CAPACITY
from noodl.apps.wsimod.engine import Ctx, dmax, dmin
from noodl.apps.wsimod.vqip import (
    Pollutants,
    empty,
    extract_vqip,
    sum_vqip,
    v_change_vqip,
    v_distill_vqip,
    write,
)

__all__ = ["DecayQueueTank", "DecayTank", "QueueTank", "ResidenceTank", "Tank", "bind_timearea"]


def _f64(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def bind_timearea(where: str, name: str, keys: tuple[int, ...], value) -> dict[int, Tensor]:
    """The day's time-area fractions on the configured keys (W56, W70; spec, Inputs).

    ``value`` holds one fraction per travel time, ``(..., L+1)`` (the driver's row of a node).
    The keys are structure (they size the ``QueueTank``'s register, W50), so the result keeps
    the configured keys in their configured order, which is the order WSIMOD walks the dict in
    (``storage.py:337``, ``sewer.py:194``); a non-zero fraction at any other time is refused
    by name. The fractions are batchable and learnable.
    """
    value = _f64(value)
    if value.ndim < 1 or value.shape[-1] <= max(keys):
        raise ValueError(
            f"{where}: {name} of shape {tuple(value.shape)}; expected (..., >= {max(keys) + 1}) "
            "fractions, one per travel time"
        )
    other = [t for t in range(value.shape[-1]) if t not in keys]
    if other and bool((value[..., other] != 0).any()):
        bad = [t for t in other if bool((value[..., t] != 0).any())]
        raise ValueError(
            f"{where}: {name} has a non-zero fraction at time(s) {bad}, which the configuration "
            f"does not list ({sorted(keys)}); the keys are structure, the fractions are not"
        )
    return {k: value[..., k] for k in keys}


class Tank:
    """``Tank`` (``tanks.py:10-356``): a VQIP store of volumetric ``capacity``.

    ``initial_storage`` is a volume (0-d) or a VQIP: a number becomes
    ``v_change_vqip(empty, x)``, a volume with no pollutants (``:36-53``). ``owner`` is the
    owning node's name and ``path`` the tank's attribute on it; both name the decision sites.
    ``area`` and ``datum`` are kept for ``get_head``, which 3a does not use.
    """

    def __init__(
        self,
        pollutants: Pollutants,
        *,
        capacity: Tensor,
        area: Tensor,
        datum: Tensor,
        initial_storage: Tensor,
        owner: str,
        path: str = "tank",
    ) -> None:
        self.p = pollutants
        self.capacity = _f64(capacity)
        self.area = _f64(area)
        self.datum = _f64(datum)
        self.owner = owner
        self.path = path
        self.site = f"node:{owner}/{path}"
        initial = _f64(initial_storage)
        if initial.ndim == 0:
            initial = v_change_vqip(pollutants, empty(pollutants, ()), initial)
        self.storage = initial.clone()
        self._ctx: Ctx | None = None

    # --- the day's context ------------------------------------------------------------------
    @property
    def ctx(self) -> Ctx:
        if self._ctx is None:
            raise RuntimeError(f"{self.site}: no day context; the owner sets tank.ctx")
        return self._ctx

    @ctx.setter
    def ctx(self, ctx: Ctx) -> None:
        self._ctx = ctx

    def _call(self, method: str) -> int:
        return self.ctx.call(f"node:{self.owner}", f"{self.path}/{method}")

    def _volume_vqip(self, volume: Tensor) -> Tensor:
        """``{"volume": volume}``."""
        volume = _f64(volume)
        rest = torch.zeros(volume.shape + (self.p.K,), dtype=torch.float64)
        return torch.cat([volume[..., None], rest], dim=-1)

    # --- state ------------------------------------------------------------------------------
    def stores(self) -> dict[str, Tensor]:
        """The tank's stores by attribute path relative to the tank."""
        return {"storage": self.storage}

    def load_stores(self, stores: Mapping[str, Tensor]) -> None:
        unknown = set(stores) - set(self.stores())
        if unknown:
            raise KeyError(f"{self.site} has no store {sorted(unknown)[0]!r}")
        for path, value in stores.items():
            *parents, last = path.split(".")
            owner = self
            for part in parents:
                owner = getattr(owner, part)
            setattr(owner, last, value)

    # --- WSIMOD's methods -------------------------------------------------------------------
    def pull_ponded(self, *, active: Tensor) -> Tensor:
        """``tanks.py:78-100``: pull ``max(V - capacity, 0)`` through ``self.pull_storage``
        (a :class:`QueueTank`'s own, so limited to ``active_storage``)."""
        call = self._call("pull_ponded")
        ponded = dmax(
            self.ctx,
            f"{self.site}/pull_ponded/excess",
            self.storage[..., 0] - self.capacity,
            0.0,
            active=active,
            call=call,
        )
        return self.pull_storage(self._volume_vqip(ponded), active=active)

    def get_avail(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``tanks.py:102-130``: the storage, at ``min(V, request)`` when a request is given."""
        reply = self.storage.clone()  # copy_vqip (tanks.py:123)
        if vqip is None:
            return reply
        volume = dmin(
            self.ctx,
            f"{self.site}/get_avail/clip",
            reply[..., 0],
            vqip[..., 0],
            active=active,
            call=self._call("get_avail"),
        )
        return v_change_vqip(self.p, reply, volume)

    def get_excess(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``tanks.py:132-158``: ``max(capacity - V, 0)``, clipped to the request when one is
        given, at the storage's concentration."""
        ctx = self.ctx
        call = self._call("get_excess")
        vol = dmax(
            ctx,
            f"{self.site}/get_excess/headroom",
            self.capacity - self.storage[..., 0],
            0.0,
            active=active,
            call=call,
        )
        if vqip is not None:
            vol = dmin(
                ctx, f"{self.site}/get_excess/clip", vqip[..., 0], vol, active=active, call=call
            )
        return v_change_vqip(self.p, self.storage, vol)

    def push_storage(self, vqip: Tensor, *, active: Tensor, force: bool = False) -> Tensor:
        """``tanks.py:160-205``: returns what was rejected. ``force`` adds everything; otherwise
        ``max(V_in - excess, 0)`` is rejected and the rest enters."""
        p = self.p
        active = torch.as_tensor(active, dtype=torch.bool)
        if force:
            self.storage = write(active, sum_vqip(p, self.storage, vqip), self.storage)
            return self.ctx.empty()
        call = self._call("push_storage")
        excess = self.get_excess(active=active)[..., 0]
        reply = dmax(
            self.ctx,
            f"{self.site}/push_storage/clip",
            vqip[..., 0] - excess,
            0.0,
            active=active,
            call=call,
        )
        reply = v_change_vqip(p, vqip, reply)
        entered = v_change_vqip(p, vqip, vqip[..., 0] - reply[..., 0])
        self.storage = write(active, sum_vqip(p, self.storage, entered), self.storage)
        return reply

    def pull_storage(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``tanks.py:207-239``: empty when ``V == 0`` (exactly); otherwise
        ``v_change(storage, min(request, V))``, taken out of the storage."""
        p, ctx = self.p, self.ctx
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("pull_storage")
        dry = ctx.decide(
            f"{self.site}/pull_storage/empty",
            self.storage[..., 0],
            "==",
            0.0,
            active=active,
            call=call,
        )
        run = active & ~dry
        volume = dmin(
            ctx,
            f"{self.site}/pull_storage/clip",
            vqip[..., 0],
            self.storage[..., 0],
            active=run,
            call=call,
        )
        reply = v_change_vqip(p, self.storage, volume)
        self.storage = write(run, extract_vqip(p, self.storage, reply), self.storage)
        return torch.where(dry[..., None], ctx.empty(), reply)

    def evaporate(self, evap: Tensor, *, active: Tensor) -> Tensor:
        """``tanks.py:303-317``: ``min(evap, avail)`` leaves by ``v_distill_vqip`` (the
        pollutants stay); returns that volume. ``avail`` is ``self.get_avail()``, a
        :class:`QueueTank`'s ``active_storage``."""
        active = torch.as_tensor(active, dtype=torch.bool)
        avail = self.get_avail(active=active)[..., 0]
        evap = dmin(
            self.ctx,
            f"{self.site}/evaporate/clip",
            evap,
            avail,
            active=active,
            call=self._call("evaporate"),
        )
        self.storage = write(active, v_distill_vqip(self.storage, evap), self.storage)
        return evap

    def end_timestep(self) -> None:
        """``tanks.py:348-351``: only the lagged ``storage_``, which is not ported."""


class ResidenceTank(Tank):
    """``ResidenceTank`` (``tanks.py:359-400``): ``pull_outflow`` takes ``V / residence_time``."""

    def __init__(self, pollutants: Pollutants, *, residence_time: Tensor, **kwargs) -> None:
        self.residence_time = _f64(residence_time)
        super().__init__(pollutants, **kwargs)

    def pull_outflow(self, *, active: Tensor) -> Tensor:
        """``tanks.py:387-400``."""
        outflow = self.storage[..., 0] / self.residence_time
        outflow = v_change_vqip(self.p, self.storage, outflow)
        return self.pull_storage(outflow, active=active)


class DecayTank(Tank):
    """``DecayTank`` (``tanks.py:403-457``): a tank that decays its storage at the end of the day.

    ``end_timestep`` (``end_timestep_decay``, ``:442-447``) resets ``total_decayed``, decays
    ``storage`` with the day's factor and adds the decayed mass to ``total_decayed``
    (``core.py:474-492``). The factor and its cap were evaluated when the owner bound the
    parameters (D5: ``decay.bind``), so no decision is taken here (the end-of-day rule,
    ``Engine.end_day``). The decayed mass is the mass-balance term ``"DecayTank.decayed"``, an
    ``"out"`` term that the owning class declares. ``total_decayed`` is a per-day store (D8).

    With ``decay=None`` (empty ``decays``) the storage is unchanged, but the end of the day
    still reads the parent's ``("temperature", t)`` (``core.py:484-487``), so a missing
    temperature fails as WSIMOD's ``KeyError`` does. ``parent`` is the node whose bound
    parameters hold ``"temperature"``.
    """

    def __init__(self, pollutants: Pollutants, *, parent, decay=None, **kwargs) -> None:
        self.parent = parent
        self.decay = decay
        super().__init__(pollutants, **kwargs)
        self.total_decayed = empty(pollutants, ())

    def stores(self) -> dict[str, Tensor]:
        """``storage`` and the per-day ``total_decayed`` (D8): reset at every ``end_timestep``,
        so the owner does not carry it, but the replay compares it."""
        return {**super().stores(), "total_decayed": self.total_decayed}

    def end_timestep(self) -> None:
        if self.decay is None:
            # tanks.py:444: reset even over no pollutants. Deliberately unbatched (the init
            # shape): nothing decays here, so nothing reads it at a batch shape.
            self.total_decayed = empty(self.p, ())
            if "temperature" not in self.parent.params:
                raise KeyError(f"{self.site}: the parent has no ('temperature', t) forcing")
            return  # generic_temperature_decay over no pollutants: storage unchanged
        ctx = self.ctx
        self.total_decayed = ctx.empty()  # tanks.py:444
        self.storage, diff = self.decay.make_decay(self.storage)  # :447
        self.total_decayed = sum_vqip(self.p, self.total_decayed, diff)  # core.py:491
        ctx.term("DecayTank.decayed", diff, torch.ones(ctx.shape, dtype=torch.bool))


class QueueTank(Tank):
    """``QueueTank`` (``tanks.py:460-642``): water waits in an internal ``AltQueueArc`` before
    it joins ``active_storage``, the only water that can be pulled.

    ``push_times`` are the ``time`` arguments the owner's structure can pass to
    :meth:`push_storage` (a ``Sewer``'s ``pipe_time`` or time-area keys, a
    ``QueueGroundwater``'s time-area keys); with ``number_of_timesteps`` they fix the register
    length ``L = max(1, number_of_timesteps + max(push_times))``.

    Every push goes through the arc's ``send_push_request``, which ends with ``update_queue``,
    so slot 0 is flushed into ``active_storage`` on every push (spec, Flushing a ``QueueTank`` on
    push): water entered with total time 0 is active at once.
    """

    wsimod_type: ClassVar[str] = "QueueTank"

    def __init__(
        self,
        pollutants: Pollutants,
        *,
        number_of_timesteps: int = 0,
        push_times: Iterable[int] = (0,),
        **kwargs,
    ) -> None:
        super().__init__(pollutants, **kwargs)
        where = self.site
        self.number_of_timesteps = check_travel_time(
            where, "number_of_timesteps", number_of_timesteps
        )
        times = [check_travel_time(where, "push time", t) for t in push_times] or [0]
        self.name = self.owner
        # tanks.py:478: the initial storage is active
        self.active_storage = self.storage.clone()
        # tanks.py:481-486: the arc files itself in the tank's own arc dicts
        self.in_arcs: dict[str, AltQueueArc] = {}
        self.out_arcs: dict[str, AltQueueArc] = {}
        spec = ArcSpec(
            "",
            "AltQueueArc",
            self.owner,
            self.owner,
            capacity=UNBOUNDED_CAPACITY,
            number_of_timesteps=self.number_of_timesteps,
        )
        self.internal_arc = AltQueueArc(
            spec,
            self,
            self,
            pollutants,
            length=self.number_of_timesteps + max(times),
            site=f"{self.site}/internal_arc",
            owner=self.owner,
        )

    def connect(self, arc: AltQueueArc, end: str) -> None:
        """``arcs.py:70-71`` on a tank: ``in_arcs``/``out_arcs`` only (no typed dicts)."""
        (self.in_arcs if end == "in" else self.out_arcs)[arc.name] = arc

    @Tank.ctx.setter
    def ctx(self, ctx: Ctx) -> None:
        self._ctx = ctx
        self.internal_arc.ctx = ctx

    def stores(self) -> dict[str, Tensor]:
        arc = self.internal_arc.stores()
        return {
            "storage": self.storage,
            "active_storage": self.active_storage,
            **{f"internal_arc.{k}": v for k, v in arc.items()},
        }

    def get_avail(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``tanks.py:505-511``: ``active_storage``. WSIMOD's method takes no request."""
        if vqip is not None:
            raise TypeError(f"{self.site}: QueueTank.get_avail takes no request (tanks.py:505)")
        return self.active_storage.clone()  # copy_vqip (tanks.py:511)

    def push_storage(
        self, vqip: Tensor, *, active: Tensor, time: int = 0, force: bool = False
    ) -> Tensor:
        """``tanks.py:513-539``: ``force`` adds to ``storage`` and ``active_storage`` and skips
        the queue; otherwise the internal arc's ``send_push_request`` with ``time``, and
        ``storage`` gains ``v_change(vqip, V - reply V)``. A push below ``FLOAT_ACCURACY``
        returns empty from the arc, so ``storage`` keeps it while the queue never sees it
        (spec, Nodes, QueueTank). The arc does not record it as dropped (W49): the tank's storage
        holds it and the mass balance counts it there."""
        p = self.p
        active = torch.as_tensor(active, dtype=torch.bool)
        if force:
            self.storage = write(active, sum_vqip(p, self.storage, vqip), self.storage)
            self.active_storage = write(
                active, sum_vqip(p, self.active_storage, vqip), self.active_storage
            )
            return self.ctx.empty()
        reply = self.internal_arc.send_push_request(
            vqip, active=active, time=time, record_drop=False
        )
        entered = v_change_vqip(p, vqip, vqip[..., 0] - reply[..., 0])
        self.storage = write(active, sum_vqip(p, self.storage, entered), self.storage)
        return reply

    def pull_storage(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``tanks.py:541-564``: ``min(request, active V)`` at ``active_storage``'s
        concentration, out of ``active_storage`` and ``storage``. No ``V == 0`` test."""
        p = self.p
        active = torch.as_tensor(active, dtype=torch.bool)
        volume = dmin(
            self.ctx,
            f"{self.site}/pull_storage/clip",
            vqip[..., 0],
            self.active_storage[..., 0],
            active=active,
            call=self._call("pull_storage"),
        )
        reply = v_change_vqip(p, self.active_storage, volume)
        self.active_storage = write(
            active, extract_vqip(p, self.active_storage, reply), self.active_storage
        )
        self.storage = write(active, extract_vqip(p, self.storage, reply), self.storage)
        return reply

    def pull_storage_exact(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``tanks.py:566-586``: per column ``min(request, active_storage)`` over the volume and
        the additive pollutants (the request's non-additive entries pass through), out of
        ``active_storage`` and ``storage``."""
        p, ctx = self.p, self.ctx
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("pull_storage_exact")
        vqip = torch.broadcast_to(vqip, torch.broadcast_shapes(vqip.shape, self.storage.shape))
        columns = [vqip[..., i] for i in range(1 + p.K)]
        for name in ("volume", *p.additive):
            i = 0 if name == "volume" else p.column(name)
            columns[i] = dmin(
                ctx,
                f"{self.site}/pull_storage_exact/clip:{name}",
                columns[i],
                self.active_storage[..., i],
                active=active,
                call=call,
            )
        shape = torch.broadcast_shapes(*(c.shape for c in columns))
        reply = torch.stack([c.expand(shape) for c in columns], dim=-1)
        self.active_storage = write(
            active, extract_vqip(p, self.active_storage, reply), self.active_storage
        )
        self.storage = write(active, extract_vqip(p, self.storage, reply), self.storage)
        return reply

    def push_check(
        self, vqip: Tensor | None = None, tag: str = "default", *, active: Tensor
    ) -> Tensor:
        """``tanks.py:588-608``: ``get_excess()``; with a request only its volume is set to
        ``min(request, excess)`` (the pollutants stay the full excess's)."""
        excess = self.get_excess(active=active)
        if vqip is None:
            return excess
        volume = dmin(
            self.ctx,
            f"{self.site}/push_check/clip",
            vqip[..., 0],
            excess[..., 0],
            active=active,
            call=self._call("push_check"),
        )
        shape = torch.broadcast_shapes(volume.shape, excess.shape[:-1])
        return torch.cat(
            [volume.expand(shape)[..., None], excess[..., 1:].expand(shape + (self.p.K,))], -1
        )

    def push_set(self, vqip: Tensor, tag: str = "default", *, active: Tensor) -> Tensor:
        """``tanks.py:610-629``: the water at the end of the queue joins ``active_storage``;
        always accepted."""
        self.active_storage = write(
            active, sum_vqip(self.p, self.active_storage, vqip), self.active_storage
        )
        return self.ctx.empty()

    def end_timestep(self) -> None:
        """``tanks.py:631-635``: shift the internal arc, then flush its slot 0 into
        ``active_storage`` (``update_queue``), for every instance."""
        self.internal_arc.end_timestep()
        self.internal_arc.update_queue(active=torch.ones(self.ctx.shape, dtype=torch.bool))


class DecayQueueTank(QueueTank):
    """``DecayQueueTank`` (``tanks.py:645-696``): a :class:`QueueTank` whose internal arc is a
    :class:`~noodl.apps.wsimod.arcs.DecayArcAlt` with ``parent`` the owning node
    (``:663-669``), so the queued water decays on entry and at every end of day.

    ``number_of_timesteps`` defaults to **1** (``:648``), against the class's docstring
    ("Defaults to 0", ``:656-658``) and ``QueueTank``'s 0 (``:463``); a decaying
    ``QueueGroundwater`` passes none (``storage.py:308-315``), so every entry waits a day longer
    than in a non-decaying one (X12). Pushes and pulls are :class:`QueueTank`'s: ``storage``
    gains the **undecayed** accepted part (``:536-538``) and the pulls read ``active_storage``.

    :meth:`end_timestep` (``_end_timestep``, ``:689-696``) is transcribed as written (X13):

    - ``storage`` loses the internal arc's ``total_decayed`` **before** the arc's own end of
      day resets it (``:694``): that day's entry decay plus the previous end of day's decay, so
      the decay reaches ``storage`` a day late. The loss is the ``"out"`` term
      ``"DecayQueueTank.decayed"``, which the owner declares;
    - ``active_storage`` never decays (the TODO at ``:692-693``);
    - there is no ``update_queue``: water that reaches slot 0 at the end of the day waits there,
      and decays again, until the next push or the owner's ``update_queue``.
    """

    wsimod_type: ClassVar[str] = "DecayQueueTank"

    def __init__(
        self, pollutants: Pollutants, *, parent, decay=None, number_of_timesteps: int = 1, **kwargs
    ) -> None:
        super().__init__(pollutants, number_of_timesteps=number_of_timesteps, **kwargs)
        self.parent = parent
        queue = self.internal_arc
        # tanks.py:663-669: the internal arc is replaced (it files itself in the same dicts)
        self.internal_arc = DecayArcAlt(
            queue.spec,
            self,
            self,
            pollutants,
            decay=decay,
            parent=parent,
            length=queue.length,
            site=queue.site,
            owner=self.owner,
        )

    @property
    def decay(self):
        """The internal arc's :class:`~noodl.apps.wsimod.decay.Decay` (the owner binds it)."""
        return self.internal_arc.decay

    def end_timestep(self) -> None:
        """``tanks.py:689-696``: extract the lagged ``total_decayed``, then the arc's end of day;
        no ``update_queue`` (see the class text)."""
        ctx, arc = self.ctx, self.internal_arc
        decayed = arc.total_decayed
        self.storage = extract_vqip(self.p, self.storage, decayed)  # :694
        ctx.term("DecayQueueTank.decayed", decayed, torch.ones(ctx.shape, dtype=torch.bool))
        arc.end_timestep()  # :696
