"""WSIMOD 0.8.1's ``Sewer`` (``wsimod/nodes/sewer.py:14-247``), batched.

A sewer is a node around one :class:`~noodl.apps.wsimod.tanks.QueueTank` under the attribute
``sewer_tank`` (``number_of_timesteps = 0``, ``sewer.py:108-113``). Water pushed with the tags
``"default"`` and ``"Sewer"`` waits ``pipe_time`` days; water pushed with ``"Land"`` and
``"Demand"`` is spread over the time-area diagram ``pipe_timearea`` (``:96-99``, ``:163-199``).
Every push flushes the queue's due slot at once, so water entered with total time 0 is active
the same day and leaves with that day's ``make_discharge`` (spec, Flushing a ``QueueTank`` on
push). ``make_discharge`` (``:201-239``) is the one orchestration method.

The arithmetic is WSIMOD's ``+ - * /`` and ``min``/``max`` in WSIMOD's order, so every value is
bitwise WSIMOD's (``replay_bound = 0``). Every method takes the caller's ``active`` mask and
writes state only where it holds (D3); data branches are :meth:`Ctx.decide` (D12).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor

from noodl.apps.wsimod.arcs import check_travel_time
from noodl.apps.wsimod.engine import Ctx, dmin
from noodl.apps.wsimod.node import NODE_CLASSES, Node, Tag
from noodl.apps.wsimod.tanks import QueueTank
from noodl.apps.wsimod.vqip import Pollutants, extract_vqip, sum_vqip, v_change_vqip

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.config import NodeSpec, WsimodNetwork

__all__ = ["Sewer", "sewer_derived"]

# sewer.py:236: where ponded water goes, and its tag
PONDED_TYPES = ("Land",)
PONDED_TAG = "Sewer"


def _f64(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def sewer_derived(spec: NodeSpec) -> dict[str, float]:
    """The ``sewer_tank``'s fields from ``spec``'s merged parameters (D5): ``capacity``,
    ``area = chamber_area`` and ``datum = chamber_floor`` (``sewer.py:108-113``; overrides set
    the same three, ``:127-132``)."""
    values = spec.parameters
    return {
        "capacity": float(values.get("capacity", 0)),
        "area": float(values.get("chamber_area", 1)),
        "datum": float(values.get("chamber_floor", 10)),
    }


class Sewer(Node):
    """``Sewer`` (``sewer.py:14-247``): a :class:`QueueTank` with a pipe time and a time-area
    diagram.

    Parameters (:meth:`bind`): ``capacity``, ``chamber_area`` and ``chamber_floor`` reach the tank.
    ``pipe_time`` and the keys of ``pipe_timearea`` are structure: they fix the tank's register
    length (W50), so they are read once from the configuration. The fractions of
    ``pipe_timearea`` are the configured ones.
    """

    wsimod_type: ClassVar[str] = "Sewer"
    store_paths: ClassVar[tuple[str, ...]] = (
        "sewer_tank.storage",
        "sewer_tank.active_storage",
        "sewer_tank.internal_arc.register",
        "sewer_tank.internal_arc.present",
    )
    step_methods: ClassVar[tuple[str, ...]] = ("make_discharge",)
    # sewer.py:116: the storage change only, from the stores (D11)
    mass_balance: ClassVar[dict[str, str]] = {}
    # make_discharge's push_distributed, and Node's default pull (pull_distributed)
    distributes: ClassVar[tuple[str, ...]] = ("out", "in")

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        super().__init__(spec, pollutants)
        self.site = f"node:{self.name}"
        self._ctx: Ctx | None = None
        values = spec.parameters
        where = f"node {self.name!r}"
        self.pipe_time = check_travel_time(where, "pipe_time", values.get("pipe_time", 0))
        timearea = values.get("pipe_timearea", {0: 1})
        self.pipe_timearea: dict[int, Tensor] = {
            check_travel_time(where, "pipe_timearea key", k): _f64(v) for k, v in timearea.items()
        }
        derived = sewer_derived(spec)
        # sewer.py:108-113
        self.sewer_tank = QueueTank(
            pollutants,
            number_of_timesteps=0,
            push_times=(self.pipe_time, *self.pipe_timearea),
            capacity=_f64(derived["capacity"]),
            area=_f64(derived["area"]),
            datum=_f64(derived["datum"]),
            initial_storage=_f64(0.0),
            owner=self.name,
            path="sewer_tank",
        )
        # sewer.py:96-104
        self.push_set_handler["Sewer"] = self.push_set_sewer
        self.push_set_handler["default"] = self.push_set_sewer
        self.push_set_handler["Land"] = self.push_set_land
        self.push_set_handler["Demand"] = self.push_set_land
        for tag in ("default", "Sewer", "Demand", "Land"):
            self.push_check_handler[tag] = self.push_check_sewer

    # --- the day's context and parameters ---------------------------------------------------
    @property
    def ctx(self) -> Ctx:
        if self._ctx is None:
            raise RuntimeError(f"node {self.name!r}: no day context")
        return self._ctx

    @ctx.setter
    def ctx(self, ctx: Ctx) -> None:
        self._ctx = ctx
        self.sewer_tank.ctx = ctx

    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """This day's parameters: ``capacity``, ``chamber_area`` and ``chamber_floor`` become the
        tank's ``capacity``, ``area`` and ``datum`` (``sewer.py:127-132``)."""
        super().bind(parameters)
        for name, field in (
            ("capacity", "capacity"),
            ("chamber_area", "area"),
            ("chamber_floor", "datum"),
        ):
            if name in self.params:
                setattr(self.sewer_tank, field, self.params[name])

    # --- state ------------------------------------------------------------------------------
    def stores(self) -> dict[str, Tensor]:
        return {f"sewer_tank.{path}": value for path, value in self.sewer_tank.stores().items()}

    def end_timestep(self) -> None:
        """``sewer.py:241-243``: the tank's shift and flush."""
        self.sewer_tank.end_timestep()

    # --- the handlers -----------------------------------------------------------------------
    def push_check_sewer(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``sewer.py:145-161``: the tank's excess, its volume cut to ``min(excess, request)``
        when a request is given."""
        excess = self.sewer_tank.get_excess(active=active)
        if vqip is None:
            return excess
        volume = dmin(
            self.ctx,
            f"{self.site}/push_check_sewer/clip",
            excess[..., 0],
            vqip[..., 0],
            active=active,
            call=self.ctx.call(self.site, "push_check_sewer"),
        )
        return v_change_vqip(self.p, excess, volume)

    def push_set_sewer(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``sewer.py:163-176``: into the tank after ``pipe_time`` days; returns the rejection."""
        return self.sewer_tank.push_storage(vqip, active=active, time=self.pipe_time)

    def push_set_land(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``sewer.py:178-199``: each time-area share ``V x fraction`` into the tank under its
        time, in the diagram's order; the rejections are summed."""
        p = self.p
        reply = self.ctx.empty()
        for time, normalised in self.pipe_timearea.items():
            share = v_change_vqip(p, vqip, vqip[..., 0] * normalised)
            reply_ = self.sewer_tank.push_storage(share, active=active, time=time)
            reply = sum_vqip(p, reply, reply_)
        return reply

    # --- orchestration ----------------------------------------------------------------------
    def make_discharge(self, *, active: Tensor) -> None:
        """``sewer.py:201-239``:

        1. ``update_queue`` of the tank's arc;
        2. ``push_distributed(active_storage)`` over every out-arc (no ``of_type``);
        3. ``pull_storage_exact`` of what was sent (``"miscalculated_tank_storage"`` when the
           reply exceeds it by more than ``FLOAT_ACCURACY``);
        4. ``pull_ponded``; ponded water above ``FLOAT_ACCURACY`` goes to ``Land`` with the tag
           ``"Sewer"``, and what ``Land`` rejects is forced back at time 0. Ponded water at or
           below it has left the tank and vanishes, recorded as dropped water.

        Step 4's forced push returns empty, so WSIMOD's ``"ponded water cant reenter"``
        (``:238-239``) never prints; the event is recorded where it would.
        """
        ctx, p, tank = self.ctx, self.p, self.sewer_tank
        active = torch.as_tensor(active, dtype=torch.bool)
        call = ctx.call(self.site, "make_discharge")
        tank.internal_arc.update_queue(active=active)
        remaining = self.push_distributed(tank.active_storage, active=active)
        sent = extract_vqip(p, tank.active_storage, remaining)
        reply = tank.pull_storage_exact(sent, active=active)
        ctx.event(
            "miscalculated_tank_storage",
            self.name,
            active & ((reply[..., 0] - sent[..., 0]) > ctx.fa).detach(),
        )
        ponded = tank.pull_ponded(active=active)
        flood = active & ctx.decide(
            f"{self.site}/make_discharge/ponded",
            ponded[..., 0],
            ">",
            ctx.fa,
            active=active,
            call=call,
        )
        # ponded water at or below FLOAT_ACCURACY has left the tank and goes nowhere: dropped
        ctx.drop(ponded, active & ~flood)
        reply_ = self.push_distributed(ponded, of_type=PONDED_TYPES, tag=PONDED_TAG, active=flood)
        reply_ = tank.push_storage(reply_, active=flood, time=0, force=True)
        ctx.event("ponded_water_cant_reenter", self.name, flood & (reply_[..., 0] != 0).detach())

    # --- validation -------------------------------------------------------------------------
    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """``pipe_time`` and every ``pipe_timearea`` key are whole numbers of days >= 0 (W50;
        an ``AltQueueArc`` shifts integer keys only). A diagram whose fractions do not sum to 1
        warns and runs, as WSIMOD prints and runs (``sewer.py:135-141``). WSIMOD checks only a
        ``pipe_timearea`` given through the overrides (``apply_overrides``); noodl, which sees
        the merged parameters (D5), warns whatever the diagram's source."""
        ignored = super().validate(spec, network, pollutants=pollutants)
        where = f"node {spec.name!r}"
        check_travel_time(where, "pipe_time", spec.parameters.get("pipe_time", 0))
        timearea = spec.parameters.get("pipe_timearea", {0: 1})
        if not isinstance(timearea, Mapping) or not timearea:
            raise ValueError(f"{where}: pipe_timearea {timearea!r} is not a non-empty mapping")
        for key in timearea:
            check_travel_time(where, "pipe_timearea key", key)
        if sum([v for k, v in timearea.items()]) != 1:  # sewer.py:136-137
            warnings.warn(
                f"{where}: the sum of pipe_timearea is not equal to 1 (sewer.py:136-141)",
                stacklevel=2,
            )
        return ignored

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        """The push checks ask the tank only; the default pull check is ``pull_check_basic``."""
        return direction == "pull" and tag == "default"


NODE_CLASSES["Sewer"] = Sewer
