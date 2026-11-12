"""WSIMOD 0.8.1's ``Catchment`` (``wsimod/nodes/catchment.py:12-138``), batched.

A catchment's water is its forcing: each day's ``flow`` and every pollutant's series, the
additive ones given per unit volume (``get_flow``, ``:59-74``). Abstractions pull from what is
left after the water already sent down **every** out-arc today (``get_avail``, ``:87-101``),
and ``route`` pushes the rest to Node, River and Waste, keeping what nobody takes as unrouted
water (``:76-85``).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor

from noodl.apps.wsimod.engine import dmin
from noodl.apps.wsimod.node import NODE_CLASSES, Node, Tag
from noodl.apps.wsimod.vqip import Pollutants, empty, sum_vqip, v_change_vqip, write

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.config import NodeSpec, WsimodNetwork

__all__ = ["Catchment"]

# catchment.py:81
ROUTE_TYPES = ("Node", "River", "Waste")


class Catchment(Node):
    """``Catchment``: forcing in, abstraction pulls out of it, ``route`` pushes the remainder.

    :meth:`bind` takes ``"flow"`` ``(...)`` and ``"concentration"`` ``(..., K)``, the day's
    forcing in the pollutant order. ``unrouted_water`` accumulates over the day's ``route``
    calls and resets at ``end_timestep`` (``:136-138``).
    """

    wsimod_type: ClassVar[str] = "Catchment"
    store_paths: ClassVar[tuple[str, ...]] = ("unrouted_water",)
    step_methods: ClassVar[tuple[str, ...]] = ("route",)
    distributes: ClassVar[tuple[str, ...]] = ("out",)  # route's push_distributed
    # catchment.py:55-56
    mass_balance: ClassVar[dict[str, str]] = {
        "Catchment.flow": "in",
        "Catchment.unrouted": "out",
    }

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        super().__init__(spec, pollutants)
        # catchment.py:49-52; the push check is ``push_set_deny`` itself, and an arc into a
        # catchment is refused by name (validate)
        self.pull_set_handler["default"] = self.pull_set_abstraction
        self.pull_check_handler["default"] = self.pull_check_abstraction
        self.push_set_handler["default"] = self.push_set_deny
        self.push_check_handler["default"] = self.push_set_deny
        self.unrouted_water = empty(pollutants, ())

    def configured_parameters(self, date: datetime) -> dict[str, Tensor]:
        data = self.spec.data
        flow = data[("flow", date)]
        concentration = [data[(name, date)] for name in self.p.names]
        return {
            "flow": torch.tensor(flow, dtype=torch.float64),
            "concentration": torch.tensor(concentration, dtype=torch.float64).reshape(self.p.K),
        }

    def stores(self) -> dict[str, Tensor]:
        return {"unrouted_water": self.unrouted_water}

    def end_timestep(self) -> None:
        """``catchment.py:136-138``."""
        self.unrouted_water = empty(self.p, ())

    def record_terms(self) -> None:
        ctx = self.ctx
        every = torch.ones(ctx.shape, dtype=torch.bool)
        ctx.term("Catchment.flow", self.get_flow(), every)
        ctx.term("Catchment.unrouted", self.unrouted_water, every)

    def get_flow(self) -> Tensor:
        """``catchment.py:59-74``: the flow, every pollutant's series, the additive ones times
        the flow."""
        flow = self.params["flow"]
        concentration = self.params["concentration"]
        amounts = torch.where(
            self.p.additive_mask[1:], concentration * flow[..., None], concentration
        )
        shape = torch.broadcast_shapes(flow.shape, amounts.shape[:-1])
        flow = flow.expand(shape)[..., None]
        return torch.cat([flow, amounts.expand(shape + (self.p.K,))], dim=-1)

    def get_avail(self) -> Tensor:
        """``catchment.py:87-101``: the flow less every out-arc's ``vqip_in`` volume, in arc
        order."""
        avail = self.get_flow()
        for arc in self.out_arcs.values():
            avail = v_change_vqip(self.p, avail, avail[..., 0] - arc.vqip_in[..., 0])
        return avail

    def route(self, *, active: Tensor) -> None:
        """``catchment.py:76-85``: push what is available to Node, River and Waste; keep the
        rest as unrouted water."""
        avail = self.get_avail()
        reply = self.push_distributed(avail, of_type=ROUTE_TYPES, active=active)
        self.unrouted_water = write(
            active, sum_vqip(self.p, self.unrouted_water, reply), self.unrouted_water
        )

    def pull_check_abstraction(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``catchment.py:103-119``: what is available, clipped to the request when one is
        given (``if vqip:``, ``:116``)."""
        avail = self.get_avail()
        if vqip is not None:
            volume = dmin(
                self.ctx,
                f"node:{self.name}/pull_check_abstraction/clip",
                avail[..., 0],
                vqip[..., 0],
                active=active,
                call=self.ctx.call(f"node:{self.name}", "pull_check_abstraction"),
            )
            avail = v_change_vqip(self.p, avail, volume)
        return avail

    def pull_set_abstraction(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``catchment.py:121-134``: the request's volume, at most what is available."""
        avail = self.get_avail()
        volume = dmin(
            self.ctx,
            f"node:{self.name}/pull_set_abstraction/clip",
            avail[..., 0],
            vqip[..., 0],
            active=active,
            call=self.ctx.call(f"node:{self.name}", "pull_set_abstraction"),
        )
        return v_change_vqip(self.p, avail, volume)

    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """An arc into a catchment is refused (defect table: its push check is
        ``push_set_deny``, ``catchment.py:51-52``). The forcing ``flow`` and every pollutant
        must exist on every date (spec, Forcing checks); a quantity-only run ignores the
        pollutant series, which are returned."""
        super().validate(spec, network, pollutants=pollutants)
        for arc in network.arcs:
            if arc.out_port == spec.name:
                raise ValueError(
                    f"arc {arc.name!r} ends at catchment {spec.name!r}: a catchment accepts "
                    "no water (catchment.py:51-52)"
                )
        for variable in ("flow", *network.pollutants.names):
            for date, label in zip(network.dates, network.date_labels, strict=True):
                if (variable, date) not in spec.data:
                    raise ValueError(
                        f"catchment {spec.name!r} has no {variable!r} forcing on {label}"
                    )
        return [] if pollutants else list(network.pollutants.names)

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        return False


NODE_CLASSES["Catchment"] = Catchment
