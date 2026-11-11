"""WSIMOD 0.8.1's demand and distribution nodes (``wsimod/nodes/demand.py``,
``nodes/distribution.py``), batched.

A :class:`Demand` generates a demand (a VQIP per component), pulls the total from its in-arcs and
pushes each component where it belongs: ``garden`` to ``Land``, ``house`` to ``Sewer``, anything
else to any node (``create_demand``, ``demand.py:79-119``). The pulled water's quality is
discarded; what is pushed is the demand's own VQIP.

A :class:`Distribution` conveys water: pulls pass to the in-arcs and a push is denied. When the
configured leakage is positive, WSIMOD wraps its two pull handlers (``distribution.py:13-98``):
the request is divided by ``1 - leakage``, and the reply loses ``leakage`` of its volume, which
is pushed to ``Groundwater`` nodes. Water that cannot leak is returned to the caller with the
reply. Decoration is structural (``leakage > 0`` when the node is built), so
:meth:`Distribution.bind` refuses a leakage batch that would change it.

Every method takes the caller's ``active`` mask; state is written only where it holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor

from noodl.apps.wsimod.classes import NODE_SCHEMAS
from noodl.apps.wsimod.constants import UNBOUNDED_CAPACITY
from noodl.apps.wsimod.node import NODE_CLASSES, Node, Tag
from noodl.apps.wsimod.vqip import (
    Pollutants,
    empty,
    extract_vqip,
    sum_vqip,
    v_change_vqip,
    write,
)

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.config import NodeSpec, WsimodNetwork

__all__ = [
    "Demand",
    "Distribution",
    "NonResidentialDemand",
    "ResidentialDemand",
    "UnlimitedDistribution",
]

# demand.py:100-104: where each demand component goes, as ``(of_type, tag)``
DIRECTIONS: dict[str, tuple[str | None, Tag]] = {
    "garden": ("Land", ("Demand", "Garden")),
    "house": ("Sewer", "Demand"),
    "default": (None, "default"),
}
# distribution.py:34-36: where the leak goes
LEAK_TYPES = "Groundwater"


def _f64(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


class Demand(Node):
    """``Demand`` (``demand.py:14-137``): a constant demand, with ``pollutant_load`` set as given.

    Parameters: ``constant_demand`` ``(...)`` and ``pollutant_load`` ``(..., K)`` (D6; absent
    pollutants are 0). The three totals are the day's state (``total_received`` is the pulled
    water, ``total_backup`` what could not be pushed, ``total_demand`` what was generated); they
    are mass-balance terms and reset at ``end_timestep`` (``:133-137``).
    """

    wsimod_type: ClassVar[str] = "Demand"
    store_paths: ClassVar[tuple[str, ...]] = ("total_demand", "total_backup", "total_received")
    step_methods: ClassVar[tuple[str, ...]] = ("create_demand",)
    # demand.py:62-64: generated water appears, backup and received water disappear
    mass_balance: ClassVar[dict[str, str]] = {
        "Demand.total_demand": "in",
        "Demand.total_backup": "out",
        "Demand.total_received": "out",
    }
    parameter_defaults: ClassVar[dict[str, float]] = {"constant_demand": 0}

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        super().__init__(spec, pollutants)
        # demand.py:46-49
        self.push_set_handler["default"] = self.push_set_deny
        self.push_check_handler["default"] = self.push_check_deny
        self.pull_set_handler["default"] = self.pull_set_deny
        self.pull_check_handler["default"] = self.pull_check_deny
        values = spec.parameters
        self.defaults: dict[str, Tensor] = {
            name: _f64(values.get(name, default))
            for name, default in self.parameter_defaults.items()
        }
        load = values.get("pollutant_load", {}) or {}
        self.defaults["pollutant_load"] = torch.tensor(
            [float(load.get(name, 0.0)) for name in pollutants.names], dtype=torch.float64
        ).reshape(pollutants.K)
        # demand.py:52-54
        self.total_demand = empty(pollutants, ())
        self.total_backup = empty(pollutants, ())
        self.total_received = empty(pollutants, ())

    def _param(self, name: str) -> Tensor:
        return self.params.get(name, self.defaults[name])

    def stores(self) -> dict[str, Tensor]:
        return {
            "total_demand": self.total_demand,
            "total_backup": self.total_backup,
            "total_received": self.total_received,
        }

    def end_timestep(self) -> None:
        """``demand.py:133-137``."""
        self.total_demand = empty(self.p, ())
        self.total_backup = empty(self.p, ())
        self.total_received = empty(self.p, ())

    def record_terms(self) -> None:
        ctx = self.ctx
        every = torch.ones(ctx.shape, dtype=torch.bool)
        for term, vqip in self.stores().items():
            ctx.term(f"Demand.{term}", vqip, every)

    def _vqip(self, volume: Tensor, pollutants: Tensor) -> Tensor:
        """``volume`` ``(...)`` with the pollutant columns ``(..., K)``, over their common batch."""
        shape = torch.broadcast_shapes(volume.shape, pollutants.shape[:-1])
        return torch.cat(
            [volume.expand(shape)[..., None], pollutants.expand(shape + pollutants.shape[-1:])],
            dim=-1,
        )

    def get_demand(self, *, active: Tensor) -> dict[str, Tensor]:
        """``demand.py:121-131``: the constant demand, its pollutants set as given."""
        return {
            "default": self._vqip(self._param("constant_demand"), self._param("pollutant_load"))
        }

    def create_demand(self, *, active: Tensor) -> None:
        """``demand.py:79-119``: pull the total demand from the in-arcs, then push each
        component in dict order (``garden``, ``house``, ``default``), each with its own
        ``push_distributed``. Received, unpushed and generated water are summed into the day's
        totals; a shortfall above ``FLOAT_ACCURACY`` and unpushed water above it are events."""
        ctx, p = self.ctx, self.p
        site = f"node:{self.name}/create_demand"
        call = ctx.call(f"node:{self.name}", "create_demand")
        demand = self.get_demand(active=active)
        total_requested = ctx.zeros()
        for item in demand.values():
            total_requested = total_requested + item[..., 0]

        received = self.pull_distributed(self._volume_vqip(total_requested), active=active)
        self.total_received = write(active, received, self.total_received)

        deficit = ctx.decide(  # demand.py:90
            f"{site}/deficit",
            total_requested - received[..., 0],
            ">",
            ctx.fa,
            active=active,
            call=call,
        )
        ctx.event("demand_deficit", self.name, active & deficit)

        for key, item in demand.items():
            of_type, tag = DIRECTIONS[key]
            remaining = self.push_distributed(item, of_type=of_type, tag=tag, active=active)
            self.total_backup = write(
                active, sum_vqip(p, self.total_backup, remaining), self.total_backup
            )
            stuck = ctx.decide(  # demand.py:114
                f"{site}/not_able_to_push:{key}",
                remaining[..., 0],
                ">",
                ctx.fa,
                active=active,
                call=call,
            )
            ctx.event("demand_not_able_to_push", self.name, active & stuck)

        for item in demand.values():  # demand.py:117-119
            self.total_demand = write(
                active, sum_vqip(p, self.total_demand, item), self.total_demand
            )

    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """A quantity-only run ignores ``pollutant_load``."""
        super().validate(spec, network, pollutants=pollutants)
        load = spec.parameters.get("pollutant_load") or {}
        return ["pollutant_load"] if load and not pollutants else []

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        """Every handler is a deny."""
        return False


class NonResidentialDemand(Demand):
    """``NonResidentialDemand`` (``demand.py:140-150``). W4: WSIMOD's ``get_demand`` calls
    itself for ever; this does the evident intent, the constant demand as ``house``."""

    wsimod_type: ClassVar[str] = "NonResidentialDemand"
    mass_balance: ClassVar[dict[str, str]] = {
        "NonResidentialDemand.total_demand": "in",
        "NonResidentialDemand.total_backup": "out",
        "NonResidentialDemand.total_received": "out",
    }

    def record_terms(self) -> None:
        ctx = self.ctx
        every = torch.ones(ctx.shape, dtype=torch.bool)
        for term, vqip in self.stores().items():
            ctx.term(f"NonResidentialDemand.{term}", vqip, every)

    def get_demand(self, *, active: Tensor) -> dict[str, Tensor]:
        return {"house": Demand.get_demand(self, active=active)["default"]}


class ResidentialDemand(Demand):
    """``ResidentialDemand`` (``demand.py:153-324``). WSIMOD renames the class ``Demand``
    (``:214``), which is the type other nodes' ``of_type`` filters see.

    :meth:`bind` takes the day's ``"temperature"`` forcing ``(...)`` (air temperature, read
    whether or not it is a pollutant) besides the parameters.
    """

    wsimod_type: ClassVar[str] = "Demand"
    parameter_defaults: ClassVar[dict[str, float]] = {
        "constant_demand": 0,
        "population": 1,
        "per_capita": 0.12,
        "gardening_efficiency": 0.6 * 0.7,  # demand.py:157
        "constant_temp": 30,
        "constant_weighting": 0.2,
    }

    def configured_parameters(self, date: datetime) -> dict[str, Tensor]:
        return {
            "temperature": torch.tensor(self.spec.data[("temperature", date)], dtype=torch.float64)
        }

    def get_demand(self, *, active: Tensor) -> dict[str, Tensor]:
        """``demand.py:236-248``: ``garden`` then ``house``."""
        return {"garden": self.get_garden_demand(active=active), "house": self.get_house_demand()}

    def get_garden_demand(self, *, active: Tensor) -> Tensor:
        """``demand.py:250-301``: the push capacity ``Land`` nodes offer for the tag
        ``("Demand", "Garden")`` times ``gardening_efficiency``, as a VQIP with no pollutants.
        ``Land`` answers 0 unless it has a ``GardenSurface`` (``land.py:93-94``, 3c)."""
        excess = self.get_connected("push", "Land", ("Demand", "Garden"), active=active).avail
        volume = excess * self._param("gardening_efficiency")  # demand.py:292
        return self._vqip(volume, torch.zeros(self.p.K, dtype=torch.float64))

    def get_house_demand(self) -> Tensor:
        """``demand.py:303-324``: ``population x per_capita`` of water, the additive loads times
        the population, and the temperature mixed from the air's and the constant's by
        ``constant_weighting``. With ``temperature`` not a pollutant the write is dropped."""
        p = self.p
        population = self._param("population")
        consumption = population * self._param("per_capita")
        load = self._param("pollutant_load")
        foul = torch.where(p.additive_mask[1:], load * population[..., None], load)
        if "temperature" in p.names:
            weighting = self._param("constant_weighting")
            mixed = (
                self.params["temperature"] * (1 - weighting)
                + self._param("constant_temp") * weighting
            )
            column = torch.tensor([name == "temperature" for name in p.names])
            foul = torch.where(column, mixed[..., None], foul)
        return self._vqip(consumption, foul)

    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """Every run pollutant but ``temperature`` must be in ``pollutant_load`` (WSIMOD's
        ``foul[pol] *= population`` raises ``KeyError`` on an additive one, ``:313-314``; a
        missing non-additive one breaks its later sums); ``temperature`` forcing must exist on
        every date (``:321``)."""
        ignored = super().validate(spec, network, pollutants=pollutants)
        load = spec.parameters.get("pollutant_load") or {}
        if pollutants:
            for name in network.pollutants.names:
                if name != "temperature" and name not in load:
                    raise ValueError(
                        f"node {spec.name!r}: pollutant_load has no {name!r} "
                        "(demand.py:313-314 raises KeyError)"
                    )
        for date, label in zip(network.dates, network.date_labels, strict=True):
            if ("temperature", date) not in spec.data:
                raise ValueError(f"node {spec.name!r} has no 'temperature' forcing on {label}")
        return ignored


class Distribution(Node):
    """``Distribution`` (``distribution.py:101-153``): conveys pulls, denies pushes, leaks.

    ``leakage`` is a parameter ``(...)``. The node is *decorated* when the configured leakage is
    positive (W17: once, however often the overrides are applied); :meth:`bind` refuses a batch
    whose leakage would change that. A decorated node's pull set and pull check run
    ``distribution.py:13-98``.
    """

    wsimod_type: ClassVar[str] = "Distribution"
    # pull_distributed over the in-arcs; a decorated node also pushes its leak (validate)
    distributes: ClassVar[tuple[str, ...]] = ("in",)
    parameter_defaults: ClassVar[dict[str, float]] = {"leakage": 0}

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        super().__init__(spec, pollutants)
        self.leakage = _f64(spec.parameters.get("leakage", 0))
        self.decorated = float(self.leakage) > 0
        # distribution.py:128-130
        self.push_set_handler["default"] = self.push_set_deny
        self.push_check_handler["default"] = self.push_check_deny
        if self.decorated:  # distribution.py:132-139
            self.pull_set_handler["default"] = self.pull_set_leaky
            self.pull_check_handler["default"] = self.pull_check_leaky

    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """The day's parameters. A ``leakage`` batch must keep the node's structure:
        every instance in ``[0, 1)``, positive on a decorated node and 0 on an undecorated
        one (WSIMOD decorates when the configured leakage is positive)."""
        super().bind(parameters)
        if "leakage" not in self.params:
            return
        value = self.params["leakage"]
        where = f"node {self.name!r}: leakage"
        if bool((value >= 1).any()) or bool((value < 0).any()):
            raise ValueError(
                f"{where} must be in [0, 1) (distribution.py:40 divides by 1 - leakage)"
            )
        if self.decorated and bool((value == 0).any()):
            raise ValueError(
                f"{where} mixes zero and non-zero values on a node whose decorators were applied "
                "(configured leakage > 0); WSIMOD wraps the pull handlers only for leakage > 0"
            )
        if not self.decorated and bool((value != 0).any()):
            raise ValueError(
                f"{where} is non-zero on a node configured without leakage; WSIMOD wraps the "
                "pull handlers only for a configured leakage > 0"
            )

    def _leakage(self) -> Tensor:
        return self.params.get("leakage", self.leakage)

    def _inflated(self, vqip: Tensor) -> Tensor:
        """``vqip["volume"] /= 1 - leakage`` (``distribution.py:40``, ``:91``): the volume
        alone changes."""
        volume = vqip[..., 0] / (1 - self._leakage())
        shape = torch.broadcast_shapes(volume.shape, vqip.shape[:-1])
        rest = vqip[..., 1:].expand(shape + (self.p.K,))
        return torch.cat([volume.expand(shape)[..., None], rest], dim=-1)

    def pull_set_leaky(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``decorate_leakage_set`` (``distribution.py:13-56``): request ``V / (1 - leakage)``
        from upstream; ``leakage`` of the reply goes to ``Groundwater``; what no groundwater
        takes is added back to the reply, so the caller gets more than it asked for."""
        ctx, p = self.ctx, self.p
        call = ctx.call(f"node:{self.name}", "pull_set_leakage")
        reply = self.pull_distributed(self._inflated(vqip), active=active)
        amount_leaked = v_change_vqip(p, reply, reply[..., 0] * self._leakage())
        reply = extract_vqip(p, reply, amount_leaked)
        unsuccessful = self.push_distributed(amount_leaked, of_type=LEAK_TYPES, active=active)
        stuck = ctx.decide(  # distribution.py:46
            f"node:{self.name}/pull_set_leakage/unsuccessful",
            unsuccessful[..., 0],
            ">",
            ctx.fa,
            active=active,
            call=call,
        )
        ctx.event("leakage_not_going_to_gw", self.name, active & stuck)
        return write(stuck, sum_vqip(p, reply, unsuccessful), reply)

    def pull_check_leaky(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``decorate_leakage_check`` (``distribution.py:59-98``): the check of the inflated
        request, less ``leakage`` of its volume."""
        p = self.p
        if vqip is not None:
            vqip = self._inflated(vqip)
        reply = self.pull_check_basic(vqip, active=active)
        amount_leaked = v_change_vqip(p, reply, reply[..., 0] * self._leakage())
        return extract_vqip(p, reply, amount_leaked)

    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """``leakage`` in ``[0, 1)`` (``distribution.py:40`` divides by ``1 - leakage``; WSIMOD
        documents ``1 > leakage >= 0`` and a negative one is never decorated). A decorated node
        pushes its leak over its ``Groundwater`` out-arcs, which need a positive preference
        somewhere."""
        ignored = super().validate(spec, network, pollutants=pollutants)
        leakage = float(spec.parameters.get("leakage", 0))
        if not 0 <= leakage < 1:
            raise ValueError(
                f"node {spec.name!r}: leakage {leakage} is not in [0, 1) "
                "(distribution.py:40 divides by 1 - leakage)"
            )
        if leakage > 0:
            types = {n.name: NODE_SCHEMAS[n.cls].renamed_type for n in network.nodes}
            arcs = [
                a
                for a in network.arcs
                if a.in_port == spec.name and types[a.out_port] == LEAK_TYPES
            ]
            if len(arcs) >= 2 and all(a.preference == 0 for a in arcs):
                raise ValueError(
                    f"node {spec.name!r}: every Groundwater out-arc has preference 0 "
                    f"({', '.join(a.name for a in arcs)}); the leak is distributed over them"
                )
        return ignored

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        """Only the pull check recurses (``pull_check_basic``); the push check is a deny."""
        return direction == "pull" and tag == "default"


class UnlimitedDistribution(Distribution):
    """``UnlimitedDistribution`` (``distribution.py:156-200``): pulls return clean water of the
    requested volume, counted as ``supplied``; the pull check is ``UNBOUNDED_CAPACITY``.

    Its ``__init__`` replaces the handlers the base class decorated (``:172-177``), so a
    configured leakage is silently lost in WSIMOD; ``leakage > 0`` is refused by name instead.
    """

    wsimod_type: ClassVar[str] = "UnlimitedDistribution"
    store_paths: ClassVar[tuple[str, ...]] = ("supplied",)
    distributes: ClassVar[tuple[str, ...]] = ()
    mass_balance: ClassVar[dict[str, str]] = {"UnlimitedDistribution.supplied": "in"}  # :182

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        super().__init__(spec, pollutants)
        # distribution.py:172-177
        self.pull_set_handler["default"] = self.pull_set_unlimited
        self.pull_check_handler["default"] = self.pull_check_unlimited
        self.supplied = empty(pollutants, ())

    def pull_set_unlimited(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``distribution.py:184-196``: the requested volume of clean water."""
        reply = v_change_vqip(self.p, self.ctx.empty(), vqip[..., 0])
        self.supplied = write(active, sum_vqip(self.p, self.supplied, reply), self.supplied)
        return reply

    def pull_check_unlimited(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``distribution.py:174-176``: ``UNBOUNDED_CAPACITY``, whatever the request."""
        return v_change_vqip(
            self.p,
            self.ctx.empty(),
            torch.full(self.ctx.shape, UNBOUNDED_CAPACITY, dtype=torch.float64),
        )

    def stores(self) -> dict[str, Tensor]:
        return {"supplied": self.supplied}

    def end_timestep(self) -> None:
        """``distribution.py:198-200``."""
        self.supplied = empty(self.p, ())

    def record_terms(self) -> None:
        ctx = self.ctx
        ctx.term(
            "UnlimitedDistribution.supplied", self.supplied, torch.ones(ctx.shape, dtype=torch.bool)
        )

    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        if float(spec.parameters.get("leakage", 0)) > 0:
            raise ValueError(
                f"node {spec.name!r}: leakage > 0 on an UnlimitedDistribution is lost by WSIMOD "
                "(distribution.py:172-177); use leakage 0"
            )
        return super().validate(spec, network, pollutants=pollutants)

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        return False


NODE_CLASSES.update(
    {
        "Demand": Demand,
        "NonResidentialDemand": NonResidentialDemand,
        "ResidentialDemand": ResidentialDemand,
        "Distribution": Distribution,
        "UnlimitedDistribution": UnlimitedDistribution,
    }
)
