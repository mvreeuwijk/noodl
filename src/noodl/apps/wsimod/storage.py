"""WSIMOD 0.8.1's storage family (``wsimod/nodes/storage.py``): ``Storage``, ``Groundwater``,
``QueueGroundwater``, ``Reservoir`` and ``RiverReservoir``, batched.

A storage node wraps one tank under the attribute ``tank`` (:mod:`noodl.apps.wsimod.tanks`) and
answers the four calls with the tank's methods (``storage.py:70-73``). The orchestration methods
(``distribute``, ``infiltrate``, ``make_abstractions``, ``satisfy_environmental``) call the tank
and the node's own ``push_distributed``/``pull_distributed``, in WSIMOD's order and with its
arithmetic, so every value is bitwise WSIMOD's.

Every method takes the caller's ``active`` mask. The tank's state is written only where it holds
(D3), and a data-dependent branch is a :meth:`Ctx.decide` (D12). WSIMOD's warning prints are
events (D13).

``decays`` ``None`` builds a plain ``Tank``; a dict, ``{}`` included, builds a ``DecayTank``
(``storage.py:53-59``) that decays its storage at the end of the day
(:mod:`noodl.apps.wsimod.tanks`). Each node binds its tank's
:class:`~noodl.apps.wsimod.decay.Decay` from its bound ``decays.constant`` and
``decays.exponent`` and the day's ``temperature`` (D15), and refuses a changed inert entry on a
node whose ``decays`` does not list the pollutant (X9). A ``Groundwater`` and a
``QueueGroundwater`` lose their configured forcing and read their temperature only from a
``filename`` override (``storage.py:202-203``, ``:291-292``, X11), so a decaying one needs such
an override (:meth:`Groundwater.validate`, :meth:`QueueGroundwater.validate`). A decaying
``QueueGroundwater``'s tank is a :class:`~noodl.apps.wsimod.tanks.DecayQueueTank`
(``storage.py:307-315``).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from noodl.apps.wsimod.arcs import _put_row, check_travel_time
from noodl.apps.wsimod.constants import UNBOUNDED_CAPACITY
from noodl.apps.wsimod.decay import Decay, DecaySpec, check_inert, read_decays
from noodl.apps.wsimod.engine import Ctx, dmax, dmin
from noodl.apps.wsimod.node import NODE_CLASSES, Node, Tag
from noodl.apps.wsimod.tanks import DecayQueueTank, DecayTank, QueueTank, Tank, bind_timearea
from noodl.apps.wsimod.vqip import (
    Pollutants,
    extract_vqip,
    safe_den,
    sum_vqip,
    v_change_vqip,
    write,
)

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.config import NodeSpec, WsimodNetwork

__all__ = [
    "Groundwater",
    "QueueGroundwater",
    "Reservoir",
    "River",
    "RiverReservoir",
    "Storage",
    "exp_python",
    "pow_half",
    "river_derived",
    "riverrc",
]

# storage.py:226, :1064, :1102: where a groundwater baseflow, a reservoir spill and the
# reservoir's downstream capacity go
DOWNSTREAM_TYPES = ("Node", "River", "Waste")
# storage.py:673, :703: what a river pulls from upstream; :933: where it discharges
RIVER_UPSTREAM_TYPES = ("River", "Node")
RIVER_DOWNSTREAM_TYPES = ("River", "Node", "Waste")


def _f64(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


class _PowHalf(torch.autograd.Function):
    """``x ** 0.5`` as Python evaluates it (the C library's ``pow``), elementwise.

    ``torch.sqrt`` (and ``torch.pow(x, 0.5)``, which torch turns into a square root) differs
    from ``float.__pow__`` in the last bit for about 1 % of the arguments on the developer
    machine, so ``Groundwater.infiltrate`` (``storage.py:236``) calls the same ``pow`` WSIMOD
    does. The gradient is that of the square root, and 0 (not infinity) where ``x == 0``: the
    infinite slope at 0+ is documented, not propagated.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        flat = x.detach().reshape(-1).tolist()
        return torch.tensor([v**0.5 for v in flat], dtype=torch.float64).reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        positive = x > 0
        root = torch.sqrt(torch.where(positive, x, torch.ones_like(x)))
        return torch.where(positive, grad * 0.5 / root, torch.zeros_like(grad))


def pow_half(x: Tensor) -> Tensor:
    """``x ** 0.5`` bitwise as Python computes it; gradient 0 at ``x == 0``."""
    return _PowHalf.apply(_f64(x))


class _ExpPython(torch.autograd.Function):
    """``math.exp(x)`` as WSIMOD calls it (``storage.py:911``), elementwise.

    ``torch.exp`` differs from the C library's ``exp`` in the last bit for about 1 % of the
    arguments on the developer machine, so ``River.get_riverrc`` calls the same ``math.exp``
    WSIMOD does, once per instance (the :func:`pow_half` pattern). The gradient is
    ``exp(x)``.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        flat = x.detach().reshape(-1).tolist()
        out = torch.tensor([math.exp(v) for v in flat], dtype=torch.float64).reshape(x.shape)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad: Tensor) -> Tensor:
        (out,) = ctx.saved_tensors
        return grad * out


def exp_python(x: Tensor) -> Tensor:
    """``math.exp(x)`` bitwise as Python computes it; gradient ``exp(x)``."""
    return _ExpPython.apply(_f64(x))


def riverrc(kt: Tensor) -> Tensor:
    """``1 - kt + kt * exp(-1 / kt)``, and 1 where ``kt == 0`` (``storage.py:900-915``).

    The evaluation order is WSIMOD's: ``(1 - kt) + (kt * exp((-1) / kt))``. Where ``kt == 0`` the
    expression runs on a safe ``kt = 1`` (double ``where``), so the gradient stays finite.
    """
    kt = _f64(kt)
    nonzero = (kt != 0).detach()
    safe = torch.where(nonzero, kt, torch.ones_like(kt))
    value = 1 - safe + safe * exp_python(-1 / safe)
    return torch.where(nonzero, value, torch.ones_like(value))


class Storage(Node):
    """``Storage`` (``storage.py:15-143``): a node around one :class:`Tank`.

    Constructor parameters come from ``spec.parameters`` (``capacity``, ``area``, ``datum``,
    ``initial_storage``: a volume or a VQIP mapping); :meth:`bind` replaces them with the day's
    (batched) values. The tank is ``self.tank``.
    """

    wsimod_type: ClassVar[str] = "Storage"
    store_paths: ClassVar[tuple[str, ...]] = ("tank.storage",)
    carried_paths: ClassVar[tuple[str, ...]] = ("tank.storage",)
    step_methods: ClassVar[tuple[str, ...]] = ("distribute",)
    distributes: ClassVar[tuple[str, ...]] = ("out",)  # distribute's push_distributed
    # storage.py:76: the storage change is the engine's, from the stores (D11); a DecayTank's
    # decayed mass leaves the system (tanks.py:449-457)
    mass_balance: ClassVar[dict[str, str]] = {"DecayTank.decayed": "out"}
    # parameter name -> WSIMOD default, the node's own numeric parameters (bind replaces them)
    parameter_defaults: ClassVar[dict[str, float]] = {}

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        super().__init__(spec, pollutants)
        self.site = f"node:{self.name}"
        self._ctx: Ctx | None = None
        values = spec.parameters
        self.defaults: dict[str, Tensor] = {
            name: _f64(values.get(name, default))
            for name, default in self.parameter_defaults.items()
        }
        # X9, X34: the decays drivers are dense over the K pollutants; a pollutant the
        # configuration does not list is inert (constant 0, exponent 1)
        self.decay_spec = self._decay_spec(values.get("decays"))
        configured = self.decay_spec
        self.defaults["decays.constant"] = _f64(
            [configured.constant.get(n, 0.0) if configured else 0.0 for n in self.p.names]
        ).reshape(self.p.K)
        self.defaults["decays.exponent"] = _f64(
            [configured.exponent.get(n, 1.0) if configured else 1.0 for n in self.p.names]
        ).reshape(self.p.K)
        self.tank = self._make_tank(values)
        if isinstance(self.tank, DecayTank):  # D8: a per-day store the replay compares
            self.store_paths = (*type(self).store_paths, "tank.total_decayed")
        self._set_handlers()

    # --- construction -----------------------------------------------------------------------
    def _initial_storage(self, values: Mapping) -> Tensor:
        """``Tank``'s ``initial_storage`` (``tanks.py:36-53``): a number is a volume, a mapping
        a VQIP (keys outside the run's pollutants are left out, as a quantity-only run does)."""
        initial = values.get("initial_storage", 0)
        if isinstance(initial, Mapping):
            known = {k: v for k, v in initial.items() if k == "volume" or k in self.p.names}
            return self.p.to_vqip(known)
        return _f64(initial)

    def _decay_spec(self, decays: Mapping | None) -> DecaySpec | None:
        """The configured ``decays`` (checked by the reader and again by :meth:`validate`) on the
        run's pollutants: a quantity-only run decays nothing, as it carries no pollutant."""
        if decays is None:
            return None
        # in a quantity-only run the keys outside ``p.names`` are dropped only here, after the
        # reader (and ``validate``) checked them against the full pollutant list
        known = {k: v for k, v in decays.items() if k in self.p.names}
        return read_decays(f"node {self.name!r}", known, self.p)

    def _make_tank(self, values: Mapping) -> Tank:
        """``storage.py:53-59``: a plain tank for ``decays`` ``None``, a ``DecayTank`` for a dict
        (``{}`` included: it decays nothing but reads the temperature at the end of the day)."""
        kwargs = dict(
            capacity=_f64(values.get("capacity", 0)),
            area=_f64(values.get("area", 0)),
            datum=_f64(values.get("datum", 0)),
            initial_storage=self._initial_storage(values),
            owner=self.name,
            path="tank",
        )
        if values.get("decays") is None:
            return Tank(self.p, **kwargs)
        spec = self.decay_spec
        decay = Decay(self.p, spec, site=f"node:{self.name}/tank") if spec.keys else None
        return DecayTank(self.p, parent=self, decay=decay, **kwargs)

    def _set_handlers(self) -> None:
        """``storage.py:70-73``."""
        self.push_set_handler["default"] = self.push_set_storage
        self.push_check_handler["default"] = self._push_check_tank
        self.pull_set_handler["default"] = self._pull_set_tank
        self.pull_check_handler["default"] = self._pull_check_tank

    @property
    def ctx(self) -> Ctx:
        if self._ctx is None:
            raise RuntimeError(f"node {self.name!r}: no day context")
        return self._ctx

    @ctx.setter
    def ctx(self, ctx: Ctx) -> None:
        self._ctx = ctx
        for tank in self._tanks():
            tank.ctx = ctx

    def _tanks(self) -> list[Tank]:
        return [self.tank]

    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """This day's parameters by name: ``capacity``, ``area`` and ``datum`` reach the tank;
        the class's own parameters (``parameter_defaults``) are read by name."""
        super().bind(parameters)
        pct = self.params.get("infiltration_pct")
        if pct is not None and bool((pct < 0).any()):
            raise ValueError(
                f"node {self.name!r}: infiltration_pct is negative (storage.py:236 takes a root)"
            )
        residence = self.params.get("residence_time")
        if residence is not None and bool((residence == 0).any()):
            raise ValueError(f"node {self.name!r}: residence_time is 0 (storage.py:224 divides)")
        for tank in self._tanks():
            for name in ("capacity", "area", "datum"):
                if name in self.params:
                    setattr(tank, name, self.params[name])
        self._bind_decay()

    def configured_parameters(self, date: datetime) -> dict[str, Tensor]:
        """The day's ``temperature`` forcing of a node with a ``decays`` dict (the replay loads
        it, as :meth:`WsimodDay._bind` does from the driver)."""
        if self.spec.parameters.get("decays") is None:
            return {}
        return {"temperature": _f64(self.spec.data[("temperature", date)])}

    def _bind_decay(self) -> None:
        """Bind the tank's :class:`Decay` from the day's ``decays.*`` and ``temperature`` (D15,
        R5). A node without decay, or whose ``decays`` does not list a pollutant, refuses a
        changed inert entry by name (X9); a quantity-only run has no pollutant to decay."""
        if self.p.K == 0:
            return
        where = f"node {self.name!r}"
        constant, exponent = self._param("decays.constant"), self._param("decays.exponent")
        decay = getattr(self.tank, "decay", None)
        if decay is None:
            check_inert(
                where,
                "decays.constant/exponent",
                torch.zeros(self.p.K, dtype=torch.bool),
                constant,
                exponent,
                self.p.names,
            )
            return
        if "temperature" not in self.params:
            raise KeyError(f"{where}: no ('temperature', t) forcing (core.py:484-487)")
        decay.bind(self.ctx, constant, exponent, self.params["temperature"])

    def _param(self, name: str) -> Tensor:
        return self.params.get(name, self.defaults[name])

    def _volume(self, volume: Tensor) -> Tensor:
        return self._volume_vqip(volume)

    # --- the handlers (storage.py:70-73, :106-118) --------------------------------------------
    def push_set_storage(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``storage.py:106-118``: the tank's ``push_storage``; returns what it rejects."""
        return self.tank.push_storage(vqip, active=active)

    def _push_check_tank(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        return self.tank.get_excess(vqip, active=active)

    def _pull_set_tank(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        return self.tank.pull_storage(vqip, active=active)

    def _pull_check_tank(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        return self.tank.get_avail(vqip, active=active)

    # --- state ------------------------------------------------------------------------------
    def stores(self) -> dict[str, Tensor]:
        return {f"tank.{path}": value for path, value in self.tank.stores().items()}

    def end_timestep(self) -> None:
        """``storage.py:134-136``."""
        self.tank.end_timestep()

    # --- orchestration ----------------------------------------------------------------------
    def _force_back(self, retained: Tensor, event: str, *, active: Tensor) -> None:
        """``_ = tank.push_storage(retained, force=True)`` and the print when ``retained`` is
        above ``FLOAT_ACCURACY`` (``storage.py:125-127``, ``:227-229``)."""
        self.tank.push_storage(retained, force=True, active=active)
        self.ctx.event(event, self.name, active & (retained[..., 0] > self.ctx.fa).detach())

    def distribute(self, *, active: Tensor) -> None:
        """``storage.py:120-127``: discharge all of the tank's storage with
        ``push_distributed`` (no ``of_type``) and force back what is not pushed."""
        storage = self.tank.pull_storage(self.tank.get_avail(active=active), active=active)
        retained = self.push_distributed(storage, active=active)
        self._force_back(retained, "storage_unable_to_push", active=active)

    # --- validation -------------------------------------------------------------------------
    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """``decays`` (the merged dict, D6) is read again: an override entry without
        ``exponent`` crashes WSIMOD too. A node with a ``decays`` dict reads ``temperature`` on
        every date (``core.py:484-487``). A quantity-only run ignores a VQIP
        ``initial_storage``'s pollutants."""
        super().validate(spec, network, pollutants=pollutants)
        decays = spec.parameters.get("decays")
        where = f"node {spec.name!r}"
        if decays is not None:
            read_decays(where, decays, network.pollutants)
            for date, label in zip(network.dates, network.date_labels, strict=True):
                if ("temperature", date) not in spec.data:
                    raise ValueError(
                        f"{where} has no 'temperature' forcing on {label} (its DecayTank reads "
                        "it at the end of every day, core.py:484-487)"
                    )
        initial = spec.parameters.get("initial_storage", 0)
        if not pollutants and isinstance(initial, Mapping):
            if any(k != "volume" for k in initial):
                return ["initial_storage"]
        return []

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        """The tank's ``get_excess`` and ``get_avail`` call no neighbour."""
        return False


class Groundwater(Storage):
    """``Groundwater`` (``storage.py:146-245``): a tank with a residence time, and infiltration
    to sewers above a threshold."""

    wsimod_type: ClassVar[str] = "Groundwater"
    step_methods: ClassVar[tuple[str, ...]] = ("infiltrate", "distribute")
    parameter_defaults: ClassVar[dict[str, float]] = {
        "residence_time": 200,
        "infiltration_threshold": 1,
        "infiltration_pct": 0,
    }

    def distribute(self, *, active: Tensor) -> None:
        """``storage.py:222-229``: ``V / residence_time`` to Node, River and Waste, the rest
        forced back."""
        avail = self.tank.get_avail(active=active)[..., 0] / self._param("residence_time")
        to_send = self.tank.pull_storage(self._volume(avail), active=active)
        retained = self.push_distributed(to_send, of_type=DOWNSTREAM_TYPES, active=active)
        self._force_back(retained, "storage_unable_to_push", active=active)

    def infiltrate(self, *, active: Tensor) -> None:
        """``storage.py:231-245``: ``(max(V - capacity x threshold, 0) x pct) ** 0.5`` pushed to
        Sewer nodes, the rest left in the tank.

        The root is Python's ``** 0.5`` (:func:`pow_half`), with gradient 0 where the argument
        is 0: the slope of the root is infinite at 0+ and is not propagated.
        """
        ctx, tank = self.ctx, self.tank
        call = ctx.call(self.site, "infiltrate")
        avail = tank.get_avail(active=active)[..., 0]
        excess = dmax(
            ctx,
            f"{self.site}/infiltrate/excess",
            avail - tank.capacity * self._param("infiltration_threshold"),
            0.0,
            active=active,
            call=call,
        )
        avail = pow_half(excess * self._param("infiltration_pct"))
        to_send = tank.pull_storage(self._volume(avail), active=active)
        retained = self.push_distributed(to_send, of_type="Sewer", active=active)
        tank.push_storage(retained, force=True, active=active)

    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """``residence_time`` divides the volume (``storage.py:224``) and the root's argument
        must not be negative (``:236``: a negative base to ``0.5`` is a complex number)."""
        _needs_filename_temperature(spec, network, "Groundwater", "storage.py:202-203")
        ignored = super().validate(spec, network, pollutants=pollutants)
        if float(spec.parameters.get("residence_time", 200)) == 0:
            raise ValueError(f"node {spec.name!r}: residence_time is 0 (storage.py:224 divides)")
        if float(spec.parameters.get("infiltration_pct", 0)) < 0:
            raise ValueError(f"node {spec.name!r}: infiltration_pct is negative (storage.py:236)")
        return ignored


def _needs_filename_temperature(
    spec: NodeSpec, network: WsimodNetwork, cls: str, lines: str
) -> None:
    """A decaying ``Groundwater`` or ``QueueGroundwater`` loses its configured forcing, so the
    temperature its decay reads (``core.py:484-487``) is a ``filename`` override's (X11); a
    missing date is refused by name."""
    if spec.parameters.get("decays") is None:
        return
    for date, label in zip(network.dates, network.date_labels, strict=True):
        if ("temperature", date) not in spec.data:
            raise ValueError(
                f"node {spec.name!r}: a {cls} with 'decays' needs a ('temperature', t) forcing "
                f"on {label} from a 'filename' override: WSIMOD loses its configured "
                f"data_input_dict ({lines})"
            )


class QueueGroundwater(Storage):
    """``QueueGroundwater`` (``storage.py:248-433``): baseflow routed by a time-area diagram
    through a :class:`QueueTank`. WSIMOD renames the class ``Groundwater`` (``:294``).

    Its push check is a **frozen plain tank's** ``get_excess``: ``Storage.__init__`` binds the
    handler to the tank it builds first (``storage.py:53-59``, ``:71``), and ``QueueGroundwater``
    then replaces ``self.tank`` by the ``QueueTank`` (``:301``) without rebinding it. The check
    therefore reports ``max(capacity - initial storage, 0)`` for ever, which ``check_tank``
    reproduces (it is never pushed into). The pull handlers are rebound (``:297-298``).

    With ``decays`` (a dict) the tank is a :class:`~noodl.apps.wsimod.tanks.DecayQueueTank`
    (``:307-315``), whose internal arc decays the queued water and whose end of day extracts
    the previous day's decay from ``storage`` and does not flush slot 0 (X13). Its
    ``number_of_timesteps`` is WSIMOD's default 1 (``tanks.py:648``, X12); the override
    ``number_of_timesteps``, which ``DecayQueueTank.apply_overrides`` consumes
    (``tanks.py:682-685``), sets it (X28) and is refused on a non-decaying node. The internal
    arc's ``total_decayed`` is carried (``"tank.internal_arc.total_decayed"``, D8). The
    temperature is a ``filename`` override's (``:291-292``, X11).
    """

    wsimod_type: ClassVar[str] = "Groundwater"
    # the DecayQueueTank's lagged extraction leaves the system (tanks.py:694)
    mass_balance: ClassVar[dict[str, str]] = {
        **Storage.mass_balance,
        "DecayQueueTank.decayed": "out",
    }
    store_paths: ClassVar[tuple[str, ...]] = (
        "tank.storage",
        "tank.active_storage",
        "tank.internal_arc.register",
        "tank.internal_arc.present",
    )
    carried_paths: ClassVar[tuple[str, ...]] = store_paths
    step_methods: ClassVar[tuple[str, ...]] = ("infiltrate", "distribute")

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        values = spec.parameters
        where = f"node {spec.name!r}"
        timearea = values.get("timearea", {0: 1})
        self.timearea: dict[int, Tensor] = {
            check_travel_time(where, "timearea key", k): _f64(v) for k, v in timearea.items()
        }
        super().__init__(spec, pollutants)
        # storage.py:301-306 (the first tank built is the frozen one; see the class text)
        self.check_tank = Tank(
            self.p,
            capacity=self.tank.capacity,
            area=self.tank.area,
            datum=self.tank.datum,
            initial_storage=self._initial_storage(values),
            owner=self.name,
            path="check_tank",
        )
        self.push_check_handler["default"] = self._push_check_frozen
        if isinstance(self.tank, DecayQueueTank):  # D8: carried, extracted a day later
            self.store_paths = (*type(self).store_paths, "tank.internal_arc.total_decayed")
            self.carried_paths = self.store_paths

    def _make_tank(self, values: Mapping) -> QueueTank:
        """``storage.py:300-315``: a ``QueueTank`` (``number_of_timesteps`` 0) for ``decays``
        ``None``, a ``DecayQueueTank`` (default 1, or the override's) for a dict."""
        kwargs = dict(
            push_times=tuple(self.timearea),
            capacity=_f64(values.get("capacity", 0)),
            area=_f64(values.get("area", 0)),
            datum=_f64(values.get("datum", 0)),
            initial_storage=self._initial_storage(values),
            owner=self.name,
            path="tank",
        )
        if values.get("decays") is None:
            return QueueTank(self.p, number_of_timesteps=0, **kwargs)
        spec = self.decay_spec
        site = f"node:{self.name}/tank/internal_arc"
        decay = Decay(self.p, spec, site=site) if spec.keys else None
        return DecayQueueTank(
            self.p,
            parent=self,
            decay=decay,
            number_of_timesteps=values.get("number_of_timesteps", 1),  # tanks.py:648, :682-685
            **kwargs,
        )

    def _set_handlers(self) -> None:
        """``storage.py:296-298``; the push check stays the frozen tank's (see the class)."""
        super()._set_handlers()
        self.push_set_handler["default"] = self.push_set_timearea
        self.pull_set_handler["default"] = self.pull_set_active
        self.pull_check_handler["default"] = self.pull_check_active

    def _tanks(self) -> list[Tank]:
        return [self.tank, self.check_tank] if hasattr(self, "check_tank") else [self.tank]

    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """The day's parameters; ``"timearea"`` ``(..., L+1)`` holds the fractions by travel
        time (W56, :func:`~noodl.apps.wsimod.tanks.bind_timearea`)."""
        super().bind(parameters)
        if "timearea" in self.params:
            self.timearea = bind_timearea(
                f"node {self.name!r}", "timearea", tuple(self.timearea), self.params["timearea"]
            )

    def _push_check_frozen(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``storage.py:71`` bound to the first tank (see the class text)."""
        return self.check_tank.get_excess(vqip, active=active)

    def push_set_timearea(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``storage.py:330-348``: split the push over the time-area diagram, each share into
        the tank under its time; the replies are summed."""
        p = self.p
        reply = self.ctx.empty()
        for time, normalised in self.timearea.items():
            share = v_change_vqip(p, vqip, vqip[..., 0] * normalised)
            reply_ = self.tank.push_storage(share, active=active, time=time)
            reply = sum_vqip(p, reply, reply_)
        return reply

    def distribute(self, *, active: Tensor) -> None:
        """``storage.py:350-364``: ``update_queue``, push all ``active_storage``, pull what was
        sent."""
        p, ctx, tank = self.p, self.ctx, self.tank
        tank.internal_arc.update_queue(active=active)
        remaining = self.push_distributed(tank.active_storage, active=active)
        ctx.event(
            "groundwater_couldnt_push_all",
            self.name,
            active & (remaining[..., 0] > ctx.fa).detach(),
        )
        sent = v_change_vqip(
            p, tank.active_storage, tank.active_storage[..., 0] - remaining[..., 0]
        )
        reply = tank.pull_storage(sent, active=active)
        ctx.event(
            "miscalculated_tank_storage",
            self.name,
            active & ((reply[..., 0] - sent[..., 0]) > ctx.fa).detach(),
        )

    def infiltrate(self, *, active: Tensor) -> None:
        """``storage.py:366-368``: nothing."""

    def pull_check_active(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``storage.py:370-384``: the active storage, at ``min(request, V)`` when a request is
        given."""
        tank = self.tank
        if vqip is None:
            return tank.active_storage
        volume = dmin(
            self.ctx,
            f"{self.site}/pull_check_active/clip",
            vqip[..., 0],
            tank.active_storage[..., 0],
            active=active,
            call=self.ctx.call(self.site, "pull_check_active"),
        )
        return v_change_vqip(self.p, tank.active_storage, volume)

    def pull_set_active(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``storage.py:386-433``: pull ``min(total storage, request)`` pro rata from every
        queue slot and the active store, each in proportion to its volume over the **total**
        storage (inconsistent with :meth:`pull_check_active`, as WSIMOD notes), then take the
        sum out of ``storage``. A pull below ``FLOAT_ACCURACY`` returns empty.

        WSIMOD walks the queue dict in insertion order; the slots are walked in ascending time,
        and a slot absent from WSIMOD's key set (``present``) is skipped. The two orders agree
        for a time-area whose keys are entered in ascending order (the queue starts as {0, 1},
        ``arcs.py:567``, so gaps are harmless; only an order such as {3: ..., 2: ...} could differ).
        """
        p, ctx, tank = self.p, self.ctx, self.tank
        active = torch.as_tensor(active, dtype=torch.bool)
        call = ctx.call(self.site, "pull_set_active")
        total_storage = tank.storage[..., 0]
        total_pull = dmin(
            ctx,
            f"{self.site}/pull_set_active/clip",
            total_storage,
            vqip[..., 0],
            active=active,
            call=call,
        )
        small = ctx.decide(
            f"{self.site}/pull_set_active/below_fa",
            total_pull,
            "<",
            ctx.fa,
            active=active,
            call=call,
        )
        run = active & ~small
        den = safe_den(total_storage)
        arc = tank.internal_arc
        pulled = ctx.empty()
        for t in range(arc.length + 1):
            row = arc.register[..., t, :]
            here = arc.present[..., t] & run
            t_pulled = v_change_vqip(p, row, row[..., 0] * total_pull / den)
            arc.register = _put_row(arc.register, t, extract_vqip(p, row, t_pulled), here)
            pulled = write(here, sum_vqip(p, pulled, t_pulled), pulled)
        storage = tank.active_storage
        a_pulled = v_change_vqip(p, storage, storage[..., 0] * total_pull / den)
        tank.active_storage = write(run, extract_vqip(p, storage, a_pulled), storage)
        pulled = write(run, sum_vqip(p, pulled, a_pulled), pulled)
        tank.storage = write(run, extract_vqip(p, tank.storage, pulled), tank.storage)
        return pulled

    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """Every time-area key and the tank's ``number_of_timesteps`` are whole numbers of days
        >= 0 (W50; an ``AltQueueArc`` shifts integer keys only, ``arcs.py:653-660``). The
        ``number_of_timesteps`` override is a ``DecayQueueTank``'s (``tanks.py:682-685``, X28):
        on a node without ``decays`` it is refused, as in 3a. A decaying node's temperature is a
        ``filename`` override's (X11)."""
        where = f"node {spec.name!r}"
        _needs_filename_temperature(spec, network, "QueueGroundwater", "storage.py:291-292")
        ignored = super().validate(spec, network, pollutants=pollutants)
        if "number_of_timesteps" in spec.parameters:
            if spec.parameters.get("decays") is None:
                raise ValueError(
                    f"{where}: the override 'number_of_timesteps' is accepted only on a "
                    "QueueGroundwater with 'decays' (its DecayQueueTank's, tanks.py:682-685)"
                )
            check_travel_time(where, "number_of_timesteps", spec.parameters["number_of_timesteps"])
        timearea = spec.parameters.get("timearea", {0: 1})
        if not isinstance(timearea, Mapping) or not timearea:
            raise ValueError(f"{where}: timearea {timearea!r} is not a non-empty mapping")
        for key in timearea:
            check_travel_time(where, "timearea key", key)
        return ignored


class Reservoir(Storage):
    """``Reservoir`` (``storage.py:958-989``): a tank refilled by pulling through its in-arcs."""

    wsimod_type: ClassVar[str] = "Reservoir"
    step_methods: ClassVar[tuple[str, ...]] = ("distribute", "make_abstractions")
    distributes: ClassVar[tuple[str, ...]] = ("out", "in")

    def make_abstractions(self, *, active: Tensor) -> None:
        """``storage.py:983-989``: pull the tank's excess (the full VQIP) from upstream, push it
        into the tank without force, then force the spill in too. The tank is never clamped:
        a tank above capacity has no excess, and anything pulled all spills back in
        (``reservoir_spill``; nothing is pushed downstream)."""
        reply = self.pull_distributed(self.tank.get_excess(active=active), active=active)
        spill = self.tank.push_storage(reply, active=active)
        self.tank.push_storage(spill, force=True, active=active)
        self.ctx.event(
            "reservoir_spill", self.name, active & (spill[..., 0] > self.ctx.fa).detach()
        )


class RiverReservoir(Reservoir):
    """``RiverReservoir`` (``storage.py:992-1133``): a reservoir that receives water from its
    in-arcs, spills downstream and owes an environmental flow. WSIMOD renames the class
    ``Reservoir`` (``:1043``)."""

    wsimod_type: ClassVar[str] = "Reservoir"
    store_paths: ClassVar[tuple[str, ...]] = ("tank.storage", "total_environmental_satisfied")
    # total_environmental_satisfied is reset every day (storage.py:1130-1133)
    carried_paths: ClassVar[tuple[str, ...]] = ("tank.storage",)
    step_methods: ClassVar[tuple[str, ...]] = (
        "distribute",
        "make_abstractions",
        "satisfy_environmental",
    )
    distributes: ClassVar[tuple[str, ...]] = ("out", "in")
    parameter_defaults: ClassVar[dict[str, float]] = {"environmental_flow": 0}

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        super().__init__(spec, pollutants)
        # storage.py:1019
        self.total_environmental_satisfied = torch.zeros((), dtype=torch.float64)

    def _set_handlers(self) -> None:
        """``storage.py:1021-1022``."""
        super()._set_handlers()
        self.push_set_handler["default"] = self.push_set_river_reservoir
        self.push_check_handler["default"] = self.push_check_river_reservoir

    def stores(self) -> dict[str, Tensor]:
        return {
            **super().stores(),
            "total_environmental_satisfied": self.total_environmental_satisfied,
        }

    def end_timestep(self) -> None:
        """``storage.py:1130-1133``."""
        super().end_timestep()
        self.total_environmental_satisfied = torch.zeros((), dtype=torch.float64)

    def push_set_river_reservoir(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``storage.py:1060-1084``: force the water in, pull the ponded water out and push it
        downstream; what goes downstream counts as environmental flow."""
        self.tank.push_storage(vqip, force=True, active=active)
        spill = self.tank.pull_ponded(active=active)
        reply = self.push_distributed(spill, of_type=DOWNSTREAM_TYPES, active=active)
        satisfied = self.total_environmental_satisfied + (spill[..., 0] - reply[..., 0])
        self.total_environmental_satisfied = write(
            active, satisfied, self.total_environmental_satisfied
        )
        return reply

    def push_check_river_reservoir(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``storage.py:1086-1110``: the tank's excess plus the downstream ``avail``, clipped to
        the request when one is given."""
        ctx = self.ctx
        downstream = self.get_connected("push", DOWNSTREAM_TYPES, active=active).avail
        excess = self.tank.get_excess(active=active)
        new_v = excess[..., 0] + downstream
        if vqip is not None:
            new_v = dmin(
                ctx,
                f"{self.site}/push_check_river_reservoir/clip",
                vqip[..., 0],
                new_v,
                active=active,
                call=ctx.call(self.site, "push_check_river_reservoir"),
            )
        return v_change_vqip(self.p, excess, new_v)

    def satisfy_environmental(self, *, active: Tensor) -> None:
        """``storage.py:1112-1128``: pull what is still owed from the tank, push it on
        (``push_distributed``, no ``of_type``) and force back the rest."""
        ctx, tank = self.ctx, self.tank
        to_satisfy = dmax(
            ctx,
            f"{self.site}/satisfy_environmental/owed",
            self._param("environmental_flow") - self.total_environmental_satisfied,
            0.0,
            active=active,
            call=ctx.call(self.site, "satisfy_environmental"),
        )
        environmental = tank.pull_storage(self._volume(to_satisfy), active=active)
        reply = self.push_distributed(environmental, active=active)
        tank.push_storage(reply, force=True, active=active)
        ctx.event(
            "environmental_not_able_to_push", self.name, active & (reply[..., 0] > ctx.fa).detach()
        )
        satisfied = self.total_environmental_satisfied + environmental[..., 0]
        self.total_environmental_satisfied = write(
            active, satisfied, self.total_environmental_satisfied
        )

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        """The push check asks the downstream arcs (``get_connected``, ``storage.py:1102``)."""
        return direction == "push" and tag == "default"


def river_derived(spec: NodeSpec) -> dict[str, float]:
    """What a ``River`` computes itself, from ``spec``'s merged ``length`` and ``width``.

    ``area = length * width`` and ``capacity = UNBOUNDED_CAPACITY`` (``storage.py:524-533``);
    WSIMOD forces both over whatever the configuration or an override says
    (``:536-540``, ``:643-658``), so the merged ``area`` and ``capacity`` are never read. A
    ``depth`` other than 2 warns that it is unused (``:514-518``).
    """
    values = spec.parameters
    if values.get("depth", 2) != 2:
        warnings.warn(
            "warning: the depth parameter is unused by River nodes because it is "
            "intended for capacity to be unbounded. It may be removed in a future version.",
            stacklevel=2,
        )
    area = float(values.get("length", 200) * values.get("width", 20))
    return {"area": area, "capacity": UNBOUNDED_CAPACITY}


class River(Storage):
    """``River`` (``storage.py:452-955``) without ``biochemical_processes`` (``:738-898``): a
    tank that is never full, discharged by the routing coefficient :func:`riverrc` and pulled
    from with a minimum residual flow.

    Biochemistry runs when ``"nitrate"`` is a pollutant (``:917-924``): that is sub-project 3b,
    and :meth:`validate` refuses it, so :meth:`calculate_discharge` is a no-op. A ``decays`` dict
    builds a ``DecayTank`` as for any :class:`Storage`: the tank decays at the end of the day,
    from the river's own ``temperature`` forcing (``storage.py:53-59``). The tank's
    ``area`` and ``capacity`` are derived (:func:`river_derived`) and re-derived by :meth:`bind`.
    The ``("RiparianBuffer", "volume")`` check (``:589``, ``:938-949``) belongs to 3c.
    """

    wsimod_type: ClassVar[str] = "River"
    step_methods: ClassVar[tuple[str, ...]] = ("distribute", "calculate_discharge")
    distributes: ClassVar[tuple[str, ...]] = ("out", "in")
    parameter_defaults: ClassVar[dict[str, float]] = {
        "length": 200,
        "width": 20,
        "velocity": 17280.0,  # 0.2 * constants.M_S_TO_M_DT
        "damp": 0.1,
        "mrf": 0,
    }

    def _make_tank(self, values: Mapping) -> Tank:
        """``storage.py:524-540``: ``area`` and ``capacity`` are derived, never configured."""
        return super()._make_tank({**values, **river_derived(self.spec)})

    def _set_handlers(self) -> None:
        """``storage.py:575-589``."""
        super()._set_handlers()
        self.push_set_handler["default"] = self.push_set_river
        self.push_check_handler["default"] = self.push_check_accept
        self.pull_check_handler["default"] = self.pull_check_river
        self.pull_set_handler["default"] = self.pull_set_river
        self.pull_check_handler[("RiparianBuffer", "volume")] = self.pull_check_fp

    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """The day's parameters; ``area = length * width`` and the unbounded capacity are
        derived from them again (``storage.py:524-533``)."""
        super().bind(parameters)
        where = f"node {self.name!r}"
        if bool((self._param("velocity") <= 0).any()):
            raise ValueError(f"{where}: velocity is not positive (storage.py:907 divides)")
        if bool((self._param("damp") < 0).any()):
            raise ValueError(f"{where}: damp is negative (exp(-1 / kt) overflows)")
        area = self._param("length") * self._param("width")
        self.tank.area = area
        self.tank.capacity = torch.full_like(area, UNBOUNDED_CAPACITY)

    def get_riverrc(self) -> Tensor:
        """``storage.py:900-915``: ``kt = damp * (length / velocity)``, then :func:`riverrc`."""
        total_time = self._param("length") / self._param("velocity")
        return riverrc(self._param("damp") * total_time)

    # --- the handlers -----------------------------------------------------------------------
    def push_set_river(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``storage.py:711-721``: the tank takes everything, by force."""
        self.tank.push_storage(vqip, force=True, active=active)
        return self.ctx.empty()

    def pull_check_river(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``storage.py:660-686``: the tank's VQIP with its volume raised by what the upstream
        River and Node arcs report, cut to ``max(volume - mrf / riverrc, 0)`` and to the request.

        The rescale (``v_change_vqip``) takes the **raised** volume as its base, so the tank's
        pollutant mass is spread over the upstream water too (spec, River).
        """
        ctx = self.ctx
        call = ctx.call(self.site, "pull_check_river")
        avail = self.tank.get_avail(active=active)
        upstream = self.get_connected("pull", RIVER_UPSTREAM_TYPES, active=active)
        raised = avail[..., 0] + upstream.avail
        avail = torch.cat([raised[..., None], avail[..., 1:]], dim=-1)
        mrf = self._param("mrf") / self.get_riverrc()
        volume = dmax(
            ctx,
            f"{self.site}/pull_check_river/mrf",
            raised - mrf,
            0.0,
            active=active,
            call=call,
        )
        if vqip is not None:
            volume = dmin(
                ctx,
                f"{self.site}/pull_check_river/clip",
                volume,
                vqip[..., 0],
                active=active,
                call=call,
            )
        return v_change_vqip(self.p, avail, volume)

    def pull_set_river(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``storage.py:688-709``: what the check allows, first from the tank, the rest from
        the upstream River and Node arcs."""
        avail = self.pull_check_river(vqip, active=active)
        pulled = self.tank.pull_storage(avail, active=active)
        to_pull = self._volume(avail[..., 0] - pulled[..., 0])
        pulled_ = self.pull_distributed(to_pull, of_type=RIVER_UPSTREAM_TYPES, active=active)
        return sum_vqip(self.p, pulled, pulled_)

    def pull_check_fp(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """``storage.py:938-949`` reads a ``river_tank`` that does not exist: the riparian
        buffer is sub-project 3c and is refused by name."""
        raise KeyError(
            f"node {self.name!r}: the ('RiparianBuffer', 'volume') pull check is sub-project 3c "
            "(storage.py:938-949 reads a nonexistent river_tank)"
        )

    # --- orchestration ----------------------------------------------------------------------
    def calculate_discharge(self, *, active: Tensor) -> None:
        """``storage.py:917-924``: runs biochemistry only with ``"nitrate"`` among the
        pollutants, which :meth:`validate` refuses (3b); nothing otherwise."""

    def distribute(self, *, active: Tensor) -> None:
        """``storage.py:926-936``: ``V * riverrc`` to River, Node and Waste, the rest forced
        back (``"river_cant_push"`` above ``FLOAT_ACCURACY``)."""
        volume = self.tank.storage[..., 0] * self.get_riverrc()
        outflow = self.tank.pull_storage(self._volume(volume), active=active)
        retained = self.push_distributed(outflow, of_type=RIVER_DOWNSTREAM_TYPES, active=active)
        self._force_back(retained, "river_cant_push", active=active)

    # --- validation -------------------------------------------------------------------------
    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """Refused: ``"nitrate"`` among the pollutants of a run that carries them (WSIMOD would
        run ``biochemical_processes``, ``storage.py:917-924``: 3b; a quantity-only run reads no
        pollutant, spec W10); ``velocity <= 0`` (``storage.py:907`` divides); ``damp < 0``
        (``exp(-1 / kt)`` overflows for a small negative ``kt``); a routing coefficient that is
        not positive (``:683`` divides by it)."""
        ignored = super().validate(spec, network, pollutants=pollutants)
        where = f"node {spec.name!r}"
        if pollutants and "nitrate" in network.pollutants.names:
            raise ValueError(
                f"{where}: a River with 'nitrate' among the pollutants runs biochemistry "
                "(storage.py:917-924), which is sub-project 3b; declare pollutant lists "
                "without nitrate; run build_model(network, pollutants=False) until sub-project 3b"
            )
        values = {**cls.parameter_defaults, **spec.parameters}
        if values["velocity"] <= 0:
            raise ValueError(f"{where}: velocity {values['velocity']} is not positive")
        if values["damp"] < 0:
            raise ValueError(f"{where}: damp {values['damp']} is negative")
        kt = values["damp"] * (values["length"] / values["velocity"])
        rc = 1 - kt + kt * math.exp(-1 / kt) if kt != 0 else 1
        if not rc > 0:
            raise ValueError(f"{where}: the routing coefficient {rc} is not positive")
        return ignored

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        """``pull_check_river`` asks the upstream arcs (``get_connected``, ``storage.py:672``);
        the push check accepts without asking."""
        return direction == "pull" and tag == "default"


NODE_CLASSES.update(
    {
        "Storage": Storage,
        "Groundwater": Groundwater,
        "QueueGroundwater": QueueGroundwater,
        "Reservoir": Reservoir,
        "RiverReservoir": RiverReservoir,
        "River": River,
    }
)
