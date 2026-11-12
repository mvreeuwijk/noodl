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
import sys
import warnings
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from noodl.apps.wsimod.arcs import _put_row, check_travel_time
from noodl.apps.wsimod.constants import MG_L_TO_KG_M3, UNBOUNDED_CAPACITY
from noodl.apps.wsimod.decay import Decay, DecaySpec, check_inert, read_decays
from noodl.apps.wsimod.engine import Ctx, dmax, dmin
from noodl.apps.wsimod.node import NODE_CLASSES, Node, Tag
from noodl.apps.wsimod.tanks import DecayQueueTank, DecayTank, QueueTank, Tank, bind_timearea
from noodl.apps.wsimod.vqip import (
    Pollutants,
    empty,
    extract_vqip,
    pow_python,
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
    "RIVER_BIOCHEMISTRY_DEFAULTS",
    "River",
    "RiverReservoir",
    "Storage",
    "exp_python",
    "pow_half",
    "river_derived",
    "riverrc",
    "window_sum",
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


# storage.py:539-558: the biochemistry parameters, WSIMOD's Python expressions as written
RIVER_BIOCHEMISTRY_DEFAULTS: dict[str, float] = {
    "uptake_PNratio": 1 / 7.2,  # :539
    "denpar_w": 0.0015,  # :541
    "T_wdays": 5,  # :543
    "halfsatINwater": 1.5 * MG_L_TO_KG_M3,  # :545-547
    "hsatTP": 0.05 * MG_L_TO_KG_M3,  # :551
    "limpppar": 0.1 * MG_L_TO_KG_M3,  # :552
    "prodNpar": 0.001,  # :553
    "prodPpar": 0.0001,  # :554-556
    "muptNpar": 0.001,  # :557
    "muptPpar": 0.0001,  # :558
}
# storage.py:560, :563: the register lengths less one (structural, set by override)
RIVER_LAG_DEFAULTS: dict[str, int] = {"max_temp_lag": 20, "max_phosphorus_lag": 365}
# the pollutants biochemical_processes reads (storage.py:738-898); din_components (:566)
RIVER_POLLUTANTS = (
    "temperature",
    "phosphate",
    "org-phosphorus",
    "ammonia",
    "nitrate",
    "org-nitrogen",
)
DIN_COMPONENTS = ("ammonia", "nitrate")


def _din(columns: Mapping[str, Tensor]) -> Tensor:
    """``sum([S[x] for x in din_components])`` (``storage.py:733-735``): Python's ``sum`` from 0,
    ammonia then nitrate. ``0 + x`` and ``+0.0 + x`` agree for every ``x``, ``-0.0`` included."""
    total = 0.0 + columns["ammonia"]
    return total + columns["nitrate"]


def _shift(register: Tensor, value: Tensor) -> Tensor:
    """The register one slot to the left with ``value`` in the last slot: WSIMOD's ``del
    lagged[0]`` (once the list is full) and ``append`` (``storage.py:753-763``, X17)."""
    batch = torch.broadcast_shapes(register.shape[:-1], value.shape)
    register = torch.broadcast_to(register, (*batch, register.shape[-1]))
    value = torch.broadcast_to(value, batch)
    return torch.cat([register[..., 1:], value[..., None]], dim=-1)


# CPython 3.12 sums floats with Neumaier's compensation (bltinmodule.c, builtin_sum_impl;
# gh-100425); 3.11 adds them one at a time. WSIMOD's means are whichever the interpreter does.
COMPENSATED_SUM = sys.version_info >= (3, 12)


def window_sum(register: Tensor, length: int) -> Tensor:
    """CPython's ``sum`` of the register's last ``length`` slots, oldest first, from ``+0.0``
    (``storage.py:806-808``, X31), elementwise over the batch.

    From Python 3.12 (:data:`COMPENSATED_SUM`) ``sum`` is Neumaier's compensated sum: for each
    item ``t = s + x`` and ``c += (s - t) + x`` where ``|s| >= |x|``, else ``c += (x - t) + s``;
    at the end ``s + c`` where ``c`` is non-zero and finite. Before 3.12 it adds one item at a
    time. CPython starts from the integer 0, and ``0 + x`` is ``+0.0 + x`` for every ``x``,
    ``-0.0`` included; a padded ``+0.0`` slot leaves ``s`` and ``c`` at ``+0.0``. The sum is
    therefore Python's over WSIMOD's list, bitwise. The branch on ``|s| >= |x|`` is detached;
    the compensation's gradient is 0, so the gradient is the plain sum's."""
    total = torch.zeros(register.shape[:-1], dtype=torch.float64)
    size = register.shape[-1]
    compensation = torch.zeros_like(total)
    for i in range(size - length, size):
        x = register[..., i]
        if not COMPENSATED_SUM:
            total = total + x
            continue
        t = total + x
        larger = (total.abs() >= x.abs()).detach()
        compensation = compensation + torch.where(larger, (total - t) + x, (x - t) + total)
        total = t
    if not COMPENSATED_SUM:
        return total
    keep = ((compensation != 0) & torch.isfinite(compensation)).detach()
    return torch.where(keep, total + compensation, total)


class River(Storage):
    """``River`` (``storage.py:452-955``): a tank that is never full, discharged by the routing
    coefficient :func:`riverrc` and pulled from with a minimum residual flow, with WSIMOD's
    in-river biochemistry (:meth:`biochemical_processes`, ``:738-898``) when ``"nitrate"`` is a
    pollutant (``:917-924``).

    A ``decays`` dict builds a ``DecayTank`` as for any :class:`Storage`: the tank decays at the
    end of the day, from the river's own ``temperature`` forcing (``storage.py:53-59``). The
    tank's ``area`` and ``capacity`` are derived (:func:`river_derived`) and re-derived by
    :meth:`bind`. The ``("RiparianBuffer", "volume")`` check (``:589``, ``:938-949``) belongs to
    3c.

    With biochemistry (D9: ``"nitrate"`` among the run's pollutants, so never in a
    quantity-only run) the river has:

    - the ten parameters of :data:`RIVER_BIOCHEMISTRY_DEFAULTS` as drivers
      ``"wsimod.River.<p>"``; WSIMOD sets them after ``Storage.__init__`` (``:539-558``), so they
      reach a configuration only through overrides (``:620-644``);
    - ``lag_sizes = (max_temp_lag + 1, max_phosphorus_lag + 1)``, structural, and the carried
      registers ``lagged_temperatures`` and ``lagged_total_phosphorus`` of those sizes, slot 0
      the oldest, zero-padded at the front (X17): after ``n`` calls their last ``min(n, size)``
      slots are WSIMOD's lists, and every window sum is Python's (:func:`window_sum`, X31);
    - ``bio_in`` and ``bio_out``, the last call's gains and losses, **assigned** at every call
      (``:923-924``) and emptied at the end of the day (``:951-955``); per-day stores. The
      mass-balance terms ``"River.bio_in"`` and ``"River.bio_out"`` **add** every call's (X16).
    """

    wsimod_type: ClassVar[str] = "River"
    step_methods: ClassVar[tuple[str, ...]] = ("distribute", "calculate_discharge")
    distributes: ClassVar[tuple[str, ...]] = ("out", "in")
    # storage.py:595-599: bio_in and bio_out are the node's in and out lists
    mass_balance: ClassVar[dict[str, str]] = {
        **Storage.mass_balance,
        "River.bio_in": "in",
        "River.bio_out": "out",
    }
    parameter_defaults: ClassVar[dict[str, float]] = {
        "length": 200,
        "width": 20,
        "velocity": 17280.0,  # 0.2 * constants.M_S_TO_M_DT
        "damp": 0.1,
        "mrf": 0,
        **RIVER_BIOCHEMISTRY_DEFAULTS,
    }

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        # storage.py:919: the biochemistry runs only with "nitrate" among the pollutants
        self.biochemistry = "nitrate" in pollutants.names
        super().__init__(spec, pollutants)
        if not self.biochemistry:  # D9: no biochemistry, no drivers, no registers
            for name in RIVER_BIOCHEMISTRY_DEFAULTS:
                self.defaults.pop(name)
            return
        values = spec.parameters
        self.max_temp_lag = int(values.get("max_temp_lag", RIVER_LAG_DEFAULTS["max_temp_lag"]))
        self.max_phosphorus_lag = int(
            values.get("max_phosphorus_lag", RIVER_LAG_DEFAULTS["max_phosphorus_lag"])
        )
        self.lag_sizes = (self.max_temp_lag + 1, self.max_phosphorus_lag + 1)
        # storage.py:561, :564: empty lists, which zero registers are (X17)
        self.lagged_temperatures = torch.zeros(self.lag_sizes[0], dtype=torch.float64)
        self.lagged_total_phosphorus = torch.zeros(self.lag_sizes[1], dtype=torch.float64)
        self.bio_in = empty(pollutants, ())  # storage.py:595-596
        self.bio_out = empty(pollutants, ())
        self.current_depth = torch.zeros((), dtype=torch.float64)  # storage.py:571
        registers = ("lagged_temperatures", "lagged_total_phosphorus")
        self.store_paths = (*self.store_paths, *registers, "bio_in", "bio_out")
        self.carried_paths = (*self.carried_paths, *registers)  # D8

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
        derived from them again (``storage.py:524-533``). With biochemistry, a batched
        ``T_wdays`` of 0 (``:748`` divides by it), ``halfsatINwater <= 0`` (``:781-783``
        divides by ``c + halfsatINwater``, 0 where ``din`` is) and an ``area`` of 0
        (``update_depth`` divides by it, ``:725``) are refused, as :meth:`validate` refuses the
        configured ones."""
        super().bind(parameters)
        where = f"node {self.name!r}"
        if bool((self._param("velocity") <= 0).any()):
            raise ValueError(f"{where}: velocity is not positive (storage.py:907 divides)")
        if bool((self._param("damp") < 0).any()):
            raise ValueError(f"{where}: damp is negative (exp(-1 / kt) overflows)")
        area = self._param("length") * self._param("width")
        self.tank.area = area
        self.tank.capacity = torch.full_like(area, UNBOUNDED_CAPACITY)
        if not self.biochemistry:
            return
        if bool((self._param("T_wdays").detach() == 0).any()):
            raise ValueError(f"{where}: T_wdays is 0 (storage.py:748 divides by it)")
        if bool((self._param("halfsatINwater").detach() <= 0).any()):
            raise ValueError(
                f"{where}: halfsatINwater is not positive (storage.py:781-783 divides by "
                "din / V + halfsatINwater, which is 0 where din is 0)"
            )
        if bool((area.detach() == 0).any()):
            raise ValueError(
                f"{where}: area = length * width is 0 (update_depth divides by it, storage.py:725)"
            )

    def configured_parameters(self, date: datetime) -> dict[str, Tensor]:
        """The day's ``temperature`` forcing of a river with a ``decays`` dict or with
        biochemistry (``get_data_input``, ``storage.py:750``)."""
        out = super().configured_parameters(date)
        if self.biochemistry:
            out["temperature"] = _f64(self.spec.data[("temperature", date)])
        return out

    def get_riverrc(self) -> Tensor:
        """``storage.py:900-915``: ``kt = damp * (length / velocity)``, then :func:`riverrc`."""
        total_time = self._param("length") / self._param("velocity")
        return riverrc(self._param("damp") * total_time)

    # --- state ------------------------------------------------------------------------------
    def stores(self) -> dict[str, Tensor]:
        out = super().stores()
        if self.biochemistry:
            out.update(
                lagged_temperatures=self.lagged_temperatures,
                lagged_total_phosphorus=self.lagged_total_phosphorus,
                bio_in=self.bio_in,
                bio_out=self.bio_out,
            )
        return out

    def end_timestep(self) -> None:
        """``end_timestep_`` (``storage.py:951-955``): the tank's, then ``bio_in`` and
        ``bio_out`` emptied. The registers are written by the calls, not here."""
        super().end_timestep()
        if self.biochemistry:
            self.bio_in = empty(self.p, ())
            self.bio_out = empty(self.p, ())

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

    # --- biochemistry (storage.py:723-898) ----------------------------------------------------
    def update_depth(self) -> Tensor:
        """``storage.py:723-725``: ``current_depth = V / area``."""
        self.current_depth = self.tank.storage[..., 0] / (
            self._param("length") * self._param("width")
        )
        return self.current_depth

    def get_din_pool(self) -> Tensor:
        """``storage.py:727-736``: ammonia plus nitrate, Python's ``sum`` from 0 (nitrite is
        not included)."""
        storage = self.tank.storage
        return _din({n: storage[..., self.p.column(n)] for n in DIN_COMPONENTS})

    def lagged_means(self) -> tuple[Tensor, Tensor, Tensor]:
        """``storage.py:806-808``: ``sum(lagged_temperatures[-10:]) / 10``, ``[-20:]) / 20`` and
        ``sum(lagged_total_phosphorus) / max_phosphorus_lag``, over the registers (X31).

        WSIMOD's quirks are kept (X14): each window divides by its full length even before the
        list has filled (a cold start), and the phosphorus "mean" adds ``max_phosphorus_lag +
        1`` entries (366) and divides by ``max_phosphorus_lag`` (365). A register shorter than
        a window is summed whole, as WSIMOD's ``[-10:]`` of a shorter list is."""
        temperatures, phosphorus = self.lagged_temperatures, self.lagged_total_phosphorus
        size = temperatures.shape[-1]
        temp_10_day = window_sum(temperatures, min(10, size)) / 10
        temp_20_day = window_sum(temperatures, min(20, size)) / 20
        total = window_sum(phosphorus, phosphorus.shape[-1]) / self.max_phosphorus_lag
        return temp_10_day, temp_20_day, total

    def biochemical_processes(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``storage.py:738-898``: the day's in-river processes on ``tank.storage``; returns
        ``(in_, out_)``, the VQIPs of what they add and remove.

        In WSIMOD's order (spec, River biochemistry, steps 1-8):

        1. ``current_depth = V / area`` (:meth:`update_depth`);
        2. the temperature relaxes towards the river's ``temperature`` forcing,
           ``(1 - 1/T_wdays) S_T + (1/T_wdays) T_air``, on every active instance, dry or
           frozen;
        3. both registers shift in ``T`` and ``phosphate + org-phosphorus``, before the guards,
           so every call appends;
        4. the guards ``dry`` (``V < FLOAT_ACCURACY``) and ``frozen`` (``T <= 0``) return empty
           VQIPs: steps 5-8 run where ``active & ~dry & ~frozen``;
        5. denitrification, ``min(denpar_w * area * 2^((T-20)/10) [* T/5 below 5] * c / (c +
           halfsatINwater), 0.5 din)``, taken from ammonia and nitrate pro rata;
        6. the means (:meth:`lagged_means`), ``tempfcn = T/20 * (T10 - T20)/5`` and
           ``totalphosfcn``, which compares the phosphorus **mass** with concentrations (X14);
        7. production (``minprodN > 0``: DIN and phosphate to the organic pools, each capped at
           half its pool) or mineralisation (the reverse, capped at half the organic pools).
           With ``din == 0`` the mineralised nitrogen leaves ``org-nitrogen`` and is added
           nowhere; the loss is in ``out_`` and the event ``"river_nitrogen_lost"`` holds
           (X15). ``minprodN == -0.0`` mineralises and moves nothing;
        8. macrophyte uptake of DIN and phosphate, ``max(0, (T/20)^0.3 (T - T20)/5)``, capped at
           half the pools.

        Every branch is a decision ``node:N/biochemical_processes/<branch>`` (``engine``'s
        vocabulary) with ``call`` the index of the day's call, recorded under the mask it is
        evaluated under. The four 50 % caps keep their operand orders: ``min(denitri_water,
        0.5 din)``, and ``min(0.5 pool, x)`` at the others. The powers are Python's
        (:func:`~noodl.apps.wsimod.vqip.pow_python`). Masked-out instances compute on safe
        denominators and bases, so their values and gradients are finite and discarded.
        """
        p, ctx = self.p, self.ctx
        active = torch.as_tensor(active, dtype=torch.bool)
        call = ctx.call(self.site, "biochemical_processes")
        site = f"{self.site}/biochemical_processes"

        def decide(branch: str, lhs, op: str, rhs, mask: Tensor) -> Tensor:
            return ctx.decide(f"{site}/{branch}", lhs, op, rhs, active=mask, call=call)

        def low(branch: str, a, b, mask: Tensor) -> Tensor:
            return dmin(ctx, f"{site}/{branch}", a, b, active=mask, call=call)

        def high(branch: str, a, b, mask: Tensor) -> Tensor:
            return dmax(ctx, f"{site}/{branch}", a, b, active=mask, call=call)

        def safe(value: Tensor, mask: Tensor) -> Tensor:
            return torch.where(mask, value, torch.ones_like(value))

        storage = self.tank.storage
        s = {n: storage[..., p.column(n)] for n in RIVER_POLLUTANTS}
        volume = storage[..., 0]
        area = self._param("length") * self._param("width")
        zero = ctx.zeros()
        # 1. storage.py:746
        current_depth = self.update_depth()
        # 2. :748-750
        t_wdays = self._param("T_wdays")
        T = (1 - 1 / t_wdays) * s["temperature"] + (1 / t_wdays) * self.params["temperature"]
        # 3. :752-763
        self.lagged_temperatures = write(
            active, _shift(self.lagged_temperatures, T), self.lagged_temperatures
        )
        total_phosphorus = s["phosphate"] + s["org-phosphorus"]
        self.lagged_total_phosphorus = write(
            active,
            _shift(self.lagged_total_phosphorus, total_phosphorus),
            self.lagged_total_phosphorus,
        )
        # 4. :766-772
        dry = decide("dry", volume, "<", ctx.fa, active)
        wet = active & ~dry
        frozen = decide("frozen", T, "<=", 0.0, wet)
        run = wet & ~frozen
        # 5. :775-801
        tempfcn = pow_python(_f64(2.0), (T - 20) / 10)
        cold = decide("cold", T, "<", 5.0, run)
        tempfcn = torch.where(cold, tempfcn * (T / 5), tempfcn)
        din = _din(s)
        din_concentration = din / safe(volume, run)
        confcn = din_concentration / safe(din_concentration + self._param("halfsatINwater"), run)
        denitri_water = self._param("denpar_w") * area * tempfcn * confcn
        river_denitrification = low("denitrification_cap", denitri_water, 0.5 * din, run)
        din_ = din - river_denitrification
        in_ = dict.fromkeys(RIVER_POLLUTANTS, zero)
        out_ = dict.fromkeys(RIVER_POLLUTANTS, zero)
        positive = decide("din:denitrification", din, ">", 0.0, run)
        for pol in DIN_COMPONENTS:
            loss = (din - din_) / safe(din, positive) * s[pol]
            out_[pol] = torch.where(positive, out_[pol] + loss, out_[pol])
            s[pol] = torch.where(positive, s[pol] - loss, s[pol])
        din = _din(s)
        # 6. :803-819
        temp_10_day, temp_20_day, total_phos_365_day = self.lagged_means()
        tempfcn = T / 20 * (temp_10_day - temp_20_day) / 5
        excess = total_phos_365_day - self._param("limpppar")
        den = excess + self._param("hsatTP")
        saturating = decide("totalphosfcn", den, ">", 0.0, run)
        totalphosfcn = torch.where(saturating, excess / safe(den, saturating), zero)
        # 7. :822-873
        minprodN = self._param("prodNpar") * totalphosfcn * tempfcn * area * current_depth
        minprodP = (
            self._param("prodPpar")
            * totalphosfcn
            * tempfcn
            * area
            * current_depth
            * self._param("uptake_PNratio")
        )
        production = decide("production", minprodN, ">", 0.0, run)
        producing, mineralising = run & production, run & ~production
        # production (inorganic -> organic), :833-854
        produced_N = low("production_cap:N", 0.5 * din, minprodN, producing)
        produced_P = low("production_cap:P", 0.5 * s["phosphate"], minprodP, producing)
        sp, ip, op = dict(s), dict(in_), dict(out_)
        op["phosphate"] = produced_P
        sp["phosphate"] = s["phosphate"] - produced_P
        ip["org-phosphorus"] = produced_P
        sp["org-phosphorus"] = s["org-phosphorus"] + produced_P
        positive = decide("din:production", din, ">", 0.0, producing)
        for pol in DIN_COMPONENTS:
            loss = produced_N * sp[pol] / safe(din, positive)
            op[pol] = torch.where(positive, op[pol] + loss, op[pol])
            sp[pol] = torch.where(positive, sp[pol] - loss, sp[pol])
        ip["org-nitrogen"] = produced_N
        sp["org-nitrogen"] = s["org-nitrogen"] + produced_N
        # mineralisation (organic -> inorganic), :856-873
        mineral_N = low("mineralisation_cap:N", 0.5 * s["org-nitrogen"], -minprodN, mineralising)
        mineral_P = low("mineralisation_cap:P", 0.5 * s["org-phosphorus"], -minprodP, mineralising)
        sm, im, om = dict(s), dict(in_), dict(out_)
        im["phosphate"] = mineral_P
        sm["phosphate"] = s["phosphate"] + mineral_P
        om["org-phosphorus"] = mineral_P
        sm["org-phosphorus"] = s["org-phosphorus"] - mineral_P
        positive = decide("din:mineralisation", din, ">", 0.0, mineralising)
        for pol in DIN_COMPONENTS:
            gain = mineral_N * sm[pol] / safe(din, positive)
            im[pol] = torch.where(positive, im[pol] + gain, im[pol])
            sm[pol] = torch.where(positive, sm[pol] + gain, sm[pol])
        om["org-nitrogen"] = mineral_N
        sm["org-nitrogen"] = s["org-nitrogen"] - mineral_N
        # X15: without DIN the mineralised nitrogen is destroyed (:866-873)
        lost = mineralising & ~positive & (mineral_N.detach() > 0)
        ctx.event("river_nitrogen_lost", self.name, lost)
        for name in RIVER_POLLUTANTS:
            s[name] = torch.where(production, sp[name], sm[name])
            in_[name] = torch.where(production, ip[name], im[name])
            out_[name] = torch.where(production, op[name], om[name])
        din = _din(s)
        # 8. :875-895
        positive_T = high("macrophyte_temperature", zero, T, run)
        tempfcn1 = pow_python(positive_T / 20, _f64(0.3))
        tempfcn2 = (T - temp_20_day) / 5
        tempfcn = high("macrophyte_factor", zero, tempfcn1 * tempfcn2, run)
        macrouptN = self._param("muptNpar") * tempfcn * area
        macrophyte_uptake_N = low("macrophyte_cap:N", 0.5 * din, macrouptN, run)
        positive = decide("din:macrophyte", din, ">", 0.0, run)
        for pol in DIN_COMPONENTS:
            loss = macrophyte_uptake_N * s[pol] / safe(din, positive)
            out_[pol] = torch.where(positive, out_[pol] + loss, out_[pol])
            s[pol] = torch.where(positive, s[pol] - loss, s[pol])
        positive_P = high("macrophyte_phosphorus", zero, totalphosfcn, run)
        macrouptP = self._param("muptPpar") * tempfcn * positive_P * area
        macrophyte_uptake_P = low("macrophyte_cap:P", 0.5 * s["phosphate"], macrouptP, run)
        out_["phosphate"] = out_["phosphate"] + macrophyte_uptake_P
        s["phosphate"] = s["phosphate"] - macrophyte_uptake_P
        # the storage: temperature where active (steps 2-3 run before the guards), the rest
        # where the processes ran; in_ and out_ are empty elsewhere
        columns = [storage[..., i] for i in range(1 + p.K)]
        gains, losses = [zero] * (1 + p.K), [zero] * (1 + p.K)
        for name in RIVER_POLLUTANTS:
            c = p.column(name)
            if name == "temperature":
                columns[c] = torch.where(active, T, columns[c])
                continue
            columns[c] = torch.where(run, s[name], columns[c])
            gains[c] = torch.where(run, in_[name], zero)
            losses[c] = torch.where(run, out_[name], zero)
        self.tank.storage = _stack(columns)
        return _stack(gains), _stack(losses)

    # --- orchestration ----------------------------------------------------------------------
    def calculate_discharge(self, *, active: Tensor) -> None:
        """``storage.py:917-924``: with ``"nitrate"`` among the pollutants, run
        :meth:`biochemical_processes` and **assign** ``bio_in``/``bio_out``; the terms
        ``"River.bio_in"`` and ``"River.bio_out"`` add every call's (X16). Nothing otherwise."""
        if not self.biochemistry:
            return
        active = torch.as_tensor(active, dtype=torch.bool)
        in_, out_ = self.biochemical_processes(active=active)
        self.bio_in = write(active, in_, self.bio_in)
        self.bio_out = write(active, out_, self.bio_out)
        self.ctx.term("River.bio_in", in_, active)
        self.ctx.term("River.bio_out", out_, active)

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
        """Refused: ``velocity <= 0`` (``storage.py:907`` divides); ``damp < 0`` (``exp(-1 /
        kt)`` overflows for a small negative ``kt``); a routing coefficient that is not
        positive (``:683`` divides by it).

        With ``"nitrate"`` among the configured pollutants WSIMOD runs the biochemistry
        (``:917-924``), in either mode, so these are refused too (X19):

        - any of the six pollutants it reads missing (WSIMOD raises ``KeyError``);
        - no ``("temperature", date)`` forcing on some date (``:750``);
        - ``T_wdays == 0`` (``:748`` divides), ``area = length * width == 0`` (``:725``
          divides), ``halfsatINwater <= 0`` (``:781-783`` divides by 0 where ``din == 0``);
        - ``max_temp_lag`` not a non-negative ``int`` (a negative one makes ``del`` on an
          empty list raise ``IndexError``) and ``max_phosphorus_lag`` not a positive ``int``
          (a divisor); a float is refused although WSIMOD runs some (X37).

        A quantity-only run ignores the configured biochemistry parameters (D9).
        """
        ignored = super().validate(spec, network, pollutants=pollutants)
        where = f"node {spec.name!r}"
        values = {**cls.parameter_defaults, **RIVER_LAG_DEFAULTS, **spec.parameters}
        if values["velocity"] <= 0:
            raise ValueError(f"{where}: velocity {values['velocity']} is not positive")
        if values["damp"] < 0:
            raise ValueError(f"{where}: damp {values['damp']} is negative")
        kt = values["damp"] * (values["length"] / values["velocity"])
        rc = 1 - kt + kt * math.exp(-1 / kt) if kt != 0 else 1
        if not rc > 0:
            raise ValueError(f"{where}: the routing coefficient {rc} is not positive")
        names = network.pollutants.names
        if "nitrate" not in names:
            return ignored
        bio = f"{where}: a River with 'nitrate' among the pollutants runs biochemistry"
        for name in RIVER_POLLUTANTS:
            if name not in names:
                raise ValueError(
                    f"{bio} (storage.py:738-898), which reads {name!r}; it is not a pollutant "
                    "(WSIMOD raises KeyError)"
                )
        for date, label in zip(network.dates, network.date_labels, strict=True):
            if ("temperature", date) not in spec.data:
                raise ValueError(
                    f"{bio}, which reads its 'temperature' forcing on every date "
                    f"(storage.py:750); there is none on {label}"
                )
        if values["T_wdays"] == 0:
            raise ValueError(f"{where}: T_wdays is 0 (storage.py:748 divides by it)")
        if values["length"] * values["width"] == 0:
            raise ValueError(
                f"{where}: area = length * width is 0 (update_depth divides by it, storage.py:725)"
            )
        if not values["halfsatINwater"] > 0:
            raise ValueError(
                f"{where}: halfsatINwater {values['halfsatINwater']!r} is not positive "
                "(storage.py:781-783 divides by din / V + halfsatINwater, 0 where din is 0)"
            )
        for name, lowest in (("max_temp_lag", 0), ("max_phosphorus_lag", 1)):
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < lowest:
                kind = "non-negative" if lowest == 0 else "positive"
                raise ValueError(
                    f"{where}: {name} {value!r} is not a {kind} integer (it sizes the "
                    "register" + (" and divides the phosphorus sum" if lowest else "") + "; X37)"
                )
        if not pollutants:  # D9: no biochemistry without pollutants
            configured = [*RIVER_BIOCHEMISTRY_DEFAULTS, *RIVER_LAG_DEFAULTS]
            ignored += [name for name in configured if name in spec.parameters]
        return ignored

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        """``pull_check_river`` asks the upstream arcs (``get_connected``, ``storage.py:672``);
        the push check accepts without asking."""
        return direction == "pull" and tag == "default"


def _stack(columns: list[Tensor]) -> Tensor:
    """The columns of a VQIP, broadcast to one batch shape, stacked on the last axis."""
    shape = torch.broadcast_shapes(*(c.shape for c in columns))
    return torch.stack([torch.broadcast_to(c, shape) for c in columns], dim=-1)


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
