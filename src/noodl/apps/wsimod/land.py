"""WSIMOD 0.8.1's ``Land`` and its surfaces (``wsimod/nodes/land.py:17-1079``), batched.

A ``Land`` node owns its surfaces (``Surface``, ``ImperviousSurface``, ``PerviousSurface``) and
three residence tanks, ``surface_runoff``, ``subsurface_runoff`` and ``percolation``
(``land.py:138-149``). ``run`` (``:200-244``) runs every surface in order, then sends the
percolation to ``Groundwater`` and the surface and subsurface runoff to ``River`` and ``Node``.

A surface is a :class:`~noodl.apps.wsimod.tanks.DecayTank` with empty ``decays`` whose
``run`` (``:425-449``) walks its ``inflows``, ``processes`` and ``outflows``, adding what each
returns to the land's ``running_inflow_mb`` and ``running_outflow_mb`` (the mass-balance terms
``"Land.running_inflow_mb"`` and ``"Land.running_outflow_mb"``, ``:158-162``; D11). In 3a these
lists hold the water functions (``precipitation_evaporation``/``push_to_sewers`` of an
impervious surface, ``ihacres``/``calculate_soil_temperature``/``route`` of a pervious one) and
``simple_deposition`` when a ``pollutant_load`` is configured, which :meth:`Land.validate`
admits only with zero loads (3c). The surface deposition of ``nhx-dry``-style data is 3c and
refused with pollutants on (D21); a quantity-only run ignores it, as its volumes do not depend
on it.

The arithmetic is WSIMOD's ``+ - * /``, ``min``/``max`` in WSIMOD's order, and IHACRES's
``exp`` and ``**`` evaluated per instance by Python's own ``math.exp`` and ``float.__pow__``
(:func:`~noodl.apps.wsimod.storage.exp_python`, :func:`~noodl.apps.wsimod.vqip.pow_python`), so
every value is bitwise WSIMOD's (``replay_bound = 0``). Every method takes the caller's
``active`` mask and writes state only where it holds (D3); data branches are
:meth:`Ctx.decide` (D12).

The surfaces' sites (engine.py, Decision records) are their owner's with the attribute path
``surfaces.<name>``: ``node:N/surfaces.<name>/precipitation_evaporation/dry``,
``node:N/surfaces.<name>/ihacres/{infiltration,excess,evaporation_kink,outflow_kink,moisture,
through_soil}``, and the tank sites of :class:`~noodl.apps.wsimod.tanks.Tank`. ``run`` adds
``node:N/run/{percolation_reply,runoff,rejected,rejected_surface,rejected_subsurface}``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor

from noodl.apps.wsimod.constants import UNBOUNDED_CAPACITY
from noodl.apps.wsimod.decay import Decay, check_inert, read_decays
from noodl.apps.wsimod.engine import Ctx, dmax, dmin
from noodl.apps.wsimod.node import NODE_CLASSES, Node, Tag
from noodl.apps.wsimod.storage import exp_python
from noodl.apps.wsimod.tanks import DecayTank, ResidenceTank, Tank
from noodl.apps.wsimod.vqip import (
    Pollutants,
    empty,
    pow_python,
    safe_den,
    sum_vqip,
    v_change_vqip,
    write,
)

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.config import NodeSpec, SurfaceSpec, WsimodNetwork

__all__ = [
    "ImperviousSurface",
    "Land",
    "PerviousSurface",
    "Surface",
    "surface_derived",
]

# land.py:208: where percolation goes; :221: where runoff goes; :711: where ponding goes
PERCOLATION_TYPES = ("Groundwater",)
RUNOFF_TYPES = ("River", "Node")
SEWER_TYPES = ("Sewer",)
GARDEN_TAG = ("Demand", "Garden")
# land.py:375-380, :496-544: the surface data the deposition functions read (3c)
DEPOSITION_KEYS = frozenset({"nhx-dry", "noy-dry", "srp-dry", "nhx-wet", "noy-wet", "srp-wet"})
# land.py:753-756: state attributes, set only through the overrides
SOIL_TEMPERATURE_DEFAULTS = {
    "soil_temp_w_prev": 0.1,
    "soil_temp_w_air": 0.6,
    "soil_temp_w_deep": 0.1,
    "soil_temp_deep": 10,
}
LAND_FORCING = ("precipitation", "et0", "temperature")


def _f64(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


# --- derived quantities (D5) -------------------------------------------------------------------
def _derive(cls: str, values: Mapping):
    """What a surface's constructor computes from its parameters, in WSIMOD's order; the values
    are floats or tensors.

    ``Surface``: ``capacity = area * depth`` (``land.py:369-371``). ``ImperviousSurface`` passes
    ``depth = pore_depth`` (``:612``). ``PerviousSurface`` keeps ``field_capacity_m`` and
    ``wilting_point_m`` on the physical depth and passes the simulated depth
    ``depth * total_porosity`` (``:796-816``), with ``subsurface_coefficient = 1 -
    percolation_coefficient`` (``:820``).
    """
    if cls == "ImperviousSurface":
        depth = values["pore_depth"]
        return {"capacity": values["area"] * depth, "depth": depth}
    if cls == "PerviousSurface":
        physical = values["depth"]
        depth = physical * values["total_porosity"]
        return {
            "capacity": values["area"] * depth,
            "depth": depth,
            "field_capacity_m": values["field_capacity"] * physical,
            "wilting_point_m": values["wilting_point"] * physical,
            "subsurface_coefficient": 1 - values["percolation_coefficient"],
        }
    depth = values["depth"]
    return {"capacity": values["area"] * depth, "depth": depth}


def surface_derived(spec: SurfaceSpec) -> dict[str, float]:
    """The derived values of ``spec``'s surface from its merged parameters (D5): ``capacity``,
    the simulated ``depth`` and, for a ``PerviousSurface``, ``field_capacity_m``,
    ``wilting_point_m`` and ``subsurface_coefficient``.

    WSIMOD's ``PerviousSurface.apply_overrides`` (``land.py:843-884``) divides the simulated depth
    by the porosity and multiplies it back, so after an override its values can differ from
    construction's by one ulp; construction's are computed here.
    """
    cls = SURFACE_CLASSES[spec.cls]
    values = {**cls.parameter_defaults, **spec.parameters}
    return {k: float(v) for k, v in _derive(spec.cls, values).items()}


def _initial_storage(p: Pollutants, values: Mapping) -> Tensor:
    """``Tank``'s ``initial_storage`` (``tanks.py:36-53``): a number is a volume, a mapping a
    VQIP (keys outside the run's pollutants are left out, as a quantity-only run does)."""
    initial = values.get("initial_storage", 0)
    if isinstance(initial, Mapping):
        return p.to_vqip({k: v for k, v in initial.items() if k == "volume" or k in p.names})
    return _f64(initial)


# --- the surfaces --------------------------------------------------------------------------------
class Surface(DecayTank):
    """``Surface`` (``land.py:303-581``): a tank of ``capacity = area * depth`` whose ``run``
    walks ``inflows + processes + outflows``.

    Parameters (``parameter_defaults``, WSIMOD's defaults) come from the spec; :meth:`bind`
    replaces them with the day's (batched) values and derives the capacity again. ``parent``
    is the :class:`Land`: its bound ``"precipitation"``, ``"et0"`` and ``"temperature"`` are the
    surface's forcing (``get_data_input``, ``:451-460``), and ``"temperature"`` is what the
    end-of-day decay reads (``DecayTank``).

    A surface is a ``DecayTank`` whose ``decays`` default to ``{}``: with entries it owns a
    :class:`~noodl.apps.wsimod.decay.Decay`, bound once per day from the Land's *air*
    temperature (``DecayObj`` reads ``parent``'s ``data_input_dict``, ``core.py:462-468``; not
    the soil temperature, ``:1060-1079``). The decayed mass of the end of day ``d`` is
    ``total_decayed``; when ``nitrite`` is a pollutant, ``Surface.run`` of day ``d + 1`` turns it
    into nitrate and nitrite (``:427-439``). A surface's ``decays`` merge per surface; WSIMOD
    shares the default ``{}`` between every surface built without a ``decays`` key, so an override
    reaches all of them (D13: a defect that noodl does not reproduce).
    """

    surface_class: ClassVar[str] = "Surface"
    parameter_defaults: ClassVar[dict[str, float]] = {"area": 1, "depth": 1, "datum": 10}
    # the per-day state WSIMOD keeps on the surface besides its storage (recorded, model.py:1135)
    flow_fields: ClassVar[tuple[str, ...]] = ()

    def __init__(self, spec: SurfaceSpec, pollutants: Pollutants, *, parent: Land) -> None:
        self.spec = spec
        self.name = spec.name
        self.params: dict[str, Tensor] = {}
        values = spec.parameters
        self.defaults: dict[str, Tensor] = {
            name: _f64(values.get(name, default))
            for name, default in self.parameter_defaults.items()
        }
        load = values.get("pollutant_load") or {}
        # land.py:381-382: a configured pollutant_load (any entry) adds simple_deposition
        self.has_load = len(load) > 0
        self.defaults["pollutant_load"] = torch.tensor(
            [float(load.get(name, 0.0)) for name in pollutants.names], dtype=torch.float64
        ).reshape(pollutants.K)
        derived = _derive(type(self).surface_class, self.defaults)
        # X9, X34: dense over the K pollutants, inert (0, 1) where the configuration is silent;
        # a quantity-only run decays nothing, as it carries no pollutant
        known = {k: v for k, v in (values.get("decays") or {}).items() if k in pollutants.names}
        where = f"node {parent.name!r}, surface {spec.name!r}"
        self.decay_spec = read_decays(where, known, pollutants)
        self.defaults["decays.constant"] = torch.tensor(
            [self.decay_spec.constant.get(n, 0.0) for n in pollutants.names], dtype=torch.float64
        ).reshape(pollutants.K)
        self.defaults["decays.exponent"] = torch.tensor(
            [self.decay_spec.exponent.get(n, 1.0) for n in pollutants.names], dtype=torch.float64
        ).reshape(pollutants.K)
        decay = None
        if self.decay_spec.keys:
            decay = Decay(
                pollutants, self.decay_spec, site=f"node:{parent.name}/surfaces.{spec.name}"
            )
        super().__init__(
            pollutants,
            parent=parent,
            decay=decay,
            capacity=derived["capacity"],
            area=self.defaults["area"],
            datum=self.defaults["datum"],
            initial_storage=_initial_storage(pollutants, values),
            owner=parent.name,
            path=f"surfaces.{spec.name}",
        )
        self._set_derived(derived)
        # land.py:375-384: the deposition inflows are 3c (refused, or ignored when quantity-only)
        self.inflows = [self.simple_deposition] if self.has_load else []
        self.processes: list = []
        self.outflows: list = []

    # --- parameters -----------------------------------------------------------------------------
    def _param(self, name: str) -> Tensor:
        return self.params.get(name, self.defaults[name])

    def _set_derived(self, derived: Mapping[str, Tensor]) -> None:
        self.capacity = _f64(derived["capacity"])
        self.depth = _f64(derived["depth"])

    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """This day's parameters by name; the derived values are computed again from them."""
        unknown = set(parameters) - set(self.defaults)
        if unknown:
            raise KeyError(f"{self.site}: no parameter {sorted(unknown)[0]!r}")
        self.params = {k: _f64(v) for k, v in parameters.items()}
        values = {name: self._param(name) for name in self.defaults}
        self.area = values["area"]
        self.datum = values["datum"]
        self._set_derived(_derive(type(self).surface_class, values))

    def bind_decay(self, ctx: Ctx, temperature: Tensor | None) -> None:
        """Bind the surface's :class:`Decay` from the day's ``decays.*`` and the Land's air
        ``temperature`` (D15, R5). A surface whose ``decays`` lists no pollutant refuses a
        changed inert entry by name (X9); a quantity-only run has no pollutant to decay."""
        if self.p.K == 0:
            return
        constant, exponent = self._param("decays.constant"), self._param("decays.exponent")
        if self.decay is None:
            check_inert(
                self.site,
                "decays.constant/exponent",
                torch.zeros(self.p.K, dtype=torch.bool),
                constant,
                exponent,
                self.p.names,
            )
            return
        if temperature is None:
            raise KeyError(f"{self.site}: the Land has no ('temperature', t) forcing (core.py:484)")
        self.decay.bind(ctx, constant, exponent, temperature)

    # --- state ----------------------------------------------------------------------------------
    def stores(self) -> dict[str, Tensor]:
        """``storage``, the per-day flows and ``total_decayed``: always held and compared in the
        replay; carried by the Land only when ``nitrite`` is a pollutant (D8)."""
        return {
            "storage": self.storage,
            **{f: getattr(self, f) for f in self.flow_fields},
            "total_decayed": self.total_decayed,
        }

    def _with_temperature(self, vqip: Tensor, temperature: Tensor) -> Tensor:
        """``vqip["temperature"] = temperature`` (``land.py:685``, ``:991``, ``:1028``); WSIMOD
        writes a key no sum reads when ``temperature`` is not a pollutant."""
        if "temperature" not in self.p.names:
            return vqip
        column = torch.zeros(1 + self.p.K, dtype=torch.bool)
        column[self.p.column("temperature")] = True
        shape = torch.broadcast_shapes(vqip.shape[:-1], temperature.shape)
        vqip = vqip.expand(shape + vqip.shape[-1:])
        return torch.where(column, temperature.expand(shape)[..., None], vqip)

    # --- WSIMOD's methods -----------------------------------------------------------------------
    def run(self, *, active: Tensor) -> None:
        """``land.py:425-449``: every function in order, its ``(in, out)`` summed into the
        parent's running mass balance.

        With ``nitrite`` among the pollutants the first thing is the carry-over of yesterday's
        decay (``:427-439``, in WSIMOD's order): the decayed nitrite joins the nitrate, then the
        decayed ammonia joins the nitrite, each added to the storage and to the parent's
        ``running_inflow_mb``. ``total_decayed`` is what yesterday's ``end_timestep`` left
        (zero on day 0). A repeated ``Land.run`` on one day repeats the carry-over and so adds
        it again (``land_run_twice``)."""
        p, parent = self.p, self.parent
        if "nitrite" in p.names:
            for source, target in (("nitrite", "nitrate"), ("ammonia", "nitrite")):
                column = torch.zeros(1 + p.K, dtype=torch.bool)
                column[p.column(target)] = True
                moved = self.total_decayed[..., p.column(source)][..., None]
                self.storage = write(
                    active,
                    torch.where(column, self.storage + moved, self.storage),
                    self.storage,
                )
                parent.running_inflow_mb = write(
                    active,
                    torch.where(column, parent.running_inflow_mb + moved, parent.running_inflow_mb),
                    parent.running_inflow_mb,
                )
        for f in self.inflows + self.processes + self.outflows:
            in_, out_ = f(active=active)
            parent.running_inflow_mb = write(
                active, sum_vqip(p, parent.running_inflow_mb, in_), parent.running_inflow_mb
            )
            parent.running_outflow_mb = write(
                active, sum_vqip(p, parent.running_outflow_mb, out_), parent.running_outflow_mb
            )

    def simple_deposition(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:510-532``: the loads (additive ones times the area) at volume 0, forced in.
        :meth:`Land.validate` admits only zero loads with pollutants on (3c)."""
        p = self.p
        load = self._param("pollutant_load")
        amounts = torch.where(p.additive_mask[1:], load * self.area[..., None], load)
        pollution = torch.cat(
            [torch.zeros(amounts.shape[:-1] + (1,), dtype=torch.float64), amounts], dim=-1
        )
        self.push_storage(pollution, force=True, active=active)
        return pollution, self.ctx.empty()


class ImperviousSurface(Surface):
    """``ImperviousSurface`` (``land.py:584-721``): ``capacity = area * pore_depth``; rain less
    evaporation is forced in at air temperature and the ponded water goes to ``Sewer`` nodes.
    """

    surface_class: ClassVar[str] = "ImperviousSurface"
    parameter_defaults: ClassVar[dict[str, float]] = {
        "area": 1,
        "datum": 10,
        "pore_depth": 0,
        "et0_to_e": 1,
    }
    flow_fields: ClassVar[tuple[str, ...]] = ("evaporation", "precipitation")

    def __init__(self, spec: SurfaceSpec, pollutants: Pollutants, *, parent: Land) -> None:
        super().__init__(spec, pollutants, parent=parent)
        # land.py:615-621
        self.evaporation = empty(pollutants, ())
        self.precipitation = empty(pollutants, ())
        self.inflows.append(self.precipitation_evaporation)
        self.outflows.append(self.push_to_sewers)

    def precipitation_evaporation(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:645-697``: if ``P < E * et0_to_e`` the pores evaporate ``(E - P) * area``,
        limited to the store; otherwise ``(P - E) * area`` is forced in at air temperature."""
        ctx, parent = self.ctx, self.parent
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("precipitation_evaporation")
        precipitation = parent.params["precipitation"]
        evaporation = parent.params["et0"] * self._param("et0_to_e")
        dry = ctx.decide(
            f"{self.site}/precipitation_evaporation/dry",
            precipitation,
            "<",
            evaporation,
            active=active,
            call=call,
        )
        pores = (evaporation - precipitation) * self.area
        pores = self.evaporate(pores, active=active & dry)
        dry_total = pores + precipitation * self.area
        net = (precipitation - evaporation) * self.area
        net = self._with_temperature(self._volume_vqip(net), parent.params["temperature"])
        self.push_storage(net, force=True, active=active & ~dry)
        wet_total = evaporation * self.area
        total = torch.where(dry, dry_total, wet_total)
        evaporation_ = self._volume_vqip(total)
        precipitation_ = self._volume_vqip(precipitation * self.area)
        self.evaporation = write(active, evaporation_, self.evaporation)
        self.precipitation = write(active, precipitation_, self.precipitation)
        return precipitation_, evaporation_

    def push_to_sewers(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:699-721``: the ponded water to the parent's ``Sewer`` arcs; what they reject
        is forced back."""
        runoff = self.pull_ponded(active=active)
        reply = self.parent.push_distributed(runoff, of_type=SEWER_TYPES, active=active)
        self.push_storage(reply, force=True, active=active)
        return self.ctx.empty(), self.ctx.empty()


class PerviousSurface(Surface):
    """``PerviousSurface`` (``land.py:724-1079``): the IHACRES soil store (``ihacres``,
    ``:913-1044``), the soil temperature (``:1060-1079``) and ``route`` (``:1046-1058``).

    The tank's ``depth`` is the simulated depth ``depth * total_porosity``; ``field_capacity_m``
    and ``wilting_point_m`` use the physical depth. The soil-temperature weights are WSIMOD state
    attributes set only through overrides (``:753-756``), read from the spec's parameters.
    """

    surface_class: ClassVar[str] = "PerviousSurface"
    parameter_defaults: ClassVar[dict[str, float]] = {
        "area": 1,
        "datum": 10,
        "depth": 0.75,
        "total_porosity": 0.4,
        "field_capacity": 0.3,
        "wilting_point": 0.12,
        "infiltration_capacity": 0.5,
        "surface_coefficient": 0.05,
        "percolation_coefficient": 0.75,
        "et0_coefficient": 0.5,
        "ihacres_p": 10,
        **SOIL_TEMPERATURE_DEFAULTS,
    }
    flow_fields: ClassVar[tuple[str, ...]] = (
        "evaporation",
        "infiltration_excess",
        "percolation",
        "precipitation",
        "subsurface_flow",
        "tank_recharge",
    )

    def __init__(self, spec: SurfaceSpec, pollutants: Pollutants, *, parent: Land) -> None:
        super().__init__(spec, pollutants, parent=parent)
        # land.py:822-828
        self.infiltration_excess = empty(pollutants, ())
        self.subsurface_flow = empty(pollutants, ())
        self.percolation = empty(pollutants, ())
        self.tank_recharge = torch.zeros((), dtype=torch.float64)
        self.evaporation = empty(pollutants, ())
        self.precipitation = empty(pollutants, ())
        # land.py:831-841
        self.inflows.append(self.ihacres)
        self.processes.append(self.calculate_soil_temperature)
        self.outflows.append(self.route)

    def _set_derived(self, derived: Mapping[str, Tensor]) -> None:
        super()._set_derived(derived)
        self.field_capacity_m = _f64(derived["field_capacity_m"])
        self.wilting_point_m = _f64(derived["wilting_point_m"])
        self.subsurface_coefficient = _f64(derived["subsurface_coefficient"])

    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """The day's parameters; the IHACRES denominators ``depth - field_capacity_m`` and
        ``depth - wilting_point_m`` must stay positive (``land.py:942``, ``:950``)."""
        super().bind(parameters)
        if "temperature" in self.p.names:
            total = (
                self._param("soil_temp_w_air")
                + self._param("soil_temp_w_deep")
                + self._param("soil_temp_w_prev")
            )
            if bool((total == 0).any()):
                raise ValueError(
                    f"{self.site}: the total weight soil_temp_w_air + soil_temp_w_deep + "
                    f"soil_temp_w_prev is 0 (land.py:1073 divides by it); got total weight {total}"
                )
        for name, value in (
            ("field_capacity", self.field_capacity_m),
            ("wilting_point", self.wilting_point_m),
        ):
            if bool(((self.depth - value) <= 0).any()):
                raise ValueError(
                    f"{self.site}: total_porosity is not above {name} (land.py:942, :950 "
                    "divide by the simulated depth less it)"
                )

    def ihacres(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:913-1044``, in WSIMOD's order:

        - ``P_inf = min(P, infiltration_capacity)``, the excess ``max(P - P_inf, 0)``;
        - ``evap = E * min(1, exp(2 * (1 - cmd / (D - wp_m))))``;
        - ``outflow = P_inf * (1 - min(1, (cmd / (D - fc_m)) ** p))``;
        - ``evap = min(evap, P + smc)``;
        - the split into surface, percolation and subsurface flow, and the tank recharge.

        If water passes through the soil store (``> 0``) it is forced in at air temperature and
        the subsurface and percolation flows are pulled out; otherwise ``evaporate(-total)``
        runs and those flows are empty. WSIMOD prints "inaccurate evaporation calculation" when
        ``|evap + P_inf * area - evaporation - excess| > FLOAT_ACCURACY`` there
        (``:1003-1021``); noodl records the event ``"inaccurate_evaporation"``.
        """
        ctx, parent = self.ctx, self.parent
        active = torch.as_tensor(active, dtype=torch.bool)
        call = self._call("ihacres")
        site = f"{self.site}/ihacres"
        area = self.area
        # get_climate (land.py:905-911) and the temperature (:924)
        precipitation = parent.params["precipitation"]
        evaporation_depth = parent.params["et0"] * self._param("et0_coefficient")
        temperature = parent.params["temperature"]
        # land.py:927-928
        infiltrated = dmin(
            ctx,
            f"{site}/infiltration",
            precipitation,
            self._param("infiltration_capacity"),
            active=active,
            call=call,
        )
        excess = dmax(
            ctx, f"{site}/excess", precipitation - infiltrated, 0.0, active=active, call=call
        )
        # get_cmd (land.py:886-893)
        deficit = self.get_excess(active=active)[..., 0] / area
        # land.py:934-954
        factor = exp_python(2 * (1 - deficit / (self.depth - self.wilting_point_m)))
        factor = dmin(ctx, f"{site}/evaporation_kink", 1.0, factor, active=active, call=call)
        evaporation = evaporation_depth * factor
        ratio = pow_python(deficit / (self.depth - self.field_capacity_m), self._param("ihacres_p"))
        ratio = dmin(ctx, f"{site}/outflow_kink", 1.0, ratio, active=active, call=call)
        outflow = infiltrated * (1 - ratio)
        # land.py:957, get_smc (:895-903)
        moisture = self.storage[..., 0] / area
        evaporation = dmin(
            ctx,
            f"{site}/moisture",
            evaporation,
            precipitation + moisture,
            active=active,
            call=call,
        )
        # land.py:961-979
        surface_coefficient = self._param("surface_coefficient")
        surface = outflow * surface_coefficient * area
        percolation = (
            outflow * (1 - surface_coefficient) * self._param("percolation_coefficient") * area
        )
        subsurface = outflow * (1 - surface_coefficient) * self.subsurface_coefficient * area
        recharge = (infiltrated - evaporation - outflow) * area
        excess = excess * area
        excess = excess + surface
        evaporation = evaporation * area
        precipitation_volume = precipitation * area
        # land.py:982-1021
        total = recharge + subsurface + percolation
        through = ctx.decide(f"{site}/through_soil", total, ">", 0.0, active=active, call=call)
        wet, dry = active & through, active & ~through
        water = self._with_temperature(self._volume_vqip(total), temperature)
        self.push_storage(water, force=True, active=wet)
        subsurface_flow = self.pull_storage(self._volume_vqip(subsurface), active=wet)
        percolation_flow = self.pull_storage(self._volume_vqip(percolation), active=wet)
        evaporated = self.evaporate(-total, active=dry)
        gap = torch.abs(evaporated + infiltrated * area - evaporation - excess)
        ctx.event("inaccurate_evaporation", parent.name, dry & (gap > ctx.fa).detach())
        nothing = ctx.empty()
        subsurface_flow = torch.where(through[..., None], subsurface_flow, nothing)
        percolation_flow = torch.where(through[..., None], percolation_flow, nothing)
        # land.py:1025-1044
        excess_ = self._with_temperature(self._volume_vqip(excess), temperature)
        precipitation_ = self._volume_vqip(precipitation_volume)
        evaporation_ = self._volume_vqip(evaporation)
        self.infiltration_excess = write(active, excess_, self.infiltration_excess)
        self.subsurface_flow = write(active, subsurface_flow, self.subsurface_flow)
        self.percolation = write(active, percolation_flow, self.percolation)
        self.tank_recharge = write(active, recharge, self.tank_recharge)
        self.evaporation = write(active, evaporation_, self.evaporation)
        self.precipitation = write(active, precipitation_, self.precipitation)
        return precipitation_, evaporation_

    def calculate_soil_temperature(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:1060-1079``: the storage's temperature becomes
        ``(T_s w_prev + T_air w_air + T_deep w_deep) / (w_air + w_deep + w_prev)``. With pollutants
        on, any weights are accepted unless they sum to 0 (``land.py:1073`` divides by the sum;
        ``bind`` refuses a zero total weight); without a ``temperature`` column there is nothing
        to write. The expression is WSIMOD's, term by term, so it is bitwise equal."""
        if "temperature" not in self.p.names:
            return self.ctx.empty(), self.ctx.empty()
        column = self.p.column("temperature")
        w_prev, w_air = self._param("soil_temp_w_prev"), self._param("soil_temp_w_air")
        w_deep = self._param("soil_temp_w_deep")
        auto = self.storage[..., column] * w_prev
        air = self.parent.params["temperature"] * w_air
        total_weight = w_air + w_deep + w_prev
        value = (auto + air + self._param("soil_temp_deep") * w_deep) / total_weight
        mask = torch.zeros(1 + self.p.K, dtype=torch.bool)
        mask[column] = True
        shape = torch.broadcast_shapes(self.storage.shape[:-1], value.shape)
        storage = self.storage.expand(shape + self.storage.shape[-1:])
        new = torch.where(mask, value.expand(shape)[..., None], storage)
        self.storage = write(active, new, self.storage)
        return self.ctx.empty(), self.ctx.empty()

    def route(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``land.py:1046-1058``: the three flows forced into the parent's residence tanks."""
        parent = self.parent
        parent.surface_runoff.push_storage(self.infiltration_excess, force=True, active=active)
        parent.subsurface_runoff.push_storage(self.subsurface_flow, force=True, active=active)
        parent.percolation.push_storage(self.percolation, force=True, active=active)
        return self.ctx.empty(), self.ctx.empty()


SURFACE_CLASSES: dict[str, type[Surface]] = {
    "Surface": Surface,
    "ImperviousSurface": ImperviousSurface,
    "PerviousSurface": PerviousSurface,
}


# --- the node ------------------------------------------------------------------------------------
class Land(Node):
    """``Land`` (``land.py:17-300``): surfaces and three residence tanks of capacity ``1e15``.

    Parameters: the residence times (``surface_residence_time``, ``subsurface_residence_time``,
    ``percolation_residence_time``) and the forcing ``"precipitation"``, ``"et0"`` and
    ``"temperature"``, all ``(...)``; a surface's parameters are bound under
    ``"surfaces.<name>.<parameter>"``.

    Handlers (``land.py:91-129``): pushes with the default tag and their checks are denied; the
    ``("Demand", "Garden")`` check and push answer empty, as without a ``GardenSurface`` (3c,
    W16), so a garden demand is 0. W4: ``push_set_sewer`` (``:246-264``) reads a nonexistent
    ``self.surface`` and there is no ``"Sewer"`` check, so ponding from a sewer crashes WSIMOD;
    noodl does the evident intent, a check that accepts everything when the land has an
    ``ImperviousSurface`` (empty otherwise) and a set that forces the water into the first one.

    The stores are the residence tanks (in ``dir()`` order, D10), each surface's storage and
    per-day flows under ``surfaces.<name>``, and the day's running mass balance.
    """

    wsimod_type: ClassVar[str] = "Land"
    step_methods: ClassVar[tuple[str, ...]] = ("run", "apply_irrigation")
    # land.py:158-162: the surfaces' and tanks' ds is the engine's, from the stores (D11)
    # a surface's decayed mass leaves its storage at the end of the day (tanks.py:449-457) and
    # re-enters as the next day's carry-over, inside "Land.running_inflow_mb" (land.py:427-439)
    mass_balance: ClassVar[dict[str, str]] = {
        "Land.running_inflow_mb": "in",
        "Land.running_outflow_mb": "out",
        "DecayTank.decayed": "out",
    }
    parameter_defaults: ClassVar[dict[str, float]] = {
        "surface_residence_time": 1,
        "subsurface_residence_time": 5,
        "percolation_residence_time": 50,
    }
    _TANKS: ClassVar[tuple[str, ...]] = ("percolation", "subsurface_runoff", "surface_runoff")

    def __init__(self, spec: NodeSpec, pollutants: Pollutants) -> None:
        super().__init__(spec, pollutants)
        self.site = f"node:{self.name}"
        self._ctx: Ctx | None = None
        values = spec.parameters
        self.defaults: dict[str, Tensor] = {
            name: _f64(values.get(name, default))
            for name, default in self.parameter_defaults.items()
        }
        # land.py:91-94 (no GardenSurface: W16), :120-123, W4's "Sewer" check
        self.push_check_handler[GARDEN_TAG] = self.push_check_deny
        self.push_set_handler[GARDEN_TAG] = self.push_set_garden
        self.surfaces: list[Surface] = [
            SURFACE_CLASSES[s.cls](s, pollutants, parent=self) for s in spec.surfaces
        ]
        self.push_set_handler["default"] = self.push_set_deny
        self.push_check_handler["default"] = self.push_check_deny
        self.push_set_handler["Sewer"] = self.push_set_sewer
        self.push_check_handler["Sewer"] = self.push_check_sewer
        # land.py:138-149
        for name in self._TANKS:
            setattr(
                self,
                name,
                ResidenceTank(
                    pollutants,
                    residence_time=self.defaults[f"{name.split('_')[0]}_residence_time"],
                    capacity=_f64(UNBOUNDED_CAPACITY),
                    area=_f64(1),
                    datum=_f64(10),
                    initial_storage=_f64(0.0),
                    owner=self.name,
                    path=name,
                ),
            )
        # land.py:155-156
        self.running_inflow_mb = empty(pollutants, ())
        self.running_outflow_mb = empty(pollutants, ())
        self.store_paths = self.store_paths_of([(s.cls, s.name) for s in spec.surfaces])
        # carried: the residence tanks and the surfaces' storage; the surfaces' flows and the
        # running mass balance are the day's (W75). A surface's total_decayed is carried when
        # "nitrite" is a pollutant: the next day's run reads it (D8)
        carried = [x for x in self.store_paths if x.endswith(".storage")]
        if "nitrite" in pollutants.names:
            carried += [x for x in self.store_paths if x.endswith(".total_decayed")]
        self.carried_paths = tuple(carried)

    @classmethod
    def store_paths_of(cls, surfaces: Sequence[tuple[str, str]]) -> tuple[str, ...]:
        """The store paths of a ``Land`` with ``surfaces`` ``[(class, name), ...]``: the
        residence tanks in ``dir()`` order (D10), each surface's ``storage``, per-day flows and
        ``total_decayed``, and the running mass balance."""
        paths = [f"{tank}.storage" for tank in cls._TANKS]
        for surface_cls, name in surfaces:
            fields = ("storage", *SURFACE_CLASSES[surface_cls].flow_fields)
            paths += [f"surfaces.{name}.{field}" for field in (*fields, "total_decayed")]
        return (*paths, "running_inflow_mb", "running_outflow_mb")

    # --- the day's context and parameters -------------------------------------------------------
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
        return [*self.surfaces, *(getattr(self, name) for name in self._TANKS)]

    def _param(self, name: str) -> Tensor:
        return self.params.get(name, self.defaults[name])

    def configured_parameters(self, date: datetime) -> dict[str, Tensor]:
        """The day's ``precipitation``, ``et0`` and ``temperature`` (``land.py:451-460``)."""
        data = self.spec.data
        return {v: torch.tensor(data[(v, date)], dtype=torch.float64) for v in LAND_FORCING}

    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """This day's parameters: the residence times reach the tanks; the keys
        ``"surfaces.<name>.<parameter>"`` are bound on that surface."""
        own = {k: v for k, v in parameters.items() if not k.startswith("surfaces.")}
        super().bind(own)
        for name in self._TANKS:
            tank = getattr(self, name)
            key = f"{name.split('_')[0]}_residence_time"
            if bool((self._param(key) == 0).any()):
                raise ValueError(f"node {self.name!r}: {key} is 0 (tanks.py:398 divides)")
            tank.residence_time = self._param(f"{name.split('_')[0]}_residence_time")
        by_surface: dict[str, dict[str, Tensor]] = {s.name: {} for s in self.surfaces}
        for key, value in parameters.items():
            if not key.startswith("surfaces."):
                continue
            _, surface, parameter = key.split(".", 2)
            if surface not in by_surface:
                raise KeyError(f"node {self.name!r} has no surface {surface!r}")
            by_surface[surface][parameter] = value
        for surface in self.surfaces:
            surface.bind(by_surface[surface.name])
        # D15: each surface binds its Decay after the Land's own parameters, with the Land's
        # air temperature (core.py:462-468: the parent's data_input_dict)
        for surface in self.surfaces:
            surface.bind_decay(self.ctx, self.params.get("temperature"))

    # --- state ----------------------------------------------------------------------------------
    def _surface(self, name: str) -> Surface:
        """``get_surface`` (``land.py:279-292``): the first surface of that name."""
        for surface in self.surfaces:
            if surface.name == name:
                return surface
        raise KeyError(f"node {self.name!r} has no surface {name!r}")

    def stores(self) -> dict[str, Tensor]:
        out = {f"{name}.storage": getattr(self, name).storage for name in self._TANKS}
        for surface in self.surfaces:
            for field, value in surface.stores().items():
                out[f"surfaces.{surface.name}.{field}"] = value
        out["running_inflow_mb"] = self.running_inflow_mb
        out["running_outflow_mb"] = self.running_outflow_mb
        return out

    def load_stores(self, stores: Mapping[str, Tensor]) -> None:
        unknown = set(stores) - set(self.store_paths)
        if unknown:
            raise KeyError(f"node {self.name!r} has no store {sorted(unknown)[0]!r}")
        for path, value in stores.items():
            value = _f64(value)
            if path.startswith("surfaces."):
                _, name, field = path.split(".", 2)
                setattr(self._surface(name), field, value)
            elif "." in path:
                tank, field = path.split(".")
                setattr(getattr(self, tank), field, value)
            else:
                setattr(self, path, value)

    def end_timestep(self) -> None:
        """``land.py:266-277``: the running mass balance is reset, then every surface (the decay
        reads the day's temperature) and every residence tank ends its day."""
        self.running_inflow_mb = empty(self.p, ())
        self.running_outflow_mb = empty(self.p, ())
        for tank in [*self.surfaces, self.surface_runoff, self.subsurface_runoff, self.percolation]:
            tank.end_timestep()

    def record_terms(self) -> None:
        ctx = self.ctx
        every = torch.ones(ctx.shape, dtype=torch.bool)
        ctx.term("Land.running_inflow_mb", self.running_inflow_mb, every)
        ctx.term("Land.running_outflow_mb", self.running_outflow_mb, every)

    # --- the handlers ---------------------------------------------------------------------------
    def push_set_garden(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``land.py:94``: ``lambda x: self.empty_vqip()``, everything accepted (W16)."""
        return self.ctx.empty()

    def _first_impervious(self) -> ImperviousSurface | None:
        return next((s for s in self.surfaces if isinstance(s, ImperviousSurface)), None)

    def push_check_sewer(self, vqip: Tensor | None = None, *, active: Tensor) -> Tensor:
        """W4, the evident intent: everything is accepted when the land has an
        ``ImperviousSurface`` (``push_check_accept``), nothing otherwise."""
        if self._first_impervious() is None:
            return self.push_check_deny(vqip, active=active)
        return self.push_check_accept(vqip, active=active)

    def push_set_sewer(self, vqip: Tensor, *, active: Tensor) -> Tensor:
        """``land.py:246-264`` as intended (W4): forced into the first ``ImperviousSurface``;
        without one, everything is returned."""
        surface = self._first_impervious()
        if surface is None:
            return vqip
        return surface.push_storage(vqip, force=True, active=active)

    # --- orchestration --------------------------------------------------------------------------
    def apply_irrigation(self, *, active: Tensor) -> None:
        """``land.py:192-198``: only ``lambda: None`` (``:97``); irrigation surfaces are 3c."""

    def run(self, *, active: Tensor) -> None:
        """``land.py:200-244``:

        1. every surface's ``run``, in order;
        2. the percolation tank's outflow to ``Groundwater``; the rejection is forced back only
           above ``FLOAT_ACCURACY`` (a smaller one vanishes: dropped water);
        3. the surface and subsurface runoff, summed, to ``River`` and ``Node`` when positive;
           a positive rejection is split pro rata by volume and each positive share forced
           back into its tank.
        """
        ctx, p = self.ctx, self.p
        active = torch.as_tensor(active, dtype=torch.bool)
        call = ctx.call(self.site, "run")
        where = f"{self.site}/run"
        for surface in self.surfaces:
            surface.run(active=active)
        percolation = self.percolation.pull_outflow(active=active)
        reply = self.push_distributed(percolation, of_type=PERCOLATION_TYPES, active=active)
        back = ctx.decide(
            f"{where}/percolation_reply", reply[..., 0], ">", ctx.fa, active=active, call=call
        )
        self.percolation.push_storage(reply, force=True, active=active & back)
        ctx.drop(reply, active & ~back)  # land.py:213-215: at or below FLOAT_ACCURACY, lost
        surface_runoff = self.surface_runoff.pull_outflow(active=active)
        subsurface_runoff = self.subsurface_runoff.pull_outflow(active=active)
        total = sum_vqip(p, surface_runoff, subsurface_runoff)
        runoff = active & ctx.decide(
            f"{where}/runoff", total[..., 0], ">", 0.0, active=active, call=call
        )
        reply = self.push_distributed(total, of_type=RUNOFF_TYPES, active=runoff)
        rejected = runoff & ctx.decide(
            f"{where}/rejected", reply[..., 0], ">", 0.0, active=runoff, call=call
        )
        den = safe_den(total[..., 0])  # total > 0 where it is used
        for tank, part, branch in (
            (self.surface_runoff, surface_runoff, "rejected_surface"),
            (self.subsurface_runoff, subsurface_runoff, "rejected_subsurface"),
        ):
            share = v_change_vqip(p, reply, reply[..., 0] * part[..., 0] / den)
            positive = rejected & ctx.decide(
                f"{where}/{branch}", share[..., 0], ">", 0.0, active=rejected, call=call
            )
            tank.push_storage(share, force=True, active=positive)

    # --- validation -----------------------------------------------------------------------------
    @classmethod
    def validate(cls, spec: NodeSpec, network: WsimodNetwork, *, pollutants: bool) -> list[str]:
        """Refused by name: a residence time of 0 (``tanks.py:398`` divides), missing forcing
        (``precipitation``, ``et0`` and ``temperature`` on every date, spec Forcing checks),
        duplicate surface names or names with a ``"."`` (the store paths), and per surface
        :func:`_validate_surface`. Returns what a quantity-only run ignores."""
        super().validate(spec, network, pollutants=pollutants)
        where = f"node {spec.name!r}"
        for name, default in cls.parameter_defaults.items():
            if float(spec.parameters.get(name, default)) == 0:
                raise ValueError(f"{where}: {name} is 0 (tanks.py:398 divides by it)")
        for variable in LAND_FORCING:
            for date, label in zip(network.dates, network.date_labels, strict=True):
                if (variable, date) not in spec.data:
                    raise ValueError(f"land {spec.name!r} has no {variable!r} forcing on {label}")
        if pollutants and spec.surfaces and "nitrite" in network.pollutants.names:
            missing = [n for n in ("nitrate", "ammonia") if n not in network.pollutants.names]
            if missing:
                raise ValueError(
                    f"{where}: 'nitrite' is a pollutant but {missing[0]!r} is not: Surface.run "
                    "moves decayed nitrite to nitrate and decayed ammonia to nitrite "
                    "(land.py:427-439, KeyError)"
                )
        names = [s.name for s in spec.surfaces]
        for name in names:
            if "." in name:
                raise ValueError(f"{where}: surface name {name!r} contains '.'")
            if names.count(name) > 1:
                raise ValueError(f"{where}: two surfaces are named {name!r}")
        ignored: list[str] = []
        for surface in spec.surfaces:
            ignored += _validate_surface(
                surface, network, f"{where}, surface {surface.name!r}", pollutants=pollutants
            )
        return ignored

    @classmethod
    def check_recurses(cls, spec: NodeSpec, direction: str, tag: Tag) -> bool:
        """The push checks answer alone; the default pull check is ``pull_check_basic``."""
        return direction == "pull" and tag == "default"


def _validate_surface(
    surface: SurfaceSpec, network: WsimodNetwork, where: str, *, pollutants: bool
) -> list[str]:
    """One surface's refusals; returns its parameters a quantity-only run ignores.

    - Surface data (D21): only the deposition keys (``land.py:375-380``) are accepted, and with
      pollutants on they are refused (3c).
    - ``pollutant_load``: with pollutants on, every entry must be 0 and additive (3c).
    - ``PerviousSurface``: ``total_porosity`` above ``field_capacity`` and ``wilting_point``
      (``:942``, ``:950`` divide by the difference), a positive simulated depth less either, a
      non-zero ``area`` (``get_cmd`` divides, ``:893``) and ``ihacres_p >= 0`` (``0 ** p`` with
      ``p < 0`` raises ``ZeroDivisionError`` on a saturated soil). With pollutants on, a
      ``temperature`` pollutant (``calculate_soil_temperature`` reads it, ``:1069``) and a
      non-zero total soil weight ``soil_temp_w_air + soil_temp_w_deep + soil_temp_w_prev``
      (``:1073``; ``ZeroDivisionError`` in WSIMOD).
    """
    ignored: list[str] = []
    prefix = f"surfaces.{surface.name}"
    read_decays(where, surface.parameters.get("decays"), network.pollutants)
    variables = {variable for variable, _ in surface.data}
    other = sorted(variables - DEPOSITION_KEYS)
    if other:
        raise ValueError(
            f"{where}: surface data {other[0]!r} is not read (a surface's data_input_dict holds "
            "deposition only, land.py:375-380)"
        )
    if variables:
        if pollutants:
            raise ValueError(
                f"{where}: surface deposition data {sorted(variables)[0]!r} is sub-project 3c; "
                "run build_model(network, pollutants=False) until sub-project 3c"
            )
        ignored.append(f"{prefix}.data_input_dict")
    load = surface.parameters.get("pollutant_load") or {}
    if load:
        if pollutants:
            for key, value in load.items():
                if float(value) != 0 or key in network.pollutants.non_additive:
                    raise ValueError(
                        f"{where}: pollutant_load {key!r}: {value!r} is deposition, sub-project "
                        "3c (only zero loads of additive pollutants are accepted); "
                        "run build_model(network, pollutants=False) until sub-project 3c"
                    )
        else:
            ignored.append(f"{prefix}.pollutant_load")
    if surface.cls != "PerviousSurface":
        return ignored
    values = {**PerviousSurface.parameter_defaults, **surface.parameters}
    porosity = values["total_porosity"]
    for name in ("field_capacity", "wilting_point"):
        if not porosity > values[name]:
            raise ValueError(
                f"{where}: total_porosity {porosity} is not above {name} {values[name]} "
                "(land.py:942, :950 divide by depth * (total_porosity - it))"
            )
    derived = surface_derived(surface)
    for name in ("field_capacity_m", "wilting_point_m"):
        if not derived["depth"] - derived[name] > 0:
            raise ValueError(
                f"{where}: the simulated depth less {name} is not positive (depth "
                f"{values['depth']}, total_porosity {porosity}; land.py:942, :950 divide by it)"
            )
    if float(values["area"]) == 0:
        raise ValueError(f"{where}: area is 0 (land.py:893 divides by it)")
    if float(values["ihacres_p"]) < 0:
        raise ValueError(
            f"{where}: ihacres_p {values['ihacres_p']} is negative (land.py:950 raises "
            "ZeroDivisionError on a saturated soil)"
        )
    weights = {k: float(values[k]) for k in SOIL_TEMPERATURE_DEFAULTS}
    # ``neutral`` only decides which soil weights are listed as ignored when pollutants=False
    neutral = (
        weights["soil_temp_w_air"] == 0
        and weights["soil_temp_w_deep"] == 0
        and weights["soil_temp_w_prev"] > 0
    )
    if pollutants:
        if "temperature" not in network.pollutants.names:
            raise ValueError(
                f"{where}: a PerviousSurface needs a 'temperature' pollutant "
                "(calculate_soil_temperature reads it, land.py:1069, KeyError)"
            )
        total = (
            weights["soil_temp_w_air"] + weights["soil_temp_w_deep"] + weights["soil_temp_w_prev"]
        )
        if total == 0:
            raise ValueError(
                f"{where}: the total weight soil_temp_w_air + soil_temp_w_deep + "
                f"soil_temp_w_prev is 0 (land.py:1073 divides by it); got {weights}"
            )
    elif not neutral and "temperature" in network.pollutants.names:
        ignored += [f"{prefix}.{k}" for k in SOIL_TEMPERATURE_DEFAULTS]
    return ignored


NODE_CLASSES["Land"] = Land
