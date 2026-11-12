"""One WSIMOD day as a noodl integrating closure: :class:`WsimodDay` (spec, The daily driver).

``model.step(state, drivers, dt=86400, t=k * 86400)`` runs day ``k`` of the configuration's
dates (D16) through the closure, which owns every ``"wsimod.*"`` state key:

1. the carried stores are loaded from the state into the nodes and arcs, and the per-day
   values reset (W68, W75);
2. day ``k``'s slice of every driver is bound: the parameters ``"wsimod.<Class>.<parameter>"``
   (rows: the nodes of that configured class, D6), ``"wsimod.<SurfaceClass>.<parameter>"``
   (rows: the surfaces of every ``Land``), ``"wsimod.Arc.capacity"`` and
   ``"wsimod.Arc.preference"`` (rows: every arc), and the forcing series
   ``"wsimod.precipitation"``, ``"wsimod.et0"``, ``"wsimod.temperature"`` ``(..., T,
   n_readers)``, ``"wsimod.flow"`` ``(..., T, n_catchments)`` and ``"wsimod.concentration"``
   ``(..., T, n_catchments, K)``;
3. the orchestration runs slot by slot, each slot's nodes in configuration order
   (``model.py:999-1003``), then ``River.distribute`` in the river order (``:1005-1007``);
4. the day is recorded (:class:`DayRecord`, ``:1063-1170``), at the state it has reached;
5. ``end_timestep`` runs on every node, then on every arc (``:1172-1176``), and the carried
   stores are dumped into the returned state.

The call structure is static (spec, Approach). The closure is built with one tracing day
(D2): batch shape ``()``, ``torch.no_grad()``, day 0's configured drivers, values discarded.
It fixes every ``QueueArc``'s request-call labels (``begin_tracing``/``end_tracing``) and so
its rows, and the day's whole call sequence (``Ctx.sequence``, which includes every
distribute-loop site); every later day must repeat it, or the day raises (an internal error).

State layout (D10; spec, State). Every key has a leading batch shape; the rows are labelled:

- ``"wsimod.storage"`` ``(..., n_tanks, 1+K)``: the nodes in configuration order, each node's
  tanks in ``dir()`` (alphabetical) order as ``record_all`` walks them (``model.py:1119-1133``),
  ``(node, attribute)``; a ``Land``'s surfaces follow its tanks as ``(node, "surface:<name>")``.
- ``"wsimod.active_storage"`` ``(..., n_queue_tanks, 1+K)``: each ``QueueTank``'s
  ``active_storage``, ``(node, attribute)``.
- ``"wsimod.travel_queue"`` ``(..., n_registers, 1+K)`` and ``"wsimod.travel_queue_present"``
  ``(..., n_registers)``: the shift registers of the ``QueueTank`` internal arcs (nodes in
  configuration order, labelled ``"<node>.<tank>.internal_arc"``) then of the user
  ``AltQueueArc`` (arc order), one row per travel time ``(arc, i)``, ``i = 0 ... L`` with the
  arc's own ``L``; ``present`` is WSIMOD's key set (W48: a bool, or 0/1 as a float).
- ``"wsimod.arc_queue"`` ``(..., n_rows, 1+K)``: every ``QueueArc``'s request rows
  ``(arc, direction, i, j)`` (D1), and ``"wsimod.arc_queue_since"`` ``(..., n_anchors)`` its
  anchors ``(arc, direction)``.
- ``"wsimod.liquor"`` (every ``WWTW`` and ``FWTW``), ``"wsimod.previous_pulled"`` (``FWTW``)
  and ``"wsimod.treated"`` (``WWTW``), ``(..., n, 1+K)`` by node.

Units (W20): volumes ``m3``, flows and capacities ``m3/d``, additive pollutants ``kg``,
non-additive ones by name (``temperature`` ``degC``); never converted.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from noodl.apps.wsimod.arcs import AltQueueArc, Arc, QueueArc
from noodl.apps.wsimod.config import WsimodNetwork
from noodl.apps.wsimod.engine import Ctx, Engine
from noodl.apps.wsimod.structure import (
    check_network,
    forcing_readers,
    orchestration_plan,
    river_order,
)
from noodl.apps.wsimod.vqip import QUANTITY_ONLY, Pollutants
from noodl.refs import Field

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.node import Node
    from noodl.contracts import StepContext

__all__ = ["DayRecord", "SECONDS_PER_DAY", "STATE_KEYS", "WsimodDay"]

F64 = torch.float64
SECONDS_PER_DAY = 86400.0
"""One step is one day (``DT_DAYS = 1``, ``constants.py:20``): ``dt`` and ``t`` in seconds."""

STATE_KEYS = (
    "wsimod.storage",
    "wsimod.active_storage",
    "wsimod.travel_queue",
    "wsimod.travel_queue_present",
    "wsimod.arc_queue",
    "wsimod.arc_queue_since",
    "wsimod.liquor",
    "wsimod.previous_pulled",
    "wsimod.treated",
)
"""The state keys :class:`WsimodDay` carries (spec, State)."""

# keys that are not water held: a QueueTank's active storage is inside its storage, and an
# FWTW's previous_pulled is the quality of yesterday's pulled water (W75)
_NOT_HELD = ("wsimod.active_storage", "wsimod.previous_pulled")
# keys of one value per row, not a VQIP
_FLAT_KEYS = ("wsimod.travel_queue_present", "wsimod.arc_queue_since")
_STATE_UNITS = {
    "wsimod.travel_queue_present": "-",
    "wsimod.arc_queue_since": "d",
}
_STATE_TEXT = {
    "wsimod.storage": "every tank's and surface's storage (VQIP)",
    "wsimod.active_storage": "every QueueTank's active storage (VQIP)",
    "wsimod.travel_queue": "the AltQueueArc shift registers, by travel time (VQIP)",
    "wsimod.travel_queue_present": "the AltQueueArc registers' key sets",
    "wsimod.arc_queue": "the QueueArc request rows (VQIP)",
    "wsimod.arc_queue_since": "days since each QueueArc direction's last update_queue",
    "wsimod.liquor": "every WWTW's and FWTW's liquor (VQIP)",
    "wsimod.previous_pulled": "every FWTW's previous day's pulled water (VQIP)",
    "wsimod.treated": "every WWTW's treated water (VQIP)",
}

# D6 parameter units (W20); a parameter not listed has none declared
_PARAMETER_UNITS = {
    "capacity": "m3",
    "area": "m2",
    "datum": "m",
    "depth": "m",
    "length": "m",
    "width": "m",
    "velocity": "m/d",
    "damp": "-",
    "mrf": "m3/d",
    "residence_time": "d",
    "surface_residence_time": "d",
    "subsurface_residence_time": "d",
    "percolation_residence_time": "d",
    "infiltration_threshold": "-",
    "infiltration_pct": "-",
    "environmental_flow": "m3/d",
    "constant_demand": "m3/d",
    "population": "-",
    "per_capita": "m3/d",
    "gardening_efficiency": "-",
    "constant_temp": "degC",
    "constant_weighting": "-",
    "leakage": "-",
    "chamber_area": "m2",
    "chamber_floor": "m",
    "treatment_throughput_capacity": "m3/d",
    "percent_solids": "-",
    "liquor_multiplier": "-",
    "constant": "-",
    "exponent": "-",
    "stormwater_storage_capacity": "m3",
    "stormwater_storage_area": "m2",
    "stormwater_storage_elevation": "m",
    "service_reservoir_storage_capacity": "m3",
    "service_reservoir_storage_area": "m2",
    "service_reservoir_storage_elevation": "m",
    "pore_depth": "m",
    "et0_to_e": "-",
    "total_porosity": "-",
    "field_capacity": "-",
    "wilting_point": "-",
    "infiltration_capacity": "m/d",
    "surface_coefficient": "-",
    "percolation_coefficient": "-",
    "et0_coefficient": "-",
    "ihacres_p": "-",
    "soil_temp_w_prev": "-",
    "soil_temp_w_air": "-",
    "soil_temp_w_deep": "-",
    "soil_temp_deep": "degC",
    "timearea": "-",
    "pipe_timearea": "-",
}
_FORCING_UNITS = {
    "precipitation": "m/d",
    "et0": "m/d",
    "temperature": "degC",
    "flow": "m3/d",
}
# record_all's surface fields (model.py:1135-1170) -> the surface attribute
_SURFACE_FIELDS = {
    "storage": "storage",
    "evaporation": "evaporation",
    "precipitation": "precipitation",
    "percolation": "percolation",
    "subsurface_r": "subsurface_flow",
    "surface_r": "infiltration_excess",
    "tank_recharge": "tank_recharge",
}


@dataclass
class DayRecord:
    """What one day records (``model.py:1063-1170``), at the state the day reached, before
    ``end_timestep``; every tensor has the day's batch shape in front. ``WsimodDay`` completes
    the record after ``end_timestep`` (D14): ``storage_end``, and ``terms``, ``dropped``,
    ``dropped_arcs`` and ``events`` again, so what ``end_timestep`` records counts in its day.

    - ``arc_flow`` ``(..., n_arcs)``: each arc's ``vqip_out`` volume (``m3/d``);
      ``arc_quality`` ``(..., n_arcs, K)`` its pollutant entries.
    - ``tank_storage`` ``(..., n_tanks)`` and ``tank_quality`` ``(..., n_tanks, K)``: the
      tanks ``record_all`` reports (``Tank``, ``QueueTank``, ``ResidenceTank``;
      ``WsimodDay.tank_names``).
    - ``surface``: ``{field: (..., n_surfaces)}`` for the surfaces of the nodes under the type
      key ``"Land"`` (``WsimodDay.surface_names``), NaN where WSIMOD records no such field (an
      ``ImperviousSurface``'s flows); ``surface_quality`` ``(..., n_surfaces, K)``.
    - ``terms``: the mass-balance terms ``"<Class>.<term>"`` (D11), VQIPs ``(..., 1+K)``;
      ``dropped`` the water WSIMOD drops below ``FLOAT_ACCURACY`` and ``dropped_arcs`` the part
      of it lost inside a ``QueueArc``, ``storage_start`` the water held in every carried
      store when the day began, ``storage_total`` when it was recorded and ``storage_end``
      after ``end_timestep``, all ``(..., 1 + K_additive)`` (the volume and the additive
      columns). ``storage_end`` is ``None`` until the record is completed: a consumer at the
      ``run_days`` yield point must not read it. ``wsimod_run`` assembles the balance from
      them, over the window ``storage_start`` to ``storage_end`` (X8): the next day's
      ``storage_start`` is this day's ``storage_end``. A ``QueueTank`` push below
      ``FLOAT_ACCURACY`` stays in the tank's storage, so it is not in ``dropped`` (W49); water
      pushed into a ``WWTW`` after its ``calculate_discharge`` is the term
      ``"WWTW.unprocessed_input"`` (W69).
    - ``events``: ``{condition: (..., n_nodes) bool}`` (D13); ``maxiter``: ``{(node, method,
      call index): (...) bool}``; ``decisions``: the branch records (D12) when
      ``WsimodDay.record_decisions``, else ``None``.
    """

    arc_flow: Tensor
    arc_quality: Tensor
    tank_storage: Tensor
    tank_quality: Tensor
    surface: dict[str, Tensor]
    surface_quality: Tensor
    terms: dict[str, Tensor]
    dropped: Tensor
    dropped_arcs: Tensor
    storage_total: Tensor
    storage_start: Tensor
    storage_end: Tensor | None
    events: dict[str, Tensor]
    maxiter: dict
    decisions: list | None


@dataclass
class _Slot:
    """One carried store: ``rows`` of ``key`` from ``start``, the store's own trailing
    ``shape`` (its batch shape in front)."""

    key: str
    element: object  # the Node or Arc that owns it
    path: str
    start: int
    rows: int
    shape: tuple[int, ...]


@dataclass
class _Driver:
    """One driver key: its row labels, its columns (``None`` for one value per row), and per
    row the element and the name it is bound under."""

    key: str
    labels: list
    columns: tuple | None
    template: Tensor
    targets: list[tuple[object, str]] = field(default_factory=list)
    forcing: bool = False
    unit: str = ""
    description: str = ""


def _drive(generator) -> object:
    """Run a generator to its end; its return value."""
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return stop.value


def _f64(x) -> Tensor:
    return torch.as_tensor(x, dtype=F64)


class WsimodDay:
    """WSIMOD 0.8.1's daily step as an integrating closure (spec, The daily driver).

    ``WsimodDay(network, pollutants=...)`` refuses what cannot run (:func:`check_network`),
    builds the :class:`Engine`, the river order and the orchestration plan, and runs the
    tracing day (D2). ``pollutants=False`` runs on the volumes alone (``K = 0``; spec W10).

    Called by ``Model.step`` with a ``StepContext``: ``t`` must be a whole number of days
    inside the configuration's dates and ``dt`` one day (D16), each refused by name
    otherwise; called with ``ctx=None`` (a query) it returns nothing. A driver key left out of
    the drivers takes the configured value (:meth:`driver_template`). :meth:`run_days` steps
    the same day without a ``Model`` and yields each day's context at the record point.

    Attributes: ``engine``, ``river`` (the river order), ``plan``
    (:func:`orchestration_plan`), ``ignored_by_quantity_only``, ``tank_names``,
    ``surface_names``, ``arc_names``, ``call_sequence`` (the traced day's ``Ctx.sequence``),
    ``loop_sites`` (its distribute loops), ``last_record`` (the last day's
    :class:`DayRecord`), ``last_call_sequence``, and ``record_decisions`` (default
    ``False``).
    """

    integrates = True
    state_keys: tuple[str, ...] = STATE_KEYS
    outputs: tuple[str, ...] = ()
    edge_kinds: tuple[str, ...] = ("arc",)

    def __init__(self, network: WsimodNetwork, *, pollutants: bool) -> None:
        self.network = network
        self.pollutants = bool(pollutants)
        self.p: Pollutants = network.pollutants if self.pollutants else QUANTITY_ONLY
        self.ignored_by_quantity_only = check_network(network, pollutants=self.pollutants)
        self.engine = Engine(network, pollutants=self.p)
        self.river = river_order(network)
        self.plan = orchestration_plan(network)
        self.record_decisions = False
        self.last_record: DayRecord | None = None
        self.last_call_sequence: list | None = None
        self.last_state: dict[str, Tensor] | None = None
        self.arc_names = [a.name for a in network.arcs]
        self.node_names = [n.name for n in network.nodes]
        nodes = self.engine.nodes
        # the configured stores, before any day runs: the initial state's carried part and
        # the per-day values every day starts from
        self._fresh = {
            name: {k: v.clone() for k, v in node.stores().items()} for name, node in nodes.items()
        }
        self._fresh_arcs = {
            name: {k: v.clone() for k, v in arc.stores().items()}
            for name, arc in self.engine.arcs.items()
        }
        self._internal_arcs = [
            getattr(node, path.split(".")[0]).internal_arc
            for node in nodes.values()
            for path in node.carried_paths
            if path.endswith(".internal_arc.register")
        ]
        self._drivers = self._driver_layout()
        self.call_sequence: list | None = None
        self._trace()
        self._slots, self._labels = self._state_layout()
        self.tank_names = [
            label
            for slot in self._slots
            if slot.key == "wsimod.storage"
            for label in [self._labels["wsimod.storage"][slot.start]]
            if not label[1].startswith("surface:")
        ]
        self.surface_names = [
            (spec.name, surface.name)
            for spec in network.nodes
            if spec.slot_key == "Land"
            for surface in spec.surfaces
        ]
        self._trace_sequence()
        self.key_labels = {key: self._state_field(key) for key in STATE_KEYS}
        self.output_units = {key: self.key_labels[key].unit for key in STATE_KEYS}
        self.input_specs = {
            d.key: self._driver_field(d) for d in self._drivers.values() if self._fits(d)
        }
        self.inputs = (*self._drivers, *STATE_KEYS)

    def to(self, device=None, dtype: torch.dtype | None = None) -> WsimodDay:
        """``Model.to`` hook: the port is float64 on the CPU only (bitwise equality with
        WSIMOD), so a move to another dtype or device is refused by name; float64 on the CPU
        changes nothing."""
        if (dtype is None or dtype == torch.float64) and (
            device is None or torch.device(device).type == "cpu"
        ):
            return self
        raise TypeError(
            "WsimodDay: the WSIMOD port is float64 on CPU only (bitwise equality with WSIMOD); "
            "to(device=..., dtype=...) is not supported"
        )

    def _move_with(self, mover) -> None:
        """The hook ``Model.to`` calls for a noodl-module closure (``layers/_device.py``)."""
        self.to(device=mover.device, dtype=mover.precision)

    def __repr__(self) -> str:
        return f"WsimodDay({len(self.engine.nodes)} nodes, {len(self.engine.arcs)} arcs)"

    # --- drivers (D6) -----------------------------------------------------------------------
    def _node_parameters(self, node: Node) -> dict[str, Tensor]:
        """A node's configured parameters by name, as :meth:`Node.bind` takes them: 0-d, or
        ``(C,)`` for a pollutant-keyed one (``pollutant_load``)."""
        from noodl.apps.wsimod.demand import Distribution
        from noodl.apps.wsimod.sewer import Sewer
        from noodl.apps.wsimod.storage import River, Storage

        out: dict[str, Tensor] = {}
        if isinstance(node, Storage):
            # a River derives its area and capacity (W35): they are no parameters
            fields = ("datum",) if isinstance(node, River) else ("capacity", "area", "datum")
            out.update({f: getattr(node.tank, f) for f in fields})
        if isinstance(node, Sewer):
            tank = node.sewer_tank
            out.update(capacity=tank.capacity, chamber_area=tank.area, chamber_floor=tank.datum)
        if isinstance(node, Distribution):
            out["leakage"] = node.leakage
        out.update(getattr(node, "defaults", {}))
        return out

    def _timearea(self, node: Node) -> tuple[str, dict[int, Tensor]] | None:
        from noodl.apps.wsimod.sewer import Sewer
        from noodl.apps.wsimod.storage import QueueGroundwater

        if isinstance(node, QueueGroundwater):
            return "timearea", node.timearea
        if isinstance(node, Sewer):
            return "pipe_timearea", node.pipe_timearea
        return None

    def _driver_layout(self) -> dict[str, _Driver]:
        network, p, nodes = self.network, self.p, self.engine.nodes
        drivers: dict[str, _Driver] = {}
        values: dict[str, list] = {}

        def add(key: str, label, element, name: str, value, columns) -> None:
            if key not in drivers:
                parameter = name.rsplit(".", 1)[-1]
                drivers[key] = _Driver(
                    key,
                    [],
                    columns,
                    torch.zeros(0),
                    unit=_PARAMETER_UNITS.get(parameter, ""),
                    description=f"{key.split('.')[1]} parameter {parameter!r} (D6)",
                )
                values[key] = []
            drivers[key].labels.append(label)
            drivers[key].targets.append((element, name))
            values[key].append(value)

        pollutant_columns = p.names
        for spec in network.nodes:
            node = nodes[spec.name]
            named = node.driver_columns()
            for name, value in self._node_parameters(node).items():
                value = _f64(value)
                if name in named:  # one value per named column (X10, X34)
                    add(f"wsimod.{spec.cls}.{name}", spec.name, node, name, value, named[name])
                elif value.ndim == 1:  # pollutant-keyed, (K,): absent without pollutants
                    if p.K == 0:
                        continue
                    add(
                        f"wsimod.{spec.cls}.{name}", spec.name, node, name, value, pollutant_columns
                    )
                else:
                    add(f"wsimod.{spec.cls}.{name}", spec.name, node, name, value, None)
            timearea = self._timearea(node)
            if timearea is not None:
                name, fractions = timearea
                add(f"wsimod.{spec.cls}.{name}", spec.name, node, name, fractions, "time")
            for surface in getattr(node, "surfaces", ()):
                for name, value in surface.defaults.items():
                    value = _f64(value)
                    if value.ndim == 1 and p.K == 0:
                        continue
                    add(
                        f"wsimod.{surface.spec.cls}.{name}",
                        (spec.name, surface.name),
                        node,
                        f"surfaces.{surface.name}.{name}",
                        value,
                        pollutant_columns if value.ndim == 1 else None,
                    )
        for key, d in drivers.items():
            if d.columns == "time":  # time-area dicts, (n, L+1), absent keys 0 (D6)
                width = 1 + max(max(fractions) for fractions in values[key])
                rows = []
                for fractions in values[key]:
                    row = torch.zeros(width, dtype=F64)
                    for t, fraction in fractions.items():
                        row[t] = fraction
                    rows.append(row)
                d.columns = tuple(range(width))
                d.template = torch.stack(rows)
            else:
                d.template = torch.stack(values[key])
        for name, unit in (("capacity", "m3/d"), ("preference", "-")):
            key = f"wsimod.Arc.{name}"
            drivers[key] = _Driver(
                key,
                list(self.arc_names),
                None,
                torch.tensor([float(getattr(a, name)) for a in network.arcs], dtype=F64),
                targets=[(self.engine.arcs[a], name) for a in self.arc_names],
                unit=unit,
                description=f"every arc's {name} (D6)",
            )
        dates = network.dates
        readers = forcing_readers(network)
        for variable in _FORCING_UNITS:
            # a catchment's pollutant series (temperature among them) are its concentrations
            names = [
                n
                for n in readers.get(variable, [])
                if variable == "flow" or nodes[n].spec.cls != "Catchment"
            ]
            if not names:
                continue
            series = [[nodes[n].spec.data[(variable, date)] for n in names] for date in dates]
            key = f"wsimod.{variable}"
            drivers[key] = _Driver(
                key,
                list(names),
                None,
                torch.tensor(series, dtype=F64).reshape(len(dates), len(names)),
                targets=[(nodes[n], variable) for n in names],
                forcing=True,
                unit=_FORCING_UNITS[variable],
                description=f"the {variable} series of the nodes that read it",
            )
        catchments = readers.get("flow", [])
        if catchments and p.K:
            series = [
                [[nodes[n].spec.data[(x, date)] for x in p.names] for n in catchments]
                for date in dates
            ]
            drivers["wsimod.concentration"] = _Driver(
                "wsimod.concentration",
                list(catchments),
                p.names,
                torch.tensor(series, dtype=F64).reshape(len(dates), len(catchments), p.K),
                targets=[(nodes[n], "concentration") for n in catchments],
                forcing=True,
                description="every catchment's pollutant series, in the pollutant order",
            )
        return drivers

    def driver_template(self) -> dict[str, Tensor]:
        """Every driver at its configured value (D6), unbatched."""
        return {key: d.template.clone() for key, d in self._drivers.items()}

    @property
    def drivers(self) -> Mapping[str, _Driver]:
        """The driver layout by key: ``labels``, ``columns`` and ``template`` of each."""
        return self._drivers

    @staticmethod
    def _fits(d: _Driver) -> bool:
        """Whether the driver's layout is a :class:`Field`'s: rows, or rows by columns."""
        return not (d.forcing and d.columns is not None)

    def _driver_field(self, d: _Driver) -> Field:
        common = dict(
            role="driver",
            layer=None,
            axis="label",
            unit=d.unit,
            required="optional",
            default=None,
            description=d.description,
            dtype=F64,
        )
        if d.forcing:  # (T, n): the dates along the rows, the readers in the columns
            n = len(d.labels)
            return Field(
                d.key,
                labels=list(self.network.date_labels),
                ordering=f"one row per date, one column per reader {d.labels}",
                n_species=n,
                species=[str(x) for x in d.labels] if n > 1 else None,
                column=n == 1,
                **common,
            )
        columns = len(d.columns) if d.columns is not None else 1
        return Field(
            d.key,
            labels=d.labels,
            ordering=f"order of its {len(d.labels)} labels",
            n_species=columns,
            species=[str(c) for c in d.columns] if d.columns is not None and columns > 1 else None,
            column=d.columns is not None and columns == 1,
            **common,
        )

    # --- state (D10) ------------------------------------------------------------------------
    def _state_layout(self) -> tuple[list[_Slot], dict[str, list]]:
        p = self.p
        width = 1 + p.K
        slots: list[_Slot] = []
        labels: dict[str, list] = {key: [] for key in STATE_KEYS}

        def put(key, element, path, rows_labels, shape) -> None:
            slots.append(_Slot(key, element, path, len(labels[key]), len(rows_labels), shape))
            labels[key].extend(rows_labels)

        for spec in self.network.nodes:
            node = self.engine.nodes[spec.name]
            stores = node.stores()
            tanks = sorted(
                path.rsplit(".", 1)[0]
                for path in node.carried_paths
                if path.endswith(".storage") and not path.startswith("surfaces.")
            )
            for tank in tanks:
                put("wsimod.storage", node, f"{tank}.storage", [(spec.name, tank)], (width,))
            for path in node.carried_paths:
                if path.startswith("surfaces.") and path.endswith(".storage"):
                    name = path.split(".")[1]
                    put("wsimod.storage", node, path, [(spec.name, f"surface:{name}")], (width,))
            for path in node.carried_paths:
                head, _, last = path.rpartition(".")
                if last == "active_storage":
                    put("wsimod.active_storage", node, path, [(spec.name, head)], (width,))
                elif path.endswith(".internal_arc.register"):
                    tank = path.split(".")[0]
                    length = stores[path].shape[-2]
                    arc = f"{spec.name}.{tank}.internal_arc"
                    rows = [(arc, i) for i in range(length)]
                    put("wsimod.travel_queue", node, path, rows, (length, width))
                    present = path.replace(".register", ".present")
                    put("wsimod.travel_queue_present", node, present, rows, (length,))
                elif path in ("liquor", "previous_pulled", "treated"):
                    put(f"wsimod.{path}", node, path, [spec.name], (width,))
        for name, arc in self.engine.arcs.items():
            if isinstance(arc, AltQueueArc):
                length = arc.register.shape[-2]
                rows = [(name, i) for i in range(length)]
                put("wsimod.travel_queue", arc, "register", rows, (length, width))
                put("wsimod.travel_queue_present", arc, "present", rows, (length,))
            elif isinstance(arc, QueueArc):
                n, j = arc.number_of_timesteps, arc.rows.shape[-2]
                rows = [
                    (name, d, i, k) for d in ("push", "pull") for i in range(n) for k in range(j)
                ]
                put("wsimod.arc_queue", arc, "rows", rows, (2, n, j, width))
                put("wsimod.arc_queue_since", arc, "since", [(name, "push"), (name, "pull")], (2,))
        return slots, labels

    def _state_field(self, key: str) -> Field:
        labels = self._labels[key]
        flat = key in _FLAT_KEYS
        width = 1 if flat else 1 + self.p.K
        return Field(
            key,
            role="state",
            layer=None,
            axis="label",
            labels=labels,
            ordering=f"order of {type(self).__name__} (D10)",
            n_species=width,
            species=None if width == 1 else ["volume", *self.p.names],
            column=not flat,
            unit=_STATE_UNITS.get(key, "m3"),
            required="step",
            default=None,
            description=(
                f"state carried by WsimodDay: {_STATE_TEXT[key]}"
                + (
                    ""
                    if flat
                    else "; the unit is the volume column's, each pollutant column is in its "
                    "own unit (kg additive, by name otherwise: W20)"
                )
            ),
            dtype=F64,
        )

    def state_labels(self, key: str) -> list:
        """The row labels of the state key ``key`` (D10)."""
        return list(self._labels[key])

    def _rank(self, key: str) -> int:
        return 1 if key in _FLAT_KEYS else 2

    def initial_state(self) -> dict[str, Tensor]:
        """The configuration's values (spec, Inputs): every tank's ``initial_storage``,
        ``QueueTank`` ``active_storage = storage`` (``tanks.py:478``), registers holding keys 0
        and 1 only (``arcs.py:567``), empty request rows with ``since = n``, and empty liquor,
        previous pulled and treated water."""
        parts: dict[str, list[Tensor]] = {key: [] for key in STATE_KEYS}
        for slot in self._slots:
            if isinstance(slot.element, QueueArc):
                arc = slot.element
                if slot.path == "rows":
                    value = torch.zeros(slot.shape, dtype=F64)
                else:
                    value = torch.full((2,), float(arc.number_of_timesteps), dtype=F64)
            elif isinstance(slot.element, Arc):
                value = self._fresh_arcs[slot.element.name][slot.path]
            else:
                value = self._fresh[slot.element.name][slot.path]
            value = _f64(value)
            rows = (slot.rows,) if slot.key in _FLAT_KEYS else (slot.rows, 1 + self.p.K)
            parts[slot.key].append(value.reshape(rows))
        out = {}
        for key in STATE_KEYS:
            empty = (0,) if key in _FLAT_KEYS else (0, 1 + self.p.K)
            out[key] = torch.cat(parts[key], dim=0) if parts[key] else torch.zeros(empty)
            out[key] = out[key].to(F64)
        return out

    # --- one day ----------------------------------------------------------------------------
    def _batch_shape(self, state: Mapping, drivers: Mapping) -> torch.Size:
        for key in drivers:
            if str(key).startswith("wsimod.") and key not in self._drivers:
                hint = difflib.get_close_matches(str(key), list(self._drivers), n=3, cutoff=0.6)
                raise KeyError(
                    f"WsimodDay: no driver {key!r}"
                    + (f"; did you mean {' or '.join(map(repr, hint))}?" if hint else "")
                    + " (the layout is model.wsimod_day.drivers)"
                )
        shapes = []
        for key in STATE_KEYS:
            value = state[key]
            rank = self._rank(key)
            rows = len(self._labels[key])
            width = () if key in _FLAT_KEYS else (1 + self.p.K,)
            if value.ndim < rank or tuple(value.shape[-rank:]) != (rows, *width):
                raise ValueError(
                    f"state {key!r} of shape {tuple(value.shape)}; expected (..., "
                    f"{', '.join(str(x) for x in (rows, *width))}) (D10)"
                )
            shapes.append(value.shape[:-rank])
        for key, d in self._drivers.items():
            if key not in drivers:
                continue
            value = _f64(drivers[key])
            trailing = d.template.shape
            rank = len(trailing)
            if value.ndim < rank or tuple(value.shape[-rank:]) != tuple(trailing):
                # D19 is a trailing-shape check: a 3a (B, n) batch with B == n == 1 + K_add
                # cannot be told apart from the new layout
                hint = ""
                if key.endswith(".liquor_multiplier"):  # D19, X10
                    hint = (
                        ": 3a's (..., n) layout, the volume entry alone, is gone; the columns "
                        "are 'volume' then the additive pollutants, (..., n, 1 + K_additive)"
                    )
                raise ValueError(
                    f"driver {key!r} of shape {tuple(value.shape)}; expected (..., "
                    f"{', '.join(str(x) for x in trailing)}){hint}"
                )
            shapes.append(value.shape[:-rank])
        return torch.broadcast_shapes(*shapes) if shapes else torch.Size(())

    def _reset_day(self, shape: torch.Size) -> None:
        """The per-day values as WSIMOD resets them (W68, W75), and every arc's
        accumulators (``arcs.py:259-265``), at the day's batch shape: every store a class
        writes under a batched mask is batched before the day starts."""
        for name, node in self.engine.nodes.items():
            fresh = {
                path: torch.broadcast_to(value, (*shape, *value.shape))
                for path, value in self._fresh[name].items()
                if path not in node.carried_paths
            }
            if fresh:
                node.load_stores(fresh)
        for arc in [*self.engine.arcs.values(), *self._internal_arcs]:
            Arc.end_timestep(arc)
            if hasattr(arc, "_seen"):
                arc._seen = dict.fromkeys(arc._seen, 0)
            arc.vqip_in = torch.broadcast_to(arc.vqip_in, (*shape, *arc.vqip_in.shape))
            arc.vqip_out = arc.vqip_in
            arc.flow_in = torch.broadcast_to(arc.flow_in, shape)
            arc.flow_out = arc.flow_in

    def load_state(self, state: Mapping[str, Tensor]) -> None:
        """Put ``state``'s carried stores into the engine's nodes and arcs, with the per-day
        values reset, as every day starts (for inspecting the engine between days)."""
        shape = self._batch_shape(state, {})
        self._reset_day(shape)
        self._load(state, shape)

    def _load(self, state: Mapping, shape: torch.Size) -> None:
        """Every carried store from ``state``, broadcast to the day's batch shape."""
        by_element: dict[int, tuple[object, dict[str, Tensor]]] = {}
        for slot in self._slots:
            value = _f64(state[slot.key])
            rank = self._rank(slot.key)
            value = torch.broadcast_to(value, (*shape, *value.shape[value.ndim - rank :]))
            if rank == 2:
                rows = value[..., slot.start : slot.start + slot.rows, :]
                batch = rows.shape[:-2]
            else:
                rows = value[..., slot.start : slot.start + slot.rows]
                batch = rows.shape[:-1]
            stored = rows.reshape(*batch, *slot.shape)
            by_element.setdefault(id(slot.element), (slot.element, {}))[1][slot.path] = stored
        for element, stores in by_element.values():
            element.load_stores(stores)

    def _dump(self, shape: torch.Size) -> dict[str, Tensor]:
        parts: dict[str, list[Tensor]] = {key: [] for key in STATE_KEYS}
        cache: dict[int, dict[str, Tensor]] = {}
        for slot in self._slots:
            element = slot.element
            if id(element) not in cache:
                cache[id(element)] = element.stores()
            value = _f64(cache[id(element)][slot.path])
            if slot.key == "wsimod.travel_queue_present":
                value = value.to(F64)
            value = torch.broadcast_to(value, (*shape, *slot.shape))
            rows = (slot.rows,) if slot.key in _FLAT_KEYS else (slot.rows, 1 + self.p.K)
            parts[slot.key].append(value.reshape(*shape, *rows))
        out = {}
        for key in STATE_KEYS:
            if parts[key]:
                out[key] = torch.cat(parts[key], dim=-1 if key in _FLAT_KEYS else -2)
            else:
                empty = (0,) if key in _FLAT_KEYS else (0, 1 + self.p.K)
                out[key] = torch.zeros((*shape, *empty), dtype=F64)
        return out

    def _bind(self, drivers: Mapping, k: int) -> None:
        nodes, p = self.engine.nodes, self.p
        bound: dict[str, dict[str, Tensor]] = {name: {} for name in nodes}
        for key, d in self._drivers.items():
            value = _f64(drivers[key]) if key in drivers else d.template
            if d.forcing:
                value = value[..., k, :] if d.columns is None else value[..., k, :, :]
            for i, (element, name) in enumerate(d.targets):
                row = value[..., i] if d.columns is None else value[..., i, :]
                if isinstance(element, Arc):
                    continue
                bound[element.name][name] = row
        if "wsimod.Arc.capacity" in self._drivers:
            capacity = _f64(
                drivers.get("wsimod.Arc.capacity", self._drivers["wsimod.Arc.capacity"].template)
            )
            preference = _f64(
                drivers.get(
                    "wsimod.Arc.preference", self._drivers["wsimod.Arc.preference"].template
                )
            )
            for i, name in enumerate(self.arc_names):
                self.engine.arcs[name].bind(capacity[..., i], preference[..., i])
        for name, node in nodes.items():
            parameters = bound[name]
            if node.spec.cls == "Catchment" and "concentration" not in parameters:
                parameters["concentration"] = torch.zeros(p.K, dtype=F64)
            node.bind(parameters)

    def _run_plan(self, active: Tensor) -> None:
        nodes = self.engine.nodes
        for _, method, names in self.plan:
            for name in names:
                getattr(nodes[name], method)(active=active)
        for name in self.river:  # model.py:1005-1007
            nodes[name].distribute(active=active)

    def _trace(self) -> None:
        """D2: one day of batch shape ``()`` on day 0's configured drivers, under
        ``torch.no_grad()``, whose values are discarded. It fixes every ``QueueArc``'s call
        labels, and so its ``J`` and the state layout."""
        queue_arcs = [a for a in self.engine.arcs.values() if isinstance(a, QueueArc)]
        for arc in queue_arcs:
            arc.begin_tracing()
        with torch.no_grad():
            ctx = self._ctx(torch.Size(()), 0, record_decisions=False)
            self.engine.begin_day(ctx)
            self._bind(self.driver_template(), 0)
            self._run_plan(torch.ones((), dtype=torch.bool))
            self.engine.end_day()
        for arc in queue_arcs:
            arc.end_tracing()

    def _trace_sequence(self) -> None:
        """The tracing day again, from the initial state with every ``J`` fixed (a
        ``QueueArc``'s ``update_queue`` visits all its rows, so its calls depend on ``J``): it
        records the day's call sequence (``Ctx.sequence``), which every later day must repeat
        (D2), and its distribute loops (``loop_sites``). Its values are discarded too."""
        with torch.no_grad():
            generator = self._day(self.initial_state(), self.driver_template(), 0)
            ctx = next(generator)
            _drive(generator)
        self.call_sequence = list(ctx.sequence)
        self.loop_sites = list(ctx.maxiter)
        self.last_record = None
        self.last_call_sequence = None

    def _ctx(self, shape: torch.Size, k: int, *, record_decisions: bool) -> Ctx:
        return Ctx(
            self.p,
            self.network.float_accuracy,
            shape,
            day=k,
            node_names=self.node_names,
            record_decisions=record_decisions,
        )

    def _day(self, state: Mapping, drivers: Mapping, k: int) -> Iterator[Ctx]:
        """Day ``k``; yields its context at the record point and returns the new state."""
        missing = [key for key in STATE_KEYS if key not in state]
        if missing:
            raise KeyError(
                f"WsimodDay: the state has no {missing[0]!r}; build it with "
                "noodl.apps.wsimod.initial_state(model)"
            )
        shape = self._batch_shape(state, drivers)
        ctx = self._ctx(shape, k, record_decisions=self.record_decisions)
        engine = self.engine
        engine.begin_day(ctx)
        self._reset_day(shape)
        self._load(state, shape)
        storage_start = self._held(shape, {})
        self._bind(drivers, k)
        self._run_plan(torch.ones(shape, dtype=torch.bool))
        self.last_record = self._record(ctx, storage_start)
        yield ctx
        engine.end_day()
        self._finish_record(ctx, shape)
        self.last_call_sequence = list(ctx.sequence)
        if self.call_sequence is not None and ctx.sequence != self.call_sequence:
            pairs = zip(ctx.sequence, self.call_sequence, strict=False)
            first = next(
                (i for i, (a, b) in enumerate(pairs) if a != b),
                min(len(ctx.sequence), len(self.call_sequence)),
            )
            mine = ctx.sequence[first] if first < len(ctx.sequence) else "no call"
            traced = self.call_sequence[first] if first < len(self.call_sequence) else "no call"
            raise RuntimeError(
                f"WsimodDay: day {k} made call {first} {mine!r}, the traced day {traced!r} "
                "(D2: the call structure must not change; an internal error)"
            )
        return self._dump(shape)

    def _day_index(self, ctx: StepContext) -> int:
        dates = self.network.date_labels
        where = (
            f"the configuration's {len(dates)} dates ({dates[0]} to {dates[-1]}) are "
            f"t = 0 to {(len(dates) - 1) * SECONDS_PER_DAY:g} s"
        )
        if ctx.dt is None or float(ctx.dt) != SECONDS_PER_DAY:
            raise ValueError(
                f"WsimodDay: dt={ctx.dt!r}; a WSIMOD step is one day, dt={SECONDS_PER_DAY:g} s "
                "(DT_DAYS = 1, constants.py:20)"
            )
        if ctx.t is None:
            raise ValueError(f"WsimodDay: model.step needs t=, the start of the day; {where}")
        t = float(ctx.t)
        k = t / SECONDS_PER_DAY
        if not math.isfinite(k) or k != int(k) or not 0 <= k < len(dates):
            raise ValueError(
                f"WsimodDay: t={t:g} s is not a whole number of days inside the dates; "
                f"{where} (D16)"
            )
        return int(k)

    def __call__(
        self,
        state: Mapping[str, Tensor],
        drivers: Mapping[str, Tensor],
        ctx: StepContext | None,
    ) -> dict[str, Tensor]:
        """Day ``ctx.t / 86400``: the new carried state; ``{}`` for a query (``ctx`` None)."""
        if ctx is None:
            return {}
        return _drive(self._day(state, drivers, self._day_index(ctx)))

    def run_days(
        self,
        state: Mapping[str, Tensor] | None = None,
        drivers: Mapping[str, Tensor] | None = None,
        *,
        days: int | None = None,
        start: int = 0,
    ) -> Iterator[tuple[int, Ctx, Engine]]:
        """Step days ``start, start + 1, ...`` (all the remaining dates, or ``days`` of them)
        from ``state`` (default :meth:`initial_state`) with ``drivers`` (default the configured
        ones), yielding ``(day, ctx, engine)`` after each day's orchestration, record and river
        order, before ``end_timestep``; the day's :class:`DayRecord` is ``last_record``, whose
        ``storage_end`` is ``None`` until the generator resumes and completes it (D14). The
        state after the last day is ``last_state``."""
        state = dict(self.initial_state() if state is None else state)
        drivers = dict(self.driver_template() if drivers is None else drivers)
        total = len(self.network.dates)
        days = total - start if days is None else int(days)
        if start < 0 or days < 0 or start + days > total:
            raise ValueError(
                f"WsimodDay.run_days: days {start} to {start + days - 1} are not inside the "
                f"configuration's {total} dates"
            )
        for k in range(start, start + days):
            generator = self._day(state, drivers, k)
            ctx = next(generator)
            yield k, ctx, self.engine
            state = _drive(generator)
            self.last_state = state

    # --- the record (model.py:1063-1170) ----------------------------------------------------
    def _held(self, shape: torch.Size, stores: dict) -> Tensor:
        """The water held in the carried stores, ``(..., 1 + K_additive)``: every carried store
        but a ``QueueTank``'s active storage and queue (inside its storage), an ``FWTW``'s
        liquor (pushed on the day it is made, ``wtw.py:603-605``), its ``previous_pulled`` (the
        quality of yesterday's pulled water, not water held, W75) and a ``QueueTank``'s
        register (inside its storage), added row by row in the state's order. ``stores`` caches
        ``node.stores()`` by element ``id``."""
        p = self.p
        additive = p.volume_or_additive_mask
        held = torch.zeros((*shape, int(additive.sum())), dtype=F64)
        for slot in self._slots:
            if slot.key in _FLAT_KEYS or slot.key in _NOT_HELD:
                continue
            if slot.key == "wsimod.liquor" and slot.element.spec.cls != "WWTW":
                continue
            if slot.key == "wsimod.travel_queue" and not isinstance(slot.element, Arc):
                continue
            if id(slot.element) not in stores:
                stores[id(slot.element)] = slot.element.stores()
            value = stores[id(slot.element)][slot.path]
            value = torch.broadcast_to(_f64(value), (*shape, *slot.shape))
            for row in value.reshape(*shape, -1, 1 + p.K).unbind(-2):
                held = held + row[..., additive]
        return held

    def _record(self, ctx: Ctx, storage_start: Tensor) -> DayRecord:
        p, shape, nodes = self.p, ctx.shape, self.engine.nodes
        for node in nodes.values():
            node.record_terms()

        def stack(values: Sequence[Tensor], trailing: tuple[int, ...]) -> Tensor:
            if not values:
                return torch.zeros((*shape, 0, *trailing), dtype=F64)
            values = [torch.broadcast_to(_f64(v), (*shape, *trailing)) for v in values]
            return torch.stack(values, dim=len(shape))

        width = (1 + p.K,)
        vqip_out = stack([a.vqip_out for a in self.engine.arcs.values()], width)
        tanks = []
        stores: dict = {id(node): node.stores() for node in nodes.values()}
        for node, attribute in self.tank_names:
            tanks.append(stores[id(nodes[node])][f"{attribute}.storage"])
        tank_vqip = stack(tanks, width)
        surface: dict[str, list[Tensor]] = {f: [] for f in _SURFACE_FIELDS}
        quality = []
        nan = torch.full(shape, float("nan"), dtype=F64)
        for node, name in self.surface_names:
            s = nodes[node]._surface(name)
            for f, attribute in _SURFACE_FIELDS.items():
                if attribute == "storage":
                    surface[f].append(s.storage[..., 0])
                elif attribute == "tank_recharge" and attribute in s.flow_fields:
                    surface[f].append(s.tank_recharge)
                elif attribute in s.flow_fields:
                    surface[f].append(getattr(s, attribute)[..., 0])
                else:  # WSIMOD records no such field for this surface (model.py:1155-1167)
                    surface[f].append(nan)
            quality.append(s.storage[..., 1:])
        held = self._held(shape, stores)
        return DayRecord(
            arc_flow=vqip_out[..., 0],
            arc_quality=vqip_out[..., 1:],
            tank_storage=tank_vqip[..., 0],
            tank_quality=tank_vqip[..., 1:],
            surface={f: stack(v, ()) for f, v in surface.items()},
            surface_quality=stack(quality, (p.K,)),
            terms=dict(ctx.terms),
            dropped=self._additive(ctx.dropped, shape),
            dropped_arcs=self._additive(ctx.dropped_arcs, shape),
            storage_total=held,
            storage_start=storage_start,
            storage_end=None,
            events=self._events(ctx),
            maxiter=dict(ctx.maxiter),
            decisions=list(ctx.decisions) if ctx.record_decisions else None,
        )

    def _additive(self, vqip: Tensor, shape: torch.Size) -> Tensor:
        """``vqip``'s volume and additive columns at the batch shape, ``(..., 1 + K_add)``."""
        p = self.p
        return torch.broadcast_to(vqip, (*shape, 1 + p.K))[..., p.volume_or_additive_mask]

    def _events(self, ctx: Ctx) -> dict[str, Tensor]:
        """``ctx.events`` as ``{condition: (..., n_nodes) bool}``."""
        off = torch.zeros((), dtype=torch.bool)
        return {
            condition: torch.stack(
                [torch.broadcast_to(table.get(n, off), ctx.shape) for n in self.node_names],
                dim=-1,
            )
            for condition, table in ctx.events.items()
        }

    def _finish_record(self, ctx: Ctx, shape: torch.Size) -> None:
        """D14: complete ``last_record`` after ``end_timestep``, on the same object (``wsimod_run``
        holds it from the yield point). ``storage_end`` is the water held now, which the next
        day loads, so the windows meet (X8). ``terms``, ``dropped``, ``dropped_arcs`` and
        ``events`` are snapshotted again, so what ``end_timestep`` records counts in this day
        (D22); it records no decision and no ``maxiter``, and the record keeps those as taken.
        ``storage_total`` keeps the record point's value (X32)."""
        record = self.last_record
        record.storage_end = self._held(shape, {})
        record.terms = dict(ctx.terms)
        record.dropped = self._additive(ctx.dropped, shape)
        record.dropped_arcs = self._additive(ctx.dropped_arcs, shape)
        record.events = self._events(ctx)
