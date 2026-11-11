"""``wsimod_run``: WSIMOD's loop over the configuration's dates, with WSIMOD's records, the
diagnostics and a real mass balance (spec, Entry points, Outputs, Diagnostics).

``wsimod_run(model, state, drivers, days=...)`` steps the model's :class:`WsimodDay` over the
first ``days`` dates (default all) from ``state`` (default the configured one) under
``drivers`` (default the configured ones), the same closure ``model.step`` runs, so the two
agree exactly. It records each day as WSIMOD's ``Model.run(record_all=True)`` does
(``model.py:1063-1170``) and returns :class:`WsimodResults`. ``torch.no_grad()`` is on unless
a state or driver tensor requires grad (W9).

Diagnostics. ``diagnostics``, a dict, receives:

- ``"maxiter"``: ``{(node, method, call index): (..., T) bool}``, whether that distribute loop
  ended at ``MAXITER`` (WSIMOD prints "Maxiter reached ... at <date>");
- ``"events"``: ``{name: (..., T, n_nodes) bool}`` for every name in :data:`EVENT_NAMES` (D13;
  a condition that never fired is all False);
- ``"decisions"``, only when ``diagnostics["decisions"]`` is ``True`` on entry: a list over the
  days of that day's :class:`~noodl.apps.wsimod.engine.Decision` records, in call order;
- ``"mass_balance"``: the :class:`WsimodMassBalance`, also ``WsimodResults.mass_balance``.

The mass balance (D11). Per day, ``inflow`` sums the terms the nodes' ``mass_balance_in`` lists
carry (catchment flows, precipitation, ``FWTW`` deficits, ``UnlimitedDistribution`` supply,
demand loads) and ``outflow`` the ``mass_balance_out`` terms (Waste, evaporation, demand
consumption, unrouted catchment water, unpushed sludge, WTW solids) plus the water WSIMOD
loses without a trace: ``"dropped"`` (the sub-``FLOAT_ACCURACY`` drops: ``QueueArc`` and
``AltQueueArc`` pushes, waiting ``QueueArc`` requests removed on update, the ``Land.run``
percolation push-back, the ``WWTW`` storm-tank rejection and ``make_discharge`` push-back, the
``Sewer`` ponded water) and ``"WWTW.unprocessed_input"`` (water pushed into a ``WWTW`` after its
``calculate_discharge``, which vanishes at ``end_timestep_``, W69). A push below
``FLOAT_ACCURACY`` into a ``QueueTank`` stays in its ``storage``, so it is storage, not a drop
(W49). Arc terms cancel and are not listed. ``storage_change`` is the carried stores' water
at the record point less at the start of the day; per-day values (``Demand``/``WTW`` totals,
``Land`` per-day flows and running balances) are not storage (W75). All are ``(..., T, 1 +
K_additive)``: the volume, then the additive pollutants' masses. ``residual = inflow - outflow
- storage_change`` and closes to rounding, apart from one known gap. One rule holds on every
day: a ``Land``'s outflow is IHACRES's evaporation, not what ``evaporate`` removed (W75), so
``residual`` equals the gap ``E_actual - E_ihacres``, summed over the lands
(WSIMOD's node check prints it, "mass balance error for volume of ..."). The gap is 0 on a day
the soil supplies what IHACRES asks; a shortfall below ``FLOAT_ACCURACY`` fires no
``"inaccurate_evaporation"`` event and is in the residual all the same.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor

from noodl.apps.wsimod.build import _day
from noodl.apps.wsimod.day import DayRecord
from noodl.model import Drivers, Model, State

__all__ = ["EVENT_NAMES", "WsimodMassBalance", "WsimodResults", "wsimod_run"]

EVENT_NAMES = (
    "storage_unable_to_push",
    "demand_deficit",
    "reservoir_spill",
    "wwtw_couldnt_push",
    "service_reservoirs_not_filled",
    "sludge_nowhere",
    "river_cant_push",
    "ponded_water_cant_reenter",
    "groundwater_couldnt_push_all",
    "miscalculated_tank_storage",
    "environmental_not_able_to_push",
    "demand_not_able_to_push",
    "leakage_not_going_to_gw",
    "more_treated_than_input",
    "excess_treated_water",
    "more_backflow_than_vqip",
    "zero_priority",
    "wtw_liquor_zero_denominator",
    "inaccurate_evaporation",
)
"""The warning conditions a run reports (D13): WSIMOD's prints and noodl's own."""

F64 = torch.float64


@dataclass
class WsimodMassBalance:
    """The system's water balance per day, each ``(..., T, 1 + K_additive)`` (volume, then the
    additive pollutants' masses). ``terms`` holds every term by its ``"<Class>.<term>"`` key
    (D11) plus ``"dropped"``; ``residual = inflow - outflow - storage_change``."""

    inflow: Tensor
    outflow: Tensor
    storage_change: Tensor
    residual: Tensor
    terms: dict[str, Tensor]
    dropped_arcs: Tensor | None = None  # part of terms["dropped"] lost inside a QueueArc


@dataclass
class WsimodResults:
    """What WSIMOD's ``Model.run`` records (``model.py:1063-1170``), with a leading batch shape.

    ``arc_flow`` ``(..., T, n_arcs)`` and ``arc_quality`` ``(..., T, n_arcs, K)``: each arc's
    ``vqip_out``. ``tank_names`` ``[(node, attribute)]`` with ``tank_storage``
    ``(..., T, n_tanks)`` and ``tank_quality``: the ``Tank``, ``QueueTank`` and
    ``ResidenceTank`` attributes ``record_all`` reports. ``surface_names`` and ``surface``
    (field to ``(..., T, n_surfaces)``, NaN where WSIMOD records nothing) cover the surfaces of
    the nodes under the slot key ``"Land"``; ``surface_quality`` ``(..., T, n_surfaces, K)``.
    ``dates`` are the labels as written (D20). Volumes are ``m3``, flows ``m3/d``, additive
    pollutants ``kg`` and non-additive ones by name; nothing is converted (W20).
    """

    dates: list[str]
    arc_names: list[str]
    arc_flow: Tensor
    arc_quality: Tensor
    tank_names: list[tuple[str, str]]
    tank_storage: Tensor
    tank_quality: Tensor
    surface_names: list[tuple[str, str]]
    surface: dict[str, Tensor]
    surface_quality: Tensor
    mass_balance: WsimodMassBalance


def _requires_grad(*groups) -> bool:
    return any(
        isinstance(v, Tensor) and v.requires_grad for g in groups if g for v in g.values()
    )


def _stack(records: list[Tensor], shape: torch.Size) -> Tensor:
    """The days' ``(..., X)`` values as ``(..., T, X)``."""
    return torch.stack([torch.broadcast_to(r, (*shape, *r.shape[len(shape) :])) for r in records],
                       dim=len(shape))


def _balance(day, records: list[DayRecord], shape: torch.Size) -> WsimodMassBalance:
    p = day.p
    additive = p.volume_or_additive_mask
    direction: dict[str, str] = {}
    for node in day.engine.nodes.values():
        direction.update(node.mass_balance)
    keys: list[str] = []
    for record in records:
        keys.extend(k for k in record.terms if k not in keys)
    unknown = [k for k in keys if k not in direction]
    if unknown:
        raise RuntimeError(f"mass-balance term {unknown[0]!r} has no declared direction")
    width = int(additive.sum())
    zero = torch.zeros((*shape, len(records), width), dtype=F64)
    terms: dict[str, Tensor] = {}
    inflow, outflow = zero, zero
    for key in keys:
        none = torch.zeros((*shape, 1 + p.K), dtype=F64)
        days = [r.terms.get(key, none)[..., additive] for r in records]
        terms[key] = _stack(days, shape)
        if direction[key] == "in":
            inflow = inflow + terms[key]
        else:
            outflow = outflow + terms[key]
    terms["dropped"] = _stack([r.dropped for r in records], shape)
    outflow = outflow + terms["dropped"]
    storage = _stack([r.storage_total for r in records], shape)
    start = _stack([r.storage_start for r in records], shape)
    change = storage - start
    return WsimodMassBalance(
        inflow=inflow,
        outflow=outflow,
        storage_change=change,
        residual=inflow - outflow - change,
        terms=terms,
        dropped_arcs=_stack([r.dropped_arcs for r in records], shape),
    )


def wsimod_run(
    model: Model,
    state: State | None = None,
    drivers: Drivers | None = None,
    *,
    days: int | None = None,
    diagnostics: dict | None = None,
) -> WsimodResults:
    """Run ``model`` over its dates, or the first ``days`` of them (``1 <= days <=
    len(dates)``), and record each day (module text).

    ``state`` defaults to :func:`~noodl.apps.wsimod.initial_state` and ``drivers`` to
    :func:`~noodl.apps.wsimod.initial_drivers`; either may carry batch dimensions. The run is
    under ``torch.no_grad()`` unless a state or driver tensor requires grad (W9). ``diagnostics``
    receives ``"maxiter"``, ``"events"``, ``"mass_balance"`` and, when it holds
    ``"decisions": True`` on entry, ``"decisions"``.
    """
    from noodl.apps.wsimod.build import initial_drivers, initial_state

    day = _day(model)
    labels = list(day.network.date_labels)
    if not labels:
        raise ValueError("wsimod_run: the model has no dates; give the configuration 'dates'")
    n = len(labels) if days is None else days
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(labels):
        raise ValueError(
            f"wsimod_run: days={days!r}; need 1 <= days <= {len(labels)}, the configuration's "
            "number of dates"
        )
    state = initial_state(model) if state is None else state
    drivers = initial_drivers(model) if drivers is None else drivers
    want_decisions = diagnostics is not None and diagnostics.get("decisions") is True
    records: list[DayRecord] = []
    previous = day.record_decisions
    day.record_decisions = want_decisions
    try:
        # autograd is not forced on: a caller's torch.no_grad() stays in force
        with nullcontext() if _requires_grad(state, drivers) else torch.no_grad():
            for _, _, _ in day.run_days(state, drivers, days=n):
                records.append(day.last_record)
    finally:
        day.record_decisions = previous
        day.last_record = records[-1] if records else None
    shape = records[0].arc_flow.shape[:-1]
    results = WsimodResults(
        dates=labels[:n],
        arc_names=list(day.arc_names),
        arc_flow=_stack([r.arc_flow for r in records], shape),
        arc_quality=_stack([r.arc_quality for r in records], shape),
        tank_names=list(day.tank_names),
        tank_storage=_stack([r.tank_storage for r in records], shape),
        tank_quality=_stack([r.tank_quality for r in records], shape),
        surface_names=list(day.surface_names),
        surface={f: _stack([r.surface[f] for r in records], shape) for f in records[0].surface},
        surface_quality=_stack([r.surface_quality for r in records], shape),
        mass_balance=_balance(day, records, shape),
    )
    if diagnostics is not None:
        off = torch.zeros((*shape, len(day.node_names)), dtype=torch.bool)
        names = [*EVENT_NAMES, *(k for r in records for k in r.events if k not in EVENT_NAMES)]
        diagnostics["maxiter"] = {
            site: _stack([r.maxiter[site] for r in records], shape) for site in records[0].maxiter
        }
        diagnostics["events"] = {
            name: _stack([r.events.get(name, off) for r in records], shape)
            for name in dict.fromkeys(names)
        }
        diagnostics["mass_balance"] = results.mass_balance
        if want_decisions:
            diagnostics["decisions"] = [r.decisions for r in records]
        else:
            diagnostics.pop("decisions", None)
    return results

