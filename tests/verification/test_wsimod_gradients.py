"""Gradients of the WSIMOD port against central differences (spec, Differentiability and
Verification 4; Task 18).

For each parameter the spec names, autograd of a scalar (an arc's flow summed over the days, a
final storage or a pollutant mass) is compared with ``(f(p + h) - f(p - h)) / 2h`` at
``h = 1e-6 * |p|``: ``|ad - fd| <= 1e-6 * max(|ad|, |fd|)``. The run is piecewise smooth (branches
are ``torch.where`` on detached decisions), so each point is also checked off the jumps from the
``"decisions"`` record: the outcome of every active decision, keyed by (day, site, call, round),
is the same in the base and the two ``+-h`` runs, and no active decision has
``|margin| <= 1e-6 * scale`` (``scale`` the largest arc flow of the base run) in any of the
three, except margin-invariant decisions (see ``_jumps``).
3b adds the decay constants and exponents, the WTW process constants, exponents and liquor
multipliers, the soil-temperature weights, the River's biochemistry and the temperature forcing
(Task 15). Each is checked below the decay cap and the River's 50 % caps (a capped process takes
half the DIN or the organic nitrogen whatever its rate: no gradient), off the ``T <= 0`` and dry
guards; negative WTW solids are an event, not a kink. The
River's decisions compare masses (kg), so its margins are measured against the largest arc mass
(``margin_scale="mass"``) instead of the largest arc flow (m3).
3c-i adds the growing surface, its nutrient pool and the surface deposition (Task 11). An entry
may run a window of its case (its dates from ``start`` on, the configured state on that date,
amended by ``initial``) and check further scalars on the same runs (``more``). The adsorption's
Newton tests and the spec's continuous switches are held to a distance in steps instead of the
scale (``STEPPED``). Two tests add the batched gradient of the loads and the deposition (Y58,
Y59) and the zero gradient of ``adosorption_nr_limit``.
This file is also the whole-model check of the hand-written gradients of ``pow_half``,
``exp_python`` and ``pow_python`` (bitwise Python math).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from noodl.apps.wsimod import (
    build_model,
    initial_drivers,
    initial_state,
    read_wsimod_config,
    wsimod_run,
)
from tests.verification._wsimod_cases import CASES

F64 = torch.float64
DATA = Path(__file__).resolve().parents[1] / "data" / "wsimod"
REL = 1e-6  # the step and the agreement
KINK = 1e-6  # |margin| of an active decision, relative to the case scale
NOISE = 1e-12  # a margin change this small (relative to the scale) is rounding
# 3c: decisions whose distance to the jump is measured in steps, 2 |margin| / |margin(+h) -
# margin(-h)| (the parameter changes of h it would take to flip them), not against the scale:
# - the adsorption's Newton tests compare concentrations (mg/l: the step |dx| with 1e-6 xn) and
#   the residual in kg/km2, in no unit of the arcs (their margins are 1e-8 to 1e-5 on every day);
# - the spec's continuous switches (Differentiability): the pool transformation's in_/out_ (the
#   sign of a day's increase, both sides 0 at 0) and erosion's eff_erodedP > 0 (nothing eroded on
#   both sides). Near a freeze or a thaw, or on a day of little runoff, they sit 1e-5 kg from 0,
#   below the mass scale's 4e-5, yet 1e4 steps and more from their switch
STEPPED = (
    "/adsorption/newton:",
    "/soil_pool_transformation/inorganic_",
    "/soil_pool_transformation/organic_",
    "/erosion/phosphorus",
)
STEPS = 1e3  # a STEPPED decision is at least this many steps from its switch


@dataclass(frozen=True)
class Spec:
    case: str  # a CASES name, or "quickstart" (its configuration)
    days: int
    pollutants: bool
    set: (
        tuple  # ("driver", key, label[, column]) | ("scale", key[, column]) | ("state", key, label)
    )
    scalar: tuple  # ("flow", arc) | ("tank", node, attribute) | ("quality", arc, pollutant)
    # | ("final", node, attribute, pollutant): a store's mass after the last day
    point: float | None = None  # the parameter's value; the configured one by default
    # the unit of the decisions' margins: "flow" is the largest arc flow of the base run (m3),
    # "mass" its largest additive-pollutant arc mass (kg). The river's biochemistry compares
    # masses (kg of nitrogen and phosphorus, 1e-5 to 1), which a volume scale (tens of m3) would
    # put above every margin. Every decision of a mass-scaled entry is held to the mass
    # threshold, the volume and temperature sites too (their margins, in m3 and degrees, are
    # far above it, so the bar is only stricter there).
    margin_scale: str = "flow"
    # other driver values held fixed in every run: ((key, label, value), ...)
    also: tuple = ()
    # 3c: the window, the run's first date as an index into the case's dates (its configured
    # initial state then starts on that date), and further scalars checked on the same three runs
    start: int = 0
    more: tuple = ()
    # other initial state values in every run, ((key, label, value), ...): a window's start
    initial: tuple = ()


def _model(case: str, pollutants: bool, start: int = 0, days: int | None = None):
    if case == "quickstart":
        network = read_wsimod_config(DATA / "quickstart")
    else:
        mapping = CASES[case].build()
        if start:  # a window: the case's dates from ``start`` on, the forcing as it is
            mapping["dates"] = mapping["dates"][start : start + days]
        network = read_wsimod_config(mapping)
    return build_model(network, pollutants=pollutants)[0]


def _configured(model, spec: Spec) -> float:
    kind = spec.set[0]
    if kind in ("scale", "month"):
        return 1.0
    day = model.wsimod_day
    if kind == "state":
        row = day.state_labels(spec.set[1]).index(spec.set[2])
        return float(initial_state(model)[spec.set[1]][row, 0])
    layout = day.drivers[spec.set[1]]
    row = layout.labels.index(spec.set[2])
    if len(spec.set) == 3:
        return float(layout.template[row])
    return float(layout.template[row, list(layout.columns).index(spec.set[3])])


def _fixed(spec: Spec) -> dict:
    """``spec.also`` as ``initial_drivers``'s ``values``."""
    values: dict = {}
    for key, label, fixed in spec.also:
        values.setdefault(key, {})[label] = fixed
    return values


def _state(model, spec: Spec, value: dict | None = None):
    """The initial state with ``spec.initial`` and then ``value`` (key -> {label: value})."""
    values: dict = {}
    for key, label, fixed in spec.initial:
        values.setdefault(key, {})[label] = fixed
    for key, rows in (value or {}).items():
        values.setdefault(key, {}).update(rows)
    return initial_state(model, values=values or None)


def _inputs(model, spec: Spec, p):
    kind = spec.set[0]
    if kind == "state":
        _, key, label = spec.set
        return _state(model, spec, {key: {label: p}}), initial_drivers(model)
    if kind == "driver":
        _, key, label, *column = spec.set
        value = {label: p} if not column else {label: {column[0]: p}}
        values = _fixed(spec)
        values.setdefault(key, {}).update(value)
        return _state(model, spec), initial_drivers(model, values=values)
    drivers = initial_drivers(model, values=_fixed(spec) or None)
    if kind == "month":  # p multiplies one label's series on the days of one month ("YYYY-MM")
        _, key, label, month = spec.set
        day = model.wsimod_day
        rows = torch.tensor([d.startswith(month) for d in day.network.date_labels])
        assert bool(rows.any()), month
        series = drivers[key]
        column = torch.zeros(series.shape[-1], dtype=torch.bool)
        column[day.drivers[key].labels.index(label)] = True
        drivers[key] = torch.where(rows[:, None] & column, series * p, series)
        return _state(model, spec), drivers
    # "scale": p multiplies the forcing series (one column of the concentrations)
    _, key, *column = spec.set
    series = drivers[key]
    if column:
        index = list(model.wsimod_day.drivers[key].columns).index(column[0])
        parts = list(series.unbind(-1))
        parts[index] = parts[index] * p
        drivers[key] = torch.stack(parts, dim=-1)
    else:
        drivers[key] = series * p
    return _state(model, spec), drivers


def _value(model, results, scalar):
    kind = scalar[0]
    if kind == "flow":
        return results.arc_flow[:, results.arc_names.index(scalar[1])].sum()
    if kind == "tank":
        return results.tank_storage[-1, results.tank_names.index((scalar[1], scalar[2]))]
    day = model.wsimod_day
    if kind == "final":  # a store's mass after the last day, from the state (a decay tank is
        # not in the tank_* results, X40); "volume" its volume
        _, node, attribute, pollutant = scalar
        row = day.state_labels("wsimod.storage").index((node, attribute))
        column = 0 if pollutant == "volume" else 1 + list(day.p.names).index(pollutant)
        return day.last_state["wsimod.storage"][row, column]
    if kind == "pool":  # a growing surface's nutrient pool after the last day
        _, node, surface, pool, nutrient = scalar
        label = (node, f"surfaces.{surface}.nutrient_pool", pool, nutrient)
        return day.last_state["wsimod.nutrient_pool"][
            day.state_labels("wsimod.nutrient_pool").index(label)
        ]
    if kind == "arcs":  # several arcs' volume (pollutant None) or pollutant, summed over the days
        _, arcs, pollutant = scalar
        columns = [results.arc_names.index(a) for a in arcs]
        if pollutant is None:
            return results.arc_flow[:, columns].sum()
        return results.arc_quality[:, columns, list(day.p.names).index(pollutant)].sum()
    index = list(day.p.names).index(scalar[2])
    return results.arc_quality[:, results.arc_names.index(scalar[1]), index].sum()


def _run(model, spec: Spec, p):
    state, drivers = _inputs(model, spec, p)
    diagnostics: dict = {"decisions": True}
    results = wsimod_run(model, state, drivers, days=spec.days, diagnostics=diagnostics)
    values = [_value(model, results, scalar) for scalar in (spec.scalar, *spec.more)]
    return values, results, diagnostics["decisions"]


def _records(decisions):
    """``{(day, site, call, round): (outcome, margin)}`` of every active decision."""
    return {
        (day, d.site, d.call, d.round): (bool(d.outcome), float(d.margin))
        for day, records in enumerate(decisions)
        for d in records
        if bool(d.active)
    }


def _jumps(runs, scale):
    """``(same, margin, where, steps)`` for the base, ``+h`` and ``-h`` runs' decision records:
    whether every active decision has the same outcome in the three, the smallest ``|margin|``
    over them and the (day, site, call, round) it belongs to, and the fewest steps to a jump of
    the ``STEPPED`` decisions, which ``margin`` leaves out.

    A decision whose margin differs by at most ``NOISE * scale`` between the ``+h`` and ``-h``
    runs does not depend on the parameter within the step, and is no jump. It is a comparison
    of a quantity that is exactly 0 with ``FLOAT_ACCURACY`` (the margin is the constant
    ``-1e-11``), or of two copies of one quantity (``max(vqip - excess, 0)`` once ``excess`` is
    ``vqip``; a full arc's remaining capacity against a zero request), which agree to rounding,
    so its branches coincide and its outcome is noise. A tie or threshold that depends on the
    parameter moves by ``slope * 2h`` between the runs, far above ``NOISE * scale``.
    A decision active in some runs only is a different outcome, unless it too compares
    quantities at rounding level (the branch that opened it was rounding noise)."""
    base, up, down = runs
    same, margin, where, steps = True, float("inf"), None, (float("inf"), None)
    for key in set().union(*runs):
        found = [run.get(key) for run in runs]
        if None in found:  # active in some runs only: a call the noise of another decision opened
            if any(f is not None and abs(f[1]) > NOISE * scale for f in found):
                same = False
        elif abs(up[key][1] - down[key][1]) <= NOISE * scale:
            if len({f[0] for f in found}) > 1 and max(abs(f[1]) for f in found) > NOISE * scale:
                same = False
            continue
        elif len({f[0] for f in found}) > 1:
            same = False
        elif any(site in key[1] for site in STEPPED):
            away = 2 * min(abs(f[1]) for f in found) / abs(up[key][1] - down[key][1])
            if away < steps[0]:
                steps = (away, key)
        else:
            smallest = min(abs(f[1]) for f in found)
            if smallest < margin:
                margin, where = smallest, key
    return same, margin, where, steps


def check(spec: Spec) -> dict:
    """The three evaluations: autograd at the point, central differences, the decision record.
    ``scalar``, ``ad`` and ``fd`` are lists, one entry per scalar (``spec.scalar``, then
    ``spec.more``)."""
    model = _model(spec.case, spec.pollutants, spec.start, spec.days)
    point = _configured(model, spec) if spec.point is None else spec.point
    assert point != 0.0
    h = REL * abs(point)
    p = torch.tensor(point, dtype=F64, requires_grad=True)
    values, results, base = _run(model, spec, p)
    ad = [
        torch.autograd.grad(v, p, retain_graph=k + 1 < len(values))[0] for k, v in enumerate(values)
    ]
    with torch.no_grad():
        up, _, up_decisions = _run(model, spec, torch.tensor(point + h, dtype=F64))
        down, _, down_decisions = _run(model, spec, torch.tensor(point - h, dtype=F64))
    scale = float(results.arc_flow.detach().abs().max())
    if spec.margin_scale == "mass":
        additive = model.wsimod_day.p.volume_or_additive_mask[1:]
        scale = float(results.arc_quality.detach().abs()[..., additive].max())
    same, margin, where, steps = _jumps(
        [_records(d) for d in (base, up_decisions, down_decisions)], scale
    )
    return {
        "point": point,
        "scalar": [float(v.detach()) for v in values],
        "ad": [float(g) for g in ad],
        "fd": [float((u - d) / (2 * h)) for u, d in zip(up, down, strict=True)],
        "scale": scale,
        "same_outcomes": same,
        "margin": margin,
        "where": where,
        "steps": steps,
    }


# quickstart's first months are dry: 200 days reach the land's response
_Q = dict(case="quickstart", days=200, pollutants=False)
_RURAL, _URBAN = ("my_land", "rural"), ("my_land", "urban")


def _river(parameter, pollutant, point, *fixed, **kwargs) -> Spec:
    """``river_nutrients`` for 120 days: the final ``pollutant`` mass of ``r_low`` against
    ``parameter`` (its configured value unless ``point``). ``fixed`` are the other rates of
    ``r_low`` held at the given values, to keep their 50 % caps off the scalar: a capped
    process takes half the DIN, ``min(rate, 0.5 din)``, whatever its rate, so it carries no
    gradient. The configured rates (``_R_LOW_OVERRIDES``) bind every cap in the last weeks."""
    return Spec(
        "river_nutrients",
        120,
        True,
        ("driver", f"wsimod.River.{parameter}", "r_low")
        if parameter
        else ("scale", "wsimod.temperature"),
        ("final", "r_low", "tank", pollutant),
        point,
        margin_scale="mass",
        also=tuple((f"wsimod.River.{name}", "r_low", value) for name, value in fixed),
        **kwargs,
    )


# DIN-limited rates: denitrification 2e-7 and macrophyte uptake 1e-6 (configured 2e-4 and 1e-2)
# stay below 0.5 din on every day of the run; production 1e-2 (configured 5e-2) likewise
_SMALL_DENITRIFICATION, _SMALL_UPTAKE, _SMALL_PRODUCTION = (
    ("denpar_w", 2e-7),
    ("muptNpar", 1e-6),
    ("prodNpar", 1e-2),
)
# production far above its cap (0.5 din binds every day): it carries no gradient, and the
# production switch's margin (minprodN, 1e-6 at 0.05) grows with it, 10x clear of the threshold
_CAPPED_PRODUCTION = ("prodNpar", 0.5)

SPECS = {
    "Arc.capacity": Spec(
        "capacity_first_come",
        40,
        False,
        ("driver", "wsimod.Arc.capacity", "hub_to_sea"),
        ("flow", "hub_to_sea"),
    ),
    "Arc.preference": Spec(
        "three_out_arcs",
        40,
        False,
        ("driver", "wsimod.Arc.preference", "junction_to_a"),
        ("flow", "junction_to_a"),
        1.37,  # 1.0 ends a distribute loop on a rounding tie on days 9 and 29
    ),
    "Groundwater.residence_time": Spec(
        set=("driver", "wsimod.Groundwater.residence_time", "my_groundwater"),
        scalar=("flow", "baseflow"),
        **_Q,
    ),
    "PerviousSurface.field_capacity": Spec(
        set=("driver", "wsimod.PerviousSurface.field_capacity", _RURAL),
        scalar=("flow", "percolation"),
        **_Q,
    ),
    "PerviousSurface.ihacres_p": Spec(
        set=("driver", "wsimod.PerviousSurface.ihacres_p", _RURAL),
        scalar=("flow", "percolation"),
        **_Q,
    ),
    "PerviousSurface.et0_coefficient": Spec(
        set=("driver", "wsimod.PerviousSurface.et0_coefficient", _RURAL),
        scalar=("flow", "percolation"),
        **_Q,
    ),
    "ImperviousSurface.et0_to_e": Spec(
        set=("driver", "wsimod.ImperviousSurface.et0_to_e", _URBAN),
        scalar=("flow", "urban_drainage"),
        **{**_Q, "days": 160},  # day 162: the sewer sits at its 0.04 capacity edge
    ),
    "Land.surface_residence_time": Spec(
        set=("driver", "wsimod.Land.surface_residence_time", "my_land"),
        scalar=("flow", "runoff"),
        point=3.0,  # 1.0 leaves a ~1e-9 tank tail on the emptying branch
        **_Q,
    ),
    "Land.subsurface_residence_time": Spec(
        set=("driver", "wsimod.Land.subsurface_residence_time", "my_land"),
        scalar=("tank", "my_land", "subsurface_runoff"),
        **_Q,
    ),
    "Land.percolation_residence_time": Spec(
        set=("driver", "wsimod.Land.percolation_residence_time", "my_land"),
        scalar=("flow", "percolation"),
        **_Q,
    ),
    "River.velocity": Spec(
        "river_chain",
        40,
        False,
        ("driver", "wsimod.River.velocity", "r_mid"),
        ("flow", "mid_to_low"),
    ),
    "River.mrf": Spec(
        "river_chain",
        40,
        False,
        ("driver", "wsimod.River.mrf", "r_mid"),
        ("flow", "mid_to_low"),
    ),
    "WWTW.treatment_throughput_capacity": Spec(
        "wwtw_storm_tank",
        40,
        False,
        ("driver", "wsimod.WWTW.treatment_throughput_capacity", "works"),
        ("flow", "works_to_river"),
        27.3,  # 30 sits on a stormwater-tank tie on days 35 and 38
    ),
    "FWTW.treatment_throughput_capacity": Spec(
        # the service reservoir fills to its capacity within a week whatever the throughput, so
        # the throughput is read on the first 3 days, where it limits what is treated
        "fwtw_deficit_sludge",
        3,
        False,
        ("driver", "wsimod.FWTW.treatment_throughput_capacity", "plant"),
        ("tank", "plant", "service_reservoir_tank"),
    ),
    "ResidentialDemand.per_capita": Spec(
        "residential_demand",
        40,
        False,
        ("driver", "wsimod.ResidentialDemand.per_capita", "homes"),
        ("flow", "supply_to_homes"),
    ),
    "Distribution.leakage": Spec(
        "distribution_leakage",
        40,
        False,
        ("driver", "wsimod.Distribution.leakage", "dist_gw"),
        ("flow", "leak_to_aq"),
    ),
    "Sewer.pipe_timearea": Spec(
        "sewer_timearea",
        40,
        False,
        ("driver", "wsimod.Sewer.pipe_timearea", "sewer", 1),
        ("flow", "sewer_to_sea"),
    ),
    "precipitation": Spec(
        set=("scale", "wsimod.precipitation"),
        scalar=("flow", "catchment_outflow"),
        point=1.02,  # 1.0 puts day 162's sewer inflow on the 0.04 capacity edge
        **_Q,
    ),
    "catchment flow": Spec(
        "three_out_arcs",
        9,  # the first 9 of the case's 40 days: day 9 is a capacity tie at scale 1
        False,
        ("scale", "wsimod.flow"),
        ("flow", "junction_to_a"),
    ),
    "initial storage": Spec(
        "capacity_first_come",
        40,
        False,
        ("state", "wsimod.storage", ("res", "tank")),
        ("flow", "res_to_hub"),
    ),
    "catchment concentration": Spec(
        "three_out_arcs",
        40,
        True,
        ("scale", "wsimod.concentration", "phosphate"),
        ("quality", "junction_to_a", "phosphate"),
    ),
    "Demand.pollutant_load": Spec(
        "demand_constant_and_loads",
        40,
        True,
        ("driver", "wsimod.Demand.pollutant_load", "town", "phosphate"),
        ("quality", "town_to_sea", "phosphate"),
    ),
    # --- 3b: decay, treatment, soil temperature, the river's biochemistry (Task 15) ---
    "Storage.decays.constant": Spec(
        # the case's flow drains the pond within two days (and the later days' water carries
        # almost no phosphate): decay acts on the day-1 phosphate the pond holds at its end
        "decay_storage",
        1,
        True,
        ("driver", "wsimod.Storage.decays.constant", "pond", "phosphate"),
        ("final", "pond", "tank", "phosphate"),
    ),
    "Storage.decays.exponent": Spec(
        # the case's flow drains the pond within two days (and the later days' water carries
        # almost no phosphate): decay acts on the day-1 phosphate the pond holds at its end
        "decay_storage",
        1,
        True,
        ("driver", "wsimod.Storage.decays.exponent", "pond", "phosphate"),
        ("final", "pond", "tank", "phosphate"),
    ),
    "RiverReservoir.decays.constant": Spec(
        "decay_storage",
        3,  # the reservoir holds phosphate through the first days only
        True,
        ("driver", "wsimod.RiverReservoir.decays.constant", "rr", "phosphate"),
        ("final", "rr", "tank", "phosphate"),
    ),
    "RiverReservoir.decays.exponent": Spec(
        "decay_storage",
        3,  # the reservoir holds phosphate through the first days only
        True,
        ("driver", "wsimod.RiverReservoir.decays.exponent", "rr", "phosphate"),
        ("final", "rr", "tank", "phosphate"),
    ),
    # the receiver's (feed) final phosphate is 0 (the reservoir is drained by day 12): the
    # phosphate the arc delivers into it, j_to_feed, is the scalar
    "DecayArc.decays.constant": Spec(
        "decay_arc",
        12,
        True,
        ("driver", "wsimod.DecayArc.decays.constant", "src2_to_j", "phosphate"),
        ("quality", "j_to_feed", "phosphate"),
    ),
    # DecayQueueTank (X40: the queue holds the water a day): the gw_to_res arc's phosphate
    "QueueGroundwater.decays.constant": Spec(
        "decay_queue_groundwater",
        20,
        True,
        ("driver", "wsimod.QueueGroundwater.decays.constant", "gw", "phosphate"),
        ("quality", "gw_to_res", "phosphate"),
    ),
    "WWTW.process_parameters.constant": Spec(
        "wtw_treatment",
        20,
        True,
        ("driver", "wsimod.WWTW.process_parameters.constant", "works", "phosphate"),
        ("quality", "works_to_river", "phosphate"),
    ),
    "WWTW.process_parameters.exponent": Spec(
        "wtw_treatment",
        20,
        True,
        ("driver", "wsimod.WWTW.process_parameters.exponent", "works", "phosphate"),
        ("quality", "works_to_river", "phosphate"),
    ),
    "WWTW.liquor_multiplier": Spec(
        "wtw_treatment",
        20,
        True,
        ("driver", "wsimod.WWTW.liquor_multiplier", "works", "phosphate"),
        ("quality", "works_to_river", "phosphate"),
    ),
    "soil_temp_w_prev": Spec(
        "soil_temperature",
        40,
        True,
        ("driver", "wsimod.PerviousSurface.soil_temp_w_prev", ("moor", "heath")),
        ("quality", "moor_to_aquifer", "temperature"),
    ),
    "soil_temp_w_air": Spec(
        "soil_temperature",
        40,
        True,
        ("driver", "wsimod.PerviousSurface.soil_temp_w_air", ("moor", "heath")),
        ("quality", "moor_to_aquifer", "temperature"),
    ),
    "soil_temp_w_deep": Spec(
        "soil_temperature",
        40,
        True,
        ("driver", "wsimod.PerviousSurface.soil_temp_w_deep", ("moor", "heath")),
        ("quality", "moor_to_aquifer", "temperature"),
    ),
    "temperature (decay)": Spec(
        "decay_storage",
        1,
        True,
        ("scale", "wsimod.temperature"),
        ("final", "pond", "tank", "phosphate"),
    ),
    "temperature (decay, reservoir)": Spec(
        "decay_storage",
        3,
        True,
        ("scale", "wsimod.temperature"),
        ("final", "rr", "tank", "phosphate"),
    ),
    "temperature (treatment)": Spec(
        "wtw_treatment",
        20,
        True,
        ("scale", "wsimod.concentration", "temperature"),
        ("quality", "works_to_river", "phosphate"),
    ),
    # the nitrate scalar: production off its cap (0.01); the org-nitrogen scalar: production on
    # its cap, which takes 0.5 din, so that denitrification and uptake reach org-N through din
    "River.denpar_w [nitrate]": _river(
        "denpar_w", "nitrate", _SMALL_DENITRIFICATION[1], _SMALL_PRODUCTION, _SMALL_UPTAKE
    ),
    "River.denpar_w [org-nitrogen]": _river(
        "denpar_w", "org-nitrogen", _SMALL_DENITRIFICATION[1], _SMALL_UPTAKE
    ),
    "River.halfsatINwater [nitrate]": _river(
        # 2e-3 (configured) gives finite-difference noise on a small slope (rel 5e-7)
        "halfsatINwater",
        "nitrate",
        1e-2,
        _SMALL_DENITRIFICATION,
        _SMALL_PRODUCTION,
        _SMALL_UPTAKE,
    ),
    "River.halfsatINwater [org-nitrogen]": _river(
        "halfsatINwater", "org-nitrogen", None, _SMALL_DENITRIFICATION, _SMALL_UPTAKE
    ),
    "River.prodNpar [nitrate]": _river(
        "prodNpar", "nitrate", _SMALL_PRODUCTION[1], _SMALL_DENITRIFICATION, _SMALL_UPTAKE
    ),
    "River.prodNpar [org-nitrogen]": _river(
        "prodNpar", "org-nitrogen", _SMALL_PRODUCTION[1], _SMALL_DENITRIFICATION, _SMALL_UPTAKE
    ),
    "River.muptNpar [nitrate]": _river(
        "muptNpar", "nitrate", _SMALL_UPTAKE[1], _SMALL_DENITRIFICATION, _SMALL_PRODUCTION
    ),
    "River.muptNpar [org-nitrogen]": _river(
        "muptNpar", "org-nitrogen", _SMALL_UPTAKE[1], _SMALL_DENITRIFICATION
    ),
    "River.T_wdays [nitrate]": _river(
        # default entry: the lagged temperatures and window_sum's gradient. Production is held
        # at 0.5, far above its cap: the production switch (minprodN = 0, day 11) then has a margin
        # of 2e-5, 18x the threshold (at the configured 0.05 it was 2e-6)
        "T_wdays",
        "nitrate",
        None,
        _SMALL_DENITRIFICATION,
        _SMALL_UPTAKE,
        _CAPPED_PRODUCTION,
    ),
    "River.T_wdays [org-nitrogen]": _river(
        # 3 puts a production switch (minprodN = 0, day 11) within 1e-6 of the margin scale
        "T_wdays",
        "org-nitrogen",
        4.5,
        _SMALL_DENITRIFICATION,
        _SMALL_UPTAKE,
        _SMALL_PRODUCTION,
    ),
    "temperature (river) [nitrate]": _river(
        None, "nitrate", None, _SMALL_DENITRIFICATION, _SMALL_UPTAKE
    ),
    "temperature (river) [org-nitrogen]": _river(
        # 0.8 to 1.15 each put a production switch (minprodN = 0, days 9 to 22) within 1e-6
        None,
        "org-nitrogen",
        1.2,
        _SMALL_DENITRIFICATION,
        _SMALL_UPTAKE,
        _SMALL_PRODUCTION,
    ),
}

# --- 3c-i: the growing surface, its nutrient pool and the surface deposition (Task 11) ---------
_G = "wsimod.GrowingSurface."
_WHEAT = ("land", "wheat")
_WHEAT_DATA = ("land", "surfaces.wheat")  # the surface data rows
_SOIL_TANK = ("land", "surface:wheat")
_LAND_OUT = ("land_to_aquifer", "land_to_brook")
_DIN = ("pool", "land", "wheat", "dissolved_inorganic_pool", "N")
_ADS_P = ("pool", "land", "wheat", "adsorbed_inorganic_pool", "P")
_OUT_VOLUME = ("arcs", _LAND_OUT, None)
_SOIL_VOLUME = ("final", "land", "surface:wheat", "volume")
# growing_spring's humus pool is empty and nothing fills it (the configured soil has no humus,
# degrhpar moves no fast nitrogen or phosphorus into it in these windows): erosion then takes all
# its phosphorus from the adsorbed pool (erode_P, nutrient_pool.py:367-394) and the out-arc
# org-phosphorus does not depend on it. 60 kg of humus phosphorus at the window's start opens the
# org-phosphorus path, and keeps every day's eroded phosphorus (the erosion/phosphorus margin)
# above the threshold
_HUMUS_P = (
    ("wsimod.nutrient_pool", ("land", "surfaces.wheat.nutrient_pool", "humus_pool", "P"), 60.0),
)


def _no_nitrate(parameter, *column) -> Spec:
    """``growing_no_nitrate`` from 2004-09-05 (run day 340) for 120 days, the soil starting at
    1000 m3: stressed (``Ks < 1``, the depletion above ``RAW``) on the first days, then wetted
    above the outflow threshold by the winter rain, so that the stress coefficient reaches the
    out-arc volume. From day 0 the crop is never stressed in 120 days (``ET_depletion_factor``
    has no gradient), and over the summer it is stressed with no outflow after it."""
    return Spec(
        "growing_no_nitrate",
        120,
        True,
        ("driver", f"{_G}{parameter}", _WHEAT, *column),
        _OUT_VOLUME,
        start=340,
        more=(_SOIL_VOLUME,),
        initial=(("wsimod.storage", _SOIL_TANK, 1000.0),),
    )


def _spring(setting, scalar, *more, point=None, start=0, days=120, **kwargs) -> Spec:
    """``growing_spring`` for ``days`` from run day ``start`` (0: 2003-10-01), its decisions
    held to the mass scale."""
    return Spec(
        "growing_spring",
        days,
        True,
        setting,
        scalar,
        point,
        margin_scale="mass",
        start=start,
        more=more,
        **kwargs,
    )


def _growing(parameter, scalar, *more, **kwargs) -> Spec:
    return _spring(("driver", f"{_G}{parameter}", _WHEAT), scalar, *more, **kwargs)


def _pool(parameter) -> Spec:
    return _spring(("driver", f"{_G}nutrient_pool.{parameter}", _WHEAT, "N"), _DIN)


def _month(variable, month, pollutant="nitrate") -> Spec:
    return _spring(
        ("month", f"wsimod.{variable}", _WHEAT_DATA, month), ("final", *_SOIL_TANK, pollutant)
    )


_NITRATE_OUT = ("arcs", _LAND_OUT, "nitrate")
_PHOSPHATE_OUT = ("arcs", _LAND_OUT, "phosphate")
_EROSION = (("arcs", _LAND_OUT, "solids"), ("arcs", _LAND_OUT, "org-phosphorus"))
# the crop window: from 2004-01-19 (run day 110), sown on window day 71 (2004-03-30); uptake1 at 5
# (configured 15): at 15 the crop empties the dissolved inorganic pool within weeks of the sowing,
# the uptake is then capped by the pool on most days and the final DIN.N is insensitive (central
# differences at the rounding floor, 3e-6)
_CROP = dict(start=110)
_UPTAKE1 = 5.0
# the cover window: from 2004-01-29 (run day 120), sown on window day 61, the covers growing to
# the end
_COVER = dict(start=120, initial=_HUMUS_P)

SPECS_3C = {
    # the crop calendar and the water stress (calc_crop_cover) without a nutrient pool
    "GrowingSurface.crop_factor_stages[3]": _no_nitrate("crop_factor_stages", 3),
    "GrowingSurface.ET_depletion_factor": _no_nitrate("ET_depletion_factor"),
    "GrowingSurface.rooting_depth": _no_nitrate("rooting_depth"),
    # uptake (with the crop)
    "GrowingSurface.uptake1": _growing("uptake1", _DIN, _NITRATE_OUT, point=_UPTAKE1, **_CROP),
    "GrowingSurface.uptake3": _growing(
        "uptake3", _DIN, _NITRATE_OUT, also=((f"{_G}uptake1", _WHEAT, _UPTAKE1),), **_CROP
    ),
    # denitrification and the soil-moisture factor, from 2003-10-01: the soil dries below field
    # capacity in November (exppar, thetapow) and is saturated in winter (satact)
    "GrowingSurface.denpar": _growing("denpar", _DIN, _NITRATE_OUT),
    "GrowingSurface.hsatINs": _growing("hsatINs", _DIN, _NITRATE_OUT),
    "GrowingSurface.exppar": _growing("exppar", _DIN, _NITRATE_OUT),
    "GrowingSurface.satact": _growing("satact", _DIN, _NITRATE_OUT),
    "GrowingSurface.thetapow": _growing("thetapow", _DIN, _NITRATE_OUT),
    # adsorption: kfr at 20 (configured 153.7, at which the equilibrium hardly moves: ads.P changes
    # by 5e-6 of itself for a doubled kfr, below what central differences resolve); nfr at 0.6
    # (configured 1/2.6, likewise, 2e-5)
    "GrowingSurface.kfr": _growing("kfr", _ADS_P, _PHOSPHATE_OUT, point=20.0),
    "GrowingSurface.nfr": _growing("nfr", _ADS_P, _PHOSPHATE_OUT, point=0.6),
    "GrowingSurface.kadsdes": _growing("kadsdes", _ADS_P, _PHOSPHATE_OUT),
    # erosion: the window from 2003-10-01 has 30 days above 5 mm of rain
    "GrowingSurface.erodibility": _growing("erodibility", *_EROSION, initial=_HUMUS_P),
    "GrowingSurface.sreroexp": _growing("sreroexp", *_EROSION, initial=_HUMUS_P),
    "GrowingSurface.srfilt": _growing("srfilt", *_EROSION, initial=_HUMUS_P),
    # the covers (calc_crop_cover in the season, Y61): through erosion
    "GrowingSurface.crop_cover_max": _growing("crop_cover_max", *_EROSION, **_COVER),
    "GrowingSurface.ground_cover_max": _growing("ground_cover_max", *_EROSION, **_COVER),
    # the pool's rates and fractions (the N column)
    "NutrientPool.minfpar": _pool("minfpar"),
    "NutrientPool.immobdpar": _pool("immobdpar"),
    "NutrientPool.fraction_manure_to_dissolved_inorganic": _pool(
        "fraction_manure_to_dissolved_inorganic"
    ),
    # the surface data, one month each: srp reaches the tank's phosphate, never its nitrate
    # (adjust_vqip_to_liquid scales the P species by P only, land.py:1594-1603)
    "nhx-fertiliser": _month("nhx-fertiliser", "2003-11"),
    "noy-dry": _month("noy-dry", "2003-11"),
    "srp-manure": _month("srp-manure", "2004-01", "phosphate"),
    # the Land's temperature through tdf: from 2004-04-04 (run day 186) for 70 days the soil stays
    # between 6 and 20 degC (off the 5 degC jump and the 0 degC kink, L12)
    "temperature (tdf)": _spring(("scale", "wsimod.temperature"), _DIN, start=186, days=70),
    # Y58, Y59: the impervious load and the wet deposition, 20 days of surface_deposition (a
    # gradient run of it costs about 0.1 s a day)
    "ImperviousSurface.pollutant_load": Spec(
        "surface_deposition",
        20,
        True,
        ("driver", "wsimod.ImperviousSurface.pollutant_load", ("suburb", "roofs"), "phosphate"),
        ("arcs", ("suburb_to_storm", "suburb_to_brook"), "phosphate"),
    ),
    "srp-wet": Spec(
        "surface_deposition",
        20,
        True,
        ("scale", "wsimod.srp-wet"),
        ("arcs", ("suburb_to_storm", "suburb_to_brook"), "phosphate"),
    ),
}
SPECS.update(SPECS_3C)


# the quickstart runs take about 20 s each: three stay in the default selection, the rest are slow;
# so are the 3b runs listed below them
SLOW = {
    "PerviousSurface.ihacres_p",
    "PerviousSurface.et0_coefficient",
    "Land.surface_residence_time",
    "Land.subsurface_residence_time",
    "Land.percolation_residence_time",
    "precipitation",
    # 3b: each family keeps its cheapest entry in the default selection (the decays of the pond, the
    # reservoir and the arc, the works' treatment, one soil weight, River.T_wdays [nitrate]); the
    # other river runs take 120 days of two rivers, 10 s and more each
    "soil_temp_w_air",
    "soil_temp_w_deep",
    "River.denpar_w [org-nitrogen]",
    "River.halfsatINwater [nitrate]",
    "River.halfsatINwater [org-nitrogen]",
    "River.prodNpar [nitrate]",
    "River.prodNpar [org-nitrogen]",
    "River.muptNpar [nitrate]",
    "River.muptNpar [org-nitrogen]",
    "River.denpar_w [nitrate]",
    "River.T_wdays [org-nitrogen]",
    "temperature (river) [nitrate]",
    "temperature (river) [org-nitrogen]",
    # 3c: every growing run takes 15-30 s alone (120 days with the pool, three runs per entry);
    # the surface_deposition entries stay in the default selection
    *(name for name, spec in SPECS_3C.items() if spec.case != "surface_deposition"),
}


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n for n in SPECS],
)
def test_gradient_matches_central_differences(name):
    spec = SPECS[name]
    r = check(spec)
    scalars = (spec.scalar, *spec.more)
    for scalar, value, ad, fd in zip(scalars, r["scalar"], r["ad"], r["fd"], strict=True):
        rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-300)
        print(
            f"GRAD | {name} | {spec.case} | {spec.start}+{spec.days} | {scalar} | "
            f"p={r['point']:.6g} | scalar={value:.9g} | ad={ad:.10g} | fd={fd:.10g} | "
            f"rel={rel:.1e} | min margin={r['margin']:.3g} (scale {r['scale']:.4g}) at "
            f"{r['where']} | steps {r['steps'][0]:.3g} at {r['steps'][1]}"
        )
    assert r["same_outcomes"], f"{name}: the +-h runs' active decisions differ"
    assert r["margin"] > KINK * r["scale"], f"{name}: an active decision is within {KINK} of a jump"
    assert r["steps"][0] > STEPS, f"{name}: a decision is within {STEPS} steps of its switch: {r}"
    for scalar, ad, fd in zip(scalars, r["ad"], r["fd"], strict=True):
        assert ad != 0.0, f"{name}: {scalar} does not depend on the parameter"
        assert abs(ad - fd) <= REL * max(abs(ad), abs(fd)), (scalar, r)


# --- 3c: batched gradients (Y58, Y59) and the adsorption limit -----------------------------------
_SUBURB_OUT = ("suburb_to_storm", "suburb_to_brook")
_BATCH_DAYS = 10  # every day reads the loads and the deposition: no month boundary is needed


def _deposition_drivers(drivers, load, dry, wet):
    """``surface_deposition``'s drivers with the roofs' phosphate load ``load`` and the dry
    ``srp`` and wet ``nhx`` series of both surfaces multiplied by ``dry`` and ``wet``: numbers, or
    one value per instance (the leading batch dimension)."""
    key = "wsimod.ImperviousSurface.pollutant_load"
    loads = drivers[key]  # (surfaces, pollutants): roofs then yards, phosphate first
    roofs = torch.zeros_like(loads, dtype=torch.bool)
    roofs[0, 0] = True
    if load.dim():
        loads = loads.expand(load.shape[0], *loads.shape)

    def per_instance(x):
        return x.reshape(-1, 1, 1) if x.dim() else x

    return {
        **drivers,
        key: torch.where(roofs, per_instance(load), loads),
        "wsimod.srp-dry": drivers["wsimod.srp-dry"] * per_instance(dry),
        "wsimod.nhx-wet": drivers["wsimod.nhx-wet"] * per_instance(wet),
    }


def test_a_batched_gradient_equals_its_single_runs():
    """Y58, Y59: two ``surface_deposition`` instances differ in the roofs' phosphate load
    (``pollutant_load``), the dry ``srp`` and the wet ``nhx`` deposition forcing. Each instance's
    out-arc phosphate and ammonia and their gradients with respect to the three equal its single
    run's (``torch.equal``), and the two instances differ."""
    model = _model("surface_deposition", True)
    names = list(model.wsimod_day.p.names)
    state, drivers = initial_state(model), initial_drivers(model)
    values = {"load": (2e-6, 7e-6), "dry": (1.0, 2.5), "wet": (1.0, 0.4)}

    def scalars(p):
        results = wsimod_run(model, state, _deposition_drivers(drivers, **p), days=_BATCH_DAYS)
        columns = [results.arc_names.index(a) for a in _SUBURB_OUT]
        quality = results.arc_quality[..., columns, :].sum(dim=(-2, -3))
        return [quality[..., names.index(n)] for n in ("phosphate", "ammonia")]

    p = {k: torch.tensor(v, dtype=F64, requires_grad=True) for k, v in values.items()}
    together = scalars(p)
    grads = [torch.autograd.grad(v.sum(), list(p.values()), retain_graph=True) for v in together]
    assert all(v.shape == (2,) for v in together)
    for i in range(2):
        q = {k: torch.tensor(v[i], dtype=F64, requires_grad=True) for k, v in values.items()}
        for k, alone in enumerate(scalars(q)):
            assert torch.equal(alone.detach(), together[k][i].detach()), (k, i)
            single = torch.autograd.grad(alone, list(q.values()), retain_graph=True)
            for name, g, batched in zip(values, single, grads[k], strict=True):
                assert torch.equal(g, batched[i]), (k, i, name, float(g), float(batched[i]))
    phosphate, ammonia = grads
    # the phosphate depends on the load and srp, the ammonia on nhx; the instances differ
    assert all(float(g.abs().min()) > 0.0 for g in (*phosphate[:2], ammonia[2]))
    assert all(not torch.equal(v[0], v[1]) for v in together)


def test_the_adsorption_limit_gradient_is_zero_through_a_run():
    """``adosorption_nr_limit`` is the Newton loop's tolerance: it decides the iteration count, a
    detached decision, and enters no value (Y25). Its gradient through 30 days of
    ``growing_spring`` is absent or zero, while the same scalar (the final ``ads.P`` and the
    out-arc phosphate) depends on the adsorption: ``kfr``, set the same way, has a gradient."""
    model = _model("growing_spring", True)
    layout = model.wsimod_day.drivers
    p = {
        name: torch.tensor(
            float(layout[_G + name].template[layout[_G + name].labels.index(_WHEAT)]),
            dtype=F64,
            requires_grad=True,
        )
        for name in ("adosorption_nr_limit", "kfr")
    }
    drivers = initial_drivers(model, values={_G + name: {_WHEAT: v} for name, v in p.items()})
    results = wsimod_run(model, initial_state(model), drivers, days=30)
    value = _value(model, results, _ADS_P) + _value(model, results, _PHOSPHATE_OUT)
    grad, kfr = torch.autograd.grad(value, (p["adosorption_nr_limit"], p["kfr"]), allow_unused=True)
    assert grad is None or torch.equal(grad, torch.zeros_like(grad))
    assert kfr is not None and float(kfr) != 0.0
