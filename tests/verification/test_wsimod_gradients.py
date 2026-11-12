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


@dataclass(frozen=True)
class Spec:
    case: str  # a CASES name, or "quickstart" (its no-deposition configuration)
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


def _model(case: str, pollutants: bool):
    if case == "quickstart":
        network = read_wsimod_config(DATA / "quickstart", config_name="config_no_deposition.yml")
    else:
        network = read_wsimod_config(CASES[case].build())
    return build_model(network, pollutants=pollutants)[0]


def _configured(model, spec: Spec) -> float:
    kind = spec.set[0]
    if kind == "scale":
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


def _inputs(model, spec: Spec, p):
    kind = spec.set[0]
    if kind == "state":
        _, key, label = spec.set
        return initial_state(model, values={key: {label: p}}), initial_drivers(model)
    if kind == "driver":
        _, key, label, *column = spec.set
        value = {label: p} if not column else {label: {column[0]: p}}
        values = _fixed(spec)
        values.setdefault(key, {}).update(value)
        return initial_state(model), initial_drivers(model, values=values)
    # "scale": p multiplies the forcing series (one column of the concentrations)
    _, key, *column = spec.set
    drivers = initial_drivers(model, values=_fixed(spec) or None)
    series = drivers[key]
    if column:
        index = list(model.wsimod_day.drivers[key].columns).index(column[0])
        parts = list(series.unbind(-1))
        parts[index] = parts[index] * p
        drivers[key] = torch.stack(parts, dim=-1)
    else:
        drivers[key] = series * p
    return initial_state(model), drivers


def _value(model, results, scalar):
    kind = scalar[0]
    if kind == "flow":
        return results.arc_flow[:, results.arc_names.index(scalar[1])].sum()
    if kind == "tank":
        return results.tank_storage[-1, results.tank_names.index((scalar[1], scalar[2]))]
    if kind == "final":  # a store's mass after the last day, from the state (a decay tank is
        # not in the tank_* results, X40)
        _, node, attribute, pollutant = scalar
        day = model.wsimod_day
        row = day.state_labels("wsimod.storage").index((node, attribute))
        return day.last_state["wsimod.storage"][row, 1 + list(day.p.names).index(pollutant)]
    index = list(model.wsimod_day.p.names).index(scalar[2])
    return results.arc_quality[:, results.arc_names.index(scalar[1]), index].sum()


def _run(model, spec: Spec, p):
    state, drivers = _inputs(model, spec, p)
    diagnostics: dict = {"decisions": True}
    results = wsimod_run(model, state, drivers, days=spec.days, diagnostics=diagnostics)
    return _value(model, results, spec.scalar), results, diagnostics["decisions"]


def _records(decisions):
    """``{(day, site, call, round): (outcome, margin)}`` of every active decision."""
    return {
        (day, d.site, d.call, d.round): (bool(d.outcome), float(d.margin))
        for day, records in enumerate(decisions)
        for d in records
        if bool(d.active)
    }


def _jumps(runs, scale):
    """``(same, margin, where)`` for the base, ``+h`` and ``-h`` runs' decision records: whether
    every active decision has the same outcome in the three, the smallest ``|margin|`` over them
    and the (day, site, call, round) it belongs to.

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
    same, margin, where = True, float("inf"), None
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
        else:
            smallest = min(abs(f[1]) for f in found)
            if smallest < margin:
                margin, where = smallest, key
    return same, margin, where


def check(spec: Spec) -> dict:
    """The three evaluations: autograd at the point, central differences, the decision record."""
    model = _model(spec.case, spec.pollutants)
    point = _configured(model, spec) if spec.point is None else spec.point
    assert point != 0.0
    h = REL * abs(point)
    p = torch.tensor(point, dtype=F64, requires_grad=True)
    value, results, base = _run(model, spec, p)
    (ad,) = torch.autograd.grad(value, p)
    with torch.no_grad():
        up, _, up_decisions = _run(model, spec, torch.tensor(point + h, dtype=F64))
        down, _, down_decisions = _run(model, spec, torch.tensor(point - h, dtype=F64))
    scale = float(results.arc_flow.detach().abs().max())
    if spec.margin_scale == "mass":
        additive = model.wsimod_day.p.volume_or_additive_mask[1:]
        scale = float(results.arc_quality.detach().abs()[..., additive].max())
    same, margin, where = _jumps([_records(d) for d in (base, up_decisions, down_decisions)], scale)
    return {
        "point": point,
        "scalar": float(value),
        "ad": float(ad),
        "fd": float((up - down) / (2 * h)),
        "scale": scale,
        "same_outcomes": same,
        "margin": margin,
        "where": where,
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
}


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n for n in SPECS],
)
def test_gradient_matches_central_differences(name):
    r = check(SPECS[name])
    rel = abs(r["ad"] - r["fd"]) / max(abs(r["ad"]), abs(r["fd"]), 1e-300)
    print(
        f"GRAD | {name} | {SPECS[name].case} | {SPECS[name].days} | p={r['point']:.6g} | "
        f"scalar={r['scalar']:.9g} | ad={r['ad']:.10g} | fd={r['fd']:.10g} | rel={rel:.1e} | "
        f"min margin={r['margin']:.3g} (scale {r['scale']:.4g})"
    )
    assert r["same_outcomes"], f"{name}: the +-h runs' active decisions differ"
    assert r["margin"] > KINK * r["scale"], f"{name}: an active decision is within {KINK} of a jump"
    assert r["ad"] != 0.0, f"{name}: the scalar does not depend on the parameter"
    assert abs(r["ad"] - r["fd"]) <= REL * max(abs(r["ad"]), abs(r["fd"])), r
