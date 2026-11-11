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
    case: str  # a CASES name, or "quickstart" (its neutral configuration)
    days: int
    pollutants: bool
    set: (
        tuple  # ("driver", key, label[, column]) | ("scale", key[, column]) | ("state", key, label)
    )
    scalar: tuple  # ("flow", arc) | ("tank", node, attribute) | ("quality", arc, pollutant)
    point: float | None = None  # the parameter's value; the configured one by default


def _model(case: str, pollutants: bool):
    if case == "quickstart":
        network = read_wsimod_config(DATA / "quickstart", config_name="config_neutral.yml")
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


def _inputs(model, spec: Spec, p):
    kind = spec.set[0]
    if kind == "state":
        _, key, label = spec.set
        return initial_state(model, values={key: {label: p}}), initial_drivers(model)
    if kind == "driver":
        _, key, label, *column = spec.set
        value = {label: p} if not column else {label: {column[0]: p}}
        return initial_state(model), initial_drivers(model, values={key: value})
    # "scale": p multiplies the forcing series (one column of the concentrations)
    _, key, *column = spec.set
    drivers = initial_drivers(model)
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
    """``(same, margin)`` for the base, ``+h`` and ``-h`` runs' decision records: whether every
    active decision has the same outcome in the three, and the smallest ``|margin|`` over them.

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
    same, margin = True, float("inf")
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
            margin = min(margin, *(abs(f[1]) for f in found))
    return same, margin


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
    same, margin = _jumps([_records(d) for d in (base, up_decisions, down_decisions)], scale)
    return {
        "point": point,
        "scalar": float(value),
        "ad": float(ad),
        "fd": float((up - down) / (2 * h)),
        "scale": scale,
        "same_outcomes": same,
        "margin": margin,
    }


# quickstart's first months are dry: 200 days reach the land's response
_Q = dict(case="quickstart", days=200, pollutants=False)
_RURAL, _URBAN = ("my_land", "rural"), ("my_land", "urban")
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
}


# the quickstart runs take about 20 s each: three stay in the default selection, the rest are slow
SLOW = {
    "PerviousSurface.ihacres_p",
    "PerviousSurface.et0_coefficient",
    "Land.surface_residence_time",
    "Land.subsurface_residence_time",
    "Land.percolation_residence_time",
    "precipitation",
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
