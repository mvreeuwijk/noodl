"""Whole-model parity: noodl's WSIMOD port against WSIMOD 0.8.1 on its two demos and on every
purpose-built case (spec, Verification 2 and 3; Task 17).

``quickstart`` runs in full (1456 days), ``oxford`` for its first 365 days by default and in full
under ``-m slow``. Each demo runs twice: quantity-only on ``config.yml`` (WSIMOD runs its own
pollutants, noodl the volumes alone, the volumes compared) and in pollutant mode on
``config.yml`` with every process live (every pollutant compared). Every class replays WSIMOD
bitwise (replay bound 0), so the whole model does: the worst difference of every run is exactly
0, and no admission is needed. The bound of ``assert_matches_wsimod`` (``1e-12 * scale``) is
only the ceiling.

The purpose-built cases (``_wsimod_cases.CASES``) run in full, in both modes, the same way,
except ``river_nutrients``: its default run is its first 400 days (X39) and its 1100-day run,
with the decisions that place every biochemistry branch, is ``-m slow`` (D24). Cases
WSIMOD cannot run (``wtw_zero_input``, ``nonresidential_demand``) or runs only until a documented
crash (``land_sewer_ponding`` with a lowered sewer capacity) are tested for the crash and for
noodl's evident-intent run (W4). ``river_double_discharge`` is the named exception to the print
check (D17, X32): WSIMOD's River balance is wrong by construction there and prints every day.
The growing cases with a nutrient pool (3c: ``growing_spring``, 365 days by default and its three
years under ``slow``, ``growing_autumn``, ``growing_decay``, ``growing_lean_soil``) are bitwise on
every series, but WSIMOD prints round-off "mass balance error" lines for an additive pollutant at
their land: the one print admission (Y35, Y41, D12) takes them where ``|x| <= GROWING_PRINT_C *
eps * S`` (``STORE_BOUNDED``), and every other case asserts that it prints none.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import io
import re
import traceback
from pathlib import Path

import pytest

pytest.importorskip("wsimod")

import torch  # noqa: E402

from noodl.apps.wsimod import (  # noqa: E402
    EVENT_NAMES,
    build_model,
    read_wsimod_config,
    wsimod_run,
)
from noodl.apps.wsimod.day import WsimodDay  # noqa: E402
from noodl.apps.wsimod.engine import Decision  # noqa: E402
from noodl.apps.wsimod.nutrient_pool import DI, DO  # noqa: E402
from noodl.apps.wsimod.structure import river_order  # noqa: E402
from tests.apps.wsimod._replay import replay  # noqa: E402
from tests.verification._wsimod_cases import (  # noqa: E402
    CASES,
    _decay_groundwater,
    _decay_queue_groundwater,
)
from tests.verification._wsimod_parity import (  # noqa: E402
    BOUND,
    EPSILON,
    Admission,
    assert_matches_wsimod,
    mass_balance_worst,
    store_bounded_lines,
    wsimod_reference_by_day,
    wsimod_until_crash_by_day,
)
from tests.verification._wsimod_reference import (  # noqa: E402
    capture_node_calls,
    record_wsimod_run,
    wsimod_model,
    wsimod_reference,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "wsimod"
CONFIG = {"quantity": "config.yml", "pollutants": "config.yml"}
FULL = 1456  # both demos' dates

DEMOS = [
    pytest.param("quickstart", "quantity", FULL, id="quickstart-quantity"),
    pytest.param("quickstart", "pollutants", FULL, id="quickstart-pollutants"),
    pytest.param("oxford", "quantity", 365, id="oxford365-quantity"),
    pytest.param("oxford", "pollutants", 365, id="oxford365-pollutants"),
    pytest.param("oxford", "quantity", FULL, id="oxford-quantity", marks=pytest.mark.slow),
    pytest.param("oxford", "pollutants", FULL, id="oxford-pollutants", marks=pytest.mark.slow),
]


@functools.cache
def theirs(case: str, mode: str, days: int):
    return wsimod_reference(DATA / case, config_name=CONFIG[mode], days=days)


@functools.cache
def ours(case: str, mode: str, days: int, decisions: bool = False):
    network = read_wsimod_config(DATA / case, config_name=CONFIG[mode])
    model, state, drivers = build_model(network, pollutants=(mode == "pollutants"))
    diagnostics: dict = {"decisions": decisions}
    results = wsimod_run(model, state, drivers, days=days, diagnostics=diagnostics)
    diagnostics["node_names"] = model.wsimod_day.node_names
    return results, diagnostics


@pytest.mark.parametrize(("case", "mode", "days"), DEMOS)
def test_demo_matches_wsimod(case, mode, days):
    """Bitwise on every day: arc flows and pollutants, every recorded tank, every surface field.
    WSIMOD prints nothing on the demos, so no ``mass balance error`` and no dated deficit or
    Maxiter line (D13); no clamp event fires, so the residual is rounding, and closes."""
    results, diagnostics = ours(case, mode, days)
    reference = theirs(case, mode, days)
    found = assert_matches_wsimod(
        results,
        reference,
        diagnostics=diagnostics,
        case=f"{case}-{days}-{mode}",
        quality=(mode == "pollutants"),
    )
    assert found["worst"] == 0.0 and found["admissions"] == []
    assert reference.printed == []
    assert len(results.dates) == days
    assert results.arc_flow.shape[1] > 0 and results.tank_storage.shape[1] > 0
    assert set(EVENT_NAMES) <= set(diagnostics["events"])
    assert not bool(diagnostics["events"]["inaccurate_evaporation"].any())
    assert mass_balance_worst(results) <= BOUND
    assert bool(results.arc_flow.abs().sum() > 0)


@pytest.mark.parametrize("case", ["quickstart", "oxford"])
def test_demo_quantity_only_equals_the_pollutant_run_volumes(case):
    """Spec, Verification 3: a quantity-only run's volumes equal the pollutant run's, exactly
    (``torch.equal``): every arc flow, tank storage and surface field."""
    days = FULL if case == "quickstart" else 365
    quantity, _ = ours(case, "quantity", days)
    pollutant_run, _ = ours(case, "pollutants", days)
    assert torch.equal(quantity.arc_flow, pollutant_run.arc_flow)
    assert torch.equal(quantity.tank_storage, pollutant_run.tank_storage)
    for field, values in quantity.surface.items():
        other = pollutant_run.surface[field]
        assert torch.equal(torch.isnan(values), torch.isnan(other)), field  # not recorded
        assert torch.equal(values.nan_to_num(), other.nan_to_num()), field


def test_oxfords_urban_surface_holds_no_storage_after_the_first_day():
    """Review Focus 4, through the whole model: the urban (impervious) surface's storage is 0
    after day 1 to rounding, and bitwise WSIMOD's. The impervious surface stores nothing between
    days; the residue (at most 7.3e-12 m3 against the 1.16e6 m3 of day 1, measured) is the
    rounding of its balance, which WSIMOD has to the bit."""
    results, _ = ours("oxford", "pollutants", 365)
    reference = theirs("oxford", "pollutants", 365)
    column = results.surface_names.index(("oxford_land", "urban"))
    storage = results.surface["storage"][:, column]
    assert float(storage[0]) > 0.0
    assert float(storage[1:].abs().max()) <= BOUND * float(storage[0])
    assert torch.equal(storage, reference.surface["storage"][:, column])


@pytest.mark.parametrize("case", ["quickstart", "oxford"])
def test_the_demos_have_no_river_order_to_tie(case):
    """Ruling W64: WSIMOD orders a river-order tie by the set-hash order of node names
    (PYTHONHASHSEED), noodl by configuration order. Neither demo has a ``River`` node, so both
    orders are empty and no tie exists (measured under PYTHONHASHSEED 0-3); neither has a
    ``QueueGroundwater`` (W54)."""
    network = read_wsimod_config(DATA / case)
    assert river_order(network) == []
    assert wsimod_model(DATA / case).river_discharge_order == []
    assert not [n for n in network.nodes if n.cls in ("River", "QueueGroundwater")]


# --- the comparison itself --------------------------------------------------------------------
def _perturbed(days: int = 30):
    results, diagnostics = ours("quickstart", "pollutants", days, True)
    return copy.deepcopy(results), diagnostics, theirs("quickstart", "pollutants", days)


def _busiest_arc(reference):
    column = int(reference.arc_flow.abs().amax(dim=0).argmax())
    return column, float(reference.arc_flow[:, column].abs().max())


def _name(label: tuple[str, str] | str) -> str:
    return label if isinstance(label, str) else f"{label[0]}.{label[1]}"


@pytest.mark.parametrize(
    ("group", "element"),
    [
        ("arc_flow", None),
        ("tank_storage", "my_groundwater.tank"),
        ("surface.storage", "my_land.urban"),
        ("arc_quality", None),
    ],
)
def test_a_difference_beyond_the_bound_is_named_by_element_and_day(group, element):
    results, diagnostics, reference = _perturbed()
    if group == "arc_flow":
        column, scale = _busiest_arc(reference)
        results.arc_flow[7, column] += 1e-9 * scale
        label = f"arc_flow:{results.arc_names[column]}"
    elif group == "tank_storage":
        names = [_name(t) for t in results.tank_names]
        column = names.index(element)
        scale = float(reference.tank_storage[:, column].abs().max())
        # the groundwater holds nothing in these 30 days: scale 0, any difference counts
        results.tank_storage[7, column] += 1e-9 * scale or 1e-3
        label = f"tank_storage:{element}"
    elif group == "surface.storage":
        names = [_name(t) for t in results.surface_names]
        column = names.index(element)
        scale = float(reference.surface["storage"][:, column].abs().max())
        assert scale > 0.0
        results.surface["storage"][7, column] += 1e-9 * scale
        label = f"surface.storage:{element}"
    else:
        column, _ = _busiest_arc(reference)
        quality = reference.arc_quality[:, column]  # (T, K)
        k = int(quality.abs().amax(dim=0).argmax())
        scale = float(quality[:, k].abs().max())
        assert scale > 0.0
        results.arc_quality[7, column, k] += 1e-9 * scale
        label = f"arc_quality:{results.arc_names[column]}:{reference.pollutants[k]}"
    with pytest.raises(AssertionError, match=re.escape(f"first divergence {label} on day 7")):
        assert_matches_wsimod(
            results, reference, diagnostics=diagnostics, case="quickstart", quality=True
        )


def test_different_arc_names_or_day_counts_are_refused():
    results, diagnostics, reference = _perturbed()
    kwargs = {"diagnostics": diagnostics, "case": "quickstart", "quality": True}
    renamed = copy.deepcopy(results)
    renamed.arc_names = [*renamed.arc_names[:-1], "other"]
    with pytest.raises(AssertionError, match="arc names differ"):
        assert_matches_wsimod(renamed, reference, **kwargs)
    shorter = copy.deepcopy(results)
    shorter.dates = shorter.dates[:-1]
    with pytest.raises(AssertionError, match="29 days against 30"):
        assert_matches_wsimod(shorter, reference, **kwargs)
    shifted = copy.deepcopy(results)
    shifted.dates = ["2000-01-01", *shifted.dates[1:]]
    with pytest.raises(AssertionError, match="the dates differ"):
        assert_matches_wsimod(shifted, reference, **kwargs)


def test_an_admission_needs_a_decision_within_its_margin_that_day():
    results, diagnostics, reference = _perturbed()
    column, scale = _busiest_arc(reference)
    results.arc_flow[7, column] += 1e-9 * scale
    admission = Admission("quickstart", f"arc_flow:{results.arc_names[column]}", 7, "site")

    def decision(margin: float) -> Decision:
        one = torch.tensor(margin, dtype=torch.float64)
        return Decision("site", 0, None, torch.tensor(True), torch.tensor(True), one)

    def run(margin: float | None) -> dict:
        days = [list(d) for d in diagnostics["decisions"]]
        if margin is not None:
            days[7] = [decision(margin * scale)]
        return assert_matches_wsimod(
            results,
            reference,
            diagnostics={**diagnostics, "decisions": days},
            case="quickstart",
            quality=True,
            admitted=[admission],
        )

    with pytest.raises(AssertionError, match="no admission"):
        run(None)  # no decision at the site that day
    with pytest.raises(AssertionError, match="no admission"):
        run(1e-3)  # a decision, but its margin is beyond 1e-9 * scale
    with pytest.raises(AssertionError, match="no admission"):  # no decisions recorded at all
        assert_matches_wsimod(
            results,
            reference,
            diagnostics={k: v for k, v in diagnostics.items() if k != "decisions"},
            case="quickstart",
            quality=True,
            admitted=[admission],
        )
    found = run(1e-10)
    assert found["admissions"] == [admission] and found["worst"] > 0.0


def _with_print(kind: str, day: int, node: int, *, names: list[str] | None = None):
    """The 10-day run with one noodl flag set (a demand deficit at node ``node``, or a Maxiter
    at the first site) and WSIMOD's matching print, naming the node by ``names``."""
    results, diagnostics, reference = _perturbed(10)
    reference = copy.deepcopy(reference)
    real = list(diagnostics["node_names"])
    events = {k: v.clone() for k, v in diagnostics["events"].items()}
    maxiter = {k: v.clone() for k, v in diagnostics["maxiter"].items()}
    date = reference.dates[day]
    if kind == "deficit":
        events["demand_deficit"][day, node] = True
        reference.printed = [f"demand deficit of 1.5 at {real[node]} on {date}"]
    else:
        site = next(iter(maxiter))
        maxiter[site][day] = True
        reference.printed = [f"Maxiter reached in {site[0]} at {date}"]
    diagnostics = {
        **diagnostics,
        "events": events,
        "maxiter": maxiter,
        "node_names": real if names is None else names,
    }
    return results, diagnostics, reference


@pytest.mark.parametrize("kind", ["deficit", "maxiter"])
def test_a_dated_print_matches_the_flag_of_its_node_and_day(kind):
    kwargs = {"case": "quickstart", "quality": True}
    results, diagnostics, reference = _with_print(kind, 3, 0)
    assert_matches_wsimod(results, reference, diagnostics=diagnostics, **kwargs)
    # the wrong day
    results, diagnostics, reference = _with_print(kind, 3, 0)
    reference.printed = [reference.printed[0].replace(reference.dates[3], reference.dates[4])]
    with pytest.raises(AssertionError, match="differ"):
        assert_matches_wsimod(results, reference, diagnostics=diagnostics, **kwargs)


def test_a_wrong_node_name_mapping_fails_the_deficit_check():
    names = list(ours("quickstart", "pollutants", 10, True)[1]["node_names"])
    assert len(names) > 1
    # flag at node 0, the print names node 0 by its right name; the mapping is reversed
    results, diagnostics, reference = _with_print("deficit", 3, 0, names=names[::-1])
    with pytest.raises(AssertionError, match="demand deficits differ"):
        assert_matches_wsimod(
            results, reference, diagnostics=diagnostics, case="quickstart", quality=True
        )


def test_a_wrong_maxiter_site_name_fails_the_maxiter_check():
    results, diagnostics, reference = _with_print("maxiter", 3, 0)
    reference.printed = [reference.printed[0].replace("reached in ", "reached in not_")]
    with pytest.raises(AssertionError, match="Maxiter days differ"):
        assert_matches_wsimod(
            results, reference, diagnostics=diagnostics, case="quickstart", quality=True
        )


def test_a_print_noodl_does_not_record_is_refused():
    results, diagnostics, reference = _perturbed(10)
    reference = copy.deepcopy(reference)
    kwargs = {"diagnostics": diagnostics, "case": "quickstart", "quality": True}
    reference.printed = ["demand deficit of 1.0 at my_land on 2009-03-05"]
    with pytest.raises(AssertionError, match="demand deficits differ"):
        assert_matches_wsimod(results, reference, **kwargs)
    reference.printed = ["mass balance error for volume of 1.0 in my_land"]
    with pytest.raises(AssertionError, match="mass balance error"):
        assert_matches_wsimod(results, reference, **kwargs)
    reference.printed = ["something unexpected"]
    with pytest.raises(AssertionError, match="no check covers: 'something unexpected'"):
        assert_matches_wsimod(results, reference, **kwargs)


# --- the purpose-built cases ---------------------------------------------------------------------
MODES = {"quantity": False, "pollutants": True}
# D12 (Y35, Y41): WSIMOD's node check prints round-off "mass balance error for <additive> of x in
# land" lines on these growing cases (always org-phosphorus or org-nitrogen, |x| ~ 1e-17 kg),
# while noodl's balance closes; they are admitted where |x| <= GROWING_PRINT_C * eps * S, S the
# largest store of that pollutant on the land's surfaces and residence tanks that day or the day
# before. Measured max |x| / (eps S) per case, over its longest horizon (Task 9b): growing_spring
# 1.277 (1096 days, day 908; 0.432 over its default 365), growing_autumn 0.404 (400),
# growing_decay 0.479 (200), growing_lean_soil 0.610 (60), surface_overrides 0.170 (100, Task 12;
# one line); hence c = ceil(1.277) = 2. Every
# other case is unmapped, and its parity item asserts that it prints no such line.
GROWING_PRINT_C = 2
STORE_BOUNDED = frozenset(
    {"growing_spring", "growing_autumn", "growing_decay", "growing_lean_soil", "surface_overrides"}
)


def _case_param(name: str, mode: str):
    return pytest.param(name, mode, id=f"{name}-{mode}")


CASE_RUNS = [
    _case_param(name, mode) for name, spec in CASES.items() if spec.wsimod for mode in spec.modes
]


# D17 (X32): the nodes whose WSIMOD mass-balance error is printed on every day by construction
EXPECTED_MASS_BALANCE_PRINTS = {"river_double_discharge": ("riv",)}
SLOW_CASE_RUNS = [
    _case_param(name, mode)
    for name, spec in CASES.items()
    if spec.wsimod and spec.slow_days is not None
    for mode in spec.modes
]


@functools.cache
def case_theirs_by_day(name: str, days: int | None = None):
    """WSIMOD's run of the case over its default ``days`` (or ``days``), with the print days."""
    return wsimod_reference_by_day(
        CASES[name].build(), days=CASES[name].days if days is None else days
    )


def case_theirs(name: str):
    return case_theirs_by_day(name)[0]


@functools.cache
def case_ours(name: str, mode: str, days: int | None = None, decisions: bool = False):
    """noodl's run of the case over its default ``days`` (or ``days``)."""
    model, state, drivers = build_model(
        read_wsimod_config(CASES[name].build()), pollutants=MODES[mode]
    )
    diagnostics: dict = {"decisions": decisions}
    results = wsimod_run(
        model,
        state,
        drivers,
        days=CASES[name].days if days is None else days,
        diagnostics=diagnostics,
    )
    diagnostics["node_names"] = model.wsimod_day.node_names
    return results, diagnostics


def _assert_case_matches(name: str, mode: str, days: int | None = None) -> None:
    results, diagnostics = case_ours(name, mode, days)
    reference, print_days = case_theirs_by_day(name, days)
    found = assert_matches_wsimod(
        results,
        reference,
        diagnostics=diagnostics,
        case=f"{name}-{mode}",
        quality=MODES[mode],
        print_days=print_days,
        expected_mass_balance_prints=EXPECTED_MASS_BALANCE_PRINTS.get(name, ()),
        store_bounded_prints={"land": GROWING_PRINT_C} if name in STORE_BOUNDED else {},
    )
    assert found["worst"] == 0.0 and found["admissions"] == []
    assert len(results.dates) == (CASES[name].days if days is None else days)
    assert bool(results.arc_flow.abs().sum() > 0)
    clamp = diagnostics["events"]["inaccurate_evaporation"]
    if name == "evaporation_clamp":
        # ruling: WSIMOD prints both "inaccurate evaporation" and "mass balance error" lines, one
        # of each on every clamp day; ``assert_matches_wsimod`` has checked the first against
        # noodl's event and the second against its residual (``residual`` + ``dropped``)
        assert int(clamp.sum()) > 5
        for pattern in ("inaccurate evaporation", "mass balance error for volume"):
            assert sum(line.startswith(pattern) for line in reference.printed) == int(clamp.sum())
    else:
        assert not bool(clamp.any())
        # WSIMOD's node check prints the water a node drops below FLOAT_ACCURACY too (the
        # percolation push-back of land_percolation_threshold); noodl counts it in "dropped"
        assert mass_balance_worst(results) <= BOUND


@pytest.mark.parametrize(("name", "mode"), CASE_RUNS)
def test_case_matches_wsimod(name, mode):
    """Bitwise on every day, both modes: arc flows, every recorded tank and surface field (and
    every pollutant in ``pollutants`` mode); every WSIMOD print is one noodl records on the same
    node and day or in the same count (``assert_matches_wsimod``); the mass balance closes.
    ``river_nutrients`` runs its first 400 days here, without decisions (X39);
    ``river_double_discharge`` expects WSIMOD's River mass-balance print on every day (D17);
    the ``STORE_BOUNDED`` growing cases admit WSIMOD's round-off lines within the store bound
    (D12), and every other case asserts that it prints none."""
    _assert_case_matches(name, mode)


@pytest.mark.slow
@pytest.mark.parametrize(("name", "mode"), SLOW_CASE_RUNS)
def test_case_matches_wsimod_on_its_slow_days(name, mode):
    """D24: every case with a ``slow_days`` horizon, over all of it: ``river_nutrients`` for 1100
    days, three turns of its 366-slot phosphorus register and three winters; ``growing_spring``
    for three years (D21), ``growing_autumn`` for 400 days and ``growing_decay`` for 200 (Y68)."""
    _assert_case_matches(name, mode, CASES[name].slow_days)


def test_the_wtw_treatment_case_makes_negative_solids_on_cold_days():
    """X22: the case's works make negative solids on some days and not on others (temperatures
    cross 20 degC), its plant on every day (its water stays near 11 degC); each is a day the
    parity test above compares bitwise with WSIMOD, which has no such event of its own."""
    _, diagnostics = case_ours("wtw_treatment", "pollutants")
    event = diagnostics["events"]["wtw_negative_solids"]
    names = diagnostics["node_names"]
    works = event[:, names.index("works")]
    assert 0 < int(works.sum()) < len(works)
    assert bool(event[:, names.index("plant")].any())
    quantity = case_ours("wtw_treatment", "quantity")[1]["events"]["wtw_negative_solids"]
    assert not bool(quantity.any())  # no additive pollutants, no solids to go negative


@pytest.mark.parametrize("name", [n for n, spec in CASES.items() if set(spec.modes) >= set(MODES)])
def test_quantity_only_equals_the_pollutant_run_volumes(name):
    """Spec, Verification 3, on every case: a quantity-only run's volumes equal the pollutant
    run's, exactly (``torch.equal``): every arc flow, tank storage and surface field, over the
    case's bookkeeping prefix (``Case.bookkeeping``, Y68)."""
    days = CASES[name].bookkeeping
    quantity, _ = case_ours(name, "quantity", days)
    pollutants, _ = case_ours(name, "pollutants", days)
    assert torch.equal(quantity.arc_flow, pollutants.arc_flow)
    assert torch.equal(quantity.tank_storage, pollutants.tank_storage)
    for field, values in quantity.surface.items():
        other = pollutants.surface[field]
        assert torch.equal(torch.isnan(values), torch.isnan(other)), field
        assert torch.equal(values.nan_to_num(), other.nan_to_num()), field


@pytest.mark.parametrize("name", ["irrigation", "irrigation_early"])
def test_the_irrigation_cases_reach_every_irrigate_branch(name):
    """From the decisions (spec, Verification 3; Task 13, Y80): on each case's bookkeeping
    prefix, each of ``crop``, ``deficit``, ``demand`` and ``saturated`` is active with both
    outcomes at both irrigation surfaces: before and after sowing, a negative and a positive
    deficit, demands above and below ``FLOAT_ACCURACY`` and saturated and unsaturated soils. A
    site that never records fails."""
    day = WsimodDay(read_wsimod_config(CASES[name].build()), pollutants=True)
    day.record_decisions = True
    seen: dict[str, set[bool]] = {}
    for _, ctx, _ in day.run_days(days=CASES[name].bookkeeping):
        for d in ctx.decisions:
            if bool(d.active.any()):
                seen.setdefault(d.site, set()).add(bool(d.outcome[d.active].any()))
                seen[d.site].add(bool((~d.outcome[d.active]).any()))
    for surface in ("irr_a", "irr_b"):
        for branch in ("crop", "deficit", "demand", "saturated"):
            site = f"node:farm/surfaces.{surface}/irrigate/{branch}"
            assert seen.get(site) == {False, True}, (name, site, seen.get(site))


def test_the_cases_cover_every_listed_feature():
    """The plan's case list, task by task (Task 17 step 1): no listed case is missing."""
    listed = """three_out_arcs node_chain forced_negative_excess maxiter_pull pull_and_push_arcs
    user_alt_queue_arc storage_distribute queue_groundwater reservoir_over_capacity
    river_reservoir_environmental catchment_abstraction queue_arc river_chain
    distribution_leakage unlimited_distribution demand_constant_and_loads reservoir_spill
    sewer_timearea foul_sewers groundwater_infiltrate wwtw_storm_tank fwtw_deficit_sludge
    queue_tank_time_zero wtw_zero_input impervious_only pervious_only evaporation_clamp
    temperature_blending land_percolation_threshold residential_with_land land_sewer_ponding
    batch_divergence soil_temperature capacity_first_come simple_pollutants float_accuracy_override
    node_type_override decay_storage decay_groundwater decay_queue_groundwater
    decay_queue_groundwater_n0 decay_arc decay_arc_mixed decay_arc_alt
    surface_decay_nitrification land_run_twice river_nutrients river_double_discharge
    surface_deposition growing_no_nitrate growing_default_depletion growing_spring growing_autumn
    growing_decay deposition_non_additive growing_no_din growing_zero_n growing_nitrite_only
    growing_empty_soil growing_lean_soil surface_overrides irrigation irrigation_early
    irrigation_without_nutrients""".split()
    assert set(listed) <= set(CASES)
    assert {n for n, spec in CASES.items() if not spec.wsimod} == {
        "wtw_zero_input",
        "nonresidential_demand",
        "growing_default_depletion",
        "growing_no_din",
        "growing_zero_n",
        "growing_nitrite_only",
        "growing_empty_soil",
        "irrigation_without_nutrients",
    }
    # D13: the cases WSIMOD runs as they are until it raises
    assert {n: (spec.crash, spec.crash_day) for n, spec in CASES.items() if spec.crash} == {
        "growing_default_depletion": (("ZeroDivisionError", "land.py:1566"), 11),
        "growing_no_din": (("ZeroDivisionError", "land.py:1460"), 0),
        "growing_zero_n": (("ZeroDivisionError", "land.py:1758"), 0),
        "growing_nitrite_only": (("ZeroDivisionError", "land.py:1758"), 0),
        "growing_empty_soil": (("ZeroDivisionError", "land.py:1899"), 1),
        "irrigation_without_nutrients": (("KeyError", "land.py:2373"), 5),
    }
    # D24: the cases whose default run is a prefix of their dates
    # (Y68: the growing cases' long horizons moved to slow)
    horizons = {
        "river_nutrients": (400, 1100),
        "growing_spring": (365, 1096),  # D21 (amended by Y68): three years, one by default
        "growing_autumn": (300, 400),
        "growing_decay": (100, 200),
        "surface_deposition": (180, 450),  # Task 9b (W15)
        "growing_no_nitrate": (365, 540),
    }
    assert {n: spec.slow_days for n, spec in CASES.items() if spec.slow_days} == {
        n: slow for n, (_, slow) in horizons.items()
    }
    for name, (days, slow) in horizons.items():
        assert CASES[name].days == days and len(CASES[name].build()["dates"]) == slow
    # Y68: the bookkeeping prefixes (test_run.py's generic items and quantity-only)
    assert {n: spec.bookkeeping_days for n, spec in CASES.items() if spec.bookkeeping_days} == {
        "growing_spring": 250,
        "growing_autumn": 60,
        "growing_decay": 60,
        "surface_deposition": 90,
        "growing_no_nitrate": 250,
        "irrigation": 60,
        "irrigation_early": 60,
    }
    # D12: the cases whose round-off prints are admitted within the store bound
    assert STORE_BOUNDED <= {n for n, spec in CASES.items() if spec.wsimod}


def tiny_dropped(results) -> torch.Tensor:
    """The volume the run's mass balance counts as dropped, summed over the days."""
    return results.mass_balance.terms["dropped"][:, 0].sum()


@pytest.mark.parametrize("mode", list(MODES))
def test_the_decay_arc_cases_exercise_what_they_are_named_for(mode):
    """``decay_arc`` decays in all four of its ``DecayArc``s (``n`` = 2, 1, 0 and 0, the last
    out of a catchment, X29) inside the precondition of D21: the event is false on every day,
    one column per arc. ``decay_arc_alt``'s in-port pushes on some days and not on others, and
    its ``DecayArcAlt`` decays. In ``quantity`` mode nothing decays and both still run."""
    results, diagnostics = case_ours("decay_arc", mode)
    event = diagnostics["events"]["decay_arc_order_unknown"]
    assert event.shape == (CASES["decay_arc"].days, len(results.arc_names))
    assert not bool(event.any())
    model = build_model(read_wsimod_config(CASES["decay_arc"].build()))[0]
    arcs = model.wsimod_day.engine.arcs
    assert {
        name: arcs[name].number_of_timesteps
        for name in arcs
        if "Decay" in type(arcs[name]).__name__
    } == {
        "src2_to_j": 0,
        "q": 2,
        "b_to_sink": 0,
        "q_pull": 1,
    }
    terms = results.mass_balance.terms
    alt, _ = case_ours("decay_arc_alt", mode)
    pushed = alt.arc_flow[:, alt.arc_names.index("src_to_head")] > 0
    assert 0 < int(pushed.sum()) < len(pushed)
    if mode == "pollutants":
        assert float(terms["DecayArc.decayed"][:, 1].sum()) > 0.0
        assert float(alt.mass_balance.terms["DecayArcAlt.decayed"][:, 1:].sum()) > 0.0
    else:
        assert terms["DecayArc.decayed"].shape[-1] == 1
        assert float(terms["DecayArc.decayed"].abs().sum()) == 0.0


def test_the_new_cases_exercise_what_they_are_named_for():
    """``capacity_first_come`` fills its shared arc to capacity while the second slot gets what
    is left; ``float_accuracy_override`` runs with the accuracy it declares, a sub-1e-6 push dropped
    by the ``QueueArc`` it reaches; ``simple_pollutants`` has the two lists of
    ``set_simple_pollutants``;
    ``node_type_override`` is a ``ResidentialDemand`` under the ``Demand`` orchestration key."""
    results, _ = case_ours("capacity_first_come", "pollutants")
    column = results.arc_names.index("hub_to_sea")
    assert float(results.arc_flow[:, column].max()) == 30.0
    from_src = results.arc_flow[:, results.arc_names.index("src_to_hub")]
    from_res = results.arc_flow[:, results.arc_names.index("res_to_hub")]
    # first come, first served: on a day the outlet is full, the reservoir (second slot) got only
    # what the catchment left, less than both its arc's capacity and what it held
    full = (from_src + from_res - 30.0).abs() < 1e-12
    assert bool(full.any()) and bool((from_res[full] < 25.0).any())
    assert bool((from_src[full] > 0).all()) and float(from_res[full].min()) < 25.0 - 1e-9
    network = read_wsimod_config(CASES["float_accuracy_override"].build())
    assert network.float_accuracy == 1e-6
    tiny = case_ours("float_accuracy_override", "pollutants")[0]
    arrived = tiny.arc_flow[:, tiny.arc_names.index("src_to_head")]
    queued = tiny.arc_flow[:, tiny.arc_names.index("q")]
    assert float(arrived[2::3].min()) == pytest.approx(5e-7, rel=1e-12)  # the push reaches q ...
    assert float(queued[2::3].abs().max()) == 0.0  # ... which drops it (1e-11 < 5e-7 < 1e-6)
    assert float(tiny_dropped(tiny)) > 0.0  # and the balance counts it as dropped
    mapping = CASES["float_accuracy_override"].build()
    del mapping["float_accuracy"]  # the default accuracy queues the push
    model, state, drivers = build_model(read_wsimod_config(mapping))
    default = wsimod_run(model, state, drivers)
    assert float(default.arc_flow[2::3, default.arc_names.index("q")].min()) > 0.0
    simple = read_wsimod_config(CASES["simple_pollutants"].build())
    assert tuple(simple.pollutants.names) == ("phosphate", "temperature")
    override = read_wsimod_config(CASES["node_type_override"].build())
    homes = next(n for n in override.nodes if n.name == "homes")
    assert (homes.cls, homes.slot_key) == ("ResidentialDemand", "Demand")


def _clamp_run():
    results, diagnostics = case_ours("evaporation_clamp", "pollutants")
    return results, diagnostics, copy.deepcopy(case_theirs("evaporation_clamp"))


def test_the_evaporation_clamp_prints_are_accepted_only_where_they_match_noodl():
    """Ruling (evaporation_clamp): WSIMOD prints "inaccurate evaporation" and "mass balance error"
    lines; they pass only as noodl's ``inaccurate_evaporation`` flags and its residual. A changed
    error value, a missing line, one more error line or one more evaporation line fails."""
    kwargs = {"case": "evaporation_clamp", "quality": True}
    results, diagnostics, reference = _clamp_run()
    assert_matches_wsimod(results, reference, diagnostics=diagnostics, **kwargs)
    errors = [i for i, x in enumerate(reference.printed) if x.startswith("mass balance error")]
    assert len(errors) > 5
    changed = copy.deepcopy(reference)
    value = float(re.search(r"volume of (\S+)", changed.printed[errors[2]]).group(1))
    changed.printed[errors[2]] = f"mass balance error for volume of {value + 1e-3} in heath"
    with pytest.raises(AssertionError, match="mass balance errors"):
        assert_matches_wsimod(results, changed, diagnostics=diagnostics, **kwargs)
    missing = copy.deepcopy(reference)
    del missing.printed[errors[0]]
    with pytest.raises(AssertionError, match="mass balance errors"):
        assert_matches_wsimod(results, missing, diagnostics=diagnostics, **kwargs)
    extra = copy.deepcopy(reference)
    extra.printed.append("mass balance error for volume of 1.0 in heath")
    with pytest.raises(AssertionError, match="mass balance errors"):
        assert_matches_wsimod(results, extra, diagnostics=diagnostics, **kwargs)
    unflagged = copy.deepcopy(reference)
    unflagged.printed = [x for x in unflagged.printed if not x.startswith("inaccurate")]
    with pytest.raises(AssertionError, match="'inaccurate_evaporation' lines"):
        assert_matches_wsimod(results, unflagged, diagnostics=diagnostics, **kwargs)


# --- the River biochemistry cases (Task 13; X16, X32, X39) --------------------------------------
BIO = "biochemical_processes"
# The outcomes each river of ``river_nutrients`` is built to show (spec, Purpose-built cases),
# per river so that no branch passes on another river's hit (X55); the sets are exact.
# ``dmin``/``dmax`` record ``b < a`` / ``b > a`` with ``b`` the second operand, so a cap
# ``min(0.5 pool, x)`` is taken when its outcome is False and left when it is True, and the
# denitrification cap ``min(denitri_water, 0.5 din)`` is taken when its outcome is True
# (``storage.py:788``, ``:840-843``, ``:860-863``, ``:882``, ``:891``). Every cap listed binds on
# some days and not on others. ``r_up`` is the low-P river: it alone is dry (the drought), crosses
# the ``totalphosfcn`` guard both ways, mineralises with its caps binding (its ``production_cap:N``
# is left out: it binds there only in the DIN-free stretch, at ``din = 0``, trivially) and takes
# the phosphorus caps. ``r_low`` (``max_temp_lag = 5``: its 10-day mean exceeds its 20-day mean
# whenever its register sum is positive) produces on almost every day, so its ``production_cap:N``
# is the real one; it never empties and its phosphorus saturates ``totalphosfcn`` from day 1.
BOTH = frozenset({True, False})
RIVER_BRANCHES: dict[str, dict[str, frozenset[bool]]] = {
    "r_up": {
        "dry": BOTH,
        "frozen": BOTH,
        "cold": BOTH,
        "totalphosfcn": BOTH,
        "production": BOTH,
        "denitrification_cap": BOTH,
        "production_cap:P": BOTH,
        "mineralisation_cap:N": BOTH,
        "mineralisation_cap:P": BOTH,
        "macrophyte_cap:N": BOTH,
        "macrophyte_cap:P": BOTH,
        "macrophyte_phosphorus": BOTH,
    },
    "r_low": {
        "dry": frozenset({False}),
        "frozen": BOTH,
        "cold": BOTH,
        "totalphosfcn": frozenset({True}),
        "production": BOTH,
        "denitrification_cap": BOTH,
        "production_cap:N": BOTH,
        "production_cap:P": BOTH,
        "macrophyte_cap:N": BOTH,
        "macrophyte_cap:P": BOTH,
    },
}


def _river_outcomes(decisions, days: int, river: str) -> dict[str, set[bool]]:
    """``{branch: the outcomes active on some day < days}`` of ``river``'s biochemistry
    decisions."""
    out: dict[str, set[bool]] = {}
    prefix = f"node:{river}/{BIO}/"
    for day in decisions[:days]:
        for d in day:
            if d.site.startswith(prefix) and bool(d.active):
                out.setdefault(d.site[len(prefix) :], set()).add(bool(d.outcome))
    return out


@pytest.mark.slow
def test_river_nutrients_reaches_every_branch():
    """R3, X39, D24: with the decisions recorded over the full 1100 days, each branch the case is
    built for occurs active within its first 400 days (the default run) and over the full run:
    frost, cold, the dry day, both ``totalphosfcn`` outcomes, production both ways, every 50 %
    cap taken, and ``"river_nitrogen_lost"``. The low-P river ``r_up`` alone has the dry days,
    the DIN-free stretch and both ``totalphosfcn`` outcomes; the full run turns its 366-slot
    register over three times, and the phosphate pulse has left it (days 426-455), so
    ``totalphosfcn`` returns to 0 (its guard False) from day 448, measured."""
    spec = CASES["river_nutrients"]
    assert spec.slow_days >= 3 * 366
    results, diagnostics = case_ours("river_nutrients", "pollutants", spec.slow_days, True)
    names = diagnostics["node_names"]
    lost = diagnostics["events"]["river_nitrogen_lost"]
    for days in (spec.days, spec.slow_days):
        for river, branches in RIVER_BRANCHES.items():
            seen = _river_outcomes(diagnostics["decisions"], days, river)
            for branch, outcomes in branches.items():
                assert seen[branch] == outcomes, (days, river, branch, seen[branch])
        assert bool(lost[:days, names.index("r_up")].any()), days
    assert not bool(lost[:, names.index("r_low")].any())  # r_low never empties, so din > 0
    late = {
        k: bool(d.outcome)
        for k, day in enumerate(diagnostics["decisions"])
        for d in day
        if k >= spec.days and d.site == f"node:r_up/{BIO}/totalphosfcn" and bool(d.active)
    }
    assert set(late.values()) == {True, False}
    assert min(k for k, outcome in late.items() if not outcome) == 448
    assert len(results.dates) == spec.slow_days


def test_river_double_discharge_runs_the_biochemistry_twice_a_day():
    """X16, X32: WSIMOD prints the river's mass-balance error on every day (for pollutants, never
    for the volume, which the biochemistry leaves alone) and nothing else, because its assigned
    ``bio_in``/``bio_out`` hold the second call's terms only; noodl runs the biochemistry as
    call 0 and call 1 on every day, its terms add both calls, and its balance closes within
    rounding (measured 2.1e-16 of scale)."""
    name = "river_double_discharge"
    spec = CASES[name]
    reference, print_days = case_theirs_by_day(name)
    lines = [
        (d, m)
        for d, line in zip(print_days, reference.printed, strict=True)
        if (m := re.fullmatch(r"mass balance error for (\S+) of (\S+) in riv", line))
    ]
    assert len(lines) == len(reference.printed) > 0  # nothing else is printed
    assert {d for d, _ in lines} == set(range(spec.days))
    assert "volume" not in {m.group(1) for _, m in lines}
    results, diagnostics = case_ours(name, "pollutants", None, True)
    site = f"node:riv/{BIO}/dry"
    per_day = [[d.call for d in day if d.site == site] for day in diagnostics["decisions"]]
    assert per_day == [[0, 1]] * spec.days
    terms = results.mass_balance.terms
    assert float(terms["River.bio_out"][:, 1:].abs().sum()) > 0.0
    assert float(terms["River.bio_out"][:, 0].abs().sum()) == 0.0  # no volume moves
    assert mass_balance_worst(results) <= BOUND


def test_an_expected_mass_balance_print_must_appear_every_day():
    """D17: the river's lines pass only as a complete set. Without the exception they are refused
    (they name a pollutant, not the volume noodl's residual is compared with); with it, removing
    one day's lines from the reference fails, naming the day; the exception needs ``print_days``
    and a node the model has."""
    name = "river_double_discharge"
    results, diagnostics = case_ours(name, "pollutants")
    reference, print_days = case_theirs_by_day(name)
    kwargs = {"diagnostics": diagnostics, "case": name, "quality": True}
    expected = {"expected_mass_balance_prints": ("riv",)}
    assert_matches_wsimod(results, reference, print_days=print_days, **expected, **kwargs)
    with pytest.raises(AssertionError, match="WSIMOD printed 'mass balance error for"):
        assert_matches_wsimod(results, reference, print_days=print_days, **kwargs)
    keep = [
        k
        for k, (d, line) in enumerate(zip(print_days, reference.printed, strict=True))
        if not (d == 7 and line.endswith(" in riv"))
    ]
    assert len(keep) < len(print_days)
    missing = copy.deepcopy(reference)
    missing.printed = [reference.printed[k] for k in keep]
    with pytest.raises(AssertionError, match="no mass balance error for 'riv' on day 7 "):
        assert_matches_wsimod(
            results, missing, print_days=[print_days[k] for k in keep], **expected, **kwargs
        )
    with pytest.raises(ValueError, match="needs print_days"):
        assert_matches_wsimod(results, reference, **expected, **kwargs)
    # a volume line of the river is not excused: it goes through the residual comparison
    extra = copy.deepcopy(reference)
    extra.printed = [*reference.printed, "mass balance error for volume of 1.0 in riv"]
    with pytest.raises(AssertionError, match="mass balance errors"):
        assert_matches_wsimod(
            results, extra, print_days=[*print_days, print_days[-1]], **expected, **kwargs
        )
    with pytest.raises(AssertionError, match="unknown nodes"):
        assert_matches_wsimod(
            results,
            reference,
            print_days=print_days,
            expected_mass_balance_prints=("nobody",),
            **kwargs,
        )


def test_an_undated_print_needs_noodls_event_and_a_dated_one_its_node_and_day():
    results, diagnostics, reference = _perturbed(10)
    kwargs = {"diagnostics": diagnostics, "case": "quickstart", "quality": True}
    reference = copy.deepcopy(reference)
    reference.printed = ["Storage unable to push"]  # noodl flagged no such event
    with pytest.raises(AssertionError, match="'storage_unable_to_push' lines"):
        assert_matches_wsimod(results, reference, **kwargs)
    reference.printed = ["warning, distribution leakage not going to GW in x at 2009-03-03"]
    with pytest.raises(AssertionError, match="leakage_not_going_to_gw differ"):
        assert_matches_wsimod(results, reference, **kwargs)
    reference.printed = ["Attempted push check to deny"]  # WSIMOD's deny handler: no record
    assert_matches_wsimod(results, reference, **kwargs)


def test_an_undated_print_moved_to_another_flagged_day_is_refused():
    """The undated prints are checked day by day: moving one ``storage_unable_to_push`` flag of
    ``storage_distribute`` to another flagged day (the totals unchanged) fails."""
    results, diagnostics = case_ours("storage_distribute", "pollutants")
    reference, print_days = case_theirs_by_day("storage_distribute")
    kwargs = {"diagnostics": diagnostics, "case": "storage_distribute", "quality": True}
    assert_matches_wsimod(results, reference, print_days=print_days, **kwargs)
    flags = diagnostics["events"]["storage_unable_to_push"]
    days = [d for d in range(flags.shape[0]) if bool(flags[d].any())]
    assert len(days) >= 2
    moved = flags.clone()
    moved[days[0]] = False
    moved[days[1]] = flags[days[0]] | flags[days[1]]  # the same count of lines overall
    with pytest.raises(AssertionError, match="storage_unable_to_push"):
        assert_matches_wsimod(
            results,
            reference,
            print_days=print_days,
            **{
                **kwargs,
                "diagnostics": {
                    **diagnostics,
                    "events": {**diagnostics["events"], "storage_unable_to_push": moved},
                },
            },
        )


@pytest.mark.parametrize("name", ["wwtw_storm_tank", "land_sewer_ponding"])
def test_the_deny_prints_come_as_one_push_set_and_three_push_checks_a_day(name):
    """The accepted ``Attempted ... deny`` prints have a fixed structure: on each of the 40 days,
    one ``push set`` line and three ``push check`` lines (the works' and the land's deny
    handlers, called once and three times a day by the neighbours' pushes and checks)."""
    reference, print_days = case_theirs_by_day(name)
    for kind, count in (("push set to deny", 1), ("push check to deny", 3)):
        per_day = [0] * CASES[name].days
        for day, line in zip(print_days, reference.printed, strict=True):
            per_day[day] += line == f"Attempted {kind}"
        assert per_day == [count] * CASES[name].days, kind


# --- the cases where WSIMOD crashes --------------------------------------------------------------
def _wsimod_run_of(model) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        model.run(dates=list(model.dates), verbose=False)


def _lower_sewer_capacity(wsimod_sewer, noodl_sewer, capacity: float, normal: float) -> None:
    """Lower the sewer's capacity to ``capacity`` between ``Land.run`` and ``make_discharge``, on
    every day, in both engines, and restore ``normal`` after it (the next day's ``Land.run`` sees
    the case's own capacity): in WSIMOD by wrapping the sewer's ``make_discharge``, in noodl by
    binding the ``capacity`` parameter in the same place (a per-day capacity driver)."""

    def lowered(node, original, set_capacity):
        def wrapped(*args, **kwargs):
            set_capacity(capacity)
            try:
                return original(*args, **kwargs)
            finally:
                set_capacity(normal)

        node.make_discharge = wrapped

    if wsimod_sewer is not None:

        def set_wsimod(value):
            wsimod_sewer.capacity = value
            wsimod_sewer.sewer_tank.capacity = value

        lowered(wsimod_sewer, wsimod_sewer.make_discharge, set_wsimod)
    if noodl_sewer is not None:

        def set_noodl(value):
            noodl_sewer.bind({"capacity": torch.tensor(value, dtype=torch.float64)})

        lowered(noodl_sewer, noodl_sewer.make_discharge, set_noodl)


def test_case_runs_where_wsimod_crashes_land_sewer_ponding():
    """``land_sewer_ponding`` with the sewer's capacity lowered to 0.5 between ``Land.run`` and
    ``make_discharge``: the sewer holds more than it can, and ponds onto the streets with the tag
    ``"Sewer"``. WSIMOD has no handler for that tag in ``Land`` and raises; noodl does what the
    code evidently means (W4): the ponded water re-enters the first impervious surface, and the
    mass balance closes."""
    spec = CASES["land_sewer_ponding"]
    model = wsimod_model(spec.build())
    _lower_sewer_capacity(model.nodes["combined"], None, 0.5, 10.0)
    with pytest.raises(KeyError, match="Sewer"):
        _wsimod_run_of(model)

    noodl_model, state, drivers = build_model(read_wsimod_config(spec.build()))
    _lower_sewer_capacity(None, noodl_model.wsimod_day.engine.nodes["combined"], 0.5, 10.0)
    diagnostics: dict = {"decisions": True}
    results = wsimod_run(noodl_model, state, drivers, diagnostics=diagnostics)
    ponded = [
        bool(d.outcome)
        for day in diagnostics["decisions"]
        for d in day
        if d.site == "node:combined/make_discharge/ponded"
    ]
    assert len(ponded) == spec.days and any(ponded)
    assert ponded[0]  # day 0: the day WSIMOD crashes on (its first ponding)
    column = results.arc_names.index("combined_to_city")
    assert float(results.arc_flow[:, column].max()) > 0.0  # water went back to the streets
    assert bool(torch.isfinite(results.arc_flow).all())
    assert mass_balance_worst(results) <= BOUND
    assert len(results.dates) == spec.days


# D13: the crash cases, with the event noodl records on the crash day (None: no event)
CRASH_EVENTS = {
    "growing_default_depletion": "crop_water_stress_zero_denominator",
    "growing_no_din": "pull_storage_no_din",
    "growing_zero_n": "transformation_no_din",
    "growing_nitrite_only": "transformation_no_din",
    "growing_empty_soil": "crop_uptake_dry_soil",
    "irrigation_without_nutrients": None,  # Y20: noodl updates no pool, silently
}
CRASH_RUNS = [
    pytest.param(name, mode, id=f"{name}-{mode}")
    for name, spec in CASES.items()
    if spec.crash is not None
    for mode in spec.modes
]


@functools.cache
def crashed(name: str):
    return wsimod_until_crash_by_day(CASES[name].build())


def _innermost_land_line(error: BaseException) -> str:
    """``"land.py:<line>"`` of the innermost frame of ``error`` in WSIMOD's ``nodes/land.py``."""
    frames = [
        f
        for f in traceback.extract_tb(error.__traceback__)
        if Path(f.filename).parts[-2:] == ("nodes", "land.py")
    ]
    assert frames, f"no frame of nodes/land.py in {error!r}"
    return f"land.py:{frames[-1].lineno}"


@pytest.mark.parametrize(("name", "mode"), CRASH_RUNS)
def test_case_runs_where_wsimod_crashes(name, mode):
    """D13: WSIMOD, run as it is, raises the case's exception at its line of ``nodes/land.py``
    on the crash day. noodl runs every day: on the days before the crash it equals WSIMOD
    bitwise (``assert_matches_wsimod``, prints included; vacuous on day 0), on the crash day it
    records the event at the land (the evident intent instead of the exception), and over the
    whole run its results are finite and its balance closes."""
    spec = CASES[name]
    reference, error, day, print_days = crashed(name)
    kind, line = spec.crash
    assert type(error).__name__ == kind
    assert _innermost_land_line(error) == line
    assert day == spec.crash_day  # the recipe's crash day, pinned (D13)
    if day:
        results, diagnostics = case_ours(name, mode, day)
        found = assert_matches_wsimod(
            results,
            reference,
            diagnostics=diagnostics,
            case=f"{name}-{mode}",
            quality=MODES[mode],
            print_days=print_days,
        )
        assert found["worst"] == 0.0 and found["admissions"] == []
    results, diagnostics = case_ours(name, mode)
    event = CRASH_EVENTS[name]
    if event is not None:
        names = diagnostics["node_names"]
        land = next(n["name"] for n in spec.build()["nodes"] if n["type_"] == "Land")
        flags = diagnostics["events"][event][:, names.index(land)]
        assert bool(flags[day]) and not bool(flags[:day].any())
    for values in (results.arc_flow, results.tank_storage, results.surface["storage"]):
        assert bool(torch.isfinite(values).all())
    assert mass_balance_worst(results) <= BOUND


def test_case_runs_where_wsimod_crashes_wtw_zero_input():
    """``wtw_zero_input``: WSIMOD's liquor blend divides by zero (``wtw.py:176-182``) on the
    first day of a treatment works with no input and no liquor; noodl keeps the previous liquor
    and records ``"wtw_liquor_zero_denominator"`` (the hand expectations of Task 11: the full
    plant on day 1 only, the works every day), everything finite, the balance closing."""
    spec = CASES["wtw_zero_input"]
    with pytest.raises(ZeroDivisionError):
        _wsimod_run_of(wsimod_model(spec.build()))
    results, diagnostics = case_ours("wtw_zero_input", "pollutants")
    names = diagnostics["node_names"]
    flags = diagnostics["events"]["wtw_liquor_zero_denominator"]
    assert flags[:, names.index("plant")].tolist() == [True] + [False] * (spec.days - 1)
    assert flags[:, names.index("works")].tolist() == [True] * spec.days
    assert bool(torch.isfinite(results.arc_flow).all())
    assert bool(torch.isfinite(results.tank_storage).all())
    assert mass_balance_worst(results) <= BOUND


def test_case_runs_where_wsimod_crashes_decay_groundwater():
    """``decay_groundwater`` without its ``filename`` override: WSIMOD loses the groundwater's
    configured forcing (``storage.py:202-203``), so its first decay reads ``None``
    (``core.py:485``, ``TypeError``); noodl refuses by name (X11). With the override both run."""
    mapping = _decay_groundwater(override=False)
    with pytest.raises(TypeError):
        _wsimod_run_of(wsimod_model(mapping))
    with pytest.raises(ValueError, match="filename"):
        build_model(read_wsimod_config(mapping))


def test_case_runs_where_wsimod_crashes_decay_queue_groundwater():
    """``decay_queue_groundwater`` without its ``filename`` override: WSIMOD loses the
    groundwater's configured forcing (``storage.py:291-292``), so its ``DecayQueueTank``'s
    first decay, at the first push or end of day, reads ``None`` (``core.py:485``,
    ``TypeError``); noodl refuses by name (X11)."""
    mapping = _decay_queue_groundwater(override=False)
    with pytest.raises(TypeError):
        _wsimod_run_of(wsimod_model(mapping))
    with pytest.raises(ValueError, match="QueueGroundwater.*filename"):
        build_model(read_wsimod_config(mapping))


def test_case_runs_where_wsimod_crashes_nonresidential_demand():
    """``nonresidential_demand``: WSIMOD's ``NonResidentialDemand.get_demand`` calls itself
    (``demand.py:143-150``) and recurses until Python gives up; noodl does the evident intent
    (W4): the constant demand as ``house``, pulled from the supply every day, and the balance
    closes."""
    spec = CASES["nonresidential_demand"]
    with pytest.raises(RecursionError):
        _wsimod_run_of(wsimod_model(spec.build()))
    results, diagnostics = case_ours("nonresidential_demand", "pollutants")
    pulled = results.arc_flow[:, results.arc_names.index("supply_to_factory")]
    # a supply of 60 serves 3 a day for 20 days, then the demand is a deficit every day
    assert pulled[:20].tolist() == [3.0] * 20 and float(pulled[20:].abs().max()) == 0.0
    names = diagnostics["node_names"]
    deficit = diagnostics["events"]["demand_deficit"][:, names.index("factory")]
    assert deficit.tolist() == [False] * 20 + [True] * (spec.days - 20)
    assert bool(torch.isfinite(results.arc_flow).all())
    assert mass_balance_worst(results) <= BOUND


@pytest.mark.parametrize(
    ("name", "times"), [("surface_decay_nitrification", 1), ("land_run_twice", 2)]
)
def test_the_surface_decay_cases_carry_yesterdays_decay_into_nitrate_and_nitrite(name, times):
    """``land.py:427-439``: day ``d + 1``'s inflow of nitrate is day ``d``'s decayed nitrite and
    of nitrite its decayed ammonia, once per ``Land.run`` (X16: twice in ``land_run_twice``).
    Rain carries neither, so they are the whole inflow; day 0 has none. The factor reaches its
    cap on warm days, so the cap decisions are exercised."""
    results, diagnostics = case_ours(name, "pollutants")
    network = read_wsimod_config(CASES[name].build())
    columns = ["volume", *network.pollutants.additive]
    inflow = results.mass_balance.terms["Land.running_inflow_mb"]
    decayed = results.mass_balance.terms["DecayTank.decayed"]
    for source, target in (("nitrite", "nitrate"), ("ammonia", "nitrite")):
        carried = inflow[:, columns.index(target)]
        assert float(carried[0]) == 0.0
        assert float(decayed[:, columns.index(source)].abs().sum()) > 0.0
        expected = times * decayed[:-1, columns.index(source)]
        assert torch.allclose(carried[1:], expected, rtol=1e-12, atol=0.0)


# --- the growing cases (Tasks 9a and 9b; spec, Verification 3) ----------------------------------
@functools.cache
def growing_capture(name: str, days: int) -> tuple:
    """WSIMOD's run of a growing case's first ``days`` under ``capture_node_calls`` on its
    ``land``: every call into it (``NodeCall``), in order."""
    model = wsimod_model(CASES[name].build())
    with capture_node_calls(model, ["land"]) as calls:
        record_wsimod_run(model, days=days)
    return tuple(calls["land"])


def _wsimod_runs(name: str, days: int) -> list:
    """Each day's ``Land.run`` call in WSIMOD's capture of the first ``days``: one a day, and
    nothing else calls into the land in these cases."""
    runs = [c for c in growing_capture(name, days) if c.method == "run"]
    assert [c.day for c in runs] == list(range(days))
    return runs


def growing_days(name: str, days: int) -> tuple[list[dict], object]:
    """noodl's day driver (pollutants on) over the case's first ``days``: after each day's
    orchestration, the first growing surface's tank (``(1 + K,)``), pool (``(5, 2)``) and
    ``days_after_sow``, ``crop_cover`` and ``total_decayed``, the outcomes of every decision
    (``{site: {outcome, ...}}`` over the active ones), the day's events at the land and its
    ``Surface.nitrite_carry_over`` inflow part (D11). Each test makes its one run (Y68)."""
    network = read_wsimod_config(CASES[name].build())
    day = WsimodDay(network, pollutants=True)
    day.record_decisions = True
    rows = []
    for _, ctx, engine in day.run_days(days=days):
        surface = engine.nodes["land"].surfaces[0]
        outcomes: dict[str, set[bool]] = {}
        for d in ctx.decisions:
            if bool(d.active):
                outcomes.setdefault(d.site, set()).add(bool(d.outcome))
        rows.append(
            {
                "surface": surface.name,
                "storage": surface.storage.clone(),
                "pool": surface.nutrient_pool.storage.clone()
                if surface.nutrient_pool is not None
                else None,
                "days_after_sow": float(surface.days_after_sow),
                "crop_cover": float(surface.crop_cover),
                "total_decayed": surface.total_decayed.clone(),
                "outcomes": outcomes,
                "events": {c: bool(t["land"]) for c, t in ctx.events.items() if "land" in t},
                "carry_over": ctx.parts["in"].get("Surface.nitrite_carry_over"),
            }
        )
    return rows, network.pollutants


_N_TANK = ("ammonia", "nitrite", "nitrate", "org-nitrogen")  # the tank's N, the pool's DI + DO
_P_TANK = ("phosphate", "org-phosphorus")


def _gaps(tank: dict[str, float], pool) -> tuple[float, float]:
    """L8: the tank's N and P (``_N_TANK``, ``_P_TANK``, summed in that order) less the
    dissolved pools' (``DI + DO``), WSIMOD's or noodl's, by the same arithmetic."""
    gaps = []
    for column, names in ((0, _N_TANK), (1, _P_TANK)):
        total = 0.0
        for name in names:
            total = total + tank[name]
        gaps.append(total - float(pool[DI][column]) - float(pool[DO][column]))
    return gaps[0], gaps[1]


def _assert_the_pools_are_flushed_twice(ours, theirs, pollutants) -> None:
    """L8 (Y45): noodl's daily tank-pool gap (the tank's N and P less ``DI + DO``) equals
    WSIMOD's from the capture, bitwise, on every day given, and it grows far above rounding."""
    surface = ours[0]["surface"]
    gaps = []
    for day, (row, post) in enumerate(zip(ours, theirs, strict=True)):
        tank = {n: float(row["storage"][pollutants.column(n)]) for n in (*_N_TANK, *_P_TANK)}
        mine = _gaps(tank, row["pool"])
        wsimod = _gaps(
            post[f"surfaces.{surface}.storage"], post[f"surfaces.{surface}.nutrient_pool"]
        )
        assert mine == wsimod, (day, mine, wsimod)
        gaps.append((mine, tank))
    # above rounding (day 0's gap is ~1e-16): the largest gap exceeds a millionth of the
    # largest tank store of N and of P
    for column, names in ((0, _N_TANK), (1, _P_TANK)):
        store = max(sum(abs(tank[n]) for n in names) for _, tank in gaps)
        assert max(abs(gap[column]) for gap, _ in gaps) > 1e-6 * store, (names, store)


def _negative_columns(tanks, additive) -> list[tuple[int, str]]:
    return [(day, n) for day, tank in enumerate(tanks) for n in additive if tank[n] < 0.0]


def _assert_the_tank_columns_keep_their_sign(name, ours, theirs, pollutants) -> None:
    """L9: noodl's answer to "did an additive column of the growing tank go negative on some
    day" is WSIMOD's from the capture, and it is the measured one the docs state (L9 row): no
    column went negative, in either."""
    surface = ours[0]["surface"]
    additive = pollutants.additive
    mine = _negative_columns(
        [{n: float(r["storage"][pollutants.column(n)]) for n in additive} for r in ours], additive
    )
    wsimod = _negative_columns([post[f"surfaces.{surface}.storage"] for post in theirs], additive)
    answer = f"a negative column on {mine[:3]}" if mine else "no column went negative"
    assert mine == wsimod, f"{name}: noodl {answer}; WSIMOD {wsimod[:3]}"
    assert not mine, f"{name}: {answer}, in WSIMOD too: update the L9 row of the docs"


def _reached(rows, site: str) -> set[bool]:
    return set().union(*(r["outcomes"].get(site, ()) for r in rows))


# Y68: L8 on the spec's 120 days; L9 on 300, past day 266, where the nutrient_pool_not_tracking
# clamp (the negative-column risk) starts
_L8_DAYS = 120
_SPRING_L9_DAYS = 300


def test_growing_spring_flushes_its_pools_twice_keeps_its_signs_and_reaches_its_bands():
    """One noodl run of ``growing_spring`` over 300 days, against WSIMOD's capture (Y68):

    - L8 (Y45): ``effective_precipitation_flushing`` takes the outflows' nutrients out of the
      dissolved pools that ``pull_storage`` already took them from (``land.py:1458``,
      ``:1605-1645``), so the tank and its pool drift apart: noodl's gap equals WSIMOD's,
      bitwise, on each of the first 120 days, and it grows far above rounding (measured by day
      120: about 7.4 kg N and 0.44 kg P): the gap is reproduced, not a hard-coded ratio of it;
    - L9: the uptake and the denitrification take the pool's N as nitrate (``:1917``,
      ``:2145``), so ``pull_storage`` could ask a tank column for more than it holds; over the
      300 days no column goes negative, in either (``"nutrient_pool_not_tracking"`` from day
      266);
    - the run reaches the soil temperature factor's three bands (``warm`` and ``thawed`` each
      both ways, ``:1820-1829``), a stressed and an unstressed crop (``:1561-1568``), rain
      above 5 mm (``:1943``) and erosion's runoff branch both ways (``:1959``)."""
    name = "growing_spring"
    ours, pollutants = growing_days(name, _SPRING_L9_DAYS)
    theirs = [c.post for c in _wsimod_runs(name, _SPRING_L9_DAYS)]
    _assert_the_pools_are_flushed_twice(ours[:_L8_DAYS], theirs[:_L8_DAYS], pollutants)
    _assert_the_tank_columns_keep_their_sign(name, ours, theirs, pollutants)
    assert any(r["events"].get("nutrient_pool_not_tracking", False) for r in ours)
    wheat = "node:land/surfaces.wheat"
    for site in (
        "calc_temperature_dependence_factor/warm",
        "calc_temperature_dependence_factor/thawed",
        "calc_crop_cover/unstressed",
        "erosion/runoff",
    ):
        assert _reached(ours, f"{wheat}/{site}") == {True, False}, site
    mapping = CASES[name].build()
    rain = mapping["nodes"][0]["data_input_dict"]
    first = mapping["dates"][:_SPRING_L9_DAYS]
    assert sum(rain[("precipitation", d)] > 5e-3 for d in first) > 10


_GROWING_DECAY_DAYS = 100  # Y68: past the harvest (run day ~77) and every branch (day 66)


def test_growing_decay_keeps_its_signs_carries_its_decay_and_leaves_the_pool_behind():
    """One noodl run of ``growing_decay`` over its 100 default days, against WSIMOD's capture:

    - L9: the tank's decay (3b) bypasses the pool, and no tank column goes negative, in either;
    - 3b's carry-over on a growing tank (``land.py:427-439``): each day's ``Land.run`` adds the
      tank's decayed ammonia as nitrite and its decayed nitrite as nitrate. noodl's
      ``Surface.nitrite_carry_over`` part equals, bitwise on every day, the ``total_decayed``
      that WSIMOD's surface held when its run began (the capture's ``pre``), and it is not zero;
    - the tank decays ammonia and nitrite and raises ``"nutrient_pool_not_tracking"`` (D19) on
      some days, not all."""
    name = "growing_decay"
    assert CASES[name].days == _GROWING_DECAY_DAYS
    ours, pollutants = growing_days(name, _GROWING_DECAY_DAYS)
    runs = _wsimod_runs(name, _GROWING_DECAY_DAYS)
    _assert_the_tank_columns_keep_their_sign(name, ours, [c.post for c in runs], pollutants)
    carried = {"nitrite": 0.0, "nitrate": 0.0}
    for row, call in zip(ours, runs, strict=True):
        decayed = call.pre[f"surfaces.{row['surface']}.total_decayed"]
        for target, source in (("nitrite", "ammonia"), ("nitrate", "nitrite")):
            mine = float(row["carry_over"][pollutants.column(target)])  # a full VQIP
            assert mine == decayed[source], (call.day, target, mine, decayed[source])
            carried[target] += mine
    assert carried["nitrite"] > 0.0 and carried["nitrate"] > 0.0, carried
    for species in ("ammonia", "nitrite"):
        assert sum(float(r["total_decayed"][pollutants.column(species)]) for r in ours) > 0.0
    tracking = [r["events"].get("nutrient_pool_not_tracking", False) for r in ours]
    assert any(tracking) and not all(tracking)


# the crop from run day 7 (days_after_sow up to 113 by day 120, across the new year); the harvest
# on run day 291 (2004-07-18), so 292 days are the shortest run that crosses it (slow: 22 s
# against the default 120 days' 9 s, measured)
_AUTUMN_HARVEST_DAY = 291


@pytest.mark.parametrize(
    "days",
    [
        pytest.param(120, id="before-the-harvest"),
        pytest.param(_AUTUMN_HARVEST_DAY + 1, id="across-the-harvest", marks=pytest.mark.slow),
    ],
)
def test_growing_autumn_grows_a_crop_without_cover(days):
    """``growing_autumn`` (L2) grows a crop with no cover (its unsorted calendar) on every day,
    and its uptake reaches the autumn ``temp_func`` and ``days_after_sow - 25``
    (``:1876-1879``), with ``days_after_sow`` both below and above 25. By default over its
    first 120 days; under ``slow`` over 292, the whole season through its harvest on run day
    291, after which the crop is gone (``days_after_sow`` is ``None``, here -1)."""
    autumn, _ = growing_days("growing_autumn", days)
    site = "node:land/surfaces.barley/calc_crop_uptake"
    reached = [r for r in autumn if f"{site}/temperature_cap" in r["outcomes"]]
    assert reached and all(r["days_after_sow"] > 0 for r in reached)
    assert any(r["days_after_sow"] < 25 for r in reached)
    assert any(r["days_after_sow"] > 25 for r in reached)
    assert all(r["crop_cover"] == 0.0 for r in autumn)  # L2: the unsorted calendar
    if days > _AUTUMN_HARVEST_DAY:
        assert autumn[_AUTUMN_HARVEST_DAY - 1]["days_after_sow"] > 0
        assert autumn[_AUTUMN_HARVEST_DAY]["days_after_sow"] == -1.0  # harvested


# L1: the first three dates are a leap year's (28 and 29 February are adjusted days 59, 1 March is
# 60, as intended); the others are common years' 28 February (day 59) and 1 March (raw day 60,
# shifted to 59: the defect)
LEAP_DATES = (
    "2004-02-28",
    "2004-02-29",
    "2004-03-01",
    "2005-02-28",
    "2005-03-01",
    "2006-02-28",
    "2006-03-01",
)


def test_growing_spring_spans_the_leap_shift():
    """L1: WSIMOD tests ``t.is_leap_year``, a bound method and so always true (``land.py:1526``),
    and shifts the day of year after 28 February in every year. ``growing_spring``'s dates (its
    full three years) hold 29 February 2004 and the 1 March of 2005 and 2006, and its crop
    factor rises over days 0-90, so the shift shows: in 2004 28 and 29 February share a factor
    and 1 March has the next one; in 2005 and 2006 28 February and 1 March share 2004's
    29 February factor. noodl's
    land replayed on the captured calls of those dates equals WSIMOD's stores bitwise, the
    ``crop_factor`` and ``et0_coefficient`` among them."""
    spec = CASES["growing_spring"]
    dates = spec.build()["dates"]
    days = [dates.index(d) for d in LEAP_DATES]
    calls = [c for c in growing_capture("growing_spring", spec.slow_days) if c.day in days]
    runs = {c.day: c.post for c in calls if c.method == "run"}
    factor = {
        d: runs[day]["surfaces.wheat.crop_factor"] for d, day in zip(LEAP_DATES, days, strict=True)
    }
    assert "surfaces.wheat.et0_coefficient" in runs[days[0]]
    assert factor["2004-02-28"] == factor["2004-02-29"] != factor["2004-03-01"]
    for year in ("2005", "2006"):  # the shift in a common year: day 59 twice
        assert factor[f"{year}-02-28"] == factor[f"{year}-03-01"] == factor["2004-02-29"]
    network = read_wsimod_config(spec.build())
    assert replay(network, "land", calls, scale=spec.scale) == 0.0


# --- the growing branches (Task 9b; spec, Verification 3) ---------------------------------------
# The transformation, uptake, erosion, denitrification and adsorption sites, and the two factors'
# branches (engine.py's vocabulary, ``node:land/surfaces.<s>/<site>``), each with both outcomes.
# ``calc_crop_uptake/temperature_*`` exist only on an autumn-sown surface.
GROWING_SITES = (
    "calc_temperature_dependence_factor/warm",
    "calc_temperature_dependence_factor/thawed",
    "calc_soil_moisture_dependence_factor/saturated",
    "calc_soil_moisture_dependence_factor/wilted",
    "calc_soil_moisture_dependence_factor/min:wp",
    "calc_soil_moisture_dependence_factor/min:fc",
    "soil_pool_transformation/inorganic_N",
    "soil_pool_transformation/organic_N",
    "soil_pool_transformation/inorganic_P",
    "soil_pool_transformation/organic_P",
    "calc_crop_uptake/crop",
    "calc_crop_uptake/temperature_cap",
    "calc_crop_uptake/temperature_floor",
    "calc_crop_uptake/available_water",
    "calc_crop_uptake/cap:P",
    "calc_crop_uptake/cap:N",
    "erosion/rain",
    "erosion/runoff",
    "erosion/transport",
    "erosion/enrichment_cap",
    "erosion/flow",
    "erosion/phosphorus",
    "denitrification/saturated",
    "denitrification/moist",
    "adsorption/empty",
    "adsorption/shortcut",
    "adsorption/change",
    "adsorption/adsorb",
)
# The eight outcomes growing_spring never shows, in its 365 days or its 1096 (measured), and
# which growing_lean_soil shows instead (besides the two temperature_* sites, which a spring
# sowing does not have at all; lean soil's 45 degC days cap the function): the spec's claim that
# growing_spring reaches every branch was measured false (Task 9a review, finding 9), and
# growing_lean_soil was built for these.
LEAN_SOIL_ONLY = {
    "calc_soil_moisture_dependence_factor/wilted": True,  # its soil starts below wilting point
    "soil_pool_transformation/inorganic_P": True,  # no dissolved P, fast-pool P mineralises
    "adsorption/empty": True,  # no P in the dissolved and adsorbed pools while it is wilted
    "adsorption/shortcut": True,  # no adsorbed P: WSIMOD prints its warning
    "adsorption/change": False,  # rye's first shortcut: an equilibrium within 1e-6 of none
    "adsorption/adsorb": False,  # clover's 600 kg of adsorbed P desorb
    "calc_crop_uptake/available_water": False,  # a crop on a soil below wilting point
    "erosion/phosphorus": False,  # nothing erodes P on its P-free days
}
# noodl-only guards: where WSIMOD raises (D13), noodl takes the evident intent and records the
# event, so only a crash case reaches their true outcome: first on its crash day, never before
# (WSIMOD stops there; noodl's guard can stay true on later days). Every case WSIMOD runs shows
# them false only.
NOODL_ONLY_GUARDS = {
    "pull_storage/no_din": "growing_no_din",  # L5, land.py:1460
    "soil_pool_transformation/no_din": "growing_zero_n",  # L6, :1758
    "calc_crop_uptake/dry_soil": "growing_empty_soil",  # L7, :1899
    "calc_crop_cover/stress_zero_denominator": "growing_default_depletion",  # L3, :1566
}
# Pinned elsewhere, not by a whole-model case: ``calc_crop_uptake/shape`` is true for any
# positive uptake parameters (its false outcome needs ``uptake2 + uptake_par <= 0``, Task 7's
# unit tests), and the Newton passes ``adsorption/newton:<j>/{residual,step,positive}`` each exit
# (residual, step, maxiter, and the rounding-only ``positive`` clamp) in Task 8b's unit tests;
# here the loop only runs, and stops by its step and by its residual.


def _site_outcomes(rows, days: int) -> dict[str, set[bool]]:
    """``{site after "surfaces.<s>/": outcomes}`` over every surface of the land, days < days."""
    out: dict[str, set[bool]] = {}
    for row in rows[:days]:
        for site, outcomes in row["outcomes"].items():
            if site.startswith("node:land/surfaces.") and "/" in site[len("node:land/") :]:
                key = site[len("node:land/") :].split("/", 1)[1]
                out.setdefault(key, set()).update(outcomes)
    return out


@pytest.mark.slow
def test_growing_spring_reaches_every_branch():
    """From the decisions (spec, Verification 3; Task 9b): on ``growing_spring``'s default 365
    days and on its full three years, each site of ``GROWING_SITES`` is active, with both
    outcomes, except the outcomes of ``LEAN_SOIL_ONLY``, which it never shows and
    ``growing_lean_soil`` does. Each ``NOODL_ONLY_GUARDS`` site, a noodl guard where WSIMOD
    raises, is false on every day of both, and first true on its crash case's crash day, never
    before. The Newton
    loop runs, stops by its step on ``growing_spring`` and by its residual on
    ``growing_lean_soil``."""
    spec = CASES["growing_spring"]
    spring, _ = growing_days("growing_spring", spec.slow_days)
    lean, _ = growing_days("growing_lean_soil", CASES["growing_lean_soil"].days)
    lean_seen = _site_outcomes(lean, len(lean))
    for days in (spec.days, spec.slow_days):
        seen = _site_outcomes(spring, days)
        for site in GROWING_SITES:
            if site.startswith("calc_crop_uptake/temperature_"):
                assert site not in seen, site  # a spring sowing has no temperature function
                assert lean_seen[site] == BOTH, site
                continue
            missing = LEAN_SOIL_ONLY.get(site)
            expected = BOTH if missing is None else BOTH - {missing}
            assert seen[site] == expected, (days, site, seen[site])
            assert lean_seen[site] >= BOTH - expected, (site, lean_seen[site])
        for site in NOODL_ONLY_GUARDS:
            assert seen[site] == {False}, (days, site)
        assert any(seen.get(f"adsorption/newton:{j}/step") == BOTH for j in range(20))
    for site in NOODL_ONLY_GUARDS:
        assert lean_seen[site] == {False}, site
    assert any(False in lean_seen.get(f"adsorption/newton:{j}/residual", ()) for j in range(20))
    for site, name in NOODL_ONLY_GUARDS.items():
        rows, _ = growing_days(name, CASES[name].crash_day + 1)
        assert _site_outcomes(rows[CASES[name].crash_day :], 1)[site] >= {True}, (name, site)
        assert True not in _site_outcomes(rows, CASES[name].crash_day).get(site, set())


# --- the store-bounded prints (Task 9b; D12, Y41) -----------------------------------------------
@pytest.mark.parametrize("name", sorted(STORE_BOUNDED))
def test_the_store_bounded_prints(name):
    """D12: a mapped case's WSIMOD run prints round-off pollutant lines at its land over its
    default days (so its mapping is measured, not idle), each of an additive pollutant and
    within ``GROWING_PRINT_C * eps * S``; the parity item admits exactly these. The volume lines
    stay W86's comparison (``test_the_store_bound_admits_nothing_else``)."""
    reference, print_days = case_theirs_by_day(name)
    lines = store_bounded_lines(reference, print_days, "land")
    assert lines
    assert {line["pollutant"] for line in lines} <= set(reference.additive) - {"volume"}
    ratio = max(abs(line["x"]) / (EPSILON * line["S"]) for line in lines)
    assert 0.0 < ratio <= GROWING_PRINT_C, ratio
    assert all(line["S"] > 0.0 for line in lines)


def test_the_store_bound_admits_nothing_else():
    """D12 on ``growing_lean_soil``: without its mapping its round-off lines fail (that is how
    an unmapped case asserts zero lines); with it, a line beyond ``c * eps * S``, a volume line
    of the land as small as the round-off and a line of a non-additive pollutant each fail, and
    the admission needs the print days."""
    name = "growing_lean_soil"
    results, diagnostics = case_ours(name, "pollutants")
    reference, print_days = case_theirs_by_day(name)
    kwargs = {"diagnostics": diagnostics, "case": name, "quality": True}
    mapped = {"land": GROWING_PRINT_C}
    found = assert_matches_wsimod(
        results, reference, print_days=print_days, store_bounded_prints=mapped, **kwargs
    )
    assert found["worst"] == 0.0
    with pytest.raises(AssertionError, match="WSIMOD printed 'mass balance error for org-"):
        assert_matches_wsimod(results, reference, print_days=print_days, **kwargs)
    with pytest.raises(ValueError, match="store_bounded_prints needs print_days"):
        assert_matches_wsimod(results, reference, store_bounded_prints=mapped, **kwargs)
    first = store_bounded_lines(reference, print_days, "land")[0]
    k = reference.printed.index(first["line"])
    over = (GROWING_PRINT_C + 0.5) * EPSILON * first["S"]
    for line, error in (
        (f"mass balance error for {first['pollutant']} of {over!r} in land", "store bound"),
        ("mass balance error for volume of 1e-17 in land", "mass balance errors"),
        ("mass balance error for temperature of 1e-17 in land", "no additive pollutant"),
    ):
        printed = [*reference.printed[:k], line, *reference.printed[k + 1 :]]
        changed = dataclasses.replace(reference, printed=printed)
        with pytest.raises(AssertionError, match=error):
            assert_matches_wsimod(
                results, changed, print_days=print_days, store_bounded_prints=mapped, **kwargs
            )
