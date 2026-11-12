"""Whole-model parity: noodl's WSIMOD port against WSIMOD 0.8.1 on its two demos and on every
purpose-built case (spec, Verification 2 and 3; Task 17).

``quickstart`` runs in full (1456 days), ``oxford`` for its first 365 days by default and in full
under ``-m slow``. Each demo runs twice: quantity-only on ``config.yml`` (WSIMOD runs its own
pollutants, noodl the volumes alone, the volumes compared) and in pollutant mode on
``config_no_deposition.yml`` (every pollutant compared). Every class replays WSIMOD bitwise
(replay bound 0), so the whole model does: the worst difference of every run is exactly 0, and
no admission is needed. The bound of ``assert_matches_wsimod`` (``1e-12 * scale``) is only the
ceiling.

The purpose-built cases (``_wsimod_cases.CASES``) run in full, in both modes, the same way. Cases
WSIMOD cannot run (``wtw_zero_input``, ``nonresidential_demand``) or runs only until a documented
crash (``land_sewer_ponding`` with a lowered sewer capacity) are tested for the crash and for
noodl's evident-intent run (W4).
"""

from __future__ import annotations

import contextlib
import copy
import functools
import io
import re
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
from noodl.apps.wsimod.engine import Decision  # noqa: E402
from noodl.apps.wsimod.structure import river_order  # noqa: E402
from tests.verification._wsimod_cases import CASES  # noqa: E402
from tests.verification._wsimod_parity import (  # noqa: E402
    BOUND,
    Admission,
    assert_matches_wsimod,
    mass_balance_worst,
    wsimod_reference_by_day,
)
from tests.verification._wsimod_reference import wsimod_model, wsimod_reference  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "wsimod"
CONFIG = {"quantity": "config.yml", "pollutants": "config_no_deposition.yml"}
FULL = 1456  # both demos' dates

# 3a runs no WTW processes on the pollutant demos (sub-project 3b); the task that ports the
# process removes its marks (plan D1)
WTW = pytest.mark.xfail(strict=True, raises=ValueError, reason="the WTW processes are Task 3")

DEMOS = [
    pytest.param("quickstart", "quantity", FULL, id="quickstart-quantity"),
    pytest.param("quickstart", "pollutants", FULL, id="quickstart-pollutants"),
    pytest.param("oxford", "quantity", 365, id="oxford365-quantity"),
    pytest.param("oxford", "pollutants", 365, id="oxford365-pollutants", marks=WTW),
    pytest.param("oxford", "quantity", FULL, id="oxford-quantity", marks=pytest.mark.slow),
    pytest.param(
        "oxford", "pollutants", FULL, id="oxford-pollutants", marks=[pytest.mark.slow, WTW]
    ),
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


@pytest.mark.parametrize(
    "case",
    [pytest.param("quickstart"), pytest.param("oxford", marks=WTW)],
)
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


@WTW
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
CASE_RUNS = [
    pytest.param(name, mode, id=f"{name}-{mode}")
    for name, spec in CASES.items()
    if spec.wsimod
    for mode in spec.modes
]


@functools.cache
def case_theirs_by_day(name: str):
    return wsimod_reference_by_day(CASES[name].build())


def case_theirs(name: str):
    return case_theirs_by_day(name)[0]


@functools.cache
def case_ours(name: str, mode: str):
    model, state, drivers = build_model(
        read_wsimod_config(CASES[name].build()), pollutants=MODES[mode]
    )
    diagnostics: dict = {}
    results = wsimod_run(model, state, drivers, diagnostics=diagnostics)
    diagnostics["node_names"] = model.wsimod_day.node_names
    return results, diagnostics


@pytest.mark.parametrize(("name", "mode"), CASE_RUNS)
def test_case_matches_wsimod(name, mode):
    """Bitwise on every day, both modes: arc flows, every recorded tank and surface field (and
    every pollutant in ``pollutants`` mode); every WSIMOD print is one noodl records on the same
    node and day or in the same count (``assert_matches_wsimod``); the mass balance closes."""
    results, diagnostics = case_ours(name, mode)
    reference, print_days = case_theirs_by_day(name)
    found = assert_matches_wsimod(
        results,
        reference,
        diagnostics=diagnostics,
        case=f"{name}-{mode}",
        quality=MODES[mode],
        print_days=print_days,
    )
    assert found["worst"] == 0.0 and found["admissions"] == []
    assert len(results.dates) == CASES[name].days
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


@pytest.mark.parametrize("name", [n for n, spec in CASES.items() if set(spec.modes) >= set(MODES)])
def test_quantity_only_equals_the_pollutant_run_volumes(name):
    """Spec, Verification 3, on every case: a quantity-only run's volumes equal the pollutant
    run's, exactly (``torch.equal``): every arc flow, tank storage and surface field."""
    quantity, _ = case_ours(name, "quantity")
    pollutants, _ = case_ours(name, "pollutants")
    assert torch.equal(quantity.arc_flow, pollutants.arc_flow)
    assert torch.equal(quantity.tank_storage, pollutants.tank_storage)
    for field, values in quantity.surface.items():
        other = pollutants.surface[field]
        assert torch.equal(torch.isnan(values), torch.isnan(other)), field
        assert torch.equal(values.nan_to_num(), other.nan_to_num()), field


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
    node_type_override""".split()
    assert set(listed) <= set(CASES)
    assert {n for n, spec in CASES.items() if not spec.wsimod} == {
        "wtw_zero_input",
        "nonresidential_demand",
    }


def tiny_dropped(results) -> torch.Tensor:
    """The volume the run's mass balance counts as dropped, summed over the days."""
    return results.mass_balance.terms["dropped"][:, 0].sum()


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
