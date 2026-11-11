"""Whole-model parity: noodl's WSIMOD port against WSIMOD 0.8.1 on its two demos (spec,
Verification 2 and 3).

``quickstart`` runs in full (1456 days), ``oxford`` for its first 365 days by default and in full
under ``-m slow``. Each demo runs twice: quantity-only on ``config.yml`` (WSIMOD runs its own
pollutants, noodl the volumes alone, the volumes compared) and neutralised on
``config_neutral.yml`` (every pollutant compared). Every class replays WSIMOD bitwise (replay bound
0), so the whole model does: the worst difference of every run is exactly 0, and no admission is
needed. The bound of ``assert_matches_wsimod`` (``1e-12 * scale``) is only the ceiling.
"""

from __future__ import annotations

import copy
import functools
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
from tests.verification._wsimod_parity import (  # noqa: E402
    BOUND,
    Admission,
    assert_matches_wsimod,
    mass_balance_worst,
)
from tests.verification._wsimod_reference import wsimod_model, wsimod_reference  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "wsimod"
CONFIG = {"quantity": "config.yml", "neutral": "config_neutral.yml"}
FULL = 1456  # both demos' dates

DEMOS = [
    pytest.param("quickstart", "quantity", FULL, id="quickstart-quantity"),
    pytest.param("quickstart", "neutral", FULL, id="quickstart-neutral"),
    pytest.param("oxford", "quantity", 365, id="oxford365-quantity"),
    pytest.param("oxford", "neutral", 365, id="oxford365-neutral"),
    pytest.param("oxford", "quantity", FULL, id="oxford-quantity", marks=pytest.mark.slow),
    pytest.param("oxford", "neutral", FULL, id="oxford-neutral", marks=pytest.mark.slow),
]


@functools.cache
def theirs(case: str, mode: str, days: int):
    return wsimod_reference(DATA / case, config_name=CONFIG[mode], days=days)


@functools.cache
def ours(case: str, mode: str, days: int, decisions: bool = False):
    network = read_wsimod_config(DATA / case, config_name=CONFIG[mode])
    model, state, drivers = build_model(network, pollutants=(mode == "neutral"))
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
        quality=(mode == "neutral"),
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
def test_quantity_only_equals_the_neutralised_volumes(case):
    """Spec, Verification 3: a quantity-only run's volumes equal the neutralised run's, exactly
    (``torch.equal``): every arc flow, tank storage and surface field."""
    days = FULL if case == "quickstart" else 365
    quantity, _ = ours(case, "quantity", days)
    neutral, _ = ours(case, "neutral", days)
    assert torch.equal(quantity.arc_flow, neutral.arc_flow)
    assert torch.equal(quantity.tank_storage, neutral.tank_storage)
    for field, values in quantity.surface.items():
        other = neutral.surface[field]
        assert torch.equal(torch.isnan(values), torch.isnan(other)), field  # not recorded
        assert torch.equal(values.nan_to_num(), other.nan_to_num()), field


def test_oxfords_urban_surface_holds_no_storage_after_the_first_day():
    """Review Focus 4, through the whole model: the urban (impervious) surface's storage is 0
    after day 1 to rounding, and bitwise WSIMOD's. The impervious surface stores nothing between
    days; the residue (at most 7.3e-12 m3 against the 1.16e6 m3 of day 1, measured) is the
    rounding of its balance, which WSIMOD has to the bit."""
    results, _ = ours("oxford", "neutral", 365)
    reference = theirs("oxford", "neutral", 365)
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
    results, diagnostics = ours("quickstart", "neutral", days, True)
    return copy.deepcopy(results), diagnostics, theirs("quickstart", "neutral", days)


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
    names = list(ours("quickstart", "neutral", 10, True)[1]["node_names"])
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
