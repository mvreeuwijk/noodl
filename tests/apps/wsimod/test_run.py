"""``wsimod_run``: WSIMOD's loop over the dates, its records, the diagnostics and the mass
balance (Task 14).

The ``small`` fixture (D9) is ``inflow`` (Catchment) -> ``river`` (River) -> ``outlet``
(Waste), 10 dates. The mass-balance tests cover the rulings of Task 14: W49 (a push below
``FLOAT_ACCURACY`` into a ``QueueTank`` is storage, not a drop), W69 (water pushed into a
``WWTW`` after its ``calculate_discharge`` vanishes and is counted), W75 (only carried stores
are storage; a ``Land``'s outflow is IHACRES's evaporation) and every drop site.
"""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from noodl.apps.wsimod import (
    EVENT_NAMES,
    WsimodResults,
    build_model,
    initial_drivers,
    initial_state,
    read_wsimod_config,
    wsimod_run,
)
from tests.verification._wsimod_cases import CASES

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tests" / "data" / "wsimod"
DAY = 86400.0
RUN_CASES = [(name, mode) for name, c in CASES.items() for mode in c.modes]


def _small():
    model, state, drivers = build_model(read_wsimod_config(DATA / "small"))
    return model, state, drivers


def _case(name: str, mode: str = "pollutants"):
    network = read_wsimod_config(CASES[name].build())
    return build_model(network, pollutants=(mode == "pollutants"))


def _tensors(*values):
    return torch.tensor(values, dtype=torch.float64)


# --- against WSIMOD ---------------------------------------------------------------------------
def test_the_small_model_matches_wsimod():
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_reference

    model, state, drivers = _small()
    ours = wsimod_run(model, state, drivers)
    theirs = wsimod_reference(DATA / "small")
    assert ours.dates == theirs.dates or len(ours.dates) == len(theirs.dates) == 10
    assert ours.arc_names == theirs.arc_names
    scale = float(theirs.arc_flow.abs().max())
    assert float((ours.arc_flow - theirs.arc_flow).abs().max()) <= 1e-12 * scale
    assert ours.tank_names == theirs.tank_names
    scale = float(theirs.tank_storage.abs().max())
    river = theirs.tank_names.index(("river", "tank"))
    assert float((ours.tank_storage - theirs.tank_storage).abs().max()) <= 1e-12 * scale
    assert float(ours.tank_storage[:, river].abs().max()) > 0.0


def _printed(lines: list[str], pattern: str) -> set[tuple[str, str]]:
    found = set()
    for line in lines:
        match = re.fullmatch(pattern, line)
        if match:
            found.add(match.groups())
    return found


@pytest.mark.parametrize(
    ("case", "event", "pattern"),
    [
        ("maxiter_pull", "maxiter", r"Maxiter reached in (\S+) at (.+)"),
        ("distribution_leakage", "demand_deficit", r"demand deficit of \S+ at (\S+) on (.+)"),
    ],
)
def test_the_prints_that_carry_the_node_and_the_date_match_the_records(case, event, pattern):
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_reference

    theirs = wsimod_reference(CASES[case].build())
    model, state, drivers = _case(case)
    diagnostics: dict = {}
    wsimod_run(model, state, drivers, diagnostics=diagnostics)
    names = model.wsimod_day.node_names
    if event == "maxiter":
        ours = {
            (site[0], theirs.dates[d])
            for site, flags in diagnostics["maxiter"].items()
            for d in range(len(theirs.dates))
            if bool(flags[d])
        }
    else:
        flags = diagnostics["events"][event]
        ours = {
            (names[i], theirs.dates[d])
            for d in range(flags.shape[0])
            for i in range(flags.shape[1])
            if bool(flags[d, i])
        }
    printed = _printed(theirs.printed, pattern)
    assert printed
    assert ours == printed


# --- running ----------------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["small", "queue_arc"])
def test_step_by_step_equals_wsimod_run(case):
    if case == "small":
        model, state, drivers = _small()
    else:
        model, state, drivers = _case("queue_arc")
    ran = wsimod_run(model, state, drivers, days=10)
    final = dict(model.wsimod_day.last_state)
    stepped, flows = state, []
    for k in range(10):
        stepped = model.step(stepped, drivers, dt=DAY, t=k * DAY)
        flows.append(model.wsimod_day.last_record.arc_flow)
    assert torch.equal(torch.stack(flows), ran.arc_flow)
    for key, value in final.items():
        assert torch.equal(stepped[key], value), key


def test_a_batch_equals_its_single_runs():
    """Review Focus 2: four instances take different branches on the same day. Instance 0 is
    over-subscribed at the junction on day 3 (its capacities leave 5e-12 on an arc below
    ``FLOAT_ACCURACY``, which the check counts and ``get_connected`` zeroes), instance 1 is
    not; instances 2 and 3 take different IHACRES branches on day 5 (a dry soil evaporates
    from its store)."""
    model, _, _ = _case("batch_divergence")
    caps = {
        "junction_to_a": _tensors(2.0, 20.0, 20.0, 20.0),
        "junction_to_b": _tensors(3.0, 30.0, 30.0, 30.0),
        "junction_to_c": _tensors(5e-12, 10.0, 10.0, 10.0),
    }
    soil = _tensors(1700.0, 1700.0, 1700.0, 300.0)
    state = initial_state(model, values={"wsimod.storage": {("farm", "surface:grass"): soil}})
    drivers = initial_drivers(model, values={"wsimod.Arc.capacity": caps})
    diagnostics = {"decisions": True}
    batch = wsimod_run(model, state, drivers, diagnostics=diagnostics)
    assert batch.arc_flow.shape == (4, 40, 4)

    def outcome(site: str, day: int) -> list[bool]:
        (record,) = [r for r in diagnostics["decisions"][day] if r.site == site]
        return record.outcome.tolist()

    site = "node:junction/push_distributed/oversubscribed"
    assert outcome(site, 2) == [True, False, False, False]
    kink = outcome("node:farm/surfaces.grass/ihacres/evaporation_kink", 4)
    assert kink[2] != kink[3]
    for i in range(4):
        single_state = initial_state(
            model, values={"wsimod.storage": {("farm", "surface:grass"): float(soil[i])}}
        )
        single_drivers = initial_drivers(
            model, values={"wsimod.Arc.capacity": {k: float(v[i]) for k, v in caps.items()}}
        )
        single = wsimod_run(model, single_state, single_drivers)
        assert torch.equal(single.arc_flow, batch.arc_flow[i])
        assert torch.equal(single.tank_storage, batch.tank_storage[i])
        for name, value in single.surface.items():
            assert torch.equal(torch.nan_to_num(value), torch.nan_to_num(batch.surface[name][i]))
        assert torch.equal(single.mass_balance.residual, batch.mass_balance.residual[i])


def test_the_results_follow_wsimods_record_layout():
    model, state, drivers = _case("residential_with_land")
    results = wsimod_run(model, state, drivers, days=7)
    assert isinstance(results, WsimodResults)
    t, k = 7, model.wsimod_day.p.K
    assert results.dates == [f"2000-01-{d:02d}" for d in range(1, 8)]
    assert results.arc_flow.shape == (t, len(results.arc_names))
    assert results.arc_quality.shape == (t, len(results.arc_names), k)
    assert results.tank_storage.shape == (t, len(results.tank_names))
    assert results.tank_quality.shape == (t, len(results.tank_names), k)
    n = len(results.surface_names)
    assert n > 0
    assert set(results.surface) == {
        "storage",
        "evaporation",
        "precipitation",
        "percolation",
        "subsurface_r",
        "surface_r",
        "tank_recharge",
    }
    assert all(v.shape == (t, n) for v in results.surface.values())
    assert results.surface_quality.shape == (t, n, k)
    assert results.mass_balance.residual.shape == (t, 1 + 2)  # volume, phosphate, ammonia
    # record_all's walk (model.py:1119-1133): no surface is a tank, and WSIMOD records nothing
    # for an impervious surface's flows
    assert all(not attribute.startswith("surface:") for _, attribute in results.tank_names)
    model, state, drivers = _case("impervious_only")
    results = wsimod_run(model, state, drivers, days=5)
    for field in ("percolation", "subsurface_r", "surface_r", "tank_recharge"):
        assert torch.isnan(results.surface[field]).all(), field
    for field in ("storage", "evaporation", "precipitation"):
        assert not torch.isnan(results.surface[field]).any(), field


def test_stepping_outside_the_dates_is_refused():
    model, state, drivers = _small()
    for t in (None, 43200.0, 10 * DAY):
        with pytest.raises(ValueError, match="t"):
            model.step(state, drivers, dt=DAY, t=t)
    for days in (0, 11):
        with pytest.raises(ValueError, match="days"):
            wsimod_run(model, state, drivers, days=days)
    mapping = copy.deepcopy(CASES["three_out_arcs"].build())
    del mapping["dates"]
    for node in mapping["nodes"]:
        node.pop("data_input_dict", None)
    with pytest.raises(ValueError, match="dates"):
        build_model(read_wsimod_config({**mapping, "orchestration": []}))


def test_wsimod_is_never_imported():
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "import noodl.apps.wsimod as w\n"
        f"model, state, drivers = w.build_model(w.read_wsimod_config(Path(r'{DATA / 'small'}')))\n"
        "w.wsimod_run(model, state, drivers)\n"
        "assert 'wsimod' not in sys.modules, sorted(m for m in sys.modules if 'wsimod' in m)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=300
    )
    assert done.returncode == 0, done.stderr


def test_a_run_without_grad_builds_no_graph():
    model, state, drivers = _small()
    plain = wsimod_run(model, state, drivers)
    assert not plain.arc_flow.requires_grad
    assert not plain.mass_balance.residual.requires_grad
    row = model.wsimod_day.state_labels("wsimod.storage").index(("river", "tank"))
    leaf = state["wsimod.storage"].clone()
    leaf[row, 0] = leaf[row, 0] + 1.0
    leaf = leaf.requires_grad_(True)
    tracked = wsimod_run(model, {**state, "wsimod.storage": leaf}, drivers)
    assert tracked.arc_flow.requires_grad
    expected = wsimod_run(model, {**state, "wsimod.storage": leaf.detach()}, drivers)
    assert not expected.arc_flow.requires_grad
    assert torch.equal(tracked.arc_flow, expected.arc_flow)
    assert torch.equal(tracked.tank_storage, expected.tank_storage)
    tracked.tank_storage.sum().backward()
    assert leaf.grad is not None and float(leaf.grad[row, 0]) != 0.0


# --- diagnostics ------------------------------------------------------------------------------
def test_diagnostics_record_maxiter_events_and_decisions():
    model, state, drivers = _case("maxiter_pull")
    diagnostics: dict = {}
    results = wsimod_run(model, state, drivers, diagnostics=diagnostics)
    assert "decisions" not in diagnostics
    assert set(diagnostics["events"]) >= set(EVENT_NAMES)
    n_nodes = len(model.wsimod_day.node_names)
    assert all(v.shape == (40, n_nodes) for v in diagnostics["events"].values())
    assert all(v.dtype == torch.bool for v in diagnostics["events"].values())
    flags = diagnostics["maxiter"]
    assert flags and all(v.shape == (40,) for v in flags.values())
    assert any(bool(v.any()) for v in flags.values())
    assert diagnostics["mass_balance"] is results.mass_balance
    asked = {"decisions": True}
    wsimod_run(model, state, drivers, days=3, diagnostics=asked)
    assert len(asked["decisions"]) == 3 and asked["decisions"][0]
    # the flag is not sticky: a later run without it records none
    assert model.wsimod_day.record_decisions is False


# --- the mass balance -------------------------------------------------------------------------
def test_the_mass_balance_closes_by_hand():
    model, state, drivers = _small()
    network = model.wsimod_day.network
    results = wsimod_run(model, state, drivers)
    flows = _tensors(*[network.nodes[0].data[("flow", d)] for d in network.dates])
    balance = results.mass_balance
    assert balance.inflow.shape == (10, 1)
    assert torch.equal(balance.inflow[:, 0], flows)
    scale = float(flows.abs().max())
    assert float(balance.residual.abs().max()) <= 1e-12 * scale
    assert set(balance.terms) == {
        "Catchment.flow",
        "Catchment.unrouted",
        "Waste.total_in",
        "dropped",
    }
    assert float(balance.outflow.sum()) > 0.0 and float(balance.storage_change.abs().sum()) > 0.0
    # volume leaves as water that reached the outlet, plus what the river keeps
    reached = balance.terms["Waste.total_in"][:, 0].sum()
    kept = balance.storage_change[:, 0].sum()
    assert float(flows.sum() - reached - kept - balance.terms["Catchment.unrouted"].sum()) == (
        pytest.approx(0.0, abs=1e-12 * scale)
    )


@pytest.mark.parametrize(("name", "mode"), RUN_CASES)
def test_the_mass_balance_closes_on_every_case(name, mode):
    """Every case, both modes: the residual is within ``1e-12 * scale`` on every day, the
    ``"inaccurate_evaporation"`` days apart (W75, tested below)."""
    model, state, drivers = _case(name, mode)
    diagnostics: dict = {}
    results = wsimod_run(model, state, drivers, diagnostics=diagnostics)
    residual = results.mass_balance.residual.abs().amax(dim=-1)
    event = diagnostics["events"]["inaccurate_evaporation"].any(dim=-1)
    bound = 1e-12 * CASES[name].scale
    assert float(residual[~event].max()) <= bound
    if name != "evaporation_clamp":
        assert not bool(event.any())
    # the balance is not trivially zero: something moves
    balance = results.mass_balance
    assert float(balance.outflow.abs().sum() + balance.inflow.abs().sum()) > 0.0 or float(
        balance.storage_change.abs().sum()
    ) > 0.0


def test_drops_are_recorded_where_waters_are_dropped():
    """The ``QueueArc`` tiny push (``queue_arc``) and the ``Land.run`` percolation push-back
    (``land_percolation_threshold``) drop water below ``FLOAT_ACCURACY``; it is in ``"dropped"``
    and the balance still closes."""
    for name in ("queue_arc", "land_percolation_threshold"):
        model, state, drivers = _case(name)
        results = wsimod_run(model, state, drivers)
        dropped = results.mass_balance.terms["dropped"]
        assert float(dropped[:, 0].sum()) > 0.0, name
        assert float(results.mass_balance.residual.abs().max()) <= 1e-12 * CASES[name].scale


def _tiny_sewer_push(float_accuracy: float, tiny: float) -> dict:
    """A reservoir holding ``tiny`` (below ``float_accuracy``) pushes it into a ``Sewer`` with
    a pipe time, i.e. into a ``QueueTank``, on day 1; then nothing moves."""
    dates = [f"2000-01-{d:02d}" for d in range(1, 6)]
    return {
        "float_accuracy": float_accuracy,
        "pollutants": [],
        "additive_pollutants": [],
        "non_additive_pollutants": [],
        "dates": dates,
        "nodes": [
            {"name": "basin", "type_": "Reservoir", "capacity": 30.0, "initial_storage": tiny},
            {"name": "sewer", "type_": "Sewer", "capacity": 25.0, "pipe_time": 1},
            {"name": "sea", "type_": "Waste"},
        ],
        "arcs": [
            {"name": "basin_to_sewer", "type_": "Arc", "in_port": "basin", "out_port": "sewer"},
            {"name": "sewer_to_sea", "type_": "Arc", "in_port": "sewer", "out_port": "sea"},
        ],
        "orchestration": [{"Reservoir": "distribute"}, {"Sewer": "make_discharge"}],
    }


def test_a_tiny_push_into_a_queue_tank_is_storage_not_a_drop():
    """W49: the push below ``FLOAT_ACCURACY`` stays in the ``QueueTank``'s ``storage`` and never
    enters its queue. The arc does not record it as dropped (``record_drop=False``), so the
    storage change counts it once: the balance closes and ``"dropped"`` is zero."""
    mapping = _tiny_sewer_push(1e-3, 5e-4)
    model, state, drivers = build_model(read_wsimod_config(mapping))
    results = wsimod_run(model, state, drivers)
    balance = results.mass_balance
    assert float(balance.terms["dropped"].abs().sum()) == 0.0
    assert float(balance.storage_change[:, 0].sum()) == 0.0  # basin's 5e-4 moved to the sewer
    assert float(balance.residual.abs().max()) <= 1e-15
    held = results.tank_storage[:, results.tank_names.index(("sewer", "sewer_tank"))]
    assert float(held[0]) == 5e-4 and float(held[-1]) == 5e-4
    # the same push into a user AltQueueArc is a real drop, and is recorded
    mapping["nodes"] = [
        {"name": "basin", "type_": "Reservoir", "capacity": 30.0, "initial_storage": 5e-4},
        {"name": "pool", "type_": "Reservoir", "capacity": 30.0},
        {"name": "sea", "type_": "Waste"},
    ]
    mapping["arcs"] = [
        {
            "name": "basin_to_pool",
            "type_": "AltQueueArc",
            "in_port": "basin",
            "out_port": "pool",
            "number_of_timesteps": 1,
        },
        {"name": "pool_to_sea", "type_": "Arc", "in_port": "pool", "out_port": "sea"},
    ]
    mapping["orchestration"] = [{"Reservoir": "distribute"}]
    model, state, drivers = build_model(read_wsimod_config(mapping))
    balance = wsimod_run(model, state, drivers).mass_balance
    assert float(balance.terms["dropped"][:, 0].sum()) == pytest.approx(5e-4)
    assert float(balance.residual.abs().max()) <= 1e-15


def test_water_pushed_into_a_wwtw_after_its_treatment_is_counted():
    """W69: with the sewer's ``make_discharge`` after the works' ``calculate_discharge``,
    ``current_input`` receives water nobody treats, and ``end_timestep_`` empties it
    (``wtw.py:447-453``). It is the term ``"WWTW.unprocessed_input"``, so the balance closes."""
    mapping = CASES["wwtw_storm_tank"].build()
    mapping["orchestration"] = [
        {"Catchment": "route"},
        {"Reservoir": "distribute"},
        {"WWTW": "calculate_discharge"},
        {"Sewer": "make_discharge"},
        {"WWTW": "make_discharge"},
    ]
    model, state, drivers = build_model(read_wsimod_config(mapping))
    results = wsimod_run(model, state, drivers)
    late = results.mass_balance.terms["WWTW.unprocessed_input"]
    assert float(late[:, 0].max()) > 1.0
    scale = CASES["wwtw_storm_tank"].scale
    assert float(results.mass_balance.residual.abs().max()) <= 1e-12 * scale
    # the standard order treats everything it receives
    model, state, drivers = _case("wwtw_storm_tank")
    standard = wsimod_run(model, state, drivers).mass_balance.terms["WWTW.unprocessed_input"]
    assert float(standard.abs().max()) == 0.0


def test_only_carried_stores_are_storage():
    """W75: a ``Demand``'s totals and a ``Land``'s per-day flows are terms of the day, not
    storage: ``storage_change`` is the carried stores' change and sums to the end less the start
    (the initial stores are the ``Land`` soil, residence tanks and basins)."""
    model, state, drivers = _case("residential_with_land")
    results = wsimod_run(model, state, drivers)
    change = results.mass_balance.storage_change[:, 0].sum()
    # what the carried stores hold at the end less at the start, from the state
    rows = model.wsimod_day.state_labels("wsimod.storage")
    first = state["wsimod.storage"][..., 0].sum()
    last = model.wsimod_day.last_state["wsimod.storage"][..., 0].sum()
    assert len(rows) > 0
    assert float(change) == pytest.approx(float(last - first), rel=1e-9, abs=1e-9)
    assert float(results.mass_balance.terms["Demand.total_demand"][:, 0].sum()) > 0.0


def test_the_evaporation_clamp_residual_is_wsimods_printed_gap():
    """W75: a ``Land``'s balance outflow is IHACRES's evaporation, not what ``evaporate``
    removed. On the ``"inaccurate_evaporation"`` days the residual is the gap between the two,
    computed here from the records alone (precipitation less the water that reached the sea
    and the change of every store, less the recorded evaporation), and on every other day it
    closes."""
    model, state, drivers = _case("evaporation_clamp", "quantity")
    diagnostics: dict = {}
    results = wsimod_run(model, state, drivers, diagnostics=diagnostics)
    names = model.wsimod_day.node_names
    event = diagnostics["events"]["inaccurate_evaporation"][:, names.index("heath")]
    assert int(event.sum()) > 5 and int((~event).sum()) > 5
    arcs = results.arc_names
    reached = results.arc_flow[:, arcs.index("ditch_to_sea")]
    stores = results.tank_storage.sum(dim=-1) + results.surface["storage"].sum(dim=-1)
    previous = torch.cat([_initial_total(model, state)[None], stores[:-1]])
    precipitation = results.surface["precipitation"].sum(dim=-1)
    evaporation = results.surface["evaporation"].sum(dim=-1)
    # P = E_actual + sea + dS; the gap is E_actual - E_ihacres
    gap = (precipitation - reached - (stores - previous)) - evaporation
    residual = results.mass_balance.residual[:, 0]
    bound = 1e-12 * CASES["evaporation_clamp"].scale
    assert float(residual[~event].abs().max()) <= bound
    assert float((residual[event] - gap[event]).abs().max()) <= bound
    assert float(residual[event].abs().min()) > 1e-11


def _initial_total(model, state) -> torch.Tensor:
    """The volume the recorded stores hold before day 1: the tanks' rows and the surfaces'."""
    rows = model.wsimod_day.state_labels("wsimod.storage")
    return state["wsimod.storage"][..., 0].sum(dim=-1) if rows else torch.zeros(())
