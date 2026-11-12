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
import os
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
    assert ours.dates == model.wsimod_day.network.date_labels
    assert len(ours.dates) == len(theirs.dates) == 10
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
@pytest.mark.parametrize(
    ("case", "days"),
    [
        ("small", 10),
        ("queue_arc", 10),
        ("decay_queue_groundwater", 40),
        ("decay_arc", 30),
        ("river_nutrients", 60),
    ],
)
def test_step_by_step_equals_wsimod_run(case, days):
    """The 3b state (``total_decayed``, the River's lag registers, the user arcs' queues) is
    carried through ``model.step`` as through ``wsimod_run`` (``decay_arc`` has 30 dates,
    ``decay_queue_groundwater`` 40)."""
    if case == "small":
        model, state, drivers = _small()
    else:
        model, state, drivers = _case(case)
    ran = wsimod_run(model, state, drivers, days=days)
    final = dict(model.wsimod_day.last_state)
    stepped, flows = state, []
    for k in range(days):
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
    kink_site = "node:farm/surfaces.grass/ihacres/evaporation_kink"
    kink = outcome(kink_site, 4)
    assert kink[2] != kink[3]
    (record,) = [r for r in diagnostics["decisions"][4] if r.site == kink_site]
    assert record.active.tolist() == [True] * 4
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


# --- 3b: batches of decay, treatment and biochemistry parameters ---------------------------------
# Per case: ``driver -> factors of the four instances`` (the configured values times a factor;
# a ``decays`` entry is scaled only where the key is listed, the rest stays inert, X9). Each
# plan makes at least two instances take different decisions on one day (asserted).
BATCH_3B = {
    "decay_storage": (
        40,
        {
            "wsimod.Reservoir.decays.constant": [1.0, 0.3, 1.6, 0.05],
            "wsimod.Reservoir.decays.exponent": [1.0, 0.97, 1.04, 1.0],
        },
        "node:res/tank/make_decay/cap:phosphate",
    ),
    "wtw_treatment": (
        40,
        {
            "wsimod.WWTW.process_parameters.constant": [1.0, 0.5, 1.4, 0.1],
            "wsimod.WWTW.process_parameters.exponent": [1.0, 0.95, 1.05, 1.0],
            "wsimod.WWTW.liquor_multiplier": [1.0, 0.5, 1.5, 0.2],
            "wsimod.Reservoir.decays.constant": [1.0, 0.2, 20.0, 50.0],
        },
        # the treatment processes clamp without a branch of their own (wtw.py:163-214): the
        # instances differ in where the works spares water and clips its storm tank
        "node:works/calculate_discharge/spare",
    ),
    "surface_decay_nitrification": (
        40,
        {
            "wsimod.PerviousSurface.decays.constant": [1.0, 0.2, 10.0, 60.0],
            "wsimod.PerviousSurface.soil_temp_w_air": [1.0, 0.5, 2.0, 0.1],
        },
        "node:plot/surfaces.turf/make_decay/cap:nitrite",
    ),
    "decay_queue_groundwater": (
        40,
        {"wsimod.QueueGroundwater.decays.constant": [1.0, 0.2, 5.0, 0.01]},
        "node:gw/tank/internal_arc/make_decay/cap:ammonia",
    ),
    "decay_arc": (
        30,
        {"wsimod.DecayArc.decays.constant": [1.0, 0.2, 15.0, 40.0]},
        "arc:q/make_decay/cap:phosphate",
    ),
    "river_nutrients": (
        100,
        {
            "wsimod.River.denpar_w": [1.0, 0.2, 3.0, 0.0],
            "wsimod.River.prodNpar": [1.0, 0.5, 2.0, 0.1],
            "wsimod.River.decays.constant": [1.0, 0.2, 3.0, 0.5],
        },
        "node:r_up/biochemical_processes/macrophyte_cap:N",
    ),
}


def _scaled(drivers: dict, plan: dict) -> dict:
    out = dict(drivers)
    for key, factors in plan.items():
        base = drivers[key]
        factor = _tensors(*factors).reshape(-1, *([1] * base.dim()))
        scaled = base * factor
        if ".decays." in key:  # only the listed keys, whose constant is not 0 (X9)
            listed = drivers[key.replace("exponent", "constant")] != 0
            scaled = torch.where(listed, scaled, base.expand_as(scaled))
        out[key] = scaled
    return out


def _comparable(results: WsimodResults, final: dict) -> dict:
    fields = {
        "arc_flow": results.arc_flow,
        "arc_quality": results.arc_quality,
        "tank_storage": results.tank_storage,
        "tank_quality": results.tank_quality,
        "residual": results.mass_balance.residual,
    }
    fields.update({f"surface.{k}": torch.nan_to_num(v) for k, v in results.surface.items()})
    fields.update({f"state.{k}": v for k, v in final.items()})
    return fields


@pytest.mark.parametrize("case", list(BATCH_3B))
def test_a_batch_of_four_3b_parameter_sets_equals_its_single_runs(case):
    days, plan, site = BATCH_3B[case]
    model, state, drivers = _case(case)
    batched = _scaled(drivers, plan)
    diagnostics = {"decisions": True}
    ran = wsimod_run(model, state, batched, days=days, diagnostics=diagnostics)
    together = _comparable(ran, dict(model.wsimod_day.last_state))
    assert ran.arc_flow.shape[0] == 4
    # at least two instances take different decisions at the site on one day, among the
    # instances that reach it
    differing = [
        d
        for d, records in enumerate(diagnostics["decisions"])
        for r in records
        if r.site == site and len(set(r.outcome[r.active].tolist())) > 1
    ]
    assert differing, site
    for i in range(4):
        single_drivers = {k: (v[i] if k in plan else v) for k, v in batched.items()}
        single = wsimod_run(model, state, single_drivers, days=days)
        alone = _comparable(single, dict(model.wsimod_day.last_state))
        assert alone.keys() == together.keys()
        for name, value in alone.items():
            assert torch.equal(value, together[name][i]), (name, i)


ROUND_TRIP = {
    "river_nutrients": {
        "wsimod.river_temperature_lag": {("r_up", 3): 9.5, ("r_low", 1): 4.0},
        "wsimod.river_phosphorus_lag": {("r_up", 10): 2e-5, ("r_low", 200): 4e-4},
    },
    "decay_queue_groundwater": {
        "wsimod.total_decayed": {("gw", "tank.internal_arc"): {"volume": 0.0, "phosphate": 0.25}},
    },
}


@pytest.mark.parametrize("case", list(ROUND_TRIP))
def test_the_new_state_round_trips(case):
    """The 3b state keys are set by label, change the run, and a run warm-started from another
    run's last state equals the continuation of that run (``torch.equal``)."""
    model, default, drivers = _case(case)
    state = initial_state(model, values=ROUND_TRIP[case])
    day = model.wsimod_day
    for key, rows in ROUND_TRIP[case].items():
        labels = day.state_labels(key)
        for label in rows:
            assert not torch.equal(
                state[key][labels.index(label)], default[key][labels.index(label)]
            )
    total, first = 10, 4

    def stepped(start, k0, k1):
        out, flows = start, []
        for k in range(k0, k1):
            out = model.step(out, drivers, dt=DAY, t=k * DAY)
            flows.append(model.wsimod_day.last_record.arc_flow)
        return out, flows

    whole, whole_flows = stepped(state, 0, total)
    middle, early = stepped(state, 0, first)
    resumed, late = stepped(middle, first, total)
    for key in whole:
        assert torch.equal(whole[key], resumed[key]), key
    assert all(torch.equal(a, b) for a, b in zip(whole_flows, early + late, strict=True))
    # the registers persist, so their set values reach the run; a day's ``total_decayed`` is
    # reset at its start (W75), so its set value only round trips
    if case == "river_nutrients":
        other, _ = stepped(default, 0, total)
        assert any(not torch.equal(whole[k], other[k]) for k in ROUND_TRIP[case])


def _without_3b_values(mapping: dict) -> dict:
    """The same configuration with every value a quantity-only run lists as ignored changed:
    ``decays`` constants and exponents, the WTW pollutant entries, the River biochemistry."""
    mapping = copy.deepcopy(mapping)
    for node in mapping["nodes"]:
        for entry in (node.get("decays") or {}).values():
            entry["constant"] *= 2.5
            entry["exponent"] *= 1.1
        for entry in (node.get("process_parameters") or {}).values():
            if "constant" in entry:
                entry["constant"] *= 0.5
                entry["exponent"] = 1.2
        liquor = node.get("liquor_multiplier") or {}
        for name in liquor:
            if name != "volume":
                liquor[name] *= 0.5
        if node["type_"] == "River":  # the biochemistry parameters are overrides' (storage.py:538)
            entry = mapping.setdefault("overrides", {}).setdefault("nodes", {})
            entry = entry.setdefault(node["name"], {"type_": "River", "name": node["name"]})
            entry.update(
                {"denpar_w": 7e-4, "prodNpar": 0.2, "muptNpar": 0.05, "hsatTP": 3e-4}
                | {"limpppar": 5e-4, "T_wdays": 7}
            )
    for surface in mapping.get("surfaces") or []:
        for entry in (surface.get("decays") or {}).values():
            entry["constant"] *= 2.5
    for arc in mapping["arcs"]:
        for entry in (arc.get("decays") or {}).values():
            entry["constant"] *= 2.5
    return mapping


@pytest.mark.parametrize(
    ("case", "days"),
    [("decay_queue_groundwater", 40), ("wtw_treatment", 40), ("river_nutrients", 60)],
)
def test_quantity_only_volumes_never_depend_on_3b_values(case, days):
    mapping = CASES[case].build()
    changed = _without_3b_values(mapping)
    assert changed != mapping
    runs = []
    for configuration in (mapping, changed):
        model, state, drivers = build_model(read_wsimod_config(configuration), pollutants=False)
        runs.append((model, wsimod_run(model, state, drivers, days=days)))
    (base_model, base), (other_model, other) = runs
    assert base_model.ignored_by_quantity_only
    assert set(base_model.ignored_by_quantity_only) <= set(other_model.ignored_by_quantity_only)
    assert float(base.arc_flow.abs().max()) > 0.0
    assert torch.equal(base.arc_flow, other.arc_flow)
    assert torch.equal(base.tank_storage, other.tank_storage)


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
    # The child must import this checkout's `src`, as pytest's `pythonpath` does for the parent,
    # not whichever noodl the interpreter has installed (another worktree's, say).
    path = os.pathsep.join(p for p in (str(ROOT / "src"), os.environ.get("PYTHONPATH")) if p)
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
        env={**os.environ, "PYTHONPATH": path},
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
    n_arcs = len(model.wsimod_day.arc_names)
    for name, flags in diagnostics["events"].items():
        # D21: an event recorded at an arc has one column per arc
        assert flags.shape == (40, n_arcs if name == "decay_arc_order_unknown" else n_nodes)
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
    """Every case, both modes, over its default ``days``: the residual is within ``1e-12 *
    scale`` on every day; ``evaporation_clamp`` apart, whose residual is the clamp gap (W75,
    tested below)."""
    model, state, drivers = _case(name, mode)
    diagnostics: dict = {}
    results = wsimod_run(model, state, drivers, days=CASES[name].days, diagnostics=diagnostics)
    event = diagnostics["events"]["inaccurate_evaporation"].any(dim=-1)
    if name == "evaporation_clamp":
        assert bool(event.any())  # its residual is the clamp gap: the tests below
        return
    assert not bool(event.any())
    residual = results.mass_balance.residual.abs().amax(dim=-1)
    assert float(residual.max()) <= 1e-12 * CASES[name].scale
    # the balance is not trivially zero: something moves
    balance = results.mass_balance
    assert (
        float(balance.outflow.abs().sum() + balance.inflow.abs().sum()) > 0.0
        or float(balance.storage_change.abs().sum()) > 0.0
    )


def test_drops_are_recorded_where_waters_are_dropped():
    """The ``QueueArc`` tiny push (``queue_arc``) and the ``Land.run`` percolation push-back
    (``land_percolation_threshold``) drop water below ``FLOAT_ACCURACY``; it is in
    ``"dropped"``. These drops sit far below the sweep's bound, so each site also has its own
    mini-config (``test_every_drop_site_is_balanced_in_a_whole_model``)."""
    for name in ("queue_arc", "land_percolation_threshold"):
        model, state, drivers = _case(name)
        results = wsimod_run(model, state, drivers)
        dropped = results.mass_balance.terms["dropped"]
        assert float(dropped[:, 0].sum()) > 0.0, name


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


# --- every drop site, balanced in a whole model ----------------------------------------------
# float_accuracy 1e-3 and a drop of 5e-4 at each site: far above the 1e-12 * scale the case
# sweep resolves, so skipping that site's ``ctx.drop`` shows as a residual of 5e-4.
TINY = 5e-4


def _mini(nodes, arcs, orchestration, days=3) -> dict:
    return {
        "float_accuracy": 1e-3,
        "pollutants": [],
        "additive_pollutants": [],
        "non_additive_pollutants": [],
        "dates": [f"2000-01-{d:02d}" for d in range(1, days + 1)],
        "nodes": nodes,
        "arcs": arcs,
        "orchestration": orchestration,
    }


def _arc(name, a, b, **kw) -> dict:
    return {"name": name, "type_": kw.pop("type_", "Arc"), "in_port": a, "out_port": b, **kw}


def _reservoir(name, **kw) -> dict:
    return {"name": name, "type_": "Reservoir", **kw}


def _waste(name) -> dict:
    return {"name": name, "type_": "Waste"}


def _works(**kw) -> dict:
    return {
        "name": "works",
        "type_": "WWTW",
        "percent_solids": 0.0002,
        "liquor_multiplier": {"volume": 0.03},
        **kw,
    }


def _catchment_on_day_2(name: str, flow: float) -> dict:
    data = {("flow", f"2000-01-{d:02d}"): (flow if d == 2 else 0.0) for d in (1, 2)}
    return {"name": name, "type_": "Catchment", "data_input_dict": data}


def _queue_arc_push() -> dict:
    """A reservoir holding 5e-4 pushes it into a ``QueueArc``: below ``FLOAT_ACCURACY``, so
    neither queued nor delivered (``arcs.py:378-379``)."""
    return _mini(
        [_reservoir("basin", capacity=30.0, initial_storage=TINY), _waste("sea")],
        [_arc("q", "basin", "sea", type_="QueueArc", number_of_timesteps=1)],
        [{"Reservoir": "distribute"}],
    )


def _queue_arc_update() -> dict:
    """A reservoir pulls 5e-4 (all its well holds) through a delay-1 ``QueueArc``: the waiting
    request is below ``FLOAT_ACCURACY`` when ``update_queue`` visits it, and is removed
    (``arcs.py:483-485``); the water has already left the well."""
    return _mini(
        [
            {"name": "well", "type_": "Storage", "capacity": 10.0, "initial_storage": TINY},
            {
                "name": "basin",
                "type_": "Abstraction",
                "node_type_override": "Reservoir",
                "capacity": 10.0,
            },
        ],
        [_arc("q", "well", "basin", type_="QueueArc", number_of_timesteps=1)],
        [{"Abstraction": "make_abstractions"}],
    )


def _sewer_ponded() -> dict:
    """A sewer whose tank starts 5e-4 above its capacity (see the state below) and whose outlet
    takes nothing: the ponded water is at or below ``FLOAT_ACCURACY`` and goes nowhere
    (``sewer.py:234-235``)."""
    return _mini(
        [{"name": "sewer", "type_": "Sewer", "capacity": 10.0}, _waste("sea")],
        [_arc("out", "sewer", "sea", capacity=0.0)],
        [{"Sewer": "make_discharge"}],
    )


def _wwtw_push_back() -> dict:
    """The works' outlet is 5e-4 short of the day's treated water: the rejection is at or below
    ``FLOAT_ACCURACY`` and vanishes (``wtw.py:347-348``)."""
    c_v = 1 - 0.0002 - 0.03
    return _mini(
        [
            _reservoir("basin", capacity=30.0, initial_storage=2.0),
            _works(treatment_throughput_capacity=30.0),
            _waste("sea"),
        ],
        [
            _arc("in", "basin", "works"),
            _arc("out", "works", "sea", capacity=2.0 * c_v - TINY),
        ],
        [{"Reservoir": "distribute"}, {"WWTW": "calculate_discharge"}, {"WWTW": "make_discharge"}],
    )


def _wwtw_storm_rejection() -> dict:
    """Two delay-1 ``QueueArc`` pushes, each checked on entry against the works' spare throughput
    (3) and storm-tank headroom (2) alone, deliver 4 + 5e-4 and 1 together: the tank rejects
    5e-4, which is below ``FLOAT_ACCURACY`` (``wtw.py:403-404``). Catchments feed the reservoirs
    on day 2 so that each makes a request, and so an ``update_queue``, that day."""
    links = [{"name": f"link_{i}", "type_": "Node"} for i in (1, 2)]
    arcs = []
    for i in (1, 2):
        arcs += [
            _arc(f"f{i}", f"feeder_{i}", f"link_{i}"),
            _arc(f"l{i}", f"link_{i}", f"basin_{i}"),
            _arc(f"q{i}", f"basin_{i}", "works", type_="QueueArc", number_of_timesteps=1),
        ]
    return _mini(
        [
            _catchment_on_day_2("feeder_1", 1.0),
            _catchment_on_day_2("feeder_2", 1.0),
            *links,
            _reservoir("basin_1", capacity=30.0, initial_storage=4.0 + TINY),
            _reservoir("basin_2", capacity=30.0, initial_storage=1.0),
            _works(treatment_throughput_capacity=3.0, stormwater_storage_capacity=2.0),
            _waste("sea"),
        ],
        [*arcs, _arc("out", "works", "sea")],
        [
            {"Catchment": "route"},
            {"Reservoir": "distribute"},
            {"WWTW": "calculate_discharge"},
            {"WWTW": "make_discharge"},
        ],
        days=2,
    )


DROP_SITES = {
    "queue_arc_push": (_queue_arc_push, None),
    "queue_arc_update": (_queue_arc_update, None),
    "sewer_ponded": (_sewer_ponded, {("sewer", "sewer_tank"): 10.0 + TINY}),
    "wwtw_push_back": (_wwtw_push_back, None),
    "wwtw_storm_rejection": (_wwtw_storm_rejection, None),
}


@pytest.mark.parametrize("site", DROP_SITES)
def test_every_drop_site_is_balanced_in_a_whole_model(site):
    build, stored = DROP_SITES[site]
    model, state, drivers = build_model(read_wsimod_config(build()), pollutants=False)
    if stored:
        state = initial_state(model, values={"wsimod.storage": stored})
    balance = wsimod_run(model, state, drivers).mass_balance
    assert float(balance.terms["dropped"][:, 0].sum()) == pytest.approx(TINY, rel=1e-9)
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


def _clamp_gap(model, state, results) -> torch.Tensor:
    """``evaporation_clamp``'s per-day gap, from the records alone: ``E_actual - E_ihacres``
    with ``E_actual = P - (water that reached the sea) - (change of every store)`` and
    ``E_ihacres`` the recorded evaporation. It is 0 on a day the soil supplies what IHACRES
    asks."""
    reached = results.arc_flow[:, results.arc_names.index("ditch_to_sea")]
    stores = results.tank_storage.sum(dim=-1) + results.surface["storage"].sum(dim=-1)
    rows = model.wsimod_day.state_labels("wsimod.storage")
    assert rows
    first = state["wsimod.storage"][..., 0].sum(dim=-1)
    previous = torch.cat([first[None], stores[:-1]])
    precipitation = results.surface["precipitation"].sum(dim=-1)
    evaporation = results.surface["evaporation"].sum(dim=-1)
    return (precipitation - reached - (stores - previous)) - evaporation


@pytest.mark.parametrize("mode", ["quantity", "pollutants"])
def test_the_evaporation_clamp_residual_is_the_gap_on_every_day(mode):
    """W75: a ``Land``'s balance outflow is IHACRES's evaporation, not what ``evaporate``
    removed. One rule on every day: the residual equals the gap between the two, computed
    from the records alone, within ``1e-12 * scale``; the gap is 0 on a day without a clamp,
    and a shortfall below ``FLOAT_ACCURACY`` that fires no event is covered too."""
    model, state, drivers = _case("evaporation_clamp", mode)
    diagnostics: dict = {}
    results = wsimod_run(model, state, drivers, diagnostics=diagnostics)
    names = model.wsimod_day.node_names
    event = diagnostics["events"]["inaccurate_evaporation"][:, names.index("heath")]
    assert int(event.sum()) > 5 and int((~event).sum()) > 5
    gap = _clamp_gap(model, state, results)
    residual = results.mass_balance.residual[:, 0]
    bound = 1e-12 * CASES["evaporation_clamp"].scale
    assert float((residual - gap).abs().max()) <= bound
    assert float(residual[event].abs().min()) > 1e-11
    assert float(gap[~event].abs().max()) <= bound


@pytest.mark.parametrize("mode", ["quantity", "pollutants"])
def test_the_evaporation_clamp_residual_is_wsimods_printed_node_gap(mode):
    """WSIMOD's node check prints "mass balance error for volume of <gap> in heath" on the
    clamp days; noodl's residual on those days equals the printed values, in order."""
    pytest.importorskip("wsimod")
    from tests.verification._wsimod_reference import wsimod_reference

    theirs = wsimod_reference(CASES["evaporation_clamp"].build())
    printed = [
        float(m.group(1))
        for line in theirs.printed
        if (m := re.fullmatch(r"mass balance error for volume of (\S+) in heath", line))
    ]
    model, state, drivers = _case("evaporation_clamp", mode)
    diagnostics: dict = {}
    results = wsimod_run(model, state, drivers, diagnostics=diagnostics)
    names = model.wsimod_day.node_names
    event = diagnostics["events"]["inaccurate_evaporation"][:, names.index("heath")]
    ours = results.mass_balance.residual[event, 0]
    assert len(printed) == int(event.sum()) > 5
    bound = 1e-12 * CASES["evaporation_clamp"].scale
    assert float((ours - torch.tensor(printed, dtype=torch.float64)).abs().max()) <= bound


# --- the contiguous window (X8, X32, D14, D22) -------------------------------------------------
def _completed_records(day, **kw) -> list:
    """Every day's :class:`DayRecord`, read after the generator resumed (D14)."""
    return [day.last_record for _ in day.run_days(**kw)]


@pytest.mark.parametrize(("name", "mode"), RUN_CASES)
def test_consecutive_balance_windows_meet(name, mode):
    """X8: a day's window ends at the state ``end_timestep`` leaves, which is the next day's
    loaded state, so consecutive windows meet exactly."""
    model, _, _ = _case(name, mode)
    records = _completed_records(model.wsimod_day, days=CASES[name].days)
    assert len(records) == CASES[name].days
    for d in range(len(records) - 1):
        assert torch.equal(records[d + 1].storage_start, records[d].storage_end), (name, d)


@pytest.mark.parametrize(("name", "mode"), RUN_CASES)
def test_end_timestep_records_no_decision_and_no_maxiter(name, mode):
    """D22: ``end_timestep`` records no decision and no ``maxiter``; a later one that does
    fails here, and ``_finish_record`` would have to re-snapshot it."""
    model, _, _ = _case(name, mode)
    day = model.wsimod_day
    day.record_decisions = True
    seen = []
    try:
        for _, ctx, _ in day.run_days(days=CASES[name].days):
            seen.append((ctx, len(ctx.decisions), dict(ctx.maxiter)))
    finally:
        day.record_decisions = False
    assert len(seen) == CASES[name].days
    for ctx, decisions, maxiter in seen:
        assert decisions > 0 and len(ctx.decisions) == decisions
        assert ctx.maxiter.keys() == maxiter.keys()
        assert all(ctx.maxiter[site] is maxiter[site] for site in maxiter)


def test_storage_end_is_unset_at_the_yield_point():
    """D14: ``run_days`` yields before ``end_timestep``; ``storage_end`` is ``None`` there and
    set, on the same record, when the generator resumes."""
    model, _, _ = _small()
    day = model.wsimod_day
    records = []
    for _ in day.run_days(days=3):
        assert day.last_record.storage_end is None
        records.append(day.last_record)
    assert all(r.storage_end is not None for r in records)
    assert all(r.storage_end.shape == r.storage_total.shape for r in records)


def test_storage_total_keeps_its_meaning():
    """X32: ``storage_total`` is still the water held at the record point, before
    ``end_timestep``; ``storage_end`` is the water held after it."""
    model, _, _ = _case("user_alt_queue_arc")
    day = model.wsimod_day
    records, held = [], []
    for _, ctx, _ in day.run_days():
        held.append(day._held(ctx.shape, {}))
        records.append(day.last_record)
    for record, value in zip(records, held, strict=True):
        assert torch.equal(record.storage_total, value)


def _remove_one_cubic_metre(node) -> None:
    """``node``'s ``end_timestep`` takes 1 m³ out of its tank, at the tank's concentration, and
    records it as the out-term ``"Test.removed"`` and the event ``"test_event"``."""
    from noodl.apps.wsimod.vqip import extract_vqip, v_change_vqip

    original = node.end_timestep

    def end_timestep():
        p, ctx, tank = node.p, node.ctx, node.tank
        removed = v_change_vqip(p, tank.storage, torch.tensor(1.0, dtype=torch.float64))
        removed = torch.cat([torch.ones_like(removed[..., :1]), removed[..., 1:]], dim=-1)
        tank.storage = extract_vqip(p, tank.storage, removed)
        ctx.term("Test.removed", removed, torch.ones(ctx.shape, dtype=torch.bool))
        ctx.event("test_event", node.name, torch.ones(ctx.shape, dtype=torch.bool))
        original()

    node.end_timestep = end_timestep
    node.mass_balance = {**type(node).mass_balance, "Test.removed": "out"}


def test_a_term_recorded_at_the_end_of_the_day_counts_in_that_day():
    """X8: a term recorded during ``end_timestep`` belongs to the day it was recorded on, and
    the water it carries leaves inside that day's window."""
    model, state, drivers = _case("storage_distribute")
    day = model.wsimod_day
    node = day.engine.nodes["res"]
    _remove_one_cubic_metre(node)
    diagnostics: dict = {}
    results = wsimod_run(model, state, drivers, diagnostics=diagnostics)
    balance = results.mass_balance
    days = CASES["storage_distribute"].days
    assert torch.equal(balance.terms["Test.removed"][:, 0], torch.ones(days, dtype=torch.float64))
    scale = CASES["storage_distribute"].scale
    assert float(balance.residual.abs().max()) <= 1e-12 * scale
    fired = diagnostics["events"]["test_event"][:, day.node_names.index("res")]
    assert bool(fired.all())


def _arc_term(arc, name: str) -> None:
    """``arc``'s ``end_timestep`` records 1 m³ of nothing as the term ``name``."""
    original = arc.end_timestep

    def end_timestep():
        ctx = arc.ctx
        vqip = torch.zeros((*ctx.shape, 1 + arc.p.K), dtype=torch.float64)
        vqip[..., 0] = 1.0
        ctx.term(name, vqip, torch.ones(ctx.shape, dtype=torch.bool))
        original()

    arc.end_timestep = end_timestep


def test_an_arc_term_needs_a_declared_direction():
    """Arcs declare their terms as nodes do (``Arc.mass_balance``): an undeclared one is
    refused by name, a declared one counts in the balance."""
    model, state, drivers = _small()
    arc = next(iter(model.wsimod_day.engine.arcs.values()))
    _arc_term(arc, "TestArc.made")
    with pytest.raises(RuntimeError, match="TestArc.made"):
        wsimod_run(model, state, drivers)
    arc.mass_balance = {"TestArc.made": "in"}
    balance = wsimod_run(model, state, drivers).mass_balance
    assert torch.equal(balance.terms["TestArc.made"][:, 0], torch.ones(10, dtype=torch.float64))
    # 1 m³ a day counted in with no water behind it: the residual is that 1 m³
    assert float((balance.residual[:, 0] - 1.0).abs().max()) <= 1e-12


def test_a_direction_other_than_in_or_out_is_refused():
    """A term's direction is ``"in"`` or ``"out"``; anything else is refused by name rather
    than counted silently as out."""
    model, state, drivers = _small()
    arc = next(iter(model.wsimod_day.engine.arcs.values()))
    _arc_term(arc, "TestArc.made")
    arc.mass_balance = {"TestArc.made": "ds"}
    with pytest.raises(RuntimeError, match=r"'TestArc.made' has direction 'ds'"):
        wsimod_run(model, state, drivers)


def test_an_unfinished_record_is_refused_by_the_balance():
    """D14: a record taken at the yield point and never completed has no ``storage_end``; the
    balance names the day instead of failing inside ``torch.stack``."""
    from noodl.apps.wsimod.run import _balance

    model, _, _ = _small()
    day = model.wsimod_day
    records = []
    for _ in day.run_days(days=2):
        records.append(day.last_record)
    records[1].storage_end = None
    with pytest.raises(RuntimeError, match="interrupted before end_timestep"):
        _balance(day, records, torch.Size(()))
