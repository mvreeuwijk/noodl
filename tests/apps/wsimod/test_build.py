"""``build_model``, the ``WsimodDay`` closure and the build-time structure checks (Task 13).

The ``small`` fixture (D9) is ``inflow`` (Catchment) -> ``river`` (River) -> ``outlet``
(Waste), quantity only, 10 dates. The refusals (D7) are raised by ``build_model`` by name.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from noodl.apps.wsimod import (
    DayRecord,
    WsimodDay,
    build_model,
    initial_drivers,
    initial_state,
    read_wsimod_config,
)
from noodl.apps.wsimod.structure import (
    forcing_readers,
    orchestration_plan,
    recursion_cycles,
    river_order,
)
from tests.verification._wsimod_cases import CASES

DATA = Path(__file__).resolve().parents[2] / "data" / "wsimod"
DAY = 86400.0


def _dates(n: int) -> list[str]:
    return [f"2000-01-{k + 1:02d}" for k in range(n)]


def _mapping(nodes, arcs, *, days: int = 3, orchestration=None) -> dict:
    mapping = {
        "nodes": nodes,
        "arcs": arcs,
        "pollutants": [],
        "additive_pollutants": [],
        "non_additive_pollutants": [],
        "dates": _dates(days),
    }
    if orchestration is not None:
        mapping["orchestration"] = orchestration
    return mapping


def _catchment(name: str, days: int = 3, *, skip=()) -> dict:
    data = {("flow", d): 10.0 for k, d in enumerate(_dates(days)) if k not in skip}
    return {"name": name, "type_": "Catchment", "data_input_dict": data}


def _node(name: str, type_: str, **kw) -> dict:
    return {"name": name, "type_": type_, **kw}


def _arc(name: str, a: str, b: str, **kw) -> dict:
    return {"name": name, "type_": "Arc", "in_port": a, "out_port": b, **kw}


def _small():
    return read_wsimod_config(DATA / "small")


# --- the model --------------------------------------------------------------------------------
def test_the_small_model_builds_and_steps():
    model, state, drivers = build_model(_small())
    row = model.wsimod_day.state_labels("wsimod.storage").index(("river", "tank"))
    after = model.step(state, drivers, dt=DAY, t=0.0)
    assert float(after["wsimod.storage"][row, 0]) != float(state["wsimod.storage"][row, 0])
    net = model.net
    edges = [net.graph.edges[e] for e in net.edges]
    assert [(e["name"], e["kind"]) for e in edges] == [
        ("inflow_to_river", "arc"),
        ("river_to_outlet", "arc"),
    ]
    assert net.nodes == ["inflow", "river", "outlet"]
    report = model.check(state, drivers)
    assert report.ok, str(report)
    text = str(report)
    assert "m3" in text and "m3/d" in text
    fields = model.refs.fields()
    assert fields["wsimod.storage"].unit == "m3"
    assert fields["wsimod.Arc.capacity"].unit == "m3/d"
    record = model.wsimod_day.last_record
    assert isinstance(record, DayRecord)
    # day 0: the catchment's 120 m3/d into the river, which passes most of it on
    assert record.arc_flow.tolist()[0] == 120.0
    assert record.tank_storage.shape == (1,)


def test_the_closure_answers_a_query_with_nothing_and_refuses_days_outside_the_dates():
    model, state, drivers = build_model(_small())
    day = model.wsimod_day
    assert day(state, drivers, None) == {}
    for t in (0.5 * DAY, 10 * DAY, -DAY):
        with pytest.raises(ValueError, match="whole number of days inside the dates"):
            model.step(state, drivers, dt=DAY, t=t)
    with pytest.raises(ValueError, match="needs t="):
        model.step(state, drivers, dt=DAY)
    with pytest.raises(ValueError, match="one day"):
        model.step(state, drivers, dt=3600.0, t=0.0)
    for start, days in ((0, 11), (-1, 2), (9, 2)):
        with pytest.raises(ValueError, match="not inside the configuration's 10 dates"):
            next(day.run_days(days=days, start=start))


def test_stepping_day_by_day_follows_the_dates_and_run_days_agrees():
    model, state, drivers = build_model(_small())
    stepped = state
    flows = []
    for k in range(10):
        stepped = model.step(stepped, drivers, dt=DAY, t=k * DAY)
        flows.append(model.wsimod_day.last_record.arc_flow.clone())
    day = WsimodDay(_small(), pollutants=True)
    ran = []
    for k, ctx, _ in day.run_days():
        assert ctx.day == k
        ran.append(day.last_record.arc_flow.clone())
    assert all(torch.equal(a, b) for a, b in zip(flows, ran, strict=True))
    for key, value in stepped.items():
        assert torch.equal(value, day.last_state[key]), key


def test_a_batch_equals_its_instances():
    """Four capacities on the river's outflow: the batched day equals each single run."""
    model, state, drivers = build_model(_small())
    capacities = torch.tensor([1e15, 50.0, 10.0, 0.0], dtype=torch.float64)
    batched = initial_drivers(
        model, values={"wsimod.Arc.capacity": {"river_to_outlet": capacities}}
    )
    out = state
    for k in range(4):
        out = model.step(out, batched, dt=DAY, t=k * DAY)
    for i, c in enumerate(capacities.tolist()):
        single = initial_drivers(model, values={"wsimod.Arc.capacity": {"river_to_outlet": c}})
        one = state
        for k in range(4):
            one = model.step(one, single, dt=DAY, t=k * DAY)
        for key in one:
            assert torch.equal(out[key][i], one[key]), (key, i)


def test_initial_state_and_drivers_take_values_by_label():
    model, state, drivers = build_model(_small())
    row = model.wsimod_day.state_labels("wsimod.storage").index(("river", "tank"))
    state = initial_state(model, values={"wsimod.storage": {("river", "tank"): 9.0}})
    assert float(state["wsimod.storage"][row, 0]) == 9.0
    drivers = initial_drivers(model, values={"wsimod.Arc.capacity": {"river_to_outlet": 0.3}})
    arc = model.wsimod_day.arc_names.index("river_to_outlet")
    assert float(drivers["wsimod.Arc.capacity"][arc]) == 0.3
    assert float(drivers["wsimod.Arc.capacity"][1 - arc]) == 1e15
    drivers = initial_drivers(model, values={"wsimod.flow": {"inflow": 7.0}})
    assert drivers["wsimod.flow"][:, 0].tolist() == [7.0] * 10
    with pytest.raises(KeyError, match="did you mean"):
        initial_state(model, values={"wsimod.storage": {("rivr", "tank"): 1.0}})
    with pytest.raises(KeyError, match="did you mean"):
        initial_drivers(model, values={"wsimod.Arc.capacity": {"river_to_outlett": 1.0}})
    with pytest.raises(KeyError, match="did you mean 'wsimod.storage'"):
        initial_state(model, values={"wsimod.storag": {}})


def test_a_queue_tank_starts_active_and_its_storage_by_name_sets_both():
    network = read_wsimod_config(CASES["queue_groundwater"].build())
    model, state, _ = build_model(network)
    day = model.wsimod_day
    label = next(x for x in day.state_labels("wsimod.active_storage"))
    state = initial_state(model, values={"wsimod.storage": {label: {"volume": 4.0}}})
    row = day.state_labels("wsimod.storage").index(label)
    active = day.state_labels("wsimod.active_storage").index(label)
    assert float(state["wsimod.storage"][row, 0]) == 4.0
    assert float(state["wsimod.active_storage"][active, 0]) == 4.0
    present = day.state_labels("wsimod.travel_queue_present")
    assert present[0] == (f"{label[0]}.{label[1]}.internal_arc", 0)
    # keys 0 and 1 exist from the start (arcs.py:567)
    assert state["wsimod.travel_queue_present"][:2].tolist() == [1.0, 1.0]


def test_the_state_layout_follows_d10():
    """Tanks in dir() order with the surfaces after them; the arc labels of D10."""
    network = read_wsimod_config(CASES["land_sewer_ponding"].build())
    model, state, _ = build_model(network)
    day = model.wsimod_day
    labels = day.state_labels("wsimod.storage")
    land = [label for label in labels if label[0] == "city"]
    tanks = [a for _, a in land if not a.startswith("surface:")]
    assert tanks == sorted(tanks) == ["percolation", "subsurface_runoff", "surface_runoff"]
    assert all(a.startswith("surface:") for _, a in land[3:])
    assert day.tank_names == [x for x in labels if not x[1].startswith("surface:")]
    for key, value in state.items():
        assert value.shape[0] == len(day.state_labels(key)), key
    queue = read_wsimod_config(CASES["queue_arc"].build())
    model, state, _ = build_model(queue)
    rows = model.wsimod_day.state_labels("wsimod.arc_queue")
    assert rows and all(len(r) == 4 and r[1] in ("push", "pull") for r in rows)
    anchors = model.wsimod_day.state_labels("wsimod.arc_queue_since")
    assert {a[1] for a in anchors} == {"push", "pull"}
    n = {a.name: float(a.number_of_timesteps) for a in queue.arcs if a.cls == "QueueArc"}
    since = state["wsimod.arc_queue_since"].tolist()
    assert since == [n[arc] for arc, _ in anchors]  # since = n at the start (D1)


def test_the_present_register_accepts_a_float_mask():
    """W48: ``"wsimod.travel_queue_present"`` is carried as 0/1; a bool mask loads too."""
    network = read_wsimod_config(CASES["user_alt_queue_arc"].build())
    model, state, drivers = build_model(network)
    as_bool = {**state, "wsimod.travel_queue_present": state["wsimod.travel_queue_present"] != 0}
    a = model.step(state, drivers, dt=DAY, t=0.0)
    b = model.step(as_bool, drivers, dt=DAY, t=0.0)
    for key in a:
        assert torch.equal(a[key], b[key]), key


# --- refusals (D7) -----------------------------------------------------------------------------
def _cycle():
    return _mapping(
        [_catchment("c"), _node("a", "Node"), _node("b", "Node"), _node("w", "Waste")],
        [
            _arc("c_a", "c", "a"),
            _arc("a_b", "a", "b"),
            _arc("b_a", "b", "a"),
            _arc("b_w", "b", "w"),
        ],
    )


def _zero_preferences(preference):
    return _mapping(
        [_catchment("c"), _node("j", "Node"), _node("w1", "Waste"), _node("w2", "Waste")],
        [
            _arc("c_j", "c", "j"),
            _arc("j_w1", "j", "w1", preference=preference[0]),
            _arc("j_w2", "j", "w2", preference=preference[1]),
        ],
    )


def _arc_into_catchment():
    return _mapping(
        [_catchment("c"), _catchment("d"), _node("w", "Waste")],
        [_arc("c_d", "c", "d"), _arc("d_w", "d", "w")],
    )


def _river_without_waste():
    return _mapping(
        [_catchment("c"), _node("r", "River"), _node("j", "Node")],
        [_arc("c_r", "c", "r"), _arc("r_j", "r", "j")],
    )


def _waste_distributes():
    return _mapping(
        [_catchment("c"), _node("w", "Waste")],
        [_arc("c_w", "c", "w")],
        orchestration=[{"Catchment": "route"}, {"Waste": "distribute"}],
    )


def _missing_flow():
    return _mapping(
        [_catchment("c", skip=(2,)), _node("w", "Waste")],
        [_arc("c_w", "c", "w")],
    )


def _land_without_temperature():
    data = {}
    for d in _dates(3):
        data[("precipitation", d)] = 0.001
        data[("et0", d)] = 0.0005
    land = {"name": "farm", "type_": "Land", "data_input_dict": data, "surfaces": []}
    return _mapping([land, _node("w", "Waste")], [_arc("farm_w", "farm", "w")])


@pytest.mark.parametrize(
    ("build", "match"),
    [
        pytest.param(_cycle, "recurse in a cycle: a -> b -> a", id="push-check-cycle"),
        pytest.param(lambda: _zero_preferences((0, 0)), "preference 0", id="zero-preferences"),
        pytest.param(lambda: _zero_preferences((1, -1)), "negative", id="negative-preference"),
        pytest.param(_arc_into_catchment, "accepts no water", id="arc-into-catchment"),
        pytest.param(_river_without_waste, "no Waste node", id="river-without-waste"),
        pytest.param(_waste_distributes, "no step method 'distribute'", id="waste-distribute"),
        pytest.param(_missing_flow, "has no 'flow' forcing on 2000-01-03", id="missing-flow"),
        pytest.param(
            _land_without_temperature,
            "land 'farm' has no 'temperature' forcing on 2000-01-01",
            id="land-temperature-k0",
        ),
    ],
)
def test_structural_refusals(build, match):
    network = read_wsimod_config(build())
    with pytest.raises(ValueError, match=match):
        build_model(network)


def test_a_topological_cycle_is_accepted():
    """``Land -> Sewer`` runoff with ``Sewer -> Land`` ponding: no check enters the cycle."""
    network = read_wsimod_config(CASES["land_sewer_ponding"].build())
    assert recursion_cycles(network) == []
    model, state, drivers = build_model(network)
    model.step(state, drivers, dt=DAY, t=0.0)
    # the same cycle of recursing pull checks below a pulling node is refused
    mapping = copy.deepcopy(_cycle())
    for arc in mapping["arcs"]:
        arc["type_"] = "Arc"
    mapping["nodes"].append(_node("res", "Reservoir", capacity=10.0))
    mapping["arcs"].append(_arc("b_res", "b", "res"))
    cycles = recursion_cycles(read_wsimod_config(mapping))
    assert ["a", "b", "a"] in cycles


def test_a_leakage_batch_mixing_zero_and_positive_is_refused():
    network = read_wsimod_config(CASES["distribution_leakage"].build())
    model, state, _ = build_model(network)
    drivers = initial_drivers(
        model,
        values={"wsimod.Distribution.leakage": {"dist_gw": torch.tensor([0.0, 0.1])}},
    )
    with pytest.raises(ValueError, match="mixes zero and non-zero"):
        model.step(state, drivers, dt=DAY, t=0.0)


def test_quantity_only_lists_the_ignored_parameters():
    network = read_wsimod_config(DATA / "quickstart")
    model, state, drivers = build_model(network, pollutants=False)
    ignored = model.ignored_by_quantity_only
    assert "my_land.surfaces.urban.pollutant_load" in ignored
    assert "my_land.surfaces.rural.pollutant_load" in ignored
    model.step(state, drivers, dt=DAY, t=0.0)
    with pytest.raises(ValueError, match="3c"):
        build_model(read_wsimod_config(DATA / "quickstart"), pollutants=True)


def _residential(type_: str, override: str | None = None) -> dict:
    data = {("temperature", d): 12.0 for d in _dates(3)}
    demand = _node("homes", type_, data_input_dict=data, population=10)
    if override is not None:
        demand["node_type_override"] = override
    return _mapping(
        [_node("res", "Reservoir", capacity=100.0, initial_storage=50.0), demand],
        [_arc("res_homes", "res", "homes")],
    )


def test_slot_keys_decide_what_runs():
    """``model.py:658-660``: ``type_: ResidentialDemand`` without the override files the node
    under the key ``"ResidentialDemand"``, which the default orchestration never names, so its
    demand is never created; written ``type_: Demand`` with the override, it runs."""
    skipped = read_wsimod_config(_residential("ResidentialDemand"))
    assert all("homes" not in names for _, _, names in orchestration_plan(skipped))
    model, state, drivers = build_model(skipped)
    model.step(state, drivers, dt=DAY, t=0.0)
    assert model.wsimod_day.last_record.arc_flow.tolist() == [0.0]
    runs = read_wsimod_config(_residential("Demand", "ResidentialDemand"))
    assert ("Demand", "create_demand", ["homes"]) in orchestration_plan(runs)
    model, state, drivers = build_model(runs)
    model.step(state, drivers, dt=DAY, t=0.0)
    assert model.wsimod_day.last_record.arc_flow.tolist() == [10 * 0.12]
    assert forcing_readers(runs) == {"temperature": ["homes"]}


def test_the_trace_fixes_the_call_sequence():
    model, state, drivers = build_model(_small())
    day = model.wsimod_day
    sequences = {}
    out = state
    for k in range(6):
        out = model.step(out, drivers, dt=DAY, t=k * DAY)
        sequences[k] = day.last_call_sequence
    assert sequences[0] == sequences[5] == day.call_sequence
    assert day.call_sequence  # the river's distribute and its arcs' calls


def test_a_changed_call_sequence_is_an_internal_error():
    model, state, drivers = build_model(_small())
    day = model.wsimod_day
    day.call_sequence = day.call_sequence[:-1]
    with pytest.raises(RuntimeError, match="D2"):
        model.step(state, drivers, dt=DAY, t=0.0)


# --- the river order (W64) ---------------------------------------------------------------------
def test_the_river_order_is_most_upstream_first_with_ties_in_configuration_order():
    network = read_wsimod_config(CASES["river_chain"].build())
    # r_lost links to no Waste and is left out (model.py:730-735, :780-807)
    assert river_order(network) == ["r_head", "r_mid", "r_side", "r_low", "r_end"]
    no_river_arcs = read_wsimod_config(
        {**CASES["river_chain"].build(), "arcs": [], "orchestration": []}
    )
    assert river_order(no_river_arcs) == []
    mapping = CASES["river_chain"].build()
    mapping["nodes"] = [n for n in mapping["nodes"] if n["name"] != "sea"]
    mapping["arcs"] = [a for a in mapping["arcs"] if "sea" not in (a["in_port"], a["out_port"])]
    with pytest.raises(ValueError, match="no Waste"):
        river_order(read_wsimod_config(mapping))


# --- drivers (D6) ------------------------------------------------------------------------------
def test_the_driver_names_follow_d6():
    network = read_wsimod_config(DATA / "quickstart")
    model, _, drivers = build_model(network, pollutants=False)
    day = model.wsimod_day
    assert drivers["wsimod.Arc.capacity"].shape == (len(network.arcs),)
    assert day.drivers["wsimod.ImperviousSurface.pore_depth"].labels == [("my_land", "urban")]
    assert day.drivers["wsimod.PerviousSurface.field_capacity"].labels == [("my_land", "rural")]
    assert day.drivers["wsimod.Sewer.pipe_timearea"].template.tolist() == [[1.0]]
    assert day.drivers["wsimod.Groundwater.residence_time"].labels == ["my_groundwater"]
    assert drivers["wsimod.precipitation"].shape == (len(network.dates), 1)
    # quantity only: no pollutant-keyed parameter, no concentrations
    assert not any(k.endswith("pollutant_load") for k in drivers)
    assert "wsimod.concentration" not in drivers


def test_a_time_area_fraction_is_a_driver():
    """W56/W70: the fractions are drivers ``(..., n, L+1)`` bound each day, learnable; a
    fraction at a time the configuration does not list is refused by name."""
    network = read_wsimod_config(CASES["foul_sewers"].build())
    model, state, drivers = build_model(network, pollutants=False)
    key = "wsimod.Sewer.pipe_timearea"
    layout = model.wsimod_day.drivers[key]
    assert layout.labels == ["foul_main", "trunk"]
    # foul_main {0: 0.6, 1: 0.4}; trunk the default {0: 1}, absent key 1 is 0 (D6)
    assert drivers[key].tolist() == [[0.6, 0.4], [1.0, 0.0]]
    fractions = drivers[key].clone().requires_grad_(True)
    model.step(state, {**drivers, key: fractions}, dt=DAY, t=0.0)
    # day 0: the share entered at time 0 is active at once and goes on to the trunk
    arc = model.wsimod_day.arc_names.index("foul_to_trunk")
    (grad,) = torch.autograd.grad(model.wsimod_day.last_record.arc_flow[arc], fractions)
    assert torch.isfinite(grad).all() and float(grad[0, 0]) != 0.0
    bad = drivers[key].clone()
    bad[1, 1] = 0.5
    match = r"pipe_timearea has a non-zero fraction at time\(s\) \[1\]"
    with pytest.raises(ValueError, match=match):
        model.step(state, {**drivers, key: bad}, dt=DAY, t=0.0)
