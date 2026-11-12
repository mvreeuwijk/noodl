"""Every application initialises the same way:

    model[, state, drivers] = <app>.build_model(...)
    state   = <app>.initial_state(model, values={...by name...})
    drivers = <app>.initial_drivers(model, values={...by name...})
    model.check(state, drivers).raise_for_errors()

and a value given by name lands exactly where the hand-built tensor would put it.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import torch

from noodl.apps import sewer, street_aq, water, wsimod
from noodl.topology import Network

F64 = torch.float64
DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.parametrize(
    "package", ["building_physics", "street_aq", "sewer", "water", "wsimod"]
)
def test_every_application_has_the_same_initialisation_functions(package):
    module = importlib.import_module(f"noodl.apps.{package}")
    for name in ("build_model", "initial_state", "initial_drivers"):
        assert callable(getattr(module, name)) and name in module.__all__
        if name != "build_model":
            sig = inspect.signature(getattr(module, name))
            assert sig.parameters["values"].kind is inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------- water
def test_water_tank_levels_and_demands_by_name():
    net = water.read_epanet_inp(DATA / "water" / "Net1.inp")
    model, state0, drivers0 = water.build_model(net)
    # "water.tank_level" is an input key of initial_state only: the level is written into
    # the tank node's head, "water.phi"
    tanks = [t.name for t in model.control_closure.tanks]
    state = water.initial_state(model, values={"water.tank_level": {tanks[0]: 30.0}})
    assert "water.tank_level" not in state
    assert float(water.tank_levels(model, state)[0]) == 30.0
    bottom = net.tanks[0].elevation
    assert float(state["water.phi"][model.node_names.index(tanks[0])]) == bottom + 30.0
    with pytest.raises(KeyError, match="not tanks"):
        water.initial_state(model, values={"water.tank_level": {"nope": 1.0}})
    assert torch.equal(state["water.link_status"], state0["water.link_status"])
    ref = model.refs.water
    junction = next(n for n in ref.sources.labels if n in ref.sources.settable)
    drivers = water.initial_drivers(model, values={ref.sources: {junction: -0.01}})
    i = model.net.node_index(junction)
    expected = drivers0["water.sources"].clone()
    expected[i] = -0.01
    assert torch.equal(drivers["water.sources"], expected)
    assert torch.equal(drivers0["water.sources"], model.driver_template["water.sources"])
    report = model.check(state, drivers, steady=True)
    assert report.ok, str(report)


def test_water_initial_drivers_without_values_is_the_builders_template():
    model, _, drivers0 = water.build_model(water.twoloop())
    drivers = water.initial_drivers(model)
    assert set(drivers) == set(drivers0)
    for k in drivers0:
        assert torch.equal(drivers[k], drivers0[k])
        assert drivers[k] is not drivers0[k]


# ---------------------------------------------------------------------------- sewer
def test_sewer_species_and_levels_by_name():
    model, state0, drivers0 = sewer.build_model(sewer.tree_steady(), storage=True)
    wq = model.refs.water_quality
    assert wq.x.species == ("bod", "sulfide")
    node = wq.x.labels[0]
    state = sewer.initial_state(
        model, drivers0,
        values={"water_quality": {node: {"sulfide": 1e-3}}, "sewer.H": {node: 0.05}},
    )
    assert float(state["water_quality.x"][0, 1]) == 1e-3
    assert float(state["water_quality.x"][0, 0]) == 0.0
    h_labels = model.refs.closure_state["sewer.H"].labels
    assert float(state["sewer.H"][h_labels.index(node)]) == 0.05
    assert not model.check(state, drivers0).errors
    report = model.check(state, drivers0, dt=60.0, probe=True)
    assert report.ok and not report.warnings, str(report)
    assert "water_quality.q" in report.closure_outputs


# ---------------------------------------------------------------------------- street
def _streets():
    return street_aq.StreetNetwork(
        streets=[street_aq.Street("s1", "a", "b", 100.0, 20.0, 20.0),
                 street_aq.Street("s2", "b", "c", 100.0, 20.0, 20.0)],
        x={"a": 0.0, "b": 100.0, "c": 200.0},
        y={"a": 0.0, "b": 0.0, "c": 0.0},
    )


def _meteo():
    return {"U_ref": torch.tensor(3.0, dtype=F64), "theta_w": torch.tensor(0.0, dtype=F64),
            "h_abl": torch.tensor(800.0, dtype=F64)}


def test_street_sources_by_street_and_species_match_the_hand_built_tensor():
    model, state, drivers0 = street_aq.build_model(_streets(), species=("no", "no2"))
    drivers = street_aq.initial_drivers(
        model, values={"street.sources": {"s1": {"no2": 4e-4}}, **_meteo()}
    )
    hand = torch.zeros(model.net.n, 2, dtype=F64)
    hand[model.net.node_index("s1"), 1] = 4e-4
    assert torch.equal(drivers["street.sources"], hand)
    report = model.check(state, drivers, dt=60.0, probe=True)
    assert report.ok and not report.warnings, str(report)
    assert {"U_ref", "theta_w", "h_abl"} <= report.driver_reads


def test_street_probe_flags_a_misspelt_meteorology_key():
    model, state, _ = street_aq.build_model(_streets())
    meteo = _meteo()
    meteo["U_rf"] = meteo.pop("U_ref")
    drivers = street_aq.initial_drivers(model, values=meteo)
    report = model.check(state, drivers, dt=60.0, probe=True)
    assert not report.ok                      # the closure needs U_ref and raises
    assert "U_ref" in str(report)


# --------------------------------------------------------------------------- wsimod
def test_wsimod_values_by_name_reproduce_the_hand_built_step():
    network = wsimod.read_wsimod_config(DATA / "wsimod" / "small")
    model, state0, drivers0 = wsimod.build_model(network)
    day = model.wsimod_day
    state = wsimod.initial_state(model, values={"wsimod.storage": {("river", "tank"): 9.0}})
    drivers = wsimod.initial_drivers(
        model, values={"wsimod.Arc.capacity": {"river_to_outlet": 3.0}}
    )
    assert model.check(state, drivers).ok

    hand_state = {k: v.clone() for k, v in state0.items()}
    row = day.state_labels("wsimod.storage").index(("river", "tank"))
    hand_state["wsimod.storage"][row, 0] = 9.0
    capacity = drivers0["wsimod.Arc.capacity"].clone()
    capacity[day.arc_names.index("river_to_outlet")] = 3.0
    hand_drivers = {**drivers0, "wsimod.Arc.capacity": capacity}
    for key in state:
        assert torch.equal(state[key], hand_state[key]), key
    out = model.step(state, drivers, dt=86400.0, t=0.0)
    ref = model.step(hand_state, hand_drivers, dt=86400.0, t=0.0)
    for key in out:
        assert torch.equal(out[key], ref[key]), key


def test_wsimod_refuses_an_unknown_name():
    model, _, _ = wsimod.build_model(wsimod.read_wsimod_config(DATA / "wsimod" / "small"))
    with pytest.raises(KeyError, match="did you mean 'river_to_outlet'"):
        wsimod.initial_drivers(model, values={"wsimod.Arc.capacity": {"river_to_outlt": 1.0}})


# ------------------------------------------------- review regressions (stock app models)
def test_a_single_named_species_may_be_given_by_its_name():
    model, _, _ = street_aq.build_model(_streets())          # species=("nox",)
    state = street_aq.initial_state(model, values={"street": {"s1": {"nox": 1e-7}}})
    drivers = street_aq.initial_drivers(model, values={"street.sources": {"s1": {"nox": 2e-3}}})
    assert float(state["street.x"][0]) == 1e-7
    assert float(drivers["street.sources"][model.net.node_index("s1")]) == 2e-3
    with pytest.raises(KeyError, match="unknown species 'no2'"):
        street_aq.initial_state(model, values={"street": {"s1": {"no2": 1.0}}})


@pytest.mark.parametrize("kw", [{}, {"storage": True}, {"air": False}, {"quality": False}])
def test_the_static_check_is_clean_on_stock_sewer_models(kw):
    model, state, drivers = sewer.build_model(sewer.tree_steady(), **kw)
    report = model.check(state, drivers)
    assert report.ok and not report.warnings, str(report)


def test_the_static_check_is_clean_on_a_stock_street_model():
    model, state, _ = street_aq.build_model(_streets())
    drivers = street_aq.initial_drivers(model, values=_meteo())
    report = model.check(state, drivers)
    assert report.ok and not report.warnings, str(report)
    assert "street.q" in report.closure_outputs


def test_the_building_builder_can_return_the_same_triple_as_the_others():
    from noodl.apps import building_physics as bp
    from noodl.apps.building_physics.elements import orifice_law_from_edges
    from noodl.heads import Stack

    def network():
        net = Network(dtype=F64)
        net.add_node("ambient", z_ref=0.0)
        bp.add_zone(net, bp.Zone("A", volume=60.0, T0=290.0))
        net.add_edge("ambient", "A", kind="airpath", z_path=0.5, Cd=0.6, area=0.05)
        net.add_edge("A", "ambient", kind="airpath", z_path=2.0, Cd=0.6, area=0.05)
        return net

    net = network()
    kw = dict(air_laws=[orifice_law_from_edges(net, "airpath")],
              heads=[Stack.from_network(net, "airpath")])
    model, state, drivers = bp.build_model(net, return_inputs=True, **kw)
    assert set(state) == set(bp.initial_state(model))
    assert set(drivers) == set(bp.initial_drivers(model))
    assert model.check(state, drivers).ok
    assert not isinstance(bp.build_model(network(), **kw), tuple)   # default unchanged


def test_water_demand_by_name_is_a_negative_source():
    model, state, drivers = water.build_model(water.twoloop())
    src = model.refs.water.sources
    by_demand = water.initial_drivers(model, demand={"J2": 0.012})
    by_source = water.initial_drivers(model, values={src: {"J2": -0.012}})
    assert torch.equal(by_demand[src], by_source[src])
    with pytest.raises(KeyError, match="J22"):
        water.initial_drivers(model, demand={"J22": 0.012})
    with pytest.raises(ValueError, match="R1"):
        water.initial_drivers(model, demand={"R1": 0.012})


def test_water_demand_is_refused_on_a_pressure_driven_model():
    model, state, drivers = water.build_model(water.twoloop(), pda=True, p_min=0.0, p_req=10.0)
    with pytest.raises(ValueError, match="pressure-driven"):
        water.initial_drivers(model, demand={"J2": 0.012})


def test_water_sources_by_value_are_refused_on_a_pressure_driven_model():
    model, state, drivers = water.build_model(water.twoloop(), pda=True, p_min=0.0, p_req=10.0)
    src = model.refs.water.sources
    with pytest.raises(ValueError, match="pressure-driven"):
        water.initial_drivers(model, values={src: {"J2": -0.012}})
    with pytest.raises(ValueError, match="pressure-driven"):
        water.initial_drivers(model, values={"water.sources": {"J2": -0.012}})
    assert water.initial_drivers(model, values={src.layer + ".phi_boundary": {"R1": 50.0}})
