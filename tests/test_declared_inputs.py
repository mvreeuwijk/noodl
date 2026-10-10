"""Inputs that closures, reactions, elements and drives read themselves -- the street
meteorology, the sewer inflows, CONTAM's densities and wind -- are declared, so
`model.refs.inputs` names them with their layout and unit and `Model.check` reports one
that is missing or mis-shaped before a run."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from noodl.apps import sewer, street_aq
from noodl.apps.building_physics import project_to_model, read_prj
from noodl.refs import Field

F64 = torch.float64
DATA = Path(__file__).resolve().parent / "data"


def _streets():
    return street_aq.StreetNetwork(
        streets=[street_aq.Street("s1", "a", "b", 100.0, 20.0, 20.0),
                 street_aq.Street("s2", "b", "c", 100.0, 20.0, 20.0)],
        x={"a": 0.0, "b": 100.0, "c": 200.0}, y={"a": 0.0, "b": 0.0, "c": 0.0},
    )


def _meteo(**shape):
    t = lambda v: torch.full(shape.get("shape", ()), v, dtype=F64)  # noqa: E731
    return {"U_ref": t(3.0), "theta_w": t(0.0), "h_abl": t(800.0)}


def _codes(report):
    return {(i.code, i.key) for i in report.issues}


# ------------------------------------------------------------------------------- street
def test_the_street_meteorology_is_declared_per_instance():
    model, state, drivers = street_aq.build_model(_streets())
    inputs = model.refs.inputs
    assert {"U_ref", "theta_w", "h_abl", "u_star", "lmo"} <= set(inputs)
    assert isinstance(inputs["theta_w"], Field) and inputs["theta_w"].scalar
    assert inputs["theta_w"].unit == "rad" and inputs["theta_w"].required == "always"
    assert inputs["U_ref"].required == "optional"          # u_star may replace it


def test_missing_meteorology_is_an_error_before_any_run():
    model, state, drivers = street_aq.build_model(_streets())
    report = model.check(state, drivers)
    assert ("missing-input", "theta_w") in _codes(report)
    assert ("missing-input", "h_abl") in _codes(report)
    assert model.check(state, {**drivers, **_meteo()}).ok


def test_per_street_meteorology_is_declared_over_the_streets_and_junctions():
    model, state, drivers = street_aq.build_model(_streets(), meteo="per_street")
    inputs = model.refs.inputs
    assert inputs["theta_w"].labels == ("s1", "s2")
    assert inputs["theta_w_junction"].labels == ("a", "b", "c")
    good = {**drivers, **_meteo(shape=(2,))}
    assert model.check(state, good).ok
    bad = {**good, "h_abl": torch.full((3,), 800.0, dtype=F64)}
    assert ("shape", "h_abl") in _codes(model.check(state, bad))


def test_the_chemistry_declares_its_photolysis_rate_and_temperature():
    reaction = street_aq.photostationary_for_streets(("no", "no2", "o3"))
    model, state, drivers = street_aq.build_model(
        _streets(), species=("no", "no2", "o3"), chemistry=reaction)
    assert {"J_NO2", "temperature"} <= set(model.refs.inputs)
    report = model.check(state, {**drivers, **_meteo()})
    assert ("missing-input", "J_NO2") in _codes(report)


def test_a_per_instance_input_is_given_as_a_tensor_not_by_label():
    model, _, _ = street_aq.build_model(_streets())
    with pytest.raises(TypeError, match="one value per instance"):
        model.refs.inputs["theta_w"].build({"s1": 0.0})
    d = model.drivers_from({"theta_w": torch.tensor([0.0, 1.0], dtype=F64)})
    assert d["theta_w"].shape == (2,)


# ------------------------------------------------------------------------------- sewer
def test_the_sewer_inputs_are_declared_with_their_layouts_and_units():
    model, state, drivers = sewer.build_model(sewer.tree_steady())
    inputs = model.refs.inputs
    assert inputs["inflow"].ordering == "full node order" and inputs["inflow"].unit == "m3/s"
    assert inputs["T_water"].unit == "degC" and inputs["T_head"].unit == "K"
    assert inputs["sewer.q_slope"].labels == tuple(
        m.name for m in sewer.tree_steady().manholes)
    assert model.check(state, drivers).ok


def test_a_missing_or_misshaped_sewer_input_is_an_error():
    model, state, drivers = sewer.build_model(sewer.tree_steady())
    missing = {k: v for k, v in drivers.items() if k != "T_water"}
    assert ("missing-input", "T_water") in _codes(model.check(state, missing))
    short = {**drivers, "inflow": drivers["inflow"][:-1]}
    assert ("shape", "inflow") in _codes(model.check(state, short))


def test_sewer_inflow_by_manhole_name_matches_the_hand_built_tensor():
    model, state, drivers = sewer.build_model(sewer.tree_steady())
    new = sewer.initial_drivers(model, values={"inflow": {"J1": 0.07}})
    hand = drivers["inflow"].clone()
    hand[model.net.node_index("J1")] = 0.07
    assert torch.equal(new["inflow"], hand)


def test_a_model_built_without_quality_declares_only_what_it_reads():
    model, state, drivers = sewer.build_model(sewer.tree_steady(), quality=False)
    assert set(model.refs.inputs) <= set(drivers)


# ------------------------------------------------------------------------------- CONTAM
def test_contam_densities_and_wind_are_declared():
    model, state, drivers = project_to_model(read_prj(DATA / "contam" / "valThreeZonesWthCtm.prj"))
    inputs = model.refs.inputs
    assert inputs["rho"].ordering == "full node order" and inputs["rho"].unit == "kg/m3"
    assert inputs["V_met"].scalar and inputs["theta_w"].unit == "deg"
    assert model.check(state, drivers).ok
