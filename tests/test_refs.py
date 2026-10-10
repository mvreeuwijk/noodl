"""Layer identity, typed layer references, and building inputs by node name."""

from __future__ import annotations

import pickle

import pytest
import torch

from benchmarks.natural_ventilation import build as build_ventilation
from noodl.apps.building_physics import (
    WallMass,
    Zone,
    add_zone,
    build_model,
    initial_drivers,
    initial_state,
    thermal_layer,
)
from noodl.apps.building_physics.elements import orifice_elements_from_edges
from noodl.drives import Stack
from noodl.elements.powerlaw import PowerLaw
from noodl.layers.potential import PotentialFlowLayer
from noodl.model import Model
from noodl.refs import Field
from noodl.topology import Network

F64 = torch.float64


def _two_zone_with_wall():
    """ambient, A (with a wall node), B: the wall node is a thermal unknown but inactive for
    species, so the interior orders of the two transport layers differ."""
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0, T0=280.0)
    add_zone(net, Zone("A", volume=50.0, T0=293.0, wall=WallMass("A_wall", 1e6, 50.0, 40.0)))
    add_zone(net, Zone("B", volume=30.0, T0=291.0))
    net.add_edge("ambient", "A", kind="airpath", z_path=0.5, Cd=0.6, area=0.02)
    net.add_edge("A", "B", kind="airpath", z_path=1.0, Cd=0.6, area=0.05)
    net.add_edge("B", "ambient", kind="airpath", z_path=2.0, Cd=0.6, area=0.02)
    el = orifice_elements_from_edges(net, "airpath")
    model = build_model(net, air_elements=[el], drives=[Stack.from_network(net, "airpath")],
                        species=("co2", "pm25"))
    return net, model


# --------------------------------------------------------------------- layer identity
def test_a_layer_registered_under_another_name_is_refused():
    net, model = _two_zone_with_wall()
    th = model.transport["thermal"]
    with pytest.raises(ValueError, match="registered as 'heat'.*named 'thermal'"):
        Model(net, {"air": model.potential["air"], "heat": th})


def test_layers_may_be_given_as_a_sequence_keyed_by_their_own_names():
    net, model = _two_zone_with_wall()
    seq = Model(net, [model.potential["air"], model.transport["thermal"]],
                closures=model.closures)
    assert list(seq.layers) == ["air", "thermal"]


def test_two_layers_with_one_name_are_refused():
    net = Network(dtype=F64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="k")
    leak = PowerLaw(C=torch.tensor(0.01), n=torch.tensor(0.65), kind="k")
    one = PotentialFlowLayer(net, "air", [leak], boundary=["a"])
    two = PotentialFlowLayer(net, "air", [leak], boundary=["a"])
    with pytest.raises(ValueError, match="two layers named 'air'"):
        Model(net, [one, two])


def test_a_renamed_thermal_layer_drives_the_density_closure_under_its_own_name():
    """The density closure reads `"<thermal layer name>.x"`; with the identity check the
    registered key and that name cannot drift apart."""
    net, _ = _two_zone_with_wall()
    th = thermal_layer(net, name="heat")
    air = PotentialFlowLayer(net, "air", [orifice_elements_from_edges(net, "airpath")],
                             drives=[Stack.from_network(net, "airpath")], boundary=["ambient"])
    from noodl.apps.building_physics import IdealGasDensity

    model = Model(net, [air, th], closures=[IdealGasDensity(th, ambient_index=0)])
    state = initial_state(model)
    drivers = initial_drivers(model)
    assert set(state) == {"heat.x"} and "heat.x_boundary" in drivers
    model.step(state, drivers, 60.0)


# ------------------------------------------------------------------------ references
def test_refs_expose_every_key_with_its_order():
    net, model = _two_zone_with_wall()
    th, sp, air = model.refs.thermal, model.refs.species, model.refs["air"]
    assert th.x == "thermal.x" and isinstance(th.x, Field)
    assert th.x.labels == ("A", "A_wall", "B")
    assert sp.x.labels == ("A", "B")               # the wall node is inactive for species
    assert sp.x.species == ("co2", "pm25") and sp.x.trailing == (2, 2)
    assert th.x_boundary.labels == ("ambient",)
    assert th.sources.labels == tuple(net.nodes)    # full node order
    assert air.phi_boundary.ordering == "boundary order"
    assert th.unit == "K" and th.quantity == "temperature"
    assert "flows" not in th.inputs                 # the air layer provides them
    with pytest.raises(AttributeError, match="has no key 'source'"):
        th.source  # noqa: B018
    with pytest.raises(KeyError, match="did you mean 'thermal'"):
        model.refs["therml"]


def test_a_field_is_its_plain_key_and_pickles_as_one():
    _, model = _two_zone_with_wall()
    key = model.refs.thermal.sources
    d = {key: 1}
    assert d["thermal.sources"] == 1 and hash(key) == hash("thermal.sources")
    # Round-trips bytes this test itself just produced; nothing untrusted is loaded.
    restored = pickle.loads(pickle.dumps(d))
    assert type(next(iter(restored))) is str


# -------------------------------------------------------------------- named building
def test_build_puts_each_value_at_its_own_node():
    net, model = _two_zone_with_wall()
    src = model.refs.thermal.sources.build({"B": 500.0, "A": 1000.0})
    expected = torch.zeros(net.n, dtype=F64)
    expected[net.node_index("A")] = 1000.0
    expected[net.node_index("B")] = 500.0
    assert torch.equal(src, expected)
    x = model.refs.thermal.x.build({"A_wall": 290.0, "B": 291.0, "A": 293.0})
    assert x.tolist() == [293.0, 290.0, 291.0]


def test_build_refuses_unknown_and_fixed_nodes():
    _, model = _two_zone_with_wall()
    src = model.refs.thermal.sources
    with pytest.raises(KeyError, match="unknown node 'Bb'; did you mean 'B'"):
        src.build({"Bb": 1.0})
    with pytest.raises(ValueError, match="'ambient' may not carry a value"):
        src.build({"ambient": 1.0})
    with pytest.raises(ValueError, match="'A_wall' may not carry a value"):
        model.refs.species.sources.build({"A_wall": {"co2": 1.0}})
    with pytest.raises(KeyError, match="no value for 1 node"):
        model.refs.thermal.x.build({"A": 293.0, "B": 291.0})


def test_build_by_species_name():
    net, model = _two_zone_with_wall()
    src = model.refs.species.sources.build({"A": {"pm25": 2.0}})
    assert src.shape == (net.n, 2)
    assert src[net.node_index("A")].tolist() == [0.0, 2.0]
    with pytest.raises(KeyError, match="unknown species 'no2'"):
        model.refs.species.sources.build({"A": {"no2": 1.0}})


def test_build_batches_and_keeps_gradients_dtype_and_device():
    net, model = _two_zone_with_wall()
    q = torch.tensor([100.0, 200.0, 300.0], dtype=F64, requires_grad=True)
    src = model.refs.thermal.sources.build({"A": q, "B": 5.0})
    assert src.shape == (3, net.n)
    assert src[:, net.node_index("B")].tolist() == [5.0, 5.0, 5.0]
    src.sum().backward()
    assert q.grad.tolist() == [1.0, 1.0, 1.0]
    f32 = model.refs.thermal.sources.build({"A": torch.tensor(1.0, dtype=torch.float32)})
    assert f32.dtype == torch.float32 and f32.device == q.device


def test_build_over_a_base_changes_only_the_named_nodes():
    _, model = _two_zone_with_wall()
    base = initial_state(model)["thermal.x"]
    x = model.refs.thermal.x.build({"B": 300.0}, base=base)
    assert x.tolist() == [293.0, 293.0, 300.0]
    assert base.tolist() == [293.0, 293.0, 291.0]


def test_named_reads_a_tensor_back_by_node():
    _, model = _two_zone_with_wall()
    state = initial_state(model)
    named = model.refs.thermal.x.named(state["thermal.x"])
    assert {k: float(v) for k, v in named.items()} == {"A": 293.0, "A_wall": 293.0, "B": 291.0}


def test_state_from_accepts_a_layer_name_and_reports_what_is_missing():
    _, model = _two_zone_with_wall()
    state = model.state_from({"thermal": {"A": 290.0, "A_wall": 290.0, "B": 290.0},
                              "species": {"A": {"co2": 0.0, "pm25": 0.0},
                                          "B": {"co2": 0.0, "pm25": 0.0}}})
    assert set(state) == {"thermal.x", "species.x"}
    with pytest.raises(KeyError, match="no initial value for 'species.x'"):
        model.state_from({"thermal": {"A": 290.0, "A_wall": 290.0, "B": 290.0}})


def test_a_mapping_for_a_key_of_unknown_layout_is_refused_but_a_tensor_passes():
    _, model = _two_zone_with_wall()
    with pytest.raises(KeyError, match="does not know this key's layout"):
        model.drivers_from({"my_gain": {"A": 1.0}})
    d = model.drivers_from({"my_gain": torch.tensor(5.0, dtype=F64)})
    assert float(d["my_gain"]) == 5.0
    # `P_ref` is declared by the density closure: one value per instance, never by label.
    with pytest.raises(TypeError, match="one value per instance"):
        model.drivers_from({"P_ref": {"A": 1.0}})
    d = model.drivers_from({"P_ref": torch.tensor(101000.0, dtype=F64)})
    assert float(d["P_ref"]) == 101000.0


# ------------------------------------------------------------------- equivalence
def test_named_inputs_reproduce_the_hand_built_run_exactly():
    """The natural-ventilation benchmark builds its drivers by hand in full node order.
    The same drivers built by name must give bitwise the same trajectory."""
    net, model, state0, sources = build_ventilation("iterate")
    hand = {
        "air.phi_boundary": torch.zeros(1, dtype=F64),
        "thermal.x_boundary": torch.tensor([285.0], dtype=F64),
        "thermal.sources": sources,
    }
    th = model.refs.thermal
    named = initial_drivers(model, values={
        th.x_boundary: {"ambient": 285.0},
        th.sources: {"A": 1000.0},
    })
    assert set(named) == set(hand)
    for key in hand:
        assert torch.equal(named[key], hand[key]), key
    assert torch.equal(initial_state(model)["thermal.x"], state0["thermal.x"])
    s_hand, s_named = dict(state0), initial_state(model)
    for _ in range(3):
        s_hand = model.step(s_hand, hand, 600.0)
        s_named = model.step(s_named, named, 600.0)
    for key in s_hand:
        assert torch.equal(s_hand[key], s_named[key]), key


def test_named_inputs_carry_a_gradient_through_a_step():
    net, model, state0, _ = build_ventilation("pingpong")
    power = torch.tensor(1000.0, dtype=F64, requires_grad=True)
    th = model.refs.thermal
    drivers = initial_drivers(model, values={th.x_boundary: {"ambient": 285.0},
                                             th.sources: {"A": power}})
    out = model.step(initial_state(model), drivers, 600.0)
    out["thermal.x"][0].backward()
    assert power.grad is not None and float(power.grad) > 0.0


# ---------------------------------------------------------------- review regressions
def test_a_single_species_column_layout_is_accepted_as_the_layer_accepts_it():
    net = _two_zone_with_wall()[0]
    model = build_model(net, air_elements=[orifice_elements_from_edges(net, "airpath")],
                        drives=[Stack.from_network(net, "airpath")], species=1)
    state = initial_state(model)
    state["species.x"] = state["species.x"].unsqueeze(-1)            # (n_i, 1)
    drivers = initial_drivers(model)
    drivers["species.x_boundary"] = drivers["species.x_boundary"].unsqueeze(-1)  # (1, 1)
    report = model.check(state, drivers)
    assert report.ok, str(report)
    sp = model.refs.species
    assert sp.x.named(state["species.x"])["A"].shape == ()
    rebuilt = sp.x.build({"A": 1e-3}, base=state["species.x"])
    assert rebuilt.tolist() == [1e-3, 0.0]


def test_a_species_count_may_be_any_integer_but_not_a_string():
    np = pytest.importorskip("numpy")
    net = _two_zone_with_wall()[0]
    kw = dict(air_elements=[orifice_elements_from_edges(net, "airpath")],
              drives=[Stack.from_network(net, "airpath")])
    assert build_model(net, species=np.int64(2), **kw).transport["species"].n_species == 2
    with pytest.raises(TypeError, match=r"pass \('co2',\)"):
        build_model(net, species="co2", **kw)


def test_a_default_never_fills_a_node_the_layer_does_not_solve_for():
    net, model = _two_zone_with_wall()
    src = model.refs.thermal.sources.build({"A": 1.0}, default=5.0)
    assert float(src[net.node_index("ambient")]) == 0.0
    assert float(src[net.node_index("B")]) == 5.0


def test_species_names_must_be_unique_and_a_species_may_be_given_once():
    from noodl.apps.building_physics import species_layer

    net = _two_zone_with_wall()[0]
    with pytest.raises(ValueError, match="repeat a name"):
        species_layer(net, n_species=2, species_names=["co2", "co2"])
    _, model = _two_zone_with_wall()
    with pytest.raises(ValueError, match="gives species 0 twice"):
        model.refs.species.sources.build({"A": {"co2": 1.0, 0: 2.0}})


def test_keys_are_named_for_the_physics_the_layer_carries():
    """`.x`/`.phi`/`.s`/`.q` are the solver's suffixes; the attributes users write name the
    quantity, taken from each layer's `quantity` tag, with the suffixes kept as aliases."""
    _, model = _two_zone_with_wall()
    th, sp, air = model.refs.thermal, model.refs.species, model.refs.air
    assert th.temperature is th.x and th.boundary_temperature is th.x_boundary
    assert sp.mass_fraction is sp.x and sp.boundary_mass_fraction is sp.x_boundary
    assert air.pressure is air.phi and air.boundary_pressure is air.phi_boundary
    assert air.flow is air.q
    assert th.temperature.attribute == "temperature"
    with pytest.raises(AttributeError, match="did you mean 'temperature'"):
        th.temprature  # noqa: B018
    assert "temperature" in dir(th)
