"""Node aliases: a flattened component network names each merged node once and keeps the
other names as aliases, which every name-based lookup must accept."""

import pytest
import torch

from noodl.apps.building_physics import (
    Zone,
    add_large_opening,
    add_zone,
    build_model,
    initial_drivers,
    initial_state,
    orifice_elements_from_edges,
)
from noodl.drives import Stack
from noodl.topology import Network

F64 = torch.float64


def two_zone(aliases=None):
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    add_zone(net, Zone("A", volume=60.0, T0=288.15))
    add_zone(net, Zone("B", volume=60.0, T0=285.15))
    add_large_opening(net, "A", "B", H=2.0, W=0.9, z_mid=1.0, Cd=0.78)
    net.add_edge("ambient", "B", kind="airpath", z_path=0.2, Cd=0.6, area=0.05)
    net.aliases = dict(aliases or {})
    el = orifice_elements_from_edges(net, "airpath")
    model = build_model(net, air_elements=[el], drives=[Stack.from_network(net, "airpath")])
    return net, model


def test_network_has_no_aliases_by_default():
    assert Network().aliases == {}


def test_node_index_resolves_an_alias():
    net, _ = two_zone({"door.a": "A"})
    assert net.node_index("door.a") == net.node_index("A")


def test_boundary_and_interior_index_resolve_aliases():
    net, _ = two_zone({"outside": "ambient"})
    assert net.boundary_index(["outside"]).tolist() == net.boundary_index(["ambient"]).tolist()
    assert net.interior_index(["outside"]).tolist() == net.interior_index(["ambient"]).tolist()


def test_unknown_name_still_raises():
    net, _ = two_zone({"door.a": "A"})
    with pytest.raises(KeyError, match="unknown node"):
        net.node_index("door.c")


def test_with_ambient_copies_aliases():
    net, _ = two_zone({"door.a": "A"})
    assert net.with_ambient("extra").aliases == {"door.a": "A"}


def test_field_build_accepts_alias_for_sources():
    _, model = two_zone({"door.a": "A"})
    th = model.refs.thermal
    by_alias = th.sources.build({"door.a": 1000.0})
    by_name = th.sources.build({"A": 1000.0})
    assert torch.equal(by_alias, by_name)


def test_field_build_accepts_alias_for_interior_and_boundary_orders():
    _, model = two_zone({"room_b": "B", "outside": "ambient"})
    th = model.refs.thermal
    x = th.temperature.build({"A": 290.0, "room_b": 291.0})
    assert torch.equal(x, th.temperature.build({"A": 290.0, "B": 291.0}))
    xb = th.boundary_temperature.build({"outside": 280.0})
    assert torch.equal(xb, th.boundary_temperature.build({"ambient": 280.0}))


def test_alias_and_canonical_in_one_mapping_is_an_error():
    _, model = two_zone({"door.a": "A"})
    with pytest.raises(ValueError, match="given twice"):
        model.refs.thermal.sources.build({"door.a": 1.0, "A": 2.0})


def test_named_returns_canonical_names_only():
    _, model = two_zone({"door.a": "A"})
    th = model.refs.thermal
    state = initial_state(model)
    assert list(th.temperature.named(state[th.temperature])) == ["A", "B"]


def test_initial_drivers_by_alias_equals_by_name():
    _, model = two_zone({"door.a": "A"})
    th = model.refs.thermal
    a = initial_drivers(model, values={th.sources: {"door.a": 500.0}})
    b = initial_drivers(model, values={th.sources: {"A": 500.0}})
    assert torch.equal(a[th.sources], b[th.sources])


def test_alias_to_a_node_outside_the_field_is_ignored_by_that_field():
    # "outside" is a boundary node: it is not a label of the interior-order temperature.
    _, model = two_zone({"outside": "ambient"})
    with pytest.raises(KeyError, match="unknown node"):
        model.refs.thermal.temperature.build({"outside": 280.0, "A": 1.0, "B": 1.0})
