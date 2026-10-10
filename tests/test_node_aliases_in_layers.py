"""Layers store canonical node names; one node given under two names is an error."""

import pytest
import torch

from noodl.apps.building_physics import orifice_elements_from_edges, thermal_layer
from noodl.apps.building_physics.thermal import IdealGasDensity
from noodl.drives import Stack
from noodl.layers.potential import PotentialFlowLayer
from noodl.model import Model
from tests.apps.building_physics.test_components_physics import components_house

F64 = torch.float64


def house():
    net, _ = components_house().flatten()
    return net


def test_thermal_layer_stores_canonical_names():
    net = house()
    layer = thermal_layer(net, fixed_temperature=["door.a"])
    assert layer.boundary == ["ambient", "A.air"]
    assert layer.boundary_idx.tolist() == [0, 1]


def test_model_refs_show_canonical_boundary_nodes():
    net = house()
    el = orifice_elements_from_edges(net, "airpath")
    air = PotentialFlowLayer(net, "air", [el], [Stack.from_network(net, "airpath")],
                             boundary=["ambient"])
    th = thermal_layer(net, fixed_temperature=["door.a"])
    model = Model(net, {"air": air, "thermal": th}, closures=[IdealGasDensity(th, ambient_index=0)])
    assert model.refs.thermal.boundary_nodes == ["ambient", "A.air"]


def test_potential_layer_accepts_alias_and_stores_canonical():
    net = house()
    el = orifice_elements_from_edges(net, "airpath")
    layer = PotentialFlowLayer(net, "air", [el], boundary=["ambient", "door.a"])
    assert net.nodes[int(layer.bound[1])] == "A.air"


def test_node_given_twice_under_two_names_is_an_error():
    net = house()
    with pytest.raises(ValueError, match=r"door\.a.*A\.air|A\.air.*door\.a") as err:
        thermal_layer(net, fixed_temperature=["door.a", "A.air"])
    assert "A.air" in str(err.value)


def test_boundary_index_rejects_duplicates_but_unknown_is_keyerror():
    net = house()
    with pytest.raises(ValueError):
        net.boundary_index(["door.a", "A.air"])
    with pytest.raises(ValueError):
        net.interior_index(["door.a", "A.air"])
    with pytest.raises(KeyError):
        net.boundary_index(["nowhere"])


def test_canonical_resolves_aliases_in_order():
    net = house()
    assert net.canonical(["door.b", "ambient", "door.a"]) == ["B.air", "ambient", "A.air"]
    assert net.canonical([]) == []
