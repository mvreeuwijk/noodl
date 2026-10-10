"""Building factories: what each makes, and that they reuse the Network helpers."""

import pytest
import torch

from noodl.apps.building_physics import (
    WallMass,
    Zone,
    add_large_opening,
    add_zone,
    crack,
    door,
    room,
    shaft,
    window,
)
from noodl.apps.building_physics.attributes import EDGE_ATTRIBUTES, NODE_ATTRIBUTES
from noodl.components import Component, ComponentError, _registry
from noodl.topology import Network

F64 = torch.float64


def flat(*children, connects=()):
    b = Component("b")
    b.inner("ambient", z_ref=0.0, T0=283.15)
    b.inner_table("facades", south=180.0)
    for c in children:
        b.add(c)
    for x, y in connects:
        b.connect(x, y)
    return b.flatten(dtype=F64)


def test_import_registers_heights_and_facades():
    assert {"z_ref"} <= _registry.node_elevations()
    assert {"z_path"} <= _registry.edge_elevations()
    assert _registry.table_lookups()["facade"] == ("facades", "azimuth")


def test_room_matches_add_zone():
    r = room("A", volume=60.0, T0=288.15, z_ref=0.5)
    assert (r.template, list(r.ports)) == ("room", ["air"])
    net, _ = flat(r)
    hand = Network(dtype=F64)
    add_zone(hand, Zone("A", volume=60.0, T0=288.15, z_ref=0.5))
    expected = dict(hand.graph.nodes["A"])
    got = dict(net.graph.nodes["A.air"])
    got.pop("position")
    assert got == expected


def test_room_with_wall_reaches_ambient_through_outer():
    r = room("A", volume=60.0, wall=WallMass("wall", capacity=1e6, ua_zone=50.0, ua_ambient=20.0))
    net, _ = flat(r)
    assert net.nodes == ["ambient", "A.air", "A.wall"]
    assert [e[:2] for e in net.edges] == [("A.air", "A.wall"), ("A.wall", "ambient")]
    assert [net.graph.edges[e]["ua"] for e in net.edges] == [50.0, 20.0]


def test_room_with_wall_and_no_ambient_is_an_error():
    b = Component("b")
    b.add(room("A", volume=1.0, wall=WallMass("wall", 1.0, 1.0, 1.0)))
    with pytest.raises(ComponentError, match=r"outer\('ambient'\)"):
        b.flatten()


def test_door_is_the_two_opening_doorway():
    d = door("d", H=2.0, W=0.9, z_mid=1.0, Cd=0.7)
    assert (d.template, list(d.ports), d.edges) == ("door", ["a", "b"], ["low", "high"])
    lo, hi = d._edges
    assert lo.attrs["z_path"] == pytest.approx(1.0 - 4.0 / 9.0)
    assert hi.attrs["z_path"] == pytest.approx(1.0 + 4.0 / 9.0)
    assert lo.attrs["area"] == pytest.approx(0.9) and lo.attrs["Cd"] == 0.7
    assert lo.attrs["opening"] == 1.0


def test_add_large_opening_names_only_when_asked():
    net = Network()
    net.add_node("a")
    net.add_node("b")
    add_large_opening(net, "a", "b", H=2.0, W=1.0, z_mid=1.0)
    add_large_opening(net, "a", "b", H=2.0, W=1.0, z_mid=1.0, names=("lo", "hi"))
    assert [net.graph.edges[e].get("name") for e in net.edges] == [None, None, "lo", "hi"]


def test_interior_crack():
    c = crack("c", area=0.01, z_path=0.3)
    assert (c.template, list(c.ports), c.edges) == ("crack", ["a", "b"], ["path"])
    assert c._edges[0].attrs == {"z_path": 0.3, "Cd": 0.6, "area": 0.01}


def test_exterior_crack_points_from_ambient_and_carries_only_given_wind_attributes():
    c = crack("c", area=0.01, z_path=0.3, exterior=True, facade="south", Cp=0.6)
    assert list(c.ports) == ["a"]
    r = room("r", volume=1.0)
    net, _ = flat(c, r, connects=[(c.ports.a, r.ports.air)])
    (u, v, k), = net.edges
    data = net.graph.edges[u, v, k]
    assert (u, v) == ("ambient", "r.air")
    assert data["azimuth"] == 180.0 and data["Cp"] == 0.6 and "Ch" not in data


def test_wind_attributes_on_an_interior_crack_are_an_error():
    with pytest.raises(ComponentError, match="exterior"):
        crack("c", area=0.01, z_path=0.3, azimuth=90.0)


def test_window_is_exterior_by_default():
    w = window("w", area=0.5, z_path=1.2, azimuth=0.0)
    assert (w.template, list(w.ports)) == ("window", ["a"])
    assert w._edges[0].attrs["azimuth"] == 0.0


def test_shaft_stacks_one_zone_per_level():
    s = shaft("stair", levels=3, level_height=3.0, volume=20.0, area=2.0)
    assert list(s.ports) == ["levels[0]", "levels[1]", "levels[2]"]
    assert s.edges == ["slab[0]", "slab[1]"]
    net, _ = flat(s)
    assert [net.graph.nodes[n]["z_ref"] for n in net.nodes[1:]] == [0.0, 3.0, 6.0]
    assert [net.graph.edges[e]["z_path"] for e in net.edges] == [3.0, 6.0]
    assert [e[:2] for e in net.edges] == [("stair.levels[0]", "stair.levels[1]"),
                                          ("stair.levels[1]", "stair.levels[2]")]


def test_shaft_needs_two_levels():
    with pytest.raises(ComponentError, match="at least 2"):
        shaft("s", levels=1, level_height=3.0, volume=1.0, area=1.0)


def test_placed_door_follows_its_floor():
    b = Component("b")
    f = b.add(Component("f"), at=(0.0, 0.0, 3.0))
    A, B = f.add(room("A", volume=1.0)), f.add(room("B", volume=1.0))
    d = f.add(door("d", H=2.0, W=1.0, z_mid=1.0))
    f.connect(A.ports.air, d.ports.a)
    f.connect(d.ports.b, B.ports.air)
    net, _ = b.flatten()
    assert [net.graph.nodes[n]["z_ref"] for n in net.nodes] == [3.0, 3.0]
    assert [net.graph.edges[e]["z_path"] for e in net.edges] == pytest.approx(
        [4.0 - 4.0 / 9.0, 4.0 + 4.0 / 9.0])


def test_add_zone_on_a_network_still_adds_ambient_itself():
    net = Network()
    add_zone(net, Zone("A", volume=1.0, wall=WallMass("w", 1.0, 1.0, 1.0)))
    assert "ambient" in net.nodes


def test_catalogue_documents_facade_and_position():
    assert "position" in NODE_ATTRIBUTES
    assert "facade" in EDGE_ATTRIBUTES["airpath, to or from the ambient node"]
