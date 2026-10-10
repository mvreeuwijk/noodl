"""Placement: absolute positions, heights shifted by CONTAM-style level offsets, facades."""

import warnings

import pytest

from noodl.components import Component, ComponentError, _registry
from tests.components.conftest import link, zone

ELEV = dict(node_elevations=("z_ref",), edge_elevations=("z_path",))


def tower():
    """building > floor (z=3) > wing (z=0.5, x=10) > room; a door in the floor at z_path 1."""
    b = Component("b")
    fl = b.add(Component("floor"), at=(0.0, 0.0, 3.0))
    wing = fl.add(Component("wing"), at=(10.0, 0.0, 0.5))
    r = wing.add(zone("r", volume=30.0, z_ref=0.2))
    wing.expose(air=r.ports.air)
    other = fl.add(zone("other", volume=10.0, z_ref=0.0))
    d = fl.add(link("d", z_path=1.0, area=0.5))
    fl.connect(wing.ports.air, d.ports.a)
    fl.connect(d.ports.b, other.ports.air)
    return b


def test_positions_accumulate_through_three_levels():
    net, _ = tower().flatten(**ELEV)
    assert net.graph.nodes["floor.wing.r.air"]["position"] == (10.0, 0.0, 3.5)
    assert net.graph.nodes["floor.other.air"]["position"] == (0.0, 0.0, 3.0)


def test_node_and_edge_elevations_are_offset():
    net, _ = tower().flatten(**ELEV)
    assert net.graph.nodes["floor.wing.r.air"]["z_ref"] == pytest.approx(3.7)
    assert net.graph.nodes["floor.other.air"]["z_ref"] == pytest.approx(3.0)
    assert net.graph.edges[net.edges[0]]["z_path"] == pytest.approx(4.0)   # door owner: floor


def test_other_attributes_are_untouched():
    net, _ = tower().flatten(**ELEV)
    assert net.graph.nodes["floor.wing.r.air"]["volume"] == 30.0
    assert net.graph.edges[net.edges[0]]["area"] == 0.5


def test_merged_node_uses_the_real_nodes_placement():
    b = Component("b")
    r = b.add(zone("r", z_ref=0.0), at=(0.0, 0.0, 6.0))
    d = b.add(link("d", z_path=1.0), at=(0.0, 0.0, 6.0))
    s = b.add(zone("s", z_ref=0.0))
    b.connect(r.ports.air, d.ports.a)
    b.connect(d.ports.b, s.ports.air)
    net, _ = b.flatten(**ELEV)
    assert net.graph.nodes["r.air"]["z_ref"] == 6.0
    assert net.graph.nodes["s.air"]["z_ref"] == 0.0
    assert net.graph.edges[net.edges[0]]["z_path"] == 7.0


def test_missing_node_elevation_takes_the_placement_height():
    b = Component("b")
    up = b.add(Component("up"), at=(0.0, 0.0, 9.0))
    up.add_node("wall", heat_capacity=1.0)            # no z_ref given
    flat = b.add(Component("flat"))
    flat.add_node("wall", heat_capacity=1.0)
    net, _ = b.flatten(**ELEV)
    assert net.graph.nodes["up.wall"]["z_ref"] == 9.0
    assert "z_ref" not in net.graph.nodes["flat.wall"]    # at z = 0 nothing is added


def test_edge_without_an_elevation_stays_without():
    b = Component("b")
    up = b.add(Component("up"), at=(0.0, 0.0, 3.0))
    up.add_node("a")
    up.add_node("c")
    up.add_edge("a", "c", kind="wall", ua=1.0)
    net, _ = b.flatten(**ELEV)
    assert "z_path" not in net.graph.edges[net.edges[0]]


def test_inner_node_uses_its_declaring_components_offset():
    b = Component("b")
    f = b.add(Component("f"), at=(0.0, 0.0, 3.0))
    f.inner("plenum", z_ref=0.5)
    net, _ = b.flatten(**ELEV)
    assert net.graph.nodes["f.plenum"]["z_ref"] == 3.5


def test_registry_is_used_and_kwargs_override_it():
    _registry.register_elevations(nodes=("z_ref",), edges=("z_path",))
    net, _ = tower().flatten()
    assert net.graph.nodes["floor.other.air"]["z_ref"] == 3.0
    with pytest.warns(UserWarning, match="register_elevations"):   # kwargs () turn them off
        net, _ = tower().flatten(node_elevations=(), edge_elevations=())
    assert net.graph.nodes["floor.other.air"]["z_ref"] == 0.0


def test_warns_when_placed_above_zero_with_no_elevations():
    with pytest.warns(UserWarning, match="register_elevations"):
        tower().flatten()


def test_no_warning_at_ground_level_or_with_elevations():
    b = Component("b")
    b.add(zone("r", z_ref=0.0), at=(5.0, 5.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b.flatten()
        tower().flatten(**ELEV)


# ---------------------------------------------------------------- facades
LOOKUP = dict(table_lookups={"facade": ("facades", "azimuth")})


def house_with_window(**window_attrs):
    b = Component("b")
    b.inner("ambient")
    b.inner_table("facades", south=180.0, east=90.0)
    r = b.add(zone("r"))
    w = b.add(Component("w"))
    w.add_terminal("a")
    w.add_edge(w.outer("ambient"), "a", kind="airpath", name="path", **window_attrs)
    w.expose("a")
    b.connect(w.ports.a, r.ports.air)
    return b


def test_facade_resolves_to_azimuth_and_is_kept():
    net, _ = house_with_window(facade="south").flatten(**LOOKUP)
    data = net.graph.edges[net.edges[0]]
    assert data["azimuth"] == 180.0 and data["facade"] == "south"


def test_nearest_table_wins():
    b = house_with_window(facade="south")
    b.children["w"].inner_table("facades", south=170.0)
    net, _ = b.flatten(**LOOKUP)
    assert net.graph.edges[net.edges[0]]["azimuth"] == 170.0


def test_key_missing_from_the_nearest_table_is_an_error_naming_the_entries():
    b = house_with_window(facade="west")
    with pytest.raises(ComponentError, match=r"facade='west'.*'east', 'south'"):
        b.flatten(**LOOKUP)


def test_facade_with_no_table_is_an_error():
    b = Component("b")
    b.inner("ambient")
    w = b.add(Component("w"))
    w.add_node("a")
    w.add_edge(w.outer("ambient"), "a", kind="airpath", facade="south")
    with pytest.raises(ComponentError,
                       match=r"no enclosing component declares inner_table\('facades'"):
        b.flatten(**LOOKUP)


def test_facade_and_azimuth_together_is_an_error():
    with pytest.raises(ComponentError, match="both facade= and azimuth="):
        house_with_window(facade="south", azimuth=10.0).flatten(**LOOKUP)


def test_rotating_the_building_is_one_table_edit():
    b = house_with_window(facade="east")
    b._tables["facades"]["east"] = 45.0
    net, _ = b.flatten(**LOOKUP)
    assert net.graph.edges[net.edges[0]]["azimuth"] == 45.0
