"""Flattening: which nodes exist after connecting, what they are called, and the errors."""

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from noodl.components import Component, ComponentError, NameMap
from tests.components.conftest import leak, link, zone

F64 = torch.float64


def two_rooms():
    b = Component("house")
    A, B = b.add(zone("A", volume=60.0)), b.add(zone("B", volume=45.0))
    d = b.add(link("door", Cd=0.78))
    b.connect(A.ports.air, d.ports.a)
    b.connect(d.ports.b, B.ports.air)
    return b


def edge_names(net):
    return [net.graph.edges[e]["name"] for e in net.edges]


def test_returns_a_network_and_a_name_map():
    net, names = two_rooms().flatten(dtype=F64)
    assert isinstance(names, NameMap) and names.net is net
    assert net.dtype == F64


def test_door_between_two_rooms_gives_one_node_per_room():
    net, _ = two_rooms().flatten()
    assert net.nodes == ["A.air", "B.air"]
    assert edge_names(net) == ["door.path"]
    assert net.edges[0][:2] == ("A.air", "B.air")
    assert net.aliases == {"door.a": "A.air", "door.b": "B.air"}


def test_attributes_are_copied_and_kind_kept():
    net, _ = two_rooms().flatten()
    assert net.graph.nodes["B.air"]["volume"] == 45.0
    data = net.graph.edges[net.edges[0]]
    assert data["kind"] == "airpath" and data["Cd"] == 0.78 and data["name"] == "door.path"


def test_root_level_names_have_no_prefix():
    c = Component("root")
    c.add_node("air", volume=1.0)
    c.add_node("wall")
    c.add_edge("air", "wall", kind="wall", name="w")
    net, _ = c.flatten()
    assert net.nodes == ["air", "wall"]
    assert edge_names(net) == ["w"]


def test_flatten_copies_attributes_both_ways():
    b = two_rooms()
    net, _ = b.flatten()
    net.graph.nodes["A.air"]["volume"] = -1.0
    net.graph.edges[net.edges[0]]["Cd"] = -1.0
    assert b.children["A"]._local["air"].attrs["volume"] == 60.0
    assert b.children["door"]._edges[0].attrs["Cd"] == 0.78
    b.children["A"]._local["air"].attrs["volume"] = 99.0
    assert net.graph.nodes["A.air"]["volume"] == -1.0


def test_flatten_twice_is_identical_and_reflects_edits():
    b = two_rooms()
    n1, _ = b.flatten()
    n2, _ = b.flatten()
    assert n1.nodes == n2.nodes and n1.edges == n2.edges and n1.aliases == n2.aliases
    assert [dict(n1.graph.nodes[n]) for n in n1.nodes] == [
        dict(n2.graph.nodes[n]) for n in n2.nodes]
    b.add(zone("C", volume=10.0))
    n3, _ = b.flatten()
    assert n3.nodes == ["A.air", "B.air", "C.air"]
    assert n1.nodes == ["A.air", "B.air"]


def test_three_doors_on_one_room():
    b = Component("b")
    hub = b.add(zone("hub", volume=1.0))
    for k in range(3):
        r = b.add(zone(f"r{k}", volume=1.0))
        d = b.add(link(f"d{k}"))
        b.connect(hub.ports.air, d.ports.a)
        b.connect(d.ports.b, r.ports.air)
    net, _ = b.flatten()
    assert [e[0] for e in net.edges] == ["hub.air"] * 3
    assert [e[1] for e in net.edges] == ["r0.air", "r1.air", "r2.air"]


def test_connects_are_transitive():
    b = Component("b")
    A = b.add(zone("A", volume=1.0))
    l1, l2 = b.add(link("l1")), b.add(link("l2"))
    b.connect(A.ports.air, l1.ports.a)
    b.connect(l1.ports.a, l2.ports.a)       # l2.a reaches A only through l1.a
    b.connect(l1.ports.b, l2.ports.b)       # a junction of two terminals
    net, _ = b.flatten()
    assert net.aliases["l2.a"] == "A.air"
    assert net.nodes == ["A.air", "l1.b"]   # the junction is named by its first terminal


def test_terminal_junction_is_named_by_the_outermost_terminal():
    b = Component("b")
    b.add_terminal("j")
    f = b.add(Component("f"))
    l1 = f.add(link("l1"))
    f.expose(x=l1.ports.b)
    b.connect("j", f.ports.x)
    A = b.add(zone("A", volume=1.0))
    f2 = b.add(Component("f2"))
    l2 = f2.add(link("l2"))
    f2.expose(y=l2.ports.a)
    f2.expose(z=l2.ports.b)
    b.connect(f2.ports.y, "j")
    f.expose(w=l1.ports.a)
    b.connect(f.ports.w, A.ports.air)
    b.connect(f2.ports.z, A.ports.air)
    net, _ = b.flatten()
    assert "j" in net.nodes                 # depth 0 beats f.l1.b and f2.l2.a
    assert net.graph.nodes["j"] == {"position": (0.0, 0.0, 0.0)}  # no attributes


def test_reexport_through_two_levels():
    b = Component("b")
    fl = b.add(Component("floor"))
    wing = fl.add(Component("wing"))
    r = wing.add(zone("r", volume=1.0))
    wing.expose(door=r.ports.air)
    fl.expose(corridor=wing.ports.door)
    d = b.add(link("d"))
    x = b.add(zone("x", volume=1.0))
    b.connect(fl.ports.corridor, d.ports.a)
    b.connect(d.ports.b, x.ports.air)
    b.add_edge(fl.ports.corridor, x.ports.air, kind="airpath", name="direct")
    net, names = b.flatten()
    assert net.aliases["d.a"] == "floor.wing.r.air"
    assert ("floor.wing.r.air", "x.air") in [e[:2] for e in net.edges]
    assert names.unconnected_ports == []


def test_terminal_used_by_a_parent_edge_is_kept():
    b = Component("b")
    A = b.add(zone("A", volume=1.0))
    d = b.add(link("d"))
    b.connect(A.ports.air, d.ports.a)
    b.add_node("n")
    b.add_edge(d.ports.b, "n", kind="airpath", name="onward")   # d.b is in use, not closed
    net, _ = b.flatten()
    assert net.nodes == ["n", "A.air", "d.b"]
    assert [net.graph.edges[e]["name"] for e in net.edges] == ["onward", "d.path"]


def test_inner_at_root_is_bare_and_deeper_is_dotted():
    b = Component("b")
    b.inner("ambient", z_ref=0.0)
    f = b.add(Component("f"))
    f.inner("plenum", T0=300.0)
    net, _ = b.flatten()
    assert net.nodes == ["ambient", "f.plenum"]
    assert net.graph.nodes["f.plenum"]["T0"] == 300.0


def test_outer_resolves_to_the_nearest_enclosing_inner():
    b = Component("b")
    b.inner("plenum", T0=290.0)
    f = b.add(Component("f"))
    f.inner("plenum", T0=300.0)
    inside = f.add(Component("inside"))
    inside.add_node("air")
    inside.add_edge(inside.outer("plenum"), "air", kind="k", name="supply")
    outside = b.add(Component("outside"))
    outside.add_node("air")
    outside.add_edge(outside.outer("plenum"), "air", kind="k", name="supply")
    net, _ = b.flatten()
    ends = {net.graph.edges[e]["name"]: e[0] for e in net.edges}
    assert ends == {"f.inside.supply": "f.plenum", "outside.supply": "plenum"}


def test_leak_reaches_the_root_ambient_from_depth_three():
    b = Component("b")
    b.inner("ambient")
    f = b.add(Component("f"))
    w = f.add(Component("w"))
    r = w.add(zone("r", volume=1.0))
    lk = w.add(leak("lk"))
    w.connect(lk.ports.a, r.ports.air)
    net, _ = b.flatten()
    assert net.edges[0][:2] == ("ambient", "f.w.r.air")


def test_unresolved_outer_is_an_error():
    b = Component("b")
    lk = b.add(leak("lk"))
    b.add(zone("r", volume=1.0))
    b.connect(lk.ports.a, b.children["r"].ports.air)
    with pytest.raises(ComponentError, match=r"lk: outer\('ambient'\) has no enclosing inner"):
        b.flatten()


def test_two_rooms_merged_directly_is_a_conflict():
    b = Component("b")
    A, B = b.add(zone("A", volume=1.0)), b.add(zone("B", volume=2.0))
    b.connect(A.ports.air, B.ports.air)
    with pytest.raises(ComponentError, match=r"'A.air' and 'B.air' into one node.*door"):
        b.flatten()


def test_all_problems_are_reported_together():
    b = Component("b")
    for f in ("f1", "f2"):
        fl = b.add(Component(f))
        A, B = fl.add(zone("A", volume=1.0)), fl.add(zone("B", volume=1.0))
        fl.connect(A.ports.air, B.ports.air)
    b.add(leak("lk"))
    with pytest.raises(ComponentError) as info:
        b.flatten()
    message = str(info.value)
    assert "3 problem(s)" in message
    assert "f1.A.air" in message and "f2.A.air" in message and "outer('ambient')" in message


def test_edge_whose_ends_merge_is_an_error():
    b = Component("b")
    A = b.add(zone("A", volume=1.0))
    d = b.add(link("d"))
    b.connect(A.ports.air, d.ports.a)
    b.connect(d.ports.b, A.ports.air)
    with pytest.raises(ComponentError, match=r"d\.path: both ends .* 'A.air'"):
        b.flatten()


def test_lone_terminal_is_dropped_with_its_edges():
    b = Component("b")
    A = b.add(zone("A", volume=1.0))
    d = b.add(link("d"))
    b.connect(A.ports.air, d.ports.a)      # d.b is never connected: a closed door
    net, names = b.flatten()
    assert net.nodes == ["A.air"]
    assert net.b == 0
    assert "d.b" not in net.aliases
    assert names._components["d"].ports == {"a": "A.air", "b": None}


def test_nodes_are_contiguous_per_component_in_tree_order():
    b = Component("b")
    for f in ("f1", "f2"):
        fl = b.add(Component(f))
        for r in ("r1", "r2"):
            room = fl.add(Component(r))
            room.add_node("air")
            room.add_node("wall")
    net, _ = b.flatten()
    assert net.nodes == [f"{f}.{r}.{n}" for f in ("f1", "f2") for r in ("r1", "r2")
                         for n in ("air", "wall")]


def test_name_map_resolve():
    _, names = two_rooms().flatten()
    assert names.resolve("door.a") == "A.air"
    assert names.resolve("A.air") == "A.air"
    with pytest.raises(KeyError, match=r"unknown node 'A.aier'.*'A.air'"):
        names.resolve("A.aier")
    assert names.paths == ["", "A", "B", "door"]


# ---------------------------------------------------------------- property-based
@st.composite
def plans(draw):
    floors = []
    for _ in range(draw(st.integers(1, 3))):
        n = draw(st.integers(1, 4))
        floors.append((n, draw(st.lists(st.booleans(), min_size=n - 1, max_size=n - 1)),
                       draw(st.lists(st.booleans(), min_size=n, max_size=n))))
    stairs = draw(st.lists(st.booleans(), min_size=len(floors) - 1, max_size=len(floors) - 1))
    return floors, stairs


def build(floors, stairs):
    """A building of floors of rooms in a row, doors between neighbours, leaks to ambient
    and stairs between floors. Returns it with the node and edge counts it must have."""
    b = Component("b")
    b.inner("ambient")
    placed = []
    for f, (n, doors, leaks) in enumerate(floors):
        fl = Component(f"floor{f}", template="floor")
        rooms = [fl.add(zone(f"r{i}", volume=10.0 + i)) for i in range(n)]
        for i, has in enumerate(doors):
            if has:
                d = fl.add(link(f"d{i}"))
                fl.connect(rooms[i].ports.air, d.ports.a)
                fl.connect(d.ports.b, rooms[i + 1].ports.air)
        for i, has in enumerate(leaks):
            if has:
                lk = fl.add(leak(f"leak{i}"))
                fl.connect(lk.ports.a, rooms[i].ports.air)
        fl.expose(stair=rooms[0].ports.air)
        placed.append(b.add(fl, at=(0.0, 0.0, 3.0 * f)))
    for f, has in enumerate(stairs):
        if has:
            s = b.add(link(f"stair{f}"))
            b.connect(placed[f].ports.stair, s.ports.a)
            b.connect(s.ports.b, placed[f + 1].ports.stair)
    n_nodes = 1 + sum(n for n, _, _ in floors)
    n_edges = sum(sum(d) + sum(lk) for _, d, lk in floors) + sum(stairs)
    return b, n_nodes, n_edges


@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(plans())
def test_random_buildings_flatten_consistently(plan):
    b, n_nodes, n_edges = build(*plan)
    net, names = b.flatten(node_elevations=("z_ref",))
    assert net.n == n_nodes and net.b == n_edges
    assert len(set(net.nodes)) == net.n
    labels = [net.graph.edges[e]["name"] for e in net.edges]
    assert len(set(labels)) == len(labels)
    assert all(c in net.nodes for c in net.aliases.values())
    assert not set(net.aliases) & set(net.nodes)
    for alias in net.aliases:
        assert names.resolve(alias) in net.nodes
    again, _ = b.flatten(node_elevations=("z_ref",))
    assert again.nodes == net.nodes and again.edges == net.edges
