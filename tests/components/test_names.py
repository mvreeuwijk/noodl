"""NameMap: nodes and edges by component, templates, boundary flows, the tree."""

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from noodl.components import Component
from tests.components.conftest import leak, link, zone
from tests.components.test_flatten import build, plans

F64 = torch.float64


def floor_plan():
    """b: ambient; floor f with rooms r1, r2 joined by door d, leak on r2; room x outside f
    joined to r1 by a stair s at the root."""
    b = Component("b")
    b.inner("ambient")
    f = b.add(Component("f", template="floor"))
    r1, r2 = f.add(zone("r1", volume=1.0)), f.add(zone("r2", volume=1.0))
    d = f.add(link("d"))
    f.connect(r1.ports.air, d.ports.a)
    f.connect(d.ports.b, r2.ports.air)
    lk = f.add(leak("lk"))
    f.connect(lk.ports.a, r2.ports.air)
    f.expose(stair=r1.ports.air)
    x = b.add(zone("x", volume=1.0))
    s = b.add(link("s"))
    b.connect(x.ports.air, s.ports.a)
    b.connect(s.ports.b, f.ports.stair)
    return b.flatten(dtype=F64)


def test_nodes_and_edges_of_a_subtree():
    _, names = floor_plan()
    assert names.nodes_of("f") == ["f.r1.air", "f.r2.air"]
    assert names.nodes_of("f.r1") == ["f.r1.air"]
    assert names.nodes_of("") == ["ambient", "f.r1.air", "f.r2.air", "x.air"]
    assert names.edges_of("f") == ["f.d.path", "f.lk.path"]
    assert names.edges_of("s") == ["s.path"]


def test_path_prefix_is_not_confused_with_a_sibling():
    b = Component("b")
    b.add(zone("f", volume=1.0))
    b.add(zone("f2", volume=1.0))
    _, names = b.flatten()
    assert names.nodes_of("f") == ["f.air"]


def test_unknown_path_suggests():
    _, names = floor_plan()
    with pytest.raises(KeyError, match=r"unknown component 'f.r3'.*'f.r1'"):
        names.nodes_of("f.r3")


def test_select_by_template_in_tree_order():
    _, names = floor_plan()
    assert names.select(template="zone") == ["f.r1", "f.r2", "x"]
    assert names.select(template="floor") == ["f"]
    assert names.select(template="nothing") == []


def test_select_then_nodes_of_gives_driver_labels():
    _, names = floor_plan()
    labels = [n for p in names.select(template="zone") for n in names.nodes_of(p)]
    assert labels == ["f.r1.air", "f.r2.air", "x.air"]


def test_boundary_edges_of_a_room_and_of_the_floor():
    _, names = floor_plan()
    room = names.boundary_edges("f.r2", "airpath")
    assert room.labels == ["f.d.path", "f.lk.path"]
    assert room.sign.tolist() == [1.0, 1.0]          # d: r1 -> r2, lk: ambient -> r2
    floor = names.boundary_edges("f", "airpath")
    assert floor.labels == ["f.lk.path", "s.path"]   # the door is internal to the floor
    assert floor.sign.tolist() == [1.0, 1.0]
    assert names.boundary_edges("", "airpath").labels == []


def test_boundary_flows_hand_computed_and_batched():
    net, names = floor_plan()
    # edge order: f.d.path, f.lk.path, s.path
    q = torch.tensor([[1.0, 2.0, 4.0], [0.5, 0.0, -1.0]], dtype=F64)
    assert names.boundary_flows("f.r1", "airpath", q).tolist() == [-1.0 + 4.0, -0.5 - 1.0]
    assert names.boundary_flows("f", "airpath", q).tolist() == [6.0, -1.0]
    assert names.boundary_flows("", "airpath", q).tolist() == [0.0, 0.0]


def test_children_boundary_flows_add_up_to_the_parents():
    net, names = floor_plan()
    q = torch.randn(net.b, dtype=F64, generator=torch.Generator().manual_seed(1))
    total = sum(names.boundary_flows(p, "airpath", q) for p in ("f.r1", "f.r2"))
    torch.testing.assert_close(total, names.boundary_flows("f", "airpath", q))


@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(plans(), st.integers(0, 2**31 - 1))
def test_boundary_flow_is_minus_the_net_outflow_of_the_subtree(plan, seed):
    b, _, n_edges = build(*plan)
    if n_edges == 0:
        return
    net, names = b.flatten(dtype=F64, node_elevations=("z_ref",))
    q = torch.randn(net.b, dtype=F64, generator=torch.Generator().manual_seed(seed))
    A = net.incidence("airpath")
    A = A.to_dense() if A.is_sparse else A
    outflow = A @ q
    for path in names.paths:
        idx = [net.node_index(n) for n in names.nodes_of(path)]
        expected = -outflow[idx].sum() if idx else torch.zeros((), dtype=F64)
        torch.testing.assert_close(names.boundary_flows(path, "airpath", q), expected)


def test_tree_describes_the_hierarchy():
    _, names = floor_plan()
    tree = names.tree()
    assert tree["path"] == "" and tree["name"] == "b"
    assert [c["path"] for c in tree["children"]] == ["f", "x", "s"]
    f = tree["children"][0]
    assert f["template"] == "floor"
    assert f["ports"] == {"stair": "f.r1.air"}
    assert f["nodes"] == [] and f["edges"] == []
    assert [c["path"] for c in f["children"]] == ["f.r1", "f.r2", "f.d", "f.lk"]
    assert f["children"][0]["nodes"] == ["f.r1.air"]
    assert f["children"][2]["edges"] == ["f.d.path"]
    assert tree["nodes"] == ["ambient"]


def test_unconnected_ports():
    b = Component("b")
    r = b.add(zone("r", volume=1.0))
    b.add(zone("lonely", volume=1.0))
    d = b.add(link("d"))
    b.connect(r.ports.air, d.ports.a)
    b.expose(out=r.ports.air)                         # root ports are never "unconnected"
    _, names = b.flatten()
    assert names.unconnected_ports == ["lonely.air", "d.b"]


def test_port_used_as_edge_endpoint_counts_as_connected():
    b = Component("b")
    r = b.add(zone("r", volume=1.0))
    b.add_node("n")
    b.add_edge(r.ports.air, "n", kind="airpath")
    _, names = b.flatten()
    assert names.unconnected_ports == []


def test_boundary_flows_rejects_a_wrong_length():
    _, names = floor_plan()
    with pytest.raises(ValueError, match=r"edge order of kind 'airpath'"):
        names.boundary_flows("f", "airpath", torch.zeros(2, 4, dtype=F64))
