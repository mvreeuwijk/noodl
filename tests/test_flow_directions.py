import json
from pathlib import Path

import torch

from noodl.cycles import flow_direction_count
from noodl.topology import Network

GOLD = json.loads((Path(__file__).parent / "golden" / "legacy" / "tutorial.json").read_text())


def _from_edges(edges):
    net = Network(dtype=torch.float64)
    for n in sorted({u for e in edges for u in e}):
        net.add_node(n)
    for u, v in edges:
        net.add_edge(u, v, kind="flow")
    return net


def test_triangle_admits_two_flow_directions():
    assert flow_direction_count(_from_edges([(0, 1), (1, 2), (2, 0)])) == 2
    assert GOLD["partitions_triangle"] == 2


def test_six_edge_graph_matches_the_legacy_count():
    net = _from_edges([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)])
    assert flow_direction_count(net) == GOLD["partitions_six_edge_graph"]


def test_a_tree_admits_exactly_one_direction_pattern_up_to_sign():
    """No cycle: the current space is {0}; Winder's count of regions is 1."""
    assert flow_direction_count(_from_edges([(0, 1), (1, 2), (1, 3)])) == 1


def test_triangle_with_a_pendant_edge_still_admits_two_flow_directions():
    """The pendant edge (2, 3) carries no cycle flow: an all-zero column of the cycle
    basis. It must be dropped before Winder's enumeration, or it poisons the alternating
    sum (the un-fixed code returned 0 here instead of 2)."""
    net = _from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])
    assert flow_direction_count(net) == 2


def test_four_cycle_with_a_pendant_edge_still_admits_two_flow_directions():
    net = _from_edges([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)])
    assert flow_direction_count(net) == 2


def test_two_triangles_joined_by_a_bridge_admit_four_flow_directions():
    """Two independent cycles sharing only a bridge edge: the cycle space is
    2-dimensional and the two families of hyperplanes (one per triangle) are orthogonal
    coordinate planes, so the count is the product of each triangle's own count: 2 x 2
    = 4 regions. The bridge edge (2, 3) itself carries no cycle flow and must be
    dropped."""
    net = _from_edges([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)])
    assert flow_direction_count(net) == 4
