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
