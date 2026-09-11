"""Shared fixtures: small typed networks in float64 for exact linear-algebra tests."""

from __future__ import annotations

import pytest
import torch

from tellegen.topology import Network


@pytest.fixture
def triangle() -> Network:
    """Three nodes, one directed loop: a->b->c->a, all edges kind "airpath", float64."""
    net = Network(dtype=torch.float64)
    for name in ("a", "b", "c"):
        net.add_node(name)
    net.add_edge("a", "b", kind="airpath")
    net.add_edge("b", "c", kind="airpath")
    net.add_edge("c", "a", kind="airpath")
    return net


@pytest.fixture
def two_zone() -> Network:
    """Ambient plus two zones in series: ambient->z1->z2->ambient, kind "airpath", float64."""
    net = Network(dtype=torch.float64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name)
    net.add_edge("ambient", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient", kind="airpath")
    return net
