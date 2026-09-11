"""Shared fixtures: small typed networks in float64 for exact linear-algebra tests."""

from __future__ import annotations

import pytest
import torch

from tellegen.drives import ConstantDrive
from tellegen.elements import PowerLaw
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


@pytest.fixture
def two_zone_layer():
    """ambient (boundary) -> z1 -> z2 -> ambient, one PowerLaw airpath element.

    Edges (kind="airpath", insertion order): ambient->z1, z1->z2, z2->ambient.
    C = [0.01, 0.02, 0.01], n = 0.65. boundary = ["ambient"], interior = ["z1", "z2"].
    A ConstantDrive("airpath", "wind") is attached; its driver tensor must be shaped
    (..., 3) with the wind magnitude in column 0 (ambient->z1) and zero elsewhere.

    Returns (net, elements, drives, boundary).
    """
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z1")
    net.add_node("z2")
    net.add_edge("ambient", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient", kind="airpath")
    elements = [PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=torch.float64), 0.65)]
    drives = [ConstantDrive("airpath", "wind")]
    boundary = ["ambient"]
    return net, elements, drives, boundary
