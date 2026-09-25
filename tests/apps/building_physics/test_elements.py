"""Building element helpers: mass-flow orifice and CONTAM's two-opening doorway."""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.building_physics.elements import (
    add_large_opening,
    mass_orifice,
    orifice_elements_from_edges,
)
from noodl.drives import Stack
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network

F64 = torch.float64
G = 9.80665


def test_mass_orifice_is_a_power_law_with_the_mass_flow_coefficient():
    el = mass_orifice(0.6, 0.01, rho=1.2)
    assert el.n.item() == pytest.approx(0.5)
    assert el.C.item() == pytest.approx(0.6 * 0.01 * math.sqrt(2.0 * 1.2))
    F = el.flow(torch.tensor([4.0], dtype=F64))
    assert F.item() == pytest.approx(0.6 * 0.01 * math.sqrt(2.0 * 1.2 * 4.0))


def test_two_orifice_doorway_reproduces_brown_solvason_with_the_neutral_plane_at_mid_height():
    """Both rooms at the same reference pressure at the doorway mid-height: exchange flow
    w = (Cd/3) W sqrt(rho g drho H^3) each way (TN 1887r1 eq. 69-70)."""
    net = Network(dtype=F64)
    net.add_node("cold", z_ref=1.0)
    net.add_node("warm", z_ref=1.0)
    lo, hi = add_large_opening(net, "cold", "warm", H=2.0, W=0.8, z_mid=1.0, Cd=0.78)
    assert net.graph.edges[lo]["z_path"] == pytest.approx(1.0 - 4.0 / 9.0)
    assert net.graph.edges[hi]["z_path"] == pytest.approx(1.0 + 4.0 / 9.0)
    rho = 1.2
    el = orifice_elements_from_edges(net, "airpath", rho=rho)
    layer = PotentialFlowLayer(
        net, "air", [el], drives=[Stack.from_network(net, "airpath")],
        boundary=["cold", "warm"],
    )
    densities = torch.tensor([1.25, 1.15], dtype=F64)
    q = layer.flows(torch.zeros(2, dtype=F64), {"rho": densities})
    w = 0.78 / 3.0 * 0.8 * math.sqrt(rho * G * (1.25 - 1.15) * 2.0**3)
    assert q[0].item() == pytest.approx(w, rel=1e-12)    # low opening: cold -> warm
    assert q[1].item() == pytest.approx(-w, rel=1e-12)   # high opening: warm -> cold


def test_add_large_opening_validates_geometry_and_records_area_and_cd():
    net = Network(dtype=F64)
    net.add_node("a")
    net.add_node("b")
    with pytest.raises(ValueError, match="large opening.*H"):
        add_large_opening(net, "a", "b", H=0.0, W=0.8, z_mid=1.0)
    lo, _ = add_large_opening(net, "a", "b", H=2.0, W=0.8, z_mid=1.0)
    assert net.graph.edges[lo]["area"] == pytest.approx(0.8)
    assert net.graph.edges[lo]["Cd"] == pytest.approx(0.78)
