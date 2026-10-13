"""Duct networks built in Python reproduce ContamX's duct numbers (probes c2a, c2h, c3b)."""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.building_physics import DuctNetwork, duct_elements_from_edges
from noodl.drives import Stack
from noodl.elements import FixedFlow, PowerLaw
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network

F64 = torch.float64
U = 2.0**-24
G = 9.8055
A20 = math.pi * 0.04 / 4


def rho(T):
    return 101325.0 / (287.055 * T)


def _system(fan_flow=0.05, T_j=(293.15,) * 4, z_j=(1.5,) * 4, T_amb=293.15):
    """c2a: an ambient terminal at junction j1, a 10 m duct to j2, a fan to j3, a 5 m duct
    to j4 and a terminal into the room; the room vents to the ambient through an orifice."""
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0, T0=T_amb)
    net.add_node("room", z_ref=0.0, T0=293.15, volume=300.0)
    ducts = DuctNetwork()
    for name, T, z in zip(("j1", "j2", "j3", "j4"), T_j, z_j, strict=True):
        ducts.junction(name, z_ref=z, T0=T)
    ducts.segment("j1", "j2", length=10.0, diameter=0.2)
    ducts.segment("j3", "j4", length=5.0, diameter=0.2)
    ducts.terminal("j1", "ambient", area=0.0314159)
    ducts.terminal("j4", "room", area=0.0314159)
    ducts.add_to(net)
    net.add_edge("j2", "j3", kind="fan")
    net.add_edge("room", "ambient", kind="vent", z_path=1.5)
    elements, drives = duct_elements_from_edges(net, g=G)
    elements += [FixedFlow(torch.tensor([fan_flow], dtype=F64), kind="fan"),
                 PowerLaw(torch.tensor([0.5], dtype=F64), torch.tensor([0.5], dtype=F64),
                          kind="vent")]
    drives.append(Stack.from_network(net, "vent", g=G))
    layer = PotentialFlowLayer(net, "air", elements, drives=drives, boundary=["ambient"])
    densities = rho(net.node_attr("T0"))
    phi, q = layer.solve(torch.zeros(1, dtype=F64), {"rho": densities}, atol=1e-14,
                         rtol=1e-14)
    return net, phi, q, densities


def test_a_junction_holds_half_of_every_segment_that_meets_it():
    ducts = DuctNetwork()
    for name in ("a", "b", "c"):
        ducts.junction(name, z_ref=0.0)
    ducts.segment("a", "b", length=10.0, diameter=0.2)
    ducts.segment("b", "c", length=1.0, diameter=0.2, area=0.05)
    v = ducts.volumes()
    assert v["a"] == pytest.approx(5 * A20, rel=1e-15)
    assert v["b"] == pytest.approx(5 * A20 + 0.025, rel=1e-15)
    assert v["c"] == pytest.approx(0.025, rel=1e-15)


def test_a_python_duct_system_gives_contamx_s_pressure_drops():
    """ContamX at 0.05 kg/s: 1.4536846 Pa over the 10 m duct, 0.7268423 Pa over the 5 m
    one, 0.1314798 Pa over each terminal (budget 32 u, as in the element tests)."""
    net, phi, q, d = _system()
    i = {n: net.node_index(n) for n in ("j1", "j2", "j3", "j4", "room", "ambient")}
    tol = 32 * U
    assert (phi[i["j1"]] - phi[i["j2"]]).item() == pytest.approx(1.4536845684051514, rel=tol)
    assert (phi[i["j3"]] - phi[i["j4"]]).item() == pytest.approx(0.7268422842025757, rel=tol)
    room_at_j4 = phi[i["room"]] - d[i["room"]] * G * 1.5
    assert (phi[i["j4"]] - room_at_j4).item() == pytest.approx(0.13147981464862823, rel=tol)
    ambient_at_j1 = -d[i["ambient"]] * G * 1.5
    assert (phi[i["j1"]] - ambient_at_j1).item() == pytest.approx(-0.13147981464862823,
                                                                  rel=tol)


def test_a_vertical_duct_carries_the_column_at_its_junctions_mean_temperature():
    """c2h: no flow (the fan stopped); j1 at 1.5 m and 273.15 K, j2 at 4.5 m and 313.15 K,
    the room at 293.15 K, the ambient at 273.15 K. ContamX: j2's gauge pressure against the
    room at its own height is 0 (the column is at 293.15 K)."""
    net, phi, q, d = _system(fan_flow=0.0, T_j=(273.15, 313.15, 313.15, 313.15),
                             z_j=(1.5, 4.5, 4.5, 1.5), T_amb=273.15)
    j2, room = net.node_index("j2"), net.node_index("room")
    gauge = phi[j2] - (phi[room] - d[room] * G * 4.5)
    assert abs(gauge.item()) < 1e-9


def test_a_duct_network_adds_to_a_component_too():
    from noodl.components import Component

    c = Component("plant")
    c.add_terminal("room")
    ducts = DuctNetwork()
    ducts.junction("j", z_ref=2.0)
    ducts.junction("k", z_ref=2.0)
    ducts.segment("j", "k", length=2.0, diameter=0.2)
    ducts.terminal("k", "room", area=0.03)
    ducts.terminal("j", Component.outer("ambient"), area=0.03, Ch=0.1225, azimuth=180.0,
                   profile=1)
    ducts.add_to(c)
    assert c.nodes == ["room", "j", "k"]
    with pytest.raises(KeyError, match="'x'"):
        ducts.segment("j", "x", length=1.0, diameter=0.2)


def test_a_duct_network_refuses_a_bad_description_by_name():
    ducts = DuctNetwork()
    ducts.junction("j", z_ref=0.0)
    with pytest.raises(ValueError, match="'j' is declared twice"):
        ducts.junction("j", z_ref=1.0)
    with pytest.raises(ValueError, match="to itself"):
        ducts.segment("j", "j", length=1.0, diameter=0.2)
    ducts.junction("k", z_ref=0.0)
    with pytest.raises(ValueError, match="length must be positive"):
        ducts.segment("j", "k", length=0.0, diameter=0.2)
    with pytest.raises(KeyError, match="'x'"):
        ducts.terminal("x", "room", area=0.03)


def test_a_terminal_with_a_wind_profile_gets_a_wind_drive():
    from noodl.drives import Wind, WindProfile

    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0, T0=293.15)
    net.add_node("room", z_ref=0.0, T0=293.15, volume=50.0)
    ducts = DuctNetwork()
    ducts.junction("j", z_ref=3.0)
    ducts.junction("k", z_ref=3.0)
    ducts.segment("j", "k", length=2.0, diameter=0.2)
    ducts.terminal("j", "ambient", area=0.03, Ch=0.5, azimuth=90.0, profile=1)
    ducts.terminal("k", "room", area=0.03)
    ducts.add_to(net)
    assert net.edge_attr("z_path", "duct_terminal").tolist() == [3.0, 3.0]
    assert net.node_attr("heat_capacity", default=-1.0).tolist() == [-1.0, -1.0, 0.0, 0.0]
    volumes = net.node_attr("volume", default=0.0).tolist()
    assert volumes == pytest.approx([0.0, 50.0, A20, A20], rel=1e-15)
    profile = WindProfile([0.0, 180.0], [0.6, -0.3])
    elements, drives = duct_elements_from_edges(net, profiles=[profile])
    assert [type(e).__name__ for e in elements] == ["AirDuct", "DuctTerminal"]
    assert [type(d).__name__ for d in drives] == ["DuctStack", "Stack", "Wind"]
    wind = drives[-1]
    assert isinstance(wind, Wind) and wind.ch.tolist() == [0.5, 1.0]
