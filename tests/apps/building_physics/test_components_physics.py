"""A building made of components is the same model as the one built by hand, and obeys the
physics the hand-built one does: same network, same results to round-off, stack-driven flow
up a warm shaft, wind by facade, zero net inflow into every room."""

import math

import pytest
import torch

from benchmarks.natural_ventilation import build as benchmark_build
from noodl.apps.building_physics import (
    WallMass,
    Zone,
    add_large_opening,
    add_zone,
    build_model,
    crack,
    door,
    initial_drivers,
    initial_state,
    orifice_elements_from_edges,
    room,
    shaft,
)
from noodl.components import Component
from noodl.drives import Stack, Wind, WindProfile
from noodl.topology import Network

F64 = torch.float64
T_O = 283.15


def components_house(T_A=288.15, T_B=285.15, walls=False, door_open=True):
    def w(n):
        return WallMass("wall", capacity=2e6, ua_zone=40.0, ua_ambient=25.0) if walls else None
    b = Component("house")
    b.inner("ambient", z_ref=0.0, T0=T_O)
    A = b.add(room("A", volume=60.0, T0=T_A, wall=w("A")))
    B = b.add(room("B", volume=60.0, T0=T_B, wall=w("B")))
    d = b.add(door("door", H=2.0, W=0.9, z_mid=1.0, Cd=0.78))
    b.connect(A.ports.air, d.ports.a)
    if door_open:
        b.connect(d.ports.b, B.ports.air)
    for name, z in (("leak_low", 0.2), ("leak_high", 1.8)):
        lk = b.add(crack(name, area=0.05, Cd=0.6, z_path=z, exterior=True))
        b.connect(lk.ports.a, B.ports.air)
    return b


def hand_house(T_A=288.15, T_B=285.15, walls=False, door_open=True):
    def w(n):
        return WallMass(f"{n}_wall", capacity=2e6, ua_zone=40.0, ua_ambient=25.0) if walls else None
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0, T0=T_O)
    add_zone(net, Zone("A", volume=60.0, T0=T_A, wall=w("A")))
    add_zone(net, Zone("B", volume=60.0, T0=T_B, wall=w("B")))
    if door_open:
        add_large_opening(net, "A", "B", H=2.0, W=0.9, z_mid=1.0, Cd=0.78)
    net.add_edge("ambient", "B", kind="airpath", z_path=0.2, Cd=0.6, area=0.05)
    net.add_edge("ambient", "B", kind="airpath", z_path=1.8, Cd=0.6, area=0.05)
    return net


def model_of(net, coupling="iterate"):
    el = orifice_elements_from_edges(net, "airpath")
    return build_model(net, air_elements=[el], drives=[Stack.from_network(net, "airpath")],
                       coupling=coupling, iterate_tol={"thermal": 0.01}, iterate_max=50)


def run(model, sources_at, steps=10, dt=600.0):
    th = model.refs.thermal
    state = initial_state(model)
    drivers = initial_drivers(model, values={th.boundary_temperature: {"ambient": T_O},
                                             th.sources: {sources_at: 1000.0}})
    for _ in range(steps):
        state = model.step(state, drivers, dt)
    return state


def without(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}


@pytest.mark.parametrize("walls", [False, True])
def test_same_network_as_the_hand_built_one(walls):
    net, _ = components_house(walls=walls).flatten(dtype=F64)
    hand = hand_house(walls=walls)
    assert (net.n, net.b) == (hand.n, hand.b)
    assert net.edge_kinds() == hand.edge_kinds()
    torch.testing.assert_close(net.incidence(), hand.incidence())
    for mine, theirs in zip(net.nodes, hand.nodes, strict=True):
        assert without(net.graph.nodes[mine], "position") == dict(hand.graph.nodes[theirs])
    for mine, theirs in zip(net.edges, hand.edges, strict=True):
        assert without(net.graph.edges[mine], "name") == dict(hand.graph.edges[theirs])


@pytest.mark.parametrize("walls", [False, True])
@pytest.mark.parametrize("coupling", ["pingpong", "iterate"])
def test_same_results_as_the_hand_built_model(walls, coupling):
    net, _ = components_house(walls=walls).flatten(dtype=F64)
    mine = run(model_of(net, coupling), "A.air")
    theirs = run(model_of(hand_house(walls=walls), coupling), "A")
    assert mine.keys() == theirs.keys()
    for key in mine:
        torch.testing.assert_close(mine[key], theirs[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("coupling", ["pingpong", "iterate"])
def test_same_results_as_the_golden_benchmark(coupling):
    _, bench_model, bench_state, sources = benchmark_build(coupling)
    net, _ = components_house(T_A=T_O + 5.0, T_B=T_O + 2.0).flatten(dtype=F64)
    model = model_of(net, coupling)
    state = initial_state(model)
    for k in range(6):
        T_amb = T_O + 6.0 * math.sin(2.0 * math.pi * ((k + 1) * 600.0 - 9.0 * 3600.0) / 86400.0)
        drivers = {"air.phi_boundary": torch.zeros(1, dtype=F64),
                   "thermal.x_boundary": torch.tensor([T_amb], dtype=F64),
                   "thermal.sources": sources}
        state = model.step(state, drivers, 600.0)
        bench_state = bench_model.step(bench_state, drivers, 600.0)
    for key in bench_state:
        torch.testing.assert_close(state[key], bench_state[key], rtol=1e-12, atol=1e-12)


def test_drivers_by_alias_and_by_template():
    net, names = components_house().flatten(dtype=F64)
    model = model_of(net)
    th = model.refs.thermal
    by_alias = initial_drivers(model, values={th.sources: {"door.a": 1000.0}})
    by_name = initial_drivers(model, values={th.sources: {"A.air": 1000.0}})
    assert torch.equal(by_alias[th.sources], by_name[th.sources])
    rooms = {n: 100.0 for p in names.select(template="room") for n in names.nodes_of(p)}
    assert rooms == {"A.air": 100.0, "B.air": 100.0}
    every = initial_drivers(model, values={th.sources: rooms})[th.sources]
    assert every[net.node_index("A.air")] == every[net.node_index("B.air")] == 100.0


def test_air_mass_is_conserved_in_every_room_and_the_house():
    net, names = components_house().flatten(dtype=F64)
    model = model_of(net)
    state = run(model, "A.air", steps=3)
    q = state[model.refs.air.flow]
    for path in ("A", "B", "door", ""):
        torch.testing.assert_close(names.boundary_flows(path, "airpath", q),
                                   torch.zeros((), dtype=F64), atol=1e-8, rtol=0)
    # the doorway carries air both ways, so the boundary flow is a real cancellation:
    door_edges = names.boundary_edges("A", "airpath")
    assert door_edges.labels == ["door.low", "door.high"]
    low, high = q[door_edges.positions]
    assert low * high < 0


def test_a_door_with_one_side_unconnected_is_a_closed_door():
    net, names = components_house(door_open=False).flatten(dtype=F64)
    assert "door.low" not in [net.graph.edges[e]["name"] for e in net.edges]
    assert names.unconnected_ports == ["door.b"]
    # the closed door leaves room A with no air path, so (as in the hand-built model) it is
    # not a thermal unknown and cannot take a source: heat room B instead
    mine = run(model_of(net), "B.air")
    theirs = run(model_of(hand_house(door_open=False)), "B")
    for key in mine:
        torch.testing.assert_close(mine[key], theirs[key], rtol=1e-12, atol=1e-12)


def two_floors(levels_z=(0.0, 3.0)):
    b = Component("b")
    b.inner("ambient", z_ref=0.0, T0=T_O)
    for k, z in enumerate(levels_z):
        f = b.add(Component(f"floor{k}", template="floor"), at=(0.0, 0.0, z))
        A, B = f.add(room("A", volume=50.0)), f.add(room("B", volume=40.0))
        d = f.add(door("d", H=2.0, W=0.9, z_mid=1.0))
        f.connect(A.ports.air, d.ports.a)
        f.connect(d.ports.b, B.ports.air)
    return b


def test_floors_get_absolute_heights_as_a_hand_built_network_would():
    net, _ = two_floors().flatten(dtype=F64)
    assert net.nodes == ["ambient", "floor0.A.air", "floor0.B.air", "floor1.A.air",
                         "floor1.B.air"]
    stack = Stack.from_network(net, "airpath")
    torch.testing.assert_close(stack.z_ref,
                               torch.tensor([0.0, 0.0, 0.0, 3.0, 3.0], dtype=F64))
    expected = [1.0 - 4 / 9, 1.0 + 4 / 9, 4.0 - 4 / 9, 4.0 + 4 / 9]
    torch.testing.assert_close(stack.z_path, torch.tensor(expected, dtype=F64))


def test_warm_shaft_draws_air_upward():
    b = Component("tower")
    b.inner("ambient", z_ref=0.0, T0=T_O)
    s = b.add(shaft("stair", levels=3, level_height=3.0, volume=30.0, area=1.0, T0=T_O + 15.0))
    bottom = b.add(crack("bottom", area=0.1, z_path=0.5, exterior=True))
    top = b.add(crack("top", area=0.1, z_path=8.5, exterior=True))
    b.connect(bottom.ports.a, s.ports["levels[0]"])
    b.connect(top.ports.a, s.ports["levels[2]"])
    net, names = b.flatten(dtype=F64)
    model = model_of(net)
    state = run(model, "stair.levels[1]", steps=1, dt=60.0)
    q = dict(zip([net.graph.edges[e]["name"] for e in net.edges],
                 state[model.refs.air.flow].tolist(), strict=True))
    assert q["stair.slab[0]"] > 0 and q["stair.slab[1]"] > 0      # upward through the shaft
    assert q["bottom.path"] > 0                                    # ambient -> shaft at the bottom
    assert q["top.path"] < 0                                       # shaft -> ambient at the top


def test_wind_by_facade_equals_wind_by_azimuth():
    def house(**wind):
        b = Component("b")
        b.inner("ambient", z_ref=0.0, T0=T_O)
        b.inner_table("facades", south=180.0, north=0.0)
        r = b.add(room("r", volume=50.0))
        for name, w in wind.items():
            c = b.add(crack(name, area=0.05, z_path=1.0, exterior=True, **w))
            b.connect(c.ports.a, r.ports.air)
        return b.flatten(dtype=F64)[0]
    # A profile makes Cp depend on the angle between wind and facade (a constant Cp would
    # hide a wrong azimuth).
    profile = WindProfile([0.0, 90.0, 180.0, 270.0], [0.6, -0.3, -0.5, -0.3])
    by_facade = house(s=dict(facade="south"), n=dict(facade="north"))
    by_azimuth = house(s=dict(azimuth=180.0), n=dict(azimuth=0.0))
    w1 = Wind.from_network(by_facade, "airpath", ambient="ambient", profile=profile)
    w2 = Wind.from_network(by_azimuth, "airpath", ambient="ambient", profile=profile)
    torch.testing.assert_close(w1.azimuth, w2.azimuth)
    drivers = {"rho_amb": torch.tensor(1.2, dtype=F64), "V_met": torch.tensor(5.0, dtype=F64),
               "theta_w": torch.tensor(200.0, dtype=F64)}
    torch.testing.assert_close(w1(drivers), w2(drivers))
    assert not torch.equal(w1(drivers)[0], w1(drivers)[1])        # the two facades differ
