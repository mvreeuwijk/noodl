"""CONTAM's simple air-handling system, against ContamX 3.4.1.7 (probes c1g-c1n)."""

from __future__ import annotations

import pytest
import torch

from noodl.apps.building_physics import (
    EXHAUST,
    OUTDOOR_AIR,
    RECIRCULATION,
    RETURN,
    SUPPLY,
    AirHandlerFlows,
    add_air_handler,
    air_handler_from_edges,
    air_handler_plenums,
    build_model,
    initial_drivers,
    initial_state,
    orifice_elements_from_edges,
)
from noodl.drives import Stack
from noodl.topology import Network

F64 = torch.float64
ROLES = torch.tensor([SUPPLY, SUPPLY, RETURN, RECIRCULATION, OUTDOOR_AIR, EXHAUST])


def _flows(fo=None, min_oa=0.0, multipliers=None, ret=0.12):
    closure = AirHandlerFlows("ahs", roles=ROLES,
                              design_flows=torch.tensor([0.1, 0.05, ret, 0, 0, 0], dtype=F64),
                              min_outdoor_air=min_oa, name="ahs")
    d = {}
    if fo is not None:
        d["outdoor_air_fraction.ahs"] = torch.tensor(fo, dtype=F64)
    if multipliers is not None:
        d["flow_multiplier.ahs"] = torch.tensor(multipliers, dtype=F64)
    out = closure({}, d)["prescribed_flow.ahs"]
    return [round(v, 9) for v in out.tolist()]


@pytest.mark.parametrize("case, kw, contamx", [
    ("no schedule: all outdoor air", {}, (0.0, 0.15, 0.12)),
    ("f_o 0.3", dict(fo=0.3), (0.105, 0.045, 0.015)),
    ("f_o 0.3, minimum 0.08", dict(fo=0.3, min_oa=0.08), (0.07, 0.08, 0.05)),
    ("supply 1 halved", dict(fo=0.3, min_oa=0.08, multipliers=[0.5, 1, 1, 1, 1, 1]),
     (0.02, 0.08, 0.10)),
    ("f_o 0, minimum above the supply", dict(fo=0.0, min_oa=0.2), (0.0, 0.15, 0.12)),
    ("f_o 0, minimum 0.05", dict(fo=0.0, min_oa=0.05), (0.1, 0.05, 0.02)),
    ("returns above the supply", dict(fo=0.3, ret=0.3), (0.105, 0.045, 0.195)),
    ("f_o below 0", dict(fo=-0.2), (0.12, 0.03, 0.0)),
    ("supplies off", dict(fo=0.3, min_oa=0.05, multipliers=[0, 0, 1, 1, 1, 1]),
     (0.0, 0.0, 0.12)),
])
def test_the_air_handler_flows_are_contamx_s(case, kw, contamx):
    """(recirculation, outdoor air, exhaust) as ContamX 3.4.1.7 reported them, exact."""
    out = _flows(**kw)
    assert tuple(out[3:]) == pytest.approx(contamx, abs=1e-12), case


def test_an_outdoor_air_fraction_above_one_is_refused_naming_the_handler():
    """ContamX stops ('Invalid recirculation air flow -0.075') at f_o = 1.5."""
    with pytest.raises(ValueError, match=r"'ahs'.*outdoor-air fraction.*recirculation"):
        _flows(fo=1.5)


def test_batched_multipliers_and_fractions():
    closure = AirHandlerFlows("ahs", roles=ROLES,
                              design_flows=torch.tensor([0.1, 0.05, 0.12, 0, 0, 0], dtype=F64))
    d = {"outdoor_air_fraction.ahs": torch.tensor([0.3, 1.0], dtype=F64),
         "flow_multiplier.ahs": torch.ones(2, 6, dtype=F64)}
    out = closure({}, d)["prescribed_flow.ahs"]
    assert out.shape == (2, 6)
    torch.testing.assert_close(out[:, 3:], torch.tensor([[0.105, 0.045, 0.015],
                                                         [0.0, 0.15, 0.12]], dtype=F64))


M_ROOM = 300.0 * 1.2041
DT = 300.0


def _house(fo=0.3, supply=0.1, plenum_volume=0.0):
    """One room, a leak to the ambient, an air handler supplying and returning `supply`."""
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0, T0=293.15)
    net.add_node("room", z_ref=0.0, T0=293.15, volume=300.0)
    net.add_edge("ambient", "room", kind="airpath", z_path=1.0, Cd=0.6, area=0.01)
    add_air_handler(net, supplies={"room": supply}, returns={"room": supply},
                    outdoor_air_fraction=fo, supply_volume=plenum_volume,
                    return_volume=plenum_volume)
    element, closure = air_handler_from_edges(net, "air_handler")
    plenums = air_handler_plenums(net, "air_handler")
    free = plenums if plenum_volume == 0.0 else ()
    model = build_model(net, air_elements=[orifice_elements_from_edges(net, "airpath"),
                                           element],
                        drives=[Stack.from_network(net, "airpath")], thermal=False,
                        species=1, closures=[closure], flow_balanced=plenums,
                        storage_free=free)
    state = initial_state(model)
    drivers = initial_drivers(model)
    drivers["rho"] = torch.full((net.n,), 1.2041, dtype=F64)
    drivers["species.x_boundary"] = torch.tensor([1e-3], dtype=F64)
    return net, model, state, drivers


def test_an_air_handler_ventilates_a_room_as_contamx_mixes_it():
    """f_o 0.3 at 0.1 kg/s: 0.03 kg/s outdoor air, 0.07 recirculated through zero-volume
    plenums, so the room sees M/dt (x - x0) = 0.03 (x_amb - x)."""
    net, model, state, drivers = _house()
    out = model.step(state, drivers, DT)
    room = net.node_index("room")
    layer = model.transport["species"]
    x = out["species.x"]
    row = layer.interior_idx.tolist().index(room)
    expected = 0.03 * 1e-3 / (M_ROOM / DT + 0.03)
    assert x[row].item() == pytest.approx(expected, rel=1e-10)
    supply, ret = air_handler_plenums(net, "air_handler")
    s_row = layer.interior_idx.tolist().index(net.node_index(supply))
    assert x[s_row].item() == pytest.approx(0.3 * 1e-3 + 0.7 * expected, rel=1e-10)


def test_a_switched_off_air_handler_keeps_its_plenums_and_steps():
    net, model, state, drivers = _house()
    state = model.step(state, drivers, DT)
    before = state["species.x"].clone()
    drivers["flow_multiplier.air_handler"] = torch.zeros(5, dtype=F64)
    after = model.step(state, drivers, DT)["species.x"]
    layer = model.transport["species"]
    for plenum in air_handler_plenums(net, "air_handler"):
        row = layer.interior_idx.tolist().index(net.node_index(plenum))
        assert after[row].item() == pytest.approx(before[row].item(), rel=1e-14)


def test_plenums_with_volume_mix_as_well_mixed_nodes():
    """V = 2 m3 plenums store air (rho V): one implicit-Euler step of the three balances
    room, supply plenum and return plenum, with the closure's flows (Ws = Wret = 0.1,
    Wrec = 0.07, Wo = Wex = 0.03; the leak carries nothing)."""
    net, model, state, drivers = _house(plenum_volume=2.0)
    out = model.step(state, drivers, DT)["species.x"]
    m_p = 2.0 * 1.2041
    # unknowns (room, supply, return); x_amb = 1e-3
    A = torch.tensor([[M_ROOM / DT + 0.1, -0.1, 0.0],
                      [0.0, m_p / DT + 0.1, -0.07],
                      [-0.1, 0.0, m_p / DT + 0.1]], dtype=F64)
    b = torch.tensor([0.0, 0.03 * 1e-3, 0.0], dtype=F64)
    expected = torch.linalg.solve(A, b)
    layer = model.transport["species"]
    rows = [layer.interior_idx.tolist().index(net.node_index(n))
            for n in ("room", *air_handler_plenums(net, "air_handler"))]
    torch.testing.assert_close(out[rows], expected, rtol=1e-10, atol=0.0)


def test_a_batched_multiplier_steps_on_and_off_air_handlers_together():
    net, model, state, drivers = _house()
    drivers["flow_multiplier.air_handler"] = torch.stack(
        [torch.ones(5, dtype=F64), torch.zeros(5, dtype=F64)])
    x = model.step(state, drivers, DT)["species.x"]
    assert x.shape[0] == 2
    room = model.transport["species"].interior_idx.tolist().index(net.node_index("room"))
    assert x[0, room].item() == pytest.approx(0.03 * 1e-3 / (M_ROOM / DT + 0.03), rel=1e-10)
    assert x[1, room].item() == 0.0


def test_zero_volume_plenums_carry_heat_algebraically():
    """With heat: the supply plenum is at the mix of the outdoor air and the returned room
    air, 0.3 T_amb + 0.7 T_room, after the step."""
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0, T0=283.15)
    net.add_node("room", z_ref=0.0, T0=293.15, volume=300.0)
    net.add_edge("ambient", "room", kind="airpath", z_path=1.0, Cd=0.6, area=0.01)
    add_air_handler(net, supplies={"room": 0.1}, returns={"room": 0.1},
                    outdoor_air_fraction=0.3)
    element, closure = air_handler_from_edges(net, "air_handler")
    plenums = air_handler_plenums(net, "air_handler")
    model = build_model(net, air_elements=[orifice_elements_from_edges(net, "airpath"),
                                           element],
                        drives=[Stack.from_network(net, "airpath")], closures=[closure],
                        flow_balanced=plenums, storage_free=plenums)
    assert model.transport["thermal"].scheme == "implicit"
    T = model.step(initial_state(model), initial_drivers(model), DT)["thermal.x"]
    rows = model.transport["thermal"].interior_idx.tolist()
    T_room, T_supply, T_return = (T[rows.index(net.node_index(n))].item()
                                  for n in ("room", *plenums))
    assert T_return == pytest.approx(T_room, rel=1e-12)
    assert T_supply == pytest.approx(0.3 * 283.15 + 0.7 * T_room, rel=1e-12)
    assert 283.15 < T_room < 293.15


def test_an_air_handler_in_a_component_reaches_the_shared_ambient():
    from noodl.components import Component

    building = Component("building")
    building.inner("ambient", z_ref=0.0, T0=293.15)
    building.add_node("office", z_ref=0.0, T0=293.15, volume=300.0)
    add_air_handler(building, supplies={"office": 0.1}, returns={"office": 0.1},
                    outdoor_air_fraction=0.3, min_outdoor_air=0.05, supply_volume=2.0)
    net, _ = building.flatten(dtype=F64)
    _, closure = air_handler_from_edges(net, "air_handler")
    assert (closure.outdoor_air_fraction, closure.min_outdoor_air) == (0.3, 0.05)
    supply, ret = air_handler_plenums(net, "air_handler")
    assert net.graph.nodes[supply]["volume"] == 2.0
    src, tgt = net.endpoints("air_handler")
    roles = net.edge_attr("role", "air_handler").round().to(torch.long)
    ambient = net.node_index("ambient")
    assert int(src[roles == OUTDOOR_AIR]) == ambient == int(tgt[roles == EXHAUST])
    assert int(tgt[roles == SUPPLY]) == net.node_index("office") == int(src[roles == RETURN])


def test_a_closure_without_its_three_plant_edges_is_refused():
    with pytest.raises(ValueError, match="'ahs': needs exactly one exhaust edge, got 0"):
        AirHandlerFlows("ahs", roles=[SUPPLY, RETURN, RECIRCULATION, OUTDOOR_AIR],
                        design_flows=[0.1, 0.1, 0.0, 0.0])
    with pytest.raises(ValueError, match="'ahs': design_flows must give one value per edge"):
        AirHandlerFlows("ahs", roles=ROLES, design_flows=[0.1, 0.1])


def test_the_closure_declares_its_inputs():
    net, model, _, _ = _house()
    inputs = model.refs.inputs
    assert "flow_multiplier.air_handler" in inputs
    assert "outdoor_air_fraction.air_handler" in inputs


def test_the_flows_are_differentiable_in_the_fraction_and_the_multipliers():
    closure = AirHandlerFlows("ahs", roles=ROLES,
                              design_flows=torch.tensor([0.1, 0.05, 0.12, 0, 0, 0], dtype=F64),
                              min_outdoor_air=0.02)

    def flows(fo, m):
        return closure({}, {"outdoor_air_fraction.ahs": fo, "flow_multiplier.ahs": m})[
            "prescribed_flow.ahs"]

    fo = torch.tensor(0.3, dtype=F64, requires_grad=True)
    m = torch.tensor([0.9, 1.1, 0.8, 1, 1, 1], dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(flows, (fo, m))


def test_a_pressure_dependent_edge_on_a_flow_balanced_plenum_is_refused():
    """A plenum pinned at zero gauge pressure would absorb a leak's imbalance silently."""
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0, T0=293.15)
    net.add_node("room", z_ref=0.0, T0=293.15, volume=300.0)
    net.add_edge("ambient", "room", kind="airpath", z_path=1.0, Cd=0.6, area=0.01)
    add_air_handler(net, supplies={"room": 0.1}, returns={"room": 0.05})
    net.add_edge("room", "ahs_supply", kind="airpath", z_path=1.0, Cd=0.6, area=0.01)
    element, closure = air_handler_from_edges(net, "air_handler")
    plenums = air_handler_plenums(net, "air_handler")
    with pytest.raises(ValueError, match=r"flow_balanced node 'ahs_supply' is an end of the "
                                         r"edge 'room' -> 'ahs_supply' of kind 'airpath'"):
        build_model(net, air_elements=[orifice_elements_from_edges(net, "airpath"), element],
                    drives=[Stack.from_network(net, "airpath")], thermal=False, species=1,
                    closures=[closure], flow_balanced=plenums, storage_free=plenums)


def test_a_fixed_flow_on_a_flow_balanced_plenum_is_refused():
    """A fixed flow is dp-independent too, but it is not part of the air handler's balance:
    only the handler's prescribed flows may touch its plenums."""
    from noodl.elements import FixedFlow

    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0, T0=293.15)
    net.add_node("room", z_ref=0.0, T0=293.15, volume=300.0)
    net.add_edge("ambient", "room", kind="airpath", z_path=1.0, Cd=0.6, area=0.01)
    add_air_handler(net, supplies={"room": 0.1}, returns={"room": 0.1})
    net.add_edge("ambient", "ahs_supply", kind="fan", z_path=1.0)
    element, closure = air_handler_from_edges(net, "air_handler")
    plenums = air_handler_plenums(net, "air_handler")
    with pytest.raises(ValueError, match=r"flow_balanced node 'ahs_supply' is an end of the "
                                         r"edge 'ambient' -> 'ahs_supply' of kind 'fan'.*"
                                         r"FixedFlow"):
        build_model(net, air_elements=[orifice_elements_from_edges(net, "airpath"),
                                       FixedFlow(torch.tensor(0.01, dtype=F64), kind="fan"),
                                       element],
                    drives=[Stack.from_network(net, "airpath")], thermal=False, species=1,
                    closures=[closure], flow_balanced=plenums, storage_free=plenums)


@pytest.mark.parametrize("roles, message", [
    ([0, 1, 2, 3, 4, 7], r"unknown role codes \[7\] at edge positions \[5\]"),
    ([0, 1, 2.4, 3, 4], r"role codes must be integers 0..4, got \[2.4"),
    ([0, -1, 2, 3, 4], r"unknown role codes \[-1\]"),
])
def test_unknown_or_fractional_role_codes_are_refused(roles, message):
    with pytest.raises(ValueError, match=message):
        AirHandlerFlows("ahs", roles=torch.tensor(roles, dtype=F64),
                        design_flows=[0.1] * len(roles))


def test_the_edge_readers_refuse_bad_roles_and_a_missing_recirculation():
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    net.add_node("room", z_ref=0.0, volume=300.0)
    add_air_handler(net, supplies={"room": 0.1}, returns={"room": 0.1})
    rec = [e for e in net.edges if net.graph.edges[e]["role"] == RECIRCULATION][0]
    net.graph.edges[rec]["role"] = 2.4
    with pytest.raises(ValueError, match=r"'air_handler': role codes must be integers"):
        air_handler_from_edges(net, "air_handler")
    net.graph.edges[rec]["role"] = float(SUPPLY)
    with pytest.raises(ValueError, match=r"'air_handler': needs exactly one recirculation"):
        air_handler_from_edges(net, "air_handler")
    with pytest.raises(ValueError, match=r"'air_handler': its plenums are the ends of its "
                                         r"one recirculation edge \(role 2\), but it has 0"):
        air_handler_plenums(net, "air_handler")


def test_a_negative_point_is_refused_naming_it_not_the_fraction():
    closure = AirHandlerFlows("ahs", roles=ROLES,
                              design_flows=torch.tensor([0.1, 0.05, 0.12, 0, 0, 0], dtype=F64))
    d = {"flow_multiplier.ahs": torch.tensor([[1, 1, 1, 1, 1, 1], [1, -1, 1, 1, 1, 1]],
                                             dtype=F64),
         "outdoor_air_fraction.ahs": torch.tensor(0.3, dtype=F64)}
    with pytest.raises(ValueError, match=r"'ahs': supply or return points \[1\] .* negative"):
        closure({}, d)
    d["flow_multiplier.ahs"] = torch.tensor([1, 1, -1, 1, 1, 1], dtype=F64)
    with pytest.raises(ValueError, match=r"supply or return points \[2\]"):
        closure({}, d)
