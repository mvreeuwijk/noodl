"""The HVAC component factories: a fan, a duct run and an air handler placed in a building
flatten to the network the builders make, and the model built from it moves the air they
prescribe."""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.building_physics import (
    air_handler,
    air_handler_from_edges,
    air_handler_plenums,
    build_model,
    crack,
    duct,
    duct_laws_from_edges,
    fan,
    fan_law_from_edges,
    initial_drivers,
    initial_state,
    orifice_law_from_edges,
    room,
)
from noodl.components import Component, ComponentError
from noodl.drives import Stack

F64 = torch.float64


def _building():
    b = Component("building")
    b.inner("ambient", z_ref=0.0, T0=293.15)
    office = b.add(room("office", volume=300.0))
    store = b.add(room("store", volume=100.0))
    ahu = b.add(air_handler("ahu", supplies=[0.1], returns=[0.1], outdoor_air_fraction=0.3,
                            supply_volume=0.0, return_volume=0.0))
    b.connect(ahu.ports["supplies[0]"], office.ports.air)
    b.connect(ahu.ports["returns[0]"], office.ports.air)
    transfer = b.add(duct("transfer", length=4.0, diameter=0.15, z_a=2.5, z_b=2.5))
    b.connect(transfer.ports.a, office.ports.air)
    b.connect(transfer.ports.b, store.ports.air)
    exhaust = b.add(fan("exhaust", flow=0.02, exterior="exhaust"))
    b.connect(exhaust.ports.a, store.ports.air)
    leak = b.add(crack("leak", area=0.01, z_path=1.0, exterior=True))
    b.connect(leak.ports.a, office.ports.air)
    return b


def _model(net):
    element, closure = air_handler_from_edges(net, "air_handler")
    plenums = air_handler_plenums(net, "air_handler")
    ducts, drives = duct_laws_from_edges(net)
    elements = [orifice_law_from_edges(net, "airpath"), *ducts,
                fan_law_from_edges(net), element]
    model = build_model(net, air_laws=elements,
                        drives=[Stack.from_network(net, "airpath"), *drives],
                        thermal=False, species=1, closures=[closure], flow_balanced=plenums,
                        storage_free=plenums)
    drivers = initial_drivers(model)
    drivers["rho"] = torch.full((net.n,), 1.2041, dtype=F64)
    return model, initial_state(model), drivers


def test_the_factories_flatten_to_junctions_plenums_and_their_edges():
    net, _ = _building().flatten(dtype=F64)
    assert {"transfer.ends[0]", "transfer.ends[1]", "ahu.supply", "ahu.return"} <= set(
        net.nodes)
    volume = net.node_attr("volume", default=0.0)
    half = 0.5 * 4.0 * math.pi * 0.15**2 / 4
    assert volume[net.node_index("transfer.ends[0]")].item() == pytest.approx(half)
    kinds = net.edge_kinds()
    assert kinds.count("duct") == 1 and kinds.count("duct_terminal") == 2
    assert kinds.count("fan") == 1 and kinds.count("air_handler") == 5
    assert air_handler_plenums(net, "air_handler") == ("ahu.supply", "ahu.return")


def test_the_building_moves_the_air_its_components_prescribe():
    net, _ = _building().flatten(dtype=F64)
    model, state, drivers = _model(net)
    out = model.steady(state, drivers, atol=1e-14, rtol=1e-14)
    air = model.potential["air"]
    torch.testing.assert_close(air.flows_of_kind(out["air.q"], "duct_terminal"),
                               torch.tensor([-0.02, 0.02], dtype=F64), rtol=1e-10, atol=0)
    torch.testing.assert_close(air.flows_of_kind(out["air.q"], "air_handler"),
                               torch.tensor([0.1, 0.1, 0.07, 0.03, 0.03], dtype=F64),
                               rtol=1e-12, atol=0)
    assert air.flows_of_kind(out["air.q"], "airpath").item() == pytest.approx(0.02,
                                                                              rel=1e-10)


def test_factory_arguments_that_make_no_sense_are_refused():
    with pytest.raises(ComponentError, match="exterior"):
        fan("f", flow=0.1, exterior="sideways")
    with pytest.raises(ComponentError, match="wind"):
        duct("d", length=1.0, diameter=0.2, azimuth=90.0)
    with pytest.raises(ComponentError, match="at least one supply"):
        air_handler("a", supplies=[], returns=[0.1])


def test_exterior_fan_and_duct_open_to_the_shared_ambient():
    b = Component("building")
    b.inner("ambient", z_ref=0.0, T0=293.15)
    hall = b.add(room("hall", volume=50.0))
    supply = b.add(fan("supply", flow=0.01, exterior="supply"))
    b.connect(supply.ports.a, hall.ports.air)
    intake = b.add(duct("intake", length=2.0, diameter=0.1, exterior=True, profile=1,
                        azimuth=90.0, Ch=0.8))
    b.connect(intake.ports.b, hall.ports.air)
    net, _ = b.flatten(dtype=F64)
    src, tgt = net.endpoints("fan")
    assert (net.nodes[int(src[0])], net.nodes[int(tgt[0])]) == ("ambient", "hall.air")
    src, tgt = net.endpoints("duct_terminal")
    ends = sorted((net.nodes[int(a)], net.nodes[int(b)]) for a, b in zip(src, tgt, strict=True))
    assert ends == [("intake.ends[0]", "ambient"), ("intake.ends[1]", "hall.air")]
    wind = net.edge_attr("profile", "duct_terminal", default=0.0).tolist()
    assert sorted(wind) == [0.0, 1.0]
    outdoor = [i for i, a in enumerate(src) if net.nodes[int(a)] == "intake.ends[0]"][0]
    assert net.edge_attr("azimuth", "duct_terminal", default=0.0).tolist()[outdoor] == 90.0
    assert net.edge_attr("Ch", "duct_terminal", default=1.0).tolist()[outdoor] == 0.8
    assert net.edge_attr("profile", "duct_terminal", default=0.0).tolist()[outdoor] == 1.0


@pytest.mark.parametrize("kwargs, text", [
    (dict(diameter=-0.1), "diameter"),
    (dict(diameter=0.1, roughness=0.5), "roughness"),
    (dict(diameter=0.1, exterior=True, azimuth=90.0), "azimuth"),
    (dict(diameter=0.1, loss=-1.0), "loss"),
])
def test_a_bad_duct_is_refused_naming_the_duct(kwargs, text):
    with pytest.raises(ComponentError, match=rf"^run7: .*{text}"):
        duct("run7", length=3.0, **kwargs)


@pytest.mark.parametrize("make, text", [
    (lambda: fan("f9", flow=float("nan")), "flow"),
    (lambda: fan("f9", flow=-0.1), "flow"),
    (lambda: air_handler("h9", supplies=[0.1], returns=[]), "return"),
    (lambda: air_handler("h9", supplies=[0.1], returns=[-0.1]), r"returns\[0\]"),
    (lambda: air_handler("h9", supplies=[float("inf")], returns=[0.1]), r"supplies\[0\]"),
    (lambda: air_handler("h9", supplies=[0.1], returns=[0.1], outdoor_air_fraction=1.5),
     "outdoor_air_fraction"),
    (lambda: air_handler("h9", supplies=[0.1], returns=[0.1], min_outdoor_air=-1.0),
     "min_outdoor_air"),
    (lambda: air_handler("h9", supplies=[0.1], returns=[0.1], supply_volume=-1.0),
     "supply_volume"),
    (lambda: air_handler("h9", supplies=[0.1], returns=[0.1], return_volume=-1.0),
     "return_volume"),
])
def test_a_bad_fan_or_air_handler_is_refused_naming_the_argument(make, text):
    with pytest.raises(ComponentError, match=rf"^(f9|h9): .*{text}"):
        make()


def test_components_placed_at_a_height_carry_it_into_the_network():
    b = Component("building")
    b.inner("ambient", z_ref=0.0, T0=293.15)
    at = (0.0, 0.0, 6.0)
    flat = b.add(room("flat", volume=60.0), at=at)
    run = b.add(duct("run", length=3.0, diameter=0.1, z_a=1.0, z_b=2.0), at=at)
    ahu = b.add(air_handler("ahu", supplies=[0.1], returns=[0.1]), at=at)
    b.connect(run.ports.a, flat.ports.air)
    b.connect(ahu.ports["supplies[0]"], flat.ports.air)
    b.connect(ahu.ports["returns[0]"], flat.ports.air)
    other = b.add(room("other", volume=60.0), at=at)
    b.connect(run.ports.b, other.ports.air)
    net, _ = b.flatten(dtype=F64)
    z = net.node_attr("z_ref", default=0.0)
    assert z[net.node_index("run.ends[0]")].item() == pytest.approx(7.0)
    assert z[net.node_index("run.ends[1]")].item() == pytest.approx(8.0)
    assert z[net.node_index("ahu.supply")].item() == pytest.approx(6.0)
    assert sorted(net.edge_attr("z_path", "duct_terminal").tolist()) == [7.0, 8.0]


def test_a_pressure_dependent_edge_on_a_plenum_is_refused_by_name():
    net, _ = _building().flatten(dtype=F64)
    net.add_edge("ahu.supply", "store.air", kind="airpath", z_path=1.0, Cd=0.6, area=0.01)
    element, closure = air_handler_from_edges(net, "air_handler")
    plenums = air_handler_plenums(net, "air_handler")
    with pytest.raises(ValueError, match="ahu.supply"):
        build_model(net, air_laws=[orifice_law_from_edges(net, "airpath"),
                                       fan_law_from_edges(net), element],
                    drives=[Stack.from_network(net, "airpath")], thermal=False, species=1,
                    closures=[closure], flow_balanced=plenums, storage_free=plenums)


def test_duct_relabels_value_errors_only(monkeypatch):
    """A bad argument is a ComponentError naming the duct; a programming KeyError passes through."""
    with pytest.raises(ComponentError, match=r"^d1: .*diameter"):
        duct("d1", length=1.0, diameter=-0.2)

    def broken(self, *args, **kwargs):
        raise KeyError("internal")

    monkeypatch.setattr("noodl.apps.building_physics.components.DuctNetwork.junction", broken)
    with pytest.raises(KeyError, match="internal"):
        duct("d2", length=1.0, diameter=0.2)
