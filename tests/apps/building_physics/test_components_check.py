"""Model.check(names=) lists the ports nobody connected."""

import torch

from noodl.apps.building_physics import build_model, orifice_elements_from_edges
from noodl.components import Component
from noodl.drives import Stack
from tests.components.conftest import leak, link, zone

F64 = torch.float64


def model_and_names(closed_door=False):
    b = Component("b")
    b.inner("ambient", z_ref=0.0, T0=283.15)
    rooms = [b.add(zone(n, volume=60.0, T0=288.15, z_ref=0.0, heat_capacity=0.0))
             for n in ("A", "B")]
    d = b.add(link("d", z_path=1.0, Cd=0.6, area=0.5))
    b.connect(rooms[0].ports.air, d.ports.a)
    b.connect(d.ports.b, rooms[1].ports.air)
    lk = b.add(leak("lk", z_path=0.2, Cd=0.6, area=0.05))
    b.connect(lk.ports.a, rooms[1].ports.air)
    if closed_door:                                # a second door from A that leads nowhere
        d2 = b.add(link("d2", z_path=1.0, Cd=0.6, area=0.5))
        b.connect(rooms[0].ports.air, d2.ports.a)
    net, names = b.flatten(dtype=F64)
    el = orifice_elements_from_edges(net, "airpath")
    return build_model(net, air_elements=[el], drives=[Stack.from_network(net, "airpath")]), names


def issues(report, code):
    return [i for i in report.issues if i.code == code]


def test_no_names_no_port_issues():
    model, _ = model_and_names(closed_door=True)
    assert issues(model.check(), "unconnected-port") == []


def test_a_fully_connected_house_has_no_port_issues():
    model, names = model_and_names()
    assert issues(model.check(names=names), "unconnected-port") == []


def test_unconnected_port_is_information():
    model, names = model_and_names(closed_door=True)
    assert names.unconnected_ports == ["d2.b"]
    found = issues(model.check(names=names), "unconnected-port")
    assert [(i.level, "d2.b" in i.message, "closed" in i.message) for i in found] == [
        ("info", True, True)]
    assert model.check(names=names).ok             # information never fails a check


def test_names_from_another_network_warns():
    model, _ = model_and_names()
    _, other = model_and_names()
    found = issues(model.check(names=other), "names-mismatch")
    assert [i.level for i in found] == ["warning"]
