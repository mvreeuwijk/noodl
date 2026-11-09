"""The building airflow's stack and wind pressures as the per-edge head `"air.edge_head"`."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics import Zone, add_zone, build_model, orifice_law_from_edges
from noodl.apps.building_physics.controls import (
    PRESSURE_DIFFERENCE,
    ControlNetwork,
    Sensor,
    StateReading,
)
from noodl.apps.building_physics.prj import project_to_model, read_prj
from noodl.apps.building_physics.simulation import _with_pressure_differences
from noodl.drives import Stack
from noodl.heads import EdgeHeads
from noodl.topology import Network

F64 = torch.float64
THREE = Path(__file__).resolve().parents[2] / "data" / "contam" / "valThreeZonesWthCtm-UseApi.prj"


def _two_zones():
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    add_zone(net, Zone("A", volume=60.0, T0=303.15))
    add_zone(net, Zone("B", volume=60.0, T0=293.15))
    net.add_edge("ambient", "A", kind="airpath", z_path=0.5, Cd=0.6, area=0.05)
    net.add_edge("A", "B", kind="airpath", z_path=1.0, Cd=0.6, area=0.05)
    net.add_edge("B", "ambient", kind="airpath", z_path=1.5, Cd=0.6, area=0.05)
    return net, orifice_law_from_edges(net, "airpath")


def test_the_contam_head_is_the_sum_of_the_stack_and_wind_terms():
    """Stack and wind on the wind-profile project: the closure's "air.edge_head" is, to 1e-14,
    what the two drives added to each edge's potential difference before."""
    project = read_prj(THREE)
    model, state, drivers = project_to_model(project)
    air = model.kirchhoff["air"]
    drv = model._apply_closures(state, drivers)
    head = drv["air.edge_head"]
    assert head.shape == (len(air.cols),)
    expected = torch.zeros_like(head)
    terms = [type(t).__name__ for t in project.heads]
    assert "Stack" in terms and "Wind" in terms
    for term in project.heads:
        sl = air.kind_slice(term.kind)
        expected[sl] = expected[sl] + term(drv)
    assert head.abs().max().item() > 1.0  # a real stack and wind pressure, not zeros
    torch.testing.assert_close(head, expected, atol=1e-14, rtol=0.0)
    # The layer adds it once: dp is the potential difference plus the head.
    phi = torch.zeros(len(project.net.nodes), dtype=F64)
    plain = air.dp(phi, {k: v for k, v in drv.items() if k != "air.edge_head"})
    torch.testing.assert_close(air.dp(phi, drv) - plain, head, atol=1e-13, rtol=0.0)


def test_the_head_closure_is_listed_after_the_density_closure():
    project = read_prj(THREE)
    model, _state, _drivers = project_to_model(project)
    kinds = [type(c).__name__ for c in model.closures]
    assert kinds.index("PrescribedTemperatureDensity") < kinds.index("EdgeHeads")


def test_a_closure_listed_after_the_head_that_writes_a_driver_it_reads_is_refused():
    class LateDensity:
        outputs = ("rho",)

        def __call__(self, state, drivers):
            return {}

    net, el = _two_zones()
    with pytest.raises(ValueError, match=r"head closure.*'rho'.*LateDensity"):
        build_model(net, air_laws=[el], heads=[Stack.from_network(net, "airpath")],
                    closures=[LateDensity()])


def test_a_closure_that_writes_something_else_after_the_head_is_accepted():
    class Other:
        outputs = ("something.else",)

        def __call__(self, state, drivers):
            return {}

    net, el = _two_zones()
    model = build_model(net, air_laws=[el], heads=[Stack.from_network(net, "airpath")],
                        closures=[Other()])
    assert [type(c).__name__ for c in model.closures][-2:] == ["EdgeHeads", "Other"]


def test_a_model_without_head_terms_has_no_head_closure():
    net, el = _two_zones()
    model = build_model(net, air_laws=[el], heads=[])
    assert not any(isinstance(c, EdgeHeads) for c in model.closures)


def test_air_dp_recorded_for_a_sensor_includes_the_head():
    """`simulate` records `KirchhoffLayer.dp` as "air.dp" for a sensor: with no potential
    difference between the nodes it is the head the closure wrote, the stack of the warm zone."""
    network = ControlNetwork([Sensor(StateReading(PRESSURE_DIFFERENCE, (1,)))])
    net, el = _two_zones()
    model, state, drivers = build_model(
        net, air_laws=[el], heads=[Stack.from_network(net, "airpath")], controls=network,
        thermal=True, return_inputs=True)
    state = {**state, "air.phi": torch.zeros(len(net.nodes), dtype=F64)}
    drivers = {**drivers, "air.phi_boundary": torch.zeros(1, dtype=F64)}
    out = _with_pressure_differences(model, state, drivers, network)
    drv = model._apply_closures(state, drivers)
    stack = Stack.from_network(net, "airpath")(drv)
    assert stack.abs().max().item() > 0.1
    torch.testing.assert_close(out[PRESSURE_DIFFERENCE], stack, atol=1e-14, rtol=0.0)
    torch.testing.assert_close(out[PRESSURE_DIFFERENCE], drv["air.edge_head"],
                               atol=1e-14, rtol=0.0)


def test_a_layer_with_other_edges_than_the_air_layer_is_refused_from_reading_its_head():
    from noodl.apps.building_physics.modelica.assemble import _check_same_edges
    from noodl.layers.kirchhoff import KirchhoffLayer

    net, el = _two_zones()
    air = KirchhoffLayer(net, "air", [el], boundary=["ambient"])
    _check_same_edges(KirchhoffLayer(net, "air", [el], boundary=["ambient", "B"]), air, "x")
    net.add_edge("A", "ambient", kind="extra", z_path=0.5, Cd=0.6, area=0.05)
    other = KirchhoffLayer(net, "air", [el, orifice_law_from_edges(net, "extra")],
                           boundary=["ambient"])
    with pytest.raises(ValueError, match=r"the t = StartTime layer has edge kinds"):
        _check_same_edges(other, air, "the t = StartTime layer")
