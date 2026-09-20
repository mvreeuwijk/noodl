"""Independent oracles for the coupling findings R1 and R2 of the 19 September review.

Two unit-capacity compartments. A: one edge zone->ambient that carries no flow, so A changes
only through the sources the coupler adds. B: zone->ambient and ambient->zone, both carrying
unit flow, so B exchanges with its boundary at rate q*(x_boundary - x_B). The two-way link
feeds A's concentration to B's boundary and B's net boundary inflow back into A's sources.
"""

from __future__ import annotations

import pytest
import torch

from noodl.couple import ValueLink, union
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64


def _t(values):
    return torch.tensor(values, dtype=F64)


def compartment(name, *, scheme="exact", circulation=False, initial=1.0, source=0.0):
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("zone", "ambient", kind="flow")
    if circulation:
        net.add_edge("ambient", "zone", kind="flow")
    layer = TransportLayer(
        net, name, capacity=_t([1.0]), flow_kind="flow", boundary=["ambient"],
        scheme=scheme, quantity="concentration", unit="kg/m3",
    )
    q = _t([1.0, 1.0] if circulation else [0.0])

    def closure(state, drivers):
        return {f"{name}.q": q}

    model = Model(net, {name: layer}, closures=[closure])
    state = {f"{name}.x": _t([initial])}
    drivers = {f"{name}.x_boundary": _t([0.0]), f"{name}.sources": _t([0.0, source])}
    return model, state, drivers


def coupled_case(scheme, substeps=1, *, source=0.0, initial_b=0.0):
    a = compartment("a", scheme=scheme)
    b = compartment("b", scheme=scheme, circulation=True, initial=initial_b, source=source)
    model, state, drivers = union(
        {"A": a, "B": b},
        [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True)],
        substeps={"B": substeps}, iterate_rtol=1e-12, iterate_atol=1e-14, iterate_max=200,
    )
    diagnostics: dict = {}
    result = model.step(state, drivers, 1.0, diagnostics=diagnostics)
    return result["A"]["a.x"].item(), result["B"]["b.x"].item(), diagnostics


R1 = pytest.mark.xfail(
    strict=True,
    reason="R1: the coupler feeds back an endpoint rate, not the integrated transfer",
)


@pytest.mark.parametrize(
    "scheme, substeps",
    [
        pytest.param("exact", 1, marks=R1),
        ("implicit", 1),  # control: a single implicit step's endpoint flux IS its integral
        pytest.param("implicit", 4, marks=R1),
        pytest.param("trapezoidal", 1, marks=R1),
    ],
)
def test_closed_two_way_exchange_conserves_the_total_amount(scheme, substeps):
    """No external source or sink anywhere: whatever leaves B must arrive in A."""
    a, b, diagnostics = coupled_case(scheme, substeps)
    assert bool(diagnostics["converged"].all())
    assert a + b == pytest.approx(1.0, abs=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason="R2: convergence is judged on the previous pass's forward value, not the returned state",
)
def test_two_way_iteration_returns_the_coupled_backward_euler_solution():
    """Both start at 1, B receives a unit source, one implicit step of one second each.

    A: a = 1 + (b - a)          ->  2a - b = 1
    B: b = 1 + 1 + (a - b)      -> -a + 2b = 2
    Solution (4/3, 5/3). The current code returns (1.5, 1.5) with converged=True.
    """
    a, b, diagnostics = coupled_case("implicit", 1, source=1.0, initial_b=1.0)
    assert bool(diagnostics["converged"].all())
    assert a == pytest.approx(4 / 3, abs=1e-9)
    assert b == pytest.approx(5 / 3, abs=1e-9)


def test_a_two_way_step_never_assembles_a_dense_topology_operator(monkeypatch):
    """The compartments carry no potential layer, so nothing on this path has a legitimate
    reason to form an (edges, nodes) matrix."""

    def forbidden(self, *args, **kwargs):
        raise AssertionError("dense topology operator assembled on the coupling path")

    monkeypatch.setattr(Network, "upwind", forbidden)
    monkeypatch.setattr(Network, "incidence", forbidden)
    monkeypatch.setattr(Network, "source_selector", forbidden)
    monkeypatch.setattr(Network, "target_selector", forbidden)
    a, b, _ = coupled_case("implicit", 1)
    assert a + b == pytest.approx(1.0, abs=1e-10)
