"""Tests for PotentialFlowLayer: assembly, non-differentiable solve, power residual."""

import pytest
import torch
from scipy.optimize import brentq

from tellegen.elements import Conductance, FixedFlow, PowerLaw
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.solvers.newton import newton
from tellegen.topology import Network


def _closed_form_series_flow(C, n, drive):
    def f(q):
        return sum((q / c) ** (1.0 / n) for c in C) - drive

    return brentq(f, 1e-9, 1e3)


def test_duplicate_element_kind_raises_value_error():
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")
    with pytest.raises(ValueError, match="airpath"):
        PotentialFlowLayer(
            net,
            "dup",
            [
                Conductance(torch.tensor([1.0], dtype=torch.float64), kind="airpath"),
                Conductance(torch.tensor([1.0], dtype=torch.float64), kind="airpath"),
            ],
        )


def test_element_kind_with_no_edges_raises_value_error():
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")
    with pytest.raises(ValueError, match="conduction"):
        PotentialFlowLayer(
            net,
            "missing",
            [Conductance(torch.tensor([1.0], dtype=torch.float64), kind="conduction")],
        )


def test_unknown_boundary_node_raises_key_error():
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")
    element = PowerLaw(torch.tensor([1.0], dtype=torch.float64), 0.5, kind="airpath")
    with pytest.raises(KeyError, match="zzz"):
        PotentialFlowLayer(net, "bad", [element], boundary=["zzz"])


def test_dp_flows_dflows_and_assemble_on_a_single_conductance_edge():
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="conduction")
    g = torch.tensor([2.0], dtype=torch.float64)
    layer = PotentialFlowLayer(
        net, "cond", [Conductance(g, kind="conduction")], boundary=["ambient"]
    )

    phi = torch.tensor([0.0, 3.0], dtype=torch.float64)  # ambient, z
    dp = layer.dp(phi, {})
    torch.testing.assert_close(dp, torch.tensor([-3.0], dtype=torch.float64))

    q = layer.flows(phi, {})
    torch.testing.assert_close(q, torch.tensor([-6.0], dtype=torch.float64))

    dq = layer.dflows(phi, {})
    torch.testing.assert_close(dq, g)

    phi_i = torch.tensor([3.0], dtype=torch.float64)
    phi_b = torch.tensor([0.0], dtype=torch.float64)
    assembled = layer.assemble(phi_i, phi_b)
    torch.testing.assert_close(assembled, phi)


def test_residual_and_jacobian_on_single_conductance_edge():
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="conduction")
    g = torch.tensor([2.0], dtype=torch.float64)
    layer = PotentialFlowLayer(
        net, "cond", [Conductance(g, kind="conduction")], boundary=["ambient"]
    )

    phi_b = torch.tensor([0.0], dtype=torch.float64)
    phi_i = torch.tensor([3.0], dtype=torch.float64)
    sources = torch.tensor([0.0, 1.0], dtype=torch.float64)  # ambient, z

    r = layer.residual(phi_i, phi_b, {}, sources)
    # q = g*(phi_ambient - phi_z) = 2*(0-3) = -6; A_I row for z is -1; A_I@q = 6; minus s_I(1) -> 5
    torch.testing.assert_close(r, torch.tensor([5.0], dtype=torch.float64))

    J = layer.jacobian(phi_i, phi_b, {})
    # dq/dphi_z = -g ; J = A_I diag(dq) A_I^T = (-1)*2*(-1) = 2
    torch.testing.assert_close(J, torch.tensor([[2.0]], dtype=torch.float64))


def test_linear_init_solves_the_linear_system_exactly_for_conductance_network():
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="conduction")
    g = torch.tensor([2.0], dtype=torch.float64)
    layer = PotentialFlowLayer(
        net, "cond", [Conductance(g, kind="conduction")], boundary=["ambient"]
    )

    phi_b = torch.tensor([0.0], dtype=torch.float64)
    sources = torch.tensor([0.0, 1.0], dtype=torch.float64)
    phi_i = layer.linear_init(phi_b, {}, sources)

    r = layer.residual(phi_i, phi_b, {}, sources)
    torch.testing.assert_close(r, torch.zeros(1, dtype=torch.float64), atol=1e-10, rtol=1e-10)


def test_zone_connected_only_by_fixed_flow_edges_raises_runtime_error():
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z1")
    net.add_node("z2")
    net.add_edge("ambient", "z1", kind="duct")
    net.add_edge("z1", "z2", kind="duct")
    net.add_edge("z2", "ambient", kind="airpath")
    layer = PotentialFlowLayer(
        net,
        "mixed",
        [
            FixedFlow(torch.tensor([0.1, 0.1], dtype=torch.float64), kind="duct"),
            PowerLaw(torch.tensor([0.02], dtype=torch.float64), 0.65, kind="airpath"),
        ],
        boundary=["ambient"],
    )
    phi_b = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="z1"):
        layer.linear_init(phi_b, {}, None)


def test_conductance_only_network_converges_in_one_newton_iteration():
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z1")
    net.add_edge("ambient", "z1", kind="conduction")
    net.add_edge("z1", "ambient", kind="conduction")
    layer = PotentialFlowLayer(
        net,
        "cond",
        [Conductance(torch.tensor([0.5, 0.3], dtype=torch.float64), kind="conduction")],
        boundary=["ambient"],
    )
    phi_b = torch.zeros(1, dtype=torch.float64)
    sources = torch.tensor([0.0, 2.0], dtype=torch.float64)
    phi0 = torch.tensor([5.0], dtype=torch.float64)

    def residual_fn(x):
        return layer.residual(x, phi_b, {}, sources)

    def jacobian_fn(x):
        return layer.jacobian(x, phi_b, {})

    result = newton(residual_fn, jacobian_fn, phi0, omega=1.0)
    assert result.iterations == 1


def test_two_zone_series_flow_matches_closed_form(two_zone_layer):
    net, elements, drives, boundary = two_zone_layer
    layer = PotentialFlowLayer(net, "zones", elements, drives=drives, boundary=boundary)
    phi_b = torch.zeros(1, dtype=torch.float64)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)

    phi, q = layer.solve(phi_b, {"wind": wind}, None, differentiable=False)

    q_ref = _closed_form_series_flow([0.01, 0.02, 0.01], 0.65, 10.0)
    torch.testing.assert_close(
        q, torch.full((3,), q_ref, dtype=torch.float64), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(q[0], q[1], atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(q[1], q[2], atol=1e-9, rtol=1e-9)


def test_batched_wind_driver_matches_looped_solves(two_zone_layer):
    net, elements, drives, boundary = two_zone_layer
    layer = PotentialFlowLayer(net, "zones", elements, drives=drives, boundary=boundary)
    torch.manual_seed(3)
    magnitude = 5.0 + 10.0 * torch.rand(50, 1, dtype=torch.float64)
    wind = torch.cat([magnitude, torch.zeros(50, 2, dtype=torch.float64)], dim=-1)
    phi_b = torch.zeros(50, 1, dtype=torch.float64)

    phi, q = layer.solve(phi_b, {"wind": wind}, None, differentiable=False)

    looped = torch.stack(
        [layer.solve(phi_b[i], {"wind": wind[i]}, None, differentiable=False)[1] for i in range(50)]
    )
    torch.testing.assert_close(q, looped, atol=1e-8, rtol=1e-6)


def test_power_residual_is_zero_at_solution_and_nonzero_when_perturbed(two_zone_layer):
    # NOTE (deviation from the brief, documented in task-7-report.md): the brief's original
    # version of this test perturbed `phi` (not `q`) and expected `power_residual` to become
    # large. That is unsatisfiable for any mathematically correct implementation of the
    # specified formula: algebraically, power_residual(phi, q, drivers) collapses to exactly
    # phi_interior . (A_interior @ q) (the dp*q and drive*q terms cancel down to phi^T(A q),
    # and the boundary term removes the boundary half of that dot product), so it depends on
    # phi only through a coefficient -- A_interior @ q -- that is fixed once q is fixed and is
    # already ~1e-17 at a converged solution. No perturbation of phi alone (with q held fixed)
    # can move that product outside of machine-epsilon territory, on this or any topology.
    # Perturbing q instead genuinely breaks conservation (A_interior @ q_perturbed != 0) and
    # is what actually exercises "nonzero for an inconsistent (phi, q) pair".
    net, elements, drives, boundary = two_zone_layer
    layer = PotentialFlowLayer(net, "zones", elements, drives=drives, boundary=boundary)
    phi_b = torch.zeros(1, dtype=torch.float64)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)

    phi, q = layer.solve(phi_b, {"wind": wind}, None, differentiable=False)
    pr = layer.power_residual(phi, q, {"wind": wind})
    assert abs(pr.item()) < 1e-8

    q_perturbed = q.clone()
    q_perturbed[0] += 0.5
    pr_bad = layer.power_residual(phi, q_perturbed, {"wind": wind})
    assert abs(pr_bad.item()) > 1e-3


def test_power_residual_is_zero_at_solution_with_interior_sources():
    # SPECIFIC GUIDANCE (controller) requires power_residual to take an optional `sources`
    # argument and to be exercised both with and without interior sources: the brief's own
    # tests above only ever call power_residual with sources=None, which cannot distinguish
    # a correctly-signed interior-source term from an incorrectly-signed one (the term is
    # multiplied by a zero source vector either way). This test uses a nonzero interior
    # source so the sign of that term is actually exercised.
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="conduction")
    g = torch.tensor([2.0], dtype=torch.float64)
    layer = PotentialFlowLayer(
        net, "cond", [Conductance(g, kind="conduction")], boundary=["ambient"]
    )

    phi_b = torch.tensor([0.0], dtype=torch.float64)
    sources = torch.tensor([0.0, 1.0], dtype=torch.float64)  # ambient, z
    phi_i = layer.linear_init(phi_b, {}, sources)
    phi = layer.assemble(phi_i, phi_b)
    q = layer.flows(phi, {})

    pr = layer.power_residual(phi, q, {}, sources)
    assert abs(pr.item()) < 1e-10

    pr_no_sources = layer.power_residual(phi, q, {})
    assert abs(pr_no_sources.item()) > 1e-3
