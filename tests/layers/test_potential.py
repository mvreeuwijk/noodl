"""Tests for PotentialFlowLayer: assembly, non-differentiable solve, power residual."""

import pytest
import torch
from scipy.optimize import brentq

from noodl.drives import ConstantDrive
from noodl.elements import Conductance, FixedFlow, PowerLaw
from noodl.elements.base import Element
from noodl.layers.potential import PotentialFlowLayer
from noodl.nodesources import NodeSource
from noodl.solvers.newton import newton
from noodl.topology import Network


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

    # atol/rtol pinned explicitly: newton()'s default tolerance is now dtype-derived
    # (sqrt(finfo(dtype).eps), about 1.5e-8 for this test's float64 tensors) rather than a
    # flat 1e-9, so the q[0] == q[1] == q[2] series-conservation check below -- which needs
    # branch flows equal to 1e-9 -- must ask newton() for that precision explicitly instead
    # of silently relying on what used to be the unconditional default.
    phi, q = layer.solve(
        phi_b, {"wind": wind}, None, differentiable=False, atol=1e-11, rtol=1e-11
    )

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
    # NOTE: perturbing `phi` (not `q`) and expecting `power_residual` to become large is
    # unsatisfiable for any mathematically correct implementation of the
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
    # power_residual takes an optional `sources` argument and must be exercised both with
    # and without interior sources: the tests above only ever call power_residual with
    # sources=None, which cannot distinguish
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


def test_floating_group_with_no_path_to_boundary_raises_runtime_error():
    # Regression: a floating GROUP (two or more mutually-connected nodes with no path, as a
    # group, to any boundary node) must be detected even though each member's own J0
    # diagonal is nonzero from its internal (within-group) edges -- not only an isolated
    # SINGLE floating node. Reproduction: ambient-z0 and
    # f1-f2 are each tied together by a Conductance edge (nonzero slope), and z0-f1 is a
    # FixedFlow ("duct") edge, whose slope is identically zero and so cannot rescue the
    # {f1, f2} group's connection to the boundary node "ambient".
    net = Network(dtype=torch.float64)
    for name in ("ambient", "z0", "f1", "f2"):
        net.add_node(name)
    net.add_edge("ambient", "z0", kind="conduction")
    net.add_edge("f1", "f2", kind="conduction")
    net.add_edge("z0", "f1", kind="duct")
    layer = PotentialFlowLayer(
        net,
        "grp",
        [
            Conductance(torch.tensor([1.0, 1.0], dtype=torch.float64), kind="conduction"),
            FixedFlow(torch.tensor([0.1], dtype=torch.float64), kind="duct"),
        ],
        boundary=["ambient"],
    )
    phi_b = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(RuntimeError, match=r"f1.*f2|f2.*f1"):
        layer.linear_init(phi_b, {}, None)


def test_drive_kind_not_in_layer_raises_value_error_naming_it():
    # Regression: a drive whose kind matches none of the layer's own
    # element kinds was previously silently dropped (dp() only ever loops over kinds that
    # ARE in the layer, so a "hydronic" drive on an airpath-only layer never gets applied
    # and no error is raised anywhere).
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")
    element = PowerLaw(torch.tensor([1.0], dtype=torch.float64), 0.5, kind="airpath")
    with pytest.raises(ValueError, match="hydronic"):
        PotentialFlowLayer(
            net, "bad", [element], drives=[ConstantDrive("hydronic", "x")], boundary=["a"]
        )


def test_wrong_width_drive_raises_value_error_naming_widths():
    # Regression: a drive tensor whose trailing width does not match its
    # kind's own edge count previously corrupted `dp` silently: torch.cat in dp() would
    # accept the wrongly-widened block, shifting every later kind's slice out from under the
    # (still correctly-sized) `_elem_slices`, so a DIFFERENT element ends up being fed part
    # of the wrong kind's drive value with no exception anywhere in the call chain.
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_node("c")
    net.add_edge("a", "b", kind="conduction")
    net.add_edge("b", "c", kind="airpath")
    layer = PotentialFlowLayer(
        net,
        "widths",
        [
            Conductance(torch.tensor([1.0], dtype=torch.float64), kind="conduction"),
            PowerLaw(torch.tensor([1.0], dtype=torch.float64), 0.5, kind="airpath"),
        ],
        drives=[ConstantDrive("airpath", "wind")],
        boundary=["a"],
    )
    phi = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    wind_bad = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)  # width 3, expected 1
    with pytest.raises(ValueError, match="airpath"):
        layer.dp(phi, {"wind": wind_bad})


def test_floating_group_with_mixed_slope_same_kind_edges_raises_runtime_error():
    # Regression: gating connectivity at KIND granularity ("does this kind have any
    # nonzero-slope edge anywhere") would let a zero-slope edge of an otherwise
    # nonzero-slope kind still connect a group -- exactly what the check must prevent. Reproduction:
    # a single Conductance kind with a closed damper (g=0 on ambient->z1) and an open one (g=1 on
    # z1->z2). {z1, z2} genuinely floats: there is no nonzero-slope path from either to "ambient",
    # even though both edges share the "conduction" kind (which does have a nonzero-slope edge
    # elsewhere).
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z1")
    net.add_node("z2")
    net.add_edge("ambient", "z1", kind="conduction")
    net.add_edge("z1", "z2", kind="conduction")
    layer = PotentialFlowLayer(
        net,
        "mix",
        [Conductance(torch.tensor([0.0, 1.0], dtype=torch.float64), kind="conduction")],
        boundary=["ambient"],
    )
    phi_b = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(RuntimeError, match=r"z1.*z2|z2.*z1"):
        layer.linear_init(phi_b, {}, None)


def test_gradcheck_solve_wrt_powerlaw_conductance(two_zone_layer):
    net, elements, drives, boundary = two_zone_layer
    # Construct the element once with its own learnable Parameter and gradcheck against
    # that exact tensor object: gradcheck perturbs it in place for the numerical Jacobian
    # and autograd tracks it directly for the analytic one, so `f` need not re-read its
    # argument explicitly -- `layer.solve` reads el.C via el.named_parameters() each call.
    el = PowerLaw(
        elements[0].C.detach().clone().requires_grad_(True), elements[0].n, learnable=True
    )
    layer = PotentialFlowLayer(net, "zones", [el], drives=drives, boundary=boundary)
    phi_b = torch.zeros(1, dtype=torch.float64)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)

    def f(C_):
        phi, _ = layer.solve(
            phi_b, {"wind": wind}, None, differentiable=True, atol=1e-12, rtol=1e-12
        )
        return phi[1:]

    assert torch.autograd.gradcheck(f, (el.C,), eps=1e-6, atol=1e-5)


def test_gradcheck_solve_wrt_wind_driver(two_zone_layer):
    net, elements, drives, boundary = two_zone_layer
    layer = PotentialFlowLayer(net, "zones", elements, drives=drives, boundary=boundary)
    phi_b = torch.zeros(1, dtype=torch.float64)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)

    def f(wind_):
        phi, _ = layer.solve(
            phi_b, {"wind": wind_}, None, differentiable=True, atol=1e-12, rtol=1e-12
        )
        return phi[1:]

    assert torch.autograd.gradcheck(f, (wind,), eps=1e-6, atol=1e-5)


def test_gradcheck_solve_wrt_sources(two_zone_layer):
    net, elements, drives, boundary = two_zone_layer
    layer = PotentialFlowLayer(net, "zones", elements, drives=drives, boundary=boundary)
    phi_b = torch.zeros(1, dtype=torch.float64)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
    sources = torch.zeros(3, dtype=torch.float64, requires_grad=True)

    def f(sources_):
        phi, _ = layer.solve(
            phi_b, {"wind": wind}, sources_, differentiable=True, atol=1e-12, rtol=1e-12
        )
        return phi[1:]

    assert torch.autograd.gradcheck(f, (sources,), eps=1e-6, atol=1e-5)


def test_gradcheck_solve_wrt_boundary_potential(two_zone_layer):
    net, elements, drives, boundary = two_zone_layer
    layer = PotentialFlowLayer(net, "zones", elements, drives=drives, boundary=boundary)
    phi_b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)

    def f(phi_b_):
        phi, _ = layer.solve(
            phi_b_, {"wind": wind}, None, differentiable=True, atol=1e-12, rtol=1e-12
        )
        return phi

    assert torch.autograd.gradcheck(f, (phi_b,), eps=1e-6, atol=1e-5)


def test_adjoint_lambda_matches_autograd_gradient(two_zone_layer):
    net, elements, drives, boundary = two_zone_layer
    el = PowerLaw(
        elements[0].C.detach().clone().requires_grad_(True), elements[0].n, learnable=True
    )
    layer = PotentialFlowLayer(net, "zones", [el], drives=drives, boundary=boundary)
    phi_b = torch.zeros(1, dtype=torch.float64)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)

    phi, q = layer.solve(phi_b, {"wind": wind}, None, differentiable=True)
    loss = phi[1:].sum()
    loss.backward()
    autograd_grad = el.C.grad.clone()

    phi_i = phi[layer.interior].detach()
    grad_phi_i = torch.ones_like(phi_i)
    lam = layer.adjoint(phi_i, phi_b, {"wind": wind}, grad_phi_i)

    r = layer.residual(phi_i, phi_b, {"wind": wind}, None)
    (dr_dC,) = torch.autograd.grad(r, el.C, grad_outputs=-lam)
    torch.testing.assert_close(dr_dC, autograd_grad, atol=1e-6, rtol=1e-5)


def test_finite_difference_matches_autograd_gradient_for_zone_pressure(two_zone_layer):
    net, elements, drives, boundary = two_zone_layer
    n = elements[0].n
    C0 = elements[0].C.detach().clone()
    phi_b = torch.zeros(1, dtype=torch.float64)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)

    def zone_pressure(C):
        el = PowerLaw(C, n, learnable=False)
        layer = PotentialFlowLayer(net, "zones", [el], drives=drives, boundary=boundary)
        phi, _ = layer.solve(
            phi_b, {"wind": wind}, None, differentiable=False, atol=1e-13, rtol=1e-13
        )
        return phi[1]  # z1 pressure

    h = 1e-6
    grads_fd = torch.zeros(3, dtype=torch.float64)
    for i in range(3):
        bump = torch.zeros(3, dtype=torch.float64)
        bump[i] = h
        grads_fd[i] = (zone_pressure(C0 + bump) - zone_pressure(C0 - bump)) / (2 * h)

    el = PowerLaw(C0.clone().requires_grad_(True), n, learnable=True)
    layer = PotentialFlowLayer(net, "zones", [el], drives=drives, boundary=boundary)
    phi, _ = layer.solve(phi_b, {"wind": wind}, None, differentiable=True)
    phi[1].backward()
    grads_ad = el.C.grad

    torch.testing.assert_close(grads_ad, grads_fd, atol=1e-5, rtol=1e-5)


def test_jacobian_is_symmetric(two_zone_layer):
    net, elements, drives, boundary = two_zone_layer
    layer = PotentialFlowLayer(net, "zones", elements, drives=drives, boundary=boundary)
    phi_b = torch.zeros(1, dtype=torch.float64)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)

    phi, _ = layer.solve(phi_b, {"wind": wind}, None, differentiable=False)
    phi_i = phi[layer.interior]
    J = layer.jacobian(phi_i, phi_b, {"wind": wind})
    torch.testing.assert_close(J, J.transpose(-1, -2), atol=1e-10, rtol=1e-10)


def test_differentiable_false_matches_differentiable_true(two_zone_layer):
    net, elements, drives, boundary = two_zone_layer
    layer = PotentialFlowLayer(net, "zones", elements, drives=drives, boundary=boundary)
    phi_b = torch.zeros(1, dtype=torch.float64)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)

    phi_nd, q_nd = layer.solve(phi_b, {"wind": wind}, None, differentiable=False)
    phi_d, q_d = layer.solve(phi_b, {"wind": wind}, None, differentiable=True)
    torch.testing.assert_close(phi_nd, phi_d, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(q_nd, q_d, atol=1e-8, rtol=1e-6)


def test_solve_raises_for_element_tensor_with_requires_grad_but_not_learnable(two_zone_layer):
    # `Element._param` deliberately supports holding a tensor with
    # `requires_grad=True` unwrapped (learnable=False) so an external graph is preserved for
    # `differentiable=False`; but such a tensor is absent from `named_parameters()`, so the
    # differentiable solve (which threads only registered parameters through
    # `Function.apply`) cannot reach it and would otherwise return a silently wrong or
    # missing gradient. `solve(differentiable=True)` must refuse instead.
    net, _elements, drives, boundary = two_zone_layer
    C = torch.tensor([0.01, 0.02, 0.01], dtype=torch.float64, requires_grad=True)
    el = PowerLaw(C, 0.65, learnable=False)
    layer = PotentialFlowLayer(net, "zones", [el], drives=drives, boundary=boundary)
    phi_b = torch.zeros(1, dtype=torch.float64)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="C"):
        layer.solve(phi_b, {"wind": wind}, None, differentiable=True)

    # differentiable=False is unaffected: the same construction is a supported, correct use.
    phi, _ = layer.solve(phi_b, {"wind": wind}, None, differentiable=False)
    assert torch.isfinite(phi).all()


def test_solve_raises_for_drive_owning_its_own_differentiable_tensor(two_zone_layer):
    # A Drive is captured by closure inside the differentiable solve, not
    # threaded through Function.apply, so a Drive that owns a learnable tensor directly
    # (instead of reading it from the `drivers` mapping every call) would get a silently
    # absent gradient. `solve(differentiable=True)` must refuse instead.
    class LearnableDrive:
        kind = "airpath"

        def __init__(self, coeff):
            self.coeff = coeff

        def __call__(self, drivers):
            return self.coeff * drivers["wind"]

    net, elements, _drives, boundary = two_zone_layer
    coeff = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    drv = LearnableDrive(coeff)
    layer = PotentialFlowLayer(net, "zones", elements, drives=[drv], boundary=boundary)
    phi_b = torch.zeros(1, dtype=torch.float64)
    wind = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="coeff"):
        layer.solve(phi_b, {"wind": wind}, None, differentiable=True)


class _DetachedLeak(Element):
    """Buggy element: mathematically a PowerLaw leak, but detaches dp inside flow(), which
    silently drops the autograd graph -- the exact hazard
    `test_solve_raises_when_an_element_flow_silently_drops_the_autograd_graph` guards
    against. Unlike FixedFlow, this element does NOT declare `dp_independent = True`: its
    flow genuinely depends on dp mathematically, it just fails to say so to autograd.

    `linear_init` is overridden with the same closed-form tangent `PowerLaw.linear_init`
    uses, bypassing `Element`'s own autograd-based default (which would call `dflow`, which
    in turn would call the buggy `flow`, and raise there instead). This matches the
    demonstrated defect precisely: the forward Newton solve converges without incident (the
    residual is exact regardless of how the Jacobian is computed), and only the
    differentiable path's own Jacobian evaluation (`_dflows_functional`, which
    differentiates `flow` directly via autograd) ever exercises the detach bug.
    """

    def __init__(self, C, n, *, kind: str = "airpath") -> None:
        super().__init__(kind)
        self.C = self._param(C, learnable=False)
        self.n = self._param(n, learnable=False)
        self.dp_transition = 1e-3

    def flow(self, dp, drivers=None):
        dp_bug = dp.detach()  # BUG: silently drops the autograd graph
        return self.C * torch.sign(dp_bug) * dp_bug.abs() ** self.n

    def linear_init(self, drivers=None):
        k = self.C * self.dp_transition ** (self.n - 1)
        return torch.zeros_like(k), k


def _mixed_leak_network(*, include_buggy: bool) -> tuple[Network, list]:
    """ambient (boundary) -- z, joined by two correct PowerLaw leaks (and, if
    `include_buggy`, a third, buggy leak of the same mathematical form)."""
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="leakA")
    net.add_edge("ambient", "z", kind="leakB")
    elements = [
        PowerLaw(torch.tensor([0.02], dtype=torch.float64), 0.65, kind="leakA"),
        PowerLaw(torch.tensor([0.015], dtype=torch.float64), 0.6, kind="leakB"),
    ]
    if include_buggy:
        net.add_edge("ambient", "z", kind="leakBuggy")
        elements.append(
            _DetachedLeak(torch.tensor([0.01], dtype=torch.float64), 0.7, kind="leakBuggy")
        )
    return net, elements


def test_solve_raises_when_an_element_flow_silently_drops_the_autograd_graph():
    # Demonstrated defect: with the old, unconditional `flow.requires_grad` short-circuit
    # in `_dflows_functional`, a mixed layer of two correct PowerLaw leaks plus this buggy
    # element solved fine (Newton converges on the exact residual regardless) but produced a
    # silently wrong adjoint gradient (a demonstrated ~50% error on d(phi_zone)/d(source)).
    # The differentiable solve must now raise instead, naming the offending element, because
    # `_DetachedLeak` does not declare `dp_independent = True`.
    net, elements = _mixed_leak_network(include_buggy=True)
    layer = PotentialFlowLayer(net, "mixed", elements, boundary=["ambient"])
    phi_b = torch.zeros(1, dtype=torch.float64)
    sources = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)

    with pytest.raises(RuntimeError, match="leakBuggy"):
        layer.solve(phi_b, {}, sources, differentiable=True)


def test_solve_matches_finite_differences_for_the_all_correct_mixed_leak_layer():
    # Companion to the test above: with the buggy element removed, the same mixed-kind
    # layer's differentiable solve must still work, and its adjoint gradient of the zone
    # pressure with respect to the interior source must match finite differences.
    net, elements = _mixed_leak_network(include_buggy=False)
    layer = PotentialFlowLayer(net, "mixed", elements, boundary=["ambient"])
    phi_b = torch.zeros(1, dtype=torch.float64)

    def zone_pressure(source_z: torch.Tensor) -> torch.Tensor:
        sources = torch.stack([torch.zeros((), dtype=torch.float64), source_z])
        phi, _ = layer.solve(phi_b, {}, sources, differentiable=False, atol=1e-13, rtol=1e-13)
        return phi[1]

    source_z0 = torch.tensor(1.0, dtype=torch.float64)
    h = 1e-6
    grad_fd = (zone_pressure(source_z0 + h) - zone_pressure(source_z0 - h)) / (2 * h)

    sources = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    phi, _ = layer.solve(phi_b, {}, sources, differentiable=True)
    phi[1].backward()
    grad_ad = sources.grad[1]

    torch.testing.assert_close(grad_ad, grad_fd, atol=1e-6, rtol=1e-5)


class _WronglyDeclaredDpIndependent(Element):
    """A PowerLaw-like leak that WRONGLY declares dp_independent = True: its dflow() is not
    identically zero, so the cross-check in `_dflows_functional` (which reads dflow() for any
    element declaring dp_independent) must catch the mismatch and raise, rather than
    trusting a wrong flag and silently zeroing this element's real Jacobian contribution."""

    dp_independent = True

    def __init__(self, C, n, *, kind: str = "airpath") -> None:
        super().__init__(kind)
        self.C = self._param(C, learnable=False)
        self.n = self._param(n, learnable=False)
        self.dp_transition = 1e-3

    def flow(self, dp, drivers=None):
        return self.C * torch.sign(dp) * dp.abs() ** self.n

    def dflow(self, dp, drivers=None):
        return self.n * self.C * dp.abs() ** (self.n - 1)

    def linear_init(self, drivers=None):
        k = self.C * self.dp_transition ** (self.n - 1)
        return torch.zeros_like(k), k


def test_solve_raises_when_dp_independent_is_wrongly_declared():
    # Design choice evaluated in the report: dp_independent is cross-checked against the
    # element's own analytic dflow() so a WRONG declaration (not just a missing one) is also
    # caught, instead of being trusted silently.
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="leak")
    elements = [
        _WronglyDeclaredDpIndependent(torch.tensor([0.02], dtype=torch.float64), 0.7, kind="leak")
    ]
    layer = PotentialFlowLayer(net, "wrong_flag", elements, boundary=["ambient"])
    phi_b = torch.zeros(1, dtype=torch.float64)
    sources = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)

    with pytest.raises(RuntimeError, match="leak"):
        layer.solve(phi_b, {}, sources, differentiable=True)


class _DetachedButLearnableLeak(Element):
    """Buggy element whose flow requires grad (through its own learnable parameter C) but
    never actually uses dp, so `_dflows_functional`'s autograd.grad call raises "not used in
    the graph" rather than "does not require grad" -- the other of the two autograd failure
    modes the guard must turn into a clear, element-naming error."""

    def __init__(self, C, n, *, kind: str = "airpath") -> None:
        super().__init__(kind)
        self.C = self._param(C, learnable=True)
        self.n = self._param(n, learnable=False)
        self.dp_transition = 1e-3

    def flow(self, dp, drivers=None):
        dp_bug = dp.detach()  # BUG: silently drops the autograd graph
        return self.C * torch.sign(dp_bug) * dp_bug.abs() ** self.n

    def linear_init(self, drivers=None):
        k = self.C * self.dp_transition ** (self.n - 1)
        return torch.zeros_like(k), k


def test_solve_raises_when_flow_requires_grad_via_a_parameter_but_ignores_dp():
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="leak")
    elements = [
        _DetachedButLearnableLeak(torch.tensor([0.02], dtype=torch.float64), 0.7, kind="leak")
    ]
    layer = PotentialFlowLayer(net, "unused_dp", elements, boundary=["ambient"])
    phi_b = torch.zeros(1, dtype=torch.float64)
    sources = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)

    with pytest.raises(RuntimeError, match="leak"):
        layer.solve(phi_b, {}, sources, differentiable=True)


class _Emitter(NodeSource):
    """q_out = C sign(phi) |phi|^n, the orifice-type emitter EPANET also offers."""

    def __init__(self, nodes, coeff, exponent=0.5):
        super().__init__(nodes)
        self.coeff = torch.nn.Parameter(torch.as_tensor(coeff, dtype=torch.float64))
        self.exponent = float(exponent)

    def flow(self, phi_nodes, drivers=None):
        safe = torch.clamp(phi_nodes.abs(), min=1e-12)
        return self.coeff * torch.sign(phi_nodes) * safe**self.exponent


def _pipe_law():
    # C and n are explicit float64 tensors: a bare Python float here is silently cast to
    # torch.get_default_dtype() (float32 in this repo, see elements/duct.py) by
    # Element._param, and the ~1e-8 relative precision loss that costs would blow the
    # rel=1e-9 tolerances the node-source tests below check against an independently
    # computed root.
    return PowerLaw(
        torch.tensor(0.05, dtype=torch.float64), torch.tensor(0.54, dtype=torch.float64),
        kind="pipe",
    )


def _emitter_layer(coeff=0.02):
    net = Network(dtype=torch.float64)
    net.add_node("J")
    net.add_node("R")
    net.add_edge("R", "J", kind="pipe")
    source = _Emitter(torch.tensor([0]), torch.tensor([coeff], dtype=torch.float64))
    layer = PotentialFlowLayer(
        net, "water", [_pipe_law()], boundary=["R"],
        node_sources=[source], linear_solver="direct",
    )
    return net, layer, source


def test_node_source_solves_against_a_hand_written_root():
    """0.05 (10 - x)^0.54 == 0.02 x^0.5 at the junction."""
    scipy_optimize = pytest.importorskip("scipy.optimize")
    _, layer, _ = _emitter_layer()
    pb = torch.tensor([10.0], dtype=torch.float64)
    phi, q = layer.solve(pb, {}, None, differentiable=False, atol=1e-13, rtol=1e-13)
    root = scipy_optimize.brentq(
        lambda x: 0.05 * (10.0 - x) ** 0.54 - 0.02 * x**0.5, 1e-9, 10.0 - 1e-9
    )
    assert float(phi[0]) == pytest.approx(root, abs=1e-7)
    # the branch flow equals the withdrawal at the converged point
    assert float(q[0]) == pytest.approx(0.02 * root**0.5, rel=1e-9)


def test_node_source_enters_the_residual_and_the_jacobian_diagonal():
    _, layer, _ = _emitter_layer()
    pb = torch.tensor([10.0], dtype=torch.float64)
    phi, _ = layer.solve(pb, {}, None, differentiable=False, atol=1e-13, rtol=1e-13)
    residual = layer.residual(phi[..., layer.interior], pb, {}, None)
    assert float(residual.detach().abs().max()) < 1e-10
    without = PotentialFlowLayer(
        layer.net, "bare", [_pipe_law()], boundary=["R"],
        linear_solver="direct",
    )
    dense = layer.jacobian(phi[..., layer.interior], pb, {})
    bare = without.jacobian(phi[..., without.interior], pb, {})
    added = float(dense[0, 0] - bare[0, 0])
    assert added == pytest.approx(0.5 * 0.02 * float(phi[0]) ** -0.5, rel=1e-9)


def test_node_source_adjoint_passes_gradcheck():
    _, layer, _ = _emitter_layer()

    def fn(pb):
        phi, _ = layer.solve(pb, {}, None, differentiable=True)
        return phi[..., layer.interior]

    pb = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (pb,), eps=1e-6, atol=1e-8, rtol=1e-5)


def test_node_source_parameter_gradient_matches_central_differences():
    _, layer, source = _emitter_layer()
    source.coeff.requires_grad_(True)
    pb = torch.tensor([10.0], dtype=torch.float64)
    phi, _ = layer.solve(pb, {}, None, differentiable=True)
    phi[..., layer.interior].sum().backward()
    analytic = float(source.coeff.grad[0])
    eps = 1e-7
    with torch.no_grad():
        source.coeff += eps
    up, _ = layer.solve(pb, {}, None, differentiable=False, atol=1e-14, rtol=1e-14)
    with torch.no_grad():
        source.coeff -= 2 * eps
    down, _ = layer.solve(pb, {}, None, differentiable=False, atol=1e-14, rtol=1e-14)
    with torch.no_grad():
        source.coeff += eps
    fd = float((up[0] - down[0]) / (2 * eps))
    assert analytic == pytest.approx(fd, rel=1e-6)


def test_node_source_on_a_boundary_node_is_refused():
    net = Network(dtype=torch.float64)
    net.add_node("J")
    net.add_node("R")
    net.add_edge("R", "J", kind="pipe")
    with pytest.raises(ValueError, match=r"acts at \['R'\]"):
        PotentialFlowLayer(
            net, "water", [_pipe_law()], boundary=["R"],
            node_sources=[_Emitter(torch.tensor([1]),
                                   torch.tensor([0.02], dtype=torch.float64))],
        )


def test_a_phi_independent_node_source_is_refused():
    class _Constant(NodeSource):
        def flow(self, phi_nodes, drivers=None):
            return torch.full_like(phi_nodes, 0.01)

    source = _Constant(torch.tensor([0]))
    with pytest.raises(RuntimeError, match="does not depend on phi_nodes"):
        source.dflow(torch.tensor([5.0], dtype=torch.float64))


class _LinearGround(NodeSource):
    """w = g * phi: a linear virtual-ground withdrawal, dflow = g everywhere (well-defined
    at phi = 0, unlike `_Emitter`'s sqrt law) -- built for the grounding tests below."""

    def __init__(self, nodes, g):
        super().__init__(nodes)
        self.g = float(g)

    def flow(self, phi_nodes, drivers=None):
        return self.g * phi_nodes


def test_grounding_counts_a_node_source_at_a_fixed_flow_only_node():
    """A node reachable from every boundary only through FixedFlow edges (dflow == 0
    everywhere, the "structurally singular subnetwork" case) is ungrounded by
    edges alone; a NodeSource with a positive slope at that node must still let it solve,
    because the node source's own diagonal shift IS the SPD contribution a boundary
    connection would otherwise have to supply."""
    net = Network(dtype=torch.float64)
    net.add_node("J")
    net.add_node("R")
    net.add_edge("J", "R", kind="vent")
    q0 = torch.tensor([0.01], dtype=torch.float64)
    elements = [FixedFlow(q0, kind="vent")]
    source = _LinearGround(torch.tensor([0]), 0.02)
    layer = PotentialFlowLayer(
        net, "test", elements, boundary=["R"], node_sources=[source],
        linear_solver="direct",
    )
    pb = torch.tensor([10.0], dtype=torch.float64)
    phi, q = layer.solve(pb, {}, None, differentiable=False, atol=1e-13, rtol=1e-13)
    assert torch.isfinite(phi).all() and torch.isfinite(q).all()
    # The residual is exactly linear here (FixedFlow contributes a phi-independent q0, the
    # node source w = g*phi): A_I q0 + g*phi_J = 0.
    expected = float(-q0[0] / 0.02)
    assert float(phi[..., layer.interior]) == pytest.approx(expected, rel=1e-9)

    # Without the fix (edge-only grounding), this same layer is refused: the FixedFlow edge
    # contributes a zero slope, so an edge-only check sees an ungrounded interior node.
    without_source = PotentialFlowLayer(
        net, "bare", elements, boundary=["R"], linear_solver="direct",
    )
    with pytest.raises(RuntimeError, match="floating nodes"):
        without_source.solve(pb, {}, None, differentiable=False)


def test_two_node_sources_on_one_node_sum_their_withdrawals():
    """EPANET semantics -- several node sources at one junction sum, exactly as if a
    single node source carried their combined coefficient."""
    net = Network(dtype=torch.float64)
    net.add_node("J")
    net.add_node("R")
    net.add_edge("R", "J", kind="pipe")
    a = _Emitter(torch.tensor([0]), torch.tensor([0.02], dtype=torch.float64))
    b = _Emitter(torch.tensor([0]), torch.tensor([0.01], dtype=torch.float64))
    two = PotentialFlowLayer(
        net, "water", [_pipe_law()], boundary=["R"], node_sources=[a, b],
        linear_solver="direct",
    )
    combined = _Emitter(torch.tensor([0]), torch.tensor([0.03], dtype=torch.float64))
    one = PotentialFlowLayer(
        net, "water", [_pipe_law()], boundary=["R"], node_sources=[combined],
        linear_solver="direct",
    )
    pb = torch.tensor([10.0], dtype=torch.float64)
    phi_two, q_two = two.solve(pb, {}, None, differentiable=False, atol=1e-13, rtol=1e-13)
    phi_one, q_one = one.solve(pb, {}, None, differentiable=False, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(phi_two, phi_one, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(q_two, q_one, rtol=1e-9, atol=1e-12)


def test_node_source_with_a_batched_parameter_solves_each_instance_independently():
    """A NodeSource whose own parameter carries a leading batch dim solves each
    instance to the same answer an unbatched, single-instance solve would give it."""
    net = Network(dtype=torch.float64)
    net.add_node("J")
    net.add_node("R")
    net.add_edge("R", "J", kind="pipe")
    coeffs = torch.tensor([[0.01], [0.02], [0.04]], dtype=torch.float64)  # (3 instances, 1 node)
    source = _Emitter(torch.tensor([0]), coeffs)
    layer = PotentialFlowLayer(
        net, "water", [_pipe_law()], boundary=["R"], node_sources=[source],
        linear_solver="direct",
    )
    pb = torch.tensor([10.0], dtype=torch.float64)
    phi, q = layer.solve(pb, {}, None, differentiable=False, atol=1e-13, rtol=1e-13)
    assert phi.shape[:-1] == (3,) and q.shape[:-1] == (3,)
    for i in range(3):
        single_source = _Emitter(torch.tensor([0]), coeffs[i])
        single_layer = PotentialFlowLayer(
            net, "water", [_pipe_law()], boundary=["R"], node_sources=[single_source],
            linear_solver="direct",
        )
        phi_i, q_i = single_layer.solve(
            pb, {}, None, differentiable=False, atol=1e-13, rtol=1e-13
        )
        torch.testing.assert_close(phi[i], phi_i, rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(q[i], q_i, rtol=1e-9, atol=1e-12)


def test_node_source_with_an_unregistered_grad_tensor_is_named_as_a_node_source():
    """The unreachable-tensor guard must call a NodeSource a "node source" (not an
    "element") and must not tell the caller to pass learnable=True -- a NodeSource has no
    such constructor kwarg; the fix is to register the tensor as an nn.Parameter."""

    class _LeakyGradSource(NodeSource):
        def __init__(self, nodes, coeff):
            super().__init__(nodes)
            self.coeff = torch.nn.Parameter(torch.as_tensor(coeff, dtype=torch.float64))
            # A stray tensor with requires_grad=True that is NOT a registered nn.Parameter:
            # invisible to named_parameters(), so the differentiable solve cannot reach it.
            self.stray = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

        def flow(self, phi_nodes, drivers=None):
            return self.coeff * self.stray * phi_nodes

    net = Network(dtype=torch.float64)
    net.add_node("J")
    net.add_node("R")
    net.add_edge("R", "J", kind="pipe")
    source = _LeakyGradSource(torch.tensor([0]), 0.02)
    layer = PotentialFlowLayer(
        net, "water", [_pipe_law()], boundary=["R"], node_sources=[source],
        linear_solver="direct",
    )
    pb = torch.tensor([10.0], dtype=torch.float64)
    with pytest.raises(ValueError, match=r"node source .*'stray'") as excinfo:
        layer.solve(pb, {}, None, differentiable=True)
    message = str(excinfo.value)
    assert "learnable=True" not in message
    assert not message.startswith("element ")


def test_element_for_returns_the_element_and_its_q_slice():
    """The public accessor a caller outside this layer uses instead of reaching into
    `_elements`/`_elem_slices` directly."""
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_node("c")
    net.add_edge("a", "b", kind="pipe")
    net.add_edge("b", "c", kind="valve")
    pipe = _pipe_law()
    valve = PowerLaw(torch.tensor(0.03, dtype=torch.float64),
                      torch.tensor(0.6, dtype=torch.float64), kind="valve")
    layer = PotentialFlowLayer(net, "net", [pipe, valve], boundary=["a"])

    el, sl = layer.element_for("valve")
    assert el is valve
    assert sl == layer.kind_slice("valve")

    el0, sl0 = layer.element_for("pipe")
    assert el0 is pipe
    assert sl0 == slice(0, 1)

    with pytest.raises(KeyError, match=r"nope.*pipe.*valve"):
        layer.element_for("nope")


def test_node_sources_default_to_empty_and_change_nothing():
    net = Network(dtype=torch.float64)
    net.add_node("J")
    net.add_node("R")
    net.add_edge("R", "J", kind="pipe")
    layer = PotentialFlowLayer(
        net, "water", [_pipe_law()], boundary=["R"],
        linear_solver="direct",
    )
    phi, _ = layer.solve(
        torch.tensor([10.0], dtype=torch.float64),
        {}, torch.tensor([-0.01, 0.0], dtype=torch.float64), differentiable=False,
        atol=1e-13, rtol=1e-13,
    )
    assert float(phi[0]) == pytest.approx(10.0 - (0.01 / 0.05) ** (1.0 / 0.54), rel=1e-9)
