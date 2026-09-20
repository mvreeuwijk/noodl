"""Implicit differentiation of a converged fixed point: the gradient is the fixed point's,
whatever iteration produced z_star, and it is checked against closed forms."""

from __future__ import annotations

import pytest
import torch

from noodl.solvers.fixed_point import differentiate_fixed_point

F64 = torch.float64


def test_scalar_contraction_started_at_its_fixed_point_returns_the_implicit_derivative():
    """z = 0.5 z + theta has z* = 2 theta, so dz*/dtheta = 2. One pass from z* gives 1."""
    theta = torch.tensor(0.7, dtype=F64, requires_grad=True)
    z_star = [2.0 * theta.detach()]

    def pass_fn(z):
        y = 0.5 * z[0] + theta
        return [y], [y]                       # identity read: z_next IS the output

    (y,) = differentiate_fixed_point(z_star, pass_fn, where="scalar test")
    assert y.item() == pytest.approx(1.4)
    (g,) = torch.autograd.grad(y, (theta,))
    assert g.item() == pytest.approx(2.0, rel=1e-10)


def test_outputs_that_are_not_the_iterate_get_the_chain_rule_through_it():
    theta = torch.tensor(0.7, dtype=F64, requires_grad=True)
    z_star = [2.0 * theta.detach()]

    def pass_fn(z):
        y = 0.5 * z[0] + theta
        return [y, y * y], [y]

    y, y2 = differentiate_fixed_point(z_star, pass_fn)
    (g,) = torch.autograd.grad(y2, (theta,))
    assert g.item() == pytest.approx(2.0 * 1.4 * 2.0, rel=1e-10)   # d(y^2)/dtheta = 2 y dy/dtheta


def test_linear_map_matches_the_closed_form_solve():
    """z = J z + B theta with rho(J) < 1: dz*/dtheta = (I - J)^{-1} B."""
    J = torch.tensor([[0.3, 0.2, 0.0], [0.1, 0.4, 0.1], [0.0, 0.2, 0.5]], dtype=F64)
    B = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=F64)
    theta = torch.tensor([0.3, -0.2], dtype=F64, requires_grad=True)
    eye = torch.eye(3, dtype=F64)
    with torch.no_grad():
        z_star = [torch.linalg.solve(eye - J, B @ theta)]

    def pass_fn(z):
        y = J @ z[0] + B @ theta
        return [y], [y]

    (y,) = differentiate_fixed_point(z_star, pass_fn)
    w = torch.tensor([0.5, -1.0, 2.0], dtype=F64)
    (g,) = torch.autograd.grad((w * y).sum(), (theta,))
    want = ((torch.linalg.solve(eye - J, B)).T @ w)
    torch.testing.assert_close(g, want, rtol=1e-10, atol=1e-12)


def test_several_interface_tensors_of_different_shapes():
    theta = torch.tensor(0.5, dtype=F64, requires_grad=True)
    z_star = [
        torch.tensor([1.0, 1.0], dtype=F64) * theta.detach() * 2,
        torch.tensor(theta.item() * 4 / 3, dtype=F64),
    ]

    def pass_fn(z):
        a = 0.5 * z[0] + theta                    # a* = 2 theta (per entry)
        b = 0.25 * z[1] + theta                   # b* = 4/3 theta
        return [a, b], [a, b]

    a, b = differentiate_fixed_point(z_star, pass_fn)
    (ga,) = torch.autograd.grad(a.sum(), (theta,), retain_graph=True)
    (gb,) = torch.autograd.grad(b, (theta,))
    assert ga.item() == pytest.approx(4.0, rel=1e-10)
    assert gb.item() == pytest.approx(4.0 / 3.0, rel=1e-10)


def test_nonlinear_coupled_map_matches_the_dense_implicit_solve():
    """The brief's other maps are all linear and most are uncoupled; P1-2's real target is a
    COUPLED nonlinear model, where one pass's derivative and the fixed point's differ. The
    oracle here is the implicit function theorem evaluated with dense Jacobians and an LU
    solve -- a wholly different code path from the utility's GMRES-on-VJPs.
    """
    theta = torch.tensor(0.4, dtype=F64, requires_grad=True)

    def step(z, th):
        return torch.stack(
            [
                0.45 * torch.tanh(z[0] + 0.5 * z[1]) + th,
                0.3 * torch.sin(z[0]) + 0.25 * z[1] + 2.0 * th,
            ]
        )

    with torch.no_grad():                       # plain iteration to the fixed point
        zs = torch.zeros(2, dtype=F64)
        for _ in range(400):
            zs = step(zs, theta)
        assert float((step(zs, theta) - zs).abs().max()) < 1e-15

    def pass_fn(z):
        new = step(z[0], theta)
        return [new], [new]

    (new,) = differentiate_fixed_point([zs], pass_fn)
    torch.testing.assert_close(new.detach(), zs, rtol=0.0, atol=1e-14)

    w = torch.tensor([1.5, -0.75], dtype=F64)
    (g,) = torch.autograd.grad((w * new).sum(), (theta,))

    eye = torch.eye(2, dtype=F64)
    a = torch.autograd.functional.jacobian(lambda q: step(q, theta.detach()), zs)
    c = torch.autograd.functional.jacobian(lambda t: step(zs, t), theta.detach())
    want = w @ torch.linalg.solve(eye - a, c)
    assert float(torch.linalg.matrix_norm(a, ord=torch.inf)) < 1.0      # it is a contraction
    torch.testing.assert_close(g, want, rtol=1e-10, atol=1e-12)


def test_a_gauss_seidel_sweep_whose_outputs_feed_each_other():
    """The coupled sweep P1-2 is really about: x is solved first, then y is solved WITH THE
    NEW x, and both are interface states. x = (y + a)/2, y = (x + b)/2 has the closed form
    x* = (2a + b)/3, y* = (a + 2b)/3, so the true sensitivities are 2/3 and 1/3 (and 1/3,
    2/3), while one unrolled pass from the fixed point reports 1/2, 0 (and 1/4, 1/2).

    This is also the case that catches the plausible-looking variant of the adjoint in which
    v is projected back onto the outputs by one more VJP of `z_next` against them: that VJP
    follows the x -> y edge as well and returns dx*/da = 5/6. See the module docstring.
    """
    a = torch.tensor(1.0, dtype=F64, requires_grad=True)
    b = torch.tensor(0.25, dtype=F64, requires_grad=True)
    with torch.no_grad():
        z_star = [
            torch.tensor((2.0 * 1.0 + 0.25) / 3.0, dtype=F64),
            torch.tensor((1.0 + 2.0 * 0.25) / 3.0, dtype=F64),
        ]

    def pass_fn(z):
        x = (z[1] + a) / 2.0            # sweep 1 reads the OLD y
        y = (x + b) / 2.0               # sweep 2 reads the NEW x
        return [x, y], [x, y]

    x, y = differentiate_fixed_point(z_star, pass_fn)
    assert x.item() == pytest.approx(0.75)          # (2 * 1 + 0.25) / 3
    assert y.item() == pytest.approx(0.5)           # (1 + 2 * 0.25) / 3
    gxa, gxb = torch.autograd.grad(x, (a, b), retain_graph=True)
    gya, gyb = torch.autograd.grad(y, (a, b))
    assert gxa.item() == pytest.approx(2.0 / 3.0, rel=1e-10)
    assert gxb.item() == pytest.approx(1.0 / 3.0, rel=1e-10)
    assert gya.item() == pytest.approx(1.0 / 3.0, rel=1e-10)
    assert gyb.item() == pytest.approx(2.0 / 3.0, rel=1e-10)


def test_an_output_that_carries_no_graph_still_comes_back_with_its_value():
    """A pass may hand on a tensor that nothing differentiable reached. `autograd.grad`
    refuses such a tensor on either side of a VJP, so the utility must drop it, not crash.
    """
    theta = torch.tensor(0.7, dtype=F64, requires_grad=True)
    const = torch.tensor([5.0, 6.0], dtype=F64)

    def pass_fn(z):
        y = 0.5 * z[0] + theta
        return [y, const], [y]

    y, c = differentiate_fixed_point([2.0 * theta.detach()], pass_fn)
    torch.testing.assert_close(c.detach(), const, rtol=0.0, atol=0.0)
    (g,) = torch.autograd.grad(y, (theta,))
    assert g.item() == pytest.approx(2.0, rel=1e-10)


def test_an_interface_the_pass_never_updates_falls_back_to_the_single_pass_derivative():
    """A prescribed interface value that no subsystem writes carries no graph, so the
    interface Jacobian is exactly zero, the adjoint system is the identity, and the answer is
    the single pass's own derivative: d y / d theta = 2 theta at fixed z.
    """
    theta = torch.tensor(0.7, dtype=F64, requires_grad=True)
    held = torch.tensor(1.4, dtype=F64)

    def pass_fn(z):
        y = 0.5 * z[0] + theta * theta
        return [y, held], [held]

    y, out_held = differentiate_fixed_point([held], pass_fn)
    assert out_held.item() == pytest.approx(1.4)
    (g,) = torch.autograd.grad(y, (theta,))
    assert g.item() == pytest.approx(1.4, rel=1e-10)          # 2 * theta, no implicit term


def test_non_convergence_of_the_adjoint_is_refused_by_name():
    J = torch.tensor([[0.3, 0.2, 0.0], [0.1, 0.4, 0.1], [0.0, 0.2, 0.5]], dtype=F64)
    theta = torch.tensor([0.3, -0.2, 0.1], dtype=F64, requires_grad=True)
    with torch.no_grad():
        z_star = [torch.linalg.solve(torch.eye(3, dtype=F64) - J, theta)]

    def pass_fn(z):
        y = J @ z[0] + theta
        return [y], [y]

    (y,) = differentiate_fixed_point(z_star, pass_fn, max_iter=1, rtol=1e-15, where="tiny map")
    with pytest.raises(RuntimeError, match=r"tiny map.*adjoint.*did not converge"):
        torch.autograd.grad(y.sum(), (theta,))


def test_second_order_differentiation_is_refused_by_name():
    theta = torch.tensor(0.7, dtype=F64, requires_grad=True)

    def pass_fn(z):
        y = 0.5 * z[0] + theta
        return [y], [y]

    (y,) = differentiate_fixed_point([2.0 * theta.detach()], pass_fn, where="scalar test")
    with pytest.raises(RuntimeError, match=r"scalar test.*second-order"):
        (g,) = torch.autograd.grad(y, (theta,), create_graph=True)
        torch.autograd.grad(g, (theta,))


def test_under_no_grad_the_plain_outputs_come_back():
    theta = torch.tensor(0.7, dtype=F64)

    def pass_fn(z):
        y = 0.5 * z[0] + theta
        return [y], [y]

    with torch.no_grad():
        (y,) = differentiate_fixed_point([torch.tensor(1.4, dtype=F64)], pass_fn)
    assert not y.requires_grad and y.item() == pytest.approx(1.4)


def test_a_pass_that_does_not_depend_on_anything_differentiable_comes_back_plain():
    theta = torch.tensor(0.7, dtype=F64)          # no grad anywhere

    def pass_fn(z):
        y = 0.5 * z[0].detach() + theta
        return [y], [y]

    (y,) = differentiate_fixed_point([torch.tensor(1.4, dtype=F64)], pass_fn)
    assert not y.requires_grad


def test_a_mismatched_next_iterate_length_is_refused_by_name():
    theta = torch.tensor(0.7, dtype=F64, requires_grad=True)

    def pass_fn(z):
        y = 0.5 * z[0] + theta
        return [y], [y, y]

    with pytest.raises(ValueError, match=r"scalar test.*2 next-iterate tensors for 1"):
        differentiate_fixed_point([2.0 * theta.detach()], pass_fn, where="scalar test")
