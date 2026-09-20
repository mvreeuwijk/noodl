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

    DO NOT DELETE THIS AS REDUNDANT. It is the ONLY test in this file that kills the
    plausible-looking variant of the adjoint in which v is projected back onto the outputs by
    one more VJP of `z_next` against them (the form the Task 6 brief specified): that VJP
    follows the x -> y edge as well and returns dx*/da = 5/6 against a true 2/3. Every other
    test here either has no inter-output edge or has it pointing the other way, and all of
    them pass against the wrong form. See the module docstring.
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


def test_a_sliced_read_of_a_packed_state_output():
    """The shape Task 8 takes: the pass hands on ONE packed state tensor and the next iterate
    is a SLICE of it, so the read is a genuine non-identity linear map rather than `z_next`
    being an `outputs` entry. `iface = M iface + b theta` packed with a diagnostic
    `|iface|^2`; the closed form is d iface*/d theta = (I - M)^{-1} b, and the diagnostic
    contributes 2 iface*^T times the same.
    """
    M = torch.tensor([[0.3, 0.2], [0.1, 0.4]], dtype=F64)
    b = torch.tensor([1.0, -0.5], dtype=F64)
    theta = torch.tensor(0.9, dtype=F64, requires_grad=True)
    eye = torch.eye(2, dtype=F64)
    with torch.no_grad():
        z_star = [torch.linalg.solve(eye - M, b * theta)]

    def pass_fn(z):
        iface = M @ z[0] + b * theta
        packed = torch.cat([iface, (iface * iface).sum().reshape(1)])
        return [packed], [packed[:2]]                 # non-identity read: a slice

    (packed,) = differentiate_fixed_point(z_star, pass_fn)
    torch.testing.assert_close(packed[:2].detach(), z_star[0], rtol=0.0, atol=1e-15)

    w = torch.tensor([0.5, -1.0, 2.0], dtype=F64)
    (g,) = torch.autograd.grad((w * packed).sum(), (theta,))
    dz = torch.linalg.solve(eye - M, b)               # d iface* / d theta
    want = (w[:2] + 2.0 * w[2] * z_star[0]) @ dz
    torch.testing.assert_close(g, want, rtol=1e-10, atol=1e-12)


def test_an_output_that_carries_no_graph_comes_back_plain():
    """A pass may hand on a tensor that nothing differentiable reached. `autograd.grad`
    refuses such a tensor on either side of a VJP, so the utility must drop it, not crash --
    and it must come back with `requires_grad` FALSE. Every output of an `autograd.Function`
    requires grad if any input does, and this repo branches on that: `solvers/select.py`
    drops the SuperLU fast path for an input that requires grad and refuses an explicit
    `method="sparse_direct"` outright, so a boundary constant carried on through a Model pass
    would cost the fast path, or crash, for no real dependence.
    """
    theta = torch.tensor(0.7, dtype=F64, requires_grad=True)
    const = torch.tensor([5.0, 6.0], dtype=F64)

    def pass_fn(z):
        y = 0.5 * z[0] + theta
        return [y, const], [y]

    y, c = differentiate_fixed_point([2.0 * theta.detach()], pass_fn)
    assert not c.requires_grad and c.grad_fn is None
    torch.testing.assert_close(c, const, rtol=0.0, atol=0.0)
    assert y.requires_grad
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


def test_a_held_interface_entry_handed_back_as_the_leaf_is_refused():
    """The trap the test above avoids, pinned deliberately: the OBVIOUS way to say "the pass
    never updates this entry" is to return `z[j]` itself, and that is wrong. It puts a unit
    row in G_z, so I - G_z is singular and the adjoint solve cannot converge. The two tests
    differ by one token -- `z[0]` here, the graph-free `held` above -- so the contract rule
    in the module docstring is pinned rather than incidental.
    """
    theta = torch.tensor(0.7, dtype=F64, requires_grad=True)
    held = torch.tensor(1.4, dtype=F64)

    def pass_fn(z):
        y = 0.5 * z[0] + theta * theta
        return [y, z[0]], [z[0]]                              # NOT recomputed: a unit row

    y, _ = differentiate_fixed_point([held], pass_fn, where="held test")
    with pytest.raises(RuntimeError, match=r"held test.*adjoint.*did not converge"):
        torch.autograd.grad(y, (theta,))


def test_an_empty_interface_returns_the_single_pass_outputs():
    """No interface means no implicit term: d new/d theta is S_theta, which the pass graph
    already carries. Short-circuited in the caller, because `backward` would otherwise die in
    `torch.cat` on an empty list with no `where` to locate it by.
    """
    theta = torch.tensor(0.7, dtype=F64, requires_grad=True)

    def pass_fn(z):
        assert z == []
        return [theta * theta], []

    (y,) = differentiate_fixed_point([], pass_fn)
    assert y.item() == pytest.approx(0.49)
    (g,) = torch.autograd.grad(y, (theta,))
    assert g.item() == pytest.approx(1.4, rel=1e-10)


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


def test_a_mismatched_next_iterate_shape_is_refused_by_name():
    """Same numel, different shape: without this check it surfaces from inside `backward` as
    a bare autograd shape error with nothing to locate it by."""
    theta = torch.tensor(0.7, dtype=F64, requires_grad=True)
    z_star = [torch.zeros(2, 3, dtype=F64)]

    def pass_fn(z):
        y = 0.5 * z[0] + theta
        return [y], [y.reshape(3, 2)]

    with pytest.raises(ValueError, match=r"shapes test.*tensor 0 of shape \(3, 2\).*\(2, 3\)"):
        differentiate_fixed_point(z_star, pass_fn, where="shapes test")


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"max_iter": 0}, r"knobs test: max_iter must be >= 1 when given, got 0"),
    ({"restart": 0}, r"knobs test: restart must be >= 1 when given, got 0"),
])
def test_out_of_range_solver_knobs_are_refused_at_call_time(kwargs, message):
    """At CALL time, not from inside `backward` where gmres would raise it: the traceback
    there points at an autograd engine frame, not at the caller that set the knob."""

    def pass_fn(z):                                     # pragma: no cover - never reached
        raise AssertionError("pass_fn must not run when a knob is out of range")

    with pytest.raises(ValueError, match=message):
        differentiate_fixed_point(
            [torch.zeros(3, dtype=F64)], pass_fn, where="knobs test", **kwargs
        )


def test_restart_is_passed_through_to_the_adjoint_gmres():
    """Each matvec is a full VJP through the pass graph, so Task 8 may want a smaller Krylov
    basis than the default min(m, 100). A restart of 2 on the 3x3 map reaches the same closed
    form as `test_linear_map_matches_the_closed_form_solve`; and `restart` really is the
    basis size, not a relabelled iteration count -- a full basis converges exactly in 3
    matvecs while a restart of 1 has not converged after 4.
    """
    J = torch.tensor([[0.3, 0.2, 0.0], [0.1, 0.4, 0.1], [0.0, 0.2, 0.5]], dtype=F64)
    B = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=F64)
    eye = torch.eye(3, dtype=F64)
    w = torch.tensor([0.5, -1.0, 2.0], dtype=F64)

    def run(**kwargs):
        theta = torch.tensor([0.3, -0.2], dtype=F64, requires_grad=True)
        with torch.no_grad():
            z_star = [torch.linalg.solve(eye - J, B @ theta)]

        def pass_fn(z):
            y = J @ z[0] + B @ theta
            return [y], [y]

        (y,) = differentiate_fixed_point(z_star, pass_fn, where="restart test", **kwargs)
        return torch.autograd.grad((w * y).sum(), (theta,))[0]

    want = (torch.linalg.solve(eye - J, B)).T @ w
    torch.testing.assert_close(run(restart=2, max_iter=30), want, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(run(restart=3, max_iter=3), want, rtol=1e-10, atol=1e-12)

    with pytest.raises(RuntimeError, match=r"restart test.*adjoint.*did not converge"):
        run(restart=1, max_iter=4)
