"""Tests for the implicit-function adjoint through the batched Newton solve."""

import pytest
import torch

from noodl.solvers.implicit import adjoint, implicit_solve


def test_gradcheck_implicit_solve_on_cube_root_wrt_parameter():
    torch.manual_seed(0)
    c = torch.rand(3, 1, dtype=torch.float64, requires_grad=True) * 5 + 1.0
    x0 = torch.ones(3, 1, dtype=torch.float64)

    def residual(x, c_):
        return x**3 - c_

    def jacobian(x, c_):
        return (3 * x**2).unsqueeze(-1)

    def f(c_):
        return implicit_solve(residual, jacobian, x0, (c_,), atol=1e-12, rtol=1e-12)

    assert torch.autograd.gradcheck(f, (c,), eps=1e-6, atol=1e-5)


def test_gradcheck_implicit_solve_batched_wrt_linear_system_matrix_entries():
    torch.manual_seed(2)
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([1.0, 2.0], dtype=torch.float64)
    x0 = torch.zeros(2, dtype=torch.float64)

    def residual(x, A_):
        return torch.einsum("ij,j->i", A_, x) - b

    def jacobian(x, A_):
        return A_

    def f(A_):
        return implicit_solve(residual, jacobian, x0, (A_,), atol=1e-12, rtol=1e-12)

    assert torch.autograd.gradcheck(f, (A,), eps=1e-6, atol=1e-5)


def test_adjoint_solves_the_transposed_system():
    J = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    grad_x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    lam = adjoint(J, grad_x)
    torch.testing.assert_close(J.T @ lam, grad_x, atol=1e-10, rtol=1e-10)


def test_second_order_differentiation_raises_instead_of_silently_dropping_a_term():
    # Review finding 1: a gradient-penalty-shaped loss `(dx/dc)**2 + c**2`, computed via
    # `torch.autograd.grad(x, c, create_graph=True)` followed by a second `.backward()`,
    # used to succeed silently and give a WRONG number: the (dx/dc)**2 term contributed
    # zero (implicit_solve's backward never builds a graph over its own gradient, since its
    # internal `torch.autograd.grad` call uses the default `create_graph=False`), while the
    # `c**2` term alone kept the second `.backward()` from raising at all. Confirmed this
    # combination is unsatisfiable to detect via `@torch.autograd.function.once_differentiable`
    # alone: that decorator's guard is keyed on whether the incoming `grad_x` itself already
    # requires grad, which is false for the implicit unit seed `create_graph=True` produces
    # here, so it does not fire for this exact repro (verified empirically with and without
    # the decorator applied, both giving the identical wrong [4.0, 6.0] silently). The actual
    # fix checks `torch.is_grad_enabled()` at `backward`'s own entry instead, which is False
    # for every ordinary (non-second-order) backward call -- including every `gradcheck` in
    # this suite -- and True only when `create_graph=True` was requested upstream.
    c = torch.tensor([[2.0], [3.0]], dtype=torch.float64, requires_grad=True)
    x0 = torch.ones(2, 1, dtype=torch.float64)

    def residual(x, c_):
        return x**3 - c_

    def jacobian(x, c_):
        return (3 * x**2).unsqueeze(-1)

    x = implicit_solve(residual, jacobian, x0, (c,), atol=1e-12, rtol=1e-12)
    with pytest.raises(RuntimeError, match="second-order"):
        torch.autograd.grad(x.sum(), c, create_graph=True)


def test_implicit_solve_ignores_a_grad_requiring_but_functionally_unused_parameter():
    """Regression for the Task 8 (milestone 5) bug: a residual that never reads one of its
    own params -- e.g. `PotentialFlowLayer.solve` threading every key of a shared coupled
    `drivers` mapping into `implicit_solve`'s params positionally, including a key only a
    DIFFERENT layer's residual reads -- used to make `_Implicit.backward` raise
    `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`
    whenever that unused param was the ONLY one requiring grad. `allow_unused=True` only
    excuses individual unused INPUTS to `torch.autograd.grad`; it does not excuse calling
    it with an OUTPUT (the residual itself) that has no graph at all. The correct gradient
    for every param in that case is None/zero: the residual provably does not depend on
    any of them."""
    a = torch.zeros(2, 1, dtype=torch.float64)
    b = torch.ones(2, 1, dtype=torch.float64, requires_grad=True)
    x0 = torch.zeros(2, 1, dtype=torch.float64)

    def residual(x, a_, b_):
        return x - a_                                          # b_ is NEVER read

    def jacobian(x, a_, b_):
        return torch.ones_like(x).unsqueeze(-1)

    x = implicit_solve(residual, jacobian, x0, (a, b), atol=1e-12, rtol=1e-12)
    (grad_b,) = torch.autograd.grad(x.sum(), b, allow_unused=True)
    assert grad_b is None


def test_implicit_solve_mixed_used_and_unused_parameters():
    """Same residual, but `a` (actually used) ALSO requires grad, alongside the unused
    `b`: `a`'s gradient must still come through the adjoint correctly (dx/da = 1, since
    x = a at the converged point) while `b`'s stays None, exactly as it would if `b` were
    the only grad-requiring param."""
    a = torch.zeros(2, 1, dtype=torch.float64, requires_grad=True)
    b = torch.ones(2, 1, dtype=torch.float64, requires_grad=True)
    x0 = torch.zeros(2, 1, dtype=torch.float64)

    def residual(x, a_, b_):
        return x - a_                                          # b_ is NEVER read

    def jacobian(x, a_, b_):
        return torch.ones_like(x).unsqueeze(-1)

    x = implicit_solve(residual, jacobian, x0, (a, b), atol=1e-12, rtol=1e-12)
    grad_a, grad_b = torch.autograd.grad(x.sum(), (a, b), allow_unused=True)
    torch.testing.assert_close(grad_a, torch.ones_like(a))
    assert grad_b is None
