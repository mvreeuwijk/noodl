"""Tests for the implicit-function adjoint through the batched Newton solve."""

import torch

from tellegen.solvers.implicit import adjoint, implicit_solve


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
