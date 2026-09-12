"""Tests for the implicit-function adjoint through the batched Newton solve."""

import torch

from tellegen.solvers.implicit import implicit_solve


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
