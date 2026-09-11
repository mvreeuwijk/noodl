"""Tests for the batched, damped Newton solver."""

import pytest
import torch

from tellegen.solvers.newton import NewtonResult, newton


def _linear_system(A, b):
    def residual(x):
        return torch.einsum("...j,ij->...i", x, A) - b

    def jacobian(x):
        return A.expand(x.shape[:-1] + A.shape)

    return residual, jacobian


def test_linear_residual_converges_in_one_iteration_with_omega_one():
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    b = torch.tensor([1.0, 2.0])
    residual, jacobian = _linear_system(A, b)
    x0 = torch.zeros(2)

    result = newton(residual, jacobian, x0, omega=1.0)

    assert isinstance(result, NewtonResult)
    assert result.iterations == 1
    assert bool(torch.all(result.converged))
    expected = torch.linalg.solve(A, b)
    torch.testing.assert_close(result.x, expected, atol=1e-9, rtol=1e-9)
