"""Tests for newton()'s operator-based contract (Task 11): the second argument is now a
callable returning a LinearOperator (or, for backward compatibility, a plain dense tensor,
auto-wrapped in DenseOperator), and the inner linear solve goes through
solvers.select.solve rather than torch.linalg.solve directly.
"""

import torch

from tellegen.operators.graph import GraphLaplacianOperator
from tellegen.solvers.newton import newton


def test_newton_accepts_a_linear_operator_returning_callable_directly():
    # A trivial 1-interior-node, 1-edge grounded chain: ambient(boundary) -- z(interior).
    # slope g=2, so the Jacobian operator is the 1x1 SPD matrix [2.0].
    src = torch.tensor([0], dtype=torch.long)
    tgt = torch.tensor([1], dtype=torch.long)
    interior_of_node = torch.tensor([-1, 0], dtype=torch.long)
    boundary_mask = torch.tensor([True, False])
    g = torch.tensor([2.0], dtype=torch.float64)

    def residual(x):
        # phi_ambient=0, phi_z=x; q = g*(0-x) = -g*x; A_I row for z is -1; A_I@q = g*x.
        return g * x

    def operator(x):
        return GraphLaplacianOperator(src, tgt, g, 1, interior_of_node, boundary_mask=boundary_mask)

    x0 = torch.tensor([5.0], dtype=torch.float64)
    result = newton(residual, operator, x0, atol=1e-12, rtol=1e-12)

    assert bool(torch.all(result.converged))
    torch.testing.assert_close(result.x, torch.zeros(1, dtype=torch.float64), atol=1e-9, rtol=0.0)


def test_on_failure_return_yields_a_result_instead_of_raising():
    c = torch.tensor([[1000.0]])
    x0 = torch.tensor([[1.0]])

    def residual(x):
        return x**3 - c

    def operator(x):
        return (3 * x**2).unsqueeze(-1)

    result = newton(residual, operator, x0, max_iter=1, on_failure="return")
    assert not bool(torch.all(result.converged))
    assert bool(result.converged[0]) is False
