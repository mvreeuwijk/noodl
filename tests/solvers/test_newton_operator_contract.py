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


def _grounded_chain():
    """The same 1-interior-node, 1-edge grounded chain as above, as a reusable fixture:
    residual(x) = g*x, operator(x) = the 1x1 SPD GraphLaplacianOperator [g]."""
    src = torch.tensor([0], dtype=torch.long)
    tgt = torch.tensor([1], dtype=torch.long)
    interior_of_node = torch.tensor([-1, 0], dtype=torch.long)
    boundary_mask = torch.tensor([True, False])
    g = torch.tensor([2.0], dtype=torch.float64)

    def residual(x):
        return g * x

    def operator(x):
        return GraphLaplacianOperator(
            src, tgt, g, 1, interior_of_node, boundary_mask=boundary_mask
        )

    return residual, operator


def test_method_direct_agrees_with_auto_and_reports_linear_iterations():
    # `method` is forwarded to select.solve, so the caller chooses the inner solver without
    # newton() knowing anything about the operator's storage. `linear_iterations` is the
    # per-instance MAX inner iteration count over the Newton steps actually taken: exactly 1
    # for a direct (LU) solve, at least 1 for any Krylov one.
    residual, operator = _grounded_chain()
    x0 = torch.tensor([5.0], dtype=torch.float64)

    auto = newton(residual, operator, x0, atol=1e-12, rtol=1e-12)
    direct = newton(residual, operator, x0, atol=1e-12, rtol=1e-12, method="direct")

    assert bool(torch.all(direct.converged))
    torch.testing.assert_close(direct.x, auto.x, atol=1e-12, rtol=0.0)

    for result in (auto, direct):
        assert isinstance(result.linear_iterations, torch.Tensor)
        assert result.linear_iterations.dtype == torch.int64
        assert result.linear_iterations.shape == result.converged.shape
    assert bool(torch.all(direct.linear_iterations == 1))
    assert bool(torch.all(auto.linear_iterations >= 1))


def test_linear_iterations_is_none_when_no_linear_solve_was_needed():
    # `None` is reserved for "no inner solve happened at all" -- an x0 already at the root --
    # and is not a stand-in for a solve whose count is unknown.
    residual, operator = _grounded_chain()
    x0 = torch.zeros(1, dtype=torch.float64)

    result = newton(residual, operator, x0, atol=1e-12, rtol=1e-12)

    assert result.iterations == 0
    assert result.linear_iterations is None
