"""Tests for newton()'s operator-based contract (Task 11): the second argument is now a
callable returning a LinearOperator (or, for backward compatibility, a plain dense tensor,
auto-wrapped in DenseOperator), and the inner linear solve goes through
solvers.select.solve rather than torch.linalg.solve directly.
"""

import torch

from tellegen.operators.graph import GraphLaplacianOperator
from tellegen.solvers.newton import inner_solve_rtol, newton


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


def _leaky_chain(dtype: torch.dtype, m: int = 128):
    """`m` interior zones on a chain, each also leaking to one shared boundary node.

    Leak slopes 1.0, chain slopes 50.0: stiff enough that Jacobi-PCG needs a real number of
    iterations, so the count below is a measurement rather than a constant. Returns
    `(residual, operator, x0, m)` for a LINEAR residual `A x - b`, so Newton's own iteration
    is trivial and what is being observed is the inner solve.
    """
    leaks = [(0, i) for i in range(1, m + 1)]
    links = [(i, i + 1) for i in range(1, m)]
    src = torch.tensor([u for u, _ in leaks + links], dtype=torch.long)
    tgt = torch.tensor([v for _, v in leaks + links], dtype=torch.long)
    slopes = torch.tensor([1.0] * len(leaks) + [50.0] * len(links), dtype=dtype)
    interior_of_node = torch.tensor([-1] + list(range(m)), dtype=torch.long)
    boundary_mask = torch.tensor([True] + [False] * m)

    op = GraphLaplacianOperator(
        src, tgt, slopes, m, interior_of_node, boundary_mask=boundary_mask
    )
    b = torch.linspace(1.0, 2.0, m, dtype=dtype)

    def residual(x):
        return op.matvec(x) - b

    def operator(x):
        return op

    return residual, operator, torch.zeros(m, dtype=dtype), m


def test_inner_solve_rtol_is_floored_by_the_working_dtype():
    # `select.solve`'s pinned rtol=1e-10 is unreachable at float32 (eps 1.2e-7). Without the
    # floor, every inner PCG on this fixture ran to its max_iter ceiling (measured: exactly
    # m=128 iterations), its MAX_ITER status was swallowed by newton's own
    # on_failure="return", and `linear_iterations` reported the ceiling rather than the work
    # actually done -- which is the number the composed-model report publishes.
    residual, operator, x0, m = _leaky_chain(torch.float32)

    result = newton(residual, operator, x0, method="auto")

    assert bool(torch.all(result.converged))
    assert int(result.linear_iterations) < m, (
        f"inner solve ran to its max_iter ceiling ({m}) instead of converging"
    )


def test_the_float32_inner_solve_floor_costs_no_accuracy_against_float64():
    # The complement of the assertion above: the floor buys an honest iteration count
    # without giving up accuracy float32 could have delivered. float64 is unaffected by the
    # floor at all (max(1e-10, 32*2.2e-16) is still 1e-10).
    r32, op32, x32, _ = _leaky_chain(torch.float32)
    r64, op64, x64, _ = _leaky_chain(torch.float64)

    result32 = newton(r32, op32, x32, method="auto")
    result64 = newton(r64, op64, x64, method="auto")

    assert bool(torch.all(result32.converged)) and bool(torch.all(result64.converged))
    torch.testing.assert_close(
        result32.x.to(torch.float64), result64.x, atol=1e-6, rtol=1e-6
    )


def test_inner_solve_rtol_leaves_float64_at_the_pinned_default():
    assert inner_solve_rtol(torch.float64) == 1e-10
    assert inner_solve_rtol(torch.float32) > 1e-10


def test_linear_iterations_is_the_max_over_newton_steps_not_the_last(monkeypatch):
    """`linear_iterations` must be the MAX inner-iteration count over every Newton step
    actually taken, not merely the LAST one. Monkeypatches `newton.select_solve` with a stub
    that returns iterations 1, 5, 2 on its first three successive calls (then keeps
    returning 2), each call an EXACT linear solve `dx = r / g` for the genuinely linear
    residual `g * x` below -- so Newton's own damping (`omega=0.4`, chosen with
    `switch_ratio` set low enough that it never switches to a full step) is the only reason
    more than one call happens at all, and the stub's own solve is otherwise trivial. If the
    implementation used the LAST call's count instead of the max, this would observe 2
    (the stub's steady-state return value) rather than 5.
    """
    import tellegen.solvers.newton as newton_mod
    from tellegen.operators.base import SolveResult, SolverStatus

    g = torch.tensor([1.0], dtype=torch.float64)

    def residual(x):
        return g * x

    def operator(x):
        return g.unsqueeze(-1)  # (1, 1) dense Jacobian; never read since select_solve is stubbed

    call_iters = [1, 5, 2]
    calls = {"n": 0}

    def stub_select_solve(op, r, **kwargs):
        it = call_iters[min(calls["n"], len(call_iters) - 1)]
        calls["n"] += 1
        batch_shape = r.shape[:-1]
        dx = r / g  # the exact linear solve for this linear residual
        return SolveResult(
            x=dx,
            converged=torch.zeros(batch_shape, dtype=torch.bool),
            iterations=torch.full(batch_shape, it, dtype=torch.long),
            residual=torch.zeros(batch_shape, dtype=r.dtype),
            status=torch.full(
                batch_shape, int(SolverStatus.MAX_ITER), dtype=torch.long
            ),
        )

    monkeypatch.setattr(newton_mod, "select_solve", stub_select_solve)

    x0 = torch.tensor([100.0], dtype=torch.float64)
    result = newton_mod.newton(
        residual, operator, x0, omega=0.4, switch_ratio=0.05, max_iter=200,
    )

    assert bool(torch.all(result.converged))
    assert calls["n"] >= 3
    assert int(result.linear_iterations) == 5
