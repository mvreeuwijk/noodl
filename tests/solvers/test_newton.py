"""Tests for the batched, damped Newton solver."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

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


def test_linear_residual_converges_in_two_or_three_iterations_with_relaxation():
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    b = torch.tensor([1.0, 2.0])
    residual, jacobian = _linear_system(A, b)
    x0 = torch.zeros(2)

    result = newton(residual, jacobian, x0, omega=0.75, switch_ratio=0.5)

    assert result.iterations in (2, 3)
    expected = torch.linalg.solve(A, b)
    torch.testing.assert_close(result.x, expected, atol=1e-9, rtol=1e-9)


def test_scalar_cube_batched_converges_for_random_c():
    torch.manual_seed(1)
    c = torch.rand(20, 1, dtype=torch.float64) * 10 + 0.1
    x0 = torch.ones(20, 1, dtype=torch.float64)

    def residual(x):
        return x**3 - c

    def jacobian(x):
        return (3 * x**2).unsqueeze(-1)

    result = newton(residual, jacobian, x0, max_iter=50)

    assert bool(torch.all(result.converged))
    torch.testing.assert_close(result.x, c ** (1.0 / 3.0), atol=1e-6, rtol=1e-6)


def test_already_converged_instance_is_not_moved():
    A = torch.eye(2)
    b = torch.tensor([1.0, 2.0])
    residual, jacobian = _linear_system(A, b)
    x0 = torch.stack([torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0])])

    result = newton(residual, jacobian, x0, atol=1e-9, rtol=1e-9)

    torch.testing.assert_close(result.x[0], torch.tensor([1.0, 2.0]), atol=1e-12, rtol=0.0)
    torch.testing.assert_close(result.x[1], torch.tensor([1.0, 2.0]), atol=1e-9, rtol=1e-9)


def test_unconverged_after_max_iter_raises_naming_batch_index():
    c = torch.tensor([[1000.0]])
    x0 = torch.tensor([[1.0]])

    def residual(x):
        return x**3 - c

    def jacobian(x):
        return (3 * x**2).unsqueeze(-1)

    with pytest.raises(RuntimeError, match=r"\[0\]"):
        newton(residual, jacobian, x0, max_iter=1)


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n=st.integers(min_value=2, max_value=5),
    batch=st.integers(min_value=1, max_value=4),
)
def test_newton_matches_direct_solve_on_random_spd_systems(seed, n, batch):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(batch, n, n, generator=g, dtype=torch.float64)
    A = M @ M.transpose(-1, -2) + n * torch.eye(n, dtype=torch.float64)
    b = torch.randn(batch, n, generator=g, dtype=torch.float64)
    x0 = torch.zeros(batch, n, dtype=torch.float64)

    def residual(x):
        return torch.einsum("bij,bj->bi", A, x) - b

    def jacobian(x):
        return A

    result = newton(residual, jacobian, x0, atol=1e-10, rtol=1e-10)
    expected = torch.linalg.solve(A, b)
    torch.testing.assert_close(result.x, expected, atol=1e-8, rtol=1e-8)


def test_converged_instance_with_singular_jacobian_does_not_crash_siblings():
    # Instance 0 starts exactly at its root (c=0, x0=0): converged on iteration 0, but its
    # Jacobian (3 * x**2) is exactly singular there. Instance 1 (c=1000) still needs several
    # iterations. A batched linalg.solve raises for the whole call if *any* matrix in the
    # batch is singular, so instance 0's frozen-but-singular Jacobian must not be fed to the
    # solve unguarded, or instance 1 could never converge.
    c = torch.tensor([[0.0], [1000.0]])
    x0 = torch.tensor([[0.0], [1.0]])

    def residual(x):
        return x**3 - c

    def jacobian(x):
        return (3 * x**2).unsqueeze(-1)

    # atol/rtol pinned explicitly: this test's assertions below check x[1] to 1e-6, which
    # is tighter than newton()'s dtype-derived default for float32 (~1.2e-4, since c/x0 here
    # are plain float32 tensors) now that the default is dtype-aware (see newton()'s
    # docstring). Before that change the implicit default was a flat 1e-9 regardless of
    # dtype, so this test's 1e-6 accuracy check silently rode on that; this test's actual
    # subject is the singular-Jacobian masking behaviour described above, not the default
    # tolerance itself, so the fix here is to state the tolerance this test needs explicitly
    # rather than loosen the 1e-6 assertion to match the new, looser float32 default.
    result = newton(residual, jacobian, x0, max_iter=50, atol=1e-9, rtol=1e-9)

    assert bool(torch.all(result.converged))
    torch.testing.assert_close(result.x[0], torch.tensor([0.0]), atol=1e-9, rtol=0.0)
    torch.testing.assert_close(result.x[1], c[1] ** (1.0 / 3.0), atol=1e-6, rtol=1e-6)


def test_sqrt_law_dead_end_does_not_cycle_once_relaxation_switches_off():
    """A node whose only edge follows ``sign(dp) sqrt(|dp|)`` (a dead-end headspace edge in
    the sewer application, a closed room behind one orifice in a building) is a scalar
    equation on which an UNDAMPED Newton step maps ``dp`` to ``-dp`` exactly, so once the
    relaxation had switched to 1 the iterate cycled at constant residual until ``max_iter``
    (observed on the milestone-4 sewer benchmark: 400 iterations at a residual of 2.9e-4).
    Here the first component is linear and converges fast enough to trigger the switch;
    the second is the square-root law. The solver must fall back to the relaxed step
    when a full step fails to shrink the residual, and converge."""
    x0 = torch.tensor([1.0, 1e-4], dtype=torch.float64)

    def residual(x):
        return torch.stack([x[..., 0], torch.sign(x[..., 1]) * x[..., 1].abs().sqrt()], dim=-1)

    def jacobian(x):
        d1 = 0.5 / x[..., 1].abs().sqrt().clamp_min(1e-30)
        return torch.diag_embed(torch.stack([torch.ones_like(d1), d1], dim=-1))

    # Converges geometrically once damped (the relaxed step halves ``dp``, so the residual
    # shrinks by sqrt(2) per iteration): 1e-2 -> 1e-8 takes about 40 of the 50 iterations.
    result = newton(residual, jacobian, x0, omega=0.75, switch_ratio=0.5, atol=1e-8, rtol=0.0)

    assert bool(torch.all(result.converged))
    assert float(result.x.abs().max()) < 1e-15
