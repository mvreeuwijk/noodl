"""Tests for solve_monotone: batched safeguarded Newton/bisection with an implicit backward."""

import pytest
import torch

from tellegen.solvers.scalar import solve_monotone


def _cube_minus_c(x, c):
    return x**3 - c


def _sign_step(x, c):
    # Zero derivative everywhere except (undefined) at the root itself: forces every
    # iteration into the bisection fallback (dfx == 0 always), so convergence needs exactly
    # ceil(log2(bracket_width / tol)) iterations -- a controlled way to make max_iter too
    # small deterministically, without relying on any particular Newton trajectory.
    return torch.sign(x - c)


def test_finds_cube_root_of_a_scalar():
    c = torch.tensor(8.0, dtype=torch.float64)
    lo = torch.tensor(0.0, dtype=torch.float64)
    hi = torch.tensor(10.0, dtype=torch.float64)
    x = solve_monotone(_cube_minus_c, lo, hi, c)
    torch.testing.assert_close(x, torch.tensor(2.0, dtype=torch.float64), atol=1e-9, rtol=1e-9)


def test_finds_batched_cube_roots_of_random_c():
    torch.manual_seed(0)
    c = torch.empty(20, dtype=torch.float64).uniform_(-50.0, 50.0)
    bound = c.abs().pow(1.0 / 3.0) + 5.0
    x = solve_monotone(_cube_minus_c, -bound, bound, c)
    torch.testing.assert_close(x**3, c, atol=1e-8, rtol=1e-8)


def test_root_exactly_at_a_bracket_end():
    c = torch.tensor(0.0, dtype=torch.float64)  # root of x^3 = 0 is x = 0 = lo
    lo = torch.tensor(0.0, dtype=torch.float64)
    hi = torch.tensor(5.0, dtype=torch.float64)
    x = solve_monotone(_cube_minus_c, lo, hi, c)
    torch.testing.assert_close(x, torch.tensor(0.0, dtype=torch.float64), atol=1e-9, rtol=1e-9)


def test_raises_when_f_lo_and_f_hi_have_the_same_sign():
    c = torch.tensor(-100.0, dtype=torch.float64)  # x^3 - c > 0 on all of [0, 1]
    lo = torch.tensor(0.0, dtype=torch.float64)
    hi = torch.tensor(1.0, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        solve_monotone(_cube_minus_c, lo, hi, c)


def test_gradcheck_root_wrt_c():
    c = torch.tensor([1.0, -8.0, 27.0], dtype=torch.float64, requires_grad=True)
    lo = torch.full((3,), -5.0, dtype=torch.float64)
    hi = torch.full((3,), 5.0, dtype=torch.float64)

    def f(c_):
        return solve_monotone(_cube_minus_c, lo, hi, c_)

    assert torch.autograd.gradcheck(f, (c,), eps=1e-6, atol=1e-6)


def test_raises_on_non_convergence_when_max_iter_is_too_small_for_the_bracket():
    # Pure bisection (see _sign_step) on a 2e15-wide bracket needs ~91 iterations to reach
    # tol=1e-12; max_iter=5 leaves it far short, and this must raise rather than silently
    # return the under-converged midpoint.
    c = torch.tensor(0.0, dtype=torch.float64)
    lo = torch.tensor(-1e15, dtype=torch.float64)
    hi = torch.tensor(1e15, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        solve_monotone(_sign_step, lo, hi, c, max_iter=5)


def test_well_conditioned_root_still_converges_silently_within_default_max_iter():
    # Same bracket and step function as the non-convergence test above, but with enough
    # iterations budgeted (default max_iter=100 comfortably covers the ~91 bisections
    # needed): must NOT raise, and must find the root to within tol.
    c = torch.tensor(0.0, dtype=torch.float64)
    lo = torch.tensor(-1e15, dtype=torch.float64)
    hi = torch.tensor(1e15, dtype=torch.float64)
    x = solve_monotone(_sign_step, lo, hi, c)
    torch.testing.assert_close(x, torch.tensor(0.0, dtype=torch.float64), atol=1e-9, rtol=1e-9)


def test_gradient_matches_the_implicit_function_rule():
    # d(root)/dc for x^3 = c is 1 / (3 x^2); check the closed form against autograd.
    c = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)
    lo = torch.tensor(0.0, dtype=torch.float64)
    hi = torch.tensor(10.0, dtype=torch.float64)
    x = solve_monotone(_cube_minus_c, lo, hi, c)
    (grad_c,) = torch.autograd.grad(x, c)
    torch.testing.assert_close(grad_c, torch.tensor(1.0 / (3.0 * 2.0**2), dtype=torch.float64))
