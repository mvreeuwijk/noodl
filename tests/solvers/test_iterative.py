"""Tests for pcg (Jacobi-preconditioned CG, per-instance status, never raises) and, appended
by Task 6, gmres (restarted, nonsymmetric-capable, per-instance status, never raises).
"""

import pytest
import torch

from tellegen.operators.base import SolverStatus
from tellegen.operators.dense import DenseOperator
from tellegen.operators.graph import GraphLaplacianOperator
from tellegen.solvers.iterative import gmres, pcg


@pytest.fixture(autouse=True)
def _set_float64_dtype():
    """Set default dtype to float64 for this module's tests, then restore."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old_dtype)


def test_pcg_matches_solve_on_small_spd_system():
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    b = torch.tensor([1.0, 2.0])
    op = DenseOperator(A, symmetric=True)
    result = pcg(op, b)
    x_ref = torch.linalg.solve(A, b)
    torch.testing.assert_close(result.x, x_ref, atol=1e-8, rtol=1e-8)
    assert bool(result.converged)
    assert int(result.status) == int(SolverStatus.CONVERGED)


def test_pcg_batching_matches_looped():
    torch.manual_seed(0)
    B, n = 5, 6
    As, bs = [], []
    for _ in range(B):
        M = torch.randn(n, n)
        As.append(M @ M.T + n * torch.eye(n))
        bs.append(torch.randn(n))
    A_batch = torch.stack(As)
    b_batch = torch.stack(bs)
    op_batch = DenseOperator(A_batch, symmetric=True)
    result = pcg(op_batch, b_batch)
    for i in range(B):
        op_i = DenseOperator(As[i], symmetric=True)
        result_i = pcg(op_i, bs[i])
        torch.testing.assert_close(result.x[i], result_i.x, atol=1e-8, rtol=1e-8)


def test_pcg_breakdown_on_indefinite_operator_does_not_affect_siblings():
    # A_indef has eigenvalues 3 (eigenvector [1, 1]) and -1 (eigenvector [1, -1]); with the
    # Jacobi preconditioner's diagonal = identity here, the first search direction is b
    # itself, so b = [1, -1] (the negative-curvature eigenvector) makes p.Ap < 0 on the very
    # first iteration -- a deliberately constructed, not incidental, breakdown.
    A_spd = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    b_spd = torch.tensor([1.0, 2.0])
    A_indef = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    b_indef = torch.tensor([1.0, -1.0])

    A_mix = torch.stack([A_spd, A_indef])
    b_mix = torch.stack([b_spd, b_indef])
    op_mix = DenseOperator(A_mix, symmetric=True)
    result = pcg(op_mix, b_mix)

    assert int(result.status[0]) == int(SolverStatus.CONVERGED)
    assert bool(result.converged[0])
    assert int(result.status[1]) == int(SolverStatus.BREAKDOWN)
    assert not bool(result.converged[1])
    # instance 0 is unaffected by instance 1's breakdown
    x_ref = torch.linalg.solve(A_spd, b_spd)
    torch.testing.assert_close(result.x[0], x_ref, atol=1e-8, rtol=1e-8)


def test_pcg_max_iter_one_yields_max_iter_status_without_raising():
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    b = torch.tensor([1.0, 2.0])
    op = DenseOperator(A, symmetric=True)
    result = pcg(op, b, max_iter=1)  # does not raise
    assert not bool(result.converged)
    assert int(result.status) == int(SolverStatus.MAX_ITER)
    assert int(result.iterations) == 1


def test_pcg_jacobi_reduces_iterations_versus_none():
    # A poorly scaled diagonal system: Jacobi preconditioning on a diagonal matrix is EXACT
    # (M^-1 A is a multiple of the identity), so it converges in 1 iteration; plain CG's
    # convergence rate depends on the condition number (1e10 here) and does not reach the
    # same tight rtol within the m = 6 exact-termination budget.
    torch.manual_seed(7)
    diag_vals = torch.tensor([1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6])
    A = torch.diag(diag_vals)
    b = torch.randn(6)
    op = DenseOperator(A, symmetric=True)
    result_jacobi = pcg(op, b, preconditioner="jacobi")
    result_none = pcg(op, b, preconditioner=None)
    n_jacobi = int(result_jacobi.iterations)
    n_none = int(result_none.iterations)
    print(f"jacobi iterations={n_jacobi} (converged={bool(result_jacobi.converged)}), "
          f"none iterations={n_none} (converged={bool(result_none.converged)})")
    assert bool(result_jacobi.converged)
    assert n_jacobi < n_none, f"jacobi={n_jacobi} none={n_none}"
    assert n_jacobi == 1


def test_pcg_per_instance_freezing_bit_identical_after_convergence():
    # instance 0 (diagonal, Jacobi-preconditioned) converges in exactly 1 iteration; instance
    # 1 (a random dense SPD system) needs the full m = 6 iterations. Running with max_iter=2
    # must leave instance 0's x BIT-IDENTICAL to the max_iter=6 (default) run, while instance
    # 1's x differs (it is still mid-solve at iteration 2).
    torch.manual_seed(42)
    diag_vals = torch.tensor([1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6])
    A_diag = torch.diag(diag_vals)
    b_diag = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    M = torch.randn(6, 6)
    A_rand = M @ M.T + 6 * torch.eye(6)
    b_rand = torch.tensor([0.3, -0.2, 0.7, 1.1, -0.5, 0.2])
    A2 = torch.stack([A_diag, A_rand])
    b2 = torch.stack([b_diag, b_rand])
    op2 = DenseOperator(A2, symmetric=True)

    result_full = pcg(op2, b2, preconditioner="jacobi")
    assert result_full.iterations.tolist() == [1, 6]
    assert bool(torch.all(result_full.converged))

    result_2 = pcg(op2, b2, preconditioner="jacobi", max_iter=2)
    assert torch.equal(result_full.x[0], result_2.x[0])  # frozen at iteration 1, bit-identical
    assert not torch.equal(result_full.x[1], result_2.x[1])  # instance 1 still converging


def test_pcg_gradcheck_fixed_iteration_count():
    # Differentiating an iterative solve directly is only for THIS test: pcg's own backward
    # (via ordinary autograd through the loop) is never the production path -- production
    # gradients for a certified-SPD potential solve go through the implicit adjoint (Task 12),
    # which differentiates the FIXED POINT, not the iteration count. max_iter=2 = the exact
    # system size (m=2), so CG converges exactly within the fixed iteration count in exact
    # arithmetic and the loop is smoothly differentiable end to end.
    src = torch.tensor([0, 1])
    tgt = torch.tensor([1, 2])
    interior_of_node = torch.tensor([0, 1, 0])
    boundary_mask = torch.tensor([False, False, True])
    slopes = torch.tensor([1.3, 0.9], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([0.5, -0.3], dtype=torch.float64, requires_grad=True)

    def f(s, bb):
        op = GraphLaplacianOperator(src, tgt, s, 2, interior_of_node, boundary_mask=boundary_mask)
        return pcg(op, bb, max_iter=2, preconditioner=None).x

    assert torch.autograd.gradcheck(f, (slopes, b), eps=1e-6, atol=1e-6)


def test_gmres_matches_solve_on_nonsymmetric_system():
    A = torch.tensor([[3.0, 1.0], [0.5, 2.0]])
    b = torch.tensor([1.0, 2.0])
    op = DenseOperator(A, symmetric=False)
    result = gmres(op, b)
    x_ref = torch.linalg.solve(A, b)
    torch.testing.assert_close(result.x, x_ref, atol=1e-6, rtol=1e-6)
    assert bool(result.converged)
    assert int(result.status) == int(SolverStatus.CONVERGED)


def test_gmres_matches_solve_on_spd_system():
    # Correctness only, not efficiency: gmres makes no symmetry assumption, so it should
    # still solve a symmetric system correctly (just less efficiently than pcg would).
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    b = torch.tensor([1.0, 2.0])
    op = DenseOperator(A, symmetric=True)
    result = gmres(op, b)
    x_ref = torch.linalg.solve(A, b)
    torch.testing.assert_close(result.x, x_ref, atol=1e-6, rtol=1e-6)


def test_gmres_restart_boundary_crossed_still_correct():
    # n = 20, restart = 3: this problem provably needs many more than 3 Arnoldi directions
    # (confirmed below via the returned iteration count), so convergence here can only come
    # from correctly carrying state across multiple restart cycles.
    torch.manual_seed(2)
    n = 20
    M = torch.randn(n, n) * 0.15
    A = torch.eye(n) + M
    b = torch.randn(n)
    op = DenseOperator(A, symmetric=False)
    result = gmres(op, b, restart=3, max_iter=200)
    x_ref = torch.linalg.solve(A, b)
    torch.testing.assert_close(result.x, x_ref, atol=1e-6, rtol=1e-6)
    assert bool(result.converged)
    assert int(result.iterations) > 3, "the restart boundary was not actually crossed"
    assert int(result.iterations) == 63


def test_gmres_per_instance_status_and_freezing():
    # instance 0 has b = 0: x = 0 is exact at iteration 0, no Arnoldi work needed at all.
    # instance 1 is the restart-crossing system from the test above (needs 63 iterations).
    # Run once to convergence, once capped at max_iter=3 (instance 0 must already be done;
    # instance 1 must not be), and instance 0's x must be bit-identical across both runs.
    torch.manual_seed(2)
    n = 20
    M = torch.randn(n, n) * 0.15
    A1 = torch.eye(n) + M
    b1 = torch.randn(n)
    A0 = torch.eye(n) * 2.0
    b0 = torch.zeros(n)

    A_batch = torch.stack([A0, A1])
    b_batch = torch.stack([b0, b1])
    op = DenseOperator(A_batch, symmetric=False)

    result_full = gmres(op, b_batch, restart=3, max_iter=200)
    assert result_full.iterations.tolist() == [0, 63]
    assert bool(torch.all(result_full.converged))

    result_3 = gmres(op, b_batch, restart=3, max_iter=3)
    assert bool(result_3.converged[0])
    assert not bool(result_3.converged[1])
    assert torch.equal(result_full.x[0], result_3.x[0])  # frozen at iteration 0, bit-identical


def test_gmres_batching_matches_looped():
    torch.manual_seed(9)
    B = 4
    As = [torch.eye(5) + 0.2 * torch.randn(5, 5) for _ in range(B)]
    bs = [torch.randn(5) for _ in range(B)]
    A_batch = torch.stack(As)
    b_batch = torch.stack(bs)
    op_batch = DenseOperator(A_batch, symmetric=False)
    result = gmres(op_batch, b_batch)
    for i in range(B):
        op_i = DenseOperator(As[i], symmetric=False)
        result_i = gmres(op_i, bs[i])
        torch.testing.assert_close(result.x[i], result_i.x, atol=1e-8, rtol=1e-8)


def test_gmres_singular_system_yields_non_converged_status_without_raising():
    # A is exactly rank 1 ([[1, 2], [2, 4]] = [1, 2] outer [1, 2]); b = [1, 3] has a
    # component orthogonal to A's range, so no x solves this exactly. gmres must not raise,
    # must not silently report a plausible wrong convergence, and must name this SINGULAR
    # (an Arnoldi pivot genuinely vanishes -- the second Krylov direction cannot be built)
    # rather than merely MAX_ITER, which would suggest "might converge given more budget".
    A = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    b = torch.tensor([1.0, 3.0])
    op = DenseOperator(A, symmetric=False)
    result = gmres(op, b, max_iter=20)  # does not raise
    assert not bool(result.converged)
    assert int(result.status) == int(SolverStatus.SINGULAR)
    assert result.residual.item() > 1e-3


def test_gmres_happy_breakdown_identity_system():
    # Happy breakdown: A = I (3x3), b = [1, 2, 3]. The Krylov space contains the exact solution
    # after just 1 iteration (the residual r = b - I*x0 = b - 0 = b is already in the span of
    # the single Arnoldi direction v0 = b/||b||, so h[1,0] == 0 exactly). gmres must report
    # this as CONVERGED with iterations == 1, NOT as SINGULAR (which is reserved for rank
    # deficiency that prevents convergence).
    A = torch.eye(3, dtype=torch.float64)
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    op = DenseOperator(A, symmetric=False)
    result = gmres(op, b)
    torch.testing.assert_close(result.x, b, atol=1e-12, rtol=1e-12)
    assert bool(result.converged)
    assert int(result.status) == int(SolverStatus.CONVERGED)
    assert int(result.iterations) == 1


def test_gmres_rejects_restart_below_one():
    # gmres must reject restart < 1 with ValueError (not silently return status=MAX_ITER).
    A = torch.tensor([[3.0, 1.0], [0.5, 2.0]])
    b = torch.tensor([1.0, 2.0])
    op = DenseOperator(A, symmetric=False)
    with pytest.raises(ValueError, match="restart"):
        gmres(op, b, restart=0)


def test_gmres_rejects_max_iter_below_one():
    # gmres must reject max_iter < 1 when explicitly given with ValueError.
    A = torch.tensor([[3.0, 1.0], [0.5, 2.0]])
    b = torch.tensor([1.0, 2.0])
    op = DenseOperator(A, symmetric=False)
    with pytest.raises(ValueError, match="max_iter"):
        gmres(op, b, max_iter=0)


def test_pcg_rejects_max_iter_below_one():
    # pcg must reject max_iter < 1 when explicitly given with ValueError (for consistency
    # with gmres).
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    b = torch.tensor([1.0, 2.0])
    op = DenseOperator(A, symmetric=True)
    with pytest.raises(ValueError, match="max_iter"):
        pcg(op, b, max_iter=0)


def test_gmres_is_differentiable_through_its_iteration_at_cycle_len_above_one():
    """gmres's internal Arnoldi/Givens/back-substitution bookkeeping mutates a handful of
    work tensors (V, H, cs, sn, g, y) in place across iterations; every plain (uncloned)
    slice read from one of those tensors that is later used as an arithmetic operand becomes
    a stale autograd version once anything else writes to the SAME tensor's storage, which
    breaks backward() with 'modified by an inplace operation'. This was invisible as long as
    gmres was only ever called under no_grad(), and also invisible on any n_i=1 fixture
    (cycle_len == 1 skips the back-substitution cross-term loop and the Givens
    apply-earlier-rotations loop entirely, both of which are where the bug lives). A 5x5
    system with restart=3 (cycle_len=3 > 1) is the smallest case that exercises both loops.
    """
    torch.manual_seed(0)
    A = torch.randn(5, 5)
    b = torch.randn(5)
    A_leaf = A.clone().requires_grad_(True)
    op = DenseOperator(A_leaf)
    result = gmres(op, b, restart=3, rtol=1e-13)
    result.x.sum().backward()
    assert torch.isfinite(A_leaf.grad).all()

    def f(A_):
        return gmres(DenseOperator(A_), b, restart=3, rtol=1e-13).x

    A_gc = A.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (A_gc,), eps=1e-6, atol=1e-5)
