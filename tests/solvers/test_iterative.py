"""Tests for pcg (Jacobi-preconditioned CG, per-instance status, never raises) and, appended
by Task 6, gmres (restarted, nonsymmetric-capable, per-instance status, never raises).
"""

import torch

from tellegen.operators.base import SolverStatus
from tellegen.operators.dense import DenseOperator
from tellegen.operators.graph import GraphLaplacianOperator
from tellegen.solvers.iterative import pcg

torch.set_default_dtype(torch.float64)


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
