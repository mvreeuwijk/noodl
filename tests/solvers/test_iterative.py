"""Tests for pcg (Jacobi-preconditioned CG, per-instance status, never raises) and gmres
(restarted, nonsymmetric-capable, per-instance status, never raises), including Task 6's
left preconditioner hook (Jacobi, ILU(0) via SciPy, and a caller-supplied callable).
"""

import pytest
import torch

from noodl.operators.base import SolverStatus
from noodl.operators.dense import DenseOperator
from noodl.operators.graph import GraphLaplacianOperator
from noodl.solvers.iterative import gmres, pcg


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


def _pinned_pcg_system() -> tuple[GraphLaplacianOperator, torch.Tensor]:
    """The fixed 3-instance SPD system the pin below runs on.

    A 6-node chain (edges 0-1-2-3-4-5) grounded at node 0, so `A_I diag(g) A_I^T` is a
    5x5 tridiagonal SPD matrix per instance, with per-instance slopes and right-hand
    sides drawn from ONE seeded generator in a fixed order (slopes, then b) so the system
    is reproducible to the last bit on any run.
    """
    src = torch.tensor([0, 1, 2, 3, 4])
    tgt = torch.tensor([1, 2, 3, 4, 5])
    boundary_mask = torch.zeros(6, dtype=torch.bool)
    boundary_mask[0] = True
    interior_of_node = torch.tensor([-1, 0, 1, 2, 3, 4])
    gen = torch.Generator().manual_seed(20260913)
    slopes = 0.5 + torch.rand(3, 5, generator=gen, dtype=torch.float64)
    b = torch.randn(3, 5, generator=gen, dtype=torch.float64)
    op = GraphLaplacianOperator(
        src, tgt, slopes, 5, interior_of_node, boundary_mask=boundary_mask
    )
    return op, b


# Tolerance for the two `residual` comparisons in `test_pcg_results_are_pinned`, and ONLY
# them: `residual` is a `vector_norm`, i.e. a cross-lane reduction whose summation order is
# the BLAS's, not ours. Measured platform disagreement is 1.2 ulp (see that test's
# docstring); 4 ulps leaves headroom for another BLAS without admitting anything a real
# arithmetic change could hide in -- such a change moves the bit-exact `x` pin first.
_NORM_ULP_RTOL = 4 * torch.finfo(torch.float64).eps


def test_pcg_results_are_pinned():
    """BIT-IDENTICAL pin of `pcg`'s output on a fixed system (milestone-1b follow-up, Task B).

    Task B rewrites `pcg`'s iteration (hoisting allocations, collapsing `torch.where` calls)
    and `GraphLaplacianOperator._apply` (cached indices, one fused `index_add`) for speed
    ALONE: the arithmetic must be untouched, and "untouched" here means the last bit, not
    `assert_close`. Reordering a floating-point accumulation (two scatter-adds into one,
    say) or replacing a `torch.where` with a multiplicative mask changes results in the last
    ulp or, where an inactive instance's step holds inf/nan, catastrophically -- and an
    `rtol`-based comparison would wave both through.

    So these values are HARD-CODED from the pre-change implementation (commit dfd4664). A
    failure here means a change altered the numerics; the fix is to drop that change, never
    to re-record the numbers.

    WHAT THE LITERALS ARE A PIN OF. They are a SINGLE-PLATFORM CPU float64 recording:
    Windows, torch 2.14.0+cpu, 14 threads. `x` is compared BIT-EXACTLY (`rtol=0, atol=0`),
    and that is the instrument: 30 iterate values produced by elementwise arithmetic and
    deterministic scatter-adds, in which a reordered accumulation or a `torch.where` replaced
    by a multiplicative mask cannot hide. `residual` is compared to `_NORM_ULP_RTOL` instead,
    for a reason established by measurement rather than assumed -- see below.

    WHY `residual` IS NOT BIT-EXACT. Run on CI (ubuntu-latest, GitHub Actions, run
    34791144790) both `x` tensors matched BIT-FOR-BIT, and exactly one of the three converged
    `residual` entries differed, by 2.6e-16 relative -- 1.2 ulp. That is the expected
    signature of a cross-lane REDUCTION: `x` comes out of elementwise ops and a scatter-add
    whose order is fixed by the index array, so it is portable, while `residual` is a
    `torch.linalg.vector_norm`, whose summation order is the BLAS's business and legitimately
    differs with vectorisation width and thread count. Pinning it to the last bit pinned
    someone else's reduction order, not noodl's arithmetic -- and for the CONVERGED
    result it pinned the last bit of a ~1e-15 quantity that is pure rounding noise.
    A few-ulp tolerance still catches everything the pin exists for: the mid-iteration
    residual is O(1), and any arithmetic change big enough to move a norm by more than a few
    ulps moves `x` off its bit-exact pin first.

    If another platform disagrees on `x`, or on `residual` by more than a few ulps, that is a
    finding to INVESTIGATE -- print the differing entries and establish whether it is a
    reduction-order difference or a real change -- and never a reason to re-record the
    numbers against that platform, which would destroy the only instrument that can see an
    arithmetic change at all.

    Two runs are pinned. The first (default `max_iter = m = 5`) pins the CONVERGED result,
    where CG's exact-termination property means all three instances finish on the same
    iteration. The second (`max_iter=3`) pins a MID-ITERATION state, which is what actually
    catches an arithmetic change inside the loop: at iteration 3 the iterate is nowhere near
    the solution, so nothing about the converged answer can mask a per-iteration difference.
    """
    op, b = _pinned_pcg_system()

    x_pinned = torch.tensor(
        [
            [
                -2.4221274757646585,
                -3.3616209595087376,
                -6.6874636775904355,
                -9.029323965256463,
                -9.55860251348158,
            ],
            [
                3.1108814899435027,
                6.165292110420957,
                8.902460988513422,
                12.808285046210193,
                14.478491652720464,
            ],
            [
                -3.3262554783401908,
                -4.274940678450948,
                -4.95443455225562,
                -5.23763788904974,
                -6.491224356278421,
            ],
        ],
        dtype=torch.float64,
    )
    residual_pinned = torch.tensor(
        [1.6899695376013564e-15, 7.476049596500161e-16, 7.527059658122819e-16],
        dtype=torch.float64,
    )

    result = pcg(op, b, rtol=1e-10)
    torch.testing.assert_close(result.x, x_pinned, rtol=0, atol=0)
    torch.testing.assert_close(
        result.residual, residual_pinned, rtol=_NORM_ULP_RTOL, atol=0
    )
    assert result.iterations.tolist() == [5, 5, 5]
    assert result.converged.tolist() == [True, True, True]
    assert result.status.tolist() == [int(SolverStatus.CONVERGED)] * 3

    x_mid_pinned = torch.tensor(
        [
            [
                0.019865200684191966,
                -0.01756284653290152,
                -2.9603748835416517,
                -5.108620580339356,
                -6.278209064432335,
            ],
            [
                -0.764399659033867,
                0.4737857339591227,
                3.2823934474299588,
                7.187697429052225,
                7.03022864696632,
            ],
            [
                -3.21169012300437,
                -3.6642680147256703,
                -3.6406502574356834,
                -2.879911546974747,
                -3.2096289500350332,
            ],
        ],
        dtype=torch.float64,
    )
    residual_mid_pinned = torch.tensor(
        [0.9619314042531656, 0.9663629091357048, 0.4525248035016952], dtype=torch.float64
    )

    result_mid = pcg(op, b, rtol=1e-10, max_iter=3)
    torch.testing.assert_close(result_mid.x, x_mid_pinned, rtol=0, atol=0)
    torch.testing.assert_close(
        result_mid.residual, residual_mid_pinned, rtol=_NORM_ULP_RTOL, atol=0
    )
    assert result_mid.iterations.tolist() == [3, 3, 3]
    assert result_mid.converged.tolist() == [False, False, False]
    assert result_mid.status.tolist() == [int(SolverStatus.MAX_ITER)] * 3


# =============================================================================================
# Task 6: gmres's left preconditioner hook (Jacobi, ILU(0) via SciPy, a caller-supplied
# callable). Arnoldi runs on M^-1 A / M^-1 b; the TRUE residual norm(b - A x) is what
# `SolveResult.residual`/`converged` report, recomputed at every cycle end -- never the
# preconditioned Givens estimate. `preconditioner=None` must stay bit-identical to gmres
# before this task: no restructuring of the unpreconditioned path, only a branch added
# around it.
# =============================================================================================


class _SparseTestOperator:
    """`DenseOperator` plus the optional `assemble_sparse` member, so gmres's `"ilu"`
    preconditioner (which requires a sparse form) can be exercised here without pulling in
    `GraphLaplacianOperator`'s own topology. Mirrors `tests/solvers/test_select.py`'s
    `_SparseFakeOperator`: the COO triplet is derived from `A`'s own nonzero pattern, since
    what gmres's ILU hook needs exercised is the CONTRACT (shared int64 indices,
    batch-leading values), not any particular operator's derivation.
    """

    def __init__(self, A: torch.Tensor, *, symmetric: bool = False) -> None:
        self.A = A
        self.shape = tuple(A.shape)
        self.dtype = A.dtype
        self.device = A.device
        self.symmetric = symmetric

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return (self.A @ x.unsqueeze(-1)).squeeze(-1)

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return (self.A.transpose(-1, -2) @ x.unsqueeze(-1)).squeeze(-1)

    def diagonal(self) -> torch.Tensor:
        return torch.diagonal(self.A, dim1=-2, dim2=-1)

    def assemble(self) -> torch.Tensor:
        return self.A

    def assemble_sparse(self):
        if self.A.dim() > 2:
            pattern = (self.A != 0).any(dim=tuple(range(self.A.dim() - 2)))
        else:
            pattern = self.A != 0
        row, col = torch.nonzero(pattern, as_tuple=True)
        values = self.A[..., row, col]
        return row.to(torch.int64), col.to(torch.int64), values

    def spd_certificate(self):
        return None


class _NoSparseFormOperator(_SparseTestOperator):
    def assemble_sparse(self):
        return None


def _diag_spread_system(
    seed: int, m: int = 8, batch: int | None = None, scale: float = 0.1
):
    """`A = D + scale * R`, `D` log-spaced 1e-3..1e3 (a large diagonal spread), `R` random
    and nonsymmetric -- the brief's construction. At the default `m`/`scale` this is mild
    enough for gmres to converge within its default `restart=30` regardless of
    preconditioning (a single Arnoldi cycle covers the whole m-dimensional space either way,
    GMRES's finite-termination property) -- fine for a value comparison against
    `torch.linalg.solve`, but not for an ITERATION-COUNT comparison, which needs `restart <
    m` (see `_stiff_diag_spread_system` below).
    """
    gen = torch.Generator().manual_seed(seed)
    d = torch.logspace(-3, 3, m, dtype=torch.float64)
    if batch is None:
        R = torch.randn(m, m, generator=gen, dtype=torch.float64)
        A = torch.diag(d) + scale * R
        b = torch.randn(m, generator=gen, dtype=torch.float64)
    else:
        R = torch.randn(batch, m, m, generator=gen, dtype=torch.float64)
        A = torch.diag(d).expand(batch, m, m) + scale * R
        b = torch.randn(batch, m, generator=gen, dtype=torch.float64)
    return A, b


def _stiff_diag_spread_system():
    """A harder instance of the same `A = D + scale * R` construction (m=20, scale=0.01,
    seed fixed), together with a `restart=15 < m` that forfeits GMRES's exact
    finite-termination property (which only holds within a single cycle of length >= m):
    measured, unpreconditioned gmres does not converge within 500 matvecs at this restart,
    while Jacobi converges in 13 and ILU in 2 -- a large, reproducible margin for the
    "fewer iterations" comparison the brief asks for.
    """
    A, b = _diag_spread_system(seed=3, m=20, scale=0.01)
    return A, b


def test_gmres_preconditioner_none_is_bit_identical_to_the_default():
    """`preconditioner=None` (explicit) must be bit-for-bit identical to today's default
    (unspecified) gmres -- the non-negotiable pin from the brief."""
    A, b = _diag_spread_system(seed=1, batch=3)
    op = _SparseTestOperator(A, symmetric=False)
    result_default = gmres(op, b, restart=5, max_iter=40)
    result_none = gmres(op, b, restart=5, max_iter=40, preconditioner=None)
    assert torch.equal(result_default.x, result_none.x)
    assert torch.equal(result_default.residual, result_none.residual)
    assert torch.equal(result_default.iterations, result_none.iterations)
    assert torch.equal(result_default.converged, result_none.converged)
    assert torch.equal(result_default.status, result_none.status)


def test_gmres_jacobi_and_ilu_match_torch_linalg_solve_on_a_batched_nonsymmetric_system():
    A, b = _diag_spread_system(seed=2, batch=4)
    op = _SparseTestOperator(A, symmetric=False)
    x_ref = torch.linalg.solve(A, b)

    result_jacobi = gmres(op, b, preconditioner="jacobi", max_iter=100)
    torch.testing.assert_close(result_jacobi.x, x_ref, atol=1e-10, rtol=1e-10)
    assert bool(torch.all(result_jacobi.converged))

    result_ilu = gmres(op, b, preconditioner="ilu", max_iter=100)
    torch.testing.assert_close(result_ilu.x, x_ref, atol=1e-10, rtol=1e-10)
    assert bool(torch.all(result_ilu.converged))


def test_gmres_jacobi_and_ilu_reduce_iterations_versus_unpreconditioned():
    """The whole point of the preconditioner: on a system with a large diagonal spread AND
    a restart short enough to forfeit GMRES's exact finite-termination property, both
    Jacobi and ILU must converge in FEWER iterations than plain gmres -- which, at this
    restart, does not converge within the given budget at all.
    """
    A, b = _stiff_diag_spread_system()
    op = _SparseTestOperator(A, symmetric=False)

    result_none = gmres(op, b, preconditioner=None, restart=15, max_iter=500)
    result_jacobi = gmres(op, b, preconditioner="jacobi", restart=15, max_iter=500)
    result_ilu = gmres(op, b, preconditioner="ilu", restart=15, max_iter=500)

    assert not bool(result_none.converged), "the unpreconditioned case must stall"
    assert bool(result_jacobi.converged)
    assert bool(result_ilu.converged)
    n_none = int(result_none.iterations)
    n_jacobi = int(result_jacobi.iterations)
    n_ilu = int(result_ilu.iterations)
    print(f"none={n_none} jacobi={n_jacobi} ilu={n_ilu}")
    assert n_jacobi < n_none, f"jacobi={n_jacobi} none={n_none}"
    assert n_ilu < n_none, f"ilu={n_ilu} none={n_none}"
    assert n_ilu < n_jacobi, f"ilu={n_ilu} jacobi={n_jacobi}"


def test_gmres_callable_preconditioner_is_applied():
    """A callable preconditioner is applied as given: a uniform scalar left-preconditioner
    does not change the mathematical solution (same Krylov subspace, a rescaled residual
    minimisation), so this both proves the callable path runs and that it is wired correctly
    (a bug that ignored the callable, or applied it to the wrong operand, would generally
    still "converge" to something -- but not to the right x to tight tolerance from a
    poorly-scaled start).
    """
    A, b = _diag_spread_system(seed=4)
    op = _SparseTestOperator(A, symmetric=False)
    x_ref = torch.linalg.solve(A, b)

    calls = {"count": 0}

    def scale_by_three(v: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        return 3.0 * v

    result = gmres(op, b, preconditioner=scale_by_three, max_iter=200)
    assert calls["count"] > 0
    torch.testing.assert_close(result.x, x_ref, atol=1e-8, rtol=1e-8)


def test_gmres_reports_the_true_residual_not_the_preconditioned_one():
    """`SolveResult.residual` must be `norm(b - A x) / norm(b)`, recomputed explicitly --
    never the preconditioned Givens estimate -- so the tolerance keeps its meaning."""
    A, b = _diag_spread_system(seed=5)
    op = _SparseTestOperator(A, symmetric=False)
    # A small max_iter well short of convergence: the reported residual must still be the
    # TRUE one at this partial state, not an artifact of the preconditioned scale.
    result = gmres(op, b, preconditioner="jacobi", max_iter=2)
    assert not bool(result.converged)
    r_true = b - op.matvec(result.x)
    expected_residual = torch.linalg.vector_norm(r_true) / torch.linalg.vector_norm(b)
    torch.testing.assert_close(result.residual, expected_residual, atol=1e-13, rtol=1e-12)


def test_gmres_per_instance_convergence_and_freezing_with_a_preconditioner():
    """Per-instance status/freezing (the pre-existing contract) is unaffected by adding a
    preconditioner: instance 0 (b=0, trivially solved) must freeze bit-identically whether
    or not its sibling is still iterating."""
    torch.manual_seed(11)
    n = 10
    A1, b1 = _diag_spread_system(seed=6, m=n)
    A0 = torch.eye(n, dtype=torch.float64) * 2.0
    b0 = torch.zeros(n, dtype=torch.float64)
    A_batch = torch.stack([A0, A1])
    b_batch = torch.stack([b0, b1])
    op = _SparseTestOperator(A_batch, symmetric=False)

    result_full = gmres(op, b_batch, preconditioner="jacobi", restart=5, max_iter=200)
    assert bool(torch.all(result_full.converged))
    assert int(result_full.iterations[0]) == 0

    result_3 = gmres(op, b_batch, preconditioner="jacobi", restart=5, max_iter=3)
    assert bool(result_3.converged[0])
    assert not bool(result_3.converged[1])
    assert torch.equal(result_full.x[0], result_3.x[0])


def test_gmres_ilu_refuses_by_name_without_a_sparse_form():
    A, b = _diag_spread_system(seed=7)
    op = _NoSparseFormOperator(A, symmetric=False)
    with pytest.raises(ValueError, match="assemble_sparse"):
        gmres(op, b, preconditioner="ilu")


def test_gmres_rejects_unknown_preconditioner_name():
    A, b = _diag_spread_system(seed=8)
    op = _SparseTestOperator(A, symmetric=False)
    with pytest.raises(ValueError, match="preconditioner"):
        gmres(op, b, preconditioner="bogus")
