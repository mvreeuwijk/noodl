"""Batched iterative solvers behind the operator/solver contract: pcg (SPD) and gmres
(nonsymmetric). Neither ever raises; both return SolveResult unconditionally, per
instance -- the raise/return boundary lives in solvers/select.py, one layer up.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from noodl.operators.base import SolveResult, SolverStatus

Tensor = torch.Tensor


def _result(
    x: Tensor,
    converged: Tensor,
    iterations: Tensor,
    residual_norm: Tensor,
    b_norm: Tensor,
    *,
    flag: Tensor,
    flag_status: SolverStatus,
) -> SolveResult:
    """Shared per-instance status/residual construction for `pcg` and `gmres`: CONVERGED
    where `converged`, else `flag_status` where the solver-specific failure `flag` (pcg's
    `breakdown`, gmres's `exhausted`) fires, else MAX_ITER; residual is the relative norm
    `residual_norm / b_norm`, falling back to the raw `residual_norm` when `b_norm` is zero.
    """
    device = x.device
    batch_shape = converged.shape
    status = torch.where(
        converged,
        torch.full(batch_shape, int(SolverStatus.CONVERGED), dtype=torch.long, device=device),
        torch.where(
            flag,
            torch.full(batch_shape, int(flag_status), dtype=torch.long, device=device),
            torch.full(batch_shape, int(SolverStatus.MAX_ITER), dtype=torch.long, device=device),
        ),
    )
    residual = torch.where(b_norm > 0, residual_norm / b_norm, residual_norm)
    return SolveResult(
        x=x, converged=converged, iterations=iterations, residual=residual, status=status
    )


def pcg(
    op,
    b: Tensor,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_iter: int | None = None,
    preconditioner: str | None = "jacobi",
    x0: Tensor | None = None,
) -> SolveResult:
    """Jacobi-preconditioned CG, per-instance convergence, status and freezing.

    Convergence: ||r|| <= max(rtol * ||b||, atol), per instance. An instance whose search
    direction sees non-positive curvature (p.Ap <= 0) is marked BREAKDOWN rather than
    clamped -- clamping (as an earlier benchmark in this codebase did) would silently hide a
    loss of definiteness instead of reporting it. CONVERGED and BREAKDOWN instances FREEZE:
    once either fires, that instance's x/r/p are never updated again, matching newton()'s own
    per-instance freezing convention exactly.
    """
    if max_iter is not None and max_iter < 1:
        raise ValueError(f"pcg: max_iter must be >= 1 when given, got {max_iter!r}")

    m = op.shape[-1]
    batch_shape = torch.broadcast_shapes(op.shape[:-2], b.shape[:-1])
    dtype, device = b.dtype, b.device
    b = b.expand(batch_shape + (m,))
    if max_iter is None:
        max_iter = m
    if x0 is None:
        x = torch.zeros(batch_shape + (m,), dtype=dtype, device=device)
    else:
        x = x0.expand(batch_shape + (m,)).clone()

    if preconditioner == "jacobi":
        diag = op.diagonal().expand(batch_shape + (m,))
        m_inv = 1.0 / diag
    elif preconditioner is None:
        m_inv = torch.ones(batch_shape + (m,), dtype=dtype, device=device)
    else:
        raise ValueError(f"pcg: unknown preconditioner {preconditioner!r}")

    b_norm = torch.linalg.vector_norm(b, dim=-1)
    tol = torch.clamp(rtol * b_norm, min=atol)

    r = b - op.matvec(x)
    z = m_inv * r
    p = z.clone()
    rz_old = (r * z).sum(-1)

    norm_r = torch.linalg.vector_norm(r, dim=-1)
    converged = norm_r <= tol
    breakdown = torch.zeros_like(converged)
    iterations = torch.zeros(batch_shape, dtype=torch.long, device=device)

    # Hoisted out of the iteration: at ensemble 1 this loop
    # is DISPATCH bound -- ~40 tensor ops per iteration on 1028-element vectors, each costing
    # more in Python/ATen dispatch than in arithmetic -- so every op removed from the body is
    # a real saving, and none of the removals below changes a single bit of the result.
    #
    # `ones` replaces the two per-iteration `torch.ones_like` allocations. It is only ever
    # SELECTED by `torch.where` (never scaled, never added), so one shared constant is
    # bit-identical to a fresh one each time.
    ones = torch.ones(batch_shape, dtype=dtype, device=device)
    # `done` is the loop's own exit test AND the complement of `active`; computing it once
    # per iteration replaces the `converged | breakdown` that used to be evaluated twice.
    done = converged | breakdown

    it = 0
    while not bool(done.all()) and it < max_iter:
        active = ~done
        Ap = op.matvec(p)
        pAp = (p * Ap).sum(-1)
        new_breakdown = active & (pAp <= 0)
        breakdown = breakdown | new_breakdown
        # `new_breakdown` is a SUBSET of `active`, so `active ^ new_breakdown` is exactly
        # the `active & ~new_breakdown` it replaces, in one boolean op instead of two.
        active = active ^ new_breakdown
        active_col = active.unsqueeze(-1)

        pAp_safe = torch.where(pAp != 0, pAp, ones)
        alpha = (rz_old / pAp_safe).unsqueeze(-1)
        # These three updates keep `torch.where` and must: a MULTIPLICATIVE `active` mask is
        # NOT bit-equivalent here. A frozen instance's `p`/`Ap` can hold inf (a `beta` that
        # overflowed on the very iteration it froze on, say), and 0 * inf is nan -- masking
        # the STEP would poison an `x`/`r`/`p` that, by the per-instance freezing contract,
        # must never change again. The mask has to SELECT the old value, not scale the new
        # one. Only `active_col` is hoisted (one `unsqueeze` instead of three).
        x = torch.where(active_col, x + alpha * p, x)
        r = torch.where(active_col, r - alpha * Ap, r)

        norm_r = torch.linalg.vector_norm(r, dim=-1)
        converged = converged | (active & (norm_r <= tol))

        z = m_inv * r
        rz_new = (r * z).sum(-1)
        rz_old_safe = torch.where(rz_old != 0, rz_old, ones)
        beta = (rz_new / rz_old_safe).unsqueeze(-1)
        p = torch.where(active_col, z + beta * p, p)
        # `rz_new` can be nan for a frozen instance too (an infinite Jacobi diagonal makes
        # `z` infinite while `r` stays finite), so this one stays a `torch.where` as well.
        rz_old = torch.where(active, rz_new, rz_old)

        # The one counter that CAN drop its `torch.where`: `active` promotes to 0/1 and
        # int64 addition is exact, so this is bit-identical to
        # `torch.where(active, iterations + 1, iterations)` with no inf/nan hazard to carry.
        iterations = iterations + active
        done = converged | breakdown
        it += 1

    return _result(
        x, converged, iterations, norm_r, b_norm, flag=breakdown, flag_status=SolverStatus.BREAKDOWN
    )


def _gmres_jacobi_preconditioner(op, batch_shape: torch.Size, m: int) -> Callable[[Tensor], Tensor]:
    """gmres's `"jacobi"` preconditioner: `M^-1 = 1 / diag(A)`, broadcast over the batch and
    flattened to gmres's own `(B, m)` working shape -- built once per call, applied every
    Arnoldi step as an elementwise product (no dependence on `k`).
    """
    diag = op.diagonal().expand(*batch_shape, m)
    m_inv_flat = (1.0 / diag).reshape(-1, m)

    def apply(v_flat: Tensor) -> Tensor:
        return m_inv_flat * v_flat

    return apply


def _gmres_ilu_preconditioner(
    op, batch_shape: torch.Size, B: int, m: int, dtype: torch.dtype, device: torch.device
) -> Callable[[Tensor], Tensor]:
    """gmres's `"ilu"` preconditioner: SciPy's incomplete LU with `drop_tol=0, fill_factor=1`
    (ILU(0) in effect: nothing is dropped, so the factorisation keeps every fill entry),
    factorised PER INSTANCE from `op.assemble_sparse()`'s COO triplet -- the same per-instance
    Python loop and batch-flattening `solvers.select._sparse_direct` uses, reused here rather
    than reinvented.

    COST: this function is called ONCE PER `gmres()` CALL,
    and it factorises `spilu` PER INSTANCE in a Python loop, same as `_sparse_direct` -- there
    is no batched SuperLU entry point, so the cost is linear in the ensemble size, same shape
    as `_sparse_direct`'s documented limitation. Unlike `_sparse_direct`, this is NOT a one-off
    per solve: `_LinearSolve.forward` and `.backward` (`noodl.operators.solve`) each call
    `gmres` independently, so a single differentiable transport step under `preconditioner=
    "ilu"` factorises TWICE per step -- once for the forward solve, once for the adjoint --
    on every step, not once and reused. Compare with `sparse_direct`: one full LU per solve,
    then an exact (to rounding) solve with no outer iteration; `"ilu"` instead pays a
    (cheaper, incomplete) factorisation on both passes and then still iterates GMRES to
    convergence, so the trade is fewer, better-conditioned Arnoldi steps against strictly
    more total factorisations than `sparse_direct` needs. See `TransportLayer._resolve_solver`
    for how this reaches a transport step, and `gmres`'s own docstring for the same note next
    to `"ilu"`.

    Both refusals below are `ValueError`, not `ImportError`/`AttributeError`: `"ilu"` is an
    EXPLICIT request (never a fall-back, unlike `select.solve`'s `"auto"`), so a missing
    capability is refused by name rather than silently substituted.
    """
    assemble_sparse = getattr(op, "assemble_sparse", None)
    triplet = assemble_sparse() if assemble_sparse is not None else None
    if triplet is None:
        raise ValueError(
            f"gmres: preconditioner='ilu' requires op.assemble_sparse() to return a "
            f"(row, col, values) COO triplet, but this operator "
            f"{'returned None' if assemble_sparse is not None else 'has no assemble_sparse'}"
        )
    try:
        import scipy.sparse
        import scipy.sparse.linalg
    except ImportError as exc:
        raise ValueError(
            f"gmres: preconditioner='ilu' requires scipy (scipy.sparse and "
            f"scipy.sparse.linalg), which could not be imported: {exc}. Install scipy, or "
            f"use preconditioner=None, 'jacobi' or a callable."
        ) from exc

    row, col, values = triplet
    nnz = int(row.shape[-1])
    values_flat = values.expand(*batch_shape, nnz).reshape(B, nnz)
    row_np = row.detach().cpu().numpy()
    col_np = col.detach().cpu().numpy()
    # SuperLU has single- and double-precision kernels only; factorise in float64 and cast
    # back, mirroring `_sparse_direct`'s handling of the same constraint.
    values_np = values_flat.detach().cpu().to(torch.float64).numpy()

    lus = []
    for i in range(B if m > 0 else 0):
        A_i = scipy.sparse.csc_matrix((values_np[i], (row_np, col_np)), shape=(m, m))
        lus.append(scipy.sparse.linalg.spilu(A_i, drop_tol=0, fill_factor=1))

    def apply(v_flat: Tensor) -> Tensor:
        v_np = v_flat.detach().cpu().to(torch.float64).numpy()
        out_np = v_np.copy()
        for i in range(B if m > 0 else 0):
            out_np[i] = lus[i].solve(v_np[i])
        return torch.from_numpy(out_np).to(device=device, dtype=dtype)

    return apply


def _gmres_callable_preconditioner(
    fn: Callable[[Tensor], Tensor], batch_shape: torch.Size, m: int, B: int
) -> Callable[[Tensor], Tensor]:
    """Wraps a caller-supplied preconditioner (applied "as given") so it can
    sit next to `mv` in gmres's flattened-batch working shape: reshape flat -> the caller's
    natural `batch_shape + (m,)`, call it, reshape back.
    """

    def apply(v_flat: Tensor) -> Tensor:
        v = v_flat.reshape(*batch_shape, m)
        out = fn(v)
        return out.reshape(B, m)

    return apply


def gmres(
    op,
    b: Tensor,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_iter: int | None = None,
    restart: int = 30,
    x0: Tensor | None = None,
    preconditioner: str | Callable[[Tensor], Tensor] | None = None,
) -> SolveResult:
    """Restarted GMRES for nonsymmetric operators: Arnoldi (modified Gram-Schmidt) with
    incremental Givens rotations, per-instance status, never raises.

    An instance whose Arnoldi breaks down -- `h_next` at rounding level RELATIVE to
    `||A v_k||`, which is the invariant-Krylov-space ("lucky breakdown") case a repeated
    spectrum produces by construction -- ends its cycle there, at the k+1 system that holds
    the exact solution, instead of extending the basis with noise. Breakdown is per
    instance, like convergence: siblings keep iterating. `exhausted` (reported as SINGULAR)
    is reserved for a breakdown that TRUNCATED a cycle and still did not converge, i.e.
    genuine rank deficiency -- not a lucky breakdown, and not a basis that merely ran out
    of room at the last step of its cycle, which is what an ill-conditioned but nonsingular
    system does routinely.

    Implementation flattens every leading batch dimension into one axis `B` for the duration
    of the Arnoldi/Givens bookkeeping (reshaped back to the caller's batch shape at the end):
    the Hessenberg system, Givens coefficients and Krylov basis are naturally rectangular
    per-instance state that is far simpler to index with one flat batch axis than with
    arbitrary leading dims, unlike pcg's per-element vector ops which need no such reshape.

    `preconditioner` is a LEFT preconditioner: Arnoldi runs on `M^-1 A` with
    `M^-1 b`, so the per-cycle Givens residual estimate (`g[:, k+1]`) is in the
    PRECONDITIONED scale. The quantity that gates convergence -- `beta`/`r`, compared
    against `tol` -- is always the TRUE, unpreconditioned residual `b - A x`, recomputed via
    the ordinary (unpreconditioned) `mv` at every cycle end and at exit;
    `preconditioner=None` takes the plain code path (`r`/`beta` are aliased, not
    recomputed). `"jacobi"` is `1 / diag(A)`; `"ilu"` is SciPy's
    incomplete LU of `op.assemble_sparse()`, per instance (see `_gmres_ilu_preconditioner`,
    including its FACTORISATION COST -- read that before choosing `"ilu"`); a callable is
    applied as given, on vectors of the operator's own `batch_shape + (m,)`. Breakdown
    detection is unaffected in meaning: it operates on whatever Arnoldi sees, which is now
    `M^-1 A v_k`.

    `iterations`: unpreconditioned, this is the exact within-cycle step at
    which the Givens estimate first met `tol`, because that estimate already IS the
    true-scale residual there. PRECONDITIONED,
    the Givens estimate is in `M^-1`-weighted units while `tol` is built from `||b||`, so
    the two scales generally disagree and the in-cycle estimate cannot be trusted to report
    the true convergence step -- measured on this module's own stiff fixture
    (`tests/solvers/test_iterative.py::_stiff_diag_spread_system`, Jacobi, `restart=15`),
    the estimate crossed `tol` at step 13 while the TRUE residual, recomputed at cycle end,
    first did at step 14. `iterations` is therefore CYCLE-GRANULAR under a preconditioner:
    the matvec count through the end of whichever cycle's TRUE recompute first satisfied
    `tol` (`total_matvecs + cycle_len` for that cycle), never the in-cycle estimate. This
    can overshoot the true step (by up to `restart - 1`) but is guaranteed to never
    undershoot it, unlike the in-cycle estimate.

    `"jacobi"` divides by `op.diagonal()` unguarded: a zero diagonal entry produces `inf`
    then `nan` in that entry of `M^-1 r`, which fails safe through the true-residual gate
    (an instance whose Arnoldi state has gone non-finite will not satisfy `beta <= tol` and
    so is reported MAX_ITER, never a wrong CONVERGED) but silently -- there is no explicit
    check or message, matching pcg's own pre-existing Jacobi. A caller with a zero (or
    near-zero) diagonal entry should treat `"jacobi"` as unusable for that operator and use
    `None`, `"ilu"` or a callable instead.
    """
    if restart < 1:
        raise ValueError(f"gmres: restart must be >= 1, got {restart!r}")
    if max_iter is not None and max_iter < 1:
        raise ValueError(f"gmres: max_iter must be >= 1 when given, got {max_iter!r}")

    m = op.shape[-1]
    batch_shape = torch.broadcast_shapes(op.shape[:-2], b.shape[:-1])
    dtype, device = b.dtype, b.device
    b = b.expand(batch_shape + (m,))
    if x0 is None:
        x = torch.zeros(batch_shape + (m,), dtype=dtype, device=device)
    else:
        x = x0.expand(batch_shape + (m,)).clone()
    if max_iter is None:
        max_iter = m

    B = 1
    for s in batch_shape:
        B *= s
    B = max(B, 1)
    x_flat = x.reshape(B, m)
    b_flat = b.reshape(B, m)

    def mv(v_flat: Tensor) -> Tensor:
        v = v_flat.reshape(batch_shape + (m,))
        return op.matvec(v).reshape(B, m)

    # Built once per call, applied every Arnoldi step. `precond_apply is None` is the
    # plain path: `mv_pre` becomes `mv` itself (no wrapper, no extra call) -- see the
    # docstring.
    if preconditioner is None:
        precond_apply = None
    elif callable(preconditioner):
        precond_apply = _gmres_callable_preconditioner(preconditioner, batch_shape, m, B)
    elif preconditioner == "jacobi":
        precond_apply = _gmres_jacobi_preconditioner(op, batch_shape, m)
    elif preconditioner == "ilu":
        precond_apply = _gmres_ilu_preconditioner(op, batch_shape, B, m, dtype, device)
    else:
        raise ValueError(f"gmres: unknown preconditioner {preconditioner!r}")
    mv_pre = mv if precond_apply is None else (lambda v: precond_apply(mv(v)))

    b_norm = torch.linalg.vector_norm(b_flat, dim=-1)
    tol = torch.clamp(rtol * b_norm, min=atol)

    r = b_flat - mv(x_flat)
    beta = torch.linalg.vector_norm(r, dim=-1)
    converged = beta <= tol
    exhausted = torch.zeros(B, dtype=torch.bool, device=device)
    total_matvecs = torch.zeros(B, dtype=torch.long, device=device)
    iterations = torch.zeros(B, dtype=torch.long, device=device)
    tiny = torch.finfo(dtype).tiny
    # Arnoldi breakdown is a RELATIVE condition, never an absolute one. `h_next` is what
    # modified Gram-Schmidt leaves of `A v_k` after projecting out the existing basis, so
    # the only meaningful scale to test it against is `||A v_k||` itself. Once that ratio
    # reaches rounding level the remainder is arithmetic noise, not a new Krylov direction,
    # and dividing by it fills `V` with a vector orthogonal to nothing -- which corrupts
    # the Hessenberg matrix, leaves a degenerate (but not `tiny`) diagonal in its rotated
    # form, and lets back-substitution amplify it without bound. A `finfo.tiny` test
    # (2.2e-308 in float64) never fires on that: on the multi-species transport systems
    # that motivated this, the ratio lands at ~2e-16, i.e. about `eps` and 292 orders of
    # magnitude above `tiny`.
    #
    # `eps ** 0.75` (1.8e-12 in float64) sits ~3 orders above every breakdown ratio
    # measured (2e-16 .. 1e-14) and ~2 orders below the smallest LEGITIMATE ratio measured
    # over 1000 random dense systems of size 4..64, ill-conditioned ones included
    # (2.6e-10). Erring loose is the safe direction: the cycle's `x` is accepted or
    # rejected on the residual `b - A x` RECOMPUTED explicitly below, never on the Givens
    # estimate, so a premature truncation costs at most one extra restart cycle and can
    # never report a wrong answer as CONVERGED.
    break_rtol = torch.finfo(dtype).eps ** 0.75

    while (not bool(torch.all(converged))) and int(total_matvecs.max().item()) < max_iter:
        active = ~converged
        cycle_len = min(restart, max_iter - int(total_matvecs.max().item()))
        if cycle_len <= 0:
            break

        V = torch.zeros(B, cycle_len + 1, m, dtype=dtype, device=device)
        H = torch.zeros(B, cycle_len + 1, cycle_len, dtype=dtype, device=device)
        cs = torch.zeros(B, cycle_len, dtype=dtype, device=device)
        sn = torch.zeros(B, cycle_len, dtype=dtype, device=device)
        g = torch.zeros(B, cycle_len + 1, dtype=dtype, device=device)

        # `r0`/`beta0` are what Arnoldi actually starts from: the TRUE residual `r`/`beta`
        # when unpreconditioned (aliased, not recomputed -- this is the bit-identical path),
        # or the PRECONDITIONED residual `M^-1 r` otherwise. `r`/`beta` themselves are never
        # touched here: they still hold the true residual from the last cycle end (or the
        # initial one), which is what the outer `converged`/`active` test above is based on.
        if precond_apply is None:
            r0, beta0 = r, beta
        else:
            r0 = precond_apply(r)
            beta0 = torch.linalg.vector_norm(r0, dim=-1)
        beta0_safe = torch.where(beta0 > 0, beta0, torch.ones_like(beta0))
        V[:, 0, :] = torch.where(active.unsqueeze(-1), r0 / beta0_safe.unsqueeze(-1), V[:, 0, :])
        g[:, 0] = torch.where(active, beta0, g[:, 0])

        step_converged_at = torch.full((B,), cycle_len, dtype=torch.long, device=device)

        # Per-instance, per-cycle breakdown. An instance that breaks down at step k has an
        # INVARIANT Krylov space: its (k+1)-dimensional least-squares problem already holds
        # the best available solution (the EXACT one, in exact arithmetic), so the cycle
        # must stop there FOR THAT INSTANCE rather than extend the basis with noise -- its
        # siblings, which have not broken down, still need the remaining steps, which is
        # why this is a mask and not a `break`. Freezing the instance leaves its H columns,
        # its Givens pair and its `g` entries at zero beyond k, and that is exactly what
        # collapses the back-substitution below onto the leading (k+1)x(k+1) system: `y` is
        # exactly zero wherever the rotated diagonal is zero, and those zero entries drop
        # out of every earlier row's cross terms.
        broken = torch.zeros(B, dtype=torch.bool, device=device)

        for k in range(cycle_len):
            live = active & ~broken
            live_col = live.unsqueeze(-1)
            # V is read here and mutated again later (a new column, but the SAME tensor's
            # storage) before backward runs; every operand read from it must be `.clone()`d
            # first, or autograd's version counter (tracked per-storage, not per-slice) sees
            # a mismatch at backward time. Same reasoning as the `h_j`/`h_j1`/`g_k` clones
            # below and the `h_kk` clone above.
            # `mv_pre` is `M^-1 A` (or plain `A` when unpreconditioned) -- Arnoldi's own
            # matvec, distinct from the plain `mv` used below to recompute the TRUE
            # residual at cycle end. Breakdown detection (`w_norm`, `h_next` below) operates
            # on whatever this is, per the docstring.
            w = mv_pre(V[:, k, :].clone())
            w_norm = torch.linalg.vector_norm(w, dim=-1)
            for j in range(k + 1):
                v_j = V[:, j, :].clone()
                h_jk = torch.einsum("bi,bi->b", w, v_j)
                H[:, j, k] = torch.where(live, h_jk, H[:, j, k])
                w = w - h_jk.unsqueeze(-1) * v_j
            h_next = torch.linalg.vector_norm(w, dim=-1)
            newly_broken = live & (h_next <= break_rtol * w_norm)
            broken = broken | newly_broken
            # `exhausted` (reported as SINGULAR) is a DIAGNOSIS, and it is only earned by a
            # breakdown that actually TRUNCATES the cycle. At the final step of a cycle the
            # test above is a numerical no-op -- the basis has simply completed, there is
            # no step left to take, and nothing is discarded -- and an ill-conditioned but
            # perfectly nonsingular system reaches exactly that state routinely: measured,
            # every one of 120 systems at cond 1e8 and m = 4, 8, 16 ends its last Arnoldi
            # step under the relative threshold. Flagging those SINGULAR would point a user
            # at rank deficiency when the truth is conditioning. `broken` above is NOT
            # gated the same way: the truncation bookkeeping is a no-op at the last step
            # anyway, and keeping it ungated leaves the freezing rule uniform.
            newly_exhausted = newly_broken & (k + 1 < cycle_len)
            exhausted = exhausted | newly_exhausted
            # A detected breakdown is treated as the exact one it numerically is: forcing
            # `h_next` to zero makes the Givens rotation below zero `g[k+1]` outright, so
            # the k+1 system's residual estimate is exactly zero and the cycle's answer is
            # that system's solution. `h_next_safe` then falls back to 1, leaving
            # `V[:, k+1, :]` holding the unnormalised noise remainder -- harmless, because
            # `H[k+1, k+1]` stays zero for a frozen instance and back-substitution
            # therefore multiplies that column by an exactly-zero `y`.
            h_next = torch.where(newly_broken, torch.zeros_like(h_next), h_next)
            h_next_safe = torch.where(h_next > tiny, h_next, torch.ones_like(h_next))
            v_next = w / h_next_safe.unsqueeze(-1)
            V[:, k + 1, :] = torch.where(live_col, v_next, V[:, k + 1, :])

            # apply every earlier cycle's Givens rotation to this new Hessenberg column.
            # cs/sn are read here (for j < k, set in an earlier k-iteration) and mutated
            # again below (cs[:, k], sn[:, k]) before backward runs -- the same
            # read-then-mutate-same-storage hazard, so these reads need `.clone()` too.
            for j in range(k):
                h_j = H[:, j, k].clone()
                h_j1 = H[:, j + 1, k].clone()
                cs_j = cs[:, j].clone()
                sn_j = sn[:, j].clone()
                H[:, j, k] = torch.where(live, cs_j * h_j + sn_j * h_j1, H[:, j, k])
                H[:, j + 1, k] = torch.where(
                    live, -sn_j * h_j + cs_j * h_j1, H[:, j + 1, k]
                )

            h_kk = H[:, k, k].clone()
            h_k1k = h_next
            denom = torch.sqrt(h_kk**2 + h_k1k**2)
            denom_safe = torch.where(denom > tiny, denom, torch.ones_like(denom))
            cs_k = torch.where(denom > tiny, h_kk / denom_safe, torch.ones_like(denom))
            sn_k = torch.where(denom > tiny, h_k1k / denom_safe, torch.zeros_like(denom))
            cs[:, k] = torch.where(live, cs_k, cs[:, k])
            sn[:, k] = torch.where(live, sn_k, sn[:, k])
            H[:, k, k] = torch.where(live, cs_k * h_kk + sn_k * h_k1k, H[:, k, k])

            g_k = g[:, k].clone()
            g[:, k] = torch.where(live, cs_k * g_k, g[:, k])
            g[:, k + 1] = torch.where(live, -sn_k * g_k, g[:, k + 1])

            residual_est = g[:, k + 1].abs()
            just_met = live & (residual_est <= tol) & (step_converged_at == cycle_len)
            step_converged_at = torch.where(
                just_met, torch.full_like(step_converged_at, k + 1), step_converged_at
            )

        # back-substitution: upper-triangular H[:, :cycle_len, :cycle_len] @ y = g[:, :cycle_len]
        # y is written at the end of each k-iteration below and read (for larger j, already
        # computed) by every subsequent iteration -- the identical read-then-mutate-same-
        # storage hazard as V/H above, so every operand read from it needs `.clone()` too.
        y = torch.zeros(B, cycle_len, dtype=dtype, device=device)
        for k in range(cycle_len - 1, -1, -1):
            s = g[:, k].clone()
            for j in range(k + 1, cycle_len):
                s = s - H[:, k, j].clone() * y[:, j].clone()
            diag = H[:, k, k]
            diag_safe = torch.where(diag.abs() > tiny, diag, torch.ones_like(diag))
            y[:, k] = torch.where(diag.abs() > tiny, s / diag_safe, torch.zeros_like(s))

        dx = torch.einsum("bk,bkm->bm", y, V[:, :cycle_len, :])
        x_candidate = x_flat + dx
        x_flat = torch.where(active.unsqueeze(-1), x_candidate, x_flat)

        r = b_flat - mv(x_flat)
        beta = torch.linalg.vector_norm(r, dim=-1)
        newly_converged = active & (beta <= tol)

        # `step_converged_at` is measured in PRECONDITIONED units (the Givens estimate is
        # ||M^-1(b - A x_k)||, compared above against `tol`, which is built from the
        # TRUE-scale ||b||): unpreconditioned, that estimate already IS the true-scale
        # residual, so the within-cycle count is exact and this is the untouched, bit-
        # identical original expression. Preconditioned, the two scales generally
        # disagree -- reproduced on the shipped stiff fixture, where the estimate crosses
        # `tol` one step before the TRUE residual (recomputed as `beta` just above) actually
        # does -- so the within-cycle count cannot be trusted there; this reports the
        # CYCLE-GRANULAR count instead (every matvec actually spent in the cycle that
        # produced the `x` whose true residual just met `tol`), which can only be >= the
        # true step and is therefore never a false "converged early".
        cycle_count = (
            step_converged_at.clamp(max=cycle_len)
            if precond_apply is None
            else torch.full_like(step_converged_at, cycle_len)
        )
        iterations = torch.where(active, total_matvecs + cycle_count, iterations)
        total_matvecs = total_matvecs + torch.where(
            active, torch.full_like(total_matvecs, cycle_len), torch.zeros_like(total_matvecs)
        )
        converged = converged | newly_converged

    iterations = torch.where(converged, iterations, total_matvecs)

    x_out = x_flat.reshape(batch_shape + (m,))
    return _result(
        x_out,
        converged.reshape(batch_shape),
        iterations.reshape(batch_shape),
        beta.reshape(batch_shape),
        b_norm.reshape(batch_shape),
        flag=exhausted.reshape(batch_shape),
        flag_status=SolverStatus.SINGULAR,
    )
