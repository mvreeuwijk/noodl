"""Batched iterative solvers behind the operator/solver contract: pcg (SPD, Task 5) and gmres
(nonsymmetric, Task 6). Neither ever raises; both return SolveResult unconditionally, per
instance -- the raise/return boundary lives in solvers/select.py (Task 7), one layer up.
"""

from __future__ import annotations

import torch

from tellegen.operators.base import SolveResult, SolverStatus

Tensor = torch.Tensor


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

    it = 0
    while not bool(torch.all(converged | breakdown)) and it < max_iter:
        active = ~(converged | breakdown)
        Ap = op.matvec(p)
        pAp = (p * Ap).sum(-1)
        new_breakdown = active & (pAp <= 0)
        breakdown = breakdown | new_breakdown
        active = active & ~new_breakdown

        pAp_safe = torch.where(pAp != 0, pAp, torch.ones_like(pAp))
        alpha = rz_old / pAp_safe
        x = torch.where(active.unsqueeze(-1), x + alpha.unsqueeze(-1) * p, x)
        r = torch.where(active.unsqueeze(-1), r - alpha.unsqueeze(-1) * Ap, r)

        norm_r = torch.linalg.vector_norm(r, dim=-1)
        converged = converged | (active & (norm_r <= tol))

        z = m_inv * r
        rz_new = (r * z).sum(-1)
        rz_old_safe = torch.where(rz_old != 0, rz_old, torch.ones_like(rz_old))
        beta = rz_new / rz_old_safe
        p = torch.where(active.unsqueeze(-1), z + beta.unsqueeze(-1) * p, p)
        rz_old = torch.where(active, rz_new, rz_old)

        iterations = torch.where(active, iterations + 1, iterations)
        it += 1

    status = torch.where(
        converged,
        torch.full(batch_shape, int(SolverStatus.CONVERGED), dtype=torch.long, device=device),
        torch.where(
            breakdown,
            torch.full(batch_shape, int(SolverStatus.BREAKDOWN), dtype=torch.long, device=device),
            torch.full(batch_shape, int(SolverStatus.MAX_ITER), dtype=torch.long, device=device),
        ),
    )
    residual = torch.where(b_norm > 0, norm_r / b_norm, norm_r)
    return SolveResult(
        x=x, converged=converged, iterations=iterations, residual=residual, status=status
    )


def gmres(
    op,
    b: Tensor,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_iter: int | None = None,
    restart: int = 30,
    x0: Tensor | None = None,
) -> SolveResult:
    """Restarted GMRES for nonsymmetric operators: Arnoldi (modified Gram-Schmidt) with
    incremental Givens rotations, per-instance status, never raises.

    Implementation flattens every leading batch dimension into one axis `B` for the duration
    of the Arnoldi/Givens bookkeeping (reshaped back to the caller's batch shape at the end):
    the Hessenberg system, Givens coefficients and Krylov basis are naturally rectangular
    per-instance state that is far simpler to index with one flat batch axis than with
    arbitrary leading dims, unlike pcg's per-element vector ops which need no such reshape.
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

    b_norm = torch.linalg.vector_norm(b_flat, dim=-1)
    tol = torch.clamp(rtol * b_norm, min=atol)

    r = b_flat - mv(x_flat)
    beta = torch.linalg.vector_norm(r, dim=-1)
    converged = beta <= tol
    exhausted = torch.zeros(B, dtype=torch.bool, device=device)
    total_matvecs = torch.zeros(B, dtype=torch.long, device=device)
    iterations = torch.zeros(B, dtype=torch.long, device=device)
    tiny = torch.finfo(dtype).tiny

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

        beta_safe = torch.where(beta > 0, beta, torch.ones_like(beta))
        V[:, 0, :] = torch.where(active.unsqueeze(-1), r / beta_safe.unsqueeze(-1), V[:, 0, :])
        g[:, 0] = torch.where(active, beta, g[:, 0])

        step_converged_at = torch.full((B,), cycle_len, dtype=torch.long, device=device)

        for k in range(cycle_len):
            # V is read here and mutated again later (a new column, but the SAME tensor's
            # storage) before backward runs; every operand read from it must be `.clone()`d
            # first, or autograd's version counter (tracked per-storage, not per-slice) sees
            # a mismatch at backward time. Same reasoning as the `h_j`/`h_j1`/`g_k` clones
            # below and the `h_kk` clone above.
            w = mv(V[:, k, :].clone())
            for j in range(k + 1):
                v_j = V[:, j, :].clone()
                h_jk = torch.einsum("bi,bi->b", w, v_j)
                H[:, j, k] = torch.where(active, h_jk, H[:, j, k])
                w = w - h_jk.unsqueeze(-1) * v_j
            h_next = torch.linalg.vector_norm(w, dim=-1)
            newly_exhausted = active & (h_next <= tiny)
            exhausted = exhausted | newly_exhausted
            h_next_safe = torch.where(h_next > tiny, h_next, torch.ones_like(h_next))
            v_next = w / h_next_safe.unsqueeze(-1)
            V[:, k + 1, :] = torch.where(active.unsqueeze(-1), v_next, V[:, k + 1, :])

            # apply every earlier cycle's Givens rotation to this new Hessenberg column
            for j in range(k):
                h_j = H[:, j, k].clone()
                h_j1 = H[:, j + 1, k].clone()
                H[:, j, k] = torch.where(active, cs[:, j] * h_j + sn[:, j] * h_j1, H[:, j, k])
                H[:, j + 1, k] = torch.where(
                    active, -sn[:, j] * h_j + cs[:, j] * h_j1, H[:, j + 1, k]
                )

            h_kk = H[:, k, k].clone()
            h_k1k = h_next
            denom = torch.sqrt(h_kk**2 + h_k1k**2)
            denom_safe = torch.where(denom > tiny, denom, torch.ones_like(denom))
            cs_k = torch.where(denom > tiny, h_kk / denom_safe, torch.ones_like(denom))
            sn_k = torch.where(denom > tiny, h_k1k / denom_safe, torch.zeros_like(denom))
            cs[:, k] = torch.where(active, cs_k, cs[:, k])
            sn[:, k] = torch.where(active, sn_k, sn[:, k])
            H[:, k, k] = torch.where(active, cs_k * h_kk + sn_k * h_k1k, H[:, k, k])

            g_k = g[:, k].clone()
            g[:, k] = torch.where(active, cs_k * g_k, g[:, k])
            g[:, k + 1] = torch.where(active, -sn_k * g_k, g[:, k + 1])

            residual_est = g[:, k + 1].abs()
            just_met = active & (residual_est <= tol) & (step_converged_at == cycle_len)
            step_converged_at = torch.where(
                just_met, torch.full_like(step_converged_at, k + 1), step_converged_at
            )

        # back-substitution: upper-triangular H[:, :cycle_len, :cycle_len] @ y = g[:, :cycle_len]
        y = torch.zeros(B, cycle_len, dtype=dtype, device=device)
        for k in range(cycle_len - 1, -1, -1):
            s = g[:, k].clone()
            for j in range(k + 1, cycle_len):
                s = s - H[:, k, j] * y[:, j]
            diag = H[:, k, k]
            diag_safe = torch.where(diag.abs() > tiny, diag, torch.ones_like(diag))
            y[:, k] = torch.where(diag.abs() > tiny, s / diag_safe, torch.zeros_like(s))

        dx = torch.einsum("bk,bkm->bm", y, V[:, :cycle_len, :])
        x_candidate = x_flat + dx
        x_flat = torch.where(active.unsqueeze(-1), x_candidate, x_flat)

        r = b_flat - mv(x_flat)
        beta = torch.linalg.vector_norm(r, dim=-1)
        newly_converged = active & (beta <= tol)

        iterations = torch.where(
            active, total_matvecs + step_converged_at.clamp(max=cycle_len), iterations
        )
        total_matvecs = total_matvecs + torch.where(
            active, torch.full_like(total_matvecs, cycle_len), torch.zeros_like(total_matvecs)
        )
        converged = converged | newly_converged

    status = torch.where(
        converged,
        torch.full((B,), int(SolverStatus.CONVERGED), dtype=torch.long, device=device),
        torch.where(
            exhausted,
            torch.full((B,), int(SolverStatus.SINGULAR), dtype=torch.long, device=device),
            torch.full((B,), int(SolverStatus.MAX_ITER), dtype=torch.long, device=device),
        ),
    )
    iterations = torch.where(converged, iterations, total_matvecs)
    residual = torch.where(b_norm > 0, beta / b_norm, beta)

    x_out = x_flat.reshape(batch_shape + (m,))
    return SolveResult(
        x=x_out,
        converged=converged.reshape(batch_shape),
        iterations=iterations.reshape(batch_shape),
        residual=residual.reshape(batch_shape),
        status=status.reshape(batch_shape),
    )
