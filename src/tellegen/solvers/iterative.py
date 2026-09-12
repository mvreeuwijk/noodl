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
