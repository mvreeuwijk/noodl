"""method="auto" eligibility resolution (design section 3.1) and the raise/return failure
boundary (design section 3.2), on top of solvers.iterative's pcg and gmres, plus the retained
dense LU reference (`method="direct"`, amendment A3.1).

solvers.iterative.pcg/gmres never raise; this is the one layer where a failed NUMERICAL solve
becomes an exception by default. `on_failure="return"` is the explicit, narrow, non-default
escape hatch for that numerical-failure case only (never applies to a backward pass -- Task 12
always raises unconditionally there, bypassing this function's on_failure entirely). It does
NOT apply to an eligibility refusal: requesting `method="cg"` on an operator that cannot
certify SPD, or letting `method="auto"` see a batch where some but not all instances certify,
is a contract violation and raises regardless of `on_failure` -- there is no SolveResult to
return in that case, only a modelling error to report.
"""

from __future__ import annotations

import torch

from tellegen.operators.base import SolveResult, SolverStatus
from tellegen.solvers.iterative import gmres, pcg

Tensor = torch.Tensor


def _describe_uncertified(op, cert: Tensor, where: str) -> str:
    """Message fragment naming which instances fail to certify SPD, and why.

    Prefers `op.spd_diagnosis()` (present on GraphLaplacianOperator, absent elsewhere) to name
    the actual negative-slope edges or ungrounded interior nodes per failing instance; falls
    back to the generic "instances {bad} do not certify SPD" when the operator has no
    `spd_diagnosis` (e.g. the test-only `_FakeOperator`, or any future non-diagnosing operator).
    """
    bad = torch.nonzero(~cert.reshape(-1), as_tuple=False).flatten().tolist()
    diagnose = getattr(op, "spd_diagnosis", None)
    if diagnose is None:
        return f"instances {bad} do not certify SPD"
    lines = []
    for rec in diagnose():
        if rec["reason"] == "negative_slope":
            lines.append(f"instance {rec['instance']}: negative slope on edges {rec['edges']}")
        else:
            lines.append(f"instance {rec['instance']}: ungrounded interior nodes {rec['nodes']}")
    return "; ".join(lines) if lines else f"instances {bad} do not certify SPD"


def _direct(A: Tensor, b: Tensor) -> SolveResult:
    """LU solve of the assembled operator with PER-INSTANCE singularity status.

    torch.linalg.solve raises for the whole batch if any one instance is singular, which is
    the very behaviour the milestone-1 Newton had to work around with an identity
    substitution. lu_factor_ex reports singularity per instance through `info` instead.
    """
    batch = torch.broadcast_shapes(A.shape[:-2], b.shape[:-1])
    m = A.shape[-1]
    A_b = A.expand(*batch, m, m)
    b_b = b.expand(*batch, m)
    LU, pivots, info = torch.linalg.lu_factor_ex(A_b)
    singular = info != 0  # (...,) per instance; info>0 = zero pivot
    x = torch.linalg.lu_solve(LU, pivots, b_b.unsqueeze(-1)).squeeze(-1)
    x = torch.where(singular.unsqueeze(-1), torch.zeros_like(x), x)
    r = torch.einsum("...ij,...j->...i", A_b, x) - b_b
    b_norm = torch.linalg.vector_norm(b_b, dim=-1)
    residual = torch.linalg.vector_norm(r, dim=-1) / b_norm.clamp_min(torch.finfo(b.dtype).tiny)
    residual = torch.where(b_norm > 0, residual, torch.zeros_like(residual))
    finite = torch.isfinite(x).all(dim=-1)
    converged = finite & ~singular
    status = torch.where(
        converged,
        torch.full_like(info, int(SolverStatus.CONVERGED)),
        torch.full_like(info, int(SolverStatus.SINGULAR)),
    )
    return SolveResult(
        x=x,
        converged=converged,
        iterations=torch.ones(info.shape, dtype=torch.int64, device=info.device),
        residual=residual,
        status=status.to(torch.int64),
    )


_METHODS = ("auto", "cg", "gmres", "direct")
_ON_FAILURE = ("raise", "return")


def solve(
    op,
    b: Tensor,
    *,
    method: str = "auto",
    on_failure: str = "raise",
    where: str = "solve",
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_iter: int | None = None,
    x0: Tensor | None = None,
    preconditioner: str | None = "jacobi",
    restart: int = 30,
) -> SolveResult:
    """Resolve `method="auto"` per the eligibility table (see the module docstring), or honour
    an explicit method, then apply the raise/return boundary on the NUMERICAL outcome.

    Accepts the explicit union of backend keyword arguments -- `rtol`, `atol`, `max_iter`,
    `x0` (both pcg and gmres), `preconditioner` (pcg only), `restart` (gmres only) -- and
    forwards to the chosen backend only the ones it accepts, silently dropping the rest (no
    `**kw`: a reviewer found `solve(op_nonsym, b, preconditioner="jacobi")` raising `TypeError`
    from gmres before this signature was made explicit). `method="direct"` accepts and ignores
    all of them.
    """
    if method not in _METHODS:
        raise ValueError(f"{where}: unknown method {method!r}; expected one of {_METHODS}")
    if on_failure not in _ON_FAILURE:
        raise ValueError(
            f"{where}: unknown on_failure {on_failure!r}; expected one of {_ON_FAILURE}"
        )

    if method == "cg":
        cert = op.spd_certificate()
        if cert is None:
            raise RuntimeError(
                f"{where}: method='cg' requested explicitly but this operator's "
                f"spd_certificate() is None (it cannot certify SPD at all); refusing "
                f"rather than returning a plausible wrong answer."
            )
        if not bool(torch.all(cert)):
            raise RuntimeError(
                f"{where}: method='cg' requested explicitly but "
                f"{_describe_uncertified(op, cert, where)}; refusing rather than returning a "
                f"plausible wrong answer."
            )
        result = pcg(
            op, b, rtol=rtol, atol=atol, max_iter=max_iter, preconditioner=preconditioner, x0=x0
        )
    elif method == "gmres":
        result = gmres(op, b, rtol=rtol, atol=atol, max_iter=max_iter, restart=restart, x0=x0)
    elif method == "direct":
        A = op.assemble()
        if A is None:
            raise ValueError(f"{where}: method='direct' requires op.assemble() to return a matrix")
        result = _direct(A, b)
    else:  # method == "auto"
        cert = op.spd_certificate()
        if cert is not None and bool(torch.any(cert)) and not bool(torch.all(cert)):
            raise RuntimeError(
                f"{where}: method='auto' refuses to split the batch; "
                f"{_describe_uncertified(op, cert, where)} while other instances certify. "
                f"Certify all instances, or pass an explicit method."
            )
        if cert is not None and bool(torch.all(cert)):
            result = pcg(
                op,
                b,
                rtol=rtol,
                atol=atol,
                max_iter=max_iter,
                preconditioner=preconditioner,
                x0=x0,
            )
        else:
            # cert is None (cannot certify at all) or cert is uniformly False (no instance
            # is eligible for cg, so there is nothing to split off): both route to gmres,
            # which makes no symmetry or SPD assumption to violate. rmatvec, if this
            # operator declares one, is reserved for the adjoint and is never called here.
            result = gmres(op, b, rtol=rtol, atol=atol, max_iter=max_iter, restart=restart, x0=x0)

    if on_failure == "raise":
        return result.raise_on_failure(where)
    return result
