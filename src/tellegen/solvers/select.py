"""method="auto" eligibility resolution (design section 3.1) and the raise/return failure
boundary (design section 3.2), on top of solvers.iterative's pcg and gmres.

solvers.iterative.pcg/gmres never raise; this is the one layer where a failed solve becomes
an exception by default. `on_failure="return"` is the explicit, narrow, non-default escape
hatch (never applies to a backward pass -- Task 12 always raises unconditionally there,
bypassing this function's on_failure entirely).
"""

from __future__ import annotations

import torch

from tellegen.operators.base import SolveResult
from tellegen.solvers.iterative import gmres, pcg

Tensor = torch.Tensor


def solve(
    op,
    b: Tensor,
    *,
    method: str = "auto",
    on_failure: str = "raise",
    where: str = "solve",
    **kw,
) -> SolveResult:
    if method not in ("auto", "cg", "gmres"):
        raise ValueError(f"{where}: unknown method {method!r}; expected 'auto', 'cg' or 'gmres'")
    if on_failure not in ("raise", "return"):
        raise ValueError(
            f"{where}: unknown on_failure {on_failure!r}; expected 'raise' or 'return'"
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
            bad = torch.nonzero(~cert.reshape(-1), as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"{where}: method='cg' requested explicitly but instances {bad} do not "
                f"certify SPD; refusing rather than returning a plausible wrong answer."
            )
        result = pcg(op, b, **kw)
    elif method == "gmres":
        result = gmres(op, b, **kw)
    else:  # method == "auto"
        cert = op.spd_certificate()
        if cert is not None and bool(torch.any(cert)) and not bool(torch.all(cert)):
            bad = torch.nonzero(~cert.reshape(-1), as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"{where}: method='auto' refuses to split the batch; instances {bad} do "
                f"not certify SPD (ungrounded, or a negative slope) while others do. "
                f"Certify all instances, or pass an explicit method."
            )
        if cert is not None and bool(torch.all(cert)):
            result = pcg(op, b, **kw)
        else:
            # cert is None (cannot certify at all) or cert is uniformly False (no instance
            # is eligible for cg, so there is nothing to split off): both route to gmres,
            # which makes no symmetry or SPD assumption to violate. rmatvec, if this
            # operator declares one, is reserved for the adjoint and is never called here.
            result = gmres(op, b, **kw)

    if on_failure == "raise":
        return result.raise_on_failure(where)
    return result
