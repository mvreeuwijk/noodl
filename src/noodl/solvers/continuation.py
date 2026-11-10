"""A starting point for a Newton solve that failed: let the system settle through a
fictitious storage that is removed in steps.

`storage_ramp` is the second rung of `KirchhoffLayer`'s fallback ladder (`fallback=`; see
`docs/concepts/solvers.md`). It is pseudo-time stepping: each row of the residual gets a
storage term `g (x - x_prev)`, as if every node held some amount that resists a change of
its potential. A large storage makes each step small and the shifted Jacobian strongly
diagonal, so Newton converges from a poor start; the storage is then divided by ten per
stage, each stage starting from the last one's root, and the last root is handed to the
ordinary solve as its starting point. Nothing here is differentiated: the gradient is the
ordinary solve's, at the exact root it converges to.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from noodl.operators.base import LinearOperator
from noodl.operators.shifted import ShiftedOperator
from noodl.operators.sparse import SparseOperator
from noodl.solvers.newton import newton

Tensor = torch.Tensor

#: The storage of each stage, as a multiple of each row's own Jacobian diagonal.
RAMP_SCALES = (1.0, 1e-1, 1e-2, 1e-3)


def shift_diagonal(op, g: Tensor):
    """`op + diag(g)` for the operators a layer's Jacobian is: a dense tensor, a
    `SparseOperator` (the shift is appended as diagonal COO entries, which its consumers
    sum), an unscaled `ShiftedOperator` with a per-row shift (the two shifts are added, so
    the base's SPD certificate still applies), or any other operator (wrapped in a
    `ShiftedOperator`). `g` is `(..., n)`, non-negative."""
    if isinstance(op, Tensor):
        return op + torch.diag_embed(g.to(op.dtype))
    if isinstance(op, SparseOperator):
        n = op.shape[-1]
        idx = torch.arange(n, dtype=op.row.dtype, device=op.row.device)
        batch = torch.broadcast_shapes(op.values.shape[:-1], g.shape[:-1])
        values = torch.cat([op.values.expand(batch + op.values.shape[-1:]),
                            g.to(op.dtype).expand(batch + (n,))], dim=-1)
        return SparseOperator(torch.cat([op.row, idx]), torch.cat([op.col, idx]), values, n)
    if (isinstance(op, ShiftedOperator) and op.scale is None and torch.is_tensor(op.shift)
            and not op.shift_first):
        return ShiftedOperator(op.base, op.shift + g)
    return ShiftedOperator(op, g)


def _diagonal(op) -> Tensor:
    if isinstance(op, Tensor):
        return torch.diagonal(op, dim1=-2, dim2=-1)
    return op.diagonal()


def storage_ramp(
    residual: Callable[[Tensor], Tensor],
    operator: Callable[[Tensor], LinearOperator | Tensor],
    x0: Tensor,
    *,
    rows: Tensor | None = None,
    scales: Sequence[float] = RAMP_SCALES,
    diagnostics: dict | None = None,
    **newton_kwargs,
) -> Tensor:
    """A starting point for a Newton solve that failed: solve
    residual(x) + g (x - x_prev) = 0 with a fictitious storage g at every row, each stage
    from the last stage's root and with g ten times smaller, and return the last root.

    `g = scale * |diag(operator(x_prev))|` on `rows` (a `(n,)` bool mask; `None` is every
    row) and zero elsewhere, `x_prev` being the stage's start: each row's own Jacobian
    diagonal sets its units, so the storage is relative. `scales` are the stages' factors.
    Each stage is one `newton` solve with `newton_kwargs` and `on_failure="return"`, under
    `no_grad`: a stage that does not converge still hands its last iterate on, and an
    instance whose iterate is not finite keeps the stage's start. `diagnostics`, when given,
    receives `"newton_iterations"`, the stages' Newton iterations summed. The result is
    detached; it is a starting point, not a solution.
    """
    newton_kwargs = {**newton_kwargs, "on_failure": "return"}
    x = x0.detach()
    iterations = 0
    with torch.no_grad():
        for scale in scales:
            x_prev = x
            diag = _diagonal(operator(x_prev)).abs()
            g = scale * diag
            if rows is not None:
                g = g * rows.to(dtype=g.dtype, device=g.device)

            def stage_residual(x, x_prev=x_prev, g=g):
                return residual(x) + g * (x - x_prev)

            def stage_operator(x, g=g):
                return shift_diagonal(operator(x), g)

            result = newton(stage_residual, stage_operator, x_prev, **newton_kwargs)
            iterations += result.iterations
            finite = torch.isfinite(result.x).all(dim=-1, keepdim=True)
            x = torch.where(finite, result.x, x_prev)
    if diagnostics is not None:
        diagnostics["newton_iterations"] = iterations
    return x
