"""Implicit-function differentiation through the batched Newton solve.

Forward: run newton() under no_grad. Backward: solve J(x*)^T lambda = grad_x (the adjoint
network, per Tellegen's theorem reciprocity between forward and adjoint), then obtain
gradients with respect to every parameter tensor by one more autograd pass through the
residual evaluated at the converged point, weighted by -lambda.

Only first-order gradients are supported: `backward` never builds a graph connecting its
returned gradients back to `grad_x` or to `params` (the internal `torch.autograd.grad` call
uses the default `create_graph=False`), so a caller who tries to differentiate through
`implicit_solve`'s gradient a second time -- e.g. a gradient-penalty loss
`(dx/dtheta)**2 + ...`, computed via `torch.autograd.grad(x, theta, create_graph=True)` --
must be told this loudly rather than silently receiving an incomplete result in which the
second-order term is simply missing (contributing zero) while any other, first-order term in
the same loss still produces a gradient.

`@torch.autograd.function.once_differentiable` was tried first, as it is the standard idiom
for this, but was found NOT to catch this case here and was dropped again: its guard fires
only when the incoming `grad_x` itself already `requires_grad`, which is false for the
ordinary implicit unit-seed `torch.autograd.grad(x.sum(), theta, create_graph=True)` produces
-- confirmed empirically (with and without the decorator, a mixed loss
`(dx/dtheta**2).sum() + (theta**2).sum()` silently returns only the second term's gradient,
identically, in both cases). It also cannot be layered underneath a manual check of its own,
since its wrapper forces `torch.no_grad()` before calling the wrapped body, hiding the very
signal (`torch.is_grad_enabled()`) that would otherwise reveal a `create_graph=True` request.

Instead, `backward` checks `torch.is_grad_enabled()` at its own entry, before any of its own
`no_grad`/`enable_grad` blocks: PyTorch's engine calls `Function.backward` with grad mode
already disabled for an ordinary (`create_graph=False`) backward pass -- including every
`gradcheck` invocation in this test suite, confirmed empirically -- and leaves it enabled
only when `create_graph=True` was requested upstream. This is therefore a reliable signal
that the caller wants (or will attempt) a second differentiation through this Function's
output, which is unsupported; `backward` raises immediately in that case instead of quietly
returning a wrong number.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from tellegen.operators.base import LinearOperator
from tellegen.operators.dense import DenseOperator
from tellegen.solvers.newton import inner_solve_rtol, newton
from tellegen.solvers.select import solve as select_solve


def _as_operator(op: LinearOperator | torch.Tensor) -> LinearOperator:
    """Auto-wrap a plain dense Jacobian tensor as a ``DenseOperator``, exactly as
    ``solvers.newton._as_operator`` does for the forward pass: every pre-Milestone-1b caller
    of ``adjoint`` hands it an explicit ``(..., m, m)`` tensor, and wrapping here is what
    keeps the operator contract invisible to them.
    """
    if isinstance(op, torch.Tensor):
        return DenseOperator(op)
    return op


class TransposeOperator:
    """Wraps a LinearOperator's TRANSPOSE action as its own LinearOperator: matvec here is
    the wrapped operator's rmatvec, and rmatvec here is the wrapped operator's matvec. This
    lets the ordinary solve() entry point solve the adjoint system op^T @ lambda = grad_x
    without a separate code path, and without ever assuming op.matvec == op.rmatvec (true
    only for a symmetric operator, and not assumed here even then -- the swap always
    happens, so a bug in a caller's own claimed rmatvec is exposed rather than masked).

    This is the general counterpart of ``layers.transport._TransposeView`` (Task 9), which
    is the same adapter specialised to that layer's own advection operator; the two agree
    method for method, except that this one can forward an SPD certificate (see
    ``spd_certificate``) where the transport-local view, whose operator never certifies,
    simply returns None.
    """

    def __init__(self, op: LinearOperator) -> None:
        self._op = op
        self.shape = op.shape
        self.dtype = op.dtype
        self.device = op.device
        # The transpose of a symmetric operator is symmetric; of a nonsymmetric one, still
        # nonsymmetric. Either way the wrapped operator's own declaration carries over.
        self.symmetric = op.symmetric

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self._op.rmatvec(x)

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self._op.matvec(x)

    def diagonal(self) -> torch.Tensor:
        return self._op.diagonal()  # diagonal entries are invariant under transpose

    def assemble(self) -> torch.Tensor | None:
        a = self._op.assemble()
        return None if a is None else a.transpose(-1, -2)

    def spd_certificate(self) -> torch.Tensor | None:
        """The wrapped operator's certificate, but only when it declares itself SYMMETRIC.

        A symmetric operator is its own transpose, so an SPD certificate for it certifies
        this view verbatim -- which is what keeps `PotentialFlowLayer.adjoint`'s
        GraphLaplacianOperator on the PCG path rather than dropping to GMRES. For a
        NONSYMMETRIC operator the transpose is a different matrix and the wrapped
        certificate says nothing about it, so None is returned: `select.solve` then routes
        to GMRES, which makes no symmetry or definiteness assumption to violate.
        """
        return self._op.spd_certificate() if self._op.symmetric else None


def adjoint(
    op: LinearOperator | torch.Tensor,
    grad_x: torch.Tensor,
    *,
    where: str = "adjoint",
    method: str = "auto",
) -> torch.Tensor:
    """Solve the adjoint system ``op^T @ lambda = grad_x`` via ``op.rmatvec`` (never
    ``op.matvec``): the two coincide only when ``op`` is symmetric, which is not assumed
    here. ``op`` may be a plain dense tensor (backward compatible with every pre-1b caller)
    or a ``LinearOperator``; a tensor is auto-wrapped in ``DenseOperator``.

    ``method`` is forwarded to ``solvers.select.solve``, with the same compatibility shim
    ``newton`` applies: a BARE TENSOR under ``method="auto"`` resolves to ``"direct"`` (LU
    of the explicit transpose), so a legacy dense caller keeps the dense numerics it has
    always had rather than silently acquiring a Krylov solver's own error floor. An explicit
    ``method`` always wins, and a real operator's ``"auto"`` goes through the eligibility
    table (PCG when the TransposeOperator certifies SPD, GMRES otherwise).

    Always raises on failure (``on_failure="raise"``, not exposed as a parameter): every
    caller of this function -- ``_Implicit.backward`` unconditionally, and
    ``PotentialFlowLayer.adjoint`` as a diagnostic entry point -- wants a wrong gradient to
    be impossible rather than silently returned.
    """
    step_method = "direct" if method == "auto" and isinstance(op, torch.Tensor) else method
    top = TransposeOperator(_as_operator(op))
    result = select_solve(
        top,
        grad_x,
        method=step_method,
        on_failure="raise",
        where=where,
        rtol=inner_solve_rtol(grad_x.dtype),
    )
    return result.x


class _Implicit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, residual, jacobian, newton_kwargs, diagnostics, *params):
        with torch.no_grad():
            result = newton(
                lambda x: residual(x, *params),
                lambda x: jacobian(x, *params),
                x0,
                **newton_kwargs,
            )
        if diagnostics is not None:
            # The forward Newton solve happens here and nowhere else, so its iteration
            # counts are only observable from inside this Function. Writing them into a
            # caller-supplied dict is the narrowest way to expose them without changing what
            # `implicit_solve` RETURNS (a plain tensor, which is what autograd needs) or
            # making every caller that does not care pay for a richer result type.
            diagnostics["newton_iterations"] = result.iterations
            diagnostics["linear_iterations"] = result.linear_iterations
        ctx.residual = residual
        ctx.jacobian = jacobian
        ctx.save_for_backward(result.x, *params)
        return result.x

    @staticmethod
    def backward(ctx, grad_x):
        if torch.is_grad_enabled():
            # See the module docstring: grad mode is enabled at this point only when the
            # caller requested `create_graph=True` (a second-order differentiation attempt),
            # which this Function does not support -- its own gradient computation below is
            # not itself graph-building. Raise now rather than silently returning a gradient
            # that looks connected but omits the (unsupported) second-order term.
            raise RuntimeError(
                "implicit_solve: second-order differentiation (create_graph=True) is not "
                "supported; only first-order gradients via the adjoint method are "
                "implemented. Detach the first-order gradient before using it in a further "
                "differentiable loss (e.g. a gradient-penalty term)."
            )
        saved = ctx.saved_tensors
        x, params = saved[0], list(saved[1:])
        with torch.no_grad():
            op = ctx.jacobian(x, *params)
            lam = adjoint(op, grad_x)
        with torch.enable_grad():
            p = [t.detach().requires_grad_(t.requires_grad) for t in params]
            r = ctx.residual(x.detach(), *p)
            needs_grad = [t for t in p if t.requires_grad]
            grads = (
                torch.autograd.grad(r, needs_grad, grad_outputs=-lam, allow_unused=True)
                if needs_grad
                else []
            )
        grads_aligned = []
        it = iter(grads)
        for t in p:
            grads_aligned.append(next(it) if t.requires_grad else None)
        return (None, None, None, None, None, *grads_aligned)


def implicit_solve(
    residual: Callable[..., torch.Tensor],
    jacobian: Callable[..., torch.Tensor],
    x0: torch.Tensor,
    params: tuple[torch.Tensor, ...],
    *,
    diagnostics: dict | None = None,
    **newton_kwargs,
) -> torch.Tensor:
    """Differentiable solve of ``residual(x, *params) = 0``; returns the converged ``x``.

    ``diagnostics``, when a dict is given, is filled with the forward Newton solve's own
    ``newton_iterations`` and ``linear_iterations`` (see ``_Implicit.forward``). Every other
    keyword is forwarded to ``newton``. It is keyword-ONLY deliberately: sitting positionally
    in front of ``**newton_kwargs`` it would silently swallow a fifth positional argument
    from any caller who thought they were passing something else.
    """
    return _Implicit.apply(x0, residual, jacobian, newton_kwargs, diagnostics, *params)
