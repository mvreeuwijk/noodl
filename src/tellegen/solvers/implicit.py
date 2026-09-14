"""Implicit-function differentiation through the batched Newton solve.

Forward: run newton() under no_grad. Backward: solve J(x*)^T lambda = grad_x (the adjoint
network, per Tellegen's theorem reciprocity between forward and adjoint), then obtain
gradients with respect to every parameter tensor by one more autograd pass through the
residual evaluated at the converged point, weighted by -lambda.

That transposed solve is matvec-free: the Jacobian at x* is whatever LinearOperator the
caller's `operator` callable returns there, and `TransposeOperator` exposes its `rmatvec`
as the transposed action, so `solvers.select.solve` handles the adjoint system with no
separate code path and nothing is ever materialised as an explicit matrix. `op.matvec` is
never reused as the transpose: the two coincide only for a symmetric operator, and a
caller's own wrong `rmatvec` must surface as a wrong gradient rather than be masked here.
The adjoint solve always RAISES on failure, whatever `on_failure` the forward was given.

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

from tellegen.operators.base import LinearOperator, as_operator
from tellegen.solvers.newton import inner_solve_rtol, newton
from tellegen.solvers.select import solve as select_solve


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

    def assemble_sparse(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """The wrapped operator's COO form with ROW and COL swapped -- its transpose.

        The optional `operators.base.SparseAssembling` member, which is what lets
        `method="sparse_direct"` solve the ADJOINT system: transposing a COO triplet is
        exactly exchanging the two index arrays, with the values untouched and no
        re-assembly or coalescing of any kind. `None` propagates unchanged (the transpose
        of "no sparse form" is still no sparse form), as does the absence of the optional
        member on the wrapped operator -- so `select.solve` reports its own ValueError
        rather than this view raising an AttributeError first.
        """
        assemble_sparse = getattr(self._op, "assemble_sparse", None)
        triplet = assemble_sparse() if assemble_sparse is not None else None
        if triplet is None:
            return None
        row, col, values = triplet
        return col, row, values

    def spd_certificate(self) -> torch.Tensor | None:
        """The wrapped operator's certificate, but only when it declares itself SYMMETRIC.

        A symmetric operator is its own transpose, so an SPD certificate for it certifies
        this view verbatim -- which is what keeps `PotentialFlowLayer.adjoint`'s
        GraphLaplacianOperator on the certified-SPD branch of the eligibility table
        (sparse-direct, or PCG where no sparse form is available) rather than dropping it to
        GMRES, so the backward pass costs what the forward pass costs. For a
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
    table in ``solvers.select``'s module docstring: sparse-direct when the TransposeOperator
    certifies SPD and offers a transposed COO form (spec section 6.2 step 2), PCG when it
    certifies but cannot, GMRES otherwise.

    Always raises on failure (``on_failure="raise"``, not exposed as a parameter): every
    caller of this function -- ``_Implicit.backward`` unconditionally, and
    ``PotentialFlowLayer.adjoint`` as a diagnostic entry point -- wants a wrong gradient to
    be impossible rather than silently returned.
    """
    step_method = "direct" if method == "auto" and isinstance(op, torch.Tensor) else method
    top = TransposeOperator(as_operator(op))
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
    def forward(ctx, x0, residual, operator, newton_kwargs, diagnostics, *params):
        with torch.no_grad():
            result = newton(
                lambda x: residual(x, *params),
                lambda x: operator(x, *params),
                x0,
                **newton_kwargs,
            )
        if diagnostics is not None:
            # The forward Newton solve happens here and nowhere else, so its iteration
            # counts are only observable from inside this Function. Writing them into a
            # caller-supplied dict is the narrowest way to expose them without changing what
            # `implicit_solve` RETURNS (a plain tensor, which is what autograd needs) or
            # making every caller that does not care pay for a richer result type. The
            # convergence STATUS goes in alongside them (design section 3.2: a solve result
            # always carries its status), even though this path can only ever report
            # success -- `implicit_solve` refuses `on_failure="return"` outright, so a
            # non-converged forward raises out of `newton` above rather than reaching here.
            diagnostics["newton_iterations"] = result.iterations
            diagnostics["linear_iterations"] = result.linear_iterations
            # The backend that actually ran, not the method requested: with "auto" the two
            # differ on runtime predicates the caller cannot see (final review I5).
            diagnostics["backend"] = result.backend
            diagnostics["converged"] = result.converged
            diagnostics["residual_norm"] = result.residual_norm
        ctx.residual = residual
        ctx.operator = operator
        ctx.newton_kwargs = newton_kwargs
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
            op = ctx.operator(x, *params)
            # on_failure is never forwarded: `adjoint` always raises (its own fixed
            # default), whatever the FORWARD pass was told. A forward solve may legitimately
            # be asked to return a non-converged instance instead of raising -- a
            # calibration loop inspecting or down-weighting it -- but a backward pass has no
            # such caller: a wrong gradient silently reaching an optimiser is strictly worse
            # than an exception (design section 3.2). `method` IS forwarded, so a layer
            # configured with linear_solver="direct" keeps the dense numerics it asked for
            # on both passes rather than only on the forward one.
            lam = adjoint(
                op,
                grad_x,
                # The forward's own `where` (a layer name, when a layer supplied one) with
                # " backward" appended, so an adjoint failure names the same solve the
                # forward would have. Defaults to "implicit_solve backward" as before.
                where=f"{ctx.newton_kwargs.get('where', 'implicit_solve')} backward",
                method=ctx.newton_kwargs.get("method", "auto"),
            )
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
    operator: Callable[..., LinearOperator | torch.Tensor],
    x0: torch.Tensor,
    params: tuple[torch.Tensor, ...],
    *,
    diagnostics: dict | None = None,
    **newton_kwargs,
) -> torch.Tensor:
    """Differentiable solve of ``residual(x, *params) = 0``; returns the converged ``x``.

    ``operator`` is called as ``operator(x, *params)`` at the current iterate and returns
    either a ``LinearOperator`` (the Milestone-1b contract) or a plain dense ``(..., m, m)``
    tensor (auto-wrapped, for every pre-1b caller); it is the same callable contract
    ``newton`` takes, and the same object is re-evaluated at the converged point to build
    the backward pass's adjoint system.

    ``diagnostics``, when a dict is given, is filled with the forward Newton solve's own
    ``newton_iterations``, ``linear_iterations``, ``backend``, ``converged`` and
    ``residual_norm`` (see ``_Implicit.forward``); ``backend`` is the inner solver that
    actually ran, as opposed to the ``method`` requested. Every other keyword is forwarded
    to ``newton``. It is
    keyword-ONLY deliberately: sitting positionally in front of ``**newton_kwargs`` it would
    silently swallow a fifth positional argument from any caller who thought they were
    passing something else.

    ``on_failure="return"`` is REFUSED here (``ValueError``), unlike on ``newton``'s own
    non-differentiable path. The implicit-function adjoint linearises at the point the
    forward returned and assumes that point solves ``residual(x, *params) = 0``; at a
    non-converged point that assumption is false, the adjoint solve nevertheless converges
    happily, and the gradient handed back is silently wrong. Design section 3.2 legislates
    exactly this: the backward pass raises unconditionally because "a wrong gradient is
    worse than no gradient" -- so the escape hatch must not be reachable on the
    differentiable path at all. Use ``differentiable=False`` (or ``newton`` directly) if a
    non-converged instance is something the caller wants to inspect rather than abort on.
    """
    if newton_kwargs.get("on_failure") == "return":
        raise ValueError(
            "implicit_solve: on_failure='return' is not supported on the differentiable "
            "path -- a non-converged forward has no defined adjoint, so the gradient would "
            "be silently wrong (design section 3.2: the backward pass raises "
            "unconditionally). Use the non-differentiable solve if a non-converged instance "
            "must be returned rather than raised on."
        )
    return _Implicit.apply(x0, residual, operator, newton_kwargs, diagnostics, *params)
