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

from tellegen.solvers.newton import newton


def adjoint(jacobian_at_solution: torch.Tensor, grad_x: torch.Tensor) -> torch.Tensor:
    J = jacobian_at_solution
    return torch.linalg.solve(J.transpose(-1, -2), grad_x.unsqueeze(-1)).squeeze(-1)


class _Implicit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, residual, jacobian, newton_kwargs, *params):
        with torch.no_grad():
            result = newton(
                lambda x: residual(x, *params),
                lambda x: jacobian(x, *params),
                x0,
                **newton_kwargs,
            )
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
            J = ctx.jacobian(x, *params)
            lam = adjoint(J, grad_x)
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
        return (None, None, None, None, *grads_aligned)


def implicit_solve(
    residual: Callable[..., torch.Tensor],
    jacobian: Callable[..., torch.Tensor],
    x0: torch.Tensor,
    params: tuple[torch.Tensor, ...],
    **newton_kwargs,
) -> torch.Tensor:
    return _Implicit.apply(x0, residual, jacobian, newton_kwargs, *params)
