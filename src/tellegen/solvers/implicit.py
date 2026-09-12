"""Implicit-function differentiation through the batched Newton solve.

Forward: run newton() under no_grad. Backward: solve J(x*)^T lambda = grad_x (the adjoint
network, per Tellegen's theorem reciprocity between forward and adjoint), then obtain
gradients with respect to every parameter tensor by one more autograd pass through the
residual evaluated at the converged point, weighted by -lambda.
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
