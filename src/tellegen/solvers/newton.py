"""Batched, damped Newton solver with per-instance convergence masking.

Every instance in the leading batch dimensions is solved independently: once an
instance's residual is below tolerance its state is frozen (the update is masked
to zero) while other instances keep iterating, and the relaxation factor omega
switches from its initial (damped) value to 1 once an instance's residual ratio
drops below ``switch_ratio``, following CONTAM's under-relaxation scheme.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class NewtonResult:
    x: torch.Tensor
    converged: torch.Tensor
    iterations: int
    residual_norm: torch.Tensor


def newton(
    residual: Callable[[torch.Tensor], torch.Tensor],
    jacobian: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    max_iter: int = 50,
    omega: float = 0.75,
    switch_ratio: float = 0.5,
) -> NewtonResult:
    x = x0
    r = residual(x)
    if r.shape[-1] == 0:
        # Zero interior unknowns (every node is a boundary node): there is nothing to
        # iterate on, and `r.abs().amax(dim=-1)` below would raise IndexError ("Expected
        # reduction dim -1 to have non-zero size") rather than reporting a valid, trivially
        # converged system. `x` (shape (..., 0)) is already the unique solution.
        zero = torch.zeros(r.shape[:-1], dtype=r.dtype, device=r.device)
        return NewtonResult(
            x=x,
            converged=torch.ones(r.shape[:-1], dtype=torch.bool, device=r.device),
            iterations=0,
            residual_norm=zero,
        )
    norm0 = r.abs().amax(dim=-1)
    tol = atol + rtol * norm0
    norm = norm0
    converged = norm < tol
    omega_i = torch.full_like(norm0, omega)
    iterations = 0
    tiny = torch.finfo(norm.dtype).tiny

    while not bool(torch.all(converged)) and iterations < max_iter:
        J = jacobian(x)
        # Guard the INPUT to linalg.solve, not just its output: J is still evaluated for
        # every instance every iteration (a batched solve can't skip individual instances),
        # so a frozen instance whose Jacobian is genuinely singular at its own converged
        # point (e.g. a branch element sitting exactly on a zero-slope point) would otherwise
        # raise for the *entire* batched call even though its result is about to be masked
        # to zero below. Substituting the identity for converged instances keeps the solve
        # well-posed without changing any result: those rows are discarded by the
        # torch.where on `step` regardless of what value they resolve to.
        eye = torch.eye(J.shape[-1], dtype=J.dtype, device=J.device)
        J_safe = torch.where(converged[..., None, None], eye, J)
        dx = torch.linalg.solve(J_safe, r)
        step = torch.where(
            converged.unsqueeze(-1), torch.zeros_like(dx), omega_i.unsqueeze(-1) * dx
        )
        x = x - step
        r = residual(x)
        prev_norm = norm
        norm = r.abs().amax(dim=-1)
        ratio = norm / prev_norm.clamp_min(tiny)
        omega_i = torch.where(ratio < switch_ratio, torch.ones_like(omega_i), omega_i)
        converged = norm < tol
        iterations += 1

    if not bool(torch.all(converged)):
        flat_converged = converged.reshape(-1)
        bad = torch.nonzero(~flat_converged, as_tuple=False).flatten()
        norms = norm.reshape(-1)[bad]
        raise RuntimeError(
            f"newton: batch indices {bad.tolist()} failed to converge after "
            f"{iterations} iterations, residual norms {norms.tolist()}"
        )
    return NewtonResult(x=x, converged=converged, iterations=iterations, residual_norm=norm)
