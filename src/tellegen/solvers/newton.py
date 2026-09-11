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
    norm0 = r.abs().amax(dim=-1)
    tol = atol + rtol * norm0
    norm = norm0
    converged = norm < tol
    omega_i = torch.full_like(norm0, omega)
    iterations = 0
    tiny = torch.finfo(norm.dtype).tiny

    while not bool(torch.all(converged)) and iterations < max_iter:
        J = jacobian(x)
        dx = torch.linalg.solve(J, r)
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
