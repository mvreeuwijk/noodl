"""Batched, damped Newton solver with per-instance convergence masking.

Every instance in the leading batch dimensions is solved independently: once an
instance's residual is below tolerance its state is frozen (the update is masked
to zero) while other instances keep iterating, and the relaxation factor omega
switches from its initial (damped) value to 1 once an instance's residual ratio
drops below ``switch_ratio``, following CONTAM's under-relaxation scheme.

``operator`` returns, at the current iterate, either a ``LinearOperator`` (the
contract of Milestone 1b) or a plain dense ``(..., m, m)`` tensor for backward
compatibility with every pre-1b caller; a returned tensor is auto-wrapped in
``DenseOperator``. The inner linear solve goes through ``solvers.select.solve``
with ``on_failure="return"`` (Newton is a low-level primitive: it never raises
mid-iteration on its own account, only via the final convergence check below).
This is what replaces the old identity-substitution trick for a converged-but-
singular instance: a dense ``torch.linalg.solve`` raises for the WHOLE batched
call if any one instance's matrix is singular, which is why the old code had to
substitute an identity for converged rows before the call ever happened. An
operator-based solve (PCG/GMRES, or a per-instance-safe direct path) is batched
elementwise over the leading dimensions with no cross-instance coupling in its
own arithmetic, so one instance being exactly singular cannot make the call
fail for its siblings; that instance's own step is simply garbage (possibly
NaN), and it is discarded by the ``torch.where`` on ``step`` below exactly as
it always was, without needing to keep the solve well-posed first.

An auto-wrapped dense tensor resolves to ``method="direct"`` (LU of the explicit
matrix, per instance) rather than to the iterative default, because that is the
whole point of the shim: a caller who hands Newton an explicit dense Jacobian is
asking for the dense numerics it has always had, bit-for-bit. Routing it to GMRES
instead would silently change every such caller's answer by the iterative
solver's own floor -- measured at 3e-8 on the 2x2 float32 system in
``tests/solvers/test_newton.py``, against assertions written to 1e-9 -- which is
a migration regression, not a convergence question. ``_direct`` still gives the
per-instance singularity handling the identity substitution used to fake, since
``lu_factor_ex`` reports a zero pivot through ``info`` instead of raising for
the whole batch.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from tellegen.operators.base import LinearOperator
from tellegen.operators.dense import DenseOperator
from tellegen.solvers.select import solve as select_solve

# The relative residual an inner linear solve is asked for, matching `solvers.select.solve`'s
# own pinned default, and the ULP multiple below which no dtype can deliver it. See
# `inner_solve_rtol`.
_INNER_RTOL = 1e-10
_INNER_RTOL_ULPS = 32


def inner_solve_rtol(dtype: torch.dtype) -> float:
    """Relative residual to ask an inner linear solve for, floored by the working dtype.

    A Krylov solver's achievable relative residual is bounded below by the rounding error it
    accumulates, a small multiple of ``finfo(dtype).eps``; asking for less does not make the
    answer better, it just spends every remaining iteration and then reports ``MAX_ITER`` on
    a solve that is in fact as converged as the dtype allows. In float64 the pinned 1e-10 is
    comfortably above that floor and is used unchanged; in float32 (this project's declared
    default dtype) eps is 1.2e-7, so 1e-10 is unreachable by several orders of magnitude --
    measured: the float32 CONTAM series case in ``tests/verification`` floors at 3.2e-8 and
    was reported as a ``linear_init`` failure until this floor was applied, and the 128-zone
    leaky chain in ``tests/solvers/test_newton_operator_contract.py`` ran every inner PCG to
    its full ``max_iter = 128`` ceiling. That second symptom is the quieter one: Newton
    passes ``on_failure="return"``, so the ``MAX_ITER`` status is swallowed and only
    ``NewtonResult.linear_iterations`` -- the number the composed-model report publishes --
    carries the damage, as the ceiling rather than the work actually done.

    This is the same "the dtype cannot be asked for precision it does not have" argument
    ``newton``'s own dtype-derived ``atol``/``rtol`` default makes, applied to the linear
    solve instead of to the Newton convergence test. It lives here, next to that default,
    rather than in any one caller: ``PotentialFlowLayer.linear_init`` needs the identical
    floor for the identical reason.
    """
    return max(_INNER_RTOL, _INNER_RTOL_ULPS * float(torch.finfo(dtype).eps))


@dataclass
class NewtonResult:
    """Newton's own per-instance outcome, plus the inner solver's cost.

    ``linear_iterations`` is per instance the MAXIMUM inner-solver iteration count over the
    Newton steps actually taken -- the worst single linear solve an instance needed, which is
    what a budget or a preconditioner decision is made against, and is monotone in the
    problem's difficulty in a way a sum over a varying number of Newton steps is not. It is
    ``None`` only when no linear solve happened at all (``x0`` already satisfied the
    convergence test), never as a stand-in for an unknown count.
    """

    x: torch.Tensor
    converged: torch.Tensor
    iterations: int
    residual_norm: torch.Tensor
    linear_iterations: torch.Tensor | None = None


def _as_operator(op: LinearOperator | torch.Tensor) -> LinearOperator:
    """Auto-wrap a plain dense Jacobian tensor as a ``DenseOperator``.

    Every pre-Milestone-1b caller passes a callable returning a dense ``(..., m, m)``
    tensor; wrapping here (rather than making each of them construct an operator) is the
    single compatibility shim that makes the operator contract invisible to them.
    ``DenseOperator``'s ``symmetric=False`` default is load-bearing: it lets this wrap
    happen with no keyword at all, and it makes no symmetry claim on a Newton Jacobian that
    is in general nonsymmetric -- so an explicit ``method="cg"`` is correctly refused, and
    ``"auto"`` falls through to GMRES for anything this function does not route to the
    direct path.
    """
    if isinstance(op, torch.Tensor):
        return DenseOperator(op)
    return op


def newton(
    residual: Callable[[torch.Tensor], torch.Tensor],
    operator: Callable[[torch.Tensor], LinearOperator | torch.Tensor],
    x0: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    max_iter: int = 50,
    omega: float = 0.75,
    switch_ratio: float = 0.5,
    method: str = "auto",
    on_failure: str = "raise",
) -> NewtonResult:
    """Solve ``residual(x) = 0`` by damped, batched Newton iteration.

    ``atol``/``rtol`` default to ``None``, meaning "derive from the working dtype": each
    defaults independently to ``sqrt(torch.finfo(dtype).eps)``, where ``dtype`` is taken
    from the actual residual tensor returned by ``residual(x0)`` (not assumed from ``x0``
    or from any fixed convention), giving about 1.2e-4 for float32 and 1.5e-8 for float64.
    This is the standard "half the significant digits" heuristic for a first-order
    convergence test: tighter than that asks the dtype for precision it does not have and
    the residual floors below the target before ``converged`` ever becomes true (observed,
    pre-fix, for every network size in ``benchmarks/newton_scaling.py`` under float32, whose
    Jacobian evaluation and linear solve both round to float32 ULP). Passing an explicit
    ``atol`` and/or ``rtol`` always overrides this default for that argument; the two are
    independent, so an explicit ``atol=1e-6`` with ``rtol`` left as ``None`` still gets the
    dtype-derived default for ``rtol``.

    ``method`` is forwarded verbatim to ``solvers.select.solve`` for every inner linear
    solve, so the caller chooses the inner solver (``"auto"``, ``"cg"``, ``"gmres"``,
    ``"direct"``) without Newton needing to know anything about the operator's storage. The
    one exception is the compatibility shim: with ``method="auto"`` a bare dense tensor
    resolves to ``"direct"`` (see the module docstring). An explicit ``method`` always wins.

    ``on_failure="raise"`` (default) raises ``RuntimeError`` naming the batch indices that
    failed to converge after ``max_iter`` iterations, exactly as before. ``on_failure="return"``
    is the explicit, non-default escape hatch (design section 3.2): it returns the
    ``NewtonResult`` with ``converged`` reflecting the true per-instance state instead of
    raising, for a caller (e.g. a calibration loop, or a differentiable forward pass that
    passes ``on_failure`` through) that would rather inspect or down-weight a failed
    instance than abort. It is never silent: ``converged`` and ``residual_norm`` still carry
    the true state either way.
    """
    x = x0
    r = residual(x)
    if atol is None or rtol is None:
        default_tol = math.sqrt(torch.finfo(r.dtype).eps)
        if atol is None:
            atol = default_tol
        if rtol is None:
            rtol = default_tol
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
    linear_iterations: torch.Tensor | None = None
    tiny = torch.finfo(norm.dtype).tiny

    while not bool(torch.all(converged)) and iterations < max_iter:
        raw = operator(x)
        # A bare dense tensor (the pre-1b compatibility shim) keeps the dense LU numerics
        # it has always had; a genuine LinearOperator goes through the eligibility table.
        step_method = "direct" if method == "auto" and isinstance(raw, torch.Tensor) else method
        result = select_solve(
            _as_operator(raw),
            r,
            method=step_method,
            on_failure="return",
            where="newton",
            rtol=inner_solve_rtol(r.dtype),
        )
        dx = result.x
        inner = result.iterations.to(torch.int64).expand(converged.shape)
        linear_iterations = (
            inner.clone() if linear_iterations is None else torch.maximum(linear_iterations, inner)
        )
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
        if on_failure == "return":
            return NewtonResult(
                x=x,
                converged=converged,
                iterations=iterations,
                residual_norm=norm,
                linear_iterations=linear_iterations,
            )
        flat_converged = converged.reshape(-1)
        bad = torch.nonzero(~flat_converged, as_tuple=False).flatten()
        norms = norm.reshape(-1)[bad]
        raise RuntimeError(
            f"newton: batch indices {bad.tolist()} failed to converge after "
            f"{iterations} iterations, residual norms {norms.tolist()}"
        )
    return NewtonResult(
        x=x,
        converged=converged,
        iterations=iterations,
        residual_norm=norm,
        linear_iterations=linear_iterations,
    )
