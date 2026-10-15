"""Batched, damped Newton solver with per-instance convergence masking.

Every instance in the leading batch dimensions is solved independently: once an
instance's residual is below tolerance its state is frozen (the update is masked
to zero) while other instances keep iterating, and the relaxation factor omega
switches from its initial (damped) value to 1 once an instance's residual ratio
drops below ``switch_ratio``, following CONTAM's under-relaxation scheme, and falls
back to the damped value for any instance whose residual failed to shrink under the
full step (the undamped step on a square-root law cycles ``dp -> -dp`` forever).

``operator`` returns, at the current iterate, either a ``LinearOperator`` (the
matvec-free operator contract) or a plain dense ``(..., m, m)`` tensor for a
caller that assembles densely; a returned tensor is auto-wrapped in
``DenseOperator``. The inner linear solve goes through ``solvers.select.solve``
with ``on_failure="return"``, so a NUMERICAL failure (non-convergence,
breakdown, singularity) never raises mid-iteration -- only the final
convergence check below can, and only when ``on_failure="raise"``. This does
NOT cover an ELIGIBILITY refusal: ``solvers.select.solve``
raises ``RuntimeError`` regardless of ``on_failure`` when an explicit
``method="cg"`` cannot certify SPD, or when ``method="auto"`` would have to
split a batch between certified and uncertified instances -- both are
modelling errors, not numerical outcomes, so they propagate out of an inner
iteration exactly as they would from a single, unbatched call.
An operator-based solve (PCG/GMRES, or a per-instance-safe direct path) is batched
elementwise over the leading dimensions with no cross-instance coupling in its
own arithmetic, so one instance being exactly singular cannot make the call
fail for its siblings; that instance's own step is simply garbage (possibly
NaN), and it is discarded by the ``torch.where`` on ``step`` below.

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

from noodl.operators.base import LinearOperator, as_operator
from noodl.solvers.select import solve as select_solve

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
    default dtype) eps is 1.2e-7, so 1e-10 is unreachable by several orders of magnitude.

    This is the same "the dtype cannot be asked for precision it does not have" argument
    ``newton``'s own dtype-derived ``atol``/``rtol`` default makes, applied to the linear
    solve instead of to the Newton convergence test. It lives here, next to that default,
    rather than in any one caller: ``PotentialFlowLayer.linear_init`` needs the identical
    floor for the identical reason.
    """
    return max(_INNER_RTOL, _INNER_RTOL_ULPS * float(torch.finfo(dtype).eps))


#: ``line_search="convex"``: a step whose directional derivative ``phi'(t)`` exceeds this
#: fraction of ``|phi'(0)|`` overshot the minimum along the Newton direction and is cut back.
#: A full Newton step on a smooth law ends near ``phi' = 0`` and a damped one below it, so
#: neither is touched.
LINE_SEARCH_OVERSHOOT = 0.1
#: The cut-back search stops once ``|phi'(t)| <= LINE_SEARCH_TOL |phi'(0)|``...
LINE_SEARCH_TOL = 1e-3
#: ...or after this many residual evaluations.
LINE_SEARCH_MAX = 30


def _convex_line_search(residual, x0, r0, dx, t_trial, x1, r1, norm1, norm0, done):
    """The cut-back of `newton`'s ``line_search="convex"``: along ``d = -dx`` from `x0`
    (residual `r0`), the trial point `x1 = x0 + t_trial d` (residual `r1`, norm `norm1`;
    `norm0` the start's) is replaced, per instance, by the minimum of ``E`` along ``d``
    where it overshot it (see `newton`). Returns ``(x, r, norm, cut)``, `cut` marking the
    instances that were cut back."""
    g0 = -(r0 * dx).sum(-1)                      # phi'(0) = F(x0) . d, < 0 for a descent d
    g1 = -(r1 * dx).sum(-1)                      # phi'(t_trial)
    over = (~done) & (g0 < 0) & ((g1 > LINE_SEARCH_OVERSHOOT * (-g0))
                                 | ((g1 > 0) & (norm1 >= norm0)))
    if not bool(over.any()):
        return x1, r1, norm1, over
    t_lo, g_lo = torch.zeros_like(g0), g0
    t_hi, g_hi = t_trial.clone(), g1
    t, x, r = t_trial.clone(), x1, r1
    side = torch.zeros_like(g0)                  # Illinois: which end was kept last
    active = over.clone()
    for _ in range(LINE_SEARCH_MAX):
        denom = (g_hi - g_lo).clamp_min(torch.finfo(g0.dtype).tiny)
        t_new = t_lo - g_lo * (t_hi - t_lo) / denom
        t_new = torch.where(torch.isfinite(t_new), t_new, 0.5 * (t_lo + t_hi))
        t = torch.where(active, t_new, t)
        x = torch.where(over.unsqueeze(-1), x0 - t.unsqueeze(-1) * dx, x1)
        r = residual(x)
        g = -(r * dx).sum(-1)
        right = g > 0
        # Illinois: halve the slope value of the end kept twice in a row
        keep_lo = right & (side > 0)
        keep_hi = ~right & (side < 0)
        g_lo = torch.where(active & keep_lo, 0.5 * g_lo, g_lo)
        g_hi = torch.where(active & keep_hi, 0.5 * g_hi, g_hi)
        t_hi = torch.where(active & right, t, t_hi)
        g_hi = torch.where(active & right, g, g_hi)
        t_lo = torch.where(active & ~right, t, t_lo)
        g_lo = torch.where(active & ~right, g, g_lo)
        side = torch.where(active, torch.where(right, torch.ones_like(side),
                                               -torch.ones_like(side)), side)
        active = active & (g.abs() > LINE_SEARCH_TOL * (-g0))
        if not bool(active.any()):
            break
    return x, r, r.abs().amax(dim=-1), over


@dataclass
class NewtonResult:
    """Newton's own per-instance outcome, plus the inner solver's cost.

    ``linear_iterations`` is per instance the MAXIMUM inner-solver iteration count over the
    Newton steps actually taken -- the worst single linear solve an instance needed, which is
    what a budget or a preconditioner decision is made against, and is monotone in the
    problem's difficulty in a way a sum over a varying number of Newton steps is not. It is
    ``None`` only when no linear solve happened at all (``x0`` already satisfied the
    convergence test), never as a stand-in for an unknown count.

    ``backend`` is the inner solver that actually RAN -- ``"sparse_direct"``, ``"pcg"``,
    ``"gmres"`` or ``"direct"`` -- as opposed to the ``method`` that was requested. With
    ``method="auto"`` the two differ, and which one a solve got depends on runtime predicates
    (the batch size, whether SciPy is importable) that a caller cannot otherwise see. It is
    the LAST inner solve's backend; every inner solve in one ``newton`` call sees the same
    method and the same operator shape, so they do not disagree. Like
    ``linear_iterations`` it is ``None`` exactly when no linear solve happened.
    """

    x: torch.Tensor
    converged: torch.Tensor
    iterations: int
    residual_norm: torch.Tensor
    linear_iterations: torch.Tensor | None = None
    backend: str | None = None


def newton_tolerances(
    dtype: torch.dtype, atol: float | None, rtol: float | None
) -> tuple[float, float]:
    """``newton``'s ``(atol, rtol)``: each given value, or ``sqrt(finfo(dtype).eps)`` for
    one left ``None``, ``dtype`` being the residual's. An instance has converged when its
    residual's max-norm is below ``atol + rtol`` times its first residual's."""
    default_tol = math.sqrt(torch.finfo(dtype).eps)
    return (default_tol if atol is None else atol, default_tol if rtol is None else rtol)


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
    where: str = "newton",
    line_search: str | None = None,
    step_floor: float | None = None,
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

    ``where`` is the caller's own name for this solve, used to prefix BOTH Newton's own
    final-convergence error and every inner ``solvers.select.solve`` refusal or failure it
    triggers. It defaults to ``"newton"``, which is all a standalone call can say; a layer
    passes something like ``"PotentialFlowLayer 'air' solve"``, because slopes change
    between Newton iterates and an instance can lose grounding mid-iteration (a fan reaching
    shutoff, a damper closing) -- and in a composed model over one network, "newton:
    method='auto' refuses to split the batch" does not say WHICH layer failed.

    ``on_failure="raise"`` (default) raises ``RuntimeError`` naming the batch indices that
    failed to converge after ``max_iter`` iterations, exactly as before. ``on_failure="return"``
    is the explicit, non-default escape hatch: it returns the
    ``NewtonResult`` with ``converged`` reflecting the true per-instance state instead of
    raising, for a caller (e.g. a calibration loop, or a differentiable forward pass that
    passes ``on_failure`` through) that would rather inspect or down-weight a failed
    instance than abort. It is never silent: ``converged`` and ``residual_norm`` still carry
    the true state either way.

    ``line_search="convex"`` declares that ``residual`` is the gradient of a convex function
    ``E`` (its Jacobian is symmetric positive semidefinite everywhere, e.g. a network whose
    branch flows are each non-decreasing in their own potential difference and whose nodal
    withdrawals are each non-decreasing in their own potential). The Newton direction ``d``
    is then a descent direction of ``E``, and along it ``phi'(t) = residual(x + t d) . d``
    is non-decreasing. A step that overshoots the minimum of ``E`` along ``d`` -- ``phi'``
    at the step above `LINE_SEARCH_OVERSHOOT` times ``|phi'(0)|``, or positive with a
    residual that did not shrink -- is cut back to that minimum by a safeguarded regula
    falsi (Illinois) on ``phi'``, at one residual evaluation per trial. Without it Newton
    can cycle for ever on a law with kinks (an S-shaped flow, or a stiff bed meeting a soft
    storage line), which a damped step does not cure; with it every accepted step lowers
    ``E``. A step that does not overshoot is taken unchanged, so a smooth problem iterates
    exactly as without the search. An instance cut back is not declared converged on that
    iteration (its secant search stops on a slope test, leaving a residual that can sit just
    under the tolerance); the next, full step finishes it. ``None`` (default) is the plain
    damped iteration.

    ``step_floor`` (ULPs, default ``None``: off) ends an instance's iteration as converged
    when its Newton step is within ``step_floor`` units in the last place of its iterate in
    every component (``|dx_i| <= step_floor eps max(|x_i|, 1)``): the residual is then at
    the round-off floor of its own terms, below which no representable step can take it,
    and an ``atol`` under that floor (a stiff law at a large potential) would otherwise
    spend every iteration standing still and fail.
    """
    x = x0
    r = residual(x)
    atol, rtol = newton_tolerances(r.dtype, atol, rtol)
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
    # One dict, refilled by every inner solve, so the resolved backend costs one string
    # assignment per Newton step rather than a dict allocation. See `NewtonResult.backend`.
    backend_out: dict = {}
    tiny = torch.finfo(norm.dtype).tiny

    while not bool(torch.all(converged)) and iterations < max_iter:
        raw = operator(x)
        # A bare dense tensor (the dense compatibility shim) keeps the dense LU numerics
        # it has always had; a genuine LinearOperator goes through the eligibility table.
        step_method = "direct" if method == "auto" and isinstance(raw, torch.Tensor) else method
        result = select_solve(
            as_operator(raw),
            r,
            method=step_method,
            on_failure="return",
            where=where,
            rtol=inner_solve_rtol(r.dtype),
            backend_out=backend_out,
        )
        dx = result.x
        inner = result.iterations.to(torch.int64).expand(converged.shape)
        linear_iterations = (
            inner.clone() if linear_iterations is None else torch.maximum(linear_iterations, inner)
        )
        step = torch.where(
            converged.unsqueeze(-1), torch.zeros_like(dx), omega_i.unsqueeze(-1) * dx
        )
        x_start, r_start = x, r
        cut = None
        x = x - step
        r = residual(x)
        prev_norm = norm
        norm = r.abs().amax(dim=-1)
        if line_search == "convex":
            x, r, norm, cut = _convex_line_search(
                residual, x_start, r_start, dx, omega_i, x, r, norm, prev_norm, converged
            )
        if step_floor is not None:
            # A step at the round-off of the iterate: no representable step improves it.
            floor = step_floor * torch.finfo(x.dtype).eps * x_start.abs().clamp_min(1.0)
            at_floor = (dx.abs() <= floor).all(dim=-1) & ~converged
        else:
            at_floor = None
        ratio = norm / prev_norm.clamp_min(tiny)
        omega_i = torch.where(ratio < switch_ratio, torch.ones_like(omega_i), omega_i)
        # A step that did not shrink an instance's residual means that instance is
        # overshooting or cycling -- an undamped step on a ``sign(dp) sqrt(|dp|)`` equation
        # maps ``dp`` to ``-dp`` exactly (a dead-end sqrt-law edge), so without this fallback
        # the iterate sat at constant residual until ``max_iter``. Return that instance to
        # the relaxed step; the switch above re-arms once its residual shrinks again.
        omega_i = torch.where(ratio >= 1.0, torch.full_like(omega_i, omega), omega_i)
        converged = norm < tol
        if cut is not None:
            # A cut-back ends where the secant search's slope test does, not where Newton's
            # quadratic convergence would: its residual can sit just under `tol` (error
            # ~ tol / slope, 1.5e-7 m on a flat full-pipe force main). One more, full
            # Newton step from there costs an iteration and removes the leftover.
            converged = converged & ~cut
        if at_floor is not None:
            converged = converged | at_floor
        iterations += 1

    if not bool(torch.all(converged)):
        if on_failure == "return":
            return NewtonResult(
                x=x,
                converged=converged,
                iterations=iterations,
                residual_norm=norm,
                linear_iterations=linear_iterations,
                backend=backend_out.get("backend"),
            )
        flat_converged = converged.reshape(-1)
        bad = torch.nonzero(~flat_converged, as_tuple=False).flatten()
        norms = norm.reshape(-1)[bad]
        raise RuntimeError(
            f"{where}: batch indices {bad.tolist()} failed to converge after "
            f"{iterations} iterations, residual norms {norms.tolist()}"
        )
    return NewtonResult(
        x=x,
        converged=converged,
        iterations=iterations,
        residual_norm=norm,
        linear_iterations=linear_iterations,
        backend=backend_out.get("backend"),
    )
