"""The fallback ladder of `KirchhoffLayer(fallback=)`.

Answers: when the Newton solve from the given start does not converge, which rungs are
tried (`FALLBACK_RUNGS`: a linear starting guess, a storage ramp), in which order, and what
counts as a failed rung. A mixin of `KirchhoffLayer`: `_solve_with_fallback` and `_ramp`
call back into the layer's `_solve`.
"""

from __future__ import annotations

import torch

from noodl.solvers.continuation import storage_ramp
from noodl.solvers.errors import (
    ConvergenceError,
    GroundingError,
    LinearSolveError,
    SingularSystemError,
)

#: The rungs of the fallback ladder (`KirchhoffLayer(fallback=)`), in the order of the docs.
FALLBACK_RUNGS = ("linear_guess", "storage_ramp")


def check_fallback(layer_name: str, fallback) -> tuple[str, ...]:
    """`fallback` as a tuple of rung names, each one of `FALLBACK_RUNGS` and given once;
    anything else is refused by name."""
    if isinstance(fallback, str):
        raise TypeError(
            f"KirchhoffLayer {layer_name!r}: fallback takes a sequence of rung names, e.g. "
            f"({fallback!r},), not the string {fallback!r}"
        )
    rungs = tuple(fallback)
    unknown = [r for r in rungs if r not in FALLBACK_RUNGS]
    if unknown:
        raise ValueError(
            f"KirchhoffLayer {layer_name!r}: unknown fallback rung(s) {unknown}; the rungs "
            f"are {list(FALLBACK_RUNGS)}"
        )
    if len(set(rungs)) != len(rungs):
        raise ValueError(
            f"KirchhoffLayer {layer_name!r}: fallback {list(rungs)} names a rung twice"
        )
    return rungs


def _converged(diagnostics: dict) -> bool:
    """Whether every instance of the solve that wrote `diagnostics` converged."""
    return bool(torch.as_tensor(diagnostics["converged"]).all())


class _FallbackMixin:
    """The `_fallback` methods of `KirchhoffLayer`; see the module docstring."""

    def _solve_with_fallback(self, rungs, phi_boundary, drivers, sources, phi0, *, q0,
                             diagnostics, **kwargs):
        """`_solve`, then, if Newton did not converge, the rungs of `rungs` in order until
        one does.

        `fallback` is the ladder climbed when Newton does not converge (`None`, the default,
        is the layer's own `fallback`; `()` none). It runs only then, in every `on_failure`
        mode, and tries its rungs in order until one converges; the whole batch is solved
        again on each rung:

        - `"linear_guess"`: solve again from the linear initial guess (`linear_init`). It is
          skipped when the failed solve already started there (a layer without flow
          unknowns and no `phi0`); the initial point with held storage of a layer without
          flow unknowns, which has no linear guess, starts again from zeros instead.
        - `"storage_ramp"`: from that same start (or, when the linear guess cannot be
          formed, from where the failed solve started), let the network settle through a
          fictitious storage: `g (x - x_prev)` added to every row, `g` a multiple of the
          row's own Jacobian diagonal, removed over four stages, each with a storage ten
          times smaller (`noodl.solvers.continuation.storage_ramp`), and solve again from
          where it settled. A potential (balance) row takes the diagonal's size, a capacity
          at the node; the row of a flow unknown takes it with its sign, an inertia of the
          flow (a relation may be written either way round, and a shift of the other sign
          would cancel its diagonal). A row with a zero diagonal (an ideal source; a node
          whose every edge is a flow unknown) gets none. The stages are a starting point and
          carry no gradient; the solve that follows is this one's own, so the gradient is at
          the exact root.

        `diagnostics`, when given, then also receives `"fallback"`, the rung that converged
        (`None` if the first solve did, or no rung did), and `"fallback_iterations"`, the
        storage stages' Newton iterations (0 for `"linear_guess"`); the other entries are
        the converged solve's. When no rung converges, the first failure stands: its error
        is raised (naming the rungs tried and the last one's error), or, under
        `on_failure="return"`/`"return_primal"`, its result and diagnostics are returned.
        Only non-convergence (`ConvergenceError`) of the first solve climbs the ladder; every
        other error of it passes through. A rung that raises any error has failed, and the
        next is tried. On a layer with flow unknowns the rank test that names a singular
        system diagnoses the first solve's failure only once every rung has failed too (it
        is raised then, in place of the first error), in every `on_failure` mode. With no
        fallback (the default) none of this runs, the rank test is made at once, and
        `diagnostics` gains no entry.
        """
        report: dict = {}
        first_error = None
        try:
            out = self._solve(phi_boundary, drivers, sources, phi0, q0=q0,
                              diagnostics=diagnostics, _report=report, **kwargs)
        except ConvergenceError as exc:
            first_error = exc
        else:
            if diagnostics is None or _converged(diagnostics):
                if diagnostics is not None:
                    diagnostics.update(fallback=None, fallback_iterations=0)
                return out
            first = (out, dict(diagnostics))
        # The rungs start from the linear guess, or from zeros at the held initial point of
        # a layer without flow unknowns (which has none).
        rung_start = "zeros" if report.get("held") else "linear"
        last_error = None
        for rung in rungs:
            if rung == "linear_guess" and report.get("start") == rung_start:
                continue
            ramp: dict = {}
            ramped = rung == "storage_ramp"
            try:
                # the storage ramp falls back to the failed solve's own start when the
                # linear guess cannot be formed (`_solve`)
                out = self._solve(phi_boundary, drivers, sources, phi0 if ramped else None,
                                  q0=q0 if ramped else None, diagnostics=diagnostics,
                                  _start="ramp" if ramped else "linear",
                                  _report=ramp, **kwargs)
            except (ConvergenceError, SingularSystemError, GroundingError,
                    LinearSolveError) as exc:
                # a rung that does not converge, whose linear guess is singular or not
                # grounded, or whose linear guess's solve does not converge, has failed; the
                # first failure still stands. Any other error (a law's own bug, say) is not
                # a failure of the rung and passes through.
                last_error = exc
                continue
            if diagnostics is None or _converged(diagnostics):
                if diagnostics is not None:
                    diagnostics.update(fallback=rung,
                                       fallback_iterations=ramp.get("ramp_iterations", 0))
                return out
        if first_error is not None:
            if "singular" in report:
                # the rank test of the first solve, diagnosed once every rung has failed
                raise report["singular"] from first_error
            if last_error is None:
                raise first_error
            raise ConvergenceError(
                f"{first_error}; the fallback {list(rungs)} did not converge either: "
                f"{last_error}"
            ) from last_error
        out, diag = first
        diagnostics.clear()
        diagnostics.update(diag, fallback=None, fallback_iterations=0)
        return out

    def _ramp(self, x0, phi_boundary, drivers, sources, step, newton_kwargs, report):
        """`storage_ramp` from `x0` on this solve's own primal residual and Jacobian
        (without gradients): the starting point of the fallback ladder's `"storage_ramp"`
        rung. The rows of the flow unknowns take a storage with the sign of their own
        diagonal (an inertia), the potential rows its size."""
        u = self._unknowns if step.layout is None else step.layout

        def residual(x):
            return self._residual(x, phi_boundary, drivers, sources, step)[0]

        def operator(x):
            return self._operator(x, phi_boundary, drivers, step)

        signed = None
        if u is not None:
            signed = torch.arange(u.n, device=x0.device) >= u.w * u.n_I
        stages: dict = {}
        x = storage_ramp(residual, operator, x0, signed=signed, diagnostics=stages,
                         **{k: v for k, v in newton_kwargs.items() if k != "on_failure"})
        if report is not None:
            report["ramp_iterations"] = stages["newton_iterations"]
        return x
