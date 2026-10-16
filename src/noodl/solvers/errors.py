"""Exceptions the solvers raise."""

from __future__ import annotations


class ConvergenceError(RuntimeError):
    """A nonlinear solve did not converge within its iteration limit. A `RuntimeError`, so
    callers that catch that keep working; catch this to react to non-convergence alone (a
    retry from another starting point, say) and let every other failure through."""
