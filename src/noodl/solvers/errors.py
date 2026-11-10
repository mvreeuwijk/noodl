"""Exceptions the solvers raise."""

from __future__ import annotations


class ConvergenceError(RuntimeError):
    """A nonlinear solve did not converge within its iteration limit. A `RuntimeError`, so
    callers that catch that keep working; catch this to react to non-convergence alone (a
    retry from another starting point, say) and let every other failure through."""


class SingularSystemError(RuntimeError):
    """The linear system of a solve is singular: its potentials and flows do not determine
    each other (a loop of ideal potential sources, a cut set of flow sources). A
    `RuntimeError`, so callers that catch that keep working."""


class GroundingError(RuntimeError):
    """A network solve is not grounded: interior nodes with no path to a boundary potential
    (floating nodes), or a negative slope on an edge. A `RuntimeError`, so callers that
    catch that keep working."""


class LinearSolveError(RuntimeError):
    """A linear solve did not converge on some instance (`SolveResult.raise_on_failure`): an
    iterative solver ran out of iterations, or a direct one could not factorise. A
    `RuntimeError`, so callers that catch that keep working."""
