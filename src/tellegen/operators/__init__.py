"""Operators and the solver contract: LinearOperator, SolveResult, and their implementations."""

from tellegen.operators.base import LinearOperator, SolveResult, SolverStatus
from tellegen.operators.dense import DenseOperator

__all__ = ["DenseOperator", "LinearOperator", "SolveResult", "SolverStatus"]
