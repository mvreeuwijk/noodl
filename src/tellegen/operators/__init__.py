"""Operators and the solver contract: LinearOperator, SolveResult, and their implementations."""

from tellegen.operators.base import LinearOperator, SolveResult, SolverStatus
from tellegen.operators.dense import DenseOperator
from tellegen.operators.graph import GraphLaplacianOperator

__all__ = ["DenseOperator", "GraphLaplacianOperator", "LinearOperator", "SolveResult", "SolverStatus"]
