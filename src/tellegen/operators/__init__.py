"""Operators and the solver contract: LinearOperator, SolveResult, and their implementations."""

from tellegen.operators.advection import AdvectionOperator
from tellegen.operators.base import (
    LinearOperator,
    SolveResult,
    SolverStatus,
    SparseAssembling,
    as_operator,
)
from tellegen.operators.dense import DenseOperator
from tellegen.operators.graph import GraphLaplacianOperator

__all__ = [
    "AdvectionOperator",
    "DenseOperator",
    "GraphLaplacianOperator",
    "LinearOperator",
    "SolveResult",
    "SolverStatus",
    "SparseAssembling",
    "as_operator",
]
