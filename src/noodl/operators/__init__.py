"""Operators and the solver contract: LinearOperator, SolveResult, and their implementations."""

from noodl.operators.advection import AdvectionOperator
from noodl.operators.base import (
    LinearOperator,
    SolveResult,
    SolverStatus,
    SparseAssembling,
    as_operator,
)
from noodl.operators.dense import DenseOperator
from noodl.operators.graph import GraphLaplacianOperator

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
