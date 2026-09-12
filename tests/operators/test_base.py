"""Tests for the LinearOperator protocol, SolveResult, and the retained DenseOperator oracle."""

from __future__ import annotations

import pytest
import torch

from tellegen.operators.base import SolverStatus


def test_solver_status_enum_values_match_the_authoritative_ordering():
    assert SolverStatus.CONVERGED == 0
    assert SolverStatus.MAX_ITER == 1
    assert SolverStatus.BREAKDOWN == 2
    assert SolverStatus.SINGULAR == 3
    assert SolverStatus.NOT_CERTIFIED == 4
