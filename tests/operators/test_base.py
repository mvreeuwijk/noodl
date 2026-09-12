"""Tests for the LinearOperator protocol, SolveResult, and the retained DenseOperator oracle."""

from __future__ import annotations

import pytest
import torch

from tellegen.operators.base import SolverStatus, SolveResult


def test_solver_status_enum_values_match_the_authoritative_ordering():
    assert SolverStatus.CONVERGED == 0
    assert SolverStatus.MAX_ITER == 1
    assert SolverStatus.BREAKDOWN == 2
    assert SolverStatus.SINGULAR == 3
    assert SolverStatus.NOT_CERTIFIED == 4


def _result(converged, status, residual, dtype=torch.float64):
    n = len(converged)
    return SolveResult(
        x=torch.zeros(n, dtype=dtype),
        converged=torch.tensor(converged, dtype=torch.bool),
        iterations=torch.zeros(n, dtype=torch.int64),
        residual=torch.tensor(residual, dtype=dtype),
        status=torch.tensor(status, dtype=torch.int64),
    )


def test_raise_on_failure_raises_naming_failing_batch_indices_statuses_and_residuals():
    result = _result(
        converged=[True, False, True, False],
        status=[0, 1, 0, 3],
        residual=[1e-10, 2.5e-2, 1e-11, 4.0],
    )
    with pytest.raises(RuntimeError, match=r"solve: batch indices \[1, 3\]") as excinfo:
        result.raise_on_failure("solve")
    assert "MAX_ITER" in str(excinfo.value)
    assert "SINGULAR" in str(excinfo.value)


def test_raise_on_failure_does_not_raise_when_every_instance_converged():
    result = _result(converged=[True, True], status=[0, 0], residual=[1e-12, 1e-13])
    returned = result.raise_on_failure("solve")
    assert returned is result
