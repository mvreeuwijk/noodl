"""Tests for the LinearOperator protocol, SolveResult, and the retained DenseOperator oracle."""

from __future__ import annotations

import pytest
import torch

from tellegen.operators.base import LinearOperator, SolveResult, SolverStatus
from tellegen.operators.dense import DenseOperator


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


class _HandRolledOperator:
    """Satisfies LinearOperator structurally, with no relation to DenseOperator."""

    def __init__(self):
        self.shape = (2, 2)
        self.dtype = torch.float64
        self.device = torch.device("cpu")
        self.symmetric = False

    def matvec(self, x):
        return x

    def rmatvec(self, x):
        return x

    def diagonal(self):
        return torch.ones(2, dtype=torch.float64)

    def assemble(self):
        return None

    def spd_certificate(self):
        return None


def test_a_hand_rolled_class_with_every_member_satisfies_the_protocol():
    assert isinstance(_HandRolledOperator(), LinearOperator)


def test_a_plain_object_missing_every_member_does_not_satisfy_the_protocol():
    class NotAnOperator:
        pass

    assert not isinstance(NotAnOperator(), LinearOperator)


def test_solver_status_and_solve_result_and_linear_operator_are_reexported_from_the_package():
    from tellegen.operators import LinearOperator as PackageLinearOperator
    from tellegen.operators import SolveResult as PackageSolveResult
    from tellegen.operators import SolverStatus as PackageSolverStatus

    assert PackageSolverStatus is SolverStatus
    assert PackageSolveResult is SolveResult
    assert PackageLinearOperator is LinearOperator


def test_dense_operator_is_reexported_from_the_operators_package():
    from tellegen.operators import DenseOperator as PackageDenseOperator

    assert PackageDenseOperator is DenseOperator


def test_dense_operator_rejects_a_non_square_tensor():
    with pytest.raises(ValueError, match=r"square"):
        DenseOperator(torch.randn(3, 4, dtype=torch.float64), symmetric=False)


def test_dense_operator_matvec_matches_explicit_matmul():
    torch.manual_seed(0)
    A = torch.randn(4, 3, 3, dtype=torch.float64)
    op = DenseOperator(A, symmetric=False)
    x = torch.randn(4, 3, dtype=torch.float64)
    expected = torch.einsum("...ij,...j->...i", A, x)
    torch.testing.assert_close(op.matvec(x), expected)


def test_dense_operator_rmatvec_equals_explicit_transpose_matvec():
    torch.manual_seed(0)
    A = torch.randn(4, 3, 3, dtype=torch.float64)
    op = DenseOperator(A, symmetric=False)
    x = torch.randn(4, 3, dtype=torch.float64)
    expected = torch.einsum("...ji,...j->...i", A, x)  # A^T @ x, written out elementwise
    torch.testing.assert_close(op.rmatvec(x), expected)


def test_dense_operator_diagonal_matches_torch_diagonal():
    torch.manual_seed(0)
    A = torch.randn(4, 3, 3, dtype=torch.float64)
    op = DenseOperator(A, symmetric=False)
    torch.testing.assert_close(op.diagonal(), torch.diagonal(A, dim1=-2, dim2=-1))


def test_dense_operator_assemble_returns_the_underlying_tensor():
    A = torch.randn(3, 3, dtype=torch.float64)
    op = DenseOperator(A, symmetric=False)
    assert op.assemble() is A


def test_dense_operator_spd_certificate_is_always_none():
    A = torch.eye(3, dtype=torch.float64)
    assert DenseOperator(A, symmetric=True).spd_certificate() is None
    B = 2 * torch.eye(3, dtype=torch.float64)
    assert DenseOperator(B, symmetric=False).spd_certificate() is None


def test_dense_operator_matvec_and_rmatvec_broadcast_over_arbitrary_leading_batch_dims():
    torch.manual_seed(0)
    A = torch.randn(5, 7, 3, 3, dtype=torch.float64)
    op = DenseOperator(A, symmetric=False)
    x = torch.randn(5, 7, 3, dtype=torch.float64)
    assert op.matvec(x).shape == (5, 7, 3)
    assert op.rmatvec(x).shape == (5, 7, 3)
    torch.testing.assert_close(op.matvec(x), torch.einsum("...ij,...j->...i", A, x))
    torch.testing.assert_close(op.rmatvec(x), torch.einsum("...ji,...j->...i", A, x))


def test_dense_operator_adjoint_identity_holds_for_matvec_and_rmatvec():
    # (Ax).y == x.(A^T y) is an algebraic identity of transposition and holds for EVERY A,
    # symmetric or not -- this checks rmatvec is genuinely the transpose action, not that A
    # is symmetric (see this task's "spec ambiguity" note above the Steps).
    torch.manual_seed(0)
    A = torch.randn(4, 3, 3, dtype=torch.float64)
    op = DenseOperator(A, symmetric=False)
    x = torch.randn(4, 3, dtype=torch.float64)
    y = torch.randn(4, 3, dtype=torch.float64)
    lhs = (op.matvec(x) * y).sum(-1)
    rhs = (x * op.rmatvec(y)).sum(-1)
    torch.testing.assert_close(lhs, rhs)


def test_matvec_equals_rmatvec_on_the_same_x_for_a_declared_symmetric_operator():
    torch.manual_seed(0)
    A = torch.randn(4, 3, 3, dtype=torch.float64)
    A_sym = A + A.transpose(-1, -2)
    op = DenseOperator(A_sym, symmetric=True)
    x = torch.randn(4, 3, dtype=torch.float64)
    torch.testing.assert_close(op.matvec(x), op.rmatvec(x))


def test_matvec_does_not_equal_rmatvec_on_the_same_x_for_a_nonsymmetric_operator():
    torch.manual_seed(0)
    A = torch.randn(4, 3, 3, dtype=torch.float64)
    op = DenseOperator(A, symmetric=False)
    x = torch.randn(4, 3, dtype=torch.float64)
    assert not torch.allclose(op.matvec(x), op.rmatvec(x))


def test_dense_operator_symmetric_defaults_to_false():
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)  # symmetric matrix, but
    op = DenseOperator(A)                                             # NOT declared: default wins
    assert op.symmetric is False
