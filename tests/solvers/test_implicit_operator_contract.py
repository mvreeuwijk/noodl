"""Tests for the operator-based adjoint (Task 12): adjoint() solves against a
TransposeOperator wrapping op.rmatvec, and never reuses op.matvec as the transpose action
(which is only valid for a symmetric operator -- transport advection, and many coupled
Jacobians, are not).
"""

import pytest
import torch
from torch.autograd.gradcheck import GradcheckError

from tellegen.operators.dense import DenseOperator
from tellegen.solvers.implicit import TransposeOperator, adjoint, implicit_solve


def test_transpose_operator_matvec_is_wrapped_ops_rmatvec():
    A = torch.tensor([[4.0, 1.0], [0.0, 3.0]], dtype=torch.float64)  # deliberately asymmetric
    op = DenseOperator(A)
    top = TransposeOperator(op)
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    torch.testing.assert_close(top.matvec(x), A.T @ x, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(top.rmatvec(x), A @ x, atol=1e-12, rtol=1e-12)


def test_adjoint_accepts_a_plain_dense_tensor_backward_compatibly():
    J = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    grad_x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    lam = adjoint(J, grad_x)
    torch.testing.assert_close(J.T @ lam, grad_x, atol=1e-10, rtol=1e-10)


def test_adjoint_on_a_nonsymmetric_dense_operator_matches_explicit_transpose_solve():
    A = torch.tensor([[4.0, 1.0], [0.0, 3.0]], dtype=torch.float64)
    op = DenseOperator(A)
    grad_x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    lam = adjoint(op, grad_x)
    expected = torch.linalg.solve(A.T, grad_x)
    torch.testing.assert_close(lam, expected, atol=1e-9, rtol=1e-9)


class _BuggyReuseMatvecOperator:
    """A hand-built operator that (WRONGLY) reuses matvec as its own rmatvec -- exactly the
    hazard adjoint() must never reproduce. Used only to demonstrate the failure mode, never
    as a real operator in production code."""

    def __init__(self, A: torch.Tensor) -> None:
        self._A = A
        self.shape = A.shape
        self.dtype = A.dtype
        self.device = A.device
        self.symmetric = False

    def matvec(self, x):
        return torch.einsum("...ij,...j->...i", self._A, x)

    def rmatvec(self, x):
        return self.matvec(x)  # BUG: not A^T @ x, but A @ x again

    def diagonal(self):
        return torch.diagonal(self._A, dim1=-2, dim2=-1)

    def assemble(self):
        return self._A

    def spd_certificate(self):
        return None


def test_gradcheck_on_a_nonsymmetric_transport_shaped_problem_via_the_adjoint():
    """The problem this test guards: for a NONSYMMETRIC operator, adjoint()'s gradient is
    only correct if it solves against the operator's own rmatvec (A^T), never its matvec
    (A) again. A symmetric-only test suite (as test_implicit.py's existing gradcheck cases
    all are: their A is always [[4,1],[1,3]], symmetric) cannot distinguish a correct
    rmatvec-based adjoint from a buggy matvec-reused-as-rmatvec one, because A == A^T makes
    the two solves identical. This test uses a genuinely nonsymmetric A precisely so that a
    regression back to "rmatvec = matvec" would be caught: the correct adjoint (using
    DenseOperator, whose rmatvec is the real A^T) matches finite differences; the buggy
    operator (_BuggyReuseMatvecOperator, whose rmatvec silently reuses matvec) does not.
    """
    torch.manual_seed(0)
    A = torch.tensor([[4.0, 1.0], [0.0, 3.0]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([1.0, 2.0], dtype=torch.float64)
    x0 = torch.zeros(2, dtype=torch.float64)

    def residual(x, A_):
        return torch.einsum("ij,j->i", A_, x) - b

    def operator_correct(x, A_):
        return DenseOperator(A_)

    def f_correct(A_):
        return implicit_solve(residual, operator_correct, x0, (A_,), atol=1e-13, rtol=1e-13)

    assert torch.autograd.gradcheck(f_correct, (A,), eps=1e-6, atol=1e-5)

    def operator_buggy(x, A_):
        return _BuggyReuseMatvecOperator(A_)

    def f_buggy(A_):
        return implicit_solve(residual, operator_buggy, x0, (A_,), atol=1e-13, rtol=1e-13)

    # torch's own GradcheckError subclasses RuntimeError, NOT AssertionError, so the bare
    # `pytest.raises(AssertionError)` the task brief writes here would let the buggy case
    # escape uncaught (verified against this .venv's torch 2.14). Both are accepted so the
    # test pins "gradcheck rejects the buggy operator" regardless of which torch version
    # raises which; the `assert` in front keeps the AssertionError route live for a
    # `raise_exception=False`-style return value.
    with pytest.raises((AssertionError, GradcheckError)):
        assert torch.autograd.gradcheck(f_buggy, (A,), eps=1e-6, atol=1e-5)


def test_implicit_solve_refuses_on_failure_return_on_the_differentiable_path():
    # Final-review finding C2. This test previously asserted the WEAKER guarantee that the
    # forward could complete non-converged under on_failure="return" as long as the backward
    # raised. That is not enough: the backward raises only if the adjoint system itself is
    # degenerate, and at a merely non-converged (but perfectly nonsingular) point the adjoint
    # solve converges happily and hands back a gradient linearised at the wrong point, with
    # nothing reporting it. The escape hatch is therefore refused at the front door on the
    # differentiable path -- design section 3.2, "a wrong gradient is worse than no
    # gradient". `newton` itself still honours on_failure="return"; only the differentiable
    # wrapper refuses it.
    c = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    x0 = torch.tensor([[5.0]], dtype=torch.float64)

    def residual(x, c_):
        return x - c_

    def operator(x, c_):
        return torch.eye(x.shape[-1], dtype=x.dtype).expand(*x.shape, x.shape[-1])

    with pytest.raises(ValueError, match=r"on_failure='return'.*no defined adjoint"):
        implicit_solve(residual, operator, x0, (c,), max_iter=2, on_failure="return")


def test_backward_raises_unconditionally_on_a_singular_adjoint_system():
    # on_failure never applies to the backward pass (design section 3.2). Here the FORWARD
    # converges exactly (identity Jacobian on a linear residual), and the operator callable
    # is degenerate only at the converged point -- which is precisely where the adjoint
    # system is built. The backward must raise rather than return whatever the singular
    # solve produced.
    #
    # The `match` is not decoration: the pre-Task-12 dense `torch.linalg.solve` ALSO raised
    # here (LinAlgError is a RuntimeError subclass), so a bare `pytest.raises(RuntimeError)`
    # cannot tell the two implementations apart. What is new is that the failure is reported
    # through the operator contract's own raise/return boundary, naming the `where` it came
    # from and the failing batch instance -- this project's binding error convention.
    c = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    x0 = torch.tensor([[5.0]], dtype=torch.float64)

    def residual(x, c_):
        return x - c_

    def operator(x, c_):
        eye = torch.eye(x.shape[-1], dtype=x.dtype).expand(*x.shape, x.shape[-1])
        at_solution = bool((x - c_).abs().max() < 1e-9)
        return torch.zeros_like(eye) if at_solution else eye

    x = implicit_solve(residual, operator, x0, (c,), atol=1e-12, rtol=1e-12)
    assert torch.isfinite(x).all()
    torch.testing.assert_close(x, c.detach())  # the forward genuinely converged

    with pytest.raises(RuntimeError, match=r"implicit_solve backward.*batch indices \[0\]"):
        torch.autograd.grad(x.sum(), c)


# -- section 6.2 step 2: the transposed sparse form the adjoint solve needs -------------------


def _asymmetric_sparse_operator():
    """A deliberately ASYMMETRIC operator carrying the optional `assemble_sparse` member, so
    that a swap of row and col is observable rather than a no-op.
    """
    A = torch.tensor([[4.0, 1.0], [0.0, 3.0]], dtype=torch.float64)

    class _SparseDense(DenseOperator):
        def assemble_sparse(self):
            row, col = torch.nonzero(A != 0, as_tuple=True)
            return row.to(torch.int64), col.to(torch.int64), A[row, col]

    return A, _SparseDense(A)


def test_transpose_operator_assemble_sparse_swaps_row_and_col():
    A, op = _asymmetric_sparse_operator()
    row, col, values = op.assemble_sparse()
    trow, tcol, tvalues = TransposeOperator(op).assemble_sparse()
    torch.testing.assert_close(trow, col)
    torch.testing.assert_close(tcol, row)
    torch.testing.assert_close(tvalues, values)


def test_transpose_operator_assemble_sparse_is_the_transpose_of_assemble():
    A, op = _asymmetric_sparse_operator()
    trow, tcol, tvalues = TransposeOperator(op).assemble_sparse()
    dense = torch.zeros(2, 2, dtype=torch.float64)
    dense[trow, tcol] = tvalues
    torch.testing.assert_close(dense, A.T, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        dense, TransposeOperator(op).assemble(), atol=1e-12, rtol=1e-12
    )


def test_transpose_operator_assemble_sparse_is_none_when_the_wrapped_form_is_none():
    """`DenseOperator.assemble_sparse()` is None, and the transpose of "no sparse form" is
    still no sparse form -- not a crash on unpacking None.
    """
    op = DenseOperator(torch.eye(2, dtype=torch.float64))
    assert TransposeOperator(op).assemble_sparse() is None


def test_transpose_operator_has_no_assemble_sparse_gap_for_an_operator_without_the_member():
    """A wrapped operator that never declares the optional member at all: the view must say
    None rather than raise, so `select.solve`'s own ValueError is what a caller sees.
    """

    class _NoMember:
        shape = (2, 2)
        dtype = torch.float64
        device = torch.device("cpu")
        symmetric = False

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

    assert TransposeOperator(_NoMember()).assemble_sparse() is None


def test_adjoint_through_sparse_direct_matches_an_explicit_transpose_solve():
    A, op = _asymmetric_sparse_operator()
    grad_x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    lam = adjoint(op, grad_x, method="sparse_direct")
    torch.testing.assert_close(lam, torch.linalg.solve(A.T, grad_x), atol=1e-12, rtol=1e-12)
