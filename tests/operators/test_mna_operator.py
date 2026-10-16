import pytest
import torch

from noodl.operators.base import LinearOperator
from noodl.operators.mna import MnaOperator
from noodl.solvers.select import solve

F64 = torch.float64


def _random(batch=(), n=5, nnz=17, seed=0):
    g = torch.Generator().manual_seed(seed)
    row = torch.randint(0, n, (nnz,), generator=g)
    col = torch.randint(0, n, (nnz,), generator=g)
    # a diagonal so the matrix is comfortably non-singular
    row = torch.cat([row, torch.arange(n)])
    col = torch.cat([col, torch.arange(n)])
    values = torch.randn(batch + (nnz + n,), generator=g, dtype=F64)
    values[..., nnz:] += 10.0
    return row, col, values


def _dense(row, col, values, n):
    out = torch.zeros(values.shape[:-1] + (n * n,), dtype=F64)
    return out.index_add(-1, row * n + col, values).reshape(values.shape[:-1] + (n, n))


@pytest.mark.parametrize("batch", [(), (3,)])
def test_actions_match_the_dense_matrix_with_duplicates_summed(batch):
    row, col, values = _random(batch)
    op = MnaOperator(row, col, values, 5)
    dense = _dense(row, col, values, 5)
    x = torch.randn(batch + (5,), dtype=F64)
    assert isinstance(op, LinearOperator)
    torch.testing.assert_close(op.matvec(x), (dense @ x.unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(
        op.rmatvec(x), (dense.transpose(-1, -2) @ x.unsqueeze(-1)).squeeze(-1)
    )
    torch.testing.assert_close(op.diagonal(), torch.diagonal(dense, dim1=-2, dim2=-1))
    torch.testing.assert_close(op.assemble(), dense)
    assert op.assemble_sparse()[2] is values
    assert op.spd_certificate() is None and op.symmetric is False
    assert op.shape == batch + (5, 5)


@pytest.mark.parametrize("method", ["direct", "sparse_direct"])
def test_select_solve_handles_it(method):
    row, col, values = _random((2,))
    op = MnaOperator(row, col, values, 5)
    b = torch.randn(2, 5, dtype=F64)
    x = solve(op, b, method=method, rtol=1e-12).x
    torch.testing.assert_close(x, torch.linalg.solve(_dense(row, col, values, 5), b))


def test_unbatched_values_act_on_a_batched_vector():
    row, col, values = _random(())
    op = MnaOperator(row, col, values, 5)
    x = torch.randn(4, 5, dtype=F64)
    torch.testing.assert_close(op.matvec(x), x @ _dense(row, col, values, 5).T)
