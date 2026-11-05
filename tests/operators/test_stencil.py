"""The shared Laplacian edge stencil against the dense `A diag(g) A^T`, batched."""

import pytest
import torch

from noodl.operators.stencil import (
    laplacian_apply,
    laplacian_coo,
    laplacian_coo_indices,
    laplacian_coo_values,
    laplacian_diagonal,
)

N = 5
# A multigraph (edge 0 and edge 4 are parallel) with a self-free, mixed-orientation edge set.
SRC = torch.tensor([0, 1, 2, 3, 0, 4])
TGT = torch.tensor([1, 2, 3, 4, 1, 2])
BATCH = (3, 2)


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _dense(g: torch.Tensor) -> torch.Tensor:
    """`A diag(g) A^T` from the incidence matrix, `(..., N, N)`."""
    a = torch.zeros(N, SRC.shape[0])
    a[SRC, torch.arange(SRC.shape[0])] += 1.0
    a[TGT, torch.arange(SRC.shape[0])] -= 1.0
    return torch.einsum("ne,...e,me->...nm", a, g, a)


def _g() -> torch.Tensor:
    gen = torch.Generator().manual_seed(7)
    return torch.rand(*BATCH, SRC.shape[0], generator=gen) + 0.1


def test_apply_matches_dense_matvec_batched():
    g = _g()
    x = torch.randn(*BATCH, N, generator=torch.Generator().manual_seed(8))
    src = SRC.expand(*BATCH, -1)
    tgt = TGT.expand(*BATCH, -1)
    out = laplacian_apply(x, src, tgt, g)
    expected = torch.einsum("...nm,...m->...n", _dense(g), x)
    assert out.shape == (*BATCH, N)
    torch.testing.assert_close(out, expected)


def test_apply_broadcasts_a_shared_conductance_over_the_batch():
    g = _g()[0, 0]
    x = torch.randn(*BATCH, N, generator=torch.Generator().manual_seed(9))
    out = laplacian_apply(x, SRC.expand(*BATCH, -1), TGT.expand(*BATCH, -1), g)
    torch.testing.assert_close(out, torch.einsum("nm,...m->...n", _dense(g), x))


def test_diagonal_matches_dense_diagonal_batched():
    g = _g()
    diag = laplacian_diagonal(SRC.expand(*BATCH, -1), TGT.expand(*BATCH, -1), g, N)
    torch.testing.assert_close(diag, torch.diagonal(_dense(g), dim1=-2, dim2=-1))


def test_coo_triplets_sum_to_the_dense_matrix_batched():
    g = _g()
    row, col, values = laplacian_coo(SRC, TGT, g)
    assert row.shape == col.shape == (4 * SRC.shape[0],)
    assert values.shape == (*BATCH, 4 * SRC.shape[0])
    dense = torch.zeros(*BATCH, N * N)
    dense = dense.index_add(-1, row * N + col, values).reshape(*BATCH, N, N)
    torch.testing.assert_close(dense, _dense(g))


def test_coo_indices_and_values_are_the_two_halves_of_coo():
    g = _g()
    row, col, values = laplacian_coo(SRC, TGT, g)
    row2, col2 = laplacian_coo_indices(SRC, TGT)
    assert torch.equal(row, row2) and torch.equal(col, col2)
    assert torch.equal(values, laplacian_coo_values(g))
