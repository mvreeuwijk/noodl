"""ShiftedOperator: `scale * base + diag(shift)` over a graph Laplacian."""

import pytest
import torch

from noodl.operators.graph import GraphLaplacianOperator
from noodl.operators.shifted import ShiftedOperator


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _chain(slopes: torch.Tensor) -> GraphLaplacianOperator:
    """Interior nodes 0 and 1, node 2 the boundary; edges (0,1) and (1,2)."""
    return GraphLaplacianOperator(
        torch.tensor([0, 1]), torch.tensor([1, 2]), slopes, 2,
        torch.tensor([0, 1, -1]), boundary_mask=torch.tensor([False, False, True]),
    )


def test_a_non_negative_shift_keeps_the_spd_certificate():
    base = _chain(torch.tensor([1.0, 1.0]))
    shifted = ShiftedOperator(base, torch.tensor([0.0, 2.0]))
    cert = shifted.spd_certificate()
    assert cert is not None
    assert bool(cert)


def test_a_negative_shift_drops_the_spd_certificate():
    base = _chain(torch.tensor([1.0, 1.0]))
    assert base.spd_certificate() is not None
    assert ShiftedOperator(base, torch.tensor([0.5, -0.1])).spd_certificate() is None


def test_a_scaled_operator_never_certifies():
    base = _chain(torch.tensor([1.0, 1.0]))
    assert ShiftedOperator.implicit_step(base, 0.1).spd_certificate() is None


def test_diagonal_shift_matches_dense_in_every_form():
    slopes = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
    shift = torch.tensor([[0.3, 0.7], [1.0, 0.0]])
    op = ShiftedOperator(_chain(slopes), shift)
    dense = op.assemble()
    expected = _chain(slopes).assemble() + torch.diag_embed(shift)
    torch.testing.assert_close(dense, expected)
    x = torch.randn(2, 2, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(op.matvec(x), torch.einsum("bij,bj->bi", dense, x))
    torch.testing.assert_close(op.rmatvec(x), op.matvec(x))
    torch.testing.assert_close(op.diagonal(), torch.diagonal(dense, dim1=-2, dim2=-1))
    row, col, values = op.assemble_sparse()
    coo = torch.zeros(2, 4).index_add(-1, row * 2 + col, values).reshape(2, 2, 2)
    torch.testing.assert_close(coo, dense)
    assert op.shape == (2, 2, 2)


@pytest.mark.parametrize("storage", [None, torch.tensor([1.0, 0.0])])
def test_implicit_step_is_storage_minus_alpha_base(storage):
    slopes = torch.tensor([1.0, 2.0])
    base = _chain(slopes)
    alpha = 0.25
    op = ShiftedOperator.implicit_step(base, alpha, storage=storage)
    eye = torch.eye(2) if storage is None else torch.diag(storage)
    expected = eye - alpha * base.assemble()
    dense = op.assemble()
    torch.testing.assert_close(dense, expected)
    x = torch.tensor([0.4, -1.3])
    torch.testing.assert_close(op.matvec(x), expected @ x)
    torch.testing.assert_close(op.diagonal(), torch.diagonal(expected))
    row, col, values = op.assemble_sparse()
    coo = torch.zeros(4).index_add(-1, row * 2 + col, values).reshape(2, 2)
    torch.testing.assert_close(coo, expected)
    assert op.symmetric is True  # inherited from the base; the wrapper adds no asymmetry
