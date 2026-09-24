"""Tests for the OPTIONAL `assemble_sparse()` operator member (spec section 6.2, Task C).

`assemble_sparse()` returns `(row, col, values)` in COO form: `row`/`col` are int64 index
arrays SHARED across the batch, `values` is batch-leading `(..., nnz)`. Duplicate
`(row, col)` pairs are allowed and are summed by the consumer (`scipy.sparse` does exactly
that), which is what lets `GraphLaplacianOperator` emit four entries per edge with no
coalescing pass.

The reference throughout is the operator's own dense `assemble()`: an operator's sparse form
must be the same matrix, to 1e-12, instance for instance.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import scipy.sparse
import torch

from noodl.operators.advection import AdvectionOperator
from noodl.operators.base import SparseAssembling
from noodl.operators.dense import DenseOperator
from noodl.operators.graph import GraphLaplacianOperator


@pytest.fixture(autouse=True)
def _set_float64_dtype():
    """Set default dtype to float64 for this module's tests, then restore."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old_dtype)


def _chain_op(slopes: torch.Tensor) -> GraphLaplacianOperator:
    """The milestone's chain fixture: 3 nodes, node 2 is the boundary node, edges (0,1), (1,2)."""
    src = torch.tensor([0, 1])
    tgt = torch.tensor([1, 2])
    interior_of_node = torch.tensor([0, 1, -1])
    boundary_mask = torch.tensor([False, False, True])
    return GraphLaplacianOperator(
        src, tgt, slopes, 2, interior_of_node, boundary_mask=boundary_mask
    )


def _to_dense(sparse_form, n: int, instance: int) -> torch.Tensor:
    """`scipy.sparse.coo_matrix((values[b], (row, col))).toarray()` as a torch tensor."""
    row, col, values = sparse_form
    v = values[instance] if values.dim() > 1 else values
    coo = scipy.sparse.coo_matrix(
        (v.detach().numpy(), (row.numpy(), col.numpy())), shape=(n, n)
    )
    return torch.from_numpy(coo.toarray())


def test_graph_assemble_sparse_matches_assemble_on_the_batched_chain_fixture():
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    op = _chain_op(slopes)
    dense = op.assemble()
    sparse_form = op.assemble_sparse()
    for b in range(slopes.shape[0]):
        torch.testing.assert_close(
            _to_dense(sparse_form, 2, b), dense[b], rtol=0.0, atol=1e-12
        )


def test_graph_assemble_sparse_returns_shared_int64_indices_and_batch_leading_values():
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    row, col, values = _chain_op(slopes).assemble_sparse()
    assert row.dtype is torch.int64 and col.dtype is torch.int64
    assert row.dim() == 1 and col.dim() == 1
    assert row.shape == col.shape
    # values is (..., nnz) with the OPERATOR's batch leading and one entry per index pair.
    assert values.shape == (2, row.shape[0])
    # Every emitted index is an INTERIOR index: an edge into a boundary node contributes
    # only its own diagonal entry, never a row or column in the (absent) boundary block.
    assert int(row.max()) < 2 and int(col.max()) < 2
    assert int(row.min()) >= 0 and int(col.min()) >= 0


def test_graph_assemble_sparse_matches_assemble_on_a_multi_edge_graph_with_boundaries():
    """A harder graph than the chain: parallel edges (duplicate COO entries), several
    boundary nodes (entries that must be dropped rather than folded into a wrong row), and
    an interior node whose only edges reach the boundary.
    """
    #        0(i) --- 1(i) === 1(i)  (parallel edge)      2(i) --- 3(b)      4(b) --- 0(i)
    src = torch.tensor([0, 0, 1, 2, 4, 3])
    tgt = torch.tensor([1, 1, 2, 3, 0, 4])  # edge (3,4) joins two BOUNDARY nodes
    boundary_mask = torch.tensor([False, False, False, True, True])
    interior_of_node = torch.tensor([0, 1, 2, -1, -1])
    slopes = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.5, 0.25, 7.0, 1.5, 2.5, 0.125]]
    )
    op = GraphLaplacianOperator(
        src, tgt, slopes, 3, interior_of_node, boundary_mask=boundary_mask
    )
    dense = op.assemble()
    sparse_form = op.assemble_sparse()
    for b in range(slopes.shape[0]):
        torch.testing.assert_close(
            _to_dense(sparse_form, 3, b), dense[b], rtol=0.0, atol=1e-12
        )


def test_graph_assemble_sparse_works_for_an_unbatched_slope_vector():
    slopes = torch.tensor([2.0, 3.0])
    op = _chain_op(slopes)
    row, col, values = op.assemble_sparse()
    assert values.shape == (row.shape[0],)
    coo = scipy.sparse.coo_matrix(
        (values.numpy(), (row.numpy(), col.numpy())), shape=(2, 2)
    )
    torch.testing.assert_close(
        torch.from_numpy(coo.toarray()), op.assemble(), rtol=0.0, atol=1e-12
    )


def test_graph_assemble_sparse_is_o_edges_with_no_python_loop():
    """The static "is it vectorised" proxy this suite uses elsewhere (see
    `test_matvec_source_has_no_python_loop_over_edges`), applied to the sparse assembly:
    O(E) means gather/scatter over edge arrays, never a Python loop over edges. The
    DOCSTRING is stripped first -- its prose says "for" in the English sense.
    """
    source = inspect.getsource(GraphLaplacianOperator.assemble_sparse)
    body = source.split('"""')[2]
    assert "for " not in body
    assert "while " not in body


def test_graph_assemble_sparse_emits_at_most_four_entries_per_edge():
    """O(E) in STORAGE as well as in time: the four-entries-per-edge stencil, minus whatever
    an incident boundary node removes -- never an n x n object anywhere on the path.
    """
    slopes = torch.ones(2, 2)
    row, _col, _values = _chain_op(slopes).assemble_sparse()
    assert row.shape[0] <= 4 * 2


def test_graph_assemble_sparse_preserves_the_autograd_graph_of_slopes():
    """`values` is a function of `slopes`, so differentiating through it must work even
    though the sparse SOLVE itself is non-differentiable: a caller assembling the sparse
    form under grad mode gets a connected tensor rather than a silently detached one.
    """
    slopes = torch.tensor([[1.0, 1.0], [2.0, 1.0]], requires_grad=True)
    _row, _col, values = _chain_op(slopes).assemble_sparse()
    assert values.requires_grad
    values.sum().backward()
    assert slopes.grad is not None


def test_dense_operator_assemble_sparse_returns_none():
    op = DenseOperator(torch.eye(3))
    assert op.assemble_sparse() is None


def test_advection_operator_assemble_sparse_matches_its_dense_assembly():
    """Task C's documented scope limit ("the transport path stays on GMRES, so the
    nonsymmetric advection operator declares no sparse form") was closed by Task 4 (B1):
    `AdvectionOperator` is still nonsymmetric and still uncertified (so `method="auto"`
    still routes it to GMRES, unaffected by this), but it now HAS a sparse form, for
    `method="sparse_direct"` and ILU to consume. See `tests/operators/test_advection_sparse.py`
    for the full parity suite (batches, conduction, kinetics, removal, self-loops, an
    inactive node, a zero-flow edge); this is the one-instance smoke test in this module's
    own reference style.
    """
    src = torch.tensor([0, 1])
    tgt = torch.tensor([1, 2])
    flow = torch.tensor([1.0, 1.0])
    transmission = torch.ones(1, 2)
    capacity = torch.ones(2)
    interior_of_node = torch.tensor([0, 1, -1])
    op = AdvectionOperator(src, tgt, flow, transmission, capacity, 2, interior_of_node)
    row, col, values = op.assemble_sparse()
    assert row.dtype is torch.int64 and col.dtype is torch.int64
    torch.testing.assert_close(
        _to_dense((row, col, values), 2, 0), op.assemble(), rtol=0.0, atol=1e-12
    )


def test_sparse_assembling_protocol_accepts_an_operator_that_declares_the_member():
    slopes = torch.ones(2)
    assert isinstance(_chain_op(slopes), SparseAssembling)
    assert isinstance(DenseOperator(torch.eye(2)), SparseAssembling)


def test_sparse_assembling_protocol_rejects_an_operator_without_the_member():
    class _NoSparseForm:
        def assemble(self):
            return None

    assert not isinstance(_NoSparseForm(), SparseAssembling)


def test_graph_assemble_sparse_row_and_col_are_transposes_of_each_other():
    """A_I diag(g) A_I^T is symmetric, so the COO index set must be its own transpose: the
    same matrix comes back from swapping row and col. This is what makes
    `TransposeOperator.assemble_sparse`'s swap correct for the adjoint solve.
    """
    slopes = torch.tensor([[1.0, 2.0]])
    row, col, values = _chain_op(slopes).assemble_sparse()
    forward = _to_dense((row, col, values), 2, 0)
    swapped = _to_dense((col, row, values), 2, 0)
    torch.testing.assert_close(forward, swapped, rtol=0.0, atol=1e-12)


def test_graph_assemble_sparse_numpy_conversion_needs_no_copy_of_an_n_by_n_object():
    """The whole point of the sparse path: nnz stays O(E), far below n^2, on a graph big
    enough for the difference to matter.
    """
    n = 200
    src = torch.arange(n - 1)
    tgt = torch.arange(1, n)
    boundary_mask = torch.zeros(n, dtype=torch.bool)
    boundary_mask[-1] = True
    interior_of_node = torch.cat([torch.arange(n - 1), torch.tensor([-1])])
    slopes = torch.ones(n - 1)
    op = GraphLaplacianOperator(
        src, tgt, slopes, n - 1, interior_of_node, boundary_mask=boundary_mask
    )
    row, col, values = op.assemble_sparse()
    assert row.shape[0] < (n - 1) ** 2 / 10
    coo = scipy.sparse.coo_matrix(
        (values.numpy(), (row.numpy(), col.numpy())), shape=(n - 1, n - 1)
    )
    assert np.allclose(coo.toarray(), op.assemble().numpy(), atol=1e-12)
