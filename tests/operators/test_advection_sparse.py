"""Parity tests for `AdvectionOperator.assemble_sparse`, `_AffineSystemOperator.assemble_sparse`
and `TransposeOperator.assemble_sparse`: the sparse COO form must scatter, via
`index_put_(accumulate=True)`, to exactly the same dense matrix as the operator's own
`assemble()`, to 1e-14 -- the dense code is the reference throughout.
"""

from __future__ import annotations

import pytest
import torch

from noodl.layers.transport import TransportLayer, _AffineSystemOperator
from noodl.solvers.implicit import TransposeOperator
from noodl.solvers.select import solve as _solve_operator
from noodl.topology import Network

F64 = torch.float64


def _random_layer(seed, *, conduction, kinetics, removal, n_species):
    g = torch.Generator().manual_seed(seed)
    net = Network(dtype=F64)
    names = ["ambient", "a", "b", "c", "d"]
    for n in names:
        net.add_node(n)
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "ambient"), ("ambient", "a"), ("b", "d")]
    for u, v in edges:
        net.add_edge(u, v, kind="flow")
    if conduction:
        net.add_edge("a", "c", kind="wall")
        net.add_edge("b", "ambient", kind="wall")
    K = n_species
    kw = {}
    if conduction:
        kw.update(conduction_kind="wall", conductance=torch.rand(2, generator=g, dtype=F64) + 0.1)
    if kinetics:
        kw["kinetics"] = torch.randn(4, K, K, generator=g, dtype=F64)
    if removal:
        kw["removal"] = torch.rand(4, K, generator=g, dtype=F64)
    layer = TransportLayer(
        net, "c", capacity=torch.rand(4, generator=g, dtype=F64) + 0.5, flow_kind="flow",
        boundary=["ambient"], n_species=K, transmission=torch.rand(6, generator=g, dtype=F64),
        **kw,
    )
    q = torch.randn(3, 6, generator=g, dtype=F64)  # batch of 3 SIGNED flow fields
    return layer._advection_operator(q)


def _dense_from_coo(
    row: torch.Tensor, col: torch.Tensor, values: torch.Tensor, m: int
) -> torch.Tensor:
    """Scatter a COO `(row, col, values)` triplet into a dense `(*batch, m, m)` tensor with
    `index_put_(accumulate=True)`, so duplicate `(row, col)` pairs are summed exactly as the
    real `scipy.sparse` consumer sums them. `row`/`col` are shared across the batch; `values`
    carries the batch leading, per the `SparseAssembling` contract.
    """
    batch_shape = values.shape[:-1]
    nnz = row.shape[-1]
    b_total = 1
    for d in batch_shape:
        b_total *= d
    dense_flat = torch.zeros(b_total, m, m, dtype=values.dtype)
    values_flat = values.reshape(b_total, nnz)
    batch_idx = torch.arange(b_total).view(-1, 1).expand(b_total, nnz).reshape(-1)
    row_idx = row.view(1, -1).expand(b_total, nnz).reshape(-1)
    col_idx = col.view(1, -1).expand(b_total, nnz).reshape(-1)
    dense_flat.index_put_((batch_idx, row_idx, col_idx), values_flat.reshape(-1), accumulate=True)
    return dense_flat.reshape(*batch_shape, m, m)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("conduction,kinetics,removal,n_species", [
    (False, False, False, 1), (True, False, False, 1), (False, True, True, 2),
    (True, True, True, 3),
])
def test_sparse_form_matches_the_dense_assembly(seed, conduction, kinetics, removal, n_species):
    op = _random_layer(seed, conduction=conduction, kinetics=kinetics, removal=removal,
                       n_species=n_species)
    row, col, values = op.assemble_sparse()
    m = op.shape[-1]
    assert row.dtype == torch.int64 and col.dtype == torch.int64
    assert values.shape == (*op.batch_shape, row.numel())
    dense = _dense_from_coo(row, col, values, m)
    torch.testing.assert_close(dense, op.assemble(), rtol=1e-14, atol=1e-15)


def test_sparse_form_matches_the_dense_assembly_with_a_self_loop_edge():
    net = Network(dtype=F64)
    for n in ("ambient", "a", "b"):
        net.add_node(n)
    net.add_edge("ambient", "a", kind="flow")
    net.add_edge("a", "b", kind="flow")
    net.add_edge("b", "ambient", kind="flow")
    net.add_edge("a", "a", kind="flow")  # self-loop
    layer = TransportLayer(net, "c", capacity=torch.tensor([10.0, 20.0], dtype=F64),
                           flow_kind="flow", boundary=["ambient"],
                           transmission=torch.rand(4, generator=torch.Generator().manual_seed(3),
                                                   dtype=F64))
    q = torch.tensor([[0.3, -0.2, 0.1, 0.5], [-0.1, 0.4, -0.3, -0.2]], dtype=F64)
    op = layer._advection_operator(q)
    row, col, values = op.assemble_sparse()
    dense = _dense_from_coo(row, col, values, op.shape[-1])
    torch.testing.assert_close(dense, op.assemble(), rtol=1e-14, atol=1e-15)


def test_sparse_form_matches_the_dense_assembly_with_an_inactive_node():
    """A node no edge of the layer's kinds touches: `_boundary_idx` differs from "not
    interior", but `assemble_sparse`'s interior filter must still agree with
    `assemble()`'s, which is built from `_interior_idx` alone.
    """
    net = Network(dtype=F64)
    for n in ("ambient", "a", "b", "isolated"):
        net.add_node(n)
    net.add_edge("ambient", "a", kind="flow")
    net.add_edge("a", "b", kind="flow")
    net.add_edge("b", "ambient", kind="flow")
    layer = TransportLayer(net, "c", capacity=torch.tensor([10.0, 20.0], dtype=F64),
                           flow_kind="flow", boundary=["ambient"])
    assert "isolated" in layer._inactive_names
    q = torch.tensor([0.3, -0.2, 0.1], dtype=F64)
    op = layer._advection_operator(q)
    row, col, values = op.assemble_sparse()
    dense = _dense_from_coo(row, col, values, op.shape[-1])
    torch.testing.assert_close(dense, op.assemble(), rtol=1e-14, atol=1e-15)


def test_sparse_form_matches_the_dense_assembly_with_a_zero_flow_edge():
    """One edge's flow is exactly zero: `flow >= 0` is TRUE there, so orientation A is
    nominally "active", but its weight `abs(flow)` is zero, so both orientations must
    contribute nothing at that edge -- matching `assemble()`, which is indifferent to
    which orientation a zero-weight edge is assigned.
    """
    net = Network(dtype=F64)
    for n in ("ambient", "a", "b"):
        net.add_node(n)
    net.add_edge("ambient", "a", kind="flow")
    net.add_edge("a", "b", kind="flow")
    net.add_edge("b", "ambient", kind="flow")
    layer = TransportLayer(net, "c", capacity=torch.tensor([10.0, 20.0], dtype=F64),
                           flow_kind="flow", boundary=["ambient"])
    q = torch.tensor([0.3, 0.0, -0.4], dtype=F64)
    op = layer._advection_operator(q)
    row, col, values = op.assemble_sparse()
    dense = _dense_from_coo(row, col, values, op.shape[-1])
    torch.testing.assert_close(dense, op.assemble(), rtol=1e-14, atol=1e-15)


def test_sparse_form_matches_the_dense_assembly_for_a_batched_conductance_only_operator():
    """P2-5 shape: a batch that lives ONLY in conductance -- flow, capacity and transmission
    unbatched.
    """
    net = Network(dtype=F64)
    for n in ("ambient", "a", "b"):
        net.add_node(n)
    net.add_edge("ambient", "a", kind="flow")
    net.add_edge("a", "b", kind="flow")
    net.add_edge("b", "ambient", kind="flow")
    net.add_edge("a", "b", kind="wall")
    conductance = torch.tensor([[2.0], [5.0], [0.5]], dtype=F64)  # (3, b_c=1), batched
    layer = TransportLayer(net, "c", capacity=torch.tensor([10.0, 20.0], dtype=F64),
                           flow_kind="flow", boundary=["ambient"],
                           conduction_kind="wall", conductance=conductance)
    q = torch.tensor([0.3, -0.2, 0.1], dtype=F64)
    op = layer._advection_operator(q)
    assert op.batch_shape == (3,)
    row, col, values = op.assemble_sparse()
    dense = _dense_from_coo(row, col, values, op.shape[-1])
    torch.testing.assert_close(dense, op.assemble(), rtol=1e-14, atol=1e-15)


def test_affine_system_sparse_form_matches_its_dense_form():
    op = _random_layer(1, conduction=True, kinetics=True, removal=True, n_species=2)
    system = _AffineSystemOperator(op, 0.7)
    row, col, values = system.assemble_sparse()
    dense = _dense_from_coo(row, col, values, op.shape[-1])
    torch.testing.assert_close(dense, system.assemble(), rtol=1e-14, atol=1e-15)


def test_transpose_view_swaps_the_sparse_form():
    op = _random_layer(2, conduction=False, kinetics=True, removal=False, n_species=2)
    row, col, values = TransposeOperator(op).assemble_sparse()
    r0, c0, v0 = op.assemble_sparse()
    assert torch.equal(row, c0) and torch.equal(col, r0) and torch.equal(values, v0)


def test_sparse_direct_matches_direct_for_the_affine_system_batched():
    op = _random_layer(0, conduction=True, kinetics=True, removal=True, n_species=2)
    system = _AffineSystemOperator(op, 0.4)
    m = system.shape[-1]
    b = torch.randn(3, m, dtype=F64, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        direct = _solve_operator(system, b, method="direct")
        sparse = _solve_operator(system, b, method="sparse_direct")
    assert bool(direct.converged.all())
    assert bool(sparse.converged.all())
    torch.testing.assert_close(sparse.x, direct.x, rtol=1e-12, atol=1e-12)


def test_sparse_direct_matches_direct_for_the_affine_system_unbatched():
    net = Network(dtype=F64)
    for n in ("ambient", "a", "b"):
        net.add_node(n)
    net.add_edge("ambient", "a", kind="flow")
    net.add_edge("a", "b", kind="flow")
    net.add_edge("b", "ambient", kind="flow")
    layer = TransportLayer(net, "c", capacity=torch.tensor([10.0, 20.0], dtype=F64),
                           flow_kind="flow", boundary=["ambient"])
    q = torch.tensor([0.3, -0.2, 0.1], dtype=F64)
    op = layer._advection_operator(q)
    system = _AffineSystemOperator(op, 0.5)
    m = system.shape[-1]
    b = torch.randn(m, dtype=F64, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        direct = _solve_operator(system, b, method="direct")
        sparse = _solve_operator(system, b, method="sparse_direct")
    assert bool(direct.converged.all())
    assert bool(sparse.converged.all())
    torch.testing.assert_close(sparse.x, direct.x, rtol=1e-12, atol=1e-12)
