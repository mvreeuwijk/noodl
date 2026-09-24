"""`abs_column_sums` against the dense assembly, so `norm1_bound` is a true 1-norm."""

from __future__ import annotations

import pytest
import torch

from noodl.layers.transport import TransportLayer
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
    q = torch.randn(3, 6, generator=g, dtype=F64)  # batch of 3 signed flow fields
    return layer._advection_operator(q)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("conduction", [False, True])
@pytest.mark.parametrize("kinetics", [False, True])
@pytest.mark.parametrize("removal", [False, True])
@pytest.mark.parametrize("n_species", [1, 2])
def test_abs_column_sums_match_the_dense_assembly(seed, conduction, kinetics, removal, n_species):
    op = _random_layer(seed, conduction=conduction, kinetics=kinetics, removal=removal,
                       n_species=n_species)
    dense = op.assemble().abs().sum(dim=-2)          # (..., m): column sums of |M|
    torch.testing.assert_close(op.abs_column_sums(), dense, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(op.norm1_bound(), dense.amax(-1), rtol=1e-12, atol=1e-12)


def test_abs_column_sums_match_the_dense_assembly_for_a_kinetics_only_batch():
    """A batch that lives ONLY in kinetics (P2-5): flow, capacity and transmission unbatched."""
    g = torch.Generator().manual_seed(7)
    net = Network(dtype=F64)
    for n in ("ambient", "a", "b"):
        net.add_node(n)
    net.add_edge("ambient", "a", kind="flow")
    net.add_edge("a", "b", kind="flow")
    net.add_edge("b", "ambient", kind="flow")
    kinetics = torch.randn(3, 2, 2, 2, generator=g, dtype=F64)        # (batch 3, n_i 2, K, K)
    layer = TransportLayer(net, "c", capacity=torch.ones(2, dtype=F64), flow_kind="flow",
                           boundary=["ambient"], n_species=2, kinetics=kinetics)
    op = layer._advection_operator(torch.ones(3, dtype=F64))
    assert op.shape == (3, 4, 4)
    dense = op.assemble()
    torch.testing.assert_close(op.abs_column_sums(), dense.abs().sum(-2), rtol=1e-12, atol=1e-12)
