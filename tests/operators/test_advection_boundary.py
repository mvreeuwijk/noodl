"""`boundary_net_inflow`: what leaves the interior arrives at the boundary nodes."""

from __future__ import annotations

import pytest
import torch

from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64


def _layer(seed, *, conduction, n_species, transmission_one):
    g = torch.Generator().manual_seed(seed)
    net = Network(dtype=F64)
    for n in ("amb", "out", "a", "b", "c"):
        net.add_node(n)
    for u, v in [("amb", "a"), ("a", "b"), ("b", "c"), ("c", "out"), ("b", "amb"), ("c", "a")]:
        net.add_edge(u, v, kind="flow")
    kw = {}
    if conduction:
        net.add_edge("a", "amb", kind="wall")
        net.add_edge("b", "c", kind="wall")
        kw.update(conduction_kind="wall", conductance=torch.rand(2, generator=g, dtype=F64) + 0.1)
    t = torch.ones(6, dtype=F64) if transmission_one else torch.rand(6, generator=g, dtype=F64)
    layer = TransportLayer(
        net, "c", capacity=torch.rand(3, generator=g, dtype=F64) + 0.5, flow_kind="flow",
        boundary=["amb", "out"], n_species=n_species, transmission=t, **kw,
    )
    q = torch.randn(4, 6, generator=g, dtype=F64)          # batch of 4 signed flow fields
    return layer, layer._advection_operator(q), g


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("conduction", [False, True])
@pytest.mark.parametrize("n_species", [1, 2])
def test_interior_loss_equals_boundary_gain_when_nothing_is_lost_in_transit(
    seed, conduction, n_species
):
    layer, op, g = _layer(seed, conduction=conduction, n_species=n_species, transmission_one=True)
    K, n_i, n_b = n_species, layer.n_i, layer.n_b
    x_i = torch.rand(4, K * n_i, generator=g, dtype=F64)
    x_b = torch.rand(4, K * n_b, generator=g, dtype=F64)
    cap = layer._capacity_stacked(F64).expand(4, K * n_i)
    interior_rate = cap * (op.matvec(x_i) + op.boundary_forcing(x_b))     # amount rate, interior
    boundary_rate = op.boundary_net_inflow(x_i, x_b)                        # amount rate, boundary
    torch.testing.assert_close(
        interior_rate.sum(-1) + boundary_rate.sum(-1), torch.zeros(4, dtype=F64),
        rtol=0, atol=1e-12,
    )


def test_hand_computed_two_node_case():
    """amb -> z (q = 2, x_amb = 3), z -> out (q = 5, x_z = 1), capacity 1: amount enters `out`
    at 5 * 1 = 5 and leaves `amb` at 2 * 3 = 6, so the net inflow to amb is -6."""
    net = Network(dtype=F64)
    for n in ("amb", "z", "out"):
        net.add_node(n)
    net.add_edge("amb", "z", kind="flow")
    net.add_edge("z", "out", kind="flow")
    layer = TransportLayer(net, "c", capacity=torch.tensor([1.0], dtype=F64), flow_kind="flow",
                           boundary=["amb", "out"])
    op = layer._advection_operator(torch.tensor([2.0, 5.0], dtype=F64))
    out = op.boundary_net_inflow(
        torch.tensor([1.0], dtype=F64), torch.tensor([3.0, 0.0], dtype=F64)
    )
    torch.testing.assert_close(out, torch.tensor([-6.0, 5.0], dtype=F64), rtol=0, atol=1e-14)


def test_transmission_below_one_reports_what_arrives():
    net = Network(dtype=F64)
    for n in ("amb", "z"):
        net.add_node(n)
    net.add_edge("z", "amb", kind="flow")
    layer = TransportLayer(net, "c", capacity=torch.tensor([1.0], dtype=F64), flow_kind="flow",
                           boundary=["amb"], transmission=torch.tensor([0.25], dtype=F64))
    op = layer._advection_operator(torch.tensor([4.0], dtype=F64))
    out = op.boundary_net_inflow(torch.tensor([2.0], dtype=F64), torch.tensor([0.0], dtype=F64))
    assert out.item() == pytest.approx(0.25 * 4.0 * 2.0)          # arrives: t * w * x_up
    interior_loss = (layer.capacity * op.matvec(torch.tensor([2.0], dtype=F64))).item()
    assert interior_loss == pytest.approx(-4.0 * 2.0)              # leaves: w * x_up
