"""AllocatedFlowLayer.step broadcasts state, requests and parameters to one batch shape."""

import pytest
import torch

from noodl.layers.allocation import AllocatedFlowLayer
from noodl.topology import Network

F64 = torch.float64


def _chain_net() -> Network:
    net = Network()
    for n in "ABC":
        net.add_node(n)
    net.add_edge("A", "B", kind="link")
    net.add_edge("B", "C", kind="link")
    return net


@pytest.mark.parametrize("mode", ["hard", "smooth", "projection"])
def test_batched_requests_broadcast_against_an_unbatched_state(mode):
    net = _chain_net()
    layer = AllocatedFlowLayer(
        net, "cap", "link", mode=mode, tau=0.01 if mode == "smooth" else None,
        s_max=torch.full((net.n,), 10.0, dtype=F64), c_arc=torch.ones(2, dtype=F64),
    )
    s = torch.zeros(net.n, dtype=F64)
    r = torch.tensor([[0.5, 0.0], [2.0, 0.5], [0.0, 0.0]], dtype=F64)
    s_new, f = layer.step(s, {"cap.requests": r}, 1.0)
    assert s_new.shape == (3, 3) and f.shape == (3, 2)
    for i in range(3):
        s_i, f_i = layer.step(s, {"cap.requests": r[i]}, 1.0)
        assert torch.allclose(s_new[i], s_i) and torch.allclose(f[i], f_i)


def test_a_batched_capacity_broadcasts_against_unbatched_requests_and_state():
    net = _chain_net()
    c_arc = torch.tensor([[1.0, 1.0], [0.2, 1.0]], dtype=F64)
    layer = AllocatedFlowLayer(
        net, "cap", "link", s_max=torch.full((net.n,), 10.0, dtype=F64), c_arc=c_arc,
    )
    s_new, f = layer.step(
        torch.zeros(net.n, dtype=F64), {"cap.requests": torch.ones(2, dtype=F64)}, 1.0
    )
    assert f.shape == (2, 2) and s_new.shape == (2, 3)
    assert f[1, 0].item() == pytest.approx(0.2)
