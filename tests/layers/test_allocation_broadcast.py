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


def _merge_net() -> Network:
    net = Network(dtype=F64)
    for n in "ABCD":
        net.add_node(n)
    net.add_edge("A", "C", kind="link")
    net.add_edge("B", "C", kind="link")
    net.add_edge("C", "D", kind="link")
    return net


@pytest.mark.parametrize("mode", ["hard", "smooth", "projection"])
def test_batched_gradients_equal_the_per_instance_ones(mode):
    """A batched step's gradient with respect to the state, requests and every parameter
    is the sum (requests: the stack) of the per-instance gradients; the unbatched tensors
    accumulate over the batch."""
    torch.manual_seed(0)
    s_max = torch.tensor([10.0, 10.0, 1.0, 10.0], dtype=F64, requires_grad=True)
    c_arc = torch.tensor([2.0, 2.0, 0.3], dtype=F64, requires_grad=True)
    pref = torch.tensor([1.0, 3.0, 1.0], dtype=F64, requires_grad=True)
    layer = AllocatedFlowLayer(
        _merge_net(), "cap", "link", mode=mode, tau=0.01 if mode == "smooth" else None,
        s_max=s_max, c_arc=c_arc, preference=pref,
    )
    s = torch.tensor([5.0, 5.0, 0.2, 0.0], dtype=F64, requires_grad=True)
    r = torch.tensor([[1.0, 1.0, 0.1], [0.5, 2.0, 0.5], [0.0, 0.0, 0.0]], dtype=F64,
                     requires_grad=True)
    w = torch.randn(3, 3, dtype=F64)
    s_new, f = layer.step(s, {"cap.requests": r}, 1.0)
    batched = torch.autograd.grad((f * w).sum() + (s_new**2).sum(), (s, r, s_max, c_arc, pref))
    expected = [torch.zeros_like(x) for x in (s, r, s_max, c_arc, pref)]
    for i in range(r.shape[0]):
        r_i = r[i].detach().clone().requires_grad_(True)
        s_i, f_i = layer.step(s, {"cap.requests": r_i}, 1.0)
        g = torch.autograd.grad((f_i * w[i]).sum() + (s_i**2).sum(),
                                (s, r_i, s_max, c_arc, pref))
        expected[1][i] = g[1]
        for k in (0, 2, 3, 4):
            expected[k] += g[k]
    for got, want in zip(batched, expected, strict=True):
        assert not torch.isnan(got).any()
        torch.testing.assert_close(got, want)


def test_a_batch_that_does_not_broadcast_is_refused_naming_the_layer():
    layer = AllocatedFlowLayer(
        _merge_net(), "cap", "link", s_max=torch.full((4,), 10.0, dtype=F64),
        c_arc=torch.ones(3, dtype=F64),
    )
    with pytest.raises(RuntimeError, match=r"AllocatedFlowLayer 'cap'.*batch shapes do not"):
        layer.step(torch.zeros(2, 4, dtype=F64), {"cap.requests": torch.ones(3, 3, dtype=F64)},
                   1.0)


def test_projection_bracket_is_per_instance_for_a_batch_mixing_scales(monkeypatch):
    """Each instance of a batch is bracketed (and bisected) as it would be alone: a request
    four orders of magnitude larger in one instance neither widens the others' bracket nor
    changes their result."""
    import noodl.layers.allocation as allocation

    seen = []
    real = allocation.solve_monotone

    def spy(f, lo, hi, *params, **kwargs):
        calls = [0]

        def counted(*args):
            calls[0] += 1
            return f(*args)

        out = real(counted, lo, hi, *params, **kwargs)
        seen.append((hi.clone(), calls[0]))
        return out

    monkeypatch.setattr(allocation, "solve_monotone", spy)
    layer = AllocatedFlowLayer(
        _merge_net(), "cap", "link", mode="projection",
        s_max=torch.tensor([1e5, 1e5, 1e4, 1e5], dtype=F64),
        c_arc=torch.tensor([1e5, 1e5, 1e4], dtype=F64),
    )
    s = torch.tensor([1e4, 1e4, 0.2, 0.0], dtype=F64)
    r = torch.tensor([[0.8, 0.9, 5.0], [8000.0, 9000.0, 1e4]], dtype=F64)
    _, f = layer.step(s, {"cap.requests": r}, 1.0)
    batch = list(seen)
    assert batch
    alone = []
    for i in range(2):
        seen.clear()
        _, f_i = layer.step(s, {"cap.requests": r[i]}, 1.0)
        alone.append(list(seen))
        assert len(seen) == len(batch)
        torch.testing.assert_close(f[i], f_i, rtol=1e-10, atol=1e-12)
    for k, (hi_b, n_b) in enumerate(batch):
        for i in range(2):
            torch.testing.assert_close(hi_b[i], alone[i][k][0])
        assert n_b <= max(alone[0][k][1], alone[1][k][1])
    assert any(hb[0].max() < hb[1].max() for hb, _ in batch)
