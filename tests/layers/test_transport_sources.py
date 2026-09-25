"""`sources` in FULL node order, and `TransportLayer.rate`."""

from __future__ import annotations

import pytest
import torch

from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64


def _chain(n_species: int = 1, scheme: str = "implicit"):
    """ambient -> z1 -> z2 -> ambient; capacities 10, 20; node order ambient, z1, z2."""
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name)
    net.add_edge("ambient", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient", kind="airpath")
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([10.0, 20.0], dtype=F64), flow_kind="airpath",
        boundary=["ambient"], n_species=n_species, scheme=scheme,
    )
    return net, layer


def test_step_takes_full_node_sources():
    _, layer = _chain()
    q = torch.ones(3, dtype=F64)
    s = torch.tensor([0.0, 3.0, 0.0], dtype=F64)
    x = layer.step(torch.zeros(2, dtype=F64), q, s, torch.zeros(1, dtype=F64), dt=1.0)
    # backward Euler: z1: 10 x1 = -x1 + 3 ; z2: 20 x2 = x1 - x2
    assert x[0].item() == pytest.approx(3.0 / 11.0)
    assert x[1].item() == pytest.approx(x[0].item() / 21.0)


def test_interior_only_sources_are_rejected_naming_the_expected_shape():
    _, layer = _chain()
    with pytest.raises(ValueError, match=r"co2.*FULL node order.*\(3,\)"):
        layer.step(
            torch.zeros(2, dtype=F64), torch.ones(3, dtype=F64), torch.zeros(2, dtype=F64),
            torch.zeros(1, dtype=F64), dt=1.0,
        )


def test_nonzero_source_on_a_boundary_node_raises_naming_layer_and_node():
    _, layer = _chain()
    s = torch.tensor([1.0, 0.0, 0.0], dtype=F64)
    with pytest.raises(ValueError, match=r"co2.*ambient"):
        layer.steady(torch.ones(3, dtype=F64), s, torch.zeros(1, dtype=F64))


def test_steady_with_full_node_sources_matches_the_hand_balance():
    _, layer = _chain()
    x = layer.steady(
        torch.ones(3, dtype=F64), torch.tensor([0.0, 3.0, 0.0], dtype=F64),
        torch.zeros(1, dtype=F64),
    )
    torch.testing.assert_close(x, torch.tensor([3.0, 3.0], dtype=F64))


def test_rate_is_zero_at_steady_state_and_matches_the_balance_elsewhere():
    _, layer = _chain()
    q = torch.ones(3, dtype=F64)
    s = torch.tensor([0.0, 3.0, 0.0], dtype=F64)
    xb = torch.zeros(1, dtype=F64)
    x_ss = layer.steady(q, s, xb)
    torch.testing.assert_close(layer.rate(x_ss, q, s, xb), torch.zeros(2, dtype=F64),
                               atol=1e-12, rtol=0.0)
    x = torch.tensor([1.0, 5.0], dtype=F64)
    expected = torch.tensor([(-1.0 + 3.0) / 10.0, (1.0 - 5.0) / 20.0], dtype=F64)
    torch.testing.assert_close(layer.rate(x, q, s, xb), expected)


def test_rate_broadcasts_a_batch_of_states():
    _, layer = _chain()
    q = torch.ones(3, dtype=F64)
    s = torch.tensor([0.0, 3.0, 0.0], dtype=F64)
    xb = torch.zeros(1, dtype=F64)
    xs = torch.tensor([[1.0, 5.0], [3.0, 3.0]], dtype=F64)
    r = layer.rate(xs, q, s, xb)
    assert r.shape == (2, 2)
    torch.testing.assert_close(r[1], torch.zeros(2, dtype=F64), atol=1e-12, rtol=0.0)


def test_multi_species_full_node_sources_have_trailing_shape_n_by_K():
    _, layer = _chain(n_species=2)
    s = torch.zeros(3, 2, dtype=F64)
    s[1, 0] = 3.0
    s[2, 1] = 4.0
    x = layer.steady(torch.ones(3, dtype=F64), s, torch.zeros(1, 2, dtype=F64))
    torch.testing.assert_close(x, torch.tensor([[3.0, 0.0], [3.0, 4.0]], dtype=F64))


def test_forcing_is_dead_code_and_has_been_removed():
    """`_forcing` (a helper that silently ignored a per-step `capacity=` override by calling
    `_capacity_stacked` without it) must not exist; pins its removal against a future
    accidental re-add."""
    assert not hasattr(TransportLayer, "_forcing")
