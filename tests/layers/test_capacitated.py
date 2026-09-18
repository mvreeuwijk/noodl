"""Tests for CapacitatedTransferLayer: hard-clip mode, registered as a first-class Model layer."""

import pytest
import torch

from tellegen.layers.capacitated import CapacitatedTransferLayer
from tellegen.model import Model
from tellegen.topology import Network

F64 = torch.float64


def _chain_net() -> Network:
    """A -> B -> C, one edge kind 'link', two edges, three nodes."""
    net = Network()
    net.add_node("A")
    net.add_node("B")
    net.add_node("C")
    net.add_edge("A", "B", kind="link")
    net.add_edge("B", "C", kind="link")
    return net


def test_construction_rejects_wrong_c_arc_shape():
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    bad_c_arc = torch.tensor([1.0, 2.0, 3.0], dtype=F64)  # 3 edges, network has 2
    with pytest.raises(ValueError, match="c_arc"):
        CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=bad_c_arc)


def test_construction_rejects_wrong_s_max_shape():
    net = _chain_net()
    bad_s_max = torch.full((2,), 10.0, dtype=F64)  # 2, network has 3 nodes
    c_arc = torch.full((2,), 1.0, dtype=F64)
    with pytest.raises(ValueError, match="s_max"):
        CapacitatedTransferLayer(net, "cap", "link", s_max=bad_s_max, c_arc=c_arc)


def test_construction_rejects_unknown_mode():
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    c_arc = torch.full((2,), 1.0, dtype=F64)
    with pytest.raises(ValueError, match="mode"):
        CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="bogus")


def test_step_hard_clip_below_capacity_passes_request_through():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    s0 = torch.zeros(net.n, dtype=F64)
    drivers = {"cap.requests": torch.tensor([2.0, 2.0], dtype=F64)}
    s1, f = layer.step(s0, drivers, dt=1.0)
    assert torch.allclose(f, torch.tensor([2.0, 2.0], dtype=F64))
    # A: only a source, loses 2. B: gains 2 from A, loses 2 to C, net 0. C: gains 2.
    assert torch.allclose(s1, torch.tensor([-2.0, 0.0, 2.0], dtype=F64))


def test_step_hard_clip_above_arc_capacity_is_capped():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 1.5, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    s0 = torch.zeros(net.n, dtype=F64)
    drivers = {"cap.requests": torch.tensor([5.0, 5.0], dtype=F64)}
    _, f = layer.step(s0, drivers, dt=1.0)
    assert torch.allclose(f, torch.tensor([1.5, 1.5], dtype=F64))


def test_step_hard_clip_above_receiver_headroom_is_capped():
    net = _chain_net()
    # C already at s_max=1.0, so the B->C edge's headroom is 0.
    s_max = torch.tensor([100.0, 100.0, 1.0], dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    s0 = torch.tensor([0.0, 0.0, 1.0], dtype=F64)
    drivers = {"cap.requests": torch.tensor([2.0, 2.0], dtype=F64)}
    _, f = layer.step(s0, drivers, dt=1.0)
    assert torch.allclose(f, torch.tensor([2.0, 0.0], dtype=F64))


def test_step_missing_request_driver_raises_keyerror():
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    c_arc = torch.full((2,), 1.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    with pytest.raises(KeyError, match="cap.requests"):
        layer.step(torch.zeros(net.n, dtype=F64), {}, dt=1.0)


def test_model_registers_capacitated_layer():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    model = Model(net, {"cap": layer})
    assert model.capacitated == {"cap": layer}


def test_model_rejects_capacitated_layer_without_dt():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    model = Model(net, {"cap": layer})
    state = {"cap.s": torch.zeros(net.n, dtype=F64)}
    drivers = {"cap.requests": torch.tensor([2.0, 2.0], dtype=F64)}
    with pytest.raises(ValueError, match="dt"):
        model._pass(state, drivers, None, {})


def test_model_steps_capacitated_layer_end_to_end():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    model = Model(net, {"cap": layer})
    state = {"cap.s": torch.zeros(net.n, dtype=F64)}
    drivers = {"cap.requests": torch.tensor([2.0, 2.0], dtype=F64)}
    new_state = model.step(state, drivers, dt=1.0)
    assert torch.allclose(new_state["cap.q"], torch.tensor([2.0, 2.0], dtype=F64))
    assert torch.allclose(
        new_state["cap.s"], torch.tensor([-2.0, 0.0, 2.0], dtype=F64)
    )
