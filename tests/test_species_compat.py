"""SpeciesTransport must be a thin wrapper that delegates to TransportLayer."""

import torch

from noodl.layers.transport import TransportLayer
from noodl.physics import SpeciesTransport
from noodl.topology import Network


def two_zone_net() -> Network:
    net = Network()
    net.add_node("ambient")
    net.add_node("A")
    net.add_node("B")
    net.add_edge("A", "B", kind="airpath")   # tree
    net.add_edge("B", "A", kind="airpath")   # exchange loop
    net.add_edge("ambient", "A", kind="airpath")  # supply loop
    net.add_edge("A", "ambient", kind="airpath")  # exhaust, tree
    return net


def test_wrapper_delegates_to_a_transport_layer():
    net = two_zone_net()
    st = SpeciesTransport(net, volumes={"A": 100.0, "B": 300.0}, boundary=["ambient"])
    assert isinstance(st._layer, TransportLayer)


def test_wrapper_matches_a_directly_built_transport_layer_on_random_flows():
    net = two_zone_net()
    volumes = {"A": 100.0, "B": 300.0}
    st = SpeciesTransport(net, volumes=volumes, boundary=["ambient"])
    direct = TransportLayer(
        net, "species", capacity=torch.tensor([100.0, 300.0]), flow_kind="airpath",
        boundary=["ambient"], n_species=1, scheme="exact",
    )
    torch.manual_seed(0)
    q = torch.rand(4) + 0.1
    c = torch.tensor([500.0, 200.0])
    sources = torch.rand(2)
    c_boundary = torch.tensor([420.0])
    dt = 300.0

    # `direct.step`'s `sources` is FULL node order (spec 4.2); the wrapper's `sources` stays
    # interior-only and pads internally the same way, so this pads by hand for the direct call.
    full_sources = torch.zeros(net.n, dtype=torch.float64)
    full_sources[direct.interior_idx] = sources.double()

    out_wrapper = st.step(c, q, sources, c_boundary, dt)
    out_direct = direct.step(c.double(), q.double(), full_sources, c_boundary.double(), dt)
    torch.testing.assert_close(out_wrapper.double(), out_direct, rtol=1e-8, atol=1e-8)
