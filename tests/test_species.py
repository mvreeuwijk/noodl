import math

import pytest
import torch

from tellegen.physics import SpeciesTransport, branch_flows
from tellegen.topology import Network


def one_zone():
    net = Network()
    net.add_node("ambient")
    net.add_node("Z")
    net.add_edge("Z", "ambient", kind="airpath")  # exhaust (tree)
    net.add_edge("ambient", "Z", kind="airpath")  # supply (loop)
    return net


def test_single_zone_approaches_analytic_steady_state():
    net = one_zone()
    V, Q, people, g = 1000.0, 0.5, 20.0, 5.2e-6  # m3, m3/s, persons, m3/s per person
    st = SpeciesTransport(net, volumes={"Z": V}, boundary=["ambient"])
    assert st.interior == ["Z"]
    q = branch_flows(net, torch.tensor([Q]))
    c = torch.tensor([420.0])
    src = torch.tensor([people * g * 1e6])
    c_out = torch.tensor([420.0])
    dt = 900.0
    for _ in range(4):
        c = st.step(c, q, src, c_out, dt)
    tau = V / Q
    expected = 420.0 + people * g * 1e6 / Q * (1 - math.exp(-4 * dt / tau))
    assert abs(c.item() - expected) < 1e-3


def test_sealed_zones_conserve_total_mass():
    net = Network()
    net.add_node("ambient")
    for z in ("A", "B"):
        net.add_node(z)
    net.add_edge("A", "B", kind="airpath")  # tree
    net.add_edge("B", "A", kind="airpath")  # exchange loop
    st = SpeciesTransport(net, volumes={"A": 100.0, "B": 300.0}, boundary=["ambient"])
    q = branch_flows(net, torch.tensor([0.05]))
    c = torch.tensor([1000.0, 400.0])
    total0 = (st.volumes * c).sum()
    for _ in range(10):
        c = st.step(c, q, torch.zeros(2), torch.tensor([420.0]), 900.0)
    assert torch.allclose((st.volumes * c).sum(), total0, rtol=1e-6)
    assert abs(c[0] - c[1]) < abs(1000.0 - 400.0)  # mixing towards each other


def test_zero_flow_is_pure_source_accumulation():
    net = one_zone()
    st = SpeciesTransport(net, volumes={"Z": 500.0}, boundary=["ambient"])
    q = torch.zeros(2)
    c = st.step(torch.tensor([600.0]), q, torch.tensor([50.0]), torch.tensor([420.0]), 900.0)
    assert torch.allclose(c, torch.tensor([600.0 + 50.0 * 900.0 / 500.0]))


def test_step_broadcasts_over_ensemble():
    net = one_zone()
    st = SpeciesTransport(net, volumes={"Z": 1000.0}, boundary=["ambient"])
    n = 7
    q = branch_flows(net, torch.rand(n, 1) + 0.1)
    c = 400 + 600 * torch.rand(n, 1)
    out = st.step(c, q, torch.zeros(n, 1), torch.full((n, 1), 420.0), 900.0)
    assert out.shape == (n, 1)
    ref = torch.stack(
        [st.step(c[i], q[i], torch.zeros(1), torch.tensor([420.0]), 900.0) for i in range(n)]
    )
    assert torch.allclose(out, ref, atol=1e-4)


def test_negative_flow_is_rejected():
    net = one_zone()
    st = SpeciesTransport(net, volumes={"Z": 1000.0}, boundary=["ambient"])
    with pytest.raises(ValueError, match="non-negative"):
        st.step(
            torch.tensor([500.0]),
            torch.tensor([-0.1, 0.1]),
            torch.zeros(1),
            torch.tensor([420.0]),
            900.0,
        )


def test_missing_volume_for_interior_node_raises():
    net = one_zone()
    with pytest.raises(KeyError):
        SpeciesTransport(net, volumes={}, boundary=["ambient"])
