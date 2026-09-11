"""Tests for PotentialFlowLayer: assembly, non-differentiable solve, power residual."""

import pytest
import torch
from scipy.optimize import brentq

from tellegen.drives import ConstantDrive
from tellegen.elements import Conductance, FixedFlow, PowerLaw
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.solvers.newton import newton
from tellegen.topology import Network


def _closed_form_series_flow(C, n, drive):
    def f(q):
        return sum((q / c) ** (1.0 / n) for c in C) - drive

    return brentq(f, 1e-9, 1e3)


def test_duplicate_element_kind_raises_value_error():
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")
    with pytest.raises(ValueError, match="airpath"):
        PotentialFlowLayer(
            net,
            "dup",
            [
                Conductance(torch.tensor([1.0], dtype=torch.float64), kind="airpath"),
                Conductance(torch.tensor([1.0], dtype=torch.float64), kind="airpath"),
            ],
        )


def test_element_kind_with_no_edges_raises_value_error():
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")
    with pytest.raises(ValueError, match="conduction"):
        PotentialFlowLayer(
            net,
            "missing",
            [Conductance(torch.tensor([1.0], dtype=torch.float64), kind="conduction")],
        )


def test_unknown_boundary_node_raises_key_error():
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")
    element = PowerLaw(torch.tensor([1.0], dtype=torch.float64), 0.5, kind="airpath")
    with pytest.raises(KeyError, match="zzz"):
        PotentialFlowLayer(net, "bad", [element], boundary=["zzz"])


def test_dp_flows_dflows_and_assemble_on_a_single_conductance_edge():
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="conduction")
    g = torch.tensor([2.0], dtype=torch.float64)
    layer = PotentialFlowLayer(net, "cond", [Conductance(g, kind="conduction")], boundary=["ambient"])

    phi = torch.tensor([0.0, 3.0], dtype=torch.float64)  # ambient, z
    dp = layer.dp(phi, {})
    torch.testing.assert_close(dp, torch.tensor([-3.0], dtype=torch.float64))

    q = layer.flows(phi, {})
    torch.testing.assert_close(q, torch.tensor([-6.0], dtype=torch.float64))

    dq = layer.dflows(phi, {})
    torch.testing.assert_close(dq, g)

    phi_i = torch.tensor([3.0], dtype=torch.float64)
    phi_b = torch.tensor([0.0], dtype=torch.float64)
    assembled = layer.assemble(phi_i, phi_b)
    torch.testing.assert_close(assembled, phi)


def test_residual_and_jacobian_on_single_conductance_edge():
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="conduction")
    g = torch.tensor([2.0], dtype=torch.float64)
    layer = PotentialFlowLayer(net, "cond", [Conductance(g, kind="conduction")], boundary=["ambient"])

    phi_b = torch.tensor([0.0], dtype=torch.float64)
    phi_i = torch.tensor([3.0], dtype=torch.float64)
    sources = torch.tensor([0.0, 1.0], dtype=torch.float64)  # ambient, z

    r = layer.residual(phi_i, phi_b, {}, sources)
    # q = g*(phi_ambient - phi_z) = 2*(0-3) = -6; A_I row for z is -1; A_I@q = 6; minus s_I(1) -> 5
    torch.testing.assert_close(r, torch.tensor([5.0], dtype=torch.float64))

    J = layer.jacobian(phi_i, phi_b, {})
    # dq/dphi_z = -g ; J = A_I diag(dq) A_I^T = (-1)*2*(-1) = 2
    torch.testing.assert_close(J, torch.tensor([[2.0]], dtype=torch.float64))
