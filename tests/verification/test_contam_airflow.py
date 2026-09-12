"""CONTAM-style closed-form airflow verification for PotentialFlowLayer.

Each case is checked against an independent closed-form or scipy.optimize.brentq reference,
once for a single parameter instance and once for a batch of 64 random instances
(torch.manual_seed(0)), each batch element checked against its own per-instance reference.
"""

import torch
from scipy.optimize import brentq

from tellegen.drives import ConstantDrive
from tellegen.elements.fan import FanCurve
from tellegen.elements.fixed import FixedFlow
from tellegen.elements.powerlaw import PowerLaw
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.topology import Network

DTYPE = torch.float64


def _series_layer(C: torch.Tensor, n: torch.Tensor) -> tuple[Network, PotentialFlowLayer]:
    """ambient_w -> z1 -> z2 -> ambient_l, one PowerLaw per edge, wind drive on edge 0."""
    net = Network(dtype=DTYPE)
    for name in ("ambient_w", "z1", "z2", "ambient_l"):
        net.add_node(name)
    net.add_edge("ambient_w", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient_l", kind="airpath")
    element = PowerLaw(C, n, dp_transition=1e-6)
    drive = ConstantDrive(kind="airpath", key="wind")
    layer = PotentialFlowLayer(
        net, "series", [element], [drive], boundary=["ambient_w", "ambient_l"]
    )
    return net, layer


def _series_reference_q(C: list[float], n: float, Pw: float) -> float:
    """Single flow q solving sum_e (q / C_e)^(1/n) = Pw, by bisection with an expanding bracket."""

    def total_dp(q: float) -> float:
        return sum((q / Ci) ** (1.0 / n) for Ci in C) - Pw

    lo, hi = 0.0, 1.0
    while total_dp(hi) < 0.0:
        hi *= 2.0
    return brentq(total_dp, lo, hi, xtol=1e-14, rtol=1e-14)


def test_series_closed_form_single_instance():
    C = [0.010, 0.008, 0.012]
    n = 0.65
    Pw = 12.0
    net, layer = _series_layer(
        torch.tensor(C, dtype=DTYPE), torch.tensor(n, dtype=DTYPE)
    )
    drivers = {"wind": torch.tensor([Pw, 0.0, 0.0], dtype=DTYPE)}
    phi_boundary = torch.zeros(2, dtype=DTYPE)
    phi, q = layer.solve(phi_boundary, drivers, differentiable=False)

    q_ref = _series_reference_q(C, n, Pw)
    torch.testing.assert_close(
        q, torch.full((3,), q_ref, dtype=DTYPE), atol=1e-6, rtol=1e-6
    )

    c_series = sum(ci ** (-1.0 / n) for ci in C) ** (-n)
    assert abs(q_ref - c_series * Pw**n) < 1e-9

    phi_z1_ref = Pw - (q_ref / C[0]) ** (1.0 / n)
    phi_z2_ref = phi_z1_ref - (q_ref / C[1]) ** (1.0 / n)
    torch.testing.assert_close(
        phi[net.node_index("z1")], torch.tensor(phi_z1_ref, dtype=DTYPE), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        phi[net.node_index("z2")], torch.tensor(phi_z2_ref, dtype=DTYPE), atol=1e-6, rtol=1e-6
    )
    residual_last = phi_z2_ref - (q_ref / C[2]) ** (1.0 / n)
    assert abs(residual_last) < 1e-6


def test_series_closed_form_batched():
    torch.manual_seed(0)
    m = 64
    C = 0.004 + 0.02 * torch.rand(m, 3, dtype=DTYPE)
    n = 0.5 + 0.3 * torch.rand(m, 1, dtype=DTYPE)
    Pw = 5.0 + 45.0 * torch.rand(m, dtype=DTYPE)

    net, layer = _series_layer(C, n)
    wind = torch.zeros(m, 3, dtype=DTYPE)
    wind[:, 0] = Pw
    drivers = {"wind": wind}
    phi_boundary = torch.zeros(m, 2, dtype=DTYPE)
    phi, q = layer.solve(phi_boundary, drivers, differentiable=False)

    q_ref = torch.empty(m, dtype=DTYPE)
    for i in range(m):
        q_ref[i] = _series_reference_q(C[i].tolist(), n[i, 0].item(), Pw[i].item())

    torch.testing.assert_close(q[:, 0], q_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(q[:, 1], q_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(q[:, 2], q_ref, atol=1e-6, rtol=1e-6)

    c_series = (C ** (-1.0 / n)).sum(dim=-1) ** (-n.squeeze(-1))
    torch.testing.assert_close(q_ref, c_series * Pw**n.squeeze(-1), atol=1e-6, rtol=1e-6)
