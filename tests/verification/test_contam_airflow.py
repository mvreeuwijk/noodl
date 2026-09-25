"""CONTAM-style closed-form airflow verification for PotentialFlowLayer.

Each case is checked against an independent closed-form or scipy.optimize.brentq reference,
once for a single parameter instance and once for a batch of 64 random instances
(torch.manual_seed(0)), each batch element checked against its own per-instance reference.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
from scipy.optimize import brentq, fsolve

from noodl.drives import ConstantDrive
from noodl.elements.fan import FanCurve
from noodl.elements.fixed import FixedFlow
from noodl.elements.powerlaw import PowerLaw
from noodl.layers.potential import PotentialFlowLayer
from noodl.solvers.newton import newton
from noodl.topology import Network

DTYPE = torch.float64


def _series_layer(
    C: torch.Tensor, n: torch.Tensor, dtype: torch.dtype = DTYPE
) -> tuple[Network, PotentialFlowLayer]:
    """ambient_w -> z1 -> z2 -> ambient_l, one PowerLaw per edge, wind drive on edge 0.

    `dtype` defaults to the module's DTYPE (float64, used by every other case in this file);
    it is exposed so `test_series_closed_form_float32_end_to_end` below can build the exact
    same network entirely in float32, the project's declared default dtype.
    """
    net = Network(dtype=dtype)
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


def _parallel_layer(
    C1: torch.Tensor, C2: torch.Tensor, n: torch.Tensor
) -> tuple[Network, PotentialFlowLayer]:
    """Two parallel airpath edges a -> c (a multigraph), same exponent n."""
    net = Network(dtype=DTYPE)
    net.add_node("a")
    net.add_node("c")
    net.add_edge("a", "c", kind="airpath")
    net.add_edge("a", "c", kind="airpath")
    C = torch.stack([C1, C2], dim=-1)
    element = PowerLaw(C, n, dp_transition=1e-6)
    layer = PotentialFlowLayer(net, "parallel", [element], boundary=["a", "c"])
    return net, layer


def test_parallel_combination_single_instance():
    C1 = torch.tensor(0.020, dtype=DTYPE)
    C2 = torch.tensor(0.015, dtype=DTYPE)
    n = torch.tensor(0.6, dtype=DTYPE)
    Dp = torch.tensor(8.0, dtype=DTYPE)

    net, layer = _parallel_layer(C1, C2, n)
    phi_boundary = torch.stack([Dp, torch.zeros((), dtype=DTYPE)])
    phi, q = layer.solve(phi_boundary, differentiable=False)

    q1_ref = C1 * torch.sign(Dp) * Dp.abs() ** n
    q2_ref = C2 * torch.sign(Dp) * Dp.abs() ** n
    torch.testing.assert_close(q[0], q1_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(q[1], q2_ref, atol=1e-6, rtol=1e-6)

    c_par = C1 + C2
    torch.testing.assert_close(
        q.sum(), c_par * torch.sign(Dp) * Dp.abs() ** n, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(q[0] / q[1], C1 / C2, atol=1e-6, rtol=1e-6)


def test_parallel_combination_batched():
    torch.manual_seed(0)
    m = 64
    C1 = 0.005 + 0.03 * torch.rand(m, dtype=DTYPE)
    C2 = 0.005 + 0.03 * torch.rand(m, dtype=DTYPE)
    # n carries a trailing size-1 edge axis (matching the (m, 1) convention used for the
    # series/fan-driven/stack batched cases elsewhere in this file) so it broadcasts against
    # C's (m, 2) edge axis inside PowerLaw; a bare (m,) n
    # fails broadcasting against C's trailing edge dim of 2 (RuntimeError: size of
    # tensor a (2) must match size of tensor b (64)). n_flat below undoes this for the
    # per-edge closed-form reference, which needs n aligned with Dp's (m,) shape instead.
    n = 0.5 + 0.3 * torch.rand(m, 1, dtype=DTYPE)
    Dp = -20.0 + 40.0 * torch.rand(m, dtype=DTYPE)

    net, layer = _parallel_layer(C1, C2, n)
    phi_boundary = torch.stack([Dp, torch.zeros(m, dtype=DTYPE)], dim=-1)
    phi, q = layer.solve(phi_boundary, differentiable=False)

    n_flat = n.squeeze(-1)
    q1_ref = C1 * torch.sign(Dp) * Dp.abs() ** n_flat
    q2_ref = C2 * torch.sign(Dp) * Dp.abs() ** n_flat
    torch.testing.assert_close(q[:, 0], q1_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(q[:, 1], q2_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        q.sum(dim=-1), (C1 + C2) * torch.sign(Dp) * Dp.abs() ** n_flat, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(q[:, 0] / q[:, 1], C1 / C2, atol=1e-6, rtol=1e-6)


def _fan_driven_layer(
    C1: torch.Tensor, C2: torch.Tensor, n: torch.Tensor, q_fan: torch.Tensor
) -> tuple[Network, PotentialFlowLayer]:
    """Zone with two leakage paths to ambient plus a FixedFlow exhaust to ambient."""
    net = Network(dtype=DTYPE)
    net.add_node("zone")
    net.add_node("ambient")
    net.add_edge("zone", "ambient", kind="airpath")
    net.add_edge("zone", "ambient", kind="airpath")
    net.add_edge("zone", "ambient", kind="fan")
    leak = PowerLaw(torch.stack([C1, C2], dim=-1), n, dp_transition=1e-6)
    fan = FixedFlow(q_fan, kind="fan")
    layer = PotentialFlowLayer(net, "fan_driven", [leak, fan], boundary=["ambient"])
    return net, layer


def test_fan_driven_zone_pressure_single_instance():
    C1 = torch.tensor(0.020, dtype=DTYPE)
    C2 = torch.tensor(0.010, dtype=DTYPE)
    n = torch.tensor(0.65, dtype=DTYPE)
    q_fan = torch.tensor(0.05, dtype=DTYPE)

    net, layer = _fan_driven_layer(C1, C2, n, q_fan)
    phi_boundary = torch.zeros(1, dtype=DTYPE)
    phi, q = layer.solve(phi_boundary, differentiable=False)

    p_ref = -((q_fan / (C1 + C2)) ** (1.0 / n))
    torch.testing.assert_close(phi[net.node_index("zone")], p_ref, atol=1e-6, rtol=1e-6)

    # atol=1e-9 matches newton()'s own default convergence tolerance (atol=1e-9, rtol=1e-9);
    # atol=1e-10 would be tighter than the solver's guaranteed accuracy, and fails
    # deterministically here (observed residual ~2.09e-10).
    residual = layer.residual(phi[..., layer.interior], phi_boundary, {}, None)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-9, rtol=0.0)


def test_fan_driven_zone_pressure_batched():
    torch.manual_seed(0)
    m = 64
    C1 = 0.005 + 0.03 * torch.rand(m, dtype=DTYPE)
    C2 = 0.005 + 0.03 * torch.rand(m, dtype=DTYPE)
    n = 0.5 + 0.3 * torch.rand(m, 1, dtype=DTYPE)
    q_fan = 0.01 + 0.09 * torch.rand(m, 1, dtype=DTYPE)

    net, layer = _fan_driven_layer(C1, C2, n, q_fan)
    phi_boundary = torch.zeros(m, 1, dtype=DTYPE)
    # atol/rtol pinned explicitly: newton()'s default is now dtype-derived (about 1.5e-8 for
    # this test's float64 tensors, looser than the flat 1e-9 it used to be unconditionally),
    # and the residual check below asks for 1e-8, which the new default cannot reliably clear
    # for every one of the 64 random instances (observed: one instance's residual floored at
    # ~1.1e-8, just over the 1e-8 bound). Ask newton() for the tighter tolerance explicitly
    # instead of loosening this residual assertion.
    phi, q = layer.solve(phi_boundary, differentiable=False, atol=1e-10, rtol=1e-10)

    n_flat = n.squeeze(-1)
    q_fan_flat = q_fan.squeeze(-1)
    p_ref = -((q_fan_flat / (C1 + C2)) ** (1.0 / n_flat))
    torch.testing.assert_close(
        phi[:, net.node_index("zone")], p_ref, atol=1e-6, rtol=1e-6
    )

    residual = layer.residual(phi[..., layer.interior], phi_boundary, {}, None)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-8, rtol=0.0)


def _fan_curve_layer(
    a0: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor,
    q_max: torch.Tensor,
    C: torch.Tensor,
    n: torch.Tensor,
) -> tuple[Network, PotentialFlowLayer]:
    """Loop ambient -> z (FanCurve supply) -> ambient (PowerLaw return)."""
    net = Network(dtype=DTYPE)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="fan")
    net.add_edge("z", "ambient", kind="airpath")
    coeffs = torch.stack([a0, a1, a2, a3], dim=-1)
    # FanCurve's own default kind is "airpath" (same as PowerLaw's); without an explicit
    # kind="fan" here it collides with `leak`'s "airpath" kind and PotentialFlowLayer raises
    # "duplicate element kind 'airpath'".
    fan = FanCurve(coeffs, q_max, kind="fan")
    leak = PowerLaw(C, n, dp_transition=1e-6)
    layer = PotentialFlowLayer(net, "fan_curve", [fan, leak], boundary=["ambient"])
    return net, layer


def _fan_curve_reference_q(
    a0: float, a1: float, a2: float, a3: float, q_max: float, C: float, n: float
) -> float:
    """brentq root of P(q) = (q / C)^(1/n) for q in (0, q_max)."""

    def f(q: float) -> float:
        p = a0 + a1 * q + a2 * q**2 + a3 * q**3
        return p - (q / C) ** (1.0 / n)

    return brentq(f, 1e-9, q_max - 1e-9, xtol=1e-14, rtol=1e-14)


def test_fan_curve_loop_single_instance():
    a0, a1, a2, a3 = 150.0, -100.0, -80.0, 40.0
    q_max = 1.0
    C, n = 0.05, 0.5

    net, layer = _fan_curve_layer(
        torch.tensor(a0, dtype=DTYPE),
        torch.tensor(a1, dtype=DTYPE),
        torch.tensor(a2, dtype=DTYPE),
        torch.tensor(a3, dtype=DTYPE),
        torch.tensor(q_max, dtype=DTYPE),
        torch.tensor(C, dtype=DTYPE),
        torch.tensor(n, dtype=DTYPE),
    )
    phi_boundary = torch.zeros(1, dtype=DTYPE)
    phi, q = layer.solve(phi_boundary, differentiable=False)

    q_ref = _fan_curve_reference_q(a0, a1, a2, a3, q_max, C, n)
    torch.testing.assert_close(
        q[net.edge_index("fan")[0]], torch.tensor(q_ref, dtype=DTYPE), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        q[net.edge_index("airpath")[0]], torch.tensor(q_ref, dtype=DTYPE), atol=1e-6, rtol=1e-6
    )
    assert 0.0 < q_ref < q_max


def test_fan_curve_loop_batched():
    torch.manual_seed(0)
    m = 64
    # Every parameter below carries an explicit trailing width-1 edge axis ((m, 1) rather
    # than (m,)): both FanCurve (1 fan edge) and PowerLaw (1 airpath edge) here represent a
    # SINGLE edge each, and the layer's per-element dp slice for a width-1 block keeps that
    # trailing dim (shape (m, 1), not (m,)), per the shape convention documented on
    # PowerLaw/FanCurve. Bare (m,) tensors throughout would broadcast a (m,) parameter against a (m,
    # 1) dp slice as (m, m) instead of (m, 1) (RuntimeError: size of tensor a (128) must match size
    # of tensor b (2), from the (m, m) shape silently doubling the edge axis during linear_init's
    # concatenation).
    a0 = 100.0 + 100.0 * torch.rand(m, 1, dtype=DTYPE)
    a1 = -150.0 - 50.0 * torch.rand(m, 1, dtype=DTYPE)
    a2 = -100.0 - 50.0 * torch.rand(m, 1, dtype=DTYPE)
    a3 = 20.0 + 30.0 * torch.rand(m, 1, dtype=DTYPE)
    q_max = torch.full((m, 1), 1.0, dtype=DTYPE)
    C = 0.02 + 0.06 * torch.rand(m, 1, dtype=DTYPE)
    n = 0.4 + 0.3 * torch.rand(m, 1, dtype=DTYPE)

    net, layer = _fan_curve_layer(a0, a1, a2, a3, q_max, C, n)
    phi_boundary = torch.zeros(m, 1, dtype=DTYPE)
    phi, q = layer.solve(phi_boundary, differentiable=False)

    q_ref = torch.empty(m, dtype=DTYPE)
    for i in range(m):
        q_ref[i] = _fan_curve_reference_q(
            a0[i].item(), a1[i].item(), a2[i].item(), a3[i].item(),
            q_max[i].item(), C[i].item(), n[i].item(),
        )
    fan_col = net.edge_index("fan")[0]
    leak_col = net.edge_index("airpath")[0]
    torch.testing.assert_close(q[:, fan_col], q_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(q[:, leak_col], q_ref, atol=1e-6, rtol=1e-6)


def _stack_layer(
    C: torch.Tensor, n: torch.Tensor
) -> tuple[Network, PotentialFlowLayer]:
    """Three zones in a vertical stack: out_low -> z1 -> z2 -> z3 -> out_high."""
    net = Network(dtype=DTYPE)
    for name in ("out_low", "z1", "z2", "z3", "out_high"):
        net.add_node(name)
    net.add_edge("out_low", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "z3", kind="airpath")
    net.add_edge("z3", "out_high", kind="airpath")
    element = PowerLaw(C, n, dp_transition=1e-6)
    drive = ConstantDrive(kind="airpath", key="stack")
    layer = PotentialFlowLayer(
        net, "stack", [element], [drive], boundary=["out_low", "out_high"]
    )
    return net, layer


def _stack_reference(
    C: list[float], n: float, drive_values: list[float]
) -> tuple[float, float, float, float, float, float, float]:
    """Independent reference for the vertical stack: solve nodal conservation directly from the
    PowerLaw element law q = C sign(dp) |dp|^n, using scipy.optimize.fsolve on residuals written
    here from scratch -- no noodl solver, layer, or Network code is called anywhere in this
    function.

    Unknowns: phi at the three interior nodes z1, z2, z3 (phi at out_low and out_high is fixed at
    0, matching the tests' boundary condition). For each of the four edges e = (u, v) in the chain
    out_low -> z1 -> z2 -> z3 -> out_high, the branch law gives
        dp_e = phi_u - phi_v + drive_values[e],  q_e = C[e] * sign(dp_e) * |dp_e| ** n.
    Conservation (net outflow = 0, no interior sources) at each interior node is "flow in = flow
    out": q_(e-1) = q_e for the three consecutive edge pairs. This is a genuine 3-equation
    nonlinear system in (phi_z1, phi_z2, phi_z3); fsolve (MINPACK hybrd, a different algorithm
    from noodl's own damped Newton in solvers/newton.py) finds the root independently.

    Note (topology, not a weakness of this reference): this network is a single unbranched chain
    with no interior sources, so conservation forces the SAME q through all four edges no matter
    what the C/n values are or how the drives are distributed -- this is a structural fact, not
    something this reference could be built to avoid. What *does* depend on each edge's own C[e]
    and n individually is each interior node's own potential (phi_z1, phi_z2, phi_z3): a wrong
    C[e] on any single edge changes that edge's own dp_e = sign(q) * (|q| / C[e]) ** (1/n) and so
    shifts every downstream node's potential, even though q itself stays equal across edges. That
    is why the caller checks per-node phi (not just per-edge q) against this reference.
    """

    def q_of(dp: float, Ci: float) -> float:
        return Ci * math.copysign(abs(dp) ** n, dp) if dp != 0.0 else 0.0

    def residual(phi_interior: list[float]) -> list[float]:
        phi_z1, phi_z2, phi_z3 = phi_interior
        nodes = [0.0, phi_z1, phi_z2, phi_z3, 0.0]  # out_low, z1, z2, z3, out_high
        dps = [nodes[i] - nodes[i + 1] + drive_values[i] for i in range(4)]
        qs = [q_of(dps[i], C[i]) for i in range(4)]
        return [qs[1] - qs[0], qs[2] - qs[1], qs[3] - qs[2]]

    phi_z1, phi_z2, phi_z3 = fsolve(residual, [0.0, 0.0, 0.0], xtol=1e-13)
    nodes = [0.0, phi_z1, phi_z2, phi_z3, 0.0]
    dps = [nodes[i] - nodes[i + 1] + drive_values[i] for i in range(4)]
    qs = [q_of(dps[i], C[i]) for i in range(4)]
    return phi_z1, phi_z2, phi_z3, qs[0], qs[1], qs[2], qs[3]


def test_stack_conservation_and_antisymmetry_single_instance():
    C = torch.tensor([0.020, 0.030, 0.025, 0.018], dtype=DTYPE)
    n = torch.tensor(0.6, dtype=DTYPE)
    drive_values = torch.tensor([2.0, 1.5, 1.5, 2.0], dtype=DTYPE)  # rho g dz per edge

    net, layer = _stack_layer(C, n)
    phi_boundary = torch.zeros(2, dtype=DTYPE)

    phi, q = layer.solve(phi_boundary, {"stack": drive_values}, differentiable=False)
    residual = layer.residual(
        phi[..., layer.interior], phi_boundary, {"stack": drive_values}, None
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-9, rtol=0.0)
    power = layer.power_residual(phi, q, {"stack": drive_values})
    assert abs(power.item()) < 1e-7

    phi_rev, q_rev = layer.solve(
        phi_boundary, {"stack": -drive_values}, differentiable=False
    )
    torch.testing.assert_close(q_rev, -q, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(phi_rev, -phi, atol=1e-8, rtol=1e-8)

    # Independent reference (see _stack_reference's docstring): solved from the element law by
    # scipy.optimize.fsolve, with no noodl code involved. drive_values above (2.0, 1.5, 1.5,
    # 2.0) sum to 7.0 (nonzero -- the drives do not cancel) and are not all equal, so each edge's
    # own dp differs even though conservation forces the same q through all four.
    phi_z1_ref, phi_z2_ref, phi_z3_ref, q0_ref, q1_ref, q2_ref, q3_ref = _stack_reference(
        C.tolist(), n.item(), drive_values.tolist()
    )
    torch.testing.assert_close(
        phi[net.node_index("z1")], torch.tensor(phi_z1_ref, dtype=DTYPE), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        phi[net.node_index("z2")], torch.tensor(phi_z2_ref, dtype=DTYPE), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        phi[net.node_index("z3")], torch.tensor(phi_z3_ref, dtype=DTYPE), atol=1e-6, rtol=1e-6
    )
    q_ref = torch.tensor([q0_ref, q1_ref, q2_ref, q3_ref], dtype=DTYPE)
    torch.testing.assert_close(q, q_ref, atol=1e-6, rtol=1e-6)


def test_stack_conservation_and_antisymmetry_batched():
    torch.manual_seed(0)
    m = 64
    C = 0.005 + 0.03 * torch.rand(m, 4, dtype=DTYPE)
    n = 0.5 + 0.3 * torch.rand(m, 1, dtype=DTYPE)
    drive_values = -3.0 + 6.0 * torch.rand(m, 4, dtype=DTYPE)

    net, layer = _stack_layer(C, n)
    phi_boundary = torch.zeros(m, 2, dtype=DTYPE)

    phi, q = layer.solve(phi_boundary, {"stack": drive_values}, differentiable=False)
    residual = layer.residual(
        phi[..., layer.interior], phi_boundary, {"stack": drive_values}, None
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-7, rtol=0.0)

    phi_rev, q_rev = layer.solve(
        phi_boundary, {"stack": -drive_values}, differentiable=False
    )
    torch.testing.assert_close(q_rev, -q, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(phi_rev, -phi, atol=1e-6, rtol=1e-6)

    # Independent per-instance reference, matching the pattern of the series case: each of the
    # 64 random instances is checked against its own scipy.optimize.fsolve solution of
    # _stack_reference (element law only, no noodl code).
    phi_z1_ref = torch.empty(m, dtype=DTYPE)
    phi_z2_ref = torch.empty(m, dtype=DTYPE)
    phi_z3_ref = torch.empty(m, dtype=DTYPE)
    q_ref = torch.empty(m, 4, dtype=DTYPE)
    for i in range(m):
        p1, p2, p3, q0, q1, q2, q3 = _stack_reference(
            C[i].tolist(), n[i, 0].item(), drive_values[i].tolist()
        )
        phi_z1_ref[i], phi_z2_ref[i], phi_z3_ref[i] = p1, p2, p3
        q_ref[i] = torch.tensor([q0, q1, q2, q3], dtype=DTYPE)

    torch.testing.assert_close(
        phi[:, net.node_index("z1")], phi_z1_ref, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        phi[:, net.node_index("z2")], phi_z2_ref, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        phi[:, net.node_index("z3")], phi_z3_ref, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(q, q_ref, atol=1e-5, rtol=1e-5)


def test_linear_init_reduces_newton_iterations_single_instance():
    C = torch.tensor([0.010, 0.008, 0.012], dtype=DTYPE)
    n = torch.tensor(0.65, dtype=DTYPE)
    Pw = torch.tensor(30.0, dtype=DTYPE)

    net, layer = _series_layer(C, n)
    drivers = {"wind": torch.tensor([Pw, 0.0, 0.0], dtype=DTYPE)}
    phi_boundary = torch.zeros(2, dtype=DTYPE)
    sources = None

    def residual_fn(phi_i: torch.Tensor) -> torch.Tensor:
        return layer.residual(phi_i, phi_boundary, drivers, sources)

    def jacobian_fn(phi_i: torch.Tensor) -> torch.Tensor:
        return layer.jacobian(phi_i, phi_boundary, drivers)

    x0_zero = torch.zeros(2, dtype=DTYPE)
    x0_linear = layer.linear_init(phi_boundary, drivers, sources)

    result_zero = newton(residual_fn, jacobian_fn, x0_zero)
    result_linear = newton(residual_fn, jacobian_fn, x0_linear)

    assert result_linear.iterations <= result_zero.iterations
    torch.testing.assert_close(result_zero.x, result_linear.x, atol=1e-8, rtol=1e-8)


def test_linear_init_reduces_newton_iterations_batched():
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
    sources = None

    def residual_fn(phi_i: torch.Tensor) -> torch.Tensor:
        return layer.residual(phi_i, phi_boundary, drivers, sources)

    def jacobian_fn(phi_i: torch.Tensor) -> torch.Tensor:
        return layer.jacobian(phi_i, phi_boundary, drivers)

    x0_zero = torch.zeros(m, 2, dtype=DTYPE)
    x0_linear = layer.linear_init(phi_boundary, drivers, sources)

    result_zero = newton(residual_fn, jacobian_fn, x0_zero)
    result_linear = newton(residual_fn, jacobian_fn, x0_linear)

    assert result_linear.iterations <= result_zero.iterations
    torch.testing.assert_close(result_zero.x, result_linear.x, atol=1e-6, rtol=1e-6)


def test_golden_helper_round_trip(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import golden

    monkeypatch.setattr(golden, "_GOLDEN_DIR", tmp_path)
    golden.save_golden("scratch", {"value": 1.5, "list": [1, 2, 3]})
    assert golden.load_golden("scratch") == {"value": 1.5, "list": [1, 2, 3]}


def test_golden_matches_stored_reference():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from golden import load_golden

    golden = load_golden("contam_airflow")

    net, layer = _series_layer(
        torch.tensor([0.010, 0.008, 0.012], dtype=DTYPE), torch.tensor(0.65, dtype=DTYPE)
    )
    drivers = {"wind": torch.tensor([12.0, 0.0, 0.0], dtype=DTYPE)}
    phi, q = layer.solve(torch.zeros(2, dtype=DTYPE), drivers, differentiable=False)
    torch.testing.assert_close(
        phi, torch.tensor(golden["series"]["phi"], dtype=DTYPE), atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(
        q, torch.tensor(golden["series"]["q"], dtype=DTYPE), atol=1e-9, rtol=1e-9
    )

    net, layer = _parallel_layer(
        torch.tensor(0.020, dtype=DTYPE),
        torch.tensor(0.015, dtype=DTYPE),
        torch.tensor(0.6, dtype=DTYPE),
    )
    phi, q = layer.solve(torch.tensor([8.0, 0.0], dtype=DTYPE), differentiable=False)
    torch.testing.assert_close(
        phi, torch.tensor(golden["parallel"]["phi"], dtype=DTYPE), atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(
        q, torch.tensor(golden["parallel"]["q"], dtype=DTYPE), atol=1e-9, rtol=1e-9
    )

    net, layer = _fan_driven_layer(
        torch.tensor(0.020, dtype=DTYPE),
        torch.tensor(0.010, dtype=DTYPE),
        torch.tensor(0.65, dtype=DTYPE),
        torch.tensor(0.05, dtype=DTYPE),
    )
    phi, q = layer.solve(torch.zeros(1, dtype=DTYPE), differentiable=False)
    torch.testing.assert_close(
        phi, torch.tensor(golden["fan_driven"]["phi"], dtype=DTYPE), atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(
        q, torch.tensor(golden["fan_driven"]["q"], dtype=DTYPE), atol=1e-9, rtol=1e-9
    )

    net, layer = _fan_curve_layer(
        torch.tensor(150.0, dtype=DTYPE),
        torch.tensor(-100.0, dtype=DTYPE),
        torch.tensor(-80.0, dtype=DTYPE),
        torch.tensor(40.0, dtype=DTYPE),
        torch.tensor(1.0, dtype=DTYPE),
        torch.tensor(0.05, dtype=DTYPE),
        torch.tensor(0.5, dtype=DTYPE),
    )
    phi, q = layer.solve(torch.zeros(1, dtype=DTYPE), differentiable=False)
    torch.testing.assert_close(
        phi, torch.tensor(golden["fan_curve"]["phi"], dtype=DTYPE), atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(
        q, torch.tensor(golden["fan_curve"]["q"], dtype=DTYPE), atol=1e-9, rtol=1e-9
    )


# ---------------------------------------------------------------------------------------
# Every case above that exercises FixedFlow ("fan_driven") passes differentiable=False
# only, so the default (differentiable=True) path -- which computes the branch Jacobian by
# autograd rather than analytically -- would otherwise never be exercised by this suite for
# an element whose flow does not depend on dp at all (a path that can crash for FixedFlow at
# potential.py's autograd.grad call, in both the learnable=False and learnable=True
# constructions). The tests below
# parametrize the fan-driven case and two others over both paths, and add a direct
# same-inputs comparison between the two paths so a future regression here is caught by
# result disagreement, not just by one path failing to run.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("differentiable", [False, True])
def test_fan_driven_zone_pressure_matches_reference_on_both_paths(differentiable):
    C1 = torch.tensor(0.020, dtype=DTYPE)
    C2 = torch.tensor(0.010, dtype=DTYPE)
    n = torch.tensor(0.65, dtype=DTYPE)
    q_fan = torch.tensor(0.05, dtype=DTYPE)

    net, layer = _fan_driven_layer(C1, C2, n, q_fan)
    phi_boundary = torch.zeros(1, dtype=DTYPE)
    phi, q = layer.solve(phi_boundary, differentiable=differentiable)

    p_ref = -((q_fan / (C1 + C2)) ** (1.0 / n))
    torch.testing.assert_close(phi[net.node_index("zone")], p_ref, atol=1e-6, rtol=1e-6)

    residual = layer.residual(phi[..., layer.interior], phi_boundary, {}, None)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-8, rtol=0.0)


@pytest.mark.parametrize("differentiable", [False, True])
def test_series_closed_form_matches_reference_on_both_paths(differentiable):
    C = [0.010, 0.008, 0.012]
    n = 0.65
    Pw = 12.0
    net, layer = _series_layer(torch.tensor(C, dtype=DTYPE), torch.tensor(n, dtype=DTYPE))
    drivers = {"wind": torch.tensor([Pw, 0.0, 0.0], dtype=DTYPE)}
    phi_boundary = torch.zeros(2, dtype=DTYPE)
    phi, q = layer.solve(phi_boundary, drivers, differentiable=differentiable)

    q_ref = _series_reference_q(C, n, Pw)
    torch.testing.assert_close(q, torch.full((3,), q_ref, dtype=DTYPE), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("differentiable", [False, True])
def test_stack_conservation_matches_reference_on_both_paths(differentiable):
    C = torch.tensor([0.020, 0.030, 0.025, 0.018], dtype=DTYPE)
    n = torch.tensor(0.6, dtype=DTYPE)
    drive_values = torch.tensor([2.0, 1.5, 1.5, 2.0], dtype=DTYPE)

    net, layer = _stack_layer(C, n)
    phi_boundary = torch.zeros(2, dtype=DTYPE)
    phi, q = layer.solve(phi_boundary, {"stack": drive_values}, differentiable=differentiable)
    residual = layer.residual(
        phi[..., layer.interior], phi_boundary, {"stack": drive_values}, None
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-9, rtol=0.0)

    phi_z1_ref, phi_z2_ref, phi_z3_ref, q0_ref, q1_ref, q2_ref, q3_ref = _stack_reference(
        C.tolist(), n.item(), drive_values.tolist()
    )
    torch.testing.assert_close(
        phi[net.node_index("z1")], torch.tensor(phi_z1_ref, dtype=DTYPE), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        phi[net.node_index("z2")], torch.tensor(phi_z2_ref, dtype=DTYPE), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        phi[net.node_index("z3")], torch.tensor(phi_z3_ref, dtype=DTYPE), atol=1e-6, rtol=1e-6
    )
    q_ref = torch.tensor([q0_ref, q1_ref, q2_ref, q3_ref], dtype=DTYPE)
    torch.testing.assert_close(q, q_ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("learnable", [False, True])
def test_fan_driven_zone_differentiable_and_nondifferentiable_paths_agree(learnable):
    """Direct regression guard for the FixedFlow-on-the-default-path crash (MUST FIX 1):
    solve the same fan-driven problem on both paths and require them to agree, rather than
    only checking each path separately against the physical reference (which would not by
    itself catch two paths that happen to agree with each other but not with either
    reference, nor would it distinguish "differentiable=True crashes" from "differentiable=
    True runs but is silently wrong"). Parametrized over FixedFlow's own learnable flag
    because the two failure modes fixed in potential.py's _dflows_functional are distinct:
    learnable=False leaves flow without a grad_fn at all (flow does not depend on dp_slice
    and q0 is not a registered parameter), while learnable=True gives flow a grad_fn through
    q0 but still never through dp_slice, which is the case allow_unused=True guards.
    """
    C1 = torch.tensor(0.020, dtype=DTYPE)
    C2 = torch.tensor(0.010, dtype=DTYPE)
    n = torch.tensor(0.65, dtype=DTYPE)
    q_fan = torch.tensor(0.05, dtype=DTYPE)

    net = Network(dtype=DTYPE)
    net.add_node("zone")
    net.add_node("ambient")
    net.add_edge("zone", "ambient", kind="airpath")
    net.add_edge("zone", "ambient", kind="airpath")
    net.add_edge("zone", "ambient", kind="fan")
    leak = PowerLaw(torch.stack([C1, C2], dim=-1), n, dp_transition=1e-6)
    fan = FixedFlow(q_fan, kind="fan", learnable=learnable)
    layer = PotentialFlowLayer(net, "fan_driven", [leak, fan], boundary=["ambient"])

    phi_boundary = torch.zeros(1, dtype=DTYPE)
    phi_d, q_d = layer.solve(phi_boundary, differentiable=True)
    phi_nd, q_nd = layer.solve(phi_boundary, differentiable=False)

    torch.testing.assert_close(phi_d, phi_nd, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(q_d, q_nd, atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------------------
# A flat atol=rtol=1e-9 Newton default is unreachable in float32 (the project's declared
# default dtype -- see noodl/topology.py's Network(dtype: torch.dtype = torch.float32) and
# benchmarks/newton_scaling.py) at every network size tried. Every fixture, both
# benchmarks, the rest of this file, the performance test and the README quick start all
# use float64, so without this case nothing end-to-end runs in the declared default.
# This series case runs entirely in float32 (network, element parameters, drivers, boundary
# conditions, and the Newton solve all float32) and checks against the same independent
# brentq reference as test_series_closed_form_single_instance, at a tolerance appropriate to
# float32's roughly 7 significant decimal digits rather than float64's roughly 15.
# ---------------------------------------------------------------------------------------


def test_series_closed_form_float32_end_to_end():
    C = [0.010, 0.008, 0.012]
    n = 0.65
    Pw = 12.0
    dtype = torch.float32
    net, layer = _series_layer(
        torch.tensor(C, dtype=dtype), torch.tensor(n, dtype=dtype), dtype=dtype
    )
    drivers = {"wind": torch.tensor([Pw, 0.0, 0.0], dtype=dtype)}
    phi_boundary = torch.zeros(2, dtype=dtype)
    phi, q = layer.solve(phi_boundary, drivers, differentiable=False)

    assert q.dtype == torch.float32
    q_ref = _series_reference_q(C, n, Pw)
    torch.testing.assert_close(q, torch.full((3,), q_ref, dtype=dtype), atol=1e-4, rtol=1e-4)

    phi_z1_ref = Pw - (q_ref / C[0]) ** (1.0 / n)
    phi_z2_ref = phi_z1_ref - (q_ref / C[1]) ** (1.0 / n)
    torch.testing.assert_close(
        phi[net.node_index("z1")], torch.tensor(phi_z1_ref, dtype=dtype), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        phi[net.node_index("z2")], torch.tensor(phi_z2_ref, dtype=dtype), atol=1e-4, rtol=1e-4
    )
