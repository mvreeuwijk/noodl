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
from tellegen.solvers.newton import newton
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
    # C's (m, 2) edge axis inside PowerLaw; the brief's original draft used a bare (m,) n,
    # which fails broadcasting against C's trailing edge dim of 2 (RuntimeError: size of
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
    # the brief's original draft used atol=1e-10, tighter than the solver's guaranteed
    # accuracy, and failed deterministically here (observed residual ~2.09e-10).
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
    phi, q = layer.solve(phi_boundary, differentiable=False)

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
    # "duplicate element kind 'airpath'" (the brief's original draft omitted this argument).
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
    # PowerLaw/FanCurve. The brief's original draft used bare (m,) tensors throughout, which
    # broadcasts a (m,) parameter against a (m, 1) dp slice as (m, m) instead of (m, 1)
    # (RuntimeError: size of tensor a (128) must match size of tensor b (2), from the (m, m)
    # shape silently doubling the edge axis during linear_init's concatenation).
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
