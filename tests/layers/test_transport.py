"""Tests for TransportLayer: exact multi-species advection-diffusion on graphs."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from tellegen.layers.transport import TransportLayer, _TransposeView
from tellegen.solvers.implicit import TransposeOperator
from tellegen.topology import Network


def flow_through_zone() -> Network:
    """ambient <-> Z, exhaust (tree) then supply (loop), both forward-oriented."""
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("Z")
    net.add_edge("Z", "ambient", kind="airpath")  # exhaust
    net.add_edge("ambient", "Z", kind="airpath")  # supply
    return net


def two_sealed_zones() -> Network:
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("A")
    net.add_node("B")
    net.add_edge("A", "B", kind="airpath")
    net.add_edge("B", "A", kind="airpath")
    return net


def test_single_zone_flow_through_matches_analytic_exponential():
    net = flow_through_zone()
    V, Q = 1000.0, 0.5
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([V]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([Q, Q], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    source = torch.tensor([1.0], dtype=torch.float64)
    dt = 300.0
    c = c0.clone()
    for _ in range(20):
        c = layer.step(c, q, source, c_out, dt)
    t = 20 * dt
    tau = V / Q
    c_ss = c_out + source / Q
    expected = c_ss + (c0 - c_ss) * math.exp(-t / tau)
    torch.testing.assert_close(c, expected, rtol=1e-4, atol=1e-4)


def test_two_sealed_zones_conserve_total_amount():
    net = two_sealed_zones()
    cap = torch.tensor([100.0, 300.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.05, 0.05], dtype=torch.float64)
    c = torch.tensor([1000.0, 400.0], dtype=torch.float64)
    total0 = (cap * c).sum()
    for _ in range(10):
        c = layer.step(c, q, torch.zeros(2, dtype=torch.float64), torch.tensor([420.0]), 900.0)
    total = (cap * c).sum()
    torch.testing.assert_close(total, total0, rtol=1e-9, atol=1e-9)


def test_reversed_flow_transports_in_reverse_direction():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q_fwd = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    forward = layer.step(c0, q_fwd, torch.zeros(1, dtype=torch.float64), c_out, 300.0)

    reversed_net = Network(dtype=torch.float64)
    reversed_net.add_node("ambient")
    reversed_net.add_node("Z")
    reversed_net.add_edge("ambient", "Z", kind="airpath")  # was Z->ambient
    reversed_net.add_edge("Z", "ambient", kind="airpath")  # was ambient->Z
    rev_layer = TransportLayer(
        reversed_net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath",
        boundary=["ambient"],
    )
    q_rev = torch.tensor([-0.5, -0.5], dtype=torch.float64)
    backward = rev_layer.step(c0, q_rev, torch.zeros(1, dtype=torch.float64), c_out, 300.0)
    torch.testing.assert_close(forward, backward, rtol=1e-10, atol=1e-10)


def test_transmission_half_halves_steady_state():
    net = flow_through_zone()
    V, Q = 1000.0, 0.5
    full = TransportLayer(net, "co2", capacity=torch.tensor([V]), flow_kind="airpath",
                           boundary=["ambient"])
    filtered = TransportLayer(
        net, "co2", capacity=torch.tensor([V]), flow_kind="airpath", boundary=["ambient"],
        transmission=torch.tensor([1.0, 0.5], dtype=torch.float64),  # halve the supply edge
    )
    q = torch.tensor([Q, Q], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    zero_source = torch.zeros(1, dtype=torch.float64)
    x_full = full.steady(q, zero_source, c_out)
    x_filtered = filtered.steady(q, zero_source, c_out)
    torch.testing.assert_close(x_filtered, 0.5 * x_full, rtol=1e-6, atol=1e-6)


def test_boundary_inflow_enters_interior():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c = torch.tensor([100.0], dtype=torch.float64)
    c_out_high = torch.tensor([800.0], dtype=torch.float64)
    out = layer.step(c, q, torch.zeros(1, dtype=torch.float64), c_out_high, 300.0)
    assert out.item() > c.item()


def test_steady_equals_long_time_step():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    x_steady = layer.steady(q, source, c_out)
    x_long = layer.step(c0, q, source, c_out, 1e6)
    torch.testing.assert_close(x_long, x_steady, rtol=1e-6, atol=1e-6)


def test_batched_step_equals_looped():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    n = 5
    q = 0.3 + 0.4 * torch.rand(n, 2, dtype=torch.float64)
    c = 100 + 300 * torch.rand(n, 1, dtype=torch.float64)
    source = torch.rand(n, 1, dtype=torch.float64)
    c_out = 400 + 40 * torch.rand(n, 1, dtype=torch.float64)
    out = layer.step(c, q, source, c_out, 300.0)
    ref = torch.stack(
        [layer.step(c[i], q[i], source[i], c_out[i], 300.0) for i in range(n)]
    )
    torch.testing.assert_close(out, ref, rtol=1e-8, atol=1e-8)


def test_wrong_shape_raises_value_error_naming_argument():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="sources"):
        layer.step(
            torch.tensor([100.0], dtype=torch.float64), q, torch.zeros(3, dtype=torch.float64),
            torch.tensor([420.0], dtype=torch.float64), 300.0,
        )


def test_gradcheck_step_wrt_x_q_sources_boundary():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    x = torch.tensor([150.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.4, 0.4], dtype=torch.float64, requires_grad=True)
    sources = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    x_b = torch.tensor([420.0], dtype=torch.float64, requires_grad=True)

    def f(x, q, sources, x_b):
        return layer.step(x, q, sources, x_b, 300.0)

    assert gradcheck(f, (x, q, sources, x_b), eps=1e-6, atol=1e-5)


def sealed_zone_with_flow_kind() -> Network:
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("Z")
    net.add_edge("Z", "ambient", kind="airpath")
    net.add_edge("ambient", "Z", kind="airpath")
    return net


def test_kinetics_matches_bateman_solution_for_decay_chain():
    net = sealed_zone_with_flow_kind()
    l1, l2 = 0.01, 0.02
    kinetics = torch.zeros(3, 3, dtype=torch.float64)
    kinetics[0, 0] = -l1  # A consumed
    kinetics[1, 0] = l1   # B produced from A
    kinetics[1, 1] = -l2  # B consumed
    kinetics[2, 1] = l2   # C produced from B
    layer = TransportLayer(
        net, "chain", capacity=torch.tensor([1000.0]), flow_kind="airpath",
        boundary=["ambient"], n_species=3, kinetics=kinetics,
    )
    q = torch.zeros(2, dtype=torch.float64)
    x = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)  # (n_i=1, K=3): A, B, C
    sources = torch.zeros(1, 3, dtype=torch.float64)
    x_b = torch.zeros(1, 3, dtype=torch.float64)
    dt = 5.0
    for _ in range(40):
        x = layer.step(x, q, sources, x_b, dt)
    t = 40 * dt
    A0 = 1.0
    A = A0 * math.exp(-l1 * t)
    B = A0 * l1 / (l2 - l1) * (math.exp(-l1 * t) - math.exp(-l2 * t))
    C = A0 - A - B
    torch.testing.assert_close(
        x[0], torch.tensor([A, B, C], dtype=torch.float64), rtol=1e-6, atol=1e-8
    )


def test_removal_rate_gives_exponential_decay():
    net = sealed_zone_with_flow_kind()
    rate = 0.02
    layer = TransportLayer(
        net, "particle", capacity=torch.tensor([500.0]), flow_kind="airpath",
        boundary=["ambient"], removal=torch.tensor([rate], dtype=torch.float64),
    )
    q = torch.zeros(2, dtype=torch.float64)
    x = torch.tensor([100.0], dtype=torch.float64)
    sources = torch.zeros(1, dtype=torch.float64)
    x_b = torch.zeros(1, dtype=torch.float64)
    dt = 10.0
    for _ in range(30):
        x = layer.step(x, q, sources, x_b, dt)
    expected = 100.0 * math.exp(-rate * 30 * dt)
    torch.testing.assert_close(
        x, torch.tensor([expected], dtype=torch.float64), rtol=1e-8, atol=1e-10
    )


def test_conduction_only_reaches_laplacian_steady_state():
    net = Network(dtype=torch.float64)
    net.add_node("Tb")
    net.add_node("T1")
    net.add_edge("T1", "Tb", kind="conduction")
    g1 = 5.0
    layer = TransportLayer(
        net, "heat", capacity=torch.tensor([1000.0]), flow_kind="conduction",
        boundary=["Tb"], conduction_kind="conduction",
        conductance=torch.tensor([g1], dtype=torch.float64),
    )
    q = torch.zeros(1, dtype=torch.float64)  # no advective flow: conduction only
    Tb = torch.tensor([15.0], dtype=torch.float64)
    S = torch.tensor([50.0], dtype=torch.float64)  # W, heat source at T1
    T_ss = layer.steady(q, S, Tb)
    expected = Tb + S / g1  # g1 (T1 - Tb) = S at steady state
    torch.testing.assert_close(T_ss, expected, rtol=1e-8, atol=1e-8)


def test_wrong_length_removal_raises_value_error_naming_argument():
    net = sealed_zone_with_flow_kind()
    with pytest.raises(ValueError, match="removal"):
        TransportLayer(
            net, "particle", capacity=torch.tensor([500.0]), flow_kind="airpath",
            boundary=["ambient"], removal=torch.tensor([0.1, 0.2], dtype=torch.float64),
        )


def test_scalar_capacity_raises_value_error_naming_argument():
    net = flow_through_zone()
    with pytest.raises(ValueError, match="capacity"):
        TransportLayer(
            net, "co2", capacity=torch.tensor(1000.0), flow_kind="airpath",
            boundary=["ambient"],
        )


def test_implicit_scheme_is_first_order_in_dt():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer_exact = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath",
                                  boundary=["ambient"], scheme="exact")
    layer_imp = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath",
                                boundary=["ambient"], scheme="implicit")
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)

    def error(dt: float, n: int) -> float:
        c_e, c_i = c0.clone(), c0.clone()
        for _ in range(n):
            c_e = layer_exact.step(c_e, q, source, c_out, dt)
            c_i = layer_imp.step(c_i, q, source, c_out, dt)
        return (c_i - c_e).abs().item()

    dt0, n0 = 200.0, 5
    e1 = error(dt0, n0)
    e2 = error(dt0 / 2, n0 * 2)
    ratio = e1 / e2
    assert 1.6 < ratio < 2.4  # first order: halving dt halves the error


def test_trapezoidal_scheme_is_second_order_in_dt():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer_exact = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath",
                                  boundary=["ambient"], scheme="exact")
    layer_trap = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath",
                                 boundary=["ambient"], scheme="trapezoidal")
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)

    def error(dt: float, n: int) -> float:
        c_e, c_t = c0.clone(), c0.clone()
        for _ in range(n):
            c_e = layer_exact.step(c_e, q, source, c_out, dt)
            c_t = layer_trap.step(c_t, q, source, c_out, dt)
        return (c_t - c_e).abs().item()

    dt0, n0 = 200.0, 5
    e1 = error(dt0, n0)
    e2 = error(dt0 / 2, n0 * 2)
    ratio = e1 / e2
    assert 3.2 < ratio < 4.8  # second order: halving dt quarters the error


def test_implicit_preserves_positivity_where_trapezoidal_goes_negative():
    """A genuinely stiff single-mode decay drives trapezoidal negative while
    backward Euler stays non-negative for any step size.

    The brief's original parameterization (two sealed, equal-capacity zones
    exchanging at a huge shared flow, zero removal, zero boundary coupling) can
    NOT demonstrate this, for any choice of q/dt/capacity: with no boundary or
    removal term, that generator M always has eigenvalues {0, lambda < 0} (the
    "total capacity-weighted mass" functional is exactly conserved, as verified
    by test_two_sealed_zones_conserve_total_amount), and the trapezoidal
    (Crank-Nicolson) amplification factor a(z) = (2 + z) / (2 - z) satisfies
    |a(z)| < 1 for every z = dt * lambda < 0. Starting from x0 = [10, 0], the
    conserved and decaying eigenmodes both have amplitude 5, so the decaying
    component is bounded strictly inside (-5, 5) and every component of x_trap
    stays in the open interval (0, 10) no matter how large dt is made; this was
    confirmed numerically for dt spanning 1 to 1e9 (trapezoidal output tends to
    the boundary from above but never crosses it). A single sealed node with a
    first-order removal rate is a genuine scalar decay problem instead (no
    conservation law forces a compensating positive mode), so it reproduces the
    textbook L-stability contrast: backward Euler's amplification factor
    1 / (1 - z) stays in (0, 1] for any z <= 0, while trapezoidal's a(z) crosses
    zero and goes negative once |z| = dt * rate exceeds 2 -- here dt * rate =
    100, far past that threshold, not a knife-edge tuning.
    """
    net = sealed_zone_with_flow_kind()
    cap = torch.tensor([1.0], dtype=torch.float64)
    rate = 100.0
    dt = 1.0  # dt * rate = 100 >> 2: deep in the regime where trapezoidal overshoots
    q = torch.zeros(2, dtype=torch.float64)  # no advection: an isolated decay mode
    x0 = torch.tensor([10.0], dtype=torch.float64)
    src = torch.zeros(1, dtype=torch.float64)
    xb = torch.tensor([0.0], dtype=torch.float64)

    imp = TransportLayer(net, "x", capacity=cap, flow_kind="airpath", boundary=["ambient"],
                          scheme="implicit", removal=torch.tensor([rate], dtype=torch.float64))
    trap = TransportLayer(net, "x", capacity=cap, flow_kind="airpath", boundary=["ambient"],
                           scheme="trapezoidal", removal=torch.tensor([rate], dtype=torch.float64))

    x_imp = imp.step(x0, q, src, xb, dt)
    x_trap = trap.step(x0, q, src, xb, dt)

    assert torch.all(x_imp >= 0.0)
    assert torch.any(x_trap < 0.0)


def test_heat_layer_reaches_algebraic_energy_balance():
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("wall_ambient")
    net.add_node("Z")
    net.add_edge("ambient", "Z", kind="airpath")       # supply
    net.add_edge("Z", "ambient", kind="airpath")        # exhaust
    net.add_edge("Z", "wall_ambient", kind="conduction")

    rho_cp = 1200.0  # J / (m3 K), rho * cp for air
    Q = 0.5          # m3 / s
    U = 8.0          # W / K, conduction to the wall boundary
    V = 50.0         # m3

    layer = TransportLayer(
        net, "heat", capacity=torch.tensor([rho_cp * V]), flow_kind="airpath",
        boundary=["ambient", "wall_ambient"], carrier=rho_cp,
        conduction_kind="conduction", conductance=torch.tensor([U], dtype=torch.float64),
    )
    q = torch.tensor([Q, Q], dtype=torch.float64)
    x_b = torch.tensor([20.0, 5.0], dtype=torch.float64)  # [T_ambient, T_wall]
    T_ss = layer.steady(q, torch.zeros(1, dtype=torch.float64), x_b)
    expected = (rho_cp * Q * 20.0 + U * 5.0) / (rho_cp * Q + U)
    torch.testing.assert_close(
        T_ss, torch.tensor([expected], dtype=torch.float64), rtol=1e-6, atol=1e-6
    )


def test_wrong_length_conductance_raises_value_error_naming_argument():
    net = Network(dtype=torch.float64)
    net.add_node("Tb")
    net.add_node("T1")
    net.add_edge("T1", "Tb", kind="conduction")
    with pytest.raises(ValueError, match="conductance"):
        TransportLayer(
            net, "heat", capacity=torch.tensor([1000.0]), flow_kind="conduction",
            boundary=["Tb"], conduction_kind="conduction",
            conductance=torch.tensor([1.0, 2.0], dtype=torch.float64),
        )


def test_exact_scheme_rejects_on_failure_return():
    """scheme='exact' has no linear solve to report a SolveResult for; on_failure='return'
    is validated but does nothing there, so it must raise ValueError rather than silently
    behave like 'raise'.
    """
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"],
        scheme="exact",
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c = torch.tensor([100.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="on_failure='return'"):
        layer.step(
            c, q, torch.zeros(1, dtype=torch.float64), torch.tensor([420.0]), 300.0,
            on_failure="return",
        )


def test_exact_scheme_rejects_on_failure_return_before_doing_any_work():
    """An ARGUMENT-VALIDITY error must precede the work, not follow it. The refusal used to
    sit inside the `scheme == "exact"` branch, after `_to_stacked` had already validated and
    reshaped `x`, so a caller who passed both a bad shape and the unusable `on_failure` was
    told about the shape (final review M9). `on_failure` is wrong whatever the shapes are.
    """
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"],
        scheme="exact",
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    bad_shape_c = torch.zeros(7, dtype=torch.float64)  # n_i is 1, so this is invalid too
    with pytest.raises(ValueError, match="on_failure='return'"):
        layer.step(
            bad_shape_c, q, torch.zeros(1, dtype=torch.float64), torch.tensor([420.0]), 300.0,
            on_failure="return",
        )


def test_transpose_view_is_the_shared_transpose_operator():
    """`_TransposeView` is a thin alias for `solvers.implicit.TransposeOperator`, not a
    separately-maintained duplicate -- see the consolidation note in `layers/transport.py`.
    """
    assert _TransposeView is TransposeOperator
