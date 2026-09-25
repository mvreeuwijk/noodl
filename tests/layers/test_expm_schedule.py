"""The exponential action's Taylor work is chosen from a norm bound, independent of the
state, so autograd through the polynomial is the polynomial's own derivative."""

from __future__ import annotations

import math

import pytest
import torch

from noodl.layers.transport import (
    ExpmResult,
    TransportLayer,
    _expm_action,
    _forced_remainder,
    _shifted_remainder,
    _taylor_remainder,
    _taylor_schedule,
    _theta_max,
    _theta_table,
)
from noodl.topology import Network

F64 = torch.float64


def _t(values, **kwargs):
    return torch.tensor(values, dtype=F64, **kwargs)


def test_remainder_is_the_tail_of_the_exponential_series():
    theta, m = 1.5, 6
    tail = math.exp(theta) - sum(theta**k / math.factorial(k) for k in range(m + 1))
    assert _taylor_remainder(theta, m) == pytest.approx(tail, rel=1e-10)
    assert _taylor_remainder(0.0, 3) == 0.0
    assert _forced_remainder(0.0, 1) == 0.5 and _forced_remainder(0.0, 2) == 0.0
    r_x = sum(theta**k / math.factorial(k) for k in range(m + 1, 60))
    r_phi = sum(theta ** (k - 1) / math.factorial(k) for k in range(m + 1, 60))
    d_x = sum(k * theta ** (k - 1) / math.factorial(k) for k in range(m + 1, 60))
    d_phi = sum((k - 1) * theta ** (k - 2) / math.factorial(k) for k in range(m + 1, 60))
    assert _forced_remainder(theta, m) == pytest.approx(max(r_x, r_phi, d_x, d_phi), rel=1e-10)
    # The shifted (homogeneous) path's own derivative tail D_x = sum_{k>m} k
    # theta^(k-1)/k! equals the VALUE tail one Taylor degree down, `_taylor_remainder(theta,
    # m - 1)`, exactly (re-index k' = k - 1). `_shifted_remainder` is `max(R_x, D_x)`, which
    # is `_taylor_remainder(theta, m - 1)` because the tail is monotone decreasing in degree.
    d_x_hand = sum(k * theta ** (k - 1) / math.factorial(k) for k in range(m + 1, 60))
    assert d_x_hand == pytest.approx(_taylor_remainder(theta, m - 1), rel=1e-10)
    assert _shifted_remainder(theta, m) == pytest.approx(
        max(_taylor_remainder(theta, m), d_x_hand), rel=1e-10
    )
    assert _shifted_remainder(theta, m) == pytest.approx(_taylor_remainder(theta, m - 1), rel=1e-10)
    # A zero-norm shifted operator must stay admissible at one term: every term of D_x
    # carries a positive power of theta, so it (and hence the max) vanishes at theta = 0.
    assert _shifted_remainder(0.0, 1) == 0.0


def test_theta_max_agrees_with_al_mohy_higham_to_the_leading_digit():
    """Table 3.1 of Al-Mohy & Higham (2011) gives theta_55 = 9.87 at unit-roundoff tolerance;
    a forward bound at 2^-53 must land in the same range (the two bounds differ in constants,
    not in kind). Homogeneous (unforced) table -- the affine step's own table is smaller."""
    assert 8.0 < _theta_max(2.0**-53, 55, forcing=False) < 14.0


@pytest.mark.parametrize("norm", [0.0, 1e-3, 1.0, 12.0, 250.0, 25_000.0])
@pytest.mark.parametrize("forcing", [False, True])
def test_schedule_meets_the_tolerance_and_never_wastes_a_substep(norm, forcing):
    s, m = _taylor_schedule(norm, 1e-12, forcing=forcing)
    assert s >= 1 and 1 <= m <= 55
    if forcing:
        assert m >= 2
    bound = _forced_remainder if forcing else _shifted_remainder
    assert bound(norm / s, m) <= 1e-12
    if s > 1:
        # one fewer substep would need more than m_max terms
        assert bound(norm / (s - 1), 55) > 1e-12 or s * m <= (s - 1) * 55


@pytest.mark.parametrize("forcing", [False, True])
def test_schedule_clamps_m_at_the_table_boundary(forcing):
    """norm = the smallest float above table[-1] * 17 makes s_min land exactly on s = 17 (via
    `math.ceil(norm / table[-1])`) while the SEPARATE division `norm / 17` rounds to a hair
    above the largest tabulated theta_{m_max} -- bisect_left then returns len(table) ==
    m_max, which must clamp to m_max rather than overflow to m_max + 1. Confirmed to
    reproduce end to end through `_taylor_schedule` before the clamp was added (result
    (17, 56)) on the homogeneous table (`theta_55 = 13.1924`); the forced table's own
    `theta_55 = 12.8508` is a different boundary value, so it needs the same clamp checked
    against its own table rather than assuming the homogeneous case covers it. Both pin it
    at (17, <= 55)."""
    tol, m_max = 1e-12, 55
    table = _theta_table(tol, m_max, forcing=forcing)
    norm = math.nextafter(table[-1] * 17, math.inf)
    s, m = _taylor_schedule(norm, tol, m_max, forcing=forcing)
    assert m <= m_max


def _chain(n_nodes=4, *, removal=None, capacity=None):
    net = Network(dtype=F64)
    net.add_node("ambient")
    names = [f"z{i}" for i in range(n_nodes)]
    for nm in names:
        net.add_node(nm)
    net.add_edge("ambient", names[0], kind="flow")
    for a, b in zip(names, names[1:]):  # noqa: B905 -- intentional pairwise (unequal-length) zip
        net.add_edge(a, b, kind="flow")
    net.add_edge(names[-1], "ambient", kind="flow")
    cap = torch.ones(n_nodes, dtype=F64) if capacity is None else capacity
    return TransportLayer(net, "c", capacity=cap, flow_kind="flow", boundary=["ambient"],
                          removal=removal)


def _dense_reference(op, x, b0, dt):
    """[x_next, integral of x over (0, dt)] from a 3-block Van Loan exponential, differentiable
    through torch.linalg.matrix_exp."""
    M = op.assemble()
    m = M.shape[-1]
    eye = torch.eye(m, dtype=F64)
    Z = torch.zeros(*M.shape[:-2], 3 * m, 3 * m, dtype=F64)
    Z[..., :m, :m] = M * dt
    Z[..., :m, m:2 * m] = eye * dt
    Z[..., m:2 * m, 2 * m:] = eye * dt
    E = torch.linalg.matrix_exp(Z)
    Ed, Phi, Psi = E[..., :m, :m], E[..., :m, m:2 * m], E[..., :m, 2 * m:]
    x_next = (Ed @ x.unsqueeze(-1)).squeeze(-1) + (Phi @ b0.unsqueeze(-1)).squeeze(-1)
    # integral of x(tau) = e^{tau M} x + int_0^tau e^{s M} b0 ds  over tau in (0, dt):
    integral = (Phi @ x.unsqueeze(-1)).squeeze(-1) + (Psi @ b0.unsqueeze(-1)).squeeze(-1)
    return x_next, integral


def test_action_and_integral_match_the_dense_reference_and_so_do_their_gradients():
    layer = _chain()
    q = _t([0.3, 0.3, -0.2, 0.3, 0.3], requires_grad=True)
    cap = _t([1.0, 2.0, 0.5, 1.5], requires_grad=True)
    op = layer._advection_operator(q, cap)
    x = _t([1.0, 0.0, 2.0, 0.5], requires_grad=True)
    b0 = _t([0.1, 0.0, 0.0, -0.05], requires_grad=True)
    out = _expm_action(op, x, b0, 7.0, integrate=True)
    assert isinstance(out, ExpmResult)
    x_ref, i_ref = _dense_reference(op, x, b0, 7.0)
    torch.testing.assert_close(out.x, x_ref, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(out.integral, i_ref, rtol=1e-10, atol=1e-12)
    w = _t([0.3, -1.2, 0.7, 2.0])
    # `op.flow` (carrier * q) is one shared non-leaf node under both `out` and `x_ref`/
    # `i_ref`; the first `grad()` call needs `retain_graph=True` or the second one hits
    # PyTorch's "backward through the graph a second time" on that shared node -- true of
    # any implementation, not specific to `_expm_action`.
    got = torch.autograd.grad(
        (out.x * w).sum() + out.integral.sum(), (x, b0, q, cap), retain_graph=True
    )
    ref = torch.autograd.grad((x_ref * w).sum() + i_ref.sum(), (x, b0, q, cap))
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g, r, rtol=1e-8, atol=1e-10)


def test_a_nonzero_nilpotent_operator_is_advanced_exactly_with_its_derivative():
    """Species 0 -> species 1 at rate k, no loss, no flow: M is nilpotent (M^2 = 0), its
    spectral radius is 0 but ||M||_1 = k. exp(dt M) = I + dt M exactly."""
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("zone", "ambient", kind="flow")
    k = torch.tensor(0.8, dtype=F64, requires_grad=True)
    kinetics = torch.stack([torch.stack([0 * k, 0 * k]), torch.stack([k, 0 * k])])
    layer = TransportLayer(net, "c", capacity=_t([1.0]), flow_kind="flow", boundary=["ambient"],
                           n_species=2, kinetics=kinetics)
    x0 = _t([[1.0, 0.0]])
    y = layer.step(x0, _t([0.0]), torch.zeros(2, 2, dtype=F64), torch.zeros(1, 2, dtype=F64), 3.0)
    torch.testing.assert_close(y, _t([[1.0, 2.4]]), rtol=1e-12, atol=1e-12)
    (dk,) = torch.autograd.grad(y[..., 1].sum(), (k,))
    assert dk.item() == pytest.approx(3.0, rel=1e-10)


def test_a_mixed_stiffness_batch_shares_one_schedule_and_matches_the_reference():
    layer = _chain()
    q = torch.ones(2, 5, dtype=F64) * 0.5
    cap = torch.stack([torch.ones(4, dtype=F64), torch.full((4,), 1e-3, dtype=F64)])
    op = layer._advection_operator(q, cap)
    x = torch.rand(2, 4, generator=torch.Generator().manual_seed(1), dtype=F64)
    b0 = torch.zeros(2, 4, dtype=F64)
    out = _expm_action(op, x, b0, 2.0)
    x_ref, _ = _dense_reference(op, x, b0, 2.0)
    torch.testing.assert_close(out.x, x_ref, rtol=1e-9, atol=1e-12)
    assert out.substeps >= 1 and out.matvecs == out.substeps * out.terms


def test_pure_decay_with_no_forcing_costs_one_term_after_the_diagonal_shift():
    """dx/dt = -500 x, x(0) = 10, dt = 50: without the shift this case takes 184,459
    matvecs. With b0 == 0 the mean-diagonal shift makes M - mu I vanish."""
    layer = _chain(1, removal=_t([[500.0]]))
    op = layer._advection_operator(_t([0.0, 0.0]))
    out = _expm_action(op, _t([10.0]), _t([0.0]), 50.0)
    assert out.x.item() == pytest.approx(10.0 * math.exp(-25_000.0), abs=1e-300)
    assert out.matvecs <= 2


def test_shifted_branch_gradient_matches_the_dense_reference():
    """The b0 == 0 shift changes the recurrence itself (M -> M - mu I, rescaled by e^{h mu}),
    so it needs its own gradient check against the dense reference: the only other gradient
    test uses integrate=True (shift disabled by construction), the nilpotent test reaches the
    shift with mu == 0 (a no-op shift numerically), and the mixed-stiffness batch test checks
    only the forward value. Here mu is strongly nonzero (removal=500) and q/capacity require
    grad; b0 is detached and all-zero, which the default (shift=None) inference recognises as
    structurally eligible."""
    layer = _chain(4, removal=torch.full((4, 1), 500.0, dtype=F64))
    q = _t([0.3, 0.3, -0.2, 0.3, 0.3], requires_grad=True)
    cap = _t([1.0, 2.0, 0.5, 1.5], requires_grad=True)
    op = layer._advection_operator(q, cap)
    x = _t([1.0, 0.0, 2.0, 0.5])
    b0 = torch.zeros(4, dtype=F64).detach()
    out = _expm_action(op, x, b0, 2.0)
    x_ref, _ = _dense_reference(op, x, b0, 2.0)
    torch.testing.assert_close(out.x, x_ref, rtol=1e-9, atol=1e-12)
    w = _t([0.3, -1.2, 0.7, 2.0])
    # `op.flow` (carrier * q) is again shared between `out` and `x_ref`; see the identical
    # note on test_action_and_integral_match_the_dense_reference_and_so_do_their_gradients.
    dq, dcap = torch.autograd.grad((out.x * w).sum(), (q, cap), retain_graph=True)
    dq_ref, dcap_ref = torch.autograd.grad((x_ref * w).sum(), (q, cap))
    torch.testing.assert_close(dq, dq_ref, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(dcap, dcap_ref, rtol=1e-8, atol=1e-10)


def test_the_work_budget_is_refused_by_name():
    layer = _chain(1, removal=_t([[500.0]]))
    op = layer._advection_operator(_t([0.0, 0.0]))
    with pytest.raises(RuntimeError, match=r"matvecs.*scheme='implicit'"):
        _expm_action(op, _t([10.0]), _t([1.0]), 50.0, max_matvecs=1000, where="budget test")


def _one_node_forced(r_val: float, *, requires_grad: bool = True):
    """dx/dt = -r x + 1 on one interior node, x(0) = 0, dt = 1: the forced case.
    Nodes: ambient (0), zone (1); the single flow edge carries q = 0 so nothing advects; the
    unit forcing is `sources` on the FULL node axis; removal r is the differentiated
    coefficient."""
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("zone", "ambient", kind="flow")
    r = torch.tensor([r_val], dtype=F64, requires_grad=requires_grad)
    layer = TransportLayer(
        net, "c", capacity=_t([1.0]), flow_kind="flow", boundary=["ambient"], removal=r,
        scheme="exact",
    )
    x = layer.step(_t([0.0]), _t([0.0]), _t([0.0, 1.0]), _t([0.0]), 1.0)
    return x, r


def _closed_form_forced(r: float) -> tuple[float, float]:
    """x(1) = (1 - e^{-r}) / r and dx/dr = (e^{-r}(1 + r) - 1) / r^2, with their r -> 0
    limits 1 and -1/2. Both are entire functions of r, evaluated here by their own (equally
    exact) Taylor series -- x(1) = sum_k (-1)^k r^k / (k+1)! and dx/dr = sum_k (-1)^(k+1)
    (k+1) r^k / (k+2)! -- rather than the closed forms above for SMALL r: the naive closed
    forms subtract two nearly-equal O(1) floats to recover an O(r) (value) or O(r^2)
    (derivative) result and lose essentially all precision at r = 1e-6 (verified against
    mpmath at 50 digits: the naive derivative formula is off by 4.5e-5 there, while this
    series matches to 1e-16).

    The series form flips that trade-off for LARGE r: its alternating terms r^k / (k+1)!
    grow before they shrink, and by r = 20 cancellation among them has eaten 4.4e+3 relative
    of the 40-term truncation (nothing in this module tests above r = 1.0 today, so this was
    an untested trap for whoever extends the parametrization). The naive closed form has the
    opposite profile: `1 - e^{-r}` and `e^{-r}(1+r) - 1` both compute exactly (no
    cancellation once `e^{-r}` is small), so it is accurate to 1e-16 for every r above the
    small-r regime where IT loses precision. `r = 1.0` is comfortably inside the series'
    accurate range and short of where the naive form is needed, so the crossover point
    itself is untested by the parametrization; it is picked here as a round number known to
    lie strictly between the two forms' respective loss regions (series good through
    r ~ O(1), naive good from r ~ O(1) up)."""
    if r > 1.0:
        e = math.exp(-r)
        return (1.0 - e) / r, (e * (1.0 + r) - 1.0) / (r * r)
    n_terms = 40
    value = sum((-1) ** k * r**k / math.factorial(k + 1) for k in range(n_terms))
    derivative = sum(
        (-1) ** (k + 1) * (k + 1) * r**k / math.factorial(k + 2) for k in range(n_terms)
    )
    return value, derivative


@pytest.mark.parametrize("r", [0.0, 1e-6, 1e-3, 1e-1, 1.0])
def test_forced_step_value_and_coefficient_derivative_match_the_closed_form_near_a_zero_operator(r):
    x, r_t = _one_node_forced(r)
    value, derivative = _closed_form_forced(r)
    assert x.item() == pytest.approx(value, rel=1e-10, abs=1e-12)
    (dr,) = torch.autograd.grad(x.sum(), (r_t,))
    assert dr.item() == pytest.approx(derivative, rel=1e-8, abs=1e-9)


@pytest.mark.parametrize("dt", [1e-9, 1e-6, 1e-3, 1e-1, 1.0, 7.0])
def test_forced_action_and_its_coefficient_gradients_match_the_dense_reference_at_every_norm(dt):
    layer = _chain()
    q = _t([0.3, 0.3, -0.2, 0.3, 0.3], requires_grad=True)
    cap = _t([1.0, 2.0, 0.5, 1.5], requires_grad=True)
    op = layer._advection_operator(q, cap)
    x = _t([1.0, 0.0, 2.0, 0.5])
    b0 = _t([0.1, 0.0, 0.0, -0.05], requires_grad=True)
    out = _expm_action(op, x, b0, dt, shift=False)
    x_ref, _ = _dense_reference(op, x, b0, dt)
    torch.testing.assert_close(out.x, x_ref, rtol=1e-10, atol=1e-14)
    w = _t([0.3, -1.2, 0.7, 2.0])
    got = torch.autograd.grad((out.x * w).sum(), (q, cap, b0), retain_graph=True)
    ref = torch.autograd.grad((x_ref * w).sum(), (q, cap, b0))
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g, r, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("spread", [1e-6, 1e-4, 1e-2, 1.0])
def test_shifted_action_gradients_match_the_dense_reference_near_a_zero_operator(spread):
    """The shifted case, mirroring the forced test above on the SHIFTED path: four nodes
    with `removal = [100, 100, 100, 100 + spread]`, `q = 0` (pure decay, no advective
    coupling), a detached all-zero `b0` (so `shift=None` infers `shift=True`), `dt = 1`,
    against `torch.linalg.matrix_exp` directly (there is no forcing to fold into a 3-block
    Van Loan reference; `b0 == 0` means `x_next = expm(dt * M) @ x0` exactly). The shift
    subtracts the mean removal rate from the diagonal, so the POST-shift norm scales with
    `spread`, not with the removal rate itself (~100) -- this is what makes the schedule pick
    as few as one or two Taylor terms for `spread = 1e-6` and only grows to ~14 terms by
    `spread = 1.0`. A schedule that bounded only the state polynomial's VALUE tail
    (`_taylor_remainder`) on this path, leaving its A-derivative tail (`D_x`) unbounded,
    would take one term fewer, and `dx/d(removal)` would degrade badly at the smallest
    spreads even though the VALUE stays accurate to ~1e-13 throughout."""
    removal = torch.tensor(
        [[100.0], [100.0], [100.0], [100.0 + spread]], dtype=F64, requires_grad=True
    )
    layer = _chain(4, removal=removal)
    q = torch.zeros(5, dtype=F64)
    op = layer._advection_operator(q, layer.capacity)
    x = _t([1.0, 0.5, -0.3, 2.0])
    b0 = torch.zeros(4, dtype=F64).detach()
    out = _expm_action(op, x, b0, 1.0)
    M = op.assemble()
    x_ref = (torch.linalg.matrix_exp(M) @ x.unsqueeze(-1)).squeeze(-1)
    # `removal ~ 100` with `dt = 1` makes both `x_ref` and its gradient tiny (~e^-100 ~
    # 3.7e-44), so `atol` must be near-zero rather than a default-sized absolute floor --
    # otherwise it swamps any relative error the schedule's derivative bound is supposed to
    # control, and the test would pass unconditionally regardless of (s, m).
    torch.testing.assert_close(out.x, x_ref, rtol=1e-9, atol=0.0)
    w = _t([0.3, -1.2, 0.7, 2.0])
    (dg,) = torch.autograd.grad((out.x * w).sum(), (removal,), retain_graph=True)
    (dg_ref,) = torch.autograd.grad((x_ref * w).sum(), (removal,))
    torch.testing.assert_close(dg, dg_ref, rtol=1e-8, atol=0.0)
