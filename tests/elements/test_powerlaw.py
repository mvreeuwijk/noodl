"""Tests for PowerLaw (laminar blend and regularised variants)."""

import math

import torch

from tellegen.elements.base import Element
from tellegen.elements.powerlaw import Orifice, PowerLaw


def test_blend_matches_sharp_power_law_away_from_transition():
    el = PowerLaw(C=2.0, n=0.6, dp_transition=1e-3)
    dp = torch.tensor([1.0, 5.0, -1.0, -5.0])
    expected = 2.0 * torch.sign(dp) * dp.abs() ** 0.6
    torch.testing.assert_close(el.flow(dp), expected)


def test_blend_matches_laminar_law_inside_transition():
    el = PowerLaw(C=2.0, n=0.6, dp_transition=1e-3)
    k = 2.0 * 1e-3 ** (0.6 - 1)
    dp = torch.tensor([-5e-4, 0.0, 5e-4])
    torch.testing.assert_close(el.flow(dp), k * dp)


def test_blend_is_odd_and_monotone_nondecreasing():
    el = PowerLaw(C=1.7, n=0.65, dp_transition=2e-3)
    dp = torch.linspace(-3.0, 3.0, 101)
    torch.testing.assert_close(el.flow(-dp), -el.flow(dp))
    q = el.flow(dp)
    assert torch.all(q[1:] - q[:-1] >= -1e-12)


def test_blend_is_continuous_in_value_at_the_transition():
    """The blend matches in VALUE at the transition; the derivative has a kink there (the
    laminar slope k and the sharp-side slope n*k generally differ), so probing eps away from
    the boundary on each side necessarily differs by O(eps * slope), not merely by rounding.
    With eps = 1e-9 and slopes here of O(15), that is O(1e-8) -- three orders of magnitude
    below the O(1e-2) a genuine value discontinuity would produce, which is what this test
    actually distinguishes.
    """
    el = PowerLaw(C=1.7, n=0.65, dp_transition=2e-3)
    eps = 1e-9
    for sign in (1.0, -1.0):
        left = el.flow(torch.tensor([sign * (2e-3 - eps)]))
        right = el.flow(torch.tensor([sign * (2e-3 + eps)]))
        assert (left - right).abs().item() < 1e-6


def test_blend_dflow_matches_autograd_default_around_transition_and_zero():
    el = PowerLaw(C=1.3, n=0.7, dp_transition=1e-3)
    dp = torch.tensor([-2.0, -1e-3, -5e-4, 0.0, 5e-4, 1e-3, 2.0])
    torch.testing.assert_close(el.dflow(dp), Element.dflow(el, dp), atol=1e-6, rtol=1e-6)


def test_blend_linear_init_matches_finite_difference_slope_at_zero():
    el = PowerLaw(C=1.3, n=0.7, dp_transition=1e-3)
    h = 1e-6
    slope = (el.flow(torch.tensor([h])) - el.flow(torch.tensor([-h]))) / (2 * h)
    c, k = el.linear_init()
    torch.testing.assert_close(c, torch.tensor(0.0))
    torch.testing.assert_close(k, slope[0], atol=1e-3, rtol=1e-3)


def test_regularised_is_odd_and_matches_autograd_dflow():
    el = PowerLaw(C=1.4, n=0.55, regularised=1e-4)
    dp = torch.linspace(-3.0, 3.0, 13)
    torch.testing.assert_close(el.flow(-dp), -el.flow(dp))
    torch.testing.assert_close(el.dflow(dp), Element.dflow(el, dp), atol=1e-6, rtol=1e-6)


def test_regularised_linear_init_matches_finite_difference_slope_at_zero():
    el = PowerLaw(C=1.4, n=0.55, regularised=1e-4)
    h = 1e-7
    slope = (el.flow(torch.tensor([h])) - el.flow(torch.tensor([-h]))) / (2 * h)
    c, k = el.linear_init()
    torch.testing.assert_close(c, torch.tensor(0.0))
    torch.testing.assert_close(k, slope[0], atol=1e-2, rtol=1e-2)


def test_regularised_converges_to_blend_law_as_eps_vanishes_away_from_zero():
    dp = torch.tensor([0.5, 1.0, 2.0, -0.5, -1.0, -2.0])
    blend = PowerLaw(C=1.1, n=0.6, dp_transition=1e-6)
    errors = []
    for eps in (1e-1, 1e-2, 1e-3, 1e-5):
        reg = PowerLaw(C=1.1, n=0.6, regularised=eps)
        errors.append((reg.flow(dp) - blend.flow(dp)).abs().max().item())
    assert errors[-1] < errors[0]  # error shrinks monotonically as eps -> 0
    assert errors[-1] < 1e-3


def test_broadcasts_batched_parameters_against_batched_dp():
    C = torch.tensor([1.0, 2.0, 3.0])
    n = torch.tensor([0.5, 0.6, 0.7])
    el = PowerLaw(C=C, n=n, dp_transition=1e-3)
    dp = torch.linspace(-2.0, 2.0, 5).unsqueeze(-1).expand(5, 3)  # (batch=5, b_kind=3)
    q = el.flow(dp)
    assert q.shape == (5, 3)
    for j in range(3):
        single = PowerLaw(C=C[j].item(), n=n[j].item(), dp_transition=1e-3)
        torch.testing.assert_close(q[:, j], single.flow(dp[:, j]))


def test_single_edge_element_with_trailing_unit_dim_broadcasts_against_b_kind_dp():
    """Binding requirement: a single-edge element's parameters carry trailing shape (..., 1)
    so they broadcast against (..., b_kind) potential differences, as later composition
    tasks (4/7) require when combining many single-edge elements into a per-kind batch."""
    C = torch.tensor([2.0])  # shape (1,): one edge's worth of parameter
    n = torch.tensor([0.6])  # shape (1,)
    el = PowerLaw(C=C, n=n, dp_transition=1e-3)
    assert el.C.shape == (1,)
    assert el.n.shape == (1,)
    dp = torch.linspace(-2.0, 2.0, 5).unsqueeze(-1).expand(5, 4)  # (batch=5, b_kind=4)
    q = el.flow(dp)
    assert q.shape == (5, 4)
    single = PowerLaw(C=2.0, n=0.6, dp_transition=1e-3)
    torch.testing.assert_close(q, single.flow(dp))


def test_learnable_powerlaw_exposes_parameters():
    params = list(PowerLaw(C=1.0, n=0.5, learnable=True).parameters())
    assert len(params) == 2
    assert all(p.requires_grad for p in params)


def test_gradcheck_flow_wrt_dp_blend_and_regularised():
    """dp values are chosen close to +-dp_transition (default 1e-3) but not exactly on it:
    the blend only promises continuity in VALUE there, not in derivative (see
    test_blend_is_continuous_in_value_at_the_transition), so it is a genuine kink and
    numerical (finite-difference) gradcheck cannot agree with either one-sided analytic
    slope exactly at that point -- that is a property of the kink, not a bug. dp = 0.0 is
    included because it sits deep inside the laminar branch (a smooth region) and is the
    exact value the inf * 0 = nan trap (see PowerLaw module docstring) would corrupt.
    """
    blend = PowerLaw(
        C=torch.tensor(1.3, dtype=torch.float64), n=torch.tensor(0.6, dtype=torch.float64)
    )
    dp = torch.tensor(
        [-2.0, -1.1e-3, 5e-4, 0.0, 5e-4, 1.1e-3, 2.0], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(lambda x: blend.flow(x), (dp,), eps=1e-6, atol=1e-6)

    reg = PowerLaw(
        C=torch.tensor(1.3, dtype=torch.float64),
        n=torch.tensor(0.6, dtype=torch.float64),
        regularised=1e-3,
    )
    dp2 = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: reg.flow(x), (dp2,), eps=1e-6, atol=1e-6)


def test_gradcheck_flow_exactly_at_transition_boundaries_and_zero_no_nan():
    """Binding requirement: torch.where evaluates both branches, so the sharp branch's
    |dp|**(n-1) must never see dp == 0 even when unselected, or its backward produces
    inf * 0 = nan. Exercise exactly dp = 0 (deep inside the smooth laminar region, where
    gradcheck can and must agree with the analytic derivative) and dp = +-dp_transition
    itself (a genuine derivative kink -- see test_blend_is_continuous_in_value_at_the_
    transition -- where finite-difference gradcheck cannot be used, but the backward pass
    must still produce a finite, non-nan one-sided gradient rather than leaking the sharp
    branch's inf * 0 through torch.where)."""
    dpt = 1e-3
    el = PowerLaw(
        C=torch.tensor(1.3, dtype=torch.float64),
        n=torch.tensor(0.6, dtype=torch.float64),
        dp_transition=dpt,
    )

    zero = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: el.flow(x), (zero,), eps=1e-6, atol=1e-6)

    dp = torch.tensor([-dpt, 0.0, dpt], dtype=torch.float64, requires_grad=True)
    q = el.flow(dp)
    assert torch.isfinite(q).all()
    (grad,) = torch.autograd.grad(q.sum(), dp)
    assert torch.isfinite(grad).all()
    assert el.dflow(dp).isfinite().all()


def test_orifice_builds_a_powerlaw_with_n_half_and_derived_C():
    """Orifice is stated as a Task 3 deliverable interface but the brief never tests it."""
    Cd, A, rho = 0.6, 0.02, 1.2
    el = Orifice(Cd, A, rho=rho)
    assert isinstance(el, PowerLaw)
    assert el.kind == "airpath"
    torch.testing.assert_close(el.n, torch.tensor(0.5))
    expected_C = Cd * A * math.sqrt(2.0 / rho)
    torch.testing.assert_close(el.C, torch.tensor(expected_C, dtype=torch.float32))

    dp = torch.tensor([4.0, -4.0])
    expected_flow = expected_C * torch.sign(dp) * dp.abs() ** 0.5
    torch.testing.assert_close(el.flow(dp), expected_flow)


def test_orifice_forwards_kind_learnable_and_transition_kwargs():
    el = Orifice(0.6, 0.02, kind="hydraulic", learnable=True, dp_transition=5e-4, regularised=1e-5)
    assert el.kind == "hydraulic"
    assert el.C.requires_grad
    assert el.n.requires_grad
    assert el.dp_transition == 5e-4
    assert el.regularised == 1e-5


def test_gradcheck_flow_wrt_learnable_C():
    n = torch.tensor(0.6, dtype=torch.float64)
    C = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    dp = torch.tensor([-2.0, -0.1, 0.1, 2.0], dtype=torch.float64)

    def f(c):
        return PowerLaw(C=c, n=n, regularised=1e-3).flow(dp)

    assert torch.autograd.gradcheck(f, (C,), eps=1e-6, atol=1e-6)
