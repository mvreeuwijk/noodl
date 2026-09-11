"""Tests for FanCurve: dp = -P(q), P a cubic pressure-flow curve inverted by solve_monotone."""

import torch

from tellegen.elements.base import Element
from tellegen.elements.fan import FanCurve


def _p(coeffs, q):
    c0, c1, c2, c3 = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2], coeffs[..., 3]
    return c0 + c1 * q + c2 * q**2 + c3 * q**3


def test_flow_recovers_q_from_p_of_q_for_sampled_q_and_is_monotone_increasing():
    coeffs = torch.tensor([500.0, -50.0, -20.0, -2.0])  # P(0) = 500, decreasing on [0, q_max]
    q_max = torch.tensor(6.0)
    fan = FanCurve(coeffs=coeffs, q_max=q_max)
    q_true = torch.tensor([0.5, 1.0, 2.5, 4.0, 5.5])
    dp = -_p(coeffs, q_true)
    torch.testing.assert_close(fan.flow(dp), q_true, atol=1e-6, rtol=1e-6)

    # q(dp) = P^-1(-dp): a composition of two decreasing maps (negation, then P's decreasing
    # inverse) is increasing, so flow is monotone INCREASING in dp under this dp = -P(q) sign
    # convention, not decreasing (corrected from the brief, which asserted <= 1e-9 here; see
    # task-5-report.md for the derivation and the matching sign fix to the two "stalled"
    # clip-boundary test values below and in test_dflow_matches_... below).
    dp_sweep = torch.linspace(-500.0, 100.0, 61)
    q = fan.flow(dp_sweep)
    assert torch.all(q[1:] - q[:-1] >= -1e-9)


def test_flow_clips_to_zero_above_shutoff_and_to_q_max_below_top_of_curve_pressure():
    coeffs = torch.tensor([500.0, -50.0, -20.0, -2.0])
    q_max = torch.tensor(6.0)
    fan = FanCurve(coeffs=coeffs, q_max=q_max)

    shutoff = torch.tensor([-501.0, -600.0, -1000.0])  # -dp > P(0) = 500
    torch.testing.assert_close(fan.flow(shutoff), torch.zeros_like(shutoff))

    p_max_flow = _p(coeffs, q_max)
    # -dp < P(q_max) needs dp > -P(q_max) = -p_max_flow, i.e. -p_max_flow + X for X > 0
    # (corrected from the brief's "- 10.0, - 100.0", which computes dp < -p_max_flow instead
    # and so never actually falls in the stalled region for this curve's coefficients).
    stalled = torch.tensor([-p_max_flow + 10.0, -p_max_flow + 100.0])
    torch.testing.assert_close(fan.flow(stalled), q_max.expand(2))


def test_dflow_matches_autograd_default_in_the_smooth_region_and_is_zero_when_clipped():
    coeffs = torch.tensor([500.0, -50.0, -20.0, -2.0])
    q_max = torch.tensor(6.0)
    fan = FanCurve(coeffs=coeffs, q_max=q_max)

    q_true = torch.tensor([0.5, 2.0, 4.0, 5.5])
    dp = -_p(coeffs, q_true)
    torch.testing.assert_close(fan.dflow(dp), Element.dflow(fan, dp), atol=1e-4, rtol=1e-4)

    p_max_flow = _p(coeffs, q_max)
    # Same stalled-side sign correction as above: need dp > -p_max_flow.
    clipped = torch.tensor([-600.0, (-p_max_flow + 50.0).item()])
    torch.testing.assert_close(fan.dflow(clipped), torch.zeros(2))


def test_linear_init_matches_flow_and_dflow_at_zero():
    coeffs = torch.tensor([500.0, -50.0, -20.0, -2.0])
    fan = FanCurve(coeffs=coeffs, q_max=torch.tensor(6.0))
    c, k = fan.linear_init()
    torch.testing.assert_close(c, fan.flow(torch.zeros(())))
    torch.testing.assert_close(k, fan.dflow(torch.zeros(())))


def test_batched_coefficients_give_one_curve_per_batch_column():
    # (b_kind=2, 4)
    coeffs = torch.tensor([[500.0, -50.0, -20.0, -2.0], [300.0, -40.0, -10.0, -1.0]])
    q_max = torch.tensor([6.0, 5.0])
    fan = FanCurve(coeffs=coeffs, q_max=q_max)
    q_true = torch.tensor([[1.0, 1.0], [3.0, 2.0], [5.0, 4.0]])  # (batch=3, b_kind=2)
    dp = -_p(coeffs, q_true)
    torch.testing.assert_close(fan.flow(dp), q_true, atol=1e-6, rtol=1e-6)


def test_gradcheck_flow_wrt_learnable_coeffs():
    coeffs = torch.tensor([500.0, -50.0, -20.0, -2.0], dtype=torch.float64, requires_grad=True)
    q_max = torch.tensor(6.0, dtype=torch.float64)
    dp = torch.tensor([-100.0, -250.0, -400.0], dtype=torch.float64)

    def f(coeffs_):
        return FanCurve(coeffs=coeffs_, q_max=q_max).flow(dp)

    assert torch.autograd.gradcheck(f, (coeffs,), eps=1e-6, atol=1e-6)
