"""Tests for Quadratic: dp = a q + b |q| q, inverted for q given dp."""

import torch

from tellegen.elements.base import Element
from tellegen.elements.quadratic import Quadratic


def test_flow_inverts_the_quadratic_drag_law():
    el = Quadratic(a=2.0, b=0.5)
    dp = torch.tensor([-10.0, -1.0, -0.01, 0.01, 1.0, 10.0])
    q = el.flow(dp)
    reconstructed = el.a * q + el.b * q.abs() * q
    torch.testing.assert_close(reconstructed, dp, atol=1e-6, rtol=1e-6)


def test_flow_is_odd_monotone_and_zero_at_zero():
    el = Quadratic(a=2.0, b=0.5)
    dp = torch.linspace(-5.0, 5.0, 101)
    torch.testing.assert_close(el.flow(-dp), -el.flow(dp))
    q = el.flow(dp)
    assert torch.all(q[1:] - q[:-1] >= -1e-12)
    torch.testing.assert_close(el.flow(torch.tensor([0.0])), torch.tensor([0.0]))


def test_dflow_matches_closed_form_and_autograd_default_away_from_zero():
    el = Quadratic(a=2.0, b=0.5)
    dp = torch.tensor([-8.0, -3.0, -0.3, 0.3, 3.0, 8.0])
    expected = 1.0 / torch.sqrt(2.0**2 + 4 * 0.5 * dp.abs())
    torch.testing.assert_close(el.dflow(dp), expected)
    torch.testing.assert_close(el.dflow(dp), Element.dflow(el, dp), atol=1e-6, rtol=1e-6)


def test_linear_init_matches_finite_difference_slope_at_zero():
    el = Quadratic(a=2.0, b=0.5)
    h = 1e-6
    slope = (el.flow(torch.tensor([h])) - el.flow(torch.tensor([-h]))) / (2 * h)
    c, k = el.linear_init()
    torch.testing.assert_close(c, torch.tensor(0.0))
    torch.testing.assert_close(k, slope[0], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k, 1.0 / el.a)


def test_broadcasts_batched_parameters_against_batched_dp():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([0.1, 0.2, 0.3])
    el = Quadratic(a=a, b=b)
    dp = torch.linspace(-4.0, 4.0, 5).unsqueeze(-1).expand(5, 3)
    q = el.flow(dp)
    assert q.shape == (5, 3)
    for j in range(3):
        single = Quadratic(a=a[j].item(), b=b[j].item())
        torch.testing.assert_close(q[:, j], single.flow(dp[:, j]))


def test_learnable_quadratic_exposes_parameters():
    params = list(Quadratic(a=2.0, b=0.5, learnable=True).parameters())
    assert len(params) == 2
    assert all(p.requires_grad for p in params)


def test_learnable_quadratic_with_integer_inputs():
    # Integer inputs should be converted to float for learnable parameters
    el = Quadratic(a=2, b=1, learnable=True)
    params = list(el.parameters())
    assert len(params) == 2
    assert all(p.requires_grad for p in params)
    assert all(p.dtype in [torch.float32, torch.float64] for p in params)
    # Should be in state_dict
    state = el.state_dict()
    assert "a" in state
    assert "b" in state


def test_gradcheck_flow_wrt_dp_and_learnable_params_away_from_zero():
    a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    dp = torch.tensor([-6.0, -0.7, 0.7, 6.0], dtype=torch.float64, requires_grad=True)

    def f(a_, b_, dp_):
        return Quadratic(a=a_, b=b_).flow(dp_)

    assert torch.autograd.gradcheck(f, (a, b, dp), eps=1e-6, atol=1e-6)
