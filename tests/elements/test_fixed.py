"""Tests for FixedFlow: q = q0 regardless of dp."""

import torch

from noodl.elements.fixed import FixedFlow


def test_flow_ignores_dp_and_broadcasts_q0_to_dps_shape():
    el = FixedFlow(q0=torch.tensor([1.0, 2.0, 3.0]))
    dp = torch.zeros(4, 3)
    q = el.flow(dp)
    assert q.shape == (4, 3)
    torch.testing.assert_close(q, torch.tensor([1.0, 2.0, 3.0]).expand(4, 3))


def test_dflow_is_zero_with_the_broadcast_shape_of_flow():
    el = FixedFlow(q0=torch.tensor([1.0, 2.0]))
    dp = torch.zeros(3, 2)
    dq = el.dflow(dp)
    assert dq.shape == (3, 2)
    torch.testing.assert_close(dq, torch.zeros(3, 2))


def test_linear_init_is_q0_and_zero():
    c, k = FixedFlow(q0=2.5).linear_init()
    torch.testing.assert_close(c, torch.tensor(2.5))
    torch.testing.assert_close(k, torch.tensor(0.0))


def test_learnable_fixed_flow_exposes_one_parameter():
    params = list(FixedFlow(q0=2.5, learnable=True).parameters())
    assert len(params) == 1
    assert params[0].requires_grad


def test_gradcheck_flow_wrt_learnable_q0():
    q0 = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    dp = torch.zeros(3, 2, dtype=torch.float64)

    def f(q0_):
        return FixedFlow(q0=q0_).flow(dp)

    assert torch.autograd.gradcheck(f, (q0,), eps=1e-6, atol=1e-6)
