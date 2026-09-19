"""Tests for Conductance: q = g dp (linear conduction / SIRANE-style exchange)."""

import torch

from noodl.elements.conductance import Conductance


def test_flow_is_linear_and_defaults_to_conduction_kind():
    el = Conductance(g=3.0)
    dp = torch.tensor([-2.0, 0.0, 5.0])
    torch.testing.assert_close(el.flow(dp), 3.0 * dp)
    assert el.kind == "conduction"


def test_dflow_is_constant_g_with_dps_broadcast_shape():
    el = Conductance(g=torch.tensor([1.0, 2.0]))
    dp = torch.zeros(4, 2)
    dq = el.dflow(dp)
    assert dq.shape == (4, 2)
    torch.testing.assert_close(dq, torch.tensor([1.0, 2.0]).expand(4, 2))


def test_linear_init_is_zero_and_g():
    c, k = Conductance(g=4.0).linear_init()
    torch.testing.assert_close(c, torch.tensor(0.0))
    torch.testing.assert_close(k, torch.tensor(4.0))


def test_broadcasts_batched_g_against_batched_dp():
    g = torch.tensor([1.0, 2.0, 3.0])
    el = Conductance(g=g)
    dp = torch.linspace(-2.0, 2.0, 5).unsqueeze(-1).expand(5, 3)
    torch.testing.assert_close(el.flow(dp), g * dp)


def test_learnable_conductance_exposes_one_parameter():
    params = list(Conductance(g=4.0, learnable=True).parameters())
    assert len(params) == 1
    assert params[0].requires_grad


def test_gradcheck_flow_wrt_dp_and_learnable_g():
    g = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    dp = torch.tensor([-3.0, 0.0, 3.0], dtype=torch.float64, requires_grad=True)

    def f(g_, dp_):
        return Conductance(g=g_).flow(dp_)

    assert torch.autograd.gradcheck(f, (g, dp), eps=1e-6, atol=1e-6)
