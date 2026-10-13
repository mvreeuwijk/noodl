"""Tests for FixedFlow: q = q0 regardless of dp."""

import pytest
import torch

from noodl.elements import FixedVolumeFlow, PrescribedFlow
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


def test_a_fixed_volume_flow_is_delivered_at_the_upstream_density():
    """ContamX's duct cvf fan: 0.04 m3/s -> 0.0481639 kg/s at 293.15 K, 0.0450878 at
    313.15 K, the density of the junction it draws from."""
    rho = torch.tensor([101325 / (287.055 * 313.15), 101325 / (287.055 * 293.15)],
                       dtype=torch.float64)
    fan = FixedVolumeFlow(torch.tensor([0.04, -0.04], dtype=torch.float64),
                          src=[0, 0], tgt=[1, 1])
    q = fan.flow(torch.zeros(2, dtype=torch.float64), {"rho": rho})
    torch.testing.assert_close(q, torch.tensor([0.04 * rho[0], -0.04 * rho[1]],
                                               dtype=torch.float64))
    assert fan.dp_independent
    assert fan.dflow(torch.zeros(2, dtype=torch.float64), {"rho": rho}).eq(0).all()


def test_a_fixed_volume_flow_matches_contamx_and_works_batched():
    """The cvf numbers ContamX printed (7 digits) at 293.15 and 313.15 K, one batch row each."""
    T = torch.tensor([[293.15, 293.15], [313.15, 313.15]], dtype=torch.float64)
    fan = FixedVolumeFlow(torch.tensor([0.04], dtype=torch.float64), src=[0], tgt=[1])
    q = fan.flow(torch.zeros(2, 1, dtype=torch.float64), {"rho": 101325 / (287.055 * T)})
    assert q.shape == (2, 1)
    torch.testing.assert_close(q[:, 0], torch.tensor([0.0481639, 0.0450878],
                                                     dtype=torch.float64), rtol=1e-6, atol=0)
    c, k = fan.linear_init({"rho": 101325 / (287.055 * T)})
    torch.testing.assert_close(c, q)
    assert k.eq(0).all()


def test_a_prescribed_flow_reads_its_flows_from_the_drivers():
    el = PrescribedFlow("prescribed_flow.ahs", kind="ahs")
    d = {"prescribed_flow.ahs": torch.tensor([[0.1, 0.05], [0.0, 0.02]], dtype=torch.float64)}
    q = el.flow(torch.zeros(2, 2, dtype=torch.float64), d)
    torch.testing.assert_close(q, d["prescribed_flow.ahs"])
    assert el.dp_independent and el.dflow(torch.zeros(2, dtype=torch.float64), d).eq(0).all()
    with pytest.raises(KeyError, match="prescribed_flow.ahs"):
        el.flow(torch.zeros(2, dtype=torch.float64), {})


def test_a_prescribed_flow_starts_in_the_layer_dtype():
    el = PrescribedFlow("prescribed_flow.ahs", kind="ahs")
    q, k = el.linear_init({"prescribed_flow.ahs": torch.tensor([0.1, 0.05])})
    assert q.dtype == k.dtype == torch.float64
    el32 = PrescribedFlow("prescribed_flow.ahs", kind="ahs", dtype=torch.float32)
    q32, _ = el32.linear_init({"prescribed_flow.ahs": torch.tensor([0.1], dtype=torch.float64)})
    assert q32.dtype == torch.float32

