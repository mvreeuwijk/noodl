"""Scaled: F = s F_inner with s a per-edge driver."""

from __future__ import annotations

import pytest
import torch

from noodl.elements import FixedFlow, PowerLaw, Scaled
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network

F64 = torch.float64
KEY = "flow_multiplier.leak"


def _law():
    return PowerLaw(torch.tensor([0.02, 0.03], dtype=F64), torch.tensor([0.5, 0.65], dtype=F64),
                    kind="leak")


def test_flow_slope_and_linear_init_are_the_inner_ones_times_the_multiplier():
    inner, dp = _law(), torch.tensor([4.0, -9.0], dtype=F64)
    s = torch.tensor([0.5, 2.0], dtype=F64)
    el = Scaled(inner, KEY)
    d = {KEY: s}
    assert el.kind == "leak"
    torch.testing.assert_close(el.flow(dp, d), s * inner.flow(dp, d), rtol=0, atol=0)
    torch.testing.assert_close(el.dflow(dp, d), s * inner.dflow(dp, d), rtol=1e-15, atol=0)
    c, k = el.linear_init(d)
    c0, k0 = inner.linear_init(d)
    torch.testing.assert_close((c, k), (s * c0, s * k0), rtol=1e-15, atol=0)


def test_a_batched_multiplier_gives_batched_flows():
    el, dp = Scaled(_law(), KEY), torch.tensor([4.0, -9.0], dtype=F64)
    s = torch.tensor([[1.0, 1.0], [0.25, 0.0]], dtype=F64)
    q = el.flow(dp, {KEY: s})
    assert q.shape == (2, 2)
    torch.testing.assert_close(q[1], torch.tensor([0.25, 0.0], dtype=F64) * q[0])


def test_the_dp_independence_declaration_follows_the_inner_element():
    fan = Scaled(FixedFlow(torch.tensor([0.1], dtype=F64), kind="fan"), "flow_multiplier.fan")
    assert fan.dp_independent
    assert not Scaled(_law(), KEY).dp_independent
    dp = torch.zeros(1, dtype=F64)
    q = fan.flow(dp, {"flow_multiplier.fan": torch.tensor([0.5], dtype=F64)})
    assert q.item() == pytest.approx(0.05, rel=1e-15)


def test_a_missing_or_misshapen_multiplier_is_named():
    el = Scaled(_law(), KEY)
    with pytest.raises(KeyError, match=KEY):
        el.flow(torch.ones(2, dtype=F64), {})
    with pytest.raises(ValueError, match="3 values.*2 edges"):
        el.flow(torch.ones(2, dtype=F64), {KEY: torch.ones(3, dtype=F64)})


def _one_zone(multiplier):
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="leak")
    net.add_edge("z", "ambient", kind="fan")
    layer = PotentialFlowLayer(
        net, "air",
        [Scaled(PowerLaw(torch.tensor(0.02, dtype=F64), torch.tensor(0.5, dtype=F64),
                         kind="leak"), "m"),
         FixedFlow(torch.tensor(0.01, dtype=F64), kind="fan")],
        boundary=["ambient"],
    )
    return layer.solve(torch.zeros(1, dtype=F64), {"m": multiplier}, atol=1e-14, rtol=1e-14)


def test_in_a_layer_a_multiplier_is_a_coefficient_scale_and_differentiable():
    s = torch.tensor(0.5, dtype=F64, requires_grad=True)
    phi, q = _one_zone(s)
    # The fan pulls 0.01 kg/s through the leak: 0.01 = s 0.02 sqrt(dp), dp = (1/s)^2.
    assert q[0].item() == pytest.approx(0.01, rel=1e-12)
    assert -phi[1].item() == pytest.approx((0.01 / (0.5 * 0.02)) ** 2, rel=1e-10)
    (g,) = torch.autograd.grad(phi[1], s)
    # phi_z = -(0.01 / (0.02 s))^2  ->  d phi_z / ds = 2 (0.01 / 0.02)^2 / s^3
    assert g.item() == pytest.approx(2 * (0.01 / 0.02) ** 2 / 0.5**3, rel=1e-8)


def test_a_zero_multiplier_on_the_only_path_names_the_floating_zone():
    with pytest.raises(RuntimeError, match=r"ungrounded interior nodes \['z'\]"):
        _one_zone(torch.tensor(0.0, dtype=F64))


def test_a_misshapen_multiplier_is_named_in_linear_init_too():
    with pytest.raises(ValueError, match="3 values.*2 edges"):
        Scaled(_law(), KEY).linear_init({KEY: torch.ones(3, dtype=F64)})


def test_a_gradient_carrying_inner_tensor_that_is_not_a_parameter_is_refused():
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="leak")
    C = torch.tensor(0.02, dtype=F64, requires_grad=True)
    inner = PowerLaw(C, torch.tensor(0.5, dtype=F64), kind="leak", learnable=False)
    layer = PotentialFlowLayer(net, "air", [Scaled(inner, "m")], boundary=["ambient"])
    with pytest.raises(ValueError, match="not a registered nn.Parameter"):
        layer.solve(torch.zeros(1, dtype=F64), {"m": torch.tensor(1.0, dtype=F64)},
                    torch.tensor([0.0, 0.01], dtype=F64), differentiable=True)


def test_a_scaled_fixed_flow_has_the_analytic_gradient_in_a_differentiable_solve():
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="leak")
    net.add_edge("z", "ambient", kind="fan")
    layer = PotentialFlowLayer(
        net, "air",
        [PowerLaw(torch.tensor(0.02, dtype=F64), torch.tensor(0.5, dtype=F64), kind="leak"),
         Scaled(FixedFlow(torch.tensor(0.01, dtype=F64), kind="fan"), "m")],
        boundary=["ambient"],
    )
    s = torch.tensor(0.5, dtype=F64, requires_grad=True)
    phi, _ = layer.solve(torch.zeros(1, dtype=F64), {"m": s}, atol=1e-14, rtol=1e-14,
                         differentiable=True)
    # s 0.01 = 0.02 sqrt(-phi_z)  ->  phi_z = -(s 0.01 / 0.02)^2, d phi_z / ds = -2 s (0.5)^2
    assert phi[1].item() == pytest.approx(-((0.5 * 0.01 / 0.02) ** 2), rel=1e-10)
    (g,) = torch.autograd.grad(phi[1], s)
    assert g.item() == pytest.approx(-2 * 0.5 * 0.25, rel=1e-8)


def test_a_batched_column_multiplier_broadcasts_through_a_layer_solve():
    s = torch.tensor([[1.0], [0.5]], dtype=F64)
    phi, q = _one_zone(s)
    assert phi.shape == (2, 2)
    expected = -((0.01 / (s.squeeze(-1) * 0.02)) ** 2)
    torch.testing.assert_close(phi[:, 1], expected, rtol=1e-9, atol=0)


@pytest.mark.parametrize("multiplier", [[0.5, 2.0], 0.5, (1.0, 3.0)])
def test_a_list_or_scalar_multiplier_is_float64(multiplier):
    """A Python list or scalar multiplier never falls through to the float32 default."""
    el = Scaled(_law(), KEY)
    q = el.flow(torch.tensor([4.0, -9.0], dtype=F64), {KEY: multiplier})
    assert q.dtype == F64
    assert el._s(2, {KEY: multiplier}).dtype == F64
