"""Damper: CONTAM's backdraft damper (PL_BDF) -- separate (C, n) per sign of dp.

Dtype (Ruling R7): ``Element._param`` casts a non-tensor value with
``torch.get_default_dtype()``, which is float32 in this repo. ``_damper()`` below builds its
(C, n) as float64 tensors, and the reference ``PowerLaw`` instances each test checks against
reuse those same float64 tensors, so ``Damper`` and its references share identical float64
precision. Without this, the two sides would each independently round the same Python float
literal to float32 and the comparisons -- some at rel=1e-6/1e-8 in the layer test below --
would pass only by the accident of float32 rounding error happening to fall under the
tolerance, not because the values genuinely agree to that precision. The same reasoning
applies to the layer test's ``FixedFlow(0.05, ...)``: a float32 q0 disagrees with the
python float 0.05 by ~1.5e-8 relative, which alone exceeds the dictated ``rel=1e-8`` on
``q[0]``, so it too is constructed as a float64 tensor.
"""

from __future__ import annotations

import pytest
import torch

from tellegen.elements import Damper, FixedFlow, PowerLaw
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.topology import Network

F64 = torch.float64
C_POS = torch.tensor(0.02, dtype=F64)
N_POS = torch.tensor(0.5, dtype=F64)
C_NEG = torch.tensor(0.002, dtype=F64)
N_NEG = torch.tensor(0.65, dtype=F64)


def _damper(learnable=False):
    return Damper(C_POS, N_POS, C_NEG, N_NEG, learnable=learnable)


def test_each_sign_reproduces_its_own_power_law():
    el = _damper()
    dp = torch.tensor([-50.0, -5.0, 5.0, 50.0], dtype=F64)
    q = el.flow(dp)
    torch.testing.assert_close(q[2:], PowerLaw(C_POS, N_POS).flow(dp[2:]))
    torch.testing.assert_close(q[:2], PowerLaw(C_NEG, N_NEG).flow(dp[:2]))
    assert q[0] < 0 and q[3] > 0 and q[3] > -q[0]


def test_dflow_matches_finite_differences_on_both_sides():
    el = _damper()
    dp = torch.tensor([-20.0, -0.5, 0.5, 20.0], dtype=F64)
    h = 1e-6
    fd = (el.flow(dp + h) - el.flow(dp - h)) / (2 * h)
    torch.testing.assert_close(el.dflow(dp), fd, rtol=1e-6, atol=1e-9)


def test_gradcheck_flow_wrt_dp_away_from_the_kinks_and_finite_at_zero():
    el = _damper()
    dp = torch.tensor([-20.0, -0.01, 0.01, 20.0], dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: el.flow(x), (dp,), eps=1e-7, atol=1e-6)
    zero = torch.zeros(1, dtype=F64, requires_grad=True)
    el.flow(zero).sum().backward()
    assert torch.isfinite(zero.grad).all()


def test_learnable_damper_registers_four_parameters_with_pos_and_neg_names():
    el = _damper(learnable=True)
    assert sorted(n for n, _ in el.named_parameters()) == ["neg.C", "neg.n", "pos.C", "pos.n"]


def test_linear_init_is_the_mean_tangent_of_the_two_laws():
    el = _damper()
    c, k = el.linear_init()
    _, kp = PowerLaw(C_POS, N_POS).linear_init()
    _, kn = PowerLaw(C_NEG, N_NEG).linear_init()
    torch.testing.assert_close(k, 0.5 * (kp + kn))
    assert c.item() == 0.0


def test_damper_in_a_layer_lets_a_fan_exhaust_through_the_easy_direction():
    """ambient -> z (damper), z -> ambient (FixedFlow exhaust 0.05 kg/s).

    The exhaust pulls 0.05 through the damper in its positive direction, so dp > 0 and the
    positive coefficients apply: dp = (0.05 / 0.02)^2 and phi_z = -dp.
    """
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="damper")
    net.add_edge("z", "ambient", kind="airpath")
    layer = PotentialFlowLayer(
        net,
        "air",
        [_damper(), FixedFlow(torch.tensor(0.05, dtype=F64), kind="airpath")],
        boundary=["ambient"],
    )
    # Newton's default atol/rtol are sqrt(eps) (~1.5e-8 for float64, tellegen.solvers.newton),
    # coarser than this test's rel=1e-8 on q[0]; pin a tight explicit pair here instead of
    # weakening the assertion to match the solver's default.
    diagnostics: dict = {}
    phi, q = layer.solve(
        torch.zeros(1, dtype=F64),
        {},
        differentiable=False,
        atol=1e-14,
        rtol=1e-14,
        diagnostics=diagnostics,
    )
    assert diagnostics["converged"].all()
    assert diagnostics["newton_iterations"] < 50
    assert q[0].item() == pytest.approx(0.05, rel=1e-8)
    assert -phi[1].item() == pytest.approx((0.05 / 0.02) ** 2, rel=1e-6)
