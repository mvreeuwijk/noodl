"""Hazen-Williams, the pump curves, the minor-loss valve and pressure-driven demand."""

import math

import pytest
import torch

from noodl.apps.water.controls import CLOSED, OPEN
from noodl.apps.water.demand import PressureDrivenDemand
from noodl.apps.water.elements import (
    HW_SI,
    KW_PER_HP,
    PIN_KEY,
    ConstantPowerPump,
    FlowControlledPump,
    HazenWilliams,
    MinorLoss,
    PiecewisePumpCurve,
    PumpCurve,
    three_point_curve,
)

F64 = torch.float64


def test_the_si_hazen_williams_constant_is_the_verified_one():
    assert HW_SI == 10.666829500036


def test_hazen_williams_resistance_is_the_manual_form():
    el = HazenWilliams(
        torch.tensor([500.0], dtype=F64), torch.tensor([0.300], dtype=F64),
        torch.tensor([130.0], dtype=F64),
    )
    expected = HW_SI * 130.0**-1.852 * 0.300**-4.871 * 500.0
    assert float(el.resistance()) == pytest.approx(expected, rel=1e-14)


def test_hazen_williams_inverts_its_own_law():
    el = HazenWilliams(
        torch.tensor([500.0], dtype=F64), torch.tensor([0.300], dtype=F64),
        torch.tensor([130.0], dtype=F64),
    )
    k = float(el.resistance())
    q = el.flow(torch.tensor([1.5], dtype=F64))
    assert float(k * q**1.852) == pytest.approx(1.5, rel=1e-12)
    assert float(el.flow(torch.tensor([-1.5], dtype=F64))) == pytest.approx(
        -float(q), rel=1e-12
    )


def test_hazen_williams_is_finite_at_dp_zero_and_matches_its_dflow():
    el = HazenWilliams(
        torch.tensor([500.0], dtype=F64), torch.tensor([0.300], dtype=F64),
        torch.tensor([130.0], dtype=F64),
    )
    dp = torch.zeros(1, dtype=F64, requires_grad=True)
    el.flow(dp).sum().backward()
    assert torch.isfinite(dp.grad).all()
    leaf = torch.tensor([0.002], dtype=F64, requires_grad=True)
    el.flow(leaf).sum().backward()
    assert float(el.dflow(torch.tensor([0.002], dtype=F64))) == pytest.approx(
        float(leaf.grad), rel=1e-9
    )


def test_the_transition_is_far_below_the_fixtures_smallest_head_loss():
    """MEASURED: the smallest HEAD LOSS on twoloop_si.inp is 5.8188e-3 m, across pipe P7.

    Not 1.9e-5 m: `wntr`'s `results.link['headloss']` reports a PIPE's hydraulic GRADIENT
    in m/m (1.9394e-5 for P7 over its 300 m) and only a PUMP's value in metres. The
    default transition is six orders below the real loss, so the blend never touches a
    pipe that carries flow; what it exists for is a pipe carrying EXACTLY zero (Net1's
    pipe 10 once a control closes its pump).

    The gradient the assertion below computes (1.939615369e-05) is THIS MODEL's own
    loss-over-length, from the closed-form Hazen-Williams inversion; the `.rpt` EPANET
    itself writes for P7 is 1.9394165e-05 -- a ~1e-4 relative gap between the model and
    EPANET's own float32-limited report, not a bug in either.
    """
    el = HazenWilliams(
        torch.tensor([300.0], dtype=F64), torch.tensor([0.150], dtype=F64),
        torch.tensor([130.0], dtype=F64),
    )
    assert el.dp_transition == 1e-9
    q = el.flow(torch.tensor([5.818846106e-3], dtype=F64))
    assert float(q) == pytest.approx(7.037718599392933e-4, rel=1e-6)
    # and the gradient this model computes for P7 (loss over length)
    assert 5.818846106e-3 / 300.0 == pytest.approx(1.939615369e-05, rel=1e-6)


def test_a_minor_loss_term_is_inverted_by_the_bracketed_root():
    el = HazenWilliams(
        torch.tensor([500.0], dtype=F64), torch.tensor([0.300], dtype=F64),
        torch.tensor([130.0], dtype=F64), minor_loss=torch.tensor([2.0], dtype=F64),
    )
    dp = torch.tensor([1.5], dtype=F64)
    q = el.flow(dp)
    k = float(el.resistance())
    m = 2.0 / (2.0 * 9.80665 * (torch.pi * 0.300**2 / 4.0) ** 2)
    assert float(k * q**1.852 + m * q**2) == pytest.approx(1.5, rel=1e-10)
    bare = HazenWilliams(
        torch.tensor([500.0], dtype=F64), torch.tensor([0.300], dtype=F64),
        torch.tensor([130.0], dtype=F64),
    )
    assert float(q) < float(bare.flow(dp))


def test_the_three_point_fit_reproduces_epanets_own_coefficients():
    """Net1's single point, 1500 GPM / 250 ft: EPANET's points (0, 1.33334 H), (q, H),
    (2 q, 0) fitted by `powercurve` (`input2.c`, `input3.c`), n = 1.99998 (wntr's own
    `get_head_curve_coefficients` keeps n = 2: (101.6, 2836.1385287628063, 2))."""
    gpm = 3.785411784e-3 / 60.0
    q, h = 1500.0 * gpm, 250.0 * 0.3048
    h0, r, n = three_point_curve(torch.tensor([q], dtype=F64), torch.tensor([h], dtype=F64))
    expected_n = math.log(1.33334 / 0.33334) / math.log(2.0)
    assert n == pytest.approx(expected_n, rel=1e-15) and abs(n - 1.99998) < 1e-5
    assert float(h0) == pytest.approx(1.33334 * h, rel=1e-15)
    assert float(r) == pytest.approx(0.33334 * h / q**expected_n, rel=1e-13)
    # the curve passes through the design point and EPANET's zero head at twice the flow
    assert float(h0 - r * q**n) == pytest.approx(h, rel=1e-13)
    assert abs(float(h0 - r * (2 * q) ** n)) < 1e-12


def test_the_pump_curve_inverts_its_fitted_law():
    pump = PumpCurve(
        torch.tensor([101.6], dtype=F64), torch.tensor([2836.1385287628063], dtype=F64)
    )
    # dp = phi_suction - phi_discharge = -(head gain)
    q = pump.flow(torch.tensor([-62.285091400146484], dtype=F64))
    # MEASURED 0.11773752856904683 against EPANET's own 0.11773740500211716, i.e. 1.05e-6
    # relative. The head gain fed in is EPANET's FLOAT32 report (-62.285091400146484), and
    # inverting a quadratic curve at a float32 head cannot do better than the head's own
    # precision; the full solve agrees to 9.553e-7 (checked in
    # `tests/verification/test_water_parity.py::test_net1_single_period`).
    assert float(q) == pytest.approx(0.11773740500211716, rel=1e-5)


def test_the_pump_shuts_off_above_its_shut_off_head():
    pump = PumpCurve(
        torch.tensor([101.6], dtype=F64), torch.tensor([2836.1385287628063], dtype=F64)
    )
    assert float(pump.flow(torch.tensor([-150.0], dtype=F64))) == 0.0
    dp = torch.tensor([-101.6], dtype=F64, requires_grad=True)
    pump.flow(dp).sum().backward()
    assert torch.isfinite(dp.grad).all()


def test_the_pump_flow_is_monotone_in_dp():
    pump = PumpCurve(
        torch.tensor([101.6] * 50, dtype=F64),
        torch.tensor([2836.1385287628063] * 50, dtype=F64),
    )
    dp = torch.linspace(-101.6, 0.0, 50, dtype=F64)
    q = pump.flow(dp)
    assert bool((q[1:] >= q[:-1]).all())
    assert bool((pump.dflow(dp) >= 0).all())


def test_a_closed_status_driver_zeroes_the_pump():
    pump = PumpCurve(
        torch.tensor([101.6], dtype=F64), torch.tensor([2836.1385287628063], dtype=F64),
        links=slice(0, 1),
    )
    drivers = {"water.status": torch.zeros(1, dtype=F64),
               "water.setting": torch.ones(1, dtype=F64)}
    assert float(pump.flow(torch.tensor([-50.0], dtype=F64), drivers)) == 0.0
    drivers["water.status"] = torch.ones(1, dtype=F64)
    assert float(pump.flow(torch.tensor([-50.0], dtype=F64), drivers)) > 0.0


def test_a_missing_status_driver_is_refused():
    pump = PumpCurve(
        torch.tensor([101.6], dtype=F64), torch.tensor([2836.1385287628063], dtype=F64),
        links=slice(0, 1),
    )
    with pytest.raises(KeyError, match="'water.status'"):
        pump.flow(torch.tensor([-50.0], dtype=F64), {})


def _net1_pump():
    return PumpCurve(
        torch.tensor([101.6], dtype=F64), torch.tensor([2836.1385287628063], dtype=F64),
        links=slice(0, 1),
    )


FLOOR, CBIG = 1e-6, 1e9
ON = {"water.status": torch.ones(1, dtype=F64), "water.setting": torch.ones(1, dtype=F64)}


def test_the_head_gain_is_the_fitted_curve_above_the_floor_flow():
    pump = _net1_pump()
    q = torch.tensor([0.05, 0.11773740500211716], dtype=F64)
    gain = pump.head_gain(q, FLOOR, CBIG)
    assert gain.tolist() == pytest.approx((101.6 - 2836.1385287628063 * q**2).tolist(),
                                          rel=1e-14)
    # and it agrees with the inverted curve `flow`
    assert float(pump.flow(-gain[1:], ON)) == pytest.approx(float(q[1]), rel=1e-12)


def test_the_head_gain_is_strictly_decreasing_and_continuous_through_zero_flow():
    pump = _net1_pump()
    q_bar = FLOOR / (2.0 * 2836.1385287628063)
    q = torch.tensor([-1e-3, -1e-9, 0.0, 0.5 * q_bar, q_bar, 2.0 * q_bar, 1e-3],
                     dtype=F64, requires_grad=True)
    gain = pump.head_gain(q, FLOOR, CBIG)
    # (near q_bar the steps are below a float64 spacing of the gain, hence <=)
    assert bool((gain[1:] <= gain[:-1]).all())
    (slope,) = torch.autograd.grad(gain.sum(), q)
    assert bool((slope < 0).all())
    # below q_bar the gain is the curve's tangent there, with EPANET's RQtol slope, through
    # zero flow and beyond (the pump's no-flow row, not the curve, shuts it)
    for k in (0, 1, 2, 3):
        assert float(slope[k]) == pytest.approx(-FLOOR, rel=1e-9)
    torch.testing.assert_close(-slope, pump.gain_slope(q.detach(), FLOOR, CBIG), rtol=1e-12,
                               atol=0)
    lo = torch.tensor([q_bar - 1e-15], dtype=F64)
    hi = torch.tensor([q_bar + 1e-15], dtype=F64)
    jump = pump.head_gain(hi, FLOOR, CBIG) - pump.head_gain(lo, FLOOR, CBIG)
    assert abs(float(jump)) < 1e-12


def test_a_pump_curve_takes_its_own_exponent_and_the_speed_affinity_laws():
    """``w^2 h0 - r w^(2-n) q^n`` per pump (`hydcoeffs.c`, `pumpcoeff`)."""
    pump = PumpCurve(torch.tensor([31.6992, 60.0], dtype=F64),
                     torch.tensor([20.0, 300.0], dtype=F64),
                     torch.tensor([1.77, 2.0], dtype=F64))
    q = torch.tensor([0.1, 0.2], dtype=F64)
    w = torch.tensor([0.8, 1.1], dtype=F64)
    n, r, h0 = pump.n, pump.r, pump.h0
    expected = w**2 * h0 - r * w ** (2 - n) * q**n
    torch.testing.assert_close(pump.head_gain(q, FLOOR, CBIG, w), expected, rtol=1e-14,
                               atol=0)
    torch.testing.assert_close(pump.q_max(w), w * (h0 / r) ** (1 / n), rtol=1e-14, atol=0)


def test_an_exponent_below_one_is_capped_at_zero_flow():
    """``n < 1``: the slope grows without bound towards zero flow; below the flow where it
    reaches ``max_slope`` (EPANET's ``CBIG``) the gain is the tangent there."""
    pump = PumpCurve(torch.tensor([30.0], dtype=F64), torch.tensor([40.0], dtype=F64),
                     torch.tensor([0.5], dtype=F64))
    q = torch.tensor([-1e-3, 0.0, 1e-12, 0.01], dtype=F64, requires_grad=True)
    gain = pump.head_gain(q, FLOOR, 1e4)
    (slope,) = torch.autograd.grad(gain.sum(), q)
    assert torch.isfinite(gain).all() and bool((slope < 0).all())
    assert float(slope[0]) == pytest.approx(-1e4, rel=1e-12)
    assert float(gain[3].detach()) == pytest.approx(30.0 - 40.0 * 0.01**0.5, rel=1e-14)


def test_a_piecewise_curve_follows_its_segments_and_scales_with_speed():
    """`curvecoeff`'s bracketing segment, extended at both ends; ``w^2 h(q / w)``."""
    flows = torch.tensor([[0.0, 0.1, 0.2, 0.3, torch.inf]], dtype=F64)
    heads = torch.tensor([[90.0, 88.0, 80.0, 60.0, 0.0]], dtype=F64)
    pump = PiecewisePumpCurve(flows, heads, [4])
    q = torch.tensor([0.05, 0.15, 0.25, 0.35, -0.1], dtype=F64).unsqueeze(-1)
    gain = pump.head_gain(q).squeeze(-1)
    assert gain.tolist() == pytest.approx([89.0, 84.0, 70.0, 50.0, 92.0], rel=1e-14)
    w = torch.tensor([0.8], dtype=F64)
    at_speed = pump.head_gain(torch.tensor([[0.12]], dtype=F64), speed=w)
    assert float(at_speed) == pytest.approx(0.64 * 84.0, rel=1e-14)   # 0.12 / 0.8 = 0.15
    slope = pump.gain_slope(torch.tensor([[0.12]], dtype=F64), speed=w)
    assert float(slope) == pytest.approx(0.8 * 80.0, rel=1e-14)


def test_a_constant_power_pump_is_epanets_8_814_p_over_q():
    """``8.814 P / q`` ft in horsepower and cfs (`updatepumpparams`, `pumpcoeff`), here
    in metres and m3/s, with the speed cubed; the tangent beyond EPANET's ``CBIG`` cap."""
    cfs = 1.0 / 0.3048**3
    pump = ConstantPowerPump(torch.tensor([10.0], dtype=F64), kw_per_hp=KW_PER_HP)
    q = torch.tensor([0.05], dtype=F64)
    p_hp = 10.0 / 0.7457
    expected = 0.3048 * 8.814 * p_hp / (0.05 * cfs)
    assert float(pump.head_gain(q, FLOOR)) == pytest.approx(expected, rel=1e-14)
    w = torch.tensor([0.9], dtype=F64)
    assert float(pump.head_gain(q, FLOOR, speed=w)) == pytest.approx(0.729 * expected,
                                                                      rel=1e-14)
    cap = pump.max_slope
    assert cap == pytest.approx(1e8 * 0.3048 * cfs, rel=1e-14)
    q_lo = (float(pump.coefficient) / cap) ** 0.5
    low = torch.tensor([-1e-3, 0.0, 0.5 * q_lo], dtype=F64, requires_grad=True)
    gain = pump.head_gain(low, FLOOR)
    (slope,) = torch.autograd.grad(gain.sum(), low)
    assert slope.tolist() == pytest.approx([-cap] * 3, rel=1e-12)
    assert float(gain[1].detach()) == pytest.approx(2.0 * cap * q_lo, rel=1e-12)


def _flow_pump(kappa=3.0):
    return FlowControlledPump(_net1_pump(), gradient_floor=FLOOR, max_slope=CBIG,
                              kappa=torch.tensor([kappa], dtype=F64))


def test_the_head_gain_form_residual_follows_the_status_and_speed_drivers():
    law = _flow_pump()
    dp, q = torch.tensor([-62.0], dtype=F64), torch.tensor([0.1], dtype=F64)
    gain = float(law.pump.head_gain(q, FLOOR, CBIG))
    closed = {**ON, "water.status": torch.zeros(1, dtype=F64)}
    stopped = {**ON, "water.setting": torch.zeros(1, dtype=F64)}
    slow = {**ON, "water.setting": torch.full((1,), 0.9, dtype=F64)}
    assert float(law.residual(dp, q, ON, None)) == pytest.approx(-62.0 + gain, rel=1e-14)
    assert float(law.residual(dp, q, closed, None)) == pytest.approx(0.3, rel=1e-15)
    assert float(law.residual(dp, q, stopped, None)) == pytest.approx(0.3, rel=1e-15)
    at_09 = float(law.pump.head_gain(q, FLOOR, CBIG, torch.tensor([0.9], dtype=F64)))
    assert float(law.residual(dp, q, slow, None)) == pytest.approx(-62.0 + at_09, rel=1e-14)
    assert float(law.q_max()) == pytest.approx((101.6 / 2836.1385287628063) ** 0.5,
                                               rel=1e-14)


def test_a_pump_beyond_its_shut_off_head_takes_the_no_flow_row():
    """``max(-kappa q, dp + gain(q))``: the no-flow row where the head asked of the pump
    exceeds its shut-off head and the open row's Newton step would reverse the flow; the
    curve at a tie and wherever the flow is forward on it."""
    law = _flow_pump()
    q = torch.tensor([0.0], dtype=F64)
    beyond = torch.tensor([-150.0], dtype=F64)        # shut-off 101.6 m
    assert float(law.residual(beyond, q, ON, None)) == 0.0                      # kappa q
    assert int(law.states(beyond, q, ON)) == CLOSED
    tie = -law.pump.head_gain(q, FLOOR, CBIG)
    assert float(law.residual(tie, q, ON, None)) == 0.0
    assert int(law.states(tie, q, ON)) == OPEN
    # a forward flow whose open Newton step stays forward keeps the curve's row, even with
    # the row below -kappa q; one whose step would reverse it takes the no-flow row
    q = torch.tensor([0.12], dtype=F64)
    above = torch.tensor([-70.0], dtype=F64)
    r = float(law.residual(above, q, ON, None))
    assert r == pytest.approx(-70.0 + float(law.pump.head_gain(q, FLOOR, CBIG)), rel=1e-14)
    assert r < -0.36
    assert float(law.residual(beyond, q, ON, None)) == pytest.approx(0.36, rel=1e-15)


def test_pumps_of_several_kinds_share_one_law():
    """`(curve, columns)` parts: each pump's gain is its own curve's, in column order."""
    power = PumpCurve(torch.tensor([101.6], dtype=F64),
                      torch.tensor([2836.1385287628063], dtype=F64))
    segments = PiecewisePumpCurve(torch.tensor([[0.0, 0.1, 0.2]], dtype=F64),
                                  torch.tensor([[90.0, 88.0, 80.0]], dtype=F64), [3])
    constant = ConstantPowerPump(torch.tensor([10.0], dtype=F64))
    law = FlowControlledPump([(segments, [2]), (power, [0]), (constant, [1])],
                             gradient_floor=FLOOR, max_slope=CBIG, links=slice(0, 3))
    q = torch.tensor([0.1, 0.05, 0.15], dtype=F64)
    gain = law.head_gain(q)
    assert gain.tolist() == pytest.approx([
        float(power.head_gain(q[:1], FLOOR, CBIG)), float(constant.head_gain(q[1:2], FLOOR)),
        84.0], rel=1e-14)
    q_max = law.q_max()
    assert math.isfinite(float(q_max[0])) and math.isinf(float(q_max[1]))
    assert math.isinf(float(q_max[2]))


def test_minor_loss_valve_inverts_its_quadratic():
    el = MinorLoss(torch.tensor([10.0], dtype=F64), torch.tensor([0.2], dtype=F64))
    q = el.flow(torch.tensor([0.5], dtype=F64))
    m = float(el.coefficient())
    assert float(m * q * q.abs()) == pytest.approx(0.5, rel=1e-12)
    dp = torch.zeros(1, dtype=F64, requires_grad=True)
    el.flow(dp).sum().backward()
    assert torch.isfinite(dp.grad).all()


# ------------------------------------------------------------------ laminar blend dtype
def test_minor_loss_laminar_blend_is_full_float64_precision():
    """Building the laminar-blend constant with `torch.as_tensor(dpt)` would be wrong: it
    casts a bare Python float with `torch.get_default_dtype()` (float32) BEFORE the
    division -- type promotion then upgrades the RESULT back to float64, so the bug is
    invisible to a `.dtype` check and only shows up as an ~1.4e-8 relative precision loss
    (measured) against the value computed with `dpt` at full float64 precision throughout,
    which is what a pipe carrying exactly zero flow (the transition's whole reason to
    exist) needs from its tangent slope.
    """
    el = MinorLoss(torch.tensor([10.0], dtype=F64), torch.tensor([0.2], dtype=F64))
    dpt = el.dp_transition
    m = float(el.coefficient())
    expected_slope = (dpt / m) ** 0.5 / dpt

    zero, slope = el.linear_init()
    assert zero.dtype == F64
    assert slope.dtype == F64
    assert float(slope) == pytest.approx(expected_slope, rel=1e-13)

    dp0 = torch.zeros(1, dtype=F64)
    assert float(el.dflow(dp0)) == pytest.approx(expected_slope, rel=1e-13)


# --------------------------------------------------------------- PDA's exact kink
def test_pressure_driven_demand_is_exactly_zero_at_and_below_p_min():
    """The Wagner curve is EXACTLY 0 at and below `P_min`, the fractional
    law strictly between, and EXACTLY `q_required` at and above `P_req`; gradients finite
    (and exactly 0 below `P_min`) everywhere."""
    demand = PressureDrivenDemand(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0.01], dtype=F64),
        torch.tensor([0.0], dtype=F64),
        p_min=10.0, p_req=30.0,
    )
    # below P_min, at P_min, between, at P_req, above P_req
    heads = torch.tensor([5.0, 10.0, 20.0, 30.0, 40.0], dtype=F64, requires_grad=True)
    q = demand.flow(heads.unsqueeze(-1))
    values = q.detach()
    assert float(values[0]) == 0.0
    assert float(values[1]) == 0.0
    assert float(values[2]) == pytest.approx(0.01 * (10.0 / 20.0) ** 0.5, rel=1e-14)
    assert float(values[3]) == pytest.approx(0.01, rel=1e-14)
    assert float(values[4]) == pytest.approx(0.01, rel=1e-14)
    q.sum().backward()
    assert torch.isfinite(heads.grad).all()
    assert float(heads.grad[0]) == 0.0
    assert float(heads.grad[1]) == 0.0


# ------------------------------------------------------------------ PBV and GPV
def test_a_breaker_valve_holds_its_setting_until_its_minor_loss_exceeds_it():
    from noodl.apps.water.elements import BreakerValve

    valve = BreakerValve(torch.tensor([5.0, 5.0, 5.0, 0.0]), torch.tensor([100.0] * 4))
    q = torch.tensor([0.1, -0.1, 0.3, 0.1], dtype=torch.float64)
    dp = torch.zeros(4, dtype=torch.float64)
    # active (m q^2 = 1 <= 5) either way the water flows; open (9 > 5); open (setting 0)
    expected = torch.tensor([5.0, 5.0, 9.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(-valve.residual(dp, q, {}, None), expected)


def test_a_general_purpose_valve_follows_its_bracketing_curve_segment():
    from noodl.apps.water.elements import GeneralPurposeValve

    inf = float("inf")
    flows = torch.tensor([[0.0, 10.0, 30.0], [0.0, 1.0, inf]], dtype=torch.float64)
    heads = torch.tensor([[0.0, 1.0, 6.0], [2.0, 4.0, 0.0]], dtype=torch.float64)
    valve = GeneralPurposeValve(flows, heads, [3, 2])
    q = torch.tensor([[5.0, 0.5], [40.0, 3.0], [-20.0, -0.5], [0.0, 0.0]],
                     dtype=torch.float64)
    head = -valve.residual(torch.zeros_like(q), q, {}, None)
    expected = torch.tensor([
        [0.5, 3.0],      # inside the first segment
        [8.5, 8.0],      # past the last point: its segment extended (`curvecoeff`)
        [-3.5, -3.0],    # reversed flow: the loss at |q|, with the flow's sign
        [0.0, 2.0],      # zero flow: sgn(0) = +1, the curve's own intercept
    ], dtype=torch.float64)
    torch.testing.assert_close(head, expected)


# ------------------------------------------------------------------ PRV and PSV
def _head_valve(kind, **kwargs):
    from noodl.apps.water.elements import PressureReducingValve, PressureSustainingValve

    cls = PressureReducingValve if kind == "prv" else PressureSustainingValve
    return cls(torch.tensor([10.0], dtype=F64), torch.tensor([100.0], dtype=F64),
               kappa=torch.tensor([2.0], dtype=F64), setting=torch.tensor([40.0], dtype=F64),
               **kwargs)


def _context(phi_from, phi_to):
    from noodl.elements.base import LawContext

    return LawContext("dc", phi_from=torch.tensor(phi_from, dtype=F64),
                      phi_to=torch.tensor(phi_to, dtype=F64))


@pytest.mark.parametrize("kind, heads, q, expected", [
    # kappa = 2, m = 100. PRV, Hset = 10 + 40 = 50 at the end node. ACTIVE: held =
    # 50.5 - 50 = 0.5 above loss = 100 * 0.3^2 - 9.5 = -0.5, and below kappa q = 0.6
    ("prv", ([60.0], [50.5]), [0.3], 0.5),
    # OPEN: loss = 9 - 8.75 = 0.25 above held = 41.25 - 50
    ("prv", ([50.0], [41.25]), [0.3], 0.25),
    # CLOSED: reverse flow, kappa q = -0.2 below g = held = 3 (q = 0 is its root)
    ("prv", ([60.0], [53.0]), [-0.1], -0.2),
    # PSV, Hset = 50 at the start node: ACTIVE, held = 50 - 49.5 above loss = 9 - 9.5
    ("psv", ([49.5], [40.0]), [0.3], 0.5),
    # OPEN: loss = 9 - 8.75 above held = 50 - 60
    ("psv", ([60.0], [51.25]), [0.3], 0.25),
])
def test_a_head_valve_residual_is_the_complementarity_of_its_flow_and_its_reason_to_close(
        kind, heads, q, expected):
    """`r = min(kappa q, max(held, loss))`, `held` the controlled head beyond its setting
    (PRV `H_v - Hset`, PSV `Hset - H_u`) and `loss` the open loss less the drop."""
    valve = _head_valve(kind)
    ctx = _context(*heads)
    dp = ctx.phi_from - ctx.phi_to
    r = valve.residual(dp, torch.tensor(q, dtype=F64), {}, ctx)
    torch.testing.assert_close(r, torch.tensor([expected], dtype=F64))


def test_a_head_valve_tie_takes_the_active_row():
    """At `held == loss` the residual is the ACTIVE row (its derivative by the controlled
    head is 1, by the start head and the flow 0), not an average of the two; and at
    `kappa q == g` it is the valve's own row, not the no-flow row."""
    from noodl.elements.base import LawContext

    valve = _head_valve("prv")
    phi_from = torch.tensor([52.0], dtype=F64, requires_grad=True)
    phi_to = torch.tensor([50.2], dtype=F64, requires_grad=True)   # held = 0.2
    q = torch.tensor([0.02**0.5], dtype=F64, requires_grad=True)    # loss = 2 - 1.8 = 0.2
    ctx = LawContext("dc", phi_from=phi_from, phi_to=phi_to)
    r = valve.residual(phi_from - phi_to, q, {}, ctx)
    assert float(r.detach()) == pytest.approx(0.2, abs=1e-12)
    grads = torch.autograd.grad(r, (phi_from, phi_to, q))
    assert [float(g) for g in grads] == [0.0, 1.0, 0.0]
    # kappa q == g == 1: the valve's own (ACTIVE) row
    q = torch.tensor([0.5], dtype=F64, requires_grad=True)
    phi_to = torch.tensor([51.0], dtype=F64, requires_grad=True)
    ctx = LawContext("dc", phi_from=torch.tensor([100.0], dtype=F64), phi_to=phi_to)
    r = valve.residual(ctx.phi_from - phi_to, q, {}, ctx)
    grads = torch.autograd.grad(r, (phi_to, q))
    assert [float(r.detach()), *(float(g) for g in grads)] == [1.0, 1.0, 0.0]


def test_a_smoothed_head_valve_is_fischer_burmeister():
    """`smoothing = eps` (the layer's potential unit) replaces the `min` by
    `a + b - sqrt(a^2 + b^2 + 2 eps^2)`, whose root is `a b = eps^2`, and the inner `max`
    by `(x + y + sqrt((x - y)^2 + 4 eps^2)) / 2`."""
    eps = 1e-3
    valve = _head_valve("prv", smoothing=eps)
    ctx = _context([60.0], [50.5])
    q = torch.tensor([0.1], dtype=F64)
    r = valve.residual(ctx.phi_from - ctx.phi_to, q, {}, ctx)
    held, loss = 0.5, 100 * 0.01 - 9.5
    g = 0.5 * (held + loss + math.sqrt((held - loss) ** 2 + 4 * eps**2))
    a = 2.0 * 0.1
    assert float(r) == pytest.approx(a + g - math.sqrt(a * a + g * g + 2 * eps**2), rel=1e-14)


def test_a_head_valve_reads_its_status_and_setting_from_the_drivers():
    from noodl.apps.water.controls import ACTIVE, CLOSED, OPEN
    from noodl.apps.water.elements import PressureReducingValve

    valve = PressureReducingValve(torch.tensor([0.0, 0.0, 0.0], dtype=F64),
                                  torch.tensor([100.0] * 3, dtype=F64),
                                  kappa=torch.tensor([2.0] * 3, dtype=F64), links=slice(1, 4))
    drivers = {"water.status": torch.tensor([1.0, CLOSED, OPEN, ACTIVE], dtype=F64),
               "water.setting": torch.tensor([0.0, 50.0, 50.0, 50.0], dtype=F64)}
    ctx = _context([60.0] * 3, [50.5] * 3)
    q = torch.tensor([0.3] * 3, dtype=F64)
    r = valve.residual(ctx.phi_from - ctx.phi_to, q, drivers, ctx)
    # CLOSED: kappa q; OPEN: the drop less the open loss; ACTIVE: H_v - Hset
    torch.testing.assert_close(r, torch.tensor([0.6, 9.5 - 9.0, 0.5], dtype=F64))


@pytest.mark.parametrize("kind, dead, heads, row, state", [
    # kappa = 2, m = 100, Hset = 50, q = 0.3 (dp = 2.5 in each case)
    # a PSV's end dead, its start (49.5) below Hset = 50: CLOSED, end tied to start (dp)
    ("psv", 1.0, ([49.5], [47.0]), 2.5, 0),
    # ... at or above it: OPEN, the open row (loss = 9 - 2.5 at q = 0.3)
    ("psv", 1.0, ([60.0], [57.5]), 6.5, 1),
    # a PRV's end dead: its own g (held = 47 - 50 against loss = 6.5): OPEN
    ("prv", 1.0, ([49.5], [47.0]), 6.5, 1),
    # a start dead, either kind: CLOSED, dp
    ("prv", -1.0, ([49.5], [47.0]), 2.5, 0),
    ("psv", -1.0, ([60.0], [57.5]), 2.5, 0),
])
def test_a_dead_end_row_and_the_reported_state_agree(kind, dead, heads, row, state):
    """With `"water.dead_end"` the residual takes the row `_dead_rows` gives and `states`
    reports that row's state (a CLOSED tie is `dp`, the open row `loss`, held `g`)."""
    from noodl.apps.water.controls import ACTIVE

    valve = _head_valve(kind, links=slice(0, 1))
    drivers = {"water.status": torch.tensor([float(ACTIVE)], dtype=F64),
               "water.setting": torch.tensor([40.0], dtype=F64),
               "water.dead_end": torch.tensor([dead], dtype=F64)}
    ctx = _context(*heads)
    dp, q = ctx.phi_from - ctx.phi_to, torch.tensor([0.3], dtype=F64)
    r = valve.residual(dp, q, drivers, ctx)
    assert float(r) == pytest.approx(row, abs=1e-12)
    assert int(valve.states(dp, q, drivers, ctx)) == state


def _mixed_pumps():
    """Four pumps of every kind in one law: power functions with n = 2 and n = 0.5, a
    segmented curve starting above zero flow, and constant power."""
    power = PumpCurve(torch.tensor([101.6, 30.0], dtype=F64),
                      torch.tensor([2836.0, 40.0], dtype=F64), torch.tensor([2.0, 0.5], dtype=F64))
    segments = PiecewisePumpCurve(torch.tensor([[0.01, 0.1, 0.2]], dtype=F64),
                                  torch.tensor([[90.0, 88.0, 80.0]], dtype=F64), [3])
    constant = ConstantPowerPump(torch.tensor([10.0], dtype=F64))
    return FlowControlledPump([(power, [0, 3]), (segments, [1]), (constant, [2])],
                              gradient_floor=FLOOR, max_slope=CBIG,
                              kappa=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=F64),
                              links=slice(0, 4))


_RUNNING = {"water.status": torch.ones(4, dtype=F64),
            "water.setting": torch.tensor([0.9, 0.8, 1.1, 1.0], dtype=F64)}
_STOPPED = {"water.status": torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=F64),
            "water.setting": torch.tensor([0.9, 0.0, 1.1, 1.0], dtype=F64),
            PIN_KEY: torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=F64)}


@pytest.mark.parametrize("dp, q, drivers, rows", [
    # running on every curve
    ([-60.0, -50.0, -30.0, -20.0], [0.05, 0.1, 0.05, 0.2], _RUNNING, "open"),
    # low and reverse flow: the RQtol and CBIG tangents, below a segmented curve's first
    # point, below constant power's CBIG flow
    ([-60.0, -50.0, -30.0, -20.0], [-1e-3, 0.005, 1e-9, 1e-12], _RUNNING, "mixed"),
    # shut beyond the shut-off head
    ([-200.0, -200.0, -1e6, -200.0], [0.0, 0.0, 0.0, 0.0], _RUNNING, "shut"),
    # CLOSED and pinned, at speed 0, running
    ([-60.0, -50.0, -30.0, -20.0], [0.05, 0.1, 0.05, 0.2], _STOPPED, "mixed"),
], ids=["running", "low-flow", "shut", "stopped-closed-pinned"])
def test_the_pump_jacobian_is_the_derivative_of_its_residual(dp, q, drivers, rows):
    """`FlowControlledPump.jacobian` (closed form) against autograd of `residual`, per
    pump, in every regime of every curve kind."""
    law = _mixed_pumps()
    dp, q = torch.tensor(dp, dtype=F64), torch.tensor(q, dtype=F64)
    d_dp, d_dq = (block.reshape(4) for block in law.jacobian(dp, q, drivers, None))
    a_dp, a_dq = torch.autograd.functional.jacobian(
        lambda d, f: law.residual(d, f, drivers, None), (dp, q))
    torch.testing.assert_close(d_dp, a_dp.diagonal(), rtol=1e-12, atol=0)
    torch.testing.assert_close(d_dq, a_dq.diagonal(), rtol=1e-12, atol=0)
    assert torch.count_nonzero(a_dp - torch.diag(a_dp.diagonal())) == 0
    assert torch.count_nonzero(a_dq - torch.diag(a_dq.diagonal())) == 0
    states = law.states(dp, q, drivers)
    if rows == "open":
        assert bool((states == OPEN).all())
    elif rows == "shut":
        assert bool((states == CLOSED).all()) and bool((d_dp == 0).all())


def test_an_exponent_below_one_needs_a_slope_cap():
    """Below 1 the slope is unbounded at zero flow: without ``max_slope`` the gain is
    refused by name rather than returned as NaN."""
    pump = PumpCurve(torch.tensor([30.0], dtype=F64), torch.tensor([40.0], dtype=F64),
                     torch.tensor([0.5], dtype=F64))
    with pytest.raises(ValueError, match="exponent below 1 .* give max_slope"):
        pump.head_gain(torch.tensor([0.0], dtype=F64), FLOOR)
    assert torch.isfinite(pump.head_gain(torch.tensor([-1.0, 0.0], dtype=F64), FLOOR,
                                         CBIG)).all()


def test_an_exponent_within_tiny_of_one_is_a_line():
    """EPANET's `pumpcoeff` sets n = 1.0 where |n - 1| < TINY: the gain is
    ``w^2 h0 - r w q`` exactly, with no floor."""
    pump = PumpCurve(torch.tensor([30.0], dtype=F64), torch.tensor([40.0], dtype=F64),
                     torch.tensor([1.0 + 5e-7], dtype=F64))
    q = torch.tensor([-0.1, 0.0, 0.3], dtype=F64)
    w = torch.tensor([0.8], dtype=F64)
    gain, slope = pump.gain_and_slope(q, FLOOR, CBIG, w)
    torch.testing.assert_close(gain, 0.64 * 30.0 - 40.0 * 0.8 * q, rtol=1e-15, atol=1e-15)
    assert slope.tolist() == pytest.approx([32.0] * 3, rel=1e-15)


def test_the_inverted_curve_of_a_stopped_pump_is_no_flow():
    """`PumpCurve.flow` at speed 0 (`"water.setting"`): no flow and a zero slope, not a
    division by zero."""
    pump = _net1_pump()
    drivers = {"water.status": torch.ones(1, dtype=F64),
               "water.setting": torch.zeros(1, dtype=F64)}
    dp = torch.tensor([-50.0], dtype=F64)
    assert float(pump.flow(dp, drivers)) == 0.0
    assert float(pump.dflow(dp, drivers)) == 0.0
