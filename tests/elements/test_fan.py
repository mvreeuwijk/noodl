"""Tests for FanCurve: dp = -P(q), P a cubic pressure-flow curve inverted by solve_monotone."""

import math

import pytest
import torch

from noodl.elements import AffinityFan, UpstreamDensityPowerLaw
from noodl.elements.base import Element
from noodl.elements.fan import FanCurve


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
    # convention, not decreasing (which is also why the two "stalled" clip-boundary test
    # values below, and in test_dflow_matches_... below, carry the signs they do).
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
    # ("- 10.0, - 100.0" would compute dp < -p_max_flow instead and so never actually fall
    # in the stalled region for this curve's coefficients).
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


def test_dflow_gradients_are_finite_at_a_zero_slope_stalled_boundary():
    # P(q) = 200 + 6q - q^2, P'(q) = 6 - 2q, P'(q_max=3) = 0: the curve flattens exactly at
    # its top-of-curve (max-flow) point, a legitimate way to define that boundary. dp = 500
    # gives -dp = -500 < P(q_max) = 200 + 18 - 9 = 209, so this is deep in the stalled region.
    coeffs = torch.tensor([200.0, 6.0, -1.0, 0.0], dtype=torch.float64, requires_grad=True)
    q_max = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    fan = FanCurve(coeffs=coeffs, q_max=q_max, learnable=True)

    fan.dflow(torch.tensor([500.0], dtype=torch.float64)).sum().backward()

    assert torch.isfinite(fan.coeffs.grad).all()
    assert torch.isfinite(fan.q_max.grad).all()


def test_dflow_gradients_are_finite_at_a_zero_slope_shut_boundary():
    # P(q) = 100 - q^2, P'(q) = -2q, P'(0) = 0: the curve flattens exactly at its shutoff
    # point (q = 0), the other legitimate zero-slope boundary. dp = -200 gives -dp = 200 >
    # P(0) = 100, so this is deep in the shut region.
    coeffs = torch.tensor([100.0, 0.0, -1.0, 0.0], dtype=torch.float64, requires_grad=True)
    q_max = torch.tensor(5.0, dtype=torch.float64)
    fan = FanCurve(coeffs=coeffs, q_max=q_max, learnable=True)

    fan.dflow(torch.tensor([-200.0], dtype=torch.float64)).sum().backward()

    assert torch.isfinite(fan.coeffs.grad).all()


F64 = torch.float64
RHO_293 = 101325.0 / (287.055 * 293.15)
CURVE = dict(coeffs=torch.tensor([200.0, -100.0, -200.0, -800.0], dtype=F64),
             q_max=torch.tensor(0.5, dtype=F64))


def _fan(**kw):
    return AffinityFan(FanCurve(**CURVE, kind="fan"), src=[0], tgt=[1], rho_ref=1.2041,
                       kind="fan", **kw)


@pytest.mark.parametrize("speed, T, dp, contamx", [
    (1.0, 293.15, -23.571704864501953, 0.4726639688014984),
    (0.8, 293.15, -15.085563659667969, 0.3781316578388214),
    (1.2, 293.15, -33.94355773925781, 0.5671964883804321),
    (1.0, 313.15, -20.800249099731445, 0.4440062642097473),
])
def test_the_fan_laws_reproduce_contamx(speed, T, dp, contamx):
    """ContamX 3.4.1.7: F = s r F0(-dp / (s^2 r)), r = rho_up / 1.2041. Budget: the four
    float32 coefficients and the rating density move F by at most 2 u, the temperature 1 u
    through r, and the float32 pressure 1 u: 4 u (2.4e-7)."""
    fan = _fan(speed_key="speed")
    rho = 101325.0 / (287.055 * torch.tensor([T, 293.15], dtype=F64))
    q = fan.flow(torch.tensor([dp], dtype=F64),
                 {"rho": rho, "speed": torch.tensor([speed], dtype=F64)})
    assert q.item() == pytest.approx(contamx, rel=4 * 2.0**-24)


def test_a_multiplier_is_fans_in_parallel():
    """ContamX, path mult 2: 0.8186134 kg/s at dp = -70.70461 Pa (two fans each at P = 70.7)."""
    fan = _fan(multiplier=torch.tensor([2.0], dtype=F64))
    q = fan.flow(torch.tensor([-70.70461], dtype=F64),
                 {"rho": torch.full((2,), RHO_293, dtype=F64)})
    assert q.item() == pytest.approx(0.8186134, rel=2e-7)


def test_below_its_off_ratio_the_fan_is_its_shutoff_orifice():
    """ContamX: speed 0.05 < off 0.1 with mult 1.5: 0.005058304 kg/s at dp = 0.1356435 Pa
    from a 303.15 K zone = 1.5 * 0.0084853 * sqrt(rho_up) * sqrt(dp)."""
    rho_up = 101325.0 / (287.055 * 303.15)
    C = 0.0084853 * math.sqrt(1.2041)
    shut = UpstreamDensityPowerLaw(torch.tensor(C, dtype=F64), torch.tensor(0.5, dtype=F64),
                                   src=[0], tgt=[1], m=0.5, rho_ref=1.2041, kind="fan")
    fan = _fan(speed_key="speed", off=0.1, multiplier=torch.tensor([1.5], dtype=F64),
               shutoff=shut)
    d = {"rho": torch.tensor([rho_up, RHO_293], dtype=F64),
         "speed": torch.tensor([0.05], dtype=F64)}
    q = fan.flow(torch.tensor([0.1356435], dtype=F64), d)
    assert q.item() == pytest.approx(0.005058304, rel=1e-6)
    assert _fan(speed_key="speed", off=0.1).flow(torch.tensor([0.1], dtype=F64), d).item() == 0


def test_the_fan_slope_follows_the_affinity_laws_and_batches():
    fan = _fan(speed_key="speed")
    rho = torch.full((2,), RHO_293, dtype=F64)
    dp = torch.tensor([[-20.0], [-20.0]], dtype=F64)
    d = {"rho": rho, "speed": torch.tensor([[1.0], [0.8]], dtype=F64)}
    q = fan.flow(dp, d)
    assert q.shape == (2, 1)
    curve = FanCurve(**CURVE, kind="fan")
    r = RHO_293 / 1.2041
    expected = 0.8 * r * curve.flow(torch.tensor([-20.0 / (0.64 * r)], dtype=F64))
    assert q[1].item() == pytest.approx(expected.item(), rel=1e-12)
    slope = fan.dflow(dp, d)
    h = 1e-5
    fd = (fan.flow(dp + h, d) - fan.flow(dp - h, d)) / (2 * h)
    torch.testing.assert_close(slope, fd, rtol=1e-6, atol=0)


def _shutoff_fan(off):
    C = 0.0084853 * math.sqrt(1.2041)
    shut = UpstreamDensityPowerLaw(torch.tensor(C, dtype=F64), torch.tensor(0.5, dtype=F64),
                                   src=[0], tgt=[1], m=0.5, rho_ref=1.2041, kind="fan")
    return _fan(speed_key="speed", off=off, multiplier=torch.tensor([1.5], dtype=F64),
                shutoff=shut), shut


def _finite_everywhere(fan, speed, dp):
    """flow, dflow and the gradients of flow with respect to dp and the speed are finite."""
    s = torch.tensor([speed], dtype=F64, requires_grad=True)
    x = torch.tensor([dp], dtype=F64, requires_grad=True)
    d = {"rho": torch.full((2,), RHO_293, dtype=F64), "speed": s}
    q = fan.flow(x, d)
    g_dp, g_s = torch.autograd.grad(q.sum(), (x, s))
    slope = fan.dflow(x, d)
    (g_slope,) = torch.autograd.grad(slope.sum(), (s,), allow_unused=True)
    for v in (q, slope, g_dp, g_s) + ((g_slope,) if g_slope is not None else ()):
        assert torch.isfinite(v).all(), (speed, dp, v)
    return q.detach(), slope.detach()


@pytest.mark.parametrize("off", [0.0, 0.1])
@pytest.mark.parametrize("dp", [-20.0, 0.0, 0.5])
def test_a_fan_at_zero_speed_is_stopped_and_finite(off, dp):
    """s = 0 divides the fan laws by zero: it is stopped whatever `off` is, and passes its
    shut-off orifice (times the multiplier)."""
    fan, shut = _shutoff_fan(off)
    q, slope = _finite_everywhere(fan, 0.0, dp)
    d = {"rho": torch.full((2,), RHO_293, dtype=F64)}
    x = torch.tensor([dp], dtype=F64)
    torch.testing.assert_close(q, 1.5 * shut.flow(x, d), rtol=1e-14, atol=0)
    torch.testing.assert_close(slope, 1.5 * shut.dflow(x, d), rtol=1e-14, atol=0)


@pytest.mark.parametrize("speed", [0.1 - 1e-9, 0.1, 0.1 + 1e-9, 0.05, 1.0])
def test_around_its_off_ratio_the_fan_and_its_gradients_stay_finite(speed):
    """The flow jumps at the off ratio (running fan to shut-off orifice, as in ContamX);
    on either side, and at a stopped edge, flow, slope and gradients are finite."""
    fan, _ = _shutoff_fan(0.1)
    q, _ = _finite_everywhere(fan, speed, -0.5)
    assert (q.item() > 0) == (speed >= 0.1)     # running: pushes; stopped: the orifice leaks back


def test_a_negative_speed_is_refused_naming_the_fan_and_the_value():
    fan, _ = _shutoff_fan(0.1)
    d = {"rho": torch.full((2,), RHO_293, dtype=F64), "speed": torch.tensor([-0.5], dtype=F64)}
    with pytest.raises(ValueError, match=r"kind 'fan'.*'speed'.*negative speed ratio -0\.5"):
        fan.flow(torch.tensor([-20.0], dtype=F64), d)
    with pytest.raises(ValueError, match=r"negative speed ratio"):
        fan.dflow(torch.tensor([-20.0], dtype=F64), d)
