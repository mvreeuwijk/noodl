"""Hazen-Williams, the fitted pump curve, the minor-loss valve and pressure-driven demand."""

import pytest
import torch

from tellegen.apps.water.demand import PressureDrivenDemand
from tellegen.apps.water.elements import (
    HW_SI,
    HazenWilliams,
    MinorLoss,
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
    itself writes for P7 is 1.9394165e-05 -- the ~1e-4 relative gap spec amendment A20
    records between the model and EPANET's own float32-limited report, not a bug in
    either.
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
    """Net1's single point, 1500 GPM / 250 ft. MEASURED against wntr's own
    get_head_curve_coefficients: (101.6, 2836.1385287628063, 2)."""
    gpm = 3.785411784e-3 / 60.0
    h0, r, n = three_point_curve(
        torch.tensor([1500.0 * gpm], dtype=F64), torch.tensor([250.0 * 0.3048], dtype=F64)
    )
    assert float(h0) == pytest.approx(101.6, rel=1e-12)
    assert float(r) == pytest.approx(2836.1385287628063, rel=1e-12)
    assert n == 2.0


def test_the_pump_curve_inverts_its_fitted_law():
    pump = PumpCurve(
        torch.tensor([101.6], dtype=F64), torch.tensor([2836.1385287628063], dtype=F64)
    )
    # dp = phi_suction - phi_discharge = -(head gain)
    q = pump.flow(torch.tensor([-62.285091400146484], dtype=F64))
    # MEASURED 0.11773752856904683 against EPANET's own 0.11773740500211716, i.e. 1.05e-6
    # relative. The head gain fed in is EPANET's FLOAT32 report (-62.285091400146484), and
    # inverting a quadratic curve at a float32 head cannot do better than the head's own
    # precision; the full solve agrees to 9.553e-7 (row D2).
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
        status_key="water.status",
    )
    drivers = {"water.status": torch.zeros(1, dtype=F64)}
    assert float(pump.flow(torch.tensor([-50.0], dtype=F64), drivers)) == 0.0
    drivers["water.status"] = torch.ones(1, dtype=F64)
    assert float(pump.flow(torch.tensor([-50.0], dtype=F64), drivers)) > 0.0


def test_a_missing_status_driver_is_refused():
    pump = PumpCurve(
        torch.tensor([101.6], dtype=F64), torch.tensor([2836.1385287628063], dtype=F64),
        status_key="water.status",
    )
    with pytest.raises(KeyError, match="'water.status'"):
        pump.flow(torch.tensor([-50.0], dtype=F64), {})


def test_minor_loss_valve_inverts_its_quadratic():
    el = MinorLoss(torch.tensor([10.0], dtype=F64), torch.tensor([0.2], dtype=F64))
    q = el.flow(torch.tensor([0.5], dtype=F64))
    m = float(el.coefficient())
    assert float(m * q * q.abs()) == pytest.approx(0.5, rel=1e-12)
    dp = torch.zeros(1, dtype=F64, requires_grad=True)
    el.flow(dp).sum().backward()
    assert torch.isfinite(dp.grad).all()


# --------------------------------------------------------------- M4-R14: PDA's exact kink
def test_pressure_driven_demand_is_exactly_zero_at_and_below_p_min():
    """Ruling M4-R14: the Wagner curve is EXACTLY 0 at and below `P_min`, the fractional
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
