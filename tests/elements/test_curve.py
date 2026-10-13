"""Measured airflow curves fitted as ContamX 3.4.1.7 fits them (probes c6a-c6c)."""

from __future__ import annotations

import pytest
import torch

from noodl.elements import CubicSplineFlow, SplineCurve

F64 = torch.float64
X = [0.0, 2.0, 5.0, 10.0, 20.0]
Y = [0.0, 0.05, 0.09, 0.14, 0.2]
RHO = 101325.0 / (287.055 * torch.tensor([293.15, 313.15], dtype=F64))


def test_the_fit_is_contamx_s_clamped_spline_between_straight_end_segments():
    """ContamX prints the fit: slope 0.025 (the first segment's) at 2 Pa, 9.1875e-3 at
    5 Pa, 0.006 (the last segment's) at 10 Pa."""
    c = SplineCurve(X, Y)
    for u in (0.0, 2.0, 5.0, 10.0, 20.0):
        assert c(u).item() == pytest.approx(Y[X.index(u)], abs=1e-15)
    h = 1e-6
    for u, slope in ((2.0, 0.025), (5.0, 9.1875e-3), (10.0, 0.006)):
        assert ((c(u + h) - c(u - h)) / (2 * h)).item() == pytest.approx(slope, rel=1e-6)


@pytest.mark.parametrize("dp, contamx", [
    (0.03999975, 0.0009999938), (3.0384, 0.0699992), (5.988952, 0.1000001),
    (8.737103, 0.1299993), (15.0, 0.17),
])
def test_inside_its_points_the_curve_matches_contamx(dp, contamx):
    assert SplineCurve(X, Y)(dp).item() == pytest.approx(contamx, rel=2e-6)


def test_outside_its_points_the_curve_continues_its_end_segments():
    """ContamX: 0.2499959 kg/s at 28.33265 Pa (the last segment, slope 0.006, extended)."""
    assert SplineCurve(X, Y)(28.33265).item() == pytest.approx(0.2499959, rel=2e-6)


def test_a_negative_argument_extends_the_first_segment_not_a_mirror():
    """ContamX: -0.1199992 kg/s at -4.79997 Pa: slope 0.025 continued, not -F(4.8)."""
    c = SplineCurve(X, Y)
    assert c(-4.79997).item() == pytest.approx(-0.1199992, rel=2e-6)
    assert c(-4.79997).item() != pytest.approx(-c(4.79997).item(), rel=1e-3)


def test_the_inverse_undoes_the_curve_and_carries_gradients():
    c = SplineCurve(X, Y)
    v = torch.tensor([-0.02, 0.03, 0.07, 0.12, 0.3], dtype=F64, requires_grad=True)
    u = c.inverse(v)
    torch.testing.assert_close(c(u), v.detach(), rtol=1e-12, atol=1e-15)
    (g,) = torch.autograd.grad(u.sum(), v)
    assert torch.isfinite(g).all() and bool((g > 0).all())


def test_the_four_element_forms():
    """ContamX (c6b): qsp F = rho_up Q(dp); psf dp = P(F); psq dp = P(F / rho_up)."""
    kw = dict(src=[0], tgt=[1])
    d = {"rho": RHO}
    fsp = CubicSplineFlow(X, Y, **kw)
    qsp = CubicSplineFlow(X, Y, volume=True, **kw)
    psf = CubicSplineFlow(Y, X, independent="flow", **kw)
    psq = CubicSplineFlow(Y, X, independent="flow", volume=True, **kw)
    dp = torch.tensor([8.0], dtype=F64)
    assert qsp.flow(dp, d).item() == pytest.approx(RHO[0].item() * fsp.flow(dp, d).item())
    assert qsp.flow(-dp, d).item() == pytest.approx(RHO[1].item() * fsp.flow(-dp, d).item())
    assert psf.flow(dp, d).item() == pytest.approx(SplineCurve(Y, X).inverse(8.0).item())
    assert psq.flow(dp, d).item() == pytest.approx(
        RHO[0].item() * SplineCurve(Y, X).inverse(8.0).item())
    mult = CubicSplineFlow(X, Y, multiplier=torch.tensor([3.0], dtype=F64), **kw)
    assert mult.flow(dp, d).item() == pytest.approx(3 * fsp.flow(dp, d).item())


def test_curves_contamx_would_refuse_are_refused():
    with pytest.raises(ValueError, match="at least four"):
        CubicSplineFlow([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], src=[0], tgt=[1], kind="sp_4")
    with pytest.raises(ValueError, match=r"sp_4.*slope.*positive"):
        CubicSplineFlow([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 1.0, 2.0], src=[0], tgt=[1],
                        kind="sp_4")
    with pytest.raises(ValueError, match="increasing"):
        SplineCurve([0.0, 2.0, 1.0], [0.0, 1.0, 2.0])
