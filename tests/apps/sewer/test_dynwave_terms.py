import math

import pytest
import torch

from noodl.apps.sewer import dynwave_terms as dt_
from noodl.apps.sewer import xsect_tables as xt

F64 = torch.float64


def t(x):
    return torch.as_tensor(x, dtype=F64)


def geo(offset1=0.0, offset2=0.0, outfall1=False, outfall2=False):
    return dt_.ConduitGeometry.from_si(
        diameter=t([0.45]), roughness=t([0.013]), slope=t([0.005]), length=t([200.0]),
        invert1=t([10.0]), invert2=t([9.0]), offset1=t([offset1]), offset2=t([offset2]),
        outfall1=t([outfall1]).bool(), outfall2=t([outfall2]).bool())


def test_dry_conduit_carries_nothing_and_gives_fudge_areas():
    g = geo()
    r = dt_.conduit_terms(g, t([0.0]), t([0.0]), t([0.0]), t([0.0]),
                          xt.area_of_depth(t([dt_.FUDGE]), g.y_full, g.a_full), 5.0)
    assert r.flow_class.item() == dt_.DRY
    assert r.flow_offset.item() == 0.0 and r.conveyance.item() == 0.0
    assert r.momentum_flow.item() == 0.0
    assert r.surface_area1.item() == pytest.approx(dt_.FUDGE * g.length.item() / 2)
    assert r.surface_area2.item() == pytest.approx(dt_.FUDGE * g.length.item() / 2)
    # dwflow.c:171 -- on the dry exit SWMM stores the end-mean area, not A(yMid)
    a_end = xt.area_of_depth(t([dt_.FUDGE]), g.y_full, g.a_full)
    torch.testing.assert_close(r.mid_area, a_end)
    torch.testing.assert_close(r.volume, a_end * g.length)


def _weights(g, y1, y2, q_last):
    """(a1, a2, a_mid, r1, r_mid, v, sigma_fr) as dwflow.c:142-194 computes them."""
    y_mid = 0.5 * (y1 + y2)
    a1 = xt.area_of_depth(t([y1]), g.y_full, g.a_full)
    a2 = xt.area_of_depth(t([y2]), g.y_full, g.a_full)
    am = xt.area_of_depth(t([y_mid]), g.y_full, g.a_full)
    rm = xt.radius_of_depth(t([y_mid]), g.y_full)
    r1 = xt.radius_of_depth(t([y1]), g.y_full)
    v = q_last / am
    fr = xt.froude_number(v, t([y_mid]), g.y_full, g.a_full)
    sigma = torch.where(fr <= 0.5, t(1.0), torch.where(fr >= 1, t(0.0), 2 * (1 - fr)))
    return a1, a2, am, r1, rm, v, sigma


def test_wet_subcritical_coefficients_follow_dwflow():
    g = geo()
    y1, y2, q_last, q_old = 0.6, 0.5, 2.0, 1.8      # ft, cfs
    a_old = xt.area_of_depth(t([0.55]), g.y_full, g.a_full)
    r = dt_.conduit_terms(g, t([y1]), t([y2]), t([q_last]), t([q_old]), a_old, 5.0)
    assert r.flow_class.item() == dt_.SUBCRITICAL
    a1, a2, am, r1, rm, v, sigma = _weights(g, y1, y2, q_last)
    h1 = 10.0 / xt.LCF + y1
    h2 = 9.0 / xt.LCF + y2
    rho = sigma if h1 >= h2 else t(1.0)               # q_last > 0, not full
    aw = a1 + (am - a1) * rho
    rw = r1 + (rm - r1) * rho
    dq1 = 5.0 * g.rough_factor / rw**1.33333 * abs(v)
    dq3 = 2 * v * (am - a_old) * sigma
    dq4 = 5.0 * v * v * (a2 - a1) / g.length * sigma
    denom = 1 + dq1
    torch.testing.assert_close(r.conveyance, 32.2 * 5.0 * aw / (g.length * denom))
    torch.testing.assert_close(r.flow_offset, (q_old + dq3 + dq4) / denom)
    torch.testing.assert_close(r.momentum_flow, r.flow_offset + r.conveyance * (h1 - h2))
    # dwflow.c:284-291 -- stored mid area, link depth and volume
    torch.testing.assert_close(r.mid_area, am)
    torch.testing.assert_close(r.volume, 0.5 * (a1 + a2) * g.length)
    assert r.depth.item() == pytest.approx(0.55)


def test_damping_full_drops_the_inertial_terms():
    """Under FULL damping (sigma = 0) the offset is c0 = q_old/denom exactly, with
    denom = 1 + dq1 = g dt a_wtd / (L k); under NONE the inertial terms dq3, dq4 are
    nonzero here (a_mid != a_old and a2 != a1), so c0 differs."""
    g = geo()
    y1, y2, q_last, q_old = 0.6, 0.5, 2.0, 1.0
    a_old = xt.area_of_depth(t([0.3]), g.y_full, g.a_full)
    args = (g, t([y1]), t([y2]), t([q_last]), t([q_old]), a_old, 5.0)
    full = dt_.conduit_terms(*args, damping="full")
    none = dt_.conduit_terms(*args, damping="none")
    a1, _, am, _, _, _, sigma = _weights(g, y1, y2, q_last)
    a_wtd = a1 + (am - a1) * sigma                   # q_last > 0, h1 >= h2, not full
    denom = 32.2 * 5.0 * a_wtd / (g.length * full.conveyance)
    torch.testing.assert_close(full.flow_offset, q_old / denom)
    assert none.flow_offset.item() != pytest.approx(full.flow_offset.item())
    # damping changes only the inertial terms, not the conveyance
    torch.testing.assert_close(full.conveyance, none.conveyance)


def test_upstream_dry_end_pins_the_head_and_zeroes_conveyance():
    """Node1 dry, node2 wet above conduit end 1: class UP_CRITICAL needs offset1 > 0;
    with zero offsets and h2 below z1 the class is UP_DRY."""
    g = geo()
    r = dt_.conduit_terms(g, t([0.0]), t([0.3]), t([0.0]), t([0.0]),
                          xt.area_of_depth(t([dt_.FUDGE]), g.y_full, g.a_full), 5.0)
    assert r.flow_class.item() == dt_.UP_DRY
    assert r.dry_upstream.item() and not r.dry_downstream.item()
    assert r.conveyance.item() == 0.0 and r.flow_offset.item() == 0.0


def test_drop_inlet_downstream_critical():
    """A 0.3 m outlet offset over a dry downstream node: y2 clamps to FUDGE (downstream
    end dry), the upstream head (10 m invert + 0.8 ft) is far above the offset invert
    (9.3 m), so not DN_DRY, and z2 > 0 gives DN_CRITICAL (dwflow.c:297-413). The
    downstream end head is then the critical-depth head, independent of node 2."""
    g = geo(offset2=0.3)
    r = dt_.conduit_terms(g, t([0.8]), t([0.0]), t([1.0]), t([1.0]),
                          xt.area_of_depth(t([0.4]), g.y_full, g.a_full), 5.0)
    assert r.flow_class.item() == dt_.DN_CRITICAL
    assert r.surface_area2.item() == 0.0
    assert r.conveyance.item() == 0.0               # pinned end: k folded into c0 (D4)
    assert r.flow_offset.item() == pytest.approx(r.momentum_flow.item())
    # dwflow.c:490-493 -- end depth 2 is min(yN, yC) at |q_last|
    yn = xt.normal_depth_ft(t([1.0]), g.y_full, g.a_full, g.s_full, g.beta)
    yc = xt.critical_depth(t([1.0]), g.y_full, g.a_full)
    torch.testing.assert_close(r.end_depth2, torch.minimum(yn, yc))


def test_upstream_critical_on_reverse_flow_with_an_inlet_offset():
    """Both ends wet, q_last < 0, z1 > 0 and y1 below min(yN, yC): UP_CRITICAL
    (dwflow.c:337-348); node 1 gets no surface area and end 1 is pinned."""
    g = geo(offset1=0.2)
    y1_ft = 0.05                                   # above the offset invert
    d1 = 0.2 / xt.LCF + y1_ft
    a_old = xt.area_of_depth(t([0.3]), g.y_full, g.a_full)
    r = dt_.conduit_terms(g, t([d1]), t([1.0 + 3.3]), t([-1.0]), t([-1.0]), a_old, 5.0)
    assert r.flow_class.item() == dt_.UP_CRITICAL
    assert r.surface_area1.item() == 0.0 and r.surface_area2.item() > 0.0
    assert r.conveyance.item() == 0.0
    assert r.flow_offset.item() == pytest.approx(r.momentum_flow.item())


def test_fasnh_scales_the_downstream_surface_area():
    """Both ends wet, q_last >= 0, z2 > 0, y2 between min and max of (yN, yC): the
    downstream surface area is scaled by fasnh = (ycMax - y2)/(ycMax - ycMin)
    (dwflow.c:357-368, 474)."""
    g = geo(offset2=0.1)
    q = t([2.0])
    yn = xt.normal_depth_ft(q, g.y_full, g.a_full, g.s_full, g.beta).item()
    yc = xt.critical_depth(q, g.y_full, g.a_full).item()
    lo, hi = min(yn, yc), max(yn, yc)
    y2 = 0.5 * (lo + hi)
    y1 = 0.6
    d2 = 0.1 / xt.LCF + y2
    a_old = xt.area_of_depth(t([0.5]), g.y_full, g.a_full)
    r = dt_.conduit_terms(g, t([y1]), t([d2]), q, q, a_old, 5.0)
    assert r.flow_class.item() in (dt_.SUBCRITICAL, dt_.SUPCRITICAL)
    # the end depth seen by SWMM is h2 - z2, which differs from y2 in the last bits
    # (fasnh divides by the small yN-yC span, so use the depth the function saw)
    y2 = r.end_depth2.item()
    assert y2 == pytest.approx(0.5 * (lo + hi), rel=1e-12)
    fasnh = (hi - y2) / (hi - lo)
    ym = 0.5 * (y1 + y2)
    w = [xt.width_of_depth(t([y]), g.y_full).item() for y in (y1, ym, y2)]
    L = g.length.item()
    assert r.surface_area1.item() == pytest.approx((w[0] + w[1]) * L / 4, rel=1e-12)
    assert r.surface_area2.item() == pytest.approx((w[1] + w[2]) * L / 4 * fasnh, rel=1e-12)


def test_outfall_offset_is_measured_from_the_outfall_depth():
    """getFlowClass bases an outfall conduit's offset on the outfall depth
    (dwflow.c:327-328): a submerged outlet offset is no drop, so no DN_CRITICAL."""
    q = t([1.0])
    a_old = xt.area_of_depth(t([0.4]), t([0.45 / xt.LCF]), t([math.pi / 4 * (0.45 / 0.3048)**2]))
    d2 = t([0.3 / xt.LCF + 0.2])                   # outfall water 0.2 ft above the offset
    plain = dt_.conduit_terms(geo(offset2=0.3), t([0.8]), t([0.0]), q, q, a_old, 5.0)
    out = dt_.conduit_terms(geo(offset2=0.3, outfall2=True), t([0.8]), d2, q, q, a_old, 5.0)
    assert plain.flow_class.item() == dt_.DN_CRITICAL
    assert out.flow_class.item() in (dt_.SUBCRITICAL, dt_.SUPCRITICAL)


def test_full_conduit_is_subcritical_with_no_inertia():
    """Both ends at or above the crown: class SUBCRITICAL (dwflow.c:457-460), sigma = 0
    for a closed conduit (dwflow.c:208), rho = 1, and the widths use the 0.96 crown
    cutoff (dwflow.c:602-603)."""
    g = geo()
    yf = g.y_full.item()
    a_old = xt.area_of_depth(t([0.5]), g.y_full, g.a_full)
    r = dt_.conduit_terms(g, t([yf + 2.0]), t([yf + 0.5]), t([3.0]), t([3.0]), a_old, 5.0)
    assert r.flow_class.item() == dt_.SUBCRITICAL
    torch.testing.assert_close(r.end_depth1, g.y_full)
    rf = 0.25 * g.y_full
    v = 3.0 / g.a_full
    denom = 1 + 5.0 * g.rough_factor / rf**1.33333 * v
    torch.testing.assert_close(r.flow_offset, 3.0 / denom)
    torch.testing.assert_close(r.conveyance, 32.2 * 5.0 * g.a_full / (g.length * denom))
    w = xt.width_of_depth(dt_.EXTRAN_CROWN_CUTOFF * g.y_full, g.y_full)
    torch.testing.assert_close(r.surface_area1, w * g.length / 2)
    assert not r.limited.item()                     # y1 >= yFull: no normal-flow check


def test_normal_flow_limit_slope_test_is_y1_below_y2():
    g = geo()
    a_old = xt.area_of_depth(t([0.5]), g.y_full, g.a_full)
    lim = dt_.conduit_terms(g, t([0.4]), t([0.6]), t([3.0]), t([3.0]), a_old, 5.0,
                            normal_flow_limited="slope")
    off = dt_.conduit_terms(g, t([0.4]), t([0.6]), t([3.0]), t([3.0]), a_old, 5.0,
                            normal_flow_limited="none")
    assert lim.limited.item() and not off.limited.item()
    a1 = xt.area_of_depth(t([0.4]), g.y_full, g.a_full)
    r1 = xt.radius_of_depth(t([0.4]), g.y_full)
    assert lim.normal_flow.item() == pytest.approx((g.beta * a1 * r1 ** (2 / 3)).item())
    assert math.isinf(off.normal_flow.item())


def test_normal_flow_froude_test_uses_the_momentum_flow():
    """FROUDE mode with y1 > y2: limited iff Fr(q/a1, y1) >= 1 on the momentum flow
    (dwflow.c:665-672); an outfall end disables the Froude test (dwflow.c:666)."""
    g = geo()
    a_old = xt.area_of_depth(t([0.3]), g.y_full, g.a_full)
    args = (t([0.3]), t([0.2]), t([2.0]), t([2.0]), a_old, 5.0)
    r = dt_.conduit_terms(g, *args, normal_flow_limited="froude")
    a1 = xt.area_of_depth(t([0.3]), g.y_full, g.a_full)
    f1 = xt.froude_number(r.momentum_flow / a1, t([0.3]), g.y_full, g.a_full)
    assert r.limited.item() == bool(f1.item() >= 1.0)
    assert f1.item() >= 1.0                         # steep, shallow: the test fires
    out = dt_.conduit_terms(geo(outfall2=True), *args, normal_flow_limited="froude")
    assert not out.limited.item()


def test_velocity_is_clamped_at_50_fps():
    g = geo()
    a_old = xt.area_of_depth(t([0.05]), g.y_full, g.a_full)
    r = dt_.conduit_terms(g, t([0.05]), t([0.05]), t([1e3]), t([0.0]), a_old, 5.0)
    assert torch.isfinite(r.flow_offset).all()
    a_mid = xt.area_of_depth(t([0.05]), g.y_full, g.a_full)
    r_mid = xt.radius_of_depth(t([0.05]), g.y_full)
    denom = 1 + 5.0 * g.rough_factor / r_mid**1.33333 * dt_.MAX_VELOCITY
    torch.testing.assert_close(r.conveyance, 32.2 * 5.0 * a_mid / (g.length * denom))


def test_terms_broadcast_over_a_batch():
    g = geo()
    a_old = xt.area_of_depth(t([0.5]), g.y_full, g.a_full)
    r = dt_.conduit_terms(g, t([[0.6], [0.4]]), t([[0.5], [0.5]]), t([[2.0], [2.0]]),
                          t([[1.8], [1.8]]), a_old, 5.0)
    assert r.flow_offset.shape == (2, 1)
    assert r.flow_class.shape == (2, 1) and r.flow_class.dtype == torch.long


def test_unknown_options_are_refused():
    g = geo()
    args = (g, t([0.5]), t([0.5]), t([1.0]), t([1.0]), t([0.1]), 5.0)
    with pytest.raises(ValueError, match="damping"):
        dt_.conduit_terms(*args, damping="some")
    with pytest.raises(ValueError, match="normal_flow_limited"):
        dt_.conduit_terms(*args, normal_flow_limited="always")


def test_gradients_are_finite_on_every_branch():
    """Both-branches-safe: wet, dry, pinned and critical conduits in one batch
    back-propagate finite gradients to the node depths."""
    g = geo(offset2=0.3)
    d1 = t([[0.8], [0.0], [0.6], [0.05]]).requires_grad_(True)
    d2 = t([[0.0], [0.0], [1.5], [0.04]]).requires_grad_(True)
    q = t([[1.0], [0.0], [2.0], [1e3]])
    a_old = xt.area_of_depth(t([0.4]), g.y_full, g.a_full)
    r = dt_.conduit_terms(g, d1, d2, q, q, a_old, 5.0)
    (r.momentum_flow.sum() + r.flow_offset.sum() + r.conveyance.sum()
     + r.surface_area1.sum() + r.surface_area2.sum()).backward()
    assert torch.isfinite(d1.grad).all() and torch.isfinite(d2.grad).all()
