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
    assert r.conveyance.item() == 0.0               # pinned end: k folded into c0
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


# ---- Preissmann slot (SURCHARGE_METHOD SLOT, dwflow.c:575-633) ----------------------------


def _slot_w(y_norm, w_max):
    """The closed-form slot width of dwflow.c:575-588 for a closed conduit."""
    if y_norm < 0.985257:
        return 0.0
    if y_norm > 1.78:
        return 0.01 * w_max
    return w_max * 0.5423 * math.exp(-(y_norm ** 2.4))


@pytest.mark.parametrize("y_norm", [0.985257, 1.0, 1.5])
def test_slot_width_follows_the_sjoberg_formula(y_norm):
    y_full = t([1.5])
    w = dt_._slot_width(y_norm * y_full, y_full, True)
    assert w.item() == pytest.approx(_slot_w(y_norm, 1.5), rel=1e-12, abs=0.0)
    assert dt_._slot_width(y_norm * y_full, y_full, False).item() == 0.0   # EXTRAN: none


def test_slot_width_values_at_the_landmarks():
    y_full = t([1.0])
    for y_norm, expect in ((0.98, 0.0), (1.0, 0.5423 * math.exp(-1.0)), (2.0, 0.01)):
        assert dt_._slot_width(t([y_norm]), y_full, True).item() == pytest.approx(
            expect, rel=1e-12, abs=1e-15)
    # near-continuity with the 1 % plateau at 1.78 y_full (dwflow.c:584): the formula gives
    # 0.010029 there
    assert dt_._slot_width(t([1.78]), y_full, True).item() == pytest.approx(0.01, abs=1e-4)


def test_area_above_the_crown_extends_by_the_slot_width():
    g = geo()
    y = g.y_full * 1.3
    a = dt_._slot_area(y, g.y_full, g.a_full, True)
    w = _slot_w(1.3, g.y_full.item())
    assert a.item() == pytest.approx(g.a_full.item() + 0.3 * g.y_full.item() * w, rel=1e-12)
    # EXTRAN: the area is capped at a_full; below the crown the section area is unchanged
    assert dt_._slot_area(y, g.y_full, g.a_full, False).item() == pytest.approx(
        g.a_full.item())
    y_low = g.y_full * 0.7
    torch.testing.assert_close(dt_._slot_area(y_low, g.y_full, g.a_full, True),
                               xt.area_of_depth(y_low, g.y_full, g.a_full))
    # the friction radius stays r_full above the crown (dwflow.c:631)
    assert dt_._hyd_rad(y, g.y_full).item() == pytest.approx(0.25 * g.y_full.item())


def test_top_width_uses_the_slot_above_the_cutoff_and_the_section_below():
    g = geo()
    y_full = g.y_full
    below = 0.9 * y_full
    torch.testing.assert_close(dt_._top_width(below, y_full, True),
                               xt.width_of_depth(below, y_full))
    above = 1.2 * y_full
    assert dt_._top_width(above, y_full, True).item() == pytest.approx(
        _slot_w(1.2, y_full.item()), rel=1e-12)
    # EXTRAN: frozen at W(0.96 yFull) instead
    frozen = xt.width_of_depth(0.96 * y_full, y_full)
    torch.testing.assert_close(dt_._top_width(above, y_full, False), frozen)


def test_slot_keeps_end_depths_above_the_crown_and_extends_the_volume():
    g = geo()
    yf = g.y_full.item()
    # 0.45 m diameter, node depths 0.55 m and 0.65 m above the inverts
    d_ft1, d_ft2 = 0.55 / xt.LCF, 0.65 / xt.LCF
    a_old = t([g.a_full.item()])
    args = (g, t([d_ft1]), t([d_ft2]), t([0.0]), t([0.0]), a_old, 5.0)
    extran = dt_.conduit_terms(*args)
    slot = dt_.conduit_terms(*args, surcharge_method="slot")
    assert (extran.end_depth1.item(), extran.end_depth2.item()) == (yf, yf)
    assert slot.end_depth1.item() == pytest.approx(d_ft1)
    assert slot.end_depth2.item() == pytest.approx(d_ft2)
    assert slot.end_depth1.item() > yf
    # link depth stays at the crown; stored volume exceeds the full volume
    assert slot.depth.item() == pytest.approx(yf)
    a1 = dt_._slot_area(t([d_ft1]), g.y_full, g.a_full, True)
    a2 = dt_._slot_area(t([d_ft2]), g.y_full, g.a_full, True)
    assert slot.volume.item() == pytest.approx(((a1 + a2) / 2 * g.length).item())
    assert slot.volume.item() > (g.a_full * g.length).item()
    assert extran.volume.item() == pytest.approx((g.a_full * g.length).item())


def test_a_full_conduit_under_slot_has_no_inertial_terms():
    """Both ends above the crown: sigma = 0 (dwflow.c:208), so q = (q_old - dq2)/(1 + dq1)
    with the slot-extended mid area and the full-pipe radius."""
    g = geo()
    yf = g.y_full.item()
    d1, d2 = 0.6 / xt.LCF, 0.5 / xt.LCF
    q_last, q_old, dt = 0.01 / xt.QCF, 0.012 / xt.QCF, 5.0
    a_old = t([0.8 * g.a_full.item()])         # a_mid != a_old: dq3 would be nonzero
    r = dt_.conduit_terms(g, t([d1]), t([d2]), t([q_last]), t([q_old]), a_old, dt,
                          surcharge_method="slot")
    y_mid = 0.5 * (d1 + d2)
    a_mid = g.a_full.item() + (y_mid - yf) * _slot_w(y_mid / yf, yf)
    v = q_last / a_mid
    dq1 = dt * g.rough_factor.item() / (0.25 * yf) ** 1.33333 * abs(v)
    h1 = d1 + g.invert1.item()
    h2 = d2 + g.invert2.item()
    dq2 = dt * xt.GRAVITY_FT * a_mid * (h2 - h1) / g.length.item()
    expected = (q_old - dq2) / (1.0 + dq1)
    assert r.momentum_flow.item() == pytest.approx(expected, rel=1e-12)
    assert r.flow_class.item() == dt_.SUBCRITICAL


def test_surface_area_uses_the_slot_width_above_the_cutoff():
    g = geo()
    yf = g.y_full.item()
    d1, d2 = 1.2 * yf, 1.4 * yf
    args = (g, t([d1]), t([d2]), t([0.0]), t([0.0]), t([g.a_full.item()]), 5.0)
    r = dt_.conduit_terms(*args, surcharge_method="slot")
    w1, w2, wm = _slot_w(1.2, yf), _slot_w(1.4, yf), _slot_w(1.3, yf)
    length = g.length.item()
    assert r.surface_area1.item() == pytest.approx((w1 + wm) * length / 4, rel=1e-12)
    assert r.surface_area2.item() == pytest.approx((wm + w2) * length / 4, rel=1e-12)
    # EXTRAN: the frozen W(0.96 yFull) is much wider than the slot
    e = dt_.conduit_terms(*args)
    assert e.surface_area1.item() > 3 * r.surface_area1.item()


def test_conduit_terms_refuses_an_unknown_surcharge_method():
    g = geo()
    with pytest.raises(ValueError, match="surcharge_method"):
        dt_.conduit_terms(g, t([0.1]), t([0.1]), t([0.0]), t([0.0]), t([0.01]), 5.0,
                          surcharge_method="preissmann")


def test_slope_test_in_range_becomes_a_switch_on_the_head_difference():
    """The slope test ``y1 < y2`` (dwflow.c:659-662) with both end depths strictly inside
    (FUDGE, yFull) is ``h1 - h2 < z1 - z2``: a switch on the conduit's own head difference,
    so `slope_switch` is set and `switch_drop` is ``z1 - z2`` (ft), whichever side of it the
    fed heads are on; ``limited`` stays SWMM's verdict on the fed heads."""
    g = geo()
    a_old = xt.area_of_depth(t([0.5]), g.y_full, g.a_full)
    for y2, verdict in ((0.6, True), (0.3, False)):
        r = dt_.conduit_terms(g, t([0.4]), t([y2]), t([3.0]), t([3.0]), a_old, 5.0,
                              normal_flow_limited="slope")
        assert r.limited.item() is verdict
        assert r.slope_switch.item()
        torch.testing.assert_close(r.switch_drop, g.invert1 - g.invert2)
        a1 = xt.area_of_depth(t([0.4]), g.y_full, g.a_full)
        r1 = xt.radius_of_depth(t([0.4]), g.y_full)
        assert r.switch_normal_flow.item() == pytest.approx(
            (g.beta * a1 * r1 ** (2 / 3)).item())


def test_slope_switch_is_off_where_the_test_is_not_a_head_switch():
    """No switch: NORMAL_FLOW_LIMITED NONE or FROUDE without an outfall (no slope test); a
    downstream end at the crown under EXTRAN (y2 clamped to yFull, so ``y1 < y2`` is
    ``y1 < yFull``, already a condition of the check); a dry downstream end (y2 clamped to
    FUDGE); and where the Froude test fires, which limits whatever the slope test says."""
    g = geo()
    a_old = xt.area_of_depth(t([0.5]), g.y_full, g.a_full)
    yf = g.y_full.item()
    for mode in ("none", "froude"):
        r = dt_.conduit_terms(g, t([0.4]), t([0.6]), t([3.0]), t([3.0]), a_old, 5.0,
                              normal_flow_limited=mode)
        assert not r.slope_switch.item()
        assert math.isinf(r.switch_normal_flow.item())
    full = dt_.conduit_terms(g, t([0.4]), t([yf + 0.1]), t([3.0]), t([3.0]), a_old, 5.0)
    assert full.limited.item() and not full.slope_switch.item()
    dry = dt_.conduit_terms(g, t([0.4]), t([0.0]), t([3.0]), t([3.0]), a_old, 5.0,
                            normal_flow_limited="slope")
    assert not dry.limited.item() and not dry.slope_switch.item()
    # FROUDE fires (steep, shallow, y1 > y2): limited outright, no switch
    a_s = xt.area_of_depth(t([0.3]), g.y_full, g.a_full)
    fr = dt_.conduit_terms(g, t([0.3]), t([0.2]), t([2.0]), t([2.0]), a_s, 5.0)
    assert fr.limited.item() and not fr.slope_switch.item()
    # outfall end, y2 still in range: the slope test is a switch even in FROUDE mode
    out = dt_.conduit_terms(geo(outfall2=True), t([0.4]), t([0.6]), t([3.0]), t([3.0]),
                            a_old, 5.0, normal_flow_limited="froude")
    assert out.limited.item() and out.slope_switch.item()


def _geo_with(barrels=1.0, length_factor=1.0, **kwargs):
    base = geo(**kwargs)
    return dt_.ConduitGeometry.from_si(
        diameter=t([0.45]), roughness=t([0.013]), slope=t([0.005]), length=t([200.0]),
        invert1=t([10.0]), invert2=t([9.0]), offset1=base.offset1 * xt.LCF,
        offset2=base.offset2 * xt.LCF, outfall1=t([False]).bool(), outfall2=t([False]).bool(),
        barrels=t([barrels]), length_factor=t([length_factor]))


def test_barrels_multiply_flow_conveyance_areas_and_volume_not_the_per_barrel_state():
    """`Conduit.barrels` (dwflow.c:101-102, 243, 291-292; dynwave.c:575-576): the momentum
    update is one barrel's, so n barrels fed n times the flow return n times the flow offset,
    conveyance, normal flow, surface areas and volume of one barrel, and the per-barrel
    mid area and Froude number unchanged."""
    y1, y2, q_last, q_old = 0.6, 0.5, 2.0, 1.8                      # ft, cfs (one barrel)
    one, three = _geo_with(1.0), _geo_with(3.0)
    a_old = xt.area_of_depth(t([0.55]), one.y_full, one.a_full)
    r1 = dt_.conduit_terms(one, t([y1]), t([y2]), t([q_last]), t([q_old]), a_old, 5.0)
    r3 = dt_.conduit_terms(three, t([y1]), t([y2]), t([3 * q_last]), t([3 * q_old]), a_old,
                           5.0)
    for name in ("flow_offset", "conveyance", "momentum_flow", "surface_area1",
                 "surface_area2", "volume", "normal_flow", "lagged_conveyance1"):
        a, b = getattr(r1, name), getattr(r3, name)
        torch.testing.assert_close(b[torch.isfinite(b)], 3 * a[torch.isfinite(a)], rtol=1e-13,
                                   atol=0)
    for name in ("mid_area", "froude", "depth", "end_depth1", "end_depth2"):
        torch.testing.assert_close(getattr(r3, name), getattr(r1, name), rtol=1e-13, atol=0)
    assert r3.flow_class.item() == r1.flow_class.item()
    assert r3.limited.item() is r1.limited.item()
    # equal heads give exactly n times the flow: c0 + k dp is linear in the total
    dp = 0.37
    q1 = r1.flow_offset + r1.conveyance * dp
    q3 = r3.flow_offset + r3.conveyance * dp
    torch.testing.assert_close(q3, 3 * q1, rtol=1e-13, atol=0)


def test_dry_barrels_multiply_the_fudge_area_too():
    g = _geo_with(2.0)
    r = dt_.conduit_terms(g, t([0.0]), t([0.0]), t([0.0]), t([0.0]),
                          xt.area_of_depth(t([dt_.FUDGE]), g.y_full, g.a_full), 5.0)
    assert r.surface_area1.item() == pytest.approx(2 * dt_.FUDGE * g.length.item() / 2)


def test_lengthening_factor_follows_conduit_get_length_factor():
    """link.c:1217-1254: ratio = (sqrt(g yFull) + vFull) t / L, at least 1. A 0.3 m conduit
    8 m long at slope 0.01 and n 0.013 with t = 5 s, by hand in feet."""
    diameter, n, slope, length, step = 0.30, 0.013, 0.01, 8.0, 5.0
    y_full = diameter / xt.LCF
    a_full = math.pi / 4 * y_full**2
    s_full = a_full * (y_full / 4) ** (2 / 3)
    v_full = xt.PHI / n * s_full * math.sqrt(slope) / a_full
    expect = (math.sqrt(32.2 * y_full) + v_full) * step / (length / xt.LCF)
    assert expect > 1.0
    got = dt_.lengthening_factor(t([diameter]), t([n]), t([slope]), t([length]), step)
    assert got.item() == pytest.approx(expect, rel=1e-6)
    # a long conduit is never shortened: the factor is 1
    assert dt_.lengthening_factor(t([diameter]), t([n]), t([slope]), t([200.0]),
                                  step).item() == 1.0


def test_lengthening_changes_mod_length_not_the_true_length_or_normal_flow():
    """link.c:1104-1118: modLength = lf L, roughness n/sqrt(lf), slope/lf, so beta is
    unchanged (normal flow unchanged) and the friction coefficient g n^2/(PHI^2 lf) is
    divided by lf. The conveyance g dt A/(modLength denom) falls with the longer length, the
    surface areas grow with it, the volume keeps the true length (dwflow.c:177, 291)."""
    lf = 1.7
    plain, long = _geo_with(1.0, 1.0), _geo_with(1.0, lf)
    torch.testing.assert_close(long.mod_length, lf * plain.length)
    torch.testing.assert_close(long.length, plain.length)
    torch.testing.assert_close(long.beta, plain.beta)
    torch.testing.assert_close(long.rough_factor, plain.rough_factor / lf)
    y1, y2, q_last, q_old = 0.6, 0.5, 2.0, 1.8
    a_old = xt.area_of_depth(t([0.55]), plain.y_full, plain.a_full)
    rp = dt_.conduit_terms(plain, t([y1]), t([y2]), t([q_last]), t([q_old]), a_old, 5.0)
    rl = dt_.conduit_terms(long, t([y1]), t([y2]), t([q_last]), t([q_old]), a_old, 5.0)
    torch.testing.assert_close(rl.volume, rp.volume)
    torch.testing.assert_close(rl.surface_area1, lf * rp.surface_area1)
    torch.testing.assert_close(rl.surface_area2, lf * rp.surface_area2)
    # the momentum update itself, by hand: dq1 uses the divided friction coefficient, dq2 and
    # dq4 the modified length
    a1, a2, am, r1, rm, v, sigma = _weights(long, y1, y2, q_last)
    aw, rw = a1 + (am - a1) * sigma, r1 + (rm - r1) * sigma
    dq1 = 5.0 * long.rough_factor / rw**1.33333 * abs(v)
    dq3 = 2 * v * (am - a_old) * sigma
    dq4 = 5.0 * v * v * (a2 - a1) / long.mod_length * sigma
    denom = 1 + dq1
    torch.testing.assert_close(rl.conveyance, 32.2 * 5.0 * aw / (long.mod_length * denom))
    torch.testing.assert_close(rl.flow_offset, (q_old + dq3 + dq4) / denom)
    # the normal-flow limit is unchanged by lengthening
    assert rl.limited.item() is rp.limited.item()
    torch.testing.assert_close(rl.normal_flow, rp.normal_flow)


def _ycs(g, q):
    yn = xt.normal_depth_ft(t([q]), g.y_full, g.a_full, g.s_full, g.beta).item()
    yc = xt.critical_depth(t([q]), g.y_full, g.a_full).item()
    return min(yn, yc), max(yn, yc)


def _at_end2(g, y2, q, y1=0.6, **kw):
    """`conduit_terms` with the downstream END depth `y2` (ft above the offset invert)."""
    a_old = xt.area_of_depth(t([0.5]), g.y_full, g.a_full)
    d2 = g.offset2.item() + y2
    return dt_.conduit_terms(g, t([y1]), t([d2]), t([q]), t([q]), a_old, 5.0, **kw)


def test_a_drop_end_reports_the_thresholds_where_its_node_area_jumps_up():
    """An outlet offset receiving forward flow (dwflow.c:357-368, 472-499): below min(yN,
    yC) the end is DN_CRITICAL and gives its node no area, between min and max the area is
    a quarter of the conduit scaled by fasnh (1 falling to 0), above max the full quarter.
    So the node's area jumps UP at both thresholds; `drop_depth2` is the one nearest the
    end depth and `drop_area2` the size of its jump, `drop_above2` the side."""
    g = geo(offset2=0.1)
    q = 2.0
    lo, hi = _ycs(g, q)
    eps = 1e-9 * (hi - lo)                      # fasnh moves by eps / (hi - lo)
    for thr, probe in ((lo, lo + 0.1 * (hi - lo)), (hi, hi + 0.1 * (hi - lo))):
        r = _at_end2(g, probe, q)
        assert r.drop_depth2.item() == pytest.approx(thr, rel=1e-12)
        assert r.drop_above2.item()
        below = _at_end2(g, thr - eps, q)
        above = _at_end2(g, thr + eps, q)
        assert not below.drop_above2.item() and above.drop_above2.item()
        jump = above.surface_area2.item() - below.surface_area2.item()
        assert jump == pytest.approx(r.drop_area2.item(), rel=1e-4)
        assert jump > 0.0
    assert _at_end2(g, lo - eps, q).flow_class.item() == dt_.DN_CRITICAL
    # end 1 has no offset: no threshold there
    assert math.isinf(_at_end2(g, lo, q).drop_depth1.item())


def test_a_drop_end_below_fudge_is_its_lower_threshold():
    """With no flow min(yN, yC) is 0, and the lower threshold is SWMM's wet test, y > FUDGE:
    a dry end below a wet one is DN_CRITICAL (h1 above the offset invert, dwflow.c:396-411),
    no area; just above FUDGE both ends are wet, a full quarter (fasnh 1)."""
    g = geo(offset2=0.1)
    r = _at_end2(g, 0.0, 0.0)
    assert r.drop_depth2.item() == pytest.approx(dt_.FUDGE)
    assert r.flow_class.item() == dt_.DN_CRITICAL and r.surface_area2.item() == 0.0
    assert not r.drop_above2.item()
    wet = _at_end2(g, dt_.FUDGE * (1 + 1e-9), 0.0)
    assert wet.drop_above2.item() and wet.surface_area2.item() > 0.0
    assert wet.drop_area2.item() == pytest.approx(wet.surface_area2.item(), rel=1e-6)


def test_an_upstream_drop_end_switches_on_reverse_flow_only_at_min_yn_yc():
    """An inlet offset (dwflow.c:337-348): on reverse flow the UPSTREAM end is UP_CRITICAL
    below min(yN, yC) (no area at node 1) -- its threshold; no max threshold there (fasnh is
    the downstream end's only)."""
    g = geo(offset1=0.2)
    lo, _ = _ycs(g, 1.0)
    a_old = xt.area_of_depth(t([0.3]), g.y_full, g.a_full)
    r = dt_.conduit_terms(g, t([0.2 / xt.LCF + 0.5 * lo]), t([1.0 + 3.3]), t([-1.0]),
                          t([-1.0]), a_old, 5.0)
    assert r.flow_class.item() == dt_.UP_CRITICAL
    assert r.drop_depth1.item() == pytest.approx(lo, rel=1e-12)
    assert not r.drop_above1.item() and r.drop_area1.item() > 0.0


def test_no_drop_threshold_without_an_offset_a_wet_other_end_or_below_the_crown():
    g0 = geo()
    r = _at_end2(g0, 0.3, 2.0)
    assert math.isinf(r.drop_depth1.item()) and math.isinf(r.drop_depth2.item())
    g = geo(offset2=0.1)
    dry_up = _at_end2(g, 0.3, 2.0, y1=0.0)                  # the other end dry
    assert math.isinf(dry_up.drop_depth2.item())
    yf = g.y_full.item()
    full = _at_end2(g, yf + 0.5, 2.0, y1=yf + 1.0)          # both ends full
    assert math.isinf(full.drop_depth2.item())


def test_a_held_drop_end_takes_swmms_class_at_its_threshold_and_slides_the_other_end():
    """`hold2` (the closure's tie rule) classifies the end on the UPPER side of its
    threshold whatever its depth: SWMM's class AT the threshold (its tests are strict). At
    min(yN, yC) the other end's area is the sliding blend ``(2 - theta)`` quarters: half
    the conduit (DN_CRITICAL's) at theta 0, a quarter (the upper side's) at theta 1. At
    max(yN, yC) the class is the same on both sides, and only fasnh is 1."""
    g = geo(offset2=0.1)
    q = 2.0
    lo, hi = _ycs(g, q)
    eps = 1e-9 * (hi - lo)
    y2 = lo - eps
    free = _at_end2(g, y2, q)
    assert free.flow_class.item() == dt_.DN_CRITICAL
    up = _at_end2(g, lo + eps, q)
    for theta in (0.0, 0.25, 1.0):
        held = _at_end2(g, y2, q, hold2=t([theta]))
        assert held.flow_class.item() in (dt_.SUBCRITICAL, dt_.SUPCRITICAL)
        assert held.drop_above2.item()
        assert held.surface_area2.item() == pytest.approx(up.surface_area2.item(), rel=1e-5)
        assert held.surface_area1.item() == pytest.approx(
            (2.0 - theta) * up.surface_area1.item(), rel=1e-5)
    assert free.surface_area1.item() == pytest.approx(2.0 * up.surface_area1.item(), rel=1e-5)
    no = _at_end2(g, y2, q, hold2=t([-1.0]))               # < 0: no hold
    assert no.flow_class.item() == dt_.DN_CRITICAL
    # the upper threshold: fasnh 1 when held, the other end untouched
    below_hi = _at_end2(g, hi - eps, q)
    held_hi = _at_end2(g, hi - eps, q, hold2=t([0.5]))
    above_hi = _at_end2(g, hi + eps, q)
    assert held_hi.surface_area2.item() == pytest.approx(above_hi.surface_area2.item(),
                                                         rel=1e-5)
    assert below_hi.surface_area2.item() < 1e-3 * above_hi.surface_area2.item()
    assert held_hi.surface_area1.item() == pytest.approx(below_hi.surface_area1.item(),
                                                         rel=1e-5)
