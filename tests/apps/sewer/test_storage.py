"""SWMM 5.2.4 storage-unit area and volume curves (`node.c`, `table.c`) in SI user units."""

import math

import pytest
import torch

from noodl.apps.sewer.storage import StorageCurves, validate_storage_curve

F64 = torch.float64


# ---------------------------------------------------------------- SWMM's loops, transcribed
def _lookup_ex(xs, ys, x):
    """table.c:464-499 `table_lookupEx`, scalar, as SWMM writes it."""
    x1, y1 = xs[0], ys[0]
    if x <= x1:
        return x / x1 * y1 if x1 > 0.0 else y1
    s = 0.0
    for x2, y2 in zip(xs[1:], ys[1:], strict=True):
        if x2 != x1:
            s = (y2 - y1) / (x2 - x1)
        if x <= x2:
            dx = x2 - x1
            return (y1 + y2) / 2.0 if abs(dx) < 1e-20 else y1 + (x - x1) * (y2 - y1) / dx
        x1, y1 = x2, y2
    s = max(s, 0.0)
    return y1 + s * (x - x1)


def _storage_volume(xs, ys, x):
    """table.c:587-647 `table_getStorageVolume`, scalar, as SWMM writes it."""
    v, dx, dy = 0.0, 0.0, 0.0
    x1, a1 = xs[0], ys[0]
    if x <= x1:
        return 0.0 if x1 < 1e-6 else (a1 / x1) * x * x / 2.0
    for xe, ye in zip(xs[1:], ys[1:], strict=True):
        if xe >= x:
            a = a1 + (x - x1) * (ye - a1) / (xe - x1)
            return v + (a1 + a) / 2.0 * (x - x1)
        dx, dy = xe - x1, ye - a1
        v += (a1 + ye) / 2.0 * dx
        x1, a1 = xe, ye
    if dx > 1e-6:
        s = dy / dx
        a = a1 + s * (x - x1)
        v = v - a1 * a1 / s / 2.0 if a < 0.0 else v + (a1 + a) / 2.0 * (x - x1)
    return v


def _curves(*curves, full=(10.0,)):
    full = list(full) * len(curves) if len(full) == 1 else list(full)
    return StorageCurves(list(curves), torch.tensor(full, dtype=F64))


DEPTHS = [0.0, 0.05, 0.2, 0.5, 0.75, 1.0, 1.6, 2.0, 2.5, 3.7]


def test_functional_area_and_volume_are_swmms_power_law():
    """node.c:1001-1003 and :954-958: A = a0 + a1 d^a2, V = a0 d + a1/(a2 + 1) d^(a2 + 1)
    (FUNCTIONAL a1 a2 a0 in the file; noodl's ("functional", a, b, c) is A = a y^b + c)."""
    sc = _curves(("functional", 20.0, 1.5, 5.0), ("functional", 3.0, 0.0, 0.0))
    for d in DEPTHS:
        depth = torch.tensor([d, d], dtype=F64)
        area, volume = sc.area(depth), sc.volume(depth)
        assert area[0].item() == pytest.approx(5.0 + 20.0 * d**1.5, rel=1e-14, abs=1e-14)
        assert volume[0].item() == pytest.approx(5.0 * d + 20.0 / 2.5 * d**2.5, rel=1e-14,
                                                 abs=1e-14)
        # a2 = 0: pow(d, 0) = 1 even at d = 0 (C's pow), so the area is a0 + a1 throughout
        assert area[1].item() == pytest.approx(3.0, rel=1e-14)
        assert volume[1].item() == pytest.approx(3.0 * d, rel=1e-14, abs=1e-14)


def test_volume_is_capped_at_the_full_volume_and_zero_at_zero_depth():
    """node.c:942-944: V(0) = 0 and V(d >= fullDepth) = fullVolume = V(fullDepth) (the area
    is not capped: node.c:979-1018 evaluates the curve at any depth)."""
    sc = _curves(("functional", 20.0, 1.0, 5.0), full=(1.5,))
    full = 5.0 * 1.5 + 10.0 * 1.5**2
    for d, v in ((0.0, 0.0), (1.5, full), (2.0, full), (9.0, full)):
        assert sc.volume(torch.tensor([d], dtype=F64)).item() == pytest.approx(v, rel=1e-14)
    assert sc.area(torch.tensor([2.0], dtype=F64)).item() == pytest.approx(45.0, rel=1e-14)
    torch.testing.assert_close(sc.full_volume, torch.tensor([full], dtype=F64))


@pytest.mark.parametrize("xs,ys", [
    ((0.0, 0.5, 1.0, 2.0), (10.0, 30.0, 60.0, 80.0)),      # starts at 0, rising
    ((0.3, 1.0, 2.0), (12.0, 40.0, 25.0)),                 # starts above 0, falls at the end
    ((0.0, 1.0), (50.0, 50.0)),                            # a prism
    ((0.4,), (30.0,)),                                     # one point
])
def test_tabular_area_and_volume_follow_swmms_table_loops(xs, ys):
    """TABULAR: the area is `table_lookupEx` (linear below the first point from 0, linear
    inside, the last segment's slope floored at 0 beyond) and the volume the end-area method
    of `table_getStorageVolume` (from 0 at the first point, the triangle below it, the
    extrapolated area beyond, with SWMM's correction where that area would turn negative),
    against SWMM's own loops transcribed in this test."""
    sc = _curves(("tabular", xs, ys), full=(100.0,))
    for d in DEPTHS + [0.4, 0.3, 4.9]:
        depth = torch.tensor([d], dtype=F64)
        assert sc.area(depth).item() == pytest.approx(_lookup_ex(xs, ys, d), rel=1e-13,
                                                      abs=1e-13), d
        expected = 0.0 if d == 0.0 else _storage_volume(xs, ys, d)
        assert sc.volume(depth).item() == pytest.approx(expected, rel=1e-13, abs=1e-13), d


def test_tabular_volume_beyond_a_falling_table_stops_where_the_area_reaches_zero():
    """table.c:637-641: past the table, an extrapolated area below 0 gives the volume to the
    zero crossing, V_last - a_last^2 / s / 2."""
    xs, ys = (0.0, 1.0, 2.0), (40.0, 40.0, 20.0)
    sc = _curves(("tabular", xs, ys), full=(100.0,))
    v = sc.volume(torch.tensor([5.0], dtype=F64)).item()
    assert v == pytest.approx(40.0 + 30.0 + 20.0 * 20.0 / 20.0 / 2.0, rel=1e-14)
    assert v == pytest.approx(_storage_volume(xs, ys, 5.0), rel=1e-14)


@pytest.mark.parametrize("curve,area", [
    (("cylindrical", 4.0, 2.0), lambda d: math.pi * 2.0 * 1.0),
    (("conical", 4.0, 2.0, 0.5), lambda d: math.pi * (2.0 + 0.5 * d) * (1.0 + 0.5 * d / 2.0)),
    (("paraboloid", 4.0, 2.0, 3.0), lambda d: math.pi * 2.0 * 1.0 * d / 3.0),
    (("pyramidal", 4.0, 2.0, 0.5), lambda d: (4.0 + 2 * 0.5 * d) * (2.0 + 2 * 0.5 * d)),
])
def test_closed_shapes_are_their_solids(curve, area):
    """CYLINDRICAL/CONICAL/PARABOLOID/PYRAMIDAL (node.c:750-783 coefficients, :1008-1013
    area, :964-971 volume): the water-surface area of the elliptic cylinder (L, W), the
    elliptic cone frustum (base L x W, side slope Z run/rise), the paraboloid (top L x W at
    height H) and the rectangular frustum (base L x W, slope Z), and the volume its
    integral (Simpson's rule is exact on these quadratics)."""
    sc = _curves(curve, full=(100.0,))
    for d in DEPTHS:
        depth = torch.tensor([d], dtype=F64)
        assert sc.area(depth).item() == pytest.approx(area(d), rel=1e-13, abs=1e-13)
        simpson = d / 6.0 * (area(0.0) + 4.0 * area(d / 2.0) + area(d))
        assert sc.volume(depth).item() == pytest.approx(simpson, rel=1e-12, abs=1e-13)


def test_non_storage_junctions_have_no_area_or_volume_and_the_gradient_is_finite():
    sc = _curves(None, ("functional", 20.0, 0.5, 0.0), ("tabular", (0.0, 1.0), (5.0, 9.0)))
    depth = torch.tensor([0.7, 0.0, 0.0], dtype=F64, requires_grad=True)
    area, volume = sc.area(depth), sc.volume(depth)
    assert area[0].item() == 0.0 and volume[0].item() == 0.0
    (area.sum() + volume.sum()).backward()
    assert torch.isfinite(depth.grad).all()


@pytest.mark.parametrize("curve,match", [
    (("functional", 1.0, 1.0, -0.5), "c .*>= 0"),
    (("functional", 1.0, -1.0, 0.5), "exponent"),
    (("tabular", (0.0, 1.0, 1.0), (1.0, 2.0, 3.0)), "strictly increasing"),
    (("tabular", (), ()), "at least one"),
    (("tabular", (0.0, 1.0), (1.0,)), "as many"),
    (("cylindrical", 0.0, 2.0), "> 0"),
    (("conical", 2.0, 2.0, -1.0), ">= 0"),
    (("paraboloid", 2.0, 2.0, 0.0), "> 0"),
    (("bowl", 1.0), "shape"),
    ("functional", "tuple"),
])
def test_bad_curves_are_refused_by_name(curve, match):
    with pytest.raises(ValueError, match=rf"SU1.*{match}"):
        validate_storage_curve("SU1", curve)


def test_a_negative_full_volume_is_refused_by_name():
    """node.c:219-222 (ERR_STORAGE_VOLUME)."""
    with pytest.raises(ValueError, match="SU1.*negative volume"):
        StorageCurves([("functional", -5.0, 1.0, 1.0)], torch.tensor([2.0], dtype=F64),
                      names=["SU1"])


@pytest.mark.parametrize("curve", [("pyramidal", 20.0, 10.0, 10.0), ("conical", 20.0, 10.0, 8.0),
                                   ("functional", 3.0, 2.5, 1.0), ("tabular", (0.0, 2.0),
                                                                   (5.0, 9.0))])
def test_the_depth_gradient_is_the_curves_own_at_any_depth(curve):
    """Every branch is both-branches safe: a closed shape's quadratic coefficient (4 Z^2 =
    400 for a pyramid of side slope 10) is never used as an exponent, so at 7 m (area
    24000 m2) and at 0 m the gradients are finite: dV/dy = A(y) (above 0), and dA/dy the
    curve's slope."""
    sc = _curves(curve, full=(100.0,))
    for d in (0.0, 0.3, 7.0):
        depth = torch.tensor([d], dtype=F64, requires_grad=True)
        area = sc.area(depth)
        (g_area,) = torch.autograd.grad(area.sum(), depth)
        depth = torch.tensor([d], dtype=F64, requires_grad=True)
        (g_vol,) = torch.autograd.grad(sc.volume(depth).sum(), depth)
        assert torch.isfinite(g_area).all() and torch.isfinite(g_vol).all(), (curve, d)
        if d > 0.0:
            assert g_vol.item() == pytest.approx(area.item(), rel=1e-12)
    pyramid = _curves(("pyramidal", 20.0, 10.0, 10.0), full=(100.0,))
    seven = torch.tensor([7.0], dtype=F64)
    assert pyramid.area(seven).item() == pytest.approx(24000.0, rel=1e-14)
