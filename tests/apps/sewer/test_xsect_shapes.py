"""SWMM's non-circular cross sections (`xsect_spec`, `xsect_shapes`): the constants of
`xsect_setParams`, the geometry functions, the root finders and the transect / shape-curve
table builders. The match with the engine itself is `tests/verification/
test_sewer_shapes_parity.py`."""

import math

import pytest
import torch

from noodl.apps.sewer import xsect_shapes as sh
from noodl.apps.sewer import xsect_spec as sp
from noodl.apps.sewer import xsect_tables as xt

F64 = torch.float64

CASES = {
    "RECT_CLOSED": (1.0, 2.0), "RECT_OPEN": (1.0, 2.0, 0.0),
    "TRAPEZOIDAL": (1.0, 2.0, 1.0, 2.0), "TRIANGULAR": (1.0, 2.0), "PARABOLIC": (1.0, 2.0),
    "POWER": (1.0, 2.0, 2.0), "RECT_TRIANGULAR": (1.0, 1.0, 0.3), "RECT_ROUND": (1.0, 1.0, 0.6),
    "MODBASKETHANDLE": (1.0, 1.0, 0.6), "HORIZ_ELLIPSE": (1.0, 1.5, 0.0),
    "VERT_ELLIPSE": (1.5, 1.0, 0.0), "ARCH": (1.0, 1.5, 0.0), "EGG": (1.0,),
    "HORSESHOE": (1.0,), "GOTHIC": (1.0,), "CATENARY": (1.0,), "SEMIELLIPTICAL": (1.0,),
    "BASKETHANDLE": (1.0,), "SEMICIRCULAR": (1.0,), "FILLED_CIRCULAR": (1.0, 0.2),
    "FORCE_MAIN": (1.0, 120.0),
}


def make_set(kinds_geoms):
    xs = [sp.Xsect(k, g) for k, g in kinds_geoms]
    ps = [x.params() for x in xs]
    vals = {k: torch.tensor([p.get(k, 0.0) for p in ps], dtype=F64) for k in sh.PARAMS}
    return sh.XsectSet([x.kind for x in xs], vals)


def single(kind):
    return make_set([(kind, CASES[kind])])


def test_closed_form_shapes_round_trip_area_and_depth_exactly():
    for kind in ("RECT_CLOSED", "RECT_OPEN", "TRAPEZOIDAL", "TRIANGULAR", "PARABOLIC", "POWER"):
        s = single(kind)
        g = s.groups[0]
        y = s.y_full * torch.linspace(0.05, 0.95, 10, dtype=F64).unsqueeze(-1)
        a = s.area(y)
        torch.testing.assert_close(g.shape.y_of_a(g.p, g.t, a), y, rtol=1e-12, atol=1e-14)


def test_rectangle_constants_and_functions_follow_xsect_c():
    p = sp.Xsect("RECT_CLOSED", (0.6, 1.2)).params()
    yf, wm = 0.6 / xt.LCF, 1.2 / xt.LCF
    assert p["a_full"] == pytest.approx(yf * wm, rel=1e-14)
    assert p["r_full"] == pytest.approx(yf * wm / (2 * (yf + wm)), rel=1e-14)
    s = single("RECT_CLOSED")
    y = s.y_full * 0.5
    assert s.area(y).item() == pytest.approx(0.5 * s.a_full.item(), rel=1e-14)
    assert s.width(s.y_full).item() == 0.0            # xsect.c:1006: no width at full height
    assert s.width(y).item() == pytest.approx(s.w_max.item())


def test_every_shape_has_positive_monotone_geometry():
    for kind in CASES:
        s = single(kind)
        y = s.y_full * torch.linspace(0.02, 0.98, 25, dtype=F64).unsqueeze(-1)
        a, r, w = s.area(y), s.radius(y), s.width(y)
        assert bool((a > 0).all()) and bool((r > 0).all()), kind
        assert bool((a[1:] > a[:-1]).all()), kind
        assert bool((w > 0).all()), kind
        assert a[-1].item() <= s.a_full.item() * (1.0 + 1e-9), kind


def test_a_mixed_set_equals_its_single_shape_sets():
    kinds = ["RECT_CLOSED", "EGG", "CIRCULAR", "TRAPEZOIDAL", "EGG"]
    geoms = [CASES.get(k, (0.9,)) for k in kinds]
    circ = sp.Xsect("CIRCULAR", (0.9,)).params()
    parts = [sp.Xsect(k, g).params() for k, g in zip(kinds, geoms, strict=True)]
    vals = {k: torch.tensor([p.get(k, 0.0) for p in parts], dtype=F64) for k in sh.PARAMS}
    mixed = sh.XsectSet(kinds, vals)
    y = mixed.y_full * 0.37
    for fn in ("area", "width", "radius"):
        got = getattr(mixed, fn)(y)
        for i in range(len(kinds)):
            one = mixed.take(torch.tensor([i]))
            assert getattr(one, fn)(y[i:i + 1]).item() == got[i].item(), (fn, i)
    assert circ["y_full"] == pytest.approx(0.9 / xt.LCF)
    # the circular entry is the tabulated circle of xsect_tables, bit for bit
    i = kinds.index("CIRCULAR")
    expect = xt.area_of_depth(y[i:i + 1], mixed.y_full[i:i + 1], mixed.a_full[i:i + 1])
    assert mixed.area(y)[i].item() == expect.item()


def test_tiled_set_repeats_the_conduits():
    s = make_set([("RECT_CLOSED", CASES["RECT_CLOSED"]), ("EGG", CASES["EGG"])])
    y = s.y_full * 0.3
    got = s.tiled(3).area(torch.cat([y, y, y]))
    torch.testing.assert_close(got, torch.cat([s.area(y)] * 3), rtol=0, atol=0)


def test_normal_depth_inverts_manning_where_the_section_factor_rises():
    # (EGG is left out: SWMM's own egg tables do not close at shallow depth -- Y(S(A(y)))
    # is 8 % low at a fifth of the depth -- and the parity test holds it to SWMM)
    for kind in ("TRAPEZOIDAL", "TRIANGULAR", "PARABOLIC", "RECT_OPEN", "HORSESHOE"):
        s = single(kind)
        y = s.y_full * torch.tensor([[0.2], [0.4], [0.6]], dtype=F64)
        beta = torch.tensor([1.7], dtype=F64)
        q = beta * s.area(y) * s.radius(y) ** (2.0 / 3.0)
        tol = 5e-3 if kind == "HORSESHOE" else 1e-6
        torch.testing.assert_close(s.normal_depth(q, beta), y, rtol=tol, atol=0)


def test_critical_depth_satisfies_the_critical_flow_equation():
    for kind in ("RECT_CLOSED", "TRIANGULAR", "PARABOLIC", "POWER", "TRAPEZOIDAL", "EGG"):
        s = single(kind)
        y = s.y_full * torch.tensor([[0.3], [0.6]], dtype=F64)
        a, w = s.area(y), s.width(y)
        q = a * torch.sqrt(32.2 * a / w)
        tol = 1e-12 if kind in ("RECT_CLOSED", "TRIANGULAR", "PARABOLIC", "POWER") else 1e-2
        torch.testing.assert_close(s.critical_depth(q), y, rtol=tol, atol=0)
    assert single("EGG").critical_depth(torch.zeros(1, dtype=F64)).item() == 0.0


def test_ridder_and_newton_find_simple_roots():
    root = sh.ridder(lambda x: x * x - 2.0, torch.tensor([0.0], dtype=F64),
                     torch.tensor([3.0], dtype=F64), 1e-9)
    assert root.item() == pytest.approx(math.sqrt(2.0), abs=1e-8)
    miss = sh.ridder(lambda x: x * x + 1.0, torch.tensor([0.0], dtype=F64),
                     torch.tensor([3.0], dtype=F64), 1e-9)
    assert miss.item() == -1.0e20                      # no bracket: findroot_Ridder's value

    def func(x):
        return x ** 3 - 2.0, 3.0 * x * x

    one = torch.ones(1, dtype=F64)
    x = sh.newton_bracketed(func, 0 * one, 3 * one, 1.5 * one, 1e-12 * one)
    assert x.item() == pytest.approx(2.0 ** (1.0 / 3.0), abs=1e-10)


def test_decreasing_tail_inverse_lookup_returns_the_maximum():
    table = sh._table("S_EGG")
    n = table.shape[0]
    peak = float(torch.argmax(table)) / (n - 1)
    big = torch.tensor([float(table.max()) * 1.0001], dtype=F64)
    assert xt.inv_lookup(big, table).item() == pytest.approx(peak)
    below = torch.tensor([0.5 * float(table.max())], dtype=F64)
    x = xt.inv_lookup(below, table).item()
    assert xt.lookup(torch.tensor([x], dtype=F64), table).item() == pytest.approx(
        0.5 * float(table.max()), rel=1e-3)


def test_lookup_rows_is_lookup_per_row():
    t = torch.stack([sh._table("A_EGG"), sh._table("A_EGG").flip(0).flip(0)])
    x = torch.tensor([[0.0, 0.31], [0.5, 1.0], [0.013, 2.0]], dtype=F64)
    got = sh.lookup_rows(x, t.repeat(1, 1)[:, :26])
    for r in range(2):
        for k in range(3):
            assert got[k, r].item() == xt.lookup(x[k, r], sh._table("A_EGG")).item()
    y = torch.tensor([[0.0, 0.2], [0.7, 1.5]], dtype=F64)
    tab = torch.stack([sh._table("Y_EGG"), sh._table("Y_EGG")])
    inv = sh.inv_lookup_rows(y, tab)
    for r in range(2):
        for k in range(2):
            assert inv[k, r].item() == xt.inv_lookup(y[k, r].reshape(1), sh._table("Y_EGG")).item()


def test_transect_of_a_rectangular_channel_has_the_rectangle_tables():
    t = sp.Transect("R", (0.0, 0.0, 4.0, 4.0), (2.0, 0.0, 0.0, 2.0), 0.0, 4.0, 0.03, 0.03, 0.03)
    tb = sp.transect_tables(t)
    assert tb["y_full"] == pytest.approx(2.0 / xt.LCF)
    assert tb["w_max"] == pytest.approx(4.0 / xt.LCF)
    a_n = torch.tensor(tb["area"], dtype=F64)
    torch.testing.assert_close(a_n, torch.linspace(0, 1, 51, dtype=F64), rtol=1e-12, atol=1e-12)
    y = torch.linspace(0, 1, 51, dtype=F64)[1:] * 2.0 / xt.LCF
    w = 4.0 / xt.LCF
    manning_r = (w * y) / (w + 2 * y)
    # SWMM solves Manning with 1.486 in the flow and divides by 1.49 (transect.c:463): a
    # constant (1.486/1.49)^1.5 = 0.996 on the radius
    torch.testing.assert_close(torch.tensor(tb["hrad"][1:], dtype=F64) * tb["r_full"], manning_r,
                               rtol=6e-3, atol=0)


def test_transect_overbank_roughness_and_length_factor_scale_the_channel():
    kw = dict(name="T", stations=(0.0, 1.0, 3.0, 4.0), elevations=(1.0, 0.0, 0.0, 1.0),
              left_bank=1.0, right_bank=3.0, n_left=0.1, n_right=0.1)
    plain = sp.transect_tables(sp.Transect(n_channel=0.03, length_factor=1.0, **kw))
    rough = sp.transect_tables(sp.Transect(n_channel=0.03, length_factor=4.0, **kw))
    assert rough["r_full"] != plain["r_full"]
    with pytest.raises(ValueError, match="fewer than two stations"):
        sp.transect_tables(sp.Transect("X", (0.0,), (0.0,), 0.0, 0.0, 0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="no depth"):
        sp.transect_tables(sp.Transect("X", (0.0, 1.0), (1.0, 1.0), 0.0, 1.0, 0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="station distances decrease"):
        sp.transect_tables(sp.Transect("X", (1.0, 0.0), (0.0, 1.0), 0.0, 1.0, 0.1, 0.1, 0.1))


def test_custom_shape_of_a_rectangle_matches_the_rectangle():
    c = sp.custom_tables(((0.0, 1.0), (1.0, 1.0)))
    assert c["a_full"] == pytest.approx(1.0) and c["w_max"] == pytest.approx(1.0)
    assert c["r_full"] == pytest.approx(0.25)              # a closed unit square: P = 4
    for bad in (((0.0, 1.0),), ((0.0, -1.0), (1.0, 1.0)), ((1.0, 1.0),)):
        with pytest.raises(ValueError):
            sp.custom_tables(bad)


def test_xsect_refusals_name_the_shape():
    for kind in ("DUMMY", "STREET"):
        with pytest.raises(ValueError, match=kind):
            sp.Xsect(kind, (1.0,)).params()
    with pytest.raises(ValueError, match="xsect_setParams"):
        sp.Xsect("RECT_CLOSED", (1.0, 0.0)).params()
    with pytest.raises(ValueError, match="xsect_setParams"):
        sp.Xsect("FILLED_CIRCULAR", (1.0, 1.0)).params()
    with pytest.raises(ValueError, match="size code"):
        sp.Xsect("ARCH", (1.0, 0.0, 999.0)).params()
    with pytest.raises(ValueError, match="unknown"):
        sp.Xsect("BOX", (1.0,)).params()


def test_standard_sizes_come_from_the_tables_in_inches():
    p = sp.Xsect("ARCH", (1.0, 0.0, 1.0)).params()       # code 1: 11 in x 18 in, 1.1 ft2
    assert p["y_full"] == pytest.approx(11.0 / 12.0)
    assert p["w_max"] == pytest.approx(18.0 / 12.0)
    assert p["a_full"] == pytest.approx(1.1)


def test_force_main_sets_its_own_section_factor():
    p = sp.Xsect("FORCE_MAIN", (0.6, 120.0)).params()
    assert p["s_full"] == pytest.approx(p["a_full"] * p["r_full"] ** 0.63)
    assert p["s_max"] == pytest.approx(1.06949 * p["s_full"])
