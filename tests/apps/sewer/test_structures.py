"""SWMM 5.2.4's orifices, weirs and outlets (`noodl.apps.sewer.structures`) against
hand-evaluated SWMM expressions (`link.c`), each regime at three heads, in SWMM's feet.

Each expected value is the closed form SWMM evaluates on that branch, written out here from
link.c (line references at each test), not a second transcription of the module's code."""

import math

import pytest
import torch

from noodl.apps.sewer.dynwave_terms import DN_CRITICAL, DRY, SUBCRITICAL, UP_CRITICAL
from noodl.apps.sewer.structures import (
    CIRCULAR,
    RECT_CLOSED,
    RECT_OPEN,
    TRANSVERSE_WEIR,
    TRAPEZOIDAL,
    TRAPEZOIDAL_WEIR,
    TRIANGULAR,
    VNOTCH_WEIR,
    Opening,
    OrificeSet,
    OutletSet,
    WeirSet,
    equivalent_length,
    orifice_flow,
    outlet_dqdh,
    outlet_flow,
    table_lookup,
    weir_flow,
)
from noodl.apps.sewer.xsect_tables import LCF, QCF

F64 = torch.float64
G = 32.2
FUDGE = 1e-4
M3 = 0.028317                      # consts.h:86 M3perFT3
ROUTE = 5.0                        # s
Z1, Z2 = 10.0 / LCF, 9.0 / LCF     # node inverts (ft)
OFFSET = 0.1 / LCF                 # crest above node 1 (ft)
CREST = Z1 + OFFSET


def t(*values):
    return torch.tensor(values, dtype=F64)


def rect_orifice(bottom=False, gated=False, cd=0.65):
    """A 0.2 m high, 0.3 m wide RECT_CLOSED orifice."""
    opening = Opening.from_si([RECT_CLOSED], [0.2], [0.3])
    return OrificeSet.build(opening, [bottom], [cd], [gated], ROUTE)


def run_orifice(orf, h1, h2, **kw):
    n = len(h1)
    return orifice_flow(orf, t(*h1), t(*h2), t(*[Z1] * n), t(*[Z2] * n), t(*[CREST] * n), **kw)


YF = 0.2 / LCF                     # rect orifice height (ft)
WF = 0.3 / LCF                     # rect orifice width (ft)
AREA = YF * WF
C_ORIF = 0.65 * AREA * math.sqrt(2 * G)                  # link.c:1753-1754
LENGTH = max(200.0, 2 * ROUTE * math.sqrt(G * YF))       # link.c:1721-1723


def test_equivalent_length_is_at_least_200_ft():
    """link.c:1721-1723: 2 RouteStep sqrt(g yFull), at least 200 ft."""
    y = t(0.5, 50.0)
    expected = [200.0, 2 * ROUTE * math.sqrt(G * 50.0)]
    assert equivalent_length(y, ROUTE).tolist() == pytest.approx(expected, rel=1e-15)


def test_side_orifice_weir_regime_free():
    """Inlet partly submerged (f < 1), outlet below the crest: ``q = cWeir f^1.5`` with
    ``cWeir = Cd sqrt(yFull/2) A sqrt(2g)``, ``dqdh = 1.5 q/(f hCrit)``, ``hCrit = yFull``
    (link.c:1795-1807, 1961-1964), class DN_CRITICAL, surface area ``W(f yFull) L``
    (link.c:1913-1918)."""
    heads = [CREST + 0.1 * YF, CREST + 0.5 * YF, CREST + 0.9 * YF]
    out = run_orifice(rect_orifice(), heads, [Z2 + 0.05] * 3)
    c_weir = 0.65 * math.sqrt(YF / 2) * AREA * math.sqrt(2 * G)
    for i, h in enumerate(heads):
        f = (h - CREST) / YF
        q = c_weir * f ** 1.5
        assert out.flow[i].item() == pytest.approx(q, rel=1e-13)
        assert out.dqdh[i].item() == pytest.approx(1.5 * q / (f * YF), rel=1e-13)
        assert out.surf_area[i].item() == pytest.approx(WF * LENGTH, rel=1e-13)
        assert out.depth[i].item() == pytest.approx(f * YF, rel=1e-13)
    assert out.flow_class.tolist() == [DN_CRITICAL] * 3


def test_side_orifice_weir_regime_submerged_villemonte():
    """f < 1 and the outlet above the crest: SWMM multiplies by Villemonte's ``(1 -
    r^1.5)^0.385``, ``r = (h2 - crest)/(h1 - crest)`` (link.c:1922-1926), not in dqdh;
    class SUBCRITICAL (crest below h2)."""
    h1 = CREST + 0.8 * YF
    h2s = [CREST + 0.1 * YF, CREST + 0.4 * YF, CREST + 0.7 * YF]
    out = run_orifice(rect_orifice(), [h1] * 3, h2s)
    c_weir = 0.65 * math.sqrt(YF / 2) * AREA * math.sqrt(2 * G)
    q_free = c_weir * 0.8 ** 1.5
    for i, h2 in enumerate(h2s):
        r = (h2 - CREST) / (h1 - CREST)
        assert out.flow[i].item() == pytest.approx(q_free * (1 - r ** 1.5) ** 0.385, rel=1e-13)
        assert out.dqdh[i].item() == pytest.approx(1.5 * q_free / (0.8 * YF), rel=1e-13)
    assert out.flow_class.tolist() == [SUBCRITICAL] * 3


def test_side_orifice_orifice_regime():
    """Inlet above the crown (f = 1): ``q = cOrif sqrt(head)``, head ``h1 - midpoint``
    while the outlet is below the midpoint, else ``h1 - h2`` (link.c:1887-1889, 1968-1971);
    a closed rectangle's width at its full height is 0, so no surface area (xsect.c:1006)."""
    crown = CREST + YF
    mid = CREST + YF / 2
    cases = [(crown + 0.3, Z2 + 0.1, crown + 0.3 - mid), (crown + 0.6, mid - 0.01,
                                                           crown + 0.6 - mid),
             (crown + 1.0, mid + 0.2, crown + 1.0 - (mid + 0.2))]
    out = run_orifice(rect_orifice(), [c[0] for c in cases], [c[1] for c in cases])
    for i, (_, _, head) in enumerate(cases):
        q = C_ORIF * math.sqrt(head)
        assert out.flow[i].item() == pytest.approx(q, rel=1e-13)
        assert out.dqdh[i].item() == pytest.approx(q / (2 * head), rel=1e-13)
    assert out.surf_area.tolist() == [0.0, 0.0, 0.0]
    assert out.flow_class.tolist() == [DN_CRITICAL, SUBCRITICAL, SUBCRITICAL]


def test_orifice_reverse_flow_takes_node_1s_crest():
    """h2 > h1: the heads swap, the flow is negative, the crest stays node 1's invert plus
    the offset (link.c:1850-1856, 1877), and the class is UP_CRITICAL when the crest is
    above the (swapped) downstream head (link.c:1907)."""
    heads2 = [CREST + 0.2 * YF, CREST + 0.6 * YF, CREST + YF + 0.5]
    out = run_orifice(rect_orifice(), [Z1 + 0.01] * 3, heads2)
    c_weir = 0.65 * math.sqrt(YF / 2) * AREA * math.sqrt(2 * G)
    expected = [-c_weir * 0.2 ** 1.5, -c_weir * 0.6 ** 1.5,
                -C_ORIF * math.sqrt(heads2[2] - (CREST + YF / 2))]
    assert out.flow.tolist() == pytest.approx(expected, rel=1e-13)
    assert out.flow_class.tolist() == [UP_CRITICAL] * 3


def test_bottom_orifice_regimes():
    """A circular bottom orifice: ``hCrit = Cd/0.414 * D/4`` (link.c:1786, 1792); head is
    ``h1 - crest`` while the outlet is below the crest, else ``h1 - h2``
    (link.c:1868-1870); weir flow ``cWeir f^1.5`` below hCrit with ``cWeir = Cd sqrt(hCrit)
    A sqrt(2g)``, orifice flow above; its surface area is the opening's (link.c:1922)."""
    d = 0.25 / LCF
    opening = Opening.from_si([CIRCULAR], [0.25], [0.25])
    orf = OrificeSet.build(opening, [True], [0.6], [False], ROUTE)
    area = math.pi / 4 * d * d
    h_crit = 0.6 / 0.414 * d / 4
    c_orif = 0.6 * area * math.sqrt(2 * G)
    c_weir = 0.6 * math.sqrt(h_crit) * area * math.sqrt(2 * G)
    heads = [CREST + 0.3 * h_crit, CREST + 0.8 * h_crit, CREST + 3.0 * h_crit]
    out = run_orifice(orf, heads, [Z2] * 3)
    expected = [c_weir * 0.3 ** 1.5, c_weir * 0.8 ** 1.5, c_orif * math.sqrt(3.0 * h_crit)]
    assert out.flow.tolist() == pytest.approx(expected, rel=1e-12)
    assert out.surf_area.tolist() == pytest.approx([area] * 3, rel=1e-12)
    # outlet above the crest: head h1 - h2 (no submergence factor once f = 1)
    sub = run_orifice(orf, [CREST + 5 * h_crit], [CREST + 2 * h_crit])
    assert sub.flow.item() == pytest.approx(c_orif * math.sqrt(3 * h_crit), rel=1e-12)


def test_gated_orifice_takes_the_armco_head_loss():
    """link.c:1974-2002: ``hLoss = (4/g) v^2 exp(-1.15 v/sqrt(head))`` with v the ungated
    flow over the opening area; the head (orifice regime) or f (weir regime) is reduced
    by it and the flow recomputed; the gate also closes reverse flow."""
    orf = rect_orifice(gated=True)
    crown = CREST + YF
    head = crown + 0.4 - (CREST + YF / 2)
    q0 = C_ORIF * math.sqrt(head)
    v = q0 / AREA
    loss = 4 / G * v * v * math.exp(-1.15 * v / math.sqrt(head))
    q_orif = C_ORIF * math.sqrt(head - loss)
    c_weir = 0.65 * math.sqrt(YF / 2) * AREA * math.sqrt(2 * G)
    f = 0.5
    qw = c_weir * f ** 1.5
    vw = qw / AREA
    loss_w = 4 / G * vw * vw * math.exp(-1.15 * vw / math.sqrt(f * YF))
    f_g = f - loss_w / YF
    out = run_orifice(orf, [crown + 0.4, CREST + f * YF, Z1 + 0.01],
                      [Z2, Z2, CREST + 0.5 * YF], block_reverse=t(1, 1, 1) > 0)
    assert out.flow[0].item() == pytest.approx(q_orif, rel=1e-13)
    assert out.dqdh[0].item() == pytest.approx(q_orif / (2 * (head - loss)), rel=1e-13)
    assert out.flow[1].item() == pytest.approx(c_weir * f_g ** 1.5, rel=1e-13)
    assert out.flow[2].item() == 0.0 and out.flow_class[2].item() == DRY


def test_weir_regime_meets_orifice_regime_at_the_crown_when_free():
    """By construction of cWeir (head ``yFull/2`` at f = 1), the side orifice's two
    regimes meet at the crown when the outlet is below the crest: SWMM's switch is
    continuous there. With the outlet above the crest it is not: just below the crown
    SWMM applies Villemonte, at the crown it takes ``h1 - midpoint`` without it
    (link.c:1887, 1922): that is SWMM's branch choice, and noodl takes the same."""
    crown = CREST + YF
    eps = 1e-9
    free = run_orifice(rect_orifice(), [crown - eps, crown], [Z2, Z2])
    assert free.flow[0].item() == pytest.approx(free.flow[1].item(), rel=1e-8)
    h2 = CREST + 0.3 * YF
    sub = run_orifice(rect_orifice(), [crown - eps, crown], [h2, h2])
    r = (h2 - CREST) / (crown - eps - CREST)
    c_weir = 0.65 * math.sqrt(YF / 2) * AREA * math.sqrt(2 * G)
    assert sub.flow[0].item() == pytest.approx(
        c_weir * ((crown - eps - CREST) / YF) ** 1.5 * (1 - r ** 1.5) ** 0.385, rel=1e-12)
    assert sub.flow[1].item() == pytest.approx(C_ORIF * math.sqrt(YF / 2), rel=1e-13)
    assert sub.flow[1].item() > sub.flow[0].item()


# --------------------------------------------------------------------------------- weirs
def make_weir(kind, shape, height, width, cd, *, slopes=None, cd2=0.0, end_con=0.0,
              surcharge=True, gated=False):
    opening = Opening.from_si([shape], [height], [width],
                              None if slopes is None else [slopes])
    return WeirSet.build(opening, [kind], [cd], [cd2], [end_con], [surcharge], [gated])


def run_weir(weir, h1, h2, **kw):
    n = len(h1)
    return weir_flow(weir, t(*h1), t(*h2), t(*[Z1] * n), t(*[Z2] * n), t(*[CREST] * n), **kw)


W_H, W_L = 1.0, 0.5                # transverse weir opening height and length (m)


def test_transverse_weir_with_end_contractions():
    """link.c:2366-2370, 2409-2413: ``q = Cd (L - 0.1 n h) h^1.5`` in m and m3/s, divided by
    M3perFT3 for cfs; ``dqdh = 1.5 q/h`` (link.c:2503); class DN_CRITICAL."""
    weir = make_weir(TRANSVERSE_WEIR, RECT_OPEN, W_H, W_L, 1.84, end_con=2)
    heads = [CREST + 0.05 / LCF, CREST + 0.3 / LCF, CREST + 0.9 / LCF]
    out = run_weir(weir, heads, [Z2] * 3)
    for i, h1 in enumerate(heads):
        h_m = (h1 - CREST) * LCF
        q = 1.84 * (W_L - 0.1 * 2 * h_m) * h_m ** 1.5 / M3
        assert out.flow[i].item() == pytest.approx(q, rel=1e-13)
        assert out.dqdh[i].item() == pytest.approx(1.5 * q / (h1 - CREST), rel=1e-13)
        assert out.depth[i].item() == pytest.approx(h1 - CREST, rel=1e-13)
    assert out.flow_class.tolist() == [DN_CRITICAL] * 3
    assert out.surf_area.tolist() == [0.0, 0.0, 0.0]          # dynwave.c:513-514


def test_end_contractions_never_make_a_negative_length():
    """link.c:2369: the reduced length is floored at 0."""
    weir = make_weir(TRANSVERSE_WEIR, RECT_OPEN, 2.0, 0.1, 1.84, end_con=2)
    assert run_weir(weir, [CREST + 1.0 / LCF], [Z2]).flow.item() == 0.0


def test_sideflow_weir_forward_and_reverse():
    """link.c:2374-2387: forward ``Cd L^0.83 h^1.67``, reverse the transverse formula;
    dqdh 1.67 q/h forward, 1.5 q/h reverse (link.c:2506-2507). Weirs flow forward only
    where h1 > h2 strictly (link.c:2241)."""
    weir = make_weir(1, RECT_OPEN, W_H, W_L, 1.7)
    heads = [CREST + 0.1 / LCF, CREST + 0.4 / LCF, CREST + 0.7 / LCF]
    fwd = run_weir(weir, heads, [Z2] * 3)
    back = run_weir(weir, [Z1] * 3, heads)
    for i, h in enumerate(heads):
        h_m = (h - CREST) * LCF
        q_f = 1.7 * W_L ** 0.83 * h_m ** 1.67 / M3
        q_b = 1.7 * W_L * h_m ** 1.5 / M3
        assert fwd.flow[i].item() == pytest.approx(q_f, rel=1e-13)
        assert fwd.dqdh[i].item() == pytest.approx(1.67 * q_f / (h - CREST), rel=1e-13)
        assert back.flow[i].item() == pytest.approx(-q_b, rel=1e-13)
        assert back.dqdh[i].item() == pytest.approx(1.5 * q_b / (h - CREST), rel=1e-13)
    assert back.flow_class.tolist() == [UP_CRITICAL] * 3


def test_vnotch_weir():
    """link.c:2390-2391: ``q = Cd s h^2.5``, ``s = sBot = wMax/yFull/2`` (xsect.c:513);
    dqdh 2.5 q/h (link.c:2510)."""
    weir = make_weir(VNOTCH_WEIR, TRIANGULAR, 0.6, 1.2, 1.38)
    s = 1.2 / 0.6 / 2
    heads = [CREST + 0.05 / LCF, CREST + 0.25 / LCF, CREST + 0.55 / LCF]
    out = run_weir(weir, heads, [Z2] * 3)
    for i, h in enumerate(heads):
        q = 1.38 * s * ((h - CREST) * LCF) ** 2.5 / M3
        assert out.flow[i].item() == pytest.approx(q, rel=1e-13)
        assert out.dqdh[i].item() == pytest.approx(2.5 * q / (h - CREST), rel=1e-13)


def test_trapezoidal_weir_with_end_sections():
    """link.c:2393-2398: ``q1 = Cd1 b h^1.5`` (b the bottom width, W(0)) and ``q2 = Cd2 s
    h^2.5`` (s the mean side slope); dqdh ``1.5 q1/h + 2.5 q2/h`` (link.c:2513)."""
    weir = make_weir(TRAPEZOIDAL_WEIR, TRAPEZOIDAL, 0.8, 0.6, 1.84, slopes=(0.5, 1.5),
                     cd2=1.38)
    heads = [CREST + 0.1 / LCF, CREST + 0.4 / LCF, CREST + 0.7 / LCF]
    out = run_weir(weir, heads, [Z2] * 3)
    for i, h in enumerate(heads):
        h_m = (h - CREST) * LCF
        q1 = 1.84 * 0.6 * h_m ** 1.5 / M3
        q2 = 1.38 * 1.0 * h_m ** 2.5 / M3
        assert out.flow[i].item() == pytest.approx(q1 + q2, rel=1e-13)
        assert out.dqdh[i].item() == pytest.approx((1.5 * q1 + 2.5 * q2) / (h - CREST),
                                                   rel=1e-13)


@pytest.mark.parametrize("kind,shape,width,power", [
    (TRANSVERSE_WEIR, RECT_OPEN, W_L, 1.5), (1, RECT_OPEN, W_L, 5.0 / 3.0),
    (VNOTCH_WEIR, TRIANGULAR, 1.2, 2.5)])
def test_weir_villemonte_submergence(kind, shape, width, power):
    """Outlet above the crest: q times ``(1 - r^p)^0.385``, p by weir type
    (link.c:2218-2221, 2311-2316); dqdh is the free weir's."""
    weir = make_weir(kind, shape, W_H, width, 1.5)
    h1 = CREST + 0.6 / LCF
    h2s = [CREST + 0.1 / LCF, CREST + 0.3 / LCF, CREST + 0.55 / LCF]
    free = run_weir(weir, [h1], [Z2])
    out = run_weir(weir, [h1] * 3, h2s)
    for i, h2 in enumerate(h2s):
        r = (h2 - CREST) / (h1 - CREST)
        assert out.flow[i].item() == pytest.approx(
            free.flow.item() * (1 - r ** power) ** 0.385, rel=1e-13)
        assert out.dqdh[i].item() == pytest.approx(free.dqdh.item(), rel=1e-13)
    assert out.flow_class.tolist() == [SUBCRITICAL] * 3


def test_weir_above_its_opening_surcharges_as_an_orifice():
    """h1 at or above the crown with CanSurcharge: ``q = cSurcharge sqrt(head)``, head ``h1 -
    midpoint`` (``h1 - h2`` once h2 is above the midpoint), ``cSurcharge = q(yFull) /
    sqrt(yFull/2)`` (link.c:2154-2160, 2289-2299, 2446); no Villemonte; dqdh q/(2 head)."""
    weir = make_weir(TRANSVERSE_WEIR, RECT_OPEN, W_H, W_L, 1.84)
    yf = W_H / LCF
    c_sur = 1.84 * W_L * W_H ** 1.5 / M3 / math.sqrt(yf / 2)
    crown, mid = CREST + yf, CREST + yf / 2
    cases = [(crown, Z2, crown - mid), (crown + 0.5, mid - 0.1, crown + 0.5 - mid),
             (crown + 0.8, mid + 0.3, crown + 0.8 - (mid + 0.3))]
    out = run_weir(weir, [c[0] for c in cases], [c[1] for c in cases])
    for i, (_, _, head) in enumerate(cases):
        q = c_sur * math.sqrt(head)
        assert out.flow[i].item() == pytest.approx(q, rel=1e-13)
        assert out.dqdh[i].item() == pytest.approx(q / (2 * head), rel=1e-13)
        assert out.depth[i].item() == pytest.approx(yf, rel=1e-13)


def test_weir_that_cannot_surcharge_caps_its_head_at_the_opening():
    """CanSurcharge NO: the head is the opening height above the crown (link.c:2303), the
    Villemonte ratio still uses ``h1 - crest`` (link.c:2313)."""
    weir = make_weir(TRANSVERSE_WEIR, RECT_OPEN, W_H, W_L, 1.84, surcharge=False)
    yf = W_H / LCF
    q_cap = 1.84 * W_L * W_H ** 1.5 / M3
    h1 = CREST + yf + 0.4
    h2s = [Z2, CREST + 0.2, CREST + 0.9 * yf]
    out = run_weir(weir, [h1] * 3, h2s)
    expected = [q_cap] + [q_cap * (1 - ((h2 - CREST) / (h1 - CREST)) ** 1.5) ** 0.385
                          for h2 in h2s[1:]]
    assert out.flow.tolist() == pytest.approx(expected, rel=1e-13)


def test_weir_surcharge_switch_is_continuous_when_free():
    """At the crown with the outlet below the crest, SWMM's two weir branches agree
    (cSurcharge is fitted there, link.c:2154-2160)."""
    weir = make_weir(TRANSVERSE_WEIR, RECT_OPEN, W_H, W_L, 1.84)
    crown = CREST + W_H / LCF
    out = run_weir(weir, [crown - 1e-9, crown], [Z2, Z2])
    assert out.flow[0].item() == pytest.approx(out.flow[1].item(), rel=1e-8)


def test_gated_weir_takes_the_armco_head_loss():
    """link.c:2416-2426: ``v = q/A(open)``, the head reduced by ``(4/g) v^2 exp(-1.15
    v/sqrt(head))`` and the flow recomputed; dqdh at the reduced head."""
    weir = make_weir(TRANSVERSE_WEIR, RECT_OPEN, W_H, W_L, 1.84, gated=True)
    head = 0.4 / LCF
    q0 = 1.84 * W_L * 0.4 ** 1.5 / M3
    v = q0 / (head * W_L / LCF)
    head_g = head - 4 / G * v * v * math.exp(-1.15 * v / math.sqrt(head))
    q = 1.84 * W_L * (head_g * LCF) ** 1.5 / M3
    out = run_weir(weir, [CREST + head], [Z2])
    assert out.flow.item() == pytest.approx(q, rel=1e-13)
    assert out.dqdh.item() == pytest.approx(1.5 * q / head_g, rel=1e-13)


# ------------------------------------------------------------------------------- outlets
def run_outlet(out, h1, h2, **kw):
    n = len(h1)
    return outlet_flow(out, t(*h1), t(*h2), t(*[Z1] * n), t(*[Z2] * n), t(*[CREST] * n), **kw)


def test_functional_outlet_on_depth_and_on_head():
    """link.c:2648-2650, 2690-2691: ``q = C1 h^C2`` (m, m3/s, divided by the CMS flow factor
    0.02832, not M3perFT3), h the depth above the crest (/DEPTH) or ``h1 - max(h2,
    crest)`` (/HEAD)."""
    depth = OutletSet.build([("functional", 0.3, 0.6)], [False], [False])
    head = OutletSet.build([("functional", 0.3, 0.6)], [True], [False])
    h1s = [CREST + 0.1 / LCF, CREST + 0.5 / LCF, CREST + 1.2 / LCF]
    h2 = CREST + 0.2 / LCF
    by_depth = run_outlet(depth, h1s, [Z2] * 3)
    by_head = run_outlet(head, h1s[1:], [h2] * 2)
    for i, h1 in enumerate(h1s):
        q = 0.3 * ((h1 - CREST) * LCF) ** 0.6 / QCF
        assert by_depth.flow[i].item() == pytest.approx(q, rel=1e-13)
    for i, h1 in enumerate(h1s[1:]):
        assert by_head.flow[i].item() == pytest.approx(
            0.3 * ((h1 - h2) * LCF) ** 0.6 / QCF, rel=1e-13)
    assert by_depth.dqdh.tolist() == [0.0, 0.0, 0.0]               # SWMM sets none


def test_tabular_outlet_holds_its_end_flows():
    """table.c:395-426: the first flow at or below the first head, linear inside, the last
    flow beyond the last head."""
    out = OutletSet.build([("tabular", (0.1, 0.2, 0.5, 1.0), (0.002, 0.01, 0.03, 0.05))],
                          [False], [False])
    depths = [0.05, 0.35, 1.7]
    res = run_outlet(out, [CREST + d / LCF for d in depths], [Z2] * 3)
    expected = [0.002, 0.01 + (0.35 - 0.2) * (0.03 - 0.01) / 0.3, 0.05]
    assert res.flow.tolist() == pytest.approx([q / QCF for q in expected], rel=1e-12)


def test_table_lookup_with_padded_rows():
    """Two curves of different lengths, padded: each row is SWMM's own table_lookup."""
    out = OutletSet.build([("tabular", (0.0, 1.0), (0.0, 2.0)),
                           ("tabular", (0.0, 0.5, 1.0, 2.0), (1.0, 2.0, 4.0, 4.5))],
                          [False, False], [False, False])
    x = t(0.25, 1.5)
    got = table_lookup(x, out.curve_x, out.curve_y)
    assert got.tolist() == pytest.approx([0.5, 4.25], rel=1e-15)
    got = table_lookup(t(3.0, 3.0), out.curve_x, out.curve_y)
    assert got.tolist() == [2.0, 4.5]
    got = table_lookup(t(1.0, 0.5), out.curve_x, out.curve_y)       # at a point: x <= x2
    assert got.tolist() == [2.0, 2.0]


def test_outlet_reverse_flow_and_its_slope():
    """Reverse flow (h2 > h1) is negative with the upstream depth taken at node 2
    (link.c:2636-2643); `outlet_dqdh` is the central difference in the upstream head, >= 0
    both ways."""
    out = OutletSet.build([("functional", 0.3, 0.6)], [False], [False])
    h = CREST + 0.5 / LCF
    back = run_outlet(out, [Z1 + 0.01], [h])
    assert back.flow.item() == pytest.approx(-0.3 * 0.5 ** 0.6 / QCF, rel=1e-13)
    step = FUDGE
    for h1, h2, sign in ((h, Z2, 1.0), (Z1 + 0.01, h, -1.0)):
        g = outlet_dqdh(out, t(h1), t(h2), t(Z1), t(Z2), t(CREST))
        hu = h1 if sign > 0 else h2
        expected = 0.3 * (((hu + step - CREST) * LCF) ** 0.6
                          - ((hu - step - CREST) * LCF) ** 0.6) / QCF / (2 * step)
        assert g.item() == pytest.approx(expected, rel=1e-9)
        assert g.item() > 0.0


# ------------------------------------------------------------------- below the crest, gates
def test_structures_below_crest_carry_nothing():
    """At or below the crest (to within SWMM's FUDGE, 1e-4 ft of head), or with no water
    upstream, every regulator is closed: flow 0, dqdh 0, class DRY (link.c:1895-1902,
    2267-2271, 2654-2660); a closed orifice still shares SWMM's dry area ``FUDGE L``
    (link.c:1900)."""
    heads = [Z1 + 0.5 * OFFSET, CREST, CREST + 0.9 * FUDGE]
    lows = [Z2, Z2, Z2]
    devices = {
        "side orifice": run_orifice(rect_orifice(), heads, lows),
        "bottom orifice": run_orifice(rect_orifice(bottom=True), heads, lows),
        "transverse weir": run_weir(make_weir(TRANSVERSE_WEIR, RECT_OPEN, 1.0, 0.5, 1.84),
                                    heads, lows),
        "v-notch weir": run_weir(make_weir(VNOTCH_WEIR, TRIANGULAR, 0.6, 1.2, 1.38),
                                 heads, lows),
        "functional outlet": run_outlet(
            OutletSet.build([("functional", 0.3, 0.6)], [False], [False]), heads, lows),
        "tabular outlet": run_outlet(
            OutletSet.build([("tabular", (0.0, 1.0), (0.01, 0.05))], [False], [False]),
            heads, lows),
    }
    for name, out in devices.items():
        assert out.flow.tolist() == [0.0, 0.0, 0.0], name
        assert out.dqdh.tolist() == [0.0, 0.0, 0.0], name
        assert out.flow_class.tolist() == [DRY] * 3, name
    assert devices["side orifice"].surf_area.tolist() == pytest.approx([FUDGE * LENGTH] * 3)
    # equal heads above the crest: no head across, no flow (Villemonte ratio 1 for weirs)
    equal = [CREST + 0.3, CREST + 0.6]
    assert run_orifice(rect_orifice(), equal, equal).flow.tolist() == [0.0, 0.0]
    assert run_weir(make_weir(TRANSVERSE_WEIR, RECT_OPEN, 1.0, 0.5, 1.84), equal,
                    equal).flow.abs().max().item() < 1e-30
    # an outlet with no upstream depth (link.c:2654): head above the crest but y1 <= FUDGE
    # cannot happen with a crest above the invert, so a zero-offset outlet tests it
    out = OutletSet.build([("functional", 0.3, 0.6)], [False], [False])
    dry = outlet_flow(out, t(Z1 + 0.5 * FUDGE), t(Z2), t(Z1), t(Z2), t(Z1 - 1.0))
    assert dry.flow.item() == 0.0


def test_flap_gates_close_reverse_flow_and_gated_outfalls_their_side():
    """link.c::link_setFlapGate: a regulator's own gate (or a gated outfall at node 2)
    closes flow from node 2 to node 1; a gated outfall at node 1 closes forward flow."""
    orf = rect_orifice()
    h = CREST + 0.5 * YF
    fwd = run_orifice(orf, [h, h], [Z2, Z2], block_forward=t(0, 1) > 0)
    assert fwd.flow[0].item() > 0.0 and fwd.flow[1].item() == 0.0
    rev = run_weir(make_weir(TRANSVERSE_WEIR, RECT_OPEN, 1.0, 0.5, 1.84), [Z1, Z1],
                   [CREST + 0.3, CREST + 0.3], block_reverse=t(0, 1) > 0)
    assert rev.flow[0].item() < 0.0 and rev.flow[1].item() == 0.0
    assert rev.flow_class[1].item() == DRY


# ----------------------------------------------------------------------------- gradients
def test_gradients_are_finite_on_every_branch():
    """Both-branches-safe: the heads' gradient is finite on every branch, closed devices,
    the crest and crown, equal heads and full submergence included."""
    crown = CREST + YF
    h1 = t(Z1, CREST, CREST + 0.5 * YF, crown, crown + 0.5, CREST + 0.3, CREST + 0.3)
    h2 = t(Z2, Z2, CREST + 0.2 * YF, Z2, crown + 0.2, CREST + 0.3, Z2)
    n = len(h1)
    z1, z2, crest = t(*[Z1] * n), t(*[Z2] * n), t(*[CREST] * n)
    devices = [
        lambda a, b: orifice_flow(rect_orifice(), a, b, z1, z2, crest),
        lambda a, b: orifice_flow(rect_orifice(bottom=True, gated=True), a, b, z1, z2, crest),
        lambda a, b: weir_flow(make_weir(TRANSVERSE_WEIR, RECT_OPEN, 0.2, 0.5, 1.84,
                                         end_con=2, gated=True), a, b, z1, z2, crest),
        lambda a, b: weir_flow(make_weir(TRAPEZOIDAL_WEIR, TRAPEZOIDAL, 0.2, 0.5, 1.84,
                                         slopes=(1, 1), cd2=1.4), a, b, z1, z2, crest),
        lambda a, b: weir_flow(make_weir(1, RECT_OPEN, 0.2, 0.5, 1.7), a, b, z1, z2, crest),
        lambda a, b: outlet_flow(OutletSet.build([("functional", 0.3, 0.6)] * n,
                                                 [False] * n, [False] * n), a, b, z1, z2,
                                 crest),
        lambda a, b: outlet_flow(OutletSet.build([("tabular", (0.0, 1.0), (0.0, 0.05))] * n,
                                                 [True] * n, [False] * n), a, b, z1, z2,
                                 crest),
    ]
    for i, fn in enumerate(devices):
        a = h1.clone().requires_grad_(True)
        b = h2.clone().requires_grad_(True)
        res = fn(a, b)
        (res.flow.sum() + res.dqdh.sum() + res.surf_area.sum()).backward()
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all(), i
