"""SWMM 5.2.4's flow regulators under dynamic wave -- orifices, weirs and outlets -- as pure
functions of one pass's end heads, in SWMM's own feet units (ft, ft2, cfs; g = 32.2).

`orifice_flow`, `weir_flow` and `outlet_flow` transcribe `link.c`'s ``orifice_getInflow``,
``weir_getInflow`` and ``outlet_getInflow`` (with ``orifice_getFlow``, ``weir_getFlow``,
``weir_getOrificeFlow``, ``weir_getdqdh`` and ``outlet_getFlow``) for a fully open device
(SWMM's ``Link.setting`` 1: control rules are not read, so the setting never changes). Each
returns the flow (positive from the device's first node to its second), SWMM's ``dqdh``
where SWMM computes one (orifices and weirs; `outlet_dqdh` is a central difference for
outlets, which have none), SWMM's flow class and, for orifices, the equivalent surface area
``Orifice.surfArea`` that `dynwave.c::findNonConduitSurfArea` (dynwave.c:503-524) shares
between the two end nodes. Weirs and outlets contribute no surface area (dynwave.c:513-514).

The heads are SWMM's node heads ``newDepth + invertElev`` of the previous trial (the caller
clamps a depth below 0 to 0, as SWMM's depths never are negative). SWMM's under-relaxation
of these flows (dynwave.c:443-447) is iteration only and is not modelled.

Units follow SWMM: lengths in ft, flows in cfs; a weir's discharge coefficient is in the
input's SI units (m^0.5/s, the formula evaluated in m and m3/s, then divided by SWMM's
``M3perFT3``, link.c:2409-2413), an outlet's rating in m and m3/s (divided by ``UCF(FLOW)``,
the CMS factor `QCF`, link.c:2672-2692). Every `where` is both-branches-safe: an argument
that would be invalid on the branch not taken is replaced by a harmless value first.

Every source reference is to SWMM 5.2.4 ``src/solver``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import torch

from noodl.apps.sewer.dynwave_terms import (
    DN_CRITICAL,
    DRY,
    FUDGE,
    SUBCRITICAL,
    UP_CRITICAL,
)
from noodl.apps.sewer.xsect_tables import (
    GRAVITY_FT,
    LCF,
    QCF,
    area_of_depth,
    width_of_depth,
)

Tensor = torch.Tensor
F64 = torch.float64

#: SWMM's ``M3perFT3`` (consts.h:86): a weir's SI flow is divided by it (link.c:2409-2413),
#: NOT by the CMS flow factor `QCF` (0.02832) an outlet uses.
M3_PER_FT3 = 0.028317
#: SWMM's ``TINY`` (consts.h:33).
TINY = 1.0e-6
#: SWMM's shortest regulator equivalent length (link.c:1722-1723, 2150-2151), ft.
MIN_EQUIVALENT_LENGTH = 200.0
#: Opening shapes (SWMM's xsect types used by regulators): orifices take CIRCULAR or
#: RECT_CLOSED (link.c:1707-1708); TRANSVERSE and SIDEFLOW weirs RECT_OPEN, V-NOTCH weirs
#: TRIANGULAR, TRAPEZOIDAL weirs TRAPEZOIDAL (link.c:2115-2133).
CIRCULAR, RECT_CLOSED, RECT_OPEN, TRIANGULAR, TRAPEZOIDAL = range(5)
#: Weir types, in SWMM's order (`enums.h` ``WeirType``).
TRANSVERSE_WEIR, SIDEFLOW_WEIR, VNOTCH_WEIR, TRAPEZOIDAL_WEIR = range(4)
#: Villemonte exponents per weir type (link.c:2218-2221).
_WEIR_POWER = (1.5, 5.0 / 3.0, 2.5, 1.5)
_ARMCO = 4.0 / GRAVITY_FT        # link.c:1981, 2422: hLoss = (4/g) v^2 exp(-1.15 v/sqrt(h))


def _where(cond: Tensor, a: Tensor, b) -> Tensor:
    return torch.where(cond, a, torch.as_tensor(b, dtype=F64) if not isinstance(b, Tensor)
                       else b)


def _sqrt(x: Tensor) -> Tensor:
    """``sqrt(max(x, 0))``, exactly 0 at and below 0 with a zero (not infinite) gradient."""
    pos = x > 0.0
    return torch.where(pos, torch.sqrt(torch.where(pos, x, torch.ones_like(x))),
                       torch.zeros_like(x))


def _villemonte(ratio: Tensor, power) -> Tensor:
    """``(1 - ratio^power)^0.385`` (link.c:1925, 2304), ratio clamped to [0, 1]: exactly 0 at
    ratio 1, with a zero (not infinite) gradient there."""
    r = torch.clamp(ratio, 0.0, 1.0)
    x = 1.0 - r ** power
    pos = x > 0.0
    return torch.where(pos, torch.where(pos, x, torch.ones_like(x)) ** 0.385,
                       torch.zeros_like(x))


@dataclass(frozen=True)
class Opening:
    """A regulator's opening (`xsect.c::xsect_setParams`, xsect.c:216-520), per device, in
    ft: ``shape`` (long, `CIRCULAR` ... `TRAPEZOIDAL`), ``y_full`` the opening height,
    ``w_max`` its top width, ``bottom`` a trapezoid's bottom width (``yBot``, 0 for the
    others), ``side_slope`` SWMM's ``sBot`` (a triangle's ``wMax/yFull/2``, a trapezoid's
    mean side slope, horizontal per vertical; 0 for the others) and ``a_full`` the full area
    (ft2)."""

    shape: Tensor
    y_full: Tensor
    w_max: Tensor
    bottom: Tensor
    side_slope: Tensor
    a_full: Tensor

    @classmethod
    def from_si(cls, shape, height, width, slopes=None) -> Opening:
        """From SI inputs: ``shape`` codes, ``height`` (m, Geom1: a circle's diameter),
        ``width`` (m, Geom2: a rectangle's width, a triangle's top width, a trapezoid's
        bottom width; ignored for a circle) and ``slopes`` (left, right side slopes of a
        trapezoid, horizontal per vertical; Geom3, Geom4)."""
        shape = torch.as_tensor(shape, dtype=torch.long)
        y_full = torch.as_tensor(height, dtype=F64) / LCF
        width = torch.as_tensor(width, dtype=F64) / LCF
        if slopes is None:
            slopes = torch.zeros(shape.shape + (2,), dtype=F64)
        slopes = torch.as_tensor(slopes, dtype=F64).reshape(shape.shape + (2,))
        zero = torch.zeros_like(y_full)
        is_tri = shape == TRIANGULAR
        is_trap = shape == TRAPEZOIDAL
        safe_y = torch.where(y_full > 0.0, y_full, torch.ones_like(y_full))
        side = torch.where(is_tri, width / safe_y / 2.0,                     # xsect.c:513
                           torch.where(is_trap, slopes.sum(-1) / 2.0, zero))  # xsect.c:491
        bottom = torch.where(is_trap, width, zero)                           # xsect.c:488
        w_max = torch.where(shape == CIRCULAR, y_full,                       # xsect.c:244
                            torch.where(is_trap, width + y_full * slopes.sum(-1),  # :497
                                        width))
        a_full = torch.where(
            shape == CIRCULAR, torch.pi / 4.0 * y_full * y_full,             # xsect.c:245
            torch.where(is_tri, y_full * y_full * side,                      # xsect.c:519
                        torch.where(is_trap, (bottom + side * y_full) * y_full,  # :499
                                    y_full * w_max)))                        # xsect.c:370, 382
        return cls(shape, y_full, w_max, bottom, side, a_full)

    def area(self, y: Tensor) -> Tensor:
        """`xsect_getAofY` (xsect.c:857-941): the flow area (ft2) at depth ``y`` (ft)."""
        y_pos = torch.clamp(y, min=0.0)
        circ = area_of_depth(y_pos, self.y_full, self.a_full)                # xsect.c:868
        rect = y_pos * self.w_max                                            # xsect.c:917, 923
        tri = y_pos * y_pos * self.side_slope                                # xsect.c:2255
        trap = (self.bottom + self.side_slope * y_pos) * y_pos               # xsect.c:2212
        a = torch.where(self.shape == CIRCULAR, circ,
                        torch.where(self.shape == TRIANGULAR, tri,
                                    torch.where(self.shape == TRAPEZOIDAL, trap, rect)))
        return torch.where(y <= 0.0, torch.zeros_like(a), a)                 # xsect.c:866

    def width(self, y: Tensor) -> Tensor:
        """`xsect_getWofY` (xsect.c:943-1027): the top width (ft) at depth ``y`` (ft); a
        closed rectangle's width is 0 at exactly its full height (xsect.c:1006-1007)."""
        y_pos = torch.clamp(y, min=0.0)
        circ = width_of_depth(y_pos, self.y_full)                           # xsect.c:953
        rect_closed = torch.where(y / self.y_full == 1.0, torch.zeros_like(y_pos),
                                  self.w_max.expand_as(y_pos))              # xsect.c:1006
        tri = 2.0 * self.side_slope * y_pos                                  # xsect.c:2265
        trap = self.bottom + 2.0 * y_pos * self.side_slope                   # xsect.c:2223
        return torch.where(
            self.shape == CIRCULAR, circ,
            torch.where(self.shape == RECT_CLOSED, rect_closed,
                        torch.where(self.shape == TRIANGULAR, tri,
                                    torch.where(self.shape == TRAPEZOIDAL, trap,
                                                self.w_max.expand_as(y_pos)))))  # :1014


def equivalent_length(y_full: Tensor, route_step: float) -> Tensor:
    """An orifice's or weir's equivalent length (ft): ``max(200, 2 RouteStep sqrt(g
    yFull))`` (link.c:1721-1724, 2149-2152), with the routing step `route_step` (s) and the
    opening height `y_full` (ft)."""
    length = 2.0 * route_step * torch.sqrt(GRAVITY_FT * y_full)
    return torch.clamp(length, min=MIN_EQUIVALENT_LENGTH)


class StructureFlow(NamedTuple):
    """One regulator evaluation, per device: ``flow`` (cfs, positive from node 1 to node
    2), ``dqdh`` (cfs/ft, >= 0; SWMM's ``Link.dqdh``), ``flow_class`` (long, SWMM's flow
    class: DRY, SUBCRITICAL, UP_/DN_CRITICAL), ``depth`` (ft, SWMM's ``Link.newDepth``) and
    ``surf_area`` (ft2; an orifice's ``Orifice.surfArea``, which `dynwave.c` halves between
    the ends; 0 for weirs and outlets)."""

    flow: Tensor
    dqdh: Tensor
    flow_class: Tensor
    depth: Tensor
    surf_area: Tensor


def _orient(h1: Tensor, h2: Tensor, z1: Tensor, z2: Tensor, strict: bool
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(forward, hu, hd, yu)``: the flow direction (``h1 >= h2``, weirs ``h1 > h2``,
    link.c:1846, 2241), the upstream and downstream heads and the upstream depth (ft)."""
    forward = h1 > h2 if strict else h1 >= h2
    hu = torch.where(forward, h1, h2)
    hd = torch.where(forward, h2, h1)
    yu = torch.where(forward, h1 - z1, h2 - z2)
    return forward, hu, hd, yu


def _flow_class(closed: Tensor, forward: Tensor, crest: Tensor, hd: Tensor) -> Tensor:
    """SWMM's regulator flow class (link.c:1886-1893, 2271-2276): DRY if closed, critical at
    the downstream end (seen from node 1) where the crest is above the downstream head."""
    critical = torch.where(forward, torch.full_like(hd, DN_CRITICAL, dtype=torch.long),
                           torch.full_like(hd, UP_CRITICAL, dtype=torch.long))
    cls = torch.where(crest > hd, critical, torch.full_like(critical, SUBCRITICAL))
    return torch.where(closed, torch.full_like(cls, DRY), cls)


def _gate(forward: Tensor, block_forward, block_reverse) -> Tensor:
    """`link.c::link_setFlapGate` (link.c:643-670) for a regulator (``direction`` 1): its own
    gate or a gated outfall at its second node closes it to reverse flow (`block_reverse`),
    a gated outfall at its first node to forward flow (`block_forward`)."""
    bf = torch.as_tensor(block_forward, dtype=torch.bool)
    br = torch.as_tensor(block_reverse, dtype=torch.bool)
    return torch.where(forward, bf, br)


# ---------------------------------------------------------------------------- orifices
@dataclass(frozen=True)
class OrificeSet:
    """Orifices (`link.c` ``Orifice``), per device: the `opening`, ``bottom`` (bool: a
    BOTTOM orifice, else SIDE), ``c_disch`` (-), ``gated`` (bool, own flap gate: the ARMCO
    head loss, link.c:1974-2002), ``length`` the equivalent length (ft), and SWMM's derived
    ``c_orif`` (ft^2.5/s), ``c_weir`` (cfs) and ``h_crit`` (ft) (`orifice_setSetting`,
    link.c:1729-1762, with the setting 1)."""

    opening: Opening
    bottom: Tensor
    c_disch: Tensor
    gated: Tensor
    length: Tensor
    c_orif: Tensor
    c_weir: Tensor
    h_crit: Tensor
    setting: Tensor | None = None

    @classmethod
    def build(cls, opening: Opening, bottom, c_disch, gated, route_step: float) -> OrificeSet:
        """SWMM's orifice constants at setting 1 (`orifice_validate`, `orifice_setSetting`,
        `orifice_getWeirCoeff`, link.c:1696-1808); `route_step` (s) sets the equivalent
        length."""
        bottom = torch.as_tensor(bottom, dtype=torch.bool)
        c_disch = torch.as_tensor(c_disch, dtype=F64)
        base = cls(opening, bottom, c_disch, torch.as_tensor(gated, dtype=torch.bool),
                   equivalent_length(opening.y_full, route_step), c_disch, c_disch, c_disch)
        return base._constants(torch.ones_like(opening.y_full))

    def _constants(self, setting: Tensor) -> OrificeSet:
        """This set with SWMM's setting-dependent constants at `setting` (``Link.setting``, a
        fraction of the opening height; > 0): the opening height ``h = setting yFull``, the
        effective discharge constant ``cOrif``, the weir constant ``cWeir`` and the critical
        height ``hCrit`` (`orifice_setSetting`, `orifice_getWeirCoeff`, link.c:1750-1808)."""
        opening, bottom, c_disch = self.opening, self.bottom, self.c_disch
        h = opening.y_full * setting                                         # link.c:1752
        f = opening.area(h) * torch.sqrt(torch.tensor(2.0 * GRAVITY_FT, dtype=F64))  # :1753
        c_orif = c_disch * f                                                 # link.c:1754
        a_over_l = torch.where(opening.shape == CIRCULAR, h / 4.0,           # link.c:1786
                               (h * opening.w_max) / (2.0 * (h + opening.w_max)))  # :1789-90
        h_crit = torch.where(bottom, c_disch / 0.414 * a_over_l, h)          # link.c:1792, 1800
        h_ref = torch.where(bottom, h_crit, h / 2.0)                         # link.c:1803
        c_weir = c_disch * torch.sqrt(h_ref) * f                             # link.c:1758, 1807
        return replace(self, c_orif=c_orif, c_weir=c_weir, h_crit=h_crit, setting=setting)

    def at_setting(self, setting: Tensor) -> OrificeSet:
        """This set at the per-device `setting` (0..1; 1 fully open). A setting of 0 closes the
        orifice (`link_getInflow` returns 0, link.c:550); its constants are then those of the
        open orifice, which `orifice_flow` does not use."""
        safe = torch.where(setting > 0.0, setting, torch.ones_like(setting))
        return replace(self._constants(safe), setting=setting)


def _orifice_get_flow(orf: OrificeSet, head: Tensor, f: Tensor) -> tuple[Tensor, Tensor]:
    """`orifice_getFlow` without the flap gate (link.c:1938-1971): ``(q, dqdh)`` for a head
    ``head`` (ft) and critical-height fraction ``f``; 0 where ``head == 0`` or ``f <= 0``."""
    off = (head == 0.0) | (f <= 0.0)                                         # link.c:1953
    weir = f < 1.0                                                           # link.c:1961
    f_safe = torch.where(off | ~weir, torch.ones_like(f), f)
    head_safe = torch.where(off | weir, torch.ones_like(head), head)
    hc_safe = torch.where(orf.h_crit > 0.0, orf.h_crit, torch.ones_like(orf.h_crit))
    q_weir = orf.c_weir * f_safe ** 1.5                                      # link.c:1963
    dq_weir = 1.5 * q_weir / (f_safe * hc_safe)                              # link.c:1964
    q_orif = orf.c_orif * torch.sqrt(head_safe)                              # link.c:1970
    dq_orif = q_orif / (2.0 * head_safe)                                     # link.c:1971
    q = torch.where(weir, q_weir, q_orif)
    dq = torch.where(weir, dq_weir, dq_orif)
    zero = torch.zeros_like(q)
    return torch.where(off, zero, q), torch.where(off, zero, dq)


def orifice_flow(orf: OrificeSet, h1: Tensor, h2: Tensor, z1: Tensor, z2: Tensor,
                 crest: Tensor, block_forward=False, block_reverse=False) -> StructureFlow:
    """`link.c::orifice_getInflow` under dynamic wave (link.c:1812-1934): heads ``h1``,
    ``h2`` (ft) at the orifice's first and second node, their inverts ``z1``, ``z2`` (ft),
    the crest elevation ``crest`` (ft: node 1's invert plus the crest offset, also for
    reverse flow, link.c:1865, 1877) and the flap-gate closures (`_gate`)."""
    forward, hu, hd, yu = _orient(h1, h2, z1, z2, strict=False)               # link.c:1846-1856
    yf_full = orf.opening.y_full
    shut = None                                              # a setting of 0 (link.c:550)
    if orf.setting is None:
        yf = yf_full
    else:
        shut = orf.setting <= 0.0
        yf = yf_full * torch.where(shut, torch.ones_like(orf.setting), orf.setting)  # :1876, 1910
    hc_safe = torch.where(orf.h_crit > 0.0, orf.h_crit, torch.ones_like(orf.h_crit))
    # bottom orifice (link.c:1862-1873)
    head_b = torch.where(hu < crest, torch.zeros_like(hu),
                         torch.where(hd > crest, hu - hd, hu - crest))       # link.c:1868-1870
    f_b = torch.clamp(head_b / hc_safe, max=1.0)                             # link.c:1872-1873
    f_b = torch.where(orf.h_crit > 0.0, f_b, torch.ones_like(f_b))           # Cd 0: head/0
    # side orifice (link.c:1876-1893)
    crown = crest + yf                                                       # link.c:1878
    mid = (crest + crown) / 2.0                                              # link.c:1879
    f_s = torch.where((hu < crown) & (crown > crest), (hu - crest) / yf,
                      torch.ones_like(hu))                                   # link.c:1882-1884
    head_s = torch.where(f_s < 1.0, hu - crest,
                         torch.where(hd < mid, hu - mid, hu - hd))           # link.c:1887-1889
    f = torch.where(orf.bottom, f_b, f_s)
    head = torch.where(orf.bottom, head_b, head_s)
    closed = (head <= FUDGE) | (yu <= FUDGE) | _gate(forward, block_forward, block_reverse)
    if shut is not None:
        closed = closed | shut
    cls = _flow_class(closed, forward, crest, hd)                            # link.c:1904-1910
    depth = torch.where(orf.bottom, yf.expand_as(f), yf * f)                 # link.c:1913-1925
    surf = torch.where(orf.bottom, orf.opening.area(yf).expand_as(f),
                       orf.opening.width(depth) * orf.length)                # link.c:1917-1927
    surf = torch.where(closed, (FUDGE * orf.length).expand_as(surf), surf)   # link.c:1898
    depth = torch.where(closed, torch.zeros_like(depth), depth)
    # orifice_getFlow (link.c:1938-2004) on safe arguments where closed
    head_o = torch.where(closed, torch.ones_like(head), head)
    f_o = torch.where(closed, torch.ones_like(f), f)
    q, dqdh = _orifice_get_flow(orf, head_o, f_o)
    # ARMCO flap-gate head loss (link.c:1974-2002), then the flow at the reduced head/f
    area = orf.opening.area(yf)                                              # link.c:1978-1979
    veloc = q / area                                                         # link.c:1980
    h_loss = _ARMCO * veloc * veloc * torch.exp(-1.15 * veloc / torch.sqrt(head_o))  # :1983
    f_g = torch.where(f_o < 1.0, torch.clamp(f_o - h_loss / hc_safe, min=0.0), f_o)  # :1989
    head_g = torch.where(f_o < 1.0, head_o, torch.clamp(head_o - h_loss, min=0.0))  # :1994
    q_g, dqdh_g = _orifice_get_flow(orf, head_g, f_g)                        # link.c:2001
    q = torch.where(orf.gated, q_g, q)
    dqdh = torch.where(orf.gated, dqdh_g, dqdh)
    # Villemonte submergence (link.c:1922-1926), not part of dqdh
    vill = (f_o < 1.0) & (hd > crest)
    above = torch.where(vill & ~closed, hu - crest, torch.ones_like(hu))
    q = torch.where(vill, q * _villemonte((hd - crest) / above, 1.5), q)
    sign = torch.where(forward, torch.ones_like(q), -torch.ones_like(q))
    zero = torch.zeros_like(q)
    return StructureFlow(torch.where(closed, zero, sign * q),                # link.c:1919
                         torch.where(closed, zero, dqdh), cls, depth, surf)


# ------------------------------------------------------------------------------- weirs
@dataclass(frozen=True)
class WeirSet:
    """Weirs (`link.c` ``Weir``), per device: the `opening`, ``weir_type`` (long,
    `TRANSVERSE_WEIR` ... `TRAPEZOIDAL_WEIR`), ``c_disch1`` and ``c_disch2`` (SI: the
    central part's and a trapezoid's end triangles' coefficients, m^0.5/s),
    ``end_contractions`` (-), ``can_surcharge`` (bool), ``gated`` (bool, own flap gate),
    ``slope`` (SWMM's ``Weir.slope``: the opening's ``sBot`` for V-notch and trapezoidal
    weirs, 0 otherwise; link.c:2113-2133) and SWMM's ``c_surcharge`` (the equivalent orifice
    coefficient, cfs/ft^0.5, link.c:2154-2161)."""

    opening: Opening
    weir_type: Tensor
    c_disch1: Tensor
    c_disch2: Tensor
    end_contractions: Tensor
    can_surcharge: Tensor
    gated: Tensor
    slope: Tensor
    c_surcharge: Tensor
    setting: Tensor | None = None

    def at_setting(self, setting: Tensor) -> WeirSet:
        """This set at the per-device `setting` (0..1; 1 fully open): the crest rises by
        ``(1 - setting) yFull`` (link.c:2255), a V-notch becomes a trapezoid below 1
        (link.c:2361), and the equivalent orifice coefficient is recomputed from the flow at a
        head of ``setting yFull`` (`weir_setSetting`, link.c:2166-2194; 0 at a setting of
        0, which closes the weir, link.c:550)."""
        safe = torch.where(setting > 0.0, setting, torch.ones_like(setting))
        weir = replace(self, setting=safe)
        h = weir.opening.y_full * safe                                       # link.c:2186
        q1, q2, _ = _weir_get_flow(weir, h, torch.ones_like(h) > 0, gated=False)  # :2188
        c = (q1 + q2) / torch.sqrt(h / 2.0)                                  # link.c:2192
        c = torch.where(setting > 0.0, c, torch.zeros_like(c))               # link.c:2182
        c = torch.where(self.can_surcharge, c, self.c_surcharge)             # link.c:2178
        return replace(weir, c_surcharge=c, setting=setting)

    @classmethod
    def build(cls, opening: Opening, weir_type, c_disch1, c_disch2, end_contractions,
              can_surcharge, gated) -> WeirSet:
        """SWMM's weir constants (`weir_validate`, link.c:2103-2162)."""
        weir_type = torch.as_tensor(weir_type, dtype=torch.long)
        notch = (weir_type == VNOTCH_WEIR) | (weir_type == TRAPEZOIDAL_WEIR)
        slope = torch.where(notch, opening.side_slope, torch.zeros_like(opening.side_slope))
        weir = cls(opening, weir_type, torch.as_tensor(c_disch1, dtype=F64),
                   torch.as_tensor(c_disch2, dtype=F64),
                   torch.as_tensor(end_contractions, dtype=F64),
                   torch.as_tensor(can_surcharge, dtype=torch.bool),
                   torch.as_tensor(gated, dtype=torch.bool), slope,
                   torch.zeros_like(opening.y_full))
        # flow at a head of the opening height, forward, ungated (link.c:2154-2156)
        q1, q2, _ = _weir_get_flow(weir, opening.y_full, torch.ones_like(opening.y_full) > 0,
                                   gated=False)
        c_surcharge = (q1 + q2) / torch.sqrt(opening.y_full / 2.0)          # link.c:2159-2160
        return cls(opening, weir_type, weir.c_disch1, weir.c_disch2, weir.end_contractions,
                   weir.can_surcharge, weir.gated, slope, c_surcharge)


def _weir_q(weir: WeirSet, head: Tensor, forward: Tensor) -> tuple[Tensor, Tensor]:
    """`weir_getFlow`'s formulas (link.c:2356-2413) at ``head`` (ft, > 0 where used):
    ``(q1, q2)`` in cfs."""
    head_pos = torch.where(head > 0.0, head, torch.ones_like(head))
    length = weir.opening.w_max * LCF                                        # link.c:2353
    h = head_pos * LCF                                                       # link.c:2354
    t = weir.weir_type
    reduced = torch.clamp(length - 0.1 * weir.end_contractions * h, min=0.0)  # :2368-2369
    transverse = weir.c_disch1 * reduced * h ** 1.5                          # link.c:2370
    side_back = weir.c_disch1 * reduced * h ** 1.5                           # link.c:2381
    reduced_safe = torch.where(reduced > 0.0, reduced, torch.ones_like(reduced))
    side_fwd = torch.where(reduced > 0.0,
                           weir.c_disch1 * reduced_safe ** 0.83 * h ** 1.67,
                           torch.zeros_like(h))                              # link.c:2386
    sideflow = torch.where(forward, side_fwd, side_back)                     # link.c:2380
    vnotch = weir.c_disch1 * weir.slope * h ** 2.5                           # link.c:2391
    if weir.setting is None:
        offset = torch.zeros_like(h)
    else:
        # a V-notch below setting 1 is a trapezoid (link.c:2361); the central width is the
        # opening's at the raised crest
        offset = (1.0 - weir.setting) * weir.opening.y_full
        t = torch.where((t == VNOTCH_WEIR) & (weir.setting < 1.0),
                        torch.full_like(t, TRAPEZOIDAL_WEIR), t)
    trap_len = weir.opening.width(offset) * LCF                              # link.c:2395-2396
    trap = weir.c_disch1 * trap_len * h ** 1.5                               # link.c:2397
    q1 = torch.where(t == TRANSVERSE_WEIR, transverse,
                     torch.where(t == SIDEFLOW_WEIR, sideflow,
                                 torch.where(t == VNOTCH_WEIR, vnotch, trap)))
    q2 = torch.where(t == TRAPEZOIDAL_WEIR, weir.c_disch2 * weir.slope * h ** 2.5,
                     torch.zeros_like(h))                                    # link.c:2398
    zero = torch.zeros_like(q1)
    q1 = torch.where(head > 0.0, q1 / M3_PER_FT3, zero)                      # link.c:2350, 2411
    q2 = torch.where(head > 0.0, q2 / M3_PER_FT3, zero)
    return q1, q2


def _weir_dqdh(weir: WeirSet, forward: Tensor, h: Tensor, q1: Tensor, q2: Tensor) -> Tensor:
    """`weir_getdqdh` (link.c:2492-2517), cfs/ft."""
    small = h.abs() < FUDGE                                                  # link.c:2497
    h_safe = torch.where(small, torch.ones_like(h), h)
    q1h = (q1 / h_safe).abs()
    q2h = (q2 / h_safe).abs()
    t = weir.weir_type
    side = torch.where(forward, 1.67 * q1h, 1.5 * q1h)                       # link.c:2506-2507
    notch = torch.where(q2h == 0.0, 2.5 * q1h, 1.5 * q1h + 2.5 * q2h)        # link.c:2510-2511
    dq = torch.where(t == TRANSVERSE_WEIR, 1.5 * q1h,
                     torch.where(t == SIDEFLOW_WEIR, side,
                                 torch.where(t == VNOTCH_WEIR, notch,
                                             1.5 * q1h + 2.5 * q2h)))        # link.c:2503-2513
    return torch.where(small, torch.zeros_like(dq), dq)


def _weir_open_area(weir: WeirSet, y: Tensor) -> Tensor:
    """`weir_getOpenArea` (link.c:2467-2488): ``A(min(z + y, yFull)) - A(z)`` with the crest
    offset ``z = (1 - setting) yFull`` (0 at setting 1)."""
    if weir.setting is None:
        return weir.opening.area(torch.minimum(y, weir.opening.y_full))
    z = (1.0 - weir.setting) * weir.opening.y_full                           # link.c:2478
    return (weir.opening.area(torch.minimum(z + y, weir.opening.y_full))
            - weir.opening.area(z))                                          # :2487


def _weir_get_flow(weir: WeirSet, head: Tensor, forward: Tensor, gated=None
                   ) -> tuple[Tensor, Tensor, Tensor]:
    """`weir_getFlow` (link.c:2321-2428): ``(q1, q2, dqdh)`` (cfs, cfs, cfs/ft) at ``head``
    (ft), with the ARMCO flap-gate loss where `gated` (default: each weir's own gate)."""
    q1, q2 = _weir_q(weir, head, forward)
    gated = weir.gated if gated is None else torch.as_tensor(gated, dtype=torch.bool)
    area = _weir_open_area(weir, head)                                       # link.c:2418
    use = gated & (area > TINY) & (head > 0.0)                               # link.c:2419
    area_safe = torch.where(use, area, torch.ones_like(area))
    head_safe = torch.where(use, head, torch.ones_like(head))
    veloc = (q1 + q2) / area_safe                                            # link.c:2421
    h_loss = _ARMCO * veloc * veloc * torch.exp(-1.15 * veloc / torch.sqrt(head_safe))
    head_g = torch.clamp(head_safe - h_loss, min=0.0)                        # link.c:2424-2425
    g1, g2 = _weir_q(weir, head_g, forward)                                  # link.c:2429
    head = torch.where(use, head_g, head)
    q1 = torch.where(use, g1, q1)
    q2 = torch.where(use, g2, q2)
    return q1, q2, _weir_dqdh(weir, forward, head, q1, q2)                  # link.c:2427


def _weir_orifice_flow(weir: WeirSet, head: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """`weir_getOrificeFlow` (link.c:2432-2463) for a surcharged weir: ``(q, dqdh)`` with the
    water `y` ft above the (raised) crest, the opening full, and the ARMCO loss where
    gated."""
    q = weir.c_surcharge * _sqrt(head)                                       # link.c:2446
    a = _weir_open_area(weir, y)                                             # link.c:2451
    use = weir.gated & (a > 0.0)
    v = q / torch.where(use, a, torch.ones_like(a))
    h_loss = _ARMCO * v * v * torch.exp(-1.15 * v / torch.sqrt(y))           # link.c:2455
    head = torch.where(use, torch.clamp(head - h_loss, min=0.0), head)       # link.c:2456-2457
    q = torch.where(use, weir.c_surcharge * _sqrt(head), q)                  # link.c:2458
    pos = head > 0.0
    dq = torch.where(pos, q / (2.0 * torch.where(pos, head, torch.ones_like(head))),
                     torch.zeros_like(q))                                    # link.c:2461-2462
    return torch.where(pos, q, torch.zeros_like(q)), dq


def weir_flow(weir: WeirSet, h1: Tensor, h2: Tensor, z1: Tensor, z2: Tensor, crest: Tensor,
              block_forward=False, block_reverse=False) -> StructureFlow:
    """`link.c::weir_getInflow` under dynamic wave (link.c:2198-2317), arguments as
    `orifice_flow` (`crest` the crest elevation at setting 1; the setting, if the set has one,
    raises it by ``(1 - setting) yFull``, link.c:2255). Roadway weirs are not covered (refused
    by the network)."""
    forward, hu, hd, _ = _orient(h1, h2, z1, z2, strict=True)                # link.c:2241-2249
    yf = weir.opening.y_full
    crown = crest + yf                                                       # link.c:2253
    shut = None
    if weir.setting is not None:
        shut = weir.setting <= 0.0                                           # hcrest >= hcrown
        crest = crest + (1.0 - weir.setting) * yf                            # link.c:2255
    head = hu - crest                                                        # link.c:2263
    closed = (head <= FUDGE) | _gate(forward, block_forward, block_reverse)  # link.c:2267-2268
    if shut is not None:
        closed = closed | shut
    cls = _flow_class(closed, forward, crest, hd)                            # link.c:2275-2281
    surcharged = hu >= crown                                                 # link.c:2289
    # surcharged and able to: equivalent orifice (link.c:2291-2299)
    mid = (crest + crown) / 2.0
    head_s = torch.where(hd < mid, hu - mid, hu - hd)                        # link.c:2294-2295
    y_open = crown - crest                                                   # link.c:2296
    q_s, dq_s = _weir_orifice_flow(weir, torch.where(closed, torch.ones_like(head_s), head_s),
                                   y_open)
    orifice = surcharged & weir.can_surcharge
    # weir equation, head capped at the opening height when it cannot surcharge (:2303)
    head_w = torch.where(surcharged, y_open.expand_as(head), head)
    head_w = torch.where(closed, torch.ones_like(head_w), head_w)
    q1, q2, dq_w = _weir_get_flow(weir, head_w, forward)                     # link.c:2308
    vill = hd > crest                                                        # link.c:2311
    above = torch.where(vill & ~closed, hu - crest, torch.ones_like(hu))
    ratio = (hd - crest) / above
    power = torch.tensor(_WEIR_POWER, dtype=F64)[weir.weir_type]
    q1 = torch.where(vill, q1 * _villemonte(ratio, power), q1)               # link.c:2314
    q2 = torch.where(vill & (q2 > 0.0), q2 * _villemonte(ratio, 2.5), q2)   # link.c:2315-2316
    q = torch.where(orifice, q_s, q1 + q2)
    dqdh = torch.where(orifice, dq_s, dq_w)
    depth = torch.where(orifice, y_open.expand_as(head), torch.minimum(head, yf))  # :2298, 2319
    sign = torch.where(forward, torch.ones_like(q), -torch.ones_like(q))
    zero = torch.zeros_like(q)
    return StructureFlow(torch.where(closed, zero, sign * q), torch.where(closed, zero, dqdh),
                         cls, torch.where(closed, zero, depth), zero)


# ----------------------------------------------------------------------------- outlets
@dataclass(frozen=True)
class OutletSet:
    """Outlets (`link.c` ``Outlet``), per device: ``tabular`` (bool: a rating curve, else
    ``q = coefficient h^exponent``), ``coefficient`` and ``exponent`` (SI: m3/s per
    m^exponent, -), the rating curves padded to one width, ``curve_x`` (m, +inf beyond a
    curve's points) and ``curve_y`` (m3/s, the last flow beyond them), ``head_based`` (bool:
    SWMM's /HEAD, the head difference across the outlet; else /DEPTH, the upstream depth
    above the crest) and ``gated`` (bool, own flap gate)."""

    tabular: Tensor
    coefficient: Tensor
    exponent: Tensor
    curve_x: Tensor
    curve_y: Tensor
    head_based: Tensor
    gated: Tensor

    @classmethod
    def build(cls, curves, head_based, gated) -> OutletSet:
        """From each outlet's rating ``("functional", coefficient, exponent)`` or
        ``("tabular", heads, flows)`` (m, m3/s; heads strictly increasing)."""
        n = len(curves)
        width = max([len(c[1]) for c in curves if c[0] == "tabular"] or [1])
        x = torch.full((n, width), torch.inf, dtype=F64)
        y = torch.zeros((n, width), dtype=F64)
        coefficient = torch.zeros(n, dtype=F64)
        exponent = torch.zeros(n, dtype=F64)
        tabular = torch.zeros(n, dtype=torch.bool)
        for i, curve in enumerate(curves):
            if curve[0] == "tabular":
                tabular[i] = True
                k = len(curve[1])
                x[i, :k] = torch.tensor(curve[1], dtype=F64)
                y[i, :k] = torch.tensor(curve[2], dtype=F64)
                y[i, k:] = float(curve[2][-1])
            else:
                coefficient[i], exponent[i] = float(curve[1]), float(curve[2])
        return cls(tabular, coefficient, exponent, x, y,
                   torch.as_tensor(head_based, dtype=torch.bool),
                   torch.as_tensor(gated, dtype=torch.bool))


def table_lookup(x: Tensor, curve_x: Tensor, curve_y: Tensor) -> Tensor:
    """`table.c::table_lookup` (table.c:395-426) per row: the first y at or below the first
    x, linear interpolation inside, the last y beyond the last x. ``curve_x`` (rows padded
    with +inf) and ``curve_y`` (padded with the last y) are (devices, points); ``x`` is
    (..., devices)."""
    k = curve_x.shape[-1]
    below = (curve_x < x.unsqueeze(-1)).sum(-1)                              # x in (x[i-1], x[i]]
    hi = torch.clamp(below, max=k - 1)
    lo = torch.clamp(below - 1, min=0)
    shape = x.shape + (k,)
    cx, cy = curve_x.expand(shape), curve_y.expand(shape)
    x1 = cx.gather(-1, lo.unsqueeze(-1)).squeeze(-1)
    x2 = cx.gather(-1, hi.unsqueeze(-1)).squeeze(-1)
    y1 = cy.gather(-1, lo.unsqueeze(-1)).squeeze(-1)
    y2 = cy.gather(-1, hi.unsqueeze(-1)).squeeze(-1)
    inside = (below > 0) & (below < k) & torch.isfinite(x2)
    dx = torch.where(inside, x2 - x1, torch.ones_like(x1))
    interp = y1 + (x - torch.where(inside, x1, x)) * (y2 - y1) / dx          # table.c:62
    first = cy[..., 0]
    last = cy.gather(-1, torch.full(x.shape + (1,), k - 1)).squeeze(-1)
    return torch.where(below == 0, first, torch.where(inside, interp, last))  # table.c:416-425


def _outlet_q(out: OutletSet, head: Tensor) -> Tensor:
    """`outlet_getFlow` (link.c:2672-2692): cfs at ``head`` (ft, > 0 where used)."""
    h = torch.where(head > 0.0, head, torch.ones_like(head)) * LCF           # link.c:2684
    rated = table_lookup(h, out.curve_x, out.curve_y)                        # link.c:2688
    functional = out.coefficient * h ** out.exponent                         # link.c:2691
    return torch.where(out.tabular, rated, functional) / QCF


def outlet_flow(out: OutletSet, h1: Tensor, h2: Tensor, z1: Tensor, z2: Tensor, crest: Tensor,
                block_forward=False, block_reverse=False, setting=None) -> StructureFlow:
    """`link.c::outlet_getInflow` under dynamic wave (link.c:2608-2668), arguments as
    `orifice_flow`; the flow is scaled by the per-device `setting` (0..1, link.c:2667), which
    closes the outlet at 0 (link.c:550). SWMM sets no ``dqdh`` for an outlet (0; see
    `outlet_dqdh`)."""
    forward, hu, hd, yu = _orient(h1, h2, z1, z2, strict=False)               # link.c:2633-2643
    head = torch.where(out.head_based, hu - torch.maximum(hd, crest), hu - crest)  # :2648-2650
    closed = (head <= FUDGE) | (yu <= FUDGE) | _gate(forward, block_forward, block_reverse)
    if setting is not None:
        closed = closed | (setting <= 0.0)
    q = _outlet_q(out, torch.where(closed, torch.ones_like(head), head))
    if setting is not None:
        q = q * setting
    sign = torch.where(forward, torch.ones_like(q), -torch.ones_like(q))
    zero = torch.zeros_like(q)
    cls = torch.where(closed, torch.full_like(head, DRY, dtype=torch.long),
                      torch.full_like(head, SUBCRITICAL, dtype=torch.long))  # link.c:2659-2665
    return StructureFlow(torch.where(closed, zero, sign * q), zero, cls,
                         torch.where(closed, zero, head), zero)


def outlet_dqdh(out: OutletSet, h1: Tensor, h2: Tensor, z1: Tensor, z2: Tensor, crest: Tensor,
                block_forward=False, block_reverse=False, step: float = FUDGE,
                setting=None) -> Tensor:
    """The outlet flow's central-difference slope in its upstream head (cfs/ft, >= 0):
    ``(Q(h_up + step) - Q(h_up - step)) / (2 step)``, the upstream node being node 1 where
    ``h1 >= h2`` and node 2 otherwise (`step` in ft, default SWMM's FUDGE). SWMM computes
    no slope for an outlet; this one only shapes noodl's passes."""
    forward = h1 >= h2
    d1 = torch.where(forward, torch.full_like(h1, step), torch.zeros_like(h1))
    d2 = torch.where(forward, torch.zeros_like(h2), torch.full_like(h2, step))
    up = outlet_flow(out, h1 + d1, h2 + d2, z1, z2, crest, block_forward, block_reverse,
                     setting).flow
    down = outlet_flow(out, h1 - d1, h2 - d2, z1, z2, crest, block_forward, block_reverse,
                       setting).flow
    slope = torch.where(forward, up - down, down - up) / (2.0 * step)
    return torch.clamp(slope, min=0.0)
