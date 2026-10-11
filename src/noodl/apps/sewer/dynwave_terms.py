"""SWMM 5.2.4's dynamic-wave conduit update, `dwflow.c::dwflow_findConduitFlow`, as a pure
function of one pass's state, in SWMM's own feet units (ft, ft2, cfs, s; g = 32.2).

The update is linear in the end-head difference once the depth-dependent coefficients are
frozen, q = c0 + k (H1 - H2): that is the form `DynamicWaveConduit` solves. What SWMM applies
AFTER the momentum update -- the normal-flow limit and the dry-node clamps -- is returned as
flags and limits for the element to apply. Trial relaxation, the 0.001 cfs reversal rule and
link bypassing are artefacts of SWMM's Picard iteration that do not move its fixed point and
are not modelled (see docs/applications/sewer.md, "Dynamic wave").

Scope of the transcription (circular conduits): no local losses (``dq5``), no
evaporation/seepage (``dq6``) and no culvert inlet control. Parallel barrels (``Conduit.barrels``)
and Courant lengthening (``modLength``) are modelled: the momentum update is one barrel's, and
the flow offset, conveyance, normal flow, surface areas and volume it returns are totals over the
barrels, as SWMM's ``newFlow``, ``dqdh``, ``surfArea`` and ``newVolume`` are. The user flow limit
and the flap gate act on the flow after these terms and are the element's
(`DynamicWaveConduit`). Both ``SURCHARGE_METHOD`` values are transcribed: EXTRAN (no
Preissmann slot: ``getSlotWidth == 0``, end depths capped at the crown, widths cut off at
``0.96 yFull``) and SLOT (end depths uncapped, the Sjoberg slot width above
``0.985257 yFull``, areas extended above the crown).

Every source reference is to SWMM 5.2.4 ``src/solver``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from noodl.apps.sewer.xsect_tables import (
    GRAVITY_FT,
    LCF,
    PHI,
    area_of_depth,
    conduit_constants,
    critical_depth,
    froude_number,
    normal_depth_ft,
    radius_of_depth,
    width_of_depth,
)

Tensor = torch.Tensor
F64 = torch.float64

#: SWMM's flow classes (`enums.h` ``FlowClassType``), in SWMM's order.
DRY, UP_DRY, DN_DRY, SUBCRITICAL, SUPCRITICAL, UP_CRITICAL, DN_CRITICAL = range(7)
#: SWMM's ``FUDGE`` (`consts.h:72`): minimum depth (ft), area (ft2) and flow (cfs).
FUDGE = 1.0e-4
#: SWMM's ``MAXVELOCITY`` (`dwflow.c:37`), ft/s.
MAX_VELOCITY = 50.0
#: SWMM's ``EXTRAN_CROWN_CUTOFF`` (`dynwave.c:64`): fraction of full depth above which a
#: closed conduit's top width is frozen.
EXTRAN_CROWN_CUTOFF = 0.96
#: SWMM's ``SLOT_CROWN_CUTOFF`` (`dynwave.c:65`): the same cutoff under the Preissmann slot.
SLOT_CROWN_CUTOFF = 0.985257
#: ``SURCHARGE_METHOD`` options (`EXTRAN`, `SLOT`; `keywords.c:148`).
SURCHARGE_METHODS = ("extran", "slot")
#: ``INERTIAL_DAMPING`` options (`NO_DAMPING`, `PARTIAL_DAMPING`, `FULL_DAMPING`).
DAMPING = ("none", "partial", "full")
#: ``NORMAL_FLOW_LIMITED`` options (`SLOPE`, `FROUDE`, `BOTH`, `NEITHER`).
NORMAL_FLOW_LIMITED = ("slope", "froude", "both", "none")


@dataclass(frozen=True)
class ConduitGeometry:
    """Per-conduit constants of the dynamic-wave update, in SWMM's internal feet units.

    ``y_full`` full depth (ft), ``a_full`` full area (ft2), ``s_full`` section factor
    ``A R^(2/3)`` at full depth (ft^(8/3)), ``beta = PHI sqrt(slope)/n`` (ft^(1/3)/s),
    ``rough_factor = g (n/PHI)^2 / lengthFactor`` (ft^(1/3)/s2, `link.c:1130`, with the
    roughness reduced by ``sqrt(lengthFactor)``, `link.c:1117`), ``length`` the true length
    (ft), ``mod_length`` SWMM's ``modLength`` = ``lengthFactor * length`` (ft, `link.c:1115`;
    the momentum terms and surface areas use it, the volume the true length),
    ``barrels`` (-, SWMM's ``Conduit.barrels``),
    ``invert1``/``invert2`` end-node invert elevations (ft), ``offset1``/``offset2``
    conduit inverts above those (ft), ``outfall1``/``outfall2`` whether the end node is an
    outfall (bool).
    """

    y_full: Tensor
    a_full: Tensor
    s_full: Tensor
    beta: Tensor
    rough_factor: Tensor
    length: Tensor
    mod_length: Tensor
    barrels: Tensor
    invert1: Tensor
    invert2: Tensor
    offset1: Tensor
    offset2: Tensor
    outfall1: Tensor
    outfall2: Tensor

    @classmethod
    def from_si(cls, diameter: Tensor, roughness: Tensor, slope: Tensor, length: Tensor,
                invert1: Tensor, invert2: Tensor, offset1: Tensor, offset2: Tensor,
                outfall1: Tensor, outfall2: Tensor, *, barrels: Tensor | None = None,
                length_factor: Tensor | None = None) -> ConduitGeometry:
        """From SI inputs: ``diameter`` (m), Manning ``roughness`` (s m^(-1/3), numerically
        SWMM's n), ``slope`` (m/m), ``length`` (m), node inverts and conduit offsets (m),
        outfall flags (bool), the ``barrels`` count (default 1) and the Courant
        ``length_factor`` (-, >= 1, default 1; `lengthening_factor`)."""
        y_full, a_full, s_full, beta = conduit_constants(diameter, roughness, slope)
        factor = torch.ones_like(y_full) if length_factor is None else length_factor
        # link.c:1117 -- roughness / sqrt(lengthFactor); beta is unchanged (link.c:1116 divides
        # the slope by the same factor)
        n_phi = roughness / torch.sqrt(factor) / PHI
        return cls(
            y_full=y_full, a_full=a_full, s_full=s_full, beta=beta,
            rough_factor=GRAVITY_FT * (n_phi * n_phi),           # link.c:1130
            length=length / LCF,
            mod_length=factor * length / LCF,                    # link.c:1115
            barrels=(torch.ones_like(y_full) if barrels is None
                     else torch.as_tensor(barrels, dtype=F64)),
            invert1=invert1 / LCF, invert2=invert2 / LCF,
            offset1=offset1 / LCF, offset2=offset2 / LCF,
            outfall1=torch.as_tensor(outfall1).bool(),
            outfall2=torch.as_tensor(outfall2).bool(),
        )


def lengthening_factor(diameter: Tensor, roughness: Tensor, slope: Tensor, length: Tensor,
                       step: float) -> Tensor:
    """`link.c::conduit_getLengthFactor` (link.c:1217-1254) for circular conduits: the factor
    (>= 1, -) by which SWMM lengthens a conduit to the Courant length ``(sqrt(g yFull) +
    vFull) t`` of the shorter of the routing step and ``LENGTHENING_STEP`` (``step``, s; the
    caller passes that minimum). ``diameter`` (m), ``roughness`` (s m^(-1/3)), ``slope``
    (m/m, the conduit's real slope), ``length`` (m)."""
    y_full, a_full, s_full, _ = conduit_constants(diameter, roughness, slope)
    v_full = PHI / roughness * s_full * torch.sqrt(torch.abs(slope)) / a_full  # link.c:1241
    ratio = (torch.sqrt(GRAVITY_FT * y_full) + v_full) * step / (length / LCF)  # link.c:1249
    return torch.where(ratio > 1.0, ratio, torch.ones_like(ratio))             # link.c:1252


class ConduitTerms(NamedTuple):
    """One pass's dynamic-wave terms per conduit, in SWMM ft units.

    Per-barrel quantities are marked; the flows, conveyances and surface areas are TOTALS over
    the conduit's ``barrels`` (SWMM's ``newFlow``, ``dqdh`` and ``surfArea * barrels``,
    `dwflow.c:243, 292`, `dynwave.c:575-576`). ``flow_class`` (long, one of the seven
    classes); ``flow_offset`` c0 (cfs) and
    ``conveyance`` k = dq/d(H1 - H2) (ft2/s, 0 where an end head is pinned, the pinned
    head then folded into c0); ``momentum_flow`` SWMM's trial q before any limit (cfs, at the
    given heads, 0 on the dry exit); ``limited`` the normal-flow check passed (bool) with
    ``normal_flow`` its q_norm (cfs, +inf where not limited), both SWMM's verdict at the
    given heads; ``slope_switch`` (bool) where that verdict is the slope test ``y1 < y2``
    alone with both end depths inside their clamps, so that it is the switch ``h1 - h2 <
    switch_drop`` on the conduit's own head difference (``switch_drop`` = z1 - z2, ft) and
    ``switch_normal_flow`` its q_norm (cfs, +inf where no switch) -- the closure applies it
    within the pass (`dynwave.DynamicWaveConduit`); ``dry_upstream`` /
    ``dry_downstream`` node depth <= FUDGE (bool); ``surface_area1/2`` (ft2) contributed to
    the end nodes; ``mid_area`` the stored ``Conduit.a1`` (ft2, PER BARREL, next step's
    ``a_old``); ``volume`` (ft3, true length); ``depth`` ``Link.newDepth`` (ft);
    ``end_depth1/2`` the end flow depths after classification (ft);
    ``lagged_conveyance1/2`` (ft2/s) the conveyance k a wet conduit whose OTHER end is pinned
    still has in this end's head -- the pass lags it
    into c0 (``conveyance`` is 0 there), 0 elsewhere. SWMM keeps it in the node's
    ``sumdqdh`` (`dwflow.c:243`, `dynwave.c:578-588`); the closure uses it to anchor a
    surcharged junction (`JunctionStorage`); ``froude`` SWMM's `Link.froude` (-), which the
    variable routing step reads: 0 on the dry exit (`dwflow.c:175`), else from the flow
    of the given state at mid depth (`dwflow.c:189`, 0 for a full closed conduit; per barrel).

    The drop-end area switch (`dynwave.DynamicWaveHydraulics`, the tie rule there). An end
    with an offset whose other end is wet gives its node no surface area below its LOWER
    threshold -- UP_/DN_CRITICAL, or dry (dwflow.c:357-411, 472-499) -- and a quarter of the
    conduit above it; the threshold is ``max(FUDGE, min(yN, yC))`` where the flow leaves the
    conduit at that end (``q_last >= 0`` downstream, ``< 0`` upstream) and FUDGE (SWMM's wet
    test) otherwise. A downstream end on forward flow has an UPPER threshold too, ``max(yN,
    yC)``, where ``fasnh`` jumps from 0 back to 1. So the node's area jumps UP at each.
    ``drop_depth1/2`` (ft, end depth above the conduit invert) is the threshold nearest the
    end's depth, +inf where there is none (no offset, the other end dry, both ends full, or
    out of reach of the EXTRAN end depth capped at yFull); ``drop_area1/2`` (ft2, total over
    the barrels) the jump there, the quarter area at the threshold times ``fasnh`` just
    above it; ``drop_above1/2`` (bool) whether this pass classified the end on the upper
    side (above the threshold, or held there by `hold1/2`).
    """

    flow_class: Tensor
    flow_offset: Tensor
    conveyance: Tensor
    momentum_flow: Tensor
    limited: Tensor
    normal_flow: Tensor
    slope_switch: Tensor
    switch_drop: Tensor
    switch_normal_flow: Tensor
    dry_upstream: Tensor
    dry_downstream: Tensor
    surface_area1: Tensor
    surface_area2: Tensor
    mid_area: Tensor
    volume: Tensor
    depth: Tensor
    end_depth1: Tensor
    end_depth2: Tensor
    lagged_conveyance1: Tensor
    lagged_conveyance2: Tensor
    froude: Tensor
    drop_depth1: Tensor
    drop_depth2: Tensor
    drop_area1: Tensor
    drop_area2: Tensor
    drop_above1: Tensor
    drop_above2: Tensor


def _slot_width(y: Tensor, y_full: Tensor, slot: bool) -> Tensor:
    """`dwflow.c::getSlotWidth` (dwflow.c:575-588) for a closed circular conduit: Preissmann
    slot width (ft) at depth ``y`` (ft); 0 under EXTRAN or below ``0.985257 yFull``
    (``wMax = yFull``)."""
    if not slot:
        return torch.zeros_like(y)                                             # dwflow.c:580
    y_norm = y / y_full
    sjoberg = y_full * 0.5423 * torch.exp(-torch.pow(y_norm, 2.4))             # dwflow.c:587
    wide = torch.where(y_norm > 1.78, 0.01 * y_full, sjoberg)                  # dwflow.c:584
    return torch.where(y_norm < SLOT_CROWN_CUTOFF, torch.zeros_like(y), wide)  # dwflow.c:580


def _top_width(y: Tensor, y_full: Tensor, slot: bool = False) -> Tensor:
    """`dwflow.c::getWidth` (dwflow.c:592-605) for a closed conduit: top width (ft) at depth
    ``y`` (ft). The slot width where it is positive; otherwise the depth frozen at
    ``CrownCutoff * yFull`` above the crown cutoff (0.96 EXTRAN, 0.985257 SLOT)."""
    cutoff = SLOT_CROWN_CUTOFF if slot else EXTRAN_CROWN_CUTOFF               # dynwave.c:159
    w_slot = _slot_width(y, y_full, slot)                                      # dwflow.c:596
    y = torch.where(y / y_full >= cutoff, cutoff * y_full, y)                  # dwflow.c:599-600
    return torch.where(w_slot > 0.0, w_slot, width_of_depth(y, y_full))


def _slot_area(y: Tensor, y_full: Tensor, a_full: Tensor, slot: bool) -> Tensor:
    """`dwflow.c::getArea` (dwflow.c:609-619): flow area (ft2) at depth ``y`` (ft), extended
    above the crown by ``(y - yFull) * wSlot(y)`` (0 under EXTRAN)."""
    w_slot = _slot_width(y, y_full, slot)                                      # dwflow.c:144
    above = a_full + (y - y_full) * w_slot                                     # dwflow.c:616
    return torch.where(y >= y_full, above, area_of_depth(torch.minimum(y, y_full), y_full,
                                                         a_full))


def _hyd_rad(y: Tensor, y_full: Tensor) -> Tensor:
    """`dwflow.c::getHydRad` (dwflow.c:623-633): hydraulic radius (ft), ``rFull`` at and
    above the crown under both surcharge methods."""
    return radius_of_depth(torch.minimum(y, y_full), y_full)                   # dwflow.c:631


def conduit_terms(geo: ConduitGeometry, depth1: Tensor, depth2: Tensor, q_last: Tensor,
                  q_old: Tensor, a_old: Tensor, dt: float, *, damping: str = "partial",
                  normal_flow_limited: str = "both",
                  surcharge_method: str = "extran", hold1: Tensor | None = None,
                  hold2: Tensor | None = None) -> ConduitTerms:
    """SWMM's conduit update (`dwflow.c:57-293`) for one pass, vectorised over conduits.

    ``depth1``, ``depth2`` are the end NODE depths above the node inverts (ft) from the
    previous pass; ``q_last`` the previous pass's flow (cfs, total over the barrels; SWMM's
    ``qLast`` is one barrel's, `dwflow.c:100-102`), which sets
    the flow class, surface areas, velocity and damping; ``q_old`` and ``a_old`` the
    step-start flow (cfs, total) and stored mid area (ft2, per barrel); ``dt`` the routing
    step (s).
    ``damping`` is ``INERTIAL_DAMPING``, ``normal_flow_limited`` ``NORMAL_FLOW_LIMITED`` and
    ``surcharge_method`` ``SURCHARGE_METHOD`` (``"extran"`` or ``"slot"``). Under ``"slot"``
    the end depths ``end_depth1/2`` may exceed ``y_full``, ``volume`` and ``mid_area`` use the
    slot-extended areas and ``depth`` stays ``min(y_mid, y_full)``.
    ``hold1``/``hold2`` (-, per conduit, default none) hold a drop end ON its area switch
    (the closure's tie rule): where ``>= 0`` the end is classified on the UPPER side whatever
    its depth -- not UP_/DN_CRITICAL, ``fasnh`` 1, SWMM's own class AT the threshold (its
    tests are strict, ``y < ycMin``) -- and the value is the weight ``theta`` (0 to 1) of
    the upper side in the sliding solution: at a ``min(yN, yC)`` threshold the OTHER end's
    surface area, a quarter of the conduit on the upper side and half of it on the
    critical side (dwflow.c:472-499), is ``(2 - theta)`` quarters; ``< 0`` is no hold.
    """
    if damping not in DAMPING:
        raise ValueError(f"conduit_terms: damping must be one of {DAMPING}, got {damping!r}")
    if normal_flow_limited not in NORMAL_FLOW_LIMITED:
        raise ValueError("conduit_terms: normal_flow_limited must be one of "
                         f"{NORMAL_FLOW_LIMITED}, got {normal_flow_limited!r}")
    if surcharge_method not in SURCHARGE_METHODS:
        raise ValueError("conduit_terms: surcharge_method must be one of "
                         f"{SURCHARGE_METHODS}, got {surcharge_method!r}")
    slot = surcharge_method == "slot"
    yf, af = geo.y_full, geo.a_full
    shape = torch.broadcast_shapes(depth1.shape, depth2.shape, q_last.shape, q_old.shape,
                                   a_old.shape, yf.shape)
    depth1, depth2, q_last, q_old, a_old = (
        x.to(F64).expand(shape) for x in (depth1, depth2, q_last, q_old, a_old))
    barrels = geo.barrels
    q_last = q_last / barrels                                                  # dwflow.c:102
    q_old = q_old / barrels
    zero = torch.zeros(shape, dtype=F64)
    one = torch.ones(shape, dtype=F64)
    fudge = torch.full(shape, FUDGE, dtype=F64)
    length = geo.mod_length                                                    # dwflow.c:136

    # dwflow.c:110-115 -- end heads, raised to the conduit inverts
    z1 = geo.invert1 + geo.offset1
    z2 = geo.invert2 + geo.offset2
    head1 = depth1 + geo.invert1
    head2 = depth2 + geo.invert2
    h1 = torch.maximum(head1, z1)
    h2 = torch.maximum(head2, z2)
    pinned1 = head1 < z1          # h1 no longer follows the node head (design D4)
    pinned2 = head2 < z2
    # dwflow.c:119-129 -- end flow depths, at least FUDGE, at most yFull unless SLOT
    y1 = torch.clamp(h1 - z1, min=FUDGE)
    y2 = torch.clamp(h2 - z2, min=FUDGE)
    if not slot:
        y1 = torch.minimum(y1, yf)                                             # dwflow.c:125
        y2 = torch.minimum(y2, yf)

    # dwflow.c:297-413 getFlowClass, on q = qLast (dwflow.c:140)
    zr1 = torch.where(geo.outfall1, torch.clamp(geo.offset1 - depth1, min=0.0),
                      geo.offset1)                                        # dwflow.c:327
    zr2 = torch.where(geo.outfall2, torch.clamp(geo.offset2 - depth2, min=0.0),
                      geo.offset2)                                        # dwflow.c:328
    q_abs = q_last.abs()
    y_norm = normal_depth_ft(q_abs, yf, af, geo.s_full, geo.beta).expand(shape)
    y_crit = critical_depth(q_abs, yf, af).expand(shape)
    yc_min = torch.minimum(y_norm, y_crit)
    yc_max = torch.maximum(y_norm, y_crit)
    wet1 = y1 > FUDGE
    wet2 = y2 > FUDGE
    reverse = q_last < 0.0
    # The drop-end area switch (class docstring). An end with an offset gives its node no
    # area below its LOWER threshold (UP_/DN_CRITICAL, or UP_/DN_DRY there): FUDGE, raised
    # to min(yN, yC) where the flow leaves the conduit at that end; a downstream end that
    # receives forward flow has an UPPER one too, max(yN, yC), where fasnh jumps from 0 to 1.
    inf = torch.full(shape, torch.inf, dtype=F64)
    lower2 = torch.where(reverse, fudge, torch.maximum(yc_min, fudge))
    upper2 = torch.where(~reverse & (yc_max > lower2), yc_max, inf)
    is_upper2 = (y2 - upper2).abs() < (y2 - lower2).abs()
    thr2 = torch.where(is_upper2, upper2, lower2)
    lower1 = torch.where(reverse, torch.maximum(yc_min, fudge), fudge)
    cand1 = (zr1 > 0.0) & wet2
    cand2 = (zr2 > 0.0) & wet1
    no_hold = torch.full(shape, -1.0, dtype=F64)
    theta1 = no_hold if hold1 is None else hold1.to(F64).expand(shape)
    theta2 = no_hold if hold2 is None else hold2.to(F64).expand(shape)
    held1 = (theta1 >= 0.0) & cand1
    held2 = (theta2 >= 0.0) & cand2
    hold_low2 = held2 & ~is_upper2
    hold_up2 = held2 & is_upper2
    # a held end is classified just above its threshold: wet, not critical, fasnh 1 at the
    # upper threshold; the other end's area then slides between the two classes where the
    # side below is critical (UP_/DN_CRITICAL give it half the conduit, the side above a
    # quarter; dwflow.c:472-499)
    wet1 = wet1 | held1
    wet2 = wet2 | hold_low2
    blend1 = held1 & ((lower1 > FUDGE) | ~(h2 < z1))
    blend2 = hold_low2 & ((thr2 > FUDGE) | ~(h1 < z2))
    # dwflow.c:335-371 -- both ends wet
    both_wet = wet1 & wet2
    up_crit = both_wet & reverse & (zr1 > 0.0) & (y1 < yc_min) & ~held1   # dwflow.c:342-347
    drop2 = both_wet & ~reverse & (zr2 > 0.0)
    dn_crit = drop2 & (y2 < yc_min) & ~hold_low2                          # dwflow.c:363
    in_band = drop2 & ~(y2 < yc_min) & (y2 < yc_max) & ~hold_up2          # dwflow.c:364-368
    span = yc_max - yc_min
    span_safe = torch.where(span < FUDGE, one, span)
    fasnh = torch.where(in_band, torch.where(span < FUDGE, zero, (yc_max - y2) / span_safe),
                        one)
    # dwflow.c:374 -- both ends dry
    dry = ~wet1 & ~wet2
    # dwflow.c:377-393 -- downstream wet, upstream dry
    only2 = ~wet1 & wet2
    up_dry = only2 & (h2 < z1)
    up_crit = up_crit | (only2 & ~(h2 < z1) & (zr1 > 0.0))
    # dwflow.c:396-411 -- upstream wet, downstream dry
    only1 = wet1 & ~wet2
    dn_dry = only1 & (h1 < z2)
    dn_crit = dn_crit | (only1 & ~(h1 < z2) & (zr2 > 0.0))
    cls = torch.full(shape, SUBCRITICAL, dtype=torch.long)
    for mask, code in ((up_crit, UP_CRITICAL), (dn_crit, DN_CRITICAL), (dry, DRY),
                       (up_dry, UP_DRY), (dn_dry, DN_DRY)):
        cls = torch.where(mask, torch.full_like(cls, code), cls)
    # dwflow.c:457-460 -- both ends full: SUBCRITICAL (fasnh stays 1)
    both_full = (y1 >= yf) & (y2 >= yf)
    cls = torch.where(both_full, torch.full_like(cls, SUBCRITICAL), cls)
    fasnh = torch.where(both_full, one, fasnh)
    is_up_crit = cls == UP_CRITICAL
    is_dn_crit = cls == DN_CRITICAL
    is_up_dry = cls == UP_DRY
    is_dn_dry = cls == DN_DRY
    is_dry = cls == DRY
    # the drop ends' thresholds (none where both ends are full, or where the EXTRAN end
    # depth, capped at yFull, cannot reach it), the area jump at each -- the quarter area
    # at the threshold (dwflow.c:472-474), times the fasnh just above a downstream lower
    # one -- and the side this pass is on
    reach1 = cand1 & ~both_full
    reach2 = cand2 & ~both_full
    if not slot:
        reach1 = reach1 & (lower1 < yf)
        reach2 = reach2 & (thr2 < yf)
    drop_depth1 = torch.where(reach1, lower1, inf)
    drop_depth2 = torch.where(reach2, thr2, inf)
    at1 = torch.where(reach1, lower1, y1)
    at2 = torch.where(reach2, thr2, y2)
    mid1 = torch.clamp(0.5 * (at1 + y2), min=FUDGE)
    mid2 = torch.clamp(0.5 * (y1 + at2), min=FUDGE)
    banded = ~reverse & ~is_upper2 & (thr2 < yc_max)
    fasnh_above = torch.where(banded, torch.where(span < FUDGE, zero,
                                                  (yc_max - thr2) / span_safe), one)
    drop_area1 = (_top_width(at1, yf, slot) + _top_width(mid1, yf, slot)) * length / 4.0
    drop_area2 = ((_top_width(mid2, yf, slot) + _top_width(at2, yf, slot)) * length / 4.0
                  * fasnh_above)
    drop_area1 = torch.where(reach1, drop_area1, zero)
    drop_area2 = torch.where(reach2, drop_area2, zero)
    drop_above1 = held1 | ((y1 >= lower1) & (y1 > FUDGE))
    drop_above2 = held2 | ((y2 >= thr2) & (y2 > FUDGE))

    # dwflow.c:465-549 findSurfArea -- end depths and heads per class
    y_cn = torch.clamp(torch.where(y_norm < y_crit, y_norm, y_crit), min=FUDGE)  # :478-480
    y1 = torch.where(is_up_crit, y_cn, torch.where(is_up_dry, fudge, y1))     # :478, :502
    y2 = torch.where(is_dn_crit, y_cn, torch.where(is_dn_dry, fudge, y2))     # :490, :522
    h1 = torch.where(is_up_crit, z1 + y1, h1)                                  # :481
    h2 = torch.where(is_dn_crit, z2 + y2, h2)                                  # :493
    pinned1 = pinned1 | is_up_crit
    pinned2 = pinned2 | is_dn_crit
    y_mid_s = torch.clamp(0.5 * (y1 + y2), min=FUDGE)                          # :468-469
    w1 = _top_width(y1, yf, slot)
    w2 = _top_width(y2, yf, slot)
    wm = _top_width(y_mid_s, yf, slot)
    quarter1 = (w1 + wm) * length / 4.0
    quarter2 = (wm + w2) * length / 4.0
    dry_area = FUDGE * length / 2.0                                            # :542
    surf1 = torch.where(cls == SUBCRITICAL, quarter1, zero)                    # :473
    surf1 = torch.where(is_dn_crit, (w1 + wm) * length * 0.5, surf1)           # :498
    surf1 = torch.where(is_up_dry & (geo.offset1 <= 0.0), quarter1, surf1)     # :515-518
    surf1 = torch.where(is_dn_dry, quarter1, surf1)                            # :531
    surf1 = torch.where(is_dry, dry_area.expand(shape), surf1)
    surf2 = torch.where(cls == SUBCRITICAL, quarter2 * fasnh, zero)            # :474
    surf2 = torch.where(is_up_crit, (wm + w2) * length * 0.5, surf2)           # :486
    surf2 = torch.where(is_up_dry, quarter2, surf2)                            # :511
    surf2 = torch.where(is_dn_dry & (geo.offset2 <= 0.0), quarter2, surf2)     # :535-538
    surf2 = torch.where(is_dry, dry_area.expand(shape), surf2)
    # a held drop end (tie rule): the other end's area slides between the two classes
    surf1 = torch.where(blend2, quarter1 * (2.0 - theta2.clamp(0.0, 1.0)), surf1)
    surf2 = torch.where(blend1, quarter2 * (2.0 - theta1.clamp(0.0, 1.0)), surf2)

    # dwflow.c:143-153 -- areas and radii at the ends and the midpoint (slot-extended)
    a1 = _slot_area(y1, yf, af, slot)
    r1 = _hyd_rad(y1, yf)
    a2 = _slot_area(y2, yf, af, slot)
    y_mid = 0.5 * (y1 + y2)
    a_mid = _slot_area(y_mid, yf, af, slot)
    r_mid = _hyd_rad(y_mid, yf)
    is_full = (y1 >= yf) & (y2 >= yf)
    # dwflow.c:165-180 -- dry exit (no flap gate, no control setting here)
    dry_exit = is_dry | is_up_dry | is_dn_dry | (a_mid <= FUDGE)

    # dwflow.c:182-189 -- velocity from the last flow, Froude number, SUPCRITICAL
    a_mid_safe = torch.where(dry_exit, one, a_mid)
    v = q_last / a_mid_safe
    v = torch.where(v.abs() > MAX_VELOCITY,
                    torch.where(q_last < 0.0, -one, one) * MAX_VELOCITY, v)
    fr = froude_number(v, y_mid, yf, af)
    cls = torch.where(~dry_exit & (cls == SUBCRITICAL) & (fr > 1.0),
                      torch.full_like(cls, SUPCRITICAL), cls)
    # dwflow.c:192-194 -- damping factor
    sigma_fr = torch.where(fr <= 0.5, one, torch.where(fr >= 1.0, zero, 2.0 * (1.0 - fr)))
    # dwflow.c:198-201 -- upstream weighting
    rho = torch.where(~is_full & (q_last > 0.0) & (h1 >= h2), sigma_fr, one)
    a_wtd = a1 + (a_mid - a1) * rho
    r_wtd = r1 + (r_mid - r1) * rho
    # dwflow.c:204-208 -- damping option; full closed conduit gets full damping
    sigma = {"none": one, "full": zero, "partial": sigma_fr}[damping]
    sigma = torch.where(is_full, zero, sigma)

    # dwflow.c:213-240 -- momentum terms (dq5 = dq6 = 0, see module docstring)
    r_wtd_safe = torch.where(dry_exit | (r_wtd <= 0.0), one, r_wtd)
    dq1 = dt * geo.rough_factor / torch.pow(r_wtd_safe, 1.33333) * v.abs()     # :214
    dq2 = dt * GRAVITY_FT * a_wtd * (h2 - h1) / length                         # :217
    a_old = torch.clamp(a_old, min=FUDGE)                                      # :132-133
    dq3 = 2.0 * v * (a_mid - a_old) * sigma                                    # :224
    dq4 = dt * v * v * (a2 - a1) / length * sigma                              # :225
    denom = 1.0 + dq1                                                          # :239
    momentum = (q_old - dq2 + dq3 + dq4) / denom                               # :240
    k = GRAVITY_FT * dt * a_wtd / (length * denom)                             # :243
    c0 = (q_old + dq3 + dq4) / denom
    # an end head that does not follow its node is folded into c0 (design D4)
    pinned = pinned1 | pinned2
    k_total = k * barrels                                                      # dwflow.c:243
    lagged1 = torch.where(pinned2 & ~pinned1 & ~dry_exit, k_total, zero)
    lagged2 = torch.where(pinned1 & ~pinned2 & ~dry_exit, k_total, zero)
    flow_offset = torch.where(pinned, momentum, c0)
    conveyance = torch.where(pinned, zero, k_total)

    # dwflow.c:248-258 + checkNormalFlow (dwflow.c:637-686), on the momentum flow
    has_outfall = geo.outfall1 | geo.outfall2                                  # :654
    checked = ((momentum > 0.0) & (y1 < yf) & ~dry_exit
               & ((cls == SUBCRITICAL) | (cls == SUPCRITICAL)))
    if normal_flow_limited == "none":
        checked = torch.zeros_like(checked)                                    # :255
    slope_on = normal_flow_limited in ("slope", "both")
    froude_on = normal_flow_limited in ("froude", "both")
    slope_check = (has_outfall | slope_on) & (y1 < y2)                         # :659-662
    a1_safe = torch.where(a1 > 0.0, a1, one)
    f1 = froude_number(momentum / a1_safe, y1, yf, af)                         # :670
    froude_check = (~slope_check & ~has_outfall & (y1 > FUDGE) & (y2 > FUDGE)
                    & (f1 >= 1.0))                                             # :665-672
    if not froude_on:
        froude_check = torch.zeros_like(froude_check)
    limited = checked & (slope_check | froude_check)
    q_norm = geo.beta * a1 * torch.pow(r1, 2.0 / 3.0)                          # :678
    inf = torch.full_like(q_norm, torch.inf)
    normal_flow = torch.where(limited, q_norm, inf)
    # The slope test as a switch on the pass's own head difference: with both end depths
    # strictly inside (FUDGE, yFull) (above FUDGE under SLOT) ``y1 < y2`` is ``h1 - h2 <
    # z1 - z2`` -- unless the Froude test fires, which limits the flow whatever the slope
    # test says (`limited` is ``checked & (slope | froude)``).
    froude_fires = (~has_outfall & (y1 > FUDGE) & (y2 > FUDGE) & (f1 >= 1.0)
                    if froude_on else torch.zeros_like(checked))               # :665-672
    in_range = (y1 > FUDGE) & (y2 > FUDGE)
    if not slot:
        in_range = in_range & (y2 < yf)
    slope_switch = checked & (has_outfall | slope_on) & in_range & ~froude_fires
    if normal_flow_limited == "none":
        slope_switch = torch.zeros_like(slope_switch)

    # dwflow.c:165-180 -- dry exit: no flow, stored area is the end mean
    a_end = (a1 + a2) / 2.0                                                    # :171, :288
    flow_offset = torch.where(dry_exit, zero, flow_offset)
    conveyance = torch.where(dry_exit, zero, conveyance)
    momentum = torch.where(dry_exit, zero, momentum)

    # per-barrel q is limited, and the pass solves the total: min(q, q_norm) * barrels is
    # min(q * barrels, q_norm * barrels) (dwflow.c:292 newFlow = q * barrels)
    return ConduitTerms(
        flow_class=cls,
        flow_offset=flow_offset * barrels,
        conveyance=conveyance,
        momentum_flow=momentum * barrels,
        limited=limited,
        normal_flow=normal_flow * barrels,
        slope_switch=slope_switch,
        switch_drop=(z1 - z2).expand(shape),
        switch_normal_flow=torch.where(slope_switch, q_norm, inf) * barrels,
        dry_upstream=depth1 <= FUDGE,                                          # :280
        dry_downstream=depth2 <= FUDGE,                                        # :281
        surface_area1=surf1 * barrels,                                         # dynwave.c:575
        surface_area2=surf2 * barrels,
        mid_area=torch.where(dry_exit, a_end, a_mid),                          # :171, :284
        volume=a_end * geo.length * barrels,                                   # :177, :291
        depth=torch.minimum(y_mid, yf),                                        # :176, :287
        end_depth1=y1,
        end_depth2=y2,
        froude=torch.where(dry_exit, zero, fr),                                # :175, :189
        lagged_conveyance1=lagged1,
        lagged_conveyance2=lagged2,
        drop_depth1=drop_depth1,
        drop_depth2=drop_depth2,
        drop_area1=drop_area1 * barrels,
        drop_area2=drop_area2 * barrels,
        drop_above1=drop_above1,
        drop_above2=drop_above2,
    )


__all__ = [
    "DAMPING", "DN_CRITICAL", "DN_DRY", "DRY", "EXTRAN_CROWN_CUTOFF", "FUDGE", "MAX_VELOCITY",
    "NORMAL_FLOW_LIMITED", "SLOT_CROWN_CUTOFF", "SUBCRITICAL", "SUPCRITICAL",
    "SURCHARGE_METHODS", "UP_CRITICAL", "UP_DRY",
    "ConduitGeometry", "ConduitTerms", "conduit_terms", "lengthening_factor",
]
