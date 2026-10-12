"""SWMM 5.2.4's culvert inlet control (`culvert.c::culvert_getInflow`), FHWA HEC-5 equations.

A conduit with a culvert code (`[XSECTIONS]` field 8, 1 to 57) is limited, whenever it is not
flowing full, to the flow its inlet can pass for the head at its upstream invert: unsubmerged
(head below 0.95 of the full depth) from the code's form-1 (an implicit critical-depth
equation, solved by Ridder's method) or form-2 power law, submerged (head above ``yFull (16 C
+ Y - scf)``) from an orifice equation, and in between interpolated linearly. It replaces the
normal-flow limit for the conduit (`dwflow.c:250-258`). A culvert code above 57 is ignored,
as SWMM does.

`CulvertSet` holds the culverts of a network (the codes, slopes and sections) and the two
flows at the ends of the transition zone, which depend on the section only; `inlet_flow`
evaluates the inlet-control flow for the heads of one pass. The form-1 equation is the one
SWMM evaluates -- and its flow is the value at the LAST evaluation of Ridder's iteration,
not at the root (``culvert.qc`` is a side effect of ``form1Eqn``), which `xsect_shapes.ridder`
tracks.
"""

from __future__ import annotations

import torch

from noodl.apps.sewer.xsect_shapes import XsectSet, ridder
from noodl.apps.sewer.xsect_tables import GRAVITY_FT

Tensor = torch.Tensor
F64 = torch.float64

#: the largest culvert code (`culvert.c:29`); a larger one is ignored
MAX_CULVERT_CODE = 57
#: SWMM's ``BIG`` (`consts.h`)
BIG = 1.0e10
#: ``Params[code] = (FORM, K, M, C, Y)`` (`culvert.c:30-135`), code 0 first
PARAMS = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0098, 2.00, 0.0398, 0.67), (1.0, 0.0018, 2.00, 0.0292, 0.74),
    (1.0, 0.0045, 2.00, 0.0317, 0.69),
    (1.0, 0.0078, 2.00, 0.0379, 0.69), (1.0, 0.0210, 1.33, 0.0463, 0.75),
    (1.0, 0.0340, 1.50, 0.0553, 0.54),
    (1.0, 0.0018, 2.50, 0.0300, 0.74), (1.0, 0.0018, 2.50, 0.0243, 0.83),
    (1.0, 0.026, 1.0, 0.0347, 0.81), (1.0, 0.061, 0.75, 0.0400, 0.80),
    (1.0, 0.061, 0.75, 0.0423, 0.82),
    (2.0, 0.510, 0.667, 0.0309, 0.80), (2.0, 0.486, 0.667, 0.0249, 0.83),
    (2.0, 0.515, 0.667, 0.0375, 0.79), (2.0, 0.495, 0.667, 0.0314, 0.82),
    (2.0, 0.486, 0.667, 0.0252, 0.865),
    (2.0, 0.545, 0.667, 0.04505, 0.73), (2.0, 0.533, 0.667, 0.0425, 0.705),
    (2.0, 0.522, 0.667, 0.0402, 0.68), (2.0, 0.498, 0.667, 0.0327, 0.75),
    (2.0, 0.497, 0.667, 0.0339, 0.803), (2.0, 0.493, 0.667, 0.0361, 0.806),
    (2.0, 0.495, 0.667, 0.0386, 0.71),
    (2.0, 0.497, 0.667, 0.0302, 0.835), (2.0, 0.495, 0.667, 0.0252, 0.881),
    (2.0, 0.493, 0.667, 0.0227, 0.887),
    (1.0, 0.0083, 2.00, 0.0379, 0.69), (1.0, 0.0145, 1.75, 0.0419, 0.64),
    (1.0, 0.0340, 1.50, 0.0496, 0.57),
    (1.0, 0.0100, 2.00, 0.0398, 0.67), (1.0, 0.0018, 2.50, 0.0292, 0.74),
    (1.0, 0.0045, 2.00, 0.0317, 0.69),
    (1.0, 0.0100, 2.00, 0.0398, 0.67), (1.0, 0.0018, 2.50, 0.0292, 0.74),
    (1.0, 0.0095, 2.00, 0.0317, 0.69),
    (1.0, 0.0083, 2.00, 0.0379, 0.69), (1.0, 0.0300, 1.00, 0.0463, 0.75),
    (1.0, 0.0340, 1.50, 0.0496, 0.57),
    (1.0, 0.0300, 1.50, 0.0496, 0.57), (1.0, 0.0088, 2.00, 0.0368, 0.68),
    (1.0, 0.0030, 2.00, 0.0269, 0.77),
    (1.0, 0.0300, 1.50, 0.0496, 0.57), (1.0, 0.0088, 2.00, 0.0368, 0.68),
    (1.0, 0.0030, 2.00, 0.0269, 0.77),
    (1.0, 0.0083, 2.00, 0.0379, 0.69), (1.0, 0.0300, 1.00, 0.0473, 0.75),
    (1.0, 0.0340, 1.50, 0.0496, 0.57),
    (2.0, 0.534, 0.555, 0.0196, 0.90), (2.0, 0.519, 0.640, 0.0210, 0.90),
    (2.0, 0.536, 0.622, 0.0368, 0.83), (2.0, 0.5035, 0.719, 0.0478, 0.80),
    (2.0, 0.547, 0.800, 0.0598, 0.75),
    (2.0, 0.475, 0.667, 0.0179, 0.97),
    (2.0, 0.560, 0.667, 0.0446, 0.85), (2.0, 0.560, 0.667, 0.0378, 0.87),
    (2.0, 0.500, 0.667, 0.0446, 0.65), (2.0, 0.500, 0.667, 0.0378, 0.71),
)
_TABLE = torch.tensor(PARAMS, dtype=F64)
#: the mitered inlets, whose slope correction is ``-7 slope`` instead of ``0.5 slope``
_MITERED = (5, 37, 46)


class CulvertSet:
    """The culverts of a network.

    ``index`` (long, the conduits that are culverts), ``code`` (long), ``slope`` (m/m, the
    conduits' positive slopes), ``y_full`` and ``a_full`` (ft, ft2; the sections') and
    ``xsect`` (the `XsectSet` of the culvert conduits, ordered as `index`). The flows at the
    ends of the transition zone (``q_low`` at ``0.95 yFull``, ``q_high`` at the submerged
    limit ``yHigh``) depend on the section only and are computed once."""

    def __init__(self, index: Tensor, code: Tensor, slope: Tensor, y_full: Tensor,
                 a_full: Tensor, xsect: XsectSet) -> None:
        self.index, self.code, self.xsect = index, code, xsect
        self.y_full, self.a_full = y_full, a_full
        p = _TABLE[code]
        self.form, self.k, self.m, self.c, self.yy = (p[:, i] for i in range(5))
        self.ad = a_full * torch.sqrt(y_full)                                 # culvert.c:202
        mitered = torch.zeros_like(code, dtype=torch.bool)
        for m in _MITERED:
            mitered = mitered | (code == m)
        self.scf = torch.where(mitered, -7.0 * slope, 0.5 * slope)            # culvert.c:205-211
        self.y_high = y_full * (16.0 * self.c + self.yy - self.scf)            # culvert.c:218
        self.y_low = 0.95 * y_full                                            # culvert.c:227
        self.q_low = self._unsubmerged(self.y_low)
        self.q_high = self._submerged(self.y_high)

    def _form1(self, h: Tensor) -> Tensor:
        """`getForm1Flow`: Ridder's method on ``form1Eqn`` over ``[0.01 h, h]`` (tolerance
        0.001 ft); the flow at its last evaluation."""
        h_plus = h / self.y_full + self.scf
        xs = self.xsect

        def residual(yc):
            ac, wc = xs.area(yc), xs.width(yc)
            yh = ac / torch.where(wc > 0.0, wc, torch.ones_like(wc))
            qc = ac * torch.sqrt(GRAVITY_FT * yh)
            return (h_plus - yc / self.y_full - yh / 2.0 / self.y_full
                    - self.k * torch.pow(qc / self.ad, self.m))

        if not bool((self.form == 1.0).any()):
            return torch.zeros_like(h)
        live = h > 0.0
        h_safe = torch.where(live, h, self.y_full)
        _, last = ridder(residual, 0.01 * h_safe, h_safe, 0.001, track=True)
        ac, wc = xs.area(last), xs.width(last)
        yh = ac / torch.where(wc > 0.0, wc, torch.ones_like(wc))
        return torch.where(live, ac * torch.sqrt(GRAVITY_FT * yh), torch.zeros_like(h))

    def _unsubmerged(self, h: Tensor) -> Tensor:
        """`getUnsubmergedFlow` for the inlet depth `h` (ft above the invert)."""
        arg = h / self.y_full / self.k
        power = self.ad * torch.pow(torch.clamp(arg, min=0.0), 1.0 / self.m)
        return torch.where(self.form == 1.0, self._form1(h), power)

    def _submerged(self, h: Tensor) -> Tensor:
        """`getSubmergedFlow`: the orifice equation, ``BIG`` where its argument is not
        positive."""
        arg = (h / self.y_full - self.yy + self.scf) / self.c
        pos = arg > 0.0
        return torch.where(pos, torch.sqrt(torch.where(pos, arg, torch.ones_like(arg)))
                           * self.ad, torch.full_like(arg, BIG))

    def inlet_flow(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """``(q, dq/dh)`` of each culvert's inlet control for the head `y` (ft) above its
        upstream invert (shape (..., n culverts)): the flow (cfs, one barrel's) of
        `culvert_getInflow` before its ``q < q0`` comparison, and SWMM's ``dQdH``
        (cfs/ft: ``q / h / M`` unsubmerged, ``0.5 q / arg / yFull / C`` submerged, the chord
        slope in the transition zone, 0 where the submerged argument is not positive)."""
        submerged = y >= self.y_high
        unsub = y <= self.y_low
        q_sub = self._submerged(y)
        y_un = torch.where(unsub, y, self.y_low)
        q_un = self._unsubmerged(y_un)
        span = self.y_high - self.y_low
        slope_tr = (self.q_high - self.q_low) / span
        q_tr = self.q_low + slope_tr * (y - self.y_low)
        arg = (y / self.y_full - self.yy + self.scf) / self.c
        pos = arg > 0.0
        d_sub = torch.where(pos, 0.5 * q_sub / torch.where(pos, arg, torch.ones_like(arg))
                            / self.y_full / self.c, torch.zeros_like(arg))
        d_un = q_un / torch.where(y_un > 0.0, y_un, torch.ones_like(y_un)) / self.m
        q = torch.where(submerged, q_sub, torch.where(unsub, q_un, q_tr))
        d = torch.where(submerged, d_sub, torch.where(unsub, d_un, slope_tr))
        return q, d
