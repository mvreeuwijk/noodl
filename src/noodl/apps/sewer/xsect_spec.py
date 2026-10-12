"""SWMM 5.2.4's cross-section definitions (`[XSECTIONS]`, `[TRANSECTS]`, SHAPE `[CURVES]`), as
plain data, and their derived constants in SWMM's own feet, operation for operation.

`Xsect` is one conduit's section as SWMM reads it (`link.c::link_readXsectParams`): the shape
keyword and the four geometry fields, lengths in m. `Xsect.params` runs `xsect.c::xsect_setParams`
(and `xsect_setCustomXsectParams`, `getTransectParams`) on it with SWMM's own ``ucf``
(m per ft, `xsect_tables.LCF`) and returns the constants of the section in ft -- ``y_full``,
``a_full``, ``r_full``, ``s_full``, ``s_max``, ``w_max``, SWMM's per-shape ``y_bot``, ``a_bot``,
``s_bot``, ``r_bot`` -- which `xsect_shapes.XsectSet` evaluates. A `Transect` is one `[TRANSECTS]`
entry (`transect.c`), whose 51-point tables `transect_tables` builds as SWMM does; a CUSTOM shape's
width curve becomes tables the same way (`shape.c`, `custom_tables`).

Shapes. ``CIRCULAR`` is the tabulated circle of `xsect_tables` (a `Pipe` without an `Xsect`).
Modelled: ``FILLED_CIRCULAR``, ``RECT_CLOSED``, ``RECT_OPEN``, ``TRAPEZOIDAL``, ``TRIANGULAR``,
``PARABOLIC``, ``POWER``, ``RECT_TRIANGULAR``, ``RECT_ROUND``, ``MODBASKETHANDLE``,
``HORIZ_ELLIPSE``, ``VERT_ELLIPSE``, ``ARCH``, ``EGG``, ``HORSESHOE``, ``GOTHIC``,
``CATENARY``, ``SEMIELLIPTICAL``, ``BASKETHANDLE``, ``SEMICIRCULAR``, ``IRREGULAR`` (a
`Transect`), ``CUSTOM`` (a shape curve) and ``FORCE_MAIN``. ``DUMMY`` and ``STREET`` are not
(`UNMODELLED_KINDS`).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch

from noodl.apps.sewer import xsect_dat as dat
from noodl.apps.sewer import xsect_tables as xt
from noodl.apps.sewer.xsect_tables import LCF

#: SWMM's cross-section shapes, in its ``XsectType`` order (`enums.h`), by `.inp` keyword.
KINDS = (
    "DUMMY", "CIRCULAR", "FILLED_CIRCULAR", "RECT_CLOSED", "RECT_OPEN", "TRAPEZOIDAL",
    "TRIANGULAR", "PARABOLIC", "POWER", "RECT_TRIANGULAR", "RECT_ROUND", "MODBASKETHANDLE",
    "HORIZ_ELLIPSE", "VERT_ELLIPSE", "ARCH", "EGG", "HORSESHOE", "GOTHIC", "CATENARY",
    "SEMIELLIPTICAL", "BASKETHANDLE", "SEMICIRCULAR", "IRREGULAR", "CUSTOM", "FORCE_MAIN",
    "STREET",
)
#: The shapes this package does not model: a link carrying one is refused by name.
UNMODELLED_KINDS = {
    "DUMMY": "a DUMMY conduit routes its upstream node's outflow as a link without a section "
             "(dynwave.c::findNonConduitFlow) and is not modelled",
    "STREET": "the STREET cross section (SWMM 5.2's street/gutter [STREETS]) is not modelled",
}
#: ``Amax`` (`xsect.c`): the area at maximum flow over the full area; 1 for an open shape.
AMAX = {
    "DUMMY": 1.0, "CIRCULAR": 0.9756, "FILLED_CIRCULAR": 0.9756, "RECT_CLOSED": 0.97,
    "RECT_OPEN": 1.0, "TRAPEZOIDAL": 1.0, "TRIANGULAR": 1.0, "PARABOLIC": 1.0, "POWER": 1.0,
    "RECT_TRIANGULAR": 0.98, "RECT_ROUND": 0.98, "MODBASKETHANDLE": 0.96,
    "HORIZ_ELLIPSE": 0.96, "VERT_ELLIPSE": 0.96, "ARCH": 0.92, "EGG": 0.96, "HORSESHOE": 0.96,
    "GOTHIC": 0.96, "CATENARY": 0.98, "SEMIELLIPTICAL": 0.98, "BASKETHANDLE": 0.96,
    "SEMICIRCULAR": 0.96, "IRREGULAR": 1.0, "CUSTOM": 0.96, "FORCE_MAIN": 0.9756,
    "STREET": 1.0,
}
#: Which of ``Geom1..Geom4`` are lengths (converted to m by the reader), per shape; the others
#: (side slopes, exponents, size codes, a roughness) are taken as written.
LENGTH_FIELDS = {
    "CIRCULAR": (0,), "FILLED_CIRCULAR": (0, 1), "RECT_CLOSED": (0, 1), "RECT_OPEN": (0, 1),
    "TRAPEZOIDAL": (0, 1), "TRIANGULAR": (0, 1), "PARABOLIC": (0, 1), "POWER": (0, 1),
    "RECT_TRIANGULAR": (0, 1, 2), "RECT_ROUND": (0, 1, 2), "MODBASKETHANDLE": (0, 1, 2),
    "HORIZ_ELLIPSE": (0, 1), "VERT_ELLIPSE": (0, 1), "ARCH": (0, 1), "EGG": (0,),
    "HORSESHOE": (0,), "GOTHIC": (0,), "CATENARY": (0,), "SEMIELLIPTICAL": (0,),
    "BASKETHANDLE": (0,), "SEMICIRCULAR": (0,), "CUSTOM": (0,), "FORCE_MAIN": (0,),
}
_PI = 3.141592654          # consts.h
_RECT_ALFMAX = 0.97        # xsect.c
_RECT_TRIANG_ALFMAX = 0.98
_RECT_ROUND_ALFMAX = 0.98
#: ``N_TRANSECT_TBL`` and ``N_SHAPE_TBL`` (`objects.h`): points of a transect's or shape's tables.
N_TABLE = 51


def is_open(kind: str) -> bool:
    """`xsect.c::xsect_isOpen`: ``Amax >= 1``."""
    return AMAX[kind] >= 1.0


@dataclass(frozen=True)
class Transect:
    """One `[TRANSECTS]` entry (`transect.c`), SI. ``stations`` (m, across the section, with
    the ``X1`` station multiplier applied) and ``elevations`` (m, with the ``yFactor``
    offset applied); ``left_bank``, ``right_bank`` (m, station multiplier applied) end the
    overbanks; ``n_left``, ``n_right``, ``n_channel`` the Manning n values SWMM holds when it
    validates the transect (`transect_validate`: the main channel's, as stored, is the
    conduit's roughness) and ``length_factor`` the ratio of the main channel to the
    flood-plain length (``Lfactor``, `X1` field 8). SWMM scales the channel's n by
    ``sqrt(length_factor)`` in the tables and the conduit's roughness likewise
    (`link.c::conduit_validate`), and divides the conduit's length by ``length_factor``
    (`conduit_getLength`)."""

    name: str
    stations: tuple[float, ...]
    elevations: tuple[float, ...]
    left_bank: float
    right_bank: float
    n_left: float
    n_right: float
    n_channel: float
    length_factor: float = 1.0


@dataclass(frozen=True)
class Xsect:
    """A conduit's cross section as SWMM reads it (`[XSECTIONS]`), SI.

    ``kind`` is a SWMM shape keyword (`KINDS`), ``geom`` its ``Geom1..Geom4`` with the
    lengths (`LENGTH_FIELDS`) in m and the rest as written. ``transect`` is the `Transect`
    of an ``IRREGULAR`` section; ``curve`` the SHAPE curve of a ``CUSTOM`` one, its ``(height
    / full height, width / full height)`` points. For ``FORCE_MAIN`` ``geom[1]`` is the
    Hazen-Williams C factor or the Darcy-Weisbach roughness height (m). For
    ``HORIZ_ELLIPSE``, ``VERT_ELLIPSE`` and ``ARCH`` ``geom[2]`` is the size code SWMM
    uses (the reader applies its rule that a zero ``Geom2`` makes ``Geom1`` the code,
    `xsect.c:553, 581, 609`), 0 for a non-standard size."""

    kind: str
    geom: tuple[float, ...] = ()
    transect: Transect | None = None
    curve: tuple[tuple[float, float], ...] | None = None

    @property
    def height(self) -> float:
        """The conduit's full height in m, ``Pipe.diameter``'s counterpart: the section's
        ``yFull`` (plus the filled bottom of a FILLED_CIRCULAR, whose diameter it is)."""
        p = self.params()
        return (p["y_full"] + p.get("y_bot", 0.0) * (self.kind == "FILLED_CIRCULAR")) * LCF

    @property
    def open(self) -> bool:
        """Whether the shape is open (`is_open`)."""
        return is_open(self.kind)

    def params(self) -> dict[str, float]:
        """The section's constants in ft (the module docstring), `xsect.c::xsect_setParams`
        with ``ucf = LCF`` (SWMM's ``p[i] / ucf``), then ``xsect_setCustomXsectParams`` /
        ``getTransectParams``. Raises ValueError where SWMM's ``xsect_setParams`` returns
        FALSE (a non-positive dimension, an unknown size code, ...)."""
        return _params(self)


def _bad(kind: str, why: str):
    raise ValueError(f"{kind}: {why} (SWMM xsect_setParams rejects this section)")


def _power(x: float, y: float) -> float:
    return math.pow(x, y)


def _stock(yf: float, a: float, r: float, smax: float, w: float) -> dict[str, float]:
    """A stock tabulated shape from its ``aFull``/``rFull``/``wMax`` ratios to ``yFull``."""
    a_full = a * yf * yf
    r_full = r * yf
    s_full = a_full * _power(r_full, 2. / 3.)
    return {"y_full": yf, "a_full": a_full, "r_full": r_full, "s_full": s_full,
            "s_max": smax * s_full, "w_max": w * yf}


#: ``(aFull / yFull^2, rFull / yFull, sMax / sFull, wMax / yFull)`` of the stock tabulated
#: shapes (`xsect.c:295-363`).
_STOCK = {
    "EGG": (0.5105, 0.1931, 1.065, 2. / 3.), "HORSESHOE": (0.8293, 0.2538, 1.077, 1.0),
    "GOTHIC": (0.6554, 0.2269, 1.065, 0.84), "CATENARY": (0.70277, 0.23172, 1.05, 0.9),
    "SEMIELLIPTICAL": (0.785, 0.242, 1.045, 1.0), "BASKETHANDLE": (0.7862, 0.2464, 1.06078, 0.944),
    "SEMICIRCULAR": (1.2697, 0.2946, 1.06637, 1.64),
}


def _tab(x: float, table: torch.Tensor) -> float:
    """`xsect.c::lookup` of `x` in `table`, as a float."""
    return float(xt.lookup(torch.tensor(x, dtype=torch.float64), table))


def _params(x: Xsect) -> dict[str, float]:
    kind, p = x.kind, list(x.geom) + [0.0] * (4 - len(x.geom))
    if kind in UNMODELLED_KINDS:
        raise ValueError(f"cross-section {kind}: {UNMODELLED_KINDS[kind]}")
    if kind not in KINDS:
        raise ValueError(f"unknown cross-section shape {kind!r}")
    ucf = LCF
    if kind == "CIRCULAR":
        yf = p[0] / ucf
        af = _PI / 4.0 * yf * yf
        rf = 0.2500 * yf
        sf = af * _power(rf, 2. / 3.)
        return {"y_full": yf, "a_full": af, "r_full": rf, "s_full": sf, "s_max": 1.08 * sf,
                "w_max": yf}
    if kind == "IRREGULAR":
        if x.transect is None:
            _bad(kind, "no transect")
        t = transect_tables(x.transect)
        return {"y_full": t["y_full"], "a_full": t["a_full"], "r_full": t["r_full"],
                "s_full": t["a_full"] * _power(t["r_full"], 2. / 3.), "s_max": t["s_max"],
                "w_max": t["w_max"], "a_bot": t["a_max"]}
    if p[0] <= 0.0:
        _bad(kind, "Geom1 must be positive")
    yf = p[0] / ucf
    if kind == "FORCE_MAIN":
        af = _PI / 4.0 * yf * yf
        rf = 0.2500 * yf
        sf = af * _power(rf, 0.63)
        # `xsect.rBot`: the C factor, or (Darcy-Weisbach, converted to ft by the reader's
        # `Xsect.geom` in m) the roughness height
        return {"y_full": yf, "a_full": af, "r_full": rf, "s_full": sf, "s_max": 1.06949 * sf,
                "w_max": yf, "roughness": p[1]}
    if kind == "FILLED_CIRCULAR":
        if p[1] >= p[0]:
            _bad(kind, "the filled depth must be below the diameter")
        wmax = yf
        af = _PI / 4.0 * yf * yf
        rf = 0.2500 * yf
        y_bot = p[1] / ucf
        a_bot = af * _tab(y_bot / yf, xt._A)
        # xsect.c:281 reads the width through ``xsect_getWofY``, whose FILLED_CIRCULAR case
        # shifts the depth by ``yBot`` again: ``(yBot + yBot) / (yFull + yBot)`` (SWMM's own)
        s_bot = wmax * _tab((y_bot + y_bot) / (yf + y_bot), xt._W)
        r_bot = a_bot / (rf * _tab(y_bot / yf, xt._R))
        af -= a_bot
        rf = af / (_PI * yf - r_bot + s_bot)
        sf = af * _power(rf, 2. / 3.)
        return {"y_full": yf - y_bot, "a_full": af, "r_full": rf, "s_full": sf,
                "s_max": 1.08 * sf, "w_max": wmax, "y_bot": y_bot, "a_bot": a_bot,
                "s_bot": s_bot, "r_bot": r_bot}
    if kind in _STOCK:
        return _stock(yf, *_STOCK[kind])
    if kind == "RECT_CLOSED":
        if p[1] <= 0.0:
            _bad(kind, "Geom2 (width) must be positive")
        wm = p[1] / ucf
        af = yf * wm
        rf = af / (2.0 * (yf + wm))
        sf = af * _power(rf, 2. / 3.)
        out = {"y_full": yf, "a_full": af, "r_full": rf, "s_full": sf, "w_max": wm}
        amax = _RECT_ALFMAX * af
        p_wet = wm + 2. * amax / wm
        if amax / af > _RECT_ALFMAX:
            p_wet += (amax / af - _RECT_ALFMAX) / (1.0 - _RECT_ALFMAX) * wm
        out["s_max"] = amax * _power(amax / p_wet, 2. / 3.)
        return out
    if kind == "RECT_OPEN":
        if p[1] <= 0.0:
            _bad(kind, "Geom2 (width) must be positive")
        if p[2] < 0.0 or p[2] > 2.0:
            _bad(kind, "Geom3 (sides to ignore) must be 0 to 2")
        wm = p[1] / ucf
        af = yf * wm
        rf = af / ((2.0 - p[2]) * yf + wm)
        sf = af * _power(rf, 2. / 3.)
        return {"y_full": yf, "a_full": af, "r_full": rf, "s_full": sf, "s_max": sf,
                "w_max": wm, "s_bot": p[2]}
    if kind == "RECT_TRIANGULAR":
        if p[1] <= 0.0 or p[2] <= 0.0:
            _bad(kind, "Geom2 and Geom3 must be positive")
        wm = p[1] / ucf
        y_bot = p[2] / ucf
        a_bot = y_bot * wm / 2.0
        s_bot = wm / y_bot / 2.0
        r_bot = math.sqrt(1. + s_bot * s_bot)
        af = wm * (yf - y_bot / 2.0)
        rf = af / (2.0 * y_bot * r_bot + 2.0 * (yf - y_bot) + wm)
        sf = af * _power(rf, 2. / 3.)
        out = {"y_full": yf, "a_full": af, "r_full": rf, "s_full": sf, "w_max": wm,
               "y_bot": y_bot, "a_bot": a_bot, "s_bot": s_bot, "r_bot": r_bot}
        amax = _RECT_TRIANG_ALFMAX * af
        out["s_max"] = amax * _power(_rect_triang_r_of_a(out, amax), 2. / 3.)
        return out
    if kind == "RECT_ROUND":
        if p[1] <= 0.0:
            _bad(kind, "Geom2 (width) must be positive")
        if p[2] < p[1] / 2.0:
            p[2] = p[1] / 2.0
        wm = p[1] / ucf
        r_bot = p[2] / ucf
        theta = 2.0 * math.asin(wm / 2.0 / r_bot)
        a_bot = r_bot * r_bot / 2.0 * (theta - math.sin(theta))
        s_bot = _PI * r_bot * r_bot * _power(r_bot / 2.0, 2. / 3.)
        y_bot = r_bot * (1.0 - math.cos(theta / 2.0))
        if y_bot > yf:
            _bad(kind, "the rounded bottom is deeper than the section")
        af = wm * (yf - y_bot) + a_bot
        rf = af / (r_bot * theta + 2.0 * (yf - y_bot) + wm)
        sf = af * _power(rf, 2. / 3.)
        out = {"y_full": yf, "a_full": af, "r_full": rf, "s_full": sf, "w_max": wm,
               "y_bot": y_bot, "a_bot": a_bot, "s_bot": s_bot, "r_bot": r_bot}
        amax = _RECT_ROUND_ALFMAX * af
        out["s_max"] = amax * _power(_rect_round_r_of_a(out, amax), 2. / 3.)
        return out
    if kind == "MODBASKETHANDLE":
        if p[1] <= 0.0:
            _bad(kind, "Geom2 (width) must be positive")
        if p[2] < p[1] / 2.0:
            p[2] = p[1] / 2.0
        wm = p[1] / ucf
        r_bot = p[2] / ucf
        theta = 2.0 * math.asin(wm / 2.0 / r_bot)
        s_bot = theta
        y_bot = r_bot * (1.0 - math.cos(theta / 2.0))
        if y_bot > yf:
            _bad(kind, "the circular top is higher than the section")
        a_bot = r_bot * r_bot / 2.0 * (theta - math.sin(theta))
        af = (yf - y_bot) * wm + a_bot
        rf = af / (r_bot * theta + 2.0 * (yf - y_bot) + wm)
        sf = af * _power(rf, 2. / 3.)
        out = {"y_full": yf, "a_full": af, "r_full": rf, "s_full": sf, "w_max": wm,
               "y_bot": y_bot, "a_bot": a_bot, "s_bot": s_bot, "r_bot": r_bot}
        a_m = AMAX["MODBASKETHANDLE"] * af
        r_m = _mod_basket_r_of_a(out, a_m)
        out["s_max"] = a_m * _power(r_m, 2. / 3.) if a_m > 0 else 0.0
        return out
    if kind == "TRAPEZOIDAL":
        if p[1] < 0.0 or p[2] < 0.0 or p[3] < 0.0:
            _bad(kind, "Geom2 to Geom4 must be non-negative")
        y_bot = p[1] / ucf
        s_bot = (p[2] + p[3]) / 2.0
        if y_bot == 0.0 and s_bot == 0.0:
            _bad(kind, "the bottom width and the side slopes are both zero")
        r_bot = math.sqrt(1.0 + p[2] * p[2]) + math.sqrt(1.0 + p[3] * p[3])
        wm = y_bot + yf * (p[2] + p[3])
        af = (y_bot + s_bot * yf) * yf
        rf = af / (y_bot + yf * r_bot)
        sf = af * _power(rf, 2. / 3.)
        return {"y_full": yf, "a_full": af, "r_full": rf, "s_full": sf, "s_max": sf,
                "w_max": wm, "y_bot": y_bot, "s_bot": s_bot, "r_bot": r_bot}
    if kind == "TRIANGULAR":
        if p[1] <= 0.0:
            _bad(kind, "Geom2 (top width) must be positive")
        wm = p[1] / ucf
        s_bot = wm / yf / 2.
        r_bot = math.sqrt(1. + s_bot * s_bot)
        af = yf * yf * s_bot
        rf = af / (2.0 * yf * r_bot)
        sf = af * _power(rf, 2. / 3.)
        return {"y_full": yf, "a_full": af, "r_full": rf, "s_full": sf, "s_max": sf,
                "w_max": wm, "s_bot": s_bot, "r_bot": r_bot}
    if kind == "PARABOLIC":
        if p[1] <= 0.0:
            _bad(kind, "Geom2 (top width) must be positive")
        wm = p[1] / ucf
        r_bot = wm / 2.0 / math.sqrt(yf)
        af = (2. / 3.) * yf * wm
        out = {"y_full": yf, "a_full": af, "w_max": wm, "r_bot": r_bot}
        out["r_full"] = _parab_r_of_y(out, yf)
        out["s_full"] = af * _power(out["r_full"], 2. / 3.)
        out["s_max"] = out["s_full"]
        return out
    if kind == "POWER":
        if p[1] <= 0.0 or p[2] <= 0.0:
            _bad(kind, "Geom2 and Geom3 must be positive")
        wm = p[1] / ucf
        s_bot = 1.0 / p[2]
        r_bot = wm / (s_bot + 1) / _power(yf, s_bot)
        af = yf * wm / (s_bot + 1)
        out = {"y_full": yf, "a_full": af, "w_max": wm, "s_bot": s_bot, "r_bot": r_bot}
        out["r_full"] = _power_r_of_y(out, yf)
        out["s_full"] = af * _power(out["r_full"], 2. / 3.)
        out["s_max"] = out["s_full"]
        return out
    if kind in ("HORIZ_ELLIPSE", "VERT_ELLIPSE"):
        code = p[2]
        if code > 0.0:                                   # a standard ellipse pipe
            index = math.floor(code) - 1
            if index < 0 or index >= len(dat.MINORAXIS_ELLIPSE):
                _bad(kind, "unknown standard size code")
            minor, major = dat.MINORAXIS_ELLIPSE[index] / 12., dat.MAJORAXIS_ELLIPSE[index] / 12.
            if kind == "HORIZ_ELLIPSE":
                yf, wm = minor, major
            else:
                yf, wm = major, minor
            af, rf = dat.AFULL_ELLIPSE[index], dat.RFULL_ELLIPSE[index]
        else:
            if p[1] < 0.0:
                _bad(kind, "Geom2 must not be negative")
            wm = p[1] / ucf
            if kind == "HORIZ_ELLIPSE":
                af, rf = 1.2692 * yf * yf, 0.3061 * yf
            else:
                af, rf = 1.2692 * wm * wm, 0.3061 * wm
        sf = af * _power(rf, 2. / 3.)
        return {"y_full": yf, "a_full": af, "r_full": rf, "s_full": sf, "s_max": sf,
                "w_max": wm}
    if kind == "ARCH":
        code = p[2]
        if code > 0.0:
            index = math.floor(code) - 1
            if index < 0 or index >= len(dat.YFULL_ARCH):
                _bad(kind, "unknown standard size code")
            yf, wm = dat.YFULL_ARCH[index] / 12., dat.WMAX_ARCH[index] / 12.
            af, rf = dat.AFULL_ARCH[index], dat.RFULL_ARCH[index]
        else:
            if p[1] < 0.0:
                _bad(kind, "Geom2 must not be negative")
            wm = p[1] / ucf
            af, rf = 0.7879 * yf * wm, 0.2991 * yf
        sf = af * _power(rf, 2. / 3.)
        return {"y_full": yf, "a_full": af, "r_full": rf, "s_full": sf, "s_max": sf,
                "w_max": wm}
    if kind == "CUSTOM":
        if x.curve is None:
            _bad(kind, "no SHAPE curve")
        c = custom_tables(x.curve)
        af = c["a_full"] * yf * yf
        rf = c["r_full"] * yf
        sf = af * _power(rf, 2. / 3.)
        return {"y_full": yf, "a_full": af, "r_full": rf, "s_full": sf,
                "s_max": c["s_max"] * yf * yf * _power(yf, 2. / 3.), "w_max": c["w_max"] * yf,
                "a_bot": c["a_max"] * yf * yf}
    raise ValueError(f"cross-section {kind!r} is not modelled")        # pragma: no cover


def _rect_triang_y_of_a(q: dict, a: float) -> float:
    if a <= q["a_bot"]:
        return math.sqrt(a / q["s_bot"])
    return q["y_bot"] + (a - q["a_bot"]) / q["w_max"]


def _rect_triang_r_of_a(q: dict, a: float) -> float:
    """`xsect.c::rect_triang_getRofA`."""
    if a <= 0.0:
        return 0.0
    y = _rect_triang_y_of_a(q, a)
    if y <= q["y_bot"]:
        return a / (2. * y * q["r_bot"])
    p = 2. * q["y_bot"] * q["r_bot"] + 2. * (y - q["y_bot"])
    alf = (a / q["a_full"]) - _RECT_TRIANG_ALFMAX
    if alf > 0.0:
        p += alf / (1.0 - _RECT_TRIANG_ALFMAX) * q["w_max"]
    return a / p


def _circ_y_of_a_norm(alpha: float) -> float:
    """`xsect.c::circ_getYofA` / ``yFull``: ``Y/Yfull`` of ``A/Afull``."""
    a = torch.tensor(alpha, dtype=torch.float64)
    return float(xt.y_circular(a) if alpha < 0.04 else xt.lookup(a, xt._Y))


def _rect_round_r_of_a(q: dict, a: float) -> float:
    """`xsect.c::rect_round_getRofA`."""
    if a <= 0.0:
        return 0.0
    if a > q["a_bot"]:
        y1 = (a - q["a_bot"]) / q["w_max"]
        theta1 = 2.0 * math.asin(q["w_max"] / 2.0 / q["r_bot"])
        p = q["r_bot"] * theta1 + 2.0 * y1
        arg = (a / q["a_full"]) - _RECT_ROUND_ALFMAX
        if arg > 0.0:
            p += arg / (1.0 - _RECT_ROUND_ALFMAX) * q["w_max"]
        return a / p
    alpha = a / (_PI * q["r_bot"] * q["r_bot"])
    y1 = (2.0 * q["r_bot"]) * _circ_y_of_a_norm(alpha)
    theta1 = 2.0 * math.acos(1.0 - y1 / q["r_bot"])
    return a / (q["r_bot"] * theta1)


def _mod_basket_y_of_a(q: dict, a: float) -> float:
    if a <= q["a_full"] - q["a_bot"]:
        return a / q["w_max"]
    alpha = (q["a_full"] - a) / (_PI * q["r_bot"] * q["r_bot"])
    y1 = 2.0 * q["r_bot"] * _circ_y_of_a_norm(alpha)
    return q["y_full"] - y1


def _mod_basket_r_of_a(q: dict, a: float) -> float:
    """`xsect.c::mod_basket_getRofA`."""
    if a <= q["a_full"] - q["a_bot"]:
        return a / (q["w_max"] + 2.0 * a / q["w_max"])
    y1 = q["y_full"] - _mod_basket_y_of_a(q, a)
    theta1 = 2.0 * math.acos(1.0 - y1 / q["r_bot"])
    p = (q["s_bot"] - theta1) * q["r_bot"]
    y1 = q["y_full"] - q["y_bot"]
    p = p + 2.0 * y1 + q["w_max"]
    return a / p


def _parab_p_of_y(q: dict, y: float) -> float:
    x = 2. * math.sqrt(y) / q["r_bot"]
    t = math.sqrt(1.0 + x * x)
    return 0.5 * q["r_bot"] * q["r_bot"] * (x * t + math.log(x + t))


def _parab_r_of_y(q: dict, y: float) -> float:
    if y <= 0.0:
        return 0.0
    return (4. / 3. * q["r_bot"] * y * math.sqrt(y)) / _parab_p_of_y(q, y)


def _power_p_of_y(q: dict, y: float) -> float:
    dy1 = 0.02 * q["y_full"]
    h = (q["s_bot"] + 1.0) * q["r_bot"] / 2.0
    m = q["s_bot"]
    p, y1, x1 = 0.0, 0.0, 0.0
    while True:
        y2 = y1 + dy1
        if y2 > y:
            y2 = y
        x2 = h * _power(y2, m)
        dx, dy = x2 - x1, y2 - y1
        p += math.sqrt(dx * dx + dy * dy)
        x1, y1 = x2, y2
        if not y2 < y:
            break
    return 2.0 * p


def _power_r_of_y(q: dict, y: float) -> float:
    if y <= 0.0:
        return 0.0
    return q["r_bot"] * _power(y, q["s_bot"] + 1.0) / _power_p_of_y(q, y)


# --------------------------------------------------------------------------- transects
@functools.cache
def transect_tables(t: Transect) -> dict:
    """`transect.c::transect_validate` / ``createTables``: the 51-point geometry tables of
    `t` in SWMM's feet, normalised as SWMM stores them. Keys: ``y_full``, ``a_full``,
    ``r_full``, ``w_max`` (ft, ft2, ft, ft), ``s_max`` (ft^(8/3)) and ``a_max`` (ft2), the
    maximum section factor of the unnormalised tables and the area where it occurs, and
    ``area``, ``hrad``, ``width`` (tuples of 51, the areas and hydraulic radii over their
    full values, the widths over the maximum width). ValueError for the data SWMM rejects
    (fewer than two stations, no depth, non-increasing stations, no main-channel n, banks
    out of order)."""
    n_st = len(t.stations)
    if n_st < 2:
        raise ValueError(
            f"transect {t.name!r}: fewer than two stations (SWMM ERR_TRANSECT_TOO_FEW)")
    if n_st >= 1500:
        raise ValueError(f"transect {t.name!r}: 1500 or more stations (SWMM MAXSTATION)")
    if not t.n_channel > 0.0:
        raise ValueError(f"transect {t.name!r}: no main-channel Manning n")
    if t.left_bank > t.right_bank:
        raise ValueError(f"transect {t.name!r}: the left bank is right of the right bank")
    station = [0.0] + [s / LCF for s in t.stations]
    elev = [0.0] + [e / LCF for e in t.elevations]
    if any(station[k] < station[k - 1] for k in range(2, n_st + 1)):
        raise ValueError(f"transect {t.name!r}: station distances decrease")
    xleft, xright = t.left_bank / LCF, t.right_bank / LCF
    n_left, n_right = t.n_left, t.n_right
    n_chan = t.n_channel * math.sqrt(t.length_factor)               # transect.c:230
    ymax = max(elev[1:])
    ymin = min(elev[1:])
    if ymin >= ymax:
        raise ValueError(f"transect {t.name!r}: no depth (all elevations equal)")
    station[0] = station[1]                                         # transect.c:248-252
    elev[0] = ymax
    n_sta = n_st + 1
    station.append(station[n_sta - 1])
    elev.append(elev[0])
    area = [0.0] * N_TABLE
    hrad = [0.0] * N_TABLE
    width = [0.0] * N_TABLE
    y_full = ymax - ymin
    dy = (ymax - ymin) / (float(N_TABLE) - 1)
    y = ymin
    for i in range(1, N_TABLE):
        y += dy
        _transect_geometry(i, y, station, elev, n_sta, xleft, xright, n_left, n_right, n_chan,
                           area, hrad, width)
    # transect.c::setMaxSectionFactor
    s_max = a_max = 0.0
    for i in range(1, N_TABLE):
        sf = area[i] * _power(hrad[i], 2. / 3.)
        if sf > s_max:
            s_max, a_max = sf, area[i]
    last = N_TABLE - 1
    a_full, r_full, w_max = area[last], hrad[last], width[last]
    for i in range(1, last + 1):
        area[i] /= a_full
        hrad[i] /= r_full
        width[i] /= w_max
    width[0] = width[1]
    return {"y_full": y_full, "a_full": a_full, "r_full": r_full, "w_max": w_max,
            "s_max": s_max, "a_max": a_max, "area": tuple(area), "hrad": tuple(hrad),
            "width": tuple(width)}


def _transect_geometry(i, y, station, elev, n_sta, xleft, xright, n_left, n_right, n_chan,
                       area, hrad, width) -> None:
    """`transect.c::getGeometry` at table entry `i`, depth (elevation) `y`."""
    wp_sum = a_sum = q_sum = 0.0
    for k in range(1, n_sta + 1):
        if elev[k - 1] >= elev[k]:
            yhi, ylo = elev[k - 1], elev[k]
        else:
            yhi, ylo = elev[k], elev[k - 1]
        if ylo >= y:
            continue
        w, a, wp = _slice_geometry(station, k, y, ylo, yhi)
        wp_sum += wp
        a_sum += a
        area[i] += a
        width[i] += w
        find_flow = elev[k] >= y
        q = _transect_flow(station, k, n_sta, xleft, xright, n_left, n_right, n_chan, a_sum,
                           wp_sum, find_flow)
        if q > 0.0:
            q_sum += q
            a_sum = 0.0
            wp_sum = 0.0
    a_sum = area[i]
    if a_sum == 0.0:
        hrad[i] = hrad[i - 1]
    else:
        hrad[i] = _power(q_sum * n_chan / 1.49 / a_sum, 1.5)


def _slice_geometry(station, k, y, ylo, yhi) -> tuple[float, float, float]:
    """`transect.c::getSliceGeom`: ``(w, a, wp)``."""
    wdt = abs(station[k] - station[k - 1])
    w = wdt
    wp = math.sqrt(wdt * wdt + (yhi - ylo) * (yhi - ylo))
    a = 0.0
    if y > yhi:
        a = wdt * ((y - yhi) + (y - ylo)) / 2.0
    elif yhi > ylo:
        ratio = (y - ylo) / (yhi - ylo)
        a = wdt * (yhi - ylo) / 2.0 * ratio * ratio
        w *= ratio
        wp *= ratio
    return w, a, wp


def _transect_flow(station, k, n_sta, xleft, xright, n_left, n_right, n_chan, a, wp,
                   find_flow) -> float:
    """`transect.c::getFlow`: the normal flow (per unit of slope) of a sub-section."""
    if not find_flow:
        if k == n_sta - 1:
            find_flow = True
        elif station[k] == xleft:
            if n_left != n_chan and station[k] != station[k - 1]:
                find_flow = True
        elif station[k] == xright:
            nxt = station[k + 1] if k + 1 < len(station) else 0.0
            if n_right != n_chan and station[k] != nxt:
                find_flow = True
    if find_flow:
        n = n_chan
        if station[k - 1] < xleft:
            n = n_left
        if station[k] > xright:
            n = n_right
        if wp == 0.0:             # C's a / 0 is inf or NaN: no flow is added (``q > 0`` fails)
            return math.nan
        return 1.486 / n * a * _power(a / wp, 2. / 3.)
    return 0.0


# ------------------------------------------------------------------------ custom shapes
@functools.cache
def custom_tables(curve: tuple[tuple[float, float], ...]) -> dict:
    """`shape.c::shape_validate`: the 51-point geometry tables of a CUSTOM shape of unit
    height from its width-against-height `curve` (``(height, width)`` points, both over the
    full height). Keys: ``a_full``, ``r_full``, ``w_max``, ``s_max``, ``a_max`` (per unit
    height: ft2, ft, ft, ft^(8/3), ft2) and ``area``, ``hrad``, ``width`` (tuples of 51,
    normalised). ValueError where SWMM rejects the curve."""
    pts = list(curve)
    pos = 0

    def first():
        nonlocal pos
        pos = 1
        return pts[0]

    def nxt():
        nonlocal pos
        if pos >= len(pts):
            return None
        pos += 1
        return pts[pos - 1]

    bad = ValueError("SHAPE curve rejected (shape.c::shape_validate): heights must lie in "
                     "[0, 1) at the start, increase, and widths be non-negative")
    if not pts:
        raise bad
    y1, w1 = first()
    if y1 < 0.0 or y1 >= 1.0 or w1 < 0.0:
        raise bad
    w_max = w1
    if y1 != 0.0:
        y2, w2 = y1, w1
        y1, w1 = 0.0, 0.0
    else:
        e = nxt()
        if e is None:
            raise bad
        y2, w2 = e
        if y2 < y1 or w2 < 0.0:
            raise bad
        if y2 > 1.0:
            y2 = 1.0
        if w2 > w_max:
            w_max = w2
    n = N_TABLE - 1
    dy = 1.0 / n
    area = [0.0] * N_TABLE
    hrad = [0.0] * N_TABLE
    width = [0.0] * N_TABLE
    width[0] = w1
    p_total = w1
    a_total = 0.0
    y = 0.0
    w = w1

    def seg_area(yy, ww, yb, wb):
        lo, hi = (wb, ww) if ww > wb else (ww, wb)
        return (lo + (hi - lo) / 2.0) * (yy - yb)

    def seg_perim(yy, ww, yb, wb):
        d = yy - yb
        dw = abs(ww - wb) / 2.0
        return 2.0 * math.sqrt(d * d + dw * dw)

    for i in range(1, n + 1):
        y_last, w_last = y, w
        y = y + dy
        if abs(y - 1.0) < 1.0e-6:
            y = 1.0
        if y > y2:
            while y > y2:                                           # getNextInterval
                if y2 > y_last:
                    a_total += seg_area(y2, w2, y_last, w_last)
                    p_total += seg_perim(y2, w2, y_last, w_last)
                    y_last, w_last = y2, w2
                y1, w1 = y2, w2
                e = nxt()
                if e is None:
                    y2 = 1.0
                    break
                y2, w2 = e
                if w2 > w_max:
                    w_max = w2
                if y2 < y1 or w2 < 0.0:
                    raise bad
                if y2 > 1.0:
                    y2 = 1.0
            y_last, w_last = y1, w1
        w = w2 if y2 == y1 else w1 + (y - y1) / (y2 - y1) * (w2 - w1)
        a_total += seg_area(y, w, y_last, w_last)
        p_total += seg_perim(y, w, y_last, w_last)
        if y == 1.0:
            p_total += w2
        width[i] = w
        area[i] = a_total
        hrad[i] = a_total / p_total if p_total > 0.0 else 0.0
    a_full, r_full = area[n], hrad[n]
    s_max = a_max = 0.0
    for i in range(1, n + 1):
        sf = area[i] * _power(hrad[i], 2. / 3.)
        if sf > s_max:
            s_max, a_max = sf, area[i]
    if a_full == 0.0 or r_full == 0.0 or w_max == 0.0:
        raise bad
    for i in range(n + 1):
        area[i] /= a_full
        hrad[i] /= r_full
        width[i] /= w_max
    return {"a_full": a_full, "r_full": r_full, "w_max": w_max, "s_max": s_max,
            "a_max": a_max, "area": tuple(area), "hrad": tuple(hrad), "width": tuple(width)}


__all__ = [
    "AMAX", "KINDS", "LENGTH_FIELDS", "N_TABLE", "UNMODELLED_KINDS", "Transect", "Xsect",
    "custom_tables", "is_open", "transect_tables",
]
