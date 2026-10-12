"""SWMM 5.2.4's cross-section geometry for every shape but the circle's own path, in torch,
operation for operation: a batched `XsectSet` over a network's conduits.

`xsect_tables.py` transcribes the CIRCULAR section (`xsect.c`'s tables and its closed forms
near the invert). SWMM evaluates every other section with the same machinery -- the shape
tables of `xsect.dat` (`xsect_dat.py`) through ``lookup``/``invLookup``, closed forms for the
rectangular, trapezoidal, triangular, parabolic and power-law sections, per-conduit 51-point
tables for a transect or a CUSTOM shape, and, wherever there is no closed form for the area
of a given section factor, ``generic_getAofS``: a bracketed Newton-Raphson iteration on the
shape's own ``S(A)`` and ``dS/dA`` (`findroot.c`) -- and this module reproduces each of those
functions with SWMM's stopping rules and starting guesses, because the dynamic wave reads them
through the flow class (normal and critical depth), the surface areas (top width) and the
momentum update (area, hydraulic radius, Froude number).

An `XsectSet` holds one section per conduit as per-shape groups; every query is a few
vectorised calls per group present (not per conduit), over the last dimension. It is built
from the per-conduit constants of `xsect_spec.Xsect.params` (ft) and, for a circular conduit
in a mixed network, from `xsect_tables.conduit_constants`; a circular-only network does not
build one (its path is `xsect_tables` itself, bit for bit).

The queries, y in ft, areas in ft2, flows in cfs: `area`, `width`, `radius` (`xsect_getAofY`,
`xsect_getWofY`, `xsect_getRofY`, with the products the dynamic wave takes in one table
search: `area_width`, `area_radius`), `froude` (`link.c::link_getFroude`), `normal_depth`
(`link.c::link_getYnorm`: ``Y(A(S))``), `critical_depth` (`xsect.c::xsect_getYcrit`: a closed
form for the rectangular, triangular, parabolic and power sections, the 25-interval
enumeration `getYcritEnum` where the section's area is within a factor of 2 of the equivalent
circle's, Ridder's root finder `getYcritRidder` otherwise).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from noodl.apps.sewer import xsect_dat as dat
from noodl.apps.sewer import xsect_tables as xt
from noodl.apps.sewer.xsect_spec import AMAX, is_open

Tensor = torch.Tensor
F64 = torch.float64
_PI = 3.141592654
_TINY = 1.0e-6               # consts.h TINY
_G = xt.GRAVITY_FT
_FUDGE = xt._FUDGE
_THIRD, _TWO_THIRDS = 1.0 / 3.0, 2.0 / 3.0

_TABLES: dict[str, Tensor] = {}


def _table(name: str) -> Tensor:
    """The `xsect_dat` table `name` as a (cached) tensor; the circle's are `xsect_tables`'."""
    if name not in _TABLES:
        _TABLES[name] = getattr(xt, "_" + name[0]) if name.endswith("_CIRC") else \
            torch.tensor(getattr(dat, name), dtype=F64)
    return _TABLES[name]


def _lu(x: Tensor, name: str) -> Tensor:
    return xt.lookup(x, _table(name))


def _ilu(x: Tensor, name: str) -> Tensor:
    return xt.inv_lookup(x, _table(name))


# ----------------------------------------------------------------- per-conduit tables
def lookup_rows(x: Tensor, tab: Tensor) -> Tensor:
    """`xsect.c::lookup` with one table per element: `x` (..., G), `tab` (G, n). The same
    arithmetic as `xsect_tables.lookup`, the segment's three entries gathered per element."""
    n = tab.shape[-1]
    delta = 1.0 / (float(n) - 1.0)
    i = torch.floor(x / delta)
    top = i >= n - 1
    ic = torch.clamp(i, 0, n - 2).to(torch.long)
    g = torch.arange(tab.shape[0])
    t0, t1 = tab[g, ic], tab[g, ic + 1]
    t2 = tab[g, torch.clamp(ic + 2, max=n - 1)]
    x0 = ic.to(F64) * delta
    x1 = (ic.to(F64) + 1.0) * delta
    y = t0 + (x - x0) * (t1 - t0) / delta
    y2 = y + (x - x0) * (x - x1) / (delta * delta) * (t0 / 2.0 - t1 + t2 / 2.0)
    y = torch.where((ic < 2) & (y2 > 0.0), y2, y)
    y = torch.clamp(y, min=0.0)
    return torch.where(top, tab[:, n - 1], y)


def inv_lookup_rows(y: Tensor, tab: Tensor) -> Tensor:
    """`xsect.c::invLookup` with one INCREASING table per element (a transect's or a CUSTOM
    shape's area table): `locate`'s bisection exactly, then the linear inverse."""
    n = tab.shape[-1]
    dx = 1.0 / (float(n) - 1.0)
    g = torch.arange(tab.shape[0])
    j1 = torch.zeros(y.shape, dtype=torch.long)
    j2 = torch.full(y.shape, n - 1, dtype=torch.long)
    for _ in range(int(math.ceil(math.log2(n))) + 1):
        active = (j2 - j1) > 1
        j = (j1 + j2) >> 1
        ge = y >= tab[g, j]
        j1 = torch.where(active & ge, j, j1)
        j2 = torch.where(active & ~ge, j, j2)
    i = torch.where(y <= tab[:, 0], torch.zeros_like(j1),
                    torch.where(y >= tab[:, n - 1], torch.full_like(j1, n - 1), j1))
    top = i >= n - 1
    i = torch.clamp(i, max=n - 2)
    t0, t1 = tab[g, i], tab[g, i + 1]
    dy = t1 - t0
    x0 = i.to(F64) * dx
    dy_safe = torch.where(dy == 0.0, torch.ones_like(dy), dy)
    x = torch.where(dy == 0.0, x0, x0 + (y - t0) * dx / dy_safe)
    x = torch.clamp(x, 0.0, 1.0)
    return torch.where(top, torch.full_like(x, (n - 1) * dx), x)


# ----------------------------------------------------------------------- root finders
def newton_bracketed(func: Callable[[Tensor], tuple[Tensor, Tensor]], x1: Tensor, x2: Tensor,
                     x0: Tensor, xacc: Tensor, max_iter: int = 60) -> Tensor:
    """`findroot.c::findroot_Newton`: Newton-Raphson safeguarded by bisection on the bracket
    ``[x1, x2]`` from the starting guess `x0`, up to ``MAXIT`` = 60 iterations, per element
    (the element's own stopping rule: `xacc` on the step, or a step that does not move
    ``x``). `func` returns ``(f, df)``."""
    x = x0
    xlo, xhi = x1, x2
    dxold = torch.abs(x2 - x1)
    dx = dxold
    f, df = func(x)
    done = torch.zeros(x.shape, dtype=torch.bool)
    for _ in range(max_iter):
        bisect = ((((x - xhi) * df - f) * ((x - xlo) * df - f) >= 0.0)
                  | (torch.abs(2.0 * f) > torch.abs(dxold * df)))
        dx_b = 0.5 * (xhi - xlo)
        x_b = xlo + dx_b
        df_safe = torch.where(df == 0.0, torch.ones_like(df), df)
        dx_n = f / df_safe
        x_n = x - dx_n
        stuck = torch.where(bisect, xlo == x_b, x == x_n)
        new_dx = torch.where(bisect, dx_b, dx_n)
        new_x = torch.where(bisect, x_b, x_n)
        dxold = torch.where(done, dxold, dx)
        dx = torch.where(done, dx, new_dx)
        x = torch.where(done, x, new_x)
        done = done | stuck | (torch.abs(new_dx) < xacc)
        if bool(done.all()):
            break
        f_new, df_new = func(x)
        f = torch.where(done, f, f_new)
        df = torch.where(done, df, df_new)
        low = f < 0.0
        xlo = torch.where(~done & low, x, xlo)
        xhi = torch.where(~done & ~low, x, xhi)
    return x


def ridder(func: Callable[[Tensor], Tensor], x1: Tensor, x2: Tensor, xacc: float,
           max_iter: int = 60, *, track: bool = False):
    """`findroot.c::findroot_Ridder`: Ridder's method on ``[x1, x2]`` (`xacc` the interval
    tolerance), per element; ``-1e20`` where the interval does not bracket a root, as SWMM
    returns. With `track`, ``(root, last)`` where `last` is the abscissa of the last
    evaluation of `func` (not always the root: the culvert flow of `culvert.py` is a side
    effect of that evaluation, as in SWMM's ``form1Eqn``)."""
    flo, fhi = func(x1), func(x2)
    out = 0.5 * (x1 + x2)
    bracket = ((flo > 0.0) & (fhi < 0.0)) | ((flo < 0.0) & (fhi > 0.0))
    result = torch.where(flo == 0.0, x1, torch.where(fhi == 0.0, x2, torch.full_like(x1, -1.0e20)))
    ready = (flo == 0.0) | (fhi == 0.0)
    live = bracket & ~ready
    result = torch.where(live, out, result)
    xlo, xhi, ans = x1, x2, out
    live = live.clone()
    last = x2
    for _ in range(max_iter):
        if not bool(live.any()):
            break
        xm = 0.5 * (xlo + xhi)
        last = torch.where(live, xm, last)
        fm = func(xm)
        s = torch.sqrt(torch.clamp(fm * fm - flo * fhi, min=0.0))
        zero = s == 0.0
        s_safe = torch.where(zero, torch.ones_like(s), s)
        xnew = xm + (xm - xlo) * (torch.where(flo >= fhi, 1.0, -1.0) * fm / s_safe)
        stop = zero | (torch.abs(xnew - ans) <= xacc)
        result = torch.where(live & stop, ans, result)
        live = live & ~stop
        ans_new = torch.where(live, xnew, ans)
        last = torch.where(live, ans_new, last)
        fnew = func(ans_new)
        sgn = lambda a, b: torch.where(b >= 0.0, torch.abs(a), -torch.abs(a))   # noqa: E731
        c1 = sgn(fm, fnew) != fm
        c2 = ~c1 & (sgn(flo, fnew) != flo)
        c3 = ~c1 & ~c2 & (sgn(fhi, fnew) != fhi)
        c4 = ~c1 & ~c2 & ~c3
        result = torch.where(live & c4, ans_new, result)
        live = live & ~c4
        nxlo = torch.where(c1, xm, torch.where(c3, ans_new, xlo))
        nflo = torch.where(c1, fm, torch.where(c3, fnew, flo))
        nxhi = torch.where(c1 | c2, ans_new, xhi)
        nfhi = torch.where(c1 | c2, fnew, fhi)
        xlo = torch.where(live, nxlo, xlo)
        flo = torch.where(live, nflo, flo)
        xhi = torch.where(live, nxhi, xhi)
        fhi = torch.where(live, nfhi, fhi)
        ans = ans_new
        narrow = live & (torch.abs(xhi - xlo) <= xacc)
        result = torch.where(narrow, ans, result)
        live = live & ~narrow
    result = torch.where(live, ans, result)
    return (result, last) if track else result


# -------------------------------------------------------------------------- shapes
class Shape:
    """One SWMM cross-section type's functions on a group of conduits (a `_Group`'s
    parameter dict `p`, SWMM's ``TXsect`` fields in ft, and, for per-conduit tables, `t`).
    A subclass overrides what SWMM's `xsect.c` does for its type; the defaults are SWMM's
    own fall-throughs (``xsect_getRofA``/``xsect_getSofA``/``generic_getdSdA``/
    ``generic_getAofS``). Depths are > 0 where a function needs it (the `XsectSet` masks)."""

    name = ""
    #: whether ``xsect_getAofS`` is the generic Newton iteration (else `a_of_s` is overridden)
    generic_a_of_s = True

    # -- the five primary functions
    @staticmethod
    def area(p, t, y):
        raise NotImplementedError

    @staticmethod
    def width(p, t, y):
        raise NotImplementedError

    @classmethod
    def radius(cls, p, t, y):                     # xsect_getRofY default
        return cls.r_of_a(p, t, cls.area(p, t, y))

    @staticmethod
    def y_of_a(p, t, a):
        raise NotImplementedError

    @classmethod
    def r_of_a(cls, p, t, a):                     # xsect_getRofA default
        cathy = cls.s_of_a(p, t, a)
        ok = (cathy >= _TINY) & (a >= _TINY)
        ratio = torch.where(ok, cathy / torch.where(ok, a, torch.ones_like(a)),
                            torch.ones_like(a))
        return torch.where(ok, torch.pow(ratio, 1.5), torch.zeros_like(a))

    @classmethod
    def s_of_a(cls, p, t, a):                     # xsect_getSofA default
        r = cls.r_of_a(p, t, a)
        small = r < _TINY
        return torch.where((a == 0.0) | small, torch.zeros_like(a),
                           a * torch.pow(torch.where(small, torch.ones_like(r), r), _TWO_THIRDS))

    @classmethod
    def ds_da(cls, p, t, a):                      # generic_getdSdA (central difference)
        alpha = a / p["a_full"]
        a1 = torch.clamp(alpha - 0.001, min=0.0) * p["a_full"]
        a2 = (alpha + 0.001) * p["a_full"]
        return (cls.s_of_a(p, t, a2) - cls.s_of_a(p, t, a1)) / (a2 - a1)

    @classmethod
    def a_max(cls, p):                            # xsect_getAmax
        return AMAX[cls.name] * p["a_full"]

    @classmethod
    def a_of_s(cls, p, t, s):
        """`xsect_getAofS` for ``s > 0`` already limited to ``sMax``: `generic_getAofS`."""
        s_full, s_max, a_full = p["s_full"], p["s_max"], p["a_full"]
        between = (s <= s_max) & (s >= s_full) & (s_max != s_full)
        a1 = torch.where(between, a_full, torch.zeros_like(a_full))
        a2 = cls.a_max(p).expand_as(a1)

        def func(a):
            return cls.s_of_a(p, t, a) - s, cls.ds_da(p, t, a)

        a = newton_bracketed(func, a1.expand(s.shape), a2.expand(s.shape),
                             (0.5 * (a1 + a2)).expand(s.shape), (1.0e-4 * a_full).expand(s.shape))
        return torch.where(s > 0.0, a, torch.zeros_like(a))

    # -- critical depth: the default enumeration / Ridder path unless a subclass has a form
    closed_form_ycrit: Callable | None = None


def _safe(x: Tensor, bad: Tensor, fill: float = 1.0) -> Tensor:
    return torch.where(bad, torch.full_like(x, fill), x)


# ---- circular family ----------------------------------------------------------------
class Circular(Shape):
    """CIRCULAR and FORCE_MAIN: the tables of `xsect_tables` (its path, bit for bit)."""

    name = "CIRCULAR"
    generic_a_of_s = False

    @staticmethod
    def area(p, t, y):
        return xt.area_of_depth(y, p["y_full"], p["a_full"])

    @staticmethod
    def width(p, t, y):
        return xt.width_of_depth(y, p["y_full"])

    @staticmethod
    def radius(p, t, y):
        return xt.radius_of_depth(y, p["y_full"])

    @staticmethod
    def y_of_a(p, t, a):
        return p["y_full"] * xt.y_of_a(a / p["a_full"])

    @staticmethod
    def a_of_s(p, t, s):
        psi = s / p["s_full"]
        alpha = torch.where(psi >= 1.0, torch.ones_like(psi), xt.a_of_s(torch.clamp(psi, max=1.0)))
        return p["a_full"] * alpha


class ForceMain(Circular):
    name = "FORCE_MAIN"


class FilledCircular(Shape):
    """FILLED_CIRCULAR: the circle with its bottom ``yBot`` filled (`filled_circ_*`): the
    functions of the full circle, shifted by the fill. ``y_full``, ``a_full`` are the
    reduced values; ``y_bot``, ``a_bot`` the filled depth and area, ``s_bot`` its width,
    ``r_bot`` its wetted perimeter."""

    name = "FILLED_CIRCULAR"

    @staticmethod
    def area(p, t, y):
        yf, af = p["y_full"] + p["y_bot"], p["a_full"] + p["a_bot"]
        return af * xt.lookup((y + p["y_bot"]) / yf, xt._A) - p["a_bot"]

    @staticmethod
    def width(p, t, y):
        return p["w_max"] * xt.lookup((y + p["y_bot"]) / (p["y_full"] + p["y_bot"]), xt._W)

    @staticmethod
    def radius(p, t, y):
        yb = p["y_bot"]
        yf, af = p["y_full"] + yb, p["a_full"] + p["a_bot"]
        plain = p["r_full"] * xt.lookup(y / p["y_full"], xt._R)
        yy = (y + yb) / yf
        a = af * xt.lookup(yy, xt._A)
        r = 0.25 * yf * xt.lookup(yy, xt._R)
        pw = a / r
        a = a - p["a_bot"]
        pw = pw - p["r_bot"] + p["s_bot"]
        return torch.where(yb == 0.0, plain, a / pw)

    @staticmethod
    def y_of_a(p, t, a):
        yf, af = p["y_full"] + p["y_bot"], p["a_full"] + p["a_bot"]
        return yf * xt.y_of_a((a + p["a_bot"]) / af) - p["y_bot"]

    @classmethod
    def r_of_a(cls, p, t, a):
        pos = a > 0.0
        y = cls.y_of_a(p, t, torch.where(pos, a, p["a_full"]))
        return torch.where(pos, cls.radius(p, t, torch.clamp(y, min=0.0)), torch.zeros_like(a))


# ---- tabulated shapes ---------------------------------------------------------------
class Tabular(Shape):
    """A `xsect.dat` shape (EGG, HORSESHOE, GOTHIC, CATENARY, SEMIELLIPTICAL, BASKETHANDLE,
    SEMICIRCULAR, HORIZ_ELLIPSE, VERT_ELLIPSE, ARCH): per type, which of its tables SWMM
    reads through ``lookup`` and which through ``invLookup``."""

    #: table names (``A`` area v. depth, ``R`` hydraulic radius v. depth, ``Y`` depth v.
    #: area, ``S`` section factor v. area, ``W`` width v. depth); None where SWMM has none
    tabs: dict[str, str | None] = {}

    @classmethod
    def area(cls, p, t, y):
        yn = y / p["y_full"]
        if cls.tabs["A"]:
            return p["a_full"] * _lu(yn, cls.tabs["A"])
        return p["a_full"] * _ilu(yn, cls.tabs["Y"])

    @classmethod
    def width(cls, p, t, y):
        return p["w_max"] * _lu(y / p["y_full"], cls.tabs["W"])

    @classmethod
    def radius(cls, p, t, y):
        if cls.tabs["R"]:
            return p["r_full"] * _lu(y / p["y_full"], cls.tabs["R"])
        return cls.r_of_a(p, t, cls.area(p, t, y))

    @classmethod
    def y_of_a(cls, p, t, a):
        alpha = a / p["a_full"]
        if cls.tabs["Y"]:
            return p["y_full"] * _lu(alpha, cls.tabs["Y"])
        return p["y_full"] * _ilu(alpha, cls.tabs["A"])

    @classmethod
    def r_of_a(cls, p, t, a):
        if cls.tabs["S"]:
            return super().r_of_a.__func__(cls, p, t, a)
        pos = a > 0.0                              # xsect_getRofA: RofY(YofA(a))
        y = cls.y_of_a(p, t, torch.where(pos, a, p["a_full"]))
        return torch.where(pos, cls.radius(p, t, y), torch.zeros_like(a))

    @classmethod
    def s_of_a(cls, p, t, a):
        if cls.tabs["S"]:
            return p["s_full"] * _lu(a / p["a_full"], cls.tabs["S"])
        return super().s_of_a.__func__(cls, p, t, a)

    @classmethod
    def ds_da(cls, p, t, a):
        if not cls.tabs["S"]:
            return super().ds_da.__func__(cls, p, t, a)
        table = _table(cls.tabs["S"])               # tabular_getdSdA
        n = table.shape[0]
        delta = 1.0 / (float(n) - 1.0)
        i = torch.clamp(torch.floor((a / p["a_full"]) / delta), max=n - 2).to(torch.long)
        return (table[i + 1] - table[i]) / delta * p["s_full"] / p["a_full"]

    @classmethod
    def a_of_s(cls, p, t, s):
        if not cls.tabs["S"]:
            return super().a_of_s.__func__(cls, p, t, s)
        return p["a_full"] * _ilu(s / p["s_full"], cls.tabs["S"])

    @classmethod
    def generic(cls):
        return not cls.tabs["S"]


def _tabular(name: str, **tabs) -> type[Tabular]:
    full = {"A": None, "R": None, "Y": None, "S": None, "W": None}
    full.update(tabs)
    return type(name.title(), (Tabular,), {"name": name, "tabs": full,
                                           "generic_a_of_s": not full["S"]})


Egg = _tabular("EGG", A="A_EGG", R="R_EGG", Y="Y_EGG", S="S_EGG", W="W_EGG")
Horseshoe = _tabular("HORSESHOE", A="A_HORSESHOE", R="R_HORSESHOE", Y="Y_HORSESHOE",
                     S="S_HORSESHOE", W="W_HORSESHOE")
Gothic = _tabular("GOTHIC", Y="Y_GOTHIC", S="S_GOTHIC", W="W_GOTHIC")
Catenary = _tabular("CATENARY", Y="Y_CATENARY", S="S_CATENARY", W="W_CATENARY")
SemiElliptical = _tabular("SEMIELLIPTICAL", Y="Y_SEMIELLIP", S="S_SEMIELLIP", W="W_SEMIELLIP")
BasketHandle = _tabular("BASKETHANDLE", A="A_BASKETHANDLE", R="R_BASKETHANDLE",
                        Y="Y_BASKETHANDLE", S="S_BASKETHANDLE", W="W_BASKETHANDLE")
SemiCircular = _tabular("SEMICIRCULAR", Y="Y_SEMICIRC", S="S_SEMICIRC", W="W_SEMICIRC")
HorizEllipse = _tabular("HORIZ_ELLIPSE", A="A_HORIZELLIPSE", R="R_HORIZELLIPSE",
                        W="W_HORIZELLIPSE")
VertEllipse = _tabular("VERT_ELLIPSE", A="A_VERTELLIPSE", R="R_VERTELLIPSE", W="W_VERTELLIPSE")
Arch = _tabular("ARCH", A="A_ARCH", R="R_ARCH", W="W_ARCH")


class Rows(Shape):
    """IRREGULAR (a transect) and CUSTOM (a shape curve): 51-point tables per conduit,
    ``t["area"]``, ``t["hrad"]``, ``t["width"]`` (G, 51)."""

    name = "IRREGULAR"

    @staticmethod
    def area(p, t, y):
        return p["a_full"] * lookup_rows(y / p["y_full"], t["area"])

    @staticmethod
    def width(p, t, y):
        return p["w_max"] * lookup_rows(y / p["y_full"], t["width"])

    @staticmethod
    def radius(p, t, y):
        return p["r_full"] * lookup_rows(y / p["y_full"], t["hrad"])

    @staticmethod
    def y_of_a(p, t, a):
        return p["y_full"] * inv_lookup_rows(a / p["a_full"], t["area"])

    @classmethod
    def r_of_a(cls, p, t, a):
        pos = a > 0.0
        y = cls.y_of_a(p, t, torch.where(pos, a, p["a_full"]))
        return torch.where(pos, cls.radius(p, t, y), torch.zeros_like(a))

    @staticmethod
    def a_max(p):
        return p["a_bot"]


class Custom(Rows):
    name = "CUSTOM"


# ---- closed-form shapes -------------------------------------------------------------
_ALF = {"RECT_CLOSED": 0.97, "RECT_TRIANGULAR": 0.98, "RECT_ROUND": 0.98}


def _rect_s_limit(p, a, alf):
    """The ``S(A)`` above the ``alfMax`` fraction: the line from ``sMax`` to ``sFull``."""
    return p["s_max"] + (p["s_full"] - p["s_max"]) * (a / p["a_full"] - alf) / (1.0 - alf)


class RectClosed(Shape):
    name = "RECT_CLOSED"

    @staticmethod
    def area(p, t, y):
        return y * p["w_max"]

    @staticmethod
    def width(p, t, y):
        return torch.where(y / p["y_full"] == 1.0, torch.zeros_like(y), p["w_max"] + 0 * y)

    @staticmethod
    def y_of_a(p, t, a):
        return a / p["w_max"]

    @staticmethod
    def r_of_a(p, t, a):                          # rect_closed_getRofA
        pw = p["w_max"] + 2. * a / p["w_max"]
        over = a / p["a_full"] > 0.97
        pw = torch.where(over, pw + (a / p["a_full"] - 0.97) / (1.0 - 0.97) * p["w_max"], pw)
        pos = a > 0.0
        return torch.where(pos, a / torch.where(pos, pw, torch.ones_like(pw)), torch.zeros_like(a))

    @classmethod
    def s_of_a(cls, p, t, a):
        over = a / p["a_full"] > 0.97
        return torch.where(over, _rect_s_limit(p, a, 0.97),
                           a * torch.pow(cls.r_of_a(p, t, a), _TWO_THIRDS))

    @classmethod
    def ds_da(cls, p, t, a):
        alpha = a / p["a_full"]
        r = cls.r_of_a(p, t, a)
        analytic = (5. / 3. - (2. / 3.) * (2.0 / p["w_max"]) * r) * torch.pow(r, _TWO_THIRDS)
        generic = Shape.ds_da.__func__(cls, p, t, a)
        return torch.where(alpha > 0.97, (p["s_full"] - p["s_max"]) / ((1.0 - 0.97) * p["a_full"]),
                           torch.where(alpha <= 1.0e-30, generic, analytic))


class RectOpen(Shape):
    name = "RECT_OPEN"

    @staticmethod
    def area(p, t, y):
        return y * p["w_max"]

    @staticmethod
    def width(p, t, y):
        return p["w_max"] + 0 * y

    @staticmethod
    def y_of_a(p, t, a):
        return a / p["w_max"]

    @staticmethod
    def r_of_a(p, t, a):
        pos = a > 0.0
        den = p["w_max"] + (2. - p["s_bot"]) * a / p["w_max"]
        return torch.where(pos, a / den, torch.zeros_like(a))

    @staticmethod
    def s_of_a(p, t, a):                          # rect_open_getSofA
        y = a / p["w_max"]
        r = a / ((2.0 - p["s_bot"]) * y + p["w_max"])
        return a * torch.pow(r, _TWO_THIRDS)

    @classmethod
    def ds_da(cls, p, t, a):
        r = cls.r_of_a(p, t, a)
        dpda = (2.0 - p["s_bot"]) / p["w_max"]
        analytic = (5. / 3. - (2. / 3.) * dpda * r) * torch.pow(r, _TWO_THIRDS)
        return torch.where(a / p["a_full"] <= 1.0e-30, Shape.ds_da.__func__(cls, p, t, a), analytic)


class RectTriangular(Shape):
    name = "RECT_TRIANGULAR"

    @staticmethod
    def area(p, t, y):
        return torch.where(y <= p["y_bot"], y * y * p["s_bot"],
                           p["a_bot"] + (y - p["y_bot"]) * p["w_max"])

    @staticmethod
    def width(p, t, y):
        return torch.where(y <= p["y_bot"], 2.0 * p["s_bot"] * y, p["w_max"] + 0 * y)

    @staticmethod
    def y_of_a(p, t, a):
        low = a <= p["a_bot"]
        return torch.where(low, torch.sqrt(torch.where(low, a, p["a_bot"]) / p["s_bot"]),
                           p["y_bot"] + (a - p["a_bot"]) / p["w_max"])

    @classmethod
    def r_of_a(cls, p, t, a):
        y = cls.y_of_a(p, t, a)
        low = y <= p["y_bot"]
        pw = 2. * p["y_bot"] * p["r_bot"] + 2. * (y - p["y_bot"])
        alf = (a / p["a_full"]) - 0.98
        pw = torch.where(alf > 0.0, pw + alf / (1.0 - 0.98) * p["w_max"], pw)
        pos = a > 0.0
        y_safe = torch.where(low, torch.where(y > 0.0, y, torch.ones_like(y)), torch.ones_like(y))
        val = torch.where(low, a / (2. * y_safe * p["r_bot"]), a / pw)
        return torch.where(pos, val, torch.zeros_like(a))

    @classmethod
    def radius(cls, p, t, y):
        a = p["a_bot"] + (y - p["y_bot"]) * p["w_max"]
        pw = 2. * p["y_bot"] * p["r_bot"] + 2. * (y - p["y_bot"])
        alf = (a / p["a_full"]) - 0.98
        pw = torch.where(alf > 0.0, pw + alf / (1.0 - 0.98) * p["w_max"], pw)
        return torch.where(y <= p["y_bot"], y * p["s_bot"] / (2. * p["r_bot"]), a / pw)

    @classmethod
    def s_of_a(cls, p, t, a):
        return torch.where(a / p["a_full"] > 0.98, _rect_s_limit(p, a, 0.98),
                           a * torch.pow(cls.r_of_a(p, t, a), _TWO_THIRDS))

    @classmethod
    def ds_da(cls, p, t, a):
        alpha = a / p["a_full"]
        r = cls.r_of_a(p, t, a)
        dpda = torch.where(a > p["a_bot"], 2.0 / p["w_max"],
                           p["r_bot"] / torch.sqrt(torch.where(a > p["a_bot"], torch.ones_like(a),
                                                               a * p["s_bot"])))
        analytic = (5. / 3. - (2. / 3.) * dpda * r) * torch.pow(r, _TWO_THIRDS)
        return torch.where(alpha > 0.98, (p["s_full"] - p["s_max"]) / ((1.0 - 0.98) * p["a_full"]),
                           torch.where(alpha <= 1.0e-30, Shape.ds_da.__func__(cls, p, t, a),
                                       analytic))


def _acos_arg(y, r):
    return torch.clamp(1.0 - y / r, -1.0, 1.0)


class RectRound(Shape):
    name = "RECT_ROUND"

    @staticmethod
    def area(p, t, y):
        theta = 2.0 * torch.acos(_acos_arg(torch.minimum(y, p["y_bot"]), p["r_bot"]))
        circ = 0.5 * p["r_bot"] * p["r_bot"] * (theta - torch.sin(theta))
        return torch.where(y > p["y_bot"], p["a_bot"] + (y - p["y_bot"]) * p["w_max"], circ)

    @staticmethod
    def width(p, t, y):
        circ = 2.0 * torch.sqrt(torch.clamp(y * (2.0 * p["r_bot"] - y), min=0.0))
        return torch.where(y > p["y_bot"], p["w_max"] + 0 * y, circ)

    @staticmethod
    def y_of_a(p, t, a):
        alpha = torch.clamp(a, max=p["a_bot"]) / (_PI * p["r_bot"] * p["r_bot"])
        low = (2.0 * p["r_bot"]) * torch.where(alpha < 0.04, xt.y_circular(alpha),
                                                xt.lookup(alpha, xt._Y))
        return torch.where(a > p["a_bot"], p["y_bot"] + (a - p["a_bot"]) / p["w_max"], low)

    @classmethod
    def r_of_a(cls, p, t, a):
        a_c = torch.clamp(a, max=p["a_bot"])
        y1 = cls.y_of_a(p, t, a_c)
        theta1 = 2.0 * torch.acos(_acos_arg(y1, p["r_bot"]))
        th_safe = torch.where(theta1 > 0.0, theta1, torch.ones_like(theta1))
        inside = a_c / (p["r_bot"] * th_safe)
        y_up = (a - p["a_bot"]) / p["w_max"]
        th_big = 2.0 * torch.asin(torch.clamp(p["w_max"] / 2.0 / p["r_bot"], max=1.0))
        pw = p["r_bot"] * th_big + 2.0 * y_up
        arg = (a / p["a_full"]) - 0.98
        pw = torch.where(arg > 0.0, pw + arg / (1.0 - 0.98) * p["w_max"], pw)
        above = a / pw
        return torch.where(a <= 0.0, torch.zeros_like(a),
                           torch.where(a > p["a_bot"], above, inside))

    @classmethod
    def radius(cls, p, t, y):
        theta1 = 2.0 * torch.acos(_acos_arg(torch.minimum(y, p["y_bot"]), p["r_bot"]))
        th_safe = torch.where(theta1 > 0.0, theta1, torch.ones_like(theta1))
        circ = 0.5 * p["r_bot"] * (1.0 - torch.sin(theta1)) / th_safe     # xsect.c:2062
        up = cls.r_of_a(p, t, cls.area(p, t, y))
        return torch.where(y <= 0.0, torch.zeros_like(y), torch.where(y > p["y_bot"], up, circ))

    @classmethod
    def s_of_a(cls, p, t, a):
        alpha = torch.clamp(a, max=p["a_bot"]) / (_PI * p["r_bot"] * p["r_bot"])
        circ = p["s_bot"] * torch.where(alpha < 0.04, xt.s_circular(alpha), xt.lookup(alpha, xt._S))
        up = a * torch.pow(cls.r_of_a(p, t, a), _TWO_THIRDS)
        return torch.where(a / p["a_full"] > 0.98, _rect_s_limit(p, a, 0.98),
                           torch.where(a > p["a_bot"], up, circ))

    @classmethod
    def ds_da(cls, p, t, a):
        r = cls.r_of_a(p, t, a)
        analytic = (5. / 3. - (2. / 3.) * (2.0 / p["w_max"]) * r) * torch.pow(r, _TWO_THIRDS)
        generic = Shape.ds_da.__func__(cls, p, t, a)
        return torch.where(a / p["a_full"] > 0.98,
                           (p["s_full"] - p["s_max"]) / ((1.0 - 0.98) * p["a_full"]),
                           torch.where(a > p["a_bot"], analytic, generic))


class ModBasketHandle(Shape):
    """MODBASKETHANDLE: a rectangle under a circular arc; ``r_bot``, ``y_bot``, ``a_bot``,
    ``s_bot`` are the ARC's radius, height, area and angle."""

    name = "MODBASKETHANDLE"

    @staticmethod
    def area(p, t, y):
        y1 = p["y_full"] - y
        theta1 = 2.0 * torch.acos(_acos_arg(torch.clamp(y1, min=0.0), p["r_bot"]))
        a1 = 0.5 * p["r_bot"] * p["r_bot"] * (theta1 - torch.sin(theta1))
        return torch.where(y <= p["y_full"] - p["y_bot"], y * p["w_max"], p["a_full"] - a1)

    @staticmethod
    def width(p, t, y):
        y1 = torch.clamp(p["y_full"] - y, min=0.0)
        arc = 2.0 * torch.sqrt(torch.clamp(y1 * (2.0 * p["r_bot"] - y1), min=0.0))
        w = torch.where(y <= p["y_full"] - p["y_bot"], p["w_max"] + 0 * y, arc)
        return torch.where(y <= 0.0, torch.zeros_like(y), w)

    @staticmethod
    def y_of_a(p, t, a):
        alpha = torch.clamp((p["a_full"] - a) / (_PI * p["r_bot"] * p["r_bot"]), min=0.0, max=1.0)
        y1 = 2.0 * p["r_bot"] * torch.where(alpha < 0.04, xt.y_circular(alpha),
                                            xt.lookup(alpha, xt._Y))
        return torch.where(a <= p["a_full"] - p["a_bot"], a / p["w_max"], p["y_full"] - y1)

    @classmethod
    def r_of_a(cls, p, t, a):
        rect = a / (p["w_max"] + 2.0 * a / p["w_max"])
        y1 = p["y_full"] - cls.y_of_a(p, t, a)
        theta1 = 2.0 * torch.acos(_acos_arg(torch.clamp(y1, min=0.0), p["r_bot"]))
        pw = (p["s_bot"] - theta1) * p["r_bot"]
        y1b = p["y_full"] - p["y_bot"]
        pw = pw + 2.0 * y1b + p["w_max"]
        return torch.where(a <= p["a_full"] - p["a_bot"], rect, a / pw)

    @classmethod
    def ds_da(cls, p, t, a):
        r = a / (p["w_max"] + 2.0 * a / p["w_max"])
        analytic = (5. / 3. - (2. / 3.) * (2.0 / p["w_max"]) * r) * torch.pow(r, _TWO_THIRDS)
        use = (a <= p["a_full"] - p["a_bot"]) & (a / p["a_full"] > 1.0e-30)
        return torch.where(use, analytic, Shape.ds_da.__func__(cls, p, t, a))


class Trapezoidal(Shape):
    """TRAPEZOIDAL: ``y_bot`` the bottom width, ``s_bot`` the mean side slope, ``r_bot`` the
    side length per unit of depth (SWMM's own field reuse)."""

    name = "TRAPEZOIDAL"

    @staticmethod
    def area(p, t, y):
        return (p["y_bot"] + p["s_bot"] * y) * y

    @staticmethod
    def width(p, t, y):
        return p["y_bot"] + 2.0 * y * p["s_bot"]

    @staticmethod
    def y_of_a(p, t, a):
        flat = p["s_bot"] == 0.0
        s_safe = torch.where(flat, torch.ones_like(p["s_bot"]), p["s_bot"])
        disc = torch.clamp(p["y_bot"] * p["y_bot"] + 4. * p["s_bot"] * a, min=0.0)
        return torch.where(flat, a / p["y_bot"], (torch.sqrt(disc) - p["y_bot"]) / (2. * s_safe))

    @classmethod
    def r_of_a(cls, p, t, a):
        return a / (p["y_bot"] + cls.y_of_a(p, t, a) * p["r_bot"])

    @classmethod
    def radius(cls, p, t, y):
        a = (p["y_bot"] + p["s_bot"] * y) * y
        den = p["y_bot"] + y * p["r_bot"]
        return torch.where(y == 0.0, torch.zeros_like(y), a / den)

    @classmethod
    def ds_da(cls, p, t, a):
        r = cls.r_of_a(p, t, a)
        dpda = p["r_bot"] / torch.sqrt(p["y_bot"] * p["y_bot"] + 4. * p["s_bot"] * a)
        analytic = (5. / 3. - (2. / 3.) * dpda * r) * torch.pow(r, _TWO_THIRDS)
        return torch.where(a / p["a_full"] <= 1.0e-30, Shape.ds_da.__func__(cls, p, t, a), analytic)


class Triangular(Shape):
    name = "TRIANGULAR"

    @staticmethod
    def area(p, t, y):
        return y * y * p["s_bot"]

    @staticmethod
    def width(p, t, y):
        return 2.0 * p["s_bot"] * y

    @staticmethod
    def y_of_a(p, t, a):
        return torch.sqrt(a / p["s_bot"])

    @classmethod
    def r_of_a(cls, p, t, a):
        return a / (2. * cls.y_of_a(p, t, a) * p["r_bot"])

    @staticmethod
    def radius(p, t, y):
        return (y * p["s_bot"]) / (2. * p["r_bot"])

    @classmethod
    def ds_da(cls, p, t, a):
        pos = a / p["a_full"] > 1.0e-30
        a_s = torch.where(pos, a, p["a_full"])
        r = cls.r_of_a(p, t, a_s)
        dpda = p["r_bot"] / torch.sqrt(a_s * p["s_bot"])
        analytic = (5. / 3. - (2. / 3.) * dpda * r) * torch.pow(r, _TWO_THIRDS)
        return torch.where(pos, analytic, Shape.ds_da.__func__(cls, p, t, a))


class Parabolic(Shape):
    """PARABOLIC (``y = c x^2``, ``r_bot = 1/sqrt(c)``)."""

    name = "PARABOLIC"

    @staticmethod
    def area(p, t, y):
        return 4. / 3. * p["r_bot"] * y * torch.sqrt(y)

    @staticmethod
    def width(p, t, y):
        return 2.0 * p["r_bot"] * torch.sqrt(y)

    @staticmethod
    def y_of_a(p, t, a):
        return torch.pow((3. / 4.) * a / p["r_bot"], _TWO_THIRDS)

    @staticmethod
    def perimeter(p, y):
        x = 2. * torch.sqrt(y) / p["r_bot"]
        tt = torch.sqrt(1.0 + x * x)
        return 0.5 * p["r_bot"] * p["r_bot"] * (x * tt + torch.log(x + tt))

    @classmethod
    def radius(cls, p, t, y):
        pos = y > 0.0
        y_s = torch.where(pos, y, torch.ones_like(y))
        return torch.where(pos, cls.area(p, t, y_s) / cls.perimeter(p, y_s), torch.zeros_like(y))

    @classmethod
    def r_of_a(cls, p, t, a):
        pos = a > 0.0
        a_s = torch.where(pos, a, torch.ones_like(a))
        return torch.where(pos, a_s / cls.perimeter(p, cls.y_of_a(p, t, a_s)), torch.zeros_like(a))


class PowerFunc(Shape):
    """POWER (``w = h y^m``): ``s_bot = 1/Geom3`` (the exponent ``m``), ``r_bot`` the area
    coefficient."""

    name = "POWER"

    @staticmethod
    def area(p, t, y):
        return p["r_bot"] * torch.pow(y, p["s_bot"] + 1.0)

    @staticmethod
    def width(p, t, y):
        return (p["s_bot"] + 1.0) * p["r_bot"] * torch.pow(y, p["s_bot"])

    @staticmethod
    def y_of_a(p, t, a):
        return torch.pow(a / p["r_bot"], 1.0 / (p["s_bot"] + 1.0))

    @staticmethod
    def perimeter(p, y):
        """`powerfunc_getPofY`: the numerical arc length, in steps of ``0.02 yFull``."""
        dy1 = 0.02 * p["y_full"]
        h = (p["s_bot"] + 1.0) * p["r_bot"] / 2.0
        m = p["s_bot"]
        pw = torch.zeros_like(y)
        y1 = torch.zeros_like(y)
        x1 = torch.zeros_like(y)
        live = torch.ones(y.shape, dtype=torch.bool)
        for _ in range(500):
            y2 = torch.minimum(y1 + dy1, y)
            x2 = h * torch.pow(y2, m)
            dx, dy = x2 - x1, y2 - y1
            pw = torch.where(live, pw + torch.sqrt(dx * dx + dy * dy), pw)
            x1 = torch.where(live, x2, x1)
            y1 = torch.where(live, y2, y1)
            live = live & (y2 < y)
            if not bool(live.any()):
                break
        return 2.0 * pw

    @classmethod
    def radius(cls, p, t, y):
        pos = y > 0.0
        y_s = torch.where(pos, y, torch.ones_like(y))
        return torch.where(pos, cls.area(p, t, y_s) / cls.perimeter(p, y_s), torch.zeros_like(y))

    @classmethod
    def r_of_a(cls, p, t, a):
        pos = a > 0.0
        a_s = torch.where(pos, a, torch.ones_like(a))
        return torch.where(pos, a_s / cls.perimeter(p, cls.y_of_a(p, t, a_s)), torch.zeros_like(a))


SHAPES: dict[str, type[Shape]] = {
    "CIRCULAR": Circular, "FORCE_MAIN": ForceMain, "FILLED_CIRCULAR": FilledCircular,
    "EGG": Egg, "HORSESHOE": Horseshoe, "GOTHIC": Gothic, "CATENARY": Catenary,
    "SEMIELLIPTICAL": SemiElliptical, "BASKETHANDLE": BasketHandle,
    "SEMICIRCULAR": SemiCircular, "HORIZ_ELLIPSE": HorizEllipse, "VERT_ELLIPSE": VertEllipse,
    "ARCH": Arch, "IRREGULAR": Rows, "CUSTOM": Custom, "RECT_CLOSED": RectClosed,
    "RECT_OPEN": RectOpen, "RECT_TRIANGULAR": RectTriangular, "RECT_ROUND": RectRound,
    "MODBASKETHANDLE": ModBasketHandle, "TRAPEZOIDAL": Trapezoidal, "TRIANGULAR": Triangular,
    "PARABOLIC": Parabolic, "POWER": PowerFunc,
}
#: ``sMax / sFull`` of the circle-table shapes (`xsect.c:248, 258`)
S_MAX_RATIO = {"CIRCULAR": 1.08, "FORCE_MAIN": 1.06949}

#: the per-parameter names an `XsectSet` stores per conduit
PARAMS = ("y_full", "a_full", "r_full", "s_full", "s_max", "w_max", "y_bot", "a_bot", "s_bot",
          "r_bot")


# ----------------------------------------------------------------------------- the set
def _sub(d: dict[str, Tensor] | None, loc: Tensor | None) -> dict[str, Tensor] | None:
    if d is None or loc is None:
        return d
    return {k: v[loc] for k, v in d.items()}


def _qcrit(shape: type[Shape], p, t, y: Tensor) -> Tensor:
    """`xsect.c::getQcritical` at depth `y` with a zero target: ``A sqrt(g A / W)``, 0 where
    ``W <= 0`` (and where ``y <= 0``, area 0)."""
    pos = y > 0.0
    ys = torch.where(pos, y, 0.5 * p["y_full"])
    a = torch.where(pos, shape.area(p, t, ys), torch.zeros_like(ys))
    w = shape.width(p, t, ys)
    wet = w > 0.0
    w_safe = torch.where(wet, w, torch.ones_like(w))
    return torch.where(wet, a * torch.sqrt(_G * a / w_safe), torch.zeros_like(a))


def _ycrit_ridder(shape: type[Shape], p, t, q: Tensor) -> Tensor:
    """`xsect.c::getYcritRidder` (then ``MIN(y, yFull)``) for the flows `q` (cfs, > 0)."""
    yf = p["y_full"]
    q2g = q * q / _G
    y0 = 1.01 * torch.pow(q2g / yf, 0.25)
    y0 = torch.where(y0 >= yf, 0.97 * yf, y0)
    y2 = (0.99 * yf).expand_as(q)
    early = _qcrit(shape, p, t, y2) < q
    q0 = _qcrit(shape, p, t, y0)
    half = (0.5 * yf).expand_as(q)
    q1 = _qcrit(shape, p, t, half)
    hi = q0 > q
    y_hi = torch.where(hi, y0, torch.where(q1 > q, half, y2))
    y_lo = torch.where(hi, torch.where(q1 < q, half, torch.zeros_like(q)), y0)
    yc = ridder(lambda y: _qcrit(shape, p, t, y) - q, y_lo, y_hi, 0.001)
    return torch.where(early, yf.expand_as(q), yc)


#: `xsect_getYcrit`'s closed forms of the critical depth from ``q^2 / g``
_YCRIT_FORMS: dict[str, Callable] = {
    "RECT_OPEN": lambda p, q2g: torch.pow(q2g / (p["w_max"] * p["w_max"]), _THIRD),
    "RECT_CLOSED": lambda p, q2g: torch.pow(q2g / (p["w_max"] * p["w_max"]), _THIRD),
    "TRIANGULAR": lambda p, q2g: torch.pow(2.0 * q2g / (p["s_bot"] * p["s_bot"]), 1. / 5.),
    "PARABOLIC": lambda p, q2g: torch.pow(27. / 32. * q2g / (p["r_bot"] * p["r_bot"]), 1. / 4.),
    "POWER": lambda p, q2g: torch.pow(q2g * (p["s_bot"] + 1.0) / (p["r_bot"] * p["r_bot"]),
                                      1. / (2.0 * p["s_bot"] + 3.0)),
}


class _Group:
    """The conduits of one shape in an `XsectSet`: their positions `idx` in the set (None if
    they are all of it, in order) and the parameters `p`, tables `t`; for the critical depth
    their enumeration stations or the positions that need Ridder's method."""

    def __init__(self, kind: str, idx: Tensor | None, p, t):
        self.kind, self.shape, self.idx, self.p, self.t = kind, SHAPES[kind], idx, p, t
        self.n = p["y_full"].shape[0]
        self.safe_y = 0.5 * p["y_full"]
        self.stations = self.enum_loc = self.ridder_loc = None
        if self.shape in (Circular, ForceMain):
            self.stations = xt.critical_flow_stations(p["y_full"], p["a_full"])
        elif kind not in _YCRIT_FORMS:
            ratio = p["a_full"] / (_PI / 4.0 * p["y_full"] ** 2)       # xsect.c:1306
            enum = (ratio >= 0.5) & (ratio <= 2.0)
            self.enum_loc = enum.nonzero().flatten()
            self.ridder_loc = (~enum).nonzero().flatten()
            if self.enum_loc.numel():
                self.stations = self._stations(_sub(p, self.enum_loc), _sub(t, self.enum_loc))

    def _stations(self, p, t) -> Tensor:
        """`getYcritEnum`'s critical flow at the 26 depths ``i yFull / 25`` (cfs), (G, 26)."""
        dy = p["y_full"] / 25.0
        y = torch.arange(26, dtype=F64).unsqueeze(-1) * dy                   # (26, G)
        return _qcrit(self.shape, p, t, y).transpose(0, 1)

    def critical_depth(self, q: Tensor) -> Tensor:
        """`xsect_getYcrit` for the flows `q` (cfs, ``>= 0``, one barrel's), (..., G)."""
        p, t, shape = self.p, self.t, self.shape
        yf = p["y_full"]
        if self.kind in _YCRIT_FORMS:
            q2g = q * q / _G
            y = _YCRIT_FORMS[self.kind](p, q2g)
            return torch.where(q2g == 0.0, torch.zeros_like(y), torch.minimum(y, yf))
        if self.enum_loc is None:                       # the circle's own stations
            return xt.critical_depth(q, yf, p["a_full"], self.stations)
        out = torch.zeros_like(q)
        if self.enum_loc.numel():
            e = self.enum_loc
            out = out.index_copy(-1, e, xt.critical_depth(
                q[..., e], yf[e], p["a_full"][e], self.stations))
        if self.ridder_loc.numel():
            r = self.ridder_loc
            pr, tr = _sub(p, r), _sub(t, r)
            qr = q[..., r]
            q_pos = torch.where(qr > 0.0, qr, torch.ones_like(qr))
            y = torch.minimum(_ycrit_ridder(shape, pr, tr, q_pos), pr["y_full"])
            out = out.index_copy(-1, r, torch.where(qr * qr / _G == 0.0, torch.zeros_like(y), y))
        return out


class XsectSet:
    """The cross sections of N conduits, evaluated by group (the module docstring).

    ``kinds`` the SWMM shape keyword of each conduit; ``values`` the per-conduit constants,
    ft units, one (N,) tensor per name of `PARAMS` (``y_full``, ``a_full``, ``r_full``,
    ``s_full``, ``s_max``, ``w_max`` and the shape's ``y_bot``, ``a_bot``, ``s_bot``,
    ``r_bot``; absent names are zero); ``tables`` the (N, 51) ``area``, ``hrad``, ``width``
    rows of the IRREGULAR and CUSTOM conduits (zero rows elsewhere). All queries take
    tensors whose last dimension is N."""

    def __init__(self, kinds: list[str], values: dict[str, Tensor],
                 tables: dict[str, Tensor] | None = None) -> None:
        self.kinds = list(kinds)
        self.n = len(self.kinds)
        zeros = torch.zeros(self.n, dtype=F64)
        self.values = {k: values.get(k, zeros) for k in PARAMS}
        self.tables = tables
        self.y_full, self.a_full = self.values["y_full"], self.values["a_full"]
        self.r_full, self.s_full = self.values["r_full"], self.values["s_full"]
        self.s_max, self.w_max = self.values["s_max"], self.values["w_max"]
        #: per conduit: whether the section is open (`xsect_isOpen`)
        self.open = torch.tensor([is_open(k) for k in self.kinds], dtype=torch.bool)
        self._tiles: dict[int, XsectSet] = {}
        self.groups: list[_Group] = []
        for kind in dict.fromkeys(self.kinds):
            idx = torch.tensor([i for i, k in enumerate(self.kinds) if k == kind])
            whole = idx.numel() == self.n
            p = {k: v[idx] for k, v in self.values.items()}
            t = None if tables is None or kind not in ("IRREGULAR", "CUSTOM") else \
                {k: v[idx] for k, v in tables.items()}
            self.groups.append(_Group(kind, None if whole else idx, p, t))

    def take(self, index: Tensor) -> XsectSet:
        """The sections of the conduits at positions `index` (long tensor, repeats allowed),
        as a new set."""
        return XsectSet([self.kinds[int(i)] for i in index],
                        {k: v[index] for k, v in self.values.items()},
                        None if self.tables is None
                        else {k: v[index] for k, v in self.tables.items()})

    def tiled(self, m: int) -> XsectSet:
        """The set repeated `m` times along the conduit dimension (for `m` depths evaluated
        in one call, concatenated on the last dimension); cached."""
        if m == 1:
            return self
        if m not in self._tiles:
            self._tiles[m] = self.take(torch.arange(self.n).repeat(m))
        return self._tiles[m]

    def _apply(self, fn: Callable, *args: Tensor):
        """`fn(group, *args sliced to the group)` over the groups, assembled on the last
        dimension (a tuple result is assembled per element)."""
        outs = [fn(g, *(a if g.idx is None or a.dim() == 0 else a[..., g.idx] for a in args))
                for g in self.groups]
        if len(self.groups) == 1 and self.groups[0].idx is None:
            return outs[0]
        shape = torch.broadcast_shapes(*(a.shape for a in args))
        if isinstance(outs[0], tuple):
            res = []
            for k in range(len(outs[0])):
                full = torch.zeros(shape, dtype=F64)
                for g, o in zip(self.groups, outs, strict=True):
                    full = full.index_copy(-1, g.idx, o[k].expand(shape[:-1] + (g.n,)))
                res.append(full)
            return tuple(res)
        full = torch.zeros(shape, dtype=F64)
        for g, o in zip(self.groups, outs, strict=True):
            full = full.index_copy(-1, g.idx, o.expand(shape[:-1] + (g.n,)))
        return full

    # ---------------------------------------------------------------- geometry queries
    def area(self, y: Tensor) -> Tensor:
        """`xsect_getAofY` (ft2) at depths `y` (ft); 0 at ``y <= 0``."""
        def f(g, yy):
            pos = yy > 0.0
            a = g.shape.area(g.p, g.t, torch.where(pos, yy, g.safe_y))
            return torch.where(pos, a, torch.zeros_like(a))
        return self._apply(f, y)

    def width(self, y: Tensor) -> Tensor:
        """`xsect_getWofY` (ft) at depths `y` (ft); 0 at ``y <= 0``."""
        def f(g, yy):
            pos = yy > 0.0
            w = g.shape.width(g.p, g.t, torch.where(pos, yy, g.safe_y))
            return torch.where(pos, w, torch.zeros_like(w))
        return self._apply(f, y)

    def radius(self, y: Tensor) -> Tensor:
        """`xsect_getRofY` (ft) at depths `y` (ft); 0 at ``y <= 0``."""
        def f(g, yy):
            pos = yy > 0.0
            r = g.shape.radius(g.p, g.t, torch.where(pos, yy, g.safe_y))
            return torch.where(pos, r, torch.zeros_like(r))
        return self._apply(f, y)

    def area_width(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """`area` and `width` at depths `y`."""
        return self.area(y), self.width(y)

    def area_radius(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """`area` and `radius` at depths `y`."""
        return self.area(y), self.radius(y)

    def froude(self, v: Tensor, y: Tensor) -> Tensor:
        """`link.c::link_getFroude` for velocities `v` (ft/s) at depths `y` (ft): ``|v| /
        sqrt(g A / W)``; 0 when empty, or within FUDGE of the crown of a closed section."""
        dead = (y <= _FUDGE) | (~self.open & (self.y_full - y <= _FUDGE))
        y_safe = torch.where(dead, 0.5 * self.y_full, y)
        a, w = self.area_width(y_safe)
        w_safe = torch.where(w > 0.0, w, torch.ones_like(w))
        return torch.where(dead, torch.zeros_like(y), torch.abs(v) / torch.sqrt(_G * (a / w_safe)))

    def courant_depth(self) -> Tensor:
        """The depth of `link.c::conduit_getLengthFactor`'s Courant length (ft): ``yFull``,
        or ``aFull / W(yFull)`` for an open section."""
        w = self.width(self.y_full)
        return torch.where(self.open, self.a_full / torch.where(w > 0.0, w, torch.ones_like(w)),
                           self.y_full)

    # ------------------------------------------------------------- flow-derived depths
    def normal_depth(self, q: Tensor, beta: Tensor) -> Tensor:
        """`link.c::link_getYnorm` (ft) for the flows `q` (cfs, one barrel's) with
        ``beta = PHI sqrt(slope) / n``: ``q`` limited to ``qMax = sMax beta``, then
        ``Y(A(S))`` with ``S = q / beta`` (`xsect_getAofS`: the shape's table inverse, the
        circle's closed forms, or `generic_getAofS`'s Newton iteration)."""
        def f(g, qq, bb):
            p, t, shape = g.p, g.t, g.shape
            if shape in (Circular, ForceMain):
                return xt.normal_depth_ft(qq, p["y_full"], p["a_full"], p["s_full"], bb,
                                          S_MAX_RATIO[g.kind])
            qa = torch.minimum(torch.abs(qq), p["s_max"] * bb)             # link.c:797-798
            live = qa > 0.0
            s = torch.where(live, qa / bb, p["s_full"])
            s = torch.minimum(s, p["s_max"])                              # xsect.c:1159
            a = shape.a_of_s(p, t, s)
            y = shape.y_of_a(p, t, a)
            return torch.where(live, y, torch.zeros_like(y))
        return self._apply(f, q, beta)

    def critical_depth(self, q: Tensor) -> Tensor:
        """`xsect.c::xsect_getYcrit` (ft) for the flows `q` (cfs, ``>= 0``, one barrel's)."""
        return self._apply(lambda g, qq: g.critical_depth(qq), q)
