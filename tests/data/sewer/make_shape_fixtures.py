"""Regenerate the SWMM cross-section, loss and transect parity fixtures (run from the repo
root with ``python tests/data/sewer/make_shape_fixtures.py``).

One small network per fixture, one independent chain per cross section: a junction with a
constant inflow, a 200 m conduit of the shape under test and an outfall. The inflow of a chain
is a stated fraction of the conduit's SWMM full flow ``sFull * beta`` (from
`noodl.apps.sewer.xsect_spec`, the constants SWMM computes) and the conduit starts at 30 % of
it (``InitFlow``), so the filling sweeps the section through its partly full depths from a wet
state; the slope (0.0008) keeps the flow subcritical, where SWMM's own iteration converges on
every step. Each fixture runs 10 minutes of 5 s steps with SWMM tightened (``MAX_TRIALS 500``,
``HEAD_TOLERANCE 1e-12``) as the other dynamic-wave parity fixtures do, and a junction area
floor (``MIN_SURFAREA 12``) that keeps the first step tame.

* `shapes_dynwave.inp` -- every modelled shape in SI units, free surface, a free outfall.
* `shapes_drop_dynwave.inp` -- the same shapes with an offset at both ends of the conduit.
* `shapes_surcharge_dynwave.inp` -- the closed shapes forced above their crowns by a FIXED
  outfall stage (EXTRAN surcharge).
* `shapes_slot_dynwave.inp` -- the same under the Preissmann slot.
* `shapes_us_dynwave.inp` -- US units (CFS): the standard-size ellipse and arch pipes and two
  channels.
* `force_main_hw_dynwave.inp`, `force_main_dw_dynwave.inp` -- force mains flowing full under
  FORCE_MAIN_EQUATION H-W and D-W.
* `losses_dynwave.inp` -- conduits with entry, exit and average loss coefficients.
* `irregular_dynwave.inp` -- natural channels from `[TRANSECTS]`.
* `culvert_dynwave.inp` -- culverts under inlet control (`culvert.c`).
"""

from __future__ import annotations

import math
from pathlib import Path

from noodl.apps.sewer.xsect_spec import Transect, Xsect, transect_tables
from noodl.apps.sewer.xsect_tables import LCF, PHI, QCF

HERE = Path(__file__).resolve().parent
LENGTH = 200.0
N_MANNING = 0.013
SLOPE = 0.0008

HEADER = """[TITLE]
{title}

[OPTIONS]
FLOW_UNITS           {units}
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
LINK_OFFSETS         DEPTH
MIN_SLOPE            0
ALLOW_PONDING        NO
SKIP_STEADY_STATE    NO
START_DATE           01/01/2026
START_TIME           00:00:00
REPORT_START_DATE    01/01/2026
REPORT_START_TIME    00:00:00
END_DATE             01/01/2026
END_TIME             00:10:00
SWEEP_START          01/01
SWEEP_END            12/31
DRY_DAYS             0
REPORT_STEP          00:05:00
WET_STEP             00:01:00
DRY_STEP             00:01:00
ROUTING_STEP         5
INERTIAL_DAMPING     PARTIAL
NORMAL_FLOW_LIMITED  BOTH
FORCE_MAIN_EQUATION  {force_main}
SURCHARGE_METHOD     {surcharge}
VARIABLE_STEP        0
LENGTHENING_STEP     {lengthening}
MIN_SURFAREA         {min_area}
MAX_TRIALS           500
HEAD_TOLERANCE       1e-12
MINIMUM_STEP         0.5
THREADS              1

[EVAPORATION]
CONSTANT         0.0
DRY_ONLY         NO
"""

# (name, SWMM shape keyword, Geom1..Geom4, flow as a fraction of the full flow)
SHAPES = [
    ("RC", "RECT_CLOSED", (0.6, 1.2, 0, 0), 0.5),
    ("RO", "RECT_OPEN", (0.5, 1.5, 0, 0), 0.5),
    ("RO2", "RECT_OPEN", (0.5, 1.5, 2, 0), 0.8),
    ("TZ", "TRAPEZOIDAL", (0.8, 1.0, 1.5, 2.0), 0.5),
    ("TZW", "TRAPEZOIDAL", (0.5, 6.0, 3.0, 4.0), 0.5),
    ("TR", "TRIANGULAR", (0.6, 1.8, 0, 0), 0.5),
    ("PB", "PARABOLIC", (0.7, 2.0, 0, 0), 0.5),
    ("PW", "POWER", (0.7, 2.0, 2.0, 0), 0.5),
    ("RT", "RECT_TRIANGULAR", (0.8, 1.2, 0.3, 0), 0.5),
    ("RR", "RECT_ROUND", (0.8, 1.0, 0.7, 0), 0.5),
    ("MB", "MODBASKETHANDLE", (1.0, 1.0, 0.7, 0), 0.5),
    ("HE", "HORIZ_ELLIPSE", (0.6, 0.9, 0, 0), 0.5),
    ("VE", "VERT_ELLIPSE", (0.9, 0.6, 0, 0), 0.5),
    ("AR", "ARCH", (0.6, 0.9, 0, 0), 0.5),
    ("EG", "EGG", (0.9, 0, 0, 0), 0.5),
    ("HS", "HORSESHOE", (0.9, 0, 0, 0), 0.5),
    ("GO", "GOTHIC", (0.9, 0, 0, 0), 0.5),
    ("CA", "CATENARY", (0.9, 0, 0, 0), 0.5),
    ("SE", "SEMIELLIPTICAL", (0.9, 0, 0, 0), 0.5),
    ("BH", "BASKETHANDLE", (0.9, 0, 0, 0), 0.5),
    ("SC", "SEMICIRCULAR", (0.9, 0, 0, 0), 0.5),
    ("FC", "FILLED_CIRCULAR", (0.9, 0.15, 0, 0), 0.5),
    ("CU", "CUSTOM", (0.9, "CURVE1", 0, 0), 0.5),
    ("FM", "FORCE_MAIN", (0.6, 120, 0, 0), 0.5),
    ("CI", "CIRCULAR", (0.6, 0, 0, 0), 0.5),
]
SHAPE_CURVE = [(0.0, 0.0), (0.2, 0.8), (0.6, 1.0), (0.9, 0.6), (1.0, 0.0)]
CURVES = "[CURVES]\n;;Name Type X Y\n" + "\n".join(
    f"CURVE1 {'SHAPE' if i == 0 else '     '} {x} {y}" for i, (x, y) in enumerate(SHAPE_CURVE)
) + "\n"
#: ``Geom1..Geom4`` positions that are lengths, per shape (the rest are slopes, codes, ...)
LENGTHS = {"FILLED_CIRCULAR": (0, 1), "RECT_CLOSED": (0, 1), "RECT_OPEN": (0, 1),
           "TRAPEZOIDAL": (0, 1), "TRIANGULAR": (0, 1), "PARABOLIC": (0, 1), "POWER": (0, 1),
           "RECT_TRIANGULAR": (0, 1, 2), "RECT_ROUND": (0, 1, 2),
           "MODBASKETHANDLE": (0, 1, 2), "HORIZ_ELLIPSE": (0, 1), "VERT_ELLIPSE": (0, 1),
           "ARCH": (0, 1), "CUSTOM": (0,), "CIRCULAR": (0,), "FORCE_MAIN": (0,)}


def _xsect(kind, geom, scale=1.0):
    """The `Xsect` of a fixture line whose lengths are `scale` m per file unit."""
    if kind == "CUSTOM":
        return Xsect("CUSTOM", (geom[0] * scale,), curve=tuple(SHAPE_CURVE))
    g = [float(v) * (scale if i in LENGTHS.get(kind, (0,)) else 1.0) for i, v in enumerate(geom)]
    if kind in ("HORIZ_ELLIPSE", "VERT_ELLIPSE", "ARCH") and float(geom[1]) == 0.0:
        g[2] = float(geom[0])                     # SWMM: a zero Geom2 makes Geom1 the size code
    return Xsect(kind, tuple(g))


def _full_flow(x, slope=SLOPE, n=N_MANNING):
    """SWMM's ``sFull * beta`` (m3/s) of the section `x` (a FORCE_MAIN at its equivalent n)."""
    p = x.params()
    if x.kind == "FORCE_MAIN":
        n = 1.067 / p["roughness"] * (p["y_full"] / slope) ** 0.04
    return p["s_full"] * PHI * math.sqrt(slope) / n * QCF


def _chains(cases, *, inflow_scale=1.0, in_off=0.0, out_off=0.0, outfall="FREE NO", init=0.5,
            barrels=1, losses=None, init_depth=0.0):
    j, o, c, x, f = ["[JUNCTIONS]"], ["[OUTFALLS]"], ["[CONDUITS]"], ["[XSECTIONS]"], \
        ["[INFLOWS]"]
    for name, kind, geom, frac in cases:
        flow = frac * _full_flow(_xsect(kind, geom)) * inflow_scale
        fall = SLOPE * LENGTH
        j.append(f"J{name:<5} {10.0 + in_off:.6f} 5 {init_depth} 0 0")
        o.append(f"O{name:<5} {10.0 - fall:.6f} {outfall}")
        c.append(f"C{name:<5} J{name} O{name} {LENGTH} {N_MANNING} {in_off} {out_off} "
                 f"{init * flow:.6g} 0")
        g = " ".join(str(v) for v in geom)
        x.append(f"C{name:<5} {kind:<16} {g} {barrels}")
        f.append(f"J{name:<5} FLOW \"\" FLOW 1.0 1.0 {flow:.6g}")
    blocks = [j, o, c, x, f]
    if losses:
        blocks.append(["[LOSSES]"] + [f"{n:<6} {a} {b} {d} NO 0" for n, (a, b, d) in
                                       losses.items()])
    return blocks


def _write(path, title, blocks, *, units="CMS", force_main="H-W", extra="", min_area=12.0,
           surcharge="EXTRAN", lengthening=0):
    text = HEADER.format(title=title, units=units, force_main=force_main, min_area=min_area,
                         surcharge=surcharge, lengthening=lengthening)
    text += "\n" + "\n\n".join("\n".join(b) for b in blocks) + "\n"
    if extra:
        text += "\n" + extra
    Path(path).write_text(text)


def _us_fixture() -> None:
    """US units (ft, CFS): standard-size ellipses and arches, and two channels."""
    cases = [
        ("HE", "HORIZ_ELLIPSE", (2, 3, 6, 0), 4),         # standard size code 6
        ("HC", "HORIZ_ELLIPSE", (4, 0, 0, 0), 6),         # Geom1 alone is the size code
        ("VE", "VERT_ELLIPSE", (2, 3, 4, 0), 5),
        ("AR", "ARCH", (2, 3, 13, 0), 5),
        ("AC", "ARCH", (11, 0, 0, 0), 5),
        ("RC", "RECT_CLOSED", (2, 4, 0, 0), 5),
        ("TZ", "TRAPEZOIDAL", (3, 4, 2, 3), 5),
    ]
    j, o, c, x, f = ["[JUNCTIONS]"], ["[OUTFALLS]"], ["[CONDUITS]"], ["[XSECTIONS]"], \
        ["[INFLOWS]"]
    length = 600.0
    for name, kind, geom, tenths in cases:
        p = _xsect(kind, geom, LCF).params()
        flow = tenths / 10.0 * p["s_full"] * PHI * math.sqrt(SLOPE) / N_MANNING     # cfs
        j.append(f"J{name:<5} 30.0 15 0 0 0")
        o.append(f"O{name:<5} {30.0 - SLOPE * length:.6f} FREE NO")
        c.append(f"C{name:<5} J{name} O{name} {length} {N_MANNING} 0 0 {0.3 * flow:.6g} 0")
        x.append(f"C{name:<5} {kind:<16} {' '.join(str(v) for v in geom)} 1")
        f.append(f"J{name:<5} FLOW \"\" FLOW 1.0 1.0 {flow:.6g}")
    _write(HERE / "shapes_us_dynwave.inp",
           "Standard-size ellipse and arch pipes and channels, US units - noodl fixture",
           [j, o, c, x, f], units="CFS", min_area=120.0)


def _force_main_fixtures() -> None:
    """Force mains flowing full against a FIXED outfall, under each friction law."""
    for law, rough in (("H-W", "130"), ("D-W", "0.5")):       # C, or the height in mm
        j, o, c, x, f = ["[JUNCTIONS]"], ["[OUTFALLS]"], ["[CONDUITS]"], ["[XSECTIONS]"], \
            ["[INFLOWS]"]
        for name, d, flow in (("A", 0.4, 0.09), ("B", 0.7, 0.30)):
            j.append(f"J{name} 10.0 5 0 0 0")
            o.append(f"O{name} {10.0 - SLOPE * LENGTH:.6f} FIXED 11.5 NO")
            c.append(f"C{name} J{name} O{name} {LENGTH} 0.013 0 0 {0.3 * flow:.4g} 0")
            x.append(f"C{name} FORCE_MAIN {d} {rough} 0 0 1")
            f.append(f"J{name} FLOW \"\" FLOW 1.0 1.0 {flow}")
        _write(HERE / f"force_main_{law.replace('-', '').lower()}_dynwave.inp",
               f"Surcharged force mains, {law} - noodl fixture", [j, o, c, x, f],
               force_main=law)


def _loss_fixture() -> None:
    cases = [("A", "CIRCULAR", (0.6, 0, 0, 0), 0.5), ("B", "CIRCULAR", (0.6, 0, 0, 0), 0.5),
             ("C", "CIRCULAR", (0.45, 0, 0, 0), 0.8), ("D", "RECT_CLOSED", (0.6, 1.2, 0, 0), 0.5)]
    blocks = _chains(cases, losses={"CA": (0.5, 0.0, 0.0), "CB": (0.0, 1.0, 0.0),
                                    "CC": (0.5, 1.0, 0.2), "CD": (0.4, 0.8, 0.1)})
    _write(HERE / "losses_dynwave.inp",
           "Entry, exit and average local losses - noodl fixture", blocks)


FLOW_FRACTION = {"A": 0.55}   # of the full flow; the others 1.1


def _culvert_fixture() -> None:
    """Culverts under inlet control: each inlet type of a different form and shape, driven
    from an unsubmerged inlet through the transition zone to a submerged one."""
    cases = [("A", "CIRCULAR", (0.6, 0, 0, 0), 1), ("B", "CIRCULAR", (0.6, 0, 0, 0), 5),
             ("C", "RECT_CLOSED", (0.6, 1.0, 0, 0), 10), ("D", "RECT_CLOSED", (0.6, 1.0, 0, 0), 12),
             ("E", "CIRCULAR", (0.6, 0, 0, 0), 49), ("F", "HORIZ_ELLIPSE", (0.6, 0.9, 0, 0), 30)]
    j, o, c, x, f = ["[JUNCTIONS]"], ["[OUTFALLS]"], ["[CONDUITS]"], ["[XSECTIONS]"], \
        ["[INFLOWS]"]
    slope, length = 0.004, 40.0
    for name, kind, geom, code in cases:
        flow = FLOW_FRACTION.get(name, 1.1) * _full_flow(_xsect(kind, geom), slope=slope)
        j.append(f"J{name} 10.0 5 0.3 0 0")
        o.append(f"O{name} {10.0 - slope * length:.6f} FREE NO")
        c.append(f"C{name} J{name} O{name} {length} {N_MANNING} 0 0 {0.3 * flow:.5g} 0")
        x.append(f"C{name} {kind:<14} {' '.join(str(v) for v in geom)} 1 {code}")
        f.append(f"J{name} FLOW \"\" FLOW 1.0 1.0 {flow:.5g}")
    _write(HERE / "culvert_dynwave.inp", "Culvert inlet control, all forms - noodl fixture",
           [j, o, c, x, f], min_area=20.0)


# name, nLeft, nRight, nChannel, left bank, right bank, Lfactor, Xfactor, Yfactor, GR pairs
# (elevation, station): a compound channel, a meandering one with a station multiplier and
# an elevation offset, and a triangular gully.
TRANSECTS = [
    ("T1", 0.080, 0.080, 0.035, 6.0, 14.0, 1.0, 1.0, 0.0,
     [(3.0, 0.0), (2.2, 3.0), (1.5, 6.0), (0.4, 7.5), (0.0, 8.5), (0.0, 11.5), (0.4, 12.5),
      (1.5, 14.0), (2.2, 17.0), (3.0, 20.0)]),
    ("T2", 0.100, 0.060, 0.030, 3.0, 7.0, 1.25, 2.0, 5.0,
     [(2.0, 0.0), (1.0, 1.5), (0.0, 3.0), (0.0, 5.0), (1.0, 7.0), (2.0, 8.5), (2.6, 11.0)]),
    ("T3", 0.0, 0.0, 0.040, 0.0, 12.0, 1.0, 1.0, 0.0,
     [(2.0, 0.0), (0.0, 6.0), (2.0, 12.0)]),
]


def _irregular_fixture() -> None:
    j, o, c, x, f = ["[JUNCTIONS]"], ["[OUTFALLS]"], ["[CONDUITS]"], ["[XSECTIONS]"], \
        ["[INFLOWS]"]
    t = ["[TRANSECTS]"]
    for name, nl, nr, nc, xl, xr, lf, xf, yf, gr in TRANSECTS:
        t.append(f"NC {nl} {nr} {nc}")
        t.append(f"X1 {name} {len(gr)} {xl} {xr} 0 0 {lf if lf != 1.0 else 0} "
                 f"{xf if xf != 1.0 else 0} {yf}")
        for k in range(0, len(gr), 4):
            t.append("GR " + " ".join(f"{e} {s}" for e, s in gr[k:k + 4]))
        # SWMM adds the elevation offset to the station elevation after dividing it by the
        # unit factor once, then divides the sum again (transect.c:353, 377)
        tr = Transect(name, tuple(s * xf for _, s in gr), tuple(e + yf / LCF for e, _ in gr),
                      xl * xf, xr * xf, nl or nc, nr or nc, nc, lf)
        tt = transect_tables(tr)
        q_full = (tt["a_full"] * tt["r_full"] ** (2 / 3) * PHI * math.sqrt(SLOPE)
                  / (nc * math.sqrt(lf)) * QCF)
        flow = 0.12 * q_full
        j.append(f"J{name} 10.0 6 0 0 0")
        o.append(f"O{name} {10.0 - SLOPE * LENGTH:.6f} FREE NO")
        c.append(f"C{name} J{name} O{name} {LENGTH} 0.99 0 0 {0.3 * flow:.5g} 0")
        x.append(f"C{name} IRREGULAR {name}")
        f.append(f"J{name} FLOW \"\" FLOW 1.0 1.0 {flow:.5g}")
    _write(HERE / "irregular_dynwave.inp",
           "Natural channels from [TRANSECTS] - noodl fixture",
           # SWMM validates the last transect only when another section follows it
           [j, o, c, x, f, t, ["[TAGS]"]],
           min_area=30.0)


def main() -> None:
    _write(HERE / "shapes_dynwave.inp",
           "Every cross-section shape, SI units, free surface, free outfall - noodl fixture",
           _chains(SHAPES), extra=CURVES)
    _write(HERE / "shapes_drop_dynwave.inp",
           "Every cross-section shape with conduit offsets at both ends - noodl fixture",
           # RECT_ROUND is left out: its hydraulic radius in the circular bottom is SWMM's
           # ``0.5 r (1 - sin t) / t`` (xsect.c:2062, which falls to 0 at the springing
           # line), and with a free fall at the outfall the end depth sits in that bottom,
           # where the lagged equations have two fixed points and SWMM and noodl may settle
           # on different ones
           _chains([s for s in SHAPES if s[0] != "RR"], in_off=0.1, out_off=0.2),
           extra=CURVES)
    closed = [s for s in SHAPES if s[1] not in
              ("RECT_OPEN", "TRAPEZOIDAL", "TRIANGULAR", "PARABOLIC", "POWER")]
    for method, name in (("EXTRAN", "shapes_surcharge_dynwave.inp"),
                         ("SLOT", "shapes_slot_dynwave.inp")):
        _write(HERE / name,
               f"Closed cross-sections forced past their crowns, {method} - noodl fixture",
               _chains(closed, inflow_scale=1.0, outfall="FIXED 11.2 NO", init=1.0,
                       init_depth=1.2), extra=CURVES,
               surcharge=method)
    _us_fixture()
    _force_main_fixtures()
    _loss_fixture()
    _irregular_fixture()
    _culvert_fixture()


if __name__ == "__main__":
    main()
