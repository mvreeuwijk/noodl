"""`[XSECTIONS]` of a conduit and `[TRANSECTS]` of the SWMM `.inp` reader: SWMM 5.2.4's
`link.c::link_readXsectParams` and `transect.c::transect_readParams`, into `xsect_spec`.

`read_xsect` reads a conduit's `[XSECTIONS]` line into an `Xsect` (lengths in m: `lf` m per
length unit of the file), with SWMM's own parsing rules; `read_transects` reads `[TRANSECTS]`
(``NC``, ``X1`` and ``GR`` lines) with SWMM's reader state, which carries from one transect to
the next exactly as `transect.c` does: the Manning n of the left and right overbanks and the
main channel are static variables, a zero on an ``NC`` line keeps the value of the previous
transect, and the main channel's n is multiplied by ``sqrt(Lfactor)`` when a transect is
validated, so a later ``NC`` line with a zero channel n inherits the scaled value.
"""

from __future__ import annotations

import math

from noodl.apps.inpfile import as_float, require_fields
from noodl.apps.sewer.xsect_spec import (
    KINDS,
    LENGTH_FIELDS,
    UNMODELLED_KINDS,
    Transect,
    Xsect,
    custom_tables,
)
from noodl.apps.sewer.xsect_tables import LCF

#: ``Ucf[RAINDEPTH]`` (units.c): the Darcy-Weisbach roughness height of a FORCE_MAIN is read in
#: inches (US, 12 per ft) or mm (SI, 304.8 per ft); metres per such unit.
_ROUGHNESS_M = {"US": 0.3048 / 12.0, "SI": 0.3048 / 304.8}


def read_transects(lines, path, *, us: bool) -> dict[str, Transect]:
    """`{name: Transect}` of the `[TRANSECTS]` lines, SI. `us` whether the file's length unit is
    the foot (SWMM's ``UCF(LENGTH)`` is then 1, else 0.3048 m per ft). Refused, naming the
    line: an unknown keyword, an unpaired ``GR`` value, stations that run backwards, and the
    transects SWMM rejects (`xsect_spec.transect_tables`: fewer than two stations, no depth, no
    main-channel n, left bank right of the right bank)."""
    ucf = 1.0 if us else LCF
    n_left = n_right = n_channel = 0.0           # transect.c:81-83 (static, persistent)
    out: dict[str, Transect] = {}
    current: dict | None = None

    def finish() -> None:
        """`transect_validate`: build the transect that is complete and scale the channel n."""
        nonlocal n_channel
        if current is None:
            return
        t = Transect(
            current["name"], tuple(s * LCF for s in current["stations"]),
            tuple(e * LCF for e in current["elev"]), current["left"] * LCF,
            current["right"] * LCF, n_left, n_right, n_channel, current["lfactor"])
        from noodl.apps.sewer.xsect_spec import transect_tables
        try:
            transect_tables(t)
        except ValueError as exc:
            raise ValueError(f"{path}: line {current['line']}: {exc}") from None
        out[t.name] = t
        n_channel = n_channel * math.sqrt(current["lfactor"])        # transect.c:230

    for line in lines:
        key = line.fields[0].upper()
        if key == "NC":
            finish()
            current = None
            require_fields(line, 4, "an NC line", path)
            n = [as_float(line, i, "a Manning n", path) for i in (1, 2, 3)]
            if any(v < 0.0 for v in n):
                raise ValueError(f"{path}: line {line.number}: a transect Manning n must not "
                                 f"be negative")
            if n[0] > 0.0:
                n_left = n[0]
            if n[1] > 0.0:
                n_right = n[1]
            if n[2] > 0.0:
                n_channel = n[2]
            if n_left == 0.0:
                n_left = n_channel
            if n_right == 0.0:
                n_right = n_channel
        elif key == "X1":
            finish()
            require_fields(line, 10, "an X1 line", path)
            name = line.fields[1]
            x = [as_float(line, i, "an X1 field", path) for i in range(2, 10)]
            lfactor = x[5] or 1.0                                  # x[7] in transect.c
            xfactor = x[6] or 1.0                                  # x[8]
            current = {
                "name": name, "line": line.number, "stations": [], "elev": [],
                "left": (x[1] / ucf) * xfactor,                    # transect.c:345, 351
                "right": (x[2] / ucf) * xfactor,
                "lfactor": lfactor, "xfactor": xfactor, "yfactor": x[7] / ucf,
            }
        elif key == "GR":
            if current is None:
                raise ValueError(f"{path}: line {line.number}: a GR line before any X1 line "
                                 f"(SWMM's ERR_TRANSECT_UNKNOWN)")
            if (len(line.fields) - 1) % 2:
                raise ValueError(f"{path}: line {line.number}: a GR line needs "
                                 f"elevation-station pairs")
            for i in range(1, len(line.fields), 2):
                y = as_float(line, i, "a station elevation", path)
                xs = as_float(line, i + 1, "a station", path)
                station = xs * current["xfactor"] / ucf             # transect.c:374
                if current["stations"] and station < current["stations"][-1]:
                    raise ValueError(f"{path}: line {line.number}: transect "
                                     f"{current['name']!r} stations must not decrease (SWMM's "
                                     f"ERR_TRANSECT_SEQUENCE)")
                current["stations"].append(station)
                current["elev"].append((y + current["yfactor"]) / ucf)    # transect.c:377
        else:
            raise ValueError(f"{path}: line {line.number}: [TRANSECTS] keyword "
                             f"{line.fields[0]!r} is not NC, X1 or GR")
    finish()
    return out


def read_xsect(line, path, *, lf: float, us: bool, transects: dict[str, Transect],
               shape_curves: dict[str, tuple], force_main_darcy: bool) -> Xsect:
    """A conduit's `Xsect` from its `[XSECTIONS]` line (`link_readXsectParams`): the shape
    keyword and Geom1..Geom4 (lengths converted by `lf` m per length unit; see
    `xsect_spec.LENGTH_FIELDS`), the transect of an IRREGULAR section, the SHAPE curve
    (`shape_curves`, ``{name: (xs, ys)}``) of a CUSTOM one.
    `force_main_darcy` whether FORCE_MAIN_EQUATION is D-W (the FORCE_MAIN roughness is then a
    height, in inches or mm, converted to m). Refused, naming the line, a shape SWMM does not
    know, one this package does not model and the data SWMM rejects."""
    name, word = line.fields[0], line.fields[1].upper()
    if word not in KINDS:
        raise ValueError(f"{path}: line {line.number}: link {name!r} has the unknown "
                         f"cross-section shape {line.fields[1]!r}")
    if word in UNMODELLED_KINDS:
        raise ValueError(f"{path}: line {line.number}: link {name!r}: cross section {word}: "
                         f"{UNMODELLED_KINDS[word]}")
    if word == "IRREGULAR":
        if line.fields[2] not in transects:
            raise ValueError(f"{path}: line {line.number}: link {name!r} names the transect "
                             f"{line.fields[2]!r}, which [TRANSECTS] does not define")
        return Xsect("IRREGULAR", transect=transects[line.fields[2]])
    require_fields(line, 6, "a cross-section", path)
    if word == "CUSTOM":
        height = as_float(line, 2, "the CUSTOM section's maximum depth", path)
        if not height > 0.0:
            raise ValueError(f"{path}: line {line.number}: link {name!r} CUSTOM depth must be "
                             f"positive")
        curve = line.fields[3]
        if curve not in shape_curves:
            raise ValueError(f"{path}: line {line.number}: link {name!r} names the SHAPE curve "
                             f"{curve!r}, which [CURVES] does not define")
        pts = tuple(zip(*shape_curves[curve], strict=True))
        try:
            custom_tables(pts)
        except ValueError as exc:
            raise ValueError(f"{path}: line {line.number}: curve {curve!r}: {exc}") from None
        return Xsect("CUSTOM", (height * lf,), curve=pts)
    p = [as_float(line, i, f"Geom{i - 1}", path) for i in range(2, 6)]
    if word in ("HORIZ_ELLIPSE", "VERT_ELLIPSE", "ARCH") and p[1] == 0.0:
        p[2] = p[0]                                  # xsect.c:553, 581, 609: Geom1 is the code
    geom = list(p)
    for i in LENGTH_FIELDS.get(word, ()):
        geom[i] = p[i] * lf
    if word == "FORCE_MAIN" and force_main_darcy:
        geom[1] = p[1] * _ROUGHNESS_M["US" if us else "SI"]      # link.c:1034
    x = Xsect(word, tuple(geom))
    try:
        x.params()
    except ValueError as exc:
        raise ValueError(f"{path}: line {line.number}: link {name!r}: {exc}") from None
    if word == "FORCE_MAIN" and not geom[1] > 0.0:
        raise ValueError(f"{path}: line {line.number}: link {name!r} FORCE_MAIN needs a "
                         f"positive roughness (the Hazen-Williams C or the Darcy-Weisbach "
                         f"height; SWMM's ERR_XSECT)")
    return x
