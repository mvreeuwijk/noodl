"""A documented SWMM 5 `.inp` subset.

READ: `[TITLE]`, `[OPTIONS]` (`FLOW_UNITS` must be `CMS`),
`[JUNCTIONS]` (invert, MaxDepth, InitDepth, SurDepth), `[OUTFALLS]` (FREE, NORMAL, FIXED, TIDAL and
TIMESERIES, with their flap gate), `[CONDUITS]` (with InitFlow and MaxFlow), `[XSECTIONS]` (only
`CIRCULAR`; barrels), `[LOSSES]` (the flap gate; the loss coefficients and seepage must be
0), `[INFLOWS]` (constant `FLOW`
baselines and constant `CONCENTRATION` pollutant baselines), `[POLLUTANTS]`, and under
DYNWAVE `[STORAGE]` (FUNCTIONAL, TABULAR and the closed shapes CYLINDRICAL, CONICAL,
PARABOLOID, PYRAMIDAL, with SurDepth; a non-zero evaporation factor and exfiltration are
refused) with the `[CURVES]` of type STORAGE that its TABULAR units name, and `[ORIFICES]`,
`[WEIRS]` and `[OUTLETS]` (their `[XSECTIONS]` openings and RATING curves; `_regulators`).
InitDepth,
InitFlow, MaxFlow, flap gates and barrels are dynamic-wave physics: under STEADY or KINWAVE
routing a non-zero or gated value is refused by name rather than skipped.
`FLOW_ROUTING` is STEADY, KINWAVE or DYNWAVE; DYNWAVE also reads `ROUTING_STEP`,
`INERTIAL_DAMPING`, `NORMAL_FLOW_LIMITED`, `SURCHARGE_METHOD`, `MIN_SURFAREA`,
`HEAD_TOLERANCE`, `MAX_TRIALS` and `MIN_SLOPE` into `DynamicWaveOptions`.
The network uses SWMM's own tabulated circular section
(``geometry="tabulated"``, `xsect_tables.py`) unless ``geometry="analytic"`` is passed.
IGNORED because no conduit or node ever references them: `[EVAPORATION]`, `[REPORT]`,
`[TAGS]`, `[MAP]`, `[COORDINATES]`.

REFUSED BY NAME, because skipping them would silently change the physics: `[DWF]`,
`[STORAGE]`, `[ORIFICES]`, `[WEIRS]` and `[OUTLETS]` under STEADY or KINWAVE, `[PUMPS]`,
`[DIVIDERS]`, `[SUBCATCHMENTS]`, `[RAINGAGES]`, `[CONTROLS]`, `[CURVES]` of any type but
STORAGE, RATING and TIDAL, `[TIMESERIES]` (but see below), `[LID_USAGE]`, roadway weirs,
weir coefficient curves,
and any other section carrying content. A
time-series inflow is refused; time variation belongs in the drivers. A conduit with a
culvert code, a non-zero entry, exit or average loss coefficient or seepage rate is refused
for every routing; one with more than one barrel under STEADY or KINWAVE.

DYNWAVE reads `VARIABLE_STEP` (the Courant factor; absent means SWMM's default 0.75,
project.c:854) and `MINIMUM_STEP`, and clips `ROUTING_STEP` to `WET_STEP` (default 300 s)
as SWMM does (project.c:246-250) and `LENGTHENING_STEP` (conduit lengthening, with the file's own
ROUTING_STEP, as SWMM validates conduits before it lowers the step to WET_STEP). It
refuses, with the line number: `ALLOW_PONDING YES`. A TIDAL outfall reads its `[CURVES]` TIDAL
curve (`Outfall.tide`, hours and stage elevations). A TIMESERIES outfall takes its stage from
the `outfall_stage` driver (time variation belongs in the drivers): the reader records the
series name (`Outfall.series`) and accepts a `[TIMESERIES]` section holding only the series
that outfalls name, which it does not read; the model builder then refuses a network
whose TIMESERIES outfall has no driver.

A conduit the reader reverses (adverse, below) has its `InitFlow` sign flipped and its flap
gate recorded as reversed (`Pipe.flap_gate_reversed`), as SWMM does (`conduit_reverse`,
link.c:1189-1190 and `Link.direction`).

Slope. SWMM defines `S0 = dy / dx` with `dx = sqrt(L^2 - dy^2)` (Ref. Man. Vol. II
Eq. 2-1/2-2), so the reader uses the 3-D chord, not `dy / L`. A conduit is normalised
upstream to downstream from the two end elevations (node invert plus that end's offset),
and one with zero or adverse fall is refused by name. Under DYNWAVE the slope follows SWMM's
`conduit_getSlope`: the fall is floored at `MIN_DELTA_Z` = 0.001 ft (0.0003048 m), `MIN_SLOPE`
floors the slope, and a conduit is reversed only when it is adverse (`z1 < z2`), so a flat one
keeps the file's orientation. Offsets are kept on the pipe (`inlet_offset`, `outlet_offset`,
m above the end node's invert) and swap with the ends on a reversal.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

from noodl.apps.inpfile import as_float, read_sections, require_fields
from noodl.apps.sewer.dynwave import DynamicWaveOptions
from noodl.apps.sewer.network import (
    Manhole,
    Orifice,
    Outfall,
    Outlet,
    Pipe,
    SewerNetwork,
    Weir,
)
from noodl.apps.sewer.storage import validate_storage_curve

#: link.c `MIN_DELTA_Z`: the smallest fall SWMM uses for a conduit slope, 0.001 ft in m.
MIN_DELTA_Z = 0.001 * 0.3048
#: SWMM's default `WET_STEP`, s (project.c:864).
DEFAULT_WET_STEP = 300.0
_DYNWAVE_KEYS = (
    "ROUTING_STEP", "WET_STEP", "VARIABLE_STEP", "MINIMUM_STEP", "LENGTHENING_STEP", "MIN_SURFAREA",
    "HEAD_TOLERANCE", "MAX_TRIALS", "MIN_SLOPE", "INERTIAL_DAMPING", "NORMAL_FLOW_LIMITED",
    "SURCHARGE_METHOD",
    "ALLOW_PONDING",
)
_DAMPING = {"NO": "none", "NONE": "none", "PARTIAL": "partial", "FULL": "full"}
_NORMAL_FLOW = {"NO": "none", "NONE": "none", "NEITHER": "none", "SLOPE": "slope",
                "FROUDE": "froude", "BOTH": "both"}


def _seconds(line, path) -> float:
    """An `[OPTIONS]` time in seconds: plain seconds, or `HH:MM:SS` / `HH:MM` (SWMM accepts
    both)."""
    text = line.fields[1]
    if ":" not in text:
        return as_float(line, 1, line.fields[0], path)
    try:
        parts = [float(x) for x in text.split(":")]
    except ValueError:
        parts = []
    if len(parts) not in (2, 3):
        raise ValueError(f"{path}: line {line.number}: bad time {text!r} for {line.fields[0]}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + part
    return seconds * 60.0 if len(parts) == 2 else seconds

_IGNORED = {
    "TITLE", "EVAPORATION", "REPORT", "TAGS", "MAP", "COORDINATES", "VERTICES",
    "POLYGONS", "SYMBOLS", "LABELS", "BACKDROP", "PROFILES", "FILES", "END",
}
_READ = {"OPTIONS", "JUNCTIONS", "OUTFALLS", "CONDUITS", "XSECTIONS", "LOSSES", "INFLOWS",
         "POLLUTANTS", "STORAGE", "CURVES", "ORIFICES", "WEIRS", "OUTLETS"}
#: The regulator sections, read under DYNWAVE only.
_REGULATORS = ("ORIFICES", "WEIRS", "OUTLETS")
#: SWMM's [WEIRS] types (keywords.c:157 `WeirTypeWords`) -> `Weir.kind`; ROADWAY is refused.
_WEIR_KINDS = {"TRANSVERSE": "transverse", "SIDEFLOW": "sideflow", "V-NOTCH": "v_notch",
               "TRAPEZOIDAL": "trapezoidal"}
#: The opening shape each weir kind takes (link.c:2111-2133).
_WEIR_XSECT = {"transverse": "RECT_OPEN", "sideflow": "RECT_OPEN", "v_notch": "TRIANGULAR",
               "trapezoidal": "TRAPEZOIDAL"}
#: SWMM's [STORAGE] shape keywords (keywords.c:108 `RelationWords`)
_STORAGE_SHAPES = ("TABULAR", "FUNCTIONAL", "CYLINDRICAL", "CONICAL", "PARABOLOID",
                   "PYRAMIDAL")
#: SWMM's curve types (keywords.c `CurveTypeWords`)
_CURVE_TYPES = ("STORAGE", "DIVERSION", "TIDAL", "RATING", "CONTROL", "SHAPE", "WEIR",
                "PUMP1", "PUMP2", "PUMP3", "PUMP4", "PUMP5")


def read_swmm_inp(
    path, *, geometry: str = "tabulated"
) -> tuple[SewerNetwork, dict[str, dict[str, float]], dict[str, dict]]:
    """`(network, {node: {pollutant: kg/m3}}, {pollutant: {...}})` for `path`.

    `geometry` defaults to ``"tabulated"``: a network read from a SWMM file is read to reproduce
    SWMM, so its conduits use SWMM 5.2's own tabulated circular section
    (`noodl.apps.sewer.xsect_tables`). Pass ``geometry="analytic"`` for the exact circle.
    """
    path = Path(path)
    sections = read_sections(path)
    for name, lines in sections.items():
        if name in _READ or name in _IGNORED or not lines or name == "TIMESERIES":
            continue                  # [TIMESERIES] is checked against the outfalls below
        raise ValueError(
            f"{path}: section [{name}] (line {lines[0].number}) is not part of the "
            f"documented subset and carries content; this reader models gravity conduits, "
            f"junctions, outfalls and constant inflows only"
        )

    routing = "KINWAVE"
    offsets = "DEPTH"
    raw: dict[str, object] = {}
    opts: dict[str, tuple] = {}
    for line in sections.get("OPTIONS", []):
        key = line.fields[0].upper()
        if key in _DYNWAVE_KEYS:
            raw[key] = line        # parsed below, and only under DYNWAVE
        if key == "FLOW_UNITS":
            require_fields(line, 2, "FLOW_UNITS", path)
            if line.fields[1].upper() != "CMS":
                raise ValueError(
                    f"{path}: line {line.number}: FLOW_UNITS must be CMS -- this "
                    f"application is SI-native and does not convert, got "
                    f"{line.fields[1]!r}"
                )
        elif key == "FLOW_ROUTING":
            require_fields(line, 2, "FLOW_ROUTING", path)
            routing = line.fields[1].upper()
        elif key == "LINK_OFFSETS":
            require_fields(line, 2, "LINK_OFFSETS", path)
            offsets = line.fields[1].upper()
            if offsets not in ("DEPTH", "ELEVATION"):
                raise ValueError(
                    f"{path}: line {line.number}: LINK_OFFSETS must be DEPTH or ELEVATION"
                )

    dynamic = routing == "DYNWAVE"
    dynamic_wave = DynamicWaveOptions()
    min_slope = 0.0
    if dynamic:
        for key, line in raw.items():
            require_fields(line, 2, key, path)
            if key in ("ROUTING_STEP", "WET_STEP", "MINIMUM_STEP", "LENGTHENING_STEP"):
                opts[key] = (line, _seconds(line, path))
            elif key in ("VARIABLE_STEP", "MIN_SURFAREA", "HEAD_TOLERANCE", "MAX_TRIALS",
                         "MIN_SLOPE"):
                opts[key] = (line, as_float(line, 1, key, path))
            else:
                opts[key] = (line, line.fields[1].upper())
        dynamic_wave = _dynamic_wave_options(opts, path)
        min_slope = opts["MIN_SLOPE"][1] / 100.0 if "MIN_SLOPE" in opts else 0.0

    invert: dict[str, float] = {}
    manholes: list[Manhole] = []
    for line in sections.get("JUNCTIONS", []):
        require_fields(line, 2, "a junction", path)
        name = line.fields[0]
        elevation = as_float(line, 1, "the invert elevation", path)
        depth = as_float(line, 2, "the maximum depth", path) if len(line.fields) > 2 else None
        initial = (as_float(line, 3, "the initial depth", path)
                   if len(line.fields) > 3 else 0.0)
        if initial < 0.0:
            raise ValueError(
                f"{path}: line {line.number}: junction {name!r} has a negative initial depth "
                f"{line.fields[3]} m"
            )
        surcharge = (as_float(line, 4, "the surcharge depth", path)
                     if len(line.fields) > 4 else 0.0)
        if surcharge < 0.0:
            raise ValueError(
                f"{path}: line {line.number}: junction {name!r} has a negative surcharge depth "
                f"{line.fields[4]} m (SWMM's SurDepth must be >= 0)"
            )
        invert[name] = elevation
        manholes.append(
            Manhole(name, elevation, ground=None if depth is None else elevation + depth,
                    surcharge_depth=surcharge, initial_depth=initial)
        )
    all_curves = _curves(sections.get("CURVES", []), path)
    tides = {name: pts for name, (kind, pts) in all_curves.items() if kind == "TIDAL"}
    curves = {name: pts for name, (kind, pts) in all_curves.items() if kind == "STORAGE"}
    ratings = {name: pts for name, (kind, pts) in all_curves.items() if kind == "RATING"}
    if sections.get("STORAGE") and not dynamic:
        raise ValueError(
            f"{path}: section [STORAGE] (line {sections['STORAGE'][0].number}): storage units "
            f"are read under dynamic-wave routing only (FLOW_ROUTING DYNWAVE); under "
            f"{routing} SWMM routes them by their volume, which is not modelled")
    for line in sections.get("STORAGE", []):
        manhole = _storage_unit(line, curves, path)
        invert[manhole.name] = manhole.invert
        manholes.append(manhole)
    outfalls: list[Outfall] = []
    for line in sections.get("OUTFALLS", []):
        require_fields(line, 2, "an outfall", path)
        name = line.fields[0]
        elevation = as_float(line, 1, "the invert elevation", path)
        invert[name] = elevation
        outfalls.append(_outfall(line, name, elevation, dynamic, tides, all_curves, path))
    _check_timeseries(sections.get("TIMESERIES", []), outfalls, path)

    shapes: dict[str, float] = {}
    barrel_count: dict[str, int] = {}
    gated: set[str] = set()
    conduit_names = {line.fields[0] for line in sections.get("CONDUITS", [])}
    for line in sections.get("LOSSES", []):
        require_fields(line, 4, "a loss entry", path)
        if line.fields[0] not in conduit_names:
            raise ValueError(
                f"{path}: line {line.number}: [LOSSES] names link {line.fields[0]!r}, "
                f"which is not a conduit in [CONDUITS]"
            )
        for i, what in ((1, "entry loss coefficient"), (2, "exit loss coefficient"),
                        (3, "average loss coefficient")):
            if as_float(line, i, f"the {what}", path) != 0.0:
                raise ValueError(
                    f"{path}: line {line.number}: link {line.fields[0]!r} has a non-zero "
                    f"{what} {line.fields[i]}; local losses are not modelled"
                )
        if len(line.fields) > 5 and as_float(line, 5, "the seepage rate", path) != 0.0:
            raise ValueError(
                f"{path}: line {line.number}: link {line.fields[0]!r} has seepage rate "
                f"{line.fields[5]}; seepage is not modelled"
            )
        if len(line.fields) > 4:
            flap = line.fields[4].upper()
            if flap not in ("YES", "NO"):
                raise ValueError(
                    f"{path}: line {line.number}: link {line.fields[0]!r} flap gate must be "
                    f"YES or NO, got {line.fields[4]!r}"
                )
            if flap == "YES":
                gated.add(line.fields[0])
    regulator_names = {line.fields[0] for key in _REGULATORS for line in sections.get(key, [])}
    regulator_xsect: dict[str, object] = {}
    for line in sections.get("XSECTIONS", []):
        require_fields(line, 3, "a cross-section", path)
        if line.fields[0] in regulator_names:
            regulator_xsect[line.fields[0]] = line
            continue
        if line.fields[1].upper() != "CIRCULAR":
            raise ValueError(
                f"{path}: line {line.number}: only CIRCULAR cross-sections are modelled, "
                f"got {line.fields[1]!r} for link {line.fields[0]!r}"
            )
        shapes[line.fields[0]] = as_float(line, 2, "the diameter", path)
        barrels = as_float(line, 6, "the barrels", path) if len(line.fields) > 6 else 1.0
        if barrels != int(barrels) or not 1 <= barrels <= 127:             # link.c:250-256
            raise ValueError(
                f"{path}: line {line.number}: link {line.fields[0]!r} has {line.fields[6]} "
                f"barrels; SWMM takes a whole number from 1 to 127"
            )
        if barrels != 1.0 and not dynamic:
            raise ValueError(
                f"{path}: line {line.number}: link {line.fields[0]!r} has {line.fields[6]} "
                f"barrels; only dynamic-wave routing models more than one"
            )
        barrel_count[line.fields[0]] = int(barrels)
        if len(line.fields) > 7 and as_float(line, 7, "the culvert code", path) != 0.0:
            raise ValueError(
                f"{path}: line {line.number}: link {line.fields[0]!r} has culvert code "
                f"{line.fields[7]}; culvert inlet control is not modelled"
            )

    pipes: list[Pipe] = []
    for line in sections.get("CONDUITS", []):
        require_fields(line, 5, "a conduit", path)
        name, node1, node2 = line.fields[0], line.fields[1], line.fields[2]
        length = as_float(line, 3, "the length", path)
        roughness = as_float(line, 4, "the Manning n", path)
        in_offset = as_float(line, 5, "the inlet offset", path) if len(line.fields) > 5 else 0.0
        out_offset = (
            as_float(line, 6, "the outlet offset", path) if len(line.fields) > 6 else 0.0
        )
        init_flow = as_float(line, 7, "the initial flow", path) if len(line.fields) > 7 else 0.0
        max_flow = as_float(line, 8, "the maximum flow", path) if len(line.fields) > 8 else 0.0
        if max_flow < 0.0:
            max_flow = 0.0              # dwflow.c:270 only acts on a positive limit
        for end in (node1, node2):
            if end not in invert:
                raise ValueError(
                    f"{path}: line {line.number}: conduit {name!r} names node {end!r}, "
                    f"which is in neither [JUNCTIONS] nor [OUTFALLS]"
                )
        if offsets == "DEPTH":
            z1, z2 = invert[node1] + in_offset, invert[node2] + out_offset
        else:
            z1, z2 = in_offset, out_offset
            # `Pipe` offsets are heights above the end node's invert in either mode.
            in_offset, out_offset = z1 - invert[node1], z2 - invert[node2]
        if name not in shapes:
            raise ValueError(
                f"{path}: line {line.number}: conduit {name!r} has no [XSECTIONS] entry"
            )
        if dynamic:
            # link.c `conduit_getSlope`: floored fall, chord slope, MIN_SLOPE floor; reverse
            # only an adverse conduit (z1 < z2), so a flat one keeps the file's orientation.
            delta = max(abs(z1 - z2), MIN_DELTA_Z)
            slope = delta / length if delta >= length else delta / math.sqrt(length**2 - delta**2)
            slope = max(slope, min_slope)
            reverse = z1 < z2
            if reverse:
                node1, node2 = node2, node1
                in_offset, out_offset = out_offset, in_offset
            pipes.append(Pipe(
                name, node1, node2, length, shapes[name], roughness, slope,
                inlet_offset=in_offset, outlet_offset=out_offset,
                barrels=barrel_count[name], max_flow=max_flow or None,
                initial_flow=-init_flow if reverse else init_flow,         # link.c:1190
                flap_gate=name in gated, flap_gate_reversed=reverse))
            continue
        upstream, downstream, high, low = (
            (node1, node2, z1, z2) if z1 > z2 else (node2, node1, z2, z1)
        )
        if z1 <= z2:
            in_offset, out_offset = out_offset, in_offset
        fall = high - low
        if fall <= 0:
            raise ValueError(
                f"{path}: line {line.number}: conduit {name!r} has zero or adverse fall "
                f"({z1} -> {z2}); a gravity sewer pipe must fall"
            )
        if fall >= length:
            raise ValueError(
                f"{path}: line {line.number}: conduit {name!r} falls {fall} m over a "
                f"length of {length} m; the 3-D chord is undefined"
            )
        slope = fall / (length**2 - fall**2) ** 0.5
        pipes.append(
            Pipe(name, upstream, downstream, length, shapes[name], roughness, slope,
                 inlet_offset=in_offset, outlet_offset=out_offset,
                 barrels=barrel_count[name], max_flow=max_flow or None,
                 initial_flow=init_flow if z1 > z2 else -init_flow,
                 flap_gate=name in gated)
        )

    regulators = _regulators(sections, regulator_xsect, ratings, invert, dynamic, offsets,
                             routing, path)

    pollutants: dict[str, dict] = {}
    for line in sections.get("POLLUTANTS", []):
        require_fields(line, 6, "a pollutant", path)
        units = line.fields[1].upper()
        if units not in ("MG/L", "UG/L"):
            raise ValueError(
                f"{path}: line {line.number}: pollutant units must be MG/L or UG/L, got "
                f"{units!r}"
            )
        # Cinit (field 10) is the concentration of the water a run starts with; the model
        # starts clean, so a nonzero one is refused. The other skipped fields only feed what
        # the reader refuses anyway: Crain (rain), Cgw (groundwater), Crdii ([RDII]), SnowOnly,
        # CoPollutant/CoFraction (runoff washoff) and Cdwf ([DWF]).
        if len(line.fields) > 10 and as_float(line, 10, "the initial concentration",
                                              path) != 0.0:
            raise ValueError(
                f"{path}: line {line.number}: pollutant {line.fields[0]!r} has a nonzero "
                f"Cinit; the model starts with clean water, set it to 0")
        pollutants[line.fields[0]] = {
            "units": units,
            "scale": 1e-3 if units == "MG/L" else 1e-6,
            "decay": as_float(line, 5, "the decay coefficient", path) / 86400.0,
        }

    inflow = {m.name: 0.0 for m in manholes}
    loads: dict[str, dict[str, float]] = {}
    for line in sections.get("INFLOWS", []):
        require_fields(line, 4, "an inflow", path)
        node, constituent, series = line.fields[0], line.fields[1], line.fields[2]
        if series not in ('""', "''", "*"):
            raise ValueError(
                f"{path}: line {line.number}: inflow at {node!r} names the time series "
                f"{series!r}; only constant baselines are read, time variation belongs in "
                f"the drivers"
            )
        kind = line.fields[3].upper()
        baseline = as_float(line, 6, "the baseline", path) if len(line.fields) > 6 else 0.0
        if constituent.upper() == "FLOW":
            if node not in inflow:
                raise ValueError(
                    f"{path}: line {line.number}: inflow at {node!r}, which is not a "
                    f"junction"
                )
            inflow[node] = baseline
        else:
            if constituent not in pollutants:
                raise ValueError(
                    f"{path}: line {line.number}: inflow names pollutant "
                    f"{constituent!r}, which has no [POLLUTANTS] entry"
                )
            if kind != "CONCENTRATION":
                raise ValueError(
                    f"{path}: line {line.number}: a pollutant inflow must be of type "
                    f"CONCENTRATION, got {kind!r}"
                )
            loads.setdefault(node, {})[constituent] = (
                baseline * pollutants[constituent]["scale"]
            )

    manholes = [
        dataclasses.replace(m, inflow=inflow[m.name])
        for m in manholes
    ]
    network = SewerNetwork(
        manholes=tuple(manholes), pipes=tuple(pipes), outfalls=tuple(outfalls),
        routing=routing, geometry=geometry, dynamic_wave=dynamic_wave,
        orifices=regulators["ORIFICES"], weirs=regulators["WEIRS"],
        outlets=regulators["OUTLETS"],
    )
    network.validate()
    return network, loads, pollutants


def _curves(lines, path) -> dict[str, tuple[str, tuple[tuple, tuple]]]:
    """`[CURVES]`: ``{name: (type, (xs, ys))}`` of the STORAGE, RATING and TIDAL curves, each
    curve's points in file order over any number of lines, its type on its first line
    (`table.c:67-108`). A curve of another type is refused (no object that uses one is
    read), as are an unpaired value and x values that do not strictly increase
    (`table.c:313`)."""
    points: dict[str, list[tuple[float, float]]] = {}
    kinds: dict[str, str] = {}
    for line in lines:
        name = line.fields[0]
        start = 1
        if name not in points:
            kind = line.fields[1].upper() if len(line.fields) > 1 else ""
            if kind not in _CURVE_TYPES:
                raise ValueError(
                    f"{path}: line {line.number}: curve {name!r} needs its type (one of "
                    f"{_CURVE_TYPES}) on its first line")
            if kind not in ("STORAGE", "RATING", "TIDAL"):
                raise ValueError(
                    f"{path}: line {line.number}: curve {name!r} is of type {kind}; only "
                    f"STORAGE, RATING and TIDAL curves are read (no object that uses "
                    f"another type is modelled)")
            points[name] = []
            kinds[name] = kind
            start = 2
        if (len(line.fields) - start) % 2:
            raise ValueError(f"{path}: line {line.number}: curve {name!r} has an unpaired "
                             f"value; give depth-area pairs")
        for i in range(start, len(line.fields), 2):
            x = as_float(line, i, "a curve x value", path)
            y = as_float(line, i + 1, "a curve y value", path)
            if points[name] and x <= points[name][-1][0]:
                raise ValueError(
                    f"{path}: line {line.number}: curve {name!r} x values (depths, heads) "
                    f"must be strictly increasing (SWMM's ERR_CURVE_SEQUENCE), got {x} "
                    f"after {points[name][-1][0]}")
            points[name].append((x, y))
    return {name: (kinds[name], (tuple(p[0] for p in pts), tuple(p[1] for p in pts)))
            for name, pts in points.items()}


def _yes_no(line, i: int, what: str, path) -> bool:
    """A YES/NO field (SWMM's `NoYesWords`)."""
    word = line.fields[i].upper()
    if word not in ("YES", "NO"):
        raise ValueError(f"{path}: line {line.number}: {what} must be YES or NO, got "
                         f"{line.fields[i]!r}")
    return word == "YES"


def _given(line, i: int) -> bool:
    """Field `i` is present and not SWMM's ``*`` (keep the default)."""
    return len(line.fields) > i and line.fields[i] != "*"


def _opening(line, name: str, shapes: tuple[str, ...], path) -> tuple[str, list[float]]:
    """A regulator's `[XSECTIONS]` line: its shape (one of `shapes`, SWMM's
    ERR_REGULATOR_SHAPE otherwise) and Geom1..Geom4 (m; absent values 0)."""
    if line is None:
        raise ValueError(f"{path}: regulator {name!r} has no [XSECTIONS] entry (its opening)")
    shape = line.fields[1].upper()
    if shape not in shapes:
        raise ValueError(f"{path}: line {line.number}: regulator {name!r} has opening shape "
                         f"{line.fields[1]!r}; it takes {' or '.join(shapes)} "
                         f"(SWMM's ERR_REGULATOR_SHAPE)")
    geoms = [as_float(line, i, f"Geom{i - 1}", path) if len(line.fields) > i else 0.0
             for i in (2, 3, 4, 5)]
    return shape, geoms


def _regulators(sections, xsect: dict, ratings: dict, invert: dict, dynamic: bool,
                offsets: str, routing: str, path) -> dict[str, tuple]:
    """`[ORIFICES]`, `[WEIRS]` and `[OUTLETS]` (link.c:1641-1692, 2010-2099, 2524-2604), read
    under DYNWAVE only, with their `[XSECTIONS]` openings and the RATING curves of tabular
    outlets. Refused by name, with the line: a roadway weir, a weir coefficient curve, an
    [XSECTIONS] entry for an outlet, a negative orifice or weir crest height, unknown nodes
    and keywords, and LINK_OFFSETS ELEVATION (SWMM then sets each end's offset from its own
    node, which is not read)."""
    found: dict[str, list] = {key: [] for key in _REGULATORS}
    for key in _REGULATORS:
        lines = sections.get(key, [])
        if not lines:
            continue
        if not dynamic:
            raise ValueError(
                f"{path}: section [{key}] (line {lines[0].number}): flow regulators are read "
                f"under dynamic-wave routing only (FLOW_ROUTING DYNWAVE), not {routing}")
        if offsets != "DEPTH":
            raise ValueError(
                f"{path}: section [{key}] (line {lines[0].number}): regulators are read with "
                f"LINK_OFFSETS DEPTH only")
        for line in lines:
            require_fields(line, 6, f"a {key[:-1].lower()} entry", path)
            name, u, v = line.fields[0], line.fields[1], line.fields[2]
            where = f"{path}: line {line.number}: {key[:-1].lower()} {name!r}"
            for end in (u, v):
                if end not in invert:
                    raise ValueError(f"{where} names node {end!r}, which is not a junction, "
                                     f"storage unit or outfall")
            found[key].append(_REGULATOR_READERS[key](line, name, u, v, where, xsect,
                                                      ratings, path))
    stray = sorted(set(xsect) - {line.fields[0] for key in ("ORIFICES", "WEIRS")
                                 for line in sections.get(key, [])})
    if stray:
        line = xsect[stray[0]]
        raise ValueError(f"{path}: line {line.number}: outlet {stray[0]!r} has an [XSECTIONS] "
                         f"entry; an outlet has no opening (SWMM's DUMMY section)")
    return {key: tuple(devices) for key, devices in found.items()}


def _orifice(line, name, u, v, where, xsect, ratings, path) -> Orifice:
    """``Name N1 N2 SIDE|BOTTOM CrestHt Cd [Gated] [CloseTime]`` (link.c:1641-1692); the
    opening is CIRCULAR or RECT_CLOSED. CloseTime only shapes control actions, which are
    not read, so it is accepted and has no effect."""
    orientation = line.fields[3].upper()
    if orientation not in ("SIDE", "BOTTOM"):
        raise ValueError(f"{where}: type must be SIDE or BOTTOM, got {line.fields[3]!r}")
    crest = as_float(line, 4, "the crest height", path)
    if crest < 0.0:
        raise ValueError(f"{where}: negative crest height {crest} m (SWMM raises it to 0 at "
                         f"the inlet only; not modelled)")
    cd = as_float(line, 5, "the discharge coefficient", path)
    if cd < 0.0:                                                             # link.c:1675
        raise ValueError(f"{where}: negative discharge coefficient {cd}")
    gated = _yes_no(line, 6, "the flap gate", path) if len(line.fields) > 6 else False
    if len(line.fields) > 7 and as_float(line, 7, "the close time", path) < 0.0:
        raise ValueError(f"{where}: negative close time {line.fields[7]}")    # link.c:1688
    shape, geoms = _opening(xsect.get(name), name, ("CIRCULAR", "RECT_CLOSED"), path)
    return Orifice(name, u, v, orientation.lower(),
                   "circular" if shape == "CIRCULAR" else "rectangular", geoms[0], crest, cd,
                   width=None if shape == "CIRCULAR" else geoms[1], flap_gate=gated)


def _weir(line, name, u, v, where, xsect, ratings, path) -> Weir:
    """``Name N1 N2 Type CrestHt Cd [Gated] [EndCon] [Cd2] [CanSurcharge] [RoadWidth
    RoadSurf] [CdCurve]`` (link.c:2010-2099; ``*`` keeps a default). ROADWAY weirs and a
    coefficient curve are refused by name; the opening is RECT_OPEN, TRIANGULAR or
    TRAPEZOIDAL by type (link.c:2111-2133)."""
    if line.fields[3].upper() == "ROADWAY":
        raise ValueError(f"{where}: ROADWAY weirs are not modelled")
    kind = _WEIR_KINDS.get(line.fields[3].upper())
    if kind is None:
        raise ValueError(f"{where}: type must be one of {sorted(_WEIR_KINDS)}, got "
                         f"{line.fields[3]!r}")
    crest = as_float(line, 4, "the crest height", path)
    if crest < 0.0:
        raise ValueError(f"{where}: negative crest height {crest} m (SWMM raises it to 0 at "
                         f"the inlet only; not modelled)")
    cd = as_float(line, 5, "the discharge coefficient", path)
    gated = _yes_no(line, 6, "the flap gate", path) if _given(line, 6) else False
    end_con = as_float(line, 7, "the end contractions", path) if _given(line, 7) else 0.0
    cd2 = as_float(line, 8, "the end discharge coefficient", path) if _given(line, 8) else 0.0
    surcharge = _yes_no(line, 9, "CanSurcharge", path) if _given(line, 9) else True
    for what, value in (("discharge coefficient", cd), ("end contractions", end_con),
                        ("end discharge coefficient", cd2)):
        if value < 0.0:                                                      # link.c:2040-2062
            raise ValueError(f"{where}: negative {what} {value}")
    if _given(line, 12):
        raise ValueError(f"{where}: names the coefficient curve {line.fields[12]!r}; weir "
                         f"coefficient curves are not modelled")
    shape, geoms = _opening(xsect.get(name), name, (_WEIR_XSECT[kind],), path)
    return Weir(name, u, v, kind, geoms[0], geoms[1], crest, cd,
                side_slopes=(geoms[2], geoms[3]) if shape == "TRAPEZOIDAL" else (0.0, 0.0),
                end_contractions=end_con, end_discharge_coefficient=cd2,
                can_surcharge=surcharge, flap_gate=gated)


def _outlet(line, name, u, v, where, xsect, ratings, path) -> Outlet:
    """``Name N1 N2 Offset TABULAR/DEPTH|TABULAR/HEAD Curve [Gated]`` or ``... FUNCTIONAL/DEPTH
    |FUNCTIONAL/HEAD C1 C2 [Gated]`` (link.c:2524-2604): a negative offset is 0 (link.c:2556),
    the curve a RATING curve of [CURVES]."""
    crest = max(as_float(line, 3, "the crest height", path), 0.0)           # link.c:2556
    relation, _, qualifier = line.fields[4].upper().partition("/")
    if relation not in ("TABULAR", "FUNCTIONAL") or qualifier not in ("", "DEPTH", "HEAD"):
        raise ValueError(f"{where}: type must be TABULAR/DEPTH, TABULAR/HEAD, "
                         f"FUNCTIONAL/DEPTH or FUNCTIONAL/HEAD, got {line.fields[4]!r}")
    if relation == "FUNCTIONAL":
        require_fields(line, 7, "a functional outlet", path)
        rating: tuple = ("functional", as_float(line, 5, "the coefficient", path),
                         as_float(line, 6, "the exponent", path))
        n = 7
    else:
        label = line.fields[5]
        if label not in ratings:
            raise ValueError(f"{where}: names curve {label!r}, which is not a RATING curve in "
                             f"[CURVES]")
        rating = ("tabular", *ratings[label])
        n = 6
    gated = _yes_no(line, n, "the flap gate", path) if len(line.fields) > n else False
    return Outlet(name, u, v, crest, rating, head_difference=qualifier == "HEAD",
                  flap_gate=gated)


_REGULATOR_READERS = {"ORIFICES": _orifice, "WEIRS": _weir, "OUTLETS": _outlet}


def _storage_unit(line, curves: dict, path) -> Manhole:
    """A `[STORAGE]` line, ``Name Elev MaxDepth InitDepth Shape ... [SurDepth [Fevap
    [exfiltration]]]`` (node.c:654-809): TABULAR names a curve, the other shapes take three
    values (FUNCTIONAL a1 a2 a0; CYLINDRICAL L W -; CONICAL L W Z; PARABOLOID L W H;
    PYRAMIDAL L W Z). Evaporation and exfiltration are refused by name."""
    require_fields(line, 6, "a storage unit", path)
    name = line.fields[0]
    where = f"{path}: line {line.number}: storage unit {name!r}"
    elevation = as_float(line, 1, "the invert elevation", path)
    max_depth = as_float(line, 2, "the maximum depth", path)
    initial = as_float(line, 3, "the initial depth", path)
    for what, value in (("maximum", max_depth), ("initial", initial)):
        if value < 0.0:
            raise ValueError(f"{where} has a negative {what} depth {value} m")
    shape = line.fields[4].upper()
    if shape not in _STORAGE_SHAPES:
        raise ValueError(f"{where} has shape {line.fields[4]!r}; SWMM's shapes are "
                         f"{_STORAGE_SHAPES}")
    if shape == "TABULAR":
        label = line.fields[5]
        if label not in curves:
            raise ValueError(f"{where} names curve {label!r}, which is not a STORAGE curve "
                             f"in [CURVES]")
        curve: tuple = ("tabular", *curves[label])
        n = 6
    else:
        if len(line.fields) < 8:                                              # node.c:710
            raise ValueError(f"{where}: {shape} needs three values after it")
        a, b, c = (as_float(line, i, f"a {shape} value", path) for i in (5, 6, 7))
        if shape == "CYLINDRICAL" and c < 0.0:                                # node.c:734
            raise ValueError(f"{where}: CYLINDRICAL third value must be >= 0, got {c}")
        curve = {"FUNCTIONAL": ("functional", a, b, c),
                 "CYLINDRICAL": ("cylindrical", a, b)}.get(shape, (shape.lower(), a, b, c))
        n = 8
    surcharge = as_float(line, n, "the surcharge depth", path) if len(line.fields) > n else 0.0
    if surcharge < 0.0:
        raise ValueError(f"{where} has a negative surcharge depth {surcharge} m")
    if len(line.fields) > n + 1 and as_float(line, n + 1, "the evaporation factor",
                                             path) != 0.0:
        raise ValueError(f"{where} has evaporation factor {line.fields[n + 1]}; storage "
                         f"evaporation is not modelled (set Fevap to 0)")
    extra = len(line.fields) - (n + 2)
    if extra > 0:
        # exfil.c:48-66: one value is Ksat, else suction head, Ksat, IMDmax; Ksat 0 means
        # no exfiltration object at all
        if extra == 2 or extra > 3:
            raise ValueError(f"{where} has {extra} exfiltration values; SWMM takes Ksat "
                             f"alone or suction head, Ksat and IMDmax")
        at = n + 2 if extra == 1 else n + 3
        if as_float(line, at, "the exfiltration Ksat", path) != 0.0:
            raise ValueError(f"{where} has exfiltration Ksat {line.fields[at]}; storage "
                             f"seepage (exfiltration) is not modelled")
    try:
        validate_storage_curve(name, curve)
    except ValueError as err:
        raise ValueError(f"{path}: line {line.number}: {err}") from err
    return Manhole(name, elevation, ground=elevation + max_depth, surcharge_depth=surcharge,
                   initial_depth=initial, storage=curve)


def _dynamic_wave_options(opts: dict, path) -> DynamicWaveOptions:
    """The DYNWAVE `[OPTIONS]`, with their refusals (each names its line)."""

    def refuse(key: str, why: str) -> None:
        line = opts[key][0]
        raise ValueError(f"{path}: line {line.number}: {key} {line.fields[1]} {why}")

    if "VARIABLE_STEP" in opts and not 0.0 <= opts["VARIABLE_STEP"][1] <= 2.0:
        refuse("VARIABLE_STEP", "-- the Courant factor must be between 0 and 2")
    if "MINIMUM_STEP" in opts and opts["MINIMUM_STEP"][1] < 0.0:
        refuse("MINIMUM_STEP", "-- must not be negative")
    if "LENGTHENING_STEP" in opts and opts["LENGTHENING_STEP"][1] < 0.0:
        refuse("LENGTHENING_STEP", "-- must not be negative")
    if opts.get("ALLOW_PONDING", (None, "NO"))[1] not in ("NO", "FALSE"):
        refuse("ALLOW_PONDING", "-- ponding is not modelled; set ALLOW_PONDING NO")
    kwargs: dict[str, object] = {}
    # SWMM lowers the routing step to the wet step when larger (project.c:246-250)
    routing_step = opts["ROUTING_STEP"][1] if "ROUTING_STEP" in opts else (
        DynamicWaveOptions().routing_step)
    kwargs["routing_step"] = min(routing_step, opts["WET_STEP"][1] if "WET_STEP" in opts
                                 else DEFAULT_WET_STEP)
    # conduits are validated (and lengthened) with the file's ROUTING_STEP, before SWMM
    # lowers it to WET_STEP (project.c:238-250)
    kwargs["lengthening_routing_step"] = routing_step
    if "LENGTHENING_STEP" in opts:
        kwargs["lengthening_step"] = opts["LENGTHENING_STEP"][1]
    for key, table, field_name in (
        ("INERTIAL_DAMPING", _DAMPING, "inertial_damping"),
        ("NORMAL_FLOW_LIMITED", _NORMAL_FLOW, "normal_flow_limited"),
        ("SURCHARGE_METHOD", {"EXTRAN": "extran", "SLOT": "slot"}, "surcharge_method"),
    ):
        if key in opts:
            if opts[key][1] not in table:
                refuse(key, f"-- must be one of {sorted(table)}")
            kwargs[field_name] = table[opts[key][1]]
    # an absent VARIABLE_STEP is SWMM's 0.75 (project.c:854), not the dataclass's 0
    kwargs["variable_step"] = opts["VARIABLE_STEP"][1] if "VARIABLE_STEP" in opts else 0.75
    if "MINIMUM_STEP" in opts:
        kwargs["minimum_step"] = opts["MINIMUM_STEP"][1]
    for key, field_name in (("MIN_SURFAREA", "min_surface_area"),
                            ("HEAD_TOLERANCE", "head_tolerance")):
        if key in opts and opts[key][1] != 0.0:
            kwargs[field_name] = opts[key][1]
    if "MAX_TRIALS" in opts and opts["MAX_TRIALS"][1] != 0.0:
        kwargs["max_trials"] = int(opts["MAX_TRIALS"][1])
    options = DynamicWaveOptions(**kwargs)
    options.validate()
    return options


def _outfall(line, name: str, elevation: float, dynamic: bool, tides: dict,
             all_curves: dict, path) -> Outfall:
    """An `[OUTFALLS]` line: type (field 2, default FREE), the type's data (field 3: a FIXED
    stage, a TIDAL curve name or a TIMESERIES name; node.c:1333-1380) and the gate flag,
    which follows the type data."""
    kind = line.fields[2].upper() if len(line.fields) > 2 else "FREE"
    if kind not in ("FREE", "NORMAL", "FIXED", "TIDAL", "TIMESERIES"):
        if dynamic:
            raise ValueError(
                f"{path}: line {line.number}: outfall {name!r} is of type {kind}; only FREE, "
                f"NORMAL, FIXED, TIDAL and TIMESERIES outfalls are modelled"
            )
        return Outfall(name, elevation)
    stage = tide = series = None
    gated_at = 3
    if kind == "FIXED":
        if len(line.fields) < 4:
            raise ValueError(
                f"{path}: line {line.number}: FIXED outfall {name!r} needs a stage"
            )
        stage = as_float(line, 3, "the fixed stage", path)
        gated_at = 4
    elif kind in ("TIDAL", "TIMESERIES") and dynamic:
        what = "a tide curve" if kind == "TIDAL" else "a time series"
        if len(line.fields) < 4:
            raise ValueError(
                f"{path}: line {line.number}: {kind} outfall {name!r} needs {what} name"
            )
        label = line.fields[3]
        gated_at = 4
        if kind == "TIDAL":
            if label not in tides:
                found = all_curves[label][0] if label in all_curves else None
                raise ValueError(
                    f"{path}: line {line.number}: TIDAL outfall {name!r} names curve "
                    f"{label!r}, which "
                    + (f"is of type {found}, not TIDAL" if found
                       else "has no [CURVES] entry of type TIDAL")
                )
            tide = tuple(zip(*tides[label], strict=True))
        else:
            series = label
    elif kind in ("TIDAL", "TIMESERIES"):
        return Outfall(name, elevation)
    flap = len(line.fields) > gated_at and line.fields[gated_at].upper() == "YES"
    return Outfall(name, elevation, kind=kind.lower(), stage=stage, flap_gate=flap,
                   tide=tide, series=series)


def _check_timeseries(lines, outfalls: list, path) -> None:
    """A `[TIMESERIES]` section holds only series that TIMESERIES outfalls name (their data
    is the `outfall_stage` driver's, not read here); any other series is refused."""
    named = {o.series for o in outfalls if o.series is not None}
    for line in lines:
        if line.fields[0] not in named:
            raise ValueError(
                f"{path}: section [TIMESERIES] (line {line.number}) names series "
                f"{line.fields[0]!r}, which no TIMESERIES outfall uses; this reader models "
                f"gravity conduits, junctions, outfalls and constant inflows only (time "
                f"variation belongs in the drivers)"
            )
