"""A documented SWMM 5 `.inp` subset.

READ: `[TITLE]`, `[OPTIONS]` (`FLOW_UNITS` must be `CMS`),
`[JUNCTIONS]` (invert, MaxDepth, SurDepth), `[OUTFALLS]` (FREE, NORMAL and FIXED),
`[CONDUITS]`, `[XSECTIONS]` (only `CIRCULAR`, one barrel), `[INFLOWS]` (constant `FLOW`
baselines and constant `CONCENTRATION` pollutant baselines), `[POLLUTANTS]`.
`FLOW_ROUTING` is STEADY, KINWAVE or DYNWAVE; DYNWAVE also reads `ROUTING_STEP`,
`INERTIAL_DAMPING`, `NORMAL_FLOW_LIMITED`, `SURCHARGE_METHOD`, `MIN_SURFAREA`,
`HEAD_TOLERANCE`, `MAX_TRIALS` and `MIN_SLOPE` into `DynamicWaveOptions`.
The network uses SWMM's own tabulated circular section
(``geometry="tabulated"``, `xsect_tables.py`) unless ``geometry="analytic"`` is passed.
IGNORED because no conduit or node ever references them: `[EVAPORATION]`, `[REPORT]`,
`[TAGS]`, `[MAP]`, `[COORDINATES]`.

REFUSED BY NAME, because skipping them would silently change the physics: `[DWF]`,
`[STORAGE]` with a non-constant area, `[PUMPS]`, `[WEIRS]`, `[ORIFICES]`, `[OUTLETS]`,
`[DIVIDERS]`, `[SUBCATCHMENTS]`, `[RAINGAGES]`, `[CONTROLS]`, `[CURVES]`, `[TIMESERIES]`,
`[LID_USAGE]`, and any other section carrying content. A time-series inflow is refused;
time variation belongs in the drivers. A conduit with more than one barrel or a culvert
code is refused for every routing.

DYNWAVE refuses, with the line number: `VARIABLE_STEP` > 0, `LENGTHENING_STEP` > 0,
`ALLOW_PONDING YES`, a junction `InitDepth`, a conduit `InitFlow` or
`MaxFlow`, a gated outfall and TIDAL/TIMESERIES outfalls (phase A: cold dry start, fixed step,
no ponding).

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
from noodl.apps.sewer.network import Manhole, Outfall, Pipe, SewerNetwork

#: link.c `MIN_DELTA_Z`: the smallest fall SWMM uses for a conduit slope, 0.001 ft in m.
MIN_DELTA_Z = 0.001 * 0.3048
_DYNWAVE_KEYS = (
    "ROUTING_STEP", "VARIABLE_STEP", "LENGTHENING_STEP", "MIN_SURFAREA", "HEAD_TOLERANCE",
    "MAX_TRIALS", "MIN_SLOPE", "INERTIAL_DAMPING", "NORMAL_FLOW_LIMITED", "SURCHARGE_METHOD",
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
_READ = {"OPTIONS", "JUNCTIONS", "OUTFALLS", "CONDUITS", "XSECTIONS", "INFLOWS",
         "POLLUTANTS"}


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
        if name in _READ or name in _IGNORED or not lines:
            continue
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
            if key in ("ROUTING_STEP", "VARIABLE_STEP", "LENGTHENING_STEP"):
                opts[key] = (line, _seconds(line, path))
            elif key in ("MIN_SURFAREA", "HEAD_TOLERANCE", "MAX_TRIALS", "MIN_SLOPE"):
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
        if (
            dynamic and len(line.fields) > 3
            and as_float(line, 3, "the initial depth", path) != 0.0
        ):
            raise ValueError(
                f"{path}: line {line.number}: junction {name!r} has a non-zero initial depth "
                f"{line.fields[3]}; dynamic wave starts from a dry network (phase A)"
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
                    surcharge_depth=surcharge)
        )
    outfalls: list[Outfall] = []
    for line in sections.get("OUTFALLS", []):
        require_fields(line, 2, "an outfall", path)
        name = line.fields[0]
        elevation = as_float(line, 1, "the invert elevation", path)
        invert[name] = elevation
        outfalls.append(_outfall(line, name, elevation, dynamic, path))

    shapes: dict[str, float] = {}
    for line in sections.get("XSECTIONS", []):
        require_fields(line, 3, "a cross-section", path)
        if line.fields[1].upper() != "CIRCULAR":
            raise ValueError(
                f"{path}: line {line.number}: only CIRCULAR cross-sections are modelled, "
                f"got {line.fields[1]!r} for link {line.fields[0]!r}"
            )
        shapes[line.fields[0]] = as_float(line, 2, "the diameter", path)
        barrels = as_float(line, 6, "the barrels", path) if len(line.fields) > 6 else 1.0
        if barrels != 1.0:
            raise ValueError(
                f"{path}: line {line.number}: link {line.fields[0]!r} has {line.fields[6]} "
                f"barrels; only single-barrel conduits are modelled"
            )
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
        if dynamic:
            for i, what in ((7, "initial flow"), (8, "maximum flow")):
                if len(line.fields) > i and as_float(line, i, f"the {what}", path) != 0.0:
                    raise ValueError(
                        f"{path}: line {line.number}: conduit {name!r} has a non-zero {what} "
                        f"{line.fields[i]}; dynamic wave starts from rest and does not cap "
                        f"flows (phase A)"
                    )
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
            if z1 < z2:
                node1, node2 = node2, node1
                in_offset, out_offset = out_offset, in_offset
            pipes.append(Pipe(name, node1, node2, length, shapes[name], roughness, slope,
                              inlet_offset=in_offset, outlet_offset=out_offset))
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
                 inlet_offset=in_offset, outlet_offset=out_offset)
        )

    pollutants: dict[str, dict] = {}
    for line in sections.get("POLLUTANTS", []):
        require_fields(line, 6, "a pollutant", path)
        units = line.fields[1].upper()
        if units not in ("MG/L", "UG/L"):
            raise ValueError(
                f"{path}: line {line.number}: pollutant units must be MG/L or UG/L, got "
                f"{units!r}"
            )
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
    )
    network.validate()
    return network, loads, pollutants


def _dynamic_wave_options(opts: dict, path) -> DynamicWaveOptions:
    """The DYNWAVE `[OPTIONS]`, with the phase-A refusals (each names its line)."""

    def refuse(key: str, why: str) -> None:
        line = opts[key][0]
        raise ValueError(f"{path}: line {line.number}: {key} {line.fields[1]} {why}")

    if opts.get("VARIABLE_STEP", (None, 0.0))[1] > 0:
        refuse("VARIABLE_STEP", "-- variable time stepping is not modelled yet; set "
                                "VARIABLE_STEP 0")
    if opts.get("LENGTHENING_STEP", (None, 0.0))[1] > 0:
        refuse("LENGTHENING_STEP", "-- conduit lengthening is not modelled yet; set "
                                   "LENGTHENING_STEP 0")
    if opts.get("ALLOW_PONDING", (None, "NO"))[1] not in ("NO", "FALSE"):
        refuse("ALLOW_PONDING", "-- ponding is not modelled yet; set ALLOW_PONDING NO")
    kwargs: dict[str, object] = {}
    if "ROUTING_STEP" in opts:
        kwargs["routing_step"] = opts["ROUTING_STEP"][1]
    for key, table, field_name in (
        ("INERTIAL_DAMPING", _DAMPING, "inertial_damping"),
        ("NORMAL_FLOW_LIMITED", _NORMAL_FLOW, "normal_flow_limited"),
        ("SURCHARGE_METHOD", {"EXTRAN": "extran", "SLOT": "slot"}, "surcharge_method"),
    ):
        if key in opts:
            if opts[key][1] not in table:
                refuse(key, f"-- must be one of {sorted(table)}")
            kwargs[field_name] = table[opts[key][1]]
    for key, field_name in (("MIN_SURFAREA", "min_surface_area"),
                            ("HEAD_TOLERANCE", "head_tolerance")):
        if key in opts and opts[key][1] != 0.0:
            kwargs[field_name] = opts[key][1]
    if "MAX_TRIALS" in opts and opts["MAX_TRIALS"][1] != 0.0:
        kwargs["max_trials"] = int(opts["MAX_TRIALS"][1])
    options = DynamicWaveOptions(**kwargs)
    options.validate()
    return options


def _outfall(line, name: str, elevation: float, dynamic: bool, path) -> Outfall:
    """An `[OUTFALLS]` line: type (field 2, default FREE), FIXED stage (field 3) and the gate
    flag, which follows the type data."""
    kind = line.fields[2].upper() if len(line.fields) > 2 else "FREE"
    if kind not in ("FREE", "NORMAL", "FIXED"):
        if dynamic:
            raise ValueError(
                f"{path}: line {line.number}: outfall {name!r} is of type {kind}; only FREE, "
                f"NORMAL and FIXED outfalls are modelled"
            )
        return Outfall(name, elevation)
    stage = None
    gated_at = 3
    if kind == "FIXED":
        if len(line.fields) < 4:
            raise ValueError(
                f"{path}: line {line.number}: FIXED outfall {name!r} needs a stage"
            )
        stage = as_float(line, 3, "the fixed stage", path)
        gated_at = 4
    if dynamic and len(line.fields) > gated_at and line.fields[gated_at].upper() == "YES":
        raise ValueError(
            f"{path}: line {line.number}: outfall {name!r} has a flap gate; gated outfalls "
            f"are not modelled yet"
        )
    return Outfall(name, elevation, kind=kind.lower(), stage=stage)
