"""A documented SWMM 5 `.inp` subset (spec 4.5).

READ: `[TITLE]`, `[OPTIONS]` (`FLOW_UNITS` must be `CMS`; `FLOW_ROUTING` is recorded),
`[JUNCTIONS]`, `[OUTFALLS]`, `[CONDUITS]`, `[XSECTIONS]` (only `CIRCULAR`), `[INFLOWS]`
(constant `FLOW` baselines and constant `CONCENTRATION` pollutant baselines),
`[POLLUTANTS]`. IGNORED because no conduit or node ever references them: `[EVAPORATION]`,
`[REPORT]`, `[TAGS]`, `[MAP]`, `[COORDINATES]`.

REFUSED BY NAME, because skipping them would silently change the physics: `[DWF]`,
`[STORAGE]` with a non-constant area, `[PUMPS]`, `[WEIRS]`, `[ORIFICES]`, `[OUTLETS]`,
`[DIVIDERS]`, `[SUBCATCHMENTS]`, `[RAINGAGES]`, `[CONTROLS]`, `[CURVES]`, `[TIMESERIES]`,
`[LID_USAGE]`, and any other section carrying content. A time-series inflow is refused;
time variation belongs in the drivers.

Slope. SWMM defines `S0 = dy / dx` with `dx = sqrt(L^2 - dy^2)` (Ref. Man. Vol. II
Eq. 2-1/2-2), so the reader uses the 3-D chord, not `dy / L`. A conduit is normalised
upstream to downstream from the two end elevations (node invert plus that end's offset),
and one with zero or adverse fall is refused by name.
"""

from __future__ import annotations

from pathlib import Path

from noodl.apps.inpfile import as_float, read_sections, require_fields
from noodl.apps.sewer.network import Manhole, Outfall, Pipe, SewerNetwork

_IGNORED = {
    "TITLE", "EVAPORATION", "REPORT", "TAGS", "MAP", "COORDINATES", "VERTICES",
    "POLYGONS", "SYMBOLS", "LABELS", "BACKDROP", "PROFILES", "FILES", "END",
}
_READ = {"OPTIONS", "JUNCTIONS", "OUTFALLS", "CONDUITS", "XSECTIONS", "INFLOWS",
         "POLLUTANTS"}


def read_swmm_inp(path) -> tuple[SewerNetwork, dict[str, dict[str, float]], dict[str, dict]]:
    """`(network, {node: {pollutant: kg/m3}}, {pollutant: {...}})` for `path`."""
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
    for line in sections.get("OPTIONS", []):
        key = line.fields[0].upper()
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

    invert: dict[str, float] = {}
    manholes: list[Manhole] = []
    for line in sections.get("JUNCTIONS", []):
        require_fields(line, 2, "a junction", path)
        name = line.fields[0]
        elevation = as_float(line, 1, "the invert elevation", path)
        depth = as_float(line, 2, "the maximum depth", path) if len(line.fields) > 2 else None
        invert[name] = elevation
        manholes.append(
            Manhole(name, elevation, ground=None if depth is None else elevation + depth)
        )
    outfalls: list[Outfall] = []
    for line in sections.get("OUTFALLS", []):
        require_fields(line, 2, "an outfall", path)
        name = line.fields[0]
        elevation = as_float(line, 1, "the invert elevation", path)
        invert[name] = elevation
        outfalls.append(Outfall(name, elevation))

    shapes: dict[str, float] = {}
    for line in sections.get("XSECTIONS", []):
        require_fields(line, 3, "a cross-section", path)
        if line.fields[1].upper() != "CIRCULAR":
            raise ValueError(
                f"{path}: line {line.number}: only CIRCULAR cross-sections are modelled, "
                f"got {line.fields[1]!r} for link {line.fields[0]!r}"
            )
        shapes[line.fields[0]] = as_float(line, 2, "the diameter", path)

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
        if name not in shapes:
            raise ValueError(
                f"{path}: line {line.number}: conduit {name!r} has no [XSECTIONS] entry"
            )
        upstream, downstream, high, low = (
            (node1, node2, z1, z2) if z1 > z2 else (node2, node1, z2, z1)
        )
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
            Pipe(name, upstream, downstream, length, shapes[name], roughness, slope)
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
        Manhole(m.name, m.invert, ground=m.ground, inflow=inflow[m.name])
        for m in manholes
    ]
    network = SewerNetwork(
        manholes=tuple(manholes), pipes=tuple(pipes), outfalls=tuple(outfalls),
        routing=routing,
    )
    network.validate()
    return network, loads, pollutants
