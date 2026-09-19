"""A documented EPANET 2.2 `.inp` subset (spec 13.5).

READ: `[JUNCTIONS] [RESERVOIRS] [TANKS] [PIPES] [PUMPS] [VALVES] [DEMANDS] [PATTERNS]
[CURVES] [CONTROLS] [OPTIONS] [TIMES]`, with the column layouts of the manual's Appendix
C.2 as quoted verbatim in `.superpowers/epanet-research/epanet-research.md` section 9.

IGNORED because nothing in the hydraulics references them: `[TITLE]`, `[REPORT]` ("for the
Windows version of EPANET, the only [REPORT] option recognized is STATUS", Manual p.142),
`[COORDINATES]`, `[VERTICES]`, `[LABELS]`, `[BACKDROP]`, `[TAGS]`, `[ENERGY]`, `[END]`, and
the water-quality sections `[QUALITY] [SOURCES] [REACTIONS] [MIXING]` (this reader returns
hydraulics; a quality run is configured through `build_water_model(quality=...)`).

REFUSED BY NAME, because skipping them would silently change the network: `[RULES]`,
`[EMITTERS]` and `[STATUS]` with content, a time-based `[CONTROLS]` line, a PRV/PSV/PBV/GPV
valve, a constant-POWER pump, a pump with a SPEED PATTERN, a `[TANKS]` row with a volume
curve, a `HEADLOSS` that is not `H-W` or `D-W`, and any other section carrying content.

UNITS. EPANET computes internally in feet and cfs whatever the file says, converting at the
boundary; this reader converts to SI instead. The flow unit chosen in `[OPTIONS] Units`
also fixes the LENGTH units of every other field: `CFS GPM MGD IMGD AFD` are US (feet for
elevations, lengths, heads and tank levels, inches for diameters) and `LPS LPM MLD CMH CMD`
are SI (metres, millimetres) -- Manual p.136, "LPS/LPM/MLD/CMH/CMD => SI/metric for
everything else".

`[OPTIONS]` KEYS THAT CHANGE THE PHYSICS -- `DEMAND MODEL`, `MINIMUM PRESSURE`,
`REQUIRED PRESSURE`, `PRESSURE EXPONENT`, `SPECIFIC GRAVITY`, `VISCOSITY` -- are carried
onto `WaterNetwork.options` (a `WaterOptions`, EPANET's own defaults) rather than dropped;
`build_water_model` defaults its own `pda`/`p_min`/`p_req`/`exponent` arguments from them.
Any OTHER `[OPTIONS]` line (`TRIALS`, `ACCURACY`, `UNBALANCED`, the in-section `QUALITY`
mode, ...) is recorded verbatim in `notes["unrecognised_options"]` instead of being
silently ignored, since most of them are solver/report cosmetics this reader does not need
but a few (an unimplemented `QUALITY CHEMICAL`, say) would silently change the network if
dropped without a trace.
"""

from __future__ import annotations

from pathlib import Path

from noodl.apps.inpfile import as_float, read_sections, require_fields
from noodl.apps.water.network import (
    Control,
    Junction,
    Pump,
    Reservoir,
    Tank,
    Valve,
    WaterNetwork,
    WaterOptions,
    WaterPipe,
)

#: `.inp` flow units -> m3/s. GPM is one US gallon per minute; the rest follow the manual's
#: Appendix C.2.11 list. `GPM` is exactly `3.785411784e-3 / 60 = 6.309019640343977e-05`,
#: which reproduces wntr's own conversion of Net1's 1500 GPM design point to
#: 0.0946352946 m3/s (measured).
FLOW_UNITS = {
    "CFS": 0.028316846592,
    "GPM": 3.785411784e-3 / 60.0,
    "MGD": 3.785411784e3 / 86400.0,
    "IMGD": 4.54609e3 / 86400.0,
    "AFD": 1233.48183754752 / 86400.0,
    "LPS": 1e-3,
    "LPM": 1e-3 / 60.0,
    "MLD": 1e3 / 86400.0,
    "CMH": 1.0 / 3600.0,
    "CMD": 1.0 / 86400.0,
}
_US_UNITS = frozenset({"CFS", "GPM", "MGD", "IMGD", "AFD"})
FOOT = 0.3048
INCH = 0.0254
#: psi -> m of head, for a US-units file's `MINIMUM PRESSURE`/`REQUIRED PRESSURE` (`wntr`'s
#: own `HydParam.Pressure` conversion, `0.3048 / 0.4333`, confirmed to 1e-15 against
#: `wntr.epanet.util.to_si`). SI files state these two already in metres.
PSI_TO_M = FOOT / 0.4333

_IGNORED = frozenset(
    {
        "TITLE", "REPORT", "COORDINATES", "VERTICES", "LABELS", "BACKDROP", "TAGS",
        "ENERGY", "END", "QUALITY", "SOURCES", "REACTIONS", "MIXING",
    }
)
_READ = frozenset(
    {
        "JUNCTIONS", "RESERVOIRS", "TANKS", "PIPES", "PUMPS", "VALVES", "DEMANDS",
        "PATTERNS", "CURVES", "CONTROLS", "OPTIONS", "TIMES",
    }
)
_REFUSED = ("RULES", "EMITTERS", "STATUS")


def _time(text: str, path, line) -> float:
    """`HH:MM`, `HH:MM:SS` or a bare number of HOURS -> seconds (Appendix C.2.24)."""
    if ":" in text:
        parts = text.split(":")
        if len(parts) > 3:
            raise ValueError(
                f"{path}: line {line.number}: {text!r} is not a time of the form "
                f"HH:MM[:SS]"
            )
        try:
            values = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(
                f"{path}: line {line.number}: {text!r} is not a time of the form "
                f"HH:MM[:SS]"
            ) from exc
        while len(values) < 3:
            values.append(0.0)
        return values[0] * 3600.0 + values[1] * 60.0 + values[2]
    try:
        return float(text) * 3600.0
    except ValueError as exc:
        raise ValueError(
            f"{path}: line {line.number}: {text!r} is not a duration"
        ) from exc


def read_epanet_inp(path) -> WaterNetwork:
    """Read the documented subset of an EPANET 2.2 `.inp` into a `WaterNetwork`, in SI."""
    path = Path(path)
    sections = read_sections(path)
    for name in _REFUSED:
        if sections.get(name):
            raise ValueError(
                f"{path}: section [{name}] (line {sections[name][0].number}) carries "
                f"content; rule-based controls, emitters and explicit link statuses are "
                f"out of scope (spec 13.5)"
            )
    for name, lines in sections.items():
        if name in _READ or name in _IGNORED or not lines:
            continue
        raise ValueError(
            f"{path}: section [{name}] (line {lines[0].number}) is not part of the "
            f"documented subset and carries content"
        )

    flow_unit = "GPM"
    headloss = "H-W"
    demand_pattern = None
    demand_multiplier = 1.0
    demand_model = "DDA"
    minimum_pressure = 0.0
    required_pressure = 0.1
    pressure_exponent = 0.5
    specific_gravity = 1.0
    viscosity = 1.0
    unrecognised: list[str] = []
    for line in sections.get("OPTIONS", []):
        key = line.fields[0].upper()
        second = line.fields[1].upper() if len(line.fields) > 1 else ""
        if key == "UNITS":
            require_fields(line, 2, "UNITS", path)
            flow_unit = second
            if flow_unit not in FLOW_UNITS:
                raise ValueError(
                    f"{path}: line {line.number}: UNITS {flow_unit!r} is not one of "
                    f"{sorted(FLOW_UNITS)}"
                )
        elif key == "HEADLOSS":
            require_fields(line, 2, "HEADLOSS", path)
            headloss = line.fields[1].upper()
            if headloss not in ("H-W", "D-W"):
                raise ValueError(
                    f"{path}: line {line.number}: HEADLOSS {headloss!r}; only H-W and D-W "
                    f"are modelled (Chezy-Manning's SI constant is UNVERIFIED)"
                )
        elif key == "PATTERN":
            demand_pattern = line.fields[1] if len(line.fields) > 1 else None
        elif key == "DEMAND" and second == "MULTIPLIER":
            demand_multiplier = as_float(line, 2, "the demand multiplier", path)
        elif key == "DEMAND" and second == "MODEL":
            require_fields(line, 3, "DEMAND MODEL", path)
            demand_model = line.fields[2].upper()
            if demand_model not in ("DDA", "PDA"):
                raise ValueError(
                    f"{path}: line {line.number}: DEMAND MODEL {demand_model!r}; only "
                    f"DDA and PDA are modelled"
                )
        elif key == "MINIMUM" and second == "PRESSURE":
            minimum_pressure = as_float(line, 2, "MINIMUM PRESSURE", path)
        elif key == "REQUIRED" and second == "PRESSURE":
            required_pressure = as_float(line, 2, "REQUIRED PRESSURE", path)
        elif key == "PRESSURE" and second == "EXPONENT":
            pressure_exponent = as_float(line, 2, "PRESSURE EXPONENT", path)
        elif key == "SPECIFIC" and second == "GRAVITY":
            specific_gravity = as_float(line, 2, "SPECIFIC GRAVITY", path)
        elif key == "VISCOSITY":
            require_fields(line, 2, "VISCOSITY", path)
            viscosity = as_float(line, 1, "VISCOSITY", path)
        else:
            unrecognised.append(line.raw.strip())

    flow = FLOW_UNITS[flow_unit]
    us = flow_unit in _US_UNITS
    length_scale = FOOT if us else 1.0
    diameter_scale = INCH if us else 1e-3
    pressure_scale = PSI_TO_M if us else 1.0

    patterns: dict[str, tuple[float, ...]] = {}
    for line in sections.get("PATTERNS", []):
        require_fields(line, 2, "a pattern", path)
        values = tuple(
            as_float(line, i, "a pattern multiplier", path)
            for i in range(1, len(line.fields))
        )
        patterns[line.fields[0]] = patterns.get(line.fields[0], ()) + values

    curves: dict[str, list[tuple[float, float]]] = {}
    for line in sections.get("CURVES", []):
        require_fields(line, 3, "a curve point", path)
        curves.setdefault(line.fields[0], []).append(
            (
                as_float(line, 1, "the curve's x value", path),
                as_float(line, 2, "the curve's y value", path),
            )
        )

    junctions: list[Junction] = []
    for line in sections.get("JUNCTIONS", []):
        require_fields(line, 2, "a junction", path)
        demand = as_float(line, 2, "the demand", path) if len(line.fields) > 2 else 0.0
        pattern = line.fields[3] if len(line.fields) > 3 else None
        junctions.append(
            Junction(
                line.fields[0],
                as_float(line, 1, "the elevation", path) * length_scale,
                demand * flow * demand_multiplier,
                pattern,
            )
        )
    # `[DEMANDS]` OVERRIDES a junction's single `[JUNCTIONS]` demand (Appendix C.2.6:
    # "supplemental demands ... override the demand given in [JUNCTIONS]").
    overrides: dict[str, tuple[float, str | None]] = {}
    for line in sections.get("DEMANDS", []):
        require_fields(line, 2, "a demand", path)
        pattern = line.fields[2] if len(line.fields) > 2 else None
        overrides[line.fields[0]] = (
            as_float(line, 1, "the demand", path) * flow * demand_multiplier,
            pattern,
        )
    known = {j.name for j in junctions}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(
            f"{path}: [DEMANDS] names {unknown}, which are not junctions of this network"
        )
    if overrides:
        junctions = [
            Junction(j.name, j.elevation, *overrides[j.name]) if j.name in overrides else j
            for j in junctions
        ]

    reservoirs: list[Reservoir] = []
    for line in sections.get("RESERVOIRS", []):
        require_fields(line, 2, "a reservoir", path)
        reservoirs.append(
            Reservoir(
                line.fields[0],
                as_float(line, 1, "the head", path) * length_scale,
                line.fields[2] if len(line.fields) > 2 else None,
            )
        )

    tanks: list[Tank] = []
    for line in sections.get("TANKS", []):
        require_fields(line, 6, "a tank", path)
        if len(line.fields) > 7 and line.fields[7] not in ("", "*"):
            raise ValueError(
                f"{path}: line {line.number}: tank {line.fields[0]!r} has the volume curve "
                f"{line.fields[7]!r}; only cylindrical tanks are modelled (spec 13.5)"
            )
        diameter = as_float(line, 5, "the diameter", path) * length_scale
        if not diameter > 0:
            raise ValueError(
                f"{path}: line {line.number}: tank {line.fields[0]!r} has diameter "
                f"{diameter}; a cylindrical tank needs a positive one"
            )
        tanks.append(
            Tank(
                line.fields[0],
                as_float(line, 1, "the bottom elevation", path) * length_scale,
                as_float(line, 2, "the initial level", path) * length_scale,
                as_float(line, 3, "the minimum level", path) * length_scale,
                as_float(line, 4, "the maximum level", path) * length_scale,
                diameter,
            )
        )

    pipes: list[WaterPipe] = []
    for line in sections.get("PIPES", []):
        require_fields(line, 6, "a pipe", path)
        roughness = as_float(line, 5, "the roughness", path)
        # Hazen-Williams C is unitless; Darcy-Weisbach roughness is in MILLIFEET (US) or
        # millimetres (SI) -- Manual Table 3.1's own note, "epsilon = Darcy-Weisbach
        # roughness (ft)" with the [PIPES] column in millifeet.
        if headloss == "D-W":
            roughness = roughness * (FOOT * 1e-3 if us else 1e-3)
        pipes.append(
            WaterPipe(
                line.fields[0], line.fields[1], line.fields[2],
                as_float(line, 3, "the length", path) * length_scale,
                as_float(line, 4, "the diameter", path) * diameter_scale,
                roughness,
                as_float(line, 6, "the minor loss", path) if len(line.fields) > 6 else 0.0,
                line.fields[7].upper() if len(line.fields) > 7 else "OPEN",
            )
        )

    pumps: list[Pump] = []
    for line in sections.get("PUMPS", []):
        require_fields(line, 5, "a pump", path)
        curve = None
        power = None
        speed = 1.0
        rest = list(line.fields[3:])
        while rest:
            keyword = rest.pop(0).upper()
            if not rest:
                raise ValueError(
                    f"{path}: line {line.number}: pump {line.fields[0]!r} has the keyword "
                    f"{keyword!r} with no value"
                )
            value = rest.pop(0)
            if keyword == "HEAD":
                curve = value
            elif keyword == "POWER":
                power = float(value)
            elif keyword == "SPEED":
                speed = float(value)
            elif keyword == "PATTERN":
                raise ValueError(
                    f"{path}: line {line.number}: pump {line.fields[0]!r} has a SPEED "
                    f"PATTERN; variable-speed pumps are out of scope (spec 13.5)"
                )
            else:
                raise ValueError(
                    f"{path}: line {line.number}: pump {line.fields[0]!r} has the unknown "
                    f"keyword {keyword!r}"
                )
        pumps.append(
            Pump(line.fields[0], line.fields[1], line.fields[2], curve, power, speed)
        )

    valves: list[Valve] = []
    for line in sections.get("VALVES", []):
        require_fields(line, 6, "a valve", path)
        kind = line.fields[4].upper()
        setting = as_float(line, 5, "the setting", path)
        # TCV's setting is a unitless loss coefficient; FCV's is a FLOW (Appendix C.2.9).
        if kind == "FCV":
            setting = setting * flow
        valves.append(
            Valve(
                line.fields[0], line.fields[1], line.fields[2],
                as_float(line, 3, "the diameter", path) * diameter_scale,
                kind, setting,
                as_float(line, 6, "the minor loss", path) if len(line.fields) > 6 else 0.0,
            )
        )

    controls: list[Control] = []
    for line in sections.get("CONTROLS", []):
        upper = [f.upper() for f in line.fields]
        if "TIME" in upper or "CLOCKTIME" in upper:
            raise ValueError(
                f"{path}: line {line.number}: a time-based control "
                f"({line.raw.strip()!r}); only tank-level controls are read, time "
                f"variation belongs in the drivers"
            )
        require_fields(line, 8, "a control", path)
        if upper[0] != "LINK" or upper[3] != "IF" or upper[4] != "NODE":
            raise ValueError(
                f"{path}: line {line.number}: only controls of the form 'LINK <id> "
                f"OPEN|CLOSED IF NODE <tank> BELOW|ABOVE <level>' are read, got "
                f"{line.raw.strip()!r}"
            )
        if upper[2] not in ("OPEN", "CLOSED"):
            raise ValueError(
                f"{path}: line {line.number}: a control's link status must be OPEN or "
                f"CLOSED, got {line.fields[2]!r}"
            )
        if upper[6] not in ("BELOW", "ABOVE"):
            raise ValueError(
                f"{path}: line {line.number}: a control's test must be BELOW or ABOVE, "
                f"got {line.fields[6]!r}"
            )
        controls.append(
            Control(
                line.fields[1], upper[2], line.fields[5], upper[6],
                as_float(line, 7, "the trigger level", path) * length_scale,
            )
        )

    duration = 0.0
    hydraulic = 3600.0
    pattern_step = 3600.0
    report_step = 3600.0
    for line in sections.get("TIMES", []):
        key = " ".join(f.upper() for f in line.fields[:2])
        if key.startswith("DURATION"):
            duration = _time(line.fields[1], path, line)
        elif key == "HYDRAULIC TIMESTEP":
            hydraulic = _time(line.fields[2], path, line)
        elif key == "PATTERN TIMESTEP":
            pattern_step = _time(line.fields[2], path, line)
        elif key == "REPORT TIMESTEP":
            report_step = _time(line.fields[2], path, line)

    network = WaterNetwork(
        junctions=tuple(junctions),
        reservoirs=tuple(reservoirs),
        tanks=tuple(tanks),
        pipes=tuple(pipes),
        pumps=tuple(pumps),
        valves=tuple(valves),
        # A pump's HEAD curve is (flow, head): x in the file's flow units, y in its length
        # units. `[CURVES]` carries no type tag of its own, so the conversion is applied
        # here, where the only curves this reader keeps are pump curves.
        curves={
            name: tuple((x * flow, y * length_scale) for x, y in points)
            for name, points in curves.items()
        },
        patterns=patterns,
        controls=tuple(controls),
        demand_pattern=demand_pattern,
        pattern_timestep=pattern_step,
        hydraulic_timestep=hydraulic,
        report_timestep=report_step,
        duration=duration,
        headloss=headloss,
        options=WaterOptions(
            demand_model=demand_model,
            minimum_pressure=minimum_pressure * pressure_scale,
            required_pressure=required_pressure * pressure_scale,
            pressure_exponent=pressure_exponent,
            specific_gravity=specific_gravity,
            viscosity=viscosity,
        ),
        notes=(
            {"flow_units": flow_unit, "unrecognised_options": "; ".join(unrecognised)}
            if unrecognised
            else {"flow_units": flow_unit}
        ),
    )
    network.validate()
    return network
