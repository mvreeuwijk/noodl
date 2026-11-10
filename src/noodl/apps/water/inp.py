"""A documented EPANET 2.2 `.inp` subset.

READ: `[JUNCTIONS] [RESERVOIRS] [TANKS] [PIPES] [PUMPS] [VALVES] [DEMANDS] [PATTERNS]
[CURVES] [CONTROLS] [RULES] [STATUS] [EMITTERS] [ENERGY] [OPTIONS] [TIMES]`, with the column
layouts of the EPANET 2.2 manual's Appendix C.2. Everything EPANET 2.2's hydraulics reads is
READ and converted to SI here; `WaterNetwork.validate()` then REFUSES, by name, each feature
the builder does not model yet (a numeric `[STATUS]` pump speed,
a constant-POWER pump, a pump with more than a three-point curve or a SPEED PATTERN, a tank
with a volume curve or OVERFLOW, a rule, and every `[CONTROLS]`
form except `LINK <pump> OPEN|CLOSED IF NODE <tank> BELOW|ABOVE <level>`), so that reading a
file never changes what it means and building one never silently drops a feature. The reader
itself refuses only what it cannot read: a malformed record, an unknown name, a curve used
in two roles whose units differ.

CURVES BY ROLE. `[CURVES]` carries no type of its own, so each curve is converted by the role
that uses it: a pump HEAD curve or a GPV head-loss curve (flow, length), a pump efficiency
curve from `[ENERGY]` (flow, percent: the percentages are kept as written) or a tank volume
curve (length, volume). A curve used in two roles with different units is refused by name.
A curve no device uses is converted as a head curve, as before.

IGNORED because nothing in the hydraulics references them: `[TITLE]`, `[REPORT]` ("for the
Windows version of EPANET, the only [REPORT] option recognized is STATUS", Manual p.142),
`[COORDINATES]`, `[VERTICES]`, `[LABELS]`, `[BACKDROP]`, `[TAGS]`, `[END]`, and the
water-quality sections `[QUALITY] [SOURCES] [REACTIONS] [MIXING]` (this reader returns
hydraulics; a quality run is configured through `build_model(quality=...)`). Any other
section carrying content is refused.

UNITS. EPANET computes internally in feet and cfs whatever the file says, converting at the
boundary; this reader converts to SI instead. The flow unit chosen in `[OPTIONS] Units`
also fixes the LENGTH units of every other field: `CFS GPM MGD IMGD AFD` are US (feet for
elevations, lengths, heads and tank levels, inches for diameters) and `LPS LPM MLD CMH CMD`
are SI (metres, millimetres) -- Manual p.136, "LPS/LPM/MLD/CMH/CMD => SI/metric for
everything else".

`[OPTIONS]` KEYS THAT CHANGE THE PHYSICS -- `DEMAND MODEL`, `MINIMUM PRESSURE`,
`REQUIRED PRESSURE`, `PRESSURE EXPONENT`, `SPECIFIC GRAVITY`, `VISCOSITY` -- are carried
onto `WaterNetwork.options` (a `WaterOptions`, EPANET's own defaults) rather than dropped;
`build_model` defaults its own `pda`/`p_min`/`p_req`/`exponent` arguments from them.
`EMITTER EXPONENT` is read too. Any OTHER `[OPTIONS]` line (`TRIALS`, `ACCURACY`,
`UNBALANCED`, the in-section `QUALITY`
mode, ...) is recorded verbatim in `notes["unrecognised_options"]` instead of being
silently ignored, since most of them are solver/report cosmetics this reader does not need
but a few (an unimplemented `QUALITY CHEMICAL`, say) would silently change the network if
dropped without a trace.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from noodl.apps.inpfile import as_float, read_sections, require_fields
from noodl.apps.water.controls import Control, status_code
from noodl.apps.water.elements import NU_WATER_FT2
from noodl.apps.water.inp_rules import read_rules
from noodl.apps.water.inp_times import clock_time, time_with_unit
from noodl.apps.water.network import (
    Emitter,
    EnergyOptions,
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
        "END", "QUALITY", "SOURCES", "REACTIONS", "MIXING",
    }
)
_READ = frozenset(
    {
        "JUNCTIONS", "RESERVOIRS", "TANKS", "PIPES", "PUMPS", "VALVES", "DEMANDS",
        "PATTERNS", "CURVES", "CONTROLS", "RULES", "STATUS", "EMITTERS", "ENERGY", "OPTIONS",
        "TIMES",
    }
)
#: Horsepower in kW (`wntr`'s `HydParam.Power`, 745.699872 W per hp).
HP_TO_KW = 0.745699872


def read_epanet_inp(path) -> WaterNetwork:
    """Read the documented subset of an EPANET 2.2 `.inp` into a `WaterNetwork`, in SI."""
    path = Path(path)
    sections = read_sections(path)
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
    emitter_exponent = 0.5
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
            if headloss not in ("H-W", "D-W", "C-M"):
                raise ValueError(
                    f"{path}: line {line.number}: HEADLOSS {headloss!r} is not H-W, D-W "
                    f"or C-M"
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
        elif key == "EMITTER" and second == "EXPONENT":
            emitter_exponent = as_float(line, 2, "EMITTER EXPONENT", path)
        elif key == "SPECIFIC" and second == "GRAVITY":
            specific_gravity = as_float(line, 2, "SPECIFIC GRAVITY", path)
        elif key == "VISCOSITY":
            require_fields(line, 2, "VISCOSITY", path)
            viscosity = as_float(line, 1, "VISCOSITY", path)
        else:
            unrecognised.append(line.raw.strip())

    flow = FLOW_UNITS[flow_unit]
    us = flow_unit in _US_UNITS
    # EPANET 2.2 (`input1.c`, `adjustdata`): a VISCOSITY above 1e-3 is a multiplier on
    # water's 1.1e-5 ft2/s, anything else is the ACTUAL kinematic viscosity, in m2/s for
    # an SI file and ft2/s for a US one. `WaterOptions.viscosity` is always the ratio.
    if viscosity <= 1e-3:
        viscosity = viscosity / (NU_WATER_FT2 * (1.0 if us else FOOT**2))
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

    # Which devices use each curve, by role (see the module docstring).
    curve_roles: dict[str, set[str]] = {}

    def use_curve(name: str | None, role: str, line) -> None:
        if name is None:
            return
        if name not in curves:
            raise ValueError(f"{path}: line {line.number}: names curve {name!r}, which has "
                             f"no [CURVES] entry")
        curve_roles.setdefault(name, set()).add(role)

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
    # Several categories of one junction add up; the model carries one demand and one pattern
    # per junction, so categories whose patterns differ are refused rather than merged.
    # A demand with no pattern uses the default pattern (`[OPTIONS] PATTERN`, else a pattern
    # named "1" if one exists, else a constant 1.0), so categories are compared by EFFECTIVE
    # pattern: "no pattern" and an explicit naming of the default are the same.
    default_pattern = demand_pattern or ("1" if "1" in patterns else None)

    def effective(name: str | None) -> str | None:
        return name if name is not None else default_pattern

    overrides: dict[str, tuple[float, str | None]] = {}
    for line in sections.get("DEMANDS", []):
        require_fields(line, 2, "a demand", path)
        pattern = line.fields[2] if len(line.fields) > 2 else None
        value = as_float(line, 1, "the demand", path) * flow * demand_multiplier
        name = line.fields[0]
        if name in overrides:
            total, first = overrides[name]
            if effective(pattern) != effective(first):
                raise ValueError(
                    f"{path}: line {line.number}: junction {name!r} has [DEMANDS] "
                    f"categories with different patterns ({first!r} and {pattern!r}); "
                    f"one demand pattern per junction is modelled, so they cannot be "
                    f"aggregated"
                )
            overrides[name] = (total + value, first if first is not None else pattern)
        else:
            overrides[name] = (value, pattern)
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
        volume_curve = (
            line.fields[7] if len(line.fields) > 7 and line.fields[7] != "*" else None
        )
        use_curve(volume_curve, "volume", line)
        overflow = False
        if len(line.fields) > 8:
            if line.fields[8].upper() not in ("YES", "NO"):
                raise ValueError(
                    f"{path}: line {line.number}: tank {line.fields[0]!r} has the overflow "
                    f"flag {line.fields[8]!r}; use YES or NO"
                )
            overflow = line.fields[8].upper() == "YES"
        diameter = as_float(line, 5, "the diameter", path) * length_scale
        if volume_curve is None and not diameter > 0:
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
                volume_curve,
                overflow,
                # MinVol (column 7), a volume: ft3 in a US file, m3 in an SI one
                as_float(line, 6, "the minimum volume", path) * length_scale**3
                if len(line.fields) > 6 else 0.0,
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
        speed_pattern = None
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
                # US files give horsepower, SI files kilowatts (`wntr` HydParam.Power).
                power = float(value) * (HP_TO_KW if us else 1.0)
            elif keyword == "SPEED":
                speed = float(value)
            elif keyword == "PATTERN":
                if value not in patterns:
                    raise ValueError(
                        f"{path}: line {line.number}: pump {line.fields[0]!r} names the "
                        f"speed pattern {value!r}, which has no [PATTERNS] entry"
                    )
                speed_pattern = value
            else:
                raise ValueError(
                    f"{path}: line {line.number}: pump {line.fields[0]!r} has the unknown "
                    f"keyword {keyword!r}"
                )
        use_curve(curve, "head", line)
        pumps.append(
            Pump(line.fields[0], line.fields[1], line.fields[2], curve, power, speed,
                 speed_pattern)
        )

    valves: list[Valve] = []
    for line in sections.get("VALVES", []):
        require_fields(line, 6, "a valve", path)
        kind = line.fields[4].upper()
        # A GPV's setting is the id of its head-loss curve; the others are numbers: TCV's a
        # unitless loss coefficient, FCV's a FLOW, PBV's a PRESSURE (Appendix C.2.9).
        curve = line.fields[5] if kind == "GPV" else None
        setting = 0.0 if kind == "GPV" else as_float(line, 5, "the setting", path)
        if kind == "FCV":
            setting = setting * flow
        elif kind in ("PRV", "PSV", "PBV"):
            setting = setting * pressure_scale
        if kind == "GPV":
            use_curve(curve, "head", line)
        valves.append(
            Valve(
                line.fields[0], line.fields[1], line.fields[2],
                as_float(line, 3, "the diameter", path) * diameter_scale,
                kind, setting,
                as_float(line, 6, "the minor loss", path) if len(line.fields) > 6 else 0.0,
                curve=curve,
            )
        )

    junction_names = {j.name for j in junctions}
    reservoir_names = {r.name for r in reservoirs}
    tank_names = {t.name for t in tanks}
    link_kinds: dict[str, str] = {p.name: "PIPE" for p in pipes}
    link_kinds.update({p.name: "PUMP" for p in pumps})
    link_kinds.update({v.name: v.kind.upper() for v in valves})

    def link_setting(link: str, value: float, line) -> float:
        """A numeric link setting in SI: a pump's relative speed is unitless, a PRV, PSV or PBV
        setting a pressure, an FCV's a flow, a TCV's a unitless loss coefficient (Appendix
        C.2.9; `wntr` `_read_status`). A pipe or a GPV takes none."""
        kind = link_kinds.get(link)
        if kind is None:
            raise ValueError(f"{path}: line {line.number}: names link {link!r}, which is "
                             f"not a link of this network")
        if kind in ("PUMP", "TCV"):
            return value
        if kind in ("PRV", "PSV", "PBV"):
            return value * pressure_scale
        if kind == "FCV":
            return value * flow
        raise ValueError(f"{path}: line {line.number}: a numeric setting for the {kind} "
                         f"{link!r} has no meaning")

    def node_trigger_scale(node: str, line) -> float:
        """Metres per file unit of a node trigger: a tank's LEVEL and a reservoir's HEAD are
        lengths, a junction's PRESSURE is psi in a US file and metres in an SI one."""
        if node in junction_names:
            return pressure_scale
        if node in tank_names or node in reservoir_names:
            return length_scale
        raise ValueError(f"{path}: line {line.number}: names node {node!r}, which is not a "
                         f"node of this network")

    controls: list[Control] = []
    for line in sections.get("CONTROLS", []):
        upper = [f.upper() for f in line.fields]
        form = (
            f"{path}: line {line.number}: only controls of the forms 'LINK <id> "
            f"OPEN|CLOSED|<setting> AT TIME <t>', '... AT CLOCKTIME <t> [AM|PM]' and "
            f"'... IF NODE <id> BELOW|ABOVE <value>' are read, got {line.raw.strip()!r}"
        )
        if len(upper) < 6:
            raise ValueError(form)
        link = line.fields[1]
        status: int | None = status_code(upper[2])
        setting: float | None = None
        if status is None:
            try:
                number = float(line.fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"{path}: line {line.number}: a control's link status must be OPEN, "
                    f"CLOSED or a number, got {line.fields[2]!r}"
                ) from exc
            setting = link_setting(link, number, line)
        if upper[3] == "AT" and upper[4] == "TIME":
            controls.append(Control(link, status, setting, "time",
                                    value=time_with_unit(line.fields, 5, path, line)))
        elif upper[3] == "AT" and upper[4] == "CLOCKTIME":
            meridiem = line.fields[6] if len(line.fields) > 6 else None
            controls.append(Control(link, status, setting, "clocktime",
                                    value=clock_time(line.fields[5], meridiem, path, line)))
        elif upper[3] == "IF" and upper[4] == "NODE":
            if len(upper) < 8:
                raise ValueError(form)
            if upper[6] not in ("BELOW", "ABOVE"):
                raise ValueError(
                    f"{path}: line {line.number}: a control's test must be BELOW or ABOVE, "
                    f"got {line.fields[6]!r}"
                )
            controls.append(Control(
                link, status, setting, "node", node=line.fields[5],
                above=upper[6] == "ABOVE",
                value=as_float(line, 7, "the trigger level", path)
                * node_trigger_scale(line.fields[5], line),
            ))
        else:
            raise ValueError(form)

    # `[STATUS]`: a link's initial status, or its initial setting (Appendix C.2.17).
    initial_status: dict[str, str | float] = {}
    for line in sections.get("STATUS", []):
        require_fields(line, 2, "a status", path)
        word = line.fields[1].upper()
        if status_code(word) is not None:
            initial_status[line.fields[0]] = word
        else:
            initial_status[line.fields[0]] = link_setting(
                line.fields[0], as_float(line, 1, "the setting", path), line
            )

    # `[EMITTERS]`: q = C p^n with n = `EMITTER EXPONENT`, C in the file's flow unit per
    # psi^n (US) or per m^n (SI). In SI, q [m3/s] = f C_us (p_psi)^n and p_psi = p_m / PSI_TO_M,
    # so C_si = f C_us / PSI_TO_M**n (EPANET `input1.c` `convertunits`: `ucf = pow(Ucf[FLOW],
    # Qexp) / Ucf[PRESSURE]`). `wntr`'s `HydParam.EmitterCoeff` hard-codes n = 1/2 and is not
    # the reference. The exponent is an `[OPTIONS]` value, read above whatever the file's
    # section order.
    emitters: list[Emitter] = []
    for line in sections.get("EMITTERS", []):
        require_fields(line, 2, "an emitter", path)
        if line.fields[0] not in junction_names:
            raise ValueError(
                f"{path}: line {line.number}: an emitter at {line.fields[0]!r}, which is "
                f"not a junction of this network"
            )
        coefficient = as_float(line, 1, "the emitter coefficient", path) * flow
        if us:
            coefficient = coefficient / PSI_TO_M**emitter_exponent
        emitters.append(Emitter(line.fields[0], coefficient))

    rules = read_rules(
        sections.get("RULES", []), path, flow=flow, length=length_scale,
        pressure=pressure_scale, power=HP_TO_KW if us else 1.0,
        junctions=junction_names, tanks=tank_names, reservoirs=reservoir_names,
        link_kinds=link_kinds, link_setting=link_setting,
    )

    # `[ENERGY]` (Appendix C.2.5): the global defaults and a per-pump price, price pattern
    # and efficiency curve. A price is dollars per kWh whatever the units and an efficiency
    # a percent. Nothing here enters the hydraulics.
    global_efficiency, global_price, global_pattern, demand_charge = 75.0, 0.0, None, 0.0
    pump_energy: dict[str, dict[str, object]] = {}
    pump_names = {p.name for p in pumps}
    for line in sections.get("ENERGY", []):
        upper = [f.upper() for f in line.fields]
        unknown = ValueError(
            f"{path}: line {line.number}: unknown [ENERGY] entry {line.raw.strip()!r}"
        )
        if upper[0] == "GLOBAL" and len(upper) >= 3:
            if upper[1].startswith("EFFIC"):
                global_efficiency = as_float(line, 2, "the global efficiency", path)
            elif upper[1] == "PRICE":
                global_price = as_float(line, 2, "the global price", path)
            elif upper[1] == "PATTERN":
                global_pattern = line.fields[2]
            else:
                raise unknown
        elif upper[0] == "DEMAND" and len(upper) >= 3 and upper[1] == "CHARGE":
            demand_charge = as_float(line, 2, "the demand charge", path)
        elif upper[0] == "PUMP" and len(upper) >= 4:
            if line.fields[1] not in pump_names:
                raise ValueError(f"{path}: line {line.number}: [ENERGY] names pump "
                                 f"{line.fields[1]!r}, which is not a pump of this network")
            entry = pump_energy.setdefault(line.fields[1], {})
            if upper[2].startswith("EFFIC"):
                use_curve(line.fields[3], "efficiency", line)
                entry["efficiency_curve"] = line.fields[3]
            elif upper[2] == "PRICE":
                entry["price"] = as_float(line, 3, "the pump price", path)
            elif upper[2] == "PATTERN":
                entry["price_pattern"] = line.fields[3]
            else:
                raise unknown
        else:
            raise unknown
    if pump_energy:
        pumps = [replace(p, **pump_energy[p.name]) if p.name in pump_energy else p
                 for p in pumps]

    # Each curve is converted by its role: head (flow, length), efficiency (flow, percent),
    # volume (length, volume). A curve two roles share must mean the same thing in both.
    curve_units = {"head": "flow and length", "efficiency": "flow and percent",
                   "volume": "length and volume"}
    converted: dict[str, tuple[tuple[float, float], ...]] = {}
    for name, points in curves.items():
        roles = sorted(curve_roles.get(name, {"head"}))
        if len(roles) > 1:
            raise ValueError(
                f"{path}: curve {name!r} is used as both a {roles[0]} curve "
                f"({curve_units[roles[0]]}) and a {roles[1]} curve ({curve_units[roles[1]]}); "
                f"its units differ between the two, so it cannot be converted"
            )
        role = roles[0]
        if role == "head":
            converted[name] = tuple((x * flow, y * length_scale) for x, y in points)
        elif role == "efficiency":
            converted[name] = tuple((x * flow, y) for x, y in points)
        else:
            converted[name] = tuple(
                (x * length_scale, y * length_scale**3) for x, y in points
            )

    duration = 0.0
    hydraulic = 3600.0
    pattern_step = 3600.0
    report_step = 3600.0
    rule_step: float | None = None
    pattern_start = 0.0
    report_start = 0.0
    start_clocktime = 0.0
    unrecognised_times: list[str] = []
    for line in sections.get("TIMES", []):
        key = " ".join(f.upper() for f in line.fields[:2])
        if key.startswith("DURATION"):
            duration = time_with_unit(line.fields, 1, path, line)
        elif key == "HYDRAULIC TIMESTEP":
            hydraulic = time_with_unit(line.fields, 2, path, line)
        elif key == "PATTERN TIMESTEP":
            pattern_step = time_with_unit(line.fields, 2, path, line)
        elif key == "REPORT TIMESTEP":
            report_step = time_with_unit(line.fields, 2, path, line)
        elif key == "RULE TIMESTEP":
            rule_step = time_with_unit(line.fields, 2, path, line)
        elif key == "PATTERN START":
            pattern_start = time_with_unit(line.fields, 2, path, line)
        elif key == "REPORT START":
            report_start = time_with_unit(line.fields, 2, path, line)
        elif key == "START CLOCKTIME":
            require_fields(line, 3, "START CLOCKTIME", path)
            start_clocktime = clock_time(
                line.fields[2], line.fields[3] if len(line.fields) > 3 else None, path, line
            )
        else:
            unrecognised_times.append(line.raw.strip())

    network = WaterNetwork(
        junctions=tuple(junctions),
        reservoirs=tuple(reservoirs),
        tanks=tuple(tanks),
        pipes=tuple(pipes),
        pumps=tuple(pumps),
        valves=tuple(valves),
        curves=converted,
        patterns=patterns,
        controls=tuple(controls),
        rules=tuple(rules),
        emitters=tuple(emitters),
        initial_status=initial_status,
        energy=EnergyOptions(global_efficiency, global_price, global_pattern, demand_charge),
        demand_pattern=demand_pattern,
        pattern_timestep=pattern_step,
        hydraulic_timestep=hydraulic,
        report_timestep=report_step,
        duration=duration,
        explicit_rule_timestep=rule_step,
        pattern_start=pattern_start,
        report_start=report_start,
        start_clocktime=start_clocktime,
        headloss=headloss,
        options=WaterOptions(
            demand_model=demand_model,
            minimum_pressure=minimum_pressure * pressure_scale,
            required_pressure=required_pressure * pressure_scale,
            pressure_exponent=pressure_exponent,
            emitter_exponent=emitter_exponent,
            specific_gravity=specific_gravity,
            viscosity=viscosity,
            flow_units=flow_unit,
        ),
        notes={
            "flow_units": flow_unit,
            **({"unrecognised_options": "; ".join(unrecognised)} if unrecognised else {}),
            **({"unrecognised_times": "; ".join(unrecognised_times)}
               if unrecognised_times else {}),
        },
    )
    network.check()
    return network
