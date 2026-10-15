"""The `[PUMPS]` and `[CONTROLS]` sections of a SWMM `.inp` (read by `inp.read_swmm_inp`).

`read_pumps` follows ``link.c::pump_readParams`` (link.c:1406-1470) and ``pump_validate``:
``Name FromNode ToNode [Curve|*] [Status] [Startup] [Shutoff]`` with a PUMP1..PUMP5 `[CURVES]`
entry for the curve or ``*`` for an ideal pump, Status ON or OFF (default ON) and the startup
and shutoff depths in the file's length unit (default 0: none). `read_controls` reads
`[CONTROLS]` into a `controls.ControlSet` (`controls.parse_rules`), with the CONTROL curves
and the time series its actions name.

Both are read under DYNWAVE only; under STEADY or KINWAVE they are refused by name, as is
everything `controls` does not model. Source references are to SWMM 5.2.4 ``src/solver``.
"""

from __future__ import annotations

from noodl.apps.inpfile import as_float, require_fields
from noodl.apps.sewer.controls import ControlSet, parse_rules
from noodl.apps.sewer.network import Pump

#: SWMM's pump curve types -> `Pump.kind` (``Pump.type = curveType - PUMP1_CURVE``).
_PUMP_CURVES = {"PUMP1": "type1", "PUMP2": "type2", "PUMP3": "type3", "PUMP4": "type4",
                "PUMP5": "type5"}


def read_pumps(lines, *, curves: dict, nodes: set[str], dynamic: bool, routing: str, path,
               lf: float) -> tuple[Pump, ...]:
    """The `Pump` list of the `[PUMPS]` `lines`. `curves` is the reader's ``{name: (type,
    (xs, ys))}`` (SI), `nodes` the node names and `lf` the metres per length unit. Refused by
    name, with the line: pumps under STEADY or KINWAVE, an unknown node, a curve that is not
    a PUMP1-PUMP5 `[CURVES]` entry, a Status other than ON or OFF, a negative depth and
    SWMM's ERR_PUMP_LIMITS (startup depth not above the shutoff depth)."""
    if not lines:
        return ()
    if not dynamic:
        raise ValueError(
            f"{path}: section [PUMPS] (line {lines[0].number}): pumps are read under "
            f"dynamic-wave routing only (FLOW_ROUTING DYNWAVE), not {routing}")
    pumps = []
    for line in lines:
        require_fields(line, 3, "a pump entry", path)
        name, u, v = line.fields[0], line.fields[1], line.fields[2]
        where = f"{path}: line {line.number}: pump {name!r}"
        for end in (u, v):
            if end not in nodes:
                raise ValueError(f"{where} names node {end!r}, which is not a junction, "
                                 f"storage unit or outfall")
        kind, curve = "ideal", ()
        if len(line.fields) > 3 and line.fields[3] != "*":
            label = line.fields[3]
            if label not in curves:
                raise ValueError(f"{where} names curve {label!r}, which is not in [CURVES]")
            ctype, points = curves[label]
            if ctype not in _PUMP_CURVES:
                raise ValueError(f"{where}: curve {label!r} is of type {ctype}, not one of "
                                 f"PUMP1..PUMP5 (SWMM's ERR_NO_CURVE)")
            kind, curve = _PUMP_CURVES[ctype], points
        on = True
        if len(line.fields) > 4:
            status = line.fields[4].upper()
            if status not in ("ON", "OFF"):
                raise ValueError(f"{where}: Status must be ON or OFF, got "
                                 f"{line.fields[4]!r}")
            on = status == "ON"
        depths = []
        for i, what in ((5, "startup depth"), (6, "shutoff depth")):
            value = as_float(line, i, f"the {what}", path) if len(line.fields) > i else 0.0
            if value < 0.0:                                                   # link.c:1456, 1462
                raise ValueError(f"{where}: negative {what} {line.fields[i]}")
            depths.append(value * lf)
        if depths[0] > 0.0 and depths[0] <= depths[1]:                        # link.c:1518
            raise ValueError(f"{where}: the startup depth {line.fields[5]} must exceed the "
                             f"shutoff depth {line.fields[6]} (SWMM's ERR_PUMP_LIMITS)")
        pumps.append(Pump(name, u, v, kind, curve, on, depths[0], depths[1], order=line.number))
    return tuple(pumps)


def read_controls(lines, *, links: dict[str, str], nodes: set[str], curves: dict, series,
                  rule_step: float, start, dynamic: bool, routing: str, path, lf: float,
                  qf: float) -> ControlSet | None:
    """The `ControlSet` of the `[CONTROLS]` `lines` (None for an empty section). `links` maps
    each link name to its kind, `nodes` the node names, `curves` the reader's curves (the
    CONTROL ones, as in the file), `series(name)` returns the `TimeSeries` of a `[TIMESERIES]`
    entry (ValueError for an unknown one), `rule_step` is RULE_STEP in s, `start` the run's
    start (datetime) and `lf`, `qf` the file's length (m) and flow (m3/s) factors. Refused by
    name: controls under STEADY or KINWAVE, and everything `controls.parse_rules` refuses."""
    if not lines:
        return None
    if not dynamic:
        raise ValueError(
            f"{path}: section [CONTROLS] (line {lines[0].number}): control rules are read "
            f"under dynamic-wave routing only (FLOW_ROUTING DYNWAVE), not {routing}")
    rules = parse_rules(lines, links=links, nodes=nodes, path=path)
    used_curves, used_series = {}, {}
    for rule in rules:
        for action in (*rule.then_actions, *rule.else_actions):
            if action.mode == "curve":
                if action.source not in curves or curves[action.source][0] != "CONTROL":
                    raise ValueError(f"{path}: rule {rule.name!r} names curve "
                                     f"{action.source!r}, which is not a CONTROL curve in "
                                     f"[CURVES]")
                used_curves[action.source] = curves[action.source][1]
            elif action.mode == "timeseries":
                try:
                    used_series[action.source] = series(action.source)
                except ValueError as err:
                    raise ValueError(f"{path}: rule {rule.name!r}: {err}") from None
    return ControlSet(tuple(rules), used_curves, used_series, length_factor=lf,
                      flow_factor=qf, start=start, rule_step=rule_step)
