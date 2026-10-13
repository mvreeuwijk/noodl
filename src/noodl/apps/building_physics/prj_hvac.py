"""CONTAM `.prj` sections for mechanical ventilation: simple air-handling systems (section
13, with their supply and return points and implicit paths among the airflow paths) and duct
networks (11 duct elements, 17 junctions, 18 junction concentrations, 19 segments): the
records, their readers, and the builder that adds the ducts to the network `prj._build`
assembles. Field positions are listed in RECORD LAYOUTS below; they follow TN 1887r1 Appendix
A and were checked by loading `tests/data/contam/duct_three_zone.prj` and
`tests/data/contam/ahs_three_zone.prj` in ContamX 3.4.1.7, whose flows the built networks
reproduce.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from noodl.apps.building_physics.air_handling import (
    EXHAUST,
    OUTDOOR_AIR,
    RECIRCULATION,
    RETURN,
    SUPPLY,
)
from noodl.apps.building_physics.inputs import Input
from noodl.apps.building_physics.thermal import RHO_0
from noodl.drives import ConstantDrive, DuctStack, Stack, Wind
from noodl.elements import (
    AirDuct,
    Damper,
    DuctTerminal,
    FixedFlow,
    FixedVolumeFlow,
    Scaled,
    UpstreamDensityPowerLaw,
)
from noodl.topology import Network

F64 = torch.float64
TERMINAL_KIND = "duct_terminal"
TEMPERATURE_UNITS = 5          # ContamW's unit type of a temperature schedule

# --------------------------------------------------------------------------------------
# RECORD LAYOUTS
#   duct element  `nr icon dtype name`, a description line, the type's data line(s), then
#                 two geometry lines: `hdia perim area major minor As Qr Pr` and
#                 `shape u_D u_P u_A u_mj u_mn u_Qr u_Pr`. hdia (m) and area (m2) are what
#                 every segment of the element uses, also for a junction's volume; Qr != 0
#                 is duct leakage (refused).
#     dct_dwc     `rough lam u_R` (ContamX 3.4.1.7 ignores lam: measured)
#     dct_plr     `lam turb expt area dia coef u_A u_D`
#     dct_fcn/qcn `lam turb expt`
#     dct_cmf     `Flow u_F` (kg/s);  dct_cvf `Flow u_F` (m3/s)
#     dct_fan     as the path `fan_fan`: two lines, then npts measured points
#     dct_bdq/bdf `lam Cp xp Cn xn`
#     dct_fsp/qsp/psf/psq   `npts u_x u_y`, then npts `x y` rows, as `csf_*`
#   junction      21 fields `nr flags jtype z# dct k# s# c# l# X Y relHt T0 P0 icon clr
#                 uH uXY uT udP vf` (vf != 0 reads a value file: refused) and, for a terminal
#                 (jtype 1, or 3 "with wind"), `T: pf pw wPset wPmod wazm bal Ad Af Fdes Ct
#                 Cb Cbmax u_A u_F ddir fdir`. z# -1 is the ambient. Wind acts on an ambient
#                 terminal only when its flags carry the value 1 (measured); bal, Af, Fdes
#                 and Cbmax are duct-balancing data and are not used. The junction's schedule is its
#                 temperature when the week schedule's unit type is 5, else (a terminal) the
#                 terminal's opening S.
#   initial junction concentrations   rows `J# x1 .. xK`; the count is values.
#   segment       17 fields `nr flags n# m# e# f# s# c# dir len Ain Aout sllc clr u_L u_A
#                 vf`: from junction n# to m#, duct element e#, filter, schedule, control,
#                 length (m), sum of dynamic loss coefficients. Ain/Aout are not used.
#   simple AHS    `nr zr zs pr ps px clr name`, then a description line: the return and
#                 supply plenum zones and the recirculation, outdoor-air and exhaust path
#                 numbers. ContamX ignores the path and zone numbers (measured): it finds the
#                 implicit paths by their path flags, 16 recirculation (return -> supply
#                 plenum), 32 outdoor air (ambient -> supply plenum), 64 exhaust (return
#                 plenum -> ambient), and by position (ContamW writes them after every other
#                 path). A supply or return point is a path with flag 8 and `pa` = the AHS
#                 number; its `Fahs` (path field 18) is its design mass flow (kg/s), its
#                 schedule multiplies it and its `mult` is ignored. The recirculation path's
#                 schedule is the outdoor-air fraction and its `Fahs` the minimum outdoor-air
#                 flow; a schedule or `Fahs` on the outdoor-air or exhaust path is ignored.
#                 The plenum zones carry flags 10 (no variable-pressure bit).
# --------------------------------------------------------------------------------------

_DUCT_SPLINES = {"dct_fsp", "dct_qsp", "dct_psf", "dct_psq"}
_DUCT_TYPES = {"dct_dwc", "dct_plr", "dct_fcn", "dct_qcn", "dct_cmf", "dct_cvf", "dct_fan",
               "dct_bdq", "dct_bdf"} | _DUCT_SPLINES
_JUNCTION_FIELDS = 21
_SEGMENT_FIELDS = 17
_WIND_FLAG = 1                 # the junction-flags value that lets the wind act on a terminal


@dataclass
class DuctElementRecord:
    nr: int
    dtype: str
    name: str
    data: list[list[float]]
    diameter: float                  # hydraulic diameter, m
    area: float                      # flow area, m2


@dataclass
class TerminalData:
    filter_nr: int
    profile_nr: int
    wPset: float
    Ch: float
    azimuth: float
    area: float                      # duct area at the terminal, m2
    loss: float                      # terminal loss + balance coefficient


@dataclass
class JunctionRecord:
    nr: int
    flags: int
    zone_nr: int                     # the zone it sits in; -1 is the ambient
    schedule_nr: int
    z: float                         # absolute height, m
    T0: float
    terminal: TerminalData | None


@dataclass
class SegmentRecord:
    nr: int
    from_junction: int
    to_junction: int
    element_nr: int
    filter_nr: int
    schedule_nr: int
    length: float
    loss: float


@dataclass
class BuildContext:
    """What `prj._build` shares with the builder here; the lists and dicts are appended
    to."""

    net: Network
    ambient: str
    zone_name: Callable[[int], str]          # CONTAM zone number (-1 ambient) -> node
    week: Callable[[int, str], object]       # schedule number -> WeekSchedule
    schedule_units: dict[int, int]           # week schedule number -> ContamW unit type
    filter_of: Callable[[int, str], object]  # filter number -> filter object
    g: float
    profiles: dict
    inputs: list
    edge_filters: dict
    built: list
    drives: list
    constant_wind_pressure: dict


@dataclass
class DuctBuild:
    junction_names: list[str]
    terminal_columns: list[tuple[int, int, int]]   # (junction nr, zone nr, network column)
    segment_columns: list[tuple[int, int]]         # (segment nr, network column)
    multiplier_edges: dict[str, int] = field(default_factory=dict)
    speed_edges: dict[str, int] = field(default_factory=dict)
    terminal_openings: int = 0


def _record(lines, section: str, count: int, read: int) -> list[str]:
    """The next record of `section`; a terminator before `count` records is refused."""
    tok = lines.record()
    if tok == ["-999"]:
        raise ValueError(f"prj: the {section!r} section promises {count} records but ends "
                         f"after {read}")
    return tok


def read_duct_elements(lines, count: int, out: dict[int, DuctElementRecord]) -> None:
    for k in range(count):
        tok = _record(lines, "duct elements", count, k)
        nr, dtype = int(tok[0]), tok[2]
        name = tok[3] if len(tok) > 3 else ""
        lines.raw()                                 # description (may be blank)
        if dtype not in _DUCT_TYPES:
            raise ValueError(f"prj: duct element {nr} ({name!r}) has unsupported type "
                             f"{dtype!r}")
        data = [[float(v) for v in lines.record()]]
        if dtype == "dct_fan":
            second = [float(v) for v in lines.record()]
            data.append(second)
            for _ in range(int(second[4])):
                data.append([float(v) for v in lines.record()])
        elif dtype in _DUCT_SPLINES:
            for _ in range(int(data[0][0])):
                data.append([float(v) for v in lines.record()])
        geometry = [float(v) for v in lines.record()]
        lines.record()                              # shape and units
        if geometry[6] != 0.0:
            raise ValueError(f"prj: duct element {nr} ({name!r}) has a leakage rate "
                             f"{geometry[6]} (duct leakage is unsupported)")
        out[nr] = DuctElementRecord(nr, dtype, name, data, geometry[0], geometry[2])
    lines.expect_end("duct elements")


def read_junctions(lines, count: int, out: list[JunctionRecord],
                   levels: dict[int, float]) -> None:
    for k in range(count):
        tok = _record(lines, "duct junctions", count, k)
        nr = int(tok[0])
        cut = tok.index("T:") if "T:" in tok else len(tok)
        base, term = tok[:cut], tok[cut + 1:]
        if len(base) != _JUNCTION_FIELDS:
            raise ValueError(f"prj: junction {nr} has {len(base)} fields before its terminal "
                             f"data, not {_JUNCTION_FIELDS}; a value file appends a name, "
                             f"which is unsupported: {' '.join(tok)}")
        flags, jtype, zone = int(base[1]), int(base[2]), int(base[3])
        pk, ps, pc, pld = (int(v) for v in base[5:9])
        if pk:
            raise ValueError(f"prj: junction {nr} references kinetic reaction {pk} "
                             f"(unsupported in a duct)")
        if pc:
            raise ValueError(f"prj: junction {nr} references control node {pc} "
                             f"(unsupported)")
        if int(base[20]):
            raise ValueError(f"prj: junction {nr} reads a value file (unsupported)")
        if pld not in levels:
            raise KeyError(f"prj: junction {nr} sits on level {pld}, which the file never "
                           f"defines")
        terminal = None
        if jtype in (1, 3):
            if len(term) < 11:
                raise ValueError(f"prj: terminal {nr} lacks its 'T:' data")
            terminal = TerminalData(int(term[0]), int(term[1]), float(term[2]),
                                    float(term[3]), float(term[4]), float(term[6]),
                                    float(term[9]) + float(term[10]))
        elif jtype != 0:
            raise ValueError(f"prj: junction {nr} has junction type {jtype}; 0 (junction), 1 "
                             f"and 3 (terminal) are supported")
        out.append(JunctionRecord(nr, flags, zone, ps, levels[pld] + float(base[11]),
                                  float(base[12]), terminal))
    lines.expect_end("duct junctions")


def read_duct_segments(lines, count: int, out: list[SegmentRecord]) -> None:
    for k in range(count):
        tok = _record(lines, "duct segments", count, k)
        nr = int(tok[0])
        if len(tok) != _SEGMENT_FIELDS:
            raise ValueError(f"prj: segment {nr} has {len(tok)} fields, not "
                             f"{_SEGMENT_FIELDS}; a value file appends a name, which is "
                             f"unsupported: {' '.join(tok)}")
        n, m, e, f, s, c = (int(v) for v in tok[2:8])
        if c:
            raise ValueError(f"prj: segment {nr} references control node {c} (unsupported)")
        if int(tok[16]):
            raise ValueError(f"prj: segment {nr} reads a value file (unsupported)")
        out.append(SegmentRecord(nr, n, m, e, f, s, float(tok[9]), float(tok[12])))
    lines.expect_end("duct segments")


def duct_element(el: DuctElementRecord, *, kind: str, src, tgt,
                 members: list[SegmentRecord], speed_key: str | None):
    """The air-layer element of the segments `members`, all of duct element `el`."""
    from noodl.apps.building_physics.prj import (
        _laminar_transition,
        fan_curve_element,
        spline_element,
    )

    d, first, n_e = el.dtype, el.data[0], len(members)
    if d == "dct_dwc":
        return AirDuct(torch.tensor([s.length for s in members], dtype=F64),
                       torch.tensor(el.diameter, dtype=F64), torch.tensor(first[0], dtype=F64),
                       torch.tensor([s.loss for s in members], dtype=F64),
                       area=torch.tensor(el.area, dtype=F64), src=src, tgt=tgt,
                       temperature_key="T", kind=kind)
    if d in ("dct_plr", "dct_fcn", "dct_qcn"):
        lam, turb, expt = first[:3]
        C, m = {"dct_plr": (turb * math.sqrt(RHO_0), 0.5), "dct_fcn": (turb, 0.0),
                "dct_qcn": (RHO_0 * turb, 1.0)}[d]
        return UpstreamDensityPowerLaw(
            torch.full((n_e,), C, dtype=F64), torch.full((n_e,), expt, dtype=F64),
            src=src, tgt=tgt, m=torch.tensor(m, dtype=F64), rho_ref=RHO_0,
            dp_transition=_laminar_transition(C, lam, expt), kind=kind)
    if d in ("dct_bdq", "dct_bdf"):
        _lam, Cp, xp, Cn, xn = first[:5]
        scale, m = (RHO_0, 1.0) if d == "dct_bdq" else (1.0, 0.0)
        return Damper(torch.tensor(scale * Cp, dtype=F64), torch.tensor(xp, dtype=F64),
                      torch.tensor(scale * Cn, dtype=F64), torch.tensor(xn, dtype=F64),
                      src=src, tgt=tgt, m=m, rho_ref=RHO_0, kind=kind)
    if d == "dct_cmf":
        return FixedFlow(torch.full((n_e,), first[0], dtype=F64), kind=kind)
    if d == "dct_cvf":
        return FixedVolumeFlow(torch.full((n_e,), first[0], dtype=F64), src=src, tgt=tgt,
                               kind=kind)
    if d == "dct_fan":
        return fan_curve_element(el.data, kind=kind, src=src, tgt=tgt,
                                 multiplier=[1.0] * n_e, speed_key=speed_key)
    try:
        return spline_element(d, el.data[1:], kind=kind, src=src, tgt=tgt,
                              multiplier=[1.0] * n_e)
    except ValueError as exc:
        raise ValueError(f"prj: duct element {el.nr}: {exc}") from exc


def build_ducts(ctx: BuildContext, elements: dict[int, DuctElementRecord],
                junctions: list[JunctionRecord],
                segments: list[SegmentRecord]) -> DuctBuild:
    """Add the junction nodes, segment and terminal edges, their elements, drives, inputs
    and filters to the build in `ctx`."""
    from noodl.apps.building_physics.prj import _check_edge_order

    net = ctx.net
    names = {j.nr: f"junction {j.nr}" for j in junctions}
    volumes = dict.fromkeys(names, 0.0)
    for s in segments:
        for end in (s.from_junction, s.to_junction):
            if end not in names:
                raise KeyError(f"prj: segment {s.nr} references junction {end}, which the "
                               f"'duct junctions' section does not define")
        if s.element_nr not in elements:
            raise KeyError(f"prj: segment {s.nr} references duct element {s.element_nr}, "
                           f"which the 'duct elements' section does not define")
        half = 0.5 * s.length * elements[s.element_nr].area
        volumes[s.from_junction] += half
        volumes[s.to_junction] += half
    for j in junctions:
        net.add_node(names[j.nr], z_ref=j.z, volume=volumes[j.nr], T0=j.T0,
                     heat_capacity=0.0)

    build = DuctBuild([names[j.nr] for j in junctions], [], [])
    by_kind: dict[str, list[SegmentRecord]] = {}
    columns: dict[str, list[int]] = {}
    for s in segments:
        kind = f"duct_{s.element_nr}"
        build.segment_columns.append((s.nr, net.b))
        columns.setdefault(kind, []).append(net.b)
        edge = net.add_edge(names[s.from_junction], names[s.to_junction], kind=kind)
        by_kind.setdefault(kind, []).append(s)
        if s.filter_nr:
            ctx.edge_filters[edge] = ctx.filter_of(s.filter_nr, f"segment {s.nr}")
    for kind, members in by_kind.items():
        el = elements[members[0].element_nr]
        _check_edge_order(net, kind, columns[kind])
        src, tgt = net.endpoints(kind)
        scheduled = [s for s in members if s.schedule_nr]
        if scheduled and el.dtype == "dct_dwc":
            raise ValueError(
                f"prj: segment {scheduled[0].nr} schedules Darcy-Colebrook duct element "
                f"{el.nr}; ContamX ignores a schedule there, so it is refused")
        speed_key = f"fan_speed.{kind}" if scheduled and el.dtype == "dct_fan" else None
        element = duct_element(el, kind=kind, src=src, tgt=tgt, members=members,
                               speed_key=speed_key)
        key = speed_key
        if scheduled and speed_key is None:
            key = f"flow_multiplier.{kind}"
            element = Scaled(element, key)
            build.multiplier_edges[kind] = len(members)
        elif speed_key is not None:
            build.speed_edges[kind] = len(members)
        for s in scheduled:
            ctx.inputs.append(Input(ctx.week(s.schedule_nr, f"segment {s.nr}"), key,
                                    index=members.index(s),
                                    name=f"segment {s.nr} schedule {s.schedule_nr}"))
        ctx.built.append(element)
        ctx.drives.append(DuctStack.from_network(net, kind, g=ctx.g))
    _build_terminals(ctx, junctions, names, build)
    return build


def _build_terminals(ctx: BuildContext, junctions: list[JunctionRecord],
                     names: dict[int, str], build: DuctBuild) -> None:
    net = ctx.net
    terminals = [j for j in junctions if j.terminal is not None]
    wind_pressure: list[float] = []
    for j in terminals:
        t = j.terminal
        try:
            zone = ctx.zone_name(j.zone_nr)
        except KeyError as exc:
            raise KeyError(f"prj: terminal {j.nr} sits in zone {j.zone_nr}, which the file "
                           f"never defines") from exc
        if (t.profile_nr or t.wPset) and not (zone == ctx.ambient and j.flags & _WIND_FLAG):
            raise ValueError(
                f"prj: terminal {j.nr} carries wind data (profile {t.profile_nr}, wPset "
                f"{t.wPset}) but ContamX applies wind only to an ambient terminal whose "
                f"flags carry the flag value 1; unsupported")
        attrs = dict(area=t.area, loss=t.loss, z_path=j.z)
        if t.profile_nr:
            attrs.update(Ch=t.Ch, azimuth=t.azimuth, profile=float(t.profile_nr))
        build.terminal_columns.append((j.nr, j.zone_nr, net.b))
        edge = net.add_edge(names[j.nr], zone, kind=TERMINAL_KIND, **attrs)
        # junction -> ambient: dp = P_j - (P_amb + wPset)
        wind_pressure.append(-t.wPset if not t.profile_nr else 0.0)
        if t.filter_nr:
            ctx.edge_filters[edge] = ctx.filter_of(t.filter_nr, f"terminal {j.nr}")
    order = [j.nr for j in terminals]
    for j in junctions:
        if not j.schedule_nr:
            continue
        schedule = ctx.week(j.schedule_nr, f"junction {j.nr}")
        if ctx.schedule_units.get(j.schedule_nr) == TEMPERATURE_UNITS:
            ctx.inputs.append(Input(schedule, "T", index=net.node_index(names[j.nr]),
                                    name=f"junction {j.nr} temperature schedule "
                                         f"{j.schedule_nr}"))
        elif j.terminal is not None:
            build.terminal_openings = len(terminals)
            ctx.inputs.append(Input(schedule, f"terminal_opening.{TERMINAL_KIND}",
                                    index=order.index(j.nr),
                                    name=f"terminal {j.nr} schedule {j.schedule_nr}"))
        else:
            raise ValueError(
                f"prj: junction {j.nr} has schedule {j.schedule_nr} of unit type "
                f"{ctx.schedule_units.get(j.schedule_nr)}; a junction's schedule can only "
                f"be its temperature (unit type {TEMPERATURE_UNITS})")
    if not terminals:
        return
    src, tgt = net.endpoints(TERMINAL_KIND)
    opening = f"terminal_opening.{TERMINAL_KIND}" if build.terminal_openings else None
    ctx.built.append(DuctTerminal(
        torch.tensor([j.terminal.area for j in terminals], dtype=F64),
        torch.tensor([j.terminal.loss for j in terminals], dtype=F64), src=src, tgt=tgt,
        temperature_key="T", opening_key=opening, kind=TERMINAL_KIND))
    ctx.drives.append(Stack.from_network(net, TERMINAL_KIND, g=ctx.g))
    if any(j.terminal.profile_nr for j in terminals):
        ctx.drives.append(Wind.from_network(net, TERMINAL_KIND, ambient=ctx.ambient,
                                            profiles=ctx.profiles))
    if any(wind_pressure):
        ctx.constant_wind_pressure[TERMINAL_KIND] = torch.tensor(wind_pressure, dtype=F64)
        ctx.drives.append(ConstantDrive(TERMINAL_KIND, f"wind_pressure.{TERMINAL_KIND}"))


@dataclass
class AirHandlerRecord:
    nr: int
    return_zone: int
    supply_zone: int
    recirculation: int               # path numbers
    outdoor_air: int
    exhaust: int
    name: str


def read_air_handlers(lines, count: int, out: list[AirHandlerRecord]) -> None:
    for k in range(count):
        tok = _record(lines, "simple AHS", count, k)
        nr, zr, zs, pr, ps, px = (int(v) for v in tok[:6])
        out.append(AirHandlerRecord(nr, zr, zs, pr, ps, px, tok[-1] if len(tok) > 6 else ""))
        lines.raw()                                 # description (may be blank)
    lines.expect_end("simple AHS")


SUPPLY_OR_RETURN_FLAG = 8
_IMPLICIT = ((16, RECIRCULATION, "recirculation"), (32, OUTDOOR_AIR, "outdoor-air"),
             (64, EXHAUST, "exhaust"))
_IMPLICIT_FLAGS = 16 | 32 | 64


def assign_air_handler_roles(handlers: list[AirHandlerRecord],
                             paths: list) -> dict[int, tuple[int, int]]:
    """Path number -> (AHS number, role) for every AHS path, checked against the flags,
    endpoints and order ContamX relies on (it ignores the AHS record's numbers and takes
    the implicit paths by their flags and position)."""
    roles: dict[int, tuple[int, int]] = {}
    by_nr = {p.nr: p for p in paths}
    implicit = [p.nr for p in paths if p.flags & _IMPLICIT_FLAGS]
    if implicit and [p.nr for p in paths[-len(implicit):]] != implicit:
        raise ValueError(
            f"prj: the implicit AHS paths {implicit} must come after every other path, as "
            f"ContamW writes them; ContamX takes them by position and mis-assigns them "
            f"otherwise (unsupported)")
    by_ahs = {h.nr: h for h in handlers}
    for h in handlers:
        for (flag, role, label), nr, ends in zip(
                _IMPLICIT, (h.recirculation, h.outdoor_air, h.exhaust),
                ((h.return_zone, h.supply_zone), (-1, h.supply_zone), (h.return_zone, -1)),
                strict=True):
            p = by_nr.get(nr)
            if p is None or not p.flags & flag or (p.from_zone, p.to_zone) != ends:
                raise ValueError(
                    f"prj: AHS {h.nr} names path {nr} as its {label} path, but that path is "
                    f"not flagged {flag} from zone {ends[0]} to zone {ends[1]}")
            if nr in roles:
                raise ValueError(f"prj: path {nr} is named by AHS {roles[nr][0]} and AHS "
                                 f"{h.nr}; each AHS needs its own implicit paths")
            roles[nr] = (h.nr, role)
    for p in paths:
        if p.flags & _IMPLICIT_FLAGS and p.nr not in roles:
            raise ValueError(f"prj: path {p.nr} is an implicit AHS path that no AHS names")
        if not p.ahs_nr:
            if p.flags & SUPPLY_OR_RETURN_FLAG:
                raise ValueError(f"prj: path {p.nr} is an AHS supply or return point "
                                 f"without an AHS")
            continue
        if p.nr in roles:
            raise ValueError(f"prj: path {p.nr} is an implicit AHS path and also names AHS "
                             f"{p.ahs_nr} as a supply or return point")
        h = by_ahs.get(p.ahs_nr)
        if h is None:
            raise KeyError(f"prj: path {p.nr} references AHS {p.ahs_nr}, which the 'simple "
                           f"AHS' section does not define")
        if p.from_zone == h.supply_zone:
            roles[p.nr] = (h.nr, SUPPLY)
        elif p.to_zone == h.return_zone:
            roles[p.nr] = (h.nr, RETURN)
        else:
            raise ValueError(f"prj: path {p.nr} of AHS {h.nr} neither leaves its supply "
                             f"zone {h.supply_zone} nor enters its return zone "
                             f"{h.return_zone}")
    return roles
