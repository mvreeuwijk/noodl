"""CONTAM 3.4 project (.prj) reader -- the documented subset of TN 1887r1 Appendix A.

What is read: run control (ambient conditions, g), species, levels, wind pressure profiles,
airflow elements (power-law family, quadratic family, doorways, dampers, constant-flow fans
and the cubic fan), zones, initial zone concentrations, airflow paths, source/sink elements
and source/sinks of the constant, cutoff, decaying and burst types. Everything else is
skipped to its -999 terminator; a RECORD that references a skipped section by a nonzero
index (schedule, control, filter, kinetic reaction, AHS) raises ValueError naming it, as does
a CFD or 1-D zone, a duct network, an unsupported element type, a constant wind pressure
(wPset with no profile) and a fan curve on a path with mult != 1 (spec sections 8 and 14).

The control-node section is the motivating case for "skipped, not refused": NIST's own sample
projects carry dozens of sensor and logger control nodes that no zone and no path points at,
and those projects must load.

Conventions (spec section 14): mass flow; orifice/leak/crack turbulence coefficient
`turb` is C_d A sqrt(2) so C = mult turb sqrt(rho); fcn/test/conn/stair/shaft C = mult turb;
qcn C = mult rho turb; laminar `lam` gives F = lam (rho/mu) dp, so the laminar/turbulent
crossing is dp_t = (C_mass mu / (lam rho))^(1/(1-n)). One element KIND per CONTAM element
(`pl_<nr>`, `qf_<nr>`, `door_<nr>`, `bd_<nr>`, `fan_<nr>`), so every element keeps its own
transition and the transport layers advect over all of them.

Dtype (Ruling R7): `torch.get_default_dtype()` is float32 in this repository, and a CONTAM
parity comparison is a float64 exercise. The `Network`, every element parameter and every
tensor `Project` carries are built with an EXPLICIT float64 dtype -- never by letting a
Python float fall through to the default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tellegen.apps.building.thermal import R_AIR, RHO_0, species_layer
from tellegen.drives import Stack, Wind, WindProfile
from tellegen.elements import Damper, FanCurve, FixedFlow, PowerLaw, Quadratic
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.model import Model
from tellegen.topology import Network

F64 = torch.float64
MU_0 = 1.81625e-5

# --------------------------------------------------------------------------------------
# RECORD LAYOUTS. Every field position this reader depends on lives here and nowhere else;
# all of it is verified against NIST's `contamxpy` 0.0.9 sample projects (13-15 Sep 2026)
# unless a comment says otherwise.
#
#   header line 1   `ContamW <version> <flag>`, then the project's own file name.
#   run control     labelled comment lines each followed by one or more data lines, to -999.
#                   `! Ta Pb Ws Wd rh day u..` heads SEVERAL data lines, one per simulation
#                   case, each labelled by its OWN trailing comment; the ambient state is
#                   the one labelled `steady simulation` (Ruling R8). `!dens grav` heads
#                   `rho g`.
#   section header  `N ! <name>:`; the section's N records then run to a `-999` line.
#                   `! contaminants:` is the exception: N index lines and NO terminator.
#   species         `# s t molwt mdiam edens decay Dm CCdef Cp Kuv u[5] name` + description.
#   levels          `# refHt delHt ni u name`, then `ni` icon lines (comments interleaved).
#   wind profile    `# npts type name`, one description line (may be blank), `npts` rows
#                   `angle cp`; the last angle is 360 and repeats the Cp at 0.
#   flow element    `# icon dtype name`, ONE description line (often blank -- it must be
#                   CONSUMED, not skipped), then the data line(s).
#     plr_orfc      `lam turb expt area dia coef Re u_A u_D`
#     plr_leak3     `lam turb expt coef pres area1 area2 area3 u u u u`
#     fan_cmf       `Flow u_F`
#     dor_door      10 fields: `lam turb expt dTmin ht wd cd u u u`. VERIFIED against
#                   `reg_solverContTrace-mz-MH-trans-3day.prj` lines 704-706, the one NIST
#                   sample that carries a doorway:
#                       18 27 dor_door IntDoor-open
#                       open interior door
#                        0.148966 2.54558 0.5 0.01 2 0.9 1 0 0 0
#                   Reading that as ht = 2, wd = 0.9, cd = 1 reproduces the record's OWN
#                   turbulent coefficient through CONTAM's identity turb = cd A sqrt(2):
#                   1 * (2 * 0.9) * sqrt(2) = 2.545584, against the file's 2.54558 -- six
#                   significant figures, so fields 4, 5 and 6 are pinned by NIST's numbers.
#     dor_pl2       `lam turb expt dH    ht wd cd u u`
#                   -- the two doorway types differ ONLY in field 3 (a minimum temperature
#                   difference versus the half-separation of the two openings); `ht wd cd`
#                   sit at 4, 5, 6 in both, which is why the code below has ONE assignment
#                   for them (Ruling R9). The `dor_pl2` half is documented from TN 1887r1
#                   Appendix A and from the milestone plan's own dictated record: no NIST
#                   sample carries a dor_pl2, so only `dor_door` is file-verified.
#     plr_bdq/bdf   `lam Cp xp Cn xn ...`
#   zone            19 fields: `# f s# c# k# l# relHt Vol T0 P0 name clr uH uT uP uV axs
#                   cdvf cfd`. `clr` is not in ContamW's own header comment but IS written.
#   path            30 fields: `# f n# m# e# f# w# a# s# c# l# X Y relHt mult wPset wPmod
#                   wazm Fahs Xmax Xmin icn dir clr u[4] cdvf cfd`. Same story: the header
#                   comment omits `clr` (always -1 in every sample), so the trailing
#                   `cdvf cfd` sit at 28 and 29, not at 27 and 28. A nonzero cdvf or cfd
#                   appends a name / four data fields, which is why the field COUNT is
#                   checked. `n# == -1` is ambient; `wPmod` is Ch; `w#` is the 1-based wind
#                   pressure profile NUMBER.
#   source/sink     `# z# e# s# c# mult CC0 (X,Y,H)min (X,Y,H)max clr u[1] cdvf cfd`.
# --------------------------------------------------------------------------------------

_ZONE_FIELDS = 19
_PATH_FIELDS = 30
_STEADY_LABEL = "steady simulation"

_POWERLAW_SQRT_RHO = {"plr_orfc", "plr_leak1", "plr_leak2", "plr_leak3", "plr_crack"}
_POWERLAW_MASS = {"plr_fcn", "plr_test1", "plr_test2", "plr_conn", "plr_stair", "plr_shaft"}
_POWERLAW_VOLUME = {"plr_qcn"}
_POWERLAW = _POWERLAW_SQRT_RHO | _POWERLAW_MASS | _POWERLAW_VOLUME
_QUADRATIC = {"qfr_qab", "qfr_fab", "qfr_crack", "qfr_test2"}
_DOOR = {"dor_door", "dor_pl2"}
_DAMPER = {"plr_bdq", "plr_bdf"}
_FAN_CONST = {"fan_cmf", "fan_cvf"}
_SOURCE_TYPES = {"ccf", "cut", "eds", "brs"}

# Sections whose records would silently change the physics if ignored. Ducts and the simple
# air-handling system are networks in their own right, not attributes of a zone or a path, so
# a project that HAS one cannot be read as the documented subset -- the refusal is on the
# section, naming it and its record count.
_REFUSED_SECTIONS = ("duct junctions", "duct segments", "duct elements", "simple AHS")


@dataclass
class PrjElement:
    nr: int
    dtype: str
    name: str
    data: list[list[float]]          # data lines, tokens as floats


@dataclass
class PrjPath:
    nr: int
    from_zone: int                   # -1 = ambient
    to_zone: int
    element_nr: int
    z: float
    mult: float
    wPset: float
    Ch: float
    azimuth: float
    profile_nr: int
    edge_columns: list[int] = field(default_factory=list)   # columns of q for this path


@dataclass
class PrjSource:
    nr: int
    zone_nr: int
    element_nr: int
    source_type: str                 # ccf | cut | eds | brs
    params: list[float]
    mult: float


@dataclass
class Project:
    net: Network
    elements: list
    drives: list
    zones: list[str]
    ambient: str
    paths: list[PrjPath]
    species: list[str]
    T_zone: torch.Tensor
    zone_volumes: torch.Tensor
    g: float
    ambient_conditions: dict[str, float]
    profiles: dict[int, WindProfile]
    x0: torch.Tensor                  # (n_zones, K)
    sources: list[PrjSource]
    kinds: list[str]
    # CONTAM numbers zones; it does not promise they are 1..N in file order, and Task 13
    # resolves a source's `z#` through this map rather than assuming `zones[nr - 1]`
    # (Ruling R5).
    zone_nr_to_name: dict[int, str]

    def path_flows(self, q: torch.Tensor) -> torch.Tensor:
        """Net mass flow per path in path-number order (a doorway sums its two edges).

        `q` is the air layer's flow vector, which `PotentialFlowLayer` lays out in element-
        KIND blocks, in `self.elements` order -- NOT in path order (Ruling R2). Each path's
        `edge_columns` already carries that translation.
        """
        cols = [torch.as_tensor(p.edge_columns, dtype=torch.long) for p in self.paths]
        return torch.stack([q[..., c].sum(-1) for c in cols], dim=-1)


class _Lines:
    """Cursor over the file's lines with the two ways CONTAM records are laid out."""

    def __init__(self, text: str) -> None:
        self.lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
        self.i = 0

    def raw(self) -> str:
        if self.i >= len(self.lines):
            raise ValueError("prj: file ended in the middle of a record (no '* end project')")
        line = self.lines[self.i]
        self.i += 1
        return line

    def record(self) -> list[str]:
        """Next non-comment, non-blank line, split on whitespace, trailing `!` comment cut."""
        while True:
            line = self.raw()
            body = line.split("!", 1)[0].strip()
            if body:
                return body.split()

    def skip_section(self) -> None:
        while self.raw().strip() != "-999":
            pass

    def expect_end(self, where: str) -> None:
        tok = self.record()
        if tok != ["-999"]:
            raise ValueError(f"prj: expected -999 after {where}, found {tok}")


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _section_header(lines: _Lines) -> tuple[int, str]:
    line = lines.raw()
    while not line.strip():
        line = lines.raw()
    count, sep, name = line.partition("!")
    if not sep or not count.split():
        raise ValueError(f"prj: expected a 'N ! <section>:' header, found {line.strip()!r}")
    return int(count.split()[0]), name.strip().rstrip(":").strip()


def _read_ambient_block(lines: _Lines) -> dict[str, float] | None:
    """Consume the data lines under one `! Ta Pb Ws Wd` header; return the steady one.

    CONTAM writes one such line per simulation case (steady, wind pressure test, ...), each
    labelled by its own trailing comment. Only the `steady simulation` line is the project's
    ambient state -- taking the first or the last of them is wrong (Ruling R8).
    """
    found: dict[str, float] | None = None
    while lines.i < len(lines.lines):
        body, _, comment = lines.lines[lines.i].partition("!")
        tok = body.split()
        if len(tok) < 4 or not all(_is_number(t) for t in tok[:4]):
            break                                   # a comment line, or the next block
        lines.i += 1
        if _STEADY_LABEL in comment:
            found = {
                "Ta": float(tok[0]), "Pb": float(tok[1]),
                "Ws": float(tok[2]), "Wd": float(tok[3]),
            }
    return found


def _read_run_control(lines: _Lines) -> tuple[dict[str, float], float]:
    ambient: dict[str, float] | None = None
    g: float | None = None
    while True:
        line = lines.raw()
        stripped = line.strip()
        if stripped == "-999":
            break
        if not stripped.startswith("!"):
            continue
        if stripped[1:].split()[:1] == ["Ta"]:
            ambient = _read_ambient_block(lines) or ambient
        elif stripped.startswith("!dens"):
            g = float(lines.raw().split()[1])
    if ambient is None:
        raise ValueError(
            "prj: the run-control section has no 'Ta Pb Ws Wd' data line labelled "
            f"{_STEADY_LABEL!r}; that line IS the project's ambient state, and the other "
            "cases (a wind pressure test, say) are not interchangeable with it"
        )
    if g is None:
        raise ValueError("prj: the run-control section lacks the '!dens grav' line")
    return ambient, g


def _read_contaminants(lines: _Lines, count: int) -> None:
    """Consume the `N ! contaminants:` section: N species INDICES, then no terminator.

    The indices are whitespace-separated and CONTAM writes them all on ONE line (`   1 2 3`),
    so this counts TOKENS, not lines. A per-line loop happens to work for a single-species
    project and then eats the `N ! species:` header of every other one -- verified on
    `reg_solverContTrace-mz-MH-trans-3day.prj` (2 contaminants, `   1 2`) and
    `test_OneFloorWpcAddMf.prj` (3, `   1 2 3`). The section has no -999 of its own: the
    species section follows it directly.
    """
    seen = 0
    while seen < count:
        seen += len(lines.record())
    if seen != count:
        raise ValueError(
            f"prj: the 'contaminants' section promises {count} species indices but its "
            f"lines carry {seen}"
        )


def _read_species(lines: _Lines, count: int, species: list[str]) -> None:
    for _ in range(count):
        tok = lines.record()
        species.append(tok[-1])                     # name is the last field of the line
        lines.raw()                                 # the species' description line
    lines.expect_end("species")


def _read_initial_concentrations(
    lines: _Lines, count: int, x0_rows: dict[int, list[float]]
) -> None:
    """One row per zone, `Z# x1 .. xK`; the section's count is VALUES, not rows.

    Every other section's header count is a record count, but this one's is zones times
    species: `test_OneFloorWpcAddMf.prj:171` heads TWO zone rows with `6` (2 zones, 3
    species). So read rows to the terminator and use the count as the integrity check it
    is. A single-species project hides the difference, which is why this was invisible on
    the shipped fixtures.
    """
    values = 0
    while True:
        tok = lines.record()
        if tok == ["-999"]:
            break
        x0_rows[int(tok[0])] = [float(v) for v in tok[1:]]
        values += len(tok) - 1
    if values != count:
        raise ValueError(
            f"prj: the 'initial zone concentrations' section promises {count} values "
            f"(zones x species) but its rows carry {values}"
        )


def _read_levels(lines: _Lines, count: int, levels: dict[int, float]) -> None:
    for _ in range(count):
        tok = lines.record()
        nr, refht, ni = int(tok[0]), float(tok[1]), int(tok[3])
        levels[nr] = refht
        skipped = 0
        while skipped < ni:                         # icon lines, with comments interleaved
            if lines.raw().split("!", 1)[0].strip():
                skipped += 1
    lines.expect_end("levels")


def _read_profiles(lines: _Lines, count: int, profiles: dict[int, WindProfile]) -> None:
    for _ in range(count):
        tok = lines.record()
        nr, npts = int(tok[0]), int(tok[1])
        lines.raw()                                 # description (may be blank)
        pts = [lines.record() for _ in range(npts)]
        profiles[nr] = WindProfile([float(p[0]) for p in pts], [float(p[1]) for p in pts])
    lines.expect_end("wind pressure profiles")


def _read_source_elements(
    lines: _Lines, count: int, source_elements: dict[int, tuple[str, list[float]]]
) -> None:
    for _ in range(count):
        tok = lines.record()
        nr, dtype = int(tok[0]), tok[2]
        lines.raw()                                 # description (may be blank)
        source_elements[nr] = (dtype, [float(v) for v in lines.record()])
    lines.expect_end("source/sink elements")


def _read_flow_elements(lines: _Lines, count: int, elements: dict[int, PrjElement]) -> None:
    for _ in range(count):
        tok = lines.record()
        nr, dtype, ename = int(tok[0]), tok[2], tok[3] if len(tok) > 3 else ""
        lines.raw()                                 # description (may be blank)
        data = [[float(v) for v in lines.record()]]
        if dtype == "fan_fan":
            # Cubic fan curve: the second data line ends with the number of measured
            # points, each of which is a line of its own. Documented from TN 1887r1
            # Appendix A; no NIST sample carries a fan_fan to verify it against.
            second = [float(v) for v in lines.record()]
            data.append(second)
            for _ in range(int(second[4])):
                data.append([float(v) for v in lines.record()])
        elif dtype.startswith("csf_") or dtype == "sup_afe":
            raise ValueError(f"prj: element {nr} has unsupported type {dtype!r}")
        elements[nr] = PrjElement(nr, dtype, ename, data)
    lines.expect_end("flow elements")


def _read_zones(lines: _Lines, count: int, zones: list[tuple]) -> None:
    for _ in range(count):
        tok = lines.record()
        nr = int(tok[0])
        if len(tok) != _ZONE_FIELDS:
            raise ValueError(
                f"prj: zone {nr} line has {len(tok)} fields, not the {_ZONE_FIELDS} of a "
                f"plain zone; zone names with spaces, and the extra fields a value file or "
                f"a CFD zone writes, are unsupported: {' '.join(tok)}"
            )
        ps, pc, pk, pl = int(tok[2]), int(tok[3]), int(tok[4]), int(tok[5])
        if ps:
            raise ValueError(f"prj: zone {nr} references schedule {ps} (unsupported)")
        if pc:
            raise ValueError(f"prj: zone {nr} references control node {pc} (unsupported)")
        if pk:
            raise ValueError(f"prj: zone {nr} references kinetic reaction {pk} (unsupported)")
        rel_ht, vol, T0 = float(tok[6]), float(tok[7]), float(tok[8])
        zname = tok[10]
        rest = tok[11:]
        axs, cdvf, cfd = int(rest[5]), int(rest[6]), int(rest[7])
        if axs:
            raise ValueError(f"prj: zone {nr} is a 1-D convection/diffusion zone (unsupported)")
        if cfd:
            raise ValueError(f"prj: zone {nr} is a CFD zone (unsupported)")
        if cdvf:
            raise ValueError(f"prj: zone {nr} reads a continuous values file (unsupported)")
        zones.append((nr, zname, rel_ht, vol, T0, pl))
    lines.expect_end("zones")


def _read_paths(
    lines: _Lines, count: int, paths: list[PrjPath], levels: dict[int, float]
) -> None:
    for _ in range(count):
        tok = lines.record()
        nr = int(tok[0])
        if len(tok) != _PATH_FIELDS:
            raise ValueError(
                f"prj: path {nr} line has {len(tok)} fields, not the {_PATH_FIELDS} of a "
                f"plain path; a continuous values file or a CFD path writes extra name and "
                f"data fields, and neither is supported: {' '.join(tok)}"
            )
        pzn, pzm, pe, pf, pw, pa, ps, pc, pld = (int(v) for v in tok[2:11])
        for label, val in (("filter", pf), ("AHS", pa), ("schedule", ps), ("control node", pc)):
            if val:
                raise ValueError(f"prj: path {nr} references {label} {val} (unsupported)")
        rel_ht, mult, wPset, wPmod, wazm = (float(v) for v in tok[13:18])
        cdvf, cfd = int(float(tok[28])), int(float(tok[29]))
        if cdvf or cfd:
            raise ValueError(
                f"prj: path {nr} uses a values file (cdvf {cdvf}) or CFD (cfd {cfd}) "
                f"(unsupported)"
            )
        if pw == 0 and wPset != 0.0:
            raise ValueError(
                f"prj: path {nr} has a constant wind pressure {wPset} (unsupported; use a "
                f"profile)"
            )
        if pld not in levels:
            raise KeyError(f"prj: path {nr} sits on level {pld}, which the file never defines")
        paths.append(
            PrjPath(nr, pzn, pzm, pe, levels[pld] + rel_ht, mult, wPset, wPmod, wazm, pw)
        )
    lines.expect_end("flow paths")


def _read_sources(
    lines: _Lines,
    count: int,
    sources: list[PrjSource],
    source_elements: dict[int, tuple[str, list[float]]],
) -> None:
    for _ in range(count):
        tok = lines.record()
        nr, pz, pe, ps, pc = (int(v) for v in tok[:5])
        if ps or pc:
            raise ValueError(
                f"prj: source {nr} references schedule {ps} / control node {pc} (unsupported)"
            )
        try:
            dtype, params = source_elements[pe]
        except KeyError as exc:
            raise KeyError(
                f"prj: source {nr} references source/sink element {pe}, which the "
                f"'source/sink elements' section does not define"
            ) from exc
        stype = dtype.split("_")[-1] if "_" in dtype else dtype
        if stype not in _SOURCE_TYPES:
            raise ValueError(f"prj: source {nr} uses element type {dtype!r} (unsupported)")
        sources.append(PrjSource(nr, pz, pe, stype, params, float(tok[5])))
    lines.expect_end("source/sinks")


def _kind_of(nr: int, dtype: str) -> str:
    """One element KIND per CONTAM element, so each keeps its own laminar transition."""
    if dtype in _POWERLAW:
        return f"pl_{nr}"
    if dtype in _QUADRATIC:
        return f"qf_{nr}"
    if dtype in _DOOR:
        return f"door_{nr}"
    if dtype in _DAMPER:
        return f"bd_{nr}"
    if dtype in _FAN_CONST or dtype == "fan_fan":
        return f"fan_{nr}"
    raise ValueError(f"prj: element {nr} has unsupported type {dtype!r}")


def read_prj(path) -> Project:
    text = Path(path).read_text()
    lines = _Lines(text)
    header = lines.raw().split()
    if not header or header[0] != "ContamW":
        raise ValueError(f"prj: not a ContamW project file (first line {header!r})")
    lines.raw()                                   # the project's own file name line
    ambient_conditions, g = _read_run_control(lines)

    species: list[str] = []
    levels: dict[int, float] = {}
    profiles: dict[int, WindProfile] = {}
    elements: dict[int, PrjElement] = {}
    source_elements: dict[int, tuple[str, list[float]]] = {}
    zones: list[tuple[int, str, float, float, float, int]] = []   # nr, name, relHt, vol, T0, level
    x0_rows: dict[int, list[float]] = {}
    paths: list[PrjPath] = []
    sources: list[PrjSource] = []

    while lines.i < len(lines.lines):
        line = lines.lines[lines.i].strip()
        if line.startswith("*"):
            break
        if not line:
            lines.i += 1
            continue
        count, name = _section_header(lines)
        if name == "contaminants":
            _read_contaminants(lines, count)
        elif name == "species":
            _read_species(lines, count, species)
        elif name == "levels plus icon data":
            _read_levels(lines, count, levels)
        elif name == "wind pressure profiles":
            _read_profiles(lines, count, profiles)
        elif name == "source/sink elements":
            _read_source_elements(lines, count, source_elements)
        elif name == "flow elements":
            _read_flow_elements(lines, count, elements)
        elif name == "zones":
            _read_zones(lines, count, zones)
        elif name == "initial zone concentrations":
            _read_initial_concentrations(lines, count, x0_rows)
        elif name == "flow paths":
            _read_paths(lines, count, paths, levels)
        elif name == "source/sinks":
            _read_sources(lines, count, sources, source_elements)
        elif name in _REFUSED_SECTIONS and count:
            raise ValueError(
                f"prj: section {name!r} has {count} records; duct networks and air-handling "
                f"systems are unsupported"
            )
        else:
            lines.skip_section()

    return _build(
        ambient_conditions=ambient_conditions, g=g, species=species, levels=levels,
        profiles=profiles, elements=elements, zones=zones, x0_rows=x0_rows, paths=paths,
        sources=sources,
    )


def _build(*, ambient_conditions, g, species, levels, profiles, elements, zones, x0_rows,
           paths, sources) -> Project:
    """Turn the parsed records into a `Network`, elements, drives and a `Project`."""
    net = Network(dtype=F64)
    ambient = "ambient"
    net.add_node(ambient, z_ref=0.0, volume=0.0, T0=ambient_conditions["Ta"])
    zone_names: list[str] = []
    zone_nr_to_name: dict[int, str] = {}
    T_list, V_list = [], []
    for nr, zname, rel_ht, vol, T0, pl in zones:
        if pl not in levels:
            raise KeyError(f"prj: zone {nr} sits on level {pl}, which the file never defines")
        net.add_node(zname, z_ref=levels[pl] + rel_ht, volume=vol, T0=T0, heat_capacity=0.0)
        zone_names.append(zname)
        zone_nr_to_name[nr] = zname
        T_list.append(T0)
        V_list.append(vol)

    def node_of(z: int) -> str:
        if z == -1:
            return ambient
        try:
            return zone_nr_to_name[z]
        except KeyError as exc:
            raise KeyError(
                f"prj: a path references zone {z}, which the file never defines"
            ) from exc

    kind_of: dict[int, str] = {}
    for nr in sorted({p.element_nr for p in paths}):
        try:
            el = elements[nr]
        except KeyError as exc:
            raise KeyError(
                f"prj: a path references flow element {nr}, which the 'flow elements' "
                f"section does not define"
            ) from exc
        kind_of[nr] = _kind_of(nr, el.dtype)

    built: list = []
    per_kind_C: dict[str, list[float]] = {}
    per_kind_n: dict[str, list[float]] = {}
    per_kind_dpt: dict[str, list[float]] = {}
    per_kind_quad: dict[str, list[tuple[float, float]]] = {}
    path_edges: list[list[int]] = []          # per path, its NETWORK edge columns
    column = 0
    for p in paths:
        el = elements[p.element_nr]
        kind = kind_of[p.element_nr]
        d = el.dtype
        src, tgt = node_of(p.from_zone), node_of(p.to_zone)
        wind_attrs = {}
        if p.profile_nr:
            wind_attrs = {"Ch": p.Ch, "azimuth": p.azimuth, "profile": float(p.profile_nr)}
        cols: list[int] = []
        first = el.data[0]
        if d in _DOOR:
            # `ht wd cd` sit at fields 4, 5, 6 for BOTH doorway types; only field 3 differs
            # (dor_door's dTmin versus dor_pl2's half-separation dH). Verified for dor_door
            # against NIST's own record and its turb = cd A sqrt(2) identity; see the layout
            # block at the top of this module (Ruling R9).
            ht, wd, cd = first[4], first[5], first[6]
            dh = (2.0 * ht / 9.0) if d == "dor_door" else first[3]
            area = wd * ht / 2.0
            for off in (-dh, dh):
                net.add_edge(src, tgt, kind=kind, z_path=p.z + off, area=area, Cd=cd,
                             **wind_attrs)
                per_kind_C.setdefault(kind, []).append(
                    p.mult * cd * area * math.sqrt(2.0 * RHO_0)
                )
                per_kind_n.setdefault(kind, []).append(0.5)
                cols.append(column)
                column += 1
            path_edges.append(cols)
            continue
        net.add_edge(src, tgt, kind=kind, z_path=p.z, **wind_attrs)
        cols.append(column)
        column += 1
        path_edges.append(cols)
        if d in _POWERLAW:
            lam, turb, expt = first[0], first[1], first[2]
            if d in _POWERLAW_SQRT_RHO:
                C = turb * math.sqrt(RHO_0)
            elif d in _POWERLAW_VOLUME:
                C = RHO_0 * turb
            else:
                C = turb
            per_kind_C.setdefault(kind, []).append(p.mult * C)
            per_kind_n.setdefault(kind, []).append(expt)
            dpt = (
                (C * MU_0 / (lam * RHO_0)) ** (1.0 / (1.0 - expt))
                if (lam > 0 and expt < 1)
                else 1e-3
            )
            per_kind_dpt.setdefault(kind, []).append(dpt)
        elif d in _QUADRATIC:
            a, b = first[0], first[1]
            if d == "qfr_qab":
                a, b = a / RHO_0, b / RHO_0**2
            per_kind_quad.setdefault(kind, []).append((a / p.mult, b / p.mult**2))
        elif d in _DAMPER:
            lam, Cp, xp, Cn, xn = first[:5]
            scale = RHO_0 if d == "plr_bdq" else 1.0
            built.append(
                Damper(
                    torch.tensor(p.mult * scale * Cp, dtype=F64),
                    torch.tensor(xp, dtype=F64),
                    torch.tensor(p.mult * scale * Cn, dtype=F64),
                    torch.tensor(xn, dtype=F64),
                    kind=kind,
                )
            )
        elif d in _FAN_CONST:
            flow = first[0] * (RHO_0 if d == "fan_cvf" else 1.0)
            built.append(FixedFlow(torch.tensor(p.mult * flow, dtype=F64), kind=kind))
        elif d == "fan_fan":
            if p.mult != 1.0:
                raise ValueError(
                    f"prj: path {p.nr} scales fan curve {p.element_nr} by mult {p.mult} "
                    f"(unsupported)"
                )
            built.append(
                FanCurve(
                    torch.tensor(el.data[1][:4], dtype=F64),
                    torch.tensor(first[4], dtype=F64),
                    kind=kind,
                )
            )

    # Power-law kinds: one PowerLaw per kind with per-edge C, n and the kind's (shared)
    # transition. The transition is the element's own, so it is the same for every edge
    # of the kind; edges only differ by `mult`.
    for kind, Cs in per_kind_C.items():
        ns = per_kind_n[kind]
        dpt = per_kind_dpt.get(kind, [1e-3])[0]
        built.append(
            PowerLaw(
                torch.tensor(Cs, dtype=F64), torch.tensor(ns, dtype=F64),
                dp_transition=dpt, kind=kind,
            )
        )
    for kind, ab in per_kind_quad.items():
        built.append(
            Quadratic(
                torch.tensor([x[0] for x in ab], dtype=F64),
                torch.tensor([x[1] for x in ab], dtype=F64),
                kind=kind,
            )
        )
    kinds = [el.kind for el in built]

    # Ruling R2: `PotentialFlowLayer` concatenates `q` in element-KIND blocks, in
    # `built` order -- NOT in path order, which is the network's own edge order. The two
    # coincide only for a project with a single element kind, so translate here, once.
    layout: dict[int, int] = {}
    offset = 0
    for el in built:
        for i, col in enumerate(net.edge_index(el.kind).tolist()):
            layout[col] = offset + i
        offset += net.edge_index(el.kind).shape[0]
    for p, cols in zip(paths, path_edges, strict=True):
        p.edge_columns = [layout[c] for c in cols]

    drives: list = []
    for el in built:
        if isinstance(el, FixedFlow):
            continue                              # a fixed flow ignores dp; a drive is moot
        drives.append(Stack.from_network(net, el.kind, g=g))
        # Ruling R3: profiles are keyed by CONTAM's own profile NUMBER, and the lookup is
        # deliberately strict -- an edge naming a number no profile carries raises inside
        # `Wind.from_network`, naming the edge and the number. That is why the drive is
        # built whenever an edge names a profile, even when `profiles` is empty: a missing
        # profiles section must be loud, not a silently vanishing wind pressure.
        if bool((net.edge_attr("profile", el.kind, default=0.0) > 0).any()):
            drives.append(Wind.from_network(net, el.kind, ambient=ambient, profiles=profiles))

    K = max(1, len(species))
    x0 = torch.zeros(len(zone_names), K, dtype=F64)
    for i, (nr, *_rest) in enumerate(zones):
        if nr in x0_rows:
            row = x0_rows[nr]
            x0[i, : len(row)] = torch.tensor(row, dtype=F64)

    return Project(
        net=net, elements=built, drives=drives, zones=zone_names, ambient=ambient, paths=paths,
        species=species, T_zone=torch.tensor(T_list, dtype=F64),
        zone_volumes=torch.tensor(V_list, dtype=F64), g=g,
        ambient_conditions=ambient_conditions, profiles=profiles, x0=x0, sources=sources,
        kinds=kinds, zone_nr_to_name=zone_nr_to_name,
    )


def project_to_model(project: Project, *, ambient: dict | None = None, species: bool = True,
                     scheme: str = "implicit"):
    """Air layer + species layer (no thermal layer: a .prj carries no thermal data), the
    initial state and the drivers for the project's ambient conditions (or `ambient`)."""
    amb = dict(project.ambient_conditions)
    if ambient:
        amb.update({k: v for k, v in ambient.items() if k in ("Ta", "Pb", "Ws", "Wd")})
    net = project.net
    air = PotentialFlowLayer(net, "air", project.elements, drives=project.drives,
                             boundary=[project.ambient], quantity="pressure", unit="Pa")
    layers: dict = {"air": air}
    if species and project.species:
        layers["species"] = species_layer(
            net, ambient=project.ambient, flow_kinds=tuple(project.kinds),
            n_species=len(project.species), scheme=scheme,
        )
    model = Model(net, layers)
    T = net.node_attr("T0")
    T[net.node_index(project.ambient)] = amb["Ta"]
    rho = amb["Pb"] / (R_AIR * T)
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=F64),
        "rho": rho,
        "rho_amb": rho[net.node_index(project.ambient)],
        "V_met": torch.tensor(amb["Ws"], dtype=F64),
        "theta_w": torch.tensor(amb["Wd"], dtype=F64),
    }
    state: dict = {}
    if "species" in layers:
        K = len(project.species)
        drivers["species.x_boundary"] = torch.zeros(1, K, dtype=F64)
        state["species.x"] = project.x0.clone()
    return model, state, drivers
