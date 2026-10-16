"""A documented subset of the MATPOWER version 2 `.m` case-file format.

A MATPOWER case file is a MATLAB function that fills a struct `mpc`. This reader parses the
text; it never runs MATLAB or Octave.

READ: `mpc.version` (must be `'2'`), `mpc.baseMVA`, `mpc.bus` (at least 13 columns),
`mpc.gen` (at least 10 columns) and `mpc.branch` (at least 11 columns). Extra columns
(solved-case or optimal-power-flow results) are ignored. Columns used (MATPOWER's names,
`.` for a column read past):

- bus: `BUS_I TYPE PD QD GS BS . VM VA BASE_KV`;
- gen: `GEN_BUS PG QG QMAX QMIN VG . GEN_STATUS`;
- branch: `F_BUS T_BUS BR_R BR_X BR_B . . . TAP SHIFT BR_STATUS`.

Bus `TYPE` 1 is a load bus (`"pq"`), 2 a voltage-controlled bus (`"pv"`), 3 the reference
bus (`"slack"`). A bus with nonzero `PD`/`QD` gets one `Load`; a bus with nonzero `GS`/`BS`
gets one `FixedShunt`. A generator or branch is in service when its status is `> 0`
(MATPOWER's rule); out-of-service elements are kept, flagged. Bus ids must be integers.

IGNORED because they do not change the power flow, recorded in file order in
`notes["ignored_fields"]`: `gencost`, `bus_name`, `gen_name`, `branch_name`, `areas`,
`gentype`, `genfuel`.

REFUSED BY NAME, because skipping them would silently change the network: `mpc.dcline`
(DC lines), isolated buses (`TYPE` 4, every one named), any other bus type, any other
`mpc.<field>`, and any statement that is neither the `function mpc = <name>` header nor a
whole-field assignment `mpc.<field> = ...`: an indexed assignment such as
`mpc.bus(:, 8) = 1;`, `define_constants`, `return`, and so on. A field assigned twice is
refused too.

SYNTAX. The file is UTF-8 text (a byte-order mark is skipped; other bytes are refused,
naming the file). `%` starts a comment except inside a single-quoted string; block comments
`%{ ... %}` are not supported (a line holding only `%{` or `%}` is refused with its number).
`...` continues a line (the rest of the physical line is dropped). Matrices `[ ... ]` may
span lines; a row ends at `;` or at the end of a line; values are separated by whitespace or
commas; `Inf` and `-Inf` are numbers. Cell arrays `{ ... }` are accepted only for ignored
fields. A row shorter than its table needs, a row of a different length from the table's
first row, or a token that is not a number, is refused with the line number. The network's
name is the function's name (`case9`), or the file's stem if the file has no `function`
header.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from noodl.apps.electrical.power.network import (
    Branch,
    Bus,
    FixedShunt,
    Generator,
    Load,
    PowerNetwork,
)

_IGNORED = ("gencost", "bus_name", "gen_name", "branch_name", "areas", "gentype", "genfuel")
_TABLES = {"bus": 13, "gen": 10, "branch": 11}
_REQUIRED = ("version", "baseMVA", "bus", "gen", "branch")
_BUS_TYPES = {1: "pq", 2: "pv", 3: "slack"}

_FUNCTION = re.compile(r"function\s+mpc\s*=\s*([A-Za-z]\w*)")
_ASSIGNMENT = re.compile(r"mpc\.([A-Za-z]\w*)\s*=\s*(.*)")
_SCALAR = re.compile(r"('[^']*'|[^\s;']+)\s*;?")
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_INFINITY = re.compile(r"([+-]?)inf", re.IGNORECASE)
_SEPARATORS = re.compile(r"[\s,]+")


def _code(line: str) -> tuple[str, bool]:
    """The code on one physical line without its comment, and whether it ends in `...`."""
    in_string = False
    for k, char in enumerate(line):
        if char == "'":
            in_string = not in_string
        elif not in_string:
            if char == "%":
                return line[:k], False
            if line.startswith("...", k):
                return line[:k], True
    return line, False


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """(first physical line number, code) for every line, with continuations joined."""
    lines: list[tuple[int, str]] = []
    pending: tuple[int, str] | None = None
    for number, raw in enumerate(text.splitlines(), start=1):
        code, continued = _code(raw)
        if pending is not None:
            number, code = pending[0], pending[1] + " " + code
        pending = (number, code) if continued else None
        if not continued:
            lines.append((number, code))
    if pending is not None:
        lines.append(pending)
    return lines


def _close(text: str, char: str) -> int:
    """The index of `char` in `text` outside single-quoted strings, or -1."""
    in_string = False
    for k, c in enumerate(text):
        if c == "'":
            in_string = not in_string
        elif c == char and not in_string:
            return k
    return -1


def _number(token: str, where: str) -> float:
    if _NUMBER.fullmatch(token):
        return float(token)
    infinity = _INFINITY.fullmatch(token)
    if infinity:
        return -math.inf if infinity.group(1) == "-" else math.inf
    raise ValueError(f"{where}: {token!r} is not a number")


def _matrix(
    field: str, segments: list[tuple[int, str]], min_cols: int, path: Path
) -> list[tuple[int, list[float]]]:
    """The rows of a numeric matrix as (line number, values), all of one length, at least
    `min_cols`."""
    rows: list[tuple[int, list[float]]] = []
    first: tuple[int, int] | None = None
    for number, text in segments:
        for piece in text.split(";"):
            tokens = [t for t in _SEPARATORS.split(piece.strip()) if t]
            if not tokens:
                continue
            where = f"{path}: line {number}: mpc.{field}"
            values = [_number(t, where) for t in tokens]
            if len(values) < min_cols:
                raise ValueError(
                    f"{where} row has {len(values)} columns; at least {min_cols} columns "
                    f"are required"
                )
            if first is None:
                first = (number, len(values))
            elif len(values) != first[1]:
                raise ValueError(
                    f"{where} row has {len(values)} columns, the row on line {first[0]} has "
                    f"{first[1]}; every row of a table needs the same number of columns"
                )
            rows.append((number, values))
    return rows


def _integer(value: float, what: str, where: str) -> int:
    if not float(value).is_integer():
        raise ValueError(f"{where}: {what} {value!r} is not an integer")
    return int(value)


def _not_understood(path: Path, number: int, statement: str) -> ValueError:
    return ValueError(
        f"{path}: line {number}: {statement!r} is not part of the MATPOWER case-file subset "
        f"this reader understands"
    )


def _statements(path: Path, text: str):
    """Yield (line number, field, opening bracket or '', value text or block segments)."""
    lines = _logical_lines(text)
    k = 0
    while k < len(lines):
        number, code = lines[k]
        statement = code.strip()
        k += 1
        if not statement:
            continue
        function = _FUNCTION.fullmatch(statement)
        if function:
            yield number, None, "", function.group(1)
            continue
        assignment = _ASSIGNMENT.fullmatch(statement)
        if not assignment:
            raise _not_understood(path, number, statement)
        field, rhs = assignment.group(1), assignment.group(2).strip()
        if rhs[:1] in ("[", "{"):
            opening = rhs[0]
            closing = "]" if opening == "[" else "}"
            segments: list[tuple[int, str]] = []
            body, body_number = rhs[1:], number
            while True:
                end = _close(body, closing)
                if end >= 0:
                    segments.append((body_number, body[:end]))
                    if body[end + 1 :].strip() not in ("", ";"):
                        raise _not_understood(path, body_number, statement)
                    break
                segments.append((body_number, body))
                if k == len(lines):
                    raise ValueError(
                        f"{path}: line {number}: mpc.{field}: {opening!r} is never closed"
                    )
                body_number, body = lines[k]
                k += 1
            yield number, field, opening, segments
        else:
            scalar = _SCALAR.fullmatch(rhs)
            if not scalar:
                raise _not_understood(path, number, statement)
            yield number, field, "", scalar.group(1)


def read_matpower(path) -> PowerNetwork:
    """Read the documented subset of a MATPOWER version 2 case file into a `PowerNetwork`,
    in the file's units (MW, MVAr, degrees, p.u. impedances on `baseMVA`)."""
    path = Path(path)
    name = ""
    fields: dict[str, tuple[int, str, object]] = {}
    ignored: list[str] = []
    seen_statement = False
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: the file is not UTF-8 text ({exc.reason})") from exc
    for number, raw in enumerate(text.splitlines(), start=1):
        if raw.strip() in ("%{", "%}"):
            raise ValueError(
                f"{path}: line {number}: block comments %{{ ... %}} are not supported; "
                f"start each comment line with %"
            )
    for number, field, opening, value in _statements(path, text):
        if field is None:
            if seen_statement:
                raise _not_understood(path, number, f"function mpc = {value}")
            name = value
            seen_statement = True
            continue
        seen_statement = True
        if field == "dcline":
            raise ValueError(
                f"{path}: line {number}: mpc.dcline: DC lines change the power flow and are "
                f"not modelled"
            )
        if field not in _IGNORED and field not in _REQUIRED:
            raise ValueError(
                f"{path}: line {number}: field mpc.{field} is not part of the MATPOWER "
                f"case-file subset this reader understands"
            )
        if field in fields or field in ignored:
            raise ValueError(f"{path}: line {number}: mpc.{field} is assigned more than once")
        if field in _IGNORED:
            ignored.append(field)
            continue
        if field in _TABLES and opening != "[":
            raise ValueError(
                f"{path}: line {number}: mpc.{field} must be a numeric matrix [ ... ]"
            )
        if field not in _TABLES and opening:
            raise ValueError(f"{path}: line {number}: mpc.{field} must be a scalar")
        fields[field] = (number, opening, value)

    missing = [f"mpc.{f}" for f in _REQUIRED if f not in fields]
    if missing:
        raise ValueError(
            f"{path}: {', '.join(missing)} missing; a MATPOWER case needs "
            f"{', '.join('mpc.' + f for f in _REQUIRED)}"
        )
    number, _, version = fields["version"]
    if version != "'2'":
        raise ValueError(
            f"{path}: line {number}: mpc.version is {version}; only MATPOWER case format "
            f"version 2 ('2') is read"
        )
    number, _, base = fields["baseMVA"]
    base_mva = _number(base, f"{path}: line {number}: mpc.baseMVA")
    tables = {
        f: _matrix(f, fields[f][2], cols, path) for f, cols in _TABLES.items()
    }

    buses: list[Bus] = []
    loads: list[Load] = []
    shunts: list[FixedShunt] = []
    isolated: list[int] = []
    for number, row in tables["bus"]:
        where = f"{path}: line {number}: mpc.bus"
        bus_id = _integer(row[0], "bus id", where)
        bus_type, pd, qd, gs, bs = row[1], row[2], row[3], row[4], row[5]
        if bus_type == 4:
            isolated.append(bus_id)
            continue
        if bus_type not in _BUS_TYPES:
            raise ValueError(
                f"{where}: bus {bus_id} has type {bus_type:g}; MATPOWER's bus types are 1 "
                f"(PQ), 2 (PV), 3 (reference) and 4 (isolated)"
            )
        buses.append(Bus(bus_id, _BUS_TYPES[int(bus_type)], row[7], row[8], row[9]))
        if pd != 0 or qd != 0:
            loads.append(Load(bus_id, pd, qd))
        if gs != 0 or bs != 0:
            shunts.append(FixedShunt(bus_id, gs, bs))
    if isolated:
        raise ValueError(
            f"{path}: isolated buses (type 4) are not modelled: {isolated}"
        )

    generators = []
    for number, row in tables["gen"]:
        where = f"{path}: line {number}: mpc.gen"
        generators.append(Generator(
            bus=_integer(row[0], "generator bus", where),
            p_mw=row[1], q_mvar=row[2], vm_set=row[5], in_service=row[7] > 0,
            q_max_mvar=row[3], q_min_mvar=row[4],
        ))
    branches = []
    for number, row in tables["branch"]:
        where = f"{path}: line {number}: mpc.branch"
        branches.append(Branch(
            from_bus=_integer(row[0], "from bus", where),
            to_bus=_integer(row[1], "to bus", where),
            r=row[2], x=row[3], b=row[4], ratio=row[8], shift_deg=row[9],
            in_service=row[10] > 0,
        ))

    try:
        return PowerNetwork(
            buses=tuple(buses), branches=tuple(branches), generators=tuple(generators),
            loads=tuple(loads), shunts=tuple(shunts), base_mva=base_mva,
            name=name or path.stem, notes={"ignored_fields": ignored},
        )
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
