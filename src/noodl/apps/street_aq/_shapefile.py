"""A minimal, dependency-free reader/writer for PolyLine ESRI Shapefiles with dBase III
attribute tables (`struct`/plain file I/O only -- no GIS library).

Private to `noodl.apps.street_aq`: used to read and write SIRANE street networks, which are
always a single shape type (PolyLine, `shape_type=3`) of single-part, two-point (straight
street segment) records. The byte layout follows the ESRI Shapefile Technical Description
(1998) and dBase III, checked against the South Kensington deck's own
`RESEAU/SIRANE_FINAL.{shp,shx,dbf,prj}`.

This is NOT a general Shapefile library: only PolyLine (type 3), only one part per record,
and `read_polylines` refuses a record with anything other than exactly two points, by name,
because a SIRANE street is a straight line between two nodes -- see the module's own
`read_polylines` docstring. `.shx` (the spatial index) and `.prj` (the CRS, informational
only for SIRANE) are written for completeness but never read: this
reader locates records in the `.shp` file directly, sequentially, using each record's own
content-length field.
"""
from __future__ import annotations

import struct
from datetime import date
from pathlib import Path

FILE_CODE = 9994
VERSION = 1000
SHAPE_TYPE_POLYLINE = 3
MAIN_HEADER_SIZE = 100
"""Bytes in a Shapefile main-file (`.shp`/`.shx`) header -- ESRI Shapefile Technical
Description, both files share this 100-byte header layout."""

_DBF_VERSION = 0x03
"""dBase III without a memo (`.dbt`) file -- the version byte the South Kensington deck's own
`SIRANE_FINAL.dbf` carries, and the only variant this module reads or writes."""

Field = tuple[str, str, int, int]
"""A DBF field descriptor: `(name, type, size, decimals)` -- `type` is `'N'` (numeric,
right-justified, `decimals` digits after the point when `decimals > 0`, otherwise a plain
integer) or `'C'` (character, left-justified, `decimals` unused)."""

Point = tuple[float, float]
Polyline = list[Point]


def _paths(stem: Path) -> tuple[Path, Path, Path]:
    stem = Path(stem)
    return stem.with_suffix(".shp"), stem.with_suffix(".shx"), stem.with_suffix(".dbf")


# ------------------------------------------------------------------------------- geometry

def _polyline_content(points: Polyline) -> bytes:
    """One PolyLine record's CONTENT (everything after the 8-byte record header): shape
    type, bounding box, a single part starting at point 0, and the points themselves --
    ESRI Shapefile Technical Description's `PolyLine` record layout, specialised to exactly
    one part (SIRANE never subdivides a street segment in its own network file)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    body = struct.pack("<i", SHAPE_TYPE_POLYLINE)
    body += struct.pack("<4d", min(xs), min(ys), max(xs), max(ys))
    body += struct.pack("<2i", 1, len(points))  # NumParts, NumPoints
    body += struct.pack("<i", 0)  # Parts[0]: the single part starts at point 0
    for x, y in points:
        body += struct.pack("<2d", x, y)
    return body


def _write_shp(path: Path, lines: list[Polyline]) -> None:
    contents = [_polyline_content(points) for points in lines]
    all_x = [p[0] for points in lines for p in points]
    all_y = [p[1] for points in lines for p in points]
    box = (min(all_x), min(all_y), max(all_x), max(all_y)) if lines else (0.0, 0.0, 0.0, 0.0)
    file_length_words = MAIN_HEADER_SIZE // 2 + sum(4 + len(c) // 2 for c in contents)

    with open(path, "wb") as f:
        f.write(struct.pack(">i", FILE_CODE))
        f.write(b"\x00" * 20)  # bytes 4-23: unused
        f.write(struct.pack(">i", file_length_words))
        f.write(struct.pack("<2i", VERSION, SHAPE_TYPE_POLYLINE))
        f.write(struct.pack("<4d", *box))
        f.write(struct.pack("<4d", 0.0, 0.0, 0.0, 0.0))  # Zmin/Zmax/Mmin/Mmax: unused
        for i, content in enumerate(contents):
            f.write(struct.pack(">2i", i + 1, len(content) // 2))
            f.write(content)


def _write_shx(path: Path, lines: list[Polyline]) -> None:
    contents = [_polyline_content(points) for points in lines]
    file_length_words = MAIN_HEADER_SIZE // 2 + 4 * len(contents)
    all_x = [p[0] for points in lines for p in points]
    all_y = [p[1] for points in lines for p in points]
    box = (min(all_x), min(all_y), max(all_x), max(all_y)) if lines else (0.0, 0.0, 0.0, 0.0)

    with open(path, "wb") as f:
        f.write(struct.pack(">i", FILE_CODE))
        f.write(b"\x00" * 20)
        f.write(struct.pack(">i", file_length_words))
        f.write(struct.pack("<2i", VERSION, SHAPE_TYPE_POLYLINE))
        f.write(struct.pack("<4d", *box))
        f.write(struct.pack("<4d", 0.0, 0.0, 0.0, 0.0))
        offset_words = MAIN_HEADER_SIZE // 2
        for content in contents:
            content_words = len(content) // 2
            f.write(struct.pack(">2i", offset_words, content_words))
            offset_words += 4 + content_words


def _read_shp(path: Path) -> list[Polyline]:
    data = path.read_bytes()
    file_code = struct.unpack(">i", data[0:4])[0]
    if file_code != FILE_CODE:
        raise ValueError(
            f"read_polylines: {path} has file code {file_code}, expected {FILE_CODE} "
            f"(not an ESRI Shapefile main file)"
        )
    shape_type = struct.unpack("<i", data[32:36])[0]
    if shape_type != SHAPE_TYPE_POLYLINE:
        raise ValueError(
            f"read_polylines: {path} has shape type {shape_type}, only PolyLine "
            f"({SHAPE_TYPE_POLYLINE}) is supported"
        )
    file_length_bytes = struct.unpack(">i", data[24:28])[0] * 2

    lines: list[Polyline] = []
    pos = MAIN_HEADER_SIZE
    while pos < file_length_bytes:
        record_number, content_words = struct.unpack(">2i", data[pos:pos + 8])
        content_start = pos + 8
        content_end = content_start + content_words * 2
        record_shape_type = struct.unpack("<i", data[content_start:content_start + 4])[0]
        if record_shape_type != SHAPE_TYPE_POLYLINE:
            raise ValueError(
                f"read_polylines: {path} record {record_number} has shape type "
                f"{record_shape_type}, only PolyLine ({SHAPE_TYPE_POLYLINE}) is supported"
            )
        num_parts, num_points = struct.unpack("<2i", data[content_start + 36:content_start + 44])
        if num_parts != 1:
            raise ValueError(
                f"read_polylines: {path} record {record_number} has {num_parts} parts; "
                f"only a single-part polyline (one street segment) is supported"
            )
        if num_points != 2:
            raise ValueError(
                f"read_polylines: {path} record {record_number} has {num_points} points; "
                f"only a 2-point (straight-segment) polyline is supported"
            )
        points_start = content_start + 44 + 4 * num_parts
        points_end = points_start + 16 * num_points
        raw = struct.unpack(f"<{2 * num_points}d", data[points_start:points_end])
        lines.append([(raw[2 * i], raw[2 * i + 1]) for i in range(num_points)])
        pos = content_end
    return lines


# ------------------------------------------------------------------------------------ dbf

def _format_dbf_value(value: object, field: Field) -> str:
    name, ftype, size, decimals = field
    if ftype == "C":
        text = str(value)
        formatted = text.ljust(size)
    elif ftype == "N":
        text = f"{float(value):.{decimals}f}" if decimals > 0 else str(int(value))
        formatted = text.rjust(size)
    else:
        raise ValueError(f"write_polylines: field {name!r} has unsupported type {ftype!r}")
    if len(formatted) > size:
        raise ValueError(
            f"write_polylines: field {name!r} value {value!r} needs {len(formatted)} "
            f"characters, wider than its field width {size}"
        )
    return formatted


def _dbf_bytes(records: list[dict], fields: list[Field]) -> bytes:
    """The complete `.dbf` file content for `records` under `fields` -- built and fully
    validated (missing fields, name lengths, value widths) in memory before `write_polylines`
    touches disk, so a bad write never leaves a partial file behind."""
    for name, _ftype, _size, _decimals in fields:
        if len(name.encode("ascii")) > 11:
            raise ValueError(
                f"write_polylines: field name {name!r} is longer than 11 characters"
            )
    row_texts: list[str] = []
    for i, record in enumerate(records):
        missing = [name for name, *_ in fields if name not in record]
        if missing:
            raise ValueError(f"write_polylines: record {i} is missing field(s) {missing}")
        row_texts.append("".join(_format_dbf_value(record[f[0]], f) for f in fields))

    header_size = 32 + 32 * len(fields) + 1
    record_size = 1 + sum(size for _, _, size, _ in fields)
    today = date.today()

    out = bytearray()
    out += struct.pack("<B", _DBF_VERSION)
    out += struct.pack("<3B", today.year - 1900, today.month, today.day)
    out += struct.pack("<i", len(records))
    out += struct.pack("<2h", header_size, record_size)
    out += b"\x00" * 20  # reserved
    for name, ftype, size, decimals in fields:
        out += name.encode("ascii").ljust(11, b"\x00")
        out += ftype.encode("ascii")
        out += b"\x00" * 4  # field data address: unused
        out += struct.pack("<2B", size, decimals)
        out += b"\x00" * 14  # reserved
    out += b"\x0d"  # field descriptor array terminator
    for row in row_texts:
        out += b" "  # not deleted
        out += row.encode("ascii")
    out += b"\x1a"  # end-of-file marker
    return bytes(out)


def _read_dbf(path: Path) -> tuple[list[dict], int]:
    data = path.read_bytes()
    n_records = struct.unpack("<i", data[4:8])[0]
    header_size, record_size = struct.unpack("<2h", data[8:12])

    fields: list[Field] = []
    pos = 32
    while data[pos] != 0x0D:
        name = data[pos:pos + 11].split(b"\x00")[0].decode("ascii")
        ftype = chr(data[pos + 11])
        size = data[pos + 16]
        decimals = data[pos + 17]
        fields.append((name, ftype, size, decimals))
        pos += 32

    records = []
    for i in range(n_records):
        start = header_size + i * record_size
        raw_record = data[start:start + record_size]
        if raw_record[0:1] == b"*":  # soft-deleted (dBase leaves the record in place)
            # Skipping it would shift every later record against the .shp's shapes.
            raise ValueError(
                f"read_polylines: {path} record {i} is soft-deleted; a SIRANE network has "
                f"no deleted records"
            )
        offset = 1  # skip the deletion flag
        record: dict = {}
        for name, ftype, size, _decimals in fields:
            raw = raw_record[offset:offset + size].decode("ascii", errors="replace").strip()
            offset += size
            if ftype == "N":
                if raw == "":
                    record[name] = None
                elif "." in raw:
                    record[name] = float(raw)
                else:
                    record[name] = int(raw)
            else:
                record[name] = raw
        records.append(record)
    return records, n_records


# --------------------------------------------------------------------------------- public

def read_polylines(stem: Path) -> tuple[list[Polyline], list[dict]]:
    """Reads a PolyLine Shapefile (`stem` with `.shp` + `.dbf` appended; `.shx`/`.prj` are
    never read -- see the module docstring) into `(lines, records)`: `lines[i]` is record
    `i`'s two `(x, y)` points in the file's own order, `records[i]` its DBF attributes as a
    plain dict keyed by field name (numeric fields as `int`/`float`, character fields as
    stripped `str`).

    Refuses (`ValueError`, naming `read_polylines`) any record that is not PolyLine, has more
    than one part, or does not have exactly two points -- a SIRANE street network is always
    single-part, straight-segment records; this is not a general Shapefile reader.
    """
    shp_path, _shx_path, dbf_path = _paths(stem)
    lines = _read_shp(shp_path)
    records, n_dbf_records = _read_dbf(dbf_path)
    if len(lines) != n_dbf_records:
        raise ValueError(
            f"read_polylines: {shp_path} has {len(lines)} records but {dbf_path} has "
            f"{n_dbf_records}"
        )
    return lines, records


def write_polylines(
    stem: Path, lines: list[Polyline], records: list[dict], fields: list[Field]
) -> None:
    """Writes `stem.shp`, `stem.shx` and `stem.dbf`: `lines[i]` (any number of points >= 2,
    a single part) paired with `records[i]`'s attributes under `fields` (`(name, type
    'N'|'C', size, decimals)`, in file order).

    Refuses (`ValueError`, naming `write_polylines`) a line with fewer than 2 points, a
    record missing one of `fields`, a value that does not fit its field's width, or a field
    name longer than 11 characters (dBase III's own limit) -- so a bad write is caught here
    rather than producing a Shapefile `read_polylines` (or SIRANE itself) would refuse or
    misread.
    """
    if len(lines) != len(records):
        raise ValueError(
            f"write_polylines: {len(lines)} polylines but {len(records)} records"
        )
    for i, points in enumerate(lines):
        if len(points) < 2:
            raise ValueError(
                f"write_polylines: line {i} has {len(points)} point(s); a polyline needs "
                f"at least 2"
            )
    stem = Path(stem)
    shp_path, shx_path, dbf_path = _paths(stem)
    dbf_bytes = _dbf_bytes(records, fields)  # validated (and built) before any file write
    stem.parent.mkdir(parents=True, exist_ok=True)
    _write_shp(shp_path, lines)
    _write_shx(shx_path, lines)
    dbf_path.write_bytes(dbf_bytes)
