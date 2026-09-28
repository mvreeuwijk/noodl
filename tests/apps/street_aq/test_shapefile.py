"""`_shapefile.py`: the minimal PolyLine Shapefile + dBase III reader/writer this app uses
for SIRANE street networks."""
import struct
from pathlib import Path

import pytest

from noodl.apps.street_aq._shapefile import (
    MAIN_HEADER_SIZE,
    _dbf_bytes,
    _read_dbf,
    read_polylines,
    write_polylines,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "street"
SOUTH_KENSINGTON = DATA / "sirane_south_kensington" / "RESEAU" / "SIRANE_FINAL"

FIELDS = [
    ("TYPE", "N", 50, 0),
    ("NDDEB", "C", 50, 0),
    ("NDFIN", "C", 50, 0),
    ("WG", "N", 12, 4),
    ("WD", "N", 12, 4),
    ("HG", "N", 12, 4),
    ("HD", "N", 12, 4),
    ("MODUL_EMIS", "N", 50, 0),
]

# Generous widths for every field the real South Kensington DBF carries -- wide enough for
# any value in that file (unlike its own `(N,9,9)` descriptors, which cannot be reused for
# writing: see the module docstring's note that `decimals` only matters to the writer).
SK_FIELDS = [
    ("TYPE", "N", 20, 0), ("NDDEB", "C", 50, 0), ("NDFIN", "C", 50, 0),
    ("WG", "N", 20, 6), ("WD", "N", 20, 6), ("HG", "N", 20, 6), ("HD", "N", 20, 6),
    ("MODUL_EMIS", "N", 20, 0), ("ID", "N", 20, 0), ("LEN", "N", 20, 6),
    ("X_S", "N", 20, 6), ("Y_S", "N", 20, 6), ("X_E", "N", 20, 6), ("Y_E", "N", 20, 6),
    ("NO", "N", 20, 6), ("NO2", "N", 20, 6), ("O3", "N", 20, 6),
]


def _record(i):
    return {
        "TYPE": 0, "NDDEB": str(i), "NDFIN": str(i + 1), "WG": 4.5, "WD": 4.5,
        "HG": 12.5, "HD": 12.5, "MODUL_EMIS": 0,
    }


def test_write_polylines_then_read_polylines_round_trips_points_and_records(tmp_path):
    stem = tmp_path / "network"
    lines = [
        [(0.0, 0.0), (10.0, 0.0)],
        [(10.0, 0.0), (10.0, -5.5)],
        [(-3.25, 6.0), (0.0, 0.0)],
    ]
    records = [_record(i) for i in range(len(lines))]

    write_polylines(stem, lines, records, FIELDS)
    read_lines, read_records = read_polylines(stem)

    assert read_lines == lines
    assert read_records == records


def test_write_polylines_writes_shp_shx_and_dbf(tmp_path):
    stem = tmp_path / "network"
    write_polylines(stem, [[(0.0, 0.0), (1.0, 1.0)]], [_record(0)], FIELDS)

    assert (tmp_path / "network.shp").exists()
    assert (tmp_path / "network.shx").exists()
    assert (tmp_path / "network.dbf").exists()


def test_read_polylines_reads_the_south_kensington_network():
    lines, records = read_polylines(SOUTH_KENSINGTON)

    assert len(lines) == 46
    assert len(records) == 46
    assert all(len(points) == 2 for points in lines)

    expected_fields = {
        "TYPE", "NDDEB", "NDFIN", "WG", "WD", "HG", "HD", "MODUL_EMIS",
        "ID", "LEN", "X_S", "Y_S", "X_E", "Y_E", "NO", "NO2", "O3",
    }
    assert expected_fields <= set(records[0])

    first = records[0]
    assert first["TYPE"] == 0
    assert first["NDDEB"] == "1"
    assert first["NDFIN"] == "7"
    assert first["WG"] == pytest.approx(9.0)
    assert first["HG"] == pytest.approx(15.762893)
    assert first["MODUL_EMIS"] == 0

    x_s, y_s = lines[0][0]
    x_e, y_e = lines[0][1]
    assert x_s == pytest.approx(-15911.75, abs=1e-2)
    assert y_s == pytest.approx(4991207.95, abs=1e-2)
    assert x_e == pytest.approx(-15593.12, abs=1e-2)
    assert y_e == pytest.approx(4991229.32, abs=1e-2)


def test_read_polylines_refuses_a_polyline_with_more_than_two_points(tmp_path):
    stem = tmp_path / "three_point"
    lines = [[(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)]]
    write_polylines(stem, lines, [_record(0)], FIELDS)

    with pytest.raises(ValueError, match="read_polylines"):
        read_polylines(stem)


def test_write_polylines_refuses_a_record_missing_a_field(tmp_path):
    stem = tmp_path / "missing_field"
    bad_record = dict(_record(0))
    del bad_record["HD"]

    with pytest.raises(ValueError, match="write_polylines"):
        write_polylines(stem, [[(0.0, 0.0), (1.0, 1.0)]], [bad_record], FIELDS)


def test_write_polylines_refuses_a_value_too_wide_for_its_field(tmp_path):
    stem = tmp_path / "too_wide"
    bad_record = dict(_record(0))
    bad_record["NDDEB"] = "x" * 51

    with pytest.raises(ValueError, match="write_polylines"):
        write_polylines(stem, [[(0.0, 0.0), (1.0, 1.0)]], [bad_record], FIELDS)


def test_write_polylines_refuses_a_line_with_fewer_than_two_points(tmp_path):
    stem = tmp_path / "one_point"

    with pytest.raises(ValueError, match="write_polylines"):
        write_polylines(stem, [[(0.0, 0.0)]], [_record(0)], FIELDS)


def test_write_polylines_shx_index_matches_the_real_south_kensington_shx(tmp_path):
    """Regression test for the `_write_shx` offset-off-by-4-words bug: every record's shx
    entry must point at the record's own HEADER in the `.shp` (record 1 -> word 50 = byte
    100), not 4 words further in at its content. The written `.shx` is compared, entry by
    entry, against the real SIRANE-produced `SIRANE_FINAL.shx` -- geometry (and so every
    offset/content-length pair) is unchanged by the round trip, only the DBF field widths
    differ (see `SK_FIELDS`), and the shx carries no DBF information."""
    lines, records = read_polylines(SOUTH_KENSINGTON)

    stem = tmp_path / "network"
    write_polylines(stem, lines, records, SK_FIELDS)

    written_shx = (stem.with_suffix(".shx")).read_bytes()
    real_shx = SOUTH_KENSINGTON.with_suffix(".shx").read_bytes()
    assert written_shx[MAIN_HEADER_SIZE:] == real_shx[MAIN_HEADER_SIZE:]


def test_write_polylines_shx_offsets_point_to_shp_record_headers(tmp_path):
    """A general (fixture-independent) check of the same invariant: parsing the written
    `.shx`, each entry's offset (in 16-bit words) lands exactly on a `.shp` record header
    whose own record number is i+1."""
    lines = [[(0.0, 0.0), (1.0, 0.0)], [(1.0, 0.0), (1.0, 2.0)], [(2.0, 2.0), (0.0, 0.0)]]
    records = [_record(i) for i in range(len(lines))]
    stem = tmp_path / "network"
    write_polylines(stem, lines, records, FIELDS)

    shx = stem.with_suffix(".shx").read_bytes()
    shp = stem.with_suffix(".shp").read_bytes()
    for i in range(len(lines)):
        entry_start = MAIN_HEADER_SIZE + 8 * i
        offset_words, content_words = struct.unpack(">2i", shx[entry_start:entry_start + 8])
        header_start = offset_words * 2
        record_number, header_content_words = struct.unpack(
            ">2i", shp[header_start:header_start + 8]
        )
        assert record_number == i + 1
        assert header_content_words == content_words


def test_read_dbf_refuses_a_soft_deleted_record(tmp_path):
    """Skipping a soft-deleted record would pair every later record with the wrong shape
    (the .shp still holds it, and the header count still includes it)."""
    dbf_bytes = bytearray(_dbf_bytes([_record(0), _record(1)], FIELDS))

    header_size = struct.unpack("<h", dbf_bytes[8:10])[0]
    record_size = struct.unpack("<h", dbf_bytes[10:12])[0]
    second_record_start = header_size + record_size
    dbf_bytes[second_record_start] = ord("*")  # dBase's own soft-delete marker

    path = tmp_path / "deleted.dbf"
    path.write_bytes(bytes(dbf_bytes))

    with pytest.raises(ValueError, match="read_polylines: .*record 1 is soft-deleted"):
        _read_dbf(path)
