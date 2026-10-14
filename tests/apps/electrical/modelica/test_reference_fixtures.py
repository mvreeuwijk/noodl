"""The exported electrical reference fixtures are complete and small."""

from __future__ import annotations

from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data" / "modelica_electrical"
MAX_CSV_BYTES = 300_000


def test_fourteen_json_csv_pairs():
    jsons = {p.stem for p in DATA.glob("*.json")}
    csvs = {p.stem for p in DATA.glob("*.csv")}
    assert len(jsons) == 14
    assert jsons == csvs


def test_every_csv_is_under_the_size_cap():
    sizes = {p.name: p.stat().st_size for p in DATA.glob("*.csv")}
    assert sizes
    assert {n: s for n, s in sizes.items() if s >= MAX_CSV_BYTES} == {}
