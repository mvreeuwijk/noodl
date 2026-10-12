"""`benchmarks.swmm_corpus`: the pieces that prepare SWMM's copy of a corpus file."""

from __future__ import annotations

from pathlib import Path

from benchmarks.swmm_corpus import absolutise_files, cut_to_window


def test_time_series_file_paths_become_absolute_so_swmm_finds_them_from_its_copy():
    base = Path("/corpus/model")
    lines = ["[TIMESERIES]", 'TS1 FILE "rel/a.dat" ; note', "TS2 0 1 1 2", "TS3 file b.dat",
             "[CURVES]", "FILE x"]
    out = absolutise_files(lines, base)
    assert out[1].startswith('TS1 FILE "') and out[1].endswith('a.dat"')
    assert str(base) in out[1] and out[1].replace("\\", "/").endswith("model/rel/a.dat\"")
    assert out[2] == "TS2 0 1 1 2" and out[4:] == lines[4:]
    assert out[3].startswith('TS3 file "') and out[3].replace("\\", "/").endswith("b.dat\"")
    assert absolutise_files(lines, None) == lines


def test_cut_to_window_keeps_the_start_and_resolves_file_paths():
    text = ("[OPTIONS]\nFLOW_UNITS LPS\nSTART_DATE 06/10/2026\nSTART_TIME 11:57:00\n"
            "END_DATE 06/11/2026\n[TIMESERIES]\nTS FILE d.dat\n")
    out, units = cut_to_window(text, 600.0, Path("/corpus"))
    assert units == "LPS"
    assert "START_TIME 11:57:00" in out and "END_TIME 12:07:00" in out
    assert "d.dat" in out and "FILE d.dat" not in out
