import math
from pathlib import Path

import pytest
import torch

from noodl.apps.building.epw import read_epw, write_wth
from noodl.apps.building.wth import read_wth

# Two hours of an EPW file: 8 header lines, then rows with the 35 documented columns.
# Column indices (0-based): 6 dry-bulb C, 9 station pressure Pa, 20 wind direction deg,
# 21 wind speed m/s.
HEADER = "\n".join([
    "LOCATION,Probe,-,FRA,IWEC Data,071490,48.73,2.40,1.0,89.0",
    "DESIGN CONDITIONS,0", "TYPICAL/EXTREME PERIODS,0", "GROUND TEMPERATURES,0",
    "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0", "COMMENTS 1,probe", "COMMENTS 2,probe",
    "DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31",
])


def _row(month, day, hour, temp_c, pressure, wd, ws, year="2024"):
    cols = [year, str(month), str(day), str(hour), "60", "?"] + ["0"] * 29
    cols[6], cols[9], cols[20], cols[21] = str(temp_c), str(pressure), str(wd), str(ws)
    return ",".join(cols)


def _write(tmp_path, rows):
    p = tmp_path / "probe.epw"
    p.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_read_epw_converts_units_and_times(tmp_path):
    p = _write(tmp_path, [_row(1, 1, 1, 5.0, 101000, 270, 3.0),
                          _row(1, 1, 2, 6.0, 101100, 90, 4.0)])
    w = read_epw(p)
    assert w.t.tolist() == [3600.0, 7200.0]            # EPW hour 1 is the hour ENDING 01:00
    assert math.isclose(w.Ta[0].item(), 278.15)
    assert w.Pb.tolist() == [101000.0, 101100.0]
    assert w.Wd.tolist() == [270.0, 90.0] and w.Ws.tolist() == [3.0, 4.0]


def test_read_epw_window_starts_at_the_requested_day(tmp_path):
    rows = [_row(1, d, h, 0.0, 101325, 0, 1.0) for d in (1, 2, 3) for h in range(1, 25)]
    w = read_epw(_write(tmp_path, rows), start_day=2, n_hours=24)
    assert w.t.numel() == 24 and w.t[0].item() == 3600.0


def test_write_wth_round_trips_through_read_wth(tmp_path):
    p = _write(tmp_path, [_row(1, 1, h, 10.0 + h, 101325, 45 * h % 360, 2.0) for h in range(1, 25)])
    w = read_epw(p)
    out = write_wth(w, tmp_path / "probe.wth")
    back = read_wth(out)
    assert torch.allclose(back.Ta, w.Ta) and torch.allclose(back.Ws, w.Ws)
    assert torch.allclose(back.Wd, w.Wd) and torch.allclose(back.Pb, w.Pb)


def test_read_epw_is_monotonic_across_a_stitched_leap_year_boundary(tmp_path):
    # A "typical year" file stitches each month in from a different real source year, and
    # those years' leap status need not agree (e.g. a real Paris TMYx EPW file has
    # April from 1953, a non-leap year, and May from 1976, a leap year). Reading day-of-year
    # from each row's own `year` column would insert a 29 February on the leap side only,
    # shifting every later day by one and breaking monotonicity right at this boundary.
    rows = [_row(4, 30, h, 15.0, 101325, 0, 1.0, year="1953") for h in range(1, 25)]
    rows += [_row(5, 1, h, 15.0, 101325, 0, 1.0, year="1976") for h in range(1, 25)]
    w = read_epw(_write(tmp_path, rows))
    diffs = torch.diff(w.t)
    assert torch.all(diffs == 3600.0), diffs.tolist()


def test_read_epw_paris_file_has_a_strictly_monotonic_time_axis():
    path = Path(__file__).resolve().parents[1] / "data" / "paris.epw"
    if not path.exists():
        pytest.skip("data/paris.epw has not been fetched")
    w = read_epw(path)
    diffs = torch.diff(w.t)
    assert torch.all(diffs == 3600.0), diffs.tolist()
