"""CONTAM .wth reader (TN 1887r1 section 3.15 format)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tellegen.apps.building.wth import Weather, read_wth

DATA = Path(__file__).resolve().parents[2] / "data" / "contam"

SYNTHETIC = """WeatherFile ContamW 2.0
synthetic
1/1\t1/2
!Date\tDofW\tDtype\tDST\tTgrnd
1/1\t1\t1\t0\t283.15
1/2\t2\t1\t0\t283.15
!Date\tTime\tTa\tPb\tWs\tWd\tHr\tIth\tIdn\tTs\tRn\tSn
1/1\t00:00:00\t280.0\t101000\t2.0\t90\t0\t0\t0\t0\t0\t0
1/1\t12:00:00\t290.0\t101200\t4.0\t180\t0\t0\t0\t0\t0\t0
1/1\t24:00:00\t282.0\t101100\t3.0\t270\t0\t0\t0\t0\t0\t0
1/2\t00:00:00\t282.0\t101100\t3.0\t270\t0\t0\t0\t0\t0\t0
1/2\t24:00:00\t286.0\t101300\t1.0\t0\t0\t0\t0\t0\t0\t0
"""


def test_reads_the_nist_sample():
    w = read_wth(DATA / "valThreeZonesWthCtm.wth")
    assert isinstance(w, Weather)
    assert w.t.tolist() == [0.0, 86400.0]
    assert w.Ta.tolist() == pytest.approx([293.15, 293.15])
    assert w.Ws.tolist() == pytest.approx([5.23, 5.23])
    assert w.Wd.tolist() == pytest.approx([270.0, 270.0])


def test_interpolates_linearly_between_rows_and_across_days(tmp_path):
    f = tmp_path / "s.wth"
    f.write_text(SYNTHETIC)
    w = read_wth(f)
    assert w.t.tolist() == [0.0, 43200.0, 86400.0, 86400.0, 172800.0]
    at = w.at(6 * 3600.0)
    assert at["Ta"] == pytest.approx(285.0)
    assert at["Ws"] == pytest.approx(3.0)
    assert w.at(36 * 3600.0)["Ta"] == pytest.approx(284.0)
    d = w.drivers_at(6 * 3600.0)
    assert set(d) == {"T_amb", "P_amb", "V_met", "theta_w"}
    assert d["V_met"].dtype == torch.float64 and d["V_met"].dim() == 0


def test_rejects_a_file_that_is_not_a_contam_weather_file(tmp_path):
    f = tmp_path / "bad.wth"
    f.write_text("EPW,SomeCity\n")
    with pytest.raises(ValueError, match="WeatherFile ContamW"):
        read_wth(f)
