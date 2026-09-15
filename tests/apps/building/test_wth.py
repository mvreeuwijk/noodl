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


def test_wind_direction_interpolates_on_the_shorter_arc_across_the_0_360_wrap():
    # A step from 350 deg to 10 deg sweeps forward through 0 (a 20 deg arc), not
    # backwards through 180 (a 340 deg arc): the midpoint must be 0, not 180.
    w = Weather(
        t=torch.tensor([0.0, 100.0], dtype=torch.float64),
        Ta=torch.tensor([280.0, 280.0], dtype=torch.float64),
        Pb=torch.tensor([101000.0, 101000.0], dtype=torch.float64),
        Ws=torch.tensor([1.0, 1.0], dtype=torch.float64),
        Wd=torch.tensor([350.0, 10.0], dtype=torch.float64),
    )
    assert w.at(50.0)["Wd"] == pytest.approx(0.0, abs=1e-9)
