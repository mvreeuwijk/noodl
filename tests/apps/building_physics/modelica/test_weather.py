"""`noodl.apps.building_physics.modelica.weather`: MBL ReaderTMY3's airflow outputs.

Reference values are written out here from the MBL and MSL formulas (cited), not computed by
the module under test. OpenModelica parity of the same outputs is
`tests/verification/test_modelica_parity.py::test_weather_and_wind_pressure_parity`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from noodl.apps.building_physics.modelica import weather
from noodl.apps.building_physics.modelica.schema import ModelicaImportError, load
from tests.apps.building_physics.modelica.weather_files import (
    document,
    reader,
    tmy3_row,
    write_model,
    write_mos,
)

F64 = torch.float64
YEAR = 365 * 24 * 3600.0
H = 3600.0


def _reader_doc(tmp_path, rows, parameters=None, experiment=None, files=True):
    doc = load(write_model(tmp_path, document([reader(parameters)], experiment=experiment),
                           rows, files=files))
    return doc.components[0], doc


# ------------------------------------------------------------------------- table file
def test_table_is_read_with_its_declared_shape(tmp_path):
    rows = [tmy3_row(k * H, T=float(k)) for k in range(4)]
    table = weather.read_table(write_mos(tmp_path / "w.mos", rows))
    assert table.time.tolist() == [0.0, H, 2 * H, 3 * H]
    assert table.data[:, 1].tolist() == [0.0, 1.0, 2.0, 3.0]
    # getTimeSpan.c:135-142: (first stamp, last stamp + mean increment)
    assert table.span == (0.0, 4 * H)


@pytest.mark.parametrize(("text", "match"), [
    ("#1\ndouble other(2,30)\n", "no table"),
    ("#1\ndouble tab1(2,30)\n" + "\t".join(["0"] * 30) + "\n", "declares 2 x 30"),
    ("#1\ndouble tab1(2,30)\n" + "\t".join(["5"] + ["0"] * 29) + "\n"
     + "\t".join(["5"] + ["0"] * 29) + "\n", "strictly increasing"),
])
def test_malformed_tables_are_refused(tmp_path, text, match):
    path = tmp_path / "w.mos"
    path.write_text(text, newline="\n")
    with pytest.raises(ModelicaImportError, match=match):
        weather.read_table(path)


# ----------------------------------------------------------------------- Akima spline
def test_akima_reproduces_straight_lines_inside_and_outside_the_table():
    x = np.array([0.0, 1.0, 2.5, 3.0, 5.0])
    spline = weather.AkimaSpline(x, 2.0 * x + 1.0)
    u = torch.tensor([-1.0, 0.3, 1.7, 2.9, 4.2, 6.0], dtype=F64)
    torch.testing.assert_close(spline(u), 2.0 * u + 1.0, rtol=0, atol=1e-14)


def test_akima_follows_the_msl_slopes():
    """ModelicaStandardTables.c:6574-6608 by hand for y = 0, 0, 1, 1, 1 at x = 0..4:
    divided differences d[2..5] = 0, 1, 0, 0, extended d[0], d[1] = -2, -1 and d[6], d[7] = 0, 0;
    knot slopes -0.5, 0.5, 0, 0, 0. On [1, 2]: c1 = 3 - 2(0.5) - 0 = 2, c0 = 0.5 + 0 - 2 =
    -1.5, so y(1.5) = ((-1.5 * 0.5 + 2) * 0.5 + 0.5) * 0.5 = 0.5625. Left of the table
    the slope of the first knot (y(-1) = 0.5), right of it the last interval's end slope (0)."""
    spline = weather.AkimaSpline([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 1.0, 1.0])
    u = torch.tensor([1.5, 2.0, -1.0, 5.0], dtype=F64)
    torch.testing.assert_close(spline(u), torch.tensor([0.5625, 1.0, 0.5, 1.0], dtype=F64),
                               rtol=0, atol=1e-15)


def test_akima_with_two_rows_is_linear():
    spline = weather.AkimaSpline([0.0, 2.0], [1.0, 3.0])
    u = torch.tensor([-1.0, 1.0, 3.0], dtype=F64)
    torch.testing.assert_close(spline(u), u + 1.0, rtol=0, atol=1e-15)


def test_crossings_find_where_the_spline_meets_a_level():
    """y = 90, 99.9, 99.9, 90 % at 0, 1, 2, 3 h: knot slopes +-4.95 %/h at the middle knots,
    and on [1, 2] h the cubic is 99.9 + 4.95 v (1 - v) (c0 = 0), above 100 % between
    v = (1 -+ sqrt(1 - 4 (0.1/4.95)))/2: two interior crossings of 100."""
    spline = weather.AkimaSpline([0.0, H, 2 * H, 3 * H], [90.0, 99.9, 99.9, 90.0])
    v = (1.0 - math.sqrt(1.0 - 4.0 * 0.1 / 4.95)) / 2.0
    got = spline.crossings(100.0, 0.0, 3 * H)
    assert got == pytest.approx([H * (1.0 + v), H * (2.0 - v)], rel=1e-12)
    assert spline.crossings(100.0, 0.0, H) == []  # outside the window
    assert spline.crossings(99.9, 0.0, 3 * H) == []  # at the knots only: not interior


# ------------------------------------------------------------------- calendar time
def test_calendar_time_of_a_one_year_file_wraps_after_each_year():
    """PartialConvertTime.mo:25-36: calTim = t - (k - 1) lenWea, k = integer(t0/lenWea) + 1 at the
    start, incremented when t passes k lenWea (strictly)."""
    span = (0.0, YEAR)
    t = torch.tensor([0.0, 10.0, YEAR, YEAR + 10.0, 2 * YEAR + 5.0], dtype=F64)
    assert weather.calendar_time(t, 0.0, span).tolist() == [0.0, 10.0, YEAR, 10.0, 5.0]


def test_calendar_time_starting_on_a_year_boundary_starts_at_zero():
    t = torch.tensor([YEAR, YEAR + 5.0], dtype=F64)
    assert weather.calendar_time(t, YEAR, (0.0, YEAR)).tolist() == [0.0, 5.0]


def test_calendar_time_of_a_file_that_is_not_whole_years_is_the_model_time():
    t = torch.tensor([0.0, 5e6], dtype=F64)
    assert weather.calendar_time(t, 0.0, (0.0, 4 * H)).tolist() == [0.0, 5e6]


# ------------------------------------------------------------------- reader outputs
def _times(*t):
    return torch.tensor(t, dtype=F64)


def test_reader_outputs_convert_the_file_units(tmp_path):
    rows = [tmy3_row(k * H) for k in range(4)]
    comp, doc = _reader_doc(tmp_path, rows)
    out = weather.reader_outputs(comp, doc, _times(0.0, 1800.0, 7200.0), lambda port: None)
    assert out["TDryBul"].tolist() == [10.0 + 273.15] * 3                  # From_degC
    assert out["relHum"].tolist() == [0.01 * 50.0] * 3                     # Gain k = 0.01
    assert out["pAtm"].tolist() == [101325.0] * 3            # pAtmSou = Parameter by default
    assert out["winSpe"].tolist() == [4.0] * 3
    assert out["winDir"].tolist() == [(math.pi / 180.0) * 270.0] * 3       # From_deg


def test_reader_outputs_read_the_pressure_from_the_file_when_asked(tmp_path):
    rows = [tmy3_row(k * H, p=100000.0 + k) for k in range(4)]
    comp, doc = _reader_doc(tmp_path, rows, {
        "pAtmSou": "Buildings.BoundaryConditions.Types.DataSource.File"})
    out = weather.reader_outputs(comp, doc, _times(H), lambda port: None, ("pAtm",))
    assert out["pAtm"].tolist() == [100001.0]


def test_reader_outputs_reduce_the_wind_direction_after_interpolating(tmp_path):
    """San Francisco's file holds -180 and -190 deg: the spline runs on the raw degrees, then
    LimiterWindDirection takes mod(., 2 pi)."""
    rows = [tmy3_row(k * H, wd=-180.0) for k in range(4)]
    comp, doc = _reader_doc(tmp_path, rows)
    out = weather.reader_outputs(comp, doc, _times(H), lambda port: None, ("winDir",))
    assert float(out["winDir"]) == pytest.approx(math.pi, abs=1e-15)


def test_reader_outputs_limit_humidity_and_wind_speed(tmp_path):
    """Akima overshoots 100 % between two 100 % knots (y(1.5 h) = 101.25 %) and undershoots
    0 m/s after a calm start (the first knot's slope is negative): LimiterRelativeHumidity and
    LimiterWindSpeed cut both."""
    rh = [90.0, 100.0, 100.0, 90.0, 90.0]
    ws = [0.0, 0.0, 2.0, 2.0, 2.0]
    rows = [tmy3_row(k * H, rh=rh[k], ws=ws[k]) for k in range(5)]
    comp, doc = _reader_doc(tmp_path, rows)
    raw_rh = weather.read_table(tmp_path / "w.mos").spline(4)(_times(1.5 * H))
    raw_ws = weather.read_table(tmp_path / "w.mos").spline(17)(_times(0.5 * H))
    assert float(raw_rh) > 100.0 and float(raw_ws) < 0.0
    out = weather.reader_outputs(comp, doc, _times(0.5 * H, 1.5 * H), lambda port: None)
    assert out["relHum"].tolist()[1] == 1.0
    assert out["winSpe"].tolist()[0] == 0.0


def test_reader_parameter_and_input_sources(tmp_path):
    rows = [tmy3_row(k * H) for k in range(4)]
    comp, doc = _reader_doc(tmp_path, rows, {
        "winSpeSou": "Buildings.BoundaryConditions.Types.DataSource.Parameter", "winSpe": 1.0,
        "winDirSou": "Buildings.BoundaryConditions.Types.DataSource.Input"})
    t = _times(0.0, 10.0)
    seen = []

    def signal(port):
        seen.append(port)
        return torch.tensor([7.0, -1.0], dtype=F64)

    out = weather.reader_outputs(comp, doc, t, signal, ("winSpe", "winDir"))
    assert seen == ["winDir_in"]
    assert out["winSpe"].tolist() == [1.0, 1.0]
    assert out["winDir"].tolist() == pytest.approx([7.0 - 2 * math.pi, 2 * math.pi - 1.0])
    with pytest.raises(ModelicaImportError, match=r"weaDat .*winDir_in"):
        weather.reader_outputs(comp, doc, t, lambda port: None, ("winDir",))


def test_reader_outside_mbl_bounds_is_refused(tmp_path):
    rows = [tmy3_row(k * H, T=80.0) for k in range(4)]  # 353.15 K > 343.15 K
    comp, doc = _reader_doc(tmp_path, rows)
    with pytest.raises(ModelicaImportError, match=r"weaDat .*TDryBul .*CheckDryBulbTemperature"):
        weather.reader_outputs(comp, doc, _times(0.0), lambda port: None)


def test_a_short_file_that_does_not_cover_the_run_is_refused(tmp_path):
    """ConvertTime.mo:22-32: a file that is not whole years long must cover the run, to 1800 s
    past its end (here 4 h)."""
    rows = [tmy3_row(k * H) for k in range(4)]
    comp, doc = _reader_doc(tmp_path, rows)
    weather.reader_outputs(comp, doc, _times(0.0, 4 * H + 1799.0), lambda port: None)
    with pytest.raises(ModelicaImportError, match=r"weaDat .*does not repeat"):
        weather.reader_outputs(comp, doc, _times(0.0, 4 * H + 1800.0), lambda port: None)


# ------------------------------------------------------------------------- events
def test_reader_events_are_the_knots_and_the_limit_crossings(tmp_path):
    rh = [90.0, 99.9, 99.9, 90.0]
    rows = [tmy3_row(k * H, rh=rh[k]) for k in range(4)]
    comp, doc = _reader_doc(tmp_path, rows)
    jumps, breaks = weather.reader_events(comp, doc, 0.0, 3 * H)
    v = (1.0 - math.sqrt(1.0 - 4.0 * 0.1 / 4.95)) / 2.0
    assert jumps == set()
    assert sorted(breaks) == pytest.approx([H, H * (1.0 + v), H * (2.0 - v), 2 * H],
                                           rel=1e-12)


def test_reader_events_put_the_year_wrap_among_the_jumps(tmp_path):
    rows = [tmy3_row(0.0), tmy3_row(YEAR / 2)]  # span (0, YEAR): repeats
    comp, doc = _reader_doc(tmp_path, rows)
    jumps, breaks = weather.reader_events(comp, doc, YEAR - 100.0, YEAR + 100.0)
    assert jumps == {YEAR}
    assert breaks == set()


def test_a_reader_with_no_file_output_has_no_events(tmp_path):
    rows = [tmy3_row(k * H) for k in range(4)]
    params = {f"{q}Sou": "Buildings.BoundaryConditions.Types.DataSource.Parameter"
              for q in ("TDryBul", "relHum", "winSpe", "winDir")}
    comp, doc = _reader_doc(tmp_path, rows, params)
    assert weather.reader_events(comp, doc, 0.0, 3 * H) == (set(), set())


# ------------------------------------------------------------------- file resolution
def test_weather_file_is_refused_when_the_export_did_not_record_it(tmp_path):
    comp, doc = _reader_doc(tmp_path, [tmy3_row(0.0), tmy3_row(H)], files=False)
    with pytest.raises(ModelicaImportError, match=r"weaDat .*does not travel"):
        weather.resolve_file(comp, doc)


def test_weather_file_is_refused_when_it_was_changed(tmp_path):
    comp, doc = _reader_doc(tmp_path, [tmy3_row(0.0), tmy3_row(H)])
    (tmp_path / "w.mos").write_bytes((tmp_path / "w.mos").read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ModelicaImportError, match=r"weaDat .*sha256"):
        weather.resolve_file(comp, doc)


def test_weather_file_is_refused_when_missing_or_not_a_bare_name(tmp_path):
    comp, doc = _reader_doc(tmp_path, [tmy3_row(0.0), tmy3_row(H)])
    (tmp_path / "w.mos").unlink()
    with pytest.raises(ModelicaImportError, match=r"weaDat .*not found"):
        weather.resolve_file(comp, doc)
    bad = comp.__class__(name=comp.name, cls=comp.cls, parameters=comp.parameters,
                         files={"filNam": {"path": "../w.mos", "sha256": "a" * 64}})
    with pytest.raises(ModelicaImportError, match=r"weaDat .*bare file name"):
        weather.resolve_file(bad, doc)
