"""`read_modelica` on documents with a ReaderTMY3 and Outside boundaries (graph, assemble,
run). Expected values are written out from the MBL formulas (cited); OpenModelica parity is
`tests/verification/test_modelica_parity.py`."""

from __future__ import annotations

import json
import math

import pytest
import torch

from noodl.apps.building_physics import read_modelica
from noodl.apps.building_physics.modelica import ModelicaImportError, assemble
from noodl.apps.building_physics.modelica.run import simulate
from noodl.apps.building_physics.modelica.schema import load
from tests.apps.building_physics.modelica.weather_files import (
    document,
    reader,
    tmy3_row,
    write_model,
)

F64 = torch.float64
PI = math.pi
H = 3600.0
P_ATM = 101325.0
LOW_RISE = "Buildings.Fluid.Sources.Outside_CpLowRise"
ORIFICE = {"name": "ori", "class": "Buildings.Airflow.Multizone.Orifice",
           "parameters": {"A": 0.01, "CD": 0.65, "m": 0.5, "dp_turbulent": 0.1}}


def _wall(name: str, azi: float, s: float = 5.0) -> dict:
    return {"name": name, "class": LOW_RISE,
            "parameters": {"Cp0": 0.6, "s": s, "azi": azi, "nPorts": 1}}


def _two_walls(rows, tmp_path, reader_parameters=None, signals=(), medium="Buildings.Media.Air",
               experiment=None, files=True, walls=None):
    """weaDat -> west (azi = W) and east (azi = E), an orifice from west to east."""
    west, east = walls or (_wall("west", PI / 2), _wall("east", -PI / 2))
    doc = document(
        [reader(reader_parameters), west, east, ORIFICE],
        [("weaDat.weaBus", "west.weaBus"), ("east.weaBus", "weaDat.weaBus"),
         ("west.ports[1]", "ori.port_a"), ("ori.port_b", "east.ports[1]")],
        signals, medium=medium, experiment=experiment)
    return write_model(tmp_path, doc, rows, files=files)


def _a180(s):  # windPressureLowRise.mo:19-21
    G = math.log(s)
    return math.log(1.248 - 0.703 + 0.131 * math.sin(2 * PI * G) ** 3 + 0.071 * G ** 2)


def _closed_form(cp, v):
    """Outside_CpLowRise.mo:21-45 with Buildings.Media.Air (d = p dStp/pStp): p = pAtm /
    (1 - 0.5 Cp v^2 dStp/pStp)."""
    return P_ATM / (1.0 - 0.5 * cp * v * v * 1.2 / 101325.0)


def test_low_rise_walls_set_their_boundary_pressures_from_the_wind(tmp_path):
    """A 4 m/s west wind (270 deg): normal onto the west-facing wall (Cp = Cp0), from behind
    onto the east-facing one (incidence pi: Cp = Cp0 a180)."""
    rows = [tmy3_row(k * H) for k in range(4)]
    model, state, drivers, names = read_modelica(_two_walls(rows, tmp_path), return_names=True)
    hist = simulate(model, state, drivers, names.times[:3])
    p_west, p_east = _closed_form(0.6, 4.0), _closed_form(0.6 * _a180(5.0), 4.0)
    p = hist["p"]
    torch.testing.assert_close(p[:, names.nodes["west"]], torch.full((3,), p_west, dtype=F64),
                               rtol=1e-13, atol=0)
    torch.testing.assert_close(p[:, names.nodes["east"]], torch.full((3,), p_east, dtype=F64),
                               rtol=1e-13, atol=0)
    assert names.weather["west.pWin"][:3].tolist() == pytest.approx([p_west - P_ATM] * 3,
                                                                    rel=1e-9, abs=0.0)
    (col, sign), = names.edges["ori"]
    assert bool((sign * hist["air.q"][:, col] > 0).all())  # west -> east


def test_outside_humidity_comes_from_the_relative_humidity(tmp_path):
    """BaseClasses/Outside.mo:49-53: X = X_pTphi(p, T, relHum) at the boundary's own
    pressure; 10 degC is above the triple point, so pSat is over liquid water."""
    rows = [tmy3_row(k * H) for k in range(4)]
    room = {"name": "room", "class": "Buildings.Fluid.MixingVolumes.MixingVolume",
            "parameters": {"V": 62.5, "nPorts": 2, "m_flow_nominal": 0.01,
                           "energyDynamics": "Modelica.Fluid.Types.Dynamics.FixedInitial",
                           "massDynamics": "Modelica.Fluid.Types.Dynamics.FixedInitial"}}
    ori2 = dict(ORIFICE, name="ori2")
    doc = document(
        [reader(), _wall("west", PI / 2), _wall("east", -PI / 2), room, ORIFICE, ori2],
        [("weaDat.weaBus", "west.weaBus"), ("east.weaBus", "weaDat.weaBus"),
         ("west.ports[1]", "ori.port_a"), ("ori.port_b", "room.ports[1]"),
         ("room.ports[2]", "ori2.port_a"), ("ori2.port_b", "east.ports[1]")])
    model, state, drivers, names = read_modelica(write_model(tmp_path, doc, rows),
                                                 return_names=True)
    hist = simulate(model, state, drivers, names.times[:2])
    T, phi, k = 283.15, 0.5, 0.621964713077499
    p = _closed_form(0.6, 4.0)
    ps = 611.657 * math.exp(17.2799 - 4102.99 / (T - 35.719))
    x_west = phi * k / (k * phi + p / ps - phi)
    assert float(hist["X_w"][-1, names.nodes["west"]]) == pytest.approx(x_west, rel=1e-13,
                                                                        abs=0.0)
    assert bool(torch.isfinite(hist["T"]).all() and torch.isfinite(hist["X_w"]).all())


def test_reader_input_sources_come_from_signals(tmp_path):
    rows = [tmy3_row(k * H) for k in range(4)]
    ramp = {"name": "dir", "class": "Modelica.Blocks.Sources.Ramp",
            "parameters": {"height": 2 * PI, "duration": 3600.0, "offset": 0.0,
                           "startTime": 0.0},
            "drives": "weaDat.winDir_in"}
    path = _two_walls(rows, tmp_path, {
        "winDirSou": "Buildings.BoundaryConditions.Types.DataSource.Input"}, signals=[ramp],
        experiment={"StartTime": 0.0, "StopTime": 3600.0, "Interval": 600.0})
    _, _, _, names = read_modelica(path, return_names=True)
    t = names.times[:-1]
    torch.testing.assert_close(names.weather["weaDat.weaBus.winDir"][:-1],
                               torch.remainder(2 * PI * t / 3600.0, 2 * PI), rtol=1e-14,
                               atol=1e-14)


def test_a_reader_whose_file_did_not_travel_is_refused_by_name(tmp_path):
    rows = [tmy3_row(k * H) for k in range(4)]
    with pytest.raises(ModelicaImportError, match=r"weaDat \(.*ReaderTMY3\): .*does not travel"):
        read_modelica(_two_walls(rows, tmp_path, files=False))


@pytest.mark.parametrize(("connection", "match"), [
    (("weaDat.weaBus.HDirNor", "gai.u"), r"weaDat \(.*\): weather-bus output HDirNor"),
    (("weaDat.weaBus.TDryBul", "gai.u"), r"weaDat \(.*\): its weather bus is wired to gai.u"),
    (("weaDat.weaBus", "bou.weaBus"), r"weaDat \(.*\): its weather bus is wired to bou.weaBus"),
])
def test_other_weather_bus_wiring_is_refused_by_name(tmp_path, connection, match):
    rows = [tmy3_row(k * H) for k in range(4)]
    bou = {"name": "bou", "class": "Buildings.Fluid.Sources.Boundary_pT",
           "parameters": {"p": P_ATM, "T": 293.15, "nPorts": 1}}
    gai = {"name": "gai", "class": "Modelica.Blocks.Math.Gain", "parameters": {"k": 1.0},
           "drives": "bou.p_in"}
    doc = document([reader(), _wall("west", PI / 2), bou, ORIFICE],
                   [("weaDat.weaBus", "west.weaBus"), ("west.ports[1]", "ori.port_a"),
                    ("ori.port_b", "bou.ports[1]"), connection], [gai])
    with pytest.raises(ModelicaImportError, match=match):
        read_modelica(write_model(tmp_path, doc, rows))


def test_two_readers_on_one_boundary_are_refused(tmp_path):
    rows = [tmy3_row(k * H) for k in range(4)]
    second = dict(reader(), name="weaDat2")
    doc = document([reader(), second, _wall("west", PI / 2), _wall("east", -PI / 2), ORIFICE],
                   [("weaDat.weaBus", "west.weaBus"), ("weaDat2.weaBus", "west.weaBus"),
                    ("weaDat.weaBus", "east.weaBus"), ("west.ports[1]", "ori.port_a"),
                    ("ori.port_b", "east.ports[1]")])
    with pytest.raises(ModelicaImportError, match=r"west \(.*\): .*more than one weather reader"):
        read_modelica(write_model(tmp_path, doc, rows))


def _signal_outside(medium: str) -> dict:
    out = {"name": "out", "class": "Buildings.Fluid.Sources.Outside",
           "parameters": {"nPorts": 1}}
    bou = {"name": "bou", "class": "Buildings.Fluid.Sources.Boundary_pT",
           "parameters": {"p": P_ATM, "T": 293.15, "nPorts": 1}}
    signals = [
        {"name": "pa", "class": "Modelica.Blocks.Sources.Constant",
         "parameters": {"k": P_ATM + 5.0}, "drives": "out.weaBus.pAtm"},
        {"name": "ta", "class": "Modelica.Blocks.Sources.Constant",
         "parameters": {"k": 283.15}, "drives": "out.weaBus.TDryBul"},
    ]
    return document([out, bou, ORIFICE],
                    [("out.ports[1]", "ori.port_a"), ("ori.port_b", "bou.ports[1]")],
                    signals, medium=medium)


def test_moist_outside_driven_by_signals_needs_relative_humidity(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(_signal_outside("Buildings.Media.Air")))
    with pytest.raises(ModelicaImportError, match=r"out \(.*Outside\): weaBus.relHum"):
        read_modelica(path)
    path.write_text(json.dumps(_signal_outside("Modelica.Media.Air.SimpleAir")))
    model, state, drivers, names = read_modelica(path, return_names=True)
    hist = simulate(model, state, drivers, names.times[:2])
    assert float(hist["p"][0, names.nodes["out"]]) == P_ATM + 5.0


def test_weather_knots_and_limit_crossings_are_step_boundaries_without_restart(tmp_path):
    rh = [90.0, 99.9, 99.9, 90.0]
    rows = [tmy3_row(k * H, rh=rh[k]) for k in range(4)]
    doc = load(_two_walls(rows, tmp_path,
                          experiment={"StartTime": 0.0, "StopTime": 10800.0,
                                      "Interval": 5400.0}))
    grid = assemble.driver_grid(doc).tolist()
    v = (1.0 - math.sqrt(1.0 - 4.0 * 0.1 / 4.95)) / 2.0
    for t in (H, 2 * H, H * (1 + v), H * (2 - v), 5400.0):
        assert min(abs(g - t) for g in grid) < 1e-6, t
    jumps, kinks = assemble.grid_events(doc, assemble.driver_grid(doc))
    assert jumps.numel() == 0 and kinks.numel() == 0


def test_weather_columns_without_a_flow_element(tmp_path):
    """MBL's Fluid.Sources examples are boundaries with no flow element: read_modelica refuses
    them, weather_columns still evaluates their weather and wind pressure."""
    rows = [tmy3_row(k * H) for k in range(4)]
    doc = document([reader(), _wall("west", PI / 2), _wall("east", -PI / 2)],
                   [("weaDat.weaBus", "west.weaBus"), ("east.weaBus", "weaDat.weaBus")])
    path = write_model(tmp_path, doc, rows)
    with pytest.raises(ModelicaImportError, match="no flow element"):
        read_modelica(path)
    cols = assemble.weather_columns(load(path), torch.tensor([0.0, 600.0], dtype=F64))
    assert {"weaDat.weaBus.TDryBul", "weaDat.weaBus.pAtm", "weaDat.weaBus.relHum",
            "weaDat.weaBus.winSpe", "weaDat.weaBus.winDir", "west.pWin", "east.pWin"} <= set(cols)
    assert cols["west.pWin"].tolist() == pytest.approx([_closed_form(0.6, 4.0) - P_ATM] * 2,
                                                       rel=1e-9, abs=0.0)


def test_calm_weather_gives_no_wind_pressure_and_no_flow(tmp_path):
    """San Francisco's first hours are calm: wind speed 0 at the knots, and the Akima spline
    below 0 between them, cut to 0 by LimiterWindSpeed."""
    ws = [0.0, 0.0, 2.0, 2.0]
    rows = [tmy3_row(k * H, ws=ws[k]) for k in range(4)]
    model, state, drivers, names = read_modelica(
        _two_walls(rows, tmp_path, experiment={"StartTime": 0.0, "StopTime": 3600.0,
                                               "Interval": 600.0}), return_names=True)
    hist = simulate(model, state, drivers, names.times)
    assert names.weather["weaDat.weaBus.winSpe"].tolist() == [0.0] * names.times.numel()
    assert names.weather["west.pWin"].tolist() == [0.0] * names.times.numel()
    assert hist["air.q"].abs().max().item() == 0.0
    assert bool(torch.isfinite(hist["p"]).all())


# ------------------------------------------------------- Outside_Cp and Outside_CpData
def test_outside_cp_takes_its_coefficient_from_the_parameter_or_the_input(tmp_path):
    """Outside_Cp.mo:9, 32-37: `pWin = 0.5 Cp v^2 d(pAtm, T, X_default)`, `Cp = 0.6` unless
    given; with `use_Cp_in` the coefficient is the signal driving `Cp_in`. Buildings.Media.Air:
    `d = pAtm dStp/pStp` = 1.2 kg/m3 at 101325 Pa."""
    rows = [tmy3_row(k * H) for k in range(4)]
    west = {"name": "west", "class": "Buildings.Fluid.Sources.Outside_Cp",
            "parameters": {"nPorts": 1}}
    east = {"name": "east", "class": "Buildings.Fluid.Sources.Outside_Cp",
            "parameters": {"nPorts": 1, "use_Cp_in": True}}
    cp_in = {"name": "cpEas", "class": "Modelica.Blocks.Sources.Constant",
             "parameters": {"k": -0.25}, "drives": "east.Cp_in"}
    path = _two_walls(rows, tmp_path, signals=[cp_in], walls=(west, east))
    model, state, drivers, names = read_modelica(path, return_names=True)
    hist = simulate(model, state, drivers, names.times[:2])
    for name, cp in (("west", 0.6), ("east", -0.25)):
        p_win = 0.5 * cp * 16.0 * 1.2
        assert names.weather[f"{name}.pWin"][:2].tolist() == pytest.approx(
            [p_win] * 2, rel=1e-13, abs=0.0)
        assert hist["p"][:, names.nodes[name]].tolist() == pytest.approx(
            [P_ATM + p_win] * 2, rel=1e-15, abs=0.0)


def _cp_data(name: str, azi: float, **extra) -> dict:
    return {"name": name, "class": "Buildings.Fluid.Sources.Outside_CpData",
            "parameters": {"azi": azi, "incAngSurNor": [0.0, PI / 2, PI, 3 * PI / 2],
                           "Cp": [0.5, -0.3, -0.2, -0.3], "nPorts": 1, **extra}}


def test_outside_cp_data_interpolates_its_profile_and_scales_by_cs(tmp_path):
    """Outside_CpData.mo:16-21, 39-55: the table extended to x = {-pi, 0, pi/2, pi, 3pi/2,
    2pi, 5pi/2}, y = {-0.2, 0.5, -0.3, -0.2, -0.3, 0.5, -0.3}, slopes the mean of the adjacent
    secants (splineDerivatives.mo:40-51). A wind from 315 deg is at incidence pi/4 on the
    west-facing wall: midway between knots 0 and pi/2, with slopes -0.45/pi and -0.7/pi the
    cubic Hermite gives 0.1 + (pi/2)(0.25/pi)/8 = 0.115625. On the east-facing wall it is
    5pi/4, slopes 0 and 0.7/pi: -0.25 - (pi/2)(0.7/pi)/8 = -0.29375. `pWin = Cs 0.5 Cp d v^2`
    with `d` at the boundary's own pressure (`p = pAtm / (1 - Cs 0.5 Cp v^2 dStp/pStp)`)."""
    rows = [tmy3_row(k * H, wd=315.0) for k in range(4)]
    walls = (_cp_data("west", PI / 2, Cs=0.8), _cp_data("east", -PI / 2, Cs=0.8))
    model, state, drivers, names = read_modelica(_two_walls(rows, tmp_path, walls=walls),
                                                 return_names=True)
    hist = simulate(model, state, drivers, names.times[:2])
    for name, cp in (("west", 0.115625), ("east", -0.29375)):
        p = _closed_form(0.8 * cp, 4.0)
        assert hist["p"][:, names.nodes[name]].tolist() == pytest.approx([p] * 2, rel=1e-13,
                                                                          abs=0.0)
        assert names.weather[f"{name}.pWin"][:2].tolist() == pytest.approx(
            [p - P_ATM] * 2, rel=1e-9, abs=0.0)


def test_outside_cp_data_with_a_table_mbl_rejects_is_refused_by_name(tmp_path):
    """Outside_CpData.mo:73-82: the first incidence angle must be 0."""
    rows = [tmy3_row(k * H) for k in range(4)]
    east = _cp_data("east", -PI / 2)
    east["parameters"]["incAngSurNor"] = [0.1, PI / 2, PI, 3 * PI / 2]
    walls = (_cp_data("west", PI / 2), east)
    with pytest.raises(ModelicaImportError,
                       match=r"refused 1 item:\n  - east \(.*Outside_CpData\): the first "
                             r"incidence angle must be 0"):
        read_modelica(_two_walls(rows, tmp_path, walls=walls))
