"""Subcatchment hydrology (`runoff.py`, `inp_hydrology.py`): the rain gage, the nonlinear
reservoir and the Runge-Kutta solver, infiltration, the runoff clock, the reader's refusals,
and the gradient of the runoff. Agreement with SWMM 5.2.4 itself is in
`tests/verification/test_sewer_runoff_parity.py`."""

import dataclasses
import datetime
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave
from noodl.apps.sewer.inflows import TimeSeries
from noodl.apps.sewer.runoff import (
    ONE_SECOND,
    CurveNumber,
    Evaporation,
    GreenAmpt,
    Horton,
    Hydrology,
    RainGage,
    RunoffInflows,
    RunoffModel,
    Subareas,
    Subcatchment,
    odesolve_integrate,
)
from noodl.apps.sewer.xsect_tables import QCF

DATA = Path(__file__).resolve().parents[2] / "data" / "sewer"
HORTON = DATA / "runoff_horton_dynwave.inp"
START = datetime.datetime(2026, 1, 1, 6, 0, 0)


def _edit(tmp_path, edits=(), name="edited.inp", base=HORTON):
    text = base.read_text()
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / name
    path.write_text(text)
    return path


def _read(tmp_path, edits=()):
    return read_swmm_inp(_edit(tmp_path, edits))[0]


# ---- the Runge-Kutta solver ---------------------------------------------------------------
def test_ode_solver_integrates_a_linear_decay_to_its_tolerance():
    """y' = -y from 1: y(1) = 1/e within the step control's tolerance (ODETOL 1e-4)."""
    y = odesolve_integrate(1.0, 0.0, 1.0, 1e-4, 1.0, lambda y: -y)
    assert y == pytest.approx(math.exp(-1.0), rel=1e-4)
    tight = odesolve_integrate(1.0, 0.0, 1.0, 1e-10, 1.0, lambda y: -y)
    assert tight == pytest.approx(math.exp(-1.0), rel=1e-10)


def test_ode_solver_leaves_the_value_when_it_fails():
    """SWMM writes `ystart` only on success: a stiff problem that needs more than its 10000
    steps leaves the starting value."""
    assert odesolve_integrate(1.0, 0.0, 1.0, 1e-4, 1.0, lambda y: -1e6 * y) == 1.0


def test_ode_solver_is_differentiable_in_the_initial_value():
    y0 = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    y = odesolve_integrate(y0, 0.0, 0.5, 1e-9, 0.5, lambda y: -3.0 * y)
    y.backward()
    assert float(y0.grad) == pytest.approx(math.exp(-1.5), rel=1e-7)


# ---- the rain gage -------------------------------------------------------------------------
def _hydrology(series, kind="volume", interval=300.0, subs=None, **kw):
    """One subcatchment on a gage with `series` {seconds: value}."""
    times, values = zip(*sorted(series.items()), strict=True)
    gage = RainGage("G", TimeSeries("R", tuple(float(t) for t in times), tuple(values)), kind,
                    interval)
    sub = subs or Subcatchment(
        "S", "G", "J", area=10000.0, pct_imperv=100.0, width=100.0, slope=0.01,
        subareas=Subareas(0.013, 0.2, 0.001, 0.003, 25.0))
    defaults = dict(start=START, total_duration=7200.0, wet_step=60.0, dry_step=600.0,
                    us_units=False)
    defaults.update(kw)
    return Hydrology((gage,), (sub,), subcatchments_override(sub), **defaults) \
        if False else Hydrology((gage,), (sub,), **defaults)


def subcatchments_override(sub):  # pragma: no cover - keeps the helper readable
    return (sub,)


def _rain_at(h, seconds):
    """The gage's rainfall (mm/hr) at `seconds` into the run."""
    m = RunoffModel(h)
    g = m.gages[0]
    g.set_state(seconds)
    return g.rainfall


def test_gage_volume_intensity_cumulative_and_zero_skipping():
    """A VOLUME entry is a depth over the interval, the rain applies from its date for one
    interval and is 0 until the next non-zero entry; CUMULATIVE differences restart when the
    total falls; INTENSITY is as given."""
    vol = _hydrology({0: 6.0, 300: 0.0, 600: 0.0, 900: 3.0})            # mm per 300 s
    assert _rain_at(vol, 10.0) == pytest.approx(6.0 / 300.0 * 3600.0)
    assert _rain_at(vol, 310.0) == 0.0
    assert _rain_at(vol, 950.0) == pytest.approx(3.0 / 300.0 * 3600.0)
    assert _rain_at(vol, 1300.0) == 0.0                                   # after the last
    cum = _hydrology({0: 0.0, 300: 2.0, 600: 5.0, 900: 1.0}, kind="cumulative")
    assert _rain_at(cum, 10.0) == 0.0
    assert _rain_at(cum, 310.0) == pytest.approx(2.0 / 300.0 * 3600.0)
    assert _rain_at(cum, 610.0) == pytest.approx(3.0 / 300.0 * 3600.0)
    assert _rain_at(cum, 910.0) == pytest.approx(1.0 / 300.0 * 3600.0)   # the total fell
    inten = _hydrology({0: 20.0, 600: 0.0}, kind="intensity")
    assert _rain_at(inten, 100.0) == 20.0
    assert _rain_at(inten, 400.0) == 0.0 and _rain_at(inten, 700.0) == 0.0     # 300 s only


def test_gage_record_starting_after_the_simulation_is_dry_before_it():
    h = _hydrology({600: 4.0, 900: 4.0})
    assert _rain_at(h, 100.0) == 0.0
    assert _rain_at(h, 600.0) > 0.0 or _rain_at(h, 601.0) > 0.0


def test_gage_looks_a_second_ahead_at_a_boundary():
    """A date within `ONE_SECOND` before an entry already sees it (gage_setState adds a second)."""
    h = _hydrology({0: 4.0, 300: 8.0})
    assert _rain_at(h, 300.0 - 2.0 * ONE_SECOND) == pytest.approx(4.0 / 300.0 * 3600.0)
    assert _rain_at(h, 300.0 - 0.5 * ONE_SECOND) == pytest.approx(8.0 / 300.0 * 3600.0)


def test_co_gages_take_the_rainfall_of_the_first():
    sub2 = dataclasses.replace(
        Subcatchment("S2", "G2", "J", area=1.0, pct_imperv=50.0, width=1.0, slope=0.01))
    base = _hydrology({0: 4.0, 300: 0.0})
    series = base.gages[0].series
    h = Hydrology((base.gages[0], RainGage("G2", series, "volume", 300.0)),
                  base.subcatchments + (sub2,), START, 3600.0, 60.0, 600.0, False)
    m = RunoffModel(h)
    assert m.gages[1].co_gage is m.gages[0] and m.gages[0].co_gage is None


# ---- the runoff clock ---------------------------------------------------------------------
def test_runoff_steps_follow_the_weather():
    """The dry step is cut at the next rain date, the wet step applies while it rains and
    until the runoff stops, the dry step again after, and the last step ends the run."""
    h = _hydrology({600: 6.0, 900: 0.0}, total_duration=3000.0)
    m = RunoffModel(h)
    m.advance_to(1e9)
    starts = [s for s, _ in m.steps]
    assert m.steps[0] == (0.0, 600.0)                       # dry step cut at the rain
    assert m.steps[1] == (600.0, 60.0)                      # raining: the wet step
    assert all(length == 60.0 for _, length in m.steps[1:6])
    assert m.steps[-1][0] + m.steps[-1][1] == 3000.0
    assert starts == sorted(starts)
    m2 = RunoffModel(h)
    m2.advance_to(0.0)
    assert m2.steps == []                                    # nothing is due at t = 0


def test_runoff_is_zero_before_the_rain_and_positive_after():
    h = _hydrology({600: 6.0, 900: 0.0}, total_duration=3000.0)
    m = RunoffModel(h)
    m.advance_to(500000.0)
    assert m.runoff["S"] == 0.0
    m.advance_to(700000.0)
    assert m.runoff["S"] > 0.0


def test_weighted_outflow_interpolates_between_runoff_times():
    """routing.c addWetWeatherInflows: (1 - f) old + f new, f clamped to [0, 1]."""
    h = _hydrology({0: 6.0, 300: 6.0}, total_duration=3000.0)
    m = RunoffModel(h)
    m.advance_to(100000.0)
    assert (m.old_ms, m.new_ms) == (60000.0, 120000.0)
    old, new = m.subs[0].old_runoff, m.subs[0].new_runoff
    assert new > old > 0.0
    assert m.weighted(90000.0)[0] == pytest.approx(0.5 * (old + new))
    assert m.weighted(10000.0)[0] == old and m.weighted(500000.0)[0] == new
    assert m.outflow_by_node(120000.0)["J"] == pytest.approx(new * QCF)


def test_imperviousness_and_storage_close_the_volume_balance():
    """A fully impervious, no-infiltration subcatchment sheds exactly the rain that is not
    left on the surface: the runoff volume plus the final storage is the rain volume."""
    sub = Subcatchment("S", "G", "J", area=2000.0, pct_imperv=100.0, width=50.0, slope=0.02,
                       subareas=Subareas(0.013, 0.2, 0.0005, 0.003, 25.0))
    h = _hydrology({0: 2.0, 300: 4.0, 600: 0.0}, subs=sub, total_duration=7200.0)
    m = RunoffModel(h)
    volume = 0.0
    last = 0.0
    while m.new_ms < 7200000.0:
        m.step()
        start, length = m.steps[-1]
        volume += 0.5 * (last + m.subs[0].new_runoff) * length * 0.3048 ** 3 * 1.0
        last = m.subs[0].new_runoff
    rain = (2.0 + 4.0) / 1000.0 * 2000.0
    # the water left on the 75 % of the surface with depression storage (0.5 mm) and the
    # trapezoid rule on the hydrograph bound the closure; the rest must have run off
    assert volume == pytest.approx(rain, rel=0.1)
    assert volume < rain


def test_the_dry_step_is_raised_to_the_wet_step_and_wet_lowered_to_the_gage(tmp_path):
    net = _read(tmp_path, [("0:02     1.0      TIMESERIES R1", "0:00:30  1.0      TIMESERIES R1")])
    h = net.hydrology
    assert h.wet_step == 30.0 and h.dry_step == 300.0
    assert net.dynamic_wave.routing_step == 10.0
    net = _read(tmp_path, [("0:02     1.0      TIMESERIES R1", "0:00:04  1.0      TIMESERIES R1"),
                           ("0:00:04", "0:00:04")])
    assert net.hydrology.wet_step == 4.0 and net.dynamic_wave.routing_step == 4.0


def test_total_duration_follows_swmm_floating_point():
    """SWMM's total duration is floor(((EndDate - StartDate) + (EndTime - StartTime)) * 86400),
    which is 1199 s for 12:20:00 after 12:00:00 (the engine's last runoff step shows it)."""
    h = read_swmm_inp(DATA / "runoff_cn_mixed_lps_dynwave.inp")[0].hydrology
    assert h.total_duration == 1199.0
    assert read_swmm_inp(HORTON)[0].hydrology.total_duration == 1200.0


# ---- infiltration ---------------------------------------------------------------------------
def _infil_run(model, rain_mm_hr=80.0, minutes=20):
    """The runoff volume (m3, in cfs seconds) of a 100 % pervious subcatchment under a steady
    rain of `minutes`."""
    sub = Subcatchment("S", "G", "J", area=1000.0, pct_imperv=0.0, width=30.0, slope=0.02,
                       subareas=Subareas(0.013, 0.1, 0.001, 0.001, 25.0), infiltration=model)
    h = _hydrology({0: rain_mm_hr, minutes * 60: 0.0}, kind="intensity", subs=sub,
                   total_duration=minutes * 60.0, wet_step=60.0, dry_step=60.0,
                   interval=minutes * 60.0)
    m = RunoffModel(h)
    volume, last = 0.0, 0.0
    while m.new_ms < h.total_duration * 1000.0:
        m.step()
        volume += 0.5 * (last + m.subs[0].new_runoff) * m.steps[-1][1]
        last = m.subs[0].new_runoff
    return volume


@pytest.mark.parametrize("model", [
    Horton(50e-3 / 3600, 5e-3 / 3600, 4.0 / 3600, 7.0 * 86400),
    Horton(50e-3 / 3600, 5e-3 / 3600, 4.0 / 3600, 7.0 * 86400, 0.02, modified=True),
    GreenAmpt(0.1, 10e-3 / 3600, 0.3),
    GreenAmpt(0.1, 10e-3 / 3600, 0.3, modified=True),
    CurveNumber(80.0, 5 * 86400.0),
], ids=["horton", "modified-horton", "green-ampt", "modified-green-ampt", "curve-number"])
def test_infiltration_models_take_water_and_leave_the_rest_as_runoff(model):
    """With rain above the capacity every model lets some water in and sheds the rest; without
    infiltration all of it that is not stored runs off."""
    runs = _infil_run(model)
    assert runs > 0.0
    assert _infil_run(None) > runs


def test_horton_follows_its_decay_curve():
    """One step of a wet Horton soil integrates f = f_min + (f_0 - f_min) exp(-k t)."""
    from noodl.apps.sewer.runoff import ZERO, _Env, _HortonState

    p = Horton(40e-3 / 3600, 4e-3 / 3600, 3.0 / 3600, 5 * 86400.0)
    state = _HortonState(p)
    f0, fmin, k = p.f0 / 0.3048, p.fmin / 0.3048, p.decay
    dt = 60.0
    got = state.get_infil(dt, 1.0, 0.0, 0.0, _Env())               # plenty of rain (ft/s)
    expect = (fmin * dt + (f0 - fmin) / k * (1.0 - math.exp(-k * dt))) / dt
    assert got == pytest.approx(expect, rel=1e-12)
    assert state.tp == pytest.approx(dt) and ZERO > 0.0


# ---- evaporation ---------------------------------------------------------------------------
def test_evaporation_takes_from_ponded_water_first_and_dry_only_stops_in_rain():
    sub = Subcatchment("S", "G", "J", area=1000.0, pct_imperv=100.0, width=30.0, slope=0.02,
                       subareas=Subareas(0.013, 0.1, 0.002, 0.001, 0.0))
    rain = {0: 3.0, 120: 0.0}
    dry = _hydrology(rain, subs=sub, total_duration=1800.0, evaporation=Evaporation(
        "constant", (50.0,) * 12))
    wet = _hydrology(rain, subs=sub, total_duration=1800.0, evaporation=Evaporation(
        "constant", (50.0,) * 12, dry_only=True))
    none = _hydrology(rain, subs=sub, total_duration=1800.0)
    out = {}
    for name, h in (("evap", dry), ("dry_only", wet), ("none", none)):
        m = RunoffModel(h)
        peak = 0.0
        while m.new_ms < 1800000.0:
            m.step()
            peak = max(peak, m.subs[0].new_runoff)
        out[name] = (peak, m.subs[0].depth[1])
    assert out["evap"][0] < out["none"][0]
    assert out["evap"][1] <= out["none"][1]


# ---- the reader -----------------------------------------------------------------------------
def test_reader_gives_the_fixtures_hydrology(tmp_path):
    net = read_swmm_inp(HORTON)[0]
    h = net.hydrology
    assert [s.name for s in h.subcatchments] == ["S1", "S2", "S3"]
    s1, s2, s3 = h.subcatchments
    assert s2.outlet == "S1" and s2.outlet_is_subcatchment and not s1.outlet_is_subcatchment
    assert s1.area == pytest.approx(2.5e4 / 2.29568e-5 * 2.2956e-5 * 1.0, rel=1e-3)
    assert s1.slope == pytest.approx(0.015) and s1.width == pytest.approx(200.0, rel=1e-12)
    assert isinstance(s1.infiltration, Horton) and s2.infiltration.max_volume > 0.0
    assert s1.subareas.route_to == "outlet" and s1.subareas.pct_zero == 25.0
    assert h.gages[0].kind == "volume" and h.gages[0].interval == 120.0
    assert (h.wet_step, h.dry_step, h.total_duration) == (60.0, 300.0, 1200.0)
    assert h.start == datetime.datetime(2026, 1, 1, 6, 0, 0) and not h.us_units
    assert isinstance(net.inflows, RunoffInflows) and net.hydrology is net.inflows.hydrology
    assert net.inflows.nodes == ("J1", "J2")          # S2 drains onto S1, not to a node
    assert read_swmm_inp(DATA / "tree_dynwave.inp")[0].hydrology is None


def test_reader_keywords_match_by_prefix_as_swmm_does(tmp_path):
    net = _read(tmp_path, [("RG1              VOLUME ", "RG1              volumetric")])
    assert net.hydrology.gages[0].kind == "volume"
    net = _read(tmp_path, [("25         OUTLET\nS2", "25         IMPERV    40\nS2"),
                           ("0          OUTLET\n\n[INFIL", "20         PERV      60\n\n[INFIL")])
    assert net.hydrology.subcatchments[0].subareas.route_to == "impervious"
    assert net.hydrology.subcatchments[0].subareas.pct_routed == 40.0
    assert net.hydrology.subcatchments[2].subareas.route_to == "pervious"
    assert net.hydrology.subcatchments[2].subareas.pct_routed == 60.0


def test_a_subcatchment_without_infiltration_or_subareas_is_dry_of_runoff(tmp_path):
    line = "S1               80         10 "
    net = _read(tmp_path, [(line, ";" + line)])
    assert net.hydrology.subcatchments[0].infiltration is None
    line = "S3               0.015      0.10 "
    net = _read(tmp_path, [(line, ";" + line)])
    assert net.hydrology.subcatchments[2].subareas is None
    m = RunoffModel(net.hydrology)
    m.advance_to(1.2e6)
    assert m.runoff["S3"] == 0.0 and m.runoff["S1"] > 0.0


SNOW1 = ("200      1.5      0\n", "200      1.5      0     Snow1\n")
GROUND = ("[JUNCTIONS]", "[GROUNDWATER]\nS1 A1 J1 0 0.001 0.01 0 0 0\n\n[JUNCTIONS]")
QUALITY = ("[JUNCTIONS]", "[POLLUTANTS]\nTSS MG/L 3 0 0 0 NO * 0 0 0\n[LANDUSES]\nRes\n\n"
           "[JUNCTIONS]")
OPTION = "SKIP_STEADY_STATE    NO"


def _option(line):
    return (OPTION, OPTION + "\n" + line)


@pytest.mark.parametrize(("edits", "message"), [
    ([("[JUNCTIONS]", "[LID_USAGE]\nS1 bioswale 1 1 0 0 0 0\n\n[JUNCTIONS]")], "LID_USAGE"),
    ([GROUND], "GROUNDWATER"),
    ([SNOW1], "snow pack"),
    ([QUALITY], "LANDUSES"),
    ([("[JUNCTIONS]", "[POLLUTANTS]\nTSS MG/L 3 0 0 0 NO * 0 0 0\n\n[JUNCTIONS]")], "Crain"),
    ([("1.0      TIMESERIES R1", "1.0      FILE rain.dat STA1 IN")], "from a file"),
    ([("CONSTANT         0.0", "TEMPERATURE")], "TEMPERATURE"),
    ([("CONSTANT         0.0", "FILE")], "FILE"),
    ([("S1               RG1 ", "S1               RGX ")], "rain gage 'RGX'"),
    ([("RG1              J1 ", "RG1              Nowhere ")], "neither a node nor a sub"),
    ([("RG1              S1 ", "RG1              J1 "),
      ("[JUNCTIONS]", "[SUBCATCHMENTS]\nJ1 RG1 J2 1 50 100 1 0\n\n[JUNCTIONS]")],
     "both a node and a subcatchment"),
    ([("2.5      60 ", "2.5      -60")], "negative"),
    ([("S1               80         10 ", "S1               10         80 ")],
     "ERR_INFIL_PARAMS"),
    ([("25         OUTLET\nS2", "25         ELSEWHERE\nS2")], "RouteTo"),
    ([("END_TIME             06:20:00", "END_TIME             05:00:00")], "end after"),
    ([("FIXED      9.6              NO", "FIXED      9.6              NO       S1")],
     "Route To"),
    ([("ROUTING_STEP         10", "ROUTING_STEP         10\nINFILTRATION         SOIL")],
     "INFILTRATION"),
], ids=["lid", "groundwater", "snowpack", "washoff", "rain-concentration", "gage-file",
        "evap-temperature", "evap-file", "unknown-gage", "unknown-outlet", "ambiguous-outlet",
        "negative", "infil-range", "route-to", "duration", "outfall-route", "infil-option"])
def test_reader_refuses_by_name(tmp_path, edits, message):
    with pytest.raises(ValueError, match=message):
        _read(tmp_path, edits)


def test_ignore_options_let_the_sections_through(tmp_path):
    with pytest.raises(ValueError, match="IGNORE_GROUNDWATER"):
        _read(tmp_path, [GROUND])
    assert _read(tmp_path, [GROUND, _option("IGNORE_GROUNDWATER   YES")]).hydrology
    assert _read(tmp_path, [SNOW1, _option("IGNORE_SNOWMELT      YES")]).hydrology
    assert _read(tmp_path, [QUALITY, _option("IGNORE_QUALITY       YES")]).hydrology


def test_unused_snowpacks_gages_and_inert_sections_are_ignored(tmp_path):
    inert = ("[JUNCTIONS]", "[SNOWPACKS]\nSnow1 PLOWABLE 0 0 0 0 0 0 0\n\n[TEMPERATURE]\n"
             "TIMESERIES T1\n\n[AQUIFERS]\nA1 0.5 0.15 0.3 0.1 0.2 0.1 0.1 0.5 0 0 0\n\n"
             "[LID_CONTROLS]\nbio BC\n\n[JUNCTIONS]")
    assert _read(tmp_path, [inert]).hydrology is not None
    # a gage no subcatchment names is not read at all (not even its FILE source)
    spare = ("[SUBCATCHMENTS]", "[RAINGAGES]\nRG9 VOLUME 0:05 1.0 FILE x.dat STA IN\n\n"
             "[SUBCATCHMENTS]")
    net = _read(tmp_path, [spare])
    assert [g.name for g in net.hydrology.gages] == ["RG1"]


def test_ignore_rainfall_gives_no_rain(tmp_path):
    net = _read(tmp_path, [_option("IGNORE_RAINFALL      YES")])
    m = RunoffModel(net.hydrology)
    m.advance_to(1.2e6)
    assert all(v == 0.0 for v in m.runoff.values()) and not m.is_raining


def test_subareas_before_their_subcatchment_are_refused(tmp_path):
    text = HORTON.read_text()
    block_start = text.index("[SUBAREAS]")
    block_end = text.index("[INFILTRATION]")
    block = text[block_start:block_end]
    moved = text[:block_start] + text[block_end:]
    moved = moved.replace("[SUBCATCHMENTS]", block + "[SUBCATCHMENTS]", 1)
    path = tmp_path / "moved.inp"
    path.write_text(moved)
    with pytest.raises(ValueError, match="before its line in"):
        read_swmm_inp(path)


# ---- the inflow driver -------------------------------------------------------------------------
def test_inflow_driver_is_step_aware_and_called_with_each_steps_length():
    """`simulate_dynamic_wave` calls a `step_aware` inflow as `inflow(t, step)`: the first step
    too, a truncated last step with its true length, and variable steps with theirs."""
    net = read_swmm_inp(DATA / "tree_dynwave.inp")[0]
    model, state, drivers = build_model(net, iterate_tol=1e-9, iterate_max=100)
    calls = []

    def inflow(t, step):
        calls.append((t, step))
        return drivers["inflow"]

    inflow.step_aware = True
    simulate_dynamic_wave(model, state, drivers, duration=25.0, dt=10.0, inflow=inflow)
    assert calls == [(0.0, 10.0), (10.0, 10.0), (20.0, 5.0)]
    plain = []
    simulate_dynamic_wave(model, state, drivers, steps=2, dt=10.0,
                          inflow=lambda t: plain.append(t) or drivers["inflow"])
    assert plain == [0.0, 10.0]


def test_driver_adds_runoff_to_the_schedule_and_drops_outfalls(tmp_path):
    net = read_swmm_inp(HORTON)[0]
    names = [m.name for m in net.manholes]
    driver = net.inflows.driver(names)
    q = driver(0.0, 10.0)
    assert q.shape == (len(names),) and float(q.abs().sum()) == 0.0
    q = driver(300.0, 10.0)
    assert float(q[names.index("J1")]) > 0.0 and float(q[names.index("J2")]) > 0.0
    assert float(q[names.index("J3")]) == 0.0
    with pytest.raises(ValueError, match="not among the model's nodes"):
        net.inflows.driver(["J1"])
    # a subcatchment draining to the outfall adds nothing (the outfall takes no lateral inflow)
    moved = _read(tmp_path, [("RG1              J2 ", "RG1              Outfall ")])
    assert moved.inflows.nodes == ("J1",)
    every = [m.name for m in moved.manholes] + ["Outfall"]
    q = moved.inflows.driver(every)(300.0, 10.0)
    assert float(q[every.index("Outfall")]) == 0.0 and float(q[every.index("J1")]) > 0.0


def test_runoff_inflows_combine_with_the_time_series_schedule(tmp_path):
    inflow_line = ("[JUNCTIONS]", "[INFLOWS]\nJ1 FLOW \"\" FLOW 1.0 1.0 0.01\n\n[JUNCTIONS]")
    net = _read(tmp_path, [inflow_line])
    assert net.inflows.schedule is not None and net.inflows.nodes[0] == "J1"
    names = [m.name for m in net.manholes]
    # the baseline has no time series or pattern, so it is the constant Manhole.inflow
    assert net.inflows.schedule.flows(0.0)["J1"] == pytest.approx(0.01)
    q = net.inflows.driver(names)(0.0, 10.0)
    assert float(q[names.index("J1")]) == pytest.approx(0.01)


def test_runoff_inflows_flows_and_with_duration():
    net = read_swmm_inp(HORTON)[0]
    flows = net.inflows.flows(300.0, 10.0)
    assert set(flows) <= {"J1", "J2"} and flows["J1"] > 0.0
    assert net.inflows.flows(130.0)["J1"] >= 0.0
    cut = net.inflows.with_duration(305.0)
    assert cut.hydrology.total_duration == 305.0 and cut.nodes == net.inflows.nodes


# ---- differentiability --------------------------------------------------------------------------
def _total_runoff(h, t_ms):
    m = RunoffModel(h)
    m.advance_to(t_ms)
    return sum(float(v) for v in m.runoff.values())


def _scaled(h, name, path, factor):
    subs = []
    for s in h.subcatchments:
        if s.name == name:
            if len(path) == 1:
                s = dataclasses.replace(s, **{path[0]: getattr(s, path[0]) * factor})
            else:
                obj = getattr(s, path[0])
                obj = dataclasses.replace(obj, **{path[1]: getattr(obj, path[1]) * factor})
                s = dataclasses.replace(s, **{path[0]: obj})
        subs.append(s)
    return dataclasses.replace(h, subcatchments=tuple(subs))


@pytest.mark.parametrize("key", [
    "S1.area", "S1.pct_imperv", "S1.width", "S1.slope", "S1.subareas.n_imperv",
    "S1.subareas.n_perv", "S1.subareas.s_imperv", "S1.subareas.s_perv",
    "S1.infiltration.f0", "S1.infiltration.fmin", "S1.infiltration.decay",
    "S2.infiltration.f0", "S3.infiltration.dry_time",
])
def test_runoff_gradient_matches_finite_differences(key):
    """Autograd through the whole runoff history against a central difference of the same code
    (tolerance 2e-3: the Runge-Kutta step control, which has a tolerance of 1e-4, is held fixed
    when differentiating)."""
    h = read_swmm_inp(HORTON)[0].hydrology
    t_ms = 600000.0
    h2, leaves = h.with_gradients()
    m = RunoffModel(h2)
    m.advance_to(t_ms)
    sum(m.runoff.values()).backward()
    got = float(leaves[key].grad) if leaves[key].grad is not None else 0.0
    name, rest = key.split(".", 1)
    path = tuple(rest.split("."))
    eps = 1e-6
    value = {s.name: s for s in h.subcatchments}[name]
    for part in path:
        value = getattr(value, part)
    fd = (_total_runoff(_scaled(h, name, path, 1 + eps), t_ms)
          - _total_runoff(_scaled(h, name, path, 1 - eps), t_ms)) / (2 * eps * value)
    assert got == pytest.approx(fd, rel=2e-3, abs=1e-8)


def test_gradient_reaches_the_routed_depths(tmp_path):
    """The lateral inflow is a differentiable driver: the depth of a junction after a few
    routing steps has a gradient with respect to the area of its subcatchment, positive where
    the area adds water, and it agrees with a finite difference."""
    path = _edit(tmp_path, [("R1               01/01/2026 06:02      3.0",
                             "R1               01/01/2026 06:00      9.0")])
    net = read_swmm_inp(path)[0]
    h2, leaves = net.hydrology.with_gradients(fields=("area",), subareas=(), infiltration=False,
                                              only=("S1",))
    model, state, drivers = build_model(net, iterate_tol=1e-10, iterate_max=500)
    sched = model.inflow_schedule
    model.inflow_schedule = RunoffInflows(h2, sched.schedule, sched.nodes)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=12, record=("water.phi",))
    j1 = list(model.net.nodes).index("J1")
    depth = hist["water.phi"][-1, j1]
    depth.backward()
    grad = float(leaves["S1.area"].grad)
    assert grad > 0.0
    area = float(net.hydrology.subcatchments[0].area)
    outs = []
    for factor in (1.0 + 1e-4, 1.0 - 1e-4):
        n2 = dataclasses.replace(net, inflows=RunoffInflows(
            _scaled(net.hydrology, "S1", ("area",), factor), sched.schedule, sched.nodes))
        m2, s2, d2 = build_model(n2, iterate_tol=1e-10, iterate_max=500)
        _, h = simulate_dynamic_wave(m2, s2, d2, steps=12, record=("water.phi",))
        outs.append(float(h["water.phi"][-1, j1]))
    assert grad == pytest.approx((outs[0] - outs[1]) / (2e-4 * area), rel=1e-3)


def test_a_snow_catch_factor_is_refused_only_where_snow_packs_make_it_matter(tmp_path):
    packs = ("[JUNCTIONS]", "[SNOWPACKS]\nSnow1 PLOWABLE 0 0 0 0 0 0 0\n\n[JUNCTIONS]")
    factor = ("1.0      TIMESERIES R1", "0.8      TIMESERIES R1")
    assert _read(tmp_path, [factor]).hydrology.gages[0].snow_factor == 0.8   # no snow packs
    with pytest.raises(ValueError, match="snow catch factor 0.8"):
        _read(tmp_path, [factor, packs])
    assert _read(tmp_path, [factor, packs, _option("IGNORE_SNOWMELT      YES")]).hydrology
