"""Time-varying lateral inflows: the readers of `[TIMESERIES]`, `[PATTERNS]`, `[INFLOWS]`,
`[DWF]` and `[FILES]`, and `InflowSchedule`. Agreement with live SWMM is in
tests/verification/test_sewer_inflows_parity.py."""

import datetime
from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave
from noodl.apps.sewer.inflows import (
    FLOW_TOL,
    SAMPLE_OFFSET,
    DryWeatherInflow,
    ExternalInflow,
    InflowSchedule,
    TimePattern,
    TimeSeries,
    parse_clock,
    parse_date,
    patterns_from_lines,
    tokens,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "sewer"
J1 = 'J1               FLOW             ""               FLOW     1.0      1.0      0.05'
START = datetime.datetime(2026, 1, 1)


def _inp(tmp_path, j1=J1, extra="", edits=()):
    """`base` with its J1 `[INFLOWS]` line replaced by `j1` and `extra` sections appended."""
    text = (DATA / "tree_kinwave.inp").read_text()
    assert text.count(J1) == 1
    text = text.replace(J1, j1) + "\n" + extra
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / "m.inp"
    path.write_text(text)
    return path


# ------------------------------------------------------------------ the pure pieces
def test_tokens_keep_quoted_fields_and_drop_comments():
    assert tokens('J1 FLOW "" FLOW 1 ; a comment "x"') == ["J1", "FLOW", "", "FLOW", "1"]
    assert tokens('"P-1"  HOURLY') == ["P-1", "HOURLY"]


def test_parse_date_and_clock():
    assert parse_date("01/02/2026") == datetime.date(2026, 1, 2)
    assert parse_date("2026-01-02") == datetime.date(2026, 1, 2)
    assert parse_date("02-JAN-2026") == datetime.date(2026, 1, 2)
    assert parse_date("JAN-02-2026") == datetime.date(2026, 1, 2)
    assert parse_date("10:30") is None and parse_date("-5") is None and parse_date("3") is None
    with pytest.raises(ValueError, match="not a date"):
        parse_date("13/45/2026")
    assert parse_clock("1.5") == 5400.0 and parse_clock("1:30") == 5400.0
    assert parse_clock("0:00:30") == 30.0 and parse_clock("abc") is None


def test_time_series_interpolates_and_is_zero_outside():
    series = TimeSeries("S", (10.0, 20.0, 40.0), (1.0, 3.0, 0.0))
    assert [series(t) for t in (0.0, 9.9, 10.0, 15.0, 20.0, 30.0, 40.0, 40.1)] == [
        0.0, 0.0, 1.0, 2.0, 3.0, 1.5, 0.0, 0.0]
    with pytest.raises(ValueError, match="strictly increase"):
        TimeSeries("S", (10.0, 10.0), (1.0, 2.0))


def test_patterns_continue_over_lines_pad_with_ones_and_ignore_extras():
    lines = [(1, ["H", "HOURLY", "2", "3"]), (2, ["H", "4"]), (3, ["M", "MONTHLY"] + ["5"] * 14),
             (4, ["W", "WEEKEND", "7"])]
    found = patterns_from_lines(lines, "m.inp")
    assert found["H"].factors == (2.0, 3.0, 4.0) + (1.0,) * 21
    assert found["M"].factors == (5.0,) * 12
    saturday, thursday = datetime.datetime(2026, 1, 3, 0), datetime.datetime(2026, 1, 1, 0)
    assert found["W"].factor(saturday) == 7.0 and found["W"].factor(thursday) == 1.0
    with pytest.raises(ValueError, match="needs its type"):
        patterns_from_lines([(1, ["X", "2"])], "m.inp")


def test_daily_pattern_starts_on_sunday():
    pattern = TimePattern("D", "daily", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
    assert pattern.factor(datetime.datetime(2026, 1, 4)) == 1.0      # a Sunday
    assert pattern.factor(datetime.datetime(2026, 1, 5)) == 2.0      # a Monday
    assert pattern.factor(datetime.datetime(2026, 1, 10)) == 7.0     # a Saturday


def test_the_weekend_pattern_replaces_the_hourly_one_on_weekends_only():
    hourly = TimePattern("H", "hourly", (2.0,) * 24)
    weekend = TimePattern("W", "weekend", (5.0,) * 24)
    dwf = DryWeatherInflow("J1", 10.0, (hourly, weekend))
    assert dwf.flow(datetime.datetime(2026, 1, 1, 3)) == 20.0        # Thursday
    assert dwf.flow(datetime.datetime(2026, 1, 3, 3)) == 50.0        # Saturday
    only_weekend = DryWeatherInflow("J1", 10.0, (weekend,))
    assert only_weekend.flow(datetime.datetime(2026, 1, 1, 3)) == 10.0
    with pytest.raises(ValueError, match="one pattern per kind"):
        DryWeatherInflow("J1", 1.0, (hourly, hourly))


def test_the_schedule_samples_a_millisecond_after_the_step_start_and_drops_small_flows():
    series = TimeSeries("S", (0.0, 3600.0), (0.0, 3.6))             # 0.001 m3/s per s
    big = ExternalInflow("A", series)
    small = ExternalInflow("B", None, 1.0, 0.5 * FLOW_TOL)
    schedule = InflowSchedule(START, (big, small), (DryWeatherInflow("B", 2.0 * FLOW_TOL),))
    flows = schedule.flows(100.0)
    assert flows["A"] == pytest.approx(0.001 * (100.0 + SAMPLE_OFFSET), rel=1e-12)
    assert flows["B"] == pytest.approx(2.0 * FLOW_TOL, rel=1e-12)   # the series part is dropped
    assert schedule.nodes == ("A", "B")


def test_driver_returns_a_full_node_vector_and_refuses_an_unknown_node():
    schedule = InflowSchedule(START, (ExternalInflow("B", None, 1.0, 0.25),))
    inflow = schedule.driver(["A", "B", "C"])
    assert torch.equal(inflow(7.0), torch.tensor([0.0, 0.25, 0.0], dtype=torch.float64))
    assert torch.equal(schedule.at(0.0, ["B"]), torch.tensor([0.25], dtype=torch.float64))
    with pytest.raises(ValueError, match=r"\['B'\].*not among"):
        schedule.driver(["A", "C"])


# ------------------------------------------------------------------------ the reader
def test_constant_inflows_give_no_schedule_and_the_old_manhole_values():
    net, _, _ = read_swmm_inp(DATA / "tree_steady.inp")
    assert net.inflows is None
    assert {m.name: m.inflow for m in net.manholes}["J1"] == 0.05


def test_a_series_inflow_attaches_a_schedule_and_the_start_value(tmp_path):
    extra = "[TIMESERIES]\nS1 0 0.0 1 3.6\n"
    j1 = 'J1  FLOW  S1  FLOW  9.0  2.0  0.01'      # Mfactor 9 is ignored for a flow
    net, _, _ = read_swmm_inp(_inp(tmp_path, j1, extra))
    schedule = net.inflows
    assert schedule is not None and schedule.start == datetime.datetime(2026, 1, 1)
    # Sfactor 2 scales the series (3.6 per hour = 0.001/s), the baseline adds
    assert schedule.flows(1800.0)["J1"] == pytest.approx(0.01 + 2.0 * 1.8, rel=1e-6)
    assert schedule.flows(7200.0)["J1"] == pytest.approx(0.01)       # past the series: 0
    j1_manhole = next(m for m in net.manholes if m.name == "J1")
    assert j1_manhole.inflow == pytest.approx(schedule.flows(0.0)["J1"])
    assert schedule.flows(0.0)["J2"] == 0.08                         # a constant joins in


def test_the_last_inflow_line_of_a_node_wins(tmp_path):
    net, _, _ = read_swmm_inp(_inp(tmp_path, J1 + "\nJ1  FLOW  \"\"  FLOW  1  1  0.2"))
    assert next(m for m in net.manholes if m.name == "J1").inflow == 0.2


def test_a_baseline_pattern_multiplies_the_baseline_only(tmp_path):
    extra = "[PATTERNS]\nHP HOURLY " + " ".join(["3"] * 24) + "\n[TIMESERIES]\nS1 0 1 2 1\n"
    net, _, _ = read_swmm_inp(_inp(tmp_path, "J1 FLOW S1 FLOW 1 1 0.1 HP", extra))
    assert net.inflows.flows(0.0)["J1"] == pytest.approx(1.0 + 0.3)


def test_start_date_and_time_set_the_calendar(tmp_path):
    extra = "[PATTERNS]\nMP MONTHLY " + " ".join(str(i) for i in range(1, 13)) + "\n"
    edits = [("START_DATE           01/01/2026", "START_DATE           03/15/2026"),
             ("START_TIME           00:00:00", "START_TIME           23:59:30")]
    net, _, _ = read_swmm_inp(_inp(tmp_path, "J1 FLOW \"\" FLOW 1 1 1 MP", extra, edits=edits))
    assert net.inflows.start == datetime.datetime(2026, 3, 15, 23, 59, 30)
    assert net.inflows.flows(0.0)["J1"] == 3.0                       # March
    assert net.inflows.flows(31.0)["J1"] == 3.0                      # 00:00:01 on the 16th
    edits[0] = ("START_DATE           01/01/2026", "START_DATE           03/31/2026")
    net, _, _ = read_swmm_inp(_inp(tmp_path, "J1 FLOW \"\" FLOW 1 1 1 MP", extra, edits=edits))
    assert net.inflows.flows(31.0)["J1"] == 4.0                      # April 1st


def test_the_start_defaults_to_swmms_when_the_options_omit_it(tmp_path):
    extra = "[TIMESERIES]\nS1 0 1 1 1\n"
    edits = [("START_DATE           01/01/2026\n", ""), ("START_TIME           00:00:00\n", "")]
    net, _, _ = read_swmm_inp(_inp(tmp_path, "J1 FLOW S1 FLOW", extra, edits=edits))
    assert net.inflows.start == datetime.datetime(2004, 1, 1)


def test_a_file_series_is_resolved_beside_the_inp_and_a_missing_one_is_refused(tmp_path):
    (tmp_path / "hyd.dat").write_text("; flows\n01/01/2026 00:10 1.0\n00:20 3.0\n")
    extra = "[TIMESERIES]\nS1 FILE hyd.dat\n"
    net, _, _ = read_swmm_inp(_inp(tmp_path, "J1 FLOW S1 FLOW 1 1 0", extra))
    assert net.inflows.flows(900.0 - SAMPLE_OFFSET)["J1"] == pytest.approx(2.0)
    (tmp_path / "hyd.dat").unlink()
    with pytest.raises(ValueError, match=r"S1.*hyd\.dat.*cannot be read"):
        read_swmm_inp(_inp(tmp_path, "J1 FLOW S1 FLOW 1 1 0", extra))
    (tmp_path / "hyd.dat").write_text("00:10 1.0 5 6\n")
    with pytest.raises(ValueError, match=r"hyd\.dat: line 1.*needs 'time value'"):
        read_swmm_inp(_inp(tmp_path, "J1 FLOW S1 FLOW 1 1 0", extra))


@pytest.mark.parametrize("extra, j1, message", [
    ("", "J1 FLOW S1 FLOW", r"time series 'S1', which has no \[TIMESERIES\] entry"),
    ("[TIMESERIES]\nS1 0 1 0 2\n", "J1 FLOW S1 FLOW", "strictly increase"),
    ("[TIMESERIES]\nS1 0 1 01/01/2026 1 2\n", "J1 FLOW S1 FLOW", "before the first date"),
    ("[TIMESERIES]\nS1 xx 1\n", "J1 FLOW S1 FLOW", "time series time must be hours"),
    ("[TIMESERIES]\nS1 0 abc\n", "J1 FLOW S1 FLOW", "time series value must be a number"),
    ("[TIMESERIES]\nS1 13/45/2026 0 1\n", "J1 FLOW S1 FLOW", "not a date"),
    ("[TIMESERIES]\nS1 0 1\nS1 FILE x.dat\n", "J1 FLOW S1 FLOW", "mixes a file"),
    ("", 'J1 FLOW "" FLOW 1 1 1 NOPAT', "pattern 'NOPAT', which has no \\[PATTERNS\\]"),
    ("", "J9 FLOW \"\" FLOW 1 1 1", "'J9', which is not a junction"),
    ("", "J1 FLOW", "needs a node, a constituent and a time series"),
    ("", "J1 FLOW \"\" FLOW 1 1 abc", "the baseline must be a number"),
])
def test_what_inflows_do_not_take_is_refused_by_name(tmp_path, extra, j1, message):
    with pytest.raises(ValueError, match=message):
        read_swmm_inp(_inp(tmp_path, j1, extra))


def test_dwf_uses_the_first_four_pattern_fields_and_one_pattern_per_kind(tmp_path):
    pats = ("[PATTERNS]\nM1 MONTHLY " + " ".join(["2"] * 12) + "\nM2 MONTHLY "
            + " ".join(["3"] * 12) + "\nD1 DAILY " + " ".join(["5"] * 7)
            + "\nH1 HOURLY " + " ".join(["7"] * 24) + "\n")
    extra = pats + "[DWF]\nJ3 FLOW 1 M1 M2 D1 H1 M1\nJ4 FLOW 1 M1 \"\" D1\n"
    net, _, _ = read_swmm_inp(_inp(tmp_path, extra=extra))
    assert net.inflows.flows(0.0)["J3"] == pytest.approx(3.0 * 5.0 * 7.0)   # M2 replaced M1
    assert net.inflows.flows(0.0)["J4"] == pytest.approx(10.0)
    # five fields: the fifth (M1 again) is not read, so the result above holds


def test_a_constant_dwf_is_a_constant_inflow_without_a_schedule(tmp_path):
    net, _, _ = read_swmm_inp(_inp(tmp_path, extra="[DWF]\nJ3 FLOW 0.07\n"))
    assert net.inflows is None
    assert next(m for m in net.manholes if m.name == "J3").inflow == 0.07


@pytest.mark.parametrize("extra, message", [
    ("[DWF]\nJ3 FLOW 1 NOPAT\n", "pattern 'NOPAT'"),
    ("[DWF]\nJ3 FLOW 1 *\n", "pattern '\\*'"),
    ("[DWF]\nJ3 BOD 1\n", "only the dry-weather FLOW"),
    ("[DWF]\nNowhere FLOW 1\n", "'Nowhere', which is not a junction"),
    ("[DWF]\nJ3 FLOW\n", "a node, FLOW and an average value"),
    ("[DWF]\nJ3 FLOW abc\n", "average flow must be a number"),
])
def test_what_dwf_does_not_take_is_refused_by_name(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        read_swmm_inp(_inp(tmp_path, extra=extra))


def test_a_dry_weather_flow_into_an_outfall_is_dropped(tmp_path):
    """Dynamic-wave routing never solves an outfall's depth from its inflow, so a
    dry-weather flow at one reaches no link and no junction's inflow."""
    net, _, _ = read_swmm_inp(_inp(tmp_path, extra="[DWF]\nOutfall FLOW 1\n"))
    base, _, _ = read_swmm_inp(_inp(tmp_path))
    assert net.manholes == base.manholes


def test_a_dry_weather_pollutant_concentration_with_a_dwf_flow_is_refused(tmp_path):
    base = "tree_kinwave_pollut.inp"
    text = (DATA / base).read_text()
    old = ("Tracer           MG/L   0.0        0.0        0.0        5.0        NO         *"
           "                0.0        0.0        0.0")
    path = tmp_path / "m.inp"
    path.write_text(text.replace(old, old[:-14] + "7.0        0.0") + "\n[DWF]\nJ3 FLOW 0.01\n")
    with pytest.raises(ValueError, match="Cdwf.*'Tracer'|'Tracer'.*Cdwf"):
        read_swmm_inp(path)


def test_pollutant_inflows_take_concen_constants_only(tmp_path):
    base = "tree_kinwave_pollut.inp"
    text = (DATA / base).read_text()
    line = next(x for x in text.splitlines() if x.startswith("J1               Tracer"))
    path = tmp_path / "m.inp"

    def read(new, extra=""):
        path.write_text(text.replace(line, new) + extra)
        return read_swmm_inp(path)

    _, loads, _ = read("J1 Tracer \"\" CONCEN 1.0 1.0 100.0")
    assert loads == {"J1": {"Tracer": pytest.approx(0.1)}}
    _, loads, _ = read("J1 Tracer \"\" CONCENTRATION 1.0 1.0 100.0")
    assert loads == {"J1": {"Tracer": pytest.approx(0.1)}}
    for new, message in [
        ("J1 Tracer \"\" MASS 1.0 1.0 100.0", "must be of type CONCEN.*mass inflows"),
        ("J1 Tracer S1 CONCEN 1.0 1.0 100.0", "pollutant 'Tracer'.*time series or a pattern"),
        ("J1 Tracer \"\" CONCEN 1.0 1.0 100.0 P1", "time series or a pattern"),
        ("J1 Tracer \"\" CONCEN 2.0 1.0 100.0", "Mfactor 2.0"),
        ("J1 Nope \"\" CONCEN 1.0 1.0 100.0", "no \\[POLLUTANTS\\] entry"),
    ]:
        with pytest.raises(ValueError, match=message):
            read(new, "\n[TIMESERIES]\nS1 0 1\n[PATTERNS]\nP1 HOURLY 1\n")


@pytest.mark.parametrize("line, message", [
    ("USE HOTSTART hot.hsf", "USE HOTSTART supplies the initial state"),
    ("USE INFLOWS in.dat", "USE INFLOWS supplies interface inflows"),
    ("USE RDII rdii.dat", "USE RDII supplies RDII flows"),
    ("use runoff r.dat", "USE RUNOFF supplies runoff flows"),
])
def test_files_that_change_the_physics_are_refused_by_name(tmp_path, line, message):
    with pytest.raises(ValueError, match=message):
        read_swmm_inp(_inp(tmp_path, extra=f"[FILES]\n{line}\n"))


def test_files_that_only_save_are_ignored(tmp_path):
    extra = "[FILES]\nSAVE HOTSTART h.hsf\nSAVE OUTFLOWS o.dat\nUSE RAINFALL r.dat\n"
    net, _, _ = read_swmm_inp(_inp(tmp_path, extra=extra))
    assert net.routing == "kinematic_wave"


def test_an_unused_time_series_is_not_read(tmp_path):
    extra = "[TIMESERIES]\nUnused FILE missing.dat\nTide 0 1 6 2\n"
    net, _, _ = read_swmm_inp(_inp(tmp_path, extra=extra))
    assert net.inflows is None


def test_bad_start_options_are_refused(tmp_path):
    for old, new, message in [
        ("START_DATE           01/01/2026", "START_DATE           soon", "START_DATE must be"),
        ("START_DATE           01/01/2026", "START_DATE           13/45/2026", "not a date"),
        ("START_TIME           00:00:00", "START_TIME           25:00:00", "START_TIME must be"),
    ]:
        with pytest.raises(ValueError, match=message):
            read_swmm_inp(_inp(tmp_path, edits=[(old, new)]))


# ------------------------------------------------------------ the consumer
def test_the_dynamic_wave_defaults_to_the_models_schedule(tmp_path):
    """Without `inflow=`, `simulate_dynamic_wave` samples the model's schedule at each step
    start; it equals passing `schedule.driver(...)` explicitly, and differs from the constant
    `drivers["inflow"]`."""
    extra = "[TIMESERIES]\nS1 0 0.0 0.1 0.04\n"
    path = _inp(tmp_path, "J1 FLOW S1 FLOW 1 1 0", extra,
                edits=[("FLOW_ROUTING         KINWAVE", "FLOW_ROUTING         DYNWAVE"),
                       ("VARIABLE_STEP        0.75", "VARIABLE_STEP        0")])
    net, _, _ = read_swmm_inp(path)
    assert net.inflows is not None
    model, state, drivers = build_model(net)
    default = simulate_dynamic_wave(model, state, drivers, steps=4, record=("water.q",))[1]
    explicit = simulate_dynamic_wave(
        model, state, drivers, steps=4, record=("water.q",),
        inflow=net.inflows.driver(list(model.net.nodes)))[1]
    assert torch.equal(default["water.q"], explicit["water.q"])
    constant = simulate_dynamic_wave(model, state, drivers, steps=4, record=("water.q",),
                                     inflow=drivers["inflow"])[1]
    assert not torch.allclose(default["water.q"], constant["water.q"])


def test_a_kinematic_wave_file_carries_the_schedule_for_the_drivers(tmp_path):
    extra = "[TIMESERIES]\nS1 0 0.0 1 0.36\n"
    net, _, _ = read_swmm_inp(_inp(tmp_path, "J1 FLOW S1 FLOW 1 1 0", extra))
    model, state, drivers = build_model(net, air=False, quality=False)
    assert model.inflow_schedule is net.inflows
    at = model.inflow_schedule.driver(list(model.net.nodes))
    assert at(1800.0)[list(model.net.nodes).index("J1")] == pytest.approx(0.18, rel=1e-5)
