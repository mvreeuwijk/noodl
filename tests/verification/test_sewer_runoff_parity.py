"""Subcatchment runoff against live SWMM 5.2.4: `[RAINGAGES]`, `[SUBCATCHMENTS]`, `[SUBAREAS]`,
`[INFILTRATION]`, `[EVAPORATION]` and `[ADJUSTMENTS]` read by `read_swmm_inp`, run by
`RunoffModel` and routed by `simulate_dynamic_wave`.

Three comparisons per fixture, each against the engine through pyswmm:

- the runoff of every subcatchment after every routing step (SWMM's ``Subcatchment.runoff``,
  the runoff at the end of its latest runoff step) against `RunoffModel`, which pins the
  runoff clock (wet, dry and gage-limited steps, the last step cut at the end of the run), the
  nonlinear reservoir and its Runge-Kutta solution, every infiltration model and the
  evaporation, to 1e-6 relative + 1e-9 m3/s;
- the lateral inflow of every node at every routing step (``Node.lateral_inflow``, the sum of
  the interpolated runoff of its subcatchments, routing.c `addWetWeatherInflows`) against
  `RunoffInflows.flows`; and
- the full dynamic-wave run, depths within 1e-7 m and flows within 1e-6 relative + 1e-9 m3/s,
  with SWMM's HEAD_TOLERANCE 1e-12 and MAX_TRIALS 500 like the other parity fixtures and
  noodl's tight iteration tolerances. noodl is given no ``inflow``: the runoff of the file is
  the model's default.

Fixtures (tests/data/sewer; `make_runoff_fixtures.py` writes all but the first):
`runoff_horton_dynwave.inp` (SI, Horton and modified-Horton MaxInfil, a subcatchment draining onto
another, a record that starts after the run does), `runoff_cn_mixed_lps_dynwave.inp` (LPS, Curve
Number with modified Horton and Horton lines, RouteTo IMPERVIOUS and PERVIOUS, co-gages, an
intensity gage, a subcatchment draining to itself, monthly evaporation, a WET_STEP that is not a
multiple of the routing step, a duration the wet step does not divide),
`runoff_greenampt_us_dynwave.inp` (CFS, Green-Ampt and modified, a cumulative gage that
restarts, DRY_ONLY evaporation, n = 0, fully pervious and impervious subcatchments),
`runoff_calendar_dynwave.inp` (a
month boundary under monthly evaporation, a soil recovery pattern, [ADJUSTMENTS] of rainfall,
conductivity and evaporation and the N-PERV, DSTORE and INFIL patterns, a gage faster than WET_STEP
which lowers it and the routing step, a record that starts before the run) and
`runoff_evap_series_dynwave.inp` (time-series evaporation that starts before the run,
MODIFIED_GREEN_AMPT as the option).
"""

import re
import shutil
from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave
from noodl.apps.sewer.inp import unit_factors
from noodl.apps.sewer.runoff import QCF, RunoffModel

pyswmm = pytest.importorskip("pyswmm")

DATA = Path(__file__).resolve().parents[1] / "data" / "sewer"
TOL_DEPTH = 1e-7
TOL_FLOW_REL, TOL_FLOW_ABS = 1e-6, 1e-9
ITERATE_TOL, ITERATE_MAX = 1e-13, 2000

HORTON = "runoff_horton_dynwave.inp"
CN = "runoff_cn_mixed_lps_dynwave.inp"
GA = "runoff_greenampt_us_dynwave.inp"
CAL = "runoff_calendar_dynwave.inp"
EVS = "runoff_evap_series_dynwave.inp"
ALL = [HORTON, CN, GA, CAL, EVS]


def _copy(tmp_path, name, edits=()):
    """The fixture `name` (and its data files) copied to `tmp_path` with `edits` applied."""
    for f in DATA.glob("runoff_*"):
        shutil.copy(f, tmp_path / f.name)
    path = tmp_path / name
    text = path.read_text()
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new)
    path.write_text(text)
    return path


def _rows(path):
    """SWMM's per-step subcatchment runoff and node lateral inflow (m3/s), link flows (file
    orientation, m3/s) and node depths (m), the connections and the share of steps not
    converging. pyswmm yields after every step but the last."""
    units = re.search(r"FLOW_UNITS\s+(\w+)", path.read_text()).group(1)
    lf, qf = unit_factors(units)
    rpt = path.with_suffix(".rpt")
    rows = []
    with pyswmm.Simulation(str(path), str(rpt), str(path.with_suffix(".out"))) as sim:
        assert str(sim.engine_version) == "5.2.4"
        conn = {link.linkid: link.connections for link in pyswmm.Links(sim)}
        subs = list(pyswmm.Subcatchments(sim))
        for _ in sim:
            nodes = list(pyswmm.Nodes(sim))
            rows.append((
                {s.subcatchmentid: s.runoff * qf for s in subs},
                {n.nodeid: n.lateral_inflow * qf for n in nodes},
                {link.linkid: link.flow * qf for link in pyswmm.Links(sim)},
                {n.nodeid: n.depth * lf for n in nodes},
            ))
    m = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", rpt.read_text())
    assert m is not None and float(m.group(1)) == 0.0, "SWMM did not converge on every step"
    return rows, conn


def _check_runoff(path, rows, expect_rain=True):
    """SWMM's subcatchment runoff and lateral inflow after every step against noodl's."""
    net, _, _ = read_swmm_inp(path)
    step = net.dynamic_wave.routing_step
    model = RunoffModel(net.hydrology)
    biggest = 0.0
    for k, (runoff, _, _, _) in enumerate(rows, start=1):
        model.advance_to(k * step * 1000.0)
        for name, ref in runoff.items():
            got = float(model.runoff[name]) * QCF
            assert abs(got - ref) <= TOL_FLOW_ABS + TOL_FLOW_REL * abs(ref), (k, name, got, ref)
            biggest = max(biggest, abs(ref))
    flows = None
    for k, (_, lateral, _, _) in enumerate(rows):
        flows = net.inflows.flows(k * step, step)
        for node, ref in lateral.items():
            got = flows.get(node, 0.0)
            assert abs(got - ref) <= TOL_FLOW_ABS + TOL_FLOW_REL * abs(ref), (k, node, got, ref)
    assert (biggest > 1e-4) == expect_rain


def _check_routed(path, rows, conn, steps):
    """noodl's first `steps` steps against SWMM's, with the runoff as the model's default
    inflow."""
    net, _, _ = read_swmm_inp(path)
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    assert model.inflow_schedule is net.inflows
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=steps,
                                    record=("water.phi", "water.q"))
    hyd = model.dynamic_wave
    nodes = list(hyd.junction_names) + list(hyd.outfall_names)
    idx = torch.cat([model.junction_idx, hyd.outfall_idx])
    invert = hyd.node_invert[idx]
    names = list(model.net.nodes)
    ends = {name: (names[int(u)], names[int(v)])
            for name, u, v in zip(model.link_names, hyd.edge_u, hyd.edge_v, strict=True)}
    sign = {name: (1.0 if tuple(conn[name]) == pair else -1.0) for name, pair in ends.items()}
    worst_depth, worst_flow, flowing = 0.0, -TOL_FLOW_ABS, 0.0
    for k, (_, _, flows, depths) in enumerate(rows[:steps]):
        y = hist["water.phi"][k, idx] - invert
        for j, n in enumerate(nodes):
            worst_depth = max(worst_depth, abs(y[j].item() - depths[n]))
        for i, name in enumerate(model.link_names):
            ref = sign[name] * flows[name]
            flowing = max(flowing, abs(ref))
            miss = abs(hist["water.q"][k, i].item() - ref) - (TOL_FLOW_REL * abs(ref)
                                                             + TOL_FLOW_ABS)
            worst_flow = max(worst_flow, miss)
    assert flowing > 0.001, "the runoff should load the pipes"
    assert worst_depth <= TOL_DEPTH, worst_depth
    assert worst_flow <= 0.0, worst_flow


@pytest.mark.parametrize("name", ALL)
def test_runoff_and_lateral_inflows_match_swmm_at_every_step(tmp_path, name):
    """Each fixture's subcatchment runoff and node inflow, step by step."""
    rows, _ = _rows(_copy(tmp_path, name))
    _check_runoff(tmp_path / name, rows)


def test_routed_state_matches_swmm_over_the_first_four_minutes(tmp_path):
    """The Horton fixture routed by noodl with its runoff as the default inflow: depths and
    flows against SWMM over its first 24 steps (the rain has been falling for two minutes)."""
    path = _copy(tmp_path, HORTON)
    rows, conn = _rows(path)
    _check_routed(path, rows, conn, 24)


@pytest.mark.slow
@pytest.mark.parametrize("name", ALL)
def test_routed_state_matches_swmm_over_the_whole_window(tmp_path, name):
    """Each fixture over its whole window."""
    path = _copy(tmp_path, name)
    rows, conn = _rows(path)
    _check_routed(path, rows, conn, len(rows))


# ---- variants of the fixtures, each a feature SWMM must agree on -------------------------------
#: a snow pack, a temperature series that keeps it rain (10 C), and the series
SNOW = """[SNOWPACKS]
Snow1 PLOWABLE 0.01 0.01 0.0 0.1 0.0 0.0 0.0
Snow1 IMPERVIOUS 0.01 0.01 0.0 0.1 0.0 0.0 0.0
Snow1 PERVIOUS 0.01 0.01 0.0 0.1 0.0 0.0 0.0
Snow1 REMOVAL 1.0 0 0 0 0 0

[TEMPERATURE]
TIMESERIES T1

[TIMESERIES]
T1 01/01/2026 00:00 10.0
T1 01/02/2026 00:00 10.0

"""
#: Green-Ampt under a conductivity adjustment (K scales, the upper zone with its square root)
#: and a soil recovery pattern
MONTHLY_GA = """[PATTERNS]
Rec MONTHLY 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4

[ADJUSTMENTS]
CONDUCTIVITY 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6

"""
#: the N-PERV, DSTORE and INFIL lines moved above [SUBCATCHMENTS]: SWMM resets them there
MOVE_PATTERNS = [
    ("""N-PERV           S1       Nv
DSTORE           S1       Dv
INFIL            S2       Iv
""", ""),
    ("[RAINGAGES]", """[ADJUSTMENTS]
N-PERV           S1       Nv
DSTORE           S1       Dv
INFIL            S2       Iv

[RAINGAGES]""")]
VARIANTS = {
    # SWMM's runoff step is DRY_STEP while nothing happens, cut at the next rain date
    "dry_step_20_s": (HORTON, [("DRY_STEP             00:05:00",
                                "DRY_STEP             00:00:20")], True),
    "dry_step_an_hour": (HORTON, [("DRY_STEP             00:05:00",
                                   "DRY_STEP             01:00:00")], True),
    "rainfall_ignored": (HORTON, [("SKIP_STEADY_STATE    NO", """SKIP_STEADY_STATE    NO
IGNORE_RAINFALL      YES""")], False),
    # S1 100 % impervious with no depression storage at all, S3 fully pervious
    "all_impervious_no_storage": (HORTON, [("2.5      60 ", "2.5      100"),
                                           ("1.5        4.0        25 ",
                                            "0.0        0.0        100")], True),
    "all_pervious": (HORTON, [("0.8      90 ", "0.8      0  ")], True),
    "snow_pack_ignored": (HORTON, [("200      1.5      0\n", "200      1.5      0     Snow1\n"),
                                   ("SKIP_STEADY_STATE    NO", """SKIP_STEADY_STATE    NO
IGNORE_SNOWMELT      YES"""),
                                   ("[JUNCTIONS]", SNOW + "[JUNCTIONS]")], True),
    "unused_snow_packs": (HORTON, [("[JUNCTIONS]", SNOW + "[JUNCTIONS]")], True),
    "adjustment_patterns_before_the_subcatchments_do_nothing": (CAL, MOVE_PATTERNS, True),
    "no_recovery_pattern": (CAL, [("RECOVERY         Rec\n", "")], True),
    "no_evaporation_in_the_second_month": (CAL, [("6.0 12.0 5 ", "6.0 0.0 5 ")], True),
    "constant_evaporation": (CN, [("MONTHLY          2.0 2.0 8.0",
                                   "CONSTANT         12.0 ;")], True),
    "dry_only_monthly": (CN, [("DRY_ONLY         NO", "DRY_ONLY         YES")], True),
    "green_ampt_without_a_per_line_model": (GA, [("0.40  MODIFIED_GREEN_AMPT", "0.40")], True),
    "green_ampt_conductivity_and_recovery": (GA, [
        ("DRY_ONLY         YES", "RECOVERY         Rec\nDRY_ONLY         YES"),
        ("[TIMESERIES]", MONTHLY_GA + "[TIMESERIES]")], True),
    "interval_in_decimal_hours": (CN, [("INTENSITY 0:05 ", "INTENSITY 0.0833333 ")], True),
    "wet_step_7_s": (CN, [("WET_STEP             00:00:45", "WET_STEP             00:00:07")],
                     True),
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variants_match_swmm(tmp_path, variant):
    """Runoff and node inflow at every step of an edited fixture."""
    name, edits, rain = VARIANTS[variant]
    path = _copy(tmp_path, name, edits)
    rows, _ = _rows(path)
    _check_runoff(path, rows, expect_rain=rain)


def test_the_patterns_of_the_calendar_fixture_change_the_runoff(tmp_path):
    """The N-PERV, DSTORE and INFIL patterns are in effect in the fixture (the variant with them
    before the subcatchments is SWMM's own, in which they are not)."""
    base = read_swmm_inp(_copy(tmp_path, CAL))[0].hydrology
    moved = read_swmm_inp(_copy(
        tmp_path, CAL, MOVE_PATTERNS))[0].hydrology
    assert base.subcatchments[0].n_perv_pattern is not None
    assert moved.subcatchments[0].n_perv_pattern is None
    a, b = RunoffModel(base), RunoffModel(moved)
    a.advance_to(1.6e6)
    b.advance_to(1.6e6)
    assert a.runoff["S1"] != b.runoff["S1"] and a.runoff["S2"] != b.runoff["S2"]


def test_the_cut_last_runoff_step_is_swmms(tmp_path):
    """12:20:00 after 12:00:00 is 1199 s to SWMM's floating point; the 1199-second run's last
    runoff step (29 s) is what SWMM's reports show at the end of the CN fixture."""
    path = _copy(tmp_path, CN)
    rows, _ = _rows(path)
    net, _, _ = read_swmm_inp(path)
    assert net.hydrology.total_duration == 1199.0
    model = RunoffModel(net.hydrology)
    model.advance_to(1.19e6)
    assert model.steps[-1] == (1170.0, 29.0)
    ref = rows[-1][0]["S1"]
    assert float(model.runoff["S1"]) * QCF == pytest.approx(ref, rel=1e-6)
