"""Time-varying lateral inflows against live SWMM 5.2.4: `[TIMESERIES]`, `[PATTERNS]`,
`[INFLOWS]` and `[DWF]` read by `read_swmm_inp` and consumed by `simulate_dynamic_wave`.

Two comparisons per fixture:

- the lateral inflow of every node at every routing step, `InflowSchedule.flows(t)` against
  SWMM's own ``Node.lateral_inflow`` (the quantity routing.c adds), which pins the sampling
  time (start of step + 1 ms), the interpolation, the pattern calendar and the flow tolerance
  independently of the hydraulics; and
- the full dynamic-wave run, depths within 1e-7 m and flows within 1e-6 relative + 1e-9 m3/s
  of SWMM, run with SWMM's HEAD_TOLERANCE 1e-12 and MAX_TRIALS 500 like the other parity
  fixtures (`test_sewer_dynwave_parity.py`) and noodl's tight iteration tolerances. noodl is
  given no ``inflow``: the schedule on the model is the default.

Fixtures (tests/data/sewer): `inflow_timeseries_dynwave.inp` (an inline dated series, an inline
series in hours with Sfactor and baseline, a FILE series), `inflow_dwf_dynwave.inp` (dry-weather
flow with monthly, daily, hourly and weekend patterns across midnight, a baseline pattern, a
repeated pattern kind and a skipped field) and `inflow_us_units_dynwave.inp` (MGD, with
Mfactor, Sfactor and a baseline pattern across an hour boundary).
"""

import re
import shutil
from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave
from noodl.apps.sewer.inp import unit_factors

pyswmm = pytest.importorskip("pyswmm")

DATA = Path(__file__).resolve().parents[1] / "data" / "sewer"
TOL_DEPTH = 1e-7
TOL_FLOW_REL, TOL_FLOW_ABS = 1e-6, 1e-9
ITERATE_TOL, ITERATE_MAX = 1e-13, 2000
#: The lateral inflow is compared to this absolute (m3/s) and relative tolerance. SWMM keeps
#: it in cfs internally and the toolkit returns it as a single-precision-sized float.
TOL_LATERAL_REL, TOL_LATERAL_ABS = 1e-7, 1e-12

TS = "inflow_timeseries_dynwave.inp"
DWF = "inflow_dwf_dynwave.inp"
US = "inflow_us_units_dynwave.inp"

#: Calendar variants of the dry-weather fixture: (edits to the file, what it crosses).
DWF_VARIANTS = {
    "friday_to_saturday": ([], "23:55 Friday (weekday hourly) to 00:05 Saturday (weekend)"),
    "month_and_weekend": (
        [("01/30/2026", "01/31/2026"), ("END_DATE             01/31/2026",
                                       "END_DATE             02/01/2026")],
        "January to February and Saturday to Sunday"),
    "sunday_to_monday": (
        [("START_DATE           01/30/2026", "START_DATE           02/01/2026"),
         ("REPORT_START_DATE    01/30/2026", "REPORT_START_DATE    02/01/2026"),
         ("END_DATE             01/31/2026", "END_DATE             02/02/2026")],
        "Sunday to Monday (the weekend hourly pattern stops applying)"),
    "hour_boundary": (
        [("23:55:00", "12:55:00"), ("00:05:00", "13:05:00"),
         ("END_DATE             01/31/2026", "END_DATE             01/30/2026")],
        "12:55 to 13:05 on a Friday (hourly pattern steps)"),
}


def _copy(tmp_path, name, edits=()):
    """The fixture `name` (and its data files) copied to `tmp_path` with `edits` applied."""
    for f in DATA.glob("inflow_*"):
        shutil.copy(f, tmp_path / f.name)
    path = tmp_path / name
    text = path.read_text()
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new)
    path.write_text(text)
    return path


def _rows(path):
    """SWMM's per-step lateral inflows (m3/s), link flows (file orientation, m3/s) and node
    depths (m), the connections and the share of steps not converging. pyswmm yields after every
    step but the last."""
    units = re.search(r"FLOW_UNITS\s+(\w+)", path.read_text()).group(1)
    lf, qf = unit_factors(units)
    rpt = path.with_suffix(".rpt")
    rows = []
    with pyswmm.Simulation(str(path), str(rpt), str(path.with_suffix(".out"))) as sim:
        assert str(sim.engine_version) == "5.2.4"
        conn = {link.linkid: link.connections for link in pyswmm.Links(sim)}
        for _ in sim:
            nodes = list(pyswmm.Nodes(sim))
            rows.append((
                {n.nodeid: n.lateral_inflow * qf for n in nodes},
                {link.linkid: link.flow * qf for link in pyswmm.Links(sim)},
                {n.nodeid: n.depth * lf for n in nodes},
            ))
    m = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", rpt.read_text())
    assert m is not None and float(m.group(1)) == 0.0, "SWMM did not converge on every step"
    return rows, conn



def _check_lateral(path, rows):
    """SWMM's lateral inflow at every step against `InflowSchedule.flows`."""
    net, _, _ = read_swmm_inp(path)
    assert net.inflows is not None
    step = net.dynamic_wave.routing_step
    biggest = 0.0
    for k, (lateral, _, _) in enumerate(rows):
        flows = net.inflows.flows(k * step)
        for node, ref in lateral.items():
            got = flows.get(node, 0.0)
            assert abs(got - ref) <= TOL_LATERAL_ABS + TOL_LATERAL_REL * abs(ref), (
                k, node, got, ref)
            biggest = max(biggest, abs(ref))
    assert biggest > 0.01


def _check_routed(path, rows, conn, steps):
    """noodl's first `steps` steps against SWMM's, with the schedule as the model's default
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
    worst_depth, worst_flow, wet = 0.0, -TOL_FLOW_ABS, 0.0
    for k, (_, flows, depths) in enumerate(rows[:steps]):
        y = hist["water.phi"][k, idx] - invert
        for j, n in enumerate(nodes):
            worst_depth = max(worst_depth, abs(y[j].item() - depths[n]))
            wet = max(wet, depths[n])
        for i, name in enumerate(model.link_names):
            ref = sign[name] * flows[name]
            miss = abs(hist["water.q"][k, i].item() - ref) - (TOL_FLOW_REL * abs(ref)
                                                             + TOL_FLOW_ABS)
            worst_flow = max(worst_flow, miss)
    assert wet > 0.02, "the fixture should wet the pipes"
    assert worst_depth <= TOL_DEPTH, worst_depth
    assert worst_flow <= 0.0, worst_flow


@pytest.mark.parametrize("name", [TS, DWF, US])
def test_lateral_inflows_match_swmm_at_every_step(tmp_path, name):
    """The fixtures' lateral inflows, per node and step: an inline dated series (entries over
    two lines, the date carried), an inline series in hours since the start (START_TIME is
    not midnight) with Sfactor and a baseline, a FILE series beside the `.inp`; dry-weather
    flow with monthly, daily, hourly and weekend patterns (a repeated hourly pattern, the last
    of a kind wins; a skipped field) and a baseline with a quoted pattern name across the
    midnight at which Friday's hourly pattern gives way to Saturday's weekend one; and an
    MGD file where Mfactor has no effect on a flow, Sfactor scales the series only and the
    baseline pattern the baseline only across the hour at which it changes."""
    rows, _ = _rows(_copy(tmp_path, name))
    _check_lateral(tmp_path / name, rows)


@pytest.mark.parametrize("variant", sorted(DWF_VARIANTS))
def test_dry_weather_calendar_matches_swmm_across(tmp_path, variant):
    """The dry-weather fixture's start moved so the window crosses `DWF_VARIANTS[variant]`."""
    edits, _ = DWF_VARIANTS[variant]
    path = _copy(tmp_path, DWF, edits)
    rows, _ = _rows(path)
    _check_lateral(path, rows)


def test_routed_state_matches_swmm_over_the_first_two_minutes(tmp_path):
    """The time-series fixture routed by noodl with the schedule as its default inflow, depths
    and flows against SWMM over its first twelve steps (the inflows have started rising)."""
    path = _copy(tmp_path, TS)
    rows, conn = _rows(path)
    _check_routed(path, rows, conn, 12)


@pytest.mark.slow
@pytest.mark.parametrize("name", [TS, DWF, US])
def test_routed_state_matches_swmm_over_the_whole_window(tmp_path, name):
    """Each fixture over its ten minutes (59 comparable steps)."""
    path = _copy(tmp_path, name)
    rows, conn = _rows(path)
    _check_routed(path, rows, conn, len(rows))
