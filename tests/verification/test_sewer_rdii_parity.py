# ruff: noqa: E501
"""RDII (`[HYDROGRAPHS]`, `[RDII]`) against live SWMM 5.2.4, read by `read_swmm_inp`, computed by
`RdiiModel` and routed by `simulate_dynamic_wave`.

Each fixture is written here (a three-junction tree, J1 and J2 -> J3 -> Outfall, dynamic wave,
SWMM's HEAD_TOLERANCE 1e-12 and MAX_TRIALS 500 like the other parity fixtures) and run through
pyswmm. Two comparisons per fixture:

- the lateral inflow of every node at every routing step (``Node.lateral_inflow``: the RDII
  record in force, plus runoff when the fixture has subcatchments) against
  `RunoffInflows.flows`, to 1e-6 relative + 1e-9 m3/s; and
- the dynamic-wave run with the file's inflow as the model's default, depths within 1e-7 m
  and flows within 1e-6 relative + 1e-9 m3/s, with noodl's tight iteration tolerances.

The fixtures: all three responses of one set; a month boundary under monthly parameters (the
rain's month is that of the record, the abstraction's that of the interval); an initial
abstraction with recovery; US units (acres, inches, intensity gage); RDII with subcatchment
runoff on the same nodes; a response whose limb is shorter than the WET_STEP (several rain
intervals per record, the intervals not dividing the step); two sets with different intervals on
one gage and on co-gages; the older one-line format; a gage faster than WET_STEP; the rainfall
adjustment SWMM leaves behind for the runoff; and IGNORE_RDII and IGNORE_RAINFALL.
"""

import re

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave
from noodl.apps.sewer.inp import unit_factors

pyswmm = pytest.importorskip("pyswmm")

TOL_DEPTH = 1e-7
TOL_FLOW_REL, TOL_FLOW_ABS = 1e-6, 1e-9
ITERATE_TOL, ITERATE_MAX = 1e-13, 2000

HEAD = """[TITLE]
RDII test fixture

[OPTIONS]
FLOW_UNITS           {units}
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
LINK_OFFSETS         DEPTH
MIN_SLOPE            0
ALLOW_PONDING        NO
SKIP_STEADY_STATE    NO
START_DATE           {start_date}
START_TIME           {start_time}
REPORT_START_DATE    {start_date}
REPORT_START_TIME    {start_time}
END_DATE             {end_date}
END_TIME             {end_time}
REPORT_STEP          00:05:00
WET_STEP             {wet}
DRY_STEP             00:30:00
ROUTING_STEP         {routing}
INERTIAL_DAMPING     PARTIAL
NORMAL_FLOW_LIMITED  BOTH
FORCE_MAIN_EQUATION  H-W
VARIABLE_STEP        0
LENGTHENING_STEP     0
MIN_SURFAREA         0
MAX_TRIALS           500
HEAD_TOLERANCE       1e-12
SYS_FLOW_TOL         5
LAT_FLOW_TOL         5
MINIMUM_STEP         0.5
THREADS              1
{options}
[EVAPORATION]
CONSTANT         0.0
DRY_ONLY         NO

"""
NETWORK_SI = """[JUNCTIONS]
J1               12.0       5          0          0          0
J2               11.0       5          0          0          0
J3               10.0       5          0          0          0

[OUTFALLS]
Outfall          9.0        FIXED      9.6              NO

[CONDUITS]
C1               J1               J3               200        0.013      0          0          0          0
C2               J2               J3               200        0.013      0          0          0          0
C3               J3               Outfall          200        0.013      0          0          0          0

[XSECTIONS]
C1               CIRCULAR     1.20             0          0          0          1
C2               CIRCULAR     1.00             0          0          0          1
C3               CIRCULAR     1.60             0          0          0          1

"""
NETWORK_US = """[JUNCTIONS]
J1               40.0       16         0          0          0
J2               36.0       16         0          0          0
J3               33.0       16         0          0          0

[OUTFALLS]
Outfall          30.0       FIXED      31.6             NO

[CONDUITS]
C1               J1               J3               660        0.013      0          0          0          0
C2               J2               J3               660        0.013      0          0          0          0
C3               J3               Outfall          660        0.013      0          0          0          0

[XSECTIONS]
C1               CIRCULAR     4.0              0          0          0          1
C2               CIRCULAR     3.3              0          0          0          1
C3               CIRCULAR     5.3              0          0          0          1

"""
RAIN = """R1 01/01/2026 00:05 3.0
R1 00:10 5.0
R1 00:15 8.0
R1 00:20 6.0
R1 00:25 0.0
R1 00:30 0.0
R1 00:35 4.0
R1 00:40 2.0
R1 00:45 0.0
"""
SUBS_SI = """[SUBCATCHMENTS]
S1               RG1              J1               2.5      60       200      1.5      0
S2               RG1              J3               1.2      30       120      2.5      0

[SUBAREAS]
S1               0.013      0.15       1.5        4.0        25         OUTLET
S2               0.012      0.20       1.2        5.0        30         OUTLET

[INFILTRATION]
S1               80         10         4          7          0
S2               60         8          3          5          40

"""
THREE = """UH1 RG1
UH1 ALL SHORT 0.05 0.5 1.0
UH1 ALL MEDIUM 0.10 2.0 2.0
UH1 ALL LONG 0.15 3.0 3.0
"""


def fixture(*, units="CMS", hydrographs=THREE, rdii="J1 UH1 5\nJ2 UH1 2", gage="VOLUME 0:05",
            series=None, start=("01/01/2026", "00:00:00"), end=("01/01/2026", "06:00:00"),
            wet="00:05:00", routing=30, options="", subs="", extra="", gages=None):
    """The text of a fixture: the tree under `hydrographs` and `rdii` with rain `series` read by
    the gage `gage` (or by `gages`, the full [RAINGAGES] text)."""
    text = HEAD.format(
        units=units, start_date=start[0], start_time=start[1], end_date=end[0],
        end_time=end[1], wet=wet, routing=routing, options=options)
    text += "[RAINGAGES]\n" + (gages or f"RG1              {gage}     1.0      TIMESERIES R1\n") + "\n"
    text += subs + (NETWORK_US if units == "CFS" else NETWORK_SI)
    text += "[HYDROGRAPHS]\n" + hydrographs + "\n"
    if rdii:
        text += "[RDII]\n" + rdii + "\n"
    series = RAIN if series is None else series
    text += "[TIMESERIES]\n" + series + "\n" + extra
    return text


def _rows(path):
    """SWMM's per-step node lateral inflow (m3/s), link flows (file orientation, m3/s) and node
    depths (m), and the connections; pyswmm yields after every step but the last."""
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


def _check_inflows(path, rows, expect_flow=True, floor=1e-4):
    """SWMM's node lateral inflow after every step against noodl's; the largest inflow."""
    net, _, _ = read_swmm_inp(path)
    step = net.dynamic_wave.routing_step
    biggest = 0.0
    for k, (lateral, _, _) in enumerate(rows):
        flows = net.inflows.flows(k * step, step)
        for node, ref in lateral.items():
            if node == "Outfall":           # SWMM reports an outfall's inflow; routing ignores it
                continue
            got = flows.get(node, 0.0)
            assert abs(got - ref) <= TOL_FLOW_ABS + TOL_FLOW_REL * abs(ref), (k, node, got, ref)
            biggest = max(biggest, abs(ref))
    assert (biggest > floor) == expect_flow, biggest
    return biggest


def _check_routed(path, rows, conn, steps):
    """noodl's first `steps` steps against SWMM's, with the file's inflow as the default."""
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
    for k, (_, flows, depths) in enumerate(rows[:steps]):
        y = hist["water.phi"][k, idx] - invert
        for j, n in enumerate(nodes):
            worst_depth = max(worst_depth, abs(y[j].item() - depths[n]))
        for i, name in enumerate(model.link_names):
            ref = sign[name] * flows[name]
            flowing = max(flowing, abs(ref))
            miss = abs(hist["water.q"][k, i].item() - ref) - (TOL_FLOW_REL * abs(ref)
                                                             + TOL_FLOW_ABS)
            worst_flow = max(worst_flow, miss)
    assert flowing > 0.001, "the RDII should load the pipes"
    assert worst_depth <= TOL_DEPTH, worst_depth
    assert worst_flow <= 0.0, worst_flow


def _write(tmp_path, text, name="rdii.inp"):
    path = tmp_path / name
    path.write_text(text)
    return path


# ---- the fixtures ---------------------------------------------------------------------------
#: JAN and FEB differ in the medium and long responses; the rain falls on both sides of midnight
MONTHLY = """UH1 RG1
UH1 ALL SHORT 0.05 0.5 1.0
UH1 ALL MEDIUM 0.10 1.0 2.0
UH1 ALL LONG 0.15 1.5 3.0
UH1 JAN MEDIUM 0.20 0.75 1.5 1.0 0 0
UH1 FEB LONG 0.05 2.5 2.0
UH1 FEB SHORT 0.20 0.25 4.0 0.5 2.0 0.1
"""
MONTHLY_RAIN = """R1 01/31/2026 23:20 3.0
R1 23:25 5.0
R1 23:30 8.0
R1 23:35 6.0
R1 23:40 0.0
R1 23:45 0.0
R1 23:50 4.0
R1 23:55 2.0
R1 02/01/2026 00:00 4.0
R1 00:05 7.0
R1 00:10 3.0
R1 00:15 0.0
"""
#: Dmax 2 mm, recovery 6 mm/day, 0.5 mm used at the start; three bursts an hour and a half apart
ABSTRACTION = """UH1 RG1
UH1 ALL SHORT 0.10 0.5 1.0 2.0 6.0 0.5
UH1 ALL MEDIUM 0.20 1.0 2.0 4.0 12.0 1.0
UH1 ALL LONG 0.10 2.0 2.0 8.0 24.0 0.0
"""
ABSTRACTION_RAIN = """R1 01/01/2026 00:05 1.0
R1 00:10 2.0
R1 00:15 0.5
R1 00:20 0.0
R1 01:50 1.5
R1 01:55 2.5
R1 02:00 0.0
R1 03:30 3.0
R1 03:35 4.0
R1 03:40 0.0
"""
US_HYD = """UH1 RG1
UH1 ALL SHORT 0.05 0.5 1.0 0.1 0.3 0.0
UH1 ALL MEDIUM 0.10 2.0 2.0
UH1 ALL LONG 0.15 3.0 3.0
"""
US_RAIN = """R1 01/01/2026 00:05 0.30
R1 00:10 0.50
R1 00:15 0.80
R1 00:20 0.60
R1 00:25 0.0
R1 00:30 0.0
R1 00:35 0.40
R1 00:40 0.20
R1 00:45 0.0
"""
#: SHORT: both limbs 180 s, shorter than the 300 s WET_STEP and not dividing it
SHORT_LIMB = """UH1 RG1
UH1 ALL SHORT 0.10 0.05 1.0
UH1 ALL MEDIUM 0.10 0.5 1.0
UH1 ALL LONG 0.10 1.5 2.0
"""
TWO_SETS = """UH1 RG1
UH1 ALL SHORT 0.05 0.5 1.0
UH1 ALL MEDIUM 0.10 2.0 2.0
UH2 RG1
UH2 ALL SHORT 0.10 0.05 1.0
UH2 ALL LONG 0.10 1.5 2.0
"""
OLD_FORMAT = """UH1 RG1
UH1 ALL 0.05 0.5 1.0 0.10 2.0 2.0 0.15 3.0 3.0 1.0 4.0 0.0
"""
RAIN_ADJUSTMENT = """[ADJUSTMENTS]
RAINFALL 1.0 2.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0

"""
#: entries three minutes off the five-minute grid of the records, so that the rain intervals one
#: record takes (a response with a short limb) straddle a change of rainfall
OFF_GRID = """R1 01/01/2026 00:03 3.0
R1 00:08 6.0
R1 00:13 9.0
R1 00:18 2.0
R1 00:23 0.0
R1 00:28 0.0
R1 00:33 5.0
R1 00:38 7.0
R1 00:43 1.0
R1 00:48 0.0
"""
FEB = ("01/31/2026", "23:00:00"), ("02/01/2026", "03:00:00")

CASES = {
    "three_responses": dict(),
    "month_boundary": dict(hydrographs=MONTHLY, series=MONTHLY_RAIN, start=FEB[0], end=FEB[1]),
    "initial_abstraction": dict(hydrographs=ABSTRACTION, series=ABSTRACTION_RAIN,
                                end=("01/01/2026", "05:00:00")),
    "us_units": dict(units="CFS", routing=15, hydrographs=US_HYD, gage="INTENSITY 0:05", series=US_RAIN,
                     rdii="J1 UH1 12\nJ2 UH1 5"),
    "short_limb": dict(hydrographs=SHORT_LIMB, end=("01/01/2026", "03:00:00")),
    "short_limb_off_grid": dict(hydrographs=SHORT_LIMB, series=OFF_GRID,
                                end=("01/01/2026", "03:00:00")),
    "two_sets_off_grid": dict(hydrographs=TWO_SETS, series=OFF_GRID, rdii="J1 UH1 5\nJ2 UH2 4",
                              end=("01/01/2026", "03:00:00")),
    "tails_below_the_zero_rdii_flow": dict(units="CFS", routing=15, hydrographs=US_HYD,
                                           gage="INTENSITY 0:05", series=US_RAIN,
                                           rdii="J1 UH1 0.5\nJ2 UH1 0.2"),
    "two_sets_one_gage": dict(hydrographs=TWO_SETS, rdii="J1 UH1 5\nJ2 UH2 4",
                              end=("01/01/2026", "03:00:00")),
    "co_gages": dict(hydrographs=TWO_SETS.replace("UH2 RG1", "UH2 RG2"),
                     rdii="J1 UH1 5\nJ2 UH2 4", end=("01/01/2026", "03:00:00"),
                     gages=("RG1              VOLUME 0:05 1.0 TIMESERIES R1\n"
                            "RG2              VOLUME 0:05 1.0 TIMESERIES R1\n")),
    "old_format": dict(hydrographs=OLD_FORMAT),
    "fast_gage": dict(gage="VOLUME 0:02", series="R1 01/01/2026 00:02 1.0\nR1 00:04 3.0\n"
                      "R1 00:06 2.0\nR1 00:08 0.0\nR1 00:12 5.0\nR1 00:14 0.0\n",
                      end=("01/01/2026", "03:00:00")),
    "wet_step_longer_than_the_gage_interval": dict(wet="00:10:00", end=("01/01/2026", "03:00:00")),
    "runoff_and_rdii": dict(subs=SUBS_SI),
    "rain_adjustment_left_for_the_runoff": dict(
        hydrographs=MONTHLY, series=MONTHLY_RAIN, start=FEB[0], end=FEB[1],
        subs=SUBS_SI, extra=RAIN_ADJUSTMENT),
    "rain_adjustment_rdii_only": dict(
        hydrographs=MONTHLY, series=MONTHLY_RAIN, start=FEB[0], end=FEB[1],
        extra=RAIN_ADJUSTMENT),
    "a_set_without_nodes_still_advances_the_gage": dict(
        hydrographs=TWO_SETS, rdii="J1 UH1 5", end=("01/01/2026", "03:00:00")),
    "later_line_replaces_the_node": dict(rdii="J1 UH1 50\nJ2 UH1 2\nJ1 UH1 5"),
    "rdii_at_the_outfall_is_dropped": dict(rdii="J1 UH1 5\nOutfall UH1 9"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_lateral_inflows_match_swmm_at_every_step(tmp_path, case):
    """Each fixture's node inflow, step by step."""
    path = _write(tmp_path, fixture(**CASES[case]))
    rows, _ = _rows(path)
    _check_inflows(path, rows, floor=1e-6 if case.startswith("tails") else 1e-4)


@pytest.mark.parametrize("option", ["IGNORE_RDII YES", "IGNORE_RAINFALL YES"])
def test_ignored_rdii_is_zero_in_swmm_and_here(tmp_path, option):
    """IGNORE_RDII and IGNORE_RAINFALL give no RDII at all."""
    path = _write(tmp_path, fixture(options=option + "\n"))
    rows, _ = _rows(path)
    _check_inflows(path, rows, expect_flow=False)


def test_the_inflow_is_the_record_in_force_not_an_interpolation(tmp_path):
    """RDII is constant over the RDII step (the WET_STEP): the same flow at every routing step of
    a record, a jump at its end."""
    path = _write(tmp_path, fixture())
    rows, _ = _rows(path)
    _check_inflows(path, rows)
    per_record = 10                                                 # 300 s / 30 s
    flows = [row[0]["J1"] for row in rows]
    assert flows[3 * per_record:4 * per_record] == [flows[3 * per_record]] * per_record
    assert flows[4 * per_record] != flows[3 * per_record]


def test_routed_state_matches_swmm_over_the_first_ten_minutes(tmp_path):
    """The three-response fixture routed by noodl with its RDII as the default inflow."""
    path = _write(tmp_path, fixture())
    rows, conn = _rows(path)
    _check_routed(path, rows, conn, 60)


def test_rdii_at_an_outfall_does_not_reach_the_routing(tmp_path):
    """SWMM reports the RDII of an outfall but routes none of it."""
    path = _write(tmp_path, fixture(**CASES["rdii_at_the_outfall_is_dropped"]))
    rows, conn = _rows(path)
    assert rows[40][0]["Outfall"] > 0.0
    _check_routed(path, rows, conn, 40)


def test_routed_state_with_runoff_and_rdii_matches_swmm(tmp_path):
    """RDII and runoff on the same nodes, routed."""
    path = _write(tmp_path, fixture(subs=SUBS_SI))
    rows, conn = _rows(path)
    _check_routed(path, rows, conn, 30)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["three_responses", "month_boundary", "us_units",
                                  "short_limb", "runoff_and_rdii"])
def test_routed_state_matches_swmm_over_the_whole_window(tmp_path, case):
    """Selected fixtures over their whole window."""
    path = _write(tmp_path, fixture(**CASES[case]))
    rows, conn = _rows(path)
    _check_routed(path, rows, conn, len(rows))
