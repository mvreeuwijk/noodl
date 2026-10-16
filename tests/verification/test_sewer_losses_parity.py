"""Evaporation and seepage losses of the dynamic wave against live SWMM 5.2.4 (pyswmm).

Each fixture is written here as text (never copied from a model collection) and run in SWMM
with MAX_TRIALS 2000 and HEAD_TOLERANCE 1e-14 m, so every routing step is converged to the fixed
point of SWMM's own trial equations; the test asserts that SWMM converged on every step. The
parity suite's 500 and 1e-12 are not tight enough here: once a conduit loses water the loss
feedback (dq6) slows the contraction of SWMM's relaxed trials, and the same fixture run at 1e-12
and at 1e-16 differs by 3e-10 m in depth and 1.6e-9 m3/s in flow, SWMM against itself, which is
more than the flow target's floor. noodl is built with ``iterate_tol=1e-13`` and
``iterate_max=2000``. The targets are the parity suite's: depth within 1e-7 m, flows within 1e-6
relative plus 1e-9 m3/s, per step and per node or link, over every junction, outfall and link.

What the fixtures exercise (SWMM 5.2.4 ``src/solver`` references; docs/applications/sewer.md,
"Evaporation and seepage losses"):

- an open conduit evaporates ``topWidth * length * Evap.rate`` from the mean of the step-start
  and current link depth (link.c:1334-1399), taken half from each end node and as the momentum
  term dq6 (dwflow.c:236, dynwave.c:555-572); a closed conduit does not;
- the monthly rate and the ``[ADJUSTMENTS] EVAPORATION`` change at a month boundary inside the
  run, and ``DRY_ONLY YES`` does not stop a conduit or a storage unit evaporating while it rains
  (it is the subcatchments' alone, subcatch.c:685);
- conduit seepage ``seepRate * width * length * hydconFactor`` at the width of the section
  capped at its widest point, in a CIRCULAR, a TRAPEZOIDAL and a RECT_CLOSED section, with the
  monthly conductivity adjustment;
- a storage unit's evaporation ``Fevap * Evap.rate * area`` and its exfiltration (exfil.c): Ksat
  alone, and Green-Ampt with a suction head and IMDmax through the bottom and the banks of a
  TABULAR and a CYLINDRICAL unit.
"""

import re
from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave

pyswmm = pytest.importorskip("pyswmm")

TOL_DEPTH = 1e-7
TOL_FLOW_REL, TOL_FLOW_ABS = 1e-6, 1e-9
ITERATE_TOL, ITERATE_MAX = 1e-13, 2000

HEADER = """[TITLE]
Evaporation and seepage losses - noodl parity fixture

[OPTIONS]
FLOW_UNITS           CMS
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
SWEEP_START          01/01
SWEEP_END            12/31
DRY_DAYS             0
REPORT_STEP          00:05:00
WET_STEP             00:01:00
DRY_STEP             00:01:00
ROUTING_STEP         5
INERTIAL_DAMPING     PARTIAL
NORMAL_FLOW_LIMITED  BOTH
FORCE_MAIN_EQUATION  H-W
VARIABLE_STEP        0
LENGTHENING_STEP     0
MIN_SURFAREA         0
MAX_TRIALS           2000
HEAD_TOLERANCE       1e-14
SYS_FLOW_TOL         5
LAT_FLOW_TOL         5
MINIMUM_STEP         0.5
THREADS              1
"""

#: An open trapezoid, an open rectangle, a circle and a closed rectangle in a row, every
#: junction fed so that none is dry (a dry junction with a net outflow sits a hair below its
#: invert on noodl's stiff bed, which shows at 4e-9 m3/s in the flow of the conduit above it
#: with or without a loss; test_a_dry_junction_with_a_net_outflow_is_the_stiff_beds_alone).
CHANNELS = """
[JUNCTIONS]
J1               10.0       3          0          0          0
J2               9.4        3          0          0          0
J3               8.9        3          0          0          0
J4               8.5        3          0          0          0

[OUTFALLS]
Outfall          8.0        FIXED      8.2              NO

[CONDUITS]
C1               J1               J2               300        0.020      0          0
C2               J2               J3               200        0.015      0          0
C3               J3               J4               150        0.013      0          0
C4               J4               Outfall          100        0.013      0          0

[XSECTIONS]
C1               TRAPEZOIDAL  0.8          0.5        1.0        1.0        1
C2               RECT_OPEN    0.8          0.9        0          0          1
C3               CIRCULAR     0.60         0          0          0          1
C4               RECT_CLOSED  0.6          0.8        0          0          1

[INFLOWS]
J1               FLOW             ""               FLOW     1.0      1.0      0.04
J2               FLOW             ""               FLOW     1.0      1.0      0.02
J3               FLOW             ""               FLOW     1.0      1.0      0.01
"""

DEFAULT_EVAPORATION = "CONSTANT         0.0\nDRY_ONLY         NO\n"


def fixture(*, body=CHANNELS, evaporation=DEFAULT_EVAPORATION, extra="",
            start=("01/01/2026", "00:00:00"), end=("01/01/2026", "00:20:00")):
    """The text of one fixture: the options, `[EVAPORATION]` `evaporation`, the `body`'s
    nodes, links and inflows, and any `extra` sections."""
    return (HEADER.format(start_date=start[0], start_time=start[1], end_date=end[0],
                          end_time=end[1])
            + "\n[EVAPORATION]\n" + evaporation + body + extra)


def swmm_rows(inp: Path):
    """SWMM on `inp`: per step the link flows, node depths and node volumes; the link
    connections. Asserts SWMM 5.2.4 and that it converged on every step."""
    rpt = inp.with_suffix(".rpt")
    rows = []
    with pyswmm.Simulation(str(inp), str(rpt), str(inp.with_suffix(".out"))) as sim:
        conn = {link.linkid: link.connections for link in pyswmm.Links(sim)}
        for _ in sim:
            nodes = list(pyswmm.Nodes(sim))
            rows.append(({link.linkid: link.flow for link in pyswmm.Links(sim)},
                         {node.nodeid: node.depth for node in nodes},
                         {node.nodeid: node.volume for node in nodes}))
        assert str(sim.engine_version) == "5.2.4"
    m = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", rpt.read_text())
    assert m is not None and float(m.group(1)) == 0.0, "SWMM did not converge on every step"
    return rows, conn


def compare(tmp_path, text, name="fixture"):
    """noodl against SWMM on `text` at every step: ``(worst depth difference (m), worst flow
    excess over the flow target (m3/s, <= 0 passes), worst storage-volume excess over
    ``TOL_DEPTH`` times the unit's area (m3, <= 0 passes), SWMM's rows, noodl's history,
    the network)``."""
    inp = tmp_path / f"{name}.inp"
    inp.write_text(text)
    rows, conn = swmm_rows(inp)
    net, _, _ = read_swmm_inp(inp)
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    record = ("water.phi", "water.q", "sewer.storage_volume")
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=len(rows), record=record)
    hyd = model.dynamic_wave
    junctions = list(hyd.junction_names)
    nodes = junctions + list(hyd.outfall_names)
    idx = torch.cat([model.junction_idx, hyd.outfall_idx])
    invert = hyd.node_invert[idx]
    names = list(model.net.nodes)
    ends = {n: (names[int(u)], names[int(v)]) for n, u, v in
            zip(model.link_names, hyd.edge_u, hyd.edge_v, strict=True)}
    sign = {n: (1.0 if tuple(conn[n]) == pair else -1.0) for n, pair in ends.items()}
    worst_depth, worst_flow, worst_volume = 0.0, -TOL_FLOW_ABS, -float("inf")
    for k, (flows, depths, volumes) in enumerate(rows):
        # SWMM's node depth is never negative: a dry junction with a net outflow (a conduit
        # evaporating into it) sits a hair below its invert on noodl's stiff bed, and is
        # reported at 0 (`sewer.junction_depth`)
        y = torch.clamp(hist["water.phi"][k, idx] - invert, min=0.0)
        for j, n in enumerate(nodes):
            worst_depth = max(worst_depth, abs(y[j].item() - depths[n]))
        for i, n in enumerate(model.link_names):
            swmm = sign[n] * flows[n]
            worst_flow = max(worst_flow, abs(hist["water.q"][k, i].item() - swmm)
                             - (TOL_FLOW_REL * abs(swmm) + TOL_FLOW_ABS))
        if hyd.storage_curves is not None:
            depth = torch.tensor([depths[n] for n in junctions], dtype=torch.float64)
            area = hyd.storage_curves.area(depth)
            for j, m in enumerate(net.manholes):
                if m.storage is not None:
                    worst_volume = max(worst_volume, abs(hist["sewer.storage_volume"][k, j].item()
                                                         - volumes[m.name])
                                       - TOL_DEPTH * area[j].item())
    return worst_depth, worst_flow, worst_volume, rows, hist, net


def assert_parity(result):
    worst_depth, worst_flow, worst_volume = result[:3]
    assert worst_depth <= TOL_DEPTH, result[:3]
    assert worst_flow <= 0.0, result[:3]
    assert worst_volume <= 0.0, result[:3]


def lost_water(a, b, link="C4"):
    """The largest difference (m3/s) between the flow of `link` in two SWMM runs, the rows `a`
    and `b`: the fixture's effect, so that a test knows the loss is not negligible."""
    return max(abs(x[0][link] - y[0][link]) for x, y in zip(a, b, strict=True))



def constant(rate, dry_only="NO"):
    """An ``[EVAPORATION]`` body: CONSTANT `rate` (mm/day) and DRY_ONLY."""
    return f"CONSTANT         {rate}\nDRY_ONLY         {dry_only}\n"


@pytest.mark.slow
def test_conduit_evaporation_matches_swmm(tmp_path):
    """A constant 4000 mm/day (4.6e-8 m/s) over an open trapezoid and rectangle (the circle
    and the closed rectangle lose nothing): depths, flows against SWMM, and the loss is real
    (a flow several 1e-3 m3/s below the lossless run's)."""
    result = compare(tmp_path, fixture(evaporation=constant(4000)), "evap")
    assert_parity(result)
    lossless = compare(tmp_path, fixture(), "lossless")
    assert_parity(lossless)
    assert lost_water(result[3], lossless[3]) > 1e-3


@pytest.mark.slow
def test_a_closed_network_loses_nothing_to_evaporation(tmp_path):
    """The same rate over circular conduits only: SWMM evaporates open sections only
    (link.c:1359), and so does noodl (no climate is even attached)."""
    closed = (CHANNELS.replace("TRAPEZOIDAL  0.8          0.5        1.0        1.0",
                               "CIRCULAR     0.6          0          0          0")
              .replace("RECT_OPEN    0.8          0.9        0          0",
                       "CIRCULAR     0.7          0          0          0"))
    result = compare(tmp_path, fixture(body=closed, evaporation=constant(4000)), "closed")
    assert_parity(result)
    assert result[5].climate is None


@pytest.mark.slow
def test_a_monthly_rate_and_its_adjustment_change_at_the_month_boundary(tmp_path):
    """MONTHLY evaporation (January 800, every other month 3500 mm/day) plus an
    ``[ADJUSTMENTS] EVAPORATION`` of 400 mm/day from February, across midnight of 1 February
    (halfway through the run): `climate_setState` reads the month at the start of each routing
    step."""
    rates = " ".join(["800"] + ["3500"] * 11)
    adjust = " ".join(["0"] + ["400"] * 11)
    text = fixture(evaporation=f"MONTHLY          {rates}\nDRY_ONLY         NO\n",
                   extra=f"\n[ADJUSTMENTS]\nEVAPORATION      {adjust}\n",
                   start=("01/31/2026", "23:50:00"), end=("02/01/2026", "00:10:00"))
    result = compare(tmp_path, text, "monthly")
    assert_parity(result)
    flat = fixture(start=("01/31/2026", "23:50:00"), end=("02/01/2026", "00:10:00"))
    assert lost_water(result[3], compare(tmp_path, flat, "monthly_lossless")[3]) > 2e-3


@pytest.mark.slow
def test_seepage_matches_swmm_in_every_section_kind(tmp_path):
    """Seepage of 40 (open trapezoid), 25 (closed rectangle, in place of the open one) and 60
    (circle) mm/hr, with the monthly conductivity adjustment at 0.6 (`Adjust.hydconFactor`): the
    width is the section's at the depth limited to its widest point (the circle's half depth,
    the closed rectangle's wMax), and the loss is a sizeable part of the inflow. (A seeping
    conduit at the foot of the chain leaves SWMM itself oscillating on a few steps.)"""
    body = CHANNELS.replace("RECT_OPEN    0.8          0.9        0          0",
                            "RECT_CLOSED  0.8          0.9        0          0")
    losses = ("\n[LOSSES]\nC1 0 0 0 NO 40\nC2 0 0 0 NO 25\nC3 0 0 0 NO 60\n"
              "\n[ADJUSTMENTS]\nCONDUCTIVITY    " + " ".join(["0.6"] * 12) + "\n")
    result = compare(tmp_path, fixture(body=body, extra=losses), "seepage")
    assert_parity(result)
    plain = compare(tmp_path, fixture(body=body), "seepage_lossless")
    assert lost_water(result[3], plain[3]) > 1e-3
    pipes = {p.name: p.seepage for p in result[5].pipes}
    assert pipes["C1"] == pytest.approx(40.0 / 3.6e6)             # mm/hr to m/s
    assert result[5].climate.adjustments.conductivity == (0.6,) * 12


@pytest.mark.slow
def test_a_dry_junction_with_a_net_outflow_is_the_stiff_beds_alone(tmp_path):
    """Why `CHANNELS` feeds every junction. A lateral outflow of 4.2e-3 m3/s at J3, which is
    dry through the front's arrival, moves the flow of C2 above it by 4e-9 m3/s -- 1e-6 of
    its flow, at the edge of the target -- with no loss at all: it is the stiff bed
    (`dynwave.K_BED`) that stands in for SWMM's ``yNew < 0 -> 0`` on a dry junction, not the
    loss. Depths are unaffected (clamped at 0 as SWMM's are)."""
    dry = CHANNELS.replace('J3               FLOW             ""               FLOW     1.0      '
                           '1.0      0.01', 'J3               FLOW             ""               '
                           'FLOW     1.0      1.0      -0.0042')
    result = compare(tmp_path, fixture(body=dry), "dry_outflow")
    assert result[0] <= TOL_DEPTH
    assert result[1] <= 1e-8


# ---------------------------------------------------------------- DRY_ONLY and the runoff clock
RAIN = """
[RAINGAGES]
RG1              VOLUME    0:01     1.0      TIMESERIES R1

[SUBCATCHMENTS]
S1               RG1              J1               0.4      60       120      1.5      0

[SUBAREAS]
S1               0.013      0.15       1.5        4.0        25         OUTLET

[INFILTRATION]
S1               80         10         4          7          0

[TIMESERIES]
R1               01/01/2026 00:04      0.6
R1                          00:05      1.2
R1                          00:06      1.2
R1                          00:07      0.8
R1                          00:08      0.0
R1                          00:20      0.0
"""


@pytest.mark.slow
def test_dry_only_does_not_stop_conduit_evaporation_while_it_rains(tmp_path):
    """``DRY_ONLY YES`` with a rain period (00:03 to 00:08) over a subcatchment: SWMM zeroes
    only the subcatchments' evaporation while it rains (subcatch.c:685); a conduit's
    ``Evap.rate`` is untouched (link.c:1359). The climate comes from the runoff steps
    (`losses.ClimateClock` shares the inflow driver's `RunoffModel`). The same file with
    ``DRY_ONLY NO`` differs only in the runoff, so the two answers must differ."""
    yes = compare(tmp_path, fixture(evaporation=constant(4000, "YES"), extra=RAIN), "dry_yes")
    assert_parity(yes)
    no = compare(tmp_path, fixture(evaporation=constant(4000, "NO"), extra=RAIN), "dry_no")
    assert_parity(no)
    assert yes[5].hydrology is not None and yes[5].climate is None
    assert yes[5].hydrology.evaporation.dry_only
    dry_only_effect = max(abs(a[0]["C4"] - b[0]["C4"]) for a, b in zip(yes[3], no[3], strict=True))
    assert dry_only_effect > 1e-7           # the subcatchment evaporates differently, so the
    #                                         runoff differs; the losses did not switch off


# ---------------------------------------------------------------------------------- storage
STORAGE = """
[JUNCTIONS]
J1               12.0       5          0          0          0

[STORAGE]
S1               10.0       3          0          {shape1}   0         {fevap1}   {exfil1}
S2               9.0        3          0          {shape2}   0         {fevap2}   {exfil2}

[CURVES]
Bowl             Storage    0          8
Bowl                        0.3        8
Bowl                        0.6        20
Bowl                        1.2        35
Bowl                        2.0        35

[OUTFALLS]
Outfall          7.0        FREE                        NO

[CONDUITS]
C1               J1               S1               100        0.013      0          0
C2               S1               S2               100        0.013      0          0
C3               S2               Outfall          100        0.013      0          0

[XSECTIONS]
C1               CIRCULAR     0.50         0          0          0          1
C2               CIRCULAR     0.15         0          0          0          1
C3               CIRCULAR     0.30         0          0          0          1

[INFLOWS]
J1               FLOW             ""               FLOW     1.0      1.0      0.04
S2               FLOW             ""               FLOW     1.0      1.0      0.01
"""

BOWL = "TABULAR    Bowl"
FUNCTIONAL = "FUNCTIONAL 20         1.5        5"
CONICAL = "CONICAL    4          2          0.5"
CYLINDER = "CYLINDRICAL 4         3          0"


def storage(shape1=BOWL, shape2=FUNCTIONAL, fevap1=0, fevap2=0, exfil1="", exfil2=""):
    return STORAGE.format(shape1=shape1, shape2=shape2, fevap1=fevap1, fevap2=fevap2,
                          exfil1=exfil1, exfil2=exfil2)


def test_storage_evaporation_matches_swmm(tmp_path):
    """Fevap 1.5 on a TABULAR unit and 0.8 on a FUNCTIONAL one at 4000 mm/day: the loss is
    ``Fevap Evap.rate area`` from the step-start depth (node.c:1072-1085), area and volume of
    the unit against SWMM, and a lossless run holds visibly more."""
    text = fixture(body=storage(fevap1=1.5, fevap2=0.8), evaporation=constant(4000))
    result = compare(tmp_path, text, "storage_evap")
    assert_parity(result)
    plain = compare(tmp_path, fixture(body=storage(), evaporation=constant(4000)), "storage_no")
    assert max(abs(a[2]["S1"] - b[2]["S1"]) for a, b in zip(result[3], plain[3],
                                                            strict=True)) > 0.01


@pytest.mark.slow
def test_storage_exfiltration_through_ksat_matches_swmm(tmp_path):
    """A single Ksat (3000 mm/hr) on a TABULAR unit (a bottom of 8 m2, a bank from 0.3 m
    to 1.2 m) and 1500 mm/hr on a CONICAL one (banks from the floor): ``Ksat hydconFactor``
    over the bottom and the bank area (exfil.c:158-182), with a conductivity adjustment of 0.7."""
    adjust = "\n[ADJUSTMENTS]\nCONDUCTIVITY    " + " ".join(["0.7"] * 12) + "\n"
    text = fixture(body=storage(exfil2="1500", shape2=CONICAL, exfil1="3000"), extra=adjust)
    result = compare(tmp_path, text, "storage_ksat")
    assert_parity(result)
    plain = compare(tmp_path, fixture(body=storage(shape2=CONICAL)), "storage_ksat_no")
    assert max(abs(a[2]["S1"] - b[2]["S1"]) for a, b in zip(result[3], plain[3],
                                                            strict=True)) > 0.01


@pytest.mark.slow
def test_storage_exfiltration_through_green_ampt_matches_swmm(tmp_path):
    """Suction head, Ksat and IMDmax (modified Green-Ampt at the depth of the water above the
    bottom and half the depth above the banks' foot, exfil.c:163-194) on a TABULAR unit
    (100 mm, 2500 mm/hr, 0.3), a FUNCTIONAL unit (60 mm, 1800 mm/hr, 0.45) and evaporation
    on both: the infiltration state runs from step to step."""
    text = fixture(body=storage(fevap1=1.0, fevap2=1.0, exfil1="100 2500 0.3",
                                exfil2="60 1800 0.45"), evaporation=constant(1500))
    result = compare(tmp_path, text, "storage_ga")
    assert_parity(result)


@pytest.mark.slow
def test_a_cylinder_has_only_a_bottom_to_exfiltrate_through(tmp_path):
    """A CYLINDRICAL unit's surface area is its bottom (exfil.c:119-135): no banks. Its
    Green-Ampt rate is the bottom's alone."""
    text = fixture(body=storage(shape1=CYLINDER, exfil1="80 2200 0.2"))
    assert_parity(compare(tmp_path, text, "storage_cylinder"))
