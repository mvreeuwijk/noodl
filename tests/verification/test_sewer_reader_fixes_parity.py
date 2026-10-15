"""Live SWMM 5.2.4 parity of three reader/validation rules: negative conduit offsets (clamped
to 0, link.c:498-502 and 1057-1067), networks with a junction no link reaches or a group of
nodes without an outfall (SWMM asks for one outfall in all, flowrout.c:325), and a TIMESERIES
outfall whose stage comes from the file's own series (node.c:1454-1458, held at its ends).

Each model is written here (SI units, dynamic wave, 5 s steps), run by pyswmm at
HEAD_TOLERANCE 1e-12 m and MAX_TRIALS 500 and by noodl at ``iterate_tol`` 1e-13 m, and the
node depths and link flows compared every step (the targets of
`test_sewer_dynwave_parity.py`). SWMM must converge on every step.
"""

import re

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave

pyswmm = pytest.importorskip("pyswmm")

TOL_DEPTH = 1e-7
TOL_FLOW_REL, TOL_FLOW_ABS = 1e-6, 1e-9
ITERATE_TOL, ITERATE_MAX = 1e-13, 2000

HEADER = """[OPTIONS]
FLOW_UNITS           CMS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
LINK_OFFSETS         {offsets}
MIN_SLOPE            0
ALLOW_PONDING        NO
SKIP_STEADY_STATE    NO
START_DATE           01/01/2026
START_TIME           00:00:00
REPORT_START_DATE    01/01/2026
REPORT_START_TIME    00:00:00
END_DATE             01/01/2026
END_TIME             00:{minutes:02d}:00
REPORT_STEP          00:00:05
WET_STEP             00:00:05
DRY_STEP             00:00:05
ROUTING_STEP         5
INERTIAL_DAMPING     PARTIAL
NORMAL_FLOW_LIMITED  BOTH
FORCE_MAIN_EQUATION  H-W
VARIABLE_STEP        0
LENGTHENING_STEP     0
MIN_SURFAREA         0
MAX_TRIALS           500
HEAD_TOLERANCE       1e-12
MINIMUM_STEP         0.5
THREADS              1

"""


def _model(offsets="DEPTH", minutes=5, junctions=(), outfalls=(), conduits=(), inflows=(),
           extra=""):
    """An SI dynamic-wave .inp from its sections' data lines."""
    text = HEADER.format(offsets=offsets, minutes=minutes)
    text += "[JUNCTIONS]\n" + "\n".join(junctions) + "\n\n"
    text += "[OUTFALLS]\n" + "\n".join(outfalls) + "\n\n"
    text += "[CONDUITS]\n" + "\n".join(c[0] for c in conduits) + "\n\n"
    text += "[XSECTIONS]\n" + "\n".join(c[1] for c in conduits) + "\n\n"
    text += "[INFLOWS]\n" + "\n".join(f"{n} FLOW \"\" FLOW 1.0 1.0 {q}" for n, q in inflows)
    return text + "\n\n" + extra


def _conduit(name, a, b, length, in_off=0, out_off=0, diameter=0.3):
    return (f"{name} {a} {b} {length} 0.013 {in_off} {out_off} 0 0",
            f"{name} CIRCULAR {diameter} 0 0 0 1")


def _swmm(tmp_path, text):
    """SWMM's per-step link flows (file orientation) and node depths, and the connections."""
    inp = tmp_path / "m.inp"
    inp.write_text(text)
    rows = []
    with pyswmm.Simulation(str(inp), str(tmp_path / "m.rpt"), str(tmp_path / "m.out")) as sim:
        conn = {link.linkid: link.connections for link in pyswmm.Links(sim)}
        for _ in sim:
            rows.append(({link.linkid: link.flow for link in pyswmm.Links(sim)},
                         {node.nodeid: node.depth for node in pyswmm.Nodes(sim)}))
        assert str(sim.engine_version) == "5.2.4"
    report = (tmp_path / "m.rpt").read_text()
    not_converging = float(re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", report)[1])
    assert not_converging == 0.0, "SWMM did not converge on every step"
    return rows, conn


def _assert_parity(tmp_path, text, **build):
    """noodl against SWMM on `text` over every step; returns the noodl history, model."""
    rows, conn = _swmm(tmp_path, text)
    net, _, _ = read_swmm_inp(tmp_path / "m.inp")
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX,
                                        **build)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=len(rows),
                                    record=("water.phi", "water.q"))
    hyd = model.dynamic_wave
    nodes = list(hyd.junction_names) + list(hyd.outfall_names)
    idx = torch.cat([model.junction_idx, hyd.outfall_idx])
    names = list(model.net.nodes)
    ends = {n: (names[int(u)], names[int(v)])
            for n, u, v in zip(model.link_names, hyd.edge_u, hyd.edge_v, strict=True)}
    for k, (flows, depths) in enumerate(rows):
        y = hist["water.phi"][k, idx] - hyd.node_invert[idx]
        for j, node in enumerate(nodes):
            assert abs(y[j].item() - depths[node]) <= TOL_DEPTH, (k, node)
        for i, link in enumerate(model.link_names):
            sign = 1.0 if tuple(conn[link]) == ends[link] else -1.0
            ref = sign * flows[link]
            assert abs(hist["water.q"][k, i].item() - ref) <= (
                TOL_FLOW_REL * abs(ref) + TOL_FLOW_ABS), (k, link)
    return hist, model, rows


@pytest.mark.parametrize("offsets", ["DEPTH", "ELEVATION"])
def test_negative_conduit_offsets_match_swmm(tmp_path, offsets):
    """C1 enters 0.3 m below J1's invert and C2 leaves 0.2 m below J2's (in ELEVATION
    files the conduit end elevations are below the node inverts): SWMM clamps both to 0."""
    deep = offsets == "ELEVATION"
    text = _model(
        offsets=offsets,
        junctions=["J1 12.0 3 0 0 0", "J2 11.0 3 0 0 0"],
        outfalls=["O 10.0 FREE"],
        conduits=[_conduit("C1", "J1", "J2", 100, 11.7 if deep else -0.3, 10.98 if deep else 0),
                  _conduit("C2", "J2", "O", 100, 10.9 if deep else 0, 9.8 if deep else -0.2)],
        inflows=[("J1", 0.03)])
    (tmp_path / "m.inp").write_text(text)
    net, _, _ = read_swmm_inp(tmp_path / "m.inp")
    assert all(p.inlet_offset == 0.0 == p.outlet_offset for p in net.pipes)
    hist, model, _ = _assert_parity(tmp_path, text)
    assert hist["water.q"][-1].max().item() > 1e-3           # water moved


def test_a_junction_without_links_and_a_group_without_an_outfall_match_swmm(tmp_path):
    """JX has no link (it fills from its own inflow to its 1 m full depth and floods); JA
    -> JB drains into a dead end with no outfall (JB takes the water of JA); the outfall
    component is the ordinary one. SWMM's depths and flows follow from the nodes' own
    continuity, which the junction-storage row reproduces."""
    text = _model(
        minutes=8,
        junctions=["J1 5.0 3 0 0 0", "JX 5.0 1 0 0 0", "JA 6.0 3 0 0 0", "JB 5.5 3 0 0 0"],
        outfalls=["O 4.0 FREE"],
        conduits=[_conduit("C1", "J1", "O", 80), _conduit("CA", "JA", "JB", 60)],
        inflows=[("J1", 0.02), ("JX", 0.01), ("JA", 0.005)])
    hist, model, rows = _assert_parity(tmp_path, text)
    last = rows[-1][1]
    assert last["JX"] == pytest.approx(1.0, abs=1e-9)        # full and flooding
    assert last["JB"] > 0.05                                  # the dead end collects water


def test_a_timeseries_outfall_takes_its_stage_from_the_file(tmp_path):
    """The stage ramps from below the conduit's invert to above its crown within the
    window, then is held (the series ends at 0:04 and the run goes on to 0:06): SWMM
    reads it at the step's end time and holds its last value."""
    text = _model(
        minutes=6,
        junctions=["J1 5.0 3 0 0 0", "J2 4.5 3 0 0 0"],
        outfalls=["O 4.0 TIMESERIES Tide"],
        conduits=[_conduit("C1", "J1", "J2", 100), _conduit("C2", "J2", "O", 100)],
        inflows=[("J1", 0.02)],
        extra="[TIMESERIES]\nTide 0:00 3.5\nTide 0:02 4.2\nTide 0:04 5.1\n")
    hist, model, _ = _assert_parity(tmp_path, text)
    stage = model.dynamic_wave.outfall_stage(360.0, model.outfall_stage)
    assert stage.tolist() == pytest.approx([5.1])
