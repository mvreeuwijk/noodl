"""Dynamic-wave parity: noodl's fixed-point solve against live SWMM 5.2.4, every 5 s step.

SWMM is run with MAX_TRIALS 500 and a HEAD_TOLERANCE of 1e-12 m (in the fixtures), so each of
its routing steps is converged to the fixed point of its own trial equations; noodl solves the
same equations (docs/applications/sewer.md, "Dynamic wave") to `ITERATE_TOL`. The comparison
is only meaningful if SWMM converged on EVERY step, which each test asserts from SWMM's report
("% of Steps Not Converging" in its "Routing Time Step Summary").

Why 1e-12 m and not the 1e-9 m of the plan: SWMM stops a step when two successive trials
differ by less than HEAD_TOLERANCE, and its trials are averaged (relaxed) iterates, so the
distance to its fixed point is many times the tolerance. At 1e-9 m the tree's outfall
conduit C5 was 1e-8 m3/s apart while it first wets (steps 10-13) -- ten times the 1e-9 m3/s
floor below -- and J4's depth drifted to 1.8e-7 m apart by 30 min. Tightening SWMM to
1e-12 m alone brought the tree to 1.2e-9 m and below the flow floor over all 359 steps,
which locates the difference in SWMM's stopping rule, not in noodl. Tightening the reference
keeps the targets below unchanged.

noodl's side is tightened for the same reason: `Model(coupling="iterate")` stops when the
heads change by less than iterate_tol between passes, and where a pass contracts slowly (the
flat conduit C2 of the backwater line, the outfall head of the loop at low flow) the distance
to the fixed point is many times that change. At the default 1e-11 m that left J2 of the
backwater line 6e-9 m from SWMM and C2's flow 2e-9 m3/s off (q ~ 2e-4 m3/s); at 1e-13 m both
fall to 8e-11 m and below the flow floor. The loop's outfall head contracts by only ~0.92 per
pass near t = 500 s, so the pass budget is raised as well (327 passes needed at 1e-11 m).

Flow sign. SWMM reverses an adverse conduit under DYNWAVE (`link.c::conduit_reverse` swaps
node1/node2 and sets ``direction = -1``), and noodl's reader orients it the same way. The
toolkit reports BOTH quantities in the FILE's orientation: `swmm_getLinkResult(SM_LINKFLOW)`
returns ``newFlow * direction * UCF(FLOW)`` (swmm-toolkit 0.17.0, OWA toolkit.c), and pyswmm's
`Link.connections` swaps the internal node pair back when ``direction == -1``
(`pyswmm/swmm5.py::getLinkConnections`). So a SWMM flow is positive from
``connections[0]`` to ``connections[1]``, and ``sign`` maps it onto noodl's orientation:
+1 where the reader kept the file's From/To, -1 where it reversed the conduit.

Steps. pyswmm's iterator yields after every routing step except the last: the final
`swmm_step` returns 0 and ends the loop. 30 min at 5 s therefore yields 359 rows, row k being
the state after step k + 1, the same indexing as noodl's history.
"""

import re
import shutil
from pathlib import Path

import pytest

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave

pyswmm = pytest.importorskip("pyswmm")

DATA = Path(__file__).resolve().parents[1] / "data" / "sewer"
NETWORKS = ["tree_dynwave.inp", "backwater_dynwave.inp", "loop_dynwave.inp"]
#: Depth target (m): one decade above the ~1e-8 m fixed-point error SWMM's own stopping rule
#: leaves at HEAD_TOLERANCE 1e-9 m (it is 1e-12 m here, see the module docstring).
TOL_DEPTH = 1e-7
#: Flow target: relative, plus an absolute floor at the scale of SWMM's FUDGE flow.
TOL_FLOW_REL, TOL_FLOW_ABS = 1e-6, 1e-9
#: The fast comparison: the first two minutes of the tree, from the dry start through the
#: filling of every junction and the first flow out of the outfall conduit.
FAST_STEPS = 24
#: noodl's head convergence (m) and pass budget for the comparison (module docstring).
ITERATE_TOL, ITERATE_MAX = 1e-13, 2000


def _swmm(tmp_path, name):
    """Run the fixture in SWMM; per-step link flows (file orientation, m3/s) and node depths
    (m), the file-orientation connections, the engine version and the report's percentage
    of steps not converging."""
    inp = tmp_path / name
    shutil.copy(DATA / name, inp)
    rpt = tmp_path / f"{inp.stem}.rpt"
    rows = []
    with pyswmm.Simulation(str(inp), str(rpt), str(tmp_path / f"{inp.stem}.out")) as sim:
        conn = {link.linkid: link.connections for link in pyswmm.Links(sim)}
        for _ in sim:
            rows.append((
                {link.linkid: link.flow for link in pyswmm.Links(sim)},
                {node.nodeid: node.depth for node in pyswmm.Nodes(sim)},
            ))
        engine = str(sim.engine_version)
    text = rpt.read_text()
    m = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", text)
    return rows, conn, engine, (float(m.group(1)) if m else None)


def _worst_errors(name, rows, conn, steps):
    """noodl over `steps` steps against SWMM's first `steps` rows: the largest junction depth
    difference (m) and the largest flow excess over the flow target (m3/s, <= 0 passes)."""
    net, _, _ = read_swmm_inp(DATA / name)
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=steps)
    junctions = [m.name for m in net.manholes]
    sign = {p.name: (1.0 if tuple(conn[p.name]) == (p.u, p.v) else -1.0) for p in net.pipes}
    worst_depth = worst_flow = 0.0
    for k, (flows, depths) in enumerate(rows[:steps]):
        y = hist["water.phi"][k, model.junction_idx] - model.junction_invert
        for j, name_j in enumerate(junctions):
            worst_depth = max(worst_depth, abs(y[j].item() - depths[name_j]))
        for i, pipe in enumerate(net.pipes):
            q_s = sign[pipe.name] * flows[pipe.name]
            q_n = hist["water.q"][k, i].item()
            excess = abs(q_n - q_s) - (TOL_FLOW_REL * abs(q_s) + TOL_FLOW_ABS)
            worst_flow = max(worst_flow, excess)
    return worst_depth, worst_flow


def _converged_swmm(tmp_path, name):
    rows, conn, engine, not_converging = _swmm(tmp_path, name)
    assert engine == "5.2.4"
    assert not_converging == 0.0, f"SWMM did not converge on every step ({not_converging} %)"
    return rows, conn


def test_dynamic_wave_matches_swmm_while_the_tree_fills(tmp_path):
    """The first `FAST_STEPS` steps of the tree at the full targets: the cold start (zero
    net inflow at the first step's start, as SWMM), the dry-to-wet flow classes and the
    outfall conduit's first flow."""
    rows, conn = _converged_swmm(tmp_path, "tree_dynwave.inp")
    worst_depth, worst_flow = _worst_errors("tree_dynwave.inp", rows, conn, FAST_STEPS)
    assert worst_depth <= TOL_DEPTH, worst_depth
    assert worst_flow <= 0.0, worst_flow


@pytest.mark.slow
@pytest.mark.parametrize("name", NETWORKS)
def test_dynamic_wave_matches_swmm_at_every_step(tmp_path, name):
    rows, conn = _converged_swmm(tmp_path, name)
    worst_depth, worst_flow = _worst_errors(name, rows, conn, len(rows))
    assert worst_depth <= TOL_DEPTH, worst_depth
    assert worst_flow <= 0.0, worst_flow


def test_loop_sign_convention(tmp_path):
    """Cd is adverse in the file (J3 9.60 m -> J4 9.65 m); SWMM reverses it under DYNWAVE and
    so does the reader. pyswmm reports the FILE's orientation for both the connections and
    the flow (module docstring): Cd carries J3's water to J4, so its SWMM flow is positive,
    and noodl's orientation is the reverse of the connections."""
    rows, conn, _, _ = _swmm(tmp_path, "loop_dynwave.inp")
    net, _, _ = read_swmm_inp(DATA / "loop_dynwave.inp")
    cd = next(p for p in net.pipes if p.name == "Cd")
    assert (cd.u, cd.v) == ("J4", "J3")
    assert tuple(conn["Cd"]) == ("J3", "J4")
    assert rows[-1][0]["Cd"] > 1e-3
