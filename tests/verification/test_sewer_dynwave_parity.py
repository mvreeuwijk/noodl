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

Newton tolerance. The head solve keeps `simulate_dynamic_wave`'s default atol, `solve_atol`
(2.5e-9 m3/s here), which exceeds `TOL_FLOW_ABS`. The flow target is still met because a
pass is linear in the heads between its kinks: one Newton step lands at round-off, far below
atol, and only a pass whose starting residual is already below atol stops without a step,
which bounds a wet junction's head error by atol dt / (2 A_s) ~ 5e-9 m and the conduit
flows correspondingly; every full run below meets the flow target with it.

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
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave
from noodl.apps.sewer.xsect_tables import QCF

pyswmm = pytest.importorskip("pyswmm")

DATA = Path(__file__).resolve().parents[1] / "data" / "sewer"
NETWORKS = ["tree_dynwave.inp", "backwater_dynwave.inp", "loop_dynwave.inp"]
#: Depth target (m): over twenty times the worst agreement measured against the 1e-12 m
#: reference over all three full runs (4.5e-9 m, backwater J2), so a miss is a real
#: discrepancy, not round-off or the iteration tolerances on either side.
TOL_DEPTH = 1e-7
#: Flow target: relative, plus an absolute floor at the scale of SWMM's FUDGE flow.
TOL_FLOW_REL, TOL_FLOW_ABS = 1e-6, 1e-9
#: The fast comparison: the first two minutes of the tree, from the dry start through the
#: filling of every junction and the first flow out of the outfall conduit.
FAST_STEPS = 24
#: noodl's head convergence (m) and pass budget for the comparison (module docstring).
ITERATE_TOL, ITERATE_MAX = 1e-13, 2000


def _swmm(tmp_path, name):
    """Run the fixture in SWMM; per-step link flows (file orientation, m3/s), node depths
    (m) and node flooding (m3/s), the file-orientation connections, the engine version and
    the report's percentage of steps not converging.

    pyswmm's `Node.flooding` is the toolkit's ``SM_OVERFLOW``, ``Node.overflow *
    UCF(FLOW)`` (OWA toolkit.c): the flooding RATE of the step just taken (m3/s in a CMS
    file), the ``dV/dt`` that `dynwave.c::getFloodedDepth` sets (dynwave.c:783-791), not a
    volume -- the same quantity as noodl's ``sewer.overflow``."""
    inp = tmp_path / name
    shutil.copy(DATA / name, inp)
    rpt = tmp_path / f"{inp.stem}.rpt"
    rows = []
    with pyswmm.Simulation(str(inp), str(rpt), str(tmp_path / f"{inp.stem}.out")) as sim:
        conn = {link.linkid: link.connections for link in pyswmm.Links(sim)}
        for _ in sim:
            nodes = list(pyswmm.Nodes(sim))
            rows.append((
                {link.linkid: link.flow for link in pyswmm.Links(sim)},
                {node.nodeid: node.depth for node in nodes},
                {node.nodeid: node.flooding for node in nodes},
            ))
        engine = str(sim.engine_version)
    text = rpt.read_text()
    m = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", text)
    return rows, conn, engine, (float(m.group(1)) if m else None)


def _run_noodl(name, steps, split=None, **solve_kwargs):
    """noodl on the fixture `name` over `steps` steps, recording heads, flows and
    ``sewer.overflow``: the network, the model, the history and, when `split` is a step
    count, the state after `split` steps (the run is then two calls, which is the same
    run). `solve_kwargs` reach the head solve."""
    net, _, _ = read_swmm_inp(DATA / name)
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    record = ("water.phi", "water.q", "sewer.overflow")
    parts, at_split = [], None
    for n in ([steps] if split is None else [split, steps - split]):
        state, hist = simulate_dynamic_wave(model, state, drivers, steps=n, record=record,
                                            **solve_kwargs)
        parts.append(hist)
        at_split = state if at_split is None else at_split
    hist = {key: torch.cat([h[key] for h in parts]) for key in record}
    return net, model, hist, (at_split if split is not None else None)


def _differences(name, rows, conn, steps, run=None, **solve_kwargs):
    """noodl over `steps` steps (`run`, a `_run_noodl` result, or a new run with
    `solve_kwargs`) against SWMM's first `steps` rows, per step: a dict from ``("depth",
    node)`` to the junction or outfall depth difference (m), and from ``("flow",
    conduit)`` and ``("overflow", junction)`` to the excess (m3/s) of the flow or overflow
    difference over the flow target (<= 0 passes)."""
    net, model, hist, _ = run or _run_noodl(name, steps, **solve_kwargs)
    hyd = model.dynamic_wave
    # junctions from the solved heads; outfalls from the closure-written boundary heads
    junctions = list(hyd.junction_names)
    nodes = junctions + list(hyd.outfall_names)
    idx = torch.cat([model.junction_idx, hyd.outfall_idx])
    invert = hyd.node_invert[idx]
    sign = {p.name: (1.0 if tuple(conn[p.name]) == (p.u, p.v) else -1.0) for p in net.pipes}

    def excess(noodl, swmm):
        return abs(noodl - swmm) - (TOL_FLOW_REL * abs(swmm) + TOL_FLOW_ABS)

    out = []
    for k, (flows, depths, flooding) in enumerate(rows[:steps]):
        y = hist["water.phi"][k, idx] - invert
        diff = {("depth", n): abs(y[j].item() - depths[n]) for j, n in enumerate(nodes)}
        for i, pipe in enumerate(net.pipes):
            diff["flow", pipe.name] = excess(hist["water.q"][k, i].item(),
                                             sign[pipe.name] * flows[pipe.name])
        for j, n in enumerate(junctions):
            diff["overflow", n] = excess(hist["sewer.overflow"][k, j].item(), flooding[n])
        out.append(diff)
    return out


def _worst(diffs, keep=lambda k, key: True):
    """The largest depth difference (m), flow excess and overflow excess (m3/s) over the
    per-step `diffs` of `_differences`, over the (0-based step, key) pairs `keep` accepts."""
    worst = {"depth": 0.0, "flow": -TOL_FLOW_ABS, "overflow": -TOL_FLOW_ABS}
    for k, diff in enumerate(diffs):
        for key, value in diff.items():
            if keep(k, key):
                worst[key[0]] = max(worst[key[0]], value)
    return worst["depth"], worst["flow"], worst["overflow"]


def _worst_errors(name, rows, conn, steps, **solve_kwargs):
    """`_worst` of `_differences` over every step and quantity: the largest depth
    difference (m), flow excess and overflow excess (m3/s; <= 0 passes)."""
    return _worst(_differences(name, rows, conn, steps, **solve_kwargs))


def _assert_parity(worst):
    worst_depth, worst_flow, worst_overflow = worst
    assert worst_depth <= TOL_DEPTH, worst
    assert worst_flow <= 0.0, worst
    assert worst_overflow <= 0.0, worst


def _converged_swmm(tmp_path, name, surcharged=False):
    """SWMM's rows, after checking the engine and that it converged on every step. Without
    `surcharged` no conduit end may reach its crown (the free-surface fixtures must stay
    free-surface, or they would test something else); with it some end must."""
    rows, conn, engine, not_converging = _swmm(tmp_path, name)
    assert engine == "5.2.4"
    assert not_converging == 0.0, f"SWMM did not converge on every step ({not_converging} %)"
    net, _, _ = read_swmm_inp(DATA / name)
    reached = False
    for k, (_, depths, _) in enumerate(rows):
        for p in net.pipes:
            for node, offset in ((p.u, p.inlet_offset), (p.v, p.outlet_offset)):
                fill = (depths[node] - offset) / p.diameter
                reached |= fill >= 1.0
                assert surcharged or fill < 1.0, (
                    f"{name}: SWMM surcharges conduit {p.name} at {node} at step {k + 1} "
                    f"(end depth {fill:.3f} of its diameter); reduce the fixture's inflows")
    assert reached == surcharged, f"{name}: surcharge expected={surcharged}, reached={reached}"
    return rows, conn


def test_dynamic_wave_matches_swmm_while_the_tree_fills(tmp_path):
    """The first `FAST_STEPS` steps of the tree at the full targets: the cold start (zero
    net inflow at the first step's start, as SWMM), the dry-to-wet flow classes and the
    outfall conduit's first flow."""
    rows, conn = _converged_swmm(tmp_path, "tree_dynwave.inp")
    _assert_parity(_worst_errors("tree_dynwave.inp", rows, conn, FAST_STEPS))


@pytest.mark.slow
@pytest.mark.parametrize("name", NETWORKS)
def test_dynamic_wave_matches_swmm_at_every_step(tmp_path, name):
    rows, conn = _converged_swmm(tmp_path, name)
    _assert_parity(_worst_errors(name, rows, conn, len(rows)))


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


SLOT_NAME = "tree_slot_dynwave.inp"
#: The first step at which J4's depth passes C4's crown (0.30 m) in SWMM is 95; 110 steps
#: cover the approach, the crossing and 15 steps of slot-pressurised flow.
SLOT_FAST_STEPS = 110


def test_slot_surcharge_matches_swmm_after_the_crown(tmp_path):
    """`SURCHARGE_METHOD SLOT` on the tree at 2.5x the dynwave inflows (J4 passes C4's crown
    in step 95): noodl against SWMM at the same targets as the free-surface runs, SWMM
    converging on every step."""
    rows, conn = _converged_swmm(tmp_path, SLOT_NAME, surcharged=True)
    _assert_parity(_worst_errors(SLOT_NAME, rows, conn, SLOT_FAST_STEPS))


@pytest.mark.slow
def test_slot_surcharge_matches_swmm_at_every_step(tmp_path):
    rows, conn = _converged_swmm(tmp_path, SLOT_NAME, surcharged=True)
    _assert_parity(_worst_errors(SLOT_NAME, rows, conn, len(rows)))


EXTRAN_NAME = "tree_extran_dynwave.inp"
#: SWMM's surcharged junctions (depth above the highest attached crown) first appear at
#: steps 21 (J2), 139 (J4) and 270 (J3); 150 steps cover the first two crossings.
EXTRAN_FAST_STEPS = 150
#: The head solve's tolerance (m3/s) for the EXTRAN runs. A surcharged junction stores
#: nothing, so a balance residual r moves its head by r / sum(k), with sum(k) ~ 0.01 m2/s
#: of full-conduit conveyance: at the default `solve_atol` (2.5e-9 m3/s here, the stiff
#: bed's round-off floor) that is up to 2.5e-7 m (2.45e-7 m measured at J4, step 352). No
#: junction of this run sits on its bed after the first steps, so 1e-12 m3/s is reachable.
EXTRAN_ATOL = 1e-12


def test_extran_surcharge_matches_swmm_through_two_crossings(tmp_path):
    """`SURCHARGE_METHOD EXTRAN` on the tree at 3x the dynwave inflows: J2 and J4 jump
    above their crowns (steps 21 and 139) and hold instantaneous continuity there. SWMM
    converges on every step of this run (EXTRAN surcharge need not: see
    tests/apps/sewer/test_dynwave.py::test_crown_crossing_converges_or_is_named), so all
    steps are compared at the free-surface targets."""
    rows, conn = _converged_swmm(tmp_path, EXTRAN_NAME, surcharged=True)
    _assert_parity(_worst_errors(EXTRAN_NAME, rows, conn, EXTRAN_FAST_STEPS,
                                 atol=EXTRAN_ATOL))


@pytest.mark.slow
def test_extran_surcharge_matches_swmm_at_every_step(tmp_path):
    rows, conn = _converged_swmm(tmp_path, EXTRAN_NAME, surcharged=True)
    _assert_parity(_worst_errors(EXTRAN_NAME, rows, conn, len(rows),
                                 atol=EXTRAN_ATOL))


#: The ramp-down: the tree's inflows held 120 steps (10 min), ramped to nothing over 12
#: steps, then off for 120 steps (21 min in all).
RAMP_FACTORS = [1.0] * 120 + [1.0 - (k + 1) / 12 for k in range(12)] + [0.0] * 120
#: SWMM's first non-converging step of the ramp-down (1-based; see the test).
RAMP_SWMM_FIRST_FAILURE = 145


def _swmm_scripted(tmp_path, name, factors, steps):
    """`name` in SWMM with its [INFLOWS] removed and each junction's baseline inflow times
    ``factors[k]`` fed through pyswmm's `generated_inflow` before step k + 1 (the lateral
    inflow at the END of the step, as noodl's per-step series), run for `steps` steps:
    rows of (flows, depths) as `_swmm` (row k after step k + 1, none after the last), the
    connections and the report's percentage of steps not converging."""
    net, _, _ = read_swmm_inp(DATA / name)
    base = {m.name: m.inflow for m in net.manholes if m.inflow != 0.0}
    text = (DATA / name).read_text()
    text = re.sub(r"\[INFLOWS\].*?\n\n", "[INFLOWS]\n\n", text, flags=re.S)
    secs = steps * int(net.dynamic_wave.routing_step)
    text = re.sub(r"END_TIME\s+\S+",
                  f"END_TIME             {secs // 3600:02d}:{secs % 3600 // 60:02d}:"
                  f"{secs % 60:02d}", text)
    inp = tmp_path / f"scripted_{steps}.inp"
    inp.write_text(text)
    rpt = inp.with_suffix(".rpt")
    rows = []
    with pyswmm.Simulation(str(inp), str(rpt), str(inp.with_suffix(".out"))) as sim:
        nodes = {node.nodeid: node for node in pyswmm.Nodes(sim)}
        links = list(pyswmm.Links(sim))
        conn = {link.linkid: link.connections for link in links}
        iterator = iter(sim)
        for k in range(steps):
            for junction, q in base.items():
                nodes[junction].generated_inflow(factors[k] * q)
            if next(iterator, None) is None:
                break
            rows.append(({link.linkid: link.flow for link in links},
                         {n: node.depth for n, node in nodes.items()}))
    m = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", rpt.read_text())
    return rows, conn, (float(m.group(1)) if m else None)


@pytest.mark.slow
def test_ramp_down_matches_swmm_until_swmm_stops_converging(tmp_path):
    """Every inflow of the tree ramps down to nothing (`RAMP_FACTORS`), in SWMM (pyswmm's
    `generated_inflow`) and in noodl (a per-step inflow series).

    Compared steps (E9): SWMM converges on every step up to 144, which covers the fill, the
    whole ramp and the drain until J4 meets the outfall's depth; those are asserted at the
    free-surface targets. From step 145 on SWMM fails to converge on 82 of the remaining 108
    steps (found by ending the run after each step; its report says 32.5 % of all 252), as
    C5 (J4 -> free outfall) reaches the switch of its normal-flow slope test: its limited
    flow is Manning's normal flow at J4's depth, at which the outfall's own depth (set from
    that flow) is J4's depth again, up to the ~1e-5 m interpolation mismatch of SWMM's
    tables, so neither side of the test need be consistent (docs/applications/sewer.md,
    "Dynamic wave": the normal-flow slope switch). SWMM's last trial is then wherever its
    alternating trials stopped, J4 within ~6e-4 m of the outfall depth on either side, and
    later steps start from that state, so no later step is a fixed-point comparison: steps
    145 onwards are excluded from the targets. They are asserted only to the scale of
    SWMM's own oscillation (1e-3 m, 5e-4 m3/s), with noodl converging on every one of them,
    on SWMM's fixed point where one exists and on the switching surface where none does."""
    steps = len(RAMP_FACTORS)
    before = RAMP_SWMM_FIRST_FAILURE - 1
    _, _, until = _swmm_scripted(tmp_path, "tree_dynwave.inp", RAMP_FACTORS, before)
    assert until == 0.0, f"SWMM did not converge on every step up to {before} ({until} %)"
    _, _, through = _swmm_scripted(tmp_path, "tree_dynwave.inp", RAMP_FACTORS, before + 1)
    assert through > 0.0                       # step 145 is SWMM's first failure
    rows, conn, _ = _swmm_scripted(tmp_path, "tree_dynwave.inp", RAMP_FACTORS, steps)
    assert len(rows) == steps - 1              # the last step yields no row (module docstring)

    net, _, _ = read_swmm_inp(DATA / "tree_dynwave.inp")
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    inflow = torch.stack([f * drivers["inflow"] for f in RAMP_FACTORS]).unsqueeze(1)
    diags: list = []
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=steps, inflow=inflow,
                                    diagnostics=diags)
    assert all(bool(d["converged"]) for d in diags)
    hyd = model.dynamic_wave
    nodes = [m.name for m in net.manholes] + list(hyd.outfall_names)
    idx = torch.cat([model.junction_idx, hyd.outfall_idx])
    sign = {p.name: (1.0 if tuple(conn[p.name]) == (p.u, p.v) else -1.0) for p in net.pipes}
    worst = {"compared": [0.0, 0.0], "excluded": [0.0, 0.0]}
    for k, (flows, depths) in enumerate(rows):
        y = hist["water.phi"][k, 0, idx] - hyd.node_invert[idx]
        part = worst["compared" if k < before else "excluded"]
        for j, name_j in enumerate(nodes):
            part[0] = max(part[0], abs(y[j].item() - depths[name_j]))
        for i, pipe in enumerate(net.pipes):
            q_s = sign[pipe.name] * flows[pipe.name]
            q_n = hist["water.q"][k, 0, i].item()
            if k < before:
                excess = abs(q_n - q_s) - (TOL_FLOW_REL * abs(q_s) + TOL_FLOW_ABS)
            else:
                excess = abs(q_n - q_s)
            part[1] = max(part[1], excess)
    assert worst["compared"][0] <= TOL_DEPTH, worst
    assert worst["compared"][1] <= 0.0, worst
    assert worst["excluded"][0] <= 1e-3 and worst["excluded"][1] <= 5e-4, worst


# ------------------------------------------------------------------------------ flooding
FLOOD_NAME = "flooding_dynwave.inp"
#: SWMM's flooded steps (1-based) per junction, over the 359 steps of each flooding fixture.
#: EXTRAN: J2 (MaxDepth 0.3 m + SurDepth 1.7 m, so y_max 2.0 m) floods as it jumps above
#: its crown and releases once C2 carries its inflow; J3 (MaxDepth 0.6 m) floods from the
#: step it passes its crown. SLOT: only J3 floods (J2 peaks at 1.49 m).
FLOODED_STEPS = {
    FLOOD_NAME: {"J2": list(range(21, 28)), "J3": list(range(270, 360))},
    "flooding_slot_dynwave.inp": {"J3": list(range(307, 360))},
}
#: The fast comparison: J2's flood onset, the seven flooded steps and its release.
FLOOD_FAST_STEPS = 40


def _flooded_steps(rows):
    """SWMM's flooded steps (1-based) per junction that floods at all."""
    out: dict[str, list[int]] = {}
    for k, (_, _, flooding) in enumerate(rows):
        for node, rate in flooding.items():
            if rate > 0.0:
                out.setdefault(node, []).append(k + 1)
    return out


def test_flooding_matches_swmm_through_a_flood_and_its_release(tmp_path):
    """EXTRAN flooding at a junction with a surcharge depth: J2 floods at MaxDepth +
    SurDepth = 2.0 m on steps 21-27 and then drains back below it. Depths, flows and
    ``sewer.overflow`` (against pyswmm's `Node.flooding`, `_swmm`) over the first
    `FLOOD_FAST_STEPS` steps at the free-surface targets, SWMM converging on every step."""
    rows, conn = _converged_swmm(tmp_path, FLOOD_NAME, surcharged=True)
    assert _flooded_steps(rows[:FLOOD_FAST_STEPS]) == {"J2": FLOODED_STEPS[FLOOD_NAME]["J2"]}
    _assert_parity(_worst_errors(FLOOD_NAME, rows, conn, FLOOD_FAST_STEPS, atol=EXTRAN_ATOL))


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(FLOODED_STEPS))
def test_flooding_matches_swmm_at_every_step(tmp_path, name):
    """Both flooding fixtures over all 359 steps, at the free-surface targets. No flooded
    junction here drains into an outfall alone (see
    `test_flooding_into_an_outfall_conduit_matches_swmm_where_swmm_converged`)."""
    rows, conn = _converged_swmm(tmp_path, name, surcharged=True)
    assert _flooded_steps(rows) == FLOODED_STEPS[name]
    _assert_parity(_worst_errors(name, rows, conn, len(rows), atol=EXTRAN_ATOL))


OUTFALL_FLOOD_NAME = "flooding_outfall_dynwave.inp"
#: The step (1-based) at which J4 floods in SWMM, and the last step at which SWMM's
#: unconverged C5 flow at that onset still shows above the targets (see the test).
J4_FLOOD_ONSET, J4_FLOOD_SETTLED = 236, 258
#: The quantities that only C5's flow sets: C5 itself, the outfall depth (set from C5's
#: flow) and J4's overflow (its net inflow, of which C5 is the outflow).
C5_QUANTITIES = {("flow", "C5"), ("depth", "Outfall"), ("overflow", "J4")}


@pytest.mark.slow
def test_flooding_into_an_outfall_conduit_matches_swmm_where_swmm_converged(tmp_path):
    """Every MaxDepth 0.5 m: J2 floods from step 21 and J4 from step 236, and J4's only
    outflow conduit C5 ends at the free outfall.

    Excluded (E9): C5's flow, the outfall's depth and J4's overflow on steps 236-258,
    where SWMM stops before C5's flow has converged. SWMM's stopping rule tests node depths
    only, and only those of non-outfall nodes (`dynwave.c:615-631`). A flooded junction is
    clamped at y_max on every trial, so it passes the test as soon as it is clamped twice
    in a row, and C5's flow then reaches no tested depth: its ends are J4 (clamped) and
    the outfall (untested). Reproducing SWMM's trials at step 236 from the step-235 state
    (which agrees with SWMM to 1e-10 m) shows J4 crossing its crown on trial 1, rising
    through EXTRAN's surcharged branch and clamping at y_max on trial 53. SWMM stops after
    trial 54, the first on which every node passes, with C5 at 0.21528729349 m3/s (SWMM
    reports 0.21528729360). C5's relaxed trials are then still halving their distance to
    the fixed point, 3.2e-5 m3/s, and they reach noodl's 0.21531978504 m3/s to 4e-16 by
    trial 91. That onset error is carried by the next steps' start flow, start area and
    outfall depth, and it decays by a factor ~0.67 a step. It falls below the targets after
    step 258 (J4's overflow is the last to go, at a ~3e-9 m3/s target). The test checks the
    cause directly: noodl's C5 flow at step 236 satisfies SWMM's conduit equation at the
    step's heads (outfall depth from that flow) to round-off, and SWMM's misses it by more
    than half their difference.

    Everything else (every other depth, flow and overflow at every step, and these three
    before 236 and after 258) is held to the free-surface targets, SWMM converging on
    every step by its own rule. The excluded values are held to the scale of the onset
    error. A fixture whose flooded junction drains into a tested junction has no such
    window (`test_flooding_matches_swmm_at_every_step`). Nor does the window follow from
    J4's flooding alone: with only J4's MaxDepth at 0.5 m, J4 floods at step 139 while
    the other junctions keep SWMM iterating until C5 is within 5.6e-9 m3/s of noodl."""
    rows, conn = _converged_swmm(tmp_path, OUTFALL_FLOOD_NAME, surcharged=True)
    flooded = _flooded_steps(rows)
    assert flooded["J4"][0] == J4_FLOOD_ONSET and set(flooded) == {"J2", "J4"}
    run = _run_noodl(OUTFALL_FLOOD_NAME, len(rows), split=J4_FLOOD_ONSET - 1,
                     atol=EXTRAN_ATOL)
    diffs = _differences(OUTFALL_FLOOD_NAME, rows, conn, len(rows), run=run)
    window = range(J4_FLOOD_ONSET - 1, J4_FLOOD_SETTLED)                   # 0-based rows

    def excluded(k, key):
        return k in window and key in C5_QUANTITIES

    _assert_parity(_worst(diffs, keep=lambda k, key: not excluded(k, key)))
    depth, flow, overflow = _worst(diffs, keep=excluded)
    assert depth <= 3e-5 and flow <= 4e-5 and overflow <= 4e-5, (depth, flow, overflow)

    # the cause: SWMM's C5 flow at the onset step is off its own conduit equation
    net, model, hist, start = run
    hyd = model.dynamic_wave
    k = J4_FLOOD_ONSET - 1
    c5 = [p.name for p in net.pipes].index("C5")
    sign = 1.0 if tuple(conn["C5"]) == (net.pipes[c5].u, net.pipes[c5].v) else -1.0

    def trial_flow(q):
        """SWMM's conduit flow (m3/s, `dwflow.c:57-262` before its relaxation) at step
        236's heads, with the previous trial's flow `q` (and the outfall depth from it,
        link.c:728-766), from the step-235 start flow and area (private hooks: the test
        needs SWMM's single trial map, which the pass applies only at a fixed point)."""
        phi = hist["water.phi"][k].clone()
        phi[hyd.outfall_idx] = hyd._outfall_heads(q)
        terms = hyd._terms({"water.phi": phi, "water.q": q,
                            "sewer.pipe_flow": start["sewer.pipe_flow"],
                            "sewer.mid_area": start["sewer.mid_area"]}, model.routing_step)
        flow = torch.where(terms.limited, torch.minimum(terms.momentum_flow,
                                                        terms.normal_flow),
                           terms.momentum_flow)
        return flow[c5].item() * QCF

    q_noodl = hist["water.q"][k].clone()
    q_swmm = q_noodl.clone()
    q_swmm[c5] = sign * rows[k][0]["C5"]
    gap = q_noodl[c5].item() - q_swmm[c5].item()
    assert abs(trial_flow(q_noodl) - q_noodl[c5].item()) <= 1e-12
    miss = trial_flow(q_swmm) - q_swmm[c5].item()
    assert gap > 1e-5 and miss > 0.5 * gap, (gap, miss)


# ---------------------------------------------------------------- variable routing step
VARIABLE_NAME = "tree_variable_dynwave.inp"
#: SWMM's first four steps on the variable-step tree (ms): MINIMUM_STEP, then the junction
#: depth-change limit (the filling junctions) and the Courant limit, up to ROUTING_STEP.
VARIABLE_FIRST_STEPS_MS = [500, 13236, 31624, 51592, 60000]
#: A ramp of J1's inflow (m3/s) from 0.025 to 0.06 between 600 s and 660 s, as a SWMM
#: [TIMESERIES] (linearly interpolated between its points) and as the callable noodl takes.
J1_RAMP = ((0.0, 0.025), (600.0, 0.025), (660.0, 0.06), (3600.0, 0.06))


def _j1_ramp(t):
    t += 0.001    # SWMM's date for a step is its start plus 1 ms (swmm5.c:1551)
    (t0, q0), (t1, q1) = next((a, b) for a, b in zip(J1_RAMP, J1_RAMP[1:], strict=False)
                              if a[0] <= t <= b[0])
    return q0 + (q1 - q0) * (t - t0) / (t1 - t0)


def _ramp_inflow(net, hyd):
    """noodl's inflow callable for the ramp: the fixture's baseline inflows with J1's
    replaced by `_j1_ramp`."""
    base = torch.zeros(len(hyd.node_invert), dtype=torch.float64)
    names = list(hyd.net.nodes)
    for m in net.manholes:
        base[names.index(m.name)] = m.inflow
    j1 = names.index("J1")

    def inflow(t):
        q = base.clone()
        q[j1] = _j1_ramp(t)
        return q
    return inflow


def _ramp_inp(tmp_path):
    """The variable-step tree with J1's inflow from a [TIMESERIES] (`J1_RAMP`)."""
    text = (DATA / VARIABLE_NAME).read_text()
    text = text.replace('J1               FLOW             ""               FLOW     1.0   '
                        '   1.0      0.025',
                        'J1               FLOW             J1ramp           FLOW     1.0   '
                        '   1.0      0')
    assert "J1ramp" in text
    points = "".join(f"J1ramp           {int(t) // 3600}:{int(t) % 3600 // 60:02d}:"
                     f"{int(t) % 60:02d} {q}\n" for t, q in J1_RAMP)
    text = text.replace("[REPORT]", f"[TIMESERIES]\n{points}\n[REPORT]")
    path = tmp_path / "tree_variable_ramp.inp"
    path.write_text(text)
    return path


def _swmm_timed(inp):
    """SWMM on `inp` stepped through its own `swmm_step`: rows as `_swmm` (flows in the
    file's orientation, node depths, node flooding), the elapsed time after each step in
    ms (from the toolkit's exact elapsed days: pyswmm's `current_time` is whole seconds),
    the connections, the engine version and the report's percentage of non-converging
    steps. As `_swmm`, no row follows the last step, which `swmm_step` ends with 0."""
    rpt = inp.with_suffix(".rpt")
    rows, times = [], []
    with pyswmm.Simulation(str(inp), str(rpt), str(inp.with_suffix(".out"))) as sim:
        conn = {link.linkid: link.connections for link in pyswmm.Links(sim)}
        sim.start()
        while (elapsed := sim._model.swmm_step()) > 0:
            times.append(round(elapsed * 86400.0 * 1000.0))
            nodes = list(pyswmm.Nodes(sim))
            rows.append((
                {link.linkid: link.flow for link in pyswmm.Links(sim)},
                {node.nodeid: node.depth for node in nodes},
                {node.nodeid: node.flooding for node in nodes},
            ))
        engine = str(sim.engine_version)
    m = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", rpt.read_text())
    return rows, times, conn, engine, (float(m.group(1)) if m else None)


def _variable_run(inp, steps_limit=None, inflow=None):
    """noodl on `inp` for its whole duration (SWMM's 30 min), the variable step and its
    inflow callable if any; `(net, model, hist, None)` as `_run_noodl`. With
    `steps_limit` the run stops after that many steps (`hist` as `_run_noodl`'s)."""
    net, _, _ = read_swmm_inp(inp)
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    record = ("water.phi", "water.q", "sewer.overflow")
    if inflow is not None:
        inflow = inflow(net, model.dynamic_wave)
    kwargs = {"steps": steps_limit} if steps_limit else {"duration": 1800.0}
    _, hist = simulate_dynamic_wave(model, state, drivers, record=record, inflow=inflow,
                                    **kwargs)
    return net, model, hist, None


def test_variable_step_sequence_and_state_match_swmm_over_the_first_steps(tmp_path):
    """The first five steps of the variable-step tree (`VARIABLE_STEP 0.75`, `ROUTING_STEP`
    60 s: the minimum step, two depth-change-limited steps, one Courant-limited and the
    routing step) in SWMM and noodl: the step sequence to the millisecond, and depths,
    flows and overflow at the free-surface targets (the fast case). pyswmm's `current_time`
    is whole seconds, so the exact elapsed time is read from `swmm_step`."""
    inp = tmp_path / VARIABLE_NAME
    shutil.copy(DATA / VARIABLE_NAME, inp)
    rows, times, conn, engine, not_converging = _swmm_timed(inp)
    assert engine == "5.2.4" and not_converging == 0.0
    run = _variable_run(inp, steps_limit=5)
    _, _, hist, _ = run
    assert [round(x) for x in (hist["t"] * 1000.0).tolist()] == times[:5]
    assert [b - a for a, b in zip([0, *times[:5]], times[:5], strict=False)] == (
        VARIABLE_FIRST_STEPS_MS)
    _assert_parity(_worst(_differences(VARIABLE_NAME, rows, conn, 5, run=run)))


@pytest.mark.slow
def test_variable_step_parity_at_every_step(tmp_path):
    """The variable-step tree over its 30 min: the step sequence equals SWMM's to the
    millisecond at every step (the last, clipped step, which `swmm_step` does not
    report, ends the run at 1800 s), and depths, flows and overflow agree at every step
    at the free-surface targets, SWMM converging on every step."""
    inp = tmp_path / VARIABLE_NAME
    shutil.copy(DATA / VARIABLE_NAME, inp)
    rows, times, conn, engine, not_converging = _swmm_timed(inp)
    assert engine == "5.2.4" and not_converging == 0.0
    run = _variable_run(inp)
    _, _, hist, _ = run
    ends = [round(x) for x in (hist["t"] * 1000.0).tolist()]
    assert ends[:len(times)] == times and ends[-1] == 1_800_000
    assert len(ends) == len(times) + 1
    _assert_parity(_worst(_differences(VARIABLE_NAME, rows, conn, len(rows), run=run)))


@pytest.mark.slow
def test_time_varying_inflow_matches_swmm_with_a_timeseries(tmp_path):
    """J1's inflow ramps from 0.025 to 0.06 m3/s between 600 and 660 s: SWMM reads it from
    a [TIMESERIES] and noodl from a callable of the same piecewise-linear function. SWMM
    evaluates it at each step's START (routing.c:223) plus 1 ms (`getDateTime`, swmm5.c:1551;
    `_j1_ramp` adds it), so the match to the millisecond of
    the step sequence (which the ramp changes: later steps are no longer 60 s) and to
    the free-surface targets in depths and flows at every step also shows the callable is
    called with the start time."""
    inp = _ramp_inp(tmp_path)
    rows, times, conn, engine, not_converging = _swmm_timed(inp)
    assert engine == "5.2.4" and not_converging == 0.0
    run = _variable_run(DATA / VARIABLE_NAME, inflow=_ramp_inflow)
    _, _, hist, _ = run
    ends = [round(x) for x in (hist["t"] * 1000.0).tolist()]
    assert ends[:len(times)] == times
    assert any(t != 60000 for t in
               [b - a for a, b in zip(times[20:], times[21:], strict=False)])
    _assert_parity(_worst(_differences(VARIABLE_NAME, rows, conn, len(rows), run=run)))
