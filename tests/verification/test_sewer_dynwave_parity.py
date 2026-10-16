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

import dataclasses
import re
import shutil
from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave
from noodl.apps.sewer.xsect_tables import QCF
from noodl.contracts import StepContext

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
    link)`` (every conduit and regulator, `model.link_names`) and ``("overflow",
    junction)`` to the excess (m3/s) of the flow or overflow difference over the flow
    target (<= 0 passes)."""
    net, model, hist, _ = run or _run_noodl(name, steps, **solve_kwargs)
    hyd = model.dynamic_wave
    # junctions from the solved heads; outfalls from the closure-written boundary heads
    junctions = list(hyd.junction_names)
    nodes = junctions + list(hyd.outfall_names)
    idx = torch.cat([model.junction_idx, hyd.outfall_idx])
    invert = hyd.node_invert[idx]
    # every link: the pipes, then the orifices, weirs and outlets (never reversed)
    names = list(model.net.nodes)
    ends = {name: (names[int(u)], names[int(v)]) for name, u, v in
            zip(model.link_names, hyd.edge_u, hyd.edge_v, strict=True)}
    sign = {name: (1.0 if tuple(conn[name]) == pair else -1.0) for name, pair in ends.items()}

    def excess(noodl, swmm):
        return abs(noodl - swmm) - (TOL_FLOW_REL * abs(swmm) + TOL_FLOW_ABS)

    out = []
    for k, (flows, depths, flooding) in enumerate(rows[:steps]):
        y = hist["water.phi"][k, idx] - invert
        diff = {("depth", n): abs(y[j].item() - depths[n]) for j, n in enumerate(nodes)}
        for i, name in enumerate(model.link_names):
            diff["flow", name] = excess(hist["water.q"][k, i].item(), sign[name] * flows[name])
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
    free-surface, or they would test something else); with it some end must; with
    ``surcharged=None`` the regime is not checked."""
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
                assert surcharged is not False or fill < 1.0, (
                    f"{name}: SWMM surcharges conduit {p.name} at {node} at step {k + 1} "
                    f"(end depth {fill:.3f} of its diameter); reduce the fixture's inflows")
    if surcharged is not None:
        assert reached == surcharged, (
            f"{name}: surcharge expected={surcharged}, reached={reached}")
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

    Compared steps (SWMM converged): SWMM converges on every step up to 144, which covers the fill,
    the whole ramp and the drain until J4 meets the outfall's depth; those are asserted at the
    free-surface targets. From step 145 on SWMM fails to converge on 82 of the remaining 108 steps
    (found by ending the run after each step; its report says 32.5 % of all 252), as C5 (J4 -> free
    outfall) reaches the switch of its normal-flow slope test: its limited flow is Manning's normal
    flow at J4's depth, at which the outfall's own depth (set from that flow) is J4's depth again,
    up to the ~1e-5 m interpolation mismatch of SWMM's tables, so neither side of the test need be
    consistent (docs/applications/sewer.md, "Dynamic wave": the normal-flow slope switch). SWMM's
    last trial is then wherever its alternating trials stopped, J4 within ~6e-4 m of the outfall
    depth on either side, and later steps start from that state, so no later step is a fixed-point
    comparison: steps 145 onwards are excluded from the targets. They are asserted only to the scale
    of SWMM's own oscillation (1e-3 m, 5e-4 m3/s), with noodl converging on every one of them,
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


DIVIDER_NAME = "divider_dynwave.inp"


def test_a_divider_is_a_junction_under_dynamic_wave_routing(tmp_path):
    """`flooding_dynwave.inp` with J3 a WEIR flow divider of C3 (MaxDepth 0.6 m after its
    qMin, dhMax and cWeir): SWMM routes it as a plain junction under DYNWAVE (the diversion
    is `link_getInflow`'s, which dynwave.c does not use for a conduit), so J2's flood onset
    and release agree with SWMM exactly as in the junction fixture. J3 floods only from step
    270 (`test_a_divider_floods_like_a_junction_at_every_step`)."""
    rows, conn = _converged_swmm(tmp_path, DIVIDER_NAME, surcharged=True)
    _assert_parity(_worst_errors(DIVIDER_NAME, rows, conn, FLOOD_FAST_STEPS, atol=EXTRAN_ATOL))


@pytest.mark.slow
def test_a_divider_floods_like_a_junction_at_every_step(tmp_path):
    """The divider fixture over all 359 steps: its MaxDepth (the first field after the weir
    parameters) is the flood depth, J3 floods from step 270 as the junction does, and
    depths, flows and ``sewer.overflow`` agree with SWMM at the free-surface targets."""
    rows, conn = _converged_swmm(tmp_path, DIVIDER_NAME, surcharged=True)
    assert _flooded_steps(rows) == FLOODED_STEPS[FLOOD_NAME]
    _assert_parity(_worst_errors(DIVIDER_NAME, rows, conn, len(rows), atol=EXTRAN_ATOL))


# --------------------------------------------------------------------------------- ponding
POND_NAME = "ponding_dynwave.inp"
POND_TIE_NAME = "ponding_surcharge_dynwave.inp"
#: SWMM's flooded steps (1-based) of the ponding fixtures: SWMM's ``Node.flooding`` of a
#: ponding junction is the rate its ponded volume grows (`dynwave.c:787-792`), positive while
#: it ends above its full depth and its net inflow is above the FUDGE flow.
POND_FLOODED_STEPS = {POND_NAME: {"J2": list(range(21, 360))},
                      POND_TIE_NAME: {"J2": list(range(21, 28)), "J3": list(range(271, 360))}}
#: The first step (1-based) of the tie in `POND_TIE_NAME`, see its test.
POND_TIE_STEP = 270


def test_ponding_matches_swmm_through_the_first_ponded_steps(tmp_path):
    """`ALLOW_PONDING YES` with J2 (MaxDepth 0.3 m, its crown, Aponded 0.5 m2): J2 ponds from
    step 21 instead of flooding and its depth keeps rising above 0.3 m. Depths, flows and
    ``sewer.overflow`` (the ponded volume's growth rate) against SWMM over the first 60 steps
    at the free-surface targets, SWMM converging on every step."""
    rows, conn = _converged_swmm(tmp_path, POND_NAME, surcharged=None)
    assert _flooded_steps(rows[:60]) == {"J2": POND_FLOODED_STEPS[POND_NAME]["J2"][:40]}
    _assert_parity(_worst_errors(POND_NAME, rows, conn, 60, atol=EXTRAN_ATOL))


@pytest.mark.slow
def test_ponding_matches_swmm_at_every_step(tmp_path):
    """The same fixture over all 359 steps: J2 ponds on up to the end of the run, over the
    0.5 m2 ponded area plus its conduits' (the depth is far above its rim)."""
    rows, conn = _converged_swmm(tmp_path, POND_NAME, surcharged=None)
    assert _flooded_steps(rows) == POND_FLOODED_STEPS[POND_NAME]
    _assert_parity(_worst_errors(POND_NAME, rows, conn, len(rows), atol=EXTRAN_ATOL))


#: The ponding drain: the fixture's inflows held 60 steps (J2 ponded to 0.56 m), ramped to
#: nothing over 12 steps and off for 28, so that J2 drains back through its full depth.
POND_RAMP = [1.0] * 60 + [1.0 - (k + 1) / 12 for k in range(12)] + [0.0] * 28


@pytest.mark.slow
def test_a_ponded_junction_drains_back_through_its_full_depth_as_swmm(tmp_path):
    """J2 rises to 0.56 m, then drains below its 0.3 m rim (steps 61-100): pyswmm's
    `generated_inflow` ramp (`_swmm_scripted`) against noodl's per-step inflow series. The
    falling case is the one where the ponded line, not the conduits' area, sets the depth
    above the rim; SWMM converges on every step of the 99 rows, and depths and flows agree at
    the free-surface targets."""
    steps = len(POND_RAMP)
    rows, conn, not_converging = _swmm_scripted(tmp_path, POND_NAME, POND_RAMP, steps)
    assert not_converging == 0.0 and len(rows) == steps - 1
    assert max(d["J2"] for _, d in rows) > 0.5 and rows[-1][1]["J2"] < 0.3

    net, _, _ = read_swmm_inp(DATA / POND_NAME)
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    inflow = torch.stack([f * drivers["inflow"] for f in POND_RAMP]).unsqueeze(1)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=steps, inflow=inflow,
                                    atol=EXTRAN_ATOL)
    hyd = model.dynamic_wave
    nodes = [m.name for m in net.manholes] + list(hyd.outfall_names)
    idx = torch.cat([model.junction_idx, hyd.outfall_idx])
    sign = {p.name: (1.0 if tuple(conn[p.name]) == (p.u, p.v) else -1.0) for p in net.pipes}
    for k, (flows, depths) in enumerate(rows):
        y = hist["water.phi"][k, 0, idx] - hyd.node_invert[idx]
        for j, node in enumerate(nodes):
            assert abs(y[j].item() - depths[node]) <= TOL_DEPTH, (k + 1, node)
        for i, pipe in enumerate(net.pipes):
            q_s = sign[pipe.name] * flows[pipe.name]
            q_n = hist["water.q"][k, 0, i].item()
            assert abs(q_n - q_s) <= TOL_FLOW_REL * abs(q_s) + TOL_FLOW_ABS, (k + 1, pipe.name)


@pytest.mark.slow
def test_ponding_after_a_surcharged_junction_reaches_its_full_depth_holds_swmms_tie(tmp_path):
    """J3 (crown 0.45 m, MaxDepth 0.6 m, Aponded 0.5 m2) is surcharged: at step 270 the
    surcharged update takes it above its full depth, the ponded update ends below it, and
    SWMM's trials alternate about 0.6 m (a FUDGE above and below), stopping at 0.5999695 m
    with SWMM reporting that step as not converged (0.28 % of the 359). noodl's tie rule holds
    the junction at 0.6 m exactly. Every step before the tie agrees at the full targets; from it
    the difference is the FUDGE depth (3.05e-5 m, `FUDGE` = 1e-4 ft) and what that offset does
    to the conduits' flows, held to 1e-4 m and 1e-5 m3/s, while J3's overflow (the ponded
    volume's growth) agrees to 1e-5 m3/s."""
    rows, conn, engine, not_converging = _swmm(tmp_path, POND_TIE_NAME)
    assert engine == "5.2.4" and not_converging == pytest.approx(100.0 / 359, abs=0.01)
    assert _flooded_steps(rows) == POND_FLOODED_STEPS[POND_TIE_NAME]
    diffs = _differences(POND_TIE_NAME, rows, conn, len(rows), atol=EXTRAN_ATOL)
    _assert_parity(_worst(diffs[:POND_TIE_STEP - 1]))
    depth, flow, overflow = _worst(diffs[POND_TIE_STEP - 1:])
    assert 1e-5 < depth <= 1e-4 and flow <= 1e-5 and overflow <= 1e-5, (depth, flow, overflow)


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

    Excluded (SWMM unconverged): C5's flow, the outfall's depth and J4's overflow on steps 236-258,
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


def test_routing_step_above_the_wet_step_is_lowered_as_swmm_does(tmp_path):
    """`ROUTING_STEP 120` with `WET_STEP 00:01:00`: SWMM runs at 60 s (project.c:246-250) and
    so does the reader, so the first five steps equal the 60 s fixture's, to the ms."""
    text = (DATA / VARIABLE_NAME).read_text().replace("ROUTING_STEP         60",
                                                      "ROUTING_STEP         120")
    assert "ROUTING_STEP         120" in text
    inp = tmp_path / "wet_clip.inp"
    inp.write_text(text)
    assert read_swmm_inp(inp)[0].dynamic_wave.routing_step == 60.0
    rows, times, _, engine, not_converging = _swmm_timed(inp)
    assert engine == "5.2.4" and not_converging == 0.0
    _, _, hist, _ = _variable_run(inp, steps_limit=len(times))
    assert [round(x) for x in (hist["t"] * 1000.0).tolist()] == times
    assert [b - a for a, b in zip([0, *times[:5]], times[:5], strict=False)] == (
        VARIABLE_FIRST_STEPS_MS)


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


# ----------------------------------------------------------------------------- options (C2)
OPTIONS_NAME = "options_dynwave.inp"
#: The fast comparison: the first 60 steps (5 min) of the options fixture, which cover C1's
#: MaxFlow cap, C5 closed by its gated outfall and the start of C4's reverse flow.
OPTIONS_FAST_STEPS = 60


def _options_evidence(rows):
    """What SWMM's own rows say about the options fixture, so that a match is not trivial:
    the options must act in the reference. Returns the 1-based steps at which each option
    acts (empty lists are asserted against by the callers)."""
    acts = {"max_flow": [], "outfall_gate": [], "conduit_gate": [], "init_flow": [],
            "init_depth": [], "reversed_gate": []}
    for k, (flows, depths, _) in enumerate(rows):
        step = k + 1
        if abs(abs(flows["C1"]) - 0.012) < 1e-9:
            acts["max_flow"].append(step)
        # J4's head against the FIXED stage: a gated outfall passes nothing in, so C5 is 0
        # exactly until J4 rises over 9.3 m (and can flow only forward afterwards)
        if 9.0 + depths["J4"] < 9.3 and flows["C5"] == 0.0:
            acts["outfall_gate"].append(step)
        # C4's reverse flow (J4 higher than J5) is closed by its gate: exactly 0, not tiny
        if 9.0 + depths["J4"] > 9.1 + depths["J5"] + 1e-3 and flows["C4"] == 0.0:
            acts["conduit_gate"].append(step)
        # C6 is adverse (J5 9.1 m under J3 10.0 m) and reversed; its gate passes only the
        # file's J5 -> J3 direction, so the downhill J3 -> J5 flow is closed: exactly 0
        if 10.0 + depths["J3"] > 9.1 + depths["J5"] + 1e-3 and flows["C6"] == 0.0:
            acts["reversed_gate"].append(step)
    flows, depths, _ = rows[0]
    if flows["C3"] > 0.0:
        acts["init_flow"].append(1)             # C3 starts from its InitFlow, not rest
    if depths["J1"] > 0.2:
        acts["init_depth"].append(1)            # J1 starts at 0.25 m
    return acts


def test_options_match_swmm_over_the_first_minutes(tmp_path):
    """InitDepth (J1 0.25 m), InitFlow (C3 0.03 m3/s; C6 0.003 m3/s, adverse and reversed),
    MaxFlow (C1 0.012), two barrels (C3), a 0.05 m outlet offset (C1), flap gates (C4, and
    the reversed C6), a gated FIXED outfall (stage 9.3 m, above J4's invert) and conduit
    lengthening (C2, 10 m, `LENGTHENING_STEP 5`) together, in SWMM and noodl: depths, flows
    and overflow at the depth and flow targets of the other comparisons, SWMM converging on
    every step (the fast case). The fixture is not free-surface throughout: the stage puts
    C5's outfall end above its 0.45 m crown. The reference rows show that each option acts."""
    rows, conn = _converged_swmm(tmp_path, OPTIONS_NAME, surcharged=None)
    acts = _options_evidence(rows[:OPTIONS_FAST_STEPS])
    for option in ("max_flow", "outfall_gate", "init_flow", "init_depth", "reversed_gate"):
        assert acts[option], f"SWMM's rows show no {option} acting in the first minutes"
    _assert_parity(_worst_errors(OPTIONS_NAME, rows, conn, OPTIONS_FAST_STEPS))


@pytest.mark.slow
def test_options_match_swmm_at_every_step(tmp_path):
    """All 359 steps of the options fixture at the depth and flow targets (not free-surface
    throughout: C5's outfall end is above its crown); the flap gates on C4 and on the reversed
    C6 close reverse flow in the reference (water backs up from J4 and J3)."""
    rows, conn = _converged_swmm(tmp_path, OPTIONS_NAME, surcharged=None)
    acts = _options_evidence(rows)
    assert all(acts.values()), acts
    _assert_parity(_worst_errors(OPTIONS_NAME, rows, conn, len(rows)))


def _ablated(net, option):
    """The options network with one option switched off."""
    def pipe(name, **kw):
        return dataclasses.replace(net, pipes=tuple(
            dataclasses.replace(p, **kw) if p.name == name else p for p in net.pipes))

    if option == "init_depth":
        return dataclasses.replace(net, manholes=tuple(
            dataclasses.replace(m, initial_depth=0.0) for m in net.manholes))
    if option == "init_flow":
        return pipe("C3", initial_flow=0.0)
    if option == "max_flow":
        return pipe("C1", max_flow=None)
    if option == "conduit_gate":
        return pipe("C4", flap_gate=False)
    if option == "outfall_gate":
        return dataclasses.replace(net, outfalls=tuple(
            dataclasses.replace(o, flap_gate=False) for o in net.outfalls))
    if option == "c6_init_flow":
        return pipe("C6", initial_flow=0.0)
    if option == "outlet_offset":
        return pipe("C1", outlet_offset=0.0)
    if option == "barrels":
        return pipe("C3", barrels=1)
    if option == "lengthening":
        return dataclasses.replace(net, dynamic_wave=dataclasses.replace(
            net.dynamic_wave, lengthening_step=0.0))
    raise ValueError(option)


#: Steps over which each option is switched off in `test_each_option_...`: by step 150
#: every option has moved the state (the conduit gate's backwater first shows at ~step 100).
ABLATION_STEPS = 150


@pytest.mark.slow
def test_each_option_changes_the_options_fixture_beyond_the_targets(tmp_path):
    """The comparison above is not trivial: with any one option switched off in noodl alone,
    the first `ABLATION_STEPS` steps depart from SWMM's by far more than the targets, so each
    option is exercised by the reference and by noodl's port of it. (Over all 359 steps a run
    without the gates is several times slower to converge, which is why the window is
    short.) SWMM is run once. The reversed gate on C6 is shown to act by SWMM's own rows
    (`_options_evidence`), not by an ablation: with the gate switched off noodl alone does
    not converge from step 106, where the ungated adverse conduit's drop outlet chatters
    (not investigated: that run has no SWMM reference here)."""
    rows, conn = _converged_swmm(tmp_path, OPTIONS_NAME, surcharged=None)
    net, _, _ = read_swmm_inp(DATA / OPTIONS_NAME)
    record = ("water.phi", "water.q", "sewer.overflow")
    missed = {}
    for option in ("init_depth", "init_flow", "max_flow", "conduit_gate", "outfall_gate",
                   "barrels", "lengthening", "c6_init_flow", "outlet_offset"):
        model, state, drivers = build_model(_ablated(net, option), iterate_tol=ITERATE_TOL,
                                            iterate_max=ITERATE_MAX)
        _, hist = simulate_dynamic_wave(model, state, drivers, steps=ABLATION_STEPS,
                                        record=record)
        diffs = _differences(OPTIONS_NAME, rows, conn, ABLATION_STEPS,
                             run=(net, model, hist, None))
        worst_depth, worst_flow, _ = _worst(diffs)
        if not (worst_depth > 100 * TOL_DEPTH or worst_flow > 1e-5):
            missed[option] = (worst_depth, worst_flow)
    assert not missed, f"options with no visible effect against SWMM: {missed}"


def _options_variable_inp(tmp_path):
    """The options fixture under SWMM's variable step: ROUTING_STEP 30 s, VARIABLE_STEP 0.75
    (the Courant step uses C3's per-barrel flow and volume and C2's modLength / length)."""
    text = (DATA / OPTIONS_NAME).read_text()
    for old, new in (("ROUTING_STEP         5", "ROUTING_STEP         30"),
                     ("VARIABLE_STEP        0", "VARIABLE_STEP        0.75")):
        assert text.count(old) == 1
        text = text.replace(old, new)
    path = tmp_path / "options_variable.inp"
    path.write_text(text)
    return path


def test_options_variable_step_sequence_matches_swmm(tmp_path):
    """SWMM's variable routing step on the options fixture (two barrels, a lengthened
    conduit, initial conditions): noodl's step sequence equals SWMM's to the millisecond over
    the first 12 steps, and the state at those steps meets the depth and flow targets."""
    inp = _options_variable_inp(tmp_path)
    rows, times, conn, engine, not_converging = _swmm_timed(inp)
    assert engine == "5.2.4" and not_converging == 0.0
    n = 12
    run = _variable_run(inp, steps_limit=n)
    _, _, hist, _ = run
    ends = [round(x) for x in (hist["t"] * 1000.0).tolist()]
    assert ends == times[:n]
    assert len(set(b - a for a, b in zip(times[:n], times[1:n], strict=False))) > 2
    _assert_parity(_worst(_differences(OPTIONS_NAME, rows, conn, n, run=run)))


@pytest.mark.slow
def test_options_variable_step_matches_swmm_at_every_step(tmp_path):
    inp = _options_variable_inp(tmp_path)
    rows, times, conn, engine, not_converging = _swmm_timed(inp)
    assert engine == "5.2.4" and not_converging == 0.0
    run = _variable_run(inp)
    _, _, hist, _ = run
    ends = [round(x) for x in (hist["t"] * 1000.0).tolist()]
    assert ends[:len(times)] == times and ends[-1] == 1_800_000
    _assert_parity(_worst(_differences(OPTIONS_NAME, rows, conn, len(rows), run=run)))


# ------------------------------------------------------ the drop-end area switch (C2b)
#: Variants of the options fixture on which a junction below an outlet offset fills
#: through the offset end's area threshold (`DynamicWaveHydraulics`, the drop-end area
#: switch; tests/apps/sewer/test_dynwave.py::DROP_VARIANTS holds the same edits with the
#: steps noodl's own run holds): the fixture line and its replacement.
DROP_VARIANTS = {
    "c3_offset_0.03": ("C3               J3               J4               200        0.013"
                       "      0          0  ",
                       "C3               J3               J4               200        0.013"
                       "      0          0.03"),
    "c3_offset_0.1": ("C3               J3               J4               200        0.013"
                      "      0          0  ",
                      "C3               J3               J4               200        0.013"
                      "      0          0.1"),
    "c6_gate_off": ("C6               0          0          0          YES        0",
                    "C6               0          0          0          NO         0"),
}
#: SWMM 5.2.4's non-converging steps (1-based) over the 359 steps of its own run of each
#: variant (HEAD_TOLERANCE 1e-12 m, 500 trials), each a step on which a junction's storage
#: equation has no fixed point. Step 167 of the 0.03 m variant lies in SWMM's own history:
#: noodl's run differs from it after step 60 and does not reach that threshold there.
DROP_SWMM_FAILS = {"c3_offset_0.03": [60, 62, 167], "c3_offset_0.1": [107],
                   "c6_gate_off": [106, 129]}
#: The scale of SWMM's own oscillation on its non-converging steps of these variants (m,
#: m3/s): its trials cycle between the two sides of the threshold and its last trial is
#: wherever they stopped (on step 167 of the 0.03 m variant, emulating dynwave.c's trials
#: at J5 gives a 5-cycle over 1.4e-5 m that never meets the tolerance). Measured against
#: noodl's held solution: 3.1e-4 m (J4, step 62 of the 0.03 m variant) and 2.5e-5 m3/s
#: (C3, step 107 of the 0.1 m one).
DROP_FAIL_DEPTH, DROP_FAIL_FLOW = 1e-3, 5e-5


def _drop_inp(tmp_path, name):
    old, new = DROP_VARIANTS[name]
    text = (DATA / OPTIONS_NAME).read_text()
    assert text.count(old) == 1, name
    path = tmp_path / f"{name}.inp"
    path.write_text(text.replace(old, new))
    return path


def _swmm_rows(inp):
    """SWMM on the file `inp`: rows of (flows, depths, flooding) as `_swmm`, the
    connections, the engine version and the report's percentage of steps not converging."""
    rpt = inp.with_suffix(".rpt")
    rows = []
    with pyswmm.Simulation(str(inp), str(rpt), str(inp.with_suffix(".out"))) as sim:
        conn = {link.linkid: link.connections for link in pyswmm.Links(sim)}
        for _ in sim:
            nodes = list(pyswmm.Nodes(sim))
            rows.append(({link.linkid: link.flow for link in pyswmm.Links(sim)},
                         {node.nodeid: node.depth for node in nodes},
                         {node.nodeid: node.flooding for node in nodes}))
        engine = str(sim.engine_version)
    m = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", rpt.read_text())
    return rows, conn, engine, (float(m.group(1)) if m else None)


def _swmm_not_converged_within(inp, n):
    """How many of SWMM's first `n` routing steps of `inp` (5 s each) do not converge.
    SWMM floors the run's duration (END_TIME 5n s runs n - 1 full steps and a 4 s one), so
    the run ends at 5n + e s instead: n full steps and a short tail. The count is the
    least over tails e = 1, 2, 3 s (a tail that fails only adds one), from the report's
    percentage over the n + 1 steps."""
    counts = []
    for extra in (1, 2, 3):
        secs = 5 * n + extra
        text = re.sub(r"END_TIME\s+\S+", f"END_TIME             {secs // 3600:02d}:"
                      f"{secs % 3600 // 60:02d}:{secs % 60:02d}", inp.read_text())
        cut = inp.with_name(f"{inp.stem}_cut_{n}_{extra}.inp")
        cut.write_text(text)
        rows, _, _, percent = _swmm_rows(cut)
        rpt = cut.with_suffix(".rpt").read_text()
        shortest = float(re.search(r"Minimum Time Step\s*:\s*([0-9.]+)", rpt).group(1))
        if len(rows) == n and shortest < 5.0:          # n full steps, then the tail
            counts.append(round(percent * (n + 1) / 100.0))
    assert counts, f"no END_TIME gave {n} full steps"
    return min(counts)


def _swmm_failing_steps(inp, steps):
    """SWMM's non-converging steps (1-based) among the first `steps`, by bisection on
    `_swmm_not_converged_within`."""
    cache = {0: 0}

    def count(n):
        if n not in cache:
            cache[n] = _swmm_not_converged_within(inp, n)
        return cache[n]

    found = []

    def search(lo, hi):
        if count(hi) == count(lo):
            return
        if hi - lo == 1:
            found.append(hi)
            return
        mid = (lo + hi) // 2
        search(lo, mid)
        search(mid, hi)

    search(0, steps)
    return sorted(found)


def _restart_differences(inp, rows, conn, steps):
    """noodl's step k (each k in `steps`, 1-based, k >= 2) started from SWMM's own state
    after step k - 1, all as one batch: the junction and outfall heads from SWMM's depths,
    the conduit flows, and the carried keys evaluated there (`carried`: SWMM's stored mid
    area, from the end depths and the flow class, and its net inflow). Returns whether
    every instance converged and, per k, the largest depth difference (m) and flow excess
    over the flow target (m3/s) against SWMM after step k. Whatever path SWMM took to its
    state, this asks only whether one step from it lands where SWMM's does: on a step
    where SWMM converged from a converged start, both are the fixed point of the same
    equations."""
    net, _, _ = read_swmm_inp(inp)
    model, _, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    hyd = model.dynamic_wave
    nodes = list(model.net.nodes)
    sign = torch.tensor([1.0 if tuple(conn[p.name]) == (p.u, p.v) else -1.0
                         for p in net.pipes], dtype=torch.float64)

    def heads_flows(k):                          # SWMM after step k (1-based)
        flows, depths, _ = rows[k - 1]
        phi = hyd.node_invert + torch.tensor([depths[n] for n in nodes], dtype=torch.float64)
        q = sign * torch.tensor([flows[p.name] for p in net.pipes], dtype=torch.float64)
        return phi, q

    starts = [heads_flows(k - 1) for k in steps]
    start = {"water.phi": torch.stack([s[0] for s in starts]),
             "water.q": torch.stack([s[1] for s in starts])}
    # `carried` reads the step-start flow and area only for the momentum terms it discards
    start["sewer.pipe_flow"] = start["water.q"]
    start["sewer.mid_area"] = torch.ones_like(start["water.q"])
    start.update(hyd.carried(start, drivers))
    diags: list = []
    out, _ = simulate_dynamic_wave(model, start, drivers, steps=1,
                                   inflow=drivers["inflow"].expand(len(steps), -1),
                                   diagnostics=diags)
    idx = torch.cat([model.junction_idx, hyd.outfall_idx])
    # the drop-end switch's drivers of a pass fed each instance's result (`_held`)
    drv = hyd({"water.phi": out["water.phi"], "water.q": out["water.q"]}
              | {key: start[key] for key in hyd.state_keys},
              dict(drivers, inflow=drivers["inflow"].expand(len(steps), -1)),
              StepContext(dt=hyd.options.routing_step))
    result, held = {}, {}
    for b, k in enumerate(steps):
        phi, q = heads_flows(k)
        depth = (out["water.phi"][b] - phi)[idx].abs().max().item()
        excess = ((out["water.q"][b] - q).abs()
                  - (TOL_FLOW_REL * q.abs() + TOL_FLOW_ABS)).max().item()
        result[k] = (depth, excess)
        held[k] = _held(hyd, out["water.phi"][b][model.junction_idx],
                        {key: value[b] if value.dim() > 1 else value for key, value in drv.items()})
    return bool(diags[0]["converged"].all()), result, held


def _held(hyd, heads, drv):
    """The junctions held on a drop-end area threshold by one solution's drivers `drv`
    (those of a pass fed it): ``{junction: H - H*}`` (m) where the net inflow lies inside
    the jump at the threshold head H*."""
    h_star = drv["water.area_switch_head"]
    rate2 = 2.0 * drv["water.storage_rate"]
    rise = h_star - drv["water.head_start"]
    lo = rate2 * drv["water.area_below"] * rise - drv["water.net_inflow_start"]
    hi = rate2 * drv["water.area_above"] * rise - drv["water.net_inflow_start"]
    n = drv["water.net_inflow_last"]
    return {name: (heads[j] - h_star[j]).item()
            for j, name in enumerate(hyd.junction_names)
            if bool(torch.isfinite(h_star[j])) and lo[j] < n[j] < hi[j]}


def _check_restarts(result, fails):
    """The targets on every step SWMM converged from a converged start; SWMM's oscillation
    scale on its failing steps and on the step after each (whose start is SWMM's last,
    unconverged trial)."""
    worst = {"compared": [0.0, -TOL_FLOW_ABS], "excluded": [0.0, 0.0]}
    for k, (depth, excess) in result.items():
        group = "excluded" if (k in fails or k - 1 in fails) else "compared"
        worst[group][0] = max(worst[group][0], depth)
        worst[group][1] = max(worst[group][1], excess)
    assert worst["compared"][0] <= TOL_DEPTH and worst["compared"][1] <= 0.0, worst
    assert worst["excluded"][0] <= DROP_FAIL_DEPTH, worst
    assert worst["excluded"][1] <= DROP_FAIL_FLOW, worst


def test_drop_switch_matches_swmm_one_step_from_its_state_around_its_failure(tmp_path):
    """The fast case of the drop-end area switch: C3 with a 0.1 m outlet offset, steps
    100-115 each started from SWMM's own state. SWMM converges on every one of them but
    step 107, where J4 (filling behind the closed outfall gate) meets C3's fasnh threshold
    max(yN, yC) and its storage equation has no fixed point. noodl converges on all of
    them: at the targets wherever SWMM converged from a converged start (100-106 and
    109-115), within SWMM's own oscillation on 107 and 108 (`DROP_FAIL_DEPTH`,
    `DROP_FAIL_FLOW`), holding J4 on its threshold on 107 only."""
    inp = _drop_inp(tmp_path, "c3_offset_0.1")
    rows, conn, engine, _ = _swmm_rows(inp)
    assert engine == "5.2.4"
    assert _swmm_not_converged_within(inp, 106) == 0
    assert _swmm_not_converged_within(inp, 107) == 1
    assert _swmm_not_converged_within(inp, 115) == 1
    converged, result, held = _restart_differences(inp, rows, conn, list(range(100, 116)))
    assert converged
    _check_restarts(result, [107])
    assert {k for k, h in held.items() if h} == {107}, held
    assert abs(held[107]["J4"]) <= 1e-9, held


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(DROP_VARIANTS))
def test_drop_switch_matches_swmm_wherever_swmm_converges(tmp_path, name):
    """Each drop variant over all 359 steps, compared where SWMM converges: on all but
    the steps listed in `DROP_SWMM_FAILS` (measured here); those and the steps after them
    are excluded.

    - noodl's own run converges on every step and meets the targets on every step before
      SWMM's first failure. From that step on the two histories differ by what each did
      on it -- SWMM's last trial against noodl's held solution -- and the difference
      persists as stored volume behind the closed outfall gate (up to 3e-4 m), so later
      steps of the two runs are not a fixed-point comparison.
    - Started from SWMM's own state before each step (`_restart_differences`), noodl meets
      the targets on every step SWMM converged from a converged start, is within SWMM's
      oscillation on its failing steps and the steps after them, and holds a junction on
      its threshold (to 1e-9 m) on exactly SWMM's failing steps (measured: J4 at 60 and 62
      and J5 at 167 of the 0.03 m variant, J4 at 107, J5 at 106 and 129)."""
    inp = _drop_inp(tmp_path, name)
    rows, conn, engine, _ = _swmm_rows(inp)
    assert engine == "5.2.4"
    fails = DROP_SWMM_FAILS[name]
    assert _swmm_failing_steps(inp, len(rows)) == fails
    net, _, _ = read_swmm_inp(inp)
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    record = ("water.phi", "water.q", "sewer.overflow")
    diags: list = []
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=len(rows), record=record,
                                    diagnostics=diags)
    assert all(bool(d["converged"]) for d in diags)
    diffs = _differences(name, rows, conn, fails[0] - 1, run=(net, model, hist, None))
    _assert_parity(_worst(diffs))
    converged, result, held = _restart_differences(inp, rows, conn,
                                                   list(range(2, len(rows) + 1)))
    assert converged
    _check_restarts(result, fails)
    # started from SWMM's state, noodl holds a junction on its threshold exactly on
    # SWMM's failing steps -- including step 167 of the 0.03 m variant, which lies in
    # SWMM's own history only (J5) -- and on no other step
    on = {k: h for k, h in held.items() if h}
    assert set(on) == set(fails), on
    assert all(abs(gap) <= 1e-9 for h in on.values() for gap in h.values()), on


# ------------------------------------------------------------------------- storage units
#: Storage-unit fixtures (`[STORAGE]`): the file and its edits.
#: - ``free_surface``: the tree with J5 CONICAL, J3 FUNCTIONAL and J4 TABULAR (all SurDepth
#:   0, MaxDepth 3 m), filling below every crown;
#: - ``extran``: the 3x EXTRAN tree with J3 FUNCTIONAL, MaxDepth 0.4 m below its 0.45 m
#:   crown and no SurDepth, so its full depth is NOT extended and it floods at 0.4 m from
#:   step 33, and J4 TABULAR, MaxDepth 0.3 m with SurDepth 1.0 m, so its full depth IS
#:   extended to its 0.45 m crown and it surcharges above that (to 0.92 m) with its volume
#:   held at the full volume (link.c:450-458, dynwave.c:683-687, node.c:943-945);
#: - ``slot``: the same under SURCHARGE_METHOD SLOT (no storage surcharge branch: J4 rises
#:   on its storage curve's area above its full depth);
#: - ``extran_high``: ``extran`` with J4's MaxDepth 0.55 m ABOVE its 0.45 m crown
#:   (link.c:440-463): its full depth stays 0.55 m, and J4 surcharges only above that (to
#:   0.84 m), where a plain junction would at its crown. SWMM's surcharged branch still
#:   uses the crown for its blend and its floor (dynwave.c:724-736); noodl holds the band
#:   at the full depth. J4 never falls back below 0.55 m in this run (it ends at 0.66 m).
STORAGE_VARIANTS = {
    "free_surface": ("storage_dynwave.inp", []),
    "extran": ("storage_extran_dynwave.inp", []),
    "slot": ("storage_extran_dynwave.inp",
             [("SURCHARGE_METHOD     EXTRAN", "SURCHARGE_METHOD     SLOT")]),
    "extran_high": ("storage_extran_dynwave.inp",
                    [("J4               9.0      0.3        0",
                      "J4               9.0      0.55       0")]),
}
#: J4's full depth (m) and full volume (m3, curve Small: area 1 + 2 y up to 0.5 m, then
#: 2 + 2 (y - 0.5)) per surcharging variant
STORAGE_J4_FULL = {"extran": (0.45, 0.5 * (1.0 + 1.9) * 0.45),
                   "slot": (0.45, 0.5 * (1.0 + 1.9) * 0.45),
                   "extran_high": (0.55, 0.75 + 0.5 * (2.0 + 2.1) * 0.05)}
#: The fast comparison: the free-surface fixture from its dry start through the filling of
#: all three units and the first flow out of the outfall conduit.
STORAGE_FAST_STEPS = 40


def _storage_inp(tmp_path, variant):
    name, edits = STORAGE_VARIANTS[variant]
    text = (DATA / name).read_text()
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path = tmp_path / f"storage_{variant}.inp"
    path.write_text(text)
    return path


def _swmm_storage(inp, steps=None):
    """SWMM on `inp` (asserting 5.2.4 and 0 % of steps not converging): per step the link
    flows, node depths and node flooding as `_swmm`, and each node's volume (pyswmm's
    `Node.volume`, ``newVolume * UCF(VOLUME)``, m3; 0 at a junction, node.c:362-379)."""
    rpt = inp.with_suffix(".rpt")
    rows, volumes = [], []
    with pyswmm.Simulation(str(inp), str(rpt), str(inp.with_suffix(".out"))) as sim:
        conn = {link.linkid: link.connections for link in pyswmm.Links(sim)}
        for _ in sim:
            nodes = list(pyswmm.Nodes(sim))
            rows.append(({link.linkid: link.flow for link in pyswmm.Links(sim)},
                         {node.nodeid: node.depth for node in nodes},
                         {node.nodeid: node.flooding for node in nodes}))
            volumes.append({node.nodeid: node.volume for node in nodes})
        assert str(sim.engine_version) == "5.2.4"
    m = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", rpt.read_text())
    assert m is not None and float(m.group(1)) == 0.0, "SWMM did not converge on every step"
    return rows[:steps], volumes[:steps], conn


def _storage_parity(tmp_path, variant, steps=None):
    """noodl against SWMM on a storage variant over `steps` steps (all by default): the
    worst depth difference, flow and overflow excess (`_worst`) and the worst storage
    volume excess over ``TOL_DEPTH * A(y)`` (m3, <= 0 passes: the depth target times the
    unit's area at SWMM's depth); and SWMM's rows."""
    inp = _storage_inp(tmp_path, variant)
    rows, volumes, conn = _swmm_storage(inp, steps)
    net, _, _ = read_swmm_inp(inp)
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    record = ("water.phi", "water.q", "sewer.overflow", "sewer.storage_volume")
    # the surcharged and flooded variants need the tight head solve of the EXTRAN runs
    # (EXTRAN_ATOL); the free-surface one keeps the default, as the other free-surface runs
    solve = {} if variant == "free_surface" else {"atol": EXTRAN_ATOL}
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=len(rows), record=record,
                                    **solve)
    worst = _worst(_differences(None, rows, conn, len(rows), run=(net, model, hist, None)))
    hyd = model.dynamic_wave
    units = [j for j, m in enumerate(net.manholes) if m.storage is not None]
    worst_volume = -float("inf")
    for k, vol in enumerate(volumes):
        depth = torch.tensor([rows[k][1][n] for n in hyd.junction_names], dtype=torch.float64)
        area = hyd.storage_curves.area(depth)
        for j in units:
            name = hyd.junction_names[j]
            excess = (abs(hist["sewer.storage_volume"][k, j].item() - vol[name])
                      - TOL_DEPTH * area[j].item())
            worst_volume = max(worst_volume, excess)
    return worst, worst_volume, rows, volumes


def test_storage_units_match_swmm_while_they_fill(tmp_path):
    """FUNCTIONAL, TABULAR and CONICAL storage units (`storage_dynwave.inp`) from the dry
    start: depths, flows and the units' volumes against SWMM at the free-surface targets
    over the first `STORAGE_FAST_STEPS` steps (3e-10 m measured)."""
    worst, worst_volume, rows, _ = _storage_parity(tmp_path, "free_surface",
                                                   STORAGE_FAST_STEPS)
    _assert_parity(worst)
    assert worst_volume <= 0.0
    assert min(rows[-1][1][n] for n in ("J3", "J4", "J5")) > 0.02     # every unit is wet


@pytest.mark.slow
@pytest.mark.parametrize("variant", sorted(STORAGE_VARIANTS))
def test_storage_units_match_swmm_at_every_step(tmp_path, variant):
    """Every variant over all 359 steps at the free-surface targets, with the regime it
    exists for: free surface (no unit floods), or J3 flooding at its own MaxDepth 0.4 m (not
    its crown) and J4 rising above its extended full depth 0.45 m with SWMM's volume capped
    at the full volume there."""
    worst, worst_volume, rows, volumes = _storage_parity(tmp_path, variant)
    _assert_parity(worst)
    assert worst_volume <= 0.0
    flooded = _flooded_steps(rows)
    peak = {n: max(r[1][n] for r in rows) for n in ("J3", "J4")}
    if variant == "free_surface":
        assert flooded == {}
        return
    assert set(flooded) == {"J3"}
    assert peak["J3"] == pytest.approx(0.4, abs=1e-12)
    assert peak["J4"] > 0.6
    full_depth, full = STORAGE_J4_FULL[variant]
    above = [v["J4"] for r, v in zip(rows, volumes, strict=True) if r[1]["J4"] > full_depth]
    assert above and max(abs(v - full) for v in above) < 1e-12


# ------------------------------------------------------------- orifices, weirs, outlets
#: Regulator fixtures (`structures_dynwave.inp`): a side orifice OR1 (J2 -> J3, 0.2 x 0.3 m,
#: crest 0.1 m), a transverse weir W1 with two end contractions (J4 -> J5, crest 0.3 m), a
#: V-notch weir W2 (J7 -> J4, crest 0.1 m) and an outlet OL1 with a RATING curve from the
#: 3 m2 storage unit J8 into the outfall Out2 (an outfall on a regulator: yN = yC = 0).
#: - ``free``: the file. Every regulator opens from dry; OR1 passes from its weir regime
#:   (f < 1) to its orifice regime (J2 above the 0.3 m crown) with J3 below its crest, so
#:   SWMM's switch is continuous there; no regulator is submerged.
#: - ``submerged``: Out1 FIXED at 9.75 m (between J5's invert and W1's 9.8 m crest) and C5
#:   0.45 m. The outfall first fills J5 backwards through C5, overshooting W1's crest, so
#:   W1 flows in reverse into the dry J4 (steps 51-72, its crest still node 1's) until J5
#:   falls below the crest and W1 closes; once J4 spills forward, J5 backs up above the
#:   crest: Villemonte submergence on W1 for over 200 steps (ratio ~0.46 at the end), with C5
#:   full at J5. W1's flow never passes through zero with both heads above the crest.
#: - ``outlet_through_crest``: J3 at 10.95 m (OR1's crest 11.1 m is 0.15 m above it) and C3
#:   0.30 m, so J3 fills through OR1's crest while OR1 flows (step 97): OR1's end area at J3
#:   jumps from 0 (DN_CRITICAL) to ``W L / 2`` (~9 m2, SUBCRITICAL), a threshold like a drop
#:   end's. SWMM converges on every step and noodl converges and matches without
#:   an in-pass switch for it, so none is added.
#:   (Out1 FIXED at 10.0 m instead fills J5 from the outfall and reverses W1 through zero
#:   flow under full submergence, where the flow's slope in the heads is infinite: SWMM
#:   fails to converge on 3 of 359 steps there (0.84 %; its 0.001 cfs reversal flow shows
#:   on W1 at step 99), so that variant is not compared.)
STRUCTURE_VARIANTS = {
    "free": [],
    "submerged": [
        ("Out1             8.0        FREE                        NO",
         "Out1             8.0        FIXED      9.75             NO"),
        ("C5               CIRCULAR     0.60", "C5               CIRCULAR     0.45"),
    ],
    "outlet_through_crest": [
        ("J3               10.5       5", "J3               10.95      5"),
        ("C3               CIRCULAR     0.60", "C3               CIRCULAR     0.30"),
    ],
}
#: J3's invert per variant (m)
STRUCTURE_J3_INVERT = {"free": 10.5, "submerged": 10.5, "outlet_through_crest": 10.95}
#: The fast comparison: the free fixture from its dry start through the outlet nearing its
#: rating, the orifice opening (step 32), the V-notch weir starting to flow (step 48) and
#: the transverse weir opening (step 126).
STRUCTURE_FAST_STEPS = 132


def _structures_inp(tmp_path, variant):
    text = (DATA / "structures_dynwave.inp").read_text()
    for old, new in STRUCTURE_VARIANTS[variant]:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path = tmp_path / f"structures_{variant}.inp"
    path.write_text(text)
    return path


def _structure_parity(tmp_path, variant, steps=None):
    """noodl against SWMM (5.2.4, converged on every step) on a regulator variant over
    `steps` steps (all by default): `_worst` over every node depth and every link flow,
    conduits and regulators; and SWMM's rows."""
    inp = _structures_inp(tmp_path, variant)
    rows, conn, engine, not_converging = _swmm_rows(inp)
    assert engine == "5.2.4"
    assert not_converging == 0.0, f"SWMM did not converge on every step ({not_converging} %)"
    rows = rows[:steps]
    net, _, _ = read_swmm_inp(inp)
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=len(rows),
                                    record=("water.phi", "water.q", "sewer.overflow"))
    worst = _worst(_differences(None, rows, conn, len(rows), run=(net, model, hist, None)))
    return worst, rows


def test_regulators_match_swmm_while_they_open(tmp_path):
    """The side orifice, both weirs and the rating-curve outlet from the dry start, at the
    free-surface targets over the first `STRUCTURE_FAST_STEPS` steps."""
    worst, rows = _structure_parity(tmp_path, "free", STRUCTURE_FAST_STEPS)
    _assert_parity(worst)
    flows = rows[-1][0]
    assert min(flows["OR1"], flows["W1"], flows["W2"], flows["OL1"]) > 0.0   # all flowing


@pytest.mark.slow
@pytest.mark.parametrize("variant", sorted(STRUCTURE_VARIANTS))
def test_regulators_match_swmm_at_every_step(tmp_path, variant):
    """Every variant over all 359 steps at the free-surface targets, with the regime it
    exists for: the orifice reaching its orifice regime (J2 above the opening's crown,
    11.3 m) with J3 below its crest, except in ``outlet_through_crest``, where J3 fills
    through OR1's crest while OR1 flows (its end area at J3 jumping); in
    ``submerged`` W1 first reversed (J5 above its 9.8 m crest, J4 below it), then
    submerged (J5 above the crest while it flows forward) for much of the run."""
    worst, rows = _structure_parity(tmp_path, variant)
    _assert_parity(worst)
    final = rows[-1]
    j3 = [STRUCTURE_J3_INVERT[variant] + r[1]["J3"] for r in rows]
    assert 11.0 + final[1]["J2"] > 11.3
    # OR1's outlet J3 stays below its 11.1 m crest, or fills through it while OR1 flows
    crossed = any(a < 11.1 <= b and r[0]["OR1"] > 0.0
                  for a, b, r in zip(j3, j3[1:], rows[1:], strict=False))
    assert crossed == (variant == "outlet_through_crest")
    assert (j3[-1] < 11.1) == (variant != "outlet_through_crest")
    reversed_ = sum(r[0]["W1"] < 0.0 for r in rows)
    submerged = sum(9.4 + r[1]["J5"] > 9.8 and r[0]["W1"] > 0.0 for r in rows)
    assert (reversed_ > 10 and submerged > 100) == (variant == "submerged")
    assert reversed_ == 0 or variant == "submerged"


# ------------------------------------------------------- tidal and time-series outfalls (C5)
TIDAL_NAME = "tidal_dynwave.inp"
#: The fixture's TIDAL curve (hour, stage elevation m), and the same stage as a SWMM
#: [TIMESERIES] ((h:mm:ss, m): the time is the elapsed run time on the start date, so the
#: two are one function of the elapsed time over the 30 min) and as the callable noodl takes.
TIDE_POINTS = ((0.0, 8.05), (0.1, 8.20), (0.2, 8.38), (0.3, 8.38), (0.5, 8.15),
               (1.0, 8.15), (2.0, 8.05))
#: The fast comparison: the first 6 min, the stage rising from 8.05 m (below the outfall
#: conduit's critical elevation while it is dry) through the first kink at 6 min.
TIDAL_FAST_STEPS = 72


def _tide_seconds(t):
    """The `TIDE_POINTS` stage (m) at elapsed `t` s: linear between points, the first or
    last stage outside them (SWMM's `table_tseriesLookup` with extend, table.c:745-804)."""
    pts = [(3600.0 * h, y) for h, y in TIDE_POINTS]
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, y0), (t1, y1) in zip(pts, pts[1:], strict=False):
        if t <= t1:
            return y0 + (y1 - y0) * (t - t0) / (t1 - t0)
    return pts[-1][1]


def _timeseries_inp(tmp_path, source=TIDAL_NAME):
    """`source` with its TIDAL outfall turned into a TIMESERIES outfall on `TIDE_POINTS`."""
    text = (DATA / source).read_text()
    a, b = text.index("[CURVES]"), text.index("[REPORT]")
    points = "".join(
        f"Tide1            {int(h * 3600) // 3600}:{int(h * 3600) % 3600 // 60:02d}:"
        f"{int(round(h * 3600)) % 60:02d} {y}\n" for h, y in TIDE_POINTS)
    text = text[:a] + f"[TIMESERIES]\n{points}\n" + text[b:]
    old = "Outfall          8.0        TIDAL      Tide1            NO"
    assert old in text
    text = text.replace(old, "Outfall          8.0        TIMESERIES Tide1            NO")
    path = tmp_path / ("timeseries_" + source)
    path.write_text(text)
    return path


def _tidal_inp_variable(tmp_path):
    """The variable-step tree (60 s routing step) with the same tide."""
    tidal = (DATA / TIDAL_NAME).read_text()
    text = (DATA / VARIABLE_NAME).read_text()
    a, b = tidal.index("[CURVES]"), tidal.index("[REPORT]")
    text = text.replace("[REPORT]", tidal[a:b] + "[REPORT]", 1)
    old = "Outfall          8.0        FREE                        NO"
    assert old in text
    text = text.replace(old, "Outfall          8.0        TIDAL      Tide1            NO")
    path = tmp_path / "tidal_variable.inp"
    path.write_text(text)
    return path


def _tide_run(inp, steps_limit=None, outfall_stage=None):
    """noodl on `inp` as `_variable_run` (the reader's network, 1800 s), with the TIMESERIES
    outfalls' stage functions `outfall_stage` if any."""
    net, _, _ = read_swmm_inp(inp)
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX,
                                        outfall_stage=outfall_stage)
    kwargs = {"steps": steps_limit} if steps_limit else {"duration": 1800.0}
    _, hist = simulate_dynamic_wave(model, state, drivers, record=(
        "water.phi", "water.q", "sewer.overflow"), **kwargs)
    return net, model, hist, None


def _tide_parity(tmp_path, kind, steps=None):
    """noodl against SWMM (5.2.4, converged on every step) on the fixed-step tide fixture
    (`kind` ``"tidal"``, or ``"timeseries"``: the same stage as a [TIMESERIES] driven by
    `_tide_seconds`) over `steps` steps (all by default): `_worst`, SWMM's rows and the
    outfall's depth in noodl's history."""
    if kind == "tidal":
        inp, stage = tmp_path / TIDAL_NAME, None
        shutil.copy(DATA / TIDAL_NAME, inp)
    else:
        inp, stage = _timeseries_inp(tmp_path), {"Outfall": _tide_seconds}
    rows, conn, engine, not_converging = _swmm_rows(inp)
    assert engine == "5.2.4"
    assert not_converging == 0.0, f"SWMM did not converge on every step ({not_converging} %)"
    rows = rows[:steps]
    run = _tide_run(inp, steps_limit=len(rows), outfall_stage=stage)
    return _worst(_differences(None, rows, conn, len(rows), run=run)), rows, run


@pytest.mark.parametrize("kind", ["tidal", "timeseries"])
def test_a_tidal_outfall_matches_swmm_while_the_tide_rises(tmp_path, kind):
    """The tide fixture's first `TIDAL_FAST_STEPS` steps at the free-surface targets, for the
    TIDAL outfall and for the same stage given as a TIMESERIES outfall (the series driven
    identically on both sides): the outfall depth follows the stage at each step's END
    (5 s into the run at step 1: a stage at the step start would be 0.05 m, not 0.0521 m)."""
    worst, rows, (_, model, hist, _) = _tide_parity(tmp_path, kind, TIDAL_FAST_STEPS)
    _assert_parity(worst)
    out = model.dynamic_wave.outfall_idx
    stage = torch.tensor([_tide_seconds(5.0 * (k + 1)) for k in range(TIDAL_FAST_STEPS)],
                         dtype=torch.float64)
    assert abs(rows[0][1]["Outfall"] - (_tide_seconds(5.0) - 8.0)) < 1e-9     # SWMM: step end
    assert abs(rows[0][1]["Outfall"] - (_tide_seconds(0.0) - 8.0)) > 1e-3
    # the outfall head is the stage while the stage is above the critical elevation
    assert (hist["water.phi"][:, out].flatten() - stage).abs().max() < 1e-9


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["tidal", "timeseries"])
def test_a_tidal_outfall_matches_swmm_at_every_step(tmp_path, kind):
    """All 359 steps: the tide rises from 8.05 m to 8.38 m (the outfall conduit backed up to
    0.38 m of its 0.45 m), holds, and falls to 8.15 m; depths and flows agree at the
    free-surface targets, SWMM converging on every step, for the TIDAL outfall and for the
    equivalent TIMESERIES outfall."""
    worst, rows, _ = _tide_parity(tmp_path, kind)
    _assert_parity(worst)
    assert max(r[1]["Outfall"] for r in rows) > 0.37 and rows[-1][1]["Outfall"] < 0.2
    # the tide is felt upstream: the outfall conduit's flow is not the free outfall's
    assert max(r[1]["J4"] for r in rows) > 0.19


@pytest.mark.slow
def test_a_tidal_outfall_matches_swmm_under_the_variable_step(tmp_path):
    """The variable-step tree with the same tide: the step sequence equals SWMM's to the
    millisecond (the stage is that at the END of each, however long), and depths and flows
    agree at every step."""
    inp = _tidal_inp_variable(tmp_path)
    rows, times, conn, engine, not_converging = _swmm_timed(inp)
    assert engine == "5.2.4" and not_converging == 0.0
    run = _tide_run(inp)
    _, _, hist, _ = run
    ends = [round(x) for x in (hist["t"] * 1000.0).tolist()]
    assert ends[:len(times)] == times
    assert len({b - a for a, b in zip(times, times[1:], strict=False)}) > 2   # steps vary
    _assert_parity(_worst(_differences(None, rows, conn, len(rows), run=run)))


# ------------------------------------------------------------------- quality
#: A conservative tracer (`[POLLUTANTS]`, Kdecay 0) for the quality comparisons, written into
#: tmp_path variants of the committed fixtures (the reader takes only constant pollutant
#: baselines; a time series belongs in the drivers).
_TRACER = (
    "[POLLUTANTS]\n"
    ";;Name  Units  Crain  Cgw  Crdii  Kdecay  SnowOnly  Co-Pollutant  Co-Frac  Cdwf  Cinit\n"
    "Tracer  MG/L   0.0    0.0  0.0    0.0     NO        *             0.0      0.0   0.0\n\n"
    "[JUNCTIONS]")
#: Concentration target for the steady tracer: relative. At steady state both codes mix a
#: conservative tracer by the flows alone (SWMM: a junction is a zero-volume node mixing its
#: inflows, qualrout.c:221-249, and a conduit at steady volume passes its upstream node's
#: concentration, :315-341; noodl: the implicit upwind balance of each junction), so what
#: remains is the hydraulic parity of the flow ratios and each run's distance from steady
#: state.
TOL_TRACER = 1e-8


def _tracer_inp(tmp_path, name, end, loads, series=""):
    """A tmp_path copy of the fixture `name` running to `end` (HH:MM:SS), with the tracer
    entering at the junctions of `loads` ({junction: (time series name or '""', baseline
    mg/L)}) and the `[TIMESERIES]` text `series`."""
    text = (DATA / name).read_text()
    for old, new in (("END_TIME             00:30:00", f"END_TIME             {end}"),
                     ("[JUNCTIONS]", _TRACER), ("[REPORT]", series + "[REPORT]")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    lines = text.splitlines()
    at = lines.index("[INFLOWS]")
    end_of_section = next(i for i in range(at, len(lines)) if not lines[i].strip())
    lines[end_of_section:end_of_section] = [
        f"{node}  Tracer  {ts}  CONCENTRATION  1.0  1.0  {base}"
        for node, (ts, base) in loads.items()]
    inp = tmp_path / f"{Path(name).stem}_tracer.inp"
    inp.write_text("".join(line + "\n" for line in lines))
    return inp


def _swmm_tracer(inp):
    """SWMM on the tracer variant `inp`: per step the link and node concentrations (kg/m3),
    the link flows (m3/s, file orientation) and volumes (m3); and the report's percentage of
    steps not converging."""
    rows = []
    with pyswmm.Simulation(str(inp), str(inp.with_suffix(".rpt")),
                           str(inp.with_suffix(".out"))) as sim:
        for _ in sim:
            links, nodes = list(pyswmm.Links(sim)), list(pyswmm.Nodes(sim))
            rows.append({
                "link": {lk.linkid: lk.pollut_quality["Tracer"] * 1e-3 for lk in links},
                "node": {nd.nodeid: nd.pollut_quality["Tracer"] * 1e-3 for nd in nodes},
                "flow": {lk.linkid: lk.flow for lk in links},
                "volume": {lk.linkid: lk.volume for lk in links}})
        assert str(sim.engine_version) == "5.2.4"
    text = inp.with_suffix(".rpt").read_text()
    m = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", text)
    return rows, float(m.group(1))


def _noodl_tracer(name, loads):
    """noodl on the committed fixture `name` (the tracer is a driver here) with a
    conservative tracer, its lateral concentration at step k ``loads["at"][junction](k)``
    (kg/m3), over ``loads["steps"]`` steps at the parity tolerances:
    the model and, per step, the link concentrations (each link's upwind junction's, by the
    sign of its flow; 0 from an outfall), the flows and the transport capacities."""
    net, _, _ = read_swmm_inp(DATA / name)
    model, state, drivers = build_model(net, quality=True, species=("tracer",),
                                        iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    hyd = model.dynamic_wave
    rows = []
    for k in range(loads["steps"]):
        c_in = torch.zeros(model.net.n, dtype=torch.float64)
        for node, fn in loads["at"].items():
            c_in[model.net.nodes.index(node)] = fn(k)
        drivers = dict(drivers, tracer_in=c_in)
        # the head solve at atol 1e-12: at the default solve_atol a junction can stop with
        # a balance residual of ~1e-9 m3/s (its atol), 1e-8 of these flows, which a steady
        # tracer reads directly as a mixing ratio
        state, _ = simulate_dynamic_wave(model, state, drivers, steps=1, atol=1e-12)
        full = torch.zeros(model.net.n, dtype=torch.float64)
        full[model.junction_idx] = state["water_quality.x"]
        q = state["water.q"]
        up = torch.where(q >= 0, full[hyd.edge_u], full[hyd.edge_v])
        rows.append({"link": dict(zip(model.link_names, up.tolist(), strict=True)),
                     "flow": dict(zip(model.link_names, q.tolist(), strict=True)),
                     "capacity": dict(zip(hyd.junction_names,
                                          state["water_quality.capacity"].tolist(),
                                          strict=True))})
    return model, rows


def _steady_tracer_error(model, mine, theirs):
    """The largest relative difference of the link concentrations (0 must be exactly 0)."""
    worst = 0.0
    for name in model.link_names:
        if theirs[name] == 0.0:
            assert mine[name] == 0.0, name
            continue
        worst = max(worst, abs(mine[name] - theirs[name]) / theirs[name])
    return worst


#: The tree's tracer: J1's lateral inflow carries none for an hour, while the hydraulics
#: settle (to 1e-9 relative), then its concentration ramps to 100 mg/L over ten minutes and
#: holds to the end at 3 h.
_TREE_TRACER_SERIES = ("[TIMESERIES]\n"
                       "TRC  01/01/2026 00:00 0.0\n"
                       "TRC  01/01/2026 01:00 0.0\n"
                       "TRC  01/01/2026 01:10 100.0\n"
                       "TRC  01/01/2026 03:00 100.0\n\n")


@pytest.fixture(scope="module")
def tree_tracer(tmp_path_factory):
    """The tree with the ramped tracer at J1, run in SWMM and in noodl (noodl driven, step by
    step, by SWMM's own J1 node concentration: J1 has only its lateral inflow, so that is
    the inflow concentration SWMM applied over the step, sampled at its start plus 1 ms)."""
    inp = _tracer_inp(tmp_path_factory.mktemp("tree_tracer"), "tree_dynwave.inp",
                      "03:00:00", {"J1": ("TRC", 0.0)}, _TREE_TRACER_SERIES)
    swmm, not_converging = _swmm_tracer(inp)
    assert not_converging == 0.0
    j1 = [row["node"]["J1"] for row in swmm]
    model, mine = _noodl_tracer("tree_dynwave.inp",
                                {"steps": len(swmm), "at": {"J1": lambda k: j1[k]}})
    return model, swmm, mine


def _swmm_quality_recursion(model, swmm, dt):
    """SWMM's dynamic-wave quality step (qualrout.c) on SWMM's own per-step flows and link
    volumes, for the tree (no conduit reversed, nothing loaded but J1): a junction mixes its
    inflowing links' START-of-step concentrations with its lateral load over its inflow, with
    no volume (`findNodeQual`, :221-249: `findLinkMassFlow` reads `Link.oldQual`, :210); a
    conduit mixes its upstream node's new concentration over its step-start volume with the
    inflow raised by its volume change, ``|q| + (v2 - v1)/dt`` (`findLinkQual`, :315-341,
    `getMixedQual`, :146-174). Returns the link concentrations per step."""
    net = model.dynamic_wave
    names = list(model.link_names)
    ends = {name: (model.net.nodes[int(u)], model.net.nodes[int(v)])
            for name, u, v in zip(names, net.edge_u, net.edge_v, strict=True)}
    lateral = {n: float(model.driver_template["inflow"][model.net.nodes.index(n)])
               for n in net.junction_names}
    c = dict.fromkeys(names, 0.0)
    v_old = {name: 0.0 for name in names}
    out = []
    for k, row in enumerate(swmm):
        q, v_new = row["flow"], row["volume"]
        if k == 0:
            v_old = dict(v_new)
        node = {}
        for n in net.junction_names:
            inflow = lateral[n] + sum(abs(q[name]) for name in names if ends[name][1] == n)
            load = (row["node"]["J1"] * lateral[n] if n == "J1" else 0.0) + sum(
                abs(q[name]) * c[name] for name in names if ends[name][1] == n)
            node[n] = load / inflow if inflow > 0.0 else 0.0
        new = {}
        for name in names:
            q_in = max(abs(q[name]) + (v_new[name] - v_old[name]) / dt, 0.0)
            c_in = node[ends[name][0]]
            if q_in <= 0.0:
                new[name] = c[name]
                continue
            mixed = (c[name] * v_old[name] + c_in * q_in * dt) / (v_old[name] + q_in * dt)
            new[name] = max(min(mixed, max(c[name], c_in)), 0.0)
        c, v_old = new, dict(v_new)
        out.append(dict(c))
    return out


@pytest.mark.slow
def test_tracer_on_the_tree_matches_swmm_at_steady_state(tree_tracer):
    """Two hours after J1's tracer ramps to 100 mg/L both codes hold the flow-weighted
    mixture on every conduit (C1 0.1, C3 0.1 x 0.025/0.065, C5 0.1 x 0.025/0.08 kg/m3;
    C2, C4 exactly 0), within `TOL_TRACER` relative."""
    model, swmm, mine = tree_tracer
    worst = _steady_tracer_error(model, mine[-1]["link"], swmm[-1]["link"])
    assert worst < TOL_TRACER, worst


@pytest.mark.slow
def test_tracer_on_the_tree_in_transit_follows_each_codes_scheme(tree_tracer):
    """While the tracer ramps up the two codes differ by design, and by exactly what their
    schemes predict.

    * SWMM is its own recursion (`_swmm_quality_recursion`, on its own flows and volumes):
      zero-volume junctions that mix the upstream links' START-of-step concentrations (a lag
      of one step each) and conduit reactors of the conduit volume. Asserted at 1e-12
      relative to the plateau, so the scheme is the one ported there.
    * noodl's reactors are the junctions (half of each attached conduit's volume,
      then the junction's own continuity), solved implicitly over the network. At steady
      flow both are linear time-invariant chains with the same gain g, so the area between
      each output and its final value is ``g u_final T``, T the chain's mean delay: SWMM's
      ``sum V_link / q + dt`` per junction passed, noodl's ``sum V_j / q_out``. Hence the
      time integral of (SWMM - noodl) on each conduit is ``g u_final (T_noodl - T_swmm)``,
      with no tolerance of its own: asserted at 1e-6 of it, the hydraulics' residual drift.
    * Pointwise, for step responses rising monotonically from 0 to g, the difference is at
      most ``max |du| g (T_noodl + T_swmm) / dt``, |du| the largest input change per step.
    """
    model, swmm, mine = tree_tracer
    dt = model.routing_step
    plateau = 0.1                                           # kg/m3, the 100 mg/L plateau
    ported = _swmm_quality_recursion(model, swmm, dt)
    worst_port = max(abs(p[name] - row["link"][name]) for p, row in zip(ported, swmm,
                                                                        strict=True)
                     for name in model.link_names)
    assert worst_port < 1e-12 * plateau, worst_port

    names = list(model.link_names)
    last = swmm[-1]
    q = {name: abs(last["flow"][name]) for name in names}
    cap = mine[-1]["capacity"]
    # each conduit's chain from J1, as (conduit, the junction that feeds it)
    chains = {"C1": [("C1", "J1")], "C3": [("C1", "J1"), ("C3", "J3")],
              "C5": [("C1", "J1"), ("C3", "J3"), ("C5", "J4")]}
    du = max(abs(swmm[k]["node"]["J1"] - swmm[k - 1]["node"]["J1"])
             for k in range(1, len(swmm)))
    for name, chain in chains.items():
        gain = q["C1"] / q[name]
        t_swmm = sum(last["volume"][c] / q[c] for c, _ in chain) + dt * (len(chain) - 1)
        t_noodl = sum(cap[j] / q[c] for c, j in chain)
        area = dt * sum(row["link"][name] - m["link"][name]
                        for row, m in zip(swmm, mine, strict=True))
        expected = gain * plateau * (t_noodl - t_swmm)
        assert area == pytest.approx(expected, rel=1e-6), (name, area, expected)
        bound = du * gain * (t_noodl + t_swmm) / dt
        worst = max(abs(row["link"][name] - m["link"][name])
                    for row, m in zip(swmm, mine, strict=True))
        assert worst <= bound, (name, worst, bound)


@pytest.mark.slow
def test_tracer_on_the_loop_matches_swmm_at_steady_state(tmp_path):
    """The loop with 100 mg/L entering at J1 and 40 mg/L at J3, after 3.5 h: J3 mixes Cb's
    water with its lateral inflow and sends it through the reversed conduit Cd (oriented
    J4 -> J3, its flow negative) to J4, which mixes it with Cc's. Every conduit within
    `TOL_TRACER` relative of SWMM. (The loop's hydraulics settle slowly, its outfall head
    contracting by ~0.92 per pass near t = 500 s: at 2 h both codes are still 1e-6 from
    steady state.)"""
    inp = _tracer_inp(tmp_path, "loop_dynwave.inp", "03:30:00",
                      {"J1": ('""', 100.0), "J3": ('""', 40.0)})
    swmm, not_converging = _swmm_tracer(inp)
    assert not_converging == 0.0
    model, mine = _noodl_tracer("loop_dynwave.inp", {"steps": len(swmm),
                                      "at": {"J1": lambda k: 0.1, "J3": lambda k: 0.04}})
    assert mine[-1]["flow"]["Cd"] < 0.0
    worst = _steady_tracer_error(model, mine[-1]["link"], swmm[-1]["link"])
    assert worst < TOL_TRACER, worst


# ------------------------------------------------------------------ head-solve robustness
#
# Before the convex line search and the step floor of the head solve
# (`dynwave.HEAD_SOLVE_NEWTON`), both fixtures ended the run with "KirchhoffLayer 'water'
# solve: ... failed to converge": the orifice fixture on its first step (a residual stuck at
# 1.2e-12 m3/s, the round-off floor of a junction flooding 17 m above its outfall behind a
# 0.1 m3/s cap, above the 1e-12 m3/s atol), the drop fixture at t = 65 s (a 2-cycle of the
# damped Newton between the stiff bed of a dry junction and the fast flow arriving down a
# 22 m drop).

ROBUST_ORIFICE = "robust_orifice_dynwave.inp"
ROBUST_DROP = "robust_drop_dynwave.inp"


def test_orifice_flooding_at_a_high_head_matches_swmm_at_every_step(tmp_path):
    """The head solve ends at its round-off floor (`step_floor`) instead of failing, and
    every step matches SWMM, which converges on each."""
    rows, conn = _converged_swmm(tmp_path, ROBUST_ORIFICE, surcharged=None)
    _assert_parity(_worst_errors(ROBUST_ORIFICE, rows, conn, len(rows)))


def _first_unconverged_row(tmp_path, name, rows):
    """The first row (0-based) at which SWMM's run of `name` did not converge: its
    MAX_TRIALS 500 and 499 runs agree exactly up to there (a step that converges within 499
    trials is the same computation in both)."""
    text = (DATA / name).read_text().replace("MAX_TRIALS           500",
                                             "MAX_TRIALS           499")
    fewer = tmp_path / "fewer.inp"
    fewer.write_text(text)
    other, _, _, _ = _swmm_rows(fewer)
    for k, ((_, depths, _), (_, depths2, _)) in enumerate(zip(rows, other, strict=True)):
        if any(depths[n] != depths2[n] for n in depths):
            return k
    return len(rows)


def test_drop_into_a_dry_chain_runs_and_matches_swmm_until_swmm_stops_converging(tmp_path):
    """Every step runs (the line search breaks the head solve's 2-cycle), and up to SWMM's
    own first non-converging step the two agree at the full targets."""
    rows, conn, engine, not_converging = _swmm(tmp_path, ROBUST_DROP)
    assert engine == "5.2.4"
    first = _first_unconverged_row(tmp_path, ROBUST_DROP, rows)
    assert 13 < first < len(rows), first          # past the step that used to fail
    run = _run_noodl(ROBUST_DROP, first)
    _assert_parity(_worst(_differences(ROBUST_DROP, rows, conn, first, run=run)))
