"""Cross sections, local losses and transects against live SWMM 5.2.4, every 5 s step.

`tests/data/sewer/make_shape_fixtures.py` writes one small network per fixture with one
independent chain (junction, 200 m conduit, outfall) per cross section: the SWMM shapes of
`xsect.c` (the tabulated and the closed-form ones, CUSTOM, FORCE_MAIN, FILLED_CIRCULAR), the
losses of `[LOSSES]` and the natural channels of `[TRANSECTS]`. Each is run in SWMM with its
iteration tightened (MAX_TRIALS 500, HEAD_TOLERANCE 1e-12; the test asserts SWMM converged on
every step) and in noodl with tight iteration controls; the junction and outfall depths must
agree to 1e-7 m and every link flow to 1e-6 relative plus 1e-9 m3/s, the same targets as the
circular dynamic-wave parity (`test_sewer_dynwave_parity`).

The fixtures keep SWMM's own iteration convergent (a subcritical slope, a wet start from
InitFlow, a MIN_SURFAREA of 12 m2): in a dry start or on a steep slope SWMM's relaxed
iteration limit-cycles on the wet/dry fronts and the outfall's normal-or-critical depth, and
there is no converged answer to compare with. Where the fixture is in such a regime noodl
and SWMM may also settle on different branches of the same lagged equations (a drained and a
refilling outfall conduit in one step).

Units. The US fixture is read by pyswmm in feet and cfs; depths and flows are converted by
SWMM's own factors (0.3048 m/ft, 0.02832 m3 per ft3).
"""

import re
import shutil
from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave

pyswmm = pytest.importorskip("pyswmm")

pytestmark = pytest.mark.slow

DATA = Path(__file__).resolve().parents[1] / "data" / "sewer"
TOL_DEPTH = 1e-7
TOL_FLOW_REL, TOL_FLOW_ABS = 1e-6, 1e-9
ITERATE_TOL, ITERATE_MAX = 1e-13, 2000

#: fixture -> steps compared
FIXTURES = {
    "shapes_dynwave.inp": 40,
    "shapes_drop_dynwave.inp": 40,
    "shapes_surcharge_dynwave.inp": 60,
    "shapes_slot_dynwave.inp": 60,
    "shapes_us_dynwave.inp": 40,
    "force_main_hw_dynwave.inp": 60,
    "force_main_dw_dynwave.inp": 60,
    "losses_dynwave.inp": 60,
    "irregular_dynwave.inp": 40,
    "culvert_dynwave.inp": 60,
}


def _swmm(tmp_path, name, steps):
    """SWMM's first `steps` rows (SI): per step the link flows (m3/s, file orientation) and
    node depths (m); the file-orientation connections; whether SWMM converged on every step."""
    inp = tmp_path / name
    shutil.copy(DATA / name, inp)
    us = "FLOW_UNITS           CFS" in inp.read_text()
    lf, qf = (0.3048, 0.02832) if us else (1.0, 1.0)
    rows = []
    with pyswmm.Simulation(str(inp), str(tmp_path / "r.rpt"), str(tmp_path / "r.out")) as sim:
        assert str(sim.engine_version) == "5.2.4"
        conn = {link.linkid: link.connections for link in pyswmm.Links(sim)}
        for _ in sim:
            rows.append(({link.linkid: link.flow * qf for link in pyswmm.Links(sim)},
                         {node.nodeid: node.depth * lf for node in pyswmm.Nodes(sim)}))
            if len(rows) >= steps:
                break
    m = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", (tmp_path / "r.rpt").read_text())
    return rows, conn, float(m.group(1)) if m else None


@pytest.fixture(scope="module", params=sorted(FIXTURES))
def comparison(request, tmp_path_factory):
    """``(name, {link: (worst depth difference, worst flow excess)})`` for one fixture."""
    name = request.param
    steps = FIXTURES[name]
    rows, conn, not_converging = _swmm(tmp_path_factory.mktemp("swmm"), name, steps)
    assert not_converging == 0.0, f"SWMM did not converge on every step ({not_converging} %)"
    net, _, _ = read_swmm_inp(DATA / name)
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=len(rows),
                                    record=("water.phi", "water.q"))
    hyd = model.dynamic_wave
    nodes = list(model.net.nodes)
    idx = torch.cat([model.junction_idx, hyd.outfall_idx])
    names = list(hyd.junction_names) + list(hyd.outfall_names)
    invert = hyd.node_invert[idx]
    ends = {n: (nodes[int(u)], nodes[int(v)]) for n, u, v in
            zip(model.link_names, hyd.edge_u, hyd.edge_v, strict=True)}
    sign = {n: (1.0 if tuple(conn[n]) == pair else -1.0) for n, pair in ends.items()}
    worst = {}
    for k, (flows, depths) in enumerate(rows):
        y = hist["water.phi"][k, idx] - invert
        depth_diff = {n: abs(y[j].item() - depths[n]) for j, n in enumerate(names)}
        for i, link in enumerate(model.link_names):
            ref = sign[link] * flows[link]
            excess = (abs(hist["water.q"][k, i].item() - ref)
                      - (TOL_FLOW_REL * abs(ref) + TOL_FLOW_ABS))
            d = max(depth_diff[n] for n in ends[link])
            w_d, w_f = worst.get(link, (0.0, -TOL_FLOW_ABS))
            worst[link] = (max(w_d, d), max(w_f, excess))
    return name, worst


def test_noodl_matches_swmm_on_every_link(comparison):
    name, worst = comparison
    bad = {link: w for link, w in worst.items() if w[0] > TOL_DEPTH or w[1] > 0.0}
    assert not bad, f"{name}: (worst depth m, flow excess m3/s) outside the targets: {bad}"
    assert len(worst) >= 2
