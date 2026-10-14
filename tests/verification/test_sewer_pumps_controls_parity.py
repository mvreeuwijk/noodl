"""Pumps and control rules under dynamic wave against live SWMM 5.2.4, every 5 s step.

Each case is one small system cut out of `pumps_dynwave.inp` / `controls_dynwave.inp` (the
files hold several independent systems, each named by a letter or number its lines end in;
`_system` keeps one). SWMM runs with MAX_TRIALS 500 and a HEAD_TOLERANCE of 1e-12 m and must
converge on every step; the pulses, curves and thresholds of the fixtures were chosen so that
it does: a STEPPED pump curve (PUMP1, PUMP2) crossed by a rising or falling well can have no
fixed point inside a step (the pump flow depends on the depth, the depth on the pump flow, and
the two branches never agree), and SWMM then fails to converge however tight its tolerance --
a property of the equations, not of noodl. The targets are those of the other dynamic-wave
parity tests: depth within 1e-7 m, flow within 1e-6 relative plus 1e-9 m3/s, and the link
settings equal (SWMM's ``current_setting``).
"""

import re
from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave

pyswmm = pytest.importorskip("pyswmm")

DATA = Path(__file__).resolve().parents[1] / "data" / "sewer"
TOL_DEPTH, TOL_FLOW_REL, TOL_FLOW_ABS, TOL_SETTING = 1e-7, 1e-6, 1e-9, 1e-9
ITERATE_TOL, ITERATE_MAX = 1e-13, 2000

PUMP_SYSTEMS = ["A", "H", "B", "C", "D", "F"]
CONTROL_SYSTEMS = ["1", "2", "3", "4", "5", "6", "7", "8"]


def _system(text: str, key: str) -> str:
    """The lines of `text` (a fixture with several systems) that belong to system `key`: every
    data line whose first field ends in `key`, every rule block named ``R<key>...``, and the
    lines that name no system (title, options, report)."""
    out, section, keep_rule = [], None, True
    for line in text.splitlines():
        stripped = line.split(";")[0].strip()
        if stripped.startswith("["):
            section = stripped[1:-1].upper()
            out.append(line)
            continue
        if not stripped or section in (None, "TITLE", "OPTIONS", "REPORT"):
            out.append(line)
            continue
        first = stripped.split()[0]
        if section == "CONTROLS":
            if first.upper() == "RULE":
                keep_rule = re.fullmatch(rf"R{key}[a-z]?", stripped.split()[1]) is not None
            if keep_rule:
                out.append(line)
        elif first.endswith(key):
            out.append(line)
    return "\n".join(out) + "\n"


def _write(tmp_path, name, key, extra_options=""):
    text = _system((DATA / name).read_text(), key)
    if extra_options:
        marker = "THREADS              1\n"
        text = text.replace(marker, marker + extra_options, 1)
    path = tmp_path / f"{Path(name).stem}_{key}.inp"
    path.write_text(text)
    return path


def _swmm(inp):
    rpt = inp.with_suffix(".rpt")
    rows = []
    with pyswmm.Simulation(str(inp), str(rpt), str(inp.with_suffix(".out"))) as sim:
        conn = {link.linkid: link.connections for link in pyswmm.Links(sim)}
        engine = str(sim.engine_version)
        for _ in sim:
            links, nodes = list(pyswmm.Links(sim)), list(pyswmm.Nodes(sim))
            rows.append(({link.linkid: link.flow for link in links},
                         {node.nodeid: node.depth for node in nodes},
                         {link.linkid: link.current_setting for link in links}))
    text = rpt.read_text()
    not_converging = float(re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", text).group(1))
    return rows, conn, engine, not_converging


def _compare(tmp_path, name, key, extra_options=""):
    """noodl against SWMM on system `key` of `name`: ``(worst depth, worst flow excess, worst
    setting difference, history, rows)``."""
    inp = _write(tmp_path, name, key, extra_options)
    rows, conn, engine, not_converging = _swmm(inp)
    assert engine == "5.2.4"
    assert not_converging == 0.0, f"SWMM did not converge on every step ({not_converging} %)"
    net, _, _ = read_swmm_inp(inp)
    model, state, drivers = build_model(net, iterate_tol=ITERATE_TOL, iterate_max=ITERATE_MAX)
    hyd = model.dynamic_wave
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=len(rows),
                                    record=("water.phi", "water.q", "sewer.setting"))
    nodes = list(hyd.junction_names) + list(hyd.outfall_names)
    idx = torch.cat([model.junction_idx, hyd.outfall_idx])
    names = list(model.net.nodes)
    sign = {}
    for link, u, v in zip(model.link_names, hyd.edge_u, hyd.edge_v, strict=True):
        sign[link] = 1.0 if tuple(conn[link]) == (names[int(u)], names[int(v)]) else -1.0
    depth, flow, setting = 0.0, -TOL_FLOW_ABS, 0.0
    for k, (flows, depths, settings) in enumerate(rows):
        y = hist["water.phi"][k, idx] - hyd.node_invert[idx]
        for j, n in enumerate(nodes):
            depth = max(depth, abs(y[j].item() - depths[n]))
        for i, link in enumerate(model.link_names):
            a, b = hist["water.q"][k, i].item(), sign[link] * flows[link]
            flow = max(flow, abs(a - b) - (TOL_FLOW_REL * abs(b) + TOL_FLOW_ABS))
        for i, link in enumerate(hyd.setting_names):
            setting = max(setting, abs(hist["sewer.setting"][k, i].item() - settings[link]))
    return depth, flow, setting, hist, rows


def _assert_parity(result):
    depth, flow, setting = result[:3]
    assert depth <= TOL_DEPTH, result[:3]
    assert flow <= 0.0, result[:3]
    assert setting <= TOL_SETTING, result[:3]


def _slow(keys, fast):
    return [k if k == fast else pytest.param(k, marks=pytest.mark.slow) for k in keys]


@pytest.mark.parametrize("key", _slow(PUMP_SYSTEMS, "A"))
def test_pumps_match_swmm(tmp_path, key):
    """A, H: PUMP3 with startup and shutoff depths (hysteresis, H against a lift); B: PUMP2
    steps in the well depth; C: PUMP4; D: PUMP1 steps in the volume of a storage well; F: an
    ideal pump. Each system's pump must actually run and, where it has depth limits, stop."""
    result = _compare(tmp_path, "pumps_dynwave.inp", key)
    _assert_parity(result)
    assert float(result[3]["water.q"].abs().max()) > 0.05                    # something pumped
    if key in "AHBD":                                                         # it cycled
        setting = result[3]["sewer.setting"][:, 0]
        assert float(setting.max()) == 1.0 and float(setting.min()) == 0.0


@pytest.mark.parametrize("key", _slow(CONTROL_SYSTEMS, "1"))
def test_control_rules_match_swmm(tmp_path, key):
    """Rules that switch a pump (status, speed by CURVE, PRIORITY) and that open and close an
    orifice (at CloseTime 0.01 h, by TIMESERIES, by PID) and a weir; system 8 runs at
    RULE_STEP 00:00:30 with variable premises. The setting must change during the run."""
    extra = "RULE_STEP            00:00:30\n" if key == "8" else ""
    result = _compare(tmp_path, "controls_dynwave.inp", key, extra)
    _assert_parity(result)
    setting = result[3]["sewer.setting"]
    assert float(setting.max() - setting.min()) > 0.1
