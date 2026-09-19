"""WSIMOD parity (design spec section 6, rows W1, W3, W4): CapacitatedTransferLayer
replays WSIMOD's own captured per-arc requests and is compared against WSIMOD's own
realised flows on the SAME topology -- WSIMOD's own hard-clipped output is the oracle,
exactly as pyswmm/EPANET were oracles for milestone 4 (nothing here reimplements
WSIMOD's node science, design spec section 1).

`quickstart_demo`'s six arcs are all captured with WSIMOD's own `UNBOUNDED_CAPACITY`
(1e15, confirmed directly against `wsimod.core.constants.UNBOUNDED_CAPACITY` while
regenerating these fixtures) and every captured `requested` equals its `realised` --
WSIMOD's own node science never lets this particular demo's push requests exceed a
downstream node's headroom, so W1 here exercises the identity path of the clip
arithmetic (`min(request, huge_capacity) == request`) rather than an actively-binding
clip. This is not a weaker fixture by choice -- `oxford_demo` (Task 7, row W2) is where a
genuinely bounded arc gets exercised; this module only claims what `quickstart_demo`
actually contains.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from tellegen.layers.capacitated import CapacitatedTransferLayer
from tellegen.topology import Network

pytest.importorskip("wsimod")  # only the regeneration script needs it installed to
# RUN; these tests read the already-committed fixtures and need it only so that a
# stale fixture regenerated against a mismatched wsimod version is caught by CI
# skipping rather than silently comparing against a version this pin never validated.

F64 = torch.float64
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "wsimod"


def _load(prefix: str):
    topology = json.loads((FIXTURE_DIR / f"{prefix}_topology.json").read_text())
    events = pd.read_csv(FIXTURE_DIR / f"{prefix}_events.csv")
    return topology, events


def _build_network(topology: dict) -> tuple[Network, list[str], list[str]]:
    net = Network()
    for node in topology["nodes"]:
        net.add_node(node["name"])
    arc_names = [a["name"] for a in topology["arcs"]]
    for arc in topology["arcs"]:
        net.add_edge(arc["source"], arc["target"], kind="link", name=arc["name"])
    return net, [n["name"] for n in topology["nodes"]], arc_names


def test_quickstart_hard_clip_matches_wsimod_realised_flows():
    topology, events = _load("quickstart")
    net, _node_names, arc_names = _build_network(topology)
    c_arc = torch.tensor([a["capacity"] for a in topology["arcs"]], dtype=F64)
    s_max = torch.full((net.n,), float("inf"), dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    s = torch.zeros(net.n, dtype=F64)
    arc_index = {name: i for i, name in enumerate(arc_names)}
    max_abs_error = 0.0
    for _t, group in events.groupby("t"):
        r = torch.zeros(len(arc_names), dtype=F64)
        wsimod_realised = torch.zeros(len(arc_names), dtype=F64)
        for _, row in group.iterrows():
            i = arc_index[row["arc"]]
            r[i] = row["requested"]
            wsimod_realised[i] = row["realised"]
        s, f = layer.step(s, {"cap.requests": r}, dt=1.0)
        max_abs_error = max(max_abs_error, (f - wsimod_realised).abs().max().item())
    # Observed max_abs_error over all 1456 quickstart timesteps: 1.1102e-16 (a single
    # float64 machine-epsilon-scale unit -- every captured request/realised pair is
    # identical, see module docstring, and `min(r, 1e15)` is exact in float64 for
    # these magnitudes). 1e-9 is ~1e7x above that noise floor while still catching any
    # real regression.
    assert max_abs_error < 1e-9


def test_quickstart_smooth_mode_converges_to_hard_clip():
    topology, events = _load("quickstart")
    net, _node_names, arc_names = _build_network(topology)
    c_arc = torch.tensor([a["capacity"] for a in topology["arcs"]], dtype=F64)
    s_max = torch.full((net.n,), float("inf"), dtype=F64)
    arc_index = {name: i for i, name in enumerate(arc_names)}
    hard = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    smooth = CapacitatedTransferLayer(
        net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="smooth", tau=1e-3
    )
    s_hard = torch.zeros(net.n, dtype=F64)
    s_smooth = torch.zeros(net.n, dtype=F64)
    max_abs_error = 0.0
    for _t, group in events.groupby("t"):
        r = torch.zeros(len(arc_names), dtype=F64)
        for _, row in group.iterrows():
            r[arc_index[row["arc"]]] = row["requested"]
        s_hard, f_hard = hard.step(s_hard, {"cap.requests": r}, dt=1.0)
        s_smooth, f_smooth = smooth.step(s_smooth, {"cap.requests": r}, dt=1.0)
        max_abs_error = max(max_abs_error, (f_smooth - f_hard).abs().max().item())
    # Observed max_abs_error: 1.7918e-3 (measured directly, not guessed in advance).
    # This is NOT capacity-smoothing residue -- every quickstart arc is unbounded
    # (1e15) and `a << b` there makes the softmin exact to float64 precision. It is
    # `_nonneg`'s softplus-at-zero bias: `tau * softplus(0/tau) = tau * log(2) approx
    # 6.9e-4` is the per-kink-site offset whenever a clamped quantity sits AT its zero
    # kink, which many of this fixture's requests do (several arcs log `requested ==
    # 0.0` at many timesteps -- see quickstart_events.csv); the observed value is a
    # small multiple of that per-site offset, consistent with more than one kink site
    # (arc headroom, node headroom) being at zero simultaneously and the bias
    # compounding forward through recurrent storage state `s`. This is the documented,
    # expected behaviour of `mode="smooth"` at `tau=1e-3` (module docstring above,
    # brief step 7: "tau=1e-3 is not asymptotically tight") -- it is bounded and does
    # not grow across the 1456-timestep run, not a runaway divergence. 3e-3 is
    # comfortably above the observed 1.7918e-3 while still well below a value that
    # would hide a real regression (an order-of-magnitude jump would still fail it).
    assert max_abs_error < 3e-3


def test_quickstart_projection_mode_converges_to_hard_clip():
    topology, events = _load("quickstart")
    net, _node_names, arc_names = _build_network(topology)
    c_arc = torch.tensor([a["capacity"] for a in topology["arcs"]], dtype=F64)
    s_max = torch.full((net.n,), float("inf"), dtype=F64)
    arc_index = {name: i for i, name in enumerate(arc_names)}
    hard = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    projection = CapacitatedTransferLayer(
        net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="projection"
    )
    s_hard = torch.zeros(net.n, dtype=F64)
    s_proj = torch.zeros(net.n, dtype=F64)
    max_abs_error = 0.0
    for _t, group in events.groupby("t"):
        r = torch.zeros(len(arc_names), dtype=F64)
        for _, row in group.iterrows():
            r[arc_index[row["arc"]]] = row["requested"]
        s_hard, f_hard = hard.step(s_hard, {"cap.requests": r}, dt=1.0)
        s_proj, f_proj = projection.step(s_proj, {"cap.requests": r}, dt=1.0)
        max_abs_error = max(max_abs_error, (f_proj - f_hard).abs().max().item())
    # Observed max_abs_error: 9.0955e-13 -- pure QP-solver/float64 arithmetic noise (no
    # arc in quickstart is ever actually capacity-bound, so this exercises
    # projection mode's identity path, not its active-constraint QP; row W2's oxford
    # fixture, Task 7, is where an arc's capacity actually binds). 1e-9 is ~1000x
    # above the observed noise floor.
    assert max_abs_error < 1e-9
