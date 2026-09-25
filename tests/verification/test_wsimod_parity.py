"""WSIMOD parity (rows W1, W3, W4): CapacitatedTransferLayer
replays WSIMOD's own captured per-arc requests and is compared against WSIMOD's own
realised flows on the SAME topology -- WSIMOD's own hard-clipped output is the reference,
exactly as pyswmm and EPANET are the references for the sewer and water applications
(nothing here reimplements WSIMOD's node science).

`quickstart_demo`'s six arcs are all captured with WSIMOD's own `UNBOUNDED_CAPACITY`
(1e15, confirmed directly against `wsimod.core.constants.UNBOUNDED_CAPACITY` while
regenerating these fixtures) and every captured `requested` equals its `realised` --
WSIMOD's own node science never lets this particular demo's push requests exceed a
downstream node's headroom, so W1 here exercises the identity path of the clip
arithmetic (`min(request, huge_capacity) == request`) rather than an actively-binding
clip. This is not a weaker fixture by choice -- `oxford_demo` (row W2) was hoped
to be where a genuinely bounded arc gets exercised; as shown below, it isn't either.

**Multiple events per (arc, timestep), and why every replay loop below ACCUMULATES.**
The committed fixtures hold one row per (arc, DIRECTION, timestep), so an arc that sees
both a push and a pull event within one WSIMOD timestep contributes TWO rows to the same
`(arc, t)`. Quickstart has none (7464 rows, 7464 distinct `(arc, t)` pairs); oxford has
5824 such pairs across four arcs -- `abstraction_to_farmoor`, `evenlode_to_thames`,
`thames_to_thames` and `thames_to_farmoor` (33880 rows, 28056 distinct pairs). Every
replay loop below therefore sums a group's rows into `r`/`wsimod_realised` (`+=`) rather
than assigning; an earlier version assigned, which silently kept only the LAST row of
each group ("push", sorted after "pull" in the fixture) and so replayed
`abstraction_to_farmoor` -- the ONE genuinely finite-capacity arc in either demo -- as
its always-zero push request instead of its real pull requests. The aggregate parity
numbers are unchanged by the fix (`max_abs_error` 1.8626e-9, 0 mismatched timesteps,
measured both ways), because the arc's capacity still never binds, but the numbers now
being compared are the real ones.

**oxford_demo: no arc's OWN capacity ever actually binds either.**
Of oxford_demo's 21 arcs, 20 are captured at `UNBOUNDED_CAPACITY` (1e15) exactly like
every quickstart arc. Exactly one, `abstraction_to_farmoor`, has a genuinely finite
capacity (50000.0, WSIMOD's own volume units) -- but across the full 1456-timestep
2009-2013 run, that arc's captured `requested` volume never exceeds ~30934.2, well under
its 50000 capacity, so it is NEVER clipped either (confirmed directly against the
committed `oxford_events.csv`, not assumed: `requested > capacity` is true for zero rows,
for every arc, in the whole fixture). So W1 AND W2 both only ever exercise the
capacitated layer's identity path (`min(request, capacity) == request`) for every arc
whose CAPACITY this harness can see -- neither reference demo shipped by
WSIMOD itself provides a load-bearing test of the hard-clip branch actually clipping
something on an arc's own `c_arc` capacity. This is a real gap in what W1/W2 validate,
not a minor footnote; a synthetic small-graph test with a deliberately tight `c_arc`
would be needed to exercise that branch; it is not retrofitted into these
WSIMOD-reference-only verification rows.

Oxford's `sewer_to_wwtw` arc DOES show `requested > realised` in 185 of its 1456
timesteps (max observed gap ~3.924e6) -- but this is NOT that arc's own capacity acting
(still 1e15, unbounded, confirmed the same way). It is WSIMOD's `WWTW` node applying its
own internal `treatment_throughput_capacity` / stormwater-tank overflow logic
(`wsimod/nodes/wtw.py`), a NODE-level throughput constraint this harness does
not extract (`extract_topology`, `tests/verification/_wsimod_reference.py`, reads only
`arc.capacity`) and that `CapacitatedTransferLayer` does not model in this test (`s_max`
here is a storage-headroom bound, set to infinity for every node, not a per-step
throughput-rate cap). `test_oxford_hard_clip_matches_wsimod_realised_flows` below
excludes this one arc from its strict comparison and documents exactly why at the
exclusion site -- it is not a bug in the replay or the harness, and every one of oxford's other 20
arcs across all 1456 timesteps replays to float64 noise (~1.86e-9), same as quickstart.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from noodl.layers.capacitated import CapacitatedTransferLayer
from noodl.topology import Network

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
        # ACCUMULATE (`+=`), never assign. A `(arc, t)` group can hold MORE THAN
        # ONE row: the fixtures are one row per (arc, DIRECTION, timestep), so an arc
        # that sees both a push and a pull event within one WSIMOD timestep has two
        # rows here. Summing their volumes is the correct per-timestep aggregate for
        # a replay through a single `.step`; assigning silently kept only whichever
        # row came last in the file ("push", sorted after "pull"). See the module
        # docstring's multiple-events-per-timestep paragraph.
        for _, row in group.iterrows():
            i = arc_index[row["arc"]]
            r[i] += row["requested"]
            wsimod_realised[i] += row["realised"]
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
        # ACCUMULATE (`+=`), never assign. A `(arc, t)` group can hold MORE THAN
        # ONE row: the fixtures are one row per (arc, DIRECTION, timestep), so an arc
        # that sees both a push and a pull event within one WSIMOD timestep has two
        # rows here. Summing their volumes is the correct per-timestep aggregate for
        # a replay through a single `.step`; assigning silently kept only whichever
        # row came last in the file ("push", sorted after "pull"). See the module
        # docstring's multiple-events-per-timestep paragraph.
        for _, row in group.iterrows():
            r[arc_index[row["arc"]]] += row["requested"]
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
    # expected behaviour of `mode="smooth"` at `tau=1e-3` (module docstring above:
    # tau=1e-3 is not asymptotically tight) -- it is bounded and does
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
        # ACCUMULATE (`+=`), never assign. A `(arc, t)` group can hold MORE THAN
        # ONE row: the fixtures are one row per (arc, DIRECTION, timestep), so an arc
        # that sees both a push and a pull event within one WSIMOD timestep has two
        # rows here. Summing their volumes is the correct per-timestep aggregate for
        # a replay through a single `.step`; assigning silently kept only whichever
        # row came last in the file ("push", sorted after "pull"). See the module
        # docstring's multiple-events-per-timestep paragraph.
        for _, row in group.iterrows():
            r[arc_index[row["arc"]]] += row["requested"]
        s_hard, f_hard = hard.step(s_hard, {"cap.requests": r}, dt=1.0)
        s_proj, f_proj = projection.step(s_proj, {"cap.requests": r}, dt=1.0)
        max_abs_error = max(max_abs_error, (f_proj - f_hard).abs().max().item())
    # Observed max_abs_error: 9.0955e-13 -- pure QP-solver/float64 arithmetic noise (no
    # arc in quickstart is ever actually capacity-bound, so this exercises
    # projection mode's identity path, not its active-constraint QP; row W2's oxford
    # fixture is where an arc's capacity actually binds). 1e-9 is ~1000x
    # above the observed noise floor.
    assert max_abs_error < 1e-9


# Excluded from the strict comparison below: WSIMOD's own `WWTW` node applies an
# internal `treatment_throughput_capacity` / stormwater-tank constraint on top of
# whatever it pulls in over `sewer_to_wwtw` (a NODE-level throughput cap, not that arc's
# OWN `.capacity`, which is 1e15/unbounded -- see the module docstring's oxford_demo
# section). `extract_topology` only captures per-arc capacity, and this test's `s_max`
# models storage headroom, not a per-step throughput rate, so this one node-level
# constraint is out of scope for what W2 can replay -- excluded by name, not
# papered over with a loosened global tolerance.
OXFORD_NODE_CAPACITY_ARCS = {"sewer_to_wwtw"}


def test_oxford_hard_clip_matches_wsimod_realised_flows():
    topology, events = _load("oxford")
    net, _, arc_names = _build_network(topology)
    c_arc = torch.tensor([a["capacity"] for a in topology["arcs"]], dtype=F64)
    s_max = torch.full((net.n,), float("inf"), dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    s = torch.zeros(net.n, dtype=F64)
    arc_index = {name: i for i, name in enumerate(arc_names)}
    excluded = torch.tensor([name in OXFORD_NODE_CAPACITY_ARCS for name in arc_names])
    max_abs_error = 0.0
    max_abs_error_excluded_arcs = 0.0
    n_mismatched_timesteps = 0
    for _t, group in events.groupby("t"):
        r = torch.zeros(len(arc_names), dtype=F64)
        wsimod_realised = torch.zeros(len(arc_names), dtype=F64)
        # ACCUMULATE (`+=`), never assign. A `(arc, t)` group can hold MORE THAN
        # ONE row: the fixtures are one row per (arc, DIRECTION, timestep), so an arc
        # that sees both a push and a pull event within one WSIMOD timestep has two
        # rows here. Summing their volumes is the correct per-timestep aggregate for
        # a replay through a single `.step`; assigning silently kept only whichever
        # row came last in the file ("push", sorted after "pull"). See the module
        # docstring's multiple-events-per-timestep paragraph.
        for _, row in group.iterrows():
            i = arc_index[row["arc"]]
            r[i] += row["requested"]
            wsimod_realised[i] += row["realised"]
        s, f = layer.step(s, {"cap.requests": r}, dt=1.0)
        diff = (f - wsimod_realised).abs()
        step_error = diff[~excluded].max().item()
        max_abs_error = max(max_abs_error, step_error)
        max_abs_error_excluded_arcs = max(
            max_abs_error_excluded_arcs, diff[excluded].max().item()
        )
        if step_error > 1e-6:
            n_mismatched_timesteps += 1
    print(
        f"oxford: max_abs_error={max_abs_error} over the "
        f"{len(arc_names) - len(OXFORD_NODE_CAPACITY_ARCS)} non-excluded arcs, "
        f"mismatched timesteps={n_mismatched_timesteps}; excluded arcs "
        f"{sorted(OXFORD_NODE_CAPACITY_ARCS)} max diff="
        f"{max_abs_error_excluded_arcs} (WWTW node-internal throughput cap, see "
        "module docstring)"
    )
    # Observed max_abs_error over the 20 non-excluded arcs, across all 1456 oxford
    # timesteps: 1.8626e-9 -- float64 noise at oxford's magnitude (flows reach ~1e6,
    # where relative machine epsilon alone is ~2e-10). 1e-6 is ~500x above that observed
    # floor while still well below a value that would hide a real regression. This
    # confirms every arc oxford_demo captures with a well-formed `.capacity` (whether
    # 1e15 or the one genuinely finite value, see module docstring) replays exactly.
    assert max_abs_error < 1e-6
    assert n_mismatched_timesteps == 0
