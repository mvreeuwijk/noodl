"""SewerNetwork validation, the builder and the steady driver."""

import pytest
import torch

from noodl.apps.sewer.network import (
    Manhole,
    Outfall,
    Pipe,
    SewerNetwork,
    build_model,
    initial_state,
    sewer_steady,
    tree_steady,
)
from noodl.drives import Stack

F64 = torch.float64


def test_the_fixture_tree_is_the_research_network():
    net = tree_steady()
    assert [m.name for m in net.manholes] == ["J1", "J2", "J5", "J3", "J4"]
    assert [p.name for p in net.pipes] == ["C1", "C2", "C4", "C3", "C5"]
    assert [o.name for o in net.outfalls] == ["Outfall"]
    assert [p.diameter for p in net.pipes] == [0.30, 0.30, 0.30, 0.45, 0.45]
    assert [round(p.slope, 6) for p in net.pipes] == [0.01, 0.01, 0.01, 0.005, 0.005]
    net.validate()


def test_a_manhole_with_two_outgoing_pipes_is_refused():
    net = SewerNetwork(
        manholes=(Manhole("J1", 12.0), Manhole("J2", 11.0)),
        pipes=(Pipe("A", "J1", "J2", 100, 0.3, 0.013, 0.01),
               Pipe("B", "J1", "Out", 100, 0.3, 0.013, 0.01),
               Pipe("C", "J2", "Out", 100, 0.3, 0.013, 0.01)),
        outfalls=(Outfall("Out", 9.0),),
    )
    with pytest.raises(ValueError, match=r"manhole 'J1' has 2 outgoing pipes"):
        net.validate()


def test_a_cycle_is_refused():
    net = SewerNetwork(
        manholes=(Manhole("J1", 12.0), Manhole("J2", 11.0)),
        pipes=(Pipe("A", "J1", "J2", 100, 0.3, 0.013, 0.01),
               Pipe("B", "J2", "J1", 100, 0.3, 0.013, 0.01)),
        outfalls=(Outfall("Out", 9.0),),
    )
    with pytest.raises(ValueError, match="does not reach an outfall"):
        net.validate()


def test_a_component_without_an_outfall_is_refused():
    net = SewerNetwork(
        manholes=(Manhole("J1", 12.0), Manhole("J2", 11.0), Manhole("J3", 10.0)),
        pipes=(Pipe("A", "J1", "J2", 100, 0.3, 0.013, 0.01),
               Pipe("B", "J2", "Out", 100, 0.3, 0.013, 0.01),
               Pipe("C", "J3", "J3", 100, 0.3, 0.013, 0.01)),
        outfalls=(Outfall("Out", 9.0),),
    )
    with pytest.raises(ValueError, match=r"'J3'"):
        net.validate()


@pytest.mark.parametrize("slope", [0.0, -0.001])
def test_a_zero_or_adverse_slope_is_refused(slope):
    net = SewerNetwork(
        manholes=(Manhole("J1", 12.0),),
        pipes=(Pipe("A", "J1", "Out", 100, 0.3, 0.013, slope),),
        outfalls=(Outfall("Out", 9.0),),
    )
    with pytest.raises(ValueError, match=r"pipe 'A' has slope"):
        net.validate()


def test_a_duplicate_name_is_refused():
    net = SewerNetwork(
        manholes=(Manhole("J1", 12.0), Manhole("J1", 11.0)),
        pipes=(Pipe("A", "J1", "Out", 100, 0.3, 0.013, 0.01),),
        outfalls=(Outfall("Out", 9.0),),
    )
    with pytest.raises(ValueError, match="duplicate node name 'J1'"):
        net.validate()


def test_builder_returns_a_model_a_state_and_drivers():
    model, state, drivers = build_model(tree_steady())
    assert set(model.potential) == {"air"}
    assert set(model.transport) == {"water_quality", "air_quality"}
    assert model.net.nodes == ["J1", "J2", "J5", "J3", "J4", "Outfall", "ambient"]
    for key in ("air.phi", "air.q", "water_quality.x", "air_quality.x"):
        assert key in state
    for key in ("inflow", "T_water", "T_head", "T_amb", "pH", "bod_in", "sulfide_in",
                "air.phi_boundary", "water_quality.x_boundary", "air_quality.x_boundary",
                "sewer.q_slope"):
        assert key in drivers, key


def test_the_air_layer_carries_every_kind():
    model, _, _ = build_model(tree_steady(), fans=("J3",))
    assert set(model.potential["air"].kinds) == {"headspace", "leak", "fan"}
    # one headspace edge per pipe, plus the outfall edge
    assert model.net.edge_index("headspace").numel() == 6
    assert model.net.edge_index("leak").numel() == 5
    assert model.net.edge_index("fan").numel() == 1


def test_air_false_builds_the_water_only_model():
    model, state, drivers = build_model(tree_steady(), air=False)
    assert model.potential == {}
    assert set(model.transport) == {"water_quality"}
    assert "ambient" not in model.net.nodes
    assert "air.phi_boundary" not in drivers


def test_quality_false_drops_both_quality_layers():
    model, _, _ = build_model(tree_steady(), quality=False)
    assert model.transport == {}


def test_storage_true_registers_the_closure_state_key():
    model, state, _ = build_model(tree_steady(), storage=True, dt_storage=60.0)
    assert "sewer.H" in model.closure_state_keys
    assert state["sewer.H"].shape == (5,)


def test_a_single_step_runs_and_conserves_the_air_balance():
    model, state, drivers = build_model(tree_steady())
    new = model.step(state, drivers, 60.0)
    residuals = model.residuals(new, drivers)
    assert float(residuals["air"].abs().max()) < 1e-10


def test_sewer_steady_reaches_a_fixed_point():
    model, state, drivers = build_model(tree_steady())
    final = sewer_steady(model, state, drivers)
    residuals = model.residuals(final, drivers)
    for name, value in residuals.items():
        assert float(value.abs().max()) < 1e-8, name


def test_initial_state_refuses_an_unknown_quantity():
    model, _, drivers = build_model(tree_steady())
    model.transport["water_quality"].quantity = "nonsense"
    with pytest.raises(ValueError, match="nonsense"):
        initial_state(model, drivers)


def test_read_swmm_inp_sulfide_source_uses_each_manholes_own_outgoing_pipe():
    """M4-R4/M4-R9 amendment: `tree_steady.inp`'s `[CONDUITS]` order (C1..C5) does not match
    its `[JUNCTIONS]` order (J1, J2, J5, J3, J4), so `read_swmm_inp` gives the non-identity
    `out_pipe = [0, 1, 3, 2, 4]`. The model's sulfide source at each manhole must equal
    `sulfide_rate` evaluated on THAT manhole's own outgoing pipe's hydraulics, not on the
    pipe at the same position -- exercised end to end through `read_swmm_inp` and
    `build_model`."""
    from pathlib import Path

    from noodl.apps.sewer.inp import read_swmm_inp
    from noodl.apps.sewer.quality import sulfide_rate

    data = Path(__file__).resolve().parents[2] / "data" / "sewer"
    net, _, _ = read_swmm_inp(data / "tree_steady.inp")
    assert [p.name for p in net.pipes] == ["C1", "C2", "C3", "C4", "C5"]
    assert [m.name for m in net.manholes] == ["J1", "J2", "J5", "J3", "J4"]

    model, state, drivers = build_model(net, air=False, quality=True)
    assert model.out_pipe.tolist() == [0, 1, 3, 2, 4]

    reaction = dict(model.reactions)["water_quality"]
    resolved = model._apply_closures(state, drivers)

    bod = torch.tensor([1e-3, 2e-3, 3e-3, 4e-3, 5e-3], dtype=F64)
    sulfide = torch.tensor([1e-4, 2e-4, 3e-4, 4e-4, 5e-4], dtype=F64)
    x = torch.stack([bod, sulfide], dim=-1)
    dt = 60.0
    out = reaction.apply(x, dt, resolved)

    radius = resolved["sewer.R_h"].index_select(-1, model.out_pipe)
    velocity = resolved["sewer.v"].index_select(-1, model.out_pipe)
    depth = resolved["sewer.d_m"].index_select(-1, model.out_pipe)
    slope = drivers["sewer.q_slope"]
    d_sulfide = sulfide_rate(
        bod, sulfide, drivers["T_water"], radius, slope, velocity, depth
    )
    expected_sulfide = torch.clamp(sulfide + dt * d_sulfide, min=0.0)
    assert torch.allclose(out[..., 1], expected_sulfide)


def test_the_outfall_pipe_is_found_by_where_it_drains_not_by_position():
    """M4-R19 Important 1: the brief's dictated `outfall_manhole = net.pipes[-1].u`
    assumed the LAST pipe drains to the outfall. Moving C5 (J4 -> Outfall) to the FRONT of
    `net.pipes` used to attach the outfall-flagged `headspace` edge (and the geometry it
    borrows) to J3 -- the upstream end of whatever pipe happened to be last, C3 (J3 -> J4)
    -- instead of J4. Fixed by finding the pipe whose `v` is an outfall, wherever it sits;
    both orderings must now attach the outfall edge to J4 and produce identical air-side
    steady results, compared by node NAME (node order is the same for both builds -- only
    `net.pipes`' order differs -- so this also incidentally checks position-for-position,
    but the comparison is done by name as asked)."""
    canonical = tree_steady()
    reordered = SewerNetwork(
        manholes=canonical.manholes,
        pipes=(canonical.pipes[-1], *canonical.pipes[:-1]),
        outfalls=canonical.outfalls,
    )
    reordered.validate()
    assert [p.name for p in reordered.pipes] == ["C5", "C1", "C2", "C4", "C3"]

    model_a, state_a, drivers_a = build_model(canonical)
    model_b, state_b, drivers_b = build_model(reordered)
    assert model_a.net.nodes == model_b.net.nodes

    def outfall_edge_names(model):
        src, tgt = model.net.endpoints("headspace")
        flagged = model.net.edge_attr("outfall", kind="headspace", default=0.0)
        (idx,) = flagged.nonzero().flatten().tolist()
        return model.net.nodes[int(src[idx])], model.net.nodes[int(tgt[idx])]

    assert outfall_edge_names(model_a) == ("J4", "ambient")
    assert outfall_edge_names(model_b) == ("J4", "ambient")

    new_a = model_a.step(state_a, drivers_a, 60.0)
    new_b = model_b.step(state_b, drivers_b, 60.0)
    phi_a = dict(zip(model_a.net.nodes, new_a["air.phi"].tolist(), strict=True))
    phi_b = dict(zip(model_b.net.nodes, new_b["air.phi"].tolist(), strict=True))
    assert set(phi_a) == set(phi_b)
    for name in phi_a:
        assert phi_a[name] == pytest.approx(phi_b[name], abs=1e-10), name


def test_the_headspace_stack_drive_uses_the_pipe_crown_not_the_mean_invert():
    """M4-R19 Important 2: `_stack_for`'s `headspace` branch must use the pipe CROWN
    elevation (mean invert plus diameter), not the bare mean invert, as `z_path`. Built
    with `T_head != T_amb` (the builder's own defaults, 293.15 K vs 283.15 K) so the drive
    carries a non-trivial value, and checked against the drive's own formula
    (`drives.Stack`'s docstring) evaluated by hand with the crown `z_path`."""
    net = SewerNetwork(
        manholes=(Manhole("J1", 12.0),),
        pipes=(Pipe("A", "J1", "Out", 100.0, 0.3, 0.013, 0.01),),
        outfalls=(Outfall("Out", 9.0),),
    )
    model, state, drivers = build_model(net, quality=False)
    assert float(drivers["T_head"]) != float(drivers["T_amb"])
    resolved = model._apply_closures(state, drivers)

    stack = next(
        d for d in model.potential["air"]._drives
        if isinstance(d, Stack) and d.kind == "headspace"
    )
    value = stack(resolved)

    graph = model.net
    invert = torch.tensor(
        [graph.graph.nodes[n].get("invert", 0.0) or 0.0 for n in graph.nodes], dtype=F64
    )
    ground = torch.tensor(
        [
            graph.graph.nodes[n].get("ground") or graph.graph.nodes[n].get("invert", 0.0)
            or 0.0
            for n in graph.nodes
        ],
        dtype=F64,
    )
    src, tgt = graph.endpoints("headspace")
    diameter = net.pipes[0].diameter
    z_path = 0.5 * (invert[src] + invert[tgt]) + diameter
    rho = resolved["rho_air_nodes"]
    expected = 9.80665 * (
        rho[..., src] * (ground[src] - z_path) - rho[..., tgt] * (ground[tgt] - z_path)
    )
    assert torch.allclose(value, expected)


# ------------------------------------------------------------------------------------ N2


def test_the_leak_element_is_built_float64_not_the_default_dtype():
    """N2: the leak's `PowerLaw` is built DIRECTLY (never through `Orifice`, which casts
    with `torch.get_default_dtype()` -- float32 in this repository -- regardless of its
    inputs' own dtype); both `C` and `n` must be float64."""
    model, _, _ = build_model(tree_steady())
    leak = next(el for el in model.potential["air"]._elements if el.kind == "leak")
    assert leak.C.dtype == torch.float64
    assert leak.n.dtype == torch.float64


# ------------------------------------------------------------------------------------ N4


def _two_component_forest():
    return SewerNetwork(
        manholes=(Manhole("A", invert=10.0, inflow=0.05), Manhole("B", invert=8.0, inflow=0.03)),
        pipes=(Pipe("PA", "A", "OutA", 100.0, 0.30, 0.013, 0.01),
               Pipe("PB", "B", "OutB", 100.0, 0.30, 0.013, 0.01)),
        outfalls=(Outfall("OutA", 5.0), Outfall("OutB", 3.0)),
    )


def test_a_two_component_forest_builds_and_steps_with_air_false():
    net = _two_component_forest()
    model, state, drivers = build_model(net, air=False, quality=False)
    resolved = model._apply_closures(state, drivers)
    assert resolved["sewer.q"].tolist() == pytest.approx([0.05, 0.03], abs=1e-15)


def test_a_two_component_forest_builds_and_steps_with_air_true():
    """N4: `build_model(air=True)` used to refuse any network whose pipe subgraph
    was not a single tree with exactly one outfall pipe. A forest gets one outfall-to-
    ambient headspace edge PER outfall pipe instead, each borrowing its own pipe's length
    and diameter, with the drive zeroed on both appended edges."""
    net = _two_component_forest()
    model, state, drivers = build_model(net, air=True, quality=True)
    assert model.net.edge_index("headspace").numel() == 4  # 2 pipes + 2 outfall edges
    new = model.step(state, drivers, 60.0)
    residuals = model.residuals(new, drivers)
    assert float(residuals["air"].abs().max()) < 1e-9


def test_two_manholes_draining_into_one_outfall_is_still_refused_by_name():
    """The forest generalisation (N4) must not silently accept two DIFFERENT manholes
    draining directly into the SAME outfall (M4-R19's own accepted deviation)."""
    net = SewerNetwork(
        manholes=(Manhole("A", invert=10.0), Manhole("B", invert=9.0)),
        pipes=(Pipe("PA", "A", "Out", 100.0, 0.30, 0.013, 0.01),
               Pipe("PB", "B", "Out", 100.0, 0.30, 0.013, 0.01)),
        outfalls=(Outfall("Out", 5.0),),
    )
    with pytest.raises(ValueError, match=r"more than one pipe.*draining"):
        build_model(net, air=True)


# ------------------------------------------------------------------------------------ N8


def test_a_ground_level_of_exactly_zero_is_not_dropped():
    """N8: `_stack_for` used an `or`-chain (`node.get("ground") or node.get("invert", 0.0)
    or 0.0`), which treats an explicitly given `ground=0.0` as falsy and falls through to
    the invert instead. A manhole with `invert=-5.0, ground=0.0` must give the leak's own
    `z_path` (the manhole's ground, `_stack_for`'s `kind='leak'` branch) as exactly 0.0,
    not -5.0."""
    net = SewerNetwork(
        manholes=(Manhole("A", invert=-5.0, ground=0.0),),
        pipes=(Pipe("PA", "A", "Out", 100.0, 0.30, 0.013, 0.01),),
        outfalls=(Outfall("Out", -10.0),),
    )
    model, state, drivers = build_model(net, quality=False)
    leak_stack = next(
        d for d in model.potential["air"]._drives
        if isinstance(d, Stack) and d.kind == "leak"
    )
    src, _ = model.net.endpoints("leak")
    manhole_node = model.net.nodes.index("A")
    (pos,) = (src == manhole_node).nonzero().flatten().tolist()
    assert float(leak_stack.z_path[pos]) == 0.0


# ----------------------------------------------------------------------------- N10, N11


def test_model_notes_reflects_the_closures_own_notes_dict():
    """N10: `model.notes` is the SAME dict object as the hydraulics closure's own `notes`,
    not a one-time copy taken at build time, so a note the closure adds only once a step
    actually hits it (`capacity_floor`, added lazily on a dry pipe) shows up on
    `model.notes` too. R5: `build_model`'s initial-storage query
    (`Model.initial_capacities`) already evaluates the closure once at construction, so on a
    network that is dry from the start the `capacity_floor` note can already be present
    before the first `model.step` call -- this test therefore checks the identity through the
    note surviving (and staying legible) across a step, not through its absence beforehand."""
    net = SewerNetwork(
        manholes=(Manhole("A", invert=5.0, inflow=0.0),),
        pipes=(Pipe("PA", "A", "Out", 100.0, 0.3, 0.013, 0.01),),
        outfalls=(Outfall("Out", 0.0),),
    )
    model, state, drivers = build_model(net, air=False, quality=True)
    model.step(state, drivers, 60.0)
    assert "capacity_floor" in model.notes
    assert "PA" in model.notes["capacity_floor"]


def test_fr21_one_step_with_a_lateral_bod_load_raises_only_j1():
    """FR-21 (a): `bod_in = 0.3` at J1 only. `LateralLoads`'s own contract puts the
    resulting source at EXACTLY J1's row and EXACTLY the BOD column (verified directly at
    the closure level in `test_quality.py`); this test verifies the FULL WIRING through
    `build_model` and one `model.step`.

    J1 is a HEADWATER manhole (no upstream inflow, only its own lateral source), so its own
    concentration after one implicit step of the water_quality transport layer is the
    closed form for that case: `x = source * dt / (V_wet + Q_out * dt)`
    (`source = inflow * bod_in`, `Q_out` = J1's own outgoing pipe discharge) -- NOT the
    brief's own first-order-in-dt sketch `inflow * bod_in * dt / V_wet`, which is the SAME
    formula only in the limit `Q_out * dt << V_wet`; here `Q_out * dt / V_wet = 0.414`, so
    that limit does not hold and the naive figure is measured 41% off. MEASURED: the
    transport-layer-only step (bypassing `SulfideGeneration`'s reaction, verified separately
    at S1) matches the closed form above to rel 0.0 (bit-exact -- it is that layer's OWN
    scheme, not an approximation of it)."""
    model, state, drivers = build_model(tree_steady())
    n = model.net.n
    j1_node = int(model.manhole_idx[0])
    drivers = dict(drivers)
    bod_in = torch.zeros(n, dtype=F64)
    bod_in[j1_node] = 0.3
    drivers["bod_in"] = bod_in
    resolved = model._apply_closures(state, drivers)
    sources = resolved["water_quality.sources"]

    other_mask = torch.ones(n, dtype=torch.bool)
    other_mask[j1_node] = False
    assert torch.equal(sources[other_mask], torch.zeros_like(sources[other_mask]))
    assert float(sources[j1_node, 0]) == pytest.approx(0.05 * 0.3, rel=1e-12)
    assert float(sources[j1_node, 1]) == 0.0

    layer = model.transport["water_quality"]
    q = resolved["water_quality.q"]
    cap = resolved["water_quality.capacity"]
    new_x = layer.step(
        state["water_quality.x"], q, sources, drivers["water_quality.x_boundary"], 60.0,
        capacity=cap,
    )
    v_wet = float(cap[0])
    q_out = float(q[int(model.out_pipe[0])])
    source = 0.05 * 0.3
    expected = source * 60.0 / (v_wet + q_out * 60.0)
    assert float(new_x[0, 0]) == pytest.approx(expected, rel=1e-12)


def test_initial_state_builds_sewer_h_and_a_storage_step_runs():
    """N11: `initial_state(model)` must build `"sewer.H"` itself when `storage=True`, as
    `SewerHydraulics`'s own `KeyError` message already promises."""
    model, _, drivers = build_model(tree_steady(), storage=True, dt_storage=60.0)
    state = initial_state(model, drivers)
    assert "sewer.H" in state
    new = model.step(state, drivers, 60.0)
    assert torch.isfinite(new["sewer.H"]).all()
