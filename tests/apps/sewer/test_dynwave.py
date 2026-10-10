import dataclasses
from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp
from noodl.apps.sewer.dynwave import (
    FUDGE_FLOW,
    SOLVE_ATOL,
    DynamicWaveConduit,
    JunctionStorage,
    simulate_dynamic_wave,
    solve_atol,
)
from noodl.apps.sewer.xsect_tables import LCF, QCF, area_of_depth

F64 = torch.float64
DATA = Path(__file__).resolve().parents[2] / "data" / "sewer"


def _tree(tmp_path, scale=1.0):
    text = (DATA / "tree_kinwave.inp").read_text()
    text = text.replace("FLOW_ROUTING         KINWAVE", "FLOW_ROUTING         DYNWAVE")
    text = text.replace("VARIABLE_STEP        0.75", "VARIABLE_STEP        0")
    path = tmp_path / "tree_dyn.inp"
    path.write_text(text)
    net, _, _ = read_swmm_inp(path)
    if scale != 1.0:
        net = dataclasses.replace(net, manholes=tuple(
            dataclasses.replace(m, inflow=m.inflow * scale) for m in net.manholes))
    return net


def test_conduit_element_is_linear_then_limited_then_capped():
    el = DynamicWaveConduit()
    drv = {"water.flow_offset": torch.tensor([0.1, 0.1, 0.1], dtype=F64),
           "water.conveyance": torch.tensor([2.0, 2.0, 2.0], dtype=F64),
           "water.normal_flow": torch.tensor([float("inf"), 0.15, float("inf")], dtype=F64),
           "water.limited": torch.tensor([0.0, 1.0, 0.0], dtype=F64),
           "water.dry_upstream": torch.tensor([0.0, 0.0, 1.0], dtype=F64),
           "water.dry_downstream": torch.zeros(3, dtype=F64)}
    dp = torch.tensor([0.05, 0.05, 0.05], dtype=F64)
    q = el.flow(dp, drv)
    torch.testing.assert_close(q, torch.tensor([0.2, 0.15, FUDGE_FLOW], dtype=F64))
    torch.testing.assert_close(el.dflow(dp, drv), torch.tensor([2.0, 0.0, 0.0], dtype=F64))


def test_junction_storage_is_trapezoidal_continuity_with_a_stiff_bed():
    """The bed stiffness uses the fixed `bed_area` (SWMM's surface-area floor), not the
    pass's surface area, so its round-off floor is known before the step."""
    ns = JunctionStorage(torch.tensor([0, 1]), bed=torch.tensor([1.0, 1.0], dtype=F64),
                         bed_area=0.5)
    drv = {"water.storage_area": torch.tensor([2.0, 2.0], dtype=F64),
           "water.head_start": torch.tensor([1.5, 1.0], dtype=F64),
           "water.net_inflow_start": torch.tensor([0.3, 0.0], dtype=F64),
           "water.storage_rate": torch.tensor(0.2, dtype=F64)}       # dt = 5 s
    phi = torch.tensor([1.6, 0.999], dtype=F64)
    w = ns.flow(phi, drv)
    assert w[0].item() == pytest.approx(2 * 2.0 * 0.2 * 0.1 - 0.3)
    assert w[1].item() == pytest.approx(2 * 0.2 * (2.0 + 0.5 * 1e6) * (-0.001))
    torch.testing.assert_close(
        ns.dflow(phi, drv),
        torch.tensor([2 * 0.2 * 2.0, 2 * 0.2 * (2.0 + 0.5 * 1e6)], dtype=F64))


def test_a_junction_on_its_bed_converges_at_the_default_min_surface_area():
    """Regression: with SWMM's default MIN_SURFAREA (12.566 ft2) J3 of the tree sits a few
    ulp below its invert on the first step, where the bed slope (2 A_s/dt)(1 + K_BED) times
    half an ulp of the head (1.9e-9 m3/s, A_s = 5.5 m2) exceeded the fixed 1e-12 m3/s Newton
    tolerance, and the first step failed. The bed slope now uses the fixed surface-area
    floor and the default tolerance sits above its round-off floor."""
    net, _, _ = read_swmm_inp(DATA / "tree_dynwave.inp")
    assert net.dynamic_wave.min_surface_area is None
    model, state, drivers = build_model(net)
    state, hist = simulate_dynamic_wave(model, state, drivers, steps=2)
    assert torch.isfinite(hist["water.phi"]).all()
    assert solve_atol(model) > SOLVE_ATOL


def test_default_pass_budget_covers_a_slow_outfall_head():
    """The loop fixture's outfall head contracts by ~0.92 per pass at low flow and needs 327
    passes at t = 500 s (tests/verification/test_sewer_dynwave_parity.py); the default
    budget must cover that, so a plain build_model + simulate_dynamic_wave does not fail."""
    net, _, _ = read_swmm_inp(DATA / "loop_dynwave.inp")
    model, _, _ = build_model(net)
    assert model.iterate_max == 2000


def test_build_model_dispatches_and_refuses_air_and_quality(tmp_path):
    net = _tree(tmp_path)
    model, state, drivers = build_model(net)
    assert "water" in model.potential and model.potential["water"].quantity == "head"
    with pytest.raises(ValueError, match="air"):
        build_model(net, air=True)
    with pytest.raises(ValueError, match="quality"):
        build_model(net, air=False, quality=True)


def test_cold_start_is_dry_at_the_inverts(tmp_path):
    model, state, drivers = build_model(_tree(tmp_path))
    junction = model.junction_idx
    torch.testing.assert_close(state["water.phi"][junction], model.junction_invert)
    assert torch.all(state["sewer.junction_depth"] == 0)
    # SWMM zeroes the lateral inflow before routing (node.c:264, flowrout.c:437), so the
    # first step's oldNetInflow is 0
    assert torch.all(state["sewer.net_inflow"] == 0)


def test_cold_start_sets_fixed_outfall_stage_and_its_conduit_depth():
    """SWMM's cold start (flowrout.c:336-420): outfall depths from the zero flow
    (link_setOutfallDepth), so a FIXED outfall starts at its stage; then each conduit's depth
    is the mean of its end-node depths above its offsets, which sets the stored area that the
    first step's inertial term dq3 = 2 v (a_mid - a_old) sees. Backwater: Out 9.60 m, stage
    9.85 m, so C3 (J3 9.80 m dry -> Out, 0.45 m) starts at (0 + 0.25)/2 = 0.125 m."""
    net, _, _ = read_swmm_inp(DATA / "backwater_dynwave.inp")
    model, state, drivers = build_model(net)
    hyd = model.dynamic_wave
    out = hyd.outfall_idx
    torch.testing.assert_close(state["water.phi"][out], torch.tensor([9.85], dtype=F64))
    torch.testing.assert_close(drivers["water.phi_boundary"], torch.tensor([9.85], dtype=F64))
    c3 = model.pipe_names.index("C3")
    y_ft = torch.tensor([0.125 / LCF], dtype=F64)
    a_si = area_of_depth(y_ft, hyd.geo.y_full[c3:c3 + 1], hyd.geo.a_full[c3:c3 + 1]) * QCF / LCF
    torch.testing.assert_close(state["sewer.mid_area"][c3:c3 + 1], a_si)
    c1 = model.pipe_names.index("C1")
    fudge = area_of_depth(torch.tensor([1e-4], dtype=F64), hyd.geo.y_full[c1:c1 + 1],
                          hyd.geo.a_full[c1:c1 + 1]) * QCF / LCF
    torch.testing.assert_close(state["sewer.mid_area"][c1:c1 + 1], fudge)


def test_first_step_fills_a_dry_junction_by_half_the_inflow(tmp_path):
    """J1 drains through C1 with a 1 m inlet offset, which the first step's water does not
    reach (C1 stays DRY, q = 0), and J1's area is the floor: A_s y = 0.5 dt (0 + Q)."""
    text = (DATA / "tree_kinwave.inp").read_text()
    text = text.replace("FLOW_ROUTING         KINWAVE", "FLOW_ROUTING         DYNWAVE")
    text = text.replace("VARIABLE_STEP        0.75", "VARIABLE_STEP        0")
    old = "C1               J1               J3               200        0.013      0 "
    assert old in text
    text = text.replace(old, "C1               J1               J3               200        "
                             "0.013      1.0 ")
    path = tmp_path / "offset.inp"
    path.write_text(text)
    net, _, _ = read_swmm_inp(path)
    model, state, drivers = build_model(net)
    hyd = model.dynamic_wave
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=1)
    j1 = hyd.junction_names.index("J1")
    q_in = drivers["inflow"][model.junction_idx[j1]].item()
    from noodl.apps.sewer.xsect_tables import LCF, QCF
    floor = hyd.min_area_ft2 * QCF / LCF                            # SI area, design D2
    expected = 0.5 * model.routing_step * q_in / floor
    assert state["water.q"][hyd.pipe_names.index("C1")].item() == 0.0
    assert state["sewer.junction_depth"][j1].item() == pytest.approx(expected, rel=1e-12)


def test_gradient_over_a_few_steps_from_the_cold_start(tmp_path):
    """Fast gradient check: d(sum of junction depths after 4 steps)/d(Manning n of C1),
    against central differences. C1 is wet from the first step (J1 carries inflow)."""
    model, state0, drivers = build_model(_tree(tmp_path))
    hyd = model.dynamic_wave
    i = hyd.pipe_names.index("C1")

    def loss(n1):
        hyd.set_roughness(i, n1)
        s, _ = simulate_dynamic_wave(model, state0, drivers, steps=4)
        return (s["water.phi"][model.junction_idx] - model.junction_invert).sum()

    n = torch.tensor(0.013, dtype=F64, requires_grad=True)
    loss(n).backward()
    h = 1e-7
    with torch.no_grad():
        fd = (loss(n.detach() + h) - loss(n.detach() - h)) / (2 * h)
    assert n.grad.item() != 0.0
    assert n.grad.item() == pytest.approx(fd.item(), rel=1e-5)


@pytest.mark.slow
def test_steady_state_passes_the_total_inflow(tmp_path):
    """1 h at 5 s, at HALF the fixture's inflows: at full inflow (0.16 m3/s in C5, normal
    depth 0.303 m) the water at J4 reaches the 0.30 m crown of C4, which joins J4 with a
    zero offset, and the run is refused as surcharge at t = 780 s -- the physics, not a
    defect. Half the inflow (normal depth 0.197 m in C5) stays free-surface."""
    net = _tree(tmp_path, scale=0.5)
    model, state, drivers = build_model(net)
    state, hist = simulate_dynamic_wave(model, state, drivers, steps=720)
    out = model.outfall_pipe_idx
    total = sum(m.inflow for m in net.manholes)
    assert hist["water.q"][-1, out].sum().item() == pytest.approx(total, rel=1e-6)


@pytest.mark.slow
def test_outfall_flow_rises_monotonically_towards_the_total(tmp_path):
    """The fast variant of the steady-state test: 10 minutes at half the inflows."""
    net = _tree(tmp_path, scale=0.5)
    model, state, drivers = build_model(net)
    state, hist = simulate_dynamic_wave(model, state, drivers, steps=120)
    q = hist["water.q"][:, model.outfall_pipe_idx].sum(-1)
    total = sum(m.inflow for m in net.manholes)
    assert bool((q[1:] >= q[:-1]).all())
    assert 0.6 * total < q[-1].item() <= total       # 0.736 of the total, measured


def test_each_step_satisfies_the_discrete_junction_equation(tmp_path):
    """A_s (y - y_start) = 0.5 dt (N_start + N) at every junction, on the returned state.

    The head solve is held to `SOLVE_ATOL` here: this fixture's MIN_SURFAREA is 12.566 m2
    (not ft2), so the default `solve_atol` -- raised to the stiff bed's round-off floor,
    which scales with that area -- would be 2.7e-8 m3/s, i.e. a residual up to 7e-8 m3. No
    junction of this run sits on its bed, so the tight tolerance is reachable."""
    model, state, drivers = build_model(_tree(tmp_path))
    for _ in range(40):
        before = dict(state)
        state, _ = simulate_dynamic_wave(model, state, drivers, steps=1, atol=SOLVE_ATOL)
        res = model.dynamic_wave.junction_residual(before, state, drivers)
        assert res.abs().max().item() < 1e-9


def test_dry_network_stays_finite(tmp_path):
    model, state, drivers = build_model(_tree(tmp_path, scale=0.0))
    state, hist = simulate_dynamic_wave(model, state, drivers, steps=20)
    assert torch.isfinite(hist["water.phi"]).all() and torch.isfinite(hist["water.q"]).all()
    depth = hist["water.phi"][:, model.junction_idx] - model.junction_invert
    assert depth.min().item() > -1e-9


def test_surcharge_is_refused_by_name(tmp_path):
    model, state, drivers = build_model(_tree(tmp_path, scale=4.0))
    with pytest.raises(RuntimeError, match=r"surcharge.*C\d"):
        simulate_dynamic_wave(model, state, drivers, steps=720)


def test_non_convergence_is_named(tmp_path):
    net = _tree(tmp_path)
    model, state, drivers = build_model(net, iterate_max=2, iterate_tol=1e-15)
    with pytest.raises(RuntimeError, match=r"^dynamic wave step at t=0\.0 s: .*did not converge"):
        simulate_dynamic_wave(model, state, drivers, steps=3)


def test_first_step_of_a_dry_network_moves_nothing(tmp_path):
    model, state, drivers = build_model(_tree(tmp_path, scale=0.0))
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=1)
    assert torch.all(state["water.q"] == 0)


@pytest.mark.slow
def test_batched_equals_single(tmp_path):
    net = _tree(tmp_path)
    model, state, drivers = build_model(net)
    base = drivers["inflow"]
    batch = torch.stack([base, 0.5 * base])
    _, hb = simulate_dynamic_wave(model, state, drivers, steps=30, inflow=batch)
    for i, scale in enumerate((1.0, 0.5)):
        _, hs = simulate_dynamic_wave(model, state, drivers, steps=30, inflow=scale * base)
        torch.testing.assert_close(hb["water.phi"][:, i], hs["water.phi"],
                                   rtol=1e-9, atol=1e-10)


@pytest.mark.slow
def test_gradient_through_the_steps_matches_central_differences(tmp_path):
    """d(sum of junction depths after 60 steps)/d(Manning n of C3): wet, subcritical regime
    from 300 s on (no flow-class switch inside the difference stencil)."""
    net = _tree(tmp_path)
    model, state, drivers = build_model(net)
    hyd = model.dynamic_wave
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=60)   # wet it first
    i = hyd.pipe_names.index("C3")

    def loss(n3):
        hyd.set_roughness(i, n3)
        s, _ = simulate_dynamic_wave(model, state, drivers, steps=60)
        return (s["water.phi"][model.junction_idx] - model.junction_invert).sum()

    n = torch.tensor(0.013, dtype=F64, requires_grad=True)
    loss(n).backward()
    h = 1e-7
    with torch.no_grad():
        fd = (loss(n.detach() + h) - loss(n.detach() - h)) / (2 * h)
    assert n.grad.item() == pytest.approx(fd.item(), rel=1e-5)


def test_initial_state_and_pipe_table_serve_a_dynamic_wave_model(tmp_path):
    from noodl.apps.sewer import initial_state, pipe_table

    model, state, drivers = build_model(_tree(tmp_path))
    fresh = initial_state(model, drivers)
    assert set(fresh) == set(state)
    assert set(initial_state(model)) == set(state)
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=2)
    rows = pipe_table(model, state, drivers)
    assert [r["pipe"] for r in rows] == model.pipe_names
    assert set(rows[0]) == {"pipe", "q", "h", "v", "V_wet"}


def test_kinematic_routing_refuses_the_dynamic_wave_iteration_controls():
    from noodl.apps.sewer import tree_steady

    with pytest.raises(ValueError, match="iterate_tol"):
        build_model(tree_steady(), iterate_tol=1e-9)
