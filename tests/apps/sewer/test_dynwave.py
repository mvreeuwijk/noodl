import dataclasses
import re
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
           "water.dry_downstream": torch.zeros(3, dtype=F64),
           "water.slope_switch": torch.zeros(3, dtype=F64),
           "water.switch_drop": torch.zeros(3, dtype=F64),
           "water.switch_slope": torch.zeros(3, dtype=F64),
           "water.flow_last": torch.zeros(3, dtype=F64)}
    dp = torch.tensor([0.05, 0.05, 0.05], dtype=F64)
    q = el.flow(dp, drv)
    torch.testing.assert_close(q, torch.tensor([0.2, 0.15, FUDGE_FLOW], dtype=F64))
    torch.testing.assert_close(el.dflow(dp, drv), torch.tensor([2.0, 0.0, 0.0], dtype=F64))


def _switch_drivers(q_last):
    """One conduit on the in-pass slope switch: lin = 0.1 + 2 dp, q_norm = 0.15, switch at
    dp = 0.05 (lin there 0.2), wall slope 100, fed flow `q_last` (m3/s)."""
    one = torch.ones(1, dtype=F64)
    return {"water.flow_offset": 0.1 * one, "water.conveyance": 2.0 * one,
            "water.normal_flow": 0.15 * one, "water.limited": 0.0 * one,
            "water.slope_switch": one, "water.switch_drop": 0.05 * one,
            "water.switch_slope": 100.0 * one, "water.flow_last": q_last * one,
            "water.dry_upstream": 0.0 * one, "water.dry_downstream": 0.0 * one}


def test_conduit_element_slides_on_the_slope_switch():
    """The in-pass slope switch ``min(lin, max(q_norm, q_last + S (dp - dz)))``: SWMM's
    unlimited flow above the switch and its normal-flow limit well below it, a steep wall
    between; continuous and non-decreasing in dp. Its fixed points (q = q_last) are SWMM's
    own on either side, plus the switching surface dp = dz with q between q_norm and lin."""
    el = DynamicWaveConduit()
    drv = _switch_drivers(0.17)
    dp = torch.tensor([0.06, 0.05, 0.0499, 0.04], dtype=F64)
    drv4 = {k: v.expand(4) for k, v in drv.items()}
    q = el.flow(dp, drv4)
    torch.testing.assert_close(q, torch.tensor([0.22, 0.17, 0.16, 0.15], dtype=F64))
    torch.testing.assert_close(el.dflow(dp, drv4),
                               torch.tensor([2.0, 100.0, 100.0, 0.0], dtype=F64))
    grid = torch.linspace(0.0, 0.1, 2001, dtype=F64)
    qs = el.flow(grid, {k: v.expand(2001) for k, v in drv.items()})
    assert bool((qs[1:] - qs[:-1] >= 0.0).all())
    # fixed points: on the surface for any q_last in [q_norm, lin(dz)] ...
    for q_last in (0.15, 0.17, 0.2):
        assert el.flow(torch.tensor([0.05], dtype=F64), _switch_drivers(q_last)).item() == \
            pytest.approx(q_last, abs=1e-15)
    # ... SWMM's unlimited one above it, and its limited one below it, unchanged
    assert el.flow(torch.tensor([0.07], dtype=F64), _switch_drivers(0.24)).item() == \
        pytest.approx(0.24)
    assert el.flow(torch.tensor([0.02], dtype=F64), _switch_drivers(0.14)).item() == \
        pytest.approx(0.14)                     # lin below q_norm: lin
    assert el.flow(torch.tensor([0.04], dtype=F64), _switch_drivers(0.15)).item() == \
        pytest.approx(0.15)                     # the limit


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
    against central differences. C1 is wet from the first step (J1 carries inflow).

    The gradient is the default (Anderson-accelerated) model's; the central differences
    are taken on plain relaxation. The adjoint is the fixed point's own, the same either
    way (1.88181489e-2 vs 1.88181502e-2), but the primal is resolved only to the head
    solve's band (a pass whose fed heads already meet `solve_atol` returns them unchanged,
    ~5e-9 m): where in that band an iteration lands drifts smoothly with n, and along the
    Anderson path that drift biases the differences by 1.4e-5 relative at every h from
    1e-7 to 1e-5, against 2.6e-6 along the relaxed path."""
    from noodl.apps.sewer.dynwave import build_dynamic_wave_model

    net = _tree(tmp_path)
    model, state0, drivers = build_model(net)
    plain = build_dynamic_wave_model(net, acceleration=None)[0]

    def loss(m, n1):
        hyd = m.dynamic_wave
        hyd.set_roughness(hyd.pipe_names.index("C1"), n1)
        s, _ = simulate_dynamic_wave(m, state0, drivers, steps=4)
        return (s["water.phi"][m.junction_idx] - m.junction_invert).sum()

    n = torch.tensor(0.013, dtype=F64, requires_grad=True)
    loss(model, n).backward()
    h = 1e-7
    with torch.no_grad():
        fd = (loss(plain, n.detach() + h) - loss(plain, n.detach() - h)) / (2 * h)
    assert n.grad.item() != 0.0
    assert n.grad.item() == pytest.approx(fd.item(), rel=1e-5)


@pytest.mark.slow
def test_steady_state_passes_the_total_inflow(tmp_path):
    """1 h at 5 s, at HALF the fixture's inflows (normal depth 0.197 m in C5), which stays
    free-surface throughout: at full inflow (0.16 m3/s in C5, normal depth 0.303 m) the
    water at J4 reaches the 0.30 m crown of C4, which joins J4 with a zero offset."""
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


def _passes_and_history(name, steps, solve_kwargs=None, **kwargs):
    from noodl.apps.sewer.dynwave import build_dynamic_wave_model

    net, _, _ = read_swmm_inp(DATA / name)
    model, state, drivers = build_dynamic_wave_model(net, **kwargs)
    diags: list = []
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=steps, diagnostics=diags,
                                    **(solve_kwargs or {}))
    return sum(d["passes"] for d in diags) / steps, hist


#: Plain relaxation (heads and flows judged, as required since the slope switch).
_PLAIN = {"acceleration": None}


def test_anderson_is_the_default_and_halves_the_passes_on_the_tree():
    """Design E8: Anderson mixing of the head iteration at least halves the mean passes per
    step (measured over 60 steps: 35.2 -> 14.1, plain relaxation judging heads and flows;
    34.8 when it judged the heads alone, which is now refused).

    The two runs agree to the head solve's own resolution, not to 1e-10 m: Newton stops
    once a junction's balance is within `solve_atol` (2.5e-9 m3/s here), so a pass whose
    fed heads already meet it hands them back unchanged, and EACH iteration lands
    somewhere within ``atol dt / (2 A_s)`` (~5e-9 m) of the exact pass solution. Measured
    2.0e-9 m and 4.9e-10 m3/s over 60 steps;
    `test_anderson_lands_on_the_relaxed_fixed_point_with_a_tight_head_solve` shows the
    two fixed points agree to 2e-12 m once the head solve is tight."""
    from noodl.apps.sewer.dynwave import build_dynamic_wave_model

    net, _, _ = read_swmm_inp(DATA / "tree_dynwave.inp")
    assert build_dynamic_wave_model(net)[0].iterate_acceleration == "anderson"
    fast, h_fast = _passes_and_history("tree_dynwave.inp", 30)
    plain, h_plain = _passes_and_history("tree_dynwave.inp", 30, **_PLAIN)
    assert fast <= plain / 2.0, (fast, plain)
    torch.testing.assert_close(h_fast["water.phi"], h_plain["water.phi"], rtol=0.0, atol=1e-8)
    torch.testing.assert_close(h_fast["water.q"], h_plain["water.q"], rtol=0.0, atol=2e-9)


def test_anderson_lands_on_the_relaxed_fixed_point_with_a_tight_head_solve():
    """Design E8: Anderson mixing moves no fixed point. With the head solve tight (atol
    1e-12 m3/s, so no pass stops on its warm start) and both iterations judging heads to
    1e-13 m and flows to 1e-12 m3/s, the looped network's two runs agree to 1e-10 m (1.9e-12
    m measured over 60 steps). The tree cannot be run this way: a junction sitting on its
    stiff bed has a residual floor above 1e-12 m3/s (`solve_atol`)."""
    tight = {"iterate_tol": 1e-13, "iterate_flow_tol": 1e-12, "iterate_max": 5000}
    fast, h_fast = _passes_and_history("loop_dynwave.inp", 20, {"atol": 1e-12}, **tight)
    plain, h_plain = _passes_and_history("loop_dynwave.inp", 20, {"atol": 1e-12},
                                         acceleration=None, **tight)
    assert fast <= plain / 2.0, (fast, plain)
    torch.testing.assert_close(h_fast["water.phi"], h_plain["water.phi"], rtol=0.0,
                               atol=1e-10)


@pytest.mark.slow
def test_anderson_halves_the_passes_on_the_loop():
    """Over the first 120 steps, through the slowly contracting outfall head near t = 500 s
    (plain relaxation's worst case, 327 passes at 1e-11 m)."""
    fast, h_fast = _passes_and_history("loop_dynwave.inp", 120)
    plain, h_plain = _passes_and_history("loop_dynwave.inp", 120, **_PLAIN)
    assert fast <= plain / 2.0, (fast, plain)
    # the head solve's resolution, as in the tree test above
    torch.testing.assert_close(h_fast["water.phi"], h_plain["water.phi"], rtol=0.0, atol=1e-8)


def _slot_tree(tmp_path, scale):
    """tree_dynwave.inp with SURCHARGE_METHOD SLOT and the inflows scaled by `scale`."""
    text = (DATA / "tree_dynwave.inp").read_text().replace(
        "ALLOW_PONDING        NO", "ALLOW_PONDING        NO\nSURCHARGE_METHOD     SLOT")
    path = tmp_path / "tree_slot.inp"
    path.write_text(text)
    net, _, _ = read_swmm_inp(path)
    return dataclasses.replace(net, manholes=tuple(
        dataclasses.replace(m, inflow=m.inflow * scale) for m in net.manholes))


def test_slot_surcharges_the_tree_past_the_crown_without_refusal(tmp_path):
    """At 2.5x the fixture's inflows (1.25x tree_kinwave.inp) the water at J4 passes the
    0.30 m crown of C4 at step 95 (475 s) of 120. The brief's 1.5x only reaches 0.25 m at
    J4 over 30 min, so it never surcharges. Under SLOT the pipes stay pressurised, the
    junction equation holds and link depths stay at or below the crown. Under EXTRAN the
    same run fills C4's end at J4 but no junction passes its own crown (J4's is C3's and
    C5's 0.45 m), so nothing is surcharged."""
    net = _slot_tree(tmp_path, 2.5)
    assert net.dynamic_wave.surcharge_method == "slot"
    model, state, drivers = build_model(net)
    state, hist = simulate_dynamic_wave(model, state, drivers, steps=120)
    depth = hist["water.phi"][:, model.junction_idx] - model.junction_invert
    j4 = net.manholes.index(next(m for m in net.manholes if m.name == "J4"))
    assert depth[:, j4].max().item() > 0.30                  # C4's 0.30 m crown
    assert torch.isfinite(hist["water.phi"]).all() and torch.isfinite(hist["water.q"]).all()
    # link depth stays min(y_mid, y_full); full pipes hold more than their full volume
    terms = model.dynamic_wave._terms(state, net.dynamic_wave.routing_step)
    diam = torch.tensor([p.diameter for p in net.pipes], dtype=F64) / LCF
    assert bool((terms.depth <= diam + 1e-12).all())
    assert bool((torch.maximum(terms.end_depth1, terms.end_depth2) > diam).any())
    # the same run under EXTRAN: C4 full at J4, no junction above its crown
    extran = dataclasses.replace(net, dynamic_wave=dataclasses.replace(
        net.dynamic_wave, surcharge_method="extran"))
    m2, s2, d2 = build_model(extran)
    s2, h2 = simulate_dynamic_wave(m2, s2, d2, steps=120)
    assert _depths(m2, h2)[:, j4].max().item() > 0.30
    assert not bool((_depths(m2, h2) > m2.dynamic_wave.crown_depth).any())


def test_slot_leaves_a_free_surface_run_unchanged(tmp_path):
    """Below the crown cutoff the slot adds nothing: half the fixture's inflows, 20 steps."""
    net = _slot_tree(tmp_path, 0.5)
    extran = dataclasses.replace(net, dynamic_wave=dataclasses.replace(
        net.dynamic_wave, surcharge_method="extran"))
    runs = []
    for n in (net, extran):
        model, state, drivers = build_model(n)
        runs.append(simulate_dynamic_wave(model, state, drivers, steps=20)[1])
    torch.testing.assert_close(runs[0]["water.phi"], runs[1]["water.phi"], rtol=0, atol=1e-10)
    torch.testing.assert_close(runs[0]["water.q"], runs[1]["water.q"], rtol=0, atol=1e-10)


# ---------------------------------------------------------------- EXTRAN surcharge (E1)
EXTRAN_NAME = "tree_extran_dynwave.inp"


def _extran_tree(scale=1.0, **manhole_changes):
    """tree_extran_dynwave.inp (3x tree_dynwave.inp's inflows, 1.5x tree_kinwave.inp's;
    SURCHARGE_METHOD EXTRAN), the inflows scaled by `scale`, manholes replaced field-wise."""
    net, _, _ = read_swmm_inp(DATA / EXTRAN_NAME)
    return dataclasses.replace(net, manholes=tuple(
        dataclasses.replace(m, inflow=m.inflow * scale, **manhole_changes)
        for m in net.manholes))


def _depths(model, hist):
    return hist["water.phi"][..., model.junction_idx] - model.junction_invert


def test_junction_crown_is_the_highest_attached_conduit_top():
    """dynwave.c:137-153: a junction's crown depth is the largest offset + y_full over the
    conduit ends it carries (both ends of every conduit); J4 carries C3 and C5 (0.45 m) and
    C4 (0.30 m). An outlet offset of 0.25 m on C4 lifts J4's crown to 0.55 m."""
    model, _, _ = build_model(_extran_tree())
    hyd = model.dynamic_wave
    crown = dict(zip(hyd.junction_names, hyd.crown_depth.tolist(), strict=True))
    assert crown == pytest.approx({"J1": 0.30, "J2": 0.30, "J5": 0.30, "J3": 0.45,
                                   "J4": 0.45}, abs=1e-12)
    net = _extran_tree()
    net = dataclasses.replace(net, pipes=tuple(
        dataclasses.replace(p, outlet_offset=0.25) if p.name == "C4" else p
        for p in net.pipes))
    hyd = build_model(net)[0].dynamic_wave
    assert hyd.crown_depth[hyd.junction_names.index("J4")].item() == pytest.approx(0.55)


def test_junction_storage_holds_continuity_without_storage_where_surcharged():
    """Design E1 in `JunctionStorage` (crown head 1.3 m, A_s 2 m2, dt 5 s, so 2 A_s/dt =
    0.8 m2/s; the band's stiffness (2/dt) A_bed K_HOLD = 200 m2/s). Junction 0 drains at
    its crown (w_c = 0.8 * 0.3 = 0.24 m3/s), junction 1 would still fill there (w_c = 0.8 *
    0.05 - 0.3 = -0.26 m3/s). The flag comes from the fed head H_last through the band
    b = w_c + 200 (H - 1.3) (above the crown): surcharged junctions withdraw the anchor
    s (H - H_last), 0 at the fed head; the others max(storage, b)."""
    ns = JunctionStorage(torch.tensor([0, 1]), bed=torch.ones(2, dtype=F64), bed_area=0.5,
                         crown=torch.full((2,), 1.3, dtype=F64))
    drv = {"water.storage_area": torch.full((2,), 2.0, dtype=F64),
           "water.head_start": torch.tensor([1.0, 1.25], dtype=F64),
           "water.net_inflow_start": torch.tensor([0.0, 0.3], dtype=F64),
           "water.storage_rate": torch.tensor(0.2, dtype=F64),
           "water.surcharge_slope": torch.full((2,), 0.7, dtype=F64)}

    def flag(h_last):
        drv["water.head_last"] = torch.tensor(h_last, dtype=F64)
        drv["water.surcharged"] = ns.surcharged(drv)
        return drv["water.surcharged"].tolist()

    def at(h):
        phi = torch.full((2,), h, dtype=F64)
        return ns.flow(phi, drv).tolist(), ns.dflow(phi, drv).tolist()

    # fed above the crown: both surcharged (b > 0); 0 withdrawal at the fed head
    assert flag([1.6, 1.6]) == [1.0, 1.0]
    w, dw = at(1.5)
    assert w == pytest.approx([0.7 * -0.1, 0.7 * -0.1]) and dw == pytest.approx([0.7, 0.7])
    assert at(1.6)[0] == [0.0, 0.0]
    # fed at or below the crown: free
    assert flag([1.2, 1.3]) == [0.0, 0.0]
    w, dw = at(1.2)                                              # storage below the crown
    assert w == pytest.approx([0.8 * 0.2, 0.8 * -0.05 - 0.3]) and dw == pytest.approx([0.8, 0.8])
    w, dw = at(1.3 + 1e-6)                                       # the band above it
    assert w == pytest.approx([0.24 + 2e-4, -0.26 + 2e-4])
    assert dw == pytest.approx([200.0, 200.0])
    # fed just above the crown: junction 0 surcharged, junction 1 on its band (b < 0 up to
    # 1.3 m + 0.26/200 = 1.3013 m)
    assert flag([1.3 + 1.0e-6, 1.3 + 1.0e-6]) == [1.0, 0.0]
    assert flag([1.3 + 1.0e-6, 1.3014]) == [1.0, 1.0]


def test_extran_surcharge_is_instantaneous_continuity():
    """Design E1 on the tree at 3x tree_dynwave.inp's inflows (the brief's "1.5x" is
    tree_kinwave.inp's scale; tree_dynwave.inp is half of that). In SWMM J2 passes its
    0.30 m crown at step 21 and J4 its 0.45 m crown at step 139. At every step, each
    junction above its crown has no storage: its net inflow is zero (1e-9 m3/s), while the
    junctions below their crowns keep the trapezoidal storage equation (1e-8 m3). The head
    solve is held to `SOLVE_ATOL`: a surcharged junction has no storage to turn the default
    `solve_atol` (2.5e-9 m3/s here, the stiff bed's round-off floor) into a small head
    error, so its net inflow is resolved only to that atol (1.6e-9 m3/s measured at J4)."""
    net = _extran_tree()
    model, state, drivers = build_model(net)
    hyd = model.dynamic_wave
    crown = hyd.crown_depth
    j2, j4 = hyd.junction_names.index("J2"), hyd.junction_names.index("J4")
    first = {}
    for k in range(150):
        before = dict(state)
        state, _ = simulate_dynamic_wave(model, state, drivers, steps=1, atol=SOLVE_ATOL)
        depth = state["water.phi"][model.junction_idx] - model.junction_invert
        surcharged = depth > crown
        reported = hyd(state, drivers)["sewer.surcharged"]
        assert torch.equal(reported, surcharged.to(F64))
        for j in surcharged.nonzero().flatten().tolist():
            first.setdefault(j, k + 1)
        net_in = state["sewer.net_inflow"]
        assert bool((net_in[surcharged].abs() < 1e-9).all()), (k, net_in[surcharged])
        res = hyd.junction_residual(before, state, drivers, dt=model.routing_step)
        assert res[~surcharged].abs().max().item() < 1e-8, k
    assert first.get(j2) == 21 and first.get(j4) == 139, first


def _ramp(steps_up=40, hold=20, steps_down=10, low=0.8):
    """Inflow factors: up from 0 to 1 over `steps_up` steps, `hold` steps at 1, then down
    to `low` over `steps_down` steps (the lateral inflow at the end of each step)."""
    return ([(k + 1) / steps_up for k in range(steps_up)] + [1.0] * hold
            + [1.0 - (1.0 - low) * (k + 1) / steps_down for k in range(steps_down)])


def test_crown_crossing_converges_or_is_named():
    """A ramp of inflow up through J2's crown and back down (tree_extran_dynwave.inp's
    inflows times 0..1, then down to 0.8). J2 jumps above its 0.30 m crown at step 47 and
    falls back through it at step 66: there the storage equation from the step start
    (0.389 m) ends above the crown, while at the crown J2 already loses water, so SWMM's
    surcharged solution lies below the crown. SWMM has no fixed point there (its trials
    alternate; 48 % of this ramp's steps do not converge in SWMM 5.2.4 at 500 trials),
    and a flag lagged on the depth alone chatters until the pass budget runs out. The tie
    rule (`JunctionStorage`) holds J2 at its crown, within the band (< 1 mm), releasing
    water: every step converges, in at most 26 passes (steps 66-68; 11-19 elsewhere). A
    budget of 21 passes is too small for the crossing, and the refusal names its time."""
    net = _extran_tree()
    model, state0, drivers = build_model(net)
    hyd = model.dynamic_wave
    factors = _ramp()
    inflow = torch.stack([f * drivers["inflow"] for f in factors]).unsqueeze(1)
    diags: list = []
    state, hist = simulate_dynamic_wave(model, state0, drivers, steps=len(factors),
                                        inflow=inflow, diagnostics=diags)
    assert all(bool(d["converged"]) for d in diags)
    j2 = hyd.junction_names.index("J2")
    above = _depths(model, hist)[:, 0, j2] - hyd.crown_depth[j2]
    assert above[46].item() > 1.0 and above[45].item() < 0.0       # the jump at step 47
    held = ((above > 0.0) & (above < 1e-3)).nonzero().flatten().tolist()
    assert held and held[0] == 65, held                            # step 66: held
    assert above[-1].item() < 0.0                                  # free again
    assert max(d["passes"] for d in diags) <= 30
    # a budget too small for the crossing steps: refused, naming the step's time
    small, s0, d0 = build_model(net, iterate_max=21)
    with pytest.raises(RuntimeError, match=r"^dynamic wave step at t=(\d+)\.0 s: .*converge"
                       ) as err:
        simulate_dynamic_wave(small, s0, d0, steps=len(factors), inflow=inflow)
    t = float(re.match(r"^dynamic wave step at t=(\d+\.0) s", str(err.value)).group(1))
    assert 325.0 <= t <= 335.0, t                                  # steps 66-68 start


@pytest.mark.slow
def test_extran_batched_equals_single():
    """A batch of the surcharging tree (J2 and J4 surcharge) and half its inflow (free
    surface throughout): each instance carries its own surcharge flags. The head solve runs
    at atol 1e-12 m3/s: at the default `solve_atol` a surcharged junction's head is resolved
    only to atol / sum(k) (up to ~2.5e-7 m, ruling R28), and the batch and the single run
    took different, equally valid landings within that (2.7e-8 m at J4's crossing, step
    139, once the normal-flow slope switch changed the passes); at 1e-12 they agree to
    1e-12 m."""
    model, state, drivers = build_model(_extran_tree())
    base = drivers["inflow"]
    _, hb = simulate_dynamic_wave(model, state, drivers, steps=150,
                                  inflow=torch.stack([base, 0.5 * base]), atol=1e-12)
    crown = model.dynamic_wave.crown_depth
    assert bool((_depths(model, hb)[:, 0] > crown).any())
    assert not bool((_depths(model, hb)[:, 1] > crown).any())
    for i, scale in enumerate((1.0, 0.5)):
        _, hs = simulate_dynamic_wave(model, state, drivers, steps=150, inflow=scale * base,
                                      atol=1e-12)
        torch.testing.assert_close(hb["water.phi"][:, i], hs["water.phi"],
                                   rtol=1e-9, atol=1e-10)


def test_gradient_through_a_surcharged_junction():
    """d(sum of junction depths 4 steps on)/d(Manning n of C2) from step 25 of the EXTRAN
    tree, where J2 (C2's upstream end) is surcharged and holds instantaneous continuity: the
    adjoint against central differences on plain relaxation (564.8027017 both, 1e-10 rel
    measured)."""
    from noodl.apps.sewer.dynwave import build_dynamic_wave_model

    net = _extran_tree()
    model, state0, drivers = build_model(net)
    state0, _ = simulate_dynamic_wave(model, state0, drivers, steps=25)
    state0 = {k: v.detach() for k, v in state0.items()}
    j2 = model.dynamic_wave.junction_names.index("J2")
    depth = state0["water.phi"][model.junction_idx] - model.junction_invert
    assert depth[j2].item() > model.dynamic_wave.crown_depth[j2].item()
    plain = build_dynamic_wave_model(net, acceleration=None)[0]

    def loss(m, n2):
        hyd = m.dynamic_wave
        hyd.set_roughness(hyd.pipe_names.index("C2"), n2)
        s, _ = simulate_dynamic_wave(m, state0, drivers, steps=4)
        return (s["water.phi"][m.junction_idx] - m.junction_invert).sum()

    n = torch.tensor(0.013, dtype=F64, requires_grad=True)
    loss(model, n).backward()
    h = 1e-6
    with torch.no_grad():
        fd = (loss(plain, n.detach() + h) - loss(plain, n.detach() - h)) / (2 * h)
    assert n.grad.item() == pytest.approx(fd.item(), rel=1e-6)


def test_hold_band_never_acts_below_the_crown_at_a_large_storage_area():
    """Review fix: with A_s = 2000 A_bed (2 m2 against a 0.001 m2 floor) a band of slope
    (2/dt) K_HOLD A_bed would lie above the storage line everywhere below the crown. Below
    the crown a free junction's withdrawal and slope are the storage form exactly; above
    it the band is (2/dt) K_HOLD_STORAGE A_s = 8 m2/s, steeper than the storage's 0.8."""
    from noodl.apps.sewer.dynwave import K_HOLD_STORAGE

    ns = JunctionStorage(torch.tensor([0]), bed=torch.ones(1, dtype=F64), bed_area=0.001,
                         crown=torch.full((1,), 1.3, dtype=F64))
    drv = {"water.storage_area": torch.full((1,), 2.0, dtype=F64),
           "water.head_start": torch.full((1,), 1.0, dtype=F64),
           "water.net_inflow_start": torch.full((1,), 0.0, dtype=F64),
           "water.storage_rate": torch.tensor(0.2, dtype=F64),
           "water.surcharge_slope": torch.full((1,), 0.7, dtype=F64),
           "water.head_last": torch.full((1,), 1.1, dtype=F64)}
    drv["water.surcharged"] = ns.surcharged(drv)
    assert drv["water.surcharged"].item() == 0.0

    def at(h):
        phi = torch.full((1,), h, dtype=F64)
        return ns.flow(phi, drv).item(), ns.dflow(phi, drv).item()

    for h in (1.05, 1.1, 1.3):                                 # storage form, exactly
        w, dw = at(h)
        assert w == pytest.approx(0.8 * (h - 1.0)) and dw == pytest.approx(0.8)
    stiff = 0.4 * K_HOLD_STORAGE * 2.0
    w, dw = at(1.31)                                           # band above the crown
    assert w == pytest.approx(0.8 * 0.3 + stiff * 0.01) and dw == pytest.approx(stiff)
    assert at(1.3 + 1e-12)[0] == pytest.approx(0.24)           # continuous at the crown


# ------------------------------------------------------------------- flooding (E3)
def _flooding_tree(max_depth=0.5, method="extran"):
    """The EXTRAN tree with every junction's MaxDepth `max_depth` (m) and the surcharge
    method `method`: J2's surcharge (4.3 m at step 21 with MaxDepth 5) passes it."""
    net = _extran_tree()
    net = dataclasses.replace(net, manholes=tuple(
        dataclasses.replace(m, ground=m.invert + max_depth) for m in net.manholes))
    return dataclasses.replace(net, dynamic_wave=dataclasses.replace(
        net.dynamic_wave, surcharge_method=method))


def test_full_depth_extends_a_zero_max_depth_to_the_attached_crowns():
    """link.c:440-463: a junction's full depth is its MaxDepth, raised to every attached
    conduit's top; with MaxDepth 0 that is its crown. It floods above y_max = full depth +
    SurDepth (dynwave.c:746-748)."""
    net = _extran_tree()
    net = dataclasses.replace(net, manholes=tuple(
        dataclasses.replace(m, ground=m.invert,
                            surcharge_depth=0.2 if m.name == "J3" else 0.0)
        for m in net.manholes))
    hyd = build_model(net)[0].dynamic_wave
    torch.testing.assert_close(hyd.full_depth, hyd.crown_depth)
    flood = dict(zip(hyd.junction_names, hyd.flood_depth.tolist(), strict=True))
    assert flood == pytest.approx({"J1": 0.30, "J2": 0.30, "J5": 0.30, "J3": 0.65,
                                   "J4": 0.45}, abs=1e-12)
    hyd = build_model(_extran_tree())[0].dynamic_wave              # MaxDepth 5 stands
    assert hyd.full_depth.tolist() == pytest.approx([5.0] * 5)
    assert hyd.flood_depth.tolist() == pytest.approx([5.0] * 5)


def test_junction_storage_floods_at_its_top_through_a_lagged_excess():
    """Design E3 in `JunctionStorage` (top head 1.5 m, A_s 2 m2, dt 5 s: storage slope 0.8
    m2/s; wall slope (2/dt) max(K_HOLD A_bed, K_HOLD_STORAGE A_s) = 200 m2/s). The wall is
    ``w(top) + E + 200 (H - top)``, E the lagged excess of the fed net inflow over the
    non-flooded withdrawal at the top: junction 0 (w(top) = 0.4, fed N 0.5) floods with E
    = 0.1, junction 1 (w(top) = -0.22, fed N -0.3) does not (E = 0), but its wall still
    caps it at the top within the pass."""
    ns = JunctionStorage(torch.tensor([0, 1]), bed=torch.ones(2, dtype=F64), bed_area=0.5,
                         top=torch.full((2,), 1.5, dtype=F64))
    drv = {"water.storage_area": torch.full((2,), 2.0, dtype=F64),
           "water.head_start": torch.tensor([1.0, 1.4], dtype=F64),
           "water.net_inflow_start": torch.tensor([0.0, 0.3], dtype=F64),
           "water.storage_rate": torch.tensor(0.2, dtype=F64),
           "water.net_inflow_last": torch.tensor([0.5, -0.3], dtype=F64)}
    drv["water.flood_excess"] = ns.flood_excess(drv)
    assert drv["water.flood_excess"].tolist() == pytest.approx([0.1, 0.0])

    def at(h):
        phi = torch.full((2,), h, dtype=F64)
        return ns.flow(phi, drv).tolist(), ns.dflow(phi, drv).tolist()

    w, dw = at(1.5)                                            # on the wall at the top
    assert w == pytest.approx([0.5, -0.22]) and dw == pytest.approx([200.0, 200.0])
    w, dw = at(1.4)                                            # storage well below it
    assert w == pytest.approx([0.32, -0.3]) and dw == pytest.approx([0.8, 0.8])
    w, dw = at(1.4999)            # junction 0's wall reaches below the top by E/(200-0.8)
    assert w == pytest.approx([0.5 - 0.02, 0.8 * 0.0999 - 0.3])
    assert dw == pytest.approx([200.0, 0.8])
    w, dw = at(1.51)                                           # above the top: the wall
    assert w == pytest.approx([0.5 + 2.0, -0.22 + 2.0]) and dw == pytest.approx([200.0] * 2)


@pytest.mark.parametrize("method", ["extran", "slot"])
def test_a_junction_driven_past_its_maximum_depth_floods_at_it(method):
    """MaxDepth 0.5 m everywhere: J2 (crown 0.30 m) passes 0.5 m and floods
    (dynwave.c:743-795, no ponding). It is held at y_max (1e-9 m) while the excess leaves
    as `sewer.overflow`, SWMM's ``0.5 (N_start + N)`` (m3/s), which settles to J2's net
    inflow, its lateral inflow less C2's flow (1e-6 m3/s by step 120). It first floods at
    step 21 under EXTRAN and 36 under SLOT (the slot's storage fills it more slowly). No
    other junction floods in the 120 steps (J4 does at step ~231 under EXTRAN); every step
    converges."""
    net = _flooding_tree(method=method)
    model, state, drivers = build_model(net)
    hyd = model.dynamic_wave
    j2 = hyd.junction_names.index("J2")
    diags: list = []
    state, hist = simulate_dynamic_wave(
        model, state, drivers, steps=120, diagnostics=diags,
        record=("water.phi", "water.q", "sewer.net_inflow", "sewer.overflow"))
    assert all(bool(d["converged"]) for d in diags)
    depth = _depths(model, hist)
    over = hist["sewer.overflow"]
    flooded = over[:, j2] > 0.0
    first = int(flooded.nonzero()[0])
    assert first + 1 == {"extran": 21, "slot": 36}[method], first
    assert bool(flooded[first:].all())
    assert (depth[first:, j2] - 0.5).abs().max().item() < 1e-9
    assert depth.max().item() < 0.5 + 1e-9
    others = [j for j in range(len(hyd.junction_names)) if j != j2]
    assert bool((over[:, others] == 0.0).all())
    net_in = hist["sewer.net_inflow"]
    trapezoid = 0.5 * (net_in[first - 1:-1, j2] + net_in[first:, j2])
    torch.testing.assert_close(over[first:, j2], trapezoid, rtol=0.0, atol=1e-12)
    assert over[-1, j2].item() > 0.01
    lateral = drivers["inflow"][model.junction_idx[j2]].item()
    c2 = hist["water.q"][-1, hyd.pipe_names.index("C2")].item()
    assert abs(over[-1, j2].item() - (lateral - c2)) < 1e-6                  # settled


def test_flood_releases_when_inflow_stops():
    """J2 floods (MaxDepth 0.5 m, from step 21), then its lateral inflow ramps down to
    nothing over steps 31-50 (the others stay). The lagged flood excess falls with J2's net
    inflow; once it would be negative J2 leaves y_max (step 34), its overflow is 0
    thereafter, it passes back down through its crown and drains, and no step chatters
    (at most 30 passes). (Stopping EVERY inflow drains the tree onto C5's normal-flow
    slope switch: `test_inflow_stop_converges`.)"""
    model, state, drivers = build_model(_flooding_tree())
    hyd = model.dynamic_wave
    j2 = hyd.junction_names.index("J2")
    node = int(model.junction_idx[j2])
    factors = [1.0] * 30 + [1.0 - (k + 1) / 20 for k in range(20)] + [0.0] * 20
    rows = []
    for f in factors:
        row = drivers["inflow"].clone()
        row[node] = f * drivers["inflow"][node]
        rows.append(row)
    inflow = torch.stack(rows).unsqueeze(1)
    diags: list = []
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=len(factors), inflow=inflow,
                                    diagnostics=diags,
                                    record=("water.phi", "sewer.overflow"))
    assert all(bool(d["converged"]) for d in diags)
    assert max(d["passes"] for d in diags) <= 30, max(d["passes"] for d in diags)
    depth = _depths(model, hist)[:, 0, j2]
    over = hist["sewer.overflow"][:, 0, j2]
    assert (depth[20:30] - 0.5).abs().max().item() < 1e-9
    assert bool((over[20:30] > 0.0).all())
    released = int((over == 0.0)[30:].nonzero()[0]) + 30
    assert released + 1 == 34, released
    assert bool((over[released:] == 0.0).all())
    assert bool((depth[released:] < 0.5 - 1e-6).all())
    assert depth[-1].item() < hyd.crown_depth[j2].item()


def test_gradient_through_a_flooded_junction():
    """d(sum of the outfall conduit's flow 4 steps on)/d(Manning n of C2) from step 40 of
    the flooding tree, where J2 (C2's upstream end) is held at y_max: the adjoint against
    central differences on plain relaxation."""
    from noodl.apps.sewer.dynwave import build_dynamic_wave_model

    net = _flooding_tree()
    model, state0, drivers = build_model(net)
    state0, _ = simulate_dynamic_wave(model, state0, drivers, steps=40)
    state0 = {k: v.detach() for k, v in state0.items()}
    j2 = model.dynamic_wave.junction_names.index("J2")
    depth = state0["water.phi"][model.junction_idx] - model.junction_invert
    assert abs(depth[j2].item() - 0.5) < 1e-9
    plain = build_dynamic_wave_model(net, acceleration=None)[0]
    c5 = model.dynamic_wave.pipe_names.index("C5")

    def loss(m, n2):
        hyd = m.dynamic_wave
        hyd.set_roughness(hyd.pipe_names.index("C2"), n2)
        s, _ = simulate_dynamic_wave(m, state0, drivers, steps=4, atol=1e-12)
        return s["water.q"][..., c5].sum()

    n = torch.tensor(0.013, dtype=F64, requires_grad=True)
    loss(model, n).backward()
    h = 1e-6
    with torch.no_grad():
        fd = (loss(plain, n.detach() + h) - loss(plain, n.detach() - h)) / (2 * h)
    assert abs(n.grad.item()) > 0.0
    assert n.grad.item() == pytest.approx(fd.item(), rel=1e-6)


def _drain(factors, conduits=("C5",)):
    """tree_dynwave.inp under every lateral inflow times `factors` (one per step, the
    inflow at the END of the step). Returns the diagnostics and, per conduit in `conduits`
    and per step, its side of the normal-flow limit, each checked against the drivers of
    a pass fed the step's result:

    - -1: the slope switch's limited side (y1 < y2), q = min(lin, q_norm);
    - +1: its unlimited side (y1 > y2), q = lin;
    - 0: on the switching surface (y1 = y2), q between q_norm and lin;
    - ``"lagged"``: limited by a lagged flag (the Froude test), q = min(lin, q_norm);
    - None: no limit and no switch (e.g. while the tree fills).

    So every step is SWMM's own fixed point, or sits on the switch. y1 - y2 is ``h1 - h2 -
    dz``, dz the drop between the conduit-end inverts; C5 joins J4 to the outfall."""
    from noodl.model import StepContext

    net, _, _ = read_swmm_inp(DATA / "tree_dynwave.inp")
    model, state, drivers = build_model(net)
    hyd = model.dynamic_wave
    inflow = torch.stack([f * drivers["inflow"] for f in factors]).unsqueeze(1)
    diags: list = []
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=len(factors), inflow=inflow,
                                    diagnostics=diags, record=("water.phi", "water.q")
                                    + hyd.state_keys)
    sides: dict[str, list] = {name: [] for name in conduits}
    for k in range(len(factors)):
        before = {key: (hist[key][k - 1] if k else state[key]) for key in hyd.state_keys}
        after = {key: hist[key][k] for key in ("water.phi", "water.q")}
        drv = hyd(after | before, {"inflow": inflow[k]}, StepContext(dt=hyd.options.routing_step))
        phi = after["water.phi"][0]
        for name in conduits:
            i = hyd.pipe_names.index(name)
            dp = phi[int(hyd.u[i])] - phi[int(hyd.v[i])]
            q = after["water.q"][0, i].item()
            gap = (dp - drv["water.switch_drop"][0, i]).item()
            lin = (drv["water.flow_offset"][0, i] + drv["water.conveyance"][0, i] * dp).item()
            q_norm = drv["water.normal_flow"][0, i].item()
            if drv["water.limited"][0, i].item() == 1.0:
                side = "lagged"                 # limited by a lagged flag (Froude test)
                assert abs(q - min(lin, q_norm)) <= 1e-9, (name, k, q, q_norm, lin)
            elif drv["water.slope_switch"][0, i].item() != 1.0:
                side = None
            elif abs(gap) <= 1e-9:
                side = 0
                assert min(q_norm, lin) - 1e-9 <= q <= lin + 1e-9, (name, k, q, q_norm, lin)
            else:
                side = -1 if gap < 0.0 else 1
                expected = min(lin, q_norm) if side < 0 else lin
                assert abs(q - expected) <= 1e-9, (name, k, side, q, q_norm, lin)
            sides[name].append(side)
    return diags, sides


def test_ramp_down_converges():
    """Every inflow of the tree ramps from its full value to nothing over steps 61-72 and
    stays off. As the network drains, J4 falls to the depth of the free outfall, which C5
    (J4 -> outfall) sets from its own flow, and C5's slope test y1 < y2 (always on for an
    outfall conduit) becomes degenerate: its limited flow q_norm(y1) is Manning's normal
    flow at J4's depth, at which the outfall's normal depth IS J4's depth, up to the
    interpolation mismatch of SWMM's tables (~1e-5 m). The limited side is consistent
    only where that mismatch puts the outfall above J4; elsewhere neither side is (SWMM
    5.2.4 itself then stops converging: on 82 of the 108 steps after J4 first meets the
    outfall depth of the 120-step-fill version of this ramp, at HEAD_TOLERANCE 1e-12 m and
    500 trials). The lagged flag chattered until the pass budget ran out (step 120, t
    = 595 s). With the in-pass slope switch (`DynamicWaveConduit`) every step converges --
    a failure would name its time -- each on SWMM's own fixed point where there is one
    (steps 106-119 limited), and on the switching surface where there is none (step
    120)."""
    factors = [1.0] * 60 + [1.0 - (k + 1) / 12 for k in range(12)] + [0.0] * 48
    diags, both = _drain(factors, ("C5", "C3"))
    # C3 (J3 -> J4) is an interior conduit (no outfall compliance: its wall slope is the
    # cap S_max): on its switch's unlimited side from step 2 to 90, then limited by its
    # Froude test (a lagged flag) as the drain-down steepens it; consistent on every step.
    assert set(both["C3"][1:90]) == {1} and set(both["C3"][90:]) == {"lagged"}, both["C3"]
    sides = both["C5"]
    assert all(bool(d["converged"]) for d in diags)
    assert max(d["passes"] for d in diags) <= 30, max(d["passes"] for d in diags)
    assert set(sides[60:105]) == {1} and set(sides[105:119]) == {-1}, sides
    assert sides[119] == 0, sides


def test_inflow_stop_converges():
    """Every inflow of the tree stops at step 61 (all zero from its end on). The same
    degenerate slope test as `test_ramp_down_converges`, reached sooner (the lagged flag
    chattered from step 105, t = 520 s): every step converges, from there on on the
    switching surface (steps 105-120)."""
    factors = [1.0] * 60 + [0.0] * 60
    diags, sides = _drain(factors)
    sides = sides["C5"]
    assert all(bool(d["converged"]) for d in diags)
    assert max(d["passes"] for d in diags) <= 30, max(d["passes"] for d in diags)
    assert set(sides[60:104]) == {1}, sides
    assert set(sides[104:]) == {0}, sides


def test_a_heads_only_convergence_test_is_refused():
    """`iterate_flow_tol=None` (the heads judged alone) is refused by name by
    `build_dynamic_wave_model` (`build_model` exposes no flow tolerance and always passes
    the builder's default): on the normal-flow slope switch the heads bound the flows'
    residual only through the wall slope over the junction's grounding, and a heads-only
    plain run was measured stopping 2.9e-5 m3/s short of the fixed point (the tree's
    drain-down, step 105). Plain relaxation with a flow tolerance stays allowed."""
    from noodl.apps.sewer.dynwave import build_dynamic_wave_model

    net, _, _ = read_swmm_inp(DATA / "tree_dynwave.inp")
    with pytest.raises(ValueError, match=r"iterate_flow_tol \(m3/s\) is required"):
        build_dynamic_wave_model(net, acceleration=None, iterate_flow_tol=None)
    model = build_dynamic_wave_model(net, acceleration=None)[0]
    assert model.iterate_acceleration is None and model.iterate_flow_tol == {"water": 1e-10}
    assert build_model(net)[0].iterate_flow_tol == {"water": 1e-10}


def test_gradient_on_the_slope_switch_does_not_depend_on_the_wall_slope(monkeypatch):
    """At a step that lands on C5's switching surface (the tree's drain-down after all
    inflows stop, step 108) the returned gradient is the tie rule's -- the sensitivity of a
    solution held at y1 = y2 -- and the wall slope S drops out of it: the fixed point does
    not move with S, so neither does its implicit derivative. d(C5 flow over 2 steps)/d(n
    of C3) is the same for S scaled by 0.2, 1 and 5 (to 1e-10 relative; 11 digits
    measured) and matches central differences (2e-8 relative measured)."""
    from noodl.apps.sewer.dynwave import DynamicWaveHydraulics

    net, _, _ = read_swmm_inp(DATA / "tree_dynwave.inp")
    model, state0, drivers = build_model(net)
    factors = [1.0] * 60 + [0.0] * 50
    inflow = torch.stack([f * drivers["inflow"] for f in factors]).unsqueeze(1)
    state0, _ = simulate_dynamic_wave(model, state0, drivers, steps=107, inflow=inflow,
                                      atol=1e-12)
    state0 = {k: v.detach() for k, v in state0.items()}
    hyd = model.dynamic_wave
    c5, c3 = hyd.pipe_names.index("C5"), hyd.pipe_names.index("C3")
    u, v = int(hyd.u[c5]), int(hyd.v[c5])

    def loss(n3):
        hyd.set_roughness(c3, n3)
        s, _ = simulate_dynamic_wave(model, state0, drivers, steps=2, inflow=inflow[107:109],
                                     atol=1e-12)
        gap = (s["water.phi"][..., u] - hyd.node_invert[u]) - (
            s["water.phi"][..., v] - hyd.node_invert[v])
        assert gap.abs().max().item() < 1e-12                   # on the switch
        return s["water.q"][..., c5].sum()

    original = DynamicWaveHydraulics._switch_slope
    grads = []
    for scale in (1.0, 0.2, 5.0):
        monkeypatch.setattr(DynamicWaveHydraulics, "_switch_slope",
                            lambda self, *a, scale=scale: scale * original(self, *a))
        n = torch.tensor(0.013, dtype=F64, requires_grad=True)
        loss(n).backward()
        grads.append(n.grad.item())
    monkeypatch.setattr(DynamicWaveHydraulics, "_switch_slope", original)
    assert grads[1] == pytest.approx(grads[0], rel=1e-10)
    assert grads[2] == pytest.approx(grads[0], rel=1e-10)
    h = 1e-6
    with torch.no_grad():
        fd = (loss(torch.tensor(0.013 + h, dtype=F64))
              - loss(torch.tensor(0.013 - h, dtype=F64))) / (2 * h)
    assert grads[0] == pytest.approx(fd.item(), rel=1e-6)
