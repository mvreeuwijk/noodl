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
    routing_step_size,
    simulate_dynamic_wave,
    solve_atol,
)
from noodl.apps.sewer.xsect_tables import LCF, QCF, area_of_depth

F64 = torch.float64
DATA = Path(__file__).resolve().parents[2] / "data" / "sewer"


def _tree(tmp_path, scale=1.0):
    """The kinematic tree routed by dynamic wave, converged tightly as the dynwave fixtures
    are (HEAD_TOLERANCE 1e-12 m, 2000 trials: the iteration's defaults follow the file)."""
    text = (DATA / "tree_kinwave.inp").read_text()
    text = text.replace("FLOW_ROUTING         KINWAVE", "FLOW_ROUTING         DYNWAVE")
    text = text.replace("VARIABLE_STEP        0.75", "VARIABLE_STEP        0")
    text = text.replace("MAX_TRIALS           8", "MAX_TRIALS           2000")
    text = text.replace("HEAD_TOLERANCE       0.0015", "HEAD_TOLERANCE       1e-12")
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
    budget, the file's MAX_TRIALS 500, covers that, so a plain build_model +
    simulate_dynamic_wave does not run out of passes there."""
    net, _, _ = read_swmm_inp(DATA / "loop_dynwave.inp")
    model, _, _ = build_model(net)
    assert model.iterate_max == 500


def test_build_model_dispatches_and_refuses_storage(tmp_path):
    net = _tree(tmp_path)
    model, state, drivers = build_model(net)
    assert "water" in model.kirchhoff and model.kirchhoff["water"].quantity == "head"
    assert not model.transport and "air" not in model.kirchhoff
    with pytest.raises(ValueError, match="storage"):
        build_model(net, storage=True)


@pytest.mark.parametrize("kwargs,name", [
    ({"coupling": "iterate"}, "coupling"),
    ({"dt_storage": 60.0}, "dt_storage"),
])
def test_build_model_refuses_kinematic_options_on_a_dynamic_wave_net(tmp_path, kwargs, name):
    """The kinematic path's coupling and storage step have no meaning for dynamic wave
    (its passes always iterate, its storage step is the routing step): a non-default value
    is refused by name, not ignored."""
    with pytest.raises(ValueError, match=name):
        build_model(_tree(tmp_path), **kwargs)


def test_dynamic_wave_refuses_the_analytic_geometry(tmp_path):
    """Dynamic wave always uses SWMM's tabulated circle: a network asking for the exact
    circle is refused by name rather than routed on the tables silently."""
    from noodl.apps.sewer.dynwave import build_dynamic_wave_model

    net = dataclasses.replace(_tree(tmp_path), geometry="analytic")
    with pytest.raises(ValueError, match="geometry.*analytic"):
        build_model(net)
    with pytest.raises(ValueError, match="geometry.*analytic"):
        build_dynamic_wave_model(net)


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
    floor = hyd.min_area_ft2 * QCF / LCF                            # SI area, `_AREA_SI`
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
    out = model.outfall_link_idx
    total = sum(m.inflow for m in net.manholes)
    assert hist["water.q"][-1, out].sum().item() == pytest.approx(total, rel=1e-6)


@pytest.mark.slow
def test_outfall_flow_rises_monotonically_towards_the_total(tmp_path):
    """The fast variant of the steady-state test: 10 minutes at half the inflows."""
    net = _tree(tmp_path, scale=0.5)
    model, state, drivers = build_model(net)
    state, hist = simulate_dynamic_wave(model, state, drivers, steps=120)
    q = hist["water.q"][:, model.outfall_link_idx].sum(-1)
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
    assert set(rows[0]) == {"pipe", "q", "h", "v", "R_h", "d_m", "T", "A_air", "D_h", "V_wet",
                            "V_air"}


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
    """Anderson mixing of the head iteration at least halves the mean passes per
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
    """Anderson mixing moves no fixed point. With the head solve tight (atol
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
    0.30 m crown of C4 at step 95 (475 s) of 120. A 1.5x scale only reaches 0.25 m at
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


# ---------------------------------------------------------------- EXTRAN surcharge
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
    """EXTRAN surcharge in `JunctionStorage` (crown head 1.3 m, A_s 2 m2, dt 5 s, so 2 A_s/dt =
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


def _area_switch_drivers(n_last, h_star=1.1):
    """One junction whose storage area jumps from 2 to 3 m2 at the threshold head `h_star`
    (m) while it fills (head 1 m at the step start, net inflow 0.1 m3/s then, dt 5 s): the
    storage lines are lo = 0.8 (H - 1) - 0.1 and hi = 1.2 (H - 1) - 0.1 (m3/s), -0.02 and
    +0.02 at 1.1 m; wall slope 10 m2/s, fed net inflow `n_last` (m3/s)."""
    one = torch.ones(1, dtype=F64)
    return {"water.storage_area": 2.0 * one, "water.head_start": one,
            "water.net_inflow_start": 0.1 * one, "water.storage_rate": torch.tensor(0.2, dtype=F64),
            "water.area_switch_head": h_star * one, "water.area_below": 2.0 * one,
            "water.area_above": 3.0 * one, "water.area_switch_slope": 10.0 * one,
            "water.net_inflow_last": n_last * one}


def test_junction_storage_holds_a_filling_junction_on_its_area_threshold():
    """The drop-end area switch in `JunctionStorage` (the tie rule, `DynamicWaveHydraulics`):
    ``max(lo, min(hi, N_last + S (H - H*)))``, continuous and non-decreasing in H. Its
    fixed points (withdrawal = fed net inflow) are SWMM's own on either side of the
    threshold, unchanged, plus the threshold itself, H = H*, for any net inflow inside the
    jump -- where SWMM has none (the storage equation lands above H* with the lower area and
    below it with the upper one)."""
    ns = JunctionStorage(torch.tensor([0]), bed=torch.zeros(1, dtype=F64), bed_area=0.5)

    def w(h, n_last, **kw):
        return ns.flow(torch.tensor([h], dtype=F64), _area_switch_drivers(n_last, **kw)).item()

    # on the threshold for any net inflow inside the jump [-0.02, 0.02]
    for n_last in (-0.019, 0.0, 0.019):
        assert w(1.1, n_last) == pytest.approx(n_last, abs=1e-15)
    # SWMM's fixed points below and above it, unchanged
    assert w(1.05, 0.8 * 0.05 - 0.1) == pytest.approx(0.8 * 0.05 - 0.1)
    assert w(1.2, 1.2 * 0.2 - 0.1) == pytest.approx(1.2 * 0.2 - 0.1)
    # continuous, non-decreasing, with the slope of the active piece
    grid = torch.linspace(0.9, 1.3, 4001, dtype=F64)
    drv = {k: v.expand(4001) if v.dim() else v for k, v in _area_switch_drivers(0.0).items()}
    ws = ns.flow(grid, drv)
    assert bool((ws[1:] - ws[:-1] >= 0.0).all())
    assert float((ws[1:] - ws[:-1]).abs().max()) < 10.0 * 1e-4 + 1e-12
    slopes = ns.dflow(torch.tensor([1.0, 1.1, 1.2], dtype=F64),
                      {k: v.expand(3) if v.dim() else v
                       for k, v in _area_switch_drivers(0.0).items()})
    assert slopes.tolist() == pytest.approx([0.8, 10.0, 1.2])
    # no threshold (+inf): the lagged storage form
    assert w(1.2, 0.0, h_star=float("inf")) == pytest.approx(0.8 * 0.2 - 0.1)


def test_extran_surcharge_is_instantaneous_continuity():
    """EXTRAN surcharge on the tree at 3x tree_dynwave.inp's inflows (1.5x
    tree_kinwave.inp's; tree_dynwave.inp is half of that). In SWMM J2 passes its
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
    only to atol / sum(k) (up to ~2.5e-7 m), and the batch and the single run
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


def test_hold_band_is_continuous_at_the_crown_with_the_area_switch_active():
    """Review fix: the band starts at the free storage evaluated AT the crown, area switch
    included. Crown 10 m, threshold H* 9.98 m below it, storage areas 20 (lagged) / 10
    (below) / 20 m2 (above), switch slope 5 m2/s, head 9.9 m and net inflow 0 at the step
    start, dt 2 s, fed net inflow 0: at the crown the switched storage is
    max(10 (0.1), min(20 (0.1), 5 (0.02))) = 1 m3/s, while the lagged-area line gives 2 m3/s.
    The withdrawal is continuous through the crown (no 1 m3/s jump), non-decreasing, with a
    positive slope, and the surcharge flag reads the same band."""
    from noodl.apps.sewer.dynwave import K_HOLD_STORAGE

    ns = JunctionStorage(torch.tensor([0]), bed=torch.zeros(1, dtype=F64), bed_area=0.01,
                         crown=torch.full((1,), 10.0, dtype=F64))
    one = torch.ones(1, dtype=F64)
    drv = {"water.storage_area": 20.0 * one, "water.head_start": 9.9 * one,
           "water.net_inflow_start": 0.0 * one, "water.storage_rate": torch.tensor(0.5, dtype=F64),
           "water.area_switch_head": 9.98 * one, "water.area_below": 10.0 * one,
           "water.area_above": 20.0 * one, "water.area_switch_slope": 5.0 * one,
           "water.net_inflow_last": 0.0 * one, "water.surcharge_slope": 0.7 * one,
           "water.head_last": 9.95 * one}
    drv["water.surcharged"] = ns.surcharged(drv)
    assert drv["water.surcharged"].item() == 0.0
    w_crown = ns.flow(torch.full((1,), 10.0, dtype=F64), drv).item()
    assert w_crown == pytest.approx(1.0)
    above = ns.flow(torch.full((1,), 10.0 + 1e-9, dtype=F64), drv).item()
    assert above == pytest.approx(w_crown, abs=1e-6)
    grid = torch.linspace(9.99, 10.01, 2001, dtype=F64)
    big = {k: v.expand(2001) if v.dim() else v for k, v in drv.items()}
    ws = ns.flow(grid, big)
    steps = ws[1:] - ws[:-1]
    stiff = 1.0 * K_HOLD_STORAGE * 20.0
    assert bool((steps >= 0.0).all())
    assert float(steps.max()) < stiff * 1e-5 + 1e-12
    assert bool((ns.dflow(grid, big) > 0.0).all())
    # the flag uses the same band: fed above the crown, the band is positive there
    drv["water.head_last"] = 10.0 + 1e-6 * one
    assert ns.surcharged(drv).item() == 1.0


# ------------------------------------------------------------------- flooding
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
    """Flooding in `JunctionStorage` (top head 1.5 m, A_s 2 m2, dt 5 s: storage slope 0.8
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
    # the pass budget below was measured at a 1e-11 m head tolerance (the file's is 1e-12)
    model, state, drivers = build_model(_flooding_tree(), iterate_tol=1e-11)
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


def test_iteration_controls_follow_the_file():
    """`build_dynamic_wave_model`'s iteration controls follow [OPTIONS]
    (`iterate_tolerances`): HEAD_TOLERANCE is the head tolerance, MAX_TRIALS the pass limit
    (a step that uses it up is accepted, as SWMM's), and the flow tolerance is the flow
    residual that moves the smallest junction's head by the head tolerance over a routing
    step, ``2 A_min tol / dt``, floored at `FLOW_TOL_FLOOR`. SWMM's own defaults (0.005 ft,
    8) where the file sets none; an explicit argument wins, and an explicit `iterate_max`
    keeps the refusal of a step that does not converge. The heads are never judged alone:
    on the normal-flow slope switch they bound the flows' residual only through the wall
    slope over the junction's grounding (a heads-only plain run was measured stopping
    2.9e-5 m3/s short of the fixed point on the tree's drain-down)."""
    from noodl.apps.sewer.dynwave import (
        DEFAULT_HEAD_TOLERANCE,
        FLOW_TOL_FLOOR,
        build_dynamic_wave_model,
        iterate_tolerances,
    )

    net, _, _ = read_swmm_inp(DATA / "tree_dynwave.inp")   # HEAD_TOLERANCE 1e-12, 500 trials
    model = build_dynamic_wave_model(net, acceleration=None)[0]
    assert model.iterate_tol == {"water": 1e-12} and model.iterate_max == 500
    assert model.iterate_flow_tol == {"water": FLOW_TOL_FLOOR}
    assert model.iterate_on_limit == "accept" and model.iterate_acceleration is None
    assert build_model(net)[0].iterate_flow_tol == {"water": FLOW_TOL_FLOOR}
    explicit = build_model(net, iterate_tol=1e-11, iterate_max=2000, iterate_flow_tol=3e-9)[0]
    assert explicit.iterate_tol == {"water": 1e-11} and explicit.iterate_max == 2000
    assert explicit.iterate_flow_tol == {"water": 3e-9}
    assert explicit.iterate_on_limit == "raise"
    swmm = dataclasses.replace(net.dynamic_wave, head_tolerance=None, max_trials=None)
    controls = iterate_tolerances(swmm)
    assert DEFAULT_HEAD_TOLERANCE == pytest.approx(0.005 * 0.3048, rel=1e-15)
    assert controls["iterate_tol"] == DEFAULT_HEAD_TOLERANCE
    assert controls["iterate_max"] == 8 and controls["on_limit"] == "accept"
    a_min = 12.566 * QCF / LCF                                  # MIN_SURFAREA, m2
    assert controls["iterate_flow_tol"] == pytest.approx(
        2.0 * a_min * DEFAULT_HEAD_TOLERANCE / 5.0, rel=1e-12)


def test_a_step_that_uses_up_the_files_trials_is_accepted():
    """As SWMM (`dynwave.c::dynwave_execute`), a step that uses up MAX_TRIALS is taken with
    its last pass and counted (`simulate_dynamic_wave`'s ``"converged"``); with an explicit
    `iterate_max` it is refused by name, as before."""
    net, _, _ = read_swmm_inp(DATA / "tree_dynwave.inp")
    net = dataclasses.replace(net, dynamic_wave=dataclasses.replace(net.dynamic_wave,
                                                                    max_trials=2))
    model, state, drivers = build_model(net)
    _, out = simulate_dynamic_wave(model, state, drivers, steps=6)
    assert out["converged"].dtype == torch.bool and out["converged"].shape == (6,)
    assert not bool(out["converged"].all())
    model, state, drivers = build_model(net, iterate_max=2)
    with pytest.raises(RuntimeError, match="did not converge within 2 passes"):
        simulate_dynamic_wave(model, state, drivers, steps=6)


def test_gradient_on_the_slope_switch_does_not_depend_on_the_wall_slope(monkeypatch):
    """At a step that lands on C5's switching surface (the tree's drain-down after all
    inflows stop, step 108) the returned gradient is the tie rule's -- the sensitivity of a
    solution held at y1 = y2 -- and the wall slope S drops out of it: the fixed point does
    not move with S, so neither does its implicit derivative. d(C5 flow over 2 steps)/d(n
    of C3) is the same for S scaled by 0.2, 1 and 5 (to 1e-10 relative; 11 digits
    measured) and matches central differences (2e-8 relative measured)."""
    from noodl.apps.sewer.dynwave import DynamicWaveHydraulics

    net, _, _ = read_swmm_inp(DATA / "tree_dynwave.inp")
    # pinned below the file's HEAD_TOLERANCE (1e-12 m): the on-the-switch check needs the
    # iteration to land on the switch, and at 1e-12 m the steep-wall run (S x 5) can stop one
    # tolerance short of it (1.2e-12 m measured on Linux, 2e-14 m on Windows)
    model, state0, drivers = build_model(net, iterate_tol=1e-13, iterate_max=2000)
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
                            lambda self, *a, scale=scale: (scale * original(self, *a)[0],
                                                           original(self, *a)[1]))
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


# ------------------------------------------------------------------ variable step, inflow
VARIABLE_TREE = DATA / "tree_variable_dynwave.inp"


def _variable_model():
    net, _, _ = read_swmm_inp(VARIABLE_TREE)
    return net, *build_model(net)


def test_variable_step_options_are_read_and_validated(tmp_path):
    net, _, _ = read_swmm_inp(VARIABLE_TREE)
    opts = net.dynamic_wave
    assert (opts.variable_step, opts.minimum_step, opts.routing_step) == (0.75, 0.5, 60.0)
    assert opts.min_route_step == 0.5
    # MinRouteStep is capped by the routing step and floored at 1 ms (dynwave.c:184-185)
    assert dataclasses.replace(opts, minimum_step=500.0).min_route_step == 60.0
    assert dataclasses.replace(opts, minimum_step=0.0).min_route_step == 0.001
    for bad in (dict(variable_step=-0.1), dict(variable_step=2.5), dict(minimum_step=-1.0)):
        with pytest.raises(ValueError, match="variable_step|minimum_step"):
            dataclasses.replace(opts, **bad).validate()


def test_variable_step_on_a_dry_network_is_the_routing_step():
    """No flow, no depth change: the first step is MINIMUM_STEP, every later one the
    routing step, and the last is cut to the time that remains."""
    net, model, state, drivers = _variable_model()
    _, hist = simulate_dynamic_wave(model, state, drivers, duration=200.0,
                                    inflow=torch.zeros(model.net.n, dtype=F64))
    assert hist["dt"].tolist() == [0.5, 60.0, 60.0, 60.0, 19.5]
    assert hist["t"].tolist() == [0.5, 60.5, 120.5, 180.5, 200.0]
    assert hist["water.phi"].shape[0] == 5


def test_fixed_step_duration_clips_the_last_step_to_the_time_that_remains(tmp_path):
    net = _tree(tmp_path)                     # ROUTING_STEP 5 s, VARIABLE_STEP 0
    model, state, drivers = build_model(net)
    _, hist = simulate_dynamic_wave(model, state, drivers, duration=12.0)
    assert hist["dt"].tolist() == [5.0, 5.0, 2.0] and hist["t"][-1].item() == 12.0
    # less than a millisecond remaining: SWMM takes 1 ms (swmm5.c:533-535)
    _, hist = simulate_dynamic_wave(model, state, drivers, duration=5.0004)
    assert hist["dt"].tolist() == [5.0, 0.001] and hist["t"][-1].item() == 5.0004


def test_steps_and_duration_are_exclusive_and_validated(tmp_path):
    net = _tree(tmp_path)
    model, state, drivers = build_model(net)
    with pytest.raises(ValueError, match="exactly one of steps and duration"):
        simulate_dynamic_wave(model, state, drivers)
    with pytest.raises(ValueError, match="exactly one of steps and duration"):
        simulate_dynamic_wave(model, state, drivers, steps=2, duration=10.0)
    with pytest.raises(ValueError, match="duration must be > 0"):
        simulate_dynamic_wave(model, state, drivers, duration=0.0)
    series = torch.zeros(3, 1, model.net.n, dtype=F64)
    with pytest.raises(ValueError, match="needs steps"):
        simulate_dynamic_wave(model, state, drivers, duration=10.0, inflow=series)


def test_variable_step_refuses_dt_and_a_batch_by_name():
    net, model, state, drivers = _variable_model()
    with pytest.raises(ValueError, match="variable_step = 0.75 chooses every step"):
        simulate_dynamic_wave(model, state, drivers, steps=2, dt=5.0)
    batch = torch.zeros(2, model.net.n, dtype=F64)
    with pytest.raises(ValueError, match="batched inflow cannot be combined"):
        simulate_dynamic_wave(model, state, drivers, steps=2, inflow=batch)
    with pytest.raises(ValueError, match=r"inflow\(t=0.0\) must return"):
        simulate_dynamic_wave(model, state, drivers, steps=2, inflow=lambda t: torch.zeros(3))


def test_inflow_callable_is_called_once_per_step_with_the_start_time(tmp_path):
    net = _tree(tmp_path)
    model, state, drivers = build_model(net)
    inflow = drivers["inflow"].clone()
    times = []

    def series(t):
        times.append(t)
        return inflow

    _, hist = simulate_dynamic_wave(model, state, drivers, steps=3, inflow=series)
    assert times == [0.0, 5.0, 10.0]
    _, const = simulate_dynamic_wave(model, state, drivers, steps=3, inflow=inflow)
    torch.testing.assert_close(hist["water.q"], const["water.q"], rtol=0, atol=0)
    # a batched callable expands the state like a batched array
    _, batched = simulate_dynamic_wave(model, state, drivers, steps=2,
                                       inflow=lambda t: torch.stack([inflow, 0.5 * inflow]))
    assert batched["water.q"].shape == (2, 2, len(net.pipes))


def _step_size_case(options=None, depths=None, q=None, before=None, dt=10.0):
    """`routing_step_size` on a hand-built state of the variable-step tree: junction depths
    `depths` (m, by name, others dry), conduit flows `q` (m3/s, by name), against the start
    junction depths `before` (default dry), after a step of `dt` s."""
    net, _, _ = read_swmm_inp(VARIABLE_TREE)
    if options:
        net = dataclasses.replace(net, dynamic_wave=dataclasses.replace(
            net.dynamic_wave, **options))
    model, state, _ = build_model(net)
    hyd = model.dynamic_wave
    names = list(hyd.net.nodes)
    phi = hyd.node_invert.clone()
    for name, y in (depths or {}).items():
        phi[names.index(name)] += y
    flows = torch.zeros(len(hyd.pipe_names), dtype=F64)
    for name, value in (q or {}).items():
        flows[hyd.pipe_names.index(name)] = value

    def jdepth(d):
        return torch.tensor([(d or {}).get(n, 0.0) for n in hyd.junction_names], dtype=F64)

    after = dict(state, **{"water.phi": phi, "water.q": flows, "sewer.pipe_flow": flows,
                           "sewer.junction_depth": jdepth(depths)})
    start = dict(state, **{"sewer.junction_depth": jdepth(before)})
    return routing_step_size(hyd, start, after, dt)


def test_routing_step_size_pins_each_branch_of_swmm_get_variable_step():
    # nothing moving: the routing step caps it (dynwave.c:799-806, tMin starts at maxStep)
    assert _step_size_case() == 60.0
    # a filling junction: 0.25 crown / (dy/dt) = 0.075 m / 0.005 m/s = 15 s (J1's crown 0.3 m)
    t = _step_size_case(depths={"J1": 0.05})
    assert 14.99 <= t <= 15.0
    # floored at MinRouteStep (0.5 s) however fast it rises
    assert _step_size_case(depths={"J1": 0.05}, dt=0.01) == 0.5
    # a junction at or above its crown is skipped (dynwave.c:898-899), as a dry one
    assert _step_size_case(depths={"J1": 0.35}) == 60.0
    assert _step_size_case(depths={"J1": 0.3 - 0.5e-4 * LCF}) == 60.0
    # a flowing conduit: Courant step V/|q| Fr/(1+Fr) 0.75 (dynwave.c:859-862)
    flowing = dict(depths={"J1": 0.05, "J3": 0.05}, q={"C1": 0.05}, before={"J1": 0.05,
                                                                              "J3": 0.05})
    net, _, _ = read_swmm_inp(VARIABLE_TREE)
    model, state, _ = build_model(net)
    hyd = model.dynamic_wave
    terms = hyd._terms({"water.phi": hyd.node_invert + torch.tensor(
        [0.05 if n in ("J1", "J3") else 0.0 for n in hyd.net.nodes], dtype=F64),
        "water.q": torch.tensor([0.05, 0, 0, 0, 0], dtype=F64),
        "sewer.pipe_flow": torch.zeros(5, dtype=F64), "sewer.mid_area": state["sewer.mid_area"]},
        10.0)
    fr = terms.froude[0].item()
    expected = terms.volume[0].item() / (0.05 / QCF) * fr / (1 + fr) * 0.75
    assert fr > 0.01 and expected < 60.0
    assert _step_size_case(**flowing) == pytest.approx(expected, abs=2e-3)
    # Fr <= 0.01 is skipped (dynwave.c:855): a full closed conduit has Fr = 0 whatever its flow
    assert _step_size_case(depths={"J1": 0.6, "J3": 0.6}, q={"C1": 0.05},
                           before={"J1": 0.6, "J3": 0.6}) == 60.0
    # a dry conduit (area <= FUDGE, Fr = 0) with a stale flow never limits the step
    assert _step_size_case(q={"C1": 0.05}) == 60.0


def test_routing_step_size_is_the_fixed_step_without_a_variable_step():
    assert _step_size_case(options={"variable_step": 0.0}, depths={"J1": 0.05}) == 60.0
    # below MINTIMESTEP (1 ms) the routing step is returned as it is (dynwave.c:205)
    assert _step_size_case(options={"routing_step": 0.0005}, depths={"J1": 0.05}) == 0.0005


def test_routing_step_size_first_step_is_the_minimum_step():
    net, _, _ = read_swmm_inp(VARIABLE_TREE)
    model, state, _ = build_model(net)
    assert routing_step_size(model.dynamic_wave, None, state, None) == 0.5


# --- C2: initial conditions, MaxFlow, flap gates, barrels, lengthening -------------------


def _limit_drivers(n, **over):
    """`n` identical conduits on the momentum line ``q = 0.1 + 2 dp`` with no limit, no gate
    and no dry node; `over` replaces driver entries (scalars broadcast)."""
    drv = {"water.flow_offset": 0.1, "water.conveyance": 2.0, "water.normal_flow": float("inf"),
           "water.limited": 0.0, "water.slope_switch": 0.0, "water.switch_drop": 0.0,
           "water.switch_slope": 0.0, "water.flow_last": 0.0, "water.dry_upstream": 0.0,
           "water.dry_downstream": 0.0, "water.max_flow": float("inf"),
           "water.flap_block_reverse": 0.0, "water.flap_block_forward": 0.0,
           "water.dry_flow_cap": FUDGE_FLOW}
    drv.update(over)
    return {k: torch.as_tensor(v, dtype=F64).expand(n).clone() for k, v in drv.items()}


def test_max_flow_caps_the_flow_in_both_directions_and_zeroes_its_slope():
    """`dwflow.c:270-273`: |q| <= MaxFlow, the sign kept; dflow is 0 where the cap acts."""
    el = DynamicWaveConduit()
    drv = _limit_drivers(4, **{"water.max_flow": 0.25})
    dp = torch.tensor([0.05, 0.2, -0.5, -0.1], dtype=F64)
    q = el.flow(dp, drv)
    torch.testing.assert_close(q, torch.tensor([0.2, 0.25, -0.25, -0.1], dtype=F64))
    torch.testing.assert_close(el.dflow(dp, drv), torch.tensor([2.0, 0.0, 0.0, 2.0], dtype=F64))


def test_max_flow_follows_the_normal_flow_limit_and_precedes_the_dry_cap():
    """SWMM's order (dwflow.c:255-281): the normal-flow limit, MaxFlow, the flap gate, then
    the dry-node caps, so a dry upstream node still caps at FUDGE below a large MaxFlow."""
    el = DynamicWaveConduit()
    drv = _limit_drivers(2, **{"water.max_flow": 1.0, "water.normal_flow": 0.15,
                               "water.limited": torch.tensor([1.0, 0.0], dtype=F64),
                               "water.dry_upstream": torch.tensor([0.0, 1.0], dtype=F64)})
    q = el.flow(torch.full((2,), 0.05, dtype=F64), drv)
    torch.testing.assert_close(q, torch.tensor([0.15, FUDGE_FLOW], dtype=F64))


def test_a_flap_gate_blocks_reverse_flow_and_passes_forward_flow():
    """`link_setFlapGate` (link.c:658): a gate closes on flow against its direction (here the
    u -> v direction, so q < 0 gives q = 0); the pass sees flow zero, slope zero there."""
    el = DynamicWaveConduit()
    drv = _limit_drivers(3, **{"water.flap_block_reverse": 1.0})
    dp = torch.tensor([0.05, -0.04, -0.2], dtype=F64)        # q = 0.2, 0.02, -0.3 ungated
    q = el.flow(dp, drv)
    torch.testing.assert_close(q, torch.tensor([0.2, 0.02, 0.0], dtype=F64))
    torch.testing.assert_close(el.dflow(dp, drv), torch.tensor([2.0, 2.0, 0.0], dtype=F64))
    # the reversed gate (SWMM direction -1) closes on forward flow instead; an outfall's gate
    # is the same clamp on whichever side the flow would enter the conduit
    rev = _limit_drivers(3, **{"water.flap_block_forward": 1.0})
    torch.testing.assert_close(el.flow(dp, rev), torch.tensor([0.0, 0.0, -0.3], dtype=F64))
    both = _limit_drivers(3, **{"water.flap_block_forward": 1.0,
                                "water.flap_block_reverse": 1.0})
    torch.testing.assert_close(el.flow(dp, both), torch.zeros(3, dtype=F64))


def test_limits_keep_the_element_monotone_and_continuous_in_dp():
    """The head layer's Newton step needs a continuous flow that is non-decreasing in dp
    with a consistent dflow, whatever limits act (MaxFlow, flap gate, dry caps)."""
    el = DynamicWaveConduit()
    grid = torch.linspace(-1.0, 1.0, 4001, dtype=F64)
    for over in ({"water.max_flow": 0.3},
                 {"water.flap_block_reverse": 1.0},
                 {"water.flap_block_forward": 1.0, "water.max_flow": 0.3},
                 {"water.dry_upstream": 1.0, "water.flap_block_reverse": 1.0}):
        drv = _limit_drivers(grid.numel(), **over)
        q = el.flow(grid, drv)
        assert bool((q[1:] - q[:-1] >= 0.0).all()), over
        assert float((q[1:] - q[:-1]).max()) <= 2.0 * float(grid[1] - grid[0]) + 1e-12, over
        d = el.dflow(grid, drv)
        assert bool((d >= 0.0).all())
        # the analytic slope agrees with the finite difference away from the kinks
        fd = (q[1:] - q[:-1]) / (grid[1] - grid[0])
        mid = 0.5 * (d[1:] + d[:-1])
        assert float((fd - mid).abs().sum() / fd.numel()) < 1e-2, over


def _chain_net(*, barrels=1, max_flow=None, initial_flow=0.0, flap=False, flap_reversed=False,
               outfall_gate=False, outfall_kind="free", stage=None, inflow=0.02,
               initial_depth=0.0, routing_step=5.0, lengthening_step=0.0, length=200.0,
               min_surface_area=None, variable_step=0.0, diameter=0.45):
    """One conduit J1 -> Out (0.45 m, n 0.013, 0.5 % fall over `length`) with an inflow at
    J1: the smallest network that exercises a conduit feature."""
    from noodl.apps.sewer import SewerNetwork
    from noodl.apps.sewer.dynwave import DynamicWaveOptions
    from noodl.apps.sewer.network import Manhole, Outfall, Pipe
    fall = 0.005 * length
    return SewerNetwork(
        manholes=(Manhole("J1", 10.0 + fall, ground=15.0, inflow=inflow,
                          initial_depth=initial_depth),),
        pipes=(Pipe("C1", "J1", "Out", length, diameter, 0.013, 0.005, barrels=barrels,
                    max_flow=max_flow, initial_flow=initial_flow, flap_gate=flap,
                    flap_gate_reversed=flap_reversed),),
        outfalls=(Outfall("Out", 10.0, kind=outfall_kind, stage=stage,
                          flap_gate=outfall_gate),),
        routing="dynamic_wave", geometry="tabulated",
        dynamic_wave=DynamicWaveOptions(
            routing_step=routing_step, variable_step=variable_step,
            lengthening_step=lengthening_step, min_surface_area=min_surface_area,
            head_tolerance=1e-12, max_trials=2000))


def test_two_barrels_carry_twice_one_barrels_flow_at_equal_heads():
    """One barrel and two barrels of the same conduit at the same heads and the same fed
    state: the pass's flow of the pair is exactly twice (offset, conveyance and flow all
    total over the barrels), the per-barrel stored area is the same, and the limits and
    surface areas the closure writes are the barrel-scaled ones."""
    from noodl.model import StepContext
    one, state1, drivers1 = build_model(_chain_net(barrels=1))
    two, state2, drivers2 = build_model(_chain_net(barrels=2))
    # fed state: junction 0.2 m, per-barrel flow 0.01 m3/s (so 0.02 for two barrels)
    for state, k in ((state1, 1.0), (state2, 2.0)):
        state["water.phi"] = torch.tensor([11.0 + 0.2, 10.0], dtype=F64)
        state["sewer.junction_depth"] = torch.tensor([0.2], dtype=F64)
        state["water.q"] = torch.tensor([0.01 * k], dtype=F64)
        state["sewer.pipe_flow"] = torch.tensor([0.01 * k], dtype=F64)
    d1 = one.dynamic_wave(state1, drivers1, StepContext(dt=5.0))
    d2 = two.dynamic_wave(state2, drivers2, StepContext(dt=5.0))
    for key in ("water.flow_offset", "water.conveyance"):
        torch.testing.assert_close(d2[key], 2.0 * d1[key], rtol=1e-12, atol=0)
    dp = torch.tensor([1.2], dtype=F64)
    q1 = d1["water.flow_offset"] + d1["water.conveyance"] * dp
    q2 = d2["water.flow_offset"] + d2["water.conveyance"] * dp
    torch.testing.assert_close(q2, 2.0 * q1, rtol=1e-12, atol=0)
    torch.testing.assert_close(d2["water.dry_flow_cap"], 2.0 * d1["water.dry_flow_cap"])
    # reporting: the pipe's volume is a total, its velocity one barrel's
    torch.testing.assert_close(d2["sewer.V_wet"], 2.0 * d1["sewer.V_wet"], rtol=1e-12, atol=0)
    torch.testing.assert_close(d2["sewer.v"], d1["sewer.v"], rtol=1e-12, atol=0)


def test_two_barrels_at_twice_the_inflow_run_as_one_barrel_per_barrel():
    """Scaling invariance of the whole run: two barrels fed twice the lateral inflow carry
    twice the flow with the junction at the same depth in every step (the junction's surface
    area doubles with the barrels; its floor is made negligible so that it does not break
    the scaling)."""
    runs = {}
    for barrels in (1, 2):
        net = _chain_net(barrels=barrels, inflow=0.02 * barrels, min_surface_area=1e-3)
        model, state, drivers = build_model(net)
        _, hist = simulate_dynamic_wave(model, state, drivers, steps=40,
                                        record=("water.phi", "water.q"))
        runs[barrels] = hist
    torch.testing.assert_close(runs[2]["water.q"], 2.0 * runs[1]["water.q"], rtol=1e-6,
                               atol=1e-9)
    torch.testing.assert_close(runs[2]["water.phi"], runs[1]["water.phi"], rtol=0, atol=1e-7)


def test_a_max_flow_caps_each_barrel_in_a_run():
    """SWMM limits one barrel's flow to MaxFlow (dwflow.c:270-273): two barrels of 0.015
    carry at most 0.03 in total, however hard the junction drives them."""
    net = _chain_net(barrels=2, max_flow=0.015, inflow=0.06, min_surface_area=5.0)
    model, state, drivers = build_model(net)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=60, record=("water.q",))
    q = hist["water.q"][:, 0]
    assert float(q.max()) == pytest.approx(0.03, rel=1e-12)
    assert float(q.min()) >= 0.0


def test_a_flap_gate_blocks_reverse_flow_in_a_run():
    """A FIXED outfall stage above the junction drives water back up the conduit; an ungated
    conduit carries it, a gated outfall (and a gated conduit) carries none."""
    q_end = {}
    for name, kwargs in (("open", {}), ("outfall_gate", {"outfall_gate": True}),
                         ("conduit_gate", {"flap": True})):
        net = _chain_net(outfall_kind="fixed", stage=10.0 + 1.3, inflow=0.0, **kwargs)
        model, state, drivers = build_model(net)
        _, hist = simulate_dynamic_wave(model, state, drivers, steps=40, record=("water.q",))
        q_end[name] = hist["water.q"][:, 0]
    assert float(q_end["open"].min()) < -1e-3
    assert float(q_end["outfall_gate"].abs().max()) == 0.0
    assert float(q_end["conduit_gate"].abs().max()) == 0.0
    # a conduit gate that passes the other way (a reversed conduit) lets this flow through
    net = _chain_net(outfall_kind="fixed", stage=10.0 + 1.3, inflow=0.0, flap=True,
                     flap_reversed=True)
    model, state, drivers = build_model(net)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=40, record=("water.q",))
    torch.testing.assert_close(hist["water.q"][:, 0], q_end["open"])


def test_initial_depth_and_flow_set_the_initial_state_as_swmm_does():
    """flowrout.c:337-420 on one conduit J1 -> Out: with an InitFlow the conduit starts at the
    normal depth of one barrel's flow (link_getYnorm) and J1 at the mean of the attached links'
    depths plus the first offset; with an InitDepth J1 keeps it; the conduit with neither
    starts at the mean of its end depths; the junction's net inflow is minus the initial flow
    through it."""
    from noodl.apps.sewer.xsect_tables import normal_depth_ft
    net = _chain_net(barrels=2, initial_flow=0.04)
    model, state, _ = build_model(net)
    hyd = model.dynamic_wave
    y_norm = normal_depth_ft(torch.tensor([0.02 / QCF], dtype=F64), hyd.geo.y_full,
                             hyd.geo.a_full, hyd.geo.s_full, hyd.geo.beta)
    # J1: its one link's depth counts at both ends, so the mean is that depth (flowrout.c:358)
    torch.testing.assert_close(state["sewer.junction_depth"], y_norm * LCF, rtol=1e-12,
                               atol=0)
    torch.testing.assert_close(state["sewer.mid_area"],
                               area_of_depth(y_norm, hyd.geo.y_full, hyd.geo.a_full)
                               * QCF / LCF, rtol=1e-12, atol=0)
    torch.testing.assert_close(state["water.q"], torch.tensor([0.04], dtype=F64))
    torch.testing.assert_close(state["sewer.pipe_flow"], torch.tensor([0.04], dtype=F64))
    torch.testing.assert_close(state["sewer.net_inflow"], torch.tensor([-0.04], dtype=F64))
    # an InitDepth wins over the average
    net = _chain_net(initial_depth=0.15, initial_flow=0.04)
    _, state, _ = build_model(net)
    assert state["sewer.junction_depth"].item() == pytest.approx(0.15)
    # InitDepth alone: the conduit starts at the mean of its end depths above its offsets
    net = _chain_net(initial_depth=0.15)
    model, state, _ = build_model(net)
    hyd = model.dynamic_wave
    y = torch.tensor([0.5 * 0.15 / LCF], dtype=F64)
    torch.testing.assert_close(state["sewer.mid_area"],
                               area_of_depth(y, hyd.geo.y_full, hyd.geo.a_full) * QCF / LCF)


def test_initial_depth_deeper_than_the_full_depth_is_refused_by_name():
    with pytest.raises(ValueError, match=r"J1.*initial_depth"):
        build_model(_chain_net(initial_depth=7.0))     # ground 15 m, invert 11 m: 4 m deep


def test_an_initial_flow_starts_the_run_from_that_flow():
    """A conduit carrying its InitFlow with the same lateral inflow at J1 starts near steady:
    the first step's flow stays near the initial flow instead of rising from zero."""
    net = _chain_net(initial_flow=0.04, inflow=0.04, min_surface_area=5.0)
    model, state, drivers = build_model(net)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=2, record=("water.q",))
    assert float(hist["water.q"][0, 0]) == pytest.approx(0.04, rel=0.05)


def test_lengthening_makes_a_short_conduit_numerically_longer():
    """LENGTHENING_STEP lengthens a conduit shorter than the Courant length: modLength =
    lf L with lf = (sqrt(g yFull) + vFull) t / L (link.c:1217-1254), the volume stays at the
    true length, a longer conduit is untouched."""
    from noodl.apps.sewer.dynwave_terms import lengthening_factor
    net = _chain_net(length=8.0, lengthening_step=20.0, routing_step=5.0)
    model, _, _ = build_model(net)
    geo = model.dynamic_wave.geo
    lf = lengthening_factor(torch.tensor([0.45], dtype=F64), torch.tensor([0.013], dtype=F64),
                            torch.tensor([0.005], dtype=F64), torch.tensor([8.0], dtype=F64),
                            5.0)                              # min(routing 5, lengthening 20)
    assert lf.item() > 1.0
    torch.testing.assert_close(geo.mod_length, lf * geo.length)
    assert geo.length.item() == pytest.approx(8.0 / LCF)
    # the file's ROUTING_STEP governs, not the WET_STEP-lowered one
    net2 = dataclasses.replace(net, dynamic_wave=dataclasses.replace(
        net.dynamic_wave, lengthening_routing_step=10.0))
    geo2 = build_model(net2)[0].dynamic_wave.geo
    assert geo2.mod_length.item() > geo.mod_length.item()
    # a long conduit and LENGTHENING_STEP 0 are unchanged
    for n in (_chain_net(length=200.0, lengthening_step=20.0), _chain_net(length=8.0)):
        g = build_model(n)[0].dynamic_wave.geo
        torch.testing.assert_close(g.mod_length, g.length)


def test_network_refuses_dynamic_wave_features_for_other_routing():
    from noodl.apps.sewer import tree_steady
    from noodl.apps.sewer.network import Manhole, Outfall
    base = tree_steady()
    for field, value in (("barrels", 2), ("max_flow", 0.1), ("initial_flow", 0.1),
                         ("flap_gate", True)):
        pipes = (dataclasses.replace(base.pipes[0], **{field: value}),) + base.pipes[1:]
        with pytest.raises(ValueError, match=rf"{base.pipes[0].name}.*{field}"):
            dataclasses.replace(base, pipes=pipes).validate()
    with pytest.raises(ValueError, match="J1.*initial_depth"):
        dataclasses.replace(base, manholes=(Manhole("J1", 12.0, initial_depth=0.1),)
                            + base.manholes[1:]).validate()
    with pytest.raises(ValueError, match="Outfall.*flap gate"):
        dataclasses.replace(base, outfalls=(Outfall("Outfall", 8.0, flap_gate=True),)
                            ).validate()


def test_dynamic_wave_validates_barrels_and_limits():
    for kwargs, match in (({"barrels": 0}, "barrels"), ({"barrels": 128}, "barrels"),
                          ({"barrels": 1.5}, "barrels"), ({"max_flow": -1.0}, "max_flow"),
                          ({"initial_flow": float("nan")}, "initial_flow")):
        net = _chain_net()
        pipe = dataclasses.replace(net.pipes[0], **kwargs)
        with pytest.raises(ValueError, match=rf"C1.*{match}"):
            dataclasses.replace(net, pipes=(pipe,)).validate()


def _courant_step(*, total_q_cfs, jdepth=0.012, **net_kwargs):
    """`routing_step_size` on a one-conduit network (J1 `jdepth` m deep, a dry outfall) whose
    conduit carries `total_q_cfs` (cfs, total over its barrels), routing step 60 s,
    VARIABLE_STEP 0.75, junction not moving (so only the conduit's Courant step can bind)."""
    net = _chain_net(variable_step=0.75, routing_step=60.0, length=8.0, diameter=0.1,
                     **net_kwargs)
    model, state, _ = build_model(net)
    hyd = model.dynamic_wave
    state = dict(state)
    q = torch.tensor([total_q_cfs * QCF], dtype=F64)
    state["water.q"] = state["sewer.pipe_flow"] = q
    phi = hyd.node_invert.clone()
    phi[hyd.junction_idx] += jdepth
    state["water.phi"] = phi
    state["sewer.junction_depth"] = torch.tensor([jdepth], dtype=F64)
    y_mid = torch.tensor([0.5 * jdepth / LCF], dtype=F64)
    state["sewer.mid_area"] = area_of_depth(y_mid, hyd.geo.y_full, hyd.geo.a_full) * QCF / LCF
    return routing_step_size(hyd, dict(state), state, 5.0)


def test_variable_step_uses_per_barrel_flow_and_volume():
    """getLinkStep (dynwave.c:857-863): q and volume are one barrel's. (a) the Courant step
    binds (well under the 60 s routing step) and equals the one-barrel step for n barrels at
    n times the flow, because V/q cancels the barrel count; (b) the skip ``|q| <= FUDGE``
    (dynwave.c:857) tests the PER-BARREL flow: a total of 1.2e-4 cfs is live for one barrel
    (1.2e-4 > FUDGE) but each of two barrels carries 6e-5 <= FUDGE, so the conduit does not
    limit the step and it is the 60 s routing step."""
    one = _courant_step(total_q_cfs=1.5e-3)
    two = _courant_step(total_q_cfs=3e-3, barrels=2)
    assert one < 30.0                                  # the Courant step binds
    assert two == pytest.approx(one, abs=1e-3)
    live = _courant_step(total_q_cfs=1.2e-4)
    skipped = _courant_step(total_q_cfs=1.2e-4, barrels=2)
    assert live < 60.0                                 # one barrel: 1.2e-4 cfs counts
    assert skipped == 60.0                             # two barrels: 6e-5 cfs each is skipped


def test_variable_step_scales_the_courant_step_by_mod_length_over_length():
    """dynwave.c:865: t *= modLength / length, so a lengthened conduit's step is the plain
    one times its lengthening factor (the plain step stays below the 60 s routing step)."""
    from noodl.apps.sewer.dynwave_terms import lengthening_factor
    plain = _courant_step(total_q_cfs=6e-3)
    long = _courant_step(total_q_cfs=6e-3, lengthening_step=20.0)
    lf = lengthening_factor(torch.tensor([0.1], dtype=F64), torch.tensor([0.013], dtype=F64),
                            torch.tensor([0.005], dtype=F64), torch.tensor([8.0], dtype=F64),
                            20.0).item()                      # min(routing 60 s, lengthening 20 s)
    assert lf > 1.0 and plain * lf < 60.0
    assert long == pytest.approx(plain * lf, abs=2e-3)


# ------------------------------------------------------------- the drop-end area switch
OPTIONS = DATA / "options_dynwave.inp"
#: Variants of options_dynwave.inp on which a junction below a drop fills through its
#: area threshold: (the fixture line, its replacement, the steps (1-based) on which NOODL'S
#: OWN RUN holds the junction on its threshold over all 359 steps, the junction). Those are
#: the steps on which SWMM's own run first fails to converge; SWMM's full list of
#: non-converging steps is in tests/verification/test_sewer_dynwave_parity.py
#: (``DROP_SWMM_FAILS``): it adds step 167 of the 0.03 m variant, which lies in SWMM's own
#: history (noodl's differs after step 60) and where noodl, started from SWMM's state,
#: holds J5 (asserted there).
DROP_VARIANTS = {
    # C3 (J3 -> J4, two barrels) gets a 0.03 m outlet offset: J4 fills behind the closed
    # outfall gate, through C3's DN_CRITICAL threshold min(yN, yC)
    "c3_offset_0.03": ("C3               J3               J4               200        0.013"
                       "      0          0  ",
                       "C3               J3               J4               200        0.013"
                       "      0          0.03",
                       (60, 62), "J4"),
    # the same at 0.1 m: J4 fills through C3's fasnh threshold max(yN, yC)
    "c3_offset_0.1": ("C3               J3               J4               200        0.013"
                      "      0          0  ",
                      "C3               J3               J4               200        0.013"
                      "      0          0.1",
                      (107,), "J4"),
    # C6 (J3 -> J5 once reversed, 0.05 m drop at J5) without its flap gate: J5 fills
    # through C6's DN_CRITICAL threshold
    "c6_gate_off": ("C6               0          0          0          YES        0",
                    "C6               0          0          0          NO         0",
                    (106, 129), "J5"),
}


def _drop_variant(name):
    old, new, _, _ = DROP_VARIANTS[name]
    text = OPTIONS.read_text()
    assert text.count(old) == 1, name
    return text.replace(old, new)


def _drop_sides(tmp_path, name, steps, **build_kwargs):
    """The variant `name` over `steps` steps: the diagnostics and, per step, the junctions
    held on a drop-end area threshold, ``{junction: (H - H*, theta)}`` (m, -), each other
    junction with a threshold checked to be on SWMM's own fixed point -- below it with the
    net inflow at most the lower storage line there, or above it with at least the upper
    one -- from the drivers of a pass fed the step's result."""
    from noodl.apps.sewer.dynwave import build_dynamic_wave_model
    from noodl.model import StepContext

    path = tmp_path / f"{name}.inp"
    path.write_text(_drop_variant(name))
    net, _, _ = read_swmm_inp(path)
    model, state, drivers = build_dynamic_wave_model(net, **build_kwargs)
    hyd = model.dynamic_wave
    diags: list = []
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=steps, diagnostics=diags,
                                    record=("water.phi", "water.q") + hyd.state_keys)
    held = []
    for k in range(steps):
        before = {key: (hist[key][k - 1] if k else state[key]) for key in hyd.state_keys}
        after = {key: hist[key][k] for key in ("water.phi", "water.q")}
        drv = hyd(after | before, drivers, StepContext(dt=hyd.options.routing_step))
        h_star = drv["water.area_switch_head"]
        rate2 = 2.0 * drv["water.storage_rate"]
        rise = h_star - drv["water.head_start"]
        lo = rate2 * drv["water.area_below"] * rise - drv["water.net_inflow_start"]
        hi = rate2 * drv["water.area_above"] * rise - drv["water.net_inflow_start"]
        n = drv["water.net_inflow_last"]
        heads = after["water.phi"][model.junction_idx]
        step = {}
        for j, junction in enumerate(hyd.junction_names):
            if not bool(torch.isfinite(h_star[j])):
                continue
            if lo[j] < n[j] < hi[j]:
                step[junction] = ((heads[j] - h_star[j]).item(),
                                  ((n[j] - lo[j]) / (hi[j] - lo[j])).item())
            else:
                assert (heads[j] < h_star[j] and n[j] <= lo[j]) or \
                    (heads[j] > h_star[j] and n[j] >= hi[j]), (name, k + 1, junction)
        held.append(step)
    return diags, held


def test_a_junction_filling_below_a_drop_is_held_on_its_area_threshold(tmp_path):
    """Below an outlet offset a junction's surface area jumps UP with its own depth: the
    conduit end gives it no area while DN_CRITICAL (end depth below min(yN, yC), or below
    FUDGE) and a quarter of the conduit above (dwflow.c:357-411, 472-499). While the
    junction FILLS, the storage equation can then land above the threshold with the lower
    area and below it with the upper one: no fixed point. SWMM 5.2.4 itself does not
    converge there (steps 60 and 62 of this variant, `DROP_VARIANTS`), and a lagged class
    chattered until the pass budget ran out (step 60, t = 295 s; at the parity tests'
    iterate_tol 1e-13 m the head solve failed first, C3's class also moving J3's area).
    With the tie rule (`DynamicWaveHydraulics`, held
    junctions in `JunctionStorage`) every step converges, J4 is held exactly on its
    threshold on SWMM's two failing steps and on SWMM's own fixed point on every other."""
    diags, held = _drop_sides(tmp_path, "c3_offset_0.03", 62)
    assert all(bool(d["converged"]) for d in diags)
    assert max(d["passes"] for d in diags) <= 30, max(d["passes"] for d in diags)
    on = {k + 1: step for k, step in enumerate(held) if step}
    assert set(on) == {60, 62} and all(set(s) == {"J4"} for s in on.values()), on
    for step in on.values():
        gap, theta = step["J4"]
        assert abs(gap) <= 1e-9 and 0.0 < theta < 1.0, step


def test_carried_keys_after_a_held_step_are_the_held_passes(tmp_path):
    """At a step that holds J4 on its drop-end area threshold (step 60 of the 0.03 m offset
    variant) `simulate_dynamic_wave`'s resync (`carried` with the step start) gives the
    carried keys of `model.step` alone, whose last pass took the hold: the hold is decided
    against the step-start depths and net inflows, as within the step. (The held class's
    mid area differs from the unheld one only by round-off at the threshold, 4e-15 m2
    measured, but the resync takes the same terms as the pass.)"""
    from noodl.apps.sewer.dynwave import build_dynamic_wave_model

    path = tmp_path / "c3_offset_0.03.inp"
    path.write_text(_drop_variant("c3_offset_0.03"))
    net, _, _ = read_swmm_inp(path)
    model, state, drivers = build_dynamic_wave_model(net)
    hyd = model.dynamic_wave
    dt = hyd.options.routing_step
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=59)
    alone = model.step(dict(state), drivers, dt, t=59 * dt, atol=solve_atol(model, dt))
    pinned = alone | {k: state[k] for k in hyd.state_keys}
    terms = hyd._terms(pinned, dt)
    drop = hyd._drop_switch(pinned, terms, hyd._carried(pinned, terms, hyd._inflow(drivers))
                            ["sewer.net_inflow"], dt, hyd._storage_area_ft2(pinned["water.phi"]))
    assert drop["held"].tolist() == [j == "J4" for j in hyd.junction_names]
    resynced = hyd.carried(alone, drivers, dt=dt, start=state)
    simulated, _ = simulate_dynamic_wave(model, dict(state), drivers, steps=1)
    for key in hyd.state_keys:
        assert torch.allclose(resynced[key], alone[key], rtol=0.0, atol=1e-12), key
        assert torch.allclose(simulated[key], alone[key], rtol=0.0, atol=1e-12), key


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(DROP_VARIANTS))
@pytest.mark.parametrize("acceleration", ["anderson", None])
def test_every_drop_variant_converges_held_only_where_swmm_first_fails(tmp_path, name,
                                                                       acceleration):
    """Each variant, Anderson over all 359 steps and plain relaxation through one step past
    the last held one: every step converges, and the junction is held on its threshold on
    the listed steps (those where SWMM's own run fails, `DROP_VARIANTS`) and nowhere else --
    on SWMM's own fixed point on every other step."""
    _, _, held_steps, junction = DROP_VARIANTS[name]
    steps = 359 if acceleration == "anderson" else max(held_steps) + 1
    diags, held = _drop_sides(tmp_path, name, steps, acceleration=acceleration)
    assert all(bool(d["converged"]) for d in diags)
    on = {k + 1: step for k, step in enumerate(held) if step}
    assert set(on) == set(held_steps) and all(set(s) == {junction} for s in on.values()), on
    for step in on.values():
        gap, theta = step[junction]
        assert abs(gap) <= 1e-9 and 0.0 < theta < 1.0, step


def test_gradient_on_a_held_threshold_does_not_depend_on_the_wall_slope(tmp_path,
                                                                        monkeypatch):
    """At a step that holds a junction on its drop-end area threshold (J4 at step 107 of the
    0.1 m offset variant, where SWMM has no fixed point) the returned gradient is the tie
    rule's: the sensitivity of a solution held at H = H*, with the withdrawal inside the
    jump. The wall slope S drops out of it, as the fixed point does not move with S:
    d(J4 head + 10 C3 flow)/d(n of C3) over that step is the same for S scaled by 0.2, 1
    and 5 (to 1e-10 relative; 5e-13 measured) and matches central differences (1.6e-8
    relative measured)."""
    from noodl.apps.sewer.dynwave import build_dynamic_wave_model
    from noodl.model import StepContext

    path = tmp_path / "c3_offset_0.1.inp"
    path.write_text(_drop_variant("c3_offset_0.1"))
    net, _, _ = read_swmm_inp(path)
    model, state0, drivers = build_dynamic_wave_model(net)
    state0, _ = simulate_dynamic_wave(model, state0, drivers, steps=106, atol=1e-12)
    state0 = {k: v.detach() for k, v in state0.items()}
    hyd = model.dynamic_wave
    c3 = hyd.pipe_names.index("C3")
    j4 = list(model.net.nodes).index("J4")
    j4_junction = hyd.junction_names.index("J4")

    def loss(n3):
        hyd.set_roughness(c3, n3)
        s, _ = simulate_dynamic_wave(model, state0, drivers, steps=1, atol=1e-12)
        with torch.no_grad():
            drv = hyd(s | {k: state0[k] for k in hyd.state_keys}, drivers,
                      StepContext(dt=hyd.options.routing_step))
            gap = s["water.phi"][j4] - drv["water.area_switch_head"][j4_junction]
            assert abs(gap.item()) < 1e-9                       # held on the threshold
        return s["water.phi"][j4] + 10.0 * s["water.q"][c3]

    original = JunctionStorage._area_switch
    grads = []
    for scale in (1.0, 0.2, 5.0):
        def scaled(phi, rise, storage, slope, drv, scale=scale):
            drv = dict(drv, **{"water.area_switch_slope": scale * drv["water.area_switch_slope"]})
            return original(phi, rise, storage, slope, drv)
        monkeypatch.setattr(JunctionStorage, "_area_switch", staticmethod(scaled))
        n = torch.tensor(0.013, dtype=F64, requires_grad=True)
        loss(n).backward()
        grads.append(n.grad.item())
    monkeypatch.setattr(JunctionStorage, "_area_switch", staticmethod(original))
    assert grads[1] == pytest.approx(grads[0], rel=1e-10)
    assert grads[2] == pytest.approx(grads[0], rel=1e-10)
    h = 1e-6
    with torch.no_grad():
        fd = (loss(torch.tensor(0.013 + h, dtype=F64))
              - loss(torch.tensor(0.013 - h, dtype=F64))) / (2 * h)
    assert grads[0] == pytest.approx(fd.item(), rel=1e-6)


# ------------------------------------------------------------------------- storage units
def _storage_chain(curve=("functional", 0.0, 0.0, 50.0), *, max_depth=2.0, surcharge=0.0,
                   initial_depth=0.0, inflow=0.05, method="extran", **chain):
    """`_chain_net` with J1 a storage unit (`storage_unit`) of area curve `curve` (default a
    50 m2 prism), MaxDepth `max_depth`, SurDepth `surcharge` and InitDepth `initial_depth`;
    C1 is 0.45 m, so J1's crown is 0.45 m."""
    from noodl.apps.sewer.network import storage_unit
    net = _chain_net(inflow=inflow, **chain)
    j1 = net.manholes[0]
    unit = storage_unit("J1", j1.invert, max_depth, curve, surcharge_depth=surcharge,
                        initial_depth=initial_depth, inflow=inflow)
    return dataclasses.replace(net, manholes=(unit,), dynamic_wave=dataclasses.replace(
        net.dynamic_wave, surcharge_method=method))


def _fed(model, state, depth, q=0.0):
    """`state` fed with J1 at `depth` (m) and C1 at `q` (m3/s), the step start equal."""
    hyd = model.dynamic_wave
    state = dict(state)
    phi = state["water.phi"].clone()
    phi[model.junction_idx] = model.junction_invert + depth
    state["water.phi"] = phi
    state["sewer.junction_depth"] = torch.tensor([depth], dtype=F64)
    state["water.q"] = state["sewer.pipe_flow"] = torch.tensor([q], dtype=F64)
    return hyd, state


def test_a_storage_units_area_adds_to_its_conduit_areas_at_the_fed_depth():
    """node.c:383-396 and dynwave.c:315, 582-583: a storage node's surface area is its
    curve's at the previous trial's depth plus the attached conduit ends' (which SWMM does
    NOT zero at a storage node: only regulator ends are, dynwave.c:520-523), before the
    MIN_SURFAREA floor; in SI area (ft2 x QCF/LCF, `_AREA_SI`)."""
    from noodl.model import StepContext
    curve = ("functional", 20.0, 1.5, 5.0)
    plain, s0, _ = build_model(_chain_net())
    unit, s1, d1 = build_model(_storage_chain(curve))
    for depth in (0.0, 0.2, 0.6):
        hyd0, st0 = _fed(plain, s0, depth, 0.01)
        hyd1, st1 = _fed(unit, s1, depth, 0.01)
        a1 = hyd1(st1, d1, StepContext(dt=5.0))["water.storage_area"]
        links_ft2 = hyd0._terms(st0, 5.0).surface_area1.item()      # C1's end at J1, ft2
        stor_ft2 = (5.0 + 20.0 * depth**1.5) / LCF**2                       # node.c:1017
        expected = max(links_ft2 + stor_ft2, hyd0.min_area_ft2) * QCF / LCF
        assert a1.item() == pytest.approx(expected, rel=1e-12)
    assert links_ft2 > 1.0                       # the conduit's share is really added


def test_storage_full_depth_extends_to_the_crown_only_with_a_surcharge_depth():
    """link.c:450, 458: a storage node's full depth is raised to the attached crowns only
    when its SurDepth is positive; it floods above full depth + SurDepth."""
    hyd = build_model(_storage_chain(max_depth=0.2))[0].dynamic_wave
    assert hyd.full_depth.item() == pytest.approx(0.2)            # not the 0.45 m crown
    assert hyd.flood_depth.item() == pytest.approx(0.2)
    hyd = build_model(_storage_chain(max_depth=0.2, surcharge=0.5))[0].dynamic_wave
    assert hyd.full_depth.item() == pytest.approx(0.45)
    assert hyd.flood_depth.item() == pytest.approx(0.95)


def test_storage_surcharges_above_its_full_depth_and_only_with_a_surcharge_depth():
    """dynwave.c:683-687 (EXTRAN): a storage node is surcharged when SurDepth > 0 and the
    previous trial's depth exceeds its FULL depth -- not its crown, which a plain junction
    uses (dynwave.c:690). The solve's flag (``water.surcharged``) and the reported SWMM test
    (``sewer.surcharged``) agree on fed depths clear of the hold band. A unit draining at
    its full depth (net inflow -0.2 m3/s at the step start, at rest) is used, so that the
    band is positive wherever the depth is above the full depth."""
    from noodl.model import StepContext

    def flags(net, depth):
        model, state, drivers = build_model(net)
        hyd, state = _fed(model, state, depth, 0.2)
        state["sewer.net_inflow"] = torch.tensor([-0.2], dtype=F64)
        out = hyd(state, dict(drivers, inflow=torch.zeros(2, dtype=F64)),
                  StepContext(dt=5.0))
        return out["water.surcharged"].item(), out["sewer.surcharged"].item()

    # no SurDepth: never surcharged, even above the crown (0.45 m) and the full depth
    assert flags(_storage_chain(max_depth=1.0), 0.8) == (0.0, 0.0)
    # SurDepth: above the crown but below the full depth (1.0 m) is not surcharged ...
    assert flags(_storage_chain(max_depth=1.0, surcharge=1.0), 0.8) == (0.0, 0.0)
    # ... above the full depth it is
    assert flags(_storage_chain(max_depth=1.0, surcharge=1.0), 1.2) == (1.0, 1.0)
    # a plain junction at 0.8 m is above its 0.45 m crown
    assert flags(_chain_net(), 0.8)[1] == 1.0


def test_a_storage_unit_keeps_its_own_initial_depth():
    """flowrout.c:375: SWMM does not average a storage node's initial depth from its
    conduits; an InitFlow on C1 would lift a plain junction's, the storage unit stays at its
    InitDepth (0 or not) and its initial volume is the curve's."""
    plain = build_model(_chain_net(initial_flow=0.04))[1]
    assert plain["sewer.junction_depth"].item() > 0.05
    for depth in (0.0, 0.3):
        model, state, drivers = build_model(_storage_chain(initial_flow=0.04,
                                                           initial_depth=depth))
        assert state["sewer.junction_depth"].item() == depth
        out = model.dynamic_wave(state, drivers)
        assert out["sewer.storage_volume"].item() == pytest.approx(50.0 * depth, rel=1e-14)


def test_a_storage_unit_attenuates_a_step_inflow():
    """A 200 m2 prism at J1 under a 0.05 m3/s step inflow: a plain junction passes nearly
    all of it to C1 within 200 s, the storage unit far less while it fills, never more than
    the plain junction; its reported volume is the prism's 200 y and rises every step."""
    prism = ("functional", 0.0, 0.0, 200.0)
    runs = {}
    for name, net in (("plain", _chain_net(inflow=0.05)), ("unit", _storage_chain(prism))):
        model, state, drivers = build_model(net)
        _, hist = simulate_dynamic_wave(model, state, drivers, steps=40,
                                        record=("water.q", "water.phi", "sewer.storage_volume"))
        runs[name] = hist
    q_plain, q_unit = runs["plain"]["water.q"][:, 0], runs["unit"]["water.q"][:, 0]
    assert float(q_plain[-1]) > 0.045
    assert float(q_unit[-1]) < 0.5 * float(q_plain[-1])
    assert bool((q_unit <= q_plain + 1e-12).all())
    hist = runs["unit"]
    depth = hist["water.phi"][:, 0] - 11.0
    volume = hist["sewer.storage_volume"][:, 0]
    torch.testing.assert_close(volume, 200.0 * depth, rtol=1e-12, atol=1e-12)
    assert bool((volume[1:] > volume[:-1]).all())
    assert float(runs["plain"]["sewer.storage_volume"].abs().max()) == 0.0


def test_each_step_satisfies_the_discrete_junction_equation_at_a_storage_unit():
    model, state, drivers = build_model(_storage_chain(("tabular", (0.0, 0.5, 1.0),
                                                        (5.0, 30.0, 40.0))))
    for _ in range(30):
        before = dict(state)
        state, _ = simulate_dynamic_wave(model, state, drivers, steps=1, atol=SOLVE_ATOL)
        res = model.dynamic_wave.junction_residual(before, state, drivers)
        assert res.abs().max().item() < 1e-9


def test_a_storage_unit_without_surcharge_depth_floods_at_its_max_depth():
    """MaxDepth 0.3 m below C1's 0.45 m crown and no SurDepth: the full depth is not
    extended (link.c:450), so the unit floods at 0.3 m and the excess leaves as overflow;
    its volume stays the full volume (dynwave.c:785-786)."""
    net = _storage_chain(("functional", 0.0, 0.0, 2.0), max_depth=0.3, inflow=0.2)
    model, state, drivers = build_model(net)
    state, hist = simulate_dynamic_wave(model, state, drivers, steps=20,
                                        record=("water.phi", "sewer.overflow",
                                                "sewer.storage_volume"))
    depth = hist["water.phi"][:, 0] - 11.0
    assert float(depth.max()) == pytest.approx(0.3, abs=1e-9)
    flooded = hist["sewer.overflow"][:, 0] > 0.0
    assert bool(flooded[-1])
    torch.testing.assert_close(hist["sewer.storage_volume"][flooded, 0],
                               torch.full((int(flooded.sum()),), 0.6, dtype=F64))


def test_storage_units_are_refused_by_name_outside_their_scope():
    """A storage unit under kinematic routing, a storage unit that also sets the kinematic
    shaft `surface_area`, a bad curve and a negative MaxDepth are each refused naming J1."""
    from noodl.apps.sewer.network import storage_unit
    net = _storage_chain()
    with pytest.raises(ValueError, match="J1.*storage unit.*dynamic-wave"):
        dataclasses.replace(net, routing="kinematic_wave").validate()
    bad = dataclasses.replace(net, manholes=(dataclasses.replace(net.manholes[0],
                                                                 surface_area=2.0),))
    with pytest.raises(ValueError, match="J1.*surface_area"):
        bad.validate()
    bad = dataclasses.replace(net, manholes=(dataclasses.replace(
        net.manholes[0], storage=("tabular", (1.0, 0.5), (2.0, 3.0))),))
    with pytest.raises(ValueError, match="J1.*strictly increasing"):
        bad.validate()
    with pytest.raises(ValueError, match="J1.*max_depth"):
        storage_unit("J1", 11.0, -1.0, ("functional", 0.0, 0.0, 5.0))
    # a tabular curve given as lists is stored as tuples (the manhole stays hashable)
    unit = storage_unit("J1", 11.0, 2.0, ("tabular", [0.0, 1.0], [5.0, 6.0]))
    assert unit.storage == ("tabular", (0.0, 1.0), (5.0, 6.0)) and hash(unit)


# ------------------------------------------------------------------- orifices, weirs, outlets
def _regulated(kind="orifice", *, storage_at=None, outfall_kind="free", stage=None,
               inflow=0.04, crest=0.1, j3_invert=10.5, **device):
    """J1 --C1 (0.6 m)--> J2 --[a regulator `kind`]--> J3 --C3 (0.6 m)--> Out, an inflow at
    J1; with ``kind="outlet_to_outfall"`` the outlet runs J2 -> Out directly (no J3, C3).
    The orifice is a 0.2 x 0.3 m side opening (Cd 0.65), the weir a 1.0 x 0.5 m
    transverse weir (Cd 1.84), the outlet ``q = 0.08 h^0.5``; each `crest` m above J2's
    invert (11.0 m). `storage_at` makes that junction a 3 m2 storage unit."""
    from noodl.apps.sewer import SewerNetwork
    from noodl.apps.sewer.dynwave import DynamicWaveOptions
    from noodl.apps.sewer.network import (
        Manhole,
        Orifice,
        Outfall,
        Outlet,
        Pipe,
        Weir,
        storage_unit,
    )
    manholes = [Manhole("J1", 12.0, ground=17.0, inflow=inflow),
                Manhole("J2", 11.0, ground=16.0), Manhole("J3", j3_invert, ground=15.0)]
    pipes = [Pipe("C1", "J1", "J2", 200.0, 0.6, 0.013, 0.005),
             Pipe("C3", "J3", "Out", 200.0, 0.6, 0.013, (j3_invert - 9.5) / 200.0)]
    orifices, weirs, outlets = (), (), ()
    v = "J3"
    if kind == "outlet_to_outfall":
        manholes, pipes, kind, v = manholes[:2], pipes[:1], "outlet", "Out"
    if kind == "orifice":
        args = dict(orientation="side", shape="rectangular", height=0.2, width=0.3,
                    discharge_coefficient=0.65) | device
        orifices = (Orifice("R1", "J2", v, crest_height=crest, **args),)
    elif kind == "weir":
        args = dict(kind="transverse", height=1.0, width=0.5,
                    discharge_coefficient=1.84) | device
        weirs = (Weir("R1", "J2", v, crest_height=crest, **args),)
    else:
        args = dict(rating=("functional", 0.08, 0.5)) | device
        outlets = (Outlet("R1", "J2", v, crest_height=crest, **args),)
    if storage_at is not None:
        manholes = [storage_unit(m.name, m.invert, 5.0, ("functional", 0.0, 0.0, 3.0),
                                 inflow=m.inflow) if m.name == storage_at else m
                    for m in manholes]
    return SewerNetwork(
        manholes=tuple(manholes), pipes=tuple(pipes),
        outfalls=(Outfall("Out", 9.5 if v == "J3" else 10.0, kind=outfall_kind,
                          stage=stage),),
        routing="dynamic_wave", geometry="tabulated",
        dynamic_wave=DynamicWaveOptions(routing_step=5.0, head_tolerance=1e-12,
                                        max_trials=2000),
        orifices=orifices, weirs=weirs, outlets=outlets)


def _fed_heads(model, state, depths):
    """`state` fed with the junctions at `depths` (m, junction order), the step start equal,
    no flow."""
    state = dict(state)
    phi = state["water.phi"].clone()
    phi[model.junction_idx] = model.junction_invert + torch.tensor(depths, dtype=F64)
    state["water.phi"] = phi
    state["sewer.junction_depth"] = torch.tensor(depths, dtype=F64)
    state["water.q"] = torch.zeros_like(state["water.q"])
    state["sewer.pipe_flow"] = torch.zeros_like(state["sewer.pipe_flow"])
    return state


def test_structure_element_is_swmms_flow_linearised_about_the_fed_heads():
    """Regulator flow: ``q = c0 + g (dp - dp_fed)``, so at dp = dp_fed it is c0 (SWMM's flow)
    whatever g; a flap gate clamps it at 0 (dflow 0 there); dflow is g elsewhere."""
    from noodl.apps.sewer.dynwave import StructureLink
    el = StructureLink("weir")
    one = torch.ones(3, dtype=F64)
    drv = {"water.weir.flow_offset": 0.05 * one, "water.weir.slope": 0.4 * one,
           "water.weir.dp_fed": torch.tensor([0.3, 0.3, 0.3], dtype=F64),
           "water.weir.flap_block_reverse": torch.tensor([0.0, 1.0, 0.0], dtype=F64),
           "water.weir.flap_block_forward": torch.tensor([0.0, 0.0, 1.0], dtype=F64)}
    dp = torch.tensor([0.3, 0.1, 0.4], dtype=F64)
    torch.testing.assert_close(el.flow(dp, drv), torch.tensor([0.05, 0.0, 0.0], dtype=F64))
    torch.testing.assert_close(el.dflow(dp, drv), torch.tensor([0.4, 0.0, 0.0], dtype=F64))
    assert el.flow(torch.tensor([0.5, 0.5, -1.0], dtype=F64), drv)[1].item() == (
        pytest.approx(0.13))
    with pytest.raises(ValueError, match="kind must be one of"):
        StructureLink("gate")


@pytest.mark.parametrize("kind", ["orifice", "weir", "outlet"])
def test_the_closure_feeds_swmms_regulator_flow_at_the_fed_heads(kind):
    """dynwave.c:423-454: each regulator's c0 is `link_getInflow` at the previous pass's
    node heads (m3/s = cfs x QCF), its slope SWMM's dqdh (an outlet's a central
    difference; m2/s = cfs/ft x QCF/LCF), its dp_fed the fed head difference."""
    from noodl.apps.sewer import structures as st
    from noodl.model import StepContext
    model, state, drivers = build_model(_regulated(kind))
    hyd = model.dynamic_wave
    s = hyd.structures[kind]
    fed = _fed_heads(model, state, [0.1, 0.25, 0.05])
    out = hyd(fed, drivers, StepContext(dt=5.0))
    h = fed["water.phi"] / LCF
    z = hyd.node_invert / LCF
    args = (s.params, h[s.u], h[s.v], z[s.u], z[s.v], z[s.u] + s.offset1)
    fn = {"orifice": st.orifice_flow, "weir": st.weir_flow, "outlet": st.outlet_flow}[kind]
    ref = fn(*args)
    slope = st.outlet_dqdh(*args) if kind == "outlet" else ref.dqdh
    assert ref.flow.item() > 0.0
    torch.testing.assert_close(out[f"water.{kind}.flow_offset"], ref.flow * QCF)
    torch.testing.assert_close(out[f"water.{kind}.slope"], slope * QCF / LCF)
    assert out[f"water.{kind}.dp_fed"].item() == pytest.approx(0.25 + 11.0 - 10.55)
    assert model.link_names == ["C1", "C3", "R1"]
    assert hyd.edge_kinds == ("pipe", kind)


def test_orifice_end_areas_join_their_junctions_zeroed_at_a_critical_end():
    """dynwave.c:503-524: an orifice gives each end node half its surfArea (a side
    orifice's ``W(f yFull) L``, L >= 200 ft), but none to its downstream end while it is
    DN_CRITICAL (crest above the downstream head); weirs give none."""
    from noodl.model import StepContext
    plain, s0, d0 = build_model(_regulated("weir"))
    model, state, drivers = build_model(_regulated("orifice"))
    hyd = model.dynamic_wave
    for depths, critical in (([0.1, 0.2, 0.05], True), ([0.1, 0.2, 0.65], False)):
        fed = _fed_heads(model, state, depths)
        out = hyd(fed, drivers, StepContext(dt=5.0))
        flow, _ = hyd._structure_terms(fed["water.phi"])[0]["orifice"]
        half = flow.surf_area.item() / 2.0 * QCF / LCF
        ref = plain.dynamic_wave(_fed_heads(plain, s0, depths), d0,
                                 StepContext(dt=5.0))["water.storage_area"]
        got = out["water.storage_area"]
        assert half > 1.0                                    # W(y) x 200 ft / 2, in m2
        assert got[1].item() == pytest.approx(ref[1].item() + half, rel=1e-12)
        if critical:
            assert got[2].item() == pytest.approx(ref[2].item(), rel=1e-12)
        else:
            assert got[2].item() == pytest.approx(ref[2].item() + half, rel=1e-12)


def test_regulator_end_areas_are_zeroed_at_a_storage_unit():
    """dynwave.c:520-523: at a storage node SWMM zeroes a regulator's end area
    (never a conduit's): J2 a storage unit, its orifice end gives it nothing."""
    from noodl.model import StepContext
    model, state, drivers = build_model(_regulated("orifice", storage_at="J2"))
    hyd = model.dynamic_wave
    fed = _fed_heads(model, state, [0.1, 0.2, 0.65])
    area = hyd(fed, drivers, StepContext(dt=5.0))["water.storage_area"][1].item()
    links_ft2 = hyd._terms(fed, 5.0).surface_area2[0].item()          # C1's end at J2
    assert area == pytest.approx((links_ft2 + 3.0 / LCF**2) * QCF / LCF, rel=1e-12)


def test_regulators_set_crowns_and_full_depths_as_swmm_does():
    """dynwave.c:137-153: a regulator's ends raise both nodes' crowns (node 1 offset1 +
    yFull, node 2 the crest height as given + yFull); link.c:440-463: it extends only its
    first node's full depth, and a bottom orifice none; link.c:425-433: a crest below the
    downstream invert is raised to it (offset1 only)."""
    hyd = build_model(_regulated("weir", crest=0.3))[0].dynamic_wave
    assert hyd.crown_depth.tolist() == pytest.approx([0.6, 1.3, 1.3])
    hyd = build_model(_regulated("weir", crest=0.3, j3_invert=11.5))[0].dynamic_wave
    s = hyd.structures["weir"]
    assert (s.offset1 * LCF).item() == pytest.approx(0.5)          # raised to J3's invert
    assert (s.offset2 * LCF).item() == pytest.approx(0.3)
    assert hyd.crown_depth.tolist() == pytest.approx([0.6, 1.5, 1.3])
    net = _regulated("orifice", crest=0.5)
    net = dataclasses.replace(net, manholes=tuple(
        dataclasses.replace(m, ground=m.invert + 0.2) for m in net.manholes))
    hyd = build_model(net)[0].dynamic_wave
    assert hyd.full_depth.tolist() == pytest.approx([0.6, 0.7, 0.6])
    hyd = build_model(dataclasses.replace(net, orifices=(dataclasses.replace(
        net.orifices[0], orientation="bottom"),)))[0].dynamic_wave
    assert hyd.full_depth.tolist() == pytest.approx([0.6, 0.6, 0.6])


def test_an_outfall_on_a_regulator_takes_zero_normal_and_critical_depth():
    """link.c:753-765: only a conduit computes yNorm and yCrit for its outfall; a regulator's
    outfall gets 0 for both, then node.c's FREE/FIXED rules with z its offset (the crest
    height at node 2)."""
    q = torch.tensor([0.03, 0.0, 0.0], dtype=F64)
    hyd = build_model(_regulated("outlet_to_outfall", crest=0.2))[0].dynamic_wave
    assert hyd.outfall_on_structure.tolist() == [True]
    assert hyd._outfall_heads(q[:1]).item() == 10.0
    for stage, head in ((10.5, 10.5), (10.1, 10.1)):     # node.c:1473, 1481-1483
        net = _regulated("outlet_to_outfall", crest=0.2, outfall_kind="fixed", stage=stage)
        assert build_model(net)[0].dynamic_wave._outfall_heads(q[:1]).item() == (
            pytest.approx(head))


@pytest.mark.parametrize("kind", ["orifice", "weir", "outlet"])
def test_a_run_ends_on_swmms_regulator_flow_at_its_own_heads(kind):
    """The fixed point is SWMM's: after each step the regulator's solved flow is
    `link_getInflow` at that step's own converged heads, and every junction holds SWMM's
    trapezoidal continuity with it (to the head solve's default atol, `solve_atol`: a
    junction balance within ``0.5 dt atol`` ~ 6e-9 m3 at these 12 m inverts)."""
    from noodl.apps.sewer import structures as st
    model, state, drivers = build_model(_regulated(kind), iterate_tol=1e-13)
    hyd = model.dynamic_wave
    s = hyd.structures[kind]
    fn = {"orifice": st.orifice_flow, "weir": st.weir_flow, "outlet": st.outlet_flow}[kind]
    seen = 0.0
    for _ in range(60):
        before = dict(state)
        state, _ = simulate_dynamic_wave(model, state, drivers, steps=1)
        h = state["water.phi"]
        hf = (hyd.node_invert + torch.clamp(h - hyd.node_invert, min=0.0)) / LCF
        z = hyd.node_invert / LCF
        ref = fn(s.params, hf[s.u], hf[s.v], z[s.u], z[s.v], z[s.u] + s.offset1).flow * QCF
        assert state["water.q"][2].item() == pytest.approx(ref.item(), rel=1e-7, abs=1e-10)
        res = hyd.junction_residual(before, state, drivers)
        assert res.abs().max().item() < 0.5 * 5.0 * solve_atol(model)
        seen = max(seen, ref.item())
    assert seen > 0.01


def test_regulator_columns_start_dry():
    """link.c:518 (link_initState): a regulator starts with no flow; its columns follow
    the pipes' in ``water.q``."""
    model, state, _ = build_model(_regulated("weir"))
    assert state["water.q"].tolist() == [0.0, 0.0, 0.0]
    assert model.dynamic_wave.edge_u.tolist()[-1] == list(model.net.nodes).index("J2")


def test_a_junction_drained_by_an_outlet_alone_holds_continuity():
    """An EXTRAN junction whose only outflow is an outlet (its crown the outlet's crest
    plus SWMM's TINY DUMMY height, above that of its inflow conduit's end only if the
    crest is high) surcharges once above that crown. SWMM's surcharged update divides by
    ``sumdqdh`` (dynwave.c:716-741), which an outlet leaves 0, so above 1.25 crowns its
    denominator is 0, every depth is a fixed point and SWMM freezes the depth, losing the
    inflow (seen live: a 0.015 m3/s inflow, an outlet passing 0.0023 m3/s for 30 min).
    noodl holds EXTRAN's instantaneous continuity: the outlet passes the inflow. A
    junction drained only by outlets is therefore listed in the docs' Limitations. The
    outlet carries exactly what its inflow conduit brings (no storage term)."""
    net = _regulated("outlet_to_outfall", crest=0.3, inflow=0.03)
    net = dataclasses.replace(net, pipes=(dataclasses.replace(net.pipes[0], diameter=0.25),))
    model, state, drivers = build_model(net)
    assert model.dynamic_wave.crown_depth[1].item() == pytest.approx(0.3, abs=1e-6)
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=120)
    assert model.dynamic_wave(state, drivers)["sewer.surcharged"][1].item() == 1.0
    q_in, q_out = state["water.q"].tolist()
    assert q_out == pytest.approx(q_in, abs=1e-9) and q_out > 0.029


def test_regulators_are_refused_by_name_outside_their_scope():
    """Kinematic/steady routing refuses regulators; malformed ones are refused by name."""
    net = _regulated("orifice")
    with pytest.raises(ValueError, match="orifice 'R1'.*dynamic-wave routing only"):
        dataclasses.replace(net, routing="kinematic_wave", geometry="analytic").validate()
    bad = [
        (dict(orifices=(dataclasses.replace(net.orifices[0], name="C1"),)),
         "duplicate link name"),
        (dict(orifices=(dataclasses.replace(net.orifices[0], v="J9"),)), "names node 'J9'"),
        (dict(orifices=(dataclasses.replace(net.orifices[0], shape="oval"),)), "shape"),
        (dict(orifices=(dataclasses.replace(net.orifices[0], width=None),)), "width > 0"),
        (dict(orifices=(dataclasses.replace(net.orifices[0], crest_height=-0.1),)),
         "crest_height"),
        (dict(orifices=(dataclasses.replace(net.orifices[0], v="J2"),)), "to itself"),
    ]
    for change, match in bad:
        with pytest.raises(ValueError, match=match):
            dataclasses.replace(net, **change).validate()
    weir = _regulated("weir")
    with pytest.raises(ValueError, match="roadway"):
        dataclasses.replace(weir, weirs=(dataclasses.replace(weir.weirs[0], kind="roadway"),)
                            ).validate()
    outlet = _regulated("outlet")
    with pytest.raises(ValueError, match="strictly increasing"):
        dataclasses.replace(outlet, outlets=(dataclasses.replace(
            outlet.outlets[0], rating=("tabular", (0.0, 0.0), (0.0, 1.0))),)).validate()
    two = _regulated("outlet_to_outfall")
    with pytest.raises(ValueError, match="attached to 2 links"):
        dataclasses.replace(two, pipes=(*two.pipes, dataclasses.replace(
            two.pipes[0], name="C9", u="J2", v="Out"))).validate()


def test_each_outfall_names_its_own_links_flow_column():
    """`model.outfall_link_idx` is each outfall's link's ``"water.q"`` column: a conduit's,
    or a regulator's after the pipes' (structures fixture: Out1 on C5, Out2 on the outlet
    OL1), so the outfall flows sum to the system's outflow."""
    net, _, _ = read_swmm_inp(DATA / "structures_dynwave.inp")
    model, state, drivers = build_model(net)
    names = [model.link_names[i] for i in model.outfall_link_idx.tolist()]
    assert names == ["C5", "OL1"]
    state, hist = simulate_dynamic_wave(model, state, drivers, steps=6)
    q = hist["water.q"][-1]
    assert q[model.outfall_link_idx[1]].item() == q[model.link_names.index("OL1")].item() > 0


# ------------------------------------------------------------------ tidal and time-varying outfalls
TIDE = ((0.0, 8.10), (6.0, 8.40), (18.0, 8.00))


@pytest.mark.parametrize("seconds,stage", [
    (0.0, 8.10),                     # the first entry
    (3 * 3600.0, 8.25),              # linear between entries (table.c:51-63)
    (6 * 3600.0, 8.40),              # on an entry: x <= x2 interpolates to y2 (table.c:421)
    (12 * 3600.0, 8.20),
    (20 * 3600.0, 8.00),             # beyond the last entry: its value, no wrap (table.c:424)
    (24 * 3600.0, 8.10),             # the tide repeats every 24 h of elapsed time (node.c:1449)
    (27 * 3600.0, 8.25),
    (48 * 3600.0 + 1.5, 8.10 + 1.5 / 3600.0 * 0.05),
])
def test_tidal_stage_is_swmms_table_lookup_of_the_hour_of_day(seconds, stage):
    """`node.c:1445-1452`: x = first x + 24 frac(elapsed days), then `table_lookup`:
    clamped below the first entry, linear between, the last y beyond the last x."""
    from noodl.apps.sewer.dynwave import tidal_stage
    assert tidal_stage(TIDE, seconds) == pytest.approx(stage, abs=1e-12)


def test_tidal_stage_starts_at_the_curves_first_hour_not_at_zero():
    """The abscissa is the first entry's x plus the elapsed hours (not the clock time, and
    the simulation's start time is not added): a curve starting at hour 2 begins there."""
    from noodl.apps.sewer.dynwave import tidal_stage
    tide = ((2.0, 5.0), (4.0, 7.0))
    assert tidal_stage(tide, 0.0) == 5.0
    assert tidal_stage(tide, 3600.0) == pytest.approx(6.0)
    assert tidal_stage(tide, 5 * 3600.0) == 7.0       # x = 7 h is past the last entry


def _outfall_net(**outfall):
    net, _, _ = read_swmm_inp(DATA / "tree_dynwave.inp")
    old = net.outfalls[0]
    return dataclasses.replace(net, outfalls=(dataclasses.replace(old, **outfall),))


def test_a_stage_driven_outfall_is_a_fixed_outfall_at_that_stage():
    """A tidal outfall at an instant is SWMM's FIXED outfall at the tide's stage: the same
    head solve, step for step, as a FIXED outfall of that stage."""
    fixed_model, fixed_state, fixed_drivers = build_model(_outfall_net(kind="fixed", stage=8.3))
    tidal = _outfall_net(kind="tidal", tide=((0.0, 8.3), (24.0, 8.3)))
    model, state, drivers = build_model(tidal)
    _, a = simulate_dynamic_wave(fixed_model, fixed_state, fixed_drivers, steps=12)
    _, b = simulate_dynamic_wave(model, state, drivers, steps=12)
    torch.testing.assert_close(b["water.phi"], a["water.phi"], rtol=0, atol=1e-12)
    torch.testing.assert_close(b["water.q"], a["water.q"], rtol=0, atol=1e-12)


def test_a_timeseries_outfall_follows_its_stage_function_at_each_step_end():
    """The stage function is called once per step with the END time of the step (SWMM's
    NewRoutingTime when the outfall depth is set, routing.c:301-302 before routeFlow), and
    the outfall head of each step is the stage the function gave."""
    net = _outfall_net(kind="timeseries", series="Tide")
    times = []

    def stage(t):
        times.append(t)
        return 8.6 + 0.001 * t        # far above the outfall conduit's own depth

    model, state, drivers = build_model(net, outfall_stage={"Outfall": stage})
    assert times == [0.0]                                     # the initial heads, at t = 0
    torch.testing.assert_close(state["water.phi"][model.dynamic_wave.outfall_idx],
                               torch.tensor([8.6], dtype=F64))
    times.clear()
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=3)
    assert times == [5.0, 10.0, 15.0]
    heads = hist["water.phi"][:, model.dynamic_wave.outfall_idx].flatten()
    torch.testing.assert_close(heads, torch.tensor([8.605, 8.61, 8.615], dtype=F64),
                               rtol=0, atol=1e-12)


def test_the_outfall_stage_is_taken_at_the_end_of_a_variable_step():
    net = _outfall_net(kind="timeseries", series="Tide")
    model, state, drivers = build_model(net, outfall_stage={"Outfall": lambda t: 8.5 + t / 1e4})
    # SWMM's variable step: the minimum step first, then the junction depth-change limit
    hyd = model.dynamic_wave
    hyd.options = dataclasses.replace(hyd.options, variable_step=0.75)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=4)
    heads = hist["water.phi"][:, hyd.outfall_idx].flatten()
    torch.testing.assert_close(heads, 8.5 + hist["t"] / 1e4, rtol=0, atol=1e-12)
    assert hist["dt"].unique().numel() > 1                    # the steps really varied


def test_the_outfall_stage_driver_is_required_and_validated_by_name():
    net = _outfall_net(kind="timeseries", series="Tide")
    with pytest.raises(ValueError, match="timeseries outfall.*Outfall.*outfall_stage"):
        build_model(net)
    with pytest.raises(ValueError, match="Nowhere"):
        build_model(net, outfall_stage={"Outfall": 8.5, "Nowhere": 8.0})
    tidal = _outfall_net(kind="tidal", tide=TIDE)
    with pytest.raises(ValueError, match="Outfall.*tidal.*curve"):
        build_model(tidal, outfall_stage={"Outfall": 8.5})
    free = read_swmm_inp(DATA / "tree_dynwave.inp")[0]
    with pytest.raises(ValueError, match="Outfall.*free"):
        build_model(free, outfall_stage={"Outfall": 8.5})
    model, state, drivers = build_model(net, outfall_stage={"Outfall": 8.5})
    with pytest.raises(KeyError, match="outfall_stage.*Outfall"):      # a step needs the driver
        model.step(state, {k: v for k, v in drivers.items() if k != "outfall_stage"}, 5.0)
    with pytest.raises(ValueError, match="Outfall.*finite"):
        build_model(net, outfall_stage={"Outfall": lambda t: float("nan")})
    # a constant is a stage function that never changes
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=2)
    torch.testing.assert_close(hist["water.phi"][:, model.dynamic_wave.outfall_idx],
                               torch.full((2, 1), 8.5, dtype=F64))


def test_simulate_takes_its_own_stage_function_over_the_models():
    net = _outfall_net(kind="timeseries", series="Tide")
    model, state, drivers = build_model(net, outfall_stage={"Outfall": 8.5})
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=2,
                                    outfall_stage={"Outfall": 8.7})
    torch.testing.assert_close(hist["water.phi"][:, model.dynamic_wave.outfall_idx],
                               torch.full((2, 1), 8.7, dtype=F64))


# ---------------------------------------------------------------- quality
def _tracer_run(net, steps, loads, *, species=("tracer",), loaded=None, **build):
    """`net` with a quality layer of `species`, the lateral inflow at the junctions of
    `loads` ({junction: kg/m3}) carrying the species `loaded` (default the first) and no
    other, run `steps` steps one at a time: the model, the drivers and the states (the
    initial one first)."""
    model, state, drivers = build_model(net, quality=True, species=species, **build)
    load = torch.zeros(model.net.n, dtype=F64)
    for name, value in loads.items():
        load[model.net.nodes.index(name)] = value
    drivers = dict(drivers, **{f"{s}_in": load for s in (loaded or species[:1])})
    states = [state]
    for _ in range(steps):
        state, _ = simulate_dynamic_wave(model, state, drivers, steps=1)
        states.append(state)
    return model, drivers, states


def _amount(state):
    """The stored amount of the first species per junction (kg): capacity times
    concentration."""
    x = state["water_quality.x"]
    return state["water_quality.capacity"] * (x if x.dim() == 1 else x[..., 0])


def _outfall_flux(model, state):
    """The first species' flux into the outfalls at `state` (kg/s): each outfall link's
    flow away from the network times its upwind junction's concentration (an outfall's own
    concentration, the boundary value, is 0)."""
    hyd = model.dynamic_wave
    x = state["water_quality.x"]
    x = x if x.dim() == 1 else x[..., 0]
    full = torch.zeros(model.net.n, dtype=F64)
    full[model.junction_idx] = x
    flux = 0.0
    for col, at_v in zip(model.outfall_link_idx.tolist(), hyd.outfall_at_v.tolist(),
                         strict=True):
        q = float(state["water.q"][col])
        out = q if at_v else -q                   # positive into the outfall
        inner = int(hyd.edge_u[col] if at_v else hyd.edge_v[col])
        flux += max(out, 0.0) * float(full[inner])
    return flux


def test_build_model_builds_quality_on_the_solved_flows(tmp_path):
    """``water_quality`` advects on every link of the head layer (its flows are
    ``"water.q"``, `Model.flow_layer_of`), its interior is the junctions in junction order,
    and the iteration judges it alongside the heads."""
    model, state, drivers = build_model(_tree(tmp_path), quality=True)
    layer = model.transport["water_quality"]
    assert model.flow_layer_of["water_quality"] == "water"
    assert layer.flow_kinds == model.dynamic_wave.edge_kinds
    assert layer.interior_idx.tolist() == model.junction_idx.tolist()
    assert layer.species_names == ("bod", "sulfide")
    assert set(model.iterate_tol) == {"water", "water_quality"}
    assert model.reaction_order == "before_transport"
    assert state["water_quality.x"].shape == (5, 2)
    assert "water_quality.capacity" in state


def test_quality_volume_starts_at_half_the_attached_conduits_and_follows_continuity(
        tmp_path):
    """A junction's transport control volume starts at half the summed
    wetted volume (`sewer.V_wet`, a total over barrels) of its attached conduits, floored at
    `CAPACITY_FLOOR` (half of the outfall conduit belongs to no junction), and then follows
    the junction's own continuity on the solved flows, ``V_end = V_start + dt N`` (no
    flooding here); the returned capacity is that volume."""
    from noodl.apps.sewer.hydraulics import CAPACITY_FLOOR

    net = _tree(tmp_path, scale=0.5)
    model, drivers, states = _tracer_run(net, 12, {"J1": 0.1})
    hyd = model.dynamic_wave
    first = states[0]
    half = 0.5 * hyd(first, drivers)["sewer.V_wet"]
    nodes = torch.zeros(model.net.n, dtype=F64)
    nodes = nodes.index_add(0, hyd.u, half).index_add(0, hyd.v, half)
    torch.testing.assert_close(first["sewer.quality_volume"],
                               torch.clamp(nodes[model.junction_idx], min=CAPACITY_FLOOR),
                               rtol=1e-15, atol=0.0)
    dt = model.routing_step
    for before, after in zip(states[:-1], states[1:], strict=True):
        net_inflow = hyd(after, drivers)["sewer.net_inflow"]
        torch.testing.assert_close(after["sewer.quality_volume"],
                                   before["sewer.quality_volume"] + dt * net_inflow,
                                   rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(after["water_quality.capacity"],
                                   after["sewer.quality_volume"], rtol=0.0, atol=0.0)


def test_a_storage_units_volume_is_in_its_initial_quality_volume():
    """A storage unit's own stored volume (InitDepth 0.4 m of a 50 m2 prism) joins its
    half-conduit volume in the initial transport control volume."""
    net = _storage_chain(initial_depth=0.4)
    model, state, drivers = build_model(net, quality=True, species=("tracer",))
    out = model.dynamic_wave(state, drivers)
    assert float(out["sewer.storage_volume"][0]) == pytest.approx(50.0 * 0.4, rel=1e-12)
    expected = 0.5 * out["sewer.V_wet"] + out["sewer.storage_volume"]
    torch.testing.assert_close(state["sewer.quality_volume"], expected, rtol=1e-15,
                               atol=0.0)
    torch.testing.assert_close(state["water_quality.capacity"], expected, rtol=1e-15,
                               atol=0.0)


def _uniform_run(net, steps, c=0.1):
    """`net` started at the uniform concentration `c` (kg/m3) of a conservative tracer,
    every lateral inflow at `c` too, run `steps` steps: the model, drivers and states."""
    model, state, drivers = build_model(net, quality=True, species=("tracer",))
    state = dict(state, **{"water_quality.x": torch.full_like(state["water_quality.x"], c)})
    drivers = dict(drivers, tracer_in=torch.full((model.net.n,), c, dtype=F64))
    states = [state]
    for _ in range(steps):
        state, _ = simulate_dynamic_wave(model, state, drivers, steps=1)
        states.append(state)
    return model, drivers, states


def test_a_uniform_concentration_stays_uniform_while_the_tree_fills(tmp_path):
    """Water started and entering at one concentration stays at it, at every junction and
    every step, while the tree fills from dry: the transport control volume holds the
    water the junction's continuity says arrived, so the amount form mixes like with like
    (to the flow tolerance of the iteration). Splitting each conduit's volume in half
    instead (the geometric capacity) put the tree between 0 and 5 times the load while it
    filled: a conduit filling from its upstream end holds that junction's water."""
    model, _, states = _uniform_run(_tree(tmp_path, scale=0.5), 24)
    for state in states[1:]:
        torch.testing.assert_close(state["water_quality.x"],
                                   torch.full_like(state["water_quality.x"], 0.1),
                                   rtol=1e-8, atol=0.0)


def test_a_flooded_junction_loses_its_contents_with_the_overflow():
    """The flooding tree (J2 held at y_max from step 21, its overflow lost, no ponding): a
    uniform concentration stays uniform through the flood, and the tracer amount changes by
    the load less the outfall flux less what the overflow carries away (`OverflowLoss`: the
    step-start contents times ``dt overflow / V_start``)."""
    model, drivers, states = _uniform_run(_flooding_tree(), 30)
    hyd = model.dynamic_wave
    j2 = hyd.junction_names.index("J2")
    dt = model.routing_step
    load = 0.1 * float(drivers["inflow"].sum())
    flooded = 0
    for before, after in zip(states[:-1], states[1:], strict=True):
        torch.testing.assert_close(after["water_quality.x"],
                                   torch.full_like(after["water_quality.x"], 0.1),
                                   rtol=1e-8, atol=0.0)
        overflow = hyd.overflow(before, after, drivers)
        flooded += int(overflow[j2] > 0.0)
        lost = float((dt * overflow * before["water_quality.x"]).sum())
        stored = float(_amount(after).sum() - _amount(before).sum())
        assert stored == pytest.approx(dt * (load - _outfall_flux(model, after)) - lost,
                                       rel=1e-9, abs=1e-15)
    assert flooded == 10


def test_tracer_follows_reversed_flow():
    """The loop's adverse conduit Cd is oriented J4 -> J3 (the reader reverses it, as SWMM
    does) but carries J3's water to J4, so its solved flow is negative. A tracer entering
    at J3 must reach J4 through Cd, upwind by the SIGN of the flow, and never J1 or J2,
    which lie upstream of it; J4's amount balance over the last step is the implicit
    upwind balance on the solved flows (`Cd`'s |q| from J3, `Cc` from J2, `Ce` out)."""
    net, _, _ = read_swmm_inp(DATA / "loop_dynwave.inp")
    model, drivers, states = _tracer_run(net, 60, {"J3": 0.1})
    names = list(model.link_names)
    before, after = states[-2], states[-1]
    q = {name: float(after["water.q"][i]) for i, name in enumerate(names)}
    assert q["Cd"] < -1e-4                       # reversed against its orientation
    junctions = list(model.dynamic_wave.junction_names)
    x = {n: float(after["water_quality.x"][j]) for j, n in enumerate(junctions)}
    assert x["J1"] == 0.0 and x["J2"] == 0.0
    assert x["J4"] > 1e-4
    j4 = junctions.index("J4")
    dt = model.routing_step
    stored = float(_amount(after)[j4] - _amount(before)[j4])
    moved = dt * (-q["Cd"] * x["J3"] + q["Cc"] * x["J2"] - q["Ce"] * x["J4"])
    assert stored == pytest.approx(moved, rel=1e-12)
    # and the whole loop's tracer, every step: the load at J3 less the outfall flux
    load = 0.1 * float(drivers["inflow"][model.net.nodes.index("J3")])
    for one, two in zip(states[:-1], states[1:], strict=True):
        change = float(_amount(two).sum() - _amount(one).sum())
        assert change == pytest.approx(dt * (load - _outfall_flux(model, two)),
                                       rel=1e-12, abs=1e-18)


def test_quality_conserves_the_tracer_while_the_tree_fills(tmp_path):
    """The amount form of the transport step (``V_new x_new - V_old x_old = dt F``):
    while the tree fills from dry, so that every capacity changes, the stored tracer
    changes by the lateral load minus the flux into the outfall, to round-off, every step."""
    net = _tree(tmp_path, scale=0.5)
    model, drivers, states = _tracer_run(net, 24, {"J1": 0.1})
    load = 0.1 * float(drivers["inflow"][model.net.nodes.index("J1")])
    dt = model.routing_step
    for before, after in zip(states[:-1], states[1:], strict=True):
        stored = float(_amount(after).sum() - _amount(before).sum())
        assert stored == pytest.approx(dt * (load - _outfall_flux(model, after)),
                                       rel=1e-12, abs=1e-18)
    assert _outfall_flux(model, states[-1]) > 0.0          # it reached the outfall


def test_tracer_crosses_a_regulator_without_storage():
    """An orifice (as a weir or an outlet: every edge kind of the head layer is a flow kind
    of the quality layer) carries flow but no volume (SWMM gives a non-conduit link its
    upstream node's quality, qualrout.c:282-288): the tracer crosses it upwind on its solved
    flow, J2 -> J3, and the tracer amount is conserved."""
    model, drivers, states = _tracer_run(_regulated("orifice"), 60, {"J1": 0.1})
    assert model.transport["water_quality"].flow_kinds == ("pipe", "orifice")
    x = states[-1]["water_quality.x"]
    assert float(x[2]) > 0.0                                # J3, below the regulator
    load = 0.1 * float(drivers["inflow"][model.net.nodes.index("J1")])
    for before, after in zip(states[:-1], states[1:], strict=True):
        stored = float(_amount(after).sum() - _amount(before).sum())
        assert stored == pytest.approx(
            model.routing_step * (load - _outfall_flux(model, after)), rel=1e-12, abs=1e-18)


def test_the_overflow_loss_takes_at_most_the_whole_contents():
    """One step's overflow larger than a junction's step-start contents: the lost fraction
    is capped at 1 (all of them leave) and the control volume stays at the floor or
    above; below the cap the fraction is ``dt overflow / V_start``."""
    from noodl.apps.sewer.dynwave import OverflowLoss
    from noodl.apps.sewer.hydraulics import CAPACITY_FLOOR

    model, state, _ = build_model(_flooding_tree(), quality=True, species=("tracer",))
    hyd = model.dynamic_wave
    start = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0], dtype=F64)
    out = hyd._quality_volume({"sewer.quality_volume": start},
                              torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0], dtype=F64),
                              torch.tensor([1.0, 0.1, 0.0, 0.0, 0.0], dtype=F64), 5.0)
    torch.testing.assert_close(out["water_quality.overflow_loss"],
                               torch.tensor([1.0, 0.25, 0.0, 0.0, 0.0], dtype=F64))
    assert float(out["sewer.quality_volume"][0]) == CAPACITY_FLOOR
    x = torch.full((5,), 0.1, dtype=F64)
    kept = OverflowLoss(1).apply(x, 5.0, out)
    torch.testing.assert_close(kept, torch.tensor([0.0, 0.075, 0.1, 0.1, 0.1], dtype=F64))


def test_sulfide_needs_each_junctions_unique_outgoing_conduit():
    """Sulfide generation reads each junction's outgoing conduit (its hydraulic radius,
    speed and mean depth). On the loop J1 and J4 have two and J3 none (Cd is oriented
    J4 -> J3), and above a regulator J2 has none: refused naming them. BOD decay and a
    conservative tracer need no conduit and build."""
    net, _, _ = read_swmm_inp(DATA / "loop_dynwave.inp")
    with pytest.raises(ValueError, match=r"\['J1', 'J3', 'J4'\]"):
        build_model(net, quality=True)
    with pytest.raises(ValueError, match=r"\['J2'\]"):
        build_model(_regulated("weir"), quality=True)
    with pytest.raises(ValueError, match="'sulfide' needs 'bod'"):
        build_model(net, quality=True, species=("sulfide",))
    model, state, drivers = build_model(net, quality=True, species=("bod", "tracer"))
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=2)
    assert state["water_quality.x"].shape == (4, 2)


def test_sulfide_is_generated_on_the_dynamic_wave_tree(tmp_path):
    """BOD entering at J1 generates sulfide in the conduit below it (Pomeroy-Parkhurst on
    each junction's outgoing conduit: `sewer.R_h`, `sewer.v`, `sewer.d_m`), and BOD decays
    against a conservative tracer entering with it; J2, which nothing loaded reaches, stays
    clean."""
    net = _tree(tmp_path, scale=0.5)
    model, drivers, states = _tracer_run(net, 24, {"J1": 0.3},
                                         species=("bod", "sulfide", "tracer"),
                                         loaded=("bod", "tracer"))
    out = model.dynamic_wave(states[-1], drivers)
    for key in ("sewer.R_h", "sewer.d_m"):
        assert out[key].shape == (5,) and bool((out[key] > 0).all())
    junctions = list(model.dynamic_wave.junction_names)
    x = states[-1]["water_quality.x"]
    assert float(x[junctions.index("J1"), 1]) > 0.0
    assert float(x[junctions.index("J2"), 0]) == 0.0
    j1 = junctions.index("J1")
    assert 0.0 < float(x[j1, 0]) < float(x[j1, 2])


def test_quality_runs_batched(tmp_path):
    """A batched inflow expands the multi-species state ``(junctions, species)`` too, and
    each member is its own run."""
    net = _tree(tmp_path, scale=0.5)
    model, state, drivers = build_model(net, quality=True)
    bod_in = torch.zeros(model.net.n, dtype=F64)
    bod_in[model.net.nodes.index("J1")] = 0.3
    drivers = dict(drivers, bod_in=bod_in)
    inflow = torch.stack([drivers["inflow"], 0.5 * drivers["inflow"]])
    final, _ = simulate_dynamic_wave(model, state, drivers, steps=6, inflow=inflow)
    single, _ = simulate_dynamic_wave(model, state, drivers, steps=6,
                                      inflow=0.5 * drivers["inflow"])
    assert final["water_quality.x"].shape == (2, 5, 2)
    torch.testing.assert_close(final["water_quality.x"][1], single["water_quality.x"],
                               rtol=1e-9, atol=1e-15)


# ---------------------------------------------------------------- headspace air
def _air_run(net, steps, **build):
    """`net` with the headspace air, run `steps` steps: the model, the final state and the
    drivers."""
    model, state, drivers = build_model(net, air=True, quality=False, **build)
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=steps)
    return model, state, drivers


def test_air_is_built_on_the_dynamic_wave_tree_and_balances(tmp_path):
    """The headspace air of a dynamic-wave tree: the ``air`` layer beside ``water``, its
    initial state (quiescent) in `initial_state`, and its flows balance at every manhole to
    the Newton tolerance (the leak edges carry the rest to ambient)."""
    net = _tree(tmp_path)
    model, state, drivers = build_model(net, air=True, quality=False)
    assert set(model.kirchhoff) == {"water", "air"} and not model.transport
    assert state["air.phi"].shape == (model.net.n,) and float(state["air.q"].abs().max()) == 0.0
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=30)
    assert float(state["air.q"].abs().max()) > 1e-6          # the water drags the air
    resolved = model._apply_closures(state, drivers)
    layer = model.kirchhoff["air"]
    # the air at the final state's drivers (the stored pass lags them by the head tolerance,
    # which the headspace law's square root amplifies where a flow is near zero)
    phi, q = layer.solve(drivers["air.phi_boundary"], resolved, differentiable=False,
                          atol=1e-13)
    residual = layer.residual(phi[..., layer.interior], drivers["air.phi_boundary"], resolved,
                              None)
    assert float(residual.abs().max()) < 1e-11
    power = layer.power_residual(phi, q, resolved)
    # Tellegen: the power is the residuals weighted by the pressures, which sit on the datum of
    # the ground elevations (~1e2 Pa), so its bound is the residual bound times that scale
    assert float(power.abs().max()) < 1e-11 * float(phi.abs().max())
    assert torch.isfinite(resolved["air_quality.capacity"]).all()


def test_air_flows_equal_the_kinematic_model_at_the_same_depths(tmp_path):
    """Same air-side physics: the dynamic model's air layer and the kinematic model's (the
    same `air_layer`, built from the kinematic reading of the same pipes), solved on the SAME
    headspace drivers (``sewer.A_air``, ``D_h``, ``T``, ``v`` at the dynamic wave's link
    depths, and the air densities), give the same flows to 1e-9 m3/s. The kinematic
    closure is bypassed because the two routings' depths differ (the dynamic tree is still
    filling); comparing the stored ``air.q`` instead would compare a pass lagged by the head
    tolerance through the headspace law's square root, which amplifies a driver change
    wherever the flow is near zero."""
    net = _tree(tmp_path)
    model, state, drivers = _air_run(net, 40)
    resolved = model._apply_closures(state, drivers)
    kin_net, _, _ = read_swmm_inp(DATA / "tree_kinwave.inp")
    kin, _, kin_drivers = build_model(kin_net, air=True, quality=False)
    assert [p.name for p in net.pipes] == [p.name for p in kin_net.pipes]
    boundary = drivers["air.phi_boundary"]
    _, q_dyn = model.kirchhoff["air"].solve(boundary, resolved, differentiable=False)
    _, q_kin = kin.kirchhoff["air"].solve(boundary, resolved, differentiable=False)
    torch.testing.assert_close(q_dyn, q_kin, rtol=0.0, atol=1e-9)
    assert float(q_kin.abs().max()) > 1e-6
    # the stored flows are those at the pass's own drivers: equal to the head tolerance
    torch.testing.assert_close(state["air.q"], q_dyn, rtol=0.0, atol=1e-6)


def test_air_geometry_is_the_circle_at_the_link_depth_per_barrel():
    """``A_air``, ``D_h``, ``T``, ``V_air`` at the link depth are `geometry.air_geometry`
    (the kinematic path's own call), totals over the barrels for the areas, volume and top
    width and one barrel's for the diameter: two barrels have twice the area and volume and
    a quarter of the resistance of one at the same depth."""
    from noodl.apps.sewer import geometry as geom
    from noodl.apps.sewer.air import Headspace
    from noodl.model import StepContext
    published = {}
    diameter = torch.tensor([0.45], dtype=F64)
    for barrels in (1, 2):
        model, state, drivers = build_model(_chain_net(barrels=barrels), air=True,
                                            quality=False)
        state["water.phi"] = torch.tensor([11.0 + 0.2, 10.0, 0.0], dtype=F64)   # J1, Out, ambient
        state["sewer.junction_depth"] = torch.tensor([0.2], dtype=F64)
        d = model.dynamic_wave(state, drivers, StepContext(dt=5.0))
        published[barrels] = d
        a_air, _, d_h = geom.air_geometry(d["sewer.h"], diameter)
        torch.testing.assert_close(d["sewer.A_air"], barrels * a_air, rtol=1e-14, atol=0)
        torch.testing.assert_close(d["sewer.D_h"], d_h, rtol=1e-14, atol=0)
        torch.testing.assert_close(d["sewer.V_air"], barrels * a_air * 200.0, rtol=1e-14,
                                   atol=0)
        torch.testing.assert_close(d["sewer.T"], barrels * geom.top_width(d["sewer.h"], diameter),
                                   rtol=1e-14, atol=0)
    head = Headspace(torch.tensor([200.0], dtype=F64), torch.tensor([0]))
    r1 = head.resistance(published[1])
    r2 = head.resistance(published[2])
    torch.testing.assert_close(r2, r1 / 4.0, rtol=1e-12, atol=0)


def test_air_capacity_is_the_outgoing_headspace_plus_the_shaft(tmp_path):
    """``air_quality.capacity`` of a junction is its outgoing conduit's headspace volume plus
    the air of its shaft, ``A_s (full depth - depth)`` (A_s the 1.167 m2 default)."""
    net = _tree(tmp_path)
    model, state, drivers = build_model(net, air=True, quality=False)
    hyd = model.dynamic_wave
    state, _ = simulate_dynamic_wave(model, state, drivers, steps=10)
    resolved = model._apply_closures(state, drivers)
    depth = state["water.phi"][model.junction_idx] - model.junction_invert
    shaft = 1.167 * (hyd.full_depth - depth)
    expected = resolved["sewer.V_air"][hyd.out_pipe] + shaft
    torch.testing.assert_close(resolved["air_quality.capacity"], expected, rtol=1e-13, atol=0)
    assert float(shaft.min()) > 0.0


def test_a_full_conduit_keeps_the_air_finite():
    """A conduit under a high outfall stage flows full: its headspace is closed, the capped
    depth keeps ``A_air`` and the resistance finite, and the air layer still solves (the
    manhole breathes through its leak)."""
    net = _chain_net(outfall_kind="fixed", stage=10.0 + 2.5, inflow=0.02, initial_depth=1.0)
    model, state, drivers = build_model(net, air=True, quality=False)
    state, hist = simulate_dynamic_wave(model, state, drivers, steps=10,
                                        record=("sewer.h", "air.q"))
    resolved = model._apply_closures(state, drivers)
    assert float(resolved["sewer.h"][0]) == pytest.approx(0.45, abs=1e-9)   # full
    for key in ("sewer.A_air", "sewer.D_h", "sewer.T", "sewer.V_air", "air_quality.capacity"):
        assert torch.isfinite(resolved[key]).all() and float(resolved[key].min()) >= 0.0, key
    assert 0.0 < float(resolved["sewer.A_air"][0]) < 1e-2
    assert torch.isfinite(state["air.q"]).all() and torch.isfinite(state["air.phi"]).all()
    assert torch.isfinite(hist["air.q"]).all()


def test_a_surcharged_tree_with_air_stays_finite():
    """The EXTRAN tree surcharges J2 (3x the inflows): with the air on, every air flow and
    pressure along the run is finite."""
    model, state, drivers = build_model(_extran_tree(), air=True, quality=False)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=60,
                                    record=("air.q", "air.phi", "sewer.h"))
    assert torch.isfinite(hist["air.q"]).all() and torch.isfinite(hist["air.phi"]).all()
    full = model.dynamic_wave.diameter
    assert bool((hist["sewer.h"] >= full - 1e-6).any())                      # a full conduit


def test_pipe_table_has_every_column_on_the_dynamic_wave(tmp_path):
    """`pipe_table` returns all of the kinematic path's columns for a dynamic-wave model,
    with or without the air."""
    from noodl.apps.sewer.report import _COLUMNS, pipe_table
    for air in (False, True):
        model, state, drivers = build_model(_tree(tmp_path), air=air, quality=False)
        state, _ = simulate_dynamic_wave(model, state, drivers, steps=5)
        rows = pipe_table(model, state, drivers, path=tmp_path / "pipes.csv")
        assert [r["pipe"] for r in rows] == model.pipe_names
        assert set(rows[0]) == set(_COLUMNS)
        assert all(abs(v) < float("inf") for r in rows for k, v in r.items() if k != "pipe")


def test_air_refuses_loops_branches_and_regulator_drains_by_name():
    """The air side needs exactly one outgoing conduit per junction: the loop fixture (J1 and
    J4 have two, J3 none) and a junction drained only by a regulator are refused naming the
    junctions."""
    loop, _, _ = read_swmm_inp(DATA / "loop_dynwave.inp")
    with pytest.raises(ValueError, match=r"air=True.*\['J1', 'J3', 'J4'\]"):
        build_model(loop, air=True)
    with pytest.raises(ValueError, match=r"air=True.*\['J2'\]"):
        build_model(_regulated("weir"), air=True)


def test_air_refuses_a_cycle_that_never_reaches_an_outfall():
    """Every junction with one outgoing conduit is not enough: J1 -> J2 -> J1 is a loop with
    no outfall (dynamic-wave routing allows it, as SWMM does; the air builder refuses it)."""
    from noodl.apps.sewer import SewerNetwork
    from noodl.apps.sewer.network import Manhole, Outfall, Pipe
    net = SewerNetwork(
        manholes=(Manhole("J1", 12.0, ground=15.0, inflow=0.01),
                  Manhole("J2", 11.0, ground=15.0), Manhole("J3", 10.5, ground=15.0)),
        pipes=(Pipe("C1", "J1", "J2", 100.0, 0.3, 0.013, 0.01),
               Pipe("C2", "J2", "J1", 100.0, 0.3, 0.013, 0.01),
               Pipe("C3", "J3", "Out", 100.0, 0.3, 0.013, 0.005)),
        outfalls=(Outfall("Out", 10.0),), routing="dynamic_wave", geometry="tabulated")
    build_model(net)                                  # the water alone routes it
    with pytest.raises(ValueError, match=r"junction 'J1' reaches none"):
        build_model(net, air=True)


def test_air_with_quality_couples_the_h2s_transfer(tmp_path):
    """With sulfide the air_quality layer and the two-film transfer join: sulfide loaded at
    J1 appears as headspace H2S, the transfer's sources are opposite in moles of sulfur at
    every junction, and the iteration judges the air layer's concentrations too."""
    from noodl.apps.sewer.quality import M_H2S, M_S
    net = _tree(tmp_path, scale=0.5)
    model, state, drivers = build_model(net, air=True, quality=True)
    assert set(model.transport) == {"water_quality", "air_quality"}
    assert set(model.iterate_tol) == {"water", "water_quality", "air_quality"}
    load = torch.zeros(model.net.n, dtype=F64)
    load[model.net.nodes.index("J1")] = 5e-3
    drivers = dict(drivers, sulfide_in=load, bod_in=load * 20.0)
    for _ in range(40):
        state, _ = simulate_dynamic_wave(model, state, drivers, steps=1)
    assert float(state["air_quality.x"].max()) > 0.0
    for key in ("water_quality.x", "air_quality.x", "air.q"):
        assert torch.isfinite(state[key]).all(), key
    resolved = model._apply_closures(state, drivers)
    jidx = model.junction_idx
    lateral = drivers["inflow"] * drivers["sulfide_in"]
    water = (resolved["water_quality.sources"][..., 1] - lateral)[jidx]
    air = resolved["air_quality.sources"][jidx]
    torch.testing.assert_close(water / M_S + air / M_H2S, torch.zeros_like(water),
                               rtol=0.0, atol=1e-18)
    assert float(air.abs().max()) > 0.0


def test_build_dynamic_wave_model_is_exported():
    """The docs point at `build_dynamic_wave_model`'s `acceleration`, `memory` and
    `iterate_flow_tol`: it is part of the sewer app's public names."""
    import noodl.apps.sewer as sewer
    from noodl.apps.sewer.dynwave import build_dynamic_wave_model

    assert sewer.build_dynamic_wave_model is build_dynamic_wave_model
    assert "build_dynamic_wave_model" in sewer.__all__


# ------------------------------------------------------------------------------- ponding
def _pond_run(net, steps=30):
    model, state, drivers = build_model(net, iterate_tol=1e-12, iterate_max=500)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=steps, atol=1e-12,
                                    record=("water.phi", "sewer.overflow"))
    return model, hist


def test_ponding_keeps_the_water_a_flooding_junction_would_lose():
    """J2 (MaxDepth 0.3 m, its crown, Aponded 0.5 m2) of `ponding_dynwave.inp` rises barely
    above its rim, over the ponded area plus its conduits', and its overflow is the ponded
    volume's growth; SurDepth plays no part. With `allow_ponding` off the same network is the
    one that has the junction surcharge on (EXTRAN) up to rim + SurDepth = 2 m before it
    floods, so it is far deeper."""
    net, _, _ = read_swmm_inp(DATA / "ponding_dynwave.inp")
    assert net.dynamic_wave.allow_ponding
    model, hist = _pond_run(net)
    hyd = model.dynamic_wave
    j2 = list(hyd.junction_names).index("J2")
    node = model.junction_idx[j2]
    depth = hist["water.phi"][:, node] - hyd.node_invert[node]
    rim = float(hyd.full_depth[j2])
    assert rim == pytest.approx(0.3) and bool(hyd.can_pond[j2])
    assert rim + 0.05 < depth[-1] < rim + 0.2 and depth.max() == depth[-1]
    assert hist["sewer.overflow"][-1, j2] > 0.0
    off = dataclasses.replace(
        net, dynamic_wave=dataclasses.replace(net.dynamic_wave, allow_ponding=False))
    model_off, hist_off = _pond_run(off)
    assert not bool(model_off.dynamic_wave.can_pond.any())
    depth_off = hist_off["water.phi"][:, node] - hyd.node_invert[node]
    assert depth_off[-1] > 1.0 > depth[-1]


def test_ponding_is_refused_with_quality_and_for_bad_areas():
    net, _, _ = read_swmm_inp(DATA / "ponding_dynwave.inp")
    with pytest.raises(ValueError, match="not modelled together with water quality"):
        build_model(net, quality=True)
    manholes = tuple(
        dataclasses.replace(m, ponded_area=-1.0) if m.name == "J2" else m for m in net.manholes)
    with pytest.raises(ValueError, match="ponded_area must be >= 0"):
        dataclasses.replace(net, manholes=manholes).validate()
    stor, _, _ = read_swmm_inp(DATA / "storage_dynwave.inp")
    manholes = tuple(
        dataclasses.replace(m, ponded_area=1.0) if m.storage is not None else m
        for m in stor.manholes)
    with pytest.raises(ValueError, match="storage unit .* ponded_area"):
        dataclasses.replace(stor, manholes=manholes).validate()


def _one(value, n=2):
    return torch.full((n,), value, dtype=F64)


def _pond_drivers(n_start, n_last, head_start):
    return {"water.storage_area": _one(4.0), "water.pond_area": _one(4.5),
            "water.storage_rate": torch.tensor(1.0 / 5.0, dtype=F64),
            "water.head_start": _one(head_start), "water.net_inflow_start": _one(n_start),
            "water.net_inflow_last": _one(n_last), "water.head_last": _one(head_start),
            "water.pond_switch_slope": _one(40.0), "water.ponded": _one(0.0),
            "water.surcharged": _one(0.0), "water.surcharge_slope": _one(0.01),
            "water.flood_excess": _one(0.0)}


@pytest.mark.parametrize("n_start,n_last,head_start", [
    (0.02, 0.03, 0.46), (0.02, 0.03, 0.2), (0.0, 0.0, 0.1)])
def test_a_ponding_junction_rising_through_its_full_head_is_monotone_and_unwalled(
        n_start, n_last, head_start):
    """`JunctionStorage` with a `pond_head` H* = 0.5 m beside a plain junction with the same top:
    from a step start below H* the ponding junction's withdrawal never decreases (the lower line,
    the hold at H*, then the larger-area line) with a positive slope throughout, and it has no
    flood wall above H* while the plain junction's wall is far steeper than its storage."""
    heads = torch.linspace(0.0, 1.0, 2001, dtype=F64)
    store = JunctionStorage(torch.arange(2), torch.zeros(2, dtype=F64),
                            bed_area=12.566 * 0.0929, crown=_one(0.3), top=_one(0.5),
                            surchargeable=torch.tensor([False, False]),
                            pond_head=torch.tensor([0.5, torch.inf], dtype=F64))
    assert store.ponds
    drv = _pond_drivers(n_start, n_last, head_start)
    flow = torch.stack([store.flow(h.expand(2), drv) for h in heads])
    slope = torch.stack([store.dflow(h.expand(2), drv) for h in heads])
    assert bool((slope > 0.0).all())
    pond, plain = flow[:, 0], flow[:, 1]
    assert bool((pond[1:] >= pond[:-1] - 1e-12).all())
    top = int(torch.searchsorted(heads, torch.tensor(0.5, dtype=F64)))
    wall = float((plain[-1] - plain[top]) / (heads[-1] - heads[top]))
    storage = float((plain[top - 1] - plain[top - 100]) / (heads[top - 1] - heads[top - 100]))
    assert wall > 5.0 * storage
    # the ponded line above H* carries the larger area: slope 2 A_hi / dt = 2 * 4.5 / 5
    assert float(slope[-1, 0]) == pytest.approx(2.0 * 4.5 / 5.0)
