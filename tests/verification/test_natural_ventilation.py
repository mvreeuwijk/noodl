"""Coupled airflow-heat natural ventilation against analytical solutions.

Li and Delsante (2001), Building and Environment 36:59-71: single zone, two openings h apart
with equal areas A (effective area A* = A/sqrt(2)), heat source E, envelope conductance UA,
wind pressure difference dP_w (kinematic). With volumetric q, B = E g / (rho c_p T_o),
alpha = (Cd A*)^(2/3) (B h)^(1/3), beta = UA / (3 rho c_p), gamma = (Cd A*) sqrt(2 dP_w / 3):

    buoyancy only:            q = 2^(1/3) alpha                       (eq. 9)
    with envelope loss:       q^3 + 3 beta q^2 - 2 alpha^3 = 0        (eq. 15)
    assisting wind:           q^3 + 3 beta q^2 - 3 gamma^2 q - 2 alpha^3 - 9 gamma^2 beta = 0
    opposing, upward flow:    q^3 + 3 beta q^2 + 3 gamma^2 q - 2 alpha^3 + 9 gamma^2 beta = 0
    opposing, downward flow:  q^3 + 3 beta q^2 - 3 gamma^2 q + 2 alpha^3 - 9 gamma^2 beta = 0

The model works in mass flow with `mass_orifice` (F = rho_0 q) and the Boussinesq density
closure `LinearDensity`, which is exactly the paper's rho g h (T_i - T_o) / T_o form.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from noodl.apps.building_physics.elements import add_large_opening, orifice_elements_from_edges
from noodl.apps.building_physics.thermal import (
    CP_AIR,
    RHO_0,
    Zone,
    add_zone,
    build_model,
    initial_state,
)
from noodl.drives import Stack, Wind
from noodl.topology import Network

F64 = torch.float64
G = 9.80665
T_O = 283.15
CD = 0.6

# `build_model`'s second documented coupling trap: `iterate_tol` is ABSOLUTE in the layer's
# units (K), and it cannot be met below the potential solve's own residual floor propagated
# through dT/dF (~3.6e3 K.s/kg here). Newton's dtype-derived default (sqrt(eps) = 1.49e-8)
# leaves ~5e-5 K of coupling noise, so the 1e-9 K `iterate_tol` these verification cases ask
# for stalls at the default. Ruling R21 -- tighten the SOLVE, never loosen the assertion --
# so every `steady` below carries the Newton tolerances its coupling tolerance requires.
TIGHT = {"atol": 1e-14, "rtol": 1e-14}


def _positive_real_roots(coeffs) -> list[float]:
    return sorted(float(r.real) for r in np.roots(coeffs) if abs(r.imag) < 1e-9 and r.real > 0)


def _single_zone(*, A: float, h: float, E: float, UA: float = 0.0, cp_low: float = 0.0,
                 cp_high: float = 0.0, T_start: float = T_O, coupling: str = "iterate"):
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    add_zone(net, Zone("z", volume=100.0, T0=T_start))
    net.add_edge("ambient", "z", kind="airpath", z_path=0.0, Cd=CD, area=A, Cp=cp_low, Ch=1.0)
    net.add_edge("ambient", "z", kind="airpath", z_path=h, Cd=CD, area=A, Cp=cp_high, Ch=1.0)
    if UA > 0:
        net.add_edge("z", "ambient", kind="wall", ua=UA)
    el = orifice_elements_from_edges(net, "airpath")
    drives = [Stack.from_network(net, "airpath")]
    if cp_low or cp_high:
        drives.append(Wind.from_network(net, "airpath", ambient="ambient"))
    model = build_model(
        net, air_elements=[el], drives=drives, density="linear",
        density_kwargs={"rho_0": RHO_0, "T_0": T_O},
        coupling=coupling, iterate_tol={"thermal": 1e-9}, iterate_max=300,
    )
    sources = torch.zeros(net.n, dtype=F64)
    sources[net.node_index("z")] = E
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=F64),
        "thermal.x_boundary": torch.tensor([T_O], dtype=F64),
        "thermal.sources": sources,
        "V_met": torch.tensor(0.0, dtype=F64),
        "theta_w": torch.tensor(0.0, dtype=F64),
    }
    return net, model, initial_state(model), drivers


def _params(A, h, E, UA=0.0, dPw=0.0):
    A_star = A / math.sqrt(2.0)
    B = E * G / (RHO_0 * CP_AIR * T_O)
    alpha = (CD * A_star) ** (2.0 / 3.0) * (B * h) ** (1.0 / 3.0)
    beta = UA / (3.0 * RHO_0 * CP_AIR)
    gamma = CD * A_star * math.sqrt(2.0 * abs(dPw) / 3.0)
    return alpha, beta, gamma


def _volumetric_inflow(model, ss) -> float:
    q = model.potential["air"].flows_of_kind(ss["air.q"], "airpath")
    return q[0].item() / RHO_0


def test_buoyancy_only_matches_li_delsante_closed_form(record_property):
    # `T_start` is off ambient because this is the ONE case with neither wind nor an envelope
    # conductance: at T_z == T_o the stack head is identically zero, so the first pass solves
    # the airflow as exactly q = 0 and the steady heat balance 0 = M x + N x_b + E/C is then
    # singular (M = 0 with E != 0 -- a closed adiabatic zone with a heat source has no steady
    # state at all). That is physics, not a solver defect: `TransportLayer.steady` reports it
    # as a non-converged linear solve naming the layer. Every other case here is driven at
    # T_z == T_o by wind or by conduction, so only this one needs a seed.
    A, h, E = 0.5, 3.0, 2000.0
    _, model, state, drivers = _single_zone(A=A, h=h, E=E, T_start=T_O + 5.0)
    ss = model.steady(state, drivers, **TIGHT)
    alpha, _, _ = _params(A, h, E)
    q_ref = 2.0 ** (1.0 / 3.0) * alpha
    q = _volumetric_inflow(model, ss)
    assert q == pytest.approx(q_ref, rel=1e-6)
    dT = ss["thermal.x"][0].item() - T_O
    assert dT == pytest.approx(E / (RHO_0 * CP_AIR * q), rel=1e-8)
    record_property("check", "Buoyancy-only zone vs the Li and Delsante closed form")
    record_property("tolerance", "rel 1e-6")
    record_property("measured_rel", abs(q / q_ref - 1.0))


def test_envelope_loss_matches_the_cubic_root(record_property):
    A, h, E, UA = 0.5, 3.0, 2000.0, 150.0
    _, model, state, drivers = _single_zone(A=A, h=h, E=E, UA=UA)
    ss = model.steady(state, drivers, **TIGHT)
    alpha, beta, _ = _params(A, h, E, UA)
    (root,) = _positive_real_roots([1.0, 3.0 * beta, 0.0, -2.0 * alpha**3])
    q = _volumetric_inflow(model, ss)
    assert q == pytest.approx(root, rel=1e-6)
    record_property("check", "Envelope-loss zone vs the cubic root")
    record_property("tolerance", "rel 1e-6")
    record_property("measured_rel", abs(q / root - 1.0))


def test_assisting_wind_matches_the_cubic_root(record_property):
    A, h, E, UA, V = 0.5, 3.0, 2000.0, 150.0, 2.0
    cp_low, cp_high = 0.5, -0.3
    _, model, state, drivers = _single_zone(A=A, h=h, E=E, UA=UA, cp_low=cp_low, cp_high=cp_high)
    drivers["V_met"] = torch.tensor(V, dtype=F64)
    ss = model.steady(state, drivers, **TIGHT)
    dPw = 0.5 * (cp_low - cp_high) * V**2            # kinematic, > 0: assists the upflow
    alpha, beta, gamma = _params(A, h, E, UA, dPw)
    (root,) = _positive_real_roots(
        [1.0, 3.0 * beta, -3.0 * gamma**2, -2.0 * alpha**3 - 9.0 * gamma**2 * beta]
    )
    q = _volumetric_inflow(model, ss)
    assert q == pytest.approx(root, rel=1e-6)
    record_property("check", "Assisting-wind zone vs the cubic root")
    record_property("tolerance", "rel 1e-6")
    record_property("measured_rel", abs(q / root - 1.0))


def _opposing_wind_case(coupling: str = "iterate"):
    """Li and Delsante's beta = 0, alpha = 0.9, gamma = 1 three-root example, built HERE for
    both tests below so the two can never drift into describing different cases.

    Returns the model, its initial state, its drivers, and the `alpha` and `E` the reference
    cubics and the unstable root's temperature are computed from.
    """
    h, alpha = 1.0, 0.9
    A = math.sqrt(2.0) / CD                          # Cd A* = 1
    E = alpha**3 / h * RHO_0 * CP_AIR * T_O / G      # B h = alpha^3
    cp_low, cp_high = -0.3, 0.5                      # top windward: opposes the upflow
    V = math.sqrt(1.5 / (0.5 * 0.8))                 # |dP_w| = 1.5 -> gamma = 1
    _, model, state, drivers = _single_zone(
        A=A, h=h, E=E, cp_low=cp_low, cp_high=cp_high, coupling=coupling
    )
    drivers["V_met"] = torch.tensor(V, dtype=F64)
    return model, state, drivers, alpha, E


@pytest.mark.xfail(
    strict=True,
    reason=(
        "coupling='iterate' CONVERGES here, but both instances land on the SAME stable root "
        "-- the wind-driven downward q = 1.3993 m^3/s -- whatever the start, so the upward "
        "root Li and Delsante also call stable is unreachable. Measured cause: the "
        "quasi-steady map T -> q(T) -> T = T_o + E/(rho c_p |q|) has slope g' = -7.756 at "
        "the upward root (T - T_o = 46.2943 K, central difference at 1e-3 K), so "
        "under-relaxed substitution x <- x + omega (g(x) - x) amplifies by "
        "|1 - omega (1 - g')| = |1 - 8.756 omega|, which is 3.38 at the 0.5 relaxation "
        "HARD-CODED in Model._iterate: started exactly on the root, the iteration still "
        "walks away to 15.0422 K in 52 passes. The blocker is that FIXED 0.5, not "
        "successive substitution as a method -- any omega < 0.228 contracts, and iterating "
        "these same measured relations at omega = 0.15 from 46.0 K gives 46.414, 46.261, "
        "46.305, 46.291, 46.295, converging on the root. So: no TOLERANCE reaches it (which "
        "is what this xfail is about, and why nothing here was loosened), but adaptive or "
        "user-settable under-relaxation is a candidate remedy alongside the spec's section "
        "13 monolithic Newton, and the evidence does not choose between them. The physics "
        "is sound -- the model resolves all three roots under TIME STEPPING, see "
        "test_opposing_wind_multiplicity_resolves_per_instance_under_time_stepping, which "
        "is the per-instance multiplicity check this case is here to make."
    ),
)
def test_opposing_wind_three_root_case_converges_per_instance_to_a_stable_root():
    """Li and Delsante's beta = 0, alpha = 0.9, gamma = 1 example: three steady states
    (q = 0.45 up, 0.54 down [unstable], 1.40 down). A cold start (wind wins) and a hot start
    (buoyancy wins) are run as one batch; each instance must land on a root of its branch's
    cubic. If the fixed-point coupling fails to converge here, that is the spec's section 13
    monolithic-Newton trigger: record it in the ledger and mark this test
    xfail(strict=True) with that reason -- do not loosen the tolerances.
    """
    model, state, drivers, alpha, _ = _opposing_wind_case()
    starts = dict(state, **{"thermal.x": torch.tensor([[T_O], [T_O + 40.0]], dtype=F64)})
    diag: dict = {}
    ss = model.steady(starts, drivers, diagnostics=diag, **TIGHT)
    assert bool(diag["converged"].all()), diag
    q = model.potential["air"].flows_of_kind(ss["air.q"], "airpath")[..., 0] / RHO_0
    up = _positive_real_roots([1.0, 0.0, 3.0, -2.0 * alpha**3])          # one root ~0.45
    down = _positive_real_roots([1.0, 0.0, -3.0, 2.0 * alpha**3])        # ~0.54, ~1.40
    assert up[0] == pytest.approx(0.4547, abs=2e-3)
    assert down == pytest.approx([0.537, 1.40], abs=5e-3)
    assert q[1].item() == pytest.approx(up[0], rel=1e-4)                 # hot start: up
    assert q[0].item() < 0                                               # cold start: down
    assert -q[0].item() == pytest.approx(down[1], rel=1e-4)              # the stable one


def test_opposing_wind_multiplicity_resolves_per_instance_under_time_stepping():
    """The per-instance multiplicity check on the three-root case, made with the instrument
    that can actually make it: TIME STEPPING.

    Li and Delsante classify the three roots by the STABILITY OF THE DYNAMICS (d/dT of the
    zone energy balance), not by the contractivity of a quasi-steady fixed-point iteration,
    and the two orderings genuinely differ here -- see the xfail above. Ping-pong stepping
    has exactly the coupled steady states as its fixed points (a step leaves x unchanged only
    where rate(x, q(x)) == 0), and at a small enough dt it tracks the dynamics, so it is the
    instrument that reproduces the paper's classification.

    Four starts are run AS ONE BATCH, per instance: cold (the wind wins), hot, and the
    unstable middle root perturbed by +/- 0.5 K. Each must land on a STABLE root of its
    branch, and none on the unstable middle one -- which a start sitting a half kelvin from
    it is the sharp test of.
    """
    model, _, drivers, alpha, E = _opposing_wind_case("pingpong")
    up = _positive_real_roots([1.0, 0.0, 3.0, -2.0 * alpha**3])          # one root ~0.45
    down = _positive_real_roots([1.0, 0.0, -3.0, 2.0 * alpha**3])        # ~0.54, ~1.40
    dT_unstable = E / (RHO_0 * CP_AIR * down[0])                         # 39.133 K
    starts = [0.0, 40.0, dT_unstable + 0.5, dT_unstable - 0.5]
    state = {"thermal.x": torch.tensor([[T_O + d] for d in starts], dtype=F64)}
    dt, steps = 20.0, 1200                           # 24000 s, ~109 zone time constants
    diag: dict = {}
    for _ in range(steps):
        state = model.step(state, drivers, dt, diagnostics=diag, **TIGHT)
    assert bool(diag["layers"]["air"]["converged"].all()), diag   # per instance, every one
    q = model.potential["air"].flows_of_kind(state["air.q"], "airpath")[..., 0] / RHO_0
    assert q[0].item() == pytest.approx(-down[1], rel=1e-6)       # cold: wind wins, downward
    assert q[1].item() == pytest.approx(up[0], rel=1e-6)          # hot: buoyancy wins, upward
    assert q[2].item() == pytest.approx(up[0], rel=1e-6)          # +0.5 K off the middle root
    assert q[3].item() == pytest.approx(-down[1], rel=1e-6)       # -0.5 K off it
    assert min(abs(abs(v) - down[0]) for v in q.tolist()) > 0.05  # never the unstable middle


def _two_zone_doorway():
    """A (heated, no envelope openings) -- doorway -- B (two openings to ambient)."""
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    add_zone(net, Zone("A", volume=60.0, T0=T_O + 5.0))
    add_zone(net, Zone("B", volume=60.0, T0=T_O + 2.0))
    add_large_opening(net, "A", "B", H=2.0, W=0.9, z_mid=1.0, Cd=0.78)
    net.add_edge("ambient", "B", kind="airpath", z_path=0.2, Cd=CD, area=0.05)
    net.add_edge("ambient", "B", kind="airpath", z_path=1.8, Cd=CD, area=0.05)
    el = orifice_elements_from_edges(net, "airpath")
    model = build_model(
        net, air_elements=[el], drives=[Stack.from_network(net, "airpath")], density="linear",
        density_kwargs={"rho_0": RHO_0, "T_0": T_O}, coupling="iterate",
        iterate_tol={"thermal": 1e-9}, iterate_max=300,
    )
    E_A = 1000.0
    sources = torch.zeros(net.n, dtype=F64)
    sources[net.node_index("A")] = E_A
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=F64),
        "thermal.x_boundary": torch.tensor([T_O], dtype=F64),
        "thermal.sources": sources,
    }
    return net, model, initial_state(model), drivers, E_A


def test_two_zone_doorway_matches_an_independent_root_finding_reference(record_property):
    from scipy.optimize import fsolve

    net, model, state, drivers, E_A = _two_zone_doorway()
    ss = model.steady(state, drivers, **TIGHT)
    th = model.layers["thermal"]
    names = [net.nodes[i] for i in th.interior_idx.tolist()]
    T = dict(zip(names, ss["thermal.x"].tolist(), strict=True))

    def balances(v):
        TA, TB = v
        drho_AB = RHO_0 * (TA - TB) / T_O
        drho_oB = RHO_0 * (TB - T_O) / T_O
        F_door = 0.78 / 3.0 * 0.9 * math.sqrt(RHO_0 * G * max(drho_AB, 1e-12) * 2.0**3)
        F_env = CD * 0.05 * math.sqrt(RHO_0 * G * 1.6 * max(drho_oB, 1e-12))
        return [
            CP_AIR * F_door * (TB - TA) + E_A,
            CP_AIR * F_door * (TA - TB) + CP_AIR * F_env * (T_O - TB),
        ]

    TA_ref, TB_ref = fsolve(balances, [T_O + 10.0, T_O + 5.0], xtol=1e-12)
    assert max(abs(v) for v in balances([TA_ref, TB_ref])) < 1e-8
    assert T["A"] == pytest.approx(TA_ref, abs=1e-5)
    assert T["B"] == pytest.approx(TB_ref, abs=1e-5)
    q = model.potential["air"].flows_of_kind(ss["air.q"], "airpath")
    assert (q[0] + q[1]).abs().item() < 1e-10          # doorway: no net flow
    worst_abs = max(abs(T["A"] - TA_ref), abs(T["B"] - TB_ref))
    record_property("check", "Two-zone doorway vs an independent scipy.optimize.fsolve reference")
    record_property("tolerance", "abs 1e-5 K")
    record_property("measured_abs_K", worst_abs)


def _three_zone_stack(coupling: str, tol: float = 0.05):
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    for i, z in enumerate((0.0, 3.0, 6.0)):
        add_zone(net, Zone(f"z{i + 1}", volume=120.0, T0=T_O + 5.0, z_ref=z))
        net.add_edge("ambient", f"z{i + 1}", kind="airpath", z_path=z + 0.3, Cd=CD, area=0.02)
        net.add_edge("ambient", f"z{i + 1}", kind="airpath", z_path=z + 2.7, Cd=CD, area=0.02)
    add_large_opening(net, "z1", "z2", H=1.0, W=1.5, z_mid=3.0, Cd=0.78)
    add_large_opening(net, "z2", "z3", H=1.0, W=1.5, z_mid=6.0, Cd=0.78)
    el = orifice_elements_from_edges(net, "airpath")
    model = build_model(
        net, air_elements=[el], drives=[Stack.from_network(net, "airpath")],
        coupling=coupling, iterate_tol={"thermal": tol}, iterate_max=50,
    )
    sources = torch.zeros(net.n, dtype=F64)
    for name in ("z1", "z2", "z3"):
        sources[net.node_index(name)] = 800.0
    return net, model, initial_state(model), sources


def _run_stack(coupling: str, dt: float, hours: float = 6.0) -> dict:
    net, model, state, sources = _three_zone_stack(coupling)
    lo = model.potential["air"].kind_slice("airpath").start
    door_index = 6                                                   # z1 -> z2 low opening
    flows, T3 = [], []
    steps = int(round(hours * 3600.0 / dt))
    for k in range(steps):
        t = (k + 1) * dt
        T_amb = T_O + 5.0 * math.sin(2.0 * math.pi * t / (24.0 * 3600.0))
        drivers = {
            "air.phi_boundary": torch.zeros(1, dtype=F64),
            "thermal.x_boundary": torch.tensor([T_amb], dtype=F64),
            "thermal.sources": sources,
        }
        state = model.step(state, drivers, dt)
        flows.append(abs(state["air.q"][lo + door_index].item()))
        T3.append(state["thermal.x"][2].item())
    return {"max_flow": max(flows), "mean_flow": sum(flows) / len(flows), "max_T3": max(T3)}


def test_hensen_table_fine_steps_agree_and_coarse_pingpong_errs_more_than_coarse_onion():
    fine_pp = _run_stack("pingpong", 360.0)
    fine_on = _run_stack("iterate", 360.0)
    coarse_pp = _run_stack("pingpong", 3600.0)
    coarse_on = _run_stack("iterate", 3600.0)
    print("\nHensen-style table (max flow kg/s, mean flow kg/s, max T3 K):")
    rows = (("On-10", fine_on), ("PP-10", fine_pp), ("On-1", coarse_on), ("PP-1", coarse_pp))
    for name, r in rows:
        print(f"  {name:6s} {r['max_flow']:.4f} {r['mean_flow']:.4f} {r['max_T3']:.3f}")
    for key in ("max_flow", "mean_flow", "max_T3"):
        assert fine_pp[key] == pytest.approx(fine_on[key], rel=2e-2)
    err_pp = abs(coarse_pp["max_flow"] - fine_on["max_flow"])
    err_on = abs(coarse_on["max_flow"] - fine_on["max_flow"])
    assert err_pp > err_on


def test_golden_matches_stored_reference(record_property):
    """The `benchmarks/natural_ventilation.py` demo against its committed trajectory.

    Two simulated hours at 600 s under `coupling="iterate"`, compared term by term. The demo
    has no RNG, so it regenerates bit-exactly and the tolerance here (1e-8 relative) is far
    tighter than any physical claim: this is a change detector for the whole milestone-2
    stack at once -- `build_model`, `Stack`, the density closure, the large opening and the
    iterated coupling -- not a verification, which is what every other test in this file is.
    Regenerate with `.venv/Scripts/python benchmarks/regenerate_golden.py` only when a change
    to those numbers is intended.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from golden import load_golden

    from benchmarks.natural_ventilation import run

    ref = load_golden("natural_ventilation")
    now = run("iterate", 600.0, hours=2.0)
    worst_rel = 0.0
    for key in ("T_A", "T_B", "door_kg_s"):
        now_t = torch.tensor(now[key], dtype=F64)
        ref_t = torch.tensor(ref[key], dtype=F64)
        torch.testing.assert_close(now_t, ref_t, rtol=1e-8, atol=1e-10)
        worst_rel = max(
            worst_rel, ((now_t - ref_t) / ref_t.abs().clamp_min(1e-12)).abs().max().item()
        )
    record_property("check", "Natural-ventilation demo golden regression")
    record_property("tolerance", "rel 1e-8")
    record_property("measured_rel", worst_rel)
