"""Inverse example 1 -- calibration through the join.

Design spec `docs/superpowers/specs/2026-09-19-milestone-5-coupling-design.md`. Recovers the
CONTAM building's leakage coefficients (the `pl_3` `UpstreamDensityPowerLaw`'s `C`, one per
path) by gradient descent on an indoor-concentration observation, with the gradient running
through the whole coupled street+building step (the potential solve's implicit-function
adjoint; the coupler's own outer fixed point is differentiated with the implicit adjoint of
the interface equations, `solvers.fixed_point.differentiate_fixed_point`, not by unrolling).
"""

from __future__ import annotations

import pytest
import torch

# All three inverse examples reuse the demo's own PRIVATE fixtures (`_street`, `_building`,
# `_coupled`, `_small_street_network`) rather than rebuilding the pairing: they are the same
# street canyon and the same CONTAM project, so every number recorded here moves with
# `test_coupling_demo.py` -- change a fixture there and these examples' measurements change
# with it, by design.
from tests.verification.test_coupling_demo import (
    F64,
    SHARED,
    _building,
    _coupled,
    _small_street_network,
    _street,
)


def _city(**kwargs):
    net = _small_street_network()
    street_model, street_state, street_drivers = _street(net)
    _project, (building_model, building_state, building_drivers) = _building()
    street_triple = (street_model, street_state, street_drivers)
    building_triple = (building_model, building_state, building_drivers)
    kwargs.setdefault("iterate_rtol", 1e-8)
    kwargs.setdefault("iterate_max", 100)
    city, state, drivers = _coupled(street_triple, building_triple, street_model, **kwargs)
    return city, state, drivers, street_model, building_model


def _indoor_after(city, state, drivers, n_steps=10, dt=60.0) -> torch.Tensor:
    s = state
    for _ in range(n_steps):
        s = city.step(s, drivers, dt=dt)
    return s["building"]["species.x"].flatten()


@pytest.mark.slow
def test_calibrate_leakage_coefficients_by_gradient_through_the_coupled_model(record_property):
    city, state, drivers, _street_model, building_model = _city()
    element, _cols = building_model.potential["air"].element_for("pl_3")
    if not element.C.requires_grad:
        element.C.requires_grad_(True)
    true_C = element.C.detach().clone()
    try:
        observed = _indoor_after(city, state, drivers, n_steps=3).detach()

        with torch.no_grad():
            element.C.mul_(1.5)                      # start 50 % too leaky
        optimizer = torch.optim.Adam([element.C], lr=0.02)
        history = []
        for step_idx in range(150):
            optimizer.zero_grad()
            predicted = _indoor_after(city, state, drivers, n_steps=3)
            loss = ((predicted - observed) / observed.abs().clamp_min(1e-30)).pow(2).sum()
            loss.backward()
            if step_idx == 0 and element.C.grad is not None and element.C.grad.abs().max() == 0:
                raise AssertionError(
                    f"zero gradient on element.C after the first backward() -- "
                    f"requires_grad={element.C.requires_grad}, grad={element.C.grad}"
                )
            optimizer.step()
            history.append(loss.item())
        recovered = element.C.detach()
        rel = ((recovered - true_C).abs() / true_C).max().item()
        record_property("leakage_relative_error_after_calibration", rel)
        record_property("final_loss", history[-1])
        assert history[-1] < history[0] * 1e-3
        assert rel < 0.05
    finally:
        with torch.no_grad():
            element.C.copy_(true_C)                  # leave the shared fixture as found


def test_attribute_indoor_concentration_to_street_emissions_with_one_backward_pass(
    record_property,
):
    """Source attribution: d(indoor mass fraction) / d(per-street emission), all streets at
    once, from ONE `torch.autograd.grad` call through the coupled street+building model --
    checked against central finite differences taken street by street."""
    city, state, drivers, street_model, _building_model = _city(iterate_rtol=1e-10)
    graph = street_model.net
    names = [s.name for s in _small_street_network().streets]
    columns = torch.tensor([graph.node_index(n) for n in names], dtype=torch.long)

    def indoor_given(emissions: torch.Tensor) -> torch.Tensor:
        d = {tag: dict(v) for tag, v in drivers.items()}
        sources = d["street"]["street.sources"].clone()
        sources[columns] = emissions                     # FULL node order: atmosphere first
        d["street"]["street.sources"] = sources
        return _indoor_after(city, state, d, n_steps=3)[0]  # zone "one", the inlet zone

    base = drivers["street"]["street.sources"][columns].detach().clone()
    e = base.clone().requires_grad_(True)
    attribution, = torch.autograd.grad(indoor_given(e), e)  # ONE backward pass

    # `h = 1e-8` is 10 % of the 1e-7 kg/s emission -- an enormous step for a finite
    # difference, and valid ONLY because the indoor concentration is exactly LINEAR in
    # `street.sources`: neither the street's own flows `street.q` (closure-prescribed from
    # the canyon wind) nor the building's `air.q` (driven by wind and buoyancy) depends on
    # the emissions at all, so the whole map from sources to indoor mass fraction is linear
    # and a central difference is exact up to rounding. That is also why the agreement below
    # is ~5e-7 rather than the ~h^2 truncation error a nonlinear map would show: what is left
    # is floating-point cancellation between two nearly-equal 3-step solves, not truncation.
    h = 1e-8
    fd = torch.zeros_like(base)
    for i in range(len(names)):
        up, down = base.clone(), base.clone()
        up[i] += h
        down[i] -= h
        fd[i] = (indoor_given(up) - indoor_given(down)) / (2 * h)
    record_property(
        "attribution_kgkg_per_kgs", dict(zip(names, attribution.tolist(), strict=True))
    )
    record_property("fd_kgkg_per_kgs", dict(zip(names, fd.tolist(), strict=True)))
    idx_r1, idx_r2, idx_r3 = names.index("r1"), names.index("r2"), names.index("r3")
    # r3 (n3 (30, 0) -> n1 (30, 30): due NORTH) runs PERPENDICULAR to the theta_w=0
    # (due-east) wind, so its along-canyon velocity is ~0 (measured u_canyon ~= 8.79e-18
    # m/s, vs ~1.3625e-1 m/s for r1 and r2, which both run east-west, parallel to the
    # wind) -- it exchanges essentially no advective flux with junction n1 in EITHER
    # direction, and its emission leaves almost entirely through roof exchange instead.
    # Its own route flows are themselves ~5e-17 (not exactly zero), so this is a
    # genuinely negligible signal, not a purely structural (exactly-zero) one: the FD
    # estimate happens to land on exact 0.0 while the adjoint returns ~2e-20, both
    # consistent with "below any resolvable threshold" rather than disagreeing. The
    # load-bearing rtol=1e-4 check stays exactly as strict as the brief specifies
    # (atol=0.0) on r1/r2, the two streets with a real, resolvable signal; r3 gets its
    # own explicit, separate structural/negligible-zero check instead of being folded
    # into a global atol that would silently loosen the r1/r2 comparison too.
    assert torch.allclose(
        attribution[[idx_r1, idx_r2]], fd[[idx_r1, idx_r2]], rtol=1e-4, atol=0.0
    )
    assert attribution[idx_r3].abs() < 1e-12
    assert fd[idx_r3].abs() < 1e-12
    assert torch.all(attribution >= 0)
    assert attribution.argmax().item() == names.index(SHARED)  # the shared street dominates


def test_latent_infiltration_from_one_measured_path_and_the_cycle_ensemble(record_property):
    """Inverse example 3 -- latent infiltration via the cycle space: on the CONTAM
    three-zone building's own airflow graph (ambient->one->two<-three<-ambient, ONE
    cycle), one measured path plus conservation alone pins every other flow, and with
    no measurement at all the admissible conserved flows are exactly the cycle-space
    ensemble."""
    from noodl.cycles import branch_flows, project_measured

    _city_, state, drivers, _street_model, building_model = _city()
    net, kind = building_model.net, "pl_3"
    assert net.n_cycles == 1
    solved = building_model.step(state["building"], drivers["building"], dt=60.0)["air.q"]

    # (a) Measure ONE path (the ambient->one inlet, selected by NAME rather than
    #     position) and recover every other flow from conservation alone: the "latent
    #     infiltration" the sensor never saw.
    measured = next(i for i, e in enumerate(net.edges) if e[:2] == ("ambient", "one"))
    mask = torch.zeros(net.b, dtype=torch.bool)
    mask[measured] = True
    target = torch.zeros(net.b, dtype=F64)
    target[measured] = solved[measured]
    recovered = project_measured(net, target, mask, kind)
    assert torch.allclose(recovered, solved, rtol=1e-10, atol=1e-14)

    # (b) With NO measurement, the admissible conserved flows are the cycle space: an
    #     ensemble over its amplitude is the spread of every latent path -- equal on every
    #     edge of this single loop -- and collapses to zero once one path is measured.
    _tree, chord_cols = net.spanning_forest(kind)
    assert chord_cols.numel() == 1
    chord = int(chord_cols[0])
    record_property("chord_edge", list(net.edges[chord]))
    # `branch_flows` places the amplitude directly on the chord edge, with a +1
    # self-entry, so the physical loop's amplitude IS the chord's own signed flow.
    # Seeding from a tree edge (e.g. the measured one) would instead fix the loop's
    # rotational sense by that edge's orientation, which need not agree with the
    # chord's -- here it doesn't, since the loop traverses the chord edge backwards
    # relative to its incidence direction.
    amplitudes = solved[chord] + 0.1 * solved[chord].abs() * torch.randn(
        200, 1, dtype=F64, generator=torch.Generator().manual_seed(0)
    )
    ensemble = branch_flows(net, amplitudes, kind)             # (200, b)
    spread = ensemble.std(dim=0)
    record_property("unmeasured_spread_kg_s", spread.tolist())
    assert torch.all(spread > 0)
    assert torch.allclose(spread, spread[0].expand_as(spread), rtol=1e-10)
    assert torch.allclose(ensemble.mean(dim=0), solved, rtol=0.05)  # centred on the truth
