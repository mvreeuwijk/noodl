"""Inverse example 1 -- calibration through the join.

Design spec `docs/superpowers/specs/2026-09-19-milestone-5-coupling-design.md`. Recovers the
CONTAM building's leakage coefficients (the `pl_3` `UpstreamDensityPowerLaw`'s `C`, one per
path) by gradient descent on an indoor-concentration observation, with the gradient running
through the whole coupled street+building step (the potential solve's implicit-function
adjoint, the outer fixed-point iteration unrolled).
"""

from __future__ import annotations

import torch

from tests.verification.test_coupling_demo import (
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
    # atol=1e-10 (not 0.0): "r3" has NO advective path into the shared street under this
    # wind (n3 -> n1 flows away from n1, never into it), so its true sensitivity is a
    # structural zero -- FD lands on it exactly (0.0), while the adjoint returns ~2e-20,
    # pure floating-point noise. That is 6+ orders of magnitude below the smallest REAL
    # signal here (r1, ~2.9e-4), so atol=1e-10 forgives only the noise floor and leaves the
    # rtol=1e-4 check exactly as strict on every street with an actual gradient.
    assert torch.allclose(attribution, fd, rtol=1e-4, atol=1e-10)
    assert torch.all(attribution >= 0)
    assert attribution.argmax().item() == names.index(SHARED)  # the shared street dominates
