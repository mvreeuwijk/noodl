"""Street air quality: ensemble equivalence, species permutation, constant tracer, gradients."""

from __future__ import annotations

import math

import pytest
import torch

from noodl.apps.street_aq.chemistry import photostationary_for_streets, street_steady
from noodl.apps.street_aq.network import build_model, from_test_network

DT = torch.float64
OPTS = {"pblh_floor": False, "kappa": 0.4}
NO_O3 = ("no", "no2", "o3")


def _drivers(model, species, b=None, *, c=None):
    """Uniform meteorology (batched over `b` forcing hours when given) and per-species
    sources and background; `c` replaces them by a conservative background and no sources."""
    n, k = model.net.n, len(species)
    base = {"no": 2.0e-3, "no2": 4.0e-4, "o3": 0.0, "nox": 1e-3, "tracer": 5e-4}
    back = {"no": 0.0, "no2": 0.0, "o3": 8.0e-8, "nox": 1e-5, "tracer": 2e-5}
    sources = torch.zeros(n, k, dtype=DT)
    boundary = torch.tensor([[back[s] for s in species]], dtype=DT)
    for j, s in enumerate(species):
        sources[model.net.node_index("r1"), j] = base[s]
        sources[model.net.node_index("r3"), j] = 0.5 * base[s]
    if c is not None:
        sources = torch.zeros(n, k, dtype=DT)
        boundary = torch.full((1, k), c, dtype=DT)
    shape = () if b is None else (b,)
    hours = torch.arange(b, dtype=DT) if b else None
    u = torch.tensor(3.0, dtype=DT) if b is None else 2.0 + hours
    th = torch.tensor(0.4, dtype=DT) if b is None else 0.3 + 0.7 * hours
    d = {
        "street.x_boundary": boundary.expand(*shape, 1, k).clone(),
        "street.sources": sources.expand(*shape, n, k).clone(),
        "U_ref": u, "theta_w": th,
        "h_abl": torch.full(shape, 800.0, dtype=DT),
        "J_NO2": torch.full(shape, 5e-3, dtype=DT),
        "temperature": torch.full(shape, 298.0, dtype=DT),
    }
    return d


def _single_species_unbatched(d):
    return {k: (v[..., 0] if "x_boundary" in k or "sources" in k else v) for k, v in d.items()}


@pytest.mark.parametrize("b", [1, 2, 3, 4])  # the test network has 3 streets, 4 nodes
def test_batched_forcing_steady_equals_stack_of_single_runs(b):
    model, state, _ = build_model(from_test_network(), species=("nox",), **OPTS)
    d = _drivers(model, ("nox",), b)
    d = _single_species_unbatched(d)
    out = model.steady(state, d)["street.x"]
    assert out.shape == (b, 3)
    for i in range(b):
        di = {k: (v[i] if v.dim() > (1 if "x_boundary" in k or "sources" in k else 0) else v)
              for k, v in d.items()}
        single = model.steady(state, di)["street.x"]
        torch.testing.assert_close(out[i], single, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("perm", [(1, 0, 2), (2, 1, 0), (1, 2, 0)])
def test_species_order_permutes_the_chemistry_result(perm):
    """Photostationary chemistry finds its columns by species name: any order of
    ('no', 'no2', 'o3') gives the same fields, column-permuted."""
    def run(species):
        reaction = photostationary_for_streets(species, preset="munich")
        model, state, _ = build_model(from_test_network(), species=species,
                                      chemistry=reaction, **OPTS)
        d = _drivers(model, species)
        return street_steady(model, state, d, reaction=reaction, tol=1e-18, max_iter=200)

    ref = run(NO_O3)["street.x"]
    species = tuple(NO_O3[i] for i in perm)
    torch.testing.assert_close(run(species)["street.x"], ref[..., list(perm)],
                               rtol=1e-9, atol=1e-18)


def test_a_passive_tracer_beside_the_chemistry_is_transport_only():
    """The extra species' column equals a transport-only run of that species alone."""
    species = (*NO_O3, "tracer")
    reaction = photostationary_for_streets(species, preset="munich")
    model, state, _ = build_model(from_test_network(), species=species, chemistry=reaction,
                                  **OPTS)
    mixed = street_steady(model, state, _drivers(model, species), reaction=reaction,
                          tol=1e-18, max_iter=200)["street.x"][..., 3]
    alone_model, alone_state, _ = build_model(from_test_network(), species=("tracer",), **OPTS)
    alone = alone_model.steady(alone_state, _single_species_unbatched(
        _drivers(alone_model, ("tracer",))))["street.x"]
    torch.testing.assert_close(mixed, alone, rtol=1e-9, atol=1e-18)


@pytest.mark.parametrize("species", [("tracer",), ("tracer", "nox")])
@pytest.mark.parametrize("b", [None, 3])
def test_a_uniform_conservative_tracer_equals_the_background_everywhere(species, b):
    """No sources and every boundary at c: the steady state is c in every street."""
    c = 3.5e-5
    model, state, _ = build_model(from_test_network(), species=species, **OPTS)
    d = _drivers(model, species, b, c=c)
    if len(species) == 1:
        d = _single_species_unbatched(d)
    x = model.steady(state, d)["street.x"]
    torch.testing.assert_close(x, torch.full_like(x, c), rtol=1e-9, atol=0.0)


def test_steady_gradient_matches_central_differences():
    """d(sum of concentrations)/d(emission of r1) and d/d(wind speed): the closures are smooth
    away from the Bessel/direction switch (theta_w = 0.4 is far from it)."""
    model, state, _ = build_model(from_test_network(), species=("nox",), **OPTS)
    d = _single_species_unbatched(_drivers(model, ("nox",)))
    i = model.net.node_index("r1")

    def loss(emission, u):
        src = d["street.sources"].clone()
        src[i] = emission
        return model.steady(state, dict(d, **{"street.sources": src, "U_ref": u}))[
            "street.x"].sum()

    e = d["street.sources"][i].clone().requires_grad_(True)
    u = d["U_ref"].clone().requires_grad_(True)
    loss(e, u).backward()
    h = 1e-6
    with torch.no_grad():
        fd_e = (loss(e + h * e, u) - loss(e - h * e, u)) / (2 * h * e)
        fd_u = (loss(e, u + h) - loss(e, u - h)) / (2 * h)
    assert e.grad.item() == pytest.approx(fd_e.item(), rel=1e-5)
    assert u.grad.item() == pytest.approx(fd_u.item(), rel=1e-5)
    assert math.isfinite(u.grad.item())
