"""Tests for the composed reference model: 8 buildings joined through a street and a sewer
network (milestone 1b design, section 6: "the smallest thing that is honestly a digital twin
rather than a microbenchmark"). `build_composed` returns a `ComposedModel` carrying both the
topology (reused by later tasks' physics at whatever configuration each of them needs) and a
reference physics configuration (PowerLaw elements, sources, capacity, a migrated
`PotentialFlowLayer`, a dense-path `PotentialFlowLayer` and a `TransportLayer`), per the
milestone-1b amendments (A3.4).
"""

from __future__ import annotations

import time

import torch

from benchmarks.composed_model import ComposedModel, build_composed


def test_build_composed_produces_expected_node_and_edge_counts():
    model = build_composed()
    # Node count is exact and seed-independent: 8 * 120 + 40 + 30.
    assert model.net.n == 1030
    # Edge count is seed-dependent (random extra edges beyond each spanning tree) but tightly
    # bounded: measured 2186-2196 over seed in range(5) against this exact construction.
    assert 2100 <= model.net.b <= 2300


def test_build_composed_graph_is_connected():
    model = build_composed()
    assert model.net.n_components == 1


def test_build_composed_every_interface_node_exists_in_the_network():
    model = build_composed()
    node_set = set(model.net.nodes)
    assert set(model.interface_nodes.keys()) == {"street", "sewer"}
    for i in range(8):
        key = f"building_{i}"
        ambient_node, street_node = model.interface_nodes["street"][key]
        manhole_node, sewer_node = model.interface_nodes["sewer"][key]
        assert ambient_node in node_set
        assert street_node in node_set
        assert manhole_node in node_set
        assert sewer_node in node_set


def test_build_composed_is_reproducible_given_the_same_seed():
    model1 = build_composed(seed=0)
    model2 = build_composed(seed=0)
    assert model1.net.n == model2.net.n
    assert model1.net.b == model2.net.b
    assert model1.interface_nodes == model2.interface_nodes


def test_build_composed_builds_in_under_two_seconds():
    t0 = time.perf_counter()
    build_composed()
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0


def test_build_composed_solves_on_the_dense_path_at_ensemble_one():
    model = build_composed()
    phi, q = model.layer.solve(
        model.phi_boundary, model.drivers, model.sources, differentiable=False
    )
    r = model.layer.residual(
        phi[..., model.layer.interior], model.phi_boundary, model.drivers, model.sources
    )
    assert float(r.abs().max()) < 1e-8
    # conservation uses layer.A (columns in LAYER order), not net.incidence(): q is in layer order.
    imbalance = torch.einsum("ne,...e->...n", model.layer.A, q) - model.sources
    assert float(imbalance[..., model.layer.interior].abs().max()) < 1e-8


def test_build_composed_transport_steps_on_the_dense_path():
    model = build_composed()
    phi, q = model.layer.solve(
        model.phi_boundary, model.drivers, model.sources, differentiable=False
    )
    lo, hi = model.layer._kind_slices["airpath"]
    n_i = model.layer.interior.numel()
    x0 = torch.full((1, n_i), 400.0, dtype=torch.float64)
    x1 = model.transport.step(
        x0,
        q[..., lo:hi],
        torch.zeros_like(x0),
        torch.full((1, 2), 400.0, dtype=torch.float64),
        dt=60.0,
    )
    assert x1.shape == x0.shape and torch.isfinite(x1).all()


def test_composed_model_fixture_matches_a_direct_call(composed_model):
    model = composed_model
    ref = build_composed()
    assert isinstance(model, ComposedModel)
    assert model.ensemble == 1
    assert model.net.n == ref.net.n
    assert model.net.b == ref.net.b
    assert model.interface_nodes == ref.interface_nodes
