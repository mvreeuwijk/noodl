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

import tellegen.solvers.select as select_module
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


def test_composed_layer_and_dense_layer_agree_and_take_different_paths(monkeypatch):
    """The composed model's parity gate is only a gate if its two layers are two code paths.

    `model.layer` is the migrated default (`linear_solver="auto"`, a Jacobi-PCG solve of the
    matvec-free GraphLaplacianOperator); `model.dense_layer` is the retained milestone-1
    reference (`linear_solver="direct"`, the same operator assembled and LU-factorised). The
    spy on `select.pcg` is what proves the second does not quietly run the first's solver --
    without it, a later parity assertion would be comparing the sparse path with itself.
    """
    model = build_composed()

    pcg_calls = []
    real_pcg = select_module.pcg

    def spy_pcg(*args, **kwargs):
        pcg_calls.append(1)
        return real_pcg(*args, **kwargs)

    monkeypatch.setattr(select_module, "pcg", spy_pcg)

    phi_sparse, q_sparse = model.layer.solve(
        model.phi_boundary, model.drivers, model.sources, differentiable=False
    )
    assert pcg_calls, "model.layer must solve through pcg"
    pcg_after_sparse = len(pcg_calls)

    phi_dense, q_dense = model.dense_layer.solve(
        model.phi_boundary, model.drivers, model.sources, differentiable=False
    )
    assert len(pcg_calls) == pcg_after_sparse, "model.dense_layer must not call pcg"

    torch.testing.assert_close(phi_sparse, phi_dense, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(q_sparse, q_dense, rtol=1e-9, atol=1e-12)
