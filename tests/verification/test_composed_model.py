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


def _full(layer, s_interior, node_dim=-1):
    """Interior-order sources -> FULL node order with zeros on boundary nodes (spec 4.2)."""
    shape = list(s_interior.shape)
    shape[node_dim] = layer.net.n
    full = torch.zeros(shape, dtype=s_interior.dtype)
    return full.index_copy(node_dim % full.dim(), layer.interior_idx, s_interior)


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
        _full(model.transport, torch.zeros_like(x0)),
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

    `model.layer` is the migrated default (`linear_solver="auto"`); `model.dense_layer` is
    the retained milestone-1 reference (`linear_solver="direct"`, the same operator assembled
    and LU-factorised). The spies are what prove the second does not quietly run the first's
    solver -- without them, the parity assertions below would be comparing the sparse path
    with itself.

    UPDATED for the spec section 6.2 step 2 default: `auto` on this certified-SPD
    GraphLaplacianOperator now factorises its O(E) COO form through
    `scipy.sparse.linalg.splu` (1.10x-4.59x faster than PCG at every measured ensemble size;
    see `solvers.select`'s module docstring), so the spy that distinguishes the two paths is
    `splu` against `torch.linalg.lu_factor_ex` -- a sparse kernel against a dense one --
    rather than `select.pcg` against the dense one. `pcg` is now expected on NEITHER layer,
    which is itself asserted so a silent revert of the default would fail here.
    """
    import scipy.sparse.linalg

    model = build_composed()

    splu_calls, lu_calls, pcg_calls = [], [], []
    real_splu = scipy.sparse.linalg.splu
    real_lu = torch.linalg.lu_factor_ex
    real_pcg = select_module.pcg

    def spy_splu(*args, **kwargs):
        splu_calls.append(1)
        return real_splu(*args, **kwargs)

    def spy_lu(*args, **kwargs):
        lu_calls.append(1)
        return real_lu(*args, **kwargs)

    def spy_pcg(*args, **kwargs):
        pcg_calls.append(1)
        return real_pcg(*args, **kwargs)

    monkeypatch.setattr(scipy.sparse.linalg, "splu", spy_splu)
    monkeypatch.setattr(torch.linalg, "lu_factor_ex", spy_lu)
    monkeypatch.setattr(select_module, "pcg", spy_pcg)

    phi_sparse, q_sparse = model.layer.solve(
        model.phi_boundary, model.drivers, model.sources, differentiable=False
    )
    assert splu_calls, "model.layer must solve through the sparse LU"
    assert not lu_calls, "model.layer must not dense-LU-factorise"
    assert not pcg_calls, "model.layer no longer runs pcg on a certified-SPD operator"
    splu_after_sparse = len(splu_calls)

    phi_dense, q_dense = model.dense_layer.solve(
        model.phi_boundary, model.drivers, model.sources, differentiable=False
    )
    assert len(splu_calls) == splu_after_sparse, "model.dense_layer must not call splu"
    assert lu_calls, "model.dense_layer must LU-factorise the assembled operator"
    assert not pcg_calls, "model.dense_layer must not call pcg"

    torch.testing.assert_close(phi_sparse, phi_dense, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(q_sparse, q_dense, rtol=1e-9, atol=1e-12)


def test_build_composed_forwards_linear_solver_to_the_migrated_layer_only():
    """The section 6.2 solver comparison measures the SAME model under two inner solvers, so
    `linear_solver` has to reach `layer` and nothing else: `dense_layer` is the parity
    reference and stays on the dense LU whatever `layer` is configured with.
    """
    model = build_composed(linear_solver="sparse_direct")
    assert model.layer.linear_solver == "sparse_direct"
    assert model.dense_layer.linear_solver == "direct"
    assert build_composed().layer.linear_solver == "auto"


def test_the_report_compares_two_distinct_backends_through_a_solver_keyword():
    """The report's two solver rows must be two DIFFERENT backends, and each workload must
    have a `linear_solver` keyword for `measure_budget_row`'s `solver` to travel through --
    `isolated_peak_rss` rebuilds the model in a fresh process from those kwargs alone, so a
    workload with nowhere to put the solver would silently measure the default twice.

    Named for what it asserts: it checks the two ENDS of that path (distinct backends in
    `SOLVERS`, a `linear_solver` parameter on each workload), not that `measure_budget_row`
    actually forwards `solver` into the child's kwargs, which would need the child process
    itself and is covered by the report's own output carrying two different sets of numbers.
    """
    import inspect

    from benchmarks.composed_model import workload_backward, workload_forward
    from benchmarks.report_composed_scaling import SOLVERS

    assert len(set(SOLVERS)) == 2, "SOLVERS must compare two distinct backends"
    for workload in (workload_forward, workload_backward):
        assert "linear_solver" in inspect.signature(workload).parameters
