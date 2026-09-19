"""The Milestone 1b acceptance gate: composed-model correctness and scaling (design section 6).

Fast tests (no marker) run in CI by default; `@pytest.mark.slow` tests are the full section 6.1
budget table and the two shape gates, deselected by the existing `-m "not slow"` addopts.

Memory here is measured with `benchmarks.measure.isolated_peak_rss`, never `tracemalloc`:
`tracemalloc` does not see PyTorch's own allocator at all (amendment A4), so a
`tracemalloc`-based memory gate would measure Python bookkeeping overhead and pass vacuously.
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.composed_model import build_composed, run_steps
from benchmarks.measure import isolated_peak_rss, time_call
from benchmarks.report_composed_scaling import (
    BUDGET_TABLE,
    format_budget_row,
    format_shape_gate,
    measure_budget_row,
    measure_matvec_shape_gate,
    measure_memory_shape_gate,
)
from noodl.elements import PowerLaw
from noodl.layers.potential import PotentialFlowLayer


def test_time_call_returns_a_nonnegative_elapsed_seconds_and_the_callables_result():
    elapsed, value = time_call(lambda: 2 + 2)
    assert elapsed >= 0.0
    assert value == 4


def test_isolated_peak_rss_sees_pytorch_allocations_a_200_mib_tensor_makes():
    # Non-vacuous by construction: the workload's only allocation is a torch tensor, which
    # `tracemalloc` reports as 0 bytes (amendment A4). A subprocess RSS measurement must see
    # it. 150 MiB, not 200, leaves room for the allocator returning pages in large blocks.
    peak_bytes, result = isolated_peak_rss(
        "benchmarks.composed_model", "workload_alloc", {"mb": 200}
    )
    assert result == 200
    assert peak_bytes >= 150 * 2**20


def test_composed_model_parity_with_dense_reference_at_ensemble_one_and_four():
    """The migrated default path and the retained direct path must agree numerically.

    Two genuinely different solvers (`tests/verification/test_composed_model.py`'s spy test is
    what proves that); this asserts they land on the same answer, in both `phi` and `q`, at
    ensemble 1 and again at ensemble 4 so a batching bug in either cannot hide behind a
    single-instance comparison.
    """
    for ensemble in (1, 4):
        model = build_composed(ensemble=ensemble)
        phi_sparse, q_sparse = model.layer.solve(
            model.phi_boundary, model.drivers, model.sources, differentiable=False
        )
        phi_dense, q_dense = model.dense_layer.solve(
            model.phi_boundary, model.drivers, model.sources, differentiable=False
        )
        torch.testing.assert_close(phi_sparse, phi_dense, rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(q_sparse, q_dense, rtol=1e-9, atol=1e-12)


def test_composed_model_interface_conservation_at_every_shared_node():
    """Net flux minus source vanishes at every interior node a submodel join touches.

    The imbalance is formed with `model.layer.A`, whose columns are in LAYER edge order --
    the same order `q` comes back in. `net.incidence()` is in NETWORK edge order, and using
    it here reports a spurious ~7e-3 imbalance from nothing but the column permutation.

    Tolerance is relative to the largest flow on the edges INCIDENT to the node, since that
    is the scale the cancellation happens at; the two boundary nodes (`street_0`, `sewer_0`)
    are excluded because they absorb whatever flux the rest of the network leaves over.

    Solved at `atol=rtol=1e-13` (as the parity gate is): the imbalance at a node IS the
    Newton residual there, so a gate at 1e-12 RELATIVE to a ~1e-3 flow is a statement about
    a converged solve, not about Newton's default stopping tolerance. At the default
    tolerance the worst interior residual is 8e-11 absolute; at 1e-13 it is 2e-17, five
    orders of magnitude inside this gate.
    """
    model = build_composed(ensemble=1)
    phi, q = model.layer.solve(
        model.phi_boundary, model.drivers, model.sources, atol=1e-13, rtol=1e-13
    )
    net_flux = torch.einsum("ne,...e->...n", model.layer.A, q)
    interior_nodes = set(model.layer.interior.tolist())

    checked = 0
    worst = 0.0
    for join in ("street", "sewer"):
        for building, (local_node, shared_node) in model.interface_nodes[join].items():
            for name in (local_node, shared_node):
                node = model.net.node_index(name)
                if node not in interior_nodes:
                    continue
                incident = torch.nonzero(model.layer.A[node], as_tuple=True)[0]
                scale = float(q[..., incident].abs().amax())
                residual = float((net_flux[..., node] - model.sources[..., node]).abs().amax())
                assert residual <= 1e-12 * scale, (
                    f"{join} interface of {building}: node {name!r} carries an imbalance of "
                    f"{residual:.3e}, above 1e-12 * {scale:.3e}"
                )
                worst = max(worst, residual / scale)
                checked += 1
    print(f"\nworst relative interface imbalance: {worst:.3e} (gate <= 1e-12)")
    # 8 buildings x 2 joins x 2 endpoints, less any endpoint that is a boundary node.
    assert checked >= 30, f"only {checked} interface nodes were actually checked"


def test_gradient_across_a_join_matches_central_finite_differences():
    """A parameter in the STREET submodel (the conductance C of every street edge) must
    receive a correct gradient from a loss inside a BUILDING (the potential at its manhole
    node). Small configuration so the central-difference loop is cheap.

    Amendment A3.5's gate, at BUILDING 1 rather than the amendment's building 0. Building 0
    was a bug in the amendment: `build_composed` wires building `i`'s ambient node to
    `street_names[i % street_nodes]` and its manhole to `sewer_names[i % sewer_nodes]`, and
    the boundary is `["street_0", "sewer_0"]` -- so building 0, alone among the buildings,
    attaches directly to BOTH boundary nodes. Its submodel is enclosed between two fixed
    potentials and depends on nothing outside itself, so no street conductance can move its
    manhole: autograd returned exactly 0.0 for all ten street conductances and central
    differences returned one ULP of noise (3.5e-12), and the gate asserted 0 == 0. It would
    have passed with the cross-join adjoint entirely broken.

    Building 1's ambient attaches to `street_1` and its manhole to `sewer_1`, both interior,
    so the loss genuinely depends on every street conductance through a path that leaves the
    building submodel, crosses the street join, traverses the street network and returns
    through the sewer join. The explicit non-triviality assertion below is what stops this
    gate degenerating the same way again.
    """
    model = build_composed(
        n_buildings=2, building_nodes=12, street_nodes=6, sewer_nodes=5, ensemble=1, seed=0
    )
    net = model.net
    loss_node = net.node_index(model.interface_nodes["sewer"]["building_1"][0])  # manhole
    street = next(el for el in model.elements if el.kind == "street")
    C0 = street.C.detach().clone()

    def layer_with_street_C(C, *, learnable):
        elements = [
            PowerLaw(C, 0.65, kind="street", learnable=learnable) if el.kind == "street" else el
            for el in model.elements
        ]
        return PotentialFlowLayer(net, "grad_join", elements, boundary=model.boundary), elements

    def loss_at(C):
        layer, _ = layer_with_street_C(C, learnable=False)
        phi, _ = layer.solve(
            model.phi_boundary,
            model.drivers,
            model.sources,
            differentiable=False,
            atol=1e-13,
            rtol=1e-13,
        )
        return float(phi[0, loss_node])

    layer, elements = layer_with_street_C(C0.clone(), learnable=True)
    el = next(e for e in elements if e.kind == "street")
    phi, _ = layer.solve(
        model.phi_boundary,
        model.drivers,
        model.sources,
        differentiable=True,
        atol=1e-13,
        rtol=1e-13,
    )
    phi[0, loss_node].backward()
    grad_ad = el.C.grad.detach().clone()

    h = 1e-6
    grad_fd = torch.zeros_like(C0)
    for i in range(C0.numel()):
        bump = torch.zeros_like(C0)
        bump[i] = h
        grad_fd[i] = (loss_at(C0 + bump) - loss_at(C0 - bump)) / (2 * h)

    assert float(grad_fd.abs().max()) > 1e-9, (
        "cross-join gradient gate is vacuous here too: the finite-difference gradient is "
        f"{float(grad_fd.abs().max()):.3e}, indistinguishable from roundoff"
    )
    deviation = (grad_ad - grad_fd).abs()
    print(
        f"\ncross-join gradient (building 1): |grad| up to "
        f"{float(grad_fd.abs().max()):.3e}, max abs deviation {float(deviation.max()):.3e} "
        f"(gate rtol=1e-6 atol=1e-8)"
    )
    torch.testing.assert_close(grad_ad, grad_fd, rtol=1e-6, atol=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("ensemble", "steps", "forward_budget", "backward_budget", "memory_budget"), BUDGET_TABLE
)
def test_composed_model_meets_its_section_6_1_budget(
    ensemble, steps, forward_budget, backward_budget, memory_budget
):
    """One row of the design's section 6.1 budget table, on the reference composed model.

    A STEP is one `layer.solve` plus one implicit `transport.step` at dt=60 s on the airpath
    slice of the resulting `q`, with `x` fed forward; the backward figure is the backward of
    `phi.sum() + x_final.sum()` with respect to `sources`, so it exercises the potential
    adjoint AND the transport `_LinearSolve` adjoint.

    Every number is measured in a FRESH child process (`isolated_peak_rss`), which is what
    makes the memory figure a real peak rather than a counter this process has already
    polluted; the time comes back from the same run, so time and memory describe one
    execution rather than two. Iteration counts are printed beside the times because a
    conditioning regression is invisible in wall clock when threading masks it.

    A missed budget is a FAILED gate. Do not loosen the budgets here; design section 6.2 is
    the table of follow-ups a failure triggers.
    """
    row = measure_budget_row(ensemble, steps, forward_budget, backward_budget, memory_budget)
    print("\n" + format_budget_row(row))

    failures = []
    if not row["forward_within_budget"]:
        failures.append(
            f"forward budget missed: {row['forward_seconds']:.3f}s > {forward_budget}s "
            f"({row['forward_seconds'] / forward_budget:.2f}x)"
        )
    if not row["backward_within_budget"]:
        failures.append(
            f"backward budget missed: {row['backward_seconds']:.3f}s > {backward_budget}s "
            f"({row['backward_seconds'] / backward_budget:.2f}x)"
        )
    if not row["peak_memory_within_budget"]:
        failures.append(
            f"peak memory budget missed: {row['peak_memory_bytes'] / 1e6:.1f} MB > "
            f"{memory_budget / 1e6:.0f} MB "
            f"({row['peak_memory_bytes'] / memory_budget:.2f}x)"
        )
    # One report per row covering all three budgets, not three short-circuiting asserts: a
    # row that misses its forward budget would otherwise say nothing about whether its
    # backward and memory budgets hold, and a gate that hides two thirds of its own result
    # behind the first failure is not reporting the gate.
    if failures:
        pytest.fail("\n".join(failures))


def test_thermal_composed_configuration_steps_and_differentiates_on_a_small_model():
    """The milestone-2 gate row's configuration, at a size that runs in CI in a second.

    The gate itself is `slow` and takes tens of minutes, so nothing else here would notice a
    typo in `build_composed(thermal=True)` or in `run_steps(..., thermal=True)` until that
    run failed. This exercises both: the thermal layer is sized on the ACTIVE interior (the
    same rows the co2 layer has -- street and sewer nodes carry no airpath edge), a step
    advances both transport states, and a loss over the concatenated state reaches `sources`
    through both of them.
    """
    model = build_composed(
        n_buildings=2, building_nodes=12, street_nodes=6, sewer_nodes=5, ensemble=2,
        thermal=True,
    )
    assert model.model is not None and model.thermal is not None
    assert model.thermal.n_i == model.transport.n_i < model.net.n - len(model.boundary)
    assert model.thermal.capacity.shape == (model.thermal.n_i,)

    phi, x, diagnostics = run_steps(model, 2, differentiable=False, thermal=True)
    assert phi.shape == (2, model.net.n)
    assert x.shape == (2, model.transport.n_i + model.thermal.n_i)
    assert diagnostics["newton_iterations"] > 0     # not warm-started into a no-op solve

    sources = model.sources.detach().clone().requires_grad_(True)
    phi, x, _ = run_steps(model, 2, differentiable=True, sources=sources, thermal=True)
    (phi[..., model.layer.interior].sum() + x.sum()).backward()
    assert float(sources.grad.abs().max()) > 0.0


@pytest.mark.slow
def test_composed_model_with_thermal_layer_24_steps_within_budget():
    """The milestone-2 row: ensemble 100, 24 steps, air + species + HEAT through `Model.step`.

    `build_composed(thermal=True)` adds a second `TransportLayer` (carrier c_p, capacity in
    J/K, implicit) over the same airpath flows and runs air, species and heat together
    through `Model.step`; the backward figure differentiates a loss over BOTH transport
    states, so both transport adjoints are exercised, not just the species one.

    It is judged against the SAME section 6.1 budgets as the 100x24 row (12 s forward, 25 s
    backward, that row's memory budget), which is what makes the comparison mean anything:
    the configuration differs from that row in the added layer and the `Model.step` dispatch
    around it (the closure loop, the driver lookups, `flows_of_kind` and the per-pass dict
    copies), and in nothing else. A miss is a FAILED gate, recorded in the README and the
    ledger; the budgets are never edited here.
    """
    ensemble, steps, forward_budget, backward_budget, memory_budget = next(
        r for r in BUDGET_TABLE if r[0] == 100 and r[1] == 24
    )
    row = measure_budget_row(
        ensemble, steps, forward_budget, backward_budget, memory_budget, thermal=True
    )
    print("\n" + format_budget_row(row))

    # One report covering all three budgets, as the parametrised row above does and for the
    # same reason: a gate that hides two thirds of its own result behind the first failure is
    # not reporting the gate.
    failures = []
    if not row["forward_within_budget"]:
        failures.append(
            f"forward budget missed: {row['forward_seconds']:.3f}s > {forward_budget}s "
            f"({row['forward_seconds'] / forward_budget:.2f}x)"
        )
    if not row["backward_within_budget"]:
        failures.append(
            f"backward budget missed: {row['backward_seconds']:.3f}s > {backward_budget}s "
            f"({row['backward_seconds'] / backward_budget:.2f}x)"
        )
    if not row["peak_memory_within_budget"]:
        failures.append(
            f"peak memory budget missed: {row['peak_memory_bytes'] / 1e6:.1f} MB > "
            f"{memory_budget / 1e6:.0f} MB "
            f"({row['peak_memory_bytes'] / memory_budget:.2f}x)"
        )
    if failures:
        pytest.fail("\n".join(failures))


@pytest.mark.slow
def test_doubling_nodes_at_fixed_edge_ratio_raises_peak_memory_by_at_most_2_5x():
    """Shape gate 1: peak RSS must grow about linearly in problem size, not quadratically.

    1030 nodes (`build_composed()`'s defaults) against 2060 (`building_nodes=240`,
    `street_nodes=80`, `sewer_nodes=60`), at fixed ensemble 1 and one step. The builder
    derives each submodel's extra-edge count from its node count, so doubling the node
    counts holds the edge-to-node ratio fixed and the comparison is of shape alone. A dense
    (n_interior x n_interior) Jacobian would give 4x here.
    """
    gate = measure_memory_shape_gate()
    print("\n" + format_shape_gate(gate))
    assert gate["ratio"] <= 2.5, (
        f"peak memory grew {gate['ratio']:.2f}x when nodes doubled "
        f"({gate['small_nodes']} -> {gate['large_nodes']} nodes; budget 2.5x)"
    )


@pytest.mark.slow
def test_doubling_edges_at_fixed_nodes_raises_matvec_time_by_at_most_2_5x():
    """Shape gate 2: `GraphLaplacianOperator.matvec` must be linear in edge count.

    `build_composed` cannot vary edge count at fixed node count -- it derives every
    submodel's extra-edge count from its node count -- so the doubled-edge operator is
    built directly from the reference layer's own endpoint tensors plus an equal number of
    freshly drawn random node pairs, exactly the "random tree plus extra random edges"
    recipe the builder itself uses, at the SAME node count, interior map and boundary mask.
    `matvec` is a gather/scatter over edges, so that is precisely the quantity under test.
    """
    gate = measure_matvec_shape_gate()
    print("\n" + format_shape_gate(gate))
    assert gate["ratio"] <= 2.5, (
        f"matvec time grew {gate['ratio']:.2f}x when edges doubled "
        f"({gate['small_edges']} -> {gate['large_edges']} edges; budget 2.5x)"
    )


def test_report_script_is_importable_and_has_a_main():
    import benchmarks.report_composed_scaling as report_mod

    assert hasattr(report_mod, "main")
