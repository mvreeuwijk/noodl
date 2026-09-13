"""The composed reference model: 8 building submodels joined through street and sewer networks,
with a reference physics configuration on top.

Used by the composed-model scaling gate (Task 14) and, as a `pytest` fixture
(`tests/conftest.py`'s `composed_model`), by the migration tasks that need a realistic joined
topology and physics (Tasks 9-14). `_build_topology` builds the topology ONLY -- no elements,
drives or layers; `build_composed` then attaches one reference physics configuration to it (a
`PowerLaw` per edge kind, sources, capacity, a migrated `PotentialFlowLayer`, a dense-path
`PotentialFlowLayer` and a `TransportLayer`) and returns both as one `ComposedModel`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from benchmarks.measure import time_call
from tellegen.elements import PowerLaw
from tellegen.elements.base import Element
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.layers.transport import TransportLayer
from tellegen.topology import Network

# PowerLaw conductance scale per edge kind, and the insertion order the elements list and the
# shared RNG draws follow (topology first, then these in this order, then sources, then
# capacity -- see build_composed).
_ELEMENT_SCALES: dict[str, float] = {
    "airpath": 1e-2,
    "street": 5e-1,
    "sewer": 2e-1,
    "street_interface": 1e-2,
    "sewer_interface": 1e-3,
}


@dataclass
class ComposedModel:
    """A composed reference topology plus one reference physics configuration on it."""

    net: Network
    interface_nodes: dict[str, dict[str, tuple[str, str]]]
    boundary: list[str]
    elements: list[Element]
    phi_boundary: torch.Tensor
    drivers: dict
    sources: torch.Tensor
    capacity: torch.Tensor
    layer: PotentialFlowLayer
    dense_layer: PotentialFlowLayer  # linear_solver="direct": the retained dense reference
    transport: TransportLayer
    ensemble: int
    seed: int


def _build_topology(
    n_buildings: int,
    building_nodes: int,
    street_nodes: int,
    sewer_nodes: int,
    seed: int,
) -> tuple[Network, dict[str, dict[str, tuple[str, str]]]]:
    """Build one `Network` of `n_buildings` buildings joined through a street and a sewer.

    Each building is a random connected multigraph of `building_nodes` nodes on kind
    "airpath" (a random spanning tree plus extra random edges, in the style of
    `tests/test_topology.py`'s `_random_connected_multigraph`); its node 0 is designated the
    building's "ambient" interface node and its node 1 the "manhole" interface node. The
    street network (kind "street") and the sewer network (kind "sewer") are built the same
    way. Building `i`'s ambient node is joined to `street_names[i % street_nodes]` by one
    "street_interface" edge; building `i`'s manhole node is joined to
    `sewer_names[i % sewer_nodes]` by one "sewer_interface" edge -- so every building touches
    the street and sewer networks even when there are more buildings than street or sewer
    nodes.

    Returns `(net, interface_nodes)`, where
    `interface_nodes == {"street": {"building_0": (ambient_node, street_node), ...},
                         "sewer": {"building_0": (manhole_node, sewer_node), ...}}`,
    both inner dicts keyed `f"building_{i}"` for `i in range(n_buildings)`.

    With the defaults this is `8 * 120 + 40 + 30 == 1030` nodes (exact, seed-independent) and,
    measured over `seed in range(5)`, 2186-2196 edges (about 2200): the reference composed
    model of the milestone design's section 6.
    """
    rng = torch.Generator().manual_seed(seed)
    net = Network(dtype=torch.float64)

    def _random_tree_plus_extra(prefix: str, n_local: int, extra: int, kind: str) -> list[str]:
        names = [f"{prefix}{j}" for j in range(n_local)]
        for name in names:
            net.add_node(name)
        for j in range(1, n_local):
            k = int(torch.randint(0, j, (1,), generator=rng))
            net.add_edge(names[j], names[k], kind=kind)
        for _ in range(extra):
            u = int(torch.randint(0, n_local, (1,), generator=rng))
            v = int(torch.randint(0, n_local, (1,), generator=rng))
            if u != v:
                net.add_edge(names[u], names[v], kind=kind)
        return names

    street_names = _random_tree_plus_extra("street_", street_nodes, street_nodes, "street")
    sewer_names = _random_tree_plus_extra("sewer_", sewer_nodes, sewer_nodes, "sewer")

    building_extra = int(round(building_nodes * 1.14))
    street_interfaces: dict[str, tuple[str, str]] = {}
    sewer_interfaces: dict[str, tuple[str, str]] = {}
    for i in range(n_buildings):
        names = _random_tree_plus_extra(f"b{i}_", building_nodes, building_extra, "airpath")
        ambient_node, manhole_node = names[0], names[1]
        street_node = street_names[i % len(street_names)]
        sewer_node = sewer_names[i % len(sewer_names)]
        net.add_edge(ambient_node, street_node, kind="street_interface")
        net.add_edge(manhole_node, sewer_node, kind="sewer_interface")
        street_interfaces[f"building_{i}"] = (ambient_node, street_node)
        sewer_interfaces[f"building_{i}"] = (manhole_node, sewer_node)

    interface_nodes = {"street": street_interfaces, "sewer": sewer_interfaces}
    return net, interface_nodes


def build_composed(
    n_buildings: int = 8,
    building_nodes: int = 120,
    street_nodes: int = 40,
    sewer_nodes: int = 30,
    ensemble: int = 1,
    seed: int = 0,
    linear_solver: str = "auto",
) -> ComposedModel:
    """Build the reference composed model: topology plus one reference physics configuration.

    The physics is drawn from one `torch.Generator().manual_seed(seed)`, AFTER the topology is
    built (so the graph itself is exactly `_build_topology`'s, unaffected by how many random
    draws the physics below makes), in this order: one `PowerLaw(C, 0.65, kind=kind)` per edge
    kind, in `_ELEMENT_SCALES` order, with `C = scale * (0.5 + U(b_kind))`; then `sources`; then
    `capacity`. `C` is shared across the ensemble; only `sources` varies per instance.

    `boundary` is `["street_0", "sewer_0"]` -- one ground node each on the street and sewer
    networks. `layer` (name "composed") is the migrated, default-configured path; `dense_layer`
    (name "composed_dense") is built from the same net/elements/boundary but with
    `linear_solver="direct"`: the same operator assembled and LU-factorised, i.e. the
    milestone-1 numerics, retained as a genuinely separate code path so the composed-model
    parity gate compares two solvers rather than the sparse path with itself. `transport`
    (name "co2") is a single-species implicit-scheme `TransportLayer` on the "airpath" edges.

    `linear_solver` configures `layer` ONLY, and defaults to the layer's own default, so
    every existing caller gets exactly the model it got before. It exists for the spec's
    section 6.2 solver comparison, which measures the SAME model under `"auto"` and under
    `"sparse_direct"` and needs the two to differ in nothing else. `dense_layer` is always
    `"direct"`: it is the parity reference, and a reference that moved with the thing it
    references would not be one.
    """
    net, interface_nodes = _build_topology(
        n_buildings, building_nodes, street_nodes, sewer_nodes, seed
    )
    boundary = ["street_0", "sewer_0"]

    rng = torch.Generator().manual_seed(seed)
    dtype = net.dtype

    elements: list[Element] = []
    for kind, scale in _ELEMENT_SCALES.items():
        b_kind = int(net.edge_index(kind).numel())
        u = torch.rand(b_kind, generator=rng, dtype=dtype)
        C = scale * (0.5 + u)
        elements.append(PowerLaw(C, 0.65, kind=kind))

    phi_boundary = torch.zeros(ensemble, len(boundary), dtype=dtype)

    u_sources = torch.rand(ensemble, net.n, generator=rng, dtype=dtype)
    sources = 1e-3 * (u_sources - 0.5)
    sources[..., net.boundary_index(boundary)] = 0.0

    n_interior = net.n - len(boundary)
    u_capacity = torch.rand(n_interior, generator=rng, dtype=dtype)
    capacity = 50.0 + 100.0 * u_capacity

    drivers: dict = {}

    layer = PotentialFlowLayer(
        net, "composed", elements, boundary=boundary, linear_solver=linear_solver
    )
    dense_layer = PotentialFlowLayer(
        net, "composed_dense", elements, boundary=boundary, linear_solver="direct"
    )
    transport = TransportLayer(
        net,
        "co2",
        capacity=capacity,
        flow_kind="airpath",
        boundary=boundary,
        n_species=1,
        scheme="implicit",
    )

    return ComposedModel(
        net=net,
        interface_nodes=interface_nodes,
        boundary=boundary,
        elements=elements,
        phi_boundary=phi_boundary,
        drivers=drivers,
        sources=sources,
        capacity=capacity,
        layer=layer,
        dense_layer=dense_layer,
        transport=transport,
        ensemble=ensemble,
        seed=seed,
    )


# --- Workloads for `benchmarks.measure.isolated_peak_rss` -------------------------------
#
# Each is a NAMED module-level function taking only JSON-serialisable arguments and
# returning a JSON-serialisable result, because `isolated_peak_rss` runs it in a fresh
# child process (`python -c "... import benchmarks.composed_model ..."`) and reads its
# result back off stdout. The child rebuilding the model from scratch is the point: the
# measured peak is then the whole cost of the configuration -- topology, layer caches,
# solve and transport step -- with no state inherited from the measuring process.


def workload_alloc(mb: int) -> int:
    """Allocate and touch an `mb`-MiB float64 tensor; return `mb`.

    The calibration workload for `isolated_peak_rss` itself: its only allocation is a torch
    tensor, which `tracemalloc` reports as zero bytes, so a measurement that sees it is
    demonstrably seeing PyTorch's allocator rather than Python bookkeeping.
    """
    block = torch.empty(int(mb) * 2**20 // 8, dtype=torch.float64)
    block.fill_(1.0)
    if not float(block.sum()) > 0.0:  # touch every page so the OS commits them
        raise RuntimeError("workload_alloc: allocation was not touched")
    return int(mb)


def _step_state(model: ComposedModel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(x0, x_boundary, zero_sources)` for the transport half of a step, at 400 ppm."""
    dtype = model.net.dtype
    n_i = int(model.layer.interior.numel())
    x0 = torch.full((model.ensemble, n_i), 400.0, dtype=dtype)
    x_boundary = torch.full((model.ensemble, len(model.boundary)), 400.0, dtype=dtype)
    zero_sources = torch.zeros(model.ensemble, n_i, dtype=dtype)
    return x0, x_boundary, zero_sources


def run_steps(
    model: ComposedModel,
    steps: int,
    *,
    differentiable: bool,
    sources: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Advance `steps` coupled steps; return `(phi, x, diagnostics_of_the_last_solve)`.

    ONE STEP, as the milestone's section 6.1 budget table means it: one
    `layer.solve(phi_boundary, drivers, sources)` followed by one `transport.step` on the
    AIRPATH slice of the resulting `q` (implicit scheme, dt = 60 s), with `x` fed forward
    from one step to the next so a multi-step row is a genuine trajectory rather than the
    same step repeated.
    """
    lo, hi = model.layer._kind_slices["airpath"]
    if sources is None:
        sources = model.sources
    x, x_boundary, zero_sources = _step_state(model)
    diagnostics: dict = {}
    phi = None
    for _ in range(steps):
        phi, q = model.layer.solve(
            model.phi_boundary,
            model.drivers,
            sources,
            differentiable=differentiable,
            diagnostics=diagnostics,
        )
        x = model.transport.step(x, q[..., lo:hi], zero_sources, x_boundary, dt=60.0)
    return phi, x, diagnostics


def _iteration_counts(diagnostics: dict) -> dict:
    """The JSON-serialisable half of a `solve` diagnostics dict.

    `linear_iterations` is per instance; the MAX over the ensemble is what a conditioning
    regression shows up in first, so that is what is reported.
    """

    def _as_max_int(value) -> int | None:
        if value is None:
            return None
        return int(value.amax()) if hasattr(value, "amax") else int(value)

    return {
        "newton_iterations": _as_max_int(diagnostics.get("newton_iterations")),
        "linear_iterations_max": _as_max_int(diagnostics.get("linear_iterations")),
        "method": diagnostics.get("method"),
    }


def workload_forward(
    n_buildings: int = 8,
    building_nodes: int = 120,
    street_nodes: int = 40,
    sewer_nodes: int = 30,
    ensemble: int = 1,
    steps: int = 1,
    linear_solver: str = "auto",
) -> dict:
    """Build the composed model and run `steps` NON-differentiable steps; report the run.

    The build and one warm-up step happen before the timed section, so the reported time is
    the steady, already-warm cost the budget table is about (topology, cycle basis and
    tree-elimination levels are all construction- or first-solve-time caches). Both still
    happen inside the process whose peak RSS `isolated_peak_rss` measures, which is correct:
    the memory budget is for the whole configuration, caches included.
    """
    model = build_composed(
        n_buildings=n_buildings,
        building_nodes=building_nodes,
        street_nodes=street_nodes,
        sewer_nodes=sewer_nodes,
        ensemble=ensemble,
        linear_solver=linear_solver,
    )
    run_steps(model, 1, differentiable=False)
    elapsed, (_phi, x, diagnostics) = time_call(
        lambda: run_steps(model, steps, differentiable=False)
    )
    return {
        "elapsed_s": elapsed,
        "steps": steps,
        "ensemble": ensemble,
        "linear_solver": linear_solver,
        "n_nodes": int(model.net.n),
        "n_edges": int(model.net.b),
        "x_final_mean": float(x.mean()),
        **_iteration_counts(diagnostics),
    }


def workload_backward(
    n_buildings: int = 8,
    building_nodes: int = 120,
    street_nodes: int = 40,
    sewer_nodes: int = 30,
    ensemble: int = 1,
    steps: int = 1,
    linear_solver: str = "auto",
) -> dict:
    """Build the composed model, run `steps` DIFFERENTIABLE steps, and time `loss.backward()`.

    `loss = phi[..., interior].sum() + x_final.sum()` differentiated with respect to
    `sources` exercises both adjoints the milestone migrated: the potential layer's implicit
    adjoint (through every solve) and the transport layer's `_LinearSolve` adjoint (through
    every implicit step). Only the backward pass is timed; the forward that builds the graph
    is not, because the forward budget already covers it.
    """
    model = build_composed(
        n_buildings=n_buildings,
        building_nodes=building_nodes,
        street_nodes=street_nodes,
        sewer_nodes=sewer_nodes,
        ensemble=ensemble,
        linear_solver=linear_solver,
    )
    run_steps(model, 1, differentiable=False)  # warm the construction-time caches
    sources = model.sources.detach().clone().requires_grad_(True)
    phi, x, diagnostics = run_steps(model, steps, differentiable=True, sources=sources)
    loss = phi[..., model.layer.interior].sum() + x.sum()
    elapsed, _ = time_call(loss.backward)
    return {
        "elapsed_s": elapsed,
        "steps": steps,
        "ensemble": ensemble,
        "linear_solver": linear_solver,
        "grad_sources_absmax": float(sources.grad.abs().max()),
        **_iteration_counts(diagnostics),
    }
