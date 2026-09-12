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
    dense_layer: PotentialFlowLayer
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
) -> ComposedModel:
    """Build the reference composed model: topology plus one reference physics configuration.

    The physics is drawn from one `torch.Generator().manual_seed(seed)`, AFTER the topology is
    built (so the graph itself is exactly `_build_topology`'s, unaffected by how many random
    draws the physics below makes), in this order: one `PowerLaw(C, 0.65, kind=kind)` per edge
    kind, in `_ELEMENT_SCALES` order, with `C = scale * (0.5 + U(b_kind))`; then `sources`; then
    `capacity`. `C` is shared across the ensemble; only `sources` varies per instance.

    `boundary` is `["street_0", "sewer_0"]` -- one ground node each on the street and sewer
    networks. `layer` (name "composed") is the migrated, default-configured path; `dense_layer`
    (name "composed_dense") is built from the same net/elements/boundary and is identical to
    `layer` until a later task gives it a dense-only `linear_solver`. `transport` (name "co2")
    is a single-species implicit-scheme `TransportLayer` on the "airpath" edges.
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

    layer = PotentialFlowLayer(net, "composed", elements, boundary=boundary)
    dense_layer = PotentialFlowLayer(net, "composed_dense", elements, boundary=boundary)
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
