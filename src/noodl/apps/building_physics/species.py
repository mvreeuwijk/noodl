"""Contaminant species and first-order kinetic reactions.

A `Species` holds what CONTAM's species record carries (TN 1887 Appendix A section 2). A
`KineticReaction` is CONTAM's kinetic reaction element (section 7, and section 3.12 of
the user guide): first-order reactions between species, in mass fractions, so molar
masses do not enter. `reaction_kinetics` places each zone's reaction matrix into the
species layer's `kinetics`, which the transport step integrates implicitly, as ContamX
does (measured against ContamX 3.4.1.7).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import torch

from noodl.layers.transport import active_interior
from noodl.topology import Network

Tensor = torch.Tensor
F64 = torch.float64


@dataclass(frozen=True)
class Species:
    """One contaminant species: CONTAM's species record.

    `name`; `molar_mass` (kg/kmol); `mean_diameter` (m: 0 for a gas, > 0 for a particle,
    which a `ParticleFilter` acts on); `effective_density` (kg/m3, a particle's);
    `decay` (1/s; kept as data and NOT applied, because ContamX does not apply it
    (measured): model a decay with a `KineticReaction`); `diffusivity` (m2/s);
    `default_mass_fraction` (kg/kg, the ambient value when no ambient-concentration file is
    given); `non_trace` (True for a species that changes the air's density; the CONTAM
    reader refuses one); `uv_susceptibility` (m2/J); `description`.
    """

    name: str
    molar_mass: float = 0.0
    mean_diameter: float = 0.0
    effective_density: float = 0.0
    decay: float = 0.0
    diffusivity: float = 0.0
    default_mass_fraction: float = 0.0
    non_trace: bool = False
    uv_susceptibility: float = 0.0
    description: str = ""

    @property
    def is_particle(self) -> bool:
        """True when the species has a mean diameter (a particle), False for a gas."""
        return self.mean_diameter > 0.0


class KineticReaction:
    """First-order reactions between species: dx_p/dt += k[p, r] x_r, x in kg/kg.

    `coefficients` maps `(product, reactant)` species names to the rate coefficient (1/s):
    positive produces the product, negative destroys it, and a negative `(s, s)` is a
    first-order decay of s. Coefficients may be tensors that carry gradients.
    """

    def __init__(self, coefficients: Mapping[tuple[str, str], float | Tensor], *,
                 name: str = "") -> None:
        self.coefficients = dict(coefficients)
        self.name = name

    def matrix(self, species: Sequence[str]) -> Tensor:
        """`(K, K)` float64, rows the products and columns the reactants, in `species`
        order; zero where no coefficient is given."""
        index = {s: i for i, s in enumerate(species)}
        missing = sorted({s for pair in self.coefficients for s in pair} - set(index))
        if missing:
            raise KeyError(f"KineticReaction {self.name!r}: species {missing} are not among "
                           f"the model's species {list(species)}")
        K = len(species)
        entries = {(index[p], index[r]): torch.as_tensor(k, dtype=F64)
                   for (p, r), k in self.coefficients.items()}
        zero = torch.zeros((), dtype=F64)
        return torch.stack([torch.stack([entries.get((i, j), zero) for j in range(K)])
                            for i in range(K)])


def reaction_kinetics(net: Network, *, species: Sequence[str],
                      reactions: Mapping[Hashable, KineticReaction], ambient="ambient",
                      flow_kinds=("airpath",)) -> Tensor:
    """The species layer's `kinetics`, `(n_i, K, K)`: each node's reaction matrix in the
    layer's active interior order (`active_interior(net, flow_kinds, [ambient])`), zero
    where a node has none. Pass the same `ambient` and `flow_kinds` as to
    `species_layer`. A node that is not an active interior node of that layer is refused
    by name: the layer holds no concentration there to react."""
    interior, _ = active_interior(net, tuple(flow_kinds), [ambient])
    order = interior.tolist()
    K = len(species)
    rows = [torch.zeros(K, K, dtype=F64) for _ in order]
    for node, reaction in reactions.items():
        i = net.node_index(node)
        if i not in order:
            raise KeyError(f"reaction_kinetics: node {node!r} is not an active interior node "
                           f"of the species layer (flow kinds {tuple(flow_kinds)}), so it has "
                           f"no concentration to react")
        j = order.index(i)
        rows[j] = rows[j] + reaction.matrix(species)
    return torch.stack(rows) if rows else torch.zeros(0, K, K, dtype=F64)
