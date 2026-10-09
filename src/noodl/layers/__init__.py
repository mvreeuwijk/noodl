"""Nodal-primary layers: potential-flow solves, multi-species/heat transport, reactions,
and explicit clip/allocate transfer on capacity-limited arcs. `ConstitutiveLayer` is the one
loop-formulation layer: general branch laws over a small network, not a nodal balance."""

from .allocation import AllocatedFlowLayer, CapacitatedTransferLayer  # alias, pre-rename name
from .constitutive import ConstitutiveLayer
from .potential import PotentialFlowLayer
from .reaction import FirstOrderDecay, Reaction
from .transport import TransportLayer, active_interior

__all__ = [
    "AllocatedFlowLayer",
    "CapacitatedTransferLayer",
    "ConstitutiveLayer",
    "FirstOrderDecay",
    "PotentialFlowLayer",
    "Reaction",
    "TransportLayer",
    "active_interior",
]
