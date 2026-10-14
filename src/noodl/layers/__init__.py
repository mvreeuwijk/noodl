"""Nodal-primary layers: potential-flow solves, multi-species/heat transport, reactions,
and explicit clip/allocate transfer on capacity-limited arcs."""

from .allocation import AllocatedFlowLayer, CapacitatedTransferLayer  # alias, pre-rename name
from .potential import PotentialFlowLayer
from .reaction import FirstOrderDecay, Reaction
from .transport import TransportLayer, active_interior

__all__ = [
    "AllocatedFlowLayer",
    "CapacitatedTransferLayer",
    "FirstOrderDecay",
    "PotentialFlowLayer",
    "Reaction",
    "TransportLayer",
    "active_interior",
]
