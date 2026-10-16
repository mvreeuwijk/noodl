"""Nodal-primary layers: Kirchhoff solves, multi-species/heat transport, reactions,
and explicit clip/allocate transfer on capacity-limited arcs."""

from .allocation import AllocatedFlowLayer, CapacitatedTransferLayer  # alias, pre-rename name
from .kirchhoff import KirchhoffLayer
from .reaction import FirstOrderDecay, Reaction
from .transport import TransportLayer, UndefinedSteadyState, active_interior

__all__ = [
    "AllocatedFlowLayer",
    "CapacitatedTransferLayer",
    "FirstOrderDecay",
    "KirchhoffLayer",
    "Reaction",
    "TransportLayer",
    "UndefinedSteadyState",
    "active_interior",
]
