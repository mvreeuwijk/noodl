"""Nodal-primary layers: Kirchhoff solves, multi-species/heat transport, reactions,
and explicit clip/allocate transfer on capacity-limited arcs."""

from .allocation import AllocatedFlowLayer
from .kirchhoff import KirchhoffLayer
from .reaction import Reaction
from .transport import TransportLayer, UndefinedSteadyState, active_interior

__all__ = [
    "AllocatedFlowLayer",
    "KirchhoffLayer",
    "Reaction",
    "TransportLayer",
    "UndefinedSteadyState",
    "active_interior",
]
