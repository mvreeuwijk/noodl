"""Components: subnetworks with ports that nest, connect by merging ports, and flatten to
one `Network`. See docs/concepts/components.md."""

from noodl.components import _registry
from noodl.components._component import Component, ComponentError, OuterRef, PortRef
from noodl.components._registry import register_elevations, register_table_lookup

__all__ = [
    "Component",
    "ComponentError",
    "OuterRef",
    "PortRef",
    "_registry",
    "register_elevations",
    "register_table_lookup",
]
