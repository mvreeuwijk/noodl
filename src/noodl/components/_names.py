"""`NameMap`: the flattened network, queried by component path."""

from __future__ import annotations

import difflib
from collections.abc import Hashable
from dataclasses import dataclass, field


@dataclass
class ComponentInfo:
    name: str
    template: str | None
    position: tuple[float, float, float]
    ports: dict[str, str | None]           # port -> node it became (None: dropped)
    children: list[str] = field(default_factory=list)


class NameMap:
    """What `Component.flatten` made of the tree: which component owns each node and edge,
    every component's template, placement and ports, and the ports left unconnected."""

    def __init__(self, net, *, node_owner, edge_owner, components, unconnected_ports) -> None:
        self.net = net
        self._node_owner: dict[str, str] = dict(node_owner)
        self._edge_owner: dict[str, str] = dict(edge_owner)
        self._components: dict[str, ComponentInfo] = dict(components)
        self.unconnected_ports: list[str] = list(unconnected_ports)

    @property
    def paths(self) -> list[str]:
        """Every component path in tree order; the root is `""`."""
        return list(self._components)

    def resolve(self, label: Hashable) -> str:
        """The node name `label` refers to (itself, or the node an alias was merged into)."""
        canonical = self.net.aliases.get(label, label)
        if canonical not in self._node_owner:
            known = [*self._node_owner, *self.net.aliases]
            hint = difflib.get_close_matches(str(label), known, 3, 0.6)
            suffix = f"; did you mean {', '.join(map(repr, hint))}?" if hint else ""
            raise KeyError(f"unknown node {label!r}{suffix}")
        return canonical
