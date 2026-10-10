"""`NameMap`: the flattened network, queried by component path."""

from __future__ import annotations

import difflib
from collections.abc import Hashable
from dataclasses import dataclass, field

import torch


@dataclass
class Boundary:
    """The edges of one kind crossing a subtree's boundary: their labels, their positions
    in `net.edge_index(kind)` order, and +1 for an edge pointing into the subtree."""

    labels: list[str]
    positions: torch.Tensor
    sign: torch.Tensor


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

    def _check(self, path: str) -> None:
        if path not in self._components:
            hint = difflib.get_close_matches(path, list(self._components), 3, 0.6)
            raise KeyError(
                f"unknown component {path!r}"
                + (f"; did you mean {', '.join(map(repr, hint))}?" if hint else "")
            )

    @staticmethod
    def _within(owner: str, path: str) -> bool:
        return path == "" or owner == path or owner.startswith(path + ".")

    def nodes_of(self, path: str) -> list[str]:
        """Nodes owned by the component at `path` or below it, in node order."""
        self._check(path)
        return [n for n, o in self._node_owner.items() if self._within(o, path)]

    def edges_of(self, path: str) -> list[str]:
        """Edges added by the component at `path` or below it, in edge order."""
        self._check(path)
        return [e for e, o in self._edge_owner.items() if self._within(o, path)]

    def select(self, *, template: str) -> list[str]:
        """Paths of every component made by `template`, in tree order."""
        return [p for p, c in self._components.items() if c.template == template]

    def boundary_edges(self, path: str, kind: str) -> Boundary:
        """The edges of `kind` with exactly one end owned by the subtree at `path`."""
        self._check(path)
        net = self.net
        edges = net.edges
        labels, positions, sign = [], [], []
        for pos, col in enumerate(net.edge_index(kind).tolist()):
            u, v, k = edges[col]
            inside_u = self._within(self._node_owner[u], path)
            inside_v = self._within(self._node_owner[v], path)
            if inside_u != inside_v:
                labels.append(net.graph.edges[u, v, k]["name"])
                positions.append(pos)
                sign.append(1.0 if inside_v else -1.0)
        return Boundary(labels, torch.tensor(positions, dtype=torch.long),
                        torch.tensor(sign, dtype=net.dtype))

    def boundary_flows(self, path: str, kind: str, flows: torch.Tensor) -> torch.Tensor:
        """Net inflow into the subtree at `path` through edges of `kind`: the signed sum of
        `flows` (last dimension in `net.edge_index(kind)` order) over its boundary edges.
        The flows a surrogate of the subtree would have to reproduce."""
        b = self.boundary_edges(path, kind)
        flows = torch.as_tensor(flows)
        n = len(self.net.edge_index(kind))
        if flows.ndim == 0 or flows.shape[-1] != n:
            raise ValueError(
                f"flows has shape {tuple(flows.shape)}, but the network has {n} edges of kind "
                f"{kind!r}; the last dimension must run over them in the edge order of kind "
                f"{kind!r}"
            )
        return (flows[..., b.positions] * b.sign.to(flows.dtype)).sum(-1)

    def tree(self) -> dict:
        """The hierarchy as plain data: per component its path, name, template, position,
        ports (port -> node), own nodes and edges, and children."""
        def build(path: str) -> dict:
            c = self._components[path]
            return {
                "path": path, "name": c.name, "template": c.template,
                "position": c.position, "ports": dict(c.ports),
                "nodes": [n for n, o in self._node_owner.items() if o == path],
                "edges": [e for e, o in self._edge_owner.items() if o == path],
                "children": [build(child) for child in c.children],
            }
        return build("")

    def resolve(self, label: Hashable) -> str:
        """The node name `label` refers to (itself, or the node an alias was merged into)."""
        canonical = self.net.aliases.get(label, label)
        if canonical not in self._node_owner:
            known = [*self._node_owner, *self.net.aliases]
            hint = difflib.get_close_matches(str(label), known, 3, 0.6)
            suffix = f"; did you mean {', '.join(map(repr, hint))}?" if hint else ""
            raise KeyError(f"unknown node {label!r}{suffix}")
        return canonical
