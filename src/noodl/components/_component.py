"""A component: a piece of network with a declared boundary of ports. Components nest;
`connect` merges two ports into one node (a Modelica connection set: potentials equal,
flows summing to zero); `flatten` turns the tree into one ordinary `Network`. See
docs/concepts/components.md."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import torch


class ComponentError(ValueError):
    """A component tree that cannot be built or flattened; the message names the path."""


@dataclass(frozen=True, eq=False)
class PortRef:
    """Port `name` of `component`, as `component.ports.<name>` returns it."""

    component: Component
    name: str

    def __eq__(self, other) -> bool:
        return (isinstance(other, PortRef) and other.component is self.component
                and other.name == self.name)

    def __hash__(self) -> int:
        return hash((id(self.component), self.name))

    def __repr__(self) -> str:
        return f"<port {self.component.name}.{self.name}>"


@dataclass(frozen=True)
class OuterRef:
    """A shared node declared by the nearest enclosing `inner(name)` (Modelica `outer`)."""

    name: str


@dataclass
class _Local:
    attrs: dict[str, Any] | None   # None: a terminal (a bare connection point)


@dataclass
class _Edge:
    source: Any
    target: Any
    kind: str
    name: str
    attrs: dict[str, Any]


def _check_name(what: str, name) -> None:
    if not isinstance(name, str) or not name:
        raise ComponentError(f"{what} name must be a non-empty string, got {name!r}")
    if "." in name:
        raise ComponentError(
            f"{what} name {name!r} contains '.', which separates the parts of a path"
        )


class Ports:
    """The ports of one component: `c.ports.air`, `c.ports["levels[0]"]`, `list(c.ports)`."""

    def __init__(self, component: Component) -> None:
        self._component = component

    def __getitem__(self, name: str) -> PortRef:
        c = self._component
        if name not in c._ports:
            raise KeyError(
                f"component {c.name!r} has no port {name!r}; its ports are {list(c._ports)}"
            )
        return PortRef(c, name)

    def __getattr__(self, name: str) -> PortRef:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(exc.args[0]) from None

    def __iter__(self):
        return iter(list(self._component._ports))

    def __len__(self) -> int:
        return len(self._component._ports)

    def __contains__(self, name) -> bool:
        return name in self._component._ports


class Component:
    """A named piece of network with ports. Build it with `add_node`/`add_terminal`/
    `add_edge`, declare its boundary with `expose`, nest children with `add` and join their
    ports with `connect`. `template` records which factory made it (for `NameMap.select`)."""

    def __init__(self, name: str, *, template: str | None = None) -> None:
        _check_name("component", name)
        self.name = name
        self.template = template
        self._local: dict[str, _Local] = {}
        self._edges: list[_Edge] = []
        self._ports: dict[str, Any] = {}
        self._children: dict[str, tuple[Component, tuple[float, float, float]]] = {}
        self._connects: list[tuple[Any, Any]] = []
        self._inners: dict[str, dict[str, Any]] = {}
        self._tables: dict[str, dict[str, Any]] = {}
        self._parent: Component | None = None
        self.ports = Ports(self)

    def __repr__(self) -> str:
        return f"<Component {self.name!r} template={self.template!r}>"

    # ------------------------------------------------------------------ views
    @property
    def nodes(self) -> list[str]:
        """This component's own nodes and terminals, in the order added."""
        return list(self._local)

    @property
    def edges(self) -> list[str]:
        """This component's own edge names, in the order added."""
        return [e.name for e in self._edges]

    @property
    def children(self) -> dict[str, Component]:
        return {name: child for name, (child, _) in self._children.items()}

    # ------------------------------------------------------------------ building
    def _taken(self, name: str) -> bool:
        return name in self._local or name in self._inners or name in self._children

    def _new_local(self, name: str, attrs) -> None:
        _check_name("node", name)
        if self._taken(name):
            raise ComponentError(f"{self.name}: {name!r} already exists in this component")
        self._local[name] = _Local(attrs)

    def add_node(self, name: str, **attrs) -> None:
        """A real node carrying attributes (a zone's air, a wall)."""
        self._new_local(name, dict(attrs))

    def add_terminal(self, name: str) -> None:
        """A bare connection point with no attributes, meant to be merged with a real node
        by `connect` (each side of a door). One never connected is dropped by `flatten`
        with its edges: a closed opening."""
        self._new_local(name, None)

    def add_edge(self, source, target, *, kind: str, name: str | None = None, **attrs) -> str:
        """An edge between two of this component's nodes, ports of its direct children, or
        `outer` references. Returns the edge's name (`e<k>` when not given)."""
        if not isinstance(kind, str) or not kind:
            raise ComponentError(f"{self.name}: an edge needs a non-empty kind, got {kind!r}")
        for end in (source, target):
            self._check_endpoint(end, allow_outer=True)
        taken = {e.name for e in self._edges}
        if name is None:
            k = 0
            while f"e{k}" in taken:
                k += 1
            name = f"e{k}"
        _check_name("edge", name)
        if name in taken:
            raise ComponentError(f"{self.name}: edge {name!r} already exists in this component")
        self._edges.append(_Edge(source, target, kind, name, dict(attrs)))
        return name

    def _check_endpoint(self, end, *, allow_outer: bool) -> None:
        if isinstance(end, OuterRef):
            if not allow_outer:
                raise ComponentError(
                    f"{self.name}: outer({end.name!r}) cannot be connected; a shared node is "
                    f"reached by reference from an edge, never merged"
                )
            return
        if isinstance(end, PortRef):
            child = end.component
            entry = self._children.get(child.name)
            if entry is None or entry[0] is not child:
                raise ComponentError(
                    f"{self.name}: {end!r} is not a port of a direct child of {self.name!r}; "
                    f"add the child first, or re-expose a grandchild's port from its parent"
                )
            return
        if isinstance(end, str):
            if end not in self._local:
                hint = (f" (a shared node is reached with outer({end!r}))"
                        if end in self._inners else "")
                raise ComponentError(
                    f"{self.name}: unknown node {end!r}; this component's nodes are "
                    f"{list(self._local)}{hint}"
                )
            return
        raise ComponentError(
            f"{self.name}: {end!r} is not a node name, a port or an outer reference"
        )

    def expose(self, *names: str, **renamed) -> None:
        """Declare ports: `expose("air")` exposes an own node; `expose(corridor=r.ports.air)`
        re-exports a child's port under a new name."""
        for name in names:
            if not isinstance(name, str):
                raise ComponentError(
                    f"{self.name}: expose(*names) takes node names; re-export a child's port "
                    f"as expose(new_name=child.ports.x)"
                )
            self._check_endpoint(name, allow_outer=False)
            self._add_port(name, name)
        for new, end in renamed.items():
            if not isinstance(end, str | PortRef):
                raise ComponentError(f"{self.name}: port {new!r} must name a node or a port")
            self._check_endpoint(end, allow_outer=False)
            self._add_port(new, end)

    def _add_port(self, name: str, end) -> None:
        _check_name("port", name)
        if name in self._ports:
            raise ComponentError(f"{self.name}: port {name!r} already exists")
        self._ports[name] = end

    def add(self, child: Component, *, at=(0.0, 0.0, 0.0)) -> Component:
        """Add `child`, placed at `at` = (x, y, z) in m relative to this component.
        Returns `child`."""
        if not isinstance(child, Component):
            raise ComponentError(f"{self.name}: add() takes a Component, got {child!r}")
        if child is self:
            raise ComponentError(f"{self.name}: a component cannot contain itself")
        ancestor = self._parent
        while ancestor is not None:
            if ancestor is child:
                raise ComponentError(f"{self.name}: {child.name!r} is an ancestor of it")
            ancestor = ancestor._parent
        if child._parent is not None:
            raise ComponentError(
                f"component {child.name!r} is already part of {child._parent.name!r}; call "
                f"its factory again for a second instance"
            )
        if self._taken(child.name):
            raise ComponentError(f"{self.name}: {child.name!r} already exists in this component")
        try:
            parts = tuple(at) if not isinstance(at, str) else ()
        except TypeError:
            parts = ()
        if (len(parts) != 3
                or not all(isinstance(v, numbers.Real) and not isinstance(v, bool)
                           and math.isfinite(float(v)) for v in parts)):
            raise ComponentError(f"{self.name}: at= must be (x, y, z) in m, got {at!r}")
        position = tuple(float(v) for v in parts)
        self._children[child.name] = (child, position)
        child._parent = self
        return child

    def connect(self, a, b) -> None:
        """Merge two ports (of direct children, or this component's own nodes) into one
        node. A port may appear in many connects; connects are transitive."""
        for end in (a, b):
            self._check_endpoint(end, allow_outer=False)
        if a == b:
            raise ComponentError(f"{self.name}: connect({a!r}, {b!r}) connects a port to itself")
        self._connects.append((a, b))

    def inner(self, name: str, **attrs) -> None:
        """A shared node (ambient, ground, a plenum) that every component below reaches with
        `outer(name)`; the nearest enclosing declaration wins (Modelica inner/outer)."""
        _check_name("inner node", name)
        if self._taken(name):
            raise ComponentError(f"{self.name}: {name!r} already exists in this component")
        self._inners[name] = dict(attrs)

    def inner_table(self, name: str, **entries) -> None:
        """A shared lookup table, e.g. `inner_table("facades", south=180.0)`; edges below
        refer to an entry by key through a registered lookup attribute (`facade=`)."""
        _check_name("table", name)
        if name in self._tables:
            raise ComponentError(f"{self.name}: table {name!r} already exists")
        self._tables[name] = dict(entries)

    @staticmethod
    def outer(name: str) -> OuterRef:
        """A reference to the shared node of the nearest enclosing `inner(name)`."""
        return OuterRef(name)

    def flatten(self, *, dtype: torch.dtype = torch.float32, node_elevations=None,
                edge_elevations=None, table_lookups=None):
        """The ordinary `Network` this tree describes, and a `NameMap` to query it by
        component. See `noodl.components._flatten.flatten`."""
        from noodl.components._flatten import flatten

        return flatten(self, dtype=dtype, node_elevations=node_elevations,
                       edge_elevations=edge_elevations, table_lookups=table_lookups)
