"""Flattening a component tree into one `Network` (docs/concepts/components.md).

One walk over the tree, in this order: allocate a slot for every node, terminal and inner
node; merge connected ports (union-find); check each merged set (at most one real node);
name each node by the path of its real node; drop terminals never connected, with their
edges; check edges that would join a node to itself; then emit nodes component by component
in tree order, and edges in the same order."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch

from noodl.components import _registry
from noodl.components._component import Component, ComponentError, PortRef
from noodl.components._names import ComponentInfo, NameMap
from noodl.topology import Network

Offset = tuple[float, float, float]


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


@dataclass
class _Visit:
    comp: Component
    path: str                        # "" for the root
    offset: Offset                   # absolute placement (m)
    scope: tuple[Component, ...]     # root ... comp, for inner/outer and tables
    depth: int


@dataclass
class _Slot:
    path: str
    owner: str                       # path of the component that added it
    depth: int
    order: int
    attrs: dict | None               # None: a terminal
    offset: Offset


def _walk(root: Component) -> list[_Visit]:
    visits: list[_Visit] = []

    def go(comp, path, offset, scope, depth):
        visit = _Visit(comp, path, offset, (*scope, comp), depth)
        visits.append(visit)
        for name, (child, at) in comp._children.items():
            go(child, _join(path, name),
               tuple(o + a for o, a in zip(offset, at, strict=True)), visit.scope, depth + 1)

    go(root, "", (0.0, 0.0, 0.0), (), 0)
    return visits


def flatten(root: Component, *, dtype: torch.dtype = torch.float32, node_elevations=None,
            edge_elevations=None, table_lookups=None) -> tuple[Network, NameMap]:
    """The `Network` the tree under `root` describes, and its `NameMap`.

    Raises `ComponentError` listing every problem at once: two real nodes merged into one,
    an `outer` with no enclosing `inner`, an edge whose two ends merge, a facade that does
    not resolve. `node_elevations`/`edge_elevations`/`table_lookups` override the registry
    (`noodl.components.register_elevations`, `register_table_lookup`)."""
    return _Flattener(root, node_elevations, edge_elevations, table_lookups).run(dtype)


class _Flattener:
    def __init__(self, root, node_elevations, edge_elevations, table_lookups) -> None:
        self.root = root
        self.node_elev = frozenset(_registry.node_elevations() if node_elevations is None
                                   else node_elevations)
        self.edge_elev = frozenset(_registry.edge_elevations() if edge_elevations is None
                                   else edge_elevations)
        self.lookups = dict(_registry.table_lookups() if table_lookups is None
                            else table_lookups)
        self.visits = _walk(root)
        self.errors: list[str] = []
        self.slots: list[_Slot] = []
        self.index: dict[tuple, int] = {}
        self.used_ports: set[tuple[int, str]] = set()
        self.external: set[int] = set()      # slots reached through a port from outside
        self.parent: list[int] = []

    def where(self, visit: _Visit) -> str:
        return visit.path or self.root.name

    # ------------------------------------------------------------------ slots
    def allocate(self) -> None:
        for v in self.visits:
            for name, attrs in v.comp._inners.items():
                self._new(("inner", id(v.comp), name), v, name, dict(attrs))
            for name, local in v.comp._local.items():
                attrs = None if local.attrs is None else dict(local.attrs)
                self._new(("local", id(v.comp), name), v, name, attrs)
            for end in v.comp._ports.values():
                if isinstance(end, PortRef):            # a re-export uses the child's port
                    self.used_ports.add((id(end.component), end.name))
        self.parent = list(range(len(self.slots)))

    def _new(self, key, v: _Visit, name: str, attrs) -> None:
        self.index[key] = len(self.slots)
        self.slots.append(_Slot(_join(v.path, name), v.path, v.depth, len(self.slots), attrs,
                                v.offset))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j) -> None:
        if i is None or j is None:
            return
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)

    def port_slot(self, ref: PortRef) -> int:
        comp, name = ref.component, ref.name
        while True:
            end = comp._ports[name]
            if isinstance(end, str):
                return self.index[("local", id(comp), end)]
            comp, name = end.component, end.name

    def resolve(self, v: _Visit, end) -> int | None:
        if isinstance(end, str):
            return self.index[("local", id(v.comp), end)]
        if isinstance(end, PortRef):
            self.used_ports.add((id(end.component), end.name))
            slot = self.port_slot(end)
            self.external.add(slot)
            return slot
        for comp in reversed(v.scope):
            if end.name in comp._inners:
                return self.index[("inner", id(comp), end.name)]
        self.errors.append(
            f"{self.where(v)}: outer({end.name!r}) has no enclosing inner({end.name!r}, ...); "
            f"declare it on this component or one of its ancestors"
        )
        return None

    # ------------------------------------------------------------------ merged sets
    def representatives(self) -> tuple[dict[int, list[int]], dict[int, int]]:
        members: dict[int, list[int]] = {}
        for i in range(len(self.slots)):
            members.setdefault(self.find(i), []).append(i)
        rep: dict[int, int] = {}
        for root, ms in members.items():
            reals = [m for m in ms if self.slots[m].attrs is not None]
            if len(reals) > 1:
                self.errors.append(
                    "connecting merges " + " and ".join(repr(self.slots[m].path) for m in reals)
                    + " into one node, but each carries its own attributes: two zones cannot "
                    "be one air volume. Put a door, opening or other component between them."
                )
            rep[root] = reals[0] if reals else min(
                ms, key=lambda m: (self.slots[m].depth, self.slots[m].order))
        return members, rep

    def reject_hidden_terminals(self, dropped: set[int]) -> None:
        """A dropped terminal that is not a port can never be connected by anyone."""
        for v in self.visits:
            exposed = {end for end in v.comp._ports.values() if isinstance(end, str)}
            for name, local in v.comp._local.items():
                if local.attrs is not None or name in exposed:
                    continue
                if self.find(self.index[("local", id(v.comp), name)]) in dropped:
                    self.errors.append(
                        f"{_join(v.path, name)}: terminal {name!r} is not a port, so nothing "
                        f"can connect it; use add_node(...) for an internal junction, or "
                        f"expose it"
                    )

    def reject_user_positions(self) -> None:
        for slot in self.slots:
            if slot.attrs is not None and "position" in slot.attrs:
                self.errors.append(
                    f"'position' is set by flatten from the placement; do not set it on node "
                    f"{slot.path!r}"
                )

    # ------------------------------------------------------------------ attributes
    def node_attrs(self, slot: _Slot) -> dict:
        attrs = {} if slot.attrs is None else dict(slot.attrs)
        z = slot.offset[2]
        for name in self.node_elev:
            if name in attrs:
                attrs[name] = float(attrs[name]) + z
            elif z != 0.0:
                # A missing height is 0 relative to the placement (Stack.from_network reads
                # a missing z_ref as 0), so the node sits at its component's height.
                attrs[name] = z
        attrs["position"] = slot.offset
        return attrs

    def edge_attrs(self, v: _Visit, edge) -> dict:
        attrs = dict(edge.attrs)
        for name in self.edge_elev & attrs.keys():
            attrs[name] = float(attrs[name]) + v.offset[2]
        where = _join(v.path, edge.name)
        for attr, (table, target) in self.lookups.items():
            if attr not in attrs:
                continue
            if target in attrs:
                self.errors.append(f"{where}: both {attr}= and {target}= are set; give one")
                continue
            value = self.table_value(v, where, attr, table, attrs[attr])
            if value is not None:
                attrs[target] = value
        return attrs

    def table_value(self, v: _Visit, where: str, attr: str, table: str, key):
        for comp in reversed(v.scope):
            if table in comp._tables:
                entries = comp._tables[table]
                if key not in entries:
                    self.errors.append(
                        f"{where}: {attr}={key!r} is not in the {table!r} table of "
                        f"{comp.name!r}, which has {', '.join(map(repr, sorted(entries)))}"
                    )
                    return None
                return float(entries[key])
        self.errors.append(
            f"{where}: {attr}={key!r} but no enclosing component declares "
            f"inner_table({table!r}, ...)"
        )
        return None

    def warn_unshifted(self) -> None:
        if self.node_elev or self.edge_elev:
            return
        raised = [v for v in self.visits if v.offset[2] != 0.0]
        if raised:
            warnings.warn(
                f"{self.where(raised[0])} is placed at z = {raised[0].offset[2]} m but no "
                f"height attributes are registered, so no height is shifted; import "
                f"the application that defines the height attributes (noodl.apps.building_physics "
                f"registers z_ref and z_path), call noodl.components.register_elevations(...), "
                f"or pass node_elevations= / edge_elevations= to flatten()",
                UserWarning, stacklevel=5,
            )

    # ------------------------------------------------------------------ run
    def run(self, dtype) -> tuple[Network, NameMap]:
        self.allocate()
        self.warn_unshifted()
        for v in self.visits:
            for a, b in v.comp._connects:
                self.union(self.resolve(v, a), self.resolve(v, b))
        raw = [(v, e, self.resolve(v, e.source), self.resolve(v, e.target))
               for v in self.visits for e in v.comp._edges]
        members, rep = self.representatives()
        # A terminal nobody outside its component reached (by connect or by an edge) is a
        # closed opening: drop it with its edges.
        dropped = {r for r, ms in members.items()
                   if len(ms) == 1 and self.slots[ms[0]].attrs is None
                   and ms[0] not in self.external}
        self.reject_hidden_terminals(dropped)
        self.reject_user_positions()

        kept = []
        for v, e, s, t in raw:
            if s is None or t is None:
                continue
            rs, rt = self.find(s), self.find(t)
            if rs in dropped or rt in dropped:
                continue
            if rs == rt:
                self.errors.append(
                    f"{_join(v.path, e.name)}: both ends of this {e.kind!r} edge are the same "
                    f"node {self.slots[rep[rs]].path!r} after connecting; an opening from a "
                    f"zone to itself carries no flow and is almost certainly a wrong connect"
                )
                continue
            kept.append((v, e, rs, rt, self.edge_attrs(v, e)))
        if self.errors:
            raise ComponentError(
                f"cannot flatten {self.root.name!r}: {len(self.errors)} problem(s)\n  - "
                + "\n  - ".join(self.errors)
            )

        net = Network(dtype=dtype)
        node_owner: dict[str, str] = {}
        for v in self.visits:
            keys = [("inner", id(v.comp), n) for n in v.comp._inners]
            keys += [("local", id(v.comp), n) for n in v.comp._local]
            for key in keys:
                i = self.index[key]
                r = self.find(i)
                if rep[r] != i or r in dropped:
                    continue
                slot = self.slots[i]
                net.add_node(slot.path, **self.node_attrs(slot))
                node_owner[slot.path] = slot.owner
        edge_owner: dict[str, str] = {}
        for v, e, rs, rt, attrs in kept:
            name = _join(v.path, e.name)
            net.add_edge(self.slots[rep[rs]].path, self.slots[rep[rt]].path, kind=e.kind,
                         name=name, **attrs)
            edge_owner[name] = v.path
        net.aliases = {
            self.slots[m].path: self.slots[rep[r]].path
            for r, ms in members.items() if r not in dropped for m in ms if m != rep[r]
        }

        components: dict[str, ComponentInfo] = {}
        for v in self.visits:
            ports = {}
            for pname in v.comp._ports:
                r = self.find(self.port_slot(PortRef(v.comp, pname)))
                ports[pname] = None if r in dropped else self.slots[rep[r]].path
            components[v.path] = ComponentInfo(
                v.comp.name, v.comp.template, v.offset, ports,
                [_join(v.path, c) for c in v.comp._children])
        unconnected = [
            _join(v.path, pname) for v in self.visits[1:] for pname in v.comp._ports
            if (id(v.comp), pname) not in self.used_ports
        ]
        names = NameMap(net, node_owner=node_owner, edge_owner=edge_owner,
                        components=components, unconnected_ports=unconnected)
        return net, names
