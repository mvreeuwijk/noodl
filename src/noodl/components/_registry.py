"""Which attributes `Component.flatten` treats as heights, and which edge attributes it
looks up in an `inner_table`. The core knows no attribute names; applications register
theirs on import (`noodl.apps.building_physics` registers `z_ref`, `z_path` and facades)."""

from __future__ import annotations

_NODE_ELEVATIONS: set[str] = set()
_EDGE_ELEVATIONS: set[str] = set()
_TABLE_LOOKUPS: dict[str, tuple[str, str]] = {}


def register_elevations(*, nodes=(), edges=()) -> None:
    """Node and edge attributes holding a height (m) relative to the component's placement;
    `flatten` adds the accumulated z offset to them."""
    _NODE_ELEVATIONS.update(nodes)
    _EDGE_ELEVATIONS.update(edges)


def register_table_lookup(attr: str, *, table: str, target: str) -> None:
    """An edge attribute `attr` holding a key of the nearest enclosing `inner_table(table)`;
    `flatten` writes the table's value into the edge attribute `target`."""
    new = (table, target)
    old = _TABLE_LOOKUPS.get(attr)
    if old is not None and old != new:
        raise ValueError(f"edge attribute {attr!r} is already looked up as {old}, not {new}")
    _TABLE_LOOKUPS[attr] = new


def node_elevations() -> frozenset[str]:
    return frozenset(_NODE_ELEVATIONS)


def edge_elevations() -> frozenset[str]:
    return frozenset(_EDGE_ELEVATIONS)


def table_lookups() -> dict[str, tuple[str, str]]:
    return dict(_TABLE_LOOKUPS)
