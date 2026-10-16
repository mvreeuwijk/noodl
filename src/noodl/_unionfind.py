"""Disjoint sets (union-find), shared by `Network` and the Modelica readers."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T", bound=Hashable)


class UnionFind(Generic[T]):
    """Disjoint sets over hashable items, with full path compression.

    `find` adds an unknown item as its own set. `union(a, b)` keeps `a`'s root as the merged
    root, unless `b`'s root satisfies `keep_root` and `a`'s does not (a ground that must stay
    its class's representative). Roots are representatives only: `groups` lists the sets in
    the order their first member was added, whichever item ends up the root.
    """

    def __init__(self, items: Iterable[T] = (),
                 *, keep_root: Callable[[T], bool] | None = None) -> None:
        self.parent: dict[T, T] = {item: item for item in items}
        self._keep_root = keep_root

    def __contains__(self, item: object) -> bool:
        return item in self.parent

    def add(self, item: T) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: T) -> T:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: T, b: T) -> bool:
        """Merge the sets of `a` and `b`; False if they were already one set."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._keep_root is not None and self._keep_root(rb) and not self._keep_root(ra):
            ra, rb = rb, ra
        self.parent[rb] = ra
        return True

    def groups(self) -> dict[T, set[T]]:
        """Every set, keyed by its root."""
        out: dict[T, set[T]] = {}
        for item in self.parent:
            out.setdefault(self.find(item), set()).add(item)
        return out
