"""The water network as a graph of node indices: which nodes the links join, and which nodes
water can reach. Internal to `noodl.apps.water`; a leaf module (it imports nothing of the
application), so the laws, the cut-off analysis and the energy all share it.

Every walk here is over `ends`, the `(start, end)` node indices of every link in
`model.link_names` order (`link_topology`), and a per-link flag saying which links take
part: the open links, the links that are not CLOSED, the solid links of `_limit`, and so on.
"""

from __future__ import annotations

import functools
from collections import deque
from collections.abc import Hashable, Iterable, Mapping, Sequence

from noodl._unionfind import UnionFind


def link_topology(model) -> tuple[tuple, tuple]:
    """`(ends, fixed)`: the `(start, end)` node indices of every link in `model.link_names`
    order, and the reservoir and tank nodes (`WaterNetwork.fixed_head_nodes`); built once
    per model."""
    found = model.__dict__.get("_link_topology")
    if found is None:
        net = model.water_network
        index = {name: i for i, name in enumerate(model.node_names)}
        found = model._link_topology = (
            tuple((index[k.u], index[k.v]) for k in net.links()),
            tuple(index[f.name] for f in net.fixed_head_nodes()))
    return found


@functools.lru_cache(maxsize=256)
def joined_groups(ends: tuple, fixed: tuple, joined: tuple) -> tuple[dict, frozenset]:
    """`(group, fed)` for the links `joined` (per link, of `ends`): each node's group (the
    connected parts of the joined links) and the groups holding a node of `fixed`.
    Remembered: a solve meets the same few sets of closed check valves on iterate after
    iterate."""
    groups: UnionFind[int] = UnionFind()
    for (u, v), is_joined in zip(ends, joined, strict=True):
        groups.add(u)
        groups.add(v)
        if is_joined:
            groups.union(u, v)
    for n in fixed:
        groups.add(n)
    group = {n: groups.find(n) for n in groups.parent}
    return group, frozenset(group[n] for n in fixed)


def breadth_first(starts: Iterable[Hashable],
                  ahead: Mapping[Hashable, Sequence[tuple]]) -> dict:
    """`{place: link}` for every place reached from `starts` along `ahead` (`{place:
    [(link, next place), ...]}`, followed in that order), breadth first, in the order
    reached: the link each place was first reached by, `None` for a start."""
    reached = dict.fromkeys(starts)
    queue = deque(reached)
    while queue:
        for link, other in ahead.get(queue.popleft(), ()):
            if other not in reached:
                reached[other] = link
                queue.append(other)
    return reached
