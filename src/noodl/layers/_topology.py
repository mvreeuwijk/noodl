"""Shared stale-topology refusal for layers that cache network indices at construction."""

from __future__ import annotations


class TopologyGuard:
    """Mixin: record `Network.topology_version` at construction and refuse a changed network.

    A layer keeps the edge indices, endpoints and node count of construction time, so on a
    network that has since gained a node or edge it would silently solve the old graph.
    Call `_record_topology(net)` once `self.net`/`self.name` are set, and
    `_require_current_topology()` at each public entry point (a cheap integer comparison).
    Rebuilding is explicit: construct the layer (and the model around it) again.
    """

    _topology_version: int

    def _record_topology(self, net) -> None:
        self._topology_version = net.topology_version

    def _require_current_topology(self) -> None:
        """Raise `RuntimeError` when the network gained a node or edge after construction."""
        current = self.net.topology_version
        if current != self._topology_version:
            raise RuntimeError(
                f"{type(self).__name__} {self.name!r}: the network topology has changed "
                f"since this layer was constructed (topology version "
                f"{self._topology_version} -> {current}); the layer still describes the "
                f"old graph. Add all nodes and edges before constructing layers, or "
                f"construct the layer and model again."
            )
