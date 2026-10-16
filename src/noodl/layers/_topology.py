"""Shared stale-network refusal for layers that cache network indices at construction."""

from __future__ import annotations

from noodl.layers._device import Movable, Mover


class TopologyGuard(Movable):
    """Mixin: record the network a layer was built on and refuse one that has changed.

    A layer keeps the edge indices, endpoints, node count and attribute values of
    construction time, as tensors on the network's device and dtype of that time. It
    records, when built (`_record_topology`, once `self.net` is set):

    * `Network.topology_version`, so a network that has since gained a node or edge is
      refused at every public entry point (`_require_current_topology`, a cheap integer
      comparison);
    * `Network.attribute_fingerprint`, compared by `_attributes_changed`, which `Model.check`
      and `Model.to` call (one pass over the attributes: not on every step).

    It also freezes the network's graph (`Network.freeze`), so a direct structural change to
    `net.graph` raises. Rebuilding is explicit: construct the layer (and the model around
    it) again. `to(device, dtype)` moves the layer, its network and laws (`Mover`).
    """

    _topology_version: int

    def _record_topology(self, net) -> None:
        self._topology_version = net.topology_version
        self._attribute_fingerprint = net.attribute_fingerprint()
        net.freeze()

    def _layer_label(self) -> str:
        return f"{type(self).__name__} {getattr(self, 'name', '?')!r}"

    def _require_current_topology(self) -> None:
        """Raise `RuntimeError` when the network gained a node or edge after construction."""
        current = self.net.topology_version
        # A layer pickled before the guard existed has no recorded version: it reads as 0,
        # which is also what its network, pickled with it and equally old, loads at
        # (`Network.topology_version`), so an unchanged old pickle matches and a network
        # changed after loading is still refused.
        recorded = getattr(self, "_topology_version", 0)
        if current != recorded:
            raise RuntimeError(
                f"{self._layer_label()}: the network topology has changed "
                f"since this layer was constructed (topology version "
                f"{recorded} -> {current}); the layer still describes the "
                f"old graph. Add all nodes and edges before constructing layers, or "
                f"construct the layer and model again."
            )

    def _attributes_changed(self) -> bool:
        """True when a node or edge attribute was edited after the layer was built."""
        recorded = getattr(self, "_attribute_fingerprint", None)
        return recorded is not None and recorded != self.net.attribute_fingerprint()

    def _require_current_attributes(self) -> None:
        if self._attributes_changed():
            raise RuntimeError(
                f"{self._layer_label()}: a node or edge attribute of the network was "
                f"edited after this layer was built; the layer and its laws copied the "
                f"attributes when they were built and do not see the edit. Build the laws, "
                f"layers and model again."
            )

    def _require_current(self) -> None:
        """Topology and attributes unchanged since construction."""
        recorded = getattr(self, "_topology_version", 0)
        if self.net.topology_version != recorded:
            self._require_current_topology()
        self._require_current_attributes()

    def to(self, device=None, dtype=None):
        """Move this layer, its network, laws, drives and node sources to `device` and the
        precision of `dtype`, in place; return the layer. Other layers on the same network
        then refuse to run until they are moved too: move a model with `Model.to`."""
        self._require_current()
        Mover(device, dtype).visit(self)
        return self

    def _move_with(self, mover: Mover) -> None:
        mover.attributes(self)
