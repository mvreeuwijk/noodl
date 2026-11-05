"""Electro-thermal coupling: the Joule heat of circuit devices into a heat network, and the
heat-port temperatures back into the devices' temperature-dependent laws.

The heat network is a `TransportLayer` on the same `Network` as the circuit (conventionally
named `"thermal"`, with `flow_kind=()`: nothing advects): temperature (K) is its state, a
heat conductance (MSL `HeatTransfer.Components.ThermalConductor`) a conduction edge, a heat
capacity (`HeatCapacitor`) the `capacity` (J/K) of its node, a fixed temperature
(`Sources.FixedTemperature`) a boundary node, and the heat a device dissipates the layer's
nodal source `"<thermal>.sources"` (W), which the closure below writes. A heat port with no
heat capacity of its own is a `storage_free` node: the layer folds it into its neighbours, so
its temperature is the algebraic balance of the heat it receives.

`JouleHeating` is the coupling closure: from the state it computes each heated device's
dissipated power (MSL `LossPower = v i`) into the heat layer's sources, and copies each
heat-port temperature into the per-edge temperature driver the device's law reads
(`Resistance`'s and `JunctionDiode`'s `temperature`). The two layers are coupled with
`Model(..., coupling="iterate")`: every pass solves the circuit at the temperatures of the
previous pass and the heat network at its heat, until both stop changing.

The closure carries the sources as state (`state_keys`), so a step-start state holds the
heat dissipated at its own time. A trapezoidal step of the heat layer then integrates the
heat as the circuit's trapezoidal companion forms integrate their currents,
`dt/2 (P_start + P_end)`; a backward Euler step (`scheme="implicit"`, the step after a
restart) takes `P_end` alone. Stepping the heat layer with the circuit's rule on every step
(`Model.step(..., layer_options={"thermal": {"scheme": ...}})`) is the caller's part.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from noodl.layers.transport import TransportLayer
from noodl.layout import broadcast_batch

Tensor = torch.Tensor

__all__ = ["HeatedEdge", "JouleHeating"]


@dataclass(frozen=True)
class HeatedEdge:
    """A circuit device with a heat port: its edge `edge` (position in the electrical layer's
    `q`), its end nodes `p` and `n` (positions in `phi`), the network node `heat_node` of its
    heat port (a node of the heat layer: interior, where the heat goes, or a boundary at a
    fixed temperature, which absorbs it), and the per-edge temperature driver
    `temperature_key` its law reads, at position `slot` of that driver."""

    edge: int
    p: int
    n: int
    heat_node: int
    temperature_key: str
    slot: int


class JouleHeating:
    """A coupling closure `closure(state, drivers) -> driver updates` between a circuit layer
    (`electrical`) and a heat layer (`thermal`, a `TransportLayer`) on one network.

    For the heated devices `heated` (one `HeatedEdge` each) it returns
    `{"<thermal>.sources": s (..., n_nodes), <temperature_key>: T (..., n_slots)}`. `s` is
    the heat in W per network node, in full node order: `p_k = Re((phi[p] - phi[n])
    conj(q[edge]))`, the power device k dissipates (MSL `LossPower = v i`), summed at the
    interior heat-port nodes and zero elsewhere (a heat port at a fixed temperature hands its
    heat to that boundary, which absorbs it). Each temperature driver is the caller's
    `drivers[key]` (`base_temperatures[key]` where the drivers do not hold it) with the heated
    devices' slots replaced by their heat-port temperatures (K): the layer's state
    `"<thermal>.x"` at an interior heat port, its driver `"<thermal>.x_boundary"` at a fixed
    one. The devices without a heat port keep the temperature the drivers give them. Batch
    dimensions lead (`[..., index]`); nothing is modified in place.

    `state_keys` is `("<thermal>.sources",)`: the heat a state was solved with is carried in
    it, which a trapezoidal step of the heat layer reads as the heat at the step start
    (`Model`, `TransportLayer.step`'s `sources_prev`). A start state carries it too (zeros
    before any heat was computed).
    """

    integrates = False

    def __init__(self, heated: Sequence[HeatedEdge], thermal: TransportLayer, *,
                 electrical: str = "electrical",
                 base_temperatures: Mapping[str, Tensor]) -> None:
        self.heated = tuple(heated)
        if not self.heated:
            raise ValueError("JouleHeating: no heated device given")
        if thermal.n_species != 1:
            raise ValueError(f"JouleHeating: the heat layer {thermal.name!r} must carry one "
                             f"quantity (temperature), not {thermal.n_species}")
        self.electrical = electrical
        self.thermal = thermal.name
        self.sources_key = f"{thermal.name}.sources"
        self.state_keys = (self.sources_key,)
        self.n_nodes = thermal.net.n
        self.base_temperatures = {k: torch.as_tensor(v) for k, v in base_temperatures.items()}
        unknown = sorted({h.temperature_key for h in self.heated} - set(self.base_temperatures))
        if unknown:
            raise ValueError(f"JouleHeating: no base temperatures for {unknown}; give "
                             f"base_temperatures for every temperature_key")
        for h in self.heated:
            size = self.base_temperatures[h.temperature_key].shape[-1:]
            if not size or not 0 <= h.slot < size[0]:
                raise ValueError(f"JouleHeating: slot {h.slot} lies outside the "
                                 f"{tuple(size)} base temperatures of {h.temperature_key!r}")
        interior = {int(k): i for i, k in enumerate(thermal.interior_idx.tolist())}
        boundary = {int(k): i for i, k in enumerate(thermal.boundary_idx.tolist())}
        outside = [thermal.net.nodes[h.heat_node] if 0 <= h.heat_node < self.n_nodes
                   else h.heat_node for h in self.heated
                   if h.heat_node not in interior and h.heat_node not in boundary]
        if outside:
            raise ValueError(f"JouleHeating: heat ports {outside} are not nodes of the heat "
                             f"layer {thermal.name!r} (no conduction edge reaches them and "
                             f"they are no boundary)")
        self._edge = torch.tensor([h.edge for h in self.heated])
        self._p = torch.tensor([h.p for h in self.heated])
        self._n = torch.tensor([h.n for h in self.heated])
        # the heated devices whose heat port is an interior node, and those nodes
        heats = [k for k, h in enumerate(self.heated) if h.heat_node in interior]
        self._heats = torch.tensor(heats, dtype=torch.long)
        self._heat_nodes = torch.tensor([self.heated[k].heat_node for k in heats],
                                        dtype=torch.long)
        # temperature key -> (slots, positions in "<thermal>.x", slots, positions in
        # "<thermal>.x_boundary") of the heated devices in that driver
        self._slots: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        for key in self.base_temperatures:
            mine = [h for h in self.heated if h.temperature_key == key]
            if mine:
                inner = [h for h in mine if h.heat_node in interior]
                fixed = [h for h in mine if h.heat_node in boundary]
                self._slots[key] = (
                    torch.tensor([h.slot for h in inner], dtype=torch.long),
                    torch.tensor([interior[h.heat_node] for h in inner], dtype=torch.long),
                    torch.tensor([h.slot for h in fixed], dtype=torch.long),
                    torch.tensor([boundary[h.heat_node] for h in fixed], dtype=torch.long))

    def __call__(self, state: Mapping[str, Tensor], drivers: Mapping[str, Tensor]
                 ) -> dict[str, Tensor]:
        phi = state[f"{self.electrical}.phi"]
        q = state[f"{self.electrical}.q"]
        temperature = state[f"{self.thermal}.x"]
        dp = phi[..., self._p] - phi[..., self._n]
        power = torch.real(dp * torch.conj(q[..., self._edge]))
        heat = power.index_select(-1, self._heats.to(power.device))
        sources = heat.new_zeros(heat.shape[:-1] + (self.n_nodes,)).index_add(
            -1, self._heat_nodes.to(heat.device), heat)
        out: dict[str, Tensor] = {self.sources_key: sources}
        for key, (slots, nodes, fixed_slots, fixed_nodes) in self._slots.items():
            base = drivers.get(key, self.base_temperatures[key])
            base = torch.as_tensor(base).to(dtype=temperature.dtype, device=temperature.device)
            values = [(slots, temperature.index_select(-1, nodes.to(temperature.device)))]
            if fixed_slots.numel():
                bound = torch.as_tensor(drivers[f"{self.thermal}.x_boundary"]).to(
                    dtype=temperature.dtype, device=temperature.device)
                values.append((fixed_slots, bound.index_select(-1, fixed_nodes.to(bound.device))))
            for at, heated in values:
                _, (heated, base) = broadcast_batch((heated, 1), (base, 1))
                base = base.index_copy(-1, at.to(heated.device), heated)
            out[key] = base
        return out
