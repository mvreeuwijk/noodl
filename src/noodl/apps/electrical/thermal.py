"""Electro-thermal coupling: the Joule heat of circuit devices into a thermal network, and the
heat-port temperatures back into the devices' temperature-dependent laws.

The thermal side is a second `PotentialFlowLayer` on the same `Network` (conventionally named
`"thermal"`): temperature (K) is its potential against a reference node held at 0 K (a
boundary), heat flow (W) its flow. A heat conductance is a `noodl.elements.Conductance`
(MSL `HeatTransfer.Components.ThermalConductor`), a heat capacity a `Capacitance` from the
heated node to the 0 K reference (`HeatCapacitor`), a fixed temperature a boundary node
(`Sources.FixedTemperature`), and the heat a device dissipates a `PrescribedCurrent` from the
reference into its heat-port node whose value is `Signal(heat_key, k)`, read from the driver
the closure below writes. Both layers are integrated with the same companion forms, so a
heat capacitor follows the circuit's backward Euler / trapezoidal steps.

`JouleHeating` is the coupling closure: from the state it computes each heated device's
dissipated power (MSL `LossPower = v i`) into `heat_key`, and copies each heat-port
temperature into the per-edge temperature driver the device's law reads (`Resistance`'s and
`JunctionDiode`'s `temperature`). The two layers are coupled with `Model(...,
coupling="iterate")`: every pass solves the circuit at the temperatures of the previous pass
and the thermal network at its heat, until both stop changing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

Tensor = torch.Tensor

__all__ = ["HeatedEdge", "JouleHeating"]


@dataclass(frozen=True)
class HeatedEdge:
    """A circuit device with a heat port: its edge `edge` (position in the electrical layer's
    `q`), its end nodes `p` and `n` (positions in `phi`), the thermal node `heat_node` of its
    heat port (position in the thermal layer's `phi`), and the per-edge temperature driver
    `temperature_key` its law reads, at position `slot` of that driver."""

    edge: int
    p: int
    n: int
    heat_node: int
    temperature_key: str
    slot: int


class JouleHeating:
    """A coupling closure `closure(state, drivers) -> driver updates` between a circuit layer
    (`electrical`) and a thermal layer (`thermal`) on one network.

    For the heated devices `heated` (one `HeatedEdge` each) it returns
    `{heat_key: p (..., n_heated), <temperature_key>: T (..., n_slots)}` where
    `p_k = Re((phi[p] - phi[n]) conj(q[edge]))` (W) is the power device k dissipates (MSL
    `LossPower = v i`; on a phasor layer the product of the phasors as given), and each
    temperature driver is the caller's `drivers[key]` (`base_temperatures[key]` where the
    drivers do not hold it) with the heated devices' slots replaced by their heat-port
    temperatures `thermal phi[heat_node]` (K): the devices without a heat port keep the
    temperature the drivers give them. Batch dimensions lead (`[..., index]`); nothing is
    modified in place.
    """

    def __init__(self, heated: Sequence[HeatedEdge], *, electrical: str = "electrical",
                 thermal: str = "thermal", heat_key: str = "thermal.heat",
                 base_temperatures: Mapping[str, Tensor]) -> None:
        self.heated = tuple(heated)
        if not self.heated:
            raise ValueError("JouleHeating: no heated device given")
        self.electrical = electrical
        self.thermal = thermal
        self.heat_key = heat_key
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
        self._edge = torch.tensor([h.edge for h in self.heated])
        self._p = torch.tensor([h.p for h in self.heated])
        self._n = torch.tensor([h.n for h in self.heated])
        # temperature key -> (slots, heat nodes) in that driver
        self._slots: dict[str, tuple[Tensor, Tensor]] = {}
        for key in self.base_temperatures:
            mine = [h for h in self.heated if h.temperature_key == key]
            if mine:
                self._slots[key] = (torch.tensor([h.slot for h in mine]),
                                    torch.tensor([h.heat_node for h in mine]))

    def __call__(self, state: Mapping[str, Tensor], drivers: Mapping[str, Tensor]
                 ) -> dict[str, Tensor]:
        phi = state[f"{self.electrical}.phi"]
        q = state[f"{self.electrical}.q"]
        temperature = state[f"{self.thermal}.phi"]
        dp = phi[..., self._p] - phi[..., self._n]
        power = torch.real(dp * torch.conj(q[..., self._edge]))
        out: dict[str, Tensor] = {self.heat_key: power}
        for key, (slots, nodes) in self._slots.items():
            heated = temperature[..., nodes].real
            base = drivers.get(key, self.base_temperatures[key])
            base = torch.as_tensor(base).to(dtype=heated.dtype, device=heated.device)
            base = base.expand(*heated.shape[:-1], base.shape[-1])
            out[key] = base.index_copy(-1, slots.to(heated.device), heated)
        return out
