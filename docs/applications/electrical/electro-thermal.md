# Electro-thermal coupling

How to couple a circuit to a heat network: a device that heats up and whose resistance
follows its temperature, or a load whose power depends on a temperature. The circuit devices
are in [Build and analyse a circuit](circuits.md); the AC laws used in the second example are
in [AC analysis and power laws](ac.md).

noodl puts the heat side on the same `Network` as a heat layer: a `TransportLayer` with
nothing advecting (`flow_kind=()`), conventionally named `"thermal"`, whose state
`"thermal.x"` is the temperature (K) of its interior nodes. It is the same layer type the
building application uses for its zones and walls.

| Thermal part | In the heat layer | Modelica (MSL `Thermal.HeatTransfer`) |
|---|---|---|
| heat conductance `G` (W/K) | a conduction edge (`conduction_kind`, `conductance=G`) | `Components.ThermalConductor` |
| heat capacity `C` (J/K) | the `capacity` of its node | `Components.HeatCapacitor` |
| fixed temperature | a boundary node, its value in `"thermal.x_boundary"` | `Sources.FixedTemperature` |
| a device's Joule heat | the source `"thermal.sources"` (W) at its heat-port node | the device's `heatPort` |
| a node without heat capacity | `storage_free`: folded into its neighbours, its temperature their balance | a heat port between conductors |

## Couple with `JouleHeating`

`JouleHeating(heated, thermal, base_temperatures=...)` is the closure that couples the two
layers. Each `HeatedEdge(edge, p, n, heat_node, temperature_key, slot)` names a heated
device: its edge in the circuit layer's `q`, its end nodes, the network node of its heat
port, and the per-edge temperature driver its law reads
(`Resistance(..., temperature=key)`, `JunctionDiode(..., temperature=key)`) with its
position in it. From the state the closure writes the dissipated power
`p = Re((phi[p] - phi[n]) conj(q[edge]))` (MSL `LossPower = v i`) into the heat layer's
sources `"thermal.sources"` (full node order; a heat port on a fixed temperature hands its
heat to that boundary), and each temperature driver as the driver you give (or
`base_temperatures[key]` when the drivers lack it) with the heated devices' entries
replaced by their heat-port temperatures (devices without a heat port keep the temperature
you give). Build the model with `coupling="iterate"`, so that every step (or steady solve)
repeats circuit and heat solves until both stop changing. Anderson mixing
(`iterate_acceleration="anderson"`) mixes the heat layer's temperatures with the circuit's
potentials and currents and judges their fixed-point residual like the circuit's, as it does
for every conduction-only transport layer named in `iterate_tol`.

A resistor `R(T) = R0 (1 + alpha (T - T_ref))` on 220 V, cooled through `G = 50` W/K to
293.15 K, settles where its Joule heat `V^2 / R(T)` equals `G (T - 293.15)`, a quadratic in
the temperature rise `x`: `alpha x^2 + x - V^2 / (G R0) = 0`. Its heat port holds no heat, so
it is storage-free:

```python
import math

import torch

from noodl.apps.electrical import HeatedEdge, JouleHeating, PrescribedVoltage, Resistance
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network


def f64(x):
    """A float64 parameter or driver (a Python float would be stored as float32)."""
    return torch.tensor(x, dtype=torch.float64)


V, R0, alpha, T_ref, G, T_amb = 220.0, 100.0, 1e-3, 293.15, 50.0, 293.15
net = Network(dtype=torch.float64)
for name in ("0", "s", "h", "ambient"):
    net.add_node(name)
net.add_edge("s", "0", kind="voltage_source")
net.add_edge("s", "0", kind="resistor")
net.add_edge("h", "ambient", kind="conductor")
node = {name: k for k, name in enumerate(net.nodes)}

electrical = KirchhoffLayer(
    net, "electrical",
    [PrescribedVoltage(f64(V)),
     Resistance(f64(R0), alpha=f64(alpha), T_ref=f64(T_ref),
                temperature="electrical.resistor.T")],
    boundary=["0"])
thermal = TransportLayer(
    net, "thermal", flow_kind=(), boundary=["ambient"], capacity=f64([0.0]),
    conduction_kind="conductor", conductance=f64([G]), storage_free=["h"],
    quantity="temperature", unit="K", source_unit="W", capacity_unit="J/K",
    conductance_unit="W/K")
joule = JouleHeating(
    [HeatedEdge(edge=1, p=node["s"], n=node["0"], heat_node=node["h"],
                temperature_key="electrical.resistor.T", slot=0)],
    thermal, base_temperatures={"electrical.resistor.T": f64([T_ref])})
model = Model(net, [electrical, thermal], closures=[joule], coupling="iterate",
              iterate_tol={"electrical": 1e-10, "thermal": 1e-10},
              iterate_flow_tol={"electrical": 1e-10}, iterate_acceleration="anderson",
              iterate_max=100)

state = {"electrical.phi": torch.zeros(len(net.nodes), dtype=torch.float64),
         "electrical.q": torch.zeros(2, dtype=torch.float64),
         "thermal.x": f64([T_amb]), "thermal.sources": torch.zeros(len(net.nodes),
                                                                   dtype=torch.float64)}
drivers = {"electrical.phi_boundary": f64([0.0]), "thermal.x_boundary": f64([T_amb]),
           "electrical.resistor.T": f64([T_ref])}
out = model.steady(state, drivers)

rise = (-1.0 + math.sqrt(1.0 + 4.0 * alpha * V**2 / (G * R0))) / (2.0 * alpha)
print(f"resistor at {float(out['thermal.x'][0]):.9f} K "
      f"(closed form {T_amb + rise:.9f} K)")
print(f"current {float(out['electrical.q'][1]):.9f} A")
```

```text
resistor at 302.738068934 K (closed form 302.738068934 K)
current 2.179106576 A
```

`"thermal.x"` holds the temperatures of the layer's interior nodes in network order
(`thermal.interior_idx`), `"thermal.x_boundary"` the fixed temperatures in the order of
`boundary=`. The state carries `"thermal.sources"` too: `JouleHeating` declares it as state
(`state_keys`), so a state holds the heat dissipated at its own time.

## Step with the circuit's rule

A circuit steps by backward Euler after a restart and by the trapezoidal rule otherwise
(`scheme=`). Give the heat layer the same rule on every step with
`layer_options={"thermal": {"scheme": "backward_euler"}}` (backward Euler) or
`{"scheme": "trapezoidal"}`. On a trapezoidal step the heat layer integrates the heat as the
circuit's capacitors and inductors integrate their currents, `dt/2 (P_start + P_end)`,
`P_start` being the step-start state's `"thermal.sources"`; that is the same Crank-Nicolson
rule, so the two layers agree on what a step is. The [Modelica reader](modelica.md) builds
this structure for MSL models whose resistors or diodes have an enabled heat port
(`Analog.Examples.Resistor`, `HeatingRectifier`), and `simulate_circuit` steps it so.

## Example: a heat pump's electrical load

The coupling closure need not be `JouleHeating`: any function of the state that writes
drivers couples the two layers. A heat pump delivers a fixed 6 kW of heat to a zone (heat
capacity 1e7 J/K, loss 150 W/K to 273.15 K outdoors) and draws the electrical power
`P = Q_th / COP` from a 400 V DC bus through a 0.2 ohm feeder, with a coefficient of
performance `COP = 0.45 T_zone / (T_zone - T_out)` (45 % of the Carnot value) that falls as
the zone warms. The heat pump is a `ConstantPowerLoad` of 1 W whose `scale` driver the
closure sets to `P`; the heat input is the heat layer's source `"thermal.sources"` at the
zone. One `Network` holds both layers; the model steps a day in hours, backward Euler first:

```python
from noodl.apps.electrical import ConstantPowerLoad, Impedance

net = Network(dtype=torch.float64)
for name in ("0", "bus", "hp", "zone", "outdoor"):
    net.add_node(name)
net.add_edge("bus", "0", kind="voltage_source")
net.add_edge("bus", "hp", kind="impedance")
net.add_edge("hp", "0", kind="load")
net.add_edge("zone", "outdoor", kind="conductor")
node = {name: k for k, name in enumerate(net.nodes)}

electrical = KirchhoffLayer(
    net, "electrical",
    [PrescribedVoltage(f64(400.0)), Impedance(f64(0.2)),
     ConstantPowerLoad(f64(1.0), V_nominal=f64(400.0), direct_current=True,
                       scale="electrical.load.scale")],
    boundary=["0"])
thermal = TransportLayer(
    net, "thermal", flow_kind=(), boundary=["outdoor"], capacity=f64([1e7]),
    conduction_kind="conductor", conductance=f64([150.0]), quantity="temperature",
    unit="K", source_unit="W", capacity_unit="J/K", conductance_unit="W/K")


class HeatPumpLoad:
    """Delivers Q_th to the zone; draws P = Q_th / COP from the bus, COP = 0.45 T_zone/(T_zone - T_out)."""

    def __init__(self, q_th, t_out, zone, n_nodes):
        self.q_th, self.t_out, self.zone, self.n_nodes = q_th, t_out, zone, n_nodes

    def __call__(self, state, drivers):
        t_zone = state["thermal.x"][..., 0]
        cop = 0.45 * t_zone / (t_zone - self.t_out)
        heat = torch.zeros(t_zone.shape + (self.n_nodes,), dtype=t_zone.dtype)
        heat[..., self.zone] = self.q_th
        return {"thermal.sources": heat,
                "electrical.load.scale": (self.q_th / cop).unsqueeze(-1)}


model = Model(net, [thermal, electrical],
              closures=[HeatPumpLoad(f64(6000.0), 273.15, node["zone"], len(net.nodes))],
              coupling="iterate", iterate_tol={"thermal": 1e-9, "electrical": 1e-9})

phi = torch.zeros(len(net.nodes), dtype=torch.float64)
phi[[node["bus"], node["hp"]]] = 400.0
state = {"electrical.phi": phi, "electrical.q": torch.zeros(3, dtype=torch.float64),
         "thermal.x": f64([293.15])}
drivers = {"electrical.phi_boundary": f64([0.0]), "thermal.x_boundary": f64([273.15]),
           "electrical.load.scale": f64([1.0])}

for hour in range(24):
    first = hour == 0
    state = model.step(state, drivers, dt=3600.0, t=3600.0 * hour,
                       scheme="backward_euler" if first else "trapezoidal",
                       layer_options={"thermal": {"scheme": "backward_euler" if first
                                                  else "trapezoidal"}})

v = state["electrical.phi"]
print(f"zone after 24 h: {float(state['thermal.x'][0]):.2f} K")
print(f"heat pump draws {float(v[node['hp']] * state['electrical.q'][2]):.1f} W")
print(f"bus to heat pump drop: {float(v[node['bus']] - v[node['hp']]):.3f} V")
```

```text
zone after 24 h: 307.67 K
heat pump draws 1496.0 W
bus to heat pump drop: 0.749 V
```

The zone approaches its steady 313.15 K with the time constant `C / G` = 18.5 h; at 307.67 K
the COP is 4.01, so the 6 kW of heat cost 1.50 kW, carried as 3.75 A through the feeder. The
heat is constant, so this closure need not carry it as state. The same closure on a phasor
layer would set an AC load's power, with source amplitudes given as RMS values (see
[phasor conventions](physics.md#phasor-conventions)).
