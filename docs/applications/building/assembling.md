# Assembling a building

How to describe the parts of a building: zones and walls, duct networks, air-handling systems and reusable components. The airflow laws they use are explained under [Airflow elements](elements.md), and how the layers fit together under [Physics](physics.md). A building can also be read from a [CONTAM project](contam.md) or a [Modelica export](modelica.md).

## Zones and walls

A `Zone(name, volume, T0=293.15, z_ref=0.0, wall=None)` is a room: `volume` in m³, `T0` the initial temperature in K, `z_ref` its reference height in m. A `WallMass(name, capacity, ua_zone, ua_ambient)` is a lumped wall node: heat capacity in J/K and conductances in W/K to the zone and to the ambient. `add_zone(net, zone, ambient="ambient")` adds the zone's air node and, if `zone.wall` is set, the wall node plus `kind="wall"` edges `zone → wall → ambient`.

`WallMass` refuses a non-positive conductance at construction (see [Walls](physics.md#walls) for why). The other node and edge attributes are listed with their units under [Network attributes](../../reference/catalogue/building_physics.md#network-attributes), and the functions that assemble a model in the [catalogue](../../reference/catalogue/building_physics.md#set-up).

## Duct networks

A duct system is described with `DuctNetwork` and then added to a `Network` or a
`Component`. It has three parts, built the way CONTAM builds them:

- a **junction** (`junction(name, z_ref=, T0=)`) is a node at height `z_ref`. It holds half
  the volume of every segment that meets it (the segment's length times its area). A
  terminal adds nothing to it. The volume is why the system is described first and added
  afterwards: it is known only once every segment is;
- a **segment** (`segment(a, b, length=, diameter=, roughness=, loss=, area=)`) is an
  `AirDuct` edge between two junctions, with the hydrostatic term `DuctStack` (the column at
  the junctions' mean temperature). The roughness defaults to CONTAM's 9e-5 m
  (`DEFAULT_ROUGHNESS`) and the area to the circle of the diameter;
- a **terminal** (`terminal(junction, zone, area=, loss=)`) is a `DuctTerminal` edge from a
  junction into a zone, with a `Stack` at the junction's height. Its loss defaults to
  CONTAM's 0.125 (`DEFAULT_TERMINAL_LOSS`). A terminal in the ambient takes the wind as an
  envelope path does: from the `WindProfile` numbered `profile` at the wall's `azimuth` (or a
  component's `facade`), or from a constant pressure coefficient `Cp`, either scaled by the
  speed modifier `Ch`. Wind data that would act on nothing is refused, naming the terminal:
  `Ch`, `azimuth` or `facade` without a `profile` or with `Cp=0`, a `profile` and a `Cp`
  together, and wind on a terminal into any zone but the ambient (`DuctNetwork(ambient=...)`
  names it, `"ambient"` by default; inside a component, `outer` of that name).

Every value is checked when the segment or terminal is described, and an error names it (its
`name`, or its two ends): lengths, diameters, roughnesses and areas must be positive, a
roughness below 3.7 diameters, a segment's loss non-negative and a terminal's positive.

The laws are those under [Duct elements](elements.md#duct-elements). A fan, a damper or any other
element in a duct is an ordinary edge between two junctions, added by the caller with its own
law. `duct_laws_from_edges(net, profiles=...)` returns the segments' and terminals'
laws and head terms, with a `Wind` term when a terminal has a profile or a nonzero `Cp`.

A fan draws 0.05 kg/s from the ambient through a 10 m duct and blows it through a 5 m duct
into a room, which leaks it back to the ambient:

```python
import torch
from noodl.apps.building_physics import DuctNetwork, duct_laws_from_edges
from noodl.elements import FixedFlow, PowerLaw
from noodl.heads import EdgeHeads
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

F64 = torch.float64
net = Network(dtype=F64)
net.add_node("ambient", z_ref=0.0, T0=293.15)
net.add_node("room", z_ref=0.0, T0=293.15, volume=300.0)
ducts = DuctNetwork()
for j in ("inlet", "fan_in", "fan_out", "outlet"):
    ducts.junction(j, z_ref=1.5)
ducts.segment("inlet", "fan_in", length=10.0, diameter=0.2)
ducts.segment("fan_out", "outlet", length=5.0, diameter=0.2)
ducts.terminal("inlet", "ambient", area=0.0314)
ducts.terminal("outlet", "room", area=0.0314)
ducts.add_to(net)
net.add_edge("fan_in", "fan_out", kind="fan")
net.add_edge("room", "ambient", kind="leak")
laws, heads = duct_laws_from_edges(net)
laws += [FixedFlow(torch.tensor([0.05], dtype=F64), kind="fan"),
         PowerLaw(torch.tensor([0.5], dtype=F64), torch.tensor([0.5], dtype=F64),
                  kind="leak")]
# A model runs the EdgeHeads closure itself; a hand-built layer applies it to the drivers.
air = KirchhoffLayer(net, "air", laws, boundary=["ambient"])
rho = torch.full((net.n,), 1.204097, dtype=F64)
drivers = {"rho": rho}
drivers.update(EdgeHeads(air, heads)(None, drivers))     # writes "air.edge_head"
phi, q = air.solve(torch.zeros(1, dtype=F64), drivers)
print([round(v, 4) for v in air.flows_of_kind(q, "duct_terminal").tolist()], "kg/s")
print(f"room at {phi[net.node_index('room')].item():.4f} Pa")
print({j: round(v * 1e3, 1) for j, v in ducts.volumes().items()}, "litres")
```

```text
[-0.05, 0.05] kg/s
room at 0.0100 Pa
{'inlet': 157.1, 'fan_in': 157.1, 'fan_out': 78.5, 'outlet': 78.5} litres
```

The terminal in the ambient carries the fan's flow into the duct (negative: its positive
direction is out of the duct) and the room's terminal delivers it. Each junction holds half
of its segment: 157.1 litres for the 10 m duct of 0.2 m, 78.5 for the 5 m one. The junction
nodes also carry `heat_capacity=0`: no lumped heat capacity beyond their air.

From a `.prj`: `read_prj` reads a CONTAM duct network (duct elements, junctions, terminals,
segments; see [CONTAM](contam.md)) into the same kinds of edges, one kind per duct
element (`duct_<nr>`) and one for all terminals (`duct_terminal`). Its junctions are nodes
named `"junction <nr>"`, after the zones, so a species state carries the zones' rows first;
`project.zone_mass_fractions(x)` selects them. `project.terminal_flows(q)` is the flow out of
each terminal into its zone in junction-number order (ContamX's order and sign), and
`project.segment_flows(q)` the flow along each segment. `duct_three_zone.prj` supplies the
three-zone building from a wind-exposed intake through a fan and exhausts zone three through a
constant-volume fan:

```python
from noodl.apps.building_physics import project_to_model, read_prj
from noodl.apps.building_physics.prj import steady

project = read_prj("duct_three_zone.prj")
model, state, drivers = project_to_model(project)
solved = steady(model, state, drivers)
print(project.junctions[:3], "...")
print([round(v, 4) for v in project.terminal_flows(solved["air.q"]).tolist()], "kg/s")
x = solved["species.x"]
print(tuple(x.shape), tuple(project.zone_mass_fractions(x).shape))
```

```text
['junction 1', 'junction 2', 'junction 3'] ...
[-0.3169, 0.2088, 0.108, -0.0482, 0.0482] kg/s
(12, 1) (3, 1)
```

The intake (junction 1) draws 0.317 kg/s from the ambient, split between zone one (0.209) and
zone two (0.108); the exhaust takes 0.04 m3/s of 293.15 K air, 0.0482 kg/s, from zone three.
`simulate_project` returns the terminal flows of every step as `"terminal_flow"`.

## Air-handling systems

CONTAM's simple air-handling system is not a black box in noodl: `add_air_handler` expands it
into the network. Its supply plenum and return plenum are nodes, with their volumes. Edges of
one kind join them: a supply point from the supply plenum into each supplied zone, a return
point from each zone it draws from into the return plenum, the recirculation from the return
to the supply plenum, the outdoor air from the ambient into the supply plenum, and the exhaust
from the return plenum to the ambient. Every edge's flow is prescribed (a `PrescribedFlow`
law). The closure `AirHandlerFlows` computes the flows each step, by CONTAM's algorithm
(TN 1887 section 3.6, checked against ContamX 3.4.1.7), from the supply and return points'
design flows and the outdoor-air fraction $f_o$. With $W_s$ the supplied flow, $W_{ret}$ the
returned flow and $W_{o,min}$ the minimum outdoor air (kg/s):

$$
W_o' = \max\left(f_o W_s,\ \min(W_s, W_{o,min})\right), \qquad
W_{rec} = \min(W_{ret},\ W_s - W_o'),
$$

$$
W_o = W_s - W_{rec}, \qquad W_{ex} = W_{ret} - W_{rec}.
$$

When more air returns than is supplied, the surplus leaves through the exhaust. A minimum
outdoor-air flow above the supply is capped at the supply. An outdoor-air fraction above 1
would make the recirculation negative: ContamX stops there, and the closure raises, naming
the air handler. A negative fraction is no error: it acts through the maximum.

A schedule or a control acts through two inputs, declared in `model.refs.inputs`:
`"flow_multiplier.<kind>"` (one value per edge of the kind) multiplies the supply and return
points' design flows and is ignored on the other three edges, and
`"outdoor_air_fraction.<kind>"` sets $f_o$. Without them the multipliers are 1 and $f_o$ is
the `outdoor_air_fraction` given to `add_air_handler` (1 by default, all outdoor air). Both
take leading batch dimensions. An air handler whose multipliers are all 0 moves no air; its
plenums keep their last values.

Only these balanced flows touch the plenums, so the airflow solve does not need their
pressure: `build_model(flow_balanced=plenums)` puts them on the air layer's boundary at zero
gauge pressure, while they stay ordinary nodes of the heat and species layers. An edge whose
law depends on the pressure (a leak into a plenum, say) would let a pinned plenum absorb
an imbalance unseen, and a fixed flow lies outside the air handler's balance, so
`build_model` refuses any edge but a `PrescribedFlow` on a `flow_balanced` node, naming the
node and the edge; CONTAM, too, joins only the air handler's own paths to its plenums. They are
well-mixed nodes with air mass $\rho V$. A plenum of zero volume (CONTAM's default) holds no
air and must also be named in `storage_free=`: its balance is then algebraic, the mix of what
flows in, and the heat and species layers use the backward Euler scheme (`build_model` selects it
for heat; the species layer's default is already implicit). A plenum left out of
`flow_balanced=` or `storage_free=` is refused, naming it and the option. While an air
handler is switched off, nothing passes through a zero-volume plenum: it keeps its value, and
a heat or contaminant source there (a coil) is dropped for those steps, as it delivers
nothing without airflow; `model.step(..., diagnostics=d)` counts the dropped entries at
`d["layers"][layer]["held_sources_dropped"]`.
`closures=[closure]` adds the air handler's closure after the density closure. In a
`Component`, the outdoor air and the exhaust reach the shared ambient through
`outer("ambient")`. Give each air handler of a building its own `kind` (and its own plenum
names).

An office supplied with 0.1 kg/s, 30 % outdoor air, and stepped once with outdoor air at a
mass fraction of 1e-3:

```python
import torch
from noodl.apps.building_physics import (
    add_air_handler, air_handler_from_edges, air_handler_plenums, build_model,
    initial_drivers, initial_state, orifice_law_from_edges,
)
from noodl.heads import Stack
from noodl.topology import Network

F64 = torch.float64
net = Network(dtype=F64)
net.add_node("ambient", z_ref=0.0, T0=293.15)
net.add_node("office", z_ref=0.0, T0=293.15, volume=300.0)
net.add_edge("ambient", "office", kind="airpath", z_path=1.0, Cd=0.6, area=0.01)
add_air_handler(net, supplies={"office": 0.1}, returns={"office": 0.1},
                outdoor_air_fraction=0.3)
handler, closure = air_handler_from_edges(net, "air_handler")
plenums = air_handler_plenums(net, "air_handler")
model = build_model(net, air_laws=[orifice_law_from_edges(net, "airpath"), handler],
                    heads=[Stack.from_network(net, "airpath")], thermal=False, species=1,
                    closures=[closure], flow_balanced=plenums, storage_free=plenums)
drivers = initial_drivers(model)
drivers["rho"] = torch.full((net.n,), 1.2041, dtype=F64)
flows = closure({}, drivers)["prescribed_flow.air_handler"]
print([round(v, 3) for v in flows.tolist()])
sp = model.refs.species
drivers[sp.boundary_mass_fraction] = torch.tensor([1e-3], dtype=F64)  # outdoor air
state = model.step(initial_state(model), drivers, 300.0)
x = sp.mass_fraction.named(state[sp.mass_fraction])
print({node: f"{v.item():.4e}" for node, v in x.items()})
```

```text
[0.1, 0.1, 0.07, 0.03, 0.03]
{'office': '2.4309e-05', 'ahs_supply': '3.1702e-04', 'ahs_return': '2.4309e-05'}
```

The flows are in edge order: supply, return, recirculation, outdoor air, exhaust. The office
exchanges 0.03 kg/s with the outdoor air through the air handler. The zero-volume return
plenum is at the office's value, and the supply plenum at the mix, 0.3 of the outdoor air plus
0.7 of the returned air.

From a `.prj`: `read_prj` reads CONTAM's simple air-handling systems (see
[CONTAM](contam.md)) into the same edges, one kind per system (`ahs_<nr>`), with
its closure in `project.closures`, its plenum zones in `project.flow_balanced` and the
zero-volume ones in `project.storage_free`; `project_to_model` passes all three on. The
supply and return points carry their design flows, and their schedules become
`"flow_multiplier.ahs_<nr>"` inputs. The outdoor-air fraction schedule and the minimum
outdoor-air flow are read from the recirculation path, where ContamX reads them. Filters on
any of the system's paths act on the air passing through, as on any path.
`ahs_three_zone.prj` supplies zones one (0.1 kg/s) and two (0.05 kg/s) and draws 0.12 kg/s
from zone three, with 30 % outdoor air and a minimum of 0.02 kg/s; its return plenum has no
volume:

```python
from noodl.apps.building_physics import project_to_model, read_prj
from noodl.apps.building_physics.prj import steady

project = read_prj("ahs_three_zone.prj")
print(project.flow_balanced, project.storage_free)
model, state, drivers = project_to_model(project)
solved = steady(model, state, drivers)
print([round(v, 3) for v in project.path_flows(solved["air.q"])[4:].tolist()], "kg/s")
```

```text
['ahs(Ret)', 'ahs(Sup)'] ['ahs(Ret)']
[0.1, 0.05, 0.12, 0.105, 0.045, 0.015] kg/s
```

Paths 5 to 10 are the two supply points, the return point, the recirculation, the outdoor air
and the exhaust: 30 % of the 0.15 kg/s supplied is 0.045 kg/s of outdoor air, and the
0.12 kg/s returned splits into 0.105 kg/s recirculated and 0.015 kg/s exhausted.

## Building from components

A larger building is easier to assemble from parts than to write out node by node. The
functions below return components (see [Components](../../guide/components.md)): you place
them, connect their ports, and `flatten()` gives the same `Network` you would have built by
hand, ready for `build_model`.

| Factory | Parameters | Ports |
|---|---|---|
| `room(name, ...)` | `volume` (m3), `T0` (K), `z_ref` (m), `wall` (a `WallMass`, optional) | `air` |
| `door(name, ...)` | `H`, `W` (m), `z_mid` (m, relative to the placement), `Cd` | `a`, `b` |
| `crack(name, ...)` | `area` (m2), `z_path` (m), `Cd`, `exterior` (default `False`), wind: `azimuth` or `facade`, `Cp`, `Ch`, `profile` | `a`, `b`; only `a` if exterior |
| `window(name, ...)` | as `crack`, but `exterior` defaults to `True` | `a` (and `b` with `exterior=False`) |
| `shaft(name, ...)` | `levels`, `level_height` (m), `volume` (m3 per level), `area` (m2 of each slab opening), `T0`, `Cd` | `levels[0]`, `levels[1]`, ... |
| `fan(name, ...)` | `flow` (kg/s, not negative; swap the ports to reverse; the height does not matter to a fixed-flow fan), `exterior` (`None`, `"supply"` or `"exhaust"`) | `a`, `b`; only `a` if exterior |
| `duct(name, ...)` | `length`, `diameter`, `roughness` (m), `loss`, `terminal_area` (m2), `terminal_loss`, `z_a`, `z_b` (m), `T0` (K), `exterior`, wind (exterior only, needs `profile`): `azimuth` or `facade`, `Ch`, `profile` | `a`, `b`; only `b` if exterior |
| `air_handler(name, ...)` | `supplies`, `returns` (design flows, kg/s), `outdoor_air_fraction`, `min_outdoor_air` (kg/s), `supply_volume`, `return_volume` (m3), `T0` (K), `kind` | `supplies[i]`, `returns[j]` |

A `door` has two edges, `low` and `high`, as `add_large_opening` does. An exterior `crack` or
`window` has one edge, `path`, from the shared `ambient` node, so the building must declare
`inner("ambient", ...)`. A `shaft` is a stack of zones, one per level, joined by openings
`slab[i]` (how stairwells are usually represented in multizone models). A door's or window's height is
relative to the component's placement, so the same factory call serves every floor.
`add_air_handler` adds an air handler to a component as it does to a network (see
[Air-handling systems](#air-handling-systems)).

A stairwell serving two flats, one per floor. A door joins each shaft level to its flat:
connecting `stair.levels[i]` straight to `flat.air` would merge two real nodes, which
`flatten` refuses (see [Ports and connect](../../guide/components.md#ports-and-connect)).

```python
import torch

from noodl.apps.building_physics import door, room, shaft, window
from noodl.components import Component

F64 = torch.float64

block = Component("block")
block.inner("ambient", z_ref=0.0, T0=283.15)
stair = block.add(shaft("stair", levels=2, level_height=3.0, volume=30.0, area=0.5))
for i in range(2):
    at = (0.0, 0.0, 3.0 * i)
    flat = block.add(room(f"flat{i}", volume=60.0), at=at)
    d = block.add(door(f"door{i}", H=2.0, W=0.9, z_mid=1.0), at=at)
    win = block.add(window(f"win{i}", area=0.02, z_path=1.5, azimuth=180.0), at=at)
    block.connect(stair.ports[f"levels[{i}]"], d.ports.a)
    block.connect(d.ports.b, flat.ports.air)
    block.connect(win.ports.a, flat.ports.air)
block_net, block_names = block.flatten(dtype=F64)
print(block_net.nodes)
```

```text
['ambient', 'stair.levels[0]', 'stair.levels[1]', 'flat0.air', 'flat1.air']
```

The second flat, its door and its window are placed at `z = 3`, so their heights come out
right without being repeated in the factory calls.

A `fan` edge is built with `fan_law_from_edges`, a `duct` with `duct_laws_from_edges`,
and an `air_handler` with `air_handler_from_edges` and `air_handler_plenums` (see
[Air-handling systems](#air-handling-systems)). An office is served by an air handler, joined
to a store by a transfer duct, and the store is exhausted by a fan:

```python
import math

import torch

from noodl.apps.building_physics import (
    air_handler,
    air_handler_from_edges,
    air_handler_plenums,
    build_model,
    crack,
    duct,
    duct_laws_from_edges,
    fan,
    fan_law_from_edges,
    initial_drivers,
    initial_state,
    orifice_law_from_edges,
    room,
)
from noodl.components import Component
from noodl.heads import Stack

F64 = torch.float64

b = Component("building")
b.inner("ambient", z_ref=0.0, T0=293.15)
office = b.add(room("office", volume=300.0))
store = b.add(room("store", volume=100.0))
ahu = b.add(air_handler("ahu", supplies=[0.1], returns=[0.1], outdoor_air_fraction=0.3,
                        supply_volume=0.0, return_volume=0.0))
b.connect(ahu.ports["supplies[0]"], office.ports.air)
b.connect(ahu.ports["returns[0]"], office.ports.air)
transfer = b.add(duct("transfer", length=4.0, diameter=0.15, z_a=2.5, z_b=2.5))
b.connect(transfer.ports.a, office.ports.air)
b.connect(transfer.ports.b, store.ports.air)
exhaust = b.add(fan("exhaust", flow=0.02, exterior="exhaust"))
b.connect(exhaust.ports.a, store.ports.air)
leak = b.add(crack("leak", area=0.01, z_path=1.0, exterior=True))
b.connect(leak.ports.a, office.ports.air)

net, _ = b.flatten(dtype=F64)
handler, closure = air_handler_from_edges(net, "air_handler")
plenums = air_handler_plenums(net, "air_handler")
ducts, duct_heads = duct_laws_from_edges(net)
model = build_model(
    net,
    air_laws=[orifice_law_from_edges(net, "airpath"), *ducts,
              fan_law_from_edges(net), handler],
    heads=[Stack.from_network(net, "airpath"), *duct_heads],
    thermal=False, species=1, closures=[closure],
    flow_balanced=plenums, storage_free=plenums,
)
drivers = initial_drivers(model)
drivers["rho"] = torch.full((net.n,), 1.2041, dtype=F64)
out = model.steady(initial_state(model), drivers, atol=1e-14, rtol=1e-14)
air = model.kirchhoff["air"]
print([round(v, 4) for v in air.flows_of_kind(out["air.q"], "duct_terminal").tolist()])
print([round(v, 4) for v in air.flows_of_kind(out["air.q"], "air_handler").tolist()])
```

```text
[-0.02, 0.02]
[0.1, 0.1, 0.07, 0.03, 0.03]
```

The transfer duct carries the fan's 0.02 kg/s from the office to the store. Of the air handler's
0.1 kg/s supply, 0.03 kg/s is outdoor air and 0.07 kg/s recirculated.
