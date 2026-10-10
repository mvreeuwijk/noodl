# Components

A component is a piece of network with a declared boundary of ports. It holds its own nodes
and edges, and it can hold other components, so a building is made of floors, a floor of
rooms, a room of its air and its walls. You build the tree in plain Python, in the way you
would wire blocks in Simulink or Grasshopper, and `flatten()` turns it into one ordinary
[`Network`](networks.md) with a `NameMap` that remembers which component each node and edge
came from. The solver never sees components: layers, elements and drives work on the
flattened network exactly as they do on one built by hand.

Components live in `noodl.components`. The building application adds factory functions that
return components (`room`, `door`, `crack`, `window`, `shaft`); see
[Building physics](../applications/building_physics.md#building-from-components).

## A house from rooms and a door

Two rooms and a door between them, with two leaks from room B to the outside:

```python
import torch
from noodl.apps.building_physics import crack, door, room
from noodl.components import Component

house = Component("house")
house.inner("ambient", z_ref=0.0, T0=283.15)
A = house.add(room("A", volume=60.0, T0=288.15))
B = house.add(room("B", volume=60.0, T0=285.15))
d = house.add(door("door", H=2.0, W=0.9, z_mid=1.0))
house.connect(A.ports.air, d.ports.a)
house.connect(d.ports.b, B.ports.air)
for name, z in (("leak_low", 0.2), ("leak_high", 1.8)):
    leak = house.add(crack(name, area=0.05, Cd=0.6, z_path=z, exterior=True))
    house.connect(leak.ports.a, B.ports.air)

net, names = house.flatten(dtype=torch.float64)
print(net.nodes)
print([net.graph.edges[e]["name"] for e in net.edges])
print(names.resolve("door.a"))
```

```text
['ambient', 'A.air', 'B.air']
['door.low', 'door.high', 'leak_low.path', 'leak_high.path']
A.air
```

This is the model of the [two-zone example](../applications/building_physics.md#building-a-model-in-python),
assembled from parts. Rooms `A` and `B` each bring their air node. The door brings two
edges, a low and a high opening, between two terminals. Each leak brings one edge, from
the shared `ambient` node to a terminal. `connect` joins each terminal to a room, and
`flatten` returns a network of three nodes and four edges. `door.a` is not a node of the
flattened network; it was merged into `A.air`, and `resolve` tells you so.

## Ports and connect

A port is a name a component gives to one of its nodes so that the outside can reach it.
`room("A", ...)` has the port `air`; `door` has `a` and `b`; `shaft` has one port per level.
`c.ports.air` returns a reference to the port, and `list(c.ports)` lists them all.

`connect(p, q)` merges two ports into one node. This is the Modelica connection set: the
potentials of the ports are equal, and the flows into the merged node sum to zero, which is
the conservation law the layer already solves at every node. A port may appear in several
connects (the two leaks above both join `B.air`), and connects are transitive.

A door belongs to neither of its rooms. It has two terminals, `a` and `b`, which are bare
connection points with no attributes (`add_terminal` makes one). A terminal carries no
volume, no temperature and no height; it only exists to be merged into a real node. The
same door can therefore join any two rooms, and the same room can be joined to any number
of doors.

Two real nodes cannot be merged, because each carries its own attributes (a volume, an
initial temperature) and there is no right way to combine them. Connecting `A.air` straight
to `B.air` fails at `flatten()` with

```text
connecting merges 'A.air' and 'B.air' into one node, but each carries its own attributes:
two zones cannot be one air volume. Put a door, opening or other component between them.
```

(`flatten` collects every problem it finds and raises one `ComponentError` listing them.)

A terminal that nothing outside its component ever reaches is a closed opening. `flatten`
drops it together with its edges, so a door whose side `b` is never connected carries no
flow. `model.check(state, drivers, names=names)` lists such ports as information lines, so
a forgotten `connect` does not go unnoticed (see
[Reading results](#reading-results-by-component)).

Only the ports of a direct child can be connected. Reaching into a grandchild is an error;
re-export the port from its parent under a new name instead:

```py
floor.expose(corridor=corridor_room.ports.air)   # now floor.ports.corridor exists
```

`expose("air")` with a plain name exposes one of the component's own nodes.

## Names

Every node and edge of the flattened network is named by its path: the names of the
components down to it, joined by dots, with the root component's own name left out. In the
house above, the air of room `A` is `A.air` and the low opening of the door is `door.low`.
Names may therefore not contain a dot. Arrays use brackets as part of the name:
`shaft("stair", levels=3, ...)` has the nodes `stair.levels[0]`, `stair.levels[1]` and
`stair.levels[2]`.

When `connect` merges several ports into one node, the node keeps the path of its real node
(`A.air`), and every other port path that was merged into it, such as `door.a`, becomes an
alias. Aliases are accepted wherever a node name is: in `drivers_from`, in `initial_drivers`
values, in `refs` selectors and in `node_index`. `named()` always returns the canonical
names, so `th.sources: {"door.a": 1000.0}` heats `A.air`, and the result prints as `A.air`.

## Heights and placement

`parent.add(child, at=(x, y, z))` places a child at `(x, y, z)` metres relative to its
parent; placements add up down the tree. The default is the origin. In CONTAM the height of
a zone or a path is given relative to the level it sits on; in a component tree the level
is a component placed at its elevation, and everything inside it is relative to that.

Which attributes shift with the placement is registered by the application, because the
core does not know what a height is called. The building application registers `z_ref` on
nodes and `z_path` on edges. `flatten` adds the accumulated `z` of the component to them, so
a room placed at `z = 3` with `z_ref = 0.5` ends up at `z_ref = 3.5`. A node that has no
`z_ref` of its own sits at its component's placement. `x` and `y` change nothing physical;
they are stored as the node attribute `position` for drawing.

If a tree is placed above zero but no height attributes are registered, `flatten` warns
that no height was shifted, and names the ways to fix it: import the application that
registers them, call `noodl.components.register_elevations`, or pass `node_elevations=` and
`edge_elevations=` to `flatten`.

## Shared nodes and wind

Some nodes belong to the whole building rather than to any part of it: the ambient air, the
ground, a plenum. `inner(name, **attrs)` declares such a node on a component, and an edge
anywhere below reaches it with `outer(name)`. The nearest enclosing declaration wins, as in
Modelica, so a building can declare its own `ambient` and a district of buildings can
declare another one above it. An `outer` with no enclosing `inner` is an error at
`flatten`. The exterior openings `crack(..., exterior=True)` and `window` are edges from
`outer("ambient")` to their one terminal.

Wind is specified per path, as CONTAM does it: an exterior opening carries the attributes
`Wind.from_network` reads (`azimuth`, `Cp`, `Ch`, `profile`). Repeating the azimuth on every
opening is how an orientation error creeps in, so the building can declare its facades once
with `inner_table("facades", south=180.0, north=0.0)` and an opening can name its facade:

```python
from noodl.apps.building_physics import window

tower = Component("tower")
tower.inner("ambient", z_ref=0.0, T0=283.15)
tower.inner_table("facades", south=180.0, north=0.0)
for i in range(2):
    f = tower.add(room(f"floor{i}", volume=60.0), at=(0.0, 0.0, 3.0 * i))
    w = tower.add(window(f"win{i}", area=0.01, z_path=1.0, facade="south", Cp=0.5),
                  at=(0.0, 0.0, 3.0 * i))
    tower.connect(w.ports.a, f.ports.air)
tower_net, _ = tower.flatten(dtype=torch.float64)
print(tower_net.nodes)
print(tower_net.node_attr("z_ref").tolist())
print(tower_net.edge_attr("z_path", kind="airpath").tolist())
print(tower_net.edge_attr("azimuth", kind="airpath").tolist())
```

```text
['ambient', 'floor0.air', 'floor1.air']
[0.0, 0.0, 3.0]
[1.0, 4.0]
[180.0, 180.0]
```

The second floor is placed at `z = 3`, so its window path at `z_path = 1.0` is at 4.0 m, and
both windows read their azimuth from the table: `flatten` writes the table value into the
edge attribute `azimuth`. Turning the building is one edit of the table followed by a new
`flatten()`. Giving both `facade=` and `azimuth=` on one opening is an error, as is a facade
that is not in the table, or a `facade=` with no `inner_table("facades", ...)` above it.
Wind attributes on an interior crack are an error too: `crack("c", ..., facade="south")`
raises `c: wind attributes ['facade'] apply only to an exterior opening (exterior=True)`.

## Running it

Continue with the house. Nothing in the model code knows about components; it takes `net`:

```python
from noodl.apps.building_physics import (
    build_model, initial_drivers, initial_state, orifice_elements_from_edges,
)
from noodl.drives import Stack

el = orifice_elements_from_edges(net, "airpath")
model = build_model(net, air_elements=[el], drives=[Stack.from_network(net, "airpath")],
                    coupling="iterate", iterate_tol={"thermal": 0.01}, iterate_max=50)
air, th = model.refs.air, model.refs.thermal
rooms = [n for p in names.select(template="room") for n in names.nodes_of(p)]
state = initial_state(model)
drivers = initial_drivers(model, values={th.boundary_temperature: {"ambient": 283.15},
                                         th.sources: {"door.a": 1000.0}})
model.check(state, drivers, names=names).raise_for_errors(strict=True)
state = model.step(state, drivers, dt=60.0)
print(rooms)
print(sorted(th.temperature.named(state[th.temperature])))
print(abs(float(names.boundary_flows("B", "airpath", state[air.flow]))) < 1e-9)
```

```text
['A.air', 'B.air']
['A.air', 'B.air']
True
```

The heat source is given at the alias `door.a` and lands in room A. The last line says that
the net air flow into room B is zero to solver tolerance, as it must be for a room whose
air mass is fixed.

## Reading results by component

The `NameMap` answers questions by component path. The root is the empty string `""`.

- `names.nodes_of(path)` and `names.edges_of(path)`: the nodes and edges owned by the
  component and everything below it, in network order.
- `names.select(template="room")`: the paths of every component made by a factory. A factory
  records its name as the component's `template`.
- `names.boundary_edges(path, kind)`: the edges of one kind with exactly one end inside the
  component. The result holds their names, their positions in `net.edge_index(kind)` order
  and a sign, +1 for an edge pointing into the component.
- `names.boundary_flows(path, kind, flows)`: the net inflow through those edges, for a
  flow tensor in `net.edge_index(kind)` order. These are the flows a surrogate model of the
  component would have to reproduce at its boundary, so they are the natural training target.
- `names.tree()`: the hierarchy as plain nested dictionaries (path, name, template,
  position, ports, own nodes and edges, children), ready to print or to draw.
- `model.check(state, drivers, names=names)`: the usual setup report, plus one line for each
  port that nothing was connected to. A closed port is legitimate (a closed door), so it is
  information, not a warning.

A path that does not exist raises a `KeyError` with the closest matches, for example
`unknown component 'Bb'; did you mean 'B'?`.

## How this relates to Modelica and CONTAM

| Here | Modelica | CONTAM |
|---|---|---|
| a factory function (`room`, `door`) | a class | a zone or path type |
| its keyword arguments | modifiers | the properties of the zone or path |
| `connect(p, q)` | a connection set | no counterpart |
| `inner` / `outer` | `inner` / `outer` | the ambient, implicit |
| a component placed at `z` with relative heights | a position in the parent | a level and a height relative to it |
| wind attributes on each exterior path, `inner_table("facades", ...)` | an `outer` weather bus | wind pressure on each path, by wall azimuth |

CONTAM has no components. A building is one flat list of zones and paths, and its levels map
to the placement of a component here. Reading a CONTAM file therefore yields a flat network;
components are for building models in code. The Modelica Buildings Library has components,
and noodl follows its composition rules (connection sets, `inner`/`outer`), but it does not
generate Modelica classes; see [File formats](../formats/index.md) for what can be read.
