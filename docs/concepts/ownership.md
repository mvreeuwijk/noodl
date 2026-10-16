# Mutation, devices and serialization

This page states what you may change after a model is built, how a model is moved to another
device or dtype, and what survives copying and saving. It describes what the code does today.
Anything listed as *not supported* or *not moved* is not guarded unless the text says it is
refused.

The statements below are pinned by `tests/test_ownership.py`.

## What owns what

| Object | Kind | Owns |
|---|---|---|
| `Network` | plain Python object | the graph (`net.graph`, a `networkx.MultiDiGraph`), node and edge attributes, cached operator tensors, `dtype` and `device` |
| Layers (`KirchhoffLayer`, `TransportLayer`, `AllocatedFlowLayer`) | plain Python objects, **not** `torch.nn.Module` | a reference to the network, plus index tensors, capacities and counts copied from it at construction, and the network's topology version, device, dtype and attribute fingerprint of that time |
| Constitutive laws, node sources | `torch.nn.Module` | their coefficients (parameters when `learnable=True`, otherwise plain tensors or buffers) |
| `Model` | plain Python object, **not** a `torch.nn.Module` | the layer mapping, the closure list, the reaction list and the coupling options |
| Closures, drives, reactions | arbitrary callables or plain objects | whatever they close over |
| State and drivers | dictionaries of tensors | passed in on every call; the model keeps none |

## Mutation after construction

**Guaranteed: growing the network is refused.** A layer records `Network.topology_version`
when it is built. If a node or an edge has been added since, every public entry point of the
layer, and so `Model.step`, `steady` and the other solves, raises `RuntimeError` ("the network
topology has changed"). Re-adding a node that already exists, to set attributes, does not count
as a change (it is an attribute edit, below). Add every node and edge first, then build the
layers; to change the graph, build the layers and the model again.

**Guaranteed: the graph is frozen once a layer is built.** Building a layer freezes
`net.graph` (`Network.freeze`, the `networkx.freeze` mechanism): a direct structural change,
`net.graph.add_node(...)`, `add_edge`, `remove_node`, `remove_edge`, `clear`, raises
`networkx.NetworkXError` naming `Network.add_node`. `Network.add_node` and
`Network.add_edge` keep working, bump the topology version and so make the layers refuse, as
above. `net.frozen` says whether it is frozen; a network no layer has been built on is not.
Copies (`deepcopy`, `pickle`) of a frozen network are frozen too.

**Guaranteed: an attribute edit is detected by `Model.check` and `Model.to`.** Layers and
laws copy node and edge attributes into tensors when they are built (for example
`orifice_law_from_edges` reads `Cd` and `area` once), so an edit afterwards,
`net.graph.nodes[n]["volume"] = ...` or `net.add_node(n, volume=...)` on an existing node,
does not reach them. Each layer records `Network.attribute_fingerprint()` (a digest of every
node and edge attribute, names and values, tensors by value) when it is built.
`Model.check()` reports an `attributes-changed` error for every layer built before the edit,
and `Model.to` refuses to move such a model. The check is one pass over the attributes, so
it is not made on every step: a model stepped after an attribute edit without a check still
uses the values of construction time. An attribute holding an object without a value-like
`repr` (a custom class) counts by its type only, so an edit inside it is not seen. Rebuild the
laws, layers and model after an edit.

**Guaranteed: law parameters are live.** A law is a module that the layer reads on every call.
Changing a parameter in place (`with torch.no_grad(): law.C.mul_(2)`, or an optimiser step)
changes the next step. Learnable parameters are registered on the law, so
`law.named_parameters()` lists them and `law.state_dict()` holds them.

**Not supported, not guarded:**

- Replacing or editing `model.layers`, `model.closures` or `model.reactions` after
  construction. `Model.__init__` validates names, keys and drivers once; an appended closure
  runs without the checks against the layers (its declarations are read on first use, which
  checks its call signature against `integrates`, but its `outputs`, `state_keys` and
  `species_columns` are not checked against the layers). This page, not the closure contract
  in [Extending](../development/extending.md#the-closure-contract), is what governs a
  closure's lifecycle; units live with the keys (`noodl.refs.Field`, a closure's
  `input_specs`).
- Changing a layer's private tensors (`cols`, the law slices) or its list of laws.

**Frozen dataclasses are shallow.** The application records (`Resistor`, `Circuit`,
`SewerNetwork`, `Manhole`, ...) are `@dataclass(frozen=True)`. That forbids assigning an
attribute, nothing more: a list, dict or tensor inside one can still be edited, and
`SewerNetwork.notes` is a dict that applications and users do edit. `Circuit` converts its
`devices` to a tuple; do not assume every frozen record does. Treat a built record as a value
and make a new one to change it (`dataclasses.replace`).

## Devices and dtypes

**Guaranteed: `Model.to(device=None, dtype=None)` moves the whole model,** in place, and
returns it (`CoupledModel.to` does the same for every model of a union, and `layer.to` for
one layer with its network). One walk over the model moves:

- the network (`Network.to`; its cached operators are cleared and derived again on the new
  device and dtype);
- every layer's cached index, capacity and law-slice tensors, and its Modified Nodal Analysis
  structure;
- the laws and node sources (parameters move in place, so they stay the same
  `nn.Parameter` objects an optimiser holds; buffers and plain tensor attributes move too);
- drives, reactions and every closure of a class defined in noodl, attribute by attribute
  (built-in closures need no hook of their own);
- a user closure (a class defined outside noodl) that defines `to(device=None, dtype=None)`:
  that method is called. This is the hook for a closure that holds tensors;
- the tensors an application builder attaches to the model (`model.junction_idx`,
  `model.out_pipe`, `model.initial_flows`, a `notes` dict...), and those inside dicts, lists,
  tuples and noodl records reached from it. Dicts and lists are updated in place, so a dict
  shared between a closure and the model stays one dict.

`dtype` sets the precision. A floating tensor takes `dtype`; a complex tensor stays complex at
that precision (`float32 <-> complex64`, `float64 <-> complex128`), so a phasor layer on a
float32 network solves in complex64; an integer or boolean tensor (an index, a mask) keeps its
dtype. A stored `torch.dtype` or `torch.device` attribute is mapped the same way. Moving twice
to the same place changes nothing, and moving a float64 model to float64 gives bitwise the same
results. `Model.to` refuses a model whose network changed topology or attributes since its
layers were built.

**Guaranteed: `Network.to` on a built network makes its layers refuse.** `Network.to` moves the
network only. A layer records the network's device and dtype with its topology version and
refuses to run once they differ ("the network was moved ... Move the whole model with
Model.to"). `Model.to` repairs it.

State and drivers are the caller's: pass them on the new device and dtype. Newton's default
tolerances follow the residual's dtype, but an application that pins its own (`water_steady`'s
1e-11, the building and electrical solves' tight defaults) needs looser ones in float32
(`atol=rtol=1e-5` or so).

`tests/test_ownership.py` pins float32 moves of a building model, a water model and an
electrical model (DC and phasor) against their float64 results, and a CUDA move when a GPU is
present.

**Not moved:**

- A closure that is a plain function, a lambda or a `functools.partial`: the tensors it closes
  over stay where they are. Write it as a class with a `to` method, or build it on the target
  device and dtype.
- An object from outside noodl without a `to` method, a NumPy array, and the node and edge
  attributes of the graph (source data, copied into tensors when the layers were built).
- Tensors an application creates at run time with a fixed dtype. The transport layer computes
  its step in float64 whatever its storage dtype and returns the state's dtype, by design. The
  building application's controls (`apps.building_physics.inputs.scatter`) write their
  actuator values in float64, so a building model with controls does not run in float32.

## Copying and serialization

- **`copy.deepcopy(model)` is supported** and gives an independent model: its own network, its
  own layers and laws. The copy's topology guard refers to the copy's network, so growing the
  original network does not affect the copy.
- **`pickle` and `torch.save` of a whole model work** when every part is picklable. The
  built-in layers, laws, drives and reactions are. A closure is an arbitrary callable: a
  module-level function pickles, a `lambda` or a locally defined function does not.
- **A copy keeps integrating closures integrating.** The model reads each closure's
  declarations once into a `ClosureSpec` that holds the closure object, so `deepcopy` and
  `pickle` carry the closure and its declarations together and the copy's `step` advances the
  closure's state exactly as the original does
  (`tests/test_ownership.py::test_a_copied_model_still_advances_an_integrating_closure`).
- **`state_dict()` exists only on laws and node sources** (they are modules). It holds
  parameters and buffers, nothing else: not the law kind, the layer, the network or any
  attribute. A `Model` has no `state_dict`; there is no one-call way to save and restore a
  calibrated model other than pickling it, or saving each law's `state_dict` and rebuilding the
  rest from the same builder code.
- **Attributes attached to a model are plain Python attributes.** An application builder sets
  some after construction, for example `model.notes` (a dict that the sewer hydraulics keeps
  updating) and `model.junction_idx`. They travel with `deepcopy` and `pickle` of the model,
  and with it only. They appear in no `state_dict` and no file written by noodl. A copy keeps
  the sharing that existed (a `notes` dict that is also a closure's `notes` stays one dict
  inside the copy).
- **Layer metadata** (`quantity`, `unit`, `flow_unit`, `node_hints`) is read by `model.refs`
  and in error messages, never by the numerics. It lives on the layer and is lost by any
  route that rebuilds the layer.

## Related

[Networks](networks.md), [Layers and models](layers.md), [Differentiability](differentiability.md).
