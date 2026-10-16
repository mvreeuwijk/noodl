# Mutation, devices and serialization

This page states what you may change after a model is built, how a model is moved to another
device or dtype, and what survives copying and saving. It describes what the code does today.
Anything listed as *not supported* is not guarded unless the text says it is refused.

The statements below are pinned by `tests/test_ownership.py`.

## What owns what

| Object | Kind | Owns |
|---|---|---|
| `Network` | plain Python object | the graph (`net.graph`, a `networkx.MultiDiGraph`), node and edge attributes, cached operator tensors, `dtype` and `device` |
| Layers (`KirchhoffLayer`, `TransportLayer`, `AllocatedFlowLayer`) | plain Python objects, **not** `torch.nn.Module` | a reference to the network, plus index tensors, capacities and counts copied from it at construction |
| Constitutive laws, node sources | `torch.nn.Module` | their coefficients (parameters when `learnable=True`, otherwise plain tensors or buffers) |
| `Model` | plain Python object, **not** a `torch.nn.Module` | the layer mapping, the closure list, the reaction list and the coupling options |
| Closures, drives, reactions | arbitrary callables or plain objects | whatever they close over |
| State and drivers | dictionaries of tensors | passed in on every call; the model keeps none |

## Mutation after construction

**Guaranteed: growing the network is refused.** A layer records `Network.topology_version`
when it is built. If a node or an edge has been added since, every public entry point of the
layer, and so `Model.step`, `steady` and the other solves, raises `RuntimeError` ("the network
topology has changed"). Re-adding a node that already exists, to set attributes, does not count
as a change. Add every node and edge first, then build the layers; to change the graph, build
the layers and the model again.

**Guaranteed: law parameters are live.** A law is a module that the layer reads on every call.
Changing a parameter in place (`with torch.no_grad(): law.C.mul_(2)`, or an optimiser step)
changes the next step. Learnable parameters are registered on the law, so
`law.named_parameters()` lists them and `law.state_dict()` holds them.

**Not supported, not guarded:**

- Editing node or edge attributes (`net.graph[...]`, `net.add_node(name, volume=...)` on an
  existing node) after a layer or law has read them. Attributes are copied into tensors at
  construction (for example `orifice_law_from_edges` reads `Cd` and `area` once); the layer
  does not see the edit and does not refuse. Rebuild the law and layer.
- Changing `net.graph` directly (`net.graph.add_edge(...)`). Only `Network.add_node` and
  `Network.add_edge` bump the topology version and maintain the edge order, so a direct change
  is neither seen nor refused.
- Replacing or editing `model.layers`, `model.closures` or `model.reactions` after
  construction. `Model.__init__` validates names, keys and drivers once; an appended closure
  runs without that validation.
- Changing a layer's private tensors (`cols`, the law slices) or its list of laws.

**Frozen dataclasses are shallow.** The application records (`Resistor`, `Circuit`,
`SewerNetwork`, `Manhole`, ...) are `@dataclass(frozen=True)`. That forbids assigning an
attribute, nothing more: a list, dict or tensor inside one can still be edited, and
`SewerNetwork.notes` is a dict that applications and users do edit. `Circuit` converts its
`devices` to a tuple; do not assume every frozen record does. Treat a built record as a value
and make a new one to change it (`dataclasses.replace`).

## Devices and dtypes

`Network.to(device=None, dtype=None)` sets the network's device and dtype in place, clears its
cached operators and returns the network. Operators derived afterwards use the new values.

There is no `Model.to` and no `layer.to`. A layer copies its index tensors and capacities when
it is built, and a model is not a `torch.nn.Module`, so calling `net.to(...)` on a model that
exists does **not** move it: the layers keep the old tensors, and nothing refuses. Laws and
node sources are modules and have their own `.to`, but moving only some of the parts is not
a supported way to move a model.

To run on a device or in a dtype: call `net.to(device, dtype)` first, build the laws and layers
from tensors that are already on that device and dtype, and pass state and drivers that are too.
Build a second model rather than moving the first. The test suite pins only that `Network.to` changes the
operators derived afterwards; whole-model device moves are untested.

## Copying and serialization

- **`copy.deepcopy(model)` is supported** and gives an independent model: its own network, its
  own layers and laws. The copy's topology guard refers to the copy's network, so growing the
  original network does not affect the copy.
- **`pickle` and `torch.save` of a whole model work** when every part is picklable. The
  built-in layers, laws, drives and reactions are. A closure is an arbitrary callable: a
  module-level function pickles, a `lambda` or a locally defined function does not.
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
