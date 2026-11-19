# A new application

How to compose a new application from existing layers and laws. A "new application" is a
builder function returning `(model, state, drivers)` from existing layers and laws, plus (if it
steps forward in time) an `initial_state` function and any closures it needs. It needs no change
to the core. If you also need a new branch law, write it first: [A new constitutive
law](law.md).

## The smallest application

Here is the smallest one that exercises the `Sigmoid` law of the previous page: an ambient
boundary and two zones, joined by three `airpath` edges, one `KirchhoffLayer`, one `Model`.

```py
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.model import Model
from noodl.topology import Network
import torch

F64 = torch.float64


def build_two_zone(*, learnable: bool = True):
    """(model, state, drivers) for the smallest network the new law can be checked on.

    ambient (boundary) -> z1 -> z2 -> ambient, three `airpath` edges, one `Sigmoid` law
    covering all three (steepness `k`, shape (3,)). One `KirchhoffLayer`, one `Model`.
    No `TransportLayer`, no closures: nothing here needs one, and the guide's point is the
    smallest network that exercises the new law, not a full application.
    """
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z1")
    net.add_node("z2")
    net.add_edge("ambient", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient", kind="airpath")

    k = torch.tensor([1.0, 0.8, 1.2], dtype=F64)
    law = Sigmoid(k, learnable=learnable)
    layer = KirchhoffLayer(
        net, "air", [law], boundary=["ambient"], quantity="pressure", unit="Pa",
    )
    model = Model(net, {"air": layer})

    state = {}
    drivers = {"air.phi_boundary": torch.zeros(1, dtype=F64)}
    return model, state, drivers
```

This is deliberately the minimum: one Kirchhoff layer and no transport layer, no closures. The
checks to run on it are on the previous page ([Check it](law.md#check-it)). A real application
adds more of the same kind of thing, in a fixed place:

- **`src/noodl/apps/<name>/network.py`** is the module a real application lives in. It defines
  the domain objects (see `Street` and `StreetNetwork` in
  `src/noodl/apps/street_aq/network.py` for a worked example: frozen dataclasses that validate
  their own inputs), a **builder** (`build_model` there, `build_two_zone` here) that
  turns them into `(Model, State, Drivers)` by constructing a `Network`, one `ConstitutiveLaw` per
  edge kind, one layer per physical quantity, any closures, and the `Model` itself, and an
  **`initial_state(model)`** function that returns a state dict of the right shape for every
  layer the model owns (zeros, keyed by each transport layer's own state key).
- **Closures with `integrates` declared.** A closure that only reads the current state and
  drivers to produce new driver values needs no declaration. One that instead **integrates**
  some state over the step's own interval (advances a counter, a controller, a stored level)
  must declare `integrates = True` (see `noodl.contracts.IntegratingClosure` and
  `noodl.contracts.StepContext`) so `Model` calls it as `closure(state, drivers, ctx)` with the
  interval to advance by, and so `Model.steady()` (which has no interval) refuses it by name
  instead of silently either skipping it or integrating over an undefined `dt`.
- **Drivers.** Every boundary value and every source a layer's `solve` or `step` needs is a
  driver, `"<layer>.phi_boundary"`, `"<layer>.x_boundary"` or `"<layer>.sources"`, supplied by
  the builder as a template and overwritten by the caller (or by a closure) at each step. See
  [Layers and how they are stepped together](../../concepts/layers.md) for the full key
  vocabulary.

**Numerical blocks are not extension points (yet).** The supported extension path is laws and
applications, as above: a new `ConstitutiveLaw` and a new builder composing existing layers. A
new solver or layer class is a framework change, not an extension: it needs the same design,
testing and review as any other change under `src/noodl/layers` or `src/noodl/solvers`.

## The closure contract

A closure is any callable that adds or replaces drivers. `noodl.contracts` is where the
contract is written down; `Model` reads every closure through it, once, at construction.

**What the contract covers, and what it does not.** It fixes how a closure is called (two
arguments, or three with a `StepContext` when it declares `integrates = True`; the arity is
checked against that declaration in both directions) and which optional attributes it may
declare. **Units** are checked, as strings and not as dimensions: they live with the keys, in
`noodl.refs.Field.unit` (a layer's own keys, through `model.refs`), in the closure's
`input_specs` (the `unit` of each key it reads) and in its `output_units`
(`{key: unit}` of the driver keys it writes and the state it carries), which the model reads
through `ClosureSpec.input_specs` and `ClosureSpec.output_units`. `Model.check()` reports
`unit-mismatch` (a warning) when a closure declares a unit that differs from the unit of the
layer field or declared input the key refers to, and `unit-undeclared` (info) for a field with
no unit. Trivially equivalent spellings agree (`m3`, `m^3`, `m**3`, `m³`; `1`, `-`, an empty
unit, `dimensionless`; `°C`, `degC`; see `noodl.refs.normalize_unit`); nothing is converted and
there is no unit algebra, so `kg/s` and `kg/h` are different. A closure that declares no unit
is not checked. Nor does it govern **lifecycle and mutation** (what may change after
construction, devices, copying and pickling): that is
[Mutation, devices and serialization](../../guide/devices.md).

- **Algebraic closure**: `closure(state, drivers) -> {key: tensor}`. It sees the state and the
  drivers and returns driver entries. It never writes a layer's own state key.
- **Integrating closure**: declares `integrates = True` and is called
  `closure(state, drivers, ctx)` with a `StepContext` (`ctx.dt`, `ctx.t`). `ctx` is `None` for a
  query (`residuals`, `current_flows`): evaluate the algebraic outputs without advancing. Under
  `steady()` (`ctx.dt is None`) an integrating closure is refused by name.
- **State it carries**: `state_keys = ("tank.h",)` with `integrates` declared (`True` if the
  closure advances it, `False` if something else does).

```py
class Tank:
    integrates = True
    state_keys = ("tank.h",)
    inputs = ("tank.h", "inflow")
    outputs = ("outflow",)
    output_units = {"tank.h": "m", "outflow": "m3/s"}

    def __call__(self, state, drivers, ctx):
        ...
```

Everything else a closure may say about itself is one of the optional attributes in
`noodl.contracts.CLOSURE_ATTRIBUTES` (`integrates`, `state_keys`, `key_labels`, `inputs`,
`outputs`, `input_specs`, `output_units`, `edge_kinds`, `requires_coupling`, `reads_flows_of`,
`species_columns`), each with its meaning and its consumer. Declare them as tuples of names:
a bare string (`outputs = "rho_amb"`) or a non-bool `integrates` is refused with a
`TypeError`. `closure_spec` is the only code that reads these attributes; do not `getattr` them
elsewhere.

**Species by column.** A closure or reaction that addresses species by column index declares
`species_columns = {"layer": {"bod": 0, "sulfide": 1}}`; `check_species_columns` verifies it
against the layer's `species_names`, so a default column order over a layer that orders its
species differently is refused at construction rather than mixing the species up. Build the
columns with `noodl.layout.species_columns(layer.species_names, ["bod", "sulfide"],
where="...")`.

**Layout.** Per-node tensors are `(..., n)` or `(..., n, K)`; the batch axes come first. Do not
re-derive that rule: `noodl.layout` has `as_stacked` (stacked `(..., n, K)` first, reduced
`(..., n)` also for one species), `broadcast_batch` (batch axes of several tensors, expanded
as views), `per_instance` (a value per instance that cannot broadcast against a node axis)
and `write_at`.

## What a real application must ship

A network and a builder are not enough to trust an application's numbers. Before an
application is added to the repository, check that it has all of the following.

**Tests, beyond ordinary unit coverage of its own code:**

- [ ] **An independent reference.** A case with a known answer this application did not itself
  produce (a closed form, a published value, `scipy`, or a legacy tool's own output), checked
  to a stated tolerance. Testing the code against itself is not this.
- [ ] **Conservation.** The nodal or interior balance this application's `steady()` converges
  to is zero (to solver tolerance) at that fixed point, and stays a fixed balance under
  whatever coupling or sub-stepping the application uses.
- [ ] **Directional derivatives.** `torch.autograd.gradcheck` (or, at application scale, a
  central-difference comparison) through the application's own solve, for every parameter and
  driver a calibration or optimisation loop would actually differentiate against, not just
  the new law in isolation.
- [ ] **A scaling fixture.** A case parametrised over ensemble size (and, where relevant,
  network size), so a regression in batching or in solver selection is caught as a shape or
  cost change rather than discovered later as a silent slowdown.

The first three, in miniature, are the [check on the law page](law.md#check-it); a real
application ships all four, sized to its own domain, as the existing applications' test
directories do.

**Documentation, in the same change as the code:**

- [ ] **The common set-up.** `build_model`, `initial_state(model, *, values=None)` and
  `initial_drivers(model, *, values=None)`, exported in `__all__` like every other
  application, and layers built with a physical `quantity`, `unit`, `flow_unit`, `source_unit`
  and `capacity_unit` (and `conductance_unit` and `species_names`), so that `model.refs` names
  and units their keys.
- [ ] **Units in every docstring.** A component field, law coefficient or head-term parameter
  with a physical unit states it (`length` (m), `C` in the flow unit per Pa^n), and an
  application that reads network attributes declares them in an `attributes.py` like
  `noodl.apps.building_physics.attributes`.
- [ ] **A docstring on every public name**, whose first paragraph says what it does. The
  [catalogue](../../reference/catalogue/index.md) is generated from them by `python
  scripts/gen_catalogue.py`; a new application adds itself to `PAGES` there and to the
  catalogue section of `mkdocs.yml`.
- [ ] **An application page** under `docs/applications/` whose examples run: add it to
  `PAGES` in `tests/test_docs_usage.py`, and mark any snippet that is not meant to run on
  its own as `py`.

The catalogue and documentation tests fail until all of the documentation items are in place,
and whenever a later change leaves them behind the code.
