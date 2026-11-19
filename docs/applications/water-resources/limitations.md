# Water resources limitations

What the flow allocation layer and the WSIMOD port do not support, and where their results can
differ from WSIMOD. Anything that cannot run is refused by name when the model is built. The
evidence for what is supported is on the
[verification page](../../reference/verification/water-resources.md), and the places where
WSIMOD's own source is defective or silent are on [Differences from WSIMOD](differences.md).

## Flow allocation

- **Queue arcs** (travel time and decay of requests in transit), **pass-through bottlenecks**
  (nodes that forward only what they can send on within the step) and **pollutants** are not part
  of the allocation layer; the [WSIMOD port](wsimod.md) has them.
- **Sharing is by preference, not by call order.** WSIMOD serves competing arcs first come, first
  served; the layer shares one node's headroom proportionally. The difference is stated and
  tested ([the allocation page](allocation.md#where-the-layer-differs-from-wsimod)).
- **Capacities and storage bounds are constant in time.**
- **`"smooth"` and `"projection"` have no external reference.** They are checked against the hard
  clip and by gradient checks only.
- **Node-level limits.** `sewer_to_wwtw` in `oxford_demo` is excluded from the comparison with
  WSIMOD because a node-level limit, which the layer does not model, binds on 185 of its 1,456
  steps.

## WSIMOD port

- **Scope.** Every WSIMOD 0.8.1 class is ported except `EnfieldFoulSewer`. Not ported:
  `GrowingSurface.residue` (surface data named `*-residue` is refused), the `NutrientPool`
  override hook (the pool's parameters are drivers instead), the `("RiparianBuffer", "volume")`
  check (no such class exists in WSIMOD 0.8.1) and `Model.change_impervious_area`.
- **Input and output.** A configuration is read, not written. `extensions` and
  `unified_data_file` are refused.
- **`model.check(..., probe=True)` cannot step this model.** The probe steps without `t`, which
  the closure refuses, so it reports `probe-failed` ("model.step needs t="). Use
  `model.step(state, drivers, dt=86400, t=0)` as the check.
- **Speed.** The day is a transcription of many small operations, with no compilation, so it is slower per day than a compiled model; a batch amortises the cost.
- **Not re-entrant.** One `WsimodDay` holds one engine that every step reuses, so two steps of
  one model must not run at once (threads, or a step that calls another).
- **`Model.to`.** `model.to(dtype=torch.float64)` (or a CPU device) works and changes nothing.
  Another dtype or a device move is refused by name (`TypeError`): the port is float64 on the
  CPU, as bitwise equality with WSIMOD needs.
- **River ties.** WSIMOD breaks a tie in the river order by set-hash order, which varies with
  `PYTHONHASHSEED`; noodl takes the configuration order, so a model whose river ties matter may
  differ from a WSIMOD run in the last bit. Neither demo has a `River`.
- **A gapped time-area** (`QueueGroundwater`) may differ from WSIMOD by an ulp: WSIMOD walks the
  queue in dict insertion order, noodl in ascending time.
- **The recursion rule over-approximates** and may refuse a model WSIMOD runs.

## Gradients

Gradients are those of the branch taken: they are not defined where a branch changes, and
long runs may need windows. See [what cannot be learned](wsimod-gradients.md#what-cannot-be-learned)
and [graph size](wsimod-gradients.md#graph-size-and-fitting-in-windows). The convergence limit of
the adsorption iteration has no gradient.
