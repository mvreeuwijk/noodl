# Applications

noodl ships six worked applications. Each is a thin layer of domain physics over the shared
core, and each is checked against the established reference implementation in its field — with
the tolerances and measured errors written down rather than asserted.

| Application | Physical system | Reference model | Entry point |
|---|---|---|---|
| [Building physics](building_physics.md) | Multi-zone airflow, heat, contaminants | CONTAM / ContamX 3.4.1.7, Modelica Buildings Library | `build_model`, `read_prj` |
| [Street air quality](street_aq.md) | Urban air quality, canyon exchange, routing | MUNICH | `build_model`, `StreetNetwork` |
| [Sewers](sewer.md) | Gravity hydraulics, headspace air, sulfide | SWMM 5.2.4 | `build_model`, `read_swmm_inp` |
| [Water distribution](water.md) | Pressurised mains, pumps, tanks, demand | EPANET 2.2 | `build_model`, `read_epanet_inp` |
| [WSIMOD — rule-based water-system allocation](capacitated.md) | Requested flows clipped to arc capacity and free storage at the receiving node | WSIMOD 0.8.1 | `CapacitatedTransferLayer` |
| [Coupling](coupling.md) | Two independent models exchanging values | — | `union` |

![The four flow-determination modes](../assets/four-modes.svg)

## Why these six

They are not a feature list. They are the argument that the factorisation works.

Each uses a **different one of the four flow-determination modes**, and between them they cover
all four:

- The **building physics** and **water** applications solve for a potential — pressure in
  pascals, hydraulic head in metres. Newton on a nodal conservation residual.
- The **street air quality** application has no potential at all. The along-canyon velocity is
  a closed-form function of the wind aloft, so a closure computes every flow directly.
- The **sewer** application's water side exploits the fact that a tree has an empty cycle space:
  continuity alone fixes every discharge, in closed form, with no solve. Its *headspace air*
  side, on the same graph, is a full Newton potential solve.
- The **WSIMOD allocation** application has no potential and no continuity solve — a request, clipped
  against capacity and headroom.

And they share machinery in ways that would be coincidence if the abstraction were wrong. The
sewer's headspace air layer drives its Newton solve with the *same* `Stack` buoyancy term the
building application uses for room air. The water application's Darcy-Weisbach pipes are the
framework's own `Duct` element, the CONTAM duct law, in volumetric form. The sewer and water `.inp` readers share one tokenizer. The
`solve_monotone` root-finder that inverts Manning's equation for sewer depth also solves the
per-node QP inside the capacitated layer's projection mode.

## Reference implementations, parity and validation

Three words are used with fixed meanings throughout these pages.

- A **reference implementation** is an established, independently written code that solves the
  same equations: CONTAM and ContamX for building physics, MUNICH for street air quality, SWMM
  (through pyswmm) for sewers, EPANET 2.2 (through WNTR) for water distribution, and WSIMOD for
  rule-based water-system allocation.
- A **parity test** runs noodl and a reference implementation on the same input and compares the
  outputs at a stated tolerance. Parity shows that noodl solves the same model as the reference;
  it is a code-to-code comparison, not evidence that the model describes reality. Where the
  reference has not been run, as for MUNICH on this release, the check is instead against the
  formulas and published results the reference documents, and the application page says so.
- **Validation** means comparison against measurements. No application claims it yet.

Each page states its parity rows with an explicit tolerance and the value actually measured. The
pages are equally explicit about what the comparison *does not* show, and those caveats are
sometimes the most important thing on the page:

- The [WSIMOD allocation](capacitated.md#what-the-wsimod-parity-does-not-show) page explains that
  neither WSIMOD reference demo ever exercises a binding capacity clip, so the parity checks
  the identity path only — the binding branches are covered by synthetic tests instead.
- The [street air quality](street_aq.md#limitations-and-caveats) page records that the published
  MUNICH idealised case cannot be reproduced absolutely, because its geometry was never
  published, and states the size of the remaining difference.
- The [sewer](sewer.md#coefficient-provenance) page marks each coefficient as verified,
  calibrated, or unverified, naming the source.

Where a reference implementation and noodl disagree, the pages say which is more likely to be
right and why. Several of the remaining discrepancies are the *reference*'s approximation —
SWMM's 51-point circular-geometry lookup table, EPANET's float32 binary output — not noodl's.

## Starting a new application

The six are also a template. A new application is:

1. A **network builder** — domain dataclasses plus a function that turns them into a `Network`
   with the right edge kinds and attributes.
2. **Elements and drives** for whatever constitutive laws are not already in the core.
3. **Closures** for anything that is a function of solved state.
4. A **file reader**, if your field has a standard exchange format.
5. A **`build_*_model`** function returning `(model, state, drivers)`, and a `*_steady` helper if
   reactions mean `Model.steady` is not enough.

Every application follows exactly that shape, which makes them readable as worked examples.
