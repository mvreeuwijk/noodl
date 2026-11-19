# Design notes

For maintainers. These pages record why the code is built the way it is: the benchmarks,
profiles and measurements behind non-obvious choices, and the port logs of the applications that
follow a reference tool. Each note names the benchmark or test that reproduces its numbers. The
source keeps each rule and its reason in a few sentences and links here for the evidence.
Numbers are platform dependent: re-measure when the platform or the problem size changes, and
treat the ordering, not the digits, as the claim.

You do not need these pages to use noodl. If you want to extend it, read
[Extending noodl](../extending/index.md) first, and [Contributing](../contributing.md) before
you change a default.

## Performance and solver choices

| Note | Covers |
|---|---|
| [Core orchestration](orchestration.md) | Shape broadcasting, the incidence matrix, the fast path, Anderson mixing, and why `CoupledModel` is separate from `Model`. |
| [Solver selection](solvers.md) | Sparse-direct against preconditioned CG, the transport solver choice and operator caches. |

## Application choices

| Note | Covers |
|---|---|
| [Sewers](sewer.md) | Dense against sparse head solve, the wall slope, leak-law precision. |
| [Water distribution and street air quality](water_street.md) | The headloss inversion, the event step, the junction source. |
| [Street puffs: unsteady above-roof transport](street_puffs.md) | The puff model's design and its cost at city scale. |

## Port logs

| Note | Covers |
|---|---|
| [WSIMOD port](wsimod-port.md) | Decisions made porting WSIMOD, with the upstream source references. |
