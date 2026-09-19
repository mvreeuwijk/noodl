# Coupling

Two independently built models, exchanging named values each step, iterated to a fixed point —
and differentiable across the join.

## Two levels of coupling

These are easy to confuse, and they solve different problems.

**Within one model.** Several physics *layers* on **one** `Network` inside a single `Model`,
stepped together under `coupling="pingpong"` or `coupling="iterate"`. A building's airflow and
its heat balance are coupled this way: one graph, one model, several layers. See
[Layers and models](../concepts/layers.md#the-two-couplings-and-the-trap-in-the-default).

**Across two models.** Two *separately built* `Model` objects, each with its own `Network`, its
own layers and its own closures, coupled by `union`. A street network and a building are coupled
this way: two graphs that were never meant to be one.

This page is about the second.

```python
from noodl.couple import union, ValueLink, DriverAlias
```

## What `union` does — and deliberately does not

`union` exchanges named driver and state **values** between two ordinary `Model.step` calls each
outer step, iterating to a fixed point when a link is two-way.

It **never merges the networks, never rebuilds the layers, and never reconstructs the closures.**
That is a hard constraint, not an optimisation.

The reason is closures. Layers are reconstructible from standard arguments, but a closure is an
arbitrary application object — `StreetFlows`, a building's density closure — bound at
construction time to one specific `Network` with application-specific constructor arguments that
domain-agnostic code cannot know. Rebuilding them generically would require the coupling module
to import application internals, which would break the "neither model is modified" guarantee as
thoroughly as editing the application itself.

So `noodl/couple.py` **never imports from `noodl.apps.*`**. It reads and writes named keys and
applies registered unit conversions, and it never inspects any application's internals. That is
what lets the same machinery couple any pair of models, regardless of which layer type produces
which flow.

Each model's own `step` call stays an ordinary, unmodified call. And because PyTorch's autograd
tracks the computation graph rather than Python object structure, **gradients cross the join in
one `.backward()` exactly as if the two models were one object.**

After a union, the original models still run standalone, bit-for-bit identical — which is pinned
by a test.

![Two independent models exchanging values through a ValueLink, iterated to a fixed point](../assets/app-coupling.svg)

## A worked example

```python
import torch
from noodl.couple import ValueLink, union

link = ValueLink(
    from_model="street", from_key="street.x", from_index=0,
    to_model="building", to_key="species.x_boundary", to_index=0,
    convert="concentration_to_mass_fraction",
)

city, state, drivers = union(
    {
        "street": (street_model, street_state, street_drivers),
        "building": (building_model, building_state, building_drivers),
    },
    shared=[link],
)

new_state = city.step(state, drivers, dt=1.0)
```

With `street.x[0] = 3.0` kg/m³ and the building's `rho_amb = 1.2`, the building sees a boundary
mass fraction of $3.0 / 1.2 = 2.5$ kg/kg for that step.

*(Condensed from `tests/test_couple.py`. Each model is built in the ordinary way; the full
real-data demo in `tests/verification/test_coupling_demo.py` uses a CONTAM `.prj` and an AQ_DT
street domain and is longer only because of the wiring, not the coupling API.)*

## The API

### `union`

```python
union(
    models,                # {tag: (Model, State, Drivers)}
    shared,                # a flat sequence of ValueLink and DriverAlias
    *,
    substeps=None,         # {tag: k} — step this model k times per outer step
    relaxation=0.5,
    iterate_rtol=1e-8,
    iterate_atol=0.0,
    iterate_max=20,
) -> (CoupledModel, {tag: State}, {tag: Drivers})
```

The `(Model, State, Drivers)` triple is exactly what `build_street_model`, `project_to_model`,
`build_sewer_model` and `build_water_model` all return. `union` returns fresh copies and never
mutates its inputs.

Two-wayness is set **per link** through `ValueLink.two_way`, not by an argument to `union`.

### `ValueLink`

```python
ValueLink(
    from_model, from_key, from_index,
    to_model, to_key, to_index=0,
    convert=None,          # a registered conversion name, applied forward
    two_way=False,
    convert_back=None,     # applied to the feedback flux
    sources_key=None,      # defaults to "<from layer>.sources"
)
```

- `from_key` is a transport layer's state key `"<layer>.x"`; `from_index` a position on that
  layer's **active interior** axis.
- `to_key` **must** be a transport layer's boundary driver `"<layer>.x_boundary"` — enforced at
  construction, raising if not; `to_index` a position on the **boundary** axis.
- `sources_key`, if given, must end `.sources`.
- The two-way feedback is **added** to the source term, never overwritten — so a lateral load
  already in `sources` survives.

**One-way link timing has a subtlety.** In a union with no two-way link, the single pass reads
the forward value from the *step-start* state — explicit ping-pong. In a union that *also* has a
two-way link, every pass, one-way links included, reads from the previous pass's output, so a
one-way link there carries an end-of-step value. One-way links are never part of the convergence
test.

### `DriverAlias`

```python
DriverAlias(
    source=("street", "theta_w"),
    targets=(("building", "theta_w", "street_rad_to_contam_deg"),),
)
```

The source is authoritative; every target driver is overwritten from it through *its own*
conversion. Applied **once per step, before any pass** — drivers do not change between passes,
only glue values derived from state do.

### Helpers

`apply_conversion(name, value, drivers)` applies a registered conversion, returning `value`
unchanged for `None` and raising with the sorted list of registered names for an unknown one — an
unregistered conversion is never silently treated as identity.

`transport_boundary_inflow(...)` returns the net mass inflow at a boundary node of a transport
layer. `Model.ports()` cannot supply this: it reports `boundary_flows` for *potential* layers
only.

## Unit conversions

| Name | Formula |
|---|---|
| `concentration_to_mass_fraction` | $x = c / \rho_{\text{amb}}$ |
| `mass_fraction_to_concentration` | $c = x \cdot \rho_{\text{amb}}$ |
| `street_rad_to_contam_deg` | $W_d = (270° - \deg\theta) \bmod 360°$ |
| `contam_deg_to_street_rad` | $\theta = \big(\mathrm{rad}(270° - W_d)\big) \bmod 2\pi$ |

The density pair is a **genuine unit mismatch**, not a scale factor: the street application's
transport state is a concentration in kg/m³, while CONTAM's species convention is a mass fraction
in kg/kg, and the two are related by the local air density.

The wind-direction pair exists because the two conventions differ in *both* origin and sense.
CONTAM's $W_d$ is degrees clockwise from north, giving the direction the wind blows **from**. The
street application's $\theta_w$ is radians counter-clockwise from east, giving the direction it
blows **toward**. A west wind is $W_d = 270°$ and $\theta = 0$; a north wind is $W_d = 0°$ and
$\theta = 3\pi/2$. Both are pinned on the cardinal points by test.

Note that the **backward** link in the canonical pairing needs `convert_back=None`: once
`transport_boundary_inflow` has computed it, the feedback is a mass flux in kg/s on both sides.

## The fixed point

When any link is two-way, `step` routes through the iteration:

1. Apply the one-way links from the latest state.
2. For each two-way link, compute the raw forward value, relax it against the previous pass's
   value, write it onto the target's boundary, then compute and add the feedback flux.
3. Step **every** model once **from the step-start state** — never from the previous pass's
   output.
4. Test convergence.

That third point is the rule that keeps the scheme correct: passes never compound into
`passes × dt`.

**Convergence is judged on the *unrelaxed* residual** of the fixed-point map,
$\lvert \text{raw}_k - \text{raw}_{k-1} \rvert \le \text{atol} + \text{rtol}\,\lvert \text{raw}_k \rvert$,
per link and per batch instance, inside `no_grad`. Judging the *relaxed* increment instead would
make the effective tolerance scale with the relaxation factor, letting a heavily damped run
falsely report convergence.

Failure to converge in `iterate_max` passes raises naming the link — for example
`"street:street.x[0]->building:species.x_boundary"` — the failing instances, and the largest
change.

With `diagnostics`, you get `{"passes", "converged", "max_change"}`.

## The canonical pairing: street ↔ building

- **Forward, converted:** the street's segment concentration `street.x[i]` becomes the building's
  `species.x_boundary`, divided by the building's own `rho_amb`.
- **Backward, two-way, unconverted:** the building's net species boundary inflow at its ambient
  node is added into `street.sources` at the shared segment.
- **Aliased weather drivers:** the street's `U_ref` → the building's `V_met` with no conversion
  (the street is built at $z_{\text{ref}} = 10$ m, matching CONTAM's met-station convention), and
  `theta_w` → `theta_w` through the wind-direction conversion.
- **Multi-rate:** `substeps={"building": 60}` — the street steps once per hour, the building
  sixty times at 60 s, with the glue-derived boundary held constant across the inner steps.

Physically, infiltration draws segment air into the building and the building acts as a sink on
the street side. Both demos assert that sign.

## Validation

| Check | Tolerance | Measured |
|---|---|---|
| A two-way step is a fixed point of one step from the start state | rtol 1e-9, atol 1e-14 | holds |
| Gradient across the join vs central differences | rel 1e-5 | holds |
| `substeps={"building": k}` calls the fast model exactly `k` times per slow step | exact | holds |
| Synthetic back-coupling, 2×3 m canyon: one-way 4.16974e-08 vs two-way 4.14518e-08 kg/m³ | measured | **0.589 %** change, 28 passes |
| Real `leiden_small`, segment 783, steady 2.07911e-07 vs coupled 2.07896e-07 kg/m³ | measured | **7.2e-5** (0.0072 %) change, 22 passes |
| Loose sequential file exchange vs the two-way result | measured | 0.589 % discrepancy — equal to the street-side change, as expected for a boundary response linear in the shared value |
| Inverse 1: leakage calibration through the join | rel err < 0.05 | **1.29e-4**, final loss 3.4574e-08 |
| Inverse 2: source attribution by one adjoint pass vs central differences | rel 1e-4 | 3.6e-7, 5.3e-7; third source structurally zero |
| Inverse 3: one measured path recovers all four branch flows | rtol 1e-10 | exact |

The real-data row is worth reading carefully. A 0.0072 % change is **negligible** — and that is
the honest result, not a disappointing one. One building's infiltration should not measurably
change a whole street's concentration, and the number is correctly signed. The synthetic case,
deliberately sized so the building matters, shows 0.589 %.

Throughput, on the headline union over 6 coupled hours with 60 building sub-steps per street
hour: batch 1 takes 41.221 s and 116 outer passes; batch 10, 38.541 s and 128 passes; batch 100,
134.772 s and 128 passes — 3.5x the time for 10x the batch, sub-linear. No budget is set.

## Limitations

- **`CoupledModel.steady` is not built**, though the design names it in the intended public
  surface.
- **Relaxation is fixed at 0.5** in practice — it is a constructor parameter, but no adaptive
  scheme exists. Convergence takes 22–28 passes to rtol $10^{-10}$ on the demo fixtures. This is
  flagged as a known risk: successive substitution at fixed 0.5 relaxation can converge to the
  wrong root of a repelling fixed point, the same issue the
  [building application's](building.md#limitations) three-root case runs into.
- **Single species only.** `transport_boundary_inflow` assumes a single-species transport layer
  and raises `NotImplementedError` for more than one flow kind.
- **The sub-stepping policy holds the glue value constant** across the fast model's inner steps
  rather than interpolating. A follow-up if it proves too coarse.
- **Differentiated by unrolling** the outer passes, so memory grows with pass count. An
  implicit-function treatment of the fixed point is a recorded follow-up.
- **Ambient temperature is not coupled** in the demo — it reaches the building only through
  `rho_amb`, computed once from the `.prj`'s own `Ta`, and the AQ_DT forcing carries no
  temperature field.
- **A three-way union** (sewer + street + building) is deferred. The mechanism is expected to
  extend; it has not been built or tested.
- **`CoSim` and the WSIMOD `Node` wrapper are out of scope.** `union` is the only coupling route
  built. Coupling to a non-differentiable model outside the framework — EnergyPlus, WSIMOD
  itself — is not available.
- **`Model.current_flows`'s first-pass branch re-solves** the owning potential layer from scratch
  rather than reusing the pass's own upcoming solve. A recorded inefficiency.
