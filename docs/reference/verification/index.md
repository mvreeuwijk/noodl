# Verification

Each noodl physics application reimplements the physics of an established tool. This section
reports how closely each one reproduces that tool. This page gives the method and one summary
row per application; the evidence page for each application has every comparison, with its
tolerance, how the tolerance was justified, the measured agreement and the stated cause of any
difference that remains.

## Verification and validation

*Verification* here means comparison with another code or with a closed-form solution. It says
nothing about whether either code matches reality. *Validation*, comparison with measurements, is
reported where it exists, for example the Pescod and Price ventilation measurements on the
[sewer evidence page](sewer.md).

## How to read an evidence page

Every comparison states three things.

- **Tolerance.** The bound the test asserts. It is relative unless marked "abs", which is an
  absolute bound in the quantity's own units.
- **Justification.** Why that bound, and not a tighter or looser one. Tolerances are set from the
  reference, not fitted to the result. Most are derived from the reference's own precision:
  EPANET's and SWMM's float32 binary output, ContamX's single-precision inputs, OpenModelica's
  resolution at a DASSL tolerance of 1e-13. Where a tolerance is only a regression guard, the row
  says so.
- **Measured.** The worst agreement observed, to compare against the tolerance. A wide margin
  means the tolerance is conservative; a measured value close to the tolerance is flagged.

Each row also says how the reference was run. **Live engine** means the test runs the reference
code during the test. **Stored output** means the test reads results the reference produced
earlier and committed with the tests. **Computed in the test** covers closed forms and
finite-difference gradients.

## Summary

| Application | Reference | What is compared | Agreement | Evidence |
|---|---|---|---|---|
| Building physics | ContamX 3.4.1.7 (live engine); Modelica Buildings Library 13.0.0 simulated by OpenModelica 1.27.1 (stored output); closed forms | Path flows, zone concentrations, controls, occupants, filters; every column of 22 Modelica models; analytical airflow and natural-ventilation solutions | ContamX within the budget of its float32 inputs (flows about 1e-6 relative); Modelica 1e-6; closed forms 1e-9 | [Building physics](building.md) |
| Street air quality | MUNICH formulas and published idealised case; SIRANE v2.1 archived results | Canyon exchange formulas; the 12-street case; exchange velocity, in-canyon wind and above-roof concentration | Formulas exact to the published precision; the network-level pattern does not reach its 5 % target, with the cause stated | [Street air quality](street.md) |
| Sewers | SWMM 5.2.4 (live engine) | Kinematic-wave flows, depths, velocities; dynamic-wave depths and flows; steady and transient quality | Flows 1e-9 (measured 6e-14); dynamic-wave depths 1e-7 m; quality 1e-9, except SWMM's one-step junction lag | [Sewers](sewer.md) |
| Water distribution | EPANET 2.2 (live engine) | Heads, flows, pump head gain, tank levels; pressure-driven demand; traced source | Heads and flows 1e-6 relative in steady state (the float32 floor of EPANET's output), 1e-5 over extended periods; Net6 heads 2e-5, flows 5e-4 | [Water distribution](water.md) |
| Water resources | WSIMOD 0.8.1, pinned exactly (live engine and stored output) | Allocated arc flows; every node class and the whole model on the shipped demos and purpose-built cases; gradients by central differences | Allocation 1e-9 abs on `quickstart_demo` (measured 1e-16), 1e-6 abs on 20 of 21 arcs of `oxford_demo` (measured 1.9e-9); whole-model series exactly equal where the arithmetic allows; gradients 1e-6 relative | [Water resources](water-resources.md) |
| Electrical circuits and power flow | Closed forms; Modelica Standard Library 4.1.0 and Buildings Library 13.0.0 via OpenModelica (stored output); pandapower 3.5.6 | Device laws, transients and phasors; 14 Modelica circuits; AC and DC power flow on MATPOWER cases | Closed forms 1e-12; Modelica algebraic 1e-9 relative, dynamic 1e-6 of range; power flow 1e-8 p.u. | [Electrical](electrical.md) |
| Coupling | Closed forms, central differences, conservation | Two-way fixed point, conservation, gradient across the join, back-coupling on a synthetic canyon | Closed forms 1e-9 absolute; gradients 1e-5 relative; calibration error 1.3e-4 | [Coupling](coupling.md) |

## Intentional differences

Two differences from the reference are deliberate, and each is tested so that it stays the only
difference.

- **SWMM's one-step junction lag is not replicated.** SWMM mixes the inflow to a junction from
  the upstream conduits' concentrations at the *start* of the step (`findLinkMassFlow` reads
  `Link.oldQual`), so each junction delays the signal by one step. noodl physics solves the
  whole tree implicitly within the step. The difference is first order in the time step and
  disappears at steady state. The implicit solve is the transport operator every noodl physics
  application shares; reproducing the lag would need an explicit, upstream-ordered sweep
  specific to SWMM. The test asserts that SWMM equals the lagged recursion built on noodl
  physics' own volumes and flows, so the lag accounts for the whole difference.
- **WSIMOD's first-come sharing is not replicated.** When several arcs push into one node's
  limited headroom, WSIMOD serves them in call order: two 10-unit pushes into 12 units give
  `[10, 2]` or `[2, 10]`. The allocation layer shares by preference weight and ignores order,
  giving `[6, 6]`. Order-dependent allocation has no gradient with respect to the requests. The
  preference-weighted share is a smooth function of them, with genuine cross-gradients between
  competing arcs in projection mode. The totals and the storage agree with WSIMOD. The other
  known departures from WSIMOD (same-step outflow, bottlenecks behind a pass-through `Node`,
  node-level limits) are listed in [Differences from WSIMOD](../../applications/water-resources/differences.md).

The MUNICH quadrature weights and exchange constant are the opposite case: MUNICH's own
choices, reproduced on purpose even where they look like artefacts, because "fixing" them would
bias the results relative to MUNICH.

## Golden files are not verification

`tests/golden/*.json` hold noodl physics' own earlier output (for example the two-loop water
network, the sewer tree, the CONTAM airflow case and the MUNICH idealised case). Comparing
with them detects that a result has changed. It does not show that the result is right: a
golden file matches whatever the code computed when it was recorded, right or wrong.
