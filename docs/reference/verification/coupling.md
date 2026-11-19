# Verification: coupling

Evidence for [coupling models with `union`](../../guide/coupling.md); the summary is on the [verification overview](index.md). Each row is a test that passes at the tolerance stated. The tests are `tests/verification/test_coupling_*.py` and `tests/verification/test_composed_*.py`.

## Coupling checks

| Check | Tolerance | Measured |
|---|---|---|
| The returned state's forward value agrees with the value the recipient was stepped with, to rtol | rtol 1e-9, atol 1e-14 | holds |
| Gradient across the join vs central differences | rel 1e-5 | holds |
| `substeps={"building": k}` calls the fast model exactly `k` times per slow step | exact | holds |
| Closed two-way exchange conserves the total amount — exact/backward_euler/trapezoidal schemes, 1 and 4 recipient sub-steps | abs 1e-10 | holds |
| 2×2 analytic fixed point (one implicit step each, unit source into the recipient) solves to $(4/3, 5/3)$ | abs 1e-9 | holds |
| Structural guard: a two-way step assembles no dense topology operator (`upwind`/`incidence`/selectors) | n/a | holds |
| Synthetic back-coupling, 2×3 m canyon: one-way 4.169740e-08 vs two-way 4.145151e-08 kg/m³ | measured | **0.5897 %** change, 27 passes |
| Loose sequential file exchange vs the two-way result | measured | 0.5897 % discrepancy — equal to the street-side change, as expected for a boundary response linear in the shared value |
| Leakage calibration through the join | rel err < 0.05 | **1.288e-4**, final loss 3.4574e-08 |
| Source attribution by one adjoint pass vs central differences | rel 1e-4 | 1.3e-8, 2.0e-9; third source structurally zero |
| One measured path recovers all four branch flows | rtol 1e-10 | exact |

The synthetic case is deliberately sized so that the building matters: a 2×3 m canyon, where
the building's infiltration changes the street concentration by 0.5897 %. On a street of
realistic size one building's infiltration changes the street's concentration very little, and
the coupled result should be read with that in mind.

The synthetic case is deliberately sized so that the building matters: a 2×3 m canyon, where the building's infiltration changes the street concentration by 0.5897 %. On a street of realistic size one building's infiltration changes the street's concentration very little, and the coupled result should be read with that in mind.

## Cost of a two-way run

The street ↔ building demo (`benchmarks/coupling_street_building.py`) couples 6 hours with 60 building sub-steps per street hour. On one workstation it took about 55 s at batch 1, 48 s at batch 10 and 137 s at batch 100 (medians; wall time varied by up to about 20 s between repeats, while pass counts did not vary at all). Three things set the cost:

- **Outer passes.** The run needs 107 passes at batch 1 and 118 at batches 10 and 100: a batch is converged only when every instance is, so a batch spanning a wider range of conditions (here wind speeds of 1–4 m/s rather than 1 m/s alone) takes as many passes as its slowest instance.
- **Batch size.** Cost grows far more slowly than the batch: 100 instances cost about 2.5 times one instance. Batching many scenarios into one run is much cheaper than running them one by one.
- **Boundary transfers.** A two-way link reads the recipient's boundary transfer, which costs more than a plain step. `union` requests it only from the linked transport layers, so an unlinked layer on the recipient (a `thermal` layer, say) takes the cheaper plain step; there is nothing to set.
