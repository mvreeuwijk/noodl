# Sewer application: measurements behind design choices

Measurements that justify choices in `src/noodl/apps/sewer/`. The code states the choice
and the reasoning; the numbers live here so they can be re-measured. The figures were taken
on the development workstation (Windows 11, CPU, PyTorch float64 unless stated) and move with
hardware; only their ordering is a claim.

## Dense versus sparse head solve

`DENSE_SOLVE_MAX_JUNCTIONS = 128` in `dynwave.py`: up to this many junctions the head solve
uses the dense LU (`linear_solver="direct"`), above it the layer's `"auto"` (SciPy sparse LU).

Time per pass in ms, dense / sparse, on the binary trees of `benchmarks/sewer_dynwave_speed.py`:

| Junctions | Dense | Sparse |
| --- | --- | --- |
| 63 | 17.6 | 18.3 |
| 127 | 18.5 | 19.3 |
| 255 | 21.4 | 20.2 |

Reproduce: `python benchmarks/sewer_dynwave_speed.py`. The crossover sits between 127 and 255
junctions; the dense path avoids SuperLU's per-call conversion on small networks.

## Wall slope (`K_SWITCH`)

The normal-flow slope switch's wall (`DynamicWaveConduit`) is capped at `K_SWITCH` times the
smaller end junction's grounding. Observed on the drain-down of the tree in
`tests/apps/sewer/test_dynwave.py` (`_drain`):

- at 10 x the grounding the head solve failed (50 Newton iterations at t = 520 s);
- at 1 x it converges.

The heads bound the flow residual of that switch only through the wall slope over the
junction's grounding, so a heads-only convergence test stopped a pass short of the fixed
point: a flow residual of 3e-5 m3/s was seen. That is why `iterate_flow_tol` is judged
alongside the heads.

## Leak law precision

The leak law is built directly as `PowerLaw(..., kind="leak")` in `air.py` and not through
`Orifice`, which casts to `torch.get_default_dtype()` (float32 in this repository). Verification
check C1's residual floor, with the leak coefficient in each precision (two residual
statistics of the check):

| Precision | Residuals |
| --- | --- |
| float32 leak branch | 1.352e-12 / 1.855e-11 |
| float64 leak branch | 4.518e-13 / 6.141e-12 |

Measured when the headspace model was built; the check is the one named C1 in
`tests/apps/sewer/test_air.py`.

## Storage quasi-steady agreement

`SewerHydraulics` storage sweep under constant inflow reaches the quasi-steady solution:
relative agreement 5.7e-15 after 200 steps of 60 s. Reproduce with
`tests/apps/sewer/test_hydraulics.py::test_storage_reaches_the_quasi_steady_fixed_point`
(the test asserts a tolerance; print the relative difference to see the figure).
