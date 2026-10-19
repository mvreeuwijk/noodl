# Research for #28: Anderson mixing near convergence

This branch is for investigation only. It is **not for merging**. It holds the work behind
issue #28, on top of the PR #24 head (`a4acfc5`).

## The change on this branch (in `src/`, not accepted)
- **The rule:** `src/noodl/solvers/anderson.py` adds a rounding-noise column-drop rule
  (`_ANDERSON_NOISE_MIN`, the `ns` argument of `_anderson_mix`, and the per-entry noise
  `eps * (|x| + |g|)` scaled like the residuals).
- **Its tests:** `tests/solvers/test_anderson.py` and
  `tests/test_model.py::test_anderson_near_convergence_does_not_follow_rounding`.
- **What it fixes:** the toy perturbation regression, which fails without the rule
  (`negative slope on edges [0, 1]`).
- **What it breaks:** four sewer tests, bit-identically for multiples 1, 2 and 10:
  - `tests/apps/sewer/test_losses.py::test_evaporation_takes_water_and_a_gradient_flows_through_it`
  - `tests/apps/sewer/test_dynwave.py::test_gradient_on_a_held_threshold_does_not_depend_on_the_wall_slope`
  - `tests/verification/test_sewer_losses_parity.py::test_seepage_matches_swmm_in_every_section_kind`
  - `-m slow` `tests/verification/test_sewer_dynwave_parity.py[backwater_dynwave.inp]`
- **Why:** issue #28 has the analysis.

## Scripts (`research/anderson-28/`)
| File | What it is |
| --- | --- |
| `anderson_proj.py` | Variant 1: the noise estimate projected onto each column's own direction (`d² > MIN·Σ\|v_i\|·n_i`). It passes the toy and 3 of the 4 sewer tests; evaporation fails at any multiple ≥ 1. |
| `anderson_rev.py` | Variant 2: extends the reversal safeguard to reject reversals beyond one tolerance once every residual is within tolerance. Golden unchanged; the seepage parity fails by 1.8e-9. |
| `toy.py` | The toy perturbation sweep (101 few-ulp source perturbations). |
| `evap.py`, `seep.py` | Reproduce the evaporation gradient and the seepage comparison. |
| `all_stat.txt` | 3267 near-convergence reversing steps measured across the sewer, INP, coupling, composed and SWMM-corpus tests. |
| `plug/` | pytest logging plugins that record Anderson column decisions. |
| `ci1/` | Golden-case stopping-margin scans (`trace2.py`, `perturb_toy.py`, ...), plus `pre*.json` records of the pre-refactor code (eedf5d9). |

The scripts were written as one-off probes. Some contain absolute paths to a local scratch
directory, so adjust those before running.
