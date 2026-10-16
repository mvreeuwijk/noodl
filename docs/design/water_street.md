# Water and street air quality: measurements behind design choices

Measurements that justify choices in `noodl.apps.water` and `noodl.apps.street_aq`. The
source keeps the reasoning; the numbers live here. All were taken on the platform and
versions of the day they were written (October 2026, Windows, CPU, float64); treat them as
orders of magnitude, not as asserted values.

## Headloss inversion

`water/elements.py` inverts the headloss law by EPANET's analytic Newton iteration,
safeguarded by bisection, plus one final Newton step on the autograd graph. A generic
`noodl.solvers.solve_monotone` inversion was about 5x slower, because it needs an autograd
derivative per iteration and a collapsed bracket to stop. This comparison was made ad hoc
during development and is not packaged as a benchmark. To redo it, swap the inversion in
the headloss element for `solve_monotone` and time `benchmarks/water_eps.py`.

## Event step

`water/tanks.py`: `event_step` shortens the hydraulic step to the linearly projected control
crossing, as EPANET does (Manual section 13.1 item 17, p. 113). On Net1 over 24 h:

| step rule | worst reported-step tank-level difference to EPANET |
| --- | --- |
| fixed 1 h | 2.07 m (the pump switches a whole hour late) |
| shortened to the control crossing (2 extra sub-steps) | 8.181e-5 m |

Reproduce: `tests/verification/test_water_parity.py::test_net1_extended_period_tank_level`
(run with `-m slow` if it is marked so), or the driver `benchmarks/water_eps.py`.

## Junction source

`street_aq/above_roof.py`. Setting: `C_ext = C_bg + K F` must be neutral to a uniform
background `C_bg`.

- A junction source of `sum_out q C_street` (no subtraction of the background) re-emits
  background air. On a 7 x 7 lattice (L = 100 m, W = H = 20 m, u* = 0.5 m/s, wind 0.3 rad)
  with nothing emitted, `max C_ext / C_bg = 1.095`. The implemented source is the excess
  over background, which removes this.
- With `junction_source="net"` (the incoming-vent sink kept), the coupling has a loop
  `G[i, i] = -K_j[i, j] q_in`. On `twelve_street_grid` (same L, W, H, u*) its spectral
  radius is 0.20 at 0.3 rad and 0.31 at 0 rad. With the default `"upward"` it is exactly 0
  in both cases. Computing it: build `G` from the coupling of `above_roof` on that case and
  take `max(abs(eigvals))`.
- Against SIRANE's archived `C_ext`, at the deck's downwind cut-off, the median error is
  0.32 % for `"upward"` and 0.29 % for `"net"`
  (`tests/verification/test_sirane.py`).
