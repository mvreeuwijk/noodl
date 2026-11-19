# Water distribution: limitations

What the water distribution application does not model, where it differs from EPANET on purpose, and where EPANET's own answer is not unique. The supported subset is listed on the [overview](index.md). Evidence for every comparison with EPANET is on the [verification page](../../reference/verification/water.md).

## Not modelled, refused by name

- **Not modelled, refused by name:** a valve at a full or empty tank. A file using it raises an
  error naming the valve.
- **Rules.** `FILLTIME` and `DRAINTIME` premises and the `DAY` and `POWER` ones are refused. A
  pump closed by a rule is at speed 0 in `"water.link_setting"`, as EPANET (`setlinkstatus`)
  and as a closing control leaves it; a `SETTING` premise on it reads 0, as EPANET.
- **Controls.** `validate()` refuses a control on a reservoir (`IF NODE <reservoir>`): EPANET
  2.2 compares tank volumes and a reservoir has area 0, so such a control always holds. A pump
  closed by a control is at speed 0 in `"water.link_setting"`, as in EPANET, so a `CLOSED`
  control reaching a pump closed by `[STATUS]` (which keeps its speed 1) ends a step at its
  trigger as EPANET's does (Net6, PUMP-3881 at 1050 s); a pump closed by `[STATUS]` or by the
  solve keeps its speed (read `"water.status"` too). A control that sets a pump's
  built speed leaves the `"water.setting"` driver in force (a calibration driver at 0.9 stays
  0.9 after an `OPEN`, EPANET gives 1.0). `water_step` and the implicit schemes never shorten
  a step to a time control, which fires only if a step starts exactly on its second
  (`water_run` refuses them).
- **Water quality: not modelled.** EPANET-MSX (multi-species), EPANET 2.3 features, an Eulerian
  discretisation, the per-link average reaction-rate output and the `Wbulk`, `Wwall`, `Wtank` and
  `Wsource` summary of EPANET's report (cosmetics). A reaction order between 0 and 1, `ORDER WALL`
  other than 0 or 1, and a Darcy-Weisbach roughness not below the diameter under a roughness
  correlation are refused by name ([Water quality](quality.md#refused-by-name)).
- **An empty tank that was a part's only supply** leaves that part cut off: the pipe to it
  then refuses to drain it, and the solve is refused by name before it starts ("junction(s)
  ... have a demand that no reservoir, tank or inflow reaches"), where EPANET reports negative
  pressures (wntr's Anytown as shipped: pumps at speed 0, tanks at their minimum).
- **A volume that decreases along the curve is refused.** EPANET 2.2 accepts it (wntr 1.5.0
  solves Net1 with `(110, 80000)` after `(60, 90000)`), but a volume that falls as the level
  rises is not physical; `validate()` and `TankVolumeCurve` refuse it by name. Flat segments
  are accepted.
- **Links of extreme resistance that must carry a demand are not solved.** Anytown's 0.0001 in
  placeholder pipes alone feed three junctions; EPANET reports heads of -9.9e24 m there, and
  noodl's Newton cannot bring a row whose two terms are 9.9e24 m below its tolerance (one
  rounding step there is 2^30 m): it raises the "singular" error of a numerically singular
  Jacobian. The Anytown comparison gives those pipes 12 in.
- **One demand pattern per junction.** Several `[DEMANDS]` categories of one junction must
  share their pattern (refused by name otherwise).

## Differences from EPANET

- **Closed links carry no flow; EPANET's carry `dh / CBIG`.** EPANET gives a closed link the
  conductance `1 / CBIG` (`CBIG = 1e8` ft per cfs), so a head drop across it leaks: on Net6 at
  3600 s, 2.56e-7 m³/s through three closed pumps from a dead-ended suction (noodl's head drop
  over CBIG gives the same 2.5592e-7 m³/s). Integrated over hours that is 1.7e-6 m an hour in
  a tank's level, enough to move a level trigger that falls near a half second by one second
  (EPANET rounds the time to it); over Net6's 96 h the 609 steps all match, up to a minute
  apart late in the run, and the tank levels differ by up to 1.95e-3 m. On Anytown the empty
  tanks' closed pipes leak 1.4e-7 to 2.7e-7 m³/s out of them, EPANET's pumps deliver about as
  much less, and the heads far from the pumps differ by up to 1.8e-4 m (1.3e-7 relative with
  those leaks added as demands); on the two-loop with every pipe a check valve at five times
  the demand the two closed ones leak 5.7e-8 and 3.1e-8 m³/s (heads 1.4e-6 relative; 7.6e-8
  with the leaks added as demands). The leak is not modelled in the hydraulics; the energy
  reproduces it where EPANET's energy reads it (a pump other than a constant-power one that a
  rule opens at the end of a step, see [pump energy](extended-period.md#pump-energy-and-cost)).
- **PRV and PSV inside EPANET's `HTOL` band.** Within 1.524e-4 m below a PRV's switch EPANET
  keeps it `ACTIVE` with a negative drop (its status rules have hysteresis); noodl gives the
  ideal valve, `OPEN`. The heads differ there by less than `HTOL`. EPANET's `XPRESSURE` (an active valve that cannot deliver its
  setting) has no counterpart: such a valve is `OPEN` in the exact condition.
- **The Colebrook option is not EPANET below Re = 4000.** With `friction="colebrook"` a laminar
  D-W pipe's head loss follows a straight line to the Colebrook value at Re = 2000, not
  Hagen-Poiseuille, and Dunlop's transition cubic is not used. The default `friction="composite"`
  reproduces both.
- **A batch shares one step.** `water_run` ends the step of every instance at the earliest
  event of any, so a batched instance is its own single run only up to the
  extra step boundaries, each an extra solve of EPANET's explicit step (6.5e-3 m on Net1 at
  0.8 and 1.25 times the demand; exact to 1e-9 m against its single run forced onto the same
  boundaries). Run instances one at a time to reproduce EPANET's own sequence for each.
- **The run ends at the duration.** `water_run` solves at every time EPANET's whole steps reach
  within `DURATION`, past the last report time too (a tank or control event there; the quality
  is kept and routed there as EPANET's is). EPANET's last step is not cut at `DURATION` (it
  solves once more past it, and reports nothing there); `water_run` ends the run at the
  duration without that solve.
  A REPORT START that is not a multiple of the REPORT TIMESTEP follows EPANET's rule (the first
  solve at or after each report time, `runqual`). wntr 1.5.0 cannot read that output file, so
  it is compared through EPANET's toolkit with `runqual`'s rule applied to EPANET's solves
  not against the file.
- **PATTERN START is transcribed with EPANET's own step rule.** EPANET looks for the next
  pattern period at `n * PATTERN TIMESTEP - t` (`timestep`), which ignores PATTERN START: a
  step that starts between two of the shifted period boundaries runs to the unshifted one,
  and takes the new period's multipliers on only there. `water_run` does the same.
- **Water quality inherits the hydraulics' closed-link leaks.** EPANET's closed links conduct
  `1/CBIG` and noodl's do not (see Closed links carry no flow), so a quality front that crosses a closed link's leak, or a
  switch that falls a second apart (Net6's PUMP-3845), is carried differently: ky4's 33 admitted
  points, ky10's J-433 and Net6 after 18922 s. They are measured in the verification; the leak
  is not modelled.
- **A tie at exactly `Ctol` cannot always be reproduced.** The decision is made in EPANET's
  internal unit, but the last bit of the compared values depends on EPANET's merge history in
  float32-read volumes and ft³; Net2 as shipped (31 h) is decided otherwise than EPANET's ([verification](../../reference/verification/water-quality.md)).

## Cases with no unique head

In these cases the exact equations have many solutions, and EPANET's choice depends on its closed-link leaks and its convergence path.

- **A zero-demand group of several junctions joined by a link inside it, fenced off by
  check valves**, is held at the `1 / CBIG` limit head (75 m between reservoirs at 100 and
  50 m). EPANET's head there is not that limit: it resolves this head only to steps of 0.0546 m and
  gives values from 73.2 to 75.1 m as the internal pipe's length and its own accuracy change. A single
  fenced-off junction, with no link inside the group, matches EPANET to 3e-9, and a junction fed from the
  highest of several reservoirs through one check valve to 8.5e-8. The measurements are on the
  [verification page](../../reference/verification/water.md#where-epanets-answer-is-not-unique).
- **A group behind a closed link, fed only through it, has EPANET's head only in the limit.**
  A PRV (or any valve) `CLOSED` by `[STATUS]` in front of junctions without demand holds them
  at the `1 / CBIG` limit, the head on the other side (60 m); EPANET reports 59.930 m for the
  same network at `ACCURACY 1e-7` and 1e-8 (53.687 m at 1e-5, where it stops after two
  iterations), a difference of 1.2e-3 relative. A PRV or PSV that closes itself in front of a dead end
  is held at the limit head too; where the dead end is on the valve's start, EPANET (converged) holds
  the same head but reports the valve OPEN where noodl reports it CLOSED, because at zero flow between
  equal heads both states satisfy every link. On some dead-end networks EPANET does not converge at
  1e-7, or its head moves with its accuracy; those are compared at EPANET's early stop. The cases are
  on the [verification page](../../reference/verification/water.md#where-epanets-answer-is-not-unique).
- **A zero-demand group between a pump and a check valve to a higher reservoir has no
  unique head.** Every head from R1 plus the pump's shut-off head up to R2 satisfies every
  link with no flow anywhere. EPANET's value is set by the ratio of its closed-link leaks
  and its status path: with R1 at 50 m and R2 at 100 m it shuts the pump and reports
  74.48 m. noodl returns the lower end, the pump running at zero flow at its shut-off head
  (63.3334 m), which EPANET also returns when a CLOSED pipe replaces the check valve. With
  any demand, or R2 below that shut-off head, the two agree.
- **A zero-demand group between a stopped pump and a check valve has no unique head
  either.** With the pump CLOSED or at speed 0 every head up to R2 satisfies every link;
  noodl gives the `1 / CBIG` limit with the check valve shut (the average of R1 and R2:
  75 m for 50 and 100 m), the same for a CLOSED pump and one at speed 0, and EPANET keeps
  the check valve OPEN from its first iterate (its `cvstatus` holds a status inside `HTOL`)
  and reports R2's head.

## Scope of the comparison

- **The steady tracer is verified on a single source only.** The check uses a
  network with one source, where the answer is 100 % everywhere it reaches; mixing of several
  traced sources has not been compared against EPANET.
- **Coefficients of a model built from a file are not learnable.** `build_model` builds the
  pipe, pump, valve, tank and reaction coefficients as constants. Gradients reach the drivers
  and the initial state, not the roughness or reaction rates; to fit those, build the elements
  yourself with `learnable=True` (see [Gradients](../../guide/gradients.md)).
- **Tank area does not enter a single `water_steady` call.** Only the tank's head enters the
  steady solve, so gradients of a hand-built model with respect to tank area come from a time-stepped run
  (`water_run`), not from the steady state.
- **ky10's quality is verified to 6:54:59 only.** After it EPANET's `pumpcoeff` caps the slope of
  the constant-power pump `~@Pump-10` (`CBIG`), which dead-heads against its closed CV and
  takes it to zero flow, where noodl keeps it on its power curve (5.4e-4 m³/s, 1396 m): the
  hydraulics, and so the quality, differ from 7 h. Forcing the pump closed in noodl matches EPANET to
  10 h. The 24 h comparison fails with these values, as expected, and is not claimed; EPANET itself needs `TRIALS 5000` at `ACCURACY 1e-7` on this file, and `DAMPLIMIT 1e-2` (see [Verification](../../reference/verification/water.md#where-epanets-answer-is-not-unique)).
- **EPANET's own accuracy bounds the comparison.** A file's `ACCURACY` is clamped to 1e-5 by EPANET's reader; the reference sets 1e-7 through the toolkit, the tightest that converges everywhere. The reference tool's own convergence and its two builds are described on the [verification page](../../reference/verification/water.md#where-epanets-answer-is-not-unique).
