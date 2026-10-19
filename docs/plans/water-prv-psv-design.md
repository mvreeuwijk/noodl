# Design note: PRV and PSV in `apps.water`

Status: design only, for decision. Follow-up of #23 ("PRV and PSV valves, which switch status
on the absolute head at one end"). Nothing here is implemented.

## 0. Where the code stands

- `WaterNetwork` refuses PRV and PSV by name (`_VALVES_REFUSED` in `apps/water/network.py`);
  it also refuses `CLOSED` and `CV` pipes as "status-switching devices".
- The valves that are modelled are TCV (`MinorLoss`), FCV (`FixedFlow`), PBV
  (`BreakerValve`) and GPV (`GeneralPurposeValve`). PBV and GPV are `ImplicitLaw`s, so a network
  that holds one is solved by Modified Nodal Analysis (MNA), with the valve flow as an unknown.
- The only status switch inside a solve today is in `BreakerValve.residual`:
  `torch.where(open_, m q|q|, setting)`, evaluated on the current Newton iterate. The MNA
  Jacobian is the autograd derivative of the residual (`_mna.py`, `_colour_blocks`). That
  derivative is the derivative of whichever branch is selected, so this is already a
  semismooth Newton step.
- Pump on/off and the `[CONTROLS]` tank-level rules are applied between time steps, outside
  the solve, under `no_grad` (`TankStorageControls`). Each pump's status reaches the solve as a
  driver.
- Gradients come from the implicit-function theorem (IFT) at the converged point
  (`solvers/implicit.py`): one transposed solve with the Jacobian at `x*`.
- An implicit law sees only `dp = H_u - H_v` and `q` for each edge, never an absolute head.
  Multi-port laws (`ports = 2`, used by the electrical transformers and controlled sources) see
  the `dp`, `q` of every edge in their group. `tank_mode="storage"` already adds a head-0
  boundary node, `DATUM = "_datum"`.

## 1. The switching rule, and what EPANET 2.2 does

Notation: valve `u -> v`, flow `q` (positive `u -> v`), minor-loss coefficient `m` (so the
fully open loss is `m q|q|`). The PRV target is `Hset = El_v + setting`; the PSV target is
`Hset = El_u + setting`. The three states are:

| State | PRV | PSV |
|---|---|---|
| ACTIVE (throttling) | `H_v = Hset`, `q >= 0` | `H_u = Hset`, `q >= 0` |
| OPEN | `dp = m q|q|`, `H_v <= Hset` | `dp = m q|q|`, `H_u >= Hset` |
| CLOSED | `q = 0` | `q = 0` |

A PRV closes in two situations: the flow would reverse (`H_v > H_u`; the valve acts as a check
valve), or the downstream head is already above the setting with no flow through the valve
(the downstream side is fed from elsewhere). A PSV closes when the flow would reverse, or when
the upstream head would fall below the setting even with no flow through the valve.

**EPANET 2.2** (source at tag `v2.2`, files `hydstatus.c`, `hydcoeffs.c`, `hydsolver.c`,
`hydraul.c`, `input1.c`):

- `inithyd`/`initlinkflow` (`hydraul.c`) start every PRV, PSV and FCV with a setting in the
  ACTIVE state.
- `hydsolve` (`hydsolver.c`) calls `valvestatus()` after every linear solve, because
  `DAMPLIMIT = 0` by default. When the solve has converged, it also calls `linkstatus()` (CVs,
  pumps, links to tanks) and `pswitch()`, and keeps iterating if anything changed. `pswitch` is
  not part of the valve logic. It applies `[CONTROLS]` keyed on junction pressure
  (`LOWLEVEL`/`HILEVEL` against `Grade ± Htol`), after convergence only.
- `valvestatus` evaluates `hset = El[n2] + setting` for a PRV and `El[n1] + setting` for a PSV,
  then calls `prvstatus` or `psvstatus`. These are transition rules from the current status
  `s`, with `htol = Htol` and `hml = Km q^2`:

  ```
  prvstatus:                                   psvstatus:
   ACTIVE: q < -Qtol            -> CLOSED       ACTIVE: q < -Qtol            -> CLOSED
           h1 - hml < hset-htol -> OPEN                 h2 + hml > hset+htol -> OPEN
   OPEN:   q < -Qtol            -> CLOSED       OPEN:   q < -Qtol            -> CLOSED
           h2 >= hset+htol      -> ACTIVE               h1 < hset-htol       -> ACTIVE
   CLOSED: h1>=hset+htol && h2<hset-htol -> ACTIVE   CLOSED: h2>hset+htol && h1>h2+htol -> OPEN
           h1<hset-htol && h1>h2+htol    -> OPEN             h1>=hset+htol && h1>h2+htol -> ACTIVE
   XPRESSURE: q < -Qtol         -> CLOSED       XPRESSURE: q < -Qtol         -> CLOSED
  ```

  `HTOL = 0.0005 ft` (1.524e-4 m) and `QTOL = 1e-4 cfs` (2.83e-6 m3/s) are the defaults in
  `input1.c`. The `±htol` margins give the rules hysteresis, so the status that comes out
  depends on the iteration history.
- `prvcoeff` and `psvcoeff` (`hydcoeffs.c`) handle the ACTIVE state with a penalty:
  `Aii[n2] += CBIG` and `F[n2] += CBIG*hset` (`CBIG = 1e8`), so the controlled node's head is
  pinned to `hset`. The valve flow is then set from that node's flow excess (`Y = q + Xflow`).
  An OPEN valve uses `valvecoeff`, which is `m q|q|` with an `RQtol` floor, or the tiny linear
  resistance `CSMALL = 1e-6` when `Km = 0`. A CLOSED valve uses conductance `1/CBIG`. This is
  the source of the small leak flows EPANET reports through closed valves (measured: 2e-8 m3/s).
- `badvalve` (`hydsolver.c`) sets an ACTIVE valve that makes the matrix singular to
  `XPRESSURE`, which is open but unable to deliver the setting, and logs a warning.

**Measured with `wntr` 1.5.0 `EpanetSimulator`** (scratch scripts; not in the repository):

- Every expected state is reached on small series networks and on the two-loop fixture.
- On the two-loop fixture with `V1 J7 -> J2`, a PRV with setting 45 m comes out CLOSED, not
  ACTIVE: with the valve closed, J2 is fed through the other loop at 45.298 m, which is above the
  setting.
- **Hysteresis.** Sweep the reservoir head across the ACTIVE/OPEN boundary of a series PRV
  (setting 50 m; the boundary is at an upstream head of 50 m). EPANET stays ACTIVE until the
  upstream head is `HTOL` below the setting. Inside that band it reports `H_v = 50.000` while
  `H_u = 49.99986`, with forward flow: a negative head drop of up to 1.41e-4 m through an
  "active" valve. The exact switching rule gives OPEN there, with `H_v = H_u`. EPANET and the
  exact rule therefore differ by up to `HTOL` in head inside the band, and agree everywhere
  outside it.

## 2. Formulation inside noodl's solve

### Absolute heads

Each valve becomes a 2-port `ImplicitLaw`: port 1 is the valve edge `u -> v`, port 2 is a
*sense edge* from `v` (PRV) or `u` (PSV) to `DATUM`. The sense edge's residual is `q_s = 0`, so
it carries no flow, and its `dp_s` is the absolute head at the sensed node. The other head
follows from `H_u = dp + H_v`. This reuses the multi-port MNA blocks unchanged, and needs no
change to `_mna.py`.

The alternative is to give laws their end-node potentials (a new `LawContext` field plus extra
Jacobian columns). That is more general, but it means a change to the core MNA assembly.

### The switching condition as one equation

The three states above can be written as one complementarity condition, with no memory of the
previous status:

- PRV: `q >= 0`, `g >= 0`, `q * g = 0`, where `g = max(H_v - Hset, m q|q| - dp)`.
- PSV: the same, with `g = max(Hset - H_u, m q|q| - dp)`.

Here `g` is the "reason to be closed": the downstream head above the setting, or a head drop
too small to drive the flow. When `q > 0` and `g = 0`, the larger term inside the `max` decides
the state: ACTIVE when `H_v = Hset` and the drop exceeds the open loss, OPEN when
`dp = m q|q|` with `H_v <= Hset`.

The residual is `r = min(kappa * q, g)`, with `kappa` (m per m3/s) a fixed row scale; the
solution set does not depend on it. A converged root reproduces each EPANET state:

- `r = kappa*q` selected, with `q = 0`: CLOSED.
- the `H_v - Hset` term selected: ACTIVE.
- the `m q|q| - dp` term selected: OPEN.

### Options compared

| Option | Convergence | EPANET match | Gradient | Fit with `apps.water` |
|---|---|---|---|---|
| **A. Outer status loop** (EPANET-style: freeze status, solve, update by the §1 rules, repeat) | several full Newton solves; can cycle, which EPANET itself guards with hysteresis, CHECKFREQ and MAXCHECK | exact, including hysteresis | IFT of the frozen-status system | the status depends on the solution at the same instant, not on the previous step, so it does not fit the between-step `TankStorageControls` pattern; needs a new loop around `Model.steady` |
| **B. Semismooth Newton on `min`/`max`** (active set inside Newton) | one solve, locally quadratic; the existing damping guards against cycling | exact solution of the ideal valve; differs from EPANET only inside the `HTOL` band | IFT of the branch selected at `x*` (piecewise, see §3) | same mechanism as `BreakerValve` (`torch.where` on the iterate) |
| **C. Smoothed Fischer–Burmeister** `a + b - sqrt(a^2 + b^2 + 2 eps^2)` | smooth, robust Newton | biased by `O(eps)` near every switch: `q*g = eps^2` | continuous across the switch | new kind of approximation in a code that elsewhere reproduces EPANET exactly |

**Recommendation: B.**

- It is the PBV mechanism extended to a complementarity condition. It takes one Newton solve,
  with no new outer loop and no new status state carried across solves.
- It gives the exact ideal-valve answer, which matches EPANET outside the `HTOL` band.
- Its IFT gradient is exact everywhere except on the switching set itself.
- It is written with `torch.where` and explicit tie rules, not `torch.minimum`/`torch.maximum`.
  At a tie, PyTorch splits the gradient 0.5/0.5 between the two arguments (checked). That would
  put a blended row into the Jacobian at an exact tie.
- Newton is seeded with every valve ACTIVE, as EPANET does: the seed flow is the
  `initial_flows` velocity.
- If damped Newton fails to converge, the law can raise by name using the existing singular
  diagnosis. Two parallel PRVs with equal settings, or a PRV pinning a node that a reservoir
  already fixes, make the Jacobian singular; these are EPANET's `XPRESSURE` cases.

Option C is worth keeping as an opt-in setting for optimisation across a switch (see §5).

## 3. The gradient at the switch

The solution map is continuous and piecewise smooth in the setting, demands and roughness:
heads and flows are continuous across each switch, but their derivatives jump.

Inside a state's region, the IFT gradient is the exact derivative of that state's
smooth system:

- **ACTIVE PRV.** `dH_v/dsetting = 1`. The upstream side does not influence the downstream
  heads: in a tree, `dH_downstream/d(roughness or demand upstream) = 0` exactly. Downstream
  demands flow back through the valve as `dq_valve`.
- **OPEN.** The valve is a minor loss, and `d(anything)/dsetting = 0`.
- **CLOSED.** `dq_valve = 0`. The two sides are coupled only through the rest of the network,
  and `d/dsetting = 0`.

On the boundary, both one-sided derivatives exist and differ. Option B returns the derivative
of the branch that the `torch.where` tie rule selects; the rule is documented and
deterministic. That derivative is a limiting (B-subdifferential) element, not an average.
Finite differences straddling the switch will not match it. This is the same situation as
today's PBV at `m q^2 = setting`.

With option C, the gradient is smooth and blends the two branches over a band of width
`O(eps)`. For example, `dH_v/dsetting` rises from 0 to 1 across the band, which gives an
optimiser a signal on the OPEN side. The forward answer moves by `O(eps)` in exchange.

## 4. Verification plan

Add to `tests/verification/test_water_parity.py`, reusing `_epanet`, `_valve_edits` and the
two-loop fixture with J7. Settings below are from the `wntr` runs (`H_J1 = 49.268`,
`H_J7 = H_J2 = 48.877` with the valve fully open).

| Case | Network and valve | EPANET status (measured) |
|---|---|---|
| PRV active | two-loop, `V1 J7 J2 250 PRV 48 0` | ACTIVE, `H_J2 = 48.000` |
| PRV open (upstream below setting) | two-loop, PRV 49.9 | OPEN, `H_J7 = H_J2 = 48.877` |
| PRV open with minor loss | series `R1 -> P1 -> J1 -> PRV -> J2 -> P2 -> J3`, R1 = 40 m, setting 50, `Km = 10` | OPEN, `H_J1 - H_J2 = 0.206` |
| PRV closed by reverse flow | two-loop, PRV reversed (`V1 J2 J7`), 49.9 | CLOSED, `q = 0` |
| PRV closed by downstream above setting | two-loop, PRV 45 | CLOSED, `H_J2 = 45.298` |
| PSV active | series `R1(100) -> P1 -> J1 -> PSV 70 -> J2 -> P2 -> R2(20)` | ACTIVE, `H_J1 = 70.000` |
| PSV open | same, setting 30 | OPEN |
| PSV closed (upstream below setting) | same, R1 = 60, setting 70 | CLOSED |
| PSV closed (reverse) | same, R1 = 60, setting 30, R2 = 80 | CLOSED |
| PRV and PSV in series | `R1(100) -> J1 -PRV 60-> J2 -> J3(demand 10 L/s) -PSV 55-> J4 -> R2(10)` | both ACTIVE, `H_J2 = 60`, `H_J3 = 55` |
| Near the switch | series PRV setting 50 with R1 swept over `[53.80, 53.85]` (boundary at 53.8214) | OPEN below, ACTIVE above, hysteresis band of `HTOL` |

**Tolerances**, as for the PBV test:

- heads: 1e-6 relative;
- flows: 1e-6 m3/s absolute, because EPANET's `CBIG` penalty and its `1/CBIG` closed-valve leak
  give flows of order 1e-8 m3/s;
- the controlled head: equal to `Hset` to Newton tolerance in noodl, and within 1e-6 of
  EPANET's;
- state: compare noodl's state, read from the converged residual branch, with wntr's
  `link["status"]` (0 closed, 1 open, 2 active).

**Near-switch case.** Sample points at least `2 HTOL` from the boundary must agree to the
tolerances above. Points inside the band must agree within `HTOL = 1.524e-4 m` in head. The
docstring states that EPANET holds ACTIVE there with a negative head drop (measured).

**Gradient checks**, against central differences with a step well inside the region (margin
measured from the boundary):

- PRV active on the two-loop: d(sum of junction heads)/d(demands), d/d(roughness),
  d/d(setting). Expect exactly 1 for `dH_J2/dsetting` in the tree case.
- PRV open: `d/dsetting = 0`.
- PSV active: d/d(demands).
- One-sided checks at `±delta` either side of the boundary, confirming that the analytic
  gradient jumps between the two one-sided finite differences.

Tolerance: `max|analytic - fd| <= 1e-6 max|fd|`, as in
`test_gradient_through_a_valve_network_against_central_differences`.

**Unit tests**: one per state on hand-solvable series networks; the `.inp` reader accepts
PRV/PSV; validation (negative setting, a valve touching a reservoir on its controlled side)
raises by name.

## 5. Open questions for the user

1. **Exact or smoothed.** Should the default be exact (B), matching EPANET outside `HTOL`? And
   should the smoothed option (C, `eps` in metres) be offered at all, for calibrating settings
   across a switch?
2. **Hysteresis.** Do you accept that noodl gives the ideal-valve answer inside EPANET's `HTOL`
   band, where EPANET reports ACTIVE with a negative head drop? Or must noodl reproduce
   EPANET's history-dependent status, which forces option A?
3. **Absolute head.** Sense edges to `DATUM` keep the change inside `apps.water`, but add hidden
   zero-flow edges. Those must be kept out of `link_names`, reporting, and the quality
   layer, which today refuses a datum (`tank_mode="storage"` with quality). The alternative
   is a core change that gives laws their node potentials. Which do you prefer?
4. **Differentiable setting.** Should the setting be a learnable parameter, a driver (so
   `[CONTROLS]`/rules or patterns can change it per step), or both? PBV's is currently fixed.
5. **Unsatisfiable settings.** When a setting cannot be met (EPANET's `XPRESSURE`/`badvalve`):
   should noodl raise by name, as for pump `q_max`, or fall back to OPEN with a warning, as
   EPANET does?
6. **Scope.** Should the same complementarity residual also bring in CV pipes and
   `[STATUS] CLOSED` links in this PR? A CV is the PRV's reverse-flow half with
   `g = h(q) - dp`. Or should they stay refused?
7. **Fixed status.** PRV/PSV with a fixed `[STATUS] OPEN/CLOSED`, which EPANET models by setting
   the setting to `MISSING`: support it now, or keep refusing `[STATUS]`?
