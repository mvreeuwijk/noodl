# Water resources: theoretical background

Background for [flow allocation](allocation.md) and [WSIMOD](wsimod.md), moved from the framework theory page ([theory](../../concepts/theory.md)).

## 10. A fourth flow-determination mode: clip-and-allocate flow allocation

**Why this is a fourth way of determining flows, not a variant of the Kirchhoff layer.**
Sections 1 and 9 above both describe networks where flow at an edge is either the closed-
form output of a Newton Kirchhoff solve or of continuity alone on a tree. WSIMOD's own
`Arc`/`Node` model (Dobson, Liu and Mijic, JOSS 2023; GMD 2024) is neither: an edge is asked
to carry a REQUESTED flow (typically emitted by some upstream process closure, not a
pressure difference) and simply clips it against two independent bounds -- its own arc
capacity and the downstream node's remaining storage headroom -- with no potential variable
anywhere in the calculation. This is the fourth way an edge's flow can be determined (alongside a Kirchhoff Newton solve, a driver-prescribed
flow, and a closure-computed flow), and `AllocatedFlowLayer`
(`src/noodl/layers/allocation.py`) is its implementation: a `Model` layer type that owns
one or more edge kinds exactly like `KirchhoffLayer` does, but whose per-step output is
an explicit clip-and-allocate computation rather than a solve.

**The clip/allocation math, condensed.** Given the previous
per-node storage $s$, a per-edge capacity $c_{\text{arc}}$ and a per-edge REQUEST driver $r$:
receiver headroom is $h = (s_{\max} - s)/\Delta t$ at each edge's downstream node (a boundary node
has $s_{\max} = \infty$ and never constrains) -- a RATE, in the same units as $r$ and $f$, which is
what makes the clip bound the actual storage INCREASE $\Delta t\, h = s_{\max} - s$ at any $\Delta t$ and
not only at $\Delta t = 1$; the realised flow in hard-clip mode is
$f = \min(r, c_{\text{arc}}, h)$, vectorised over every edge via the incidence matrix in one pass, no
Python loop over nodes or edges. Where more than one edge converges on a node whose combined
tentative demand exceeds its free headroom, each edge's share is instead a preference-
weighted fraction of that headroom, resolved by a small FIXED number of vectorised passes
(`n_passes`, default 5, matching WSIMOD's own `constants.MAXITER`) rather than WSIMOD's own
per-node bounded `while`-with-early-exit -- each pass is strictly non-expansive, so a fixed
count no worse than WSIMOD's own cap is a safe over-approximation of the same fixed point,
not an approximation of a different algorithm. This sharing is deliberately
RECEIVER-LOCAL: it triggers only at the node actually oversubscribed and never propagates a
downstream bottleneck back to an earlier edge in the same step. That matches WSIMOD's
storing nodes, whose accept decision is against their own headroom, but not a WSIMOD
pass-through `Node`, which accepts only what it can forward within the step; it matches what
storage itself is for in a discrete-time capacity-limited network. The storage update is $s' = \min(s + A_{\text{in}} f - A_{\text{out}} f,\ s_{\max})$ with
`overflow = relu(...)` reported as a diagnostic, never fed back -- and deliberately has NO
floor at zero: $f$ is already clipped against the RECEIVER's headroom, never the sender's
own available storage, so a caller requesting more than is actually available upstream is
a modelling responsibility upstream of this layer (a closure sizing requests correctly),
exactly as a `KirchhoffLayer`'s boundary sources are trusted as given rather than
second-guessed.

**Two differentiability modes, and why the sharing site needed a real solve, not just a
softened `min`.** `"smooth"` replaces every `min`/`clamp`/`relu` above with its
softmin/softplus counterpart at a construction-time temperature $\tau$ -- ordinary autograd
through the vectorised passes, no new solver machinery, exactly the same style as the
building application's smoothed switches elsewhere in this codebase. `"projection"` poses
the allocation at an oversubscribed node as a genuine per-node QP -- minimise
$\sum_i \text{pref}_i\, (f_i - \text{remaining}_i)^2$ subject to $0 \le f_i \le \text{avail}_i$ per edge and
$\sum_i f_i \le \text{free headroom}$ jointly -- because the FIRST implementation of this mode
reused hard-clip's own preference-proportional share formula, which depends only on
preference weights and total headroom, never on any individual edge's own request; being
provably independent of a competitor's request, it could not carry a gradient between
competing edges no matter how it was wrapped, defeating the purpose of this mode, which
is that gradients flow through which arc absorbs a constraint. The real QP's
Lagrangian stationarity, with the box bound applied, reduces to
$f_i = \operatorname{clamp}(\text{remaining}_i - \lambda/\text{pref}_i,\ 0,\ \text{avail}_i)$ for one scalar $\lambda$ SHARED by every edge competing at
that node -- so $\partial f_i / \partial r_j$ for a competing edge $j \ne i$ is genuinely nonzero, flowing
entirely through $\lambda$. $\lambda$'s root is monotone in $\lambda$ by construction (every
edge's contribution shrinks as $\lambda$ grows), which is exactly
`noodl.solvers.scalar.solve_monotone`'s own contract (a batched, monotone,
implicit-function-differentiable scalar root) -- the same primitive the sewer application
uses for Manning-depth inversion (section 9 above) -- rather than new solver machinery or a
Fischer-Burmeister-smoothed complementarity condition fed through
`noodl.solvers.implicit.implicit_solve`, which was investigated and rejected as strictly
harder to verify for no accuracy benefit. The resulting cross-gradient,
$\partial f_{BD} / \partial r_{CD} = -0.5$ on this specific SYMMETRIC-preference four-node diamond test
fixture, was hand-derived and is checked directly (sign and magnitude) by
`test_projection_mode_sharing_has_nonzero_cross_gradient`. The general mechanism behind
it is the KKT derivation
$\partial f_i / \partial r_j = -(1/\text{pref}_i) / \sum_k (1/\text{pref}_k)$, of which −0.5 is the
symmetric-weight special case; it was also hand-checked against asymmetric preferences
(`pref=[1,3]` and `pref=[1,2,5]`), where the value differs (e.g. `pref=[1,3]` gives
−0.25).

**Verification, and what it does and does not show.** `tests/verification/
test_allocation_wsimod_replay.py` replays WSIMOD's own captured per-arc requests from its packaged
`quickstart_demo` and `oxford_demo` scripts through `AllocatedFlowLayer` and compares
against WSIMOD's own realised flows -- WSIMOD is the reference implementation here, the same
relationship pyswmm and EPANET have with the sewer and water applications above, and it is
NOT an independent measurement. This comparison has a real, specific limitation: of
`quickstart_demo`'s 6 arcs and `oxford_demo`'s 21 arcs, all but one sit at WSIMOD's own
unbounded capacity for the whole run, and the one finite-capacity arc never sees its request
approach its own capacity either -- so neither demo's numbers ever exercise the branch where
$c_{\text{arc}}$ actually binds, only the identity path $\min(x, c_{\text{arc}}) = x$, and both set
$s_{\max} = \infty$. `tests/verification/test_allocation_wsimod_capacity.py` closes that gap against
WSIMOD's own numbers: scripted networks of WSIMOD's own classes in which arc capacity and tank
headroom bind, including several requests on one arc per timestep (WSIMOD clips each against
the arc's accumulated `flow_in`, so the timestep total is $\min(\sum_k r_k, c_{\text{arc}})$ in
any order), and a `quickstart_demo` variant with three capacities lowered. It also pins where
the semantics differ: several arcs competing for one node's headroom are served
first-come-first-served by WSIMOD but preference-proportionally here, and a bottleneck behind
a WSIMOD pass-through `Node` propagates upstream within the step, which this receiver-local
layer does not do (see
[where the layer differs from WSIMOD](allocation.md#where-the-layer-differs-from-wsimod)).
