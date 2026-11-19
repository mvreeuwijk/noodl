# Flow allocation: the physics

This page explains how [flow allocation](allocation.md) computes a flow, why it has three
differentiability modes, and how competing edges share a node's headroom. The layer is the
fourth way a noodl model can determine an edge's flow, next to a Kirchhoff Newton solve, a
driver-prescribed flow and a closure-computed flow.

## Why it is not a variant of the Kirchhoff layer

In a Kirchhoff network, the flow on an edge is the output of a solve for node potentials, or of
continuity alone on a tree. WSIMOD's `Arc` and `Node` model (Dobson, Liu and Mijic, JOSS 2023;
GMD 2024) is neither. An edge is asked to carry a *requested* flow, typically emitted by an
upstream process closure rather than a pressure difference, and clips it against two
independent bounds: its own arc capacity and the downstream node's remaining storage headroom.
Water-resources networks (abstraction licences, reservoir operating rules, treatment-works
throughput) are governed by rules, and the rule is the physics.
`AllocatedFlowLayer` owns one or more edge kinds exactly as `KirchhoffLayer` does, but its
per-step output is an explicit clip-and-allocate computation, not a solve.

## The clip

Given the previous per-node storage $s$, a per-edge capacity $c_{\text{arc}}$ and a per-edge
request $r$, the receiver headroom is

$$
h = \frac{s_{\max} - s}{\Delta t}
$$

at each edge's downstream node (a boundary node has $s_{\max} = \infty$ and never constrains).
It is a rate, in the same units as $r$ and $f$, which makes the clip bound the actual storage
increase $\Delta t\, h = s_{\max} - s$ at any $\Delta t$. The realised flow in hard mode is
$f = \min(r, c_{\text{arc}}, h)$, vectorised over every edge through the incidence matrix in
one pass, with no Python loop over nodes or edges.

The storage update is $s' = \min(s + A_{\text{in}} f - A_{\text{out}} f,\ s_{\max})$, with
`overflow` reported as a diagnostic and never fed back. There is deliberately no floor at zero:
$f$ is clipped against the *receiver's* headroom, never the sender's available storage, so a
caller requesting more than is available upstream is a modelling responsibility upstream of this
layer, exactly as a `KirchhoffLayer`'s boundary sources are trusted as given.

## Proportional sharing and `n_passes`

Sharing happens only at a node that is the target of more than one edge. A node with a single
in-edge is never touched, however many out-edges lie downstream: the layer never caps an edge to
match a bottleneck further along the graph. That matches WSIMOD's storing nodes (`Storage`,
`Waste`), whose accept decision is a check against their own headroom. It does not match a
WSIMOD pass-through `Node`, which accepts only what it can forward in the same step.

Each of the `n_passes` rounds computes a tentative flow, finds the nodes whose summed demand
exceeds their free headroom, and on those nodes replaces the tentative flow with a
preference-weighted share. The preference is renormalised over the edges still actively
competing, so an edge that already got everything it asked for does not soak up a share it no
longer wants. That renormalisation is what `n_passes` is for. Three edges into one sink with
headroom 3.0, equal preference, requests `[0.5, 10.0, 10.0]`:

| Passes | Realised | Total |
|---|---|---|
| 1 | `[0.5, 1.0, 1.0]` | 2.5, which under-uses the headroom by 0.5 |
| 2 | `[0.5, 1.25, 1.25]` | 3.0: the first edge's freed weight redistributes |
| 5 | `[0.5, 1.25, 1.25]` | 3.0: already converged at pass 2 |

The remaining headroom only shrinks and the flow only grows round over round, so more passes
move a share closer to the true max-min-fair split and never past it. The default of 5 matches
WSIMOD's own iteration cap. WSIMOD uses a `while` loop with early exit; noodl uses a fixed
count so the computation batches and differentiates. Each round is strictly non-expansive, so a
fixed cap is a safe over-approximation of the same fixed point, not a different algorithm.

## The three modes

The modes compute the same thing. They differ in what the gradient does, and that is the entire
reason there is more than one.

### `"hard"`: exact, and zero gradient at the clip

Plain `minimum` and `clamp`, bit-identical to the bare clip formula. This is the mode WSIMOD
parity is measured against. It is differentiable almost everywhere, but at and near a capacity
or headroom crossing the gradient through the discrete branch choice is exactly zero: a node
moving from "headroom covers demand" to "oversubscribed" carries no gradient signal across that
boundary.

### `"smooth"`: a live gradient across the boundary

`minimum` becomes a softmin, $-\tau \log\sum e^{-a/\tau}$; `clamp` becomes a softplus; and the
branch selection itself is smoothed by a sigmoid-weighted blend of two whole branches, not just
the arithmetic inside each. Smoothing only the arithmetic would leave the discrete choice, and
its zero gradient, in place.

The cost is an $O(\tau)$ forward bias. On the WSIMOD quickstart replay the measured discrepancy
of 1.79e-3 is not capacity-smoothing residue, because every arc in that fixture is unbounded and
the softmin is exact to float64. It is the softplus-at-zero bias,
$\tau \ln 2 \approx 6.9\times10^{-4}$ at $\tau = 10^{-3}$, at kink sites many requests sit
exactly on, compounding through recurrent storage.

### `"projection"`: cross-gradients between competing edges

Smooth mode's proportional-share formula is
$h_{\text{free}} \cdot \text{pref}_i / \sum_k \text{pref}_k$. It depends only on the preference
weights and the total headroom, never on any competitor's request, so
$\partial f_i / \partial r_j = 0$ for a competing edge $j$, provably and not approximately. No
wrapping of that formula can deliver a gradient through *which arc absorbs a constraint*.
Projection mode therefore routes the sharing site through a coupled quadratic programme:
minimise $\sum_i \text{pref}_i (f_i - r_i)^2$ subject to $0 \le f_i \le \text{avail}_i$ per edge
and $\sum_i f_i \le h_{\text{free}}$ jointly. Its KKT stationarity reduces to

$$
f_i = \operatorname{clamp}\!\left(r_i - \frac{\lambda}{\text{pref}_i},\ 0,\ \text{avail}_i\right)
$$

for one scalar $\lambda$ shared by every edge competing at that node. $\lambda$ is the unique
root of a monotone scalar equation, which is the contract of `noodl.solvers.scalar.solve_monotone`
(batched, monotone, differentiated through the implicit function theorem), the same primitive
the sewer application uses for Manning-depth inversion. The shared multiplier produces a genuine
cross-gradient:

$$
\frac{\partial f_i}{\partial r_j} = -\frac{1/\text{pref}_i}{\sum_k 1/\text{pref}_k}
$$

On a symmetric-preference diamond this gives $\partial f_{BD} / \partial r_{CD} = -0.5$, derived
by hand and checked for sign and magnitude. The general formula was also checked against
asymmetric preferences (`pref=[1,3]` gives $-0.25$; `pref=[1,2,5]`) to four decimal places,
which confirms that the mechanism generalises, not that $-0.5$ itself does.

A full Newton treatment of the quadratic programme was considered and rejected: it needs a
smoothed complementarity condition, which is a harder-to-verify reimplementation of what smooth
mode already does, for no accuracy benefit.

## What the evidence shows

The replay of WSIMOD's captured per-arc requests through the layer compares against WSIMOD's own
realised flows. WSIMOD is the reference implementation, not an independent measurement. On its
two shipped demonstration models all but one arc sit at WSIMOD's unbounded capacity for the
whole run, and the one finite arc never approaches its capacity, so those runs exercise only
the identity path $\min(x, c_{\text{arc}}) = x$. Scripted networks of WSIMOD's own classes, in
which arc capacity and tank headroom bind, close that gap, and a variant of the quickstart with
three capacities lowered adds a long run. WSIMOD clips each request against the arc's accumulated
flow in the timestep, so the split between several requests depends on their order, but the
timestep total is $\min(\sum_k r_k, c_{\text{arc}})$ in any order, and that total is what one
layer step computes from the summed request. The tables are on the
[verification page](../../reference/verification/water-resources.md).
