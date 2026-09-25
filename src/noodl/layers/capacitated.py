"""Explicit-in-time clip/allocation layer for capacitated networks.

Determines edge flows by clipping requests against arc capacity and receiver headroom
rather than solving for a potential -- the fourth way of determining flows the framework
identifies (alongside potential-flow Newton solves, driver-prescribed flows and
closure-computed flows). Conservation is exact by construction: every clipped unit of
flow is either realised on the edge or left unmet at the source; overflow at a full node
is a separate, reported quantity, never silently dropped.

The docstrings below record the mathematics and the failure modes each guard exists for,
which is what a later reader needs.
"""
from __future__ import annotations

import torch

from noodl.solvers.scalar import solve_monotone
from noodl.topology import Network

F64 = torch.float64
_MODES = ("hard", "smooth", "projection")
# Fixed iteration count for `_clip_projection`'s own inner box-projection solve --
# deliberately SEPARATE from `n_passes` (the outer sharing loop's round count, a different
# algorithm with a different convergence question). A separable box QP's projected-gradient
# step (unit step size) reaches its exact optimum in a single iteration (see
# `_clip_projection`'s docstring), so 3 is generous headroom, not a tuned tolerance.
_PROJECTION_ITERS = 3
# Guards `_share_via_qp`'s third degeneracy (see its docstring):
# a node's `free_headroom` can shrink to ~1e-12 from accumulated floating-point cancellation
# in a LATER `n_passes` round once its headroom is already fully consumed, comparable to the
# `1e-12` epsilon `step` itself uses for the oversubscription test -- occasionally
# misclassifying pure numerical noise as genuine oversubscription and routing a near-zero-
# scale problem through `solve_monotone`'s real Newton solve, where it was observed (a
# 1-in-2000 randomised trial) to produce a NaN gradient. This floor is many orders of
# magnitude above float64's accumulated noise over the loop's own few arithmetic passes, and
# many orders below any physically meaningful headroom value in this class's own fixtures.
_OVERSUBSCRIBED_FLOOR = 1e-9


class CapacitatedTransferLayer:
    def __init__(
        self,
        net: Network,
        name: str,
        kind: str,
        *,
        s_max: torch.Tensor,
        c_arc: torch.Tensor,
        preference: torch.Tensor | None = None,
        mode: str = "hard",
        tau: float | None = None,
        n_passes: int = 5,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(
                f"CapacitatedTransferLayer {name!r}: mode must be one of {_MODES}, "
                f"got {mode!r}"
            )
        if mode == "smooth" and tau is None:
            raise ValueError(
                f"CapacitatedTransferLayer {name!r}: mode='smooth' requires tau"
            )
        # `tau` is a temperature: every smooth-mode helper divides by it unconditionally.
        # `tau == 0` gives a division by zero (inf/NaN); `tau < 0` is worse -- it silently
        # flips `_clip`'s softmin into a softMAX and `_select`'s sigmoid blend the wrong way
        # round, a WRONG ANSWER rather than a crash. Checked for any mode, not only
        # "smooth": a tau passed alongside "hard"/"projection" is ignored, and a nonsense
        # value there is a configuration error worth naming rather than quietly accepting.
        if tau is not None and not tau > 0:
            raise ValueError(
                f"CapacitatedTransferLayer {name!r}: tau must be > 0, got {tau!r}"
            )
        if n_passes < 1:
            raise ValueError(
                f"CapacitatedTransferLayer {name!r}: n_passes must be >= 1, got {n_passes}"
            )
        try:
            src, tgt = net.endpoints(kind)
        except KeyError as exc:
            raise ValueError(
                f"CapacitatedTransferLayer {name!r}: edge kind {kind!r} has no edges "
                f"in the network"
            ) from exc
        n_edges = src.numel()
        if c_arc.shape[-1] != n_edges:
            raise ValueError(
                f"CapacitatedTransferLayer {name!r}: c_arc has trailing size "
                f"{c_arc.shape[-1]}, expected {n_edges} (edge kind {kind!r})"
            )
        if s_max.shape[-1] != net.n:
            raise ValueError(
                f"CapacitatedTransferLayer {name!r}: s_max has trailing size "
                f"{s_max.shape[-1]}, expected {net.n} (every node in the network)"
            )
        if preference is not None:
            if preference.shape[-1] != n_edges:
                raise ValueError(
                    f"CapacitatedTransferLayer {name!r}: preference has trailing size "
                    f"{preference.shape[-1]}, expected {n_edges} (edge kind {kind!r})"
                )
            # `mode="projection"` uses `preference` as a LIVE DIVISOR (`lambda /
            # preference_i`, `_share_via_qp`'s KKT stationarity), where a zero weight gives
            # a clean forward value and a silent NaN GRADIENT in that edge's Jacobian
            # column -- the one failure this class cannot report from its own output.
            # (Hard and smooth modes only ever SUM preferences, and their `pref_sum` is
            # floored at 1e-30, so they degrade gracefully; the refusal is uniform anyway,
            # since a non-positive preference weight has no meaning in any mode.) `None`
            # means "all ones" and never reaches here.
            if not bool((preference > 0).all()):
                # `.nonzero().tolist()` (index TUPLES, not a flattened run of ints) so a
                # batched `preference` names the offending (instance, edge) pairs, not a
                # meaningless interleaved list. `> 0` is negated rather than `<= 0` tested
                # so a NaN weight is caught too.
                bad = (~(preference > 0)).nonzero().tolist()
                raise ValueError(
                    f"CapacitatedTransferLayer {name!r}: preference must be > 0 everywhere; "
                    f"entries {bad} are not (mode='projection' divides by it, and a "
                    f"non-positive weight there is a silent NaN gradient)"
                )
        self.net = net
        self.name = name
        self.kinds = [kind]
        self.kind = kind
        self.mode = mode
        self.tau = tau
        self.n_passes = int(n_passes)
        self._src = src
        self._tgt = tgt
        self.s_max = s_max
        self.c_arc = c_arc
        self.preference = torch.ones_like(c_arc) if preference is None else preference

    def _clip(self, a, b):
        """`min(a, b)` in hard mode (a bare `torch.minimum`, so hard mode is untouched by
        the existence of the other modes); in smooth mode, the softmin counterpart at
        temperature `self.tau`: `-tau * logsumexp([-a/tau, -b/tau])`, which is smooth
        everywhere and -> `min(a, b)` as `tau -> 0`. Used at every UPPER-bound kink site: arc
        capacity, receiver headroom, request-vs-availability, and the final storage
        clamp against `s_max`.
        """
        if self.mode == "smooth":
            return -self.tau * torch.logsumexp(
                torch.stack([-a / self.tau, -b / self.tau]), dim=0
            )
        return torch.minimum(a, b)

    def _nonneg(self, x):
        """`max(x, 0)` in hard mode (a bare `torch.clamp(x, min=0.0)`, so hard mode is
        untouched by the other modes); in smooth mode, the softplus counterpart at
        temperature `self.tau`: `tau * softplus(x / tau)`, smooth everywhere and ->
        `max(x, 0)` as `tau -> 0`. Used at every LOWER-bound-at-zero kink site: free
        headroom, avail, and the realised share.
        """
        if self.mode == "smooth":
            return self.tau * torch.nn.functional.softplus(x / self.tau)
        return torch.clamp(x, min=0.0)

    def _select(self, hard_cond, soft_margin, a, b):
        """Selects `a` where oversubscribed, else `b` -- the pass loop's
        `over_subscribed_e` branch (proportional share vs. plain tentative flow).

        In hard mode this is `torch.where(hard_cond, a, b)`, with `hard_cond` the plain
        boolean `demand_at_tgt > free_headroom + 1e-12` -- hard mode's arithmetic is
        untouched by the existence of the other two modes, byte for byte.

        In smooth mode, `hard_cond`/`soft_margin` is a boolean `torch.where` selecting
        between two WHOLE FORMULAS, not a `min`/`max`/`clamp` of two scalars, so it has
        no single obvious softmin/softplus counterpart. DESIGN DECISION (`mode="projection"`
        faces this identical boolean site and resolves it the other way, see below): we
        smooth the SELECTION itself, via a sigmoid-weighted blend of the two whole branches at
        temperature `self.tau` -- `w = sigmoid(margin / tau)`, `w * a + (1 - w) * b`
        (`margin = demand_at_tgt - free_headroom`, positive exactly when
        oversubscribed) -- rather than leaving the branch hard and smoothing only the
        arithmetic inside each branch. Rationale: a hard boolean select is
        differentiable almost everywhere, but AT and near the crossing point (a node
        moving from "headroom covers every tentative demand" to "oversubscribed") the
        gradient through the discrete branch choice is exactly zero, even though that
        crossing is itself a smooth function of the differentiable inputs (r, s_max,
        c_arc, ...). That is precisely the point `mode="smooth"` exists to fix -- a
        live gradient across a capacity boundary -- so leaving this one boolean hard
        would defeat the mode's purpose at the one site most likely to sit exactly on
        a boundary.

        `"projection"` makes the OPPOSITE choice at the same question, deliberately: leave
        the oversubscription BRANCH decision hard, smooth/solve only the arithmetic inside
        it. Its purpose is a QP posed against the constraint
        SURFACE -- `0 <= f <= c_arc` and the headroom bound -- not a re-litigation of
        WHETHER a node is oversubscribed at all, which is a genuinely separate question
        from what the flows should be once it is. Either choice is defensible; both are
        deliberate and documented rather than incidental.

        `"projection"` does not reach this method at all: `step` calls `_share_via_qp`
        directly, which solves the real coupled QP and reproduces that same hard branch
        decision internally via a target-substitution trick (both documented there) rather
        than through this `torch.where`. `_select` governs `"hard"` and `"smooth"` only.
        """
        if self.mode == "smooth":
            w = torch.sigmoid(soft_margin / self.tau)
            return w * a + (1 - w) * b
        return torch.where(hard_cond, a, b)

    # Path taken for mode="projection" overall: the UNROLLED box-projection fallback (this
    # method) plus, at the sharing site specifically, `noodl.solvers.scalar.solve_monotone`
    # (see `_share_via_qp` below) -- NOT `noodl.solvers.implicit.implicit_solve` anywhere,
    # decided after reading `implicit.py`/`newton.py` in full (not a blind skip).
    # `implicit_solve` wants a smooth `residual(x, *params) = 0` and an `operator` returning a
    # Jacobian/LinearOperator at the current Newton iterate; this QP's KKT stationarity
    # condition has a genuine complementarity term (`f` pinned at 0, at its upper bound, or
    # strictly interior, with a DIFFERENT active Jacobian structure in each regime), so
    # presenting the FULL multi-edge QP as a Newton root-find would require smoothing the
    # complementarity condition first (e.g. Fischer-Burmeister) to get a residual Newton can
    # differentiate through -- at which point the "new" machinery is just a harder-to-verify
    # reimplementation of what `mode="smooth"` already does with `_clip`/`_nonneg`/`_select`,
    # for no accuracy benefit and with exactly that verification risk. `solve_monotone`
    # sidesteps this: at the sharing site, the QP's KKT system reduces to ONE scalar,
    # monotone equation per node (`_share_via_qp`'s docstring derives this), which is exactly
    # `solve_monotone`'s own contract (a batched, monotone, implicit-function-differentiable
    # scalar root) -- an existing, already-tested primitive, not new solver machinery.
    def _clip_projection(
        self, r: torch.Tensor, headroom_per_edge: torch.Tensor, c_arc: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Box-constrained least-squares projection: minimises `sum((f - r)**2)` subject to
        `0 <= f <= bound`, where `bound = headroom_per_edge` alone (`c_arc=None`) or `bound =
        min(c_arc, headroom_per_edge)`.

        Used ONLY at "QP site 1" in `step` -- each edge's own arc-capacity-and-headroom bound
        (`avail`), which has NO cross-edge coupling (a single edge's own capacity bound is not
        a shared-constraint problem), so the exact closed-form solution `clamp(r, 0, bound)`
        is correct and sufficient on its own; the loop below is a genuine (if
        trivially-converging, since a separable box QP's unit-step projected-gradient step
        reaches the optimum after exactly one projection) iterative solve of that QP, not a
        disguised call to `_clip`. It is deliberately NOT used at the sharing site ("QP site
        2"): that site has REAL cross-edge coupling, which a SEPARABLE box projection cannot
        represent -- applied there it computes a value mathematically identical to `_clip`'s
        hard-clip formula, carrying no cross-gradient at all. `_share_via_qp` handles that
        site instead; see its docstring.

        Solved by projected gradient descent at unit step size (`f <- clamp(f - (f - r), 0,
        bound)`) for a small FIXED `_PROJECTION_ITERS` count, independent of `self.n_passes`
        (which counts OUTER sharing rounds, a different loop solving a different problem --
        see `step`'s own docstring). Ordinary autograd differentiates straight through the
        unrolled loop.
        """
        bound = headroom_per_edge if c_arc is None else torch.minimum(c_arc, headroom_per_edge)
        zero = torch.zeros_like(r)
        f = torch.clamp(r, zero, bound)
        for _ in range(_PROJECTION_ITERS):
            grad = f - r
            f = torch.clamp(f - grad, zero, bound)
        return f

    def _share_via_qp(self, remaining, avail_e, free_headroom, node_oversubscribed):
        """The REAL coupled per-node QP at the sharing site --
        minimise `sum_i preference_i * (f_i - remaining_i)**2` subject to
        `0 <= f_i <= avail_i` for every edge `i` targeting an oversubscribed node, AND
        `sum_i f_i <= free_headroom` for that node JOINTLY (not per edge).

        WHY A SEPARABLE CLIP CANNOT DO THIS JOB, stated here because the obvious
        simplification is to reuse `_clip_projection` or `_clip` at this site and it silently
        produces the wrong DERIVATIVE while looking right in forward value: any formula built
        on the preference-proportional split `free_headroom * preference_i / pref_sum`
        depends only on preference weights and total headroom, never on any individual edge's
        own request, so `d(share_i)/d(r_j) == 0` for a competing edge `j` -- provably, not
        approximately. That defeats the purpose of the layer's differentiable modes
        (gradients should flow through which arc absorbs a constraint), and it is invisible to a
        forward-value test: `torch.autograd.functional.jacobian` on such an implementation
        gives byte-identical zero cross-terms and byte-identical full Jacobians to
        `mode="hard"`, across 200 random trials and the diamond fixture.

        This QP's solution does NOT have that flaw. Its Lagrangian (`lambda >= 0` by
        complementary slackness on the joint inequality) is `sum_i preference_i * (f_i -
        remaining_i)**2 + lambda * (sum_i f_i - free_headroom)`; stationarity per edge, with
        the box bound applied, reduces to `f_i = clamp(remaining_i - lambda / preference_i, 0,
        avail_i)` for ONE scalar `lambda`, SHARED by every edge competing at that node. Since
        `lambda` depends on every competing edge's `remaining_j` jointly (see below),
        `d(f_i)/d(r_j)` for `i != j` is genuinely nonzero, flowing entirely through `lambda`
        -- this is the real cross-gradient wanted. The covering test,
        `test_projection_mode_sharing_has_nonzero_cross_gradient`, pins its sign: increasing a
        competitor's request should make ITS share bigger and, since the shared pool of
        `free_headroom` does not grow, every OTHER competitor's share smaller.

        `lambda`'s value is the unique root of `g(lambda, ...) = sum_i f_i(lambda) -
        free_headroom`, monotone NON-INCREASING in `lambda` (every term shrinks as `lambda`
        grows) -- exactly `noodl.solvers.scalar.solve_monotone`'s own contract (a batched,
        monotone, implicit-function-differentiable scalar root), the same primitive the sewer
        app already uses for Manning-depth inversion. `params` threaded through
        `solve_monotone` are `remaining`, `avail_e`, `self.preference` and `target` (below) --
        every tensor `g` depends on besides `lambda` itself, per `solve_monotone`'s own module
        docstring (a closure-captured tensor never reaches its backward pass).

        Bracket: `lo=0` always brackets the root from below -- `g(0, target=free_headroom) =
        demand_at_tgt - free_headroom >= 0` exactly when the node IS oversubscribed, by that
        condition's own definition. `hi` is a single GLOBAL scalar (not a tight per-node
        bound -- `solve_monotone` only needs a VALID bracket, not a tight one, and a global
        scalar avoids a per-node scatter-max op): `max(preference * clamp(remaining, min=0))
        + 1` over EVERY edge in the whole call. At that `lambda`, `remaining_i -
        lambda/preference_i <= 0` for every edge everywhere (not just this node's), so `g(hi)
        <= -free_headroom <= 0` for every node -- loose but always safe. `hi` is built with
        `.item()` (a plain Python float), deliberately detached: `solve_monotone` never
        returns a gradient for `lo`/`hi` (only for `*params`), so building it from a
        `requires_grad` tensor would be silently discarded anyway, and `.item()` makes that
        explicit rather than relying on the library's own (correct, but non-obvious) behaviour.

        NON-OVERSUBSCRIBED nodes (or nodes with no incoming edges at all) still need a VALID
        bracket -- `solve_monotone` raises if `g(lo)` and `g(hi)` share a sign, which they
        would here without adjustment (`g(0) = demand_at_tgt - free_headroom <= 0` when NOT
        oversubscribed, same sign as `g(hi) <= 0`). Rather than a dynamic per-node boolean
        mask (which would break the fixed, static-shape vectorisation every other site in this
        class relies on, and reintroduce a variable-size batch per call), the TARGET fed to
        `g` is substituted for these nodes instead: `target = free_headroom` where
        oversubscribed, else `demand_at_tgt` -- `g(0) = demand_at_tgt - target = 0` EXACTLY
        there, satisfying `solve_monotone`'s own `f_lo == 0` exemption from the sign check
        unconditionally. This makes `lambda = 0` an exact root for every non-oversubscribed
        node, so the solve trivially converges to the same `tentative` these nodes would get
        anyway -- the SAME hard, not-smoothed oversubscription-branch decision `_select`'s own
        docstring discusses for the ROUTING question (see there for why `"projection"` leaves
        it hard rather than blending it the way `"smooth"` does), just implemented here via a
        target substitution instead of `_select`'s `torch.where`, since `solve_monotone`'s own
        bracket-validity contract needs BOTH branches to share one call rather than a plain
        two-value choice.

        A SECOND, more GENERAL degeneracy (found only by a 200-trial randomised sweep -- no
        hand-written fixture in this suite reaches it, which is why
        `test_projection_mode_sharing_is_nan_free_under_random_fixtures` exists as a seeded
        sweep rather than another named case): whenever a node is NOT oversubscribed,
        `lambda* = 0` is the FORCED root (by the target-substitution trick above), and the
        residual's LOCAL derivative there is `sum_i (-1/preference_i) * [interior_i]`, where
        `interior_i` is 1 only for an edge STRICTLY between its bounds
        (`0 < remaining_i < avail_i`) and 0 for
        an edge sitting AT either bound (`remaining_i <= 0`, already fully satisfied by an
        EARLIER pass in `step`'s own outer loop -- extremely common in pass 2 onward, once a
        node's demand was already met in pass 1 -- or `remaining_i >= avail_i`, arc-capacity-
        saturated). If EVERY edge into that node is at a bound simultaneously (both the
        single-edge case above, AND a genuinely multi-edge, `in_degree >= 2` node where every
        competitor happens to be boundary-pinned -- observed directly: with `c_arc` large
        enough not to bind, `avail_i` is IDENTICAL across every edge sharing one node, so once
        one edge's demand is satisfied and its `remaining` drops to exactly 0 in a later pass,
        EVERY other still-active edge sharing that SAME node can independently also land at a
        bound), the derivative is exactly 0 and `solve_monotone`'s backward divides by it --
        the SAME `0/0 -> NaN` mechanism as the narrower case below, just triggered by
        `remaining_i == 0` (a LOWER-bound pin from an already-satisfied earlier pass) instead
        of `remaining_i >= avail_i` (an UPPER-bound pin), and by a genuinely competing node in
        a LATER pass rather than only a non-competing node in the first one.

        Because being NOT oversubscribed is *exactly* the condition under which this
        degeneracy can arise (an oversubscribed node's root is `lambda* > 0`, found only by
        actually leaving at least one edge's saturated regime, so at least one edge IS locally
        responsive at the converged point in every case this class has needed to handle), the
        fix below gates on OVERSUBSCRIPTION, not in-degree. A narrower `in_degree <= 1` gate
        is NOT sufficient and should not be substituted back in: it catches the single-edge
        chain-fixture case but misses this later-pass,
        genuinely-competing-but-currently-satisfied case entirely -- measured directly, that
        gate still produced NaN gradients in 101 of the same 200 trials.

        Fix: for every edge whose TARGET node is NOT oversubscribed (`node_oversubscribed`
        broadcast to edges), the values fed into `solve_monotone`'s OWN internal residual are
        replaced with fixed, safely-INTERIOR constants (`remaining=1.0`, `avail=2.0` -- `1.0`
        strictly between `0` and `2.0`, guaranteeing a genuine nonzero local derivative
        `-1/preference_i` there), and `target` is replaced correspondingly (`in_degree`, the
        COUNT of such edges into that node, since each dummy-substituted edge contributes
        exactly `min(1.0, 2.0) = 1.0` at `lambda=0`) -- so `lambda* = 0` is STILL the exact
        root there (unchanged in VALUE, since a non-oversubscribed node's real root was always
        0 anyway), but now via a non-degenerate residual. This is safe precisely because it
        only touches `solve_monotone`'s INTERNAL root-finding inputs: the FINAL returned share
        below is computed from the REAL `remaining`/`avail_e` (never the dummy constants), so
        a non-oversubscribed edge's realised share is unaffected, and the dummy constants are
        fresh, non-`requires_grad` leaves with no connection to `r` at all, so
        `d(lambda)/d(r) = 0` there regardless of `solve_monotone`'s own backward output --
        exactly what "share = tentative regardless of any other edge" means for a node with no
        ACTIVE competition pressure this pass, whatever its in-degree.

        A THIRD degeneracy (found by the SAME 200-trial randomised sweep, 1 trial in 2000 on
        a wider re-run): once a node's headroom is fully consumed in an early pass,
        `free_headroom` and `avail_e` do not always settle to EXACTLY 0 in later passes --
        floating-point cancellation across the subtraction chain (`headroom - committed_in`,
        `c_arc - f`, ...) can leave a residual of order `1e-12`, comparable to the SAME
        `1e-12` epsilon `step` uses for `node_oversubscribed` -- so a purely numerical-noise
        residual was occasionally misclassified as genuine oversubscription (observed: `demand
        ~2.6e-12 > free_headroom(~1.3e-12) + 1e-12`), routing an essentially-zero-scale
        problem through the REAL (non-dummy) branch, where `solve_monotone`'s Newton step
        divides by a derivative that is ITSELF subject to the same cancellation -- a second,
        distinct route to the same `0/0 -> NaN` failure, this time from floating-point noise
        rather than a structural boundary-pinning coincidence. Fix: `node_oversubscribed` is
        additionally required to clear `_OVERSUBSCRIBED_FLOOR` (`1e-9`, far above accumulated
        float64 noise over this loop's few arithmetic passes, far below any physically
        meaningful headroom value in this class's fixtures) before a node is trusted as
        GENUINELY oversubscribed here; a node whose margin is noise-scale is treated as
        not-oversubscribed instead (dummy-substituted, as above) -- correct up to floating-
        point noise itself, since there was never a meaningful amount of headroom left to
        contest either way.

        An analogous flat-residual risk remains, in principle, for a node that IS GENUINELY
        (not just noise-level) oversubscribed but whose target value happens to land exactly
        on a PLATEAU of the (piecewise-linear) residual -- every competing edge simultaneously
        boundary-pinned across a whole RANGE of `lambda`, not just at a single point -- which
        would require an exact coincidence between `free_headroom` and a specific combination
        of `avail_i` values. This was not observed in either randomised sweep (2200 trials
        total) or any fixture in this test suite; it is flagged here rather than silently
        assumed away.
        """
        in_degree = torch.zeros_like(free_headroom).index_add(
            -1, self._tgt, torch.ones_like(remaining)
        )
        node_oversubscribed = node_oversubscribed & (free_headroom > _OVERSUBSCRIBED_FLOOR)
        edge_oversubscribed = node_oversubscribed.index_select(-1, self._tgt)
        # Dummy pair is strictly interior (1.0 lies between 0 and 2.0) so the residual's
        # local derivative at lambda=0 is guaranteed nonzero for these edges -- see the
        # docstring's degeneracy paragraphs above.
        remaining_safe = torch.where(edge_oversubscribed, remaining, torch.ones_like(remaining))
        avail_safe = torch.where(edge_oversubscribed, avail_e, torch.full_like(avail_e, 2.0))
        # `in_degree` edges dummy-substituted (uniformly, since oversubscription is a NODE-
        # level property shared by every edge into it), each contributing min(1.0, 2.0) = 1.0.
        target = torch.where(node_oversubscribed, free_headroom, in_degree)
        lo = torch.zeros_like(free_headroom)
        hi_value = (self.preference * torch.clamp(remaining, min=0.0)).max().item() + 1.0
        hi = torch.full_like(free_headroom, hi_value)

        def _residual(lam, remaining_e, avail_e, preference_e, target):
            lam_e = lam.index_select(-1, self._tgt)
            f_e = torch.clamp(
                remaining_e - lam_e / preference_e, torch.zeros_like(remaining_e), avail_e
            )
            return torch.zeros_like(target).index_add(-1, self._tgt, f_e) - target

        lam = solve_monotone(
            _residual, lo, hi, remaining_safe, avail_safe, self.preference, target
        )
        lam_e = lam.index_select(-1, self._tgt)
        return torch.clamp(
            remaining - lam_e / self.preference, torch.zeros_like(remaining), avail_e
        )

    def step(self, s, drivers, dt, *, diagnostics=None):
        """One explicit step. `s` is the previous per-node storage, `dt` in seconds.

        Reads `f"{self.name}.requests"` from `drivers` (per edge, m3/s). Returns
        `(s_new, f)`: `s_new` the new per-node storage, `f` the realised per-edge flow
        (m3/s). `diagnostics`, when given, is filled with `"overflow"` (m3/s, per node).

        Proportional sharing: sharing only ever happens at a
        node that is the TARGET of more than one of this layer's edges -- "where more
        than one out-edge draws on one node's [headroom] supply". A node with a single
        in-edge is never touched by this loop, however many out-edges or requests are
        downstream of it: this layer never caps an edge to match a bottleneck further
        along the graph (that would need reasoning about paths, not incidence), exactly
        mirroring WSIMOD's own per-arc semantics: a node's
        accept decision is its own `push_check`/`pull_check` against its OWN storage
        headroom, never against its future ability to forward the flow onward.

        Each of the fixed `n_passes` rounds: computes `tentative` flow per edge (request
        still unmet, capped by remaining arc capacity and the TARGET's headroom still
        free after every edge's flow committed in EARLIER rounds); finds nodes whose
        summed tentative demand this round exceeds their free headroom; and on those
        nodes only, replaces `tentative` with a preference-weighted share of that free
        headroom (preference renormalised over the edges still actively competing --
        `tentative > 0` -- so an edge that has already gotten everything it asked for
        does not soak up a share it no longer wants). `remaining` only shrinks and `f`
        only grows round over round, so more passes can only move a share closer to the
        true max-min-fair split, never past it: an edge whose request is small enough to
        be satisfied outright in one round frees its unused preference weight for the
        REMAINING competitors in the next round -- this is what `n_passes` is for
        (WSIMOD's own bounded `while`-with-early-exit, `constants.MAXITER = 5`, replaced
        here with a fixed count for batched differentiability, each round strictly
        non-expansive so a fixed cap is a safe over-approximation, never an
        approximation of a different algorithm).
        """
        key = f"{self.name}.requests"
        r = drivers.get(key)
        if r is None:
            raise KeyError(
                f"CapacitatedTransferLayer {self.name!r}: driver {key!r} is required"
            )
        # Same shape-refusal pattern (and message shape) as `Model._kind_flows` uses for
        # the "<layer>.q" flow driver: a wrong-shaped `r` otherwise reaches `torch.minimum`
        # and surfaces as an unnamed broadcast `RuntimeError` naming neither this layer nor
        # the edge count it expected.
        n_edges = self.c_arc.shape[-1]
        if r.dim() == 0 or r.shape[-1] != n_edges:
            got = r.shape[-1] if r.dim() >= 1 else 0
            raise ValueError(
                f"CapacitatedTransferLayer {self.name!r}: driver {key!r} must have trailing "
                f"shape ({n_edges},) -- this layer owns {n_edges} edges of kind "
                f"{self.kind!r}, in that order -- got {got} (shape {tuple(r.shape)})"
            )
        # A RATE (m3/s), not a volume: `f` and every quantity derived from `headroom`
        # below (`committed_in`, `free_headroom`, `avail`, the whole sharing/QP machinery)
        # is a rate, and the storage update itself multiplies by `dt`. Dividing here is
        # what makes the receiver-headroom clip actually bound the storage INCREASE --
        # `dt * sum_in(f) <= dt * (s_max - s)/dt == s_max - s` -- at any `dt`, not only
        # at `dt == 1`, where the two conventions happen to coincide.
        headroom = (self.s_max - s) / dt
        remaining = r.clone()
        f = torch.zeros_like(r)
        for _ in range(self.n_passes):
            # Headroom already used up by flow committed in EARLIER rounds -- recomputed
            # fresh from the current `f` every round, never accumulated separately.
            committed_in = torch.zeros_like(headroom).index_add(-1, self._tgt, f)
            free_headroom = self._nonneg(headroom - committed_in)
            free_headroom_e = free_headroom.index_select(-1, self._tgt)
            if self.mode == "projection":
                # QP site 1 (no cross-edge coupling, see `_clip_projection`'s docstring):
                # `avail_e` is exposed as its OWN step, not folded into a single 3-arg call,
                # so QP site 2 below can reuse it as each edge's own upper bound in the real
                # coupled QP. Splitting it this way also FLOORS `c_arc - f` at 0 (via
                # `_clip_projection`'s own `clamp(..., 0, bound)`) before it can become a
                # bound: a fused `bound = min(c_arc - f, free_headroom_e)` would take the min
                # BEFORE flooring, so a slightly-negative `c_arc - f` from floating-point
                # overshoot could feed a NEGATIVE bound into the box projection.
                avail_e = self._clip_projection(self.c_arc - f, free_headroom_e)
                tentative = self._clip_projection(remaining, avail_e)
                # QP site 2: every edge targeting the SAME oversubscribed node shares one
                # scalar Lagrange multiplier solved via `solve_monotone`, so `d(share_i)/
                # d(r_j)` for a competing edge `j != i` is genuinely nonzero. Do NOT
                # "simplify" this back to a separable clip such as
                # `_clip_projection(tentative, proportional_share)` -- `_share_via_qp`'s
                # docstring derives why any such form has a provably zero cross-gradient.
                demand_at_tgt = torch.zeros_like(headroom).index_add(-1, self._tgt, tentative)
                node_oversubscribed = demand_at_tgt > free_headroom + 1e-12
                share = self._share_via_qp(remaining, avail_e, free_headroom, node_oversubscribed)
            else:
                avail = self._nonneg(self._clip(self.c_arc - f, free_headroom_e))
                tentative = self._clip(remaining, avail)
                # `active` gates which edges compete for `pref_sum` below; this is a hard
                # boolean threshold in EVERY mode, including smooth. Scope decision: the
                # smoothed kink sites are the `minimum`/`clamp(min=0.0)` pair and
                # the `over_subscribed_e` `where` -- this narrower gate is left hard in all
                # modes. Because `active` flips discretely the instant a competing edge's
                # (now smooth) `tentative` crosses zero, it discretely changes
                # `pref_sum_at_tgt` and therefore `proportional_share` in THAT pass, which
                # DOES introduce a real discontinuity in the OTHER competing edges' realised
                # `f` right at that crossing -- not merely a locally-zero gradient blip.
                # Measured (diamond fixture, mode="smooth"): the jump is bounded, scales
                # linearly with `tau` (same O(tau) order as the mode's other accepted
                # approximation error), and is largely -- but not completely -- cancelled by
                # the `n_passes` loop's self-correction over later rounds. Verified NaN/Inf
                # safe even where `proportional_share` blows up as `pref_sum_e` hits its
                # `1e-30` floor: the `_select` sigmoid weight on that branch is ~0 there, so
                # the extreme value never propagates into `f`.
                active = (tentative > 0).to(self.preference.dtype)
                demand_at_tgt = torch.zeros_like(headroom).index_add(-1, self._tgt, tentative)
                pref_sum_at_tgt = torch.zeros_like(headroom).index_add(
                    -1, self._tgt, self.preference * active
                )
                # `hard_cond` is hard mode's own oversubscription boolean (unused in smooth
                # mode, computed unconditionally so hard mode's arithmetic is untouched).
                hard_cond = (demand_at_tgt > free_headroom + 1e-12).index_select(-1, self._tgt)
                soft_margin = (demand_at_tgt - free_headroom).index_select(-1, self._tgt)
                pref_sum_e = pref_sum_at_tgt.index_select(-1, self._tgt)
                proportional_share = (
                    free_headroom_e * self.preference / torch.clamp(pref_sum_e, min=1e-30)
                )
                bounded_share = self._clip(proportional_share, tentative)
                share = self._select(hard_cond, soft_margin, bounded_share, tentative)
            share = self._nonneg(share)
            f = f + share
            remaining = remaining - share
        ds = -self.net.accumulate(f, self.kind)
        s_unclamped = s + dt * ds
        # Only the UPPER bound (s_max) is enforced here. `f` is already clipped against
        # each receiving node's headroom-as-a-RATE (`(s_max - s)/dt`, see `headroom`
        # above), and each pass's `free_headroom` subtracts what earlier passes already
        # committed, so `dt * sum_in(f) <= s_max - s` exactly in hard mode and the clamp
        # below is a genuine no-op there. It is NOT unconditionally a no-op, which is why
        # it stays and why `overflow` is reported rather than assumed zero: `mode="smooth"`
        # deliberately relaxes every one of those clips (softplus at zero carries a
        # `tau*ln(2)` bias, so `free_headroom` can exceed the true headroom by O(tau)), a
        # node whose free headroom is below `_OVERSUBSCRIBED_FLOOR` is treated as
        # not-oversubscribed in `mode="projection"` and can be overfilled by up to that
        # floor, and a state handed in already above its own `s_max` (nothing here forbids
        # it) overflows by exactly that excess. There is deliberately no lower bound of zero: a
        # request is not checked against the SENDER's available storage in hard-clip mode
        # so a source node may be drawn below zero -- that is a modelling
        # choice upstream of this layer (e.g. a closure sizing requests off available
        # storage), not something this layer silently papers over by floors here.
        s_new = self._clip(s_unclamped, self.s_max)
        if diagnostics is not None:
            diagnostics["overflow"] = self._nonneg(s_unclamped - self.s_max) / dt
        return s_new, f
