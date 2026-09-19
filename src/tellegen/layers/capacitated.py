"""Explicit-in-time clip/allocation layer for capacitated networks (framework spec 4.2b).

Determines edge flows by clipping requests against arc capacity and receiver headroom
rather than solving for a potential -- the fourth way of determining flows the framework
identifies (alongside potential-flow Newton solves, driver-prescribed flows and
closure-computed flows). Conservation is exact by construction: every clipped unit of
flow is either realised on the edge or left unmet at the source; overflow at a full node
is a separate, reported quantity, never silently dropped.
"""
from __future__ import annotations

import torch

from tellegen.topology import Network

F64 = torch.float64
_MODES = ("hard", "smooth", "projection")
# Fixed iteration count for `_clip_projection`'s own inner box-projection solve --
# deliberately SEPARATE from `n_passes` (the outer sharing loop's round count, a different
# algorithm with a different convergence question). A separable box QP's projected-gradient
# step (unit step size) reaches its exact optimum in a single iteration (see
# `_clip_projection`'s docstring), so 3 is generous headroom, not a tuned tolerance.
_PROJECTION_ITERS = 3


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
        if preference is not None and preference.shape[-1] != n_edges:
            raise ValueError(
                f"CapacitatedTransferLayer {name!r}: preference has trailing size "
                f"{preference.shape[-1]}, expected {n_edges} (edge kind {kind!r})"
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
        """`min(a, b)` in hard mode (byte-identical to Tasks 1-2's bare
        `torch.minimum`); in smooth mode, the softmin counterpart at temperature
        `self.tau`: `-tau * logsumexp([-a/tau, -b/tau])`, which is smooth everywhere
        and -> `min(a, b)` as `tau -> 0`. Used at every UPPER-bound kink site: arc
        capacity, receiver headroom, request-vs-availability, and the final storage
        clamp against `s_max`.
        """
        if self.mode == "smooth":
            return -self.tau * torch.logsumexp(
                torch.stack([-a / self.tau, -b / self.tau]), dim=0
            )
        return torch.minimum(a, b)

    def _nonneg(self, x):
        """`max(x, 0)` in hard mode (byte-identical to Tasks 1-2's bare
        `torch.clamp(x, min=0.0)`); in smooth mode, the softplus counterpart at
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

        In hard mode this is `torch.where(hard_cond, a, b)`, with `hard_cond` the
        EXACT SAME boolean tensor Task 2 computed (`demand_at_tgt > free_headroom +
        1e-12`), so hard mode is byte-identical to the committed Task 2 arithmetic.

        In smooth mode, `hard_cond`/`soft_margin` is a boolean `torch.where` selecting
        between two WHOLE FORMULAS, not a `min`/`max`/`clamp` of two scalars, so it has
        no single obvious softmin/softplus counterpart. DESIGN DECISION (for Task 4's
        `mode="projection"`, which faces this identical boolean site): we smooth the
        SELECTION itself, via a sigmoid-weighted blend of the two whole branches at
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
        a boundary. Task 4 may reasonably choose the opposite for `"projection"` (leave
        its analogous select hard, smoothing only the QP/projection arithmetic) if that
        mode's purpose is judged to be about the constraint SURFACE rather than the
        ROUTING decision; either is defensible, but should be a deliberate, documented
        choice, as this one is.

        TASK 4's CHOICE for `"projection"`: the opposite of `"smooth"` -- `hard_cond`
        stays the exact SAME hard boolean as hard-clip mode (this method's default
        `torch.where` branch already applies, unmodified, since it only special-cases
        `"smooth"` above). The QP/projection machinery (`_clip_projection`) is applied
        instead to the ARITHMETIC inside the branches (the tentative-flow and
        proportional-share bounds computed in `step`), not to the branch choice itself.
        Rationale: `"projection"`'s stated purpose (design spec section 3) is a QP posed
        against the constraint SURFACE -- `0 <= f <= c_arc` and the headroom bound -- not
        a re-litigation of which edge WINS a competition for scarce headroom, which is
        exactly what `hard_cond` decides. Reusing hard-clip's own boolean there keeps
        `"projection"` byte-identical to hard-clip's ROUTING outcome (the only sense in
        which "solve for the flows exactly, then differentiate" is meaningful -- an
        approximate routing decision would need `"smooth"`'s own machinery, already
        available via `mode="smooth"`) while still exercising genuinely new machinery
        (`_clip_projection`'s own iterative box-projection solve, verified against
        `_clip`'s closed form in `test_clip_projection_matches_box_clamp`) at the actual
        QP site the spec names.
        """
        if self.mode == "smooth":
            w = torch.sigmoid(soft_margin / self.tau)
            return w * a + (1 - w) * b
        return torch.where(hard_cond, a, b)

    # Path taken for mode="projection": the UNROLLED box-projection fallback, not
    # `implicit_solve` reuse -- decided after reading `implicit.py`/`newton.py` in full (not
    # a blind skip). `implicit_solve` wants a smooth `residual(x, *params) = 0` and an
    # `operator` returning a Jacobian/LinearOperator at the current Newton iterate; this QP's
    # KKT stationarity condition has a genuine complementarity term (`f` pinned at 0, at its
    # upper bound, or strictly interior, with a DIFFERENT active Jacobian structure in each
    # regime), so presenting it as a Newton root-find would require smoothing the
    # complementarity condition first (e.g. Fischer-Burmeister) to get a residual Newton can
    # differentiate through -- at which point the "new" machinery is just a harder-to-verify
    # reimplementation of what `mode="smooth"` already does with `_clip`/`_nonneg`/`_select`,
    # for no accuracy benefit and the exact risk design section 10 flags. The box-constrained
    # least-squares QP this method actually solves (`min sum (f - r)**2 s.t. 0 <= f <=
    # min(c_arc, headroom)`) is SEPARABLE per edge, so its exact closed-form solution is
    # simply `clamp(r, 0, bound)` -- the loop below is a genuine (if trivially-converging)
    # projected-gradient solve of that QP, not a disguised call to `_clip`: unit step size on
    # a separable quadratic reaches the optimum after exactly one projection, so every
    # iteration after the first is a fixed-point check, not additional correction.
    def _clip_projection(
        self, r: torch.Tensor, headroom_per_edge: torch.Tensor, c_arc: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Box-constrained least-squares projection: minimises `sum((f - r)**2)` subject to
        `0 <= f <= bound`, where `bound = headroom_per_edge` alone (`c_arc=None`, used at the
        edge-sharing site where the bound is already a headroom-coupled quantity such as
        `proportional_share`) or `bound = min(c_arc, headroom_per_edge)` (used at the
        arc-capacity-and-headroom site, mirroring `avail`'s own `_clip`/`_nonneg` pair).

        Solved by projected gradient descent at unit step size (`f <- clamp(f - (f - r), 0,
        bound)`) for a small FIXED `_PROJECTION_ITERS` count, independent of `self.n_passes`
        (which counts OUTER sharing rounds, a different loop solving a different problem --
        see `step`'s own docstring). Ordinary autograd differentiates straight through the
        unrolled loop; no `implicit_solve` reuse (see the comment above this method for why).
        """
        bound = headroom_per_edge if c_arc is None else torch.minimum(c_arc, headroom_per_edge)
        zero = torch.zeros_like(r)
        f = torch.clamp(r, zero, bound)
        for _ in range(_PROJECTION_ITERS):
            grad = f - r
            f = torch.clamp(f - grad, zero, bound)
        return f

    def step(self, s, drivers, dt, *, diagnostics=None):
        """One explicit step. `s` is the previous per-node storage, `dt` in seconds.

        Reads `f"{self.name}.requests"` from `drivers` (per edge, m3/s). Returns
        `(s_new, f)`: `s_new` the new per-node storage, `f` the realised per-edge flow
        (m3/s). `diagnostics`, when given, is filled with `"overflow"` (m3/s, per node).

        Proportional sharing (design spec section 3): sharing only ever happens at a
        node that is the TARGET of more than one of this layer's edges -- "where more
        than one out-edge draws on one node's [headroom] supply". A node with a single
        in-edge is never touched by this loop, however many out-edges or requests are
        downstream of it: this layer never caps an edge to match a bottleneck further
        along the graph (that would need reasoning about paths, not incidence), exactly
        mirroring WSIMOD's own per-arc semantics (design spec amendment A1): a node's
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
        headroom = self.s_max - s
        remaining = r.clone()
        f = torch.zeros_like(r)
        for _ in range(self.n_passes):
            # Headroom already used up by flow committed in EARLIER rounds -- recomputed
            # fresh from the current `f` every round, never accumulated separately.
            committed_in = torch.zeros_like(headroom).index_add(-1, self._tgt, f)
            free_headroom = self._nonneg(headroom - committed_in)
            free_headroom_e = free_headroom.index_select(-1, self._tgt)
            if self.mode == "projection":
                # QP site 1: `tentative`'s bound is `min(c_arc - f, free_headroom_e)`,
                # exactly `avail`'s own definition -- `_clip_projection` solves that box
                # projection directly rather than computing `avail` as a separate step.
                tentative = self._clip_projection(remaining, free_headroom_e, self.c_arc - f)
            else:
                avail = self._nonneg(self._clip(self.c_arc - f, free_headroom_e))
                tentative = self._clip(remaining, avail)
            # `active` gates which edges compete for `pref_sum` below; this is a hard
            # boolean threshold in EVERY mode, including smooth. Scope decision: the
            # brief's enumerated kink sites are the `minimum`/`clamp(min=0.0)` pair and
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
            # `hard_cond` reproduces Task 2's exact boolean (kept unused in smooth mode,
            # computed unconditionally so hard mode's arithmetic is untouched).
            hard_cond = (demand_at_tgt > free_headroom + 1e-12).index_select(-1, self._tgt)
            soft_margin = (demand_at_tgt - free_headroom).index_select(-1, self._tgt)
            pref_sum_e = pref_sum_at_tgt.index_select(-1, self._tgt)
            proportional_share = (
                free_headroom_e * self.preference / torch.clamp(pref_sum_e, min=1e-30)
            )
            # QP site 2: the oversubscribed branch's value is `min(proportional_share,
            # tentative)` in every mode (`_select`'s docstring records why `"projection"`
            # leaves the BRANCH CHOICE itself hard, unlike `"smooth"`) -- `_clip_projection`
            # solves that box projection with `c_arc=None` (the sole bound is already the
            # headroom-coupled `proportional_share`, not a raw arc/headroom pair).
            bounded_share = (
                self._clip_projection(tentative, proportional_share)
                if self.mode == "projection"
                else self._clip(proportional_share, tentative)
            )
            share = self._select(hard_cond, soft_margin, bounded_share, tentative)
            share = self._nonneg(share)
            f = f + share
            remaining = remaining - share
        ds = -self.net.accumulate(f, self.kind)
        s_unclamped = s + dt * ds
        # Only the UPPER bound (s_max) is enforced here: `f` is already clipped against
        # each receiving edge's headroom, so no node can be pushed above its s_max by this
        # step (the clamp below is then a no-op, kept as a defensive backstop against
        # floating-point overshoot). There is deliberately no lower bound of zero: a
        # request is not checked against the SENDER's available storage in hard-clip mode
        # (spec 4.2b), so a source node may be drawn below zero -- that is a modelling
        # choice upstream of this layer (e.g. a closure sizing requests off available
        # storage), not something this layer silently papers over by floors here.
        s_new = self._clip(s_unclamped, self.s_max)
        if diagnostics is not None:
            diagnostics["overflow"] = self._nonneg(s_unclamped - self.s_max) / dt
        return s_new, f
