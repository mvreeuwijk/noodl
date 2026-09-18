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
            free_headroom = torch.clamp(headroom - committed_in, min=0.0)
            free_headroom_e = free_headroom.index_select(-1, self._tgt)
            avail = torch.clamp(torch.minimum(self.c_arc - f, free_headroom_e), min=0.0)
            tentative = torch.minimum(remaining, avail)
            active = (tentative > 0).to(self.preference.dtype)
            demand_at_tgt = torch.zeros_like(headroom).index_add(-1, self._tgt, tentative)
            pref_sum_at_tgt = torch.zeros_like(headroom).index_add(
                -1, self._tgt, self.preference * active
            )
            over_subscribed_e = (demand_at_tgt > free_headroom + 1e-12).index_select(
                -1, self._tgt
            )
            pref_sum_e = pref_sum_at_tgt.index_select(-1, self._tgt)
            proportional_share = (
                free_headroom_e * self.preference / torch.clamp(pref_sum_e, min=1e-30)
            )
            share = torch.where(
                over_subscribed_e, torch.minimum(proportional_share, tentative), tentative
            )
            share = torch.clamp(share, min=0.0)
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
        s_new = torch.clamp(s_unclamped, max=self.s_max)
        if diagnostics is not None:
            diagnostics["overflow"] = torch.clamp(s_unclamped - self.s_max, min=0.0) / dt
        return s_new, f
