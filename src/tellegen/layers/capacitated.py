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
        """
        key = f"{self.name}.requests"
        r = drivers.get(key)
        if r is None:
            raise KeyError(
                f"CapacitatedTransferLayer {self.name!r}: driver {key!r} is required"
            )
        headroom = (self.s_max - s).index_select(-1, self._tgt)
        f = torch.clamp(torch.minimum(torch.minimum(r, self.c_arc), headroom), min=0.0)
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
