"""Power law whose coefficient tracks the UPSTREAM node's density: CONTAM's own convention.

    F = (rho_up / rho_ref)^m * C * sign(dp) * |dp|^n,
    rho_up = rho[src]  where dp >= 0,   rho[tgt]  where dp < 0.

CONTAM (TN 1887r1 section 3.2) evaluates every power-law airflow element with the density of
the air ACTUALLY ENTERING the path, which changes with zone temperature at about 0.086 % per
kelvin and switches endpoint whenever the flow reverses. A reader that freezes the density at
a reference value instead is exact only for an isothermal building; on a 20 K stack it is
1.7 % wrong in mass flow (half the density error, since F ~ sqrt(rho) for the orifice family).

One exponent `m` covers all three CONTAM power-law families, because they differ only in the
power of density folded into the coefficient:

    m = 0    mass-flow family    plr_fcn, plr_test1/2, plr_conn, plr_stair, plr_shaft
    m = 1/2  sqrt(rho) family    plr_orfc, plr_leak1/2/3, plr_crack, and both doorway
                                 openings (C = C_d A sqrt(2 rho))
    m = 1    volumetric family   plr_qcn (a volumetric rating turned into mass flow)

`m` may be a scalar or one value per edge.

Why this lives in the ELEMENT and not in a `Closure`-written per-edge multiplier
--------------------------------------------------------------------------------
The milestone 2 spec sketches the correction as a `ReferenceCorrection` closure supplying a
multiplicative driver on `C`. That cannot work: the multiplier is FLOW-DIRECTION dependent,
and a closure runs BEFORE the solve, so it can only see the previous pass's flows -- which
under the default ping-pong coupling do not exist at all on the first (and, for `steady()`,
only) pass. A static per-edge factor taken from the nominal from-zone would carry the full
1.8 % error with the WRONG SIGN on every path that reverses. Selecting on `sign(dp)` inside
the element is the only place the information exists. Elements already receive the whole
`drivers` mapping (`PotentialFlowLayer` passes it to `flow`, `dflow` and `linear_init`, and
rebuilds it before `torch.func.functional_call` on the differentiable path), so the density
a `Stack` drive already reads is available here with no new capability; `Damper` is the
existing precedent for a power law switched on `sign(dp)`.

What this does NOT change
-------------------------
`(rho_up / rho_ref)^m` is PIECEWISE CONSTANT in `dp` -- it depends on `dp` only through
`sign(dp)`, and `torch.where`'s condition carries no gradient. So `dflow` remains the exact
elementwise derivative (the scale factor simply multiplies `PowerLaw`'s own), the Newton
Jacobian stays `A_I diag(dflow) A_I^T`, and because the scale is strictly positive the
operator stays symmetric positive definite exactly as before. The flow itself is continuous
at `dp = 0` despite the coefficient jumping there, because both branches vanish at the
origin (the laminar blend is `k dp`); only the SLOPE jumps, as it does for `Damper`.

`dp_transition` stays at REFERENCE conditions. It is the pressure at which CONTAM's
laminar/turbulent crossing sits, derived by the reader from `C` at `rho_ref`; recomputing it
per call from `rho_up` would make the transition itself flow-direction dependent and would
cost a division on every Newton iteration for a shift of order 0.1 % in a quantity that is
already a smoothing device of order 1e-3 Pa. This is the same documented approximation
`Duct` makes for its own fixed `rho`.

Dtype (Ruling R7): `m` is registered in the dtype of the `C`/`n` this element was built with,
and the density ratio is evaluated in the dtype of the `rho` driver, so a float64 application
never silently downgrades through this element.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.elements.powerlaw import PowerLaw

Tensor = torch.Tensor


class UpstreamDensityPowerLaw(PowerLaw):
    """`PowerLaw` with `C` scaled by `(rho_upstream / rho_ref) ** m`, upstream by `sign(dp)`.

    `src`/`tgt` are node POSITIONS (as `Network.endpoints(kind)` returns them, in the
    element's own edge order) held as non-differentiable long buffers; the density itself is
    read from `drivers[rho_key]`, a full-node `(..., n_nodes)` tensor -- the same driver the
    `Stack` drive reads, so a model that already carries buoyancy needs no extra driver.
    """

    def __init__(
        self,
        C,
        n,
        *,
        src,
        tgt,
        m,
        rho_ref: float = 1.2041,
        rho_key: str = "rho",
        dp_transition: float = 1e-3,
        regularised: float | None = None,
        kind: str = "airpath",
        learnable: bool = False,
    ) -> None:
        super().__init__(
            C,
            n,
            dp_transition=dp_transition,
            regularised=regularised,
            kind=kind,
            learnable=learnable,
        )
        src_t = torch.as_tensor(src).to(torch.long)
        tgt_t = torch.as_tensor(tgt).to(torch.long)
        if src_t.shape != tgt_t.shape:
            raise ValueError(
                f"UpstreamDensityPowerLaw (kind {kind!r}): src has shape "
                f"{tuple(src_t.shape)} but tgt has shape {tuple(tgt_t.shape)}; both must "
                f"carry one node index per edge of the kind"
            )
        if src_t.ndim != 1:
            raise ValueError(
                f"UpstreamDensityPowerLaw (kind {kind!r}): src/tgt must be 1-D (one node "
                f"index per edge), got shape {tuple(src_t.shape)}"
            )
        if src_t.numel() and int(torch.minimum(src_t.min(), tgt_t.min())) < 0:
            raise ValueError(
                f"UpstreamDensityPowerLaw (kind {kind!r}): src/tgt must be non-negative "
                f"node positions, got src={src_t.tolist()} tgt={tgt_t.tolist()}"
            )
        if float(rho_ref) <= 0.0:
            raise ValueError(
                f"UpstreamDensityPowerLaw (kind {kind!r}): rho_ref must be strictly "
                f"positive, got {rho_ref!r}"
            )
        m_t = torch.as_tensor(m, dtype=self._dtype())
        if m_t.ndim > 1 or (m_t.ndim == 1 and m_t.numel() not in (1, src_t.numel())):
            raise ValueError(
                f"UpstreamDensityPowerLaw (kind {kind!r}): m has shape {tuple(m_t.shape)}; "
                f"expected a scalar or one value per edge ({src_t.numel()})"
            )
        self.register_buffer("src", src_t)
        self.register_buffer("tgt", tgt_t)
        self.register_buffer("m", m_t)
        self.rho_ref = float(rho_ref)
        self.rho_key = str(rho_key)

    # ------------------------------------------------------------------ the density ratio
    def _rho(self, drivers: Mapping[str, Tensor] | None) -> Tensor:
        if drivers is None or self.rho_key not in drivers:
            raise KeyError(
                f"UpstreamDensityPowerLaw (kind {self.kind!r}): driver "
                f"{self.rho_key!r} not found; it needs the full-node density vector the "
                f"Stack drive also reads"
            )
        rho = torch.as_tensor(drivers[self.rho_key])
        n_nodes = rho.shape[-1] if rho.ndim else 0
        needed = int(torch.maximum(self.src.max(), self.tgt.max())) + 1 if self.src.numel() else 0
        if n_nodes < needed:
            raise ValueError(
                f"UpstreamDensityPowerLaw (kind {self.kind!r}): driver "
                f"{self.rho_key!r} has {n_nodes} node values but this element's endpoints "
                f"reach node index {needed - 1}"
            )
        self._check_positive(rho)
        return rho

    def _check_positive(self, rho: Tensor) -> None:
        """Raise on a non-positive density rather than let it reach `flow` as a silent `nan`.

        `(rho / rho_ref) ** m` is a real power of a physical density; a non-positive value
        (a caller's bug upstream, not a property of a converged solve) turns `m = 1/2` into
        `nan` with no exception anywhere. This is a CHECK, not a clamp -- it only inspects the
        driver values this element actually gathers (its own `src`/`tgt` node positions) and
        raises before the arithmetic runs; it does not touch the gradient path.
        """
        nodes = torch.cat([self.src, self.tgt]) if self.src.numel() else self.src
        if nodes.numel() == 0:
            return
        gathered = rho[..., nodes]
        bad = gathered <= 0
        if bool(bad.any()):
            bad_here = bad.reshape(-1, nodes.numel()).any(dim=0)
            offenders = sorted(set(nodes[bad_here].tolist()))
            raise ValueError(
                f"UpstreamDensityPowerLaw (kind {self.kind!r}): driver {self.rho_key!r} is "
                f"not strictly positive at node index/indices {offenders}"
            )

    def _ratio(self, rho: Tensor, nodes: Tensor) -> Tensor:
        return (rho[..., nodes] / self.rho_ref) ** self.m.to(rho.dtype)

    def _scale(self, dp: Tensor, drivers: Mapping[str, Tensor] | None) -> Tensor:
        """`(rho_up / rho_ref) ** m`, upstream chosen by `sign(dp)`.

        Both `torch.where` branches are ordinary positive densities gathered at real node
        positions -- neither is a singular expression the way `PowerLaw`'s sharp branch is at
        `dp = 0` -- so no `dp_safe`-style substitution is needed here and the unselected
        branch contributes an exact zero to backward rather than an `inf * 0`. The
        CONDITION carries no gradient, which is precisely why the scale is piecewise
        constant in `dp` and `dflow` below stays exact.
        """
        n_edges = self.src.numel()
        width = dp.shape[-1] if dp.ndim else 1
        # A single-edge element broadcasts against any dp shape (PowerLaw's own convention,
        # and always correct: there is one pair of endpoints to choose between). A
        # MULTI-edge element must be fed exactly its own width: a mismatch of two widths
        # both above 1 would raise from torch anyway, but a width of 1 would broadcast
        # SILENTLY and give every edge edge 0's endpoints, which is corruption, not a crash.
        if n_edges > 1 and width != n_edges:
            raise ValueError(
                f"UpstreamDensityPowerLaw (kind {self.kind!r}): dp has {width} columns but "
                f"this element covers {n_edges} edges"
            )
        rho = self._rho(drivers)
        return torch.where(dp >= 0, self._ratio(rho, self.src), self._ratio(rho, self.tgt))

    # ------------------------------------------------------------------ Element
    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        return self._scale(dp, drivers) * super().flow(dp, drivers)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        # Exact: the scale factor is constant on each side of dp = 0, so d/ddp of the
        # product is the scale times PowerLaw's own analytic derivative.
        return self._scale(dp, drivers) * super().dflow(dp, drivers)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        """Tangent-at-zero slope, averaged over the two directions (the `Damper` rule).

        `dp = 0` is exactly where the coefficient switches, so neither one-sided slope is
        more right than the other; the mean is direction-neutral and keeps the initial
        operator symmetric positive definite. This only seeds Newton's first iterate.

        The base class's own offset `c` is scaled by the same `mean` factor rather than
        discarded: `PowerLaw.linear_init` happens to return `c = 0` today, so the offset
        this element seeds Newton with is zero either way, but that is a fact about
        `PowerLaw`'s implementation, not a contract of the base class's return type. Scaling
        `c` here keeps this element correct even if a future `PowerLaw` (or a different base
        class) ever returned a nonzero offset.
        """
        c, k = super().linear_init(drivers)
        rho = self._rho(drivers)
        mean = 0.5 * (self._ratio(rho, self.src) + self._ratio(rho, self.tgt))
        return mean * c, mean * k
