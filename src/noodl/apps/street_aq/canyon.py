"""Street-canyon boundary layer, in-canyon wind and roof exchange velocity.

Every formula here carries its source. The two families are SIRANE's (Soulhac, Perkins and
Salizzoni 2008; Soulhac et al. 2011) and MUNICH's (Kim et al.
2018 and 2022 plus the MUNICH/AtmoData sources), which the MUNICH comparison
(`tests/verification/test_munich.py`) pins.
Where the two papers disagree with each other or with the code, the CODE wins and the
disagreement is named in the docstring.

`torch.special.bessel_j0/j1/y0/y1` are NOT differentiable in torch 2.14 (their output has
no `grad_fn` at all), so all four are wrapped below in `torch.autograd.Function`s carrying
the standard derivative identities. Nothing outside those wrappers may call the raw
kernels: a gradient through one is lost silently, and `solve_monotone` -- which takes
`torch.autograd.grad` of its residual inside its own forward -- fails outright. That
applies to the wrappers' own `backward` methods too: each calls the WRAPPED `bessel_*`
below, so the derivative expression is itself differentiable and the second derivative is
real rather than a silent zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from noodl.apps.street_aq.closures import KAPPA_040, KAPPA_041, keyword_alias, normalise
from noodl.solvers.scalar import solve_monotone

Tensor = torch.Tensor

KAPPA = KAPPA_040
"""von Karman constant for the neutral form (`canyon_velocity`'s default, `kappa = 0.4`);
the same value as `closures.KAPPA_040`."""

KAPPA_MUNICH = KAPPA_041
"""Earlier name of `KAPPA_041` (0.41, the constant MUNICH uses,
`StreetNetworkTransport.cxx:21`)."""

EULER_GAMMA_TRUNCATED = 0.577
"""The Euler-Mascheroni constant TRUNCATED to three decimals -- not 0.5772156649 -- in the
Bessel shape equation (`bessel_shape_residual`). ATM `MeteorologyStreet.cxx:118`
hard-codes the same truncated value. Worth about 4e-5 relative."""

GAMMA_E = EULER_GAMMA_TRUNCATED
"""Earlier name of `EULER_GAMMA_TRUNCATED`."""

Z0_B_DEFAULT = 0.15
"""In-canyon (building) roughness, the default, m."""

Z0_S_DEFAULT = 0.01
"""Surface roughness `z0_surface`, MUNICH's default (`StreetNetworkTransport.cxx:155`)."""

EXCHANGE_SIGMA_W_RATIO = 1.0 / (math.sqrt(2.0) * math.pi)
"""`u_d / sigma_w` of the `turbulent_velocity` roof exchange (SIRANE's): 0.225079079039277.
S11 Eq. (5) p. 7386, K18 Eq. (3) p. 613, K22 Eq. (B10) p. 7387 and
`StreetNetworkTransport.cxx:3273` all read `sigma_w/(sqrt(2) pi)` -- the radical covers
only the 2, verified at glyph level in all three PDFs. The reading
`sigma_w/sqrt(2 pi)` (`1/sqrt(2 pi) = 0.398942280401433`) appears in no source."""

SIRANE_EXCHANGE = EXCHANGE_SIGMA_W_RATIO
"""Earlier name of `EXCHANGE_SIGMA_W_RATIO`."""

ASPECT_RATIO_EXCHANGE_BETA = 2.0 / (math.sqrt(2.0) * math.pi)
"""`beta` in the `aspect_ratio_scaled` roof exchange `u_d = beta sigma_w / (1 + H/W)`
(Schulte's mixing-length form): 0.450158158078553, fixed by matching the
`turbulent_velocity` form at `a_r = 1` (K18 p. 613). ATM `ComputeSchulteLm`,
`MeteorologyStreet.cxx:547-554`; MUNICH v1.0 used the literal 0.45 instead, a 0.035 %
difference."""

SCHULTE_BETA = ASPECT_RATIO_EXCHANGE_BETA
"""Earlier name of `ASPECT_RATIO_EXCHANGE_BETA`."""

C_BRACKET_LO = 1e-4
C_BRACKET_HI = 3.0
C_RATIO_MAX = 1.6
_N_ROOF_LEVELS = 100


class _BesselJ0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.special.bessel_j0(x)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return -grad * bessel_j1(x)


class _BesselY0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.special.bessel_y0(x)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return -grad * bessel_y1(x)


class _BesselJ1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.special.bessel_j1(x)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * (bessel_j0(x) - bessel_j1(x) / x)


class _BesselY1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.special.bessel_y1(x)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * (bessel_y0(x) - bessel_y1(x) / x)


def bessel_j0(x: Tensor) -> Tensor:
    """J0, differentiable (`J0' = -J1`)."""
    return _BesselJ0.apply(x)


def bessel_j1(x: Tensor) -> Tensor:
    """J1, differentiable (`J1' = J0 - J1/x`)."""
    return _BesselJ1.apply(x)


def bessel_y0(x: Tensor) -> Tensor:
    """Y0, differentiable (`Y0' = -Y1`)."""
    return _BesselY0.apply(x)


def bessel_y1(x: Tensor) -> Tensor:
    """Y1, differentiable (`Y1' = Y0 - Y1/x`)."""
    return _BesselY1.apply(x)


def bessel_shape_residual(c: Tensor, ratio: Tensor) -> Tensor:
    """`0.5 (z0/di) c - exp((pi/2) Y1(c)/J1(c) - gamma_E)`; zero at the shape parameter.

    The Soulhac-Perkins-Salizzoni closed form writes exactly this; MUNICH writes the same
    equation as `z0_s/delta_i = (2/C) exp(...)` (K22 Eq. B12; ATM `MeteorologyStreet.cxx:119-153`,
    where it is solved by a brute-force search on a 0.01 grid, so MUNICH's `C` is quantised
    -- worth 4e-4 relative in `u_M`. This solve is continuous.)
    """
    return 0.5 * ratio * c - torch.exp(
        (math.pi / 2.0) * bessel_y1(c) / bessel_j1(c) - EULER_GAMMA_TRUNCATED
    )


soulhac_residual = bessel_shape_residual
"""Earlier name of `bessel_shape_residual`."""


def bessel_shape_parameter(ratio: Tensor) -> Tensor:
    """The shape parameter `c` of the Bessel canyon profile: the root of
    `bessel_shape_residual`, batched and differentiable in `ratio`.

    Bracket `[1e-4, 3.0]`, justified as follows: the residual is `+0.5 r c > 0`
    at the low end (the exponential underflows to exactly zero there) and
    `1.5 r - 2.527 < 0` at the high end for every `r < 1.685`, and it changes sign exactly
    once in between. A ratio at or above 1.6 is refused rather than handed to the solver,
    whose "no sign change is bracketed" message would name neither the ratio nor the reason.
    """
    ratio = torch.as_tensor(ratio, dtype=torch.float64)
    if bool((ratio <= 0).any()):
        bad = torch.nonzero(ratio.reshape(-1) <= 0).flatten().tolist()
        raise ValueError(
            f"bessel_shape_parameter: the roughness ratio z0_b/di must be strictly "
            f"positive; "
            f"non-positive at flat indices {bad}"
        )
    if bool((ratio >= C_RATIO_MAX).any()):
        bad = torch.nonzero(ratio.reshape(-1) >= C_RATIO_MAX).flatten().tolist()
        worst = float(ratio.reshape(-1)[bad].max())
        raise ValueError(
            f"bessel_shape_parameter: the roughness ratio z0_b/di must be below "
            f"{C_RATIO_MAX} for "
            f"the shape equation to have a root in [{C_BRACKET_LO}, {C_BRACKET_HI}]; got "
            f"up to {worst} at flat index/indices {bad}. A roughness comparable to the "
            f"canyon half-width is not a canyon -- check z0_b and the street width."
        )
    lo = torch.full_like(ratio, C_BRACKET_LO)
    hi = torch.full_like(ratio, C_BRACKET_HI)
    return solve_monotone(bessel_shape_residual, lo, hi, ratio, tol=1e-14, max_iter=200)


soulhac_shape = bessel_shape_parameter
"""Earlier name of `bessel_shape_parameter`."""

SHAPE_CONSTANT_GRID_STEP = 0.01
"""Spacing of the grid `grid_shape_parameter` searches, `c = 0.01, 0.02, ..., 1.00`
(ATM `ComputeSiraneC`, `MeteorologyStreet.cxx:114-152`: `maxC = 1`, `nc = 100`)."""

SHAPE_CONSTANT_GRID_POINTS = 100
"""Number of points on that grid."""

SHAPE_CONSTANT_GRID_TOLERANCE = 1e-3
"""The largest `|2/c exp((pi/2) Y1(c)/J1(c) - gamma_E) - z0/di|` the grid search accepts:
MUNICH stops with "Fail to find a solution" above it (`MeteorologyStreet.cxx:146-149`)."""


def _shape_constant_grid() -> Tensor:
    """The grid, accumulated exactly as MUNICH accumulates it (`tempC += step`), so every
    point carries MUNICH's own rounding rather than `k/100`'s."""
    values, c = [], 0.0
    for _ in range(SHAPE_CONSTANT_GRID_POINTS):
        c += SHAPE_CONSTANT_GRID_STEP
        values.append(c)
    return torch.tensor(values, dtype=torch.float64)


def grid_shape_parameter(ratio: Tensor) -> Tensor:
    """The shape parameter `c` the way MUNICH evaluates it: the point of the grid
    `c = 0.01, 0.02, ..., 1.00` that minimises `|2/c exp((pi/2) Y1(c)/J1(c) - gamma_E) - z0/di|`
    (ATM `ComputeSiraneC`, `MeteorologyStreet.cxx:114-152`; the first minimum wins on a tie,
    as MUNICH's strict `>` makes it).

    The forward value is that grid point, so `c` is quantised to 0.01. The canyon-mean
    factor `f_mean` is sensitive to `c`, so against the exact root (`bessel_shape_parameter`)
    the roof wind `u_H` moves by up to about 2 % (median 0.8 % for `z0_s = 0.01` and
    `delta_i` from 2 to 40 m), though `u_M` alone moves by only about 4e-4. The
    argmin is piecewise constant in `ratio`, so its own derivative is zero almost everywhere;
    the gradient carried here is instead the EXACT root's (a straight-through estimator:
    `c_exact + (c_grid - c_exact).detach()`). That keeps the roof wind differentiable in the
    roughness and the geometry with the slope of the continuous model, which is what an
    optimiser or a sensitivity needs, while the forward value is MUNICH's. A ratio whose
    best grid residual exceeds `SHAPE_CONSTANT_GRID_TOLERANCE` is refused, as MUNICH refuses
    it.
    """
    ratio = torch.as_tensor(ratio, dtype=torch.float64)
    exact = bessel_shape_parameter(ratio)
    with torch.no_grad():
        grid = _shape_constant_grid()
        curve = 2.0 / grid * torch.exp(
            (math.pi / 2.0) * torch.special.bessel_y1(grid) / torch.special.bessel_j1(grid)
            - EULER_GAMMA_TRUNCATED
        )
        misfit = (curve - ratio.detach().unsqueeze(-1)).abs()
        best, index = misfit.min(dim=-1)
        if bool((best > SHAPE_CONSTANT_GRID_TOLERANCE).any()):
            bad = torch.nonzero(
                (best > SHAPE_CONSTANT_GRID_TOLERANCE).reshape(-1)).flatten().tolist()
            raise ValueError(
                f"grid_shape_parameter: no point of the grid c = 0.01..1.00 solves the shape "
                f"equation to within {SHAPE_CONSTANT_GRID_TOLERANCE} at flat index/indices "
                f"{bad} (ratios {[float(v) for v in ratio.reshape(-1)[bad]]}); use "
                f"shape_constant='exact_root'"
            )
        on_grid = grid[index]
    return exact + (on_grid - exact).detach()


def shape_parameter(ratio: Tensor, shape_constant: str = "exact_root") -> Tensor:
    """The Bessel shape parameter `c`, by the `shape_constant` closure option:
    `"exact_root"` is `bessel_shape_parameter` (the continuous root), `"grid_search"` is
    `grid_shape_parameter` (MUNICH's argmin on a 0.01 grid)."""
    method = normalise("shape_constant", shape_constant, "shape_parameter")
    if method == "grid_search":
        return grid_shape_parameter(ratio)
    return bessel_shape_parameter(ratio)


def _guarded_sqrt(argument: Tensor) -> Tensor:
    """`sqrt(argument)`, with the exactly-zero point kept off the autograd graph.

    Every term of MUNICH's unstable `sigma_w`/`sigma_v` argument is proportional to `u*`,
    so the argument is EXACTLY zero at a calm step (`u_star == 0`), where `sqrt` has
    infinite slope and hands back a NaN gradient. The safe input goes under the square root
    on BOTH branches; the degenerate branch returns a hard zero, which is the forward value
    there. The argument itself is formed by the caller, unchanged, so no forward value
    moves by so much as an ulp.
    """
    positive = argument > 0
    return torch.where(
        positive,
        torch.sqrt(torch.where(positive, argument, torch.ones_like(argument))),
        torch.zeros_like(argument),
    )


def _bessel_roof_factor(c: Tensor) -> Tensor:
    """`Y0(c) - J0(c) Y1(c)/J1(c)`, the bracket of `u_M` in K22 Eq. (B12)."""
    return bessel_y0(c) - bessel_j0(c) * bessel_y1(c) / bessel_j1(c)


@dataclass(frozen=True)
class BoundaryLayer:
    """Friction velocity and the turbulence it sets, batched over forcing instances.

    `u_star`, `h_abl`, `z_ref`, `d` and `z0` broadcast against one another and against the
    street axis a caller adds. `kappa` is the constant the `u_star` in this object was
    derived with, kept so that `sigma_v` and MUNICH's `w*` use the same one (0.4 for the
    neutral form, 0.41 for MUNICH -- 2.5 % on `u_star`).
    """

    u_star: Tensor
    h_abl: Tensor
    z_ref: Tensor
    d: Tensor
    z0: Tensor
    kappa: float = KAPPA

    def sigma_w(self, z: Tensor, *, lmo: Tensor | None = None,
                stability: str = "neutral") -> Tensor:
        """Vertical velocity standard deviation at height `z`.

        `stability="neutral"` is `1.3 u* (1 - 0.8 z / h_abl)`, with NO guard: it goes
        NEGATIVE for `z > 1.25 h_abl`, which a tall street under a shallow boundary layer
        can reach. Use `boundary_layer(pblh_floor=...)` to apply
        MUNICH's `pblh := max(H, PBLH)` guard, or `exchange_velocity` will refuse the
        result.

        `stability="monin_obukhov"` is MUNICH's `ComputeSigmaW`,
        `StreetNetworkTransport.cxx:3221-3260`, in three branches on the Monin-Obukhov
        length. Its neutral branch is the `stability="neutral"` formula above.
        """
        stability = normalise("stability", stability, "BoundaryLayer.sigma_w")
        z = torch.as_tensor(z, dtype=self.u_star.dtype)
        if stability == "neutral":
            return 1.3 * self.u_star * (1.0 - 0.8 * z / self.h_abl)
        if lmo is None:
            raise ValueError(
                "BoundaryLayer.sigma_w: stability='monin_obukhov' needs the Monin-Obukhov "
                "length `lmo` (m); it selects the unstable/stable/neutral branch"
            )
        lmo = torch.as_tensor(lmo, dtype=self.u_star.dtype)
        pblh = torch.maximum(self.h_abl, z)
        ratio = z / pblh
        neutral = 1.3 * self.u_star * (1.0 - 0.8 * ratio)
        stable = 1.3 * self.u_star * (1.0 - 0.5 * ratio) ** 0.75
        # |lmo| is replaced by 1.0 wherever the unstable branch is not selected, so the
        # cube root never sees a zero and the unselected branch carries no NaN into the
        # gradient (the PowerLaw idiom of the framework's differentiability contract).
        is_unstable = lmo < 0
        safe_lmo = torch.where(is_unstable, lmo.abs(), torch.ones_like(lmo))
        w_star = self.u_star * (pblh / (self.kappa * safe_lmo)) ** (1.0 / 3.0)
        sigma_wc = (
            math.sqrt(0.4) * w_star * 2.1 * ratio ** (1.0 / 3.0) * (1.0 - 0.8 * ratio)
        )
        unstable = _guarded_sqrt(sigma_wc * sigma_wc + neutral * neutral)
        return torch.where(is_unstable, unstable, torch.where(lmo < pblh, stable, neutral))

    def sigma_v(self, *, lmo: Tensor | None = None, stability: str = "neutral") -> Tensor:
        """Horizontal velocity standard deviation, MUNICH's `ComputeSigmaV`
        (`StreetNetworkTransport.cxx:3194-3216`): a 10-level average over `z in [0, PBLH]`.

        `stability="neutral"` selects the neutral branch, which collapses to exactly
        `1.2 u*` (the mean of `1 - 0.8 z/PBLH` over 10 equally spaced points on `[0, PBLH]`
        is 0.6). There is no separate neutral-only `sigma_v` form; this is the value the
        turbulence-intensity direction spread (`routing`) needs, and it is stated here
        rather than invented there. `stability="monin_obukhov"` averages the unstable,
        stable or neutral branch the Monin-Obukhov length selects.
        """
        stability = normalise("stability", stability, "BoundaryLayer.sigma_v")
        levels = torch.arange(10, dtype=self.u_star.dtype) / 9.0
        z_over_pblh = levels.reshape(*([1] * self.u_star.dim()), 10)
        u_star = self.u_star.unsqueeze(-1)
        neutral = 2.0 * u_star * (1.0 - 0.8 * z_over_pblh)
        if stability == "neutral":
            return neutral.mean(-1)
        if lmo is None:
            raise ValueError(
                "BoundaryLayer.sigma_v: stability='monin_obukhov' needs the Monin-Obukhov "
                "length `lmo` (m); it selects the unstable/stable/neutral branch"
            )
        lmo = torch.as_tensor(lmo, dtype=self.u_star.dtype).unsqueeze(-1)
        pblh = self.h_abl.unsqueeze(-1)
        stable = 2.0 * u_star * (1.0 - 0.5 * z_over_pblh) ** 0.75
        is_unstable = lmo < 0
        safe_lmo = torch.where(is_unstable, lmo.abs(), torch.ones_like(lmo))
        w_star = u_star * (pblh / (self.kappa * safe_lmo)) ** (1.0 / 3.0)
        unstable = _guarded_sqrt(0.3 * w_star * w_star + neutral * neutral)
        per_level = torch.where(
            is_unstable, unstable, torch.where(lmo < pblh, stable, neutral)
        )
        return per_level.mean(-1)


def boundary_layer(
    h_mean: Tensor,
    u_ref: Tensor,
    h_abl: Tensor,
    *,
    z_ref: Tensor | float = 30.0,
    kappa: float = KAPPA,
    pblh_floor: Tensor | float | None = None,
) -> BoundaryLayer:
    """Friction velocity and the boundary-layer height from the network-mean geometry and
    reference wind, batched and differentiable.

    `d = 2 h_mean / 3`, `z0 = h_mean / 10`, `u* = kappa U_ref / ln((z_ref - d) / z0)`, with
    `h_mean` the NETWORK-mean building height. `pblh_floor`, when given, raises `h_abl` to
    at least that value -- MUNICH's `pblh := max(H, PBLH)` guard
    (`StreetNetworkTransport.cxx:3236`); pass the network's greatest street height, which
    keeps `1 - 0.8 z/h_abl >= 0.2` for every street.

    Raises when `(z_ref - d) / z0 <= 1`, i.e. when the reference height sits inside the
    canopy: the logarithm is then zero or negative and `u*` would come out zero, negative
    or infinite. That is a real configuration -- `wind_height_m = 10` with a mean building
    height of 15 m gives exactly it -- so it is named, not clamped.
    """
    h_mean = torch.as_tensor(h_mean, dtype=torch.float64)
    u_ref = torch.as_tensor(u_ref, dtype=torch.float64)
    h_abl = torch.as_tensor(h_abl, dtype=torch.float64)
    z_ref = torch.as_tensor(z_ref, dtype=torch.float64)
    d = 2.0 * h_mean / 3.0
    z0 = h_mean / 10.0
    argument = (z_ref - d) / z0
    if bool((argument <= 1.0).any()):
        raise ValueError(
            f"boundary_layer: z_ref must clear the displacement height -- "
            f"(z_ref - d)/z0 = {float(argument.min())} <= 1 with z_ref={float(z_ref.min())} "
            f"m, mean building height {float(h_mean.max())} m, displacement "
            f"{float(d.max())} m, roughness {float(z0.max())} m; the log law has no "
            f"friction velocity there"
        )
    if pblh_floor is not None:
        h_abl = torch.maximum(h_abl, torch.as_tensor(pblh_floor, dtype=torch.float64))
    return BoundaryLayer(
        u_star=kappa * u_ref / torch.log(argument), h_abl=h_abl, z_ref=z_ref, d=d, z0=z0,
        kappa=float(kappa),
    )


def canopy_displacement_roughness(
    h_mean: Tensor,
    w_mean: Tensor,
    *,
    lambda_p: float = 0.4,
    big_delta: float = 4.43,
    small_delta: float = 1.0,
    c_db: float = 1.2,
    kappa: float = KAPPA_041,
) -> tuple[Tensor, Tensor]:
    """Macdonald (1998) displacement height and roughness length, `(d_c, z0c)`, in metres.

    K22 Eqs. (1)-(3), p. 7373; SRC `ComputeMacdonaldProfile`,
    `StreetNetworkTransport.cxx:3302-3353`; ATM `MeteorologyStreet.cxx:47-105`.
    `Delta = 4.43` and `delta = 1.0` are the STAGGERED-array values, which are the ones
    MUNICH v2.0 uses (the square-array 3.59 / 0.55 are not). `lambda_p` is the config
    input `Building_density`, default 0.4, from which the building width is derived as
    `W_b = lambda_p/(1 - lambda_p) W_mean` and `lambda_f = H_mean / (W_mean + W_b)`.
    `h_mean` and `w_mean` are NETWORK means, not per-street values.
    """
    h_mean = torch.as_tensor(h_mean, dtype=torch.float64)
    w_mean = torch.as_tensor(w_mean, dtype=torch.float64)
    w_building = lambda_p / (1.0 - lambda_p) * w_mean
    lambda_f = h_mean / (w_mean + w_building)
    d_over_h = 1.0 + big_delta ** (-lambda_p) * (lambda_p - 1.0)
    d_c = d_over_h * h_mean
    z0c = h_mean * (1.0 - d_over_h) * torch.exp(
        -(0.5 * small_delta * c_db / kappa**2 * (1.0 - d_over_h) * lambda_f) ** (-0.5)
    )
    return d_c, z0c


macdonald_profile = canopy_displacement_roughness
"""Earlier name of `canopy_displacement_roughness`."""


def roof_wind(
    u_star: Tensor,
    H: Tensor,
    W: Tensor,
    *,
    roof_wind: str | None = None,
    z0_s: Tensor | float = Z0_S_DEFAULT,
    kappa: float = KAPPA_041,
    h_mean: Tensor | None = None,
    w_mean: Tensor | None = None,
    n_levels: int = _N_ROOF_LEVELS,
    shape_constant: str = "exact_root",
    form: str | None = None,
) -> Tensor:
    """Wind speed at roof level `u_H`, from the friction velocity.

    `shape_constant` is the closure option of that name: how the Bessel shape parameter
    `C` is evaluated (`shape_parameter`), the exact root or MUNICH's 0.01-grid search.

    `roof_wind` is the closure option of that name (`closures.OPTIONS`), default
    `"bessel_canyon_mean"`; `form=` is its earlier, deprecated keyword.

    `roof_wind="bessel_canyon_mean"` is K22 Eq. (B12), p. 7387
    (ATM `MeteorologyStreet.cxx:114-201`):
    `delta_i = min(H, W/2)`, `C` from `bessel_shape_parameter(z0_s/delta_i)`,
    `u_M = u* sqrt(pi/(sqrt(2) kappa^2 C) [Y0(C) - J0(C) Y1(C)/J1(C)])` and
    `u_H = u_M f_mean` with `f_mean` the 100-level mean of
    `[J1(C) Y0(C y) - J0(C y) Y1(C)] / [J1(C) Y0(C) - J0(C) Y1(C)]` over `y = k/N`.
    (ATM compares the DIMENSIONLESS `y` against the DIMENSIONAL `z0_s` at its line 192 --
    a latent unit bug that never bites at the default `z0_s = 0.01`, so all 100 levels are
    taken here, exactly as MUNICH does at its default.)

    `roof_wind="canopy_log_law"` is K22 Eq. (B13): `u_H = (u*/kappa) ln((H - d_c)/z0c)` with
    `d_c` and `z0c` from `canopy_displacement_roughness(h_mean, w_mean)`. MUNICH returns 0 when
    `H < d_c + z0c` (SRC `:3334`); that guard is reproduced here.
    """
    form = normalise("roof_wind", keyword_alias(
        "roof_wind", "roof_wind", roof_wind, "form", form, "bessel_canyon_mean"),
        "roof_wind")
    u_star = torch.as_tensor(u_star, dtype=torch.float64)
    H = torch.as_tensor(H, dtype=torch.float64)
    W = torch.as_tensor(W, dtype=torch.float64)
    if form == "canopy_log_law":
        if h_mean is None or w_mean is None:
            raise ValueError(
                "roof_wind: roof_wind='canopy_log_law' needs the network means h_mean and w_mean "
                "(K22 Eq. 3 is written on network means, not per-street values)"
            )
        d_c, z0c = canopy_displacement_roughness(h_mean, w_mean, kappa=kappa)
        above = H > d_c + z0c
        safe = torch.where(above, (H - d_c) / z0c, torch.ones_like(H * d_c))
        return torch.where(above, (u_star / kappa) * torch.log(safe),
                           torch.zeros_like(safe))
    z0_s = torch.as_tensor(z0_s, dtype=torch.float64)
    delta_i = torch.minimum(H, W / 2.0)
    c = shape_parameter(z0_s / delta_i, shape_constant)
    u_m = u_star * torch.sqrt(
        math.pi / (math.sqrt(2.0) * kappa**2 * c) * _bessel_roof_factor(c)
    )
    y = (torch.arange(1, n_levels + 1, dtype=torch.float64) / n_levels)
    cy = c.unsqueeze(-1) * y
    c_e = c.unsqueeze(-1)
    numerator = bessel_j1(c_e) * bessel_y0(cy) - bessel_j0(cy) * bessel_y1(c_e)
    denominator = bessel_j1(c) * bessel_y0(c) - bessel_j0(c) * bessel_y1(c)
    f_mean = (numerator / denominator.unsqueeze(-1)).mean(-1)
    return u_m * f_mean


def canyon_velocity(
    W: Tensor,
    H: Tensor,
    phi: Tensor,
    *,
    u_star: Tensor | None = None,
    u_h: Tensor | None = None,
    canyon_wind: str | None = None,
    z0_b: Tensor | float = Z0_B_DEFAULT,
    z0_s: Tensor | float = Z0_S_DEFAULT,
    kappa: float = KAPPA,
    canyon_wind_min: float = 0.0,
    shape_constant: str = "exact_root",
    form: str | None = None,
) -> Tensor:
    """The SIGNED along-canyon velocity, m/s. Positive means from `u` to `v`.

    `shape_constant` (closure option) chooses how the Bessel profile's `c` is evaluated:
    `"exact_root"` (default) or `"grid_search"` (MUNICH's 0.01 grid); see `shape_parameter`.

    `phi` is the angle between the wind and the street axis; it broadcasts against `W`,
    `H` and any leading forcing batch.

    `canyon_wind` is the closure option of that name (`closures.OPTIONS`), default
    `"bessel_profile"`; `form=` is its earlier, deprecated keyword.

    `canyon_wind="bessel_profile"` (needs `u_star`) is the Soulhac-Perkins-Salizzoni (2008)
    closed form,
    exactly as K22 Eq. (B15) p. 7388 writes it:
    `di = min(W/2, H)`, `alpha = ln(di/z0_b)`, `beta = exp(c/sqrt(2) (1 - H/di))`,
    `u_h = u* sqrt(pi/(sqrt(2) kappa^2 c) [Y0(c) - J0(c) Y1(c)/J1(c)])`, and

        u = u_h cos(phi) di^2/(W H)
            [ 2 sqrt2/c (1 - beta)(1 - c^2/3 + c^4/45)
              + beta (2 alpha - 3)/alpha + (W/di - 2)(alpha - 1)/alpha ]

    The Bessel profile REFUSES `z0_b >= di`, which `bessel_shape_parameter`'s own
    `z0_b/di < 1.6`
    bound lets through: `alpha` is zero at ratio 1 and negative above it, and the answer
    comes back NaN rather than wrong-looking.

    `canyon_wind="exponential_profile"` (needs `u_h`) is K22 Eq. (B14), p. 7388 -- SINGLE
    regime, `2/a_r`
    prefactor, integrated from the street roughness `z0_s`:

        u = u_h cos(phi) (2/a_r) [1 - exp((a_r/2)(z0_s/H - 1))],   a_r = H/W

    It is NOT K18 Eqs. (9)-(11) (three aspect-ratio regimes, `2/pi` prefactor, integrated
    from 0), which is MUNICH v1.0 and differs by up to 37 % for narrow canyons. ATM
    `ComputeExpUstreet`, `MeteorologyStreet.cxx:257-263` computes B14 and nothing else.

    `canyon_wind_min` is MUNICH's `ustreet_min`, default 0.1 m/s there (SRC `:3451`,
    `Minimum_Street_Wind_Speed`) and 0.0 here by default. The
    floor keeps the sign: `U = sign * max(|u|, u_min)` with `sign = +1` wherever the
    unfloored value is `>= 0`. The `>=` matters exactly once -- at `phi = pi/2`, where the
    unfloored value is `+0` and MUNICH's own `>` classification makes the street an
    OUTFLOW; that is what produces the 270-degree panel of K22 Fig. 1.
    """
    form = normalise("canyon_wind", keyword_alias(
        "canyon_velocity", "canyon_wind", canyon_wind, "form", form, "bessel_profile"),
        "canyon_velocity")
    W = torch.as_tensor(W, dtype=torch.float64)
    H = torch.as_tensor(H, dtype=torch.float64)
    phi = torch.as_tensor(phi, dtype=torch.float64)
    if form == "bessel_profile":
        if u_star is None:
            raise ValueError(
                "canyon_velocity: canyon_wind='bessel_profile' needs u_star (the Bessel profile is "
                "written on the friction velocity, not on the roof wind)"
            )
        u_star = torch.as_tensor(u_star, dtype=torch.float64)
        z0_b = torch.as_tensor(z0_b, dtype=torch.float64)
        di = torch.minimum(W / 2.0, H)
        # `bessel_shape_parameter` refuses z0_b/di >= 1.6, which is NOT enough here: `alpha` is
        # ln(di/z0_b), exactly 0 at ratio 1 and negative on (1, 1.6), so the shape function
        # divides by zero or by a negative there and the answer comes back NaN with no
        # error anywhere. Refused by name rather than clamped.
        wide, rough, half = torch.broadcast_tensors(W, z0_b, di)
        if bool((rough >= half).any()):
            bad = torch.nonzero((rough >= half).reshape(-1)).flatten().tolist()
            raise ValueError(
                f"canyon_velocity: canyon_wind='bessel_profile' needs the in-canyon roughness z0_b "
                f"strictly below di = min(W/2, H), because alpha = ln(di/z0_b) divides "
                f"the shape function; at flat index/indices {bad} the widths are "
                f"{[float(v) for v in wide.reshape(-1)[bad]]} m and the roughnesses are "
                f"{[float(v) for v in rough.reshape(-1)[bad]]} m"
            )
        c = shape_parameter(z0_b / di, shape_constant)
        alpha = torch.log(di / z0_b)
        beta = torch.exp(c / math.sqrt(2.0) * (1.0 - H / di))
        u_roof = u_star * torch.sqrt(
            math.pi / (math.sqrt(2.0) * kappa**2 * c) * _bessel_roof_factor(c)
        )
        shape = (
            2.0 * math.sqrt(2.0) / c * (1.0 - beta) * (1.0 - c**2 / 3.0 + c**4 / 45.0)
            + beta * (2.0 * alpha - 3.0) / alpha
            + (W / di - 2.0) * (alpha - 1.0) / alpha
        )
        signed = u_roof * torch.cos(phi) * di**2 / W / H * shape
    else:
        if u_h is None:
            raise ValueError(
                "canyon_velocity: canyon_wind='exponential_profile' needs u_h (K22 Eq. B14 "
                "is written on "
                "the roof-level wind; get it from roof_wind())"
            )
        u_h = torch.as_tensor(u_h, dtype=torch.float64)
        z0_s = torch.as_tensor(z0_s, dtype=torch.float64)
        a_r = H / W
        half = 0.5 * a_r
        signed = u_h * torch.cos(phi) * (1.0 / half) * (
            1.0 - torch.exp(half * (z0_s / H - 1.0))
        )
    if canyon_wind_min <= 0.0:
        return signed
    sign = torch.where(signed >= 0, torch.ones_like(signed), -torch.ones_like(signed))
    return sign * torch.clamp(signed.abs(), min=float(canyon_wind_min))


def exchange_velocity(
    sigma_w: Tensor,
    H: Tensor,
    W: Tensor,
    *,
    roof_exchange: str | None = None,
    u_d_min: float = 0.0,
    sigma_w_min: float = 0.0,
    form: str | None = None,
) -> Tensor:
    """The roof exchange velocity `u_d`, m/s.

    `roof_exchange` is the closure option of that name (`closures.OPTIONS`), default
    `"turbulent_velocity"`; `form=` is its earlier, deprecated keyword.

    `roof_exchange="turbulent_velocity"`: `u_d = sigma_w / (sqrt(2) pi)`, independent of
    the aspect ratio.
    S11 Eq. (5), K18 Eq. (3), K22 Eq. (B10), `StreetNetworkTransport.cxx:3273`. See
    `EXCHANGE_SIGMA_W_RATIO` for why the reading `sigma_w / sqrt(2 pi)` is not used.

    `roof_exchange="aspect_ratio_scaled"`: `u_d = beta sigma_w / (1 + H/W)` with
    `beta = 2/(sqrt(2) pi)`
    (K18 Eqs. 4-8, K22 Eq. B11, ATM `ComputeSchulteLm`), MUNICH v2's default. The two
    agree exactly at `H = W`, which a test pins.

    `u_d_min` is MUNICH's `Minimum_transfer_velocity`, 0.001 m/s there (SRC `:3296`) and
    0.0 here. A NEGATIVE `sigma_w` -- which the unguarded neutral form
    `1.3 u* (1 - 0.8 z/h_abl)` produces whenever a street is taller than `1.25 h_abl` --
    is refused rather than clamped: it would make the roof term anti-diffusive, pumping
    mass INTO the street against its own gradient, and every solve would still report
    success.

    `sigma_w_min` is a floor on `sigma_w` itself, applied after that check, so
    `u_d = u_d(max(sigma_w, sigma_w_min))`: SIRANE's `SIGMA_W_MIN`, 0.30 m/s by default
    there and 0.0 here.
    """
    form = normalise("roof_exchange", keyword_alias(
        "exchange_velocity", "roof_exchange", roof_exchange, "form", form,
        "turbulent_velocity"), "exchange_velocity")
    sigma_w = torch.as_tensor(sigma_w, dtype=torch.float64)
    H = torch.as_tensor(H, dtype=torch.float64)
    W = torch.as_tensor(W, dtype=torch.float64)

    def velocity(sigma: Tensor) -> Tensor:
        if form == "turbulent_velocity":
            return sigma * EXCHANGE_SIGMA_W_RATIO
        return sigma * ASPECT_RATIO_EXCHANGE_BETA / (1.0 + H / W)

    u_d = velocity(sigma_w)
    if bool((u_d < 0).any()):
        count = int((u_d < 0).sum())
        raise ValueError(
            f"exchange_velocity: the exchange velocity is negative at {count} of "
            f"{u_d.numel()} entries (minimum {float(u_d.min())} m/s), which would make "
            f"the roof term anti-diffusive. sigma_w went negative because a street is "
            f"taller than 1.25 h_abl; pass pblh_floor=<max street height> to "
            f"boundary_layer(), or use stability='monin_obukhov', which guards it"
        )
    if sigma_w_min > 0.0:
        u_d = velocity(torch.clamp(sigma_w, min=float(sigma_w_min)))
    if u_d_min <= 0.0:
        return u_d
    return torch.clamp(u_d, min=float(u_d_min))
