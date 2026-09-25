"""Street-canyon boundary layer, in-canyon wind and roof exchange velocity.

Every formula here carries its source. The two families are SIRANE's (Soulhac, Perkins and
Salizzoni 2008; Soulhac et al. 2011), which IMPAQ implements, and MUNICH's (Kim et al.
2018 and 2022 plus the MUNICH/AtmoData sources), which the parity checks of Task 11 pin.
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

from noodl.solvers.scalar import solve_monotone

Tensor = torch.Tensor

KAPPA_IMPAQ = 0.4
"""von Karman constant as IMPAQ uses it (`impaq.canyon_velocity`, `kappa = 0.4`)."""

KAPPA_MUNICH = 0.41
"""von Karman constant as MUNICH uses it (`StreetNetworkTransport.cxx:21`)."""

GAMMA_E = 0.577
"""Euler-Mascheroni as BOTH codes hard-code it -- not 0.5772156649. IMPAQ
`canyon_velocity`; ATM `MeteorologyStreet.cxx:118`. Worth about 4e-5 relative."""

Z0_B_DEFAULT = 0.15
"""In-canyon (building) roughness, IMPAQ's loader default, m."""

Z0_S_DEFAULT = 0.01
"""Surface roughness `z0_surface`, MUNICH's default (`StreetNetworkTransport.cxx:155`)."""

SIRANE_EXCHANGE = 1.0 / (math.sqrt(2.0) * math.pi)
"""`u_d / sigma_w` in SIRANE: 0.225079079039277. S11 Eq. (5) p. 7386, K18 Eq. (3) p. 613,
K22 Eq. (B10) p. 7387 and `StreetNetworkTransport.cxx:3273` all read `sigma_w/(sqrt(2) pi)`
-- the radical covers only the 2, verified at glyph level in all three PDFs. The framework
spec's and IMPAQ's docstring's "issue C" (that this should be `sigma_w/sqrt(2 pi)`) is
RETRACTED; `1/sqrt(2 pi) = 0.398942280401433` appears in no source."""

SCHULTE_BETA = 2.0 / (math.sqrt(2.0) * math.pi)
"""`beta` in Schulte's mixing-length form: 0.450158158078553, fixed by matching the SIRANE
form at `a_r = 1` (K18 p. 613). ATM `ComputeSchulteLm`, `MeteorologyStreet.cxx:547-554`;
MUNICH v1.0 used the literal 0.45 instead, a 0.035 % difference."""

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


def soulhac_residual(c: Tensor, ratio: Tensor) -> Tensor:
    """`0.5 (z0/di) c - exp((pi/2) Y1(c)/J1(c) - gamma_E)`; zero at the shape parameter.

    IMPAQ's `canyon_velocity` writes exactly this; MUNICH writes the same equation as
    `z0_s/delta_i = (2/C) exp(...)` (K22 Eq. B12; ATM `MeteorologyStreet.cxx:119-153`,
    where it is solved by a brute-force search on a 0.01 grid, so MUNICH's `C` is quantised
    -- worth 4e-4 relative in `u_M`. This solve is continuous.)
    """
    return 0.5 * ratio * c - torch.exp(
        (math.pi / 2.0) * bessel_y1(c) / bessel_j1(c) - GAMMA_E
    )


def soulhac_shape(ratio: Tensor) -> Tensor:
    """The root `c` of `soulhac_residual`, batched and differentiable in `ratio`.

    Bracket `[1e-4, 3.0]`, justified in the plan's Task 3: the residual is `+0.5 r c > 0`
    at the low end (the exponential underflows to exactly zero there) and
    `1.5 r - 2.527 < 0` at the high end for every `r < 1.685`, and it changes sign exactly
    once in between. A ratio at or above 1.6 is refused rather than handed to the solver,
    whose "no sign change is bracketed" message would name neither the ratio nor the reason.
    """
    ratio = torch.as_tensor(ratio, dtype=torch.float64)
    if bool((ratio <= 0).any()):
        bad = torch.nonzero(ratio.reshape(-1) <= 0).flatten().tolist()
        raise ValueError(
            f"soulhac_shape: the roughness ratio z0_b/di must be strictly positive; "
            f"non-positive at flat indices {bad}"
        )
    if bool((ratio >= C_RATIO_MAX).any()):
        bad = torch.nonzero(ratio.reshape(-1) >= C_RATIO_MAX).flatten().tolist()
        worst = float(ratio.reshape(-1)[bad].max())
        raise ValueError(
            f"soulhac_shape: the roughness ratio z0_b/di must be below {C_RATIO_MAX} for "
            f"the shape equation to have a root in [{C_BRACKET_LO}, {C_BRACKET_HI}]; got "
            f"up to {worst} at flat index/indices {bad}. A roughness comparable to the "
            f"canyon half-width is not a canyon -- check z0_b and the street width."
        )
    lo = torch.full_like(ratio, C_BRACKET_LO)
    hi = torch.full_like(ratio, C_BRACKET_HI)
    return solve_monotone(soulhac_residual, lo, hi, ratio, tol=1e-14, max_iter=200)


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
    derived with, kept so that `sigma_v` and MUNICH's `w*` use the same one (0.4 for IMPAQ,
    0.41 for MUNICH -- 2.5 % on `u_star`).
    """

    u_star: Tensor
    h_abl: Tensor
    z_ref: Tensor
    d: Tensor
    z0: Tensor
    kappa: float = KAPPA_IMPAQ

    def sigma_w(self, z: Tensor, *, lmo: Tensor | None = None,
                stability: str = "impaq") -> Tensor:
        """Vertical velocity standard deviation at height `z`.

        `stability="impaq"` is `1.3 u* (1 - 0.8 z / h_abl)` (IMPAQ's
        `BoundaryLayer.sigma_w`), with NO guard: it goes NEGATIVE for `z > 1.25 h_abl`,
        which happens on 7 of the 474336 (time, street) pairs of `leiden_small` (3 of its
        2928 forcing steps, measured). Use `boundary_layer(pblh_floor=...)` to apply
        MUNICH's `pblh := max(H, PBLH)` guard, or `exchange_velocity` will refuse the
        result.

        `stability="munich"` is `ComputeSigmaW`, `StreetNetworkTransport.cxx:3221-3260`,
        in three branches on the Monin-Obukhov length. Its neutral branch is IMPAQ's
        formula exactly.
        """
        z = torch.as_tensor(z, dtype=self.u_star.dtype)
        if stability == "impaq":
            return 1.3 * self.u_star * (1.0 - 0.8 * z / self.h_abl)
        if stability != "munich":
            raise ValueError(
                f"BoundaryLayer.sigma_w: stability must be 'impaq' or 'munich', got "
                f"{stability!r}"
            )
        if lmo is None:
            raise ValueError(
                "BoundaryLayer.sigma_w: stability='munich' needs the Monin-Obukhov length "
                "`lmo` (m); it selects the unstable/stable/neutral branch"
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

    def sigma_v(self, *, lmo: Tensor | None = None, stability: str = "impaq") -> Tensor:
        """Horizontal velocity standard deviation, MUNICH's `ComputeSigmaV`
        (`StreetNetworkTransport.cxx:3194-3216`): a 10-level average over `z in [0, PBLH]`.

        `stability="impaq"` selects the neutral branch, which collapses to exactly
        `1.2 u*` (the mean of `1 - 0.8 z/PBLH` over 10 equally spaced points on `[0, PBLH]`
        is 0.6). IMPAQ has no `sigma_v` of its own; this is the value the MUNICH direction
        averaging of Task 4 needs, and it is stated here rather than invented there.
        """
        levels = torch.arange(10, dtype=self.u_star.dtype) / 9.0
        z_over_pblh = levels.reshape(*([1] * self.u_star.dim()), 10)
        u_star = self.u_star.unsqueeze(-1)
        neutral = 2.0 * u_star * (1.0 - 0.8 * z_over_pblh)
        if stability == "impaq":
            return neutral.mean(-1)
        if stability != "munich":
            raise ValueError(
                f"BoundaryLayer.sigma_v: stability must be 'impaq' or 'munich', got "
                f"{stability!r}"
            )
        if lmo is None:
            raise ValueError(
                "BoundaryLayer.sigma_v: stability='munich' needs the Monin-Obukhov length "
                "`lmo` (m); it selects the unstable/stable/neutral branch"
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
    kappa: float = KAPPA_IMPAQ,
    pblh_floor: Tensor | float | None = None,
) -> BoundaryLayer:
    """IMPAQ's `compute_boundary_layer`, batched and differentiable.

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


def macdonald_profile(
    h_mean: Tensor,
    w_mean: Tensor,
    *,
    lambda_p: float = 0.4,
    big_delta: float = 4.43,
    small_delta: float = 1.0,
    c_db: float = 1.2,
    kappa: float = KAPPA_MUNICH,
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


def roof_wind(
    u_star: Tensor,
    H: Tensor,
    W: Tensor,
    *,
    form: str = "sirane",
    z0_s: Tensor | float = Z0_S_DEFAULT,
    kappa: float = KAPPA_MUNICH,
    h_mean: Tensor | None = None,
    w_mean: Tensor | None = None,
    n_levels: int = _N_ROOF_LEVELS,
) -> Tensor:
    """Wind speed at roof level `u_H`, from the friction velocity.

    `form="sirane"` is K22 Eq. (B12), p. 7387 (ATM `MeteorologyStreet.cxx:114-201`):
    `delta_i = min(H, W/2)`, `C` from `soulhac_shape(z0_s/delta_i)`,
    `u_M = u* sqrt(pi/(sqrt(2) kappa^2 C) [Y0(C) - J0(C) Y1(C)/J1(C)])` and
    `u_H = u_M f_mean` with `f_mean` the 100-level mean of
    `[J1(C) Y0(C y) - J0(C y) Y1(C)] / [J1(C) Y0(C) - J0(C) Y1(C)]` over `y = k/N`.
    (ATM compares the DIMENSIONLESS `y` against the DIMENSIONAL `z0_s` at its line 192 --
    a latent unit bug that never bites at the default `z0_s = 0.01`, so all 100 levels are
    taken here, exactly as MUNICH does at its default.)

    `form="macdonald"` is K22 Eq. (B13): `u_H = (u*/kappa) ln((H - d_c)/z0c)` with `d_c`
    and `z0c` from `macdonald_profile(h_mean, w_mean)`. MUNICH returns 0 when
    `H < d_c + z0c` (SRC `:3334`); that guard is reproduced here.
    """
    u_star = torch.as_tensor(u_star, dtype=torch.float64)
    H = torch.as_tensor(H, dtype=torch.float64)
    W = torch.as_tensor(W, dtype=torch.float64)
    if form == "macdonald":
        if h_mean is None or w_mean is None:
            raise ValueError(
                "roof_wind: form='macdonald' needs the network means h_mean and w_mean "
                "(K22 Eq. 3 is written on network means, not per-street values)"
            )
        d_c, z0c = macdonald_profile(h_mean, w_mean, kappa=kappa)
        above = H > d_c + z0c
        safe = torch.where(above, (H - d_c) / z0c, torch.ones_like(H * d_c))
        return torch.where(above, (u_star / kappa) * torch.log(safe),
                           torch.zeros_like(safe))
    if form != "sirane":
        raise ValueError(
            f"roof_wind: form must be 'sirane' or 'macdonald', got {form!r}"
        )
    z0_s = torch.as_tensor(z0_s, dtype=torch.float64)
    delta_i = torch.minimum(H, W / 2.0)
    c = soulhac_shape(z0_s / delta_i)
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
    form: str = "soulhac",
    z0_b: Tensor | float = Z0_B_DEFAULT,
    z0_s: Tensor | float = Z0_S_DEFAULT,
    kappa: float = KAPPA_IMPAQ,
    canyon_wind_min: float = 0.0,
) -> Tensor:
    """The SIGNED along-canyon velocity, m/s. Positive means from `u` to `v`.

    `phi` is the angle between the wind and the street axis; it broadcasts against `W`,
    `H` and any leading forcing batch.

    `form="soulhac"` (needs `u_star`) is the Soulhac-Perkins-Salizzoni (2008) closed form
    exactly as IMPAQ's `canyon_velocity` writes it, and as K22 Eq. (B15) p. 7388 writes it:
    `di = min(W/2, H)`, `alpha = ln(di/z0_b)`, `beta = exp(c/sqrt(2) (1 - H/di))`,
    `u_h = u* sqrt(pi/(sqrt(2) kappa^2 c) [Y0(c) - J0(c) Y1(c)/J1(c)])`, and

        u = u_h cos(phi) di^2/(W H)
            [ 2 sqrt2/c (1 - beta)(1 - c^2/3 + c^4/45)
              + beta (2 alpha - 3)/alpha + (W/di - 2)(alpha - 1)/alpha ]

    The Soulhac form REFUSES `z0_b >= di`, which `soulhac_shape`'s own `z0_b/di < 1.6`
    bound lets through: `alpha` is zero at ratio 1 and negative above it, and the answer
    comes back NaN rather than wrong-looking.

    `form="exponential"` (needs `u_h`) is K22 Eq. (B14), p. 7388 -- SINGLE regime, `2/a_r`
    prefactor, integrated from the street roughness `z0_s`:

        u = u_h cos(phi) (2/a_r) [1 - exp((a_r/2)(z0_s/H - 1))],   a_r = H/W

    It is NOT K18 Eqs. (9)-(11) (three aspect-ratio regimes, `2/pi` prefactor, integrated
    from 0), which is MUNICH v1.0 and differs by up to 37 % for narrow canyons. ATM
    `ComputeExpUstreet`, `MeteorologyStreet.cxx:257-263` computes B14 and nothing else.

    `canyon_wind_min` is MUNICH's `ustreet_min`, default 0.1 m/s there (SRC `:3451`,
    `Minimum_Street_Wind_Speed`) and 0.0 here so that IMPAQ parity is the default. The
    floor keeps the sign: `U = sign * max(|u|, u_min)` with `sign = +1` wherever the
    unfloored value is `>= 0`. The `>=` matters exactly once -- at `phi = pi/2`, where the
    unfloored value is `+0` and MUNICH's own `>` classification makes the street an
    OUTFLOW; that is what produces the 270-degree panel of K22 Fig. 1.
    """
    W = torch.as_tensor(W, dtype=torch.float64)
    H = torch.as_tensor(H, dtype=torch.float64)
    phi = torch.as_tensor(phi, dtype=torch.float64)
    if form == "soulhac":
        if u_star is None:
            raise ValueError(
                "canyon_velocity: form='soulhac' needs u_star (the Bessel profile is "
                "written on the friction velocity, not on the roof wind)"
            )
        u_star = torch.as_tensor(u_star, dtype=torch.float64)
        z0_b = torch.as_tensor(z0_b, dtype=torch.float64)
        di = torch.minimum(W / 2.0, H)
        # `soulhac_shape` refuses z0_b/di >= 1.6, which is NOT enough here: `alpha` is
        # ln(di/z0_b), exactly 0 at ratio 1 and negative on (1, 1.6), so the shape function
        # divides by zero or by a negative there and the answer comes back NaN with no
        # error anywhere. Refused by name rather than clamped.
        wide, rough, half = torch.broadcast_tensors(W, z0_b, di)
        if bool((rough >= half).any()):
            bad = torch.nonzero((rough >= half).reshape(-1)).flatten().tolist()
            raise ValueError(
                f"canyon_velocity: form='soulhac' needs the in-canyon roughness z0_b "
                f"strictly below di = min(W/2, H), because alpha = ln(di/z0_b) divides "
                f"the shape function; at flat index/indices {bad} the widths are "
                f"{[float(v) for v in wide.reshape(-1)[bad]]} m and the roughnesses are "
                f"{[float(v) for v in rough.reshape(-1)[bad]]} m"
            )
        c = soulhac_shape(z0_b / di)
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
    elif form == "exponential":
        if u_h is None:
            raise ValueError(
                "canyon_velocity: form='exponential' needs u_h (K22 Eq. B14 is written on "
                "the roof-level wind; get it from roof_wind())"
            )
        u_h = torch.as_tensor(u_h, dtype=torch.float64)
        z0_s = torch.as_tensor(z0_s, dtype=torch.float64)
        a_r = H / W
        half = 0.5 * a_r
        signed = u_h * torch.cos(phi) * (1.0 / half) * (
            1.0 - torch.exp(half * (z0_s / H - 1.0))
        )
    else:
        raise ValueError(
            f"canyon_velocity: form must be 'soulhac' or 'exponential', got {form!r}"
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
    form: str = "sirane",
    u_d_min: float = 0.0,
) -> Tensor:
    """The roof exchange velocity `u_d`, m/s.

    `form="sirane"`: `u_d = sigma_w / (sqrt(2) pi)`, independent of the aspect ratio.
    S11 Eq. (5), K18 Eq. (3), K22 Eq. (B10), `StreetNetworkTransport.cxx:3273`. See
    `SIRANE_EXCHANGE` for why the framework spec's "issue C" is retracted.

    `form="schulte"`: `u_d = beta sigma_w / (1 + H/W)` with `beta = 2/(sqrt(2) pi)`
    (K18 Eqs. 4-8, K22 Eq. B11, ATM `ComputeSchulteLm`), MUNICH v2's default. The two
    agree exactly at `H = W`, which a test pins.

    `u_d_min` is MUNICH's `Minimum_transfer_velocity`, 0.001 m/s there (SRC `:3296`) and
    0.0 here. A NEGATIVE `sigma_w` -- which IMPAQ's unguarded
    `1.3 u* (1 - 0.8 z/h_abl)` produces whenever a street is taller than `1.25 h_abl` --
    is refused rather than clamped: it would make the roof term anti-diffusive, pumping
    mass INTO the street against its own gradient, and every solve would still report
    success.
    """
    sigma_w = torch.as_tensor(sigma_w, dtype=torch.float64)
    H = torch.as_tensor(H, dtype=torch.float64)
    W = torch.as_tensor(W, dtype=torch.float64)
    if form == "sirane":
        u_d = sigma_w * SIRANE_EXCHANGE
    elif form == "schulte":
        u_d = sigma_w * SCHULTE_BETA / (1.0 + H / W)
    else:
        raise ValueError(
            f"exchange_velocity: form must be 'sirane' or 'schulte', got {form!r}"
        )
    if bool((u_d < 0).any()):
        count = int((u_d < 0).sum())
        raise ValueError(
            f"exchange_velocity: the exchange velocity is negative at {count} of "
            f"{u_d.numel()} entries (minimum {float(u_d.min())} m/s), which would make "
            f"the roof term anti-diffusive. sigma_w went negative because a street is "
            f"taller than 1.25 h_abl; pass pblh_floor=<max street height> to "
            f"boundary_layer(), or use stability='munich', which guards it"
        )
    if u_d_min <= 0.0:
        return u_d
    return torch.clamp(u_d, min=float(u_d_min))
