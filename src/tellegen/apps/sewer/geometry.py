"""Circular-pipe geometry and Manning flow for a partially filled gravity sewer.

Exact circular-segment identities (Chow 1959, *Open-Channel Hydraulics*), used in place of
SWMM's own 51-point lookup tables (SWMM Reference Manual Vol. II section 5.1.3, which states
the tables are an explicit speed optimisation over these same trigonometric relations).

With ``theta = 2 arccos(1 - 2 h / D)`` the wetted half-angle in radians:

    A(h)  = D^2 (theta - sin theta) / 8        flow area
    P(h)  = D theta / 2                        wetted perimeter
    R(h)  = A / P = (D/4) (1 - sin theta / theta)
    T(h)  = D sin(theta / 2)                   top width
    d_m   = A / T                              hydraulic mean depth

Manning in SI (constant 1.0, NOT SWMM's 1.486, which is the foot-unit conversion
1/0.3048^(1/3): SWMM computes internally in feet and uses the same numeric ``n``):

    Q(h) = (1 / n) A(h) R(h)^(2/3) sqrt(S0)

``Q`` is strictly increasing on ``0 < h <= H_MAX_RATIO D`` and reaches its maximum there
(``H_MAX_RATIO = 0.938``); above that depth it decreases again, so the ascending branch is
the whole invertible domain and a discharge above ``Q(0.938 D)`` is refused as surcharge.

Every tensor here is built with an explicit ``torch.float64``: the repository's default
dtype is float32.
"""

from __future__ import annotations

import torch

from tellegen.solvers.scalar import solve_monotone

Tensor = torch.Tensor

F64 = torch.float64

#: Depth-to-diameter ratio at which Manning discharge peaks (Chow 1959; verified standard).
H_MAX_RATIO = 0.938

#: Floor applied to the divisor of ``A / T`` (hydraulic mean depth) and ``A_air / P_air``
#: (air hydraulic diameter) at their respective ``0/0`` points. Both are removable
#: singularities of the VALUE but genuine singularities of the DERIVATIVE, so the safe
#: divisor is substituted on BOTH branches of every ``torch.where`` below (milestone 3
#: lesson: a masked-out branch still back-propagates, and ``inf * 0`` is ``nan``).
_EPS = 1e-12

#: Below this half-angle ``1 - sin(theta)/theta`` is evaluated by its series.
_THETA_SERIES = 1e-4


def _theta(h: Tensor, diameter: Tensor) -> Tensor:
    """Wetted half-angle ``theta = 2 arccos(1 - 2 h / D)``, radians, in ``[0, 2 pi]``.

    ``arccos`` has an infinite derivative at both ends of its domain, so the naive
    composition back-propagates ``inf`` at exactly ``h = 0`` or ``h = D``, which then
    multiplies the ``0`` from ``dA/dtheta`` (etc.) into ``nan``. Clamping the argument to
    ``[-1 + _EPS, 1 - _EPS]`` (an offset clamp) was tried first and REJECTED: any
    float64-representable ``_EPS`` shifts the boundary VALUE by ``O(sqrt(_EPS))`` (e.g.
    ``_EPS = 1e-12`` moves ``theta(h=0)`` to ``~4e-6``), which is far outside the exact
    endpoint tolerances the tests demand (measured: fails both the full-pipe and
    zero-depth cases). Clamping to the true domain ``[-1, 1]`` instead is exact at the
    endpoints AND finite-gradient there for a different reason: ``torch.clamp``'s own
    backward is exactly ``0`` for an input at (or past) its bound -- not merely "small" --
    so it hard-zeroes even an ``inf``/``nan`` seed arriving from further downstream, not
    only a well-behaved one. The ``arccos`` derivative singularity is never reached by the
    gradient at all: it is gated shut one step upstream, unconditionally.
    """
    u = 1.0 - 2.0 * h / diameter
    return 2.0 * torch.arccos(torch.clamp(u, -1.0, 1.0))


def flow_area(h: Tensor, diameter: Tensor) -> Tensor:
    """Wetted cross-sectional area ``A = D^2 (theta - sin theta) / 8`` (m^2)."""
    theta = _theta(h, diameter)
    return diameter**2 * (theta - torch.sin(theta)) / 8.0


def wetted_perimeter(h: Tensor, diameter: Tensor) -> Tensor:
    """Wetted perimeter ``P = D theta / 2`` (m)."""
    return diameter * _theta(h, diameter) / 2.0


def top_width(h: Tensor, diameter: Tensor) -> Tensor:
    """Free-surface width ``T = D sin(theta / 2)`` (m)."""
    return diameter * torch.sin(_theta(h, diameter) / 2.0)


def hydraulic_radius(h: Tensor, diameter: Tensor) -> Tensor:
    """Hydraulic radius ``R = A / P = (D/4)(1 - sin theta / theta)`` (m).

    Written in the closed form rather than as ``A / P`` so that the ``0/0`` at ``h = 0`` is
    one explicitly guarded division: ``theta`` is floored on BOTH branches of the ``where``
    (the series ``1 - sin t / t = t^2/6 - t^4/120 + ...`` is used below the floor), so
    neither the value nor the gradient can be ``nan``.
    """
    theta = _theta(h, diameter)
    small = theta < _THETA_SERIES
    theta_safe = torch.where(small, torch.full_like(theta, _THETA_SERIES), theta)
    series = theta**2 / 6.0 - theta**4 / 120.0
    ratio = torch.where(small, series, 1.0 - torch.sin(theta_safe) / theta_safe)
    return diameter * ratio / 4.0


def hydraulic_mean_depth(h: Tensor, diameter: Tensor) -> Tensor:
    """Hydraulic mean depth ``d_m = A / T`` (m), the length scale of the sulfide and
    two-film correlations. ``T`` is floored on both branches of the ``where``."""
    area = flow_area(h, diameter)
    width = top_width(h, diameter)
    small = width < _EPS
    width_safe = torch.where(small, torch.full_like(width, _EPS), width)
    return torch.where(small, torch.zeros_like(area), area / width_safe)


def manning_flow(h: Tensor, diameter: Tensor, roughness: Tensor, slope: Tensor) -> Tensor:
    """Manning discharge ``Q = (1/n) A R^(2/3) sqrt(S0)`` (m^3/s), SI constant 1.0.

    ``R^(2/3)`` has an infinite local derivative at ``R = 0`` (a dry pipe, ``h = 0``). Left
    unguarded, that ``inf`` flows back into ``hydraulic_radius``'s own ``diameter * ratio /
    4`` and multiplies the ``0`` from ``ratio`` there into ``nan`` for ``d(Q)/d(diameter)``
    -- ``diameter`` reaches ``hydraulic_radius`` through a plain multiplicative factor, not
    through ``_theta``'s clamp gate, so it is not protected the way ``h``, ``n`` and ``s``
    are. ``R`` is floored on both branches of the ``where`` (the module's usual pattern):
    the value is unchanged (``0`` at a dry pipe) and the local derivative of the substituted
    branch is finite, so no ``inf`` is ever produced for any gradient to multiply by ``0``.
    """
    area = flow_area(h, diameter)
    radius = hydraulic_radius(h, diameter)
    positive = radius > 0
    safe_radius = torch.where(positive, radius, torch.ones_like(radius))
    factor = torch.where(positive, safe_radius ** (2.0 / 3.0), torch.zeros_like(radius))
    return area * factor * torch.sqrt(slope) / roughness


def capacity_flow(diameter: Tensor, roughness: Tensor, slope: Tensor) -> Tensor:
    """``Q_max = Q(H_MAX_RATIO D)``: the largest discharge the ascending Manning branch
    carries, and the surcharge threshold."""
    return manning_flow(H_MAX_RATIO * diameter, diameter, roughness, slope)


def normal_depth(
    q: Tensor,
    diameter: Tensor,
    roughness: Tensor,
    slope: Tensor,
    *,
    names: list[str] | None = None,
    tol: float = 1e-14,
) -> Tensor:
    """Manning normal depth (m) for discharge ``q``, batched over pipes and instances.

    Inverts ``Q(h) = q`` on the ascending branch ``[0, H_MAX_RATIO D]`` with
    ``solvers.scalar.solve_monotone``. The bracket is justified: ``Q`` is strictly
    increasing there and ``Q(0.938 D)`` is its maximum, so a sign change is bracketed for
    every ``0 < q <= Q_max``.

    ``q > Q_max`` is REFUSED by name (surcharge; out of scope). ``q == 0`` returns exactly
    ``0``: a positive floor is substituted into the solve on BOTH branches of the mask so
    the un-taken branch back-propagates a finite (zero-weighted) gradient rather than
    ``inf * 0 = nan``. The true sensitivity ``dh/dq = 1/(dQ/dh)`` diverges as ``q -> 0``, so
    the gradient reported at exactly zero flow is ``0``: a documented modelling choice
    (spec amendment A7).
    """
    if bool(torch.any(~torch.isfinite(q))):
        raise ValueError("sewer.normal_depth: discharge must be finite")
    if bool(torch.any(q < 0)):
        raise ValueError(
            "sewer.normal_depth: discharge must be non-negative (pipes are oriented "
            "upstream to downstream, so a negative discharge is a topology error)"
        )
    q_max = capacity_flow(diameter, roughness, slope)
    over = q > q_max
    if bool(torch.any(over)):
        flat = over.reshape(-1, over.shape[-1]).any(0)
        idx = flat.nonzero().flatten().tolist()
        label = [names[i] if names else str(i) for i in idx]
        q_flat = q.reshape(-1, q.shape[-1])
        cap_flat = q_max.expand_as(q).reshape(-1, q.shape[-1])
        worst = [float(q_flat[:, i].max()) for i in idx]
        cap = [float(cap_flat[:, i].min()) for i in idx]
        raise ValueError(
            f"sewer.normal_depth: surcharge at pipe(s) {label}: discharge {worst} m3/s "
            f"exceeds the Manning capacity {cap} m3/s at h = {H_MAX_RATIO} D; surcharged "
            f"and backwater flow are out of scope for this milestone"
        )
    active = q > 0
    q_floor = 1e-6 * q_max * torch.ones_like(q)
    q_safe = torch.where(active, q, q_floor)
    lo = torch.zeros_like(q_safe)
    hi = H_MAX_RATIO * diameter * torch.ones_like(q_safe)

    def residual(h, q_t, d_t, n_t, s_t):
        return manning_flow(h, d_t, n_t, s_t) - q_t

    h = solve_monotone(
        residual,
        lo,
        hi,
        q_safe,
        diameter * torch.ones_like(q_safe),
        roughness * torch.ones_like(q_safe),
        slope * torch.ones_like(q_safe),
        tol=tol,
        max_iter=200,
    )
    return torch.where(active, h, torch.zeros_like(h))


def air_geometry(h: Tensor, diameter: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(A_air, P_air, D_h)`` for the headspace above the water surface.

    ``A_air = pi D^2/4 - A(h)``; the air-side wetted perimeter is the DRY WALL plus the
    water surface, ``P_air = pi D - P(h) + T(h)`` (closed-conduit convention, the water
    surface acting as a moving wall -- Edwini-Bonsu and Steffler 2004, Fig. 1); the air
    hydraulic diameter is ``D_h = 4 A_air / P_air``.
    """
    area = flow_area(h, diameter)
    a_air = torch.pi * diameter**2 / 4.0 - area
    p_air = torch.pi * diameter - wetted_perimeter(h, diameter) + top_width(h, diameter)
    small = p_air < _EPS
    p_safe = torch.where(small, torch.full_like(p_air, _EPS), p_air)
    d_h = torch.where(small, torch.zeros_like(a_air), 4.0 * a_air / p_safe)
    return a_air, p_air, d_h
