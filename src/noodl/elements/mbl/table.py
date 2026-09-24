"""MBL's tabulated flow law, transcribed from MBL v13.0.0
(commit 55abf579598ca81cae0a82f337350375958e6722): ``Buildings.Airflow.Multizone.Table_m_flow``
and its ``Table_V_flow`` extension.

``Table_m_flow`` (``Table_m_flow.mo:1-27``) fits a monotone cubic Hermite spline through
user-supplied ``(dpMea_nominal, mMea_flow_nominal)`` points and evaluates it at ``dp``:

    m_flow = interpolate(u=dp, xd=dpMea_nominal, yd=mMea_flow_nominal, d=d)   (:5-9)
    d = splineDerivatives(x=dpMea_nominal, y=mMea_flow_nominal, ensureMonotonicity=true)  (:19-23)

(``d`` -- the derivatives at the knots -- is computed once, at construction, never inside the
evaluation itself, exactly as MBL's own ``protected parameter`` is.)

``Table_V_flow`` (``Table_V_flow.mo:1-5``) is the same model with the table's y-data scaled to
mass-flow units *before* ``splineDerivatives``/``interpolate`` ever see it:

    mMea_flow_nominal = VMea_flow_nominal * rho_default    (``Table_V_flow.mo:5``)

This is a different order to :mod:`noodl.elements.mbl.powerlaw`'s volume form, which multiplies
``rho_default`` in *after* evaluating the volumetric law -- correct there because that law is a
plain algebraic function of ``dp``, so pre- and post-scaling agree trivially. Here the
``ensureMonotonicity=true`` correction inside ``splineDerivatives`` (below) is only *positively
homogeneous* in ``y`` (its internal ratios ``alpha = d[i]/delta[i]``, ``beta = d[i+1]/delta[i]``
are scale-invariant, so the corrected ``d`` scales exactly with ``y`` for any positive
``rho_default``) -- true, but this module still scales early, exactly where
``Table_V_flow.mo:5`` does it, rather than leaning on that homogeneity argument.

MBL's ``interpolate`` (``Utilities/Math/Functions/interpolate.mo``) selects one interval per
evaluation point with a linear scan (``:17-22``: the largest ``j`` with ``u > xd[j]``, i.e. the
same one ``bisect_left`` would pick, capped to the first/last interval for extrapolation), then
calls ``cubicHermiteLinearExtrapolation`` (``:24-31``) on that interval. This module reproduces
the interval choice with a single vectorised ``torch.searchsorted`` (no Python loop over points)
and reproduces ``cubicHermiteLinearExtrapolation.mo``/``Modelica.Fluid.Utilities.cubicHermite``
(MSL ``Fluid/Utilities.mo:787-832``) exactly, including the boundary convention that evaluating
*exactly* at an interior knot always takes the "linear" branch of whichever interval has that
knot as its endpoint (still exactly reproducing the knot's own y-value, by construction of the
Hermite basis) rather than the "interior cubic" branch of the neighbouring interval.
"""

from __future__ import annotations

import sys

import torch

from noodl.elements.base import Element

Tensor = torch.Tensor

# Modelica/Constants.mo:20 ("small = Minimum normalized positive floating-point number",
# ModelicaServices.Machine.small): for IEEE754 double precision this is DBL_MIN,
# 2.2250738585072014e-308, identical to Python's sys.float_info.min.
_MODELICA_SMALL = sys.float_info.min


def _f64(value) -> Tensor:
    """A caller's tensor keeps its own dtype; a bare Python number/sequence becomes float64
    -- see ``noodl.elements.mbl.powerlaw._f64`` (duplicated locally: the ``mbl`` package has
    no shared-utility module yet).
    """
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=torch.float64)


def _spline_derivatives(x: Tensor, y: Tensor) -> Tensor:
    """``Utilities/Math/Functions/splineDerivatives.mo:31-67``, always with
    ``ensureMonotonicity=true`` -- the only value ``Table_m_flow.mo:19-23`` (and hence
    ``Table_V_flow``, which inherits it unchanged) ever passes, so this module never exposes
    it as an option. ``x`` (knots) and ``y`` (function values) are 1-D tensors of equal length.

    Length-1 and length-2 tables take ``:32-38``'s special-cased formulas (a single zero
    derivative, respectively the shared secant slope of the one segment); any longer table
    takes the general secant-averaging formula (``:39-51``) followed by the Fritsch-Carlson
    monotonicity correction (``:54-67``, guarded there by ``n > 2``, matched here since this
    function is only reached with ``n >= 2`` and the ``n == 2`` case already returned above).

    The correction loop is genuinely sequential -- interval ``i``'s update reads ``d[i]``,
    which interval ``i - 1``'s own update may have just overwritten (each interior knot is
    shared by two consecutive intervals) -- so it is transcribed as a Python loop over knots,
    exactly as MBL's own ``for i in 1:n - 1`` is, rather than vectorised. This runs once per
    :class:`MBLTable` construction, never inside :meth:`MBLTable.flow`, over the table's
    typically small number of knots.
    """
    n = x.shape[-1]
    if n == 1:
        return torch.zeros_like(y)  # splineDerivatives.mo:34
    if n == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])  # splineDerivatives.mo:37-38
        return torch.stack([slope, slope])

    delta = (y[1:] - y[:-1]) / (x[1:] - x[:-1])  # splineDerivatives.mo:42, length n - 1
    d: list[Tensor] = [None] * n  # type: ignore[list-item]
    d[0] = delta[0]  # :46, "d[1] := delta[1]"
    d[-1] = delta[-1]  # :47, "d[n] := delta[n - 1]"
    for i in range(1, n - 1):  # :49, "for i in 2:n - 1"
        d[i] = (delta[i - 1] + delta[i]) / 2  # :50

    small = torch.as_tensor(_MODELICA_SMALL, dtype=x.dtype)
    for i in range(n - 1):  # :56, "for i in 1:n - 1"
        if delta[i].abs() < small:  # :57
            d[i] = torch.zeros_like(delta[i])  # :58
            d[i + 1] = torch.zeros_like(delta[i])  # :59
        else:
            alpha = d[i] / delta[i]  # :61
            beta = d[i + 1] / delta[i]  # :62
            if (alpha * alpha + beta * beta) > 9:  # :64
                tau = 3.0 / torch.sqrt(alpha * alpha + beta * beta)  # :65
                d[i] = delta[i] * alpha * tau  # :66
                d[i + 1] = delta[i] * beta * tau  # :67
    return torch.stack(d)


class MBLTable(Element):
    """MBL's tabulated flow law: a monotone cubic Hermite spline through ``(dp, flow)``
    knots, linearly extrapolated outside them; ``flow(dp)`` always returns MASS flow in kg/s.

    ``form="volume"`` (``Table_V_flow``) treats ``flow_points`` as volume-flow-rate knots and
    scales them to mass-flow knots by ``rho_default`` before fitting the spline
    (``Table_V_flow.mo:5``, see the module docstring); ``form="mass"`` (``Table_m_flow``)
    treats ``flow_points`` as mass-flow knots directly. ``rho_default`` is required (and used)
    only for ``form="volume"``.
    """

    def __init__(
        self,
        dp_points,
        flow_points,
        *,
        form: str,
        rho_default: float | None = None,
        kind: str = "airpath",
    ) -> None:
        super().__init__(kind)
        if form not in ("volume", "mass"):
            raise ValueError(
                f"MBLTable (kind {kind!r}): form must be 'volume' or 'mass', got {form!r}"
            )
        dp = _f64(dp_points)
        flow = _f64(flow_points)
        if dp.shape[-1] != flow.shape[-1]:
            raise ValueError(
                f"MBLTable (kind {kind!r}): dp_points and flow_points must have the same "
                f"length (Table_m_flow.mo:25-27's own size assert), got "
                f"len(dp_points)={dp.shape[-1]} and len(flow_points)={flow.shape[-1]}"
            )
        n = dp.shape[-1]
        if n < 2:
            raise ValueError(
                f"MBLTable (kind {kind!r}): at least 2 knots are required "
                f"(interpolate.mo needs two support points per interval), got {n}"
            )
        for i in range(1, n):
            if not bool(dp[i] > dp[i - 1]):
                raise ValueError(
                    f"MBLTable (kind {kind!r}): dp_points must be strictly increasing "
                    f"(splineDerivatives.mo:19-21's own assert); dp_points[{i}] = "
                    f"{dp[i].item()!r} is not greater than dp_points[{i - 1}] = "
                    f"{dp[i - 1].item()!r}"
                )

        if form == "volume":
            if rho_default is None:
                raise ValueError(
                    f"MBLTable (kind {kind!r}): form='volume' requires rho_default "
                    f"(Table_V_flow.mo:5: mMea_flow_nominal = VMea_flow_nominal*rho_default)"
                )
            mass_points = float(rho_default) * flow
        else:
            mass_points = flow

        self.dp_points = dp
        self.mass_points = mass_points
        self.d = _spline_derivatives(dp, mass_points)
        self.form = form
        self.rho_default = None if rho_default is None else float(rho_default)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        """``interpolate.mo:16-31`` + ``cubicHermiteLinearExtrapolation.mo:13-27`` +
        ``Modelica.Fluid.Utilities.cubicHermite`` (MSL ``Fluid/Utilities.mo:807-824``),
        vectorised: the interval containing each entry of ``dp`` is found with one
        ``torch.searchsorted`` call (no Python loop over points), then every entry's cubic
        Hermite value and both linear-extrapolation values are computed unconditionally and
        selected with ``torch.where`` (cheap here: no singularity like ``MBLPowerLaw``'s
        ``|dp|^m`` at ``dp = 0`` -- a plain polynomial has no ``dp_safe`` hazard).

        Interval choice: MBL's ``interpolate`` picks ``i`` = the largest ``j`` in
        ``1:n - 1`` with ``u > xd[j]`` (``1`` if none), i.e. the count of knots (excluding the
        last) strictly less than ``u``, minus one, clamped to ``0``
        (``torch.searchsorted(xd[:-1], u, right=False)`` counts exactly that). This chooses,
        for ``u`` exactly at an interior knot, the interval whose *upper* endpoint is that
        knot -- ``cubicHermiteLinearExtrapolation``'s ``elif``/``else`` branches then return
        that endpoint's own ``y``/``y`` + 0 exactly, so the choice is immaterial to the value
        (both neighbouring intervals' Hermite pieces agree there by construction) but fixes
        which branch (`inside` vs `linear`) this implementation takes, matching MBL bit for
        bit rather than merely to floating-point tolerance.
        """
        xd = self.dp_points.to(dtype=dp.dtype)
        yd = self.mass_points.to(dtype=dp.dtype)
        d = self.d.to(dtype=dp.dtype)

        x0 = xd[:-1]
        count_less = torch.searchsorted(x0, dp, right=False)
        i0 = torch.clamp(count_less - 1, min=0)
        i1 = i0 + 1

        x1 = xd[i0]
        x2 = xd[i1]
        y1 = yd[i0]
        y2 = yd[i1]
        y1d = d[i0]
        y2d = d[i1]

        h = x2 - x1
        t = (dp - x1) / h
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        cubic = y1 * h00 + h * y1d * h10 + y2 * h01 + h * y2d * h11

        lin_left = y1 + (dp - x1) * y1d
        lin_right = y2 + (dp - x2) * y2d

        inside = (dp > x1) & (dp < x2)
        left = dp <= x1
        return torch.where(inside, cubic, torch.where(left, lin_left, lin_right))
