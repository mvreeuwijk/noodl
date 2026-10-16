"""Measured airflow curves, fitted as ContamX fits them.

ContamX's curve fit, used by its cubic-spline airflow elements and its particle filter: the
first and the last segment are straight lines through their two points; between them, a
cubic spline through the points from the second to the second-last, clamped to those two
straight segments' slopes; beyond the first and the last point the straight segments
continue. ContamX 3.4.1.7 prints this fit ("CSFchk" in its log), and every value measured
agreed with it to float32 roundoff. The curve is evaluated at the signed argument: a
negative pressure difference extends the first segment, it does not mirror the curve.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.elements.base import ConstitutiveLaw
from noodl.solvers.scalar import solve_monotone

Tensor = torch.Tensor
F64 = torch.float64


def clamped_spline_slopes(x: Tensor, y: Tensor, start: Tensor, end: Tensor) -> Tensor:
    """Knot slopes of the C2 cubic spline through (x, y) with end slopes `start`, `end`."""
    n = x.numel()
    if n == 2:
        return torch.stack([start, end])
    inv = 1.0 / (x[1:] - x[:-1])
    secant = (y[1:] - y[:-1]) * inv
    A = (torch.diag(2.0 * (inv[:-1] + inv[1:])) + torch.diag(inv[1:-1], 1)
         + torch.diag(inv[1:-1], -1))
    b = 3.0 * (secant[:-1] * inv[:-1] + secant[1:] * inv[1:])
    first = torch.zeros_like(b)
    first[0] = 1.0
    last = torch.zeros_like(b)
    last[-1] = 1.0
    b = b - first * inv[0] * start - last * inv[-1] * end
    return torch.cat([start.reshape(1), torch.linalg.solve(A, b), end.reshape(1)])


def _hermite(u: Tensor, xs: Tensor, ys: Tensor, m: Tensor) -> Tensor:
    """The cubic Hermite interpolant through (xs, ys) with slopes m, at u clamped to xs."""
    uc = u.clamp(xs[0].item(), xs[-1].item())
    j = torch.searchsorted(xs.detach(), uc.detach().reshape(-1), right=True)
    j = (j.reshape(uc.shape) - 1).clamp(0, xs.numel() - 2)
    h = xs[j + 1] - xs[j]
    t = (uc - xs[j]) / h
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * ys[j] + (t3 - 2 * t2 + t) * h * m[j]
            + (3 * t2 - 2 * t3) * ys[j + 1] + (t3 - t2) * h * m[j + 1])


def _hermite_residual(u, target, xs, ys, m):
    return _hermite(u, xs, ys, m) - target


class SplineCurve:
    """A measured curve y(x) fitted as ContamX fits it (see the module docstring).

    `x` strictly increasing, at least two points; `x` and `y` may be tensors carrying
    gradients. `curve(u)` evaluates it, `curve.inverse(v)` inverts an increasing one."""

    def __init__(self, x, y, *, name: str = "") -> None:
        x = torch.as_tensor(x, dtype=F64).reshape(-1)
        y = torch.as_tensor(y, dtype=F64).reshape(-1)
        label = f"SplineCurve {name!r}"
        if x.shape != y.shape or x.numel() < 2:
            raise ValueError(f"{label}: x and y must be two lists of equal length, at least "
                             f"2 points")
        if bool((x[1:].detach() <= x[:-1].detach()).any()):
            raise ValueError(f"{label}: x must be strictly increasing, got "
                             f"{x.detach().tolist()}")
        self.x, self.y, self.name = x, y, name
        self.start = (y[1] - y[0]) / (x[1] - x[0])
        self.end = (y[-1] - y[-2]) / (x[-1] - x[-2])
        self._xs, self._ys = x[1:-1], y[1:-1]
        self._m = (clamped_spline_slopes(self._xs, self._ys, self.start, self.end)
                   if self._xs.numel() >= 2 else None)

    def _straight(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """The first segment extended below its second point and the last one above its
        second-last: the whole curve when it has no interior cubic."""
        i0, i1 = (1, -2) if self.x.numel() > 2 else (0, -1)
        below = self.y[i0] + self.start * (u - self.x[i0])
        above = self.y[i1] + self.end * (u - self.x[i1])
        return below, above

    def __call__(self, u) -> Tensor:
        u = torch.as_tensor(u, dtype=self.x.dtype)
        below, above = self._straight(u)
        if self._m is None:
            pivot = self.x[1] if self.x.numel() == 3 else self.x[-1]
            return torch.where(u < pivot, below, above)
        inside = _hermite(u, self._xs, self._ys, self._m)
        return torch.where(u < self._xs[0], below,
                           torch.where(u > self._xs[-1], above, inside))

    def inverse(self, v) -> Tensor:
        """The u with curve(u) = v, for an increasing curve (every segment's slope > 0)."""
        v = torch.as_tensor(v, dtype=self.x.dtype)
        i0, i1 = (1, -2) if self.x.numel() > 2 else (0, -1)
        below = self.x[i0] + (v - self.y[i0]) / self.start
        above = self.x[i1] + (v - self.y[i1]) / self.end
        if self._m is None:
            pivot = self.y[1] if self.x.numel() == 3 else self.y[-1]
            return torch.where(v < pivot, below, above)
        xs, ys, m = self._xs, self._ys, self._m
        target = v.clamp(ys[0].item(), ys[-1].item())
        lo = torch.broadcast_to(xs[0].detach(), v.shape).clone()
        hi = torch.broadcast_to(xs[-1].detach(), v.shape).clone()
        inside = solve_monotone(_hermite_residual, lo, hi, target, xs, ys, m,
                                tol=1e-15, max_iter=200)
        return torch.where(v < ys[0], below, torch.where(v > ys[-1], above, inside))


def _fit_is_increasing(curve: SplineCurve) -> bool:
    """True when the fitted curve has a positive derivative everywhere: the end slopes, the
    knot slopes of the cubic part, and the minimum of each Hermite piece's quadratic
    derivative."""
    if curve._m is None:
        return True
    xs, ys, m = (t.detach() for t in (curve._xs, curve._ys, curve._m))
    if bool((m <= 0).any()):
        return False
    h = xs[1:] - xs[:-1]
    s = (ys[1:] - ys[:-1]) / h
    a, b = m[:-1], m[1:]
    c, d = 3 * s - 2 * a - b, a + b - 2 * s     # dp/dx = a + 2 c t + 3 d t^2 on t in [0, 1]
    safe = torch.where(d.abs() > 0, d, torch.ones_like(d))
    t = torch.where(d.abs() > 0, -c / (3 * safe), torch.zeros_like(d))
    inside = (t > 0) & (t < 1)
    low = a + 2 * c * t + 3 * d * t * t
    return not bool((inside & (low <= 0)).any())


class CubicSplineFlow(ConstitutiveLaw):
    """An airflow element given by measured points, fitted as ContamX fits them
    (`SplineCurve`): CONTAM's cubic-spline elements.

    independent="pressure": the flow is the curve at dp (Pa);
    independent="flow":     dp is the curve at the flow, inverted for the flow.
    `volume=True`: the curve's flow is a volume flow (m3/s), made a mass flow (kg/s) with
    the density of the air entering the edge, `drivers[rho_key]` at `src` (dp >= 0) or `tgt`.
    `multiplier` (one per edge, or one for all) multiplies the flow, like CONTAM's path
    mult. ContamX requires at least four points and a positive slope on every segment, and
    so does this element. This element also refuses a curve whose fitted spline is not
    increasing everywhere (the cubic part can overshoot between points even when every
    segment rises): ContamX may accept such a curve, but the inversion of the `independent=
    "flow"` forms would then be ambiguous. The result is float64 whatever the input dtype."""

    def __init__(self, x, y, *, independent: str = "pressure", volume: bool = False,
                 multiplier=1.0, src, tgt, rho_key: str = "rho",
                 kind: str = "airpath") -> None:
        super().__init__(kind)
        label = f"CubicSplineFlow (kind {kind!r})"
        if independent not in ("pressure", "flow"):
            raise ValueError(f"{label}: independent must be 'pressure' or 'flow', got "
                             f"{independent!r}")
        x_t = torch.as_tensor(x, dtype=F64).reshape(-1)
        y_t = torch.as_tensor(y, dtype=F64).reshape(-1)
        if x_t.numel() < 4:
            raise ValueError(f"{label}: needs at least four points (as ContamX does), got "
                             f"{x_t.numel()}")
        if bool((y_t[1:].detach() <= y_t[:-1].detach()).any()):
            raise ValueError(f"{label}: every segment's slope must be positive (ContamX "
                             f"refuses this curve), got {y_t.detach().tolist()}")
        self.curve = SplineCurve(x_t, y_t, name=kind)
        if not _fit_is_increasing(self.curve):
            raise ValueError(f"{label}: the fitted curve is not increasing (its cubic part "
                             f"overshoots between points); the inversion of a pressure-flow "
                             f"form would be ambiguous, so this curve is refused, though "
                             f"ContamX may accept it")
        self.independent, self.volume = independent, bool(volume)
        self.register_buffer("multiplier", torch.as_tensor(multiplier, dtype=F64))
        self.register_buffer("src", torch.as_tensor(src).to(torch.long).reshape(-1))
        self.register_buffer("tgt", torch.as_tensor(tgt).to(torch.long).reshape(-1))
        self.rho_key = str(rho_key)

    def flow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        q = self.curve(dp) if self.independent == "pressure" else self.curve.inverse(dp)
        if self.volume:
            if drivers is None or self.rho_key not in drivers:
                raise KeyError(f"CubicSplineFlow (kind {self.kind!r}): driver "
                               f"{self.rho_key!r} not found; a volume-flow curve needs it")
            rho = torch.as_tensor(drivers[self.rho_key]).to(dp.dtype)
            q = q * torch.where(dp >= 0, rho[..., self.src], rho[..., self.tgt])
        return self.multiplier.to(dp.dtype) * q

    def linear_init(self, drivers: Mapping[str, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        zero = torch.zeros(self.src.numel(), dtype=F64)
        return self.flow(zero, drivers), self.dflow(zero, drivers)
