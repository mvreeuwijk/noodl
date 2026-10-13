"""Fan-curve branch element: dp = -P(q), P a cubic pressure-flow curve, inverted by
solve_monotone.

P(q) = a0 + a1 q + a2 q^2 + a3 q^3 is the fan's pressure rise, strictly decreasing on
0 <= q <= q_max (shutoff pressure P(0) at q = 0, lowest pressure P(q_max) at maximum flow).
The branch law is dp = -P(q): a positive dp opposes the fan's rise, so q solves
P(q) + dp = 0. Two limits clip the solve instead of calling it: q = 0 if the back-pressure
-dp exceeds the shutoff pressure P(0); q = q_max if -dp is below P(q_max) (more flow asked
for than the curve provides at zero added resistance). `flow`'s bracket clamps the *target*
-dp into [P(q_max), P(0)] before calling `solve_monotone`, so the sub-problem always has a
genuine sign change between q = 0 and q_max; the clip masks (from the true, unclamped -dp)
are applied afterwards. This keeps the same-sign check from firing on points meant to clip.

This clamping keeps ``solve_monotone`` itself safe: because it only ever sees a clamped,
well-bracketed sub-problem (never the raw, possibly out-of-range ``dp``), its forward and
backward never produce nan/inf for any batch index, clipped or not, and `flow`'s two
``torch.where`` masks over its output cannot propagate a poisoned value from a discarded
branch.

That guarantee does NOT extend to ``dflow``, which is a separate, direct computation
(``-1 / P'(q)``), not a call into ``solve_monotone``. In the clipped regions, `flow` returns
q = 0 (shut) or q = q_max (stalled) verbatim -- exactly the two points at which a physical
fan/pump curve is allowed to flatten out (dP/dq -> 0 at its shutoff point or at the top of
its curve). ``torch.where`` evaluates both of its branches, so even though the clipped
output is masked away, the *discarded* smooth branch of `dflow` still evaluates
``-1 / P'(q)`` at that q; if ``P'(q) == 0`` there, that branch is +-inf, and its backward
contribution to ``coeffs``/``q_max`` (parameters shared with the selected branch) is
``0 * inf = nan`` instead of the intended zero, poisoning those shared parameters even though
the clipped output value itself is correct. `dflow` below guards against this the same way
``PowerLaw.dflow`` guards its own boundary: by substituting a safe interior q for the
discarded branch's *input* before taking the reciprocal, not just masking the output.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl._broadcast import broadcast_shapes
from noodl.elements.base import Element
from noodl.solvers.scalar import solve_monotone

Tensor = torch.Tensor


def _fan_residual(q: Tensor, dp: Tensor, coeffs: Tensor) -> Tensor:
    c0, c1, c2, c3 = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2], coeffs[..., 3]
    return c0 + c1 * q + c2 * q**2 + c3 * q**3 + dp


class FanCurve(Element):
    """Cubic pressure-flow curve inverted for q given dp on [0, q_max]: `coeffs` are
    (a0, a1, a2, a3) of P(q) = a0 + a1 q + a2 q^2 + a3 q^3 in Pa for q in the flow unit, and
    `q_max` is the largest flow the curve covers, in that unit.
    """

    def __init__(self, coeffs, q_max, *, kind: str = "airpath", learnable: bool = False) -> None:
        super().__init__(kind)
        self.coeffs = self._param(coeffs, learnable)
        self.q_max = self._param(q_max, learnable)

    def _pressure(self, q: Tensor) -> Tensor:
        c0, c1, c2, c3 = (self.coeffs[..., i] for i in range(4))
        return c0 + c1 * q + c2 * q**2 + c3 * q**3

    def _dpressure(self, q: Tensor) -> Tensor:
        _, c1, c2, c3 = (self.coeffs[..., i] for i in range(4))
        return c1 + 2 * c2 * q + 3 * c3 * q**2

    def _bracket(self, dp: Tensor):
        p_shutoff = self.coeffs[..., 0]
        p_max_flow = self._pressure(self.q_max)
        shape = broadcast_shapes(dp.shape, p_shutoff.shape, p_max_flow.shape)
        dp_b = torch.broadcast_to(dp, shape)
        p_shutoff_b = torch.broadcast_to(p_shutoff, shape)
        p_max_flow_b = torch.broadcast_to(p_max_flow, shape)

        shut = (-dp_b) > p_shutoff_b
        stalled = (-dp_b) < p_max_flow_b
        dp_solve = -torch.clamp(-dp_b, min=p_max_flow_b, max=p_shutoff_b)

        lo = torch.zeros(shape, dtype=dp.dtype)
        hi = torch.broadcast_to(self.q_max, shape).clone()
        return shape, dp_solve, lo, hi, shut, stalled

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        shape, dp_solve, lo, hi, shut, stalled = self._bracket(dp)
        q = solve_monotone(_fan_residual, lo, hi, dp_solve, self.coeffs, tol=1e-12, max_iter=100)
        q = torch.where(shut, torch.zeros_like(q), q)
        q = torch.where(stalled, torch.broadcast_to(self.q_max, shape), q)
        return q

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        shape, _dp_solve, _lo, _hi, shut, stalled = self._bracket(dp)
        clipped = shut | stalled
        q = self.flow(dp, drivers)
        # q_safe never lets the discarded "smooth" branch see q == 0 or q == q_max, the two
        # points at which -1 / P'(q) can be +-inf for a curve that flattens at its shutoff or
        # top-of-curve boundary (see the module docstring). The bracket midpoint is a fixed,
        # always-in-domain stand-in; it costs nothing where `clipped` is False since it is
        # discarded there, and where `clipped` is True it keeps `_dpressure` finite so
        # `torch.where`'s backward multiplies a finite value by the zero mask (giving exactly
        # 0) instead of inf by zero (giving nan) into the shared coeffs/q_max parameters.
        midpoint = 0.5 * torch.broadcast_to(self.q_max, shape)
        q_safe = torch.where(clipped, midpoint, q)
        raw = -1.0 / self._dpressure(q_safe)
        return torch.where(clipped, torch.zeros_like(raw), raw)


class AffinityFan(Element):
    """A fan curve run at another speed and air density, by the fan laws (ContamX's
    performance-curve fan, measured against ContamX 3.4.1.7):

        F = multiplier * s * r * F0(dp / (s^2 r)),    r = rho_up / rho_ref

    `curve` is the fan's `FanCurve` at its rating density `rho_ref` (kg/m3) and rated
    speed, F0 its inverted curve. The speed ratio s is `drivers[speed_key]`, one value per
    edge (1 when `speed_key` is None); s > 1 is allowed. Below `off` the fan is stopped and
    the air passes `shutoff` instead (CONTAM's shut-off orifice: the curve element's own
    power law), times `multiplier`; with no `shutoff` a stopped fan passes nothing. rho_up
    is `drivers[rho_key]` at the edge's source, since a fan pushes from source to target.
    `multiplier` (one per edge, or one for all) is fans in parallel: CONTAM's path mult.

    The flow jumps at s = `off` (and at s = 0), from the running fan to the stopped one, as
    in ContamX; the slope and the gradients are finite on either side of the jump. A
    negative speed ratio is refused.
    """

    def __init__(self, curve: FanCurve, *, src, tgt, rho_ref: float,
                 speed_key: str | None = None, off: float = 0.0, multiplier=1.0,
                 shutoff: Element | None = None, rho_key: str = "rho",
                 kind: str | None = None) -> None:
        super().__init__(curve.kind if kind is None else kind)
        if shutoff is not None and shutoff.kind != self.kind:
            raise ValueError(f"AffinityFan (kind {self.kind!r}): shutoff has kind "
                             f"{shutoff.kind!r}; it must be the fan's own kind")
        if float(rho_ref) <= 0.0:
            raise ValueError(f"AffinityFan (kind {self.kind!r}): rho_ref must be positive, "
                             f"got {rho_ref!r}")
        self.curve = curve
        self.shutoff = shutoff
        self.register_buffer("src", torch.as_tensor(src).to(torch.long).reshape(-1))
        self.register_buffer("tgt", torch.as_tensor(tgt).to(torch.long).reshape(-1))
        self.register_buffer("multiplier", torch.as_tensor(multiplier, dtype=torch.float64))
        self.rho_ref = float(rho_ref)
        self.speed_key = speed_key
        self.off = float(off)
        self.rho_key = str(rho_key)

    def _state(self, dp: Tensor, drivers: Mapping[str, Tensor] | None):
        name = f"AffinityFan (kind {self.kind!r})"
        if drivers is None or self.rho_key not in drivers:
            raise KeyError(f"{name}: driver {self.rho_key!r} not found")
        r = torch.as_tensor(drivers[self.rho_key]).to(dp.dtype)[..., self.src] / self.rho_ref
        if self.speed_key is None:
            s = torch.ones_like(r)
        elif self.speed_key not in drivers:
            raise KeyError(f"{name}: driver {self.speed_key!r} not found; it carries the "
                           f"fan's speed ratio")
        else:
            s = torch.as_tensor(drivers[self.speed_key]).to(dp.dtype)
            negative = s < 0
            if bool(negative.any()):
                raise ValueError(
                    f"{name}: driver {self.speed_key!r} gives a negative speed ratio "
                    f"{s.detach()[negative].flatten()[0].item()!r}; a fan's speed ratio is 0 "
                    f"(stopped) or more")
        # A speed of exactly 0 is stopped whatever `off` is (the fan laws divide by s), and
        # stopped edges evaluate the curve at s = 1, so the discarded branch stays finite.
        running = (s >= self.off) & (s > 0)
        s = torch.where(running, s, torch.ones_like(s))
        return r, s, self.multiplier.to(dp.dtype), running

    def flow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        r, s, mult, running = self._state(dp, drivers)
        on = mult * s * r * self.curve.flow(dp / (s * s * r), drivers)
        off = (mult * self.shutoff.flow(dp, drivers) if self.shutoff is not None
               else torch.zeros_like(on))
        return torch.where(running, on, off)

    def dflow(self, dp: Tensor, drivers: Mapping[str, Tensor] | None = None) -> Tensor:
        r, s, mult, running = self._state(dp, drivers)
        on = mult * self.curve.dflow(dp / (s * s * r), drivers) / s
        off = (mult * self.shutoff.dflow(dp, drivers) if self.shutoff is not None
               else torch.zeros_like(on))
        return torch.where(running, on, off)

    def linear_init(self, drivers: Mapping[str, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        zero = torch.zeros(self.src.numel(), dtype=torch.float64)
        return self.flow(zero, drivers), self.dflow(zero, drivers)
