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

This clamping also forecloses the ``torch.where``-poisons-gradients trap: because
``solve_monotone`` only ever sees a clamped, well-bracketed sub-problem (never the raw,
possibly out-of-range ``dp``), its forward and backward never produce nan/inf for any batch
index, clipped or not, so masking the clipped results afterward with ``torch.where`` cannot
propagate a poisoned value from a discarded branch.
"""

from __future__ import annotations

import torch

from tellegen.elements.base import Element
from tellegen.solvers.scalar import solve_monotone

Tensor = torch.Tensor


def _fan_residual(q: Tensor, dp: Tensor, coeffs: Tensor) -> Tensor:
    c0, c1, c2, c3 = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2], coeffs[..., 3]
    return c0 + c1 * q + c2 * q**2 + c3 * q**3 + dp


class FanCurve(Element):
    """Cubic pressure-flow curve inverted for q given dp on [0, q_max]."""

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
        shape = torch.broadcast_shapes(dp.shape, p_shutoff.shape, p_max_flow.shape)
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
        _shape, _dp_solve, _lo, _hi, shut, stalled = self._bracket(dp)
        q = self.flow(dp, drivers)
        raw = -1.0 / self._dpressure(q)
        return torch.where(shut | stalled, torch.zeros_like(raw), raw)
