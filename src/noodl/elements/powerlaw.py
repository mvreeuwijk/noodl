"""Power-law branch element: q = C sign(dp) |dp|^n away from dp = 0.

Two ways keep the law smooth and Newton-friendly near dp = 0:

* laminar blend (default): q = k * dp for |dp| < dp_transition, k = C * dp_transition^(n-1)
  chosen so the law is continuous in value at the transition; the sharp law elsewhere. This
  is CONTAM's blend below a transition Reynolds number.
* regularised (``regularised=eps``): q = C dp (dp^2 + eps^2)^((n-1)/2), smooth everywhere,
  approaching the sharp law as dp/eps grows.

Both variants use ``torch.where`` between two fully-computed branches. In the blend, the
*unselected* "sharp" branch never evaluates |dp|^n at dp = 0 itself: ``dp_safe`` below
substitutes the constant dp_transition for |dp| wherever the laminar branch is selected.
Without this, ``torch.where``'s backward would multiply an infinite gradient (from
|dp|^(n-1) at dp = 0 when n < 1) by a zero mask and produce ``nan`` instead of zero — this is
what keeps ``flow`` gradcheck-clean at dp = 0 itself (and at dp = +-dp_transition, the
boundary where the branch selection flips).

Shape convention: C and n follow ``Element._param``'s pass-through rule (no implicit
reshaping). A single-edge instance of this element should be constructed with C, n already
carrying trailing shape ``(..., 1)`` (e.g. ``torch.tensor([2.0])`` rather than a bare Python
float, when the caller cares about broadcasting against a batched ``(..., b_kind)`` potential
difference tensor for several edges of the same kind); ordinary broadcasting then produces a
``(..., b_kind)`` result. A bare scalar (ndim 0) also broadcasts correctly against any dp
shape, so this convention is about a consistent contract for later composition (Task 4/7),
not a hard requirement enforced here.
"""

from __future__ import annotations

import math

import torch

from noodl.elements.base import Element

Tensor = torch.Tensor


class PowerLaw(Element):
    """q = C sign(dp) |dp|^n, laminar-blended or smoothly regularised near dp = 0."""

    def __init__(
        self,
        C,
        n,
        *,
        dp_transition: float = 1e-3,
        regularised: float | None = None,
        kind: str = "airpath",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        self.C = self._param(C, learnable)
        self.n = self._param(n, learnable)
        self.dp_transition = float(dp_transition)
        self.regularised = None if regularised is None else float(regularised)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        C, n = self.C, self.n
        if self.regularised is not None:
            eps = self.regularised
            return C * dp * (dp**2 + eps**2) ** ((n - 1) / 2)
        dpt = self.dp_transition
        k = C * dpt ** (n - 1)
        mask = dp.abs() < dpt
        # dp_safe never lets the unselected "sharp" branch see |dp| == 0: torch.where
        # evaluates both branches, and |dp|**(n-1) at dp == 0 is inf for n < 1; multiplying
        # that inf by the zero mask in backward would yield nan instead of the correct zero.
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        sharp = C * torch.sign(dp) * dp_safe**n
        laminar = k * dp
        return torch.where(mask, laminar, sharp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        C, n = self.C, self.n
        if self.regularised is not None:
            # dp**2 + eps**2 >= eps**2 > 0 everywhere (eps is a fixed positive float), so
            # this branch never raises a non-positive base to a non-integer power: smooth
            # and safe at dp == 0 with no dp_safe substitution needed.
            eps = self.regularised
            return C * (dp**2 + eps**2) ** ((n - 3) / 2) * (dp**2 + eps**2 + (n - 1) * dp**2)
        dpt = self.dp_transition
        k = C * dpt ** (n - 1)
        mask = dp.abs() < dpt
        # Same dp_safe trap as flow(): torch.where evaluates the "outside" branch even where
        # it is not selected (dp == 0 is always inside mask), and its own backward needs
        # d/ddp[dp.abs()**(n-1)] = (n-1)*dp.abs()**(n-2)*sign(dp), which is inf at dp == 0 for
        # n < 2. Substituting the constant dpt there keeps that derivative finite (in fact
        # exactly 0, since dp_safe is disconnected from dp under the mask) instead of
        # producing inf * 0 = nan when a caller differentiates dflow's output again.
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        outside = n * C * dp_safe ** (n - 1)
        return torch.where(mask, k * torch.ones_like(dp), outside)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        C, n = self.C, self.n
        eps_or_dpt = self.regularised if self.regularised is not None else self.dp_transition
        k = C * eps_or_dpt ** (n - 1)
        return torch.zeros_like(k), k


def Orifice(
    Cd,
    A,
    *,
    rho: float = 1.2,
    dp_transition: float = 1e-3,
    regularised: float | None = None,
    kind: str = "airpath",
    learnable: bool = False,
) -> PowerLaw:
    """PowerLaw(C = Cd * A * sqrt(2 / rho), n = 0.5): the sharp-edged orifice equation.

    N2: `Cd`/`A` given as a `torch.Tensor` keep THEIR OWN dtype -- a caller building a
    float64 network is not silently downcast to `torch.get_default_dtype()` (float32 in
    this project). A bare Python float still takes `torch.get_default_dtype()`, unchanged
    from before. `n` is built at the resulting `C`'s own dtype so it is never the odd one
    out (mirrors `apps.building.elements.mass_orifice`, which is explicit-float64 rather
    than default-dtype and so never had this bug).
    """
    Cd_t = Cd if isinstance(Cd, torch.Tensor) else torch.as_tensor(
        Cd, dtype=torch.get_default_dtype()
    )
    A_t = A if isinstance(A, torch.Tensor) else torch.as_tensor(
        A, dtype=torch.get_default_dtype()
    )
    C = Cd_t * A_t * math.sqrt(2.0 / rho)
    n = torch.as_tensor(0.5, dtype=C.dtype)
    return PowerLaw(
        C=C,
        n=n,
        dp_transition=dp_transition,
        regularised=regularised,
        kind=kind,
        learnable=learnable,
    )
