"""Branch elements for a pressurised water distribution network (EPANET 2.2 forms).

Potential is hydraulic HEAD in metres; flow is m^3/s, signed along the edge's own
orientation. All constants are SI.
"""

from __future__ import annotations

import functools

import torch

from noodl._unionfind import UnionFind
from noodl.apps.water.controls import ACTIVE, CLOSED, OPEN
from noodl.elements.base import ConstitutiveLaw, ImplicitLaw
from noodl.solvers.scalar import solve_monotone

Tensor = torch.Tensor
F64 = torch.float64

#: SI Hazen-Williams resistance constant, ``h = K C^-1.852 d^-4.871 L q^1.852``.
#: Re-derived from the EPANET 2.2 Manual Table 3.1 US value 4.727 (feet, cfs) and equal to
#: wntr's own ``hw_k`` to 1.04e-9 relative. VERIFIED.
HW_SI = 10.666829500036

#: SI Darcy-Weisbach resistance constant ``h = K f d^-5 L q^2`` = ``8 / (g pi^2)``.
#: Confirmed against wntr's ``dw_k``; NOT re-derived from the manual's US form.
DW_SI = 0.0826

#: Standard gravity (m/s^2), used by the minor-loss and pump forms.
G = 9.80665

#: The per-link drivers of a water layer, each one value per link in `model.link_names`
#: order (pipes, pumps, then valves by kind). `"water.status"`: `CLOSED`, `OPEN` or `ACTIVE`
#: (`noodl.apps.water.controls`); `"water.setting"`: a pump's relative speed, a PBV's head
#: drop (the layer's potential unit), an FCV's flow (m3/s), a TCV's loss coefficient;
#: `"water.one_way"`: 0 both directions, +1 forward flow only, -1 reverse flow only.
STATUS_KEY = "water.status"
SETTING_KEY = "water.setting"
ONE_WAY_KEY = "water.one_way"

_DRIVER_MEANING = {
    STATUS_KEY: "each link's status (CLOSED 0, OPEN 1, ACTIVE 2)",
    SETTING_KEY: "each link's setting",
    ONE_WAY_KEY: "each link's allowed flow direction (0, +1, -1)",
}


def _driver(law, drivers, key: str) -> Tensor:
    """The per-link driver `key`, whole; refuses a missing one by the law's name."""
    if drivers is None or key not in drivers:
        raise KeyError(
            f"{type(law).__name__} (kind {law.kind!r}): driver {key!r} is required and was "
            f"not given; it carries {_DRIVER_MEANING[key]}, one value per link"
        )
    return drivers[key]


def link_driver(law, drivers, key: str, like: Tensor) -> Tensor:
    """The per-link driver `key` at `law.link_range`, the law's own columns of it, in the dtype of
    `like`. Refuses a missing driver by the law's name."""
    return _driver(law, drivers, key)[..., slice(*law.link_range)].to(like.dtype)


def link_flags(law, drivers, keys: tuple[str, ...], like: Tensor, summarise):
    """`summarise(*columns)` of the law's own columns of the drivers `keys` (in the dtype of
    `like`), remembered on the law for as long as those drivers are the same tensors,
    unchanged in place (their `_version`; an inference-mode tensor counts no versions and
    is not remembered), in the last few sets seen. A solve hands every
    evaluation the same drivers (as two views: one on the residual's path, one on the
    Jacobian's), so the per-link tests a law makes before it reads them (is any pipe
    closed? one-way? any pump off? any valve not active?) run once per solve, not once per
    residual, and a network with nothing closed, one-way or inactive pays nothing for them.
    The cache lives in the law's `_memo_cache`, which the layer's scan for held tensors
    skips, and keeps the tensors it saw, so their identities are not reused."""
    tensors = tuple(_driver(law, drivers, key) for key in keys)
    cols = slice(*law.link_range)
    try:
        versions = tuple(t._version for t in tensors)
    except RuntimeError:   # an inference-mode tensor has no version counter: not remembered
        return summarise(*(t[..., cols].to(like.dtype) for t in tensors))
    stamp = (*(id(t) for t in tensors), like.dtype)
    cache = law.__dict__.setdefault("_memo_cache", {}).setdefault("link_flags", {})
    hit = cache.get(stamp)
    if (hit is not None and hit[1] == versions
            and all(t is s for t, s in zip(tensors, hit[0], strict=True))):
        return hit[2]
    value = summarise(*(t[..., cols].to(like.dtype) for t in tensors))
    if len(cache) >= 4:
        del cache[next(iter(cache))]
    cache[stamp] = (tensors, versions, value)
    return value


#: The per-link driver `water_steady` sets on one CLOSED link per junction group that closed
#: links cut off from every reservoir and tank with no demand: 1 there, 0 elsewhere. Such a
#: group has no head of its own (EPANET gives it one through its closed links' `1 / CBIG`
#: conductance); the marked link's row is then `dp = 0` instead of `kappa q = 0`, which ties
#: the group to the far end of that link with every flow still exactly zero, and
#: `water_steady` then sets the group's head to EPANET's limit (`_cut_off_plan`). Absent: no
#: link is marked.
PIN_KEY = "water.closed_tie"


def closed_row(law, dp: Tensor, q: Tensor, drivers) -> Tensor:
    """The row of a CLOSED link: ``kappa q``, or ``dp`` where `PIN_KEY` marks the link."""
    held = law.kappa * q
    pin = None if drivers is None else drivers.get(PIN_KEY)
    if pin is None or law.link_range is None:
        return held
    pin = pin[..., slice(*law.link_range)]
    if not bool(pin.any()):
        return held
    return torch.where(pin != 0, dp, held)


@functools.lru_cache(maxsize=256)
def _joined_groups(ends: tuple, fixed: tuple, joined: tuple) -> tuple[dict, frozenset]:
    """`(group, fed)` for the links `joined` (per link, of `ends`): each node's group and
    the groups holding a node of `fixed`. Remembered: a solve meets the same few sets of
    closed check valves on iterate after iterate."""
    groups: UnionFind[int] = UnionFind()
    for (u, v), is_joined in zip(ends, joined, strict=True):
        groups.add(u)
        groups.add(v)
        if is_joined:
            groups.union(u, v)
    for n in fixed:
        groups.add(n)
    group = {n: groups.find(n) for n in groups.parent}
    return group, frozenset(group[n] for n in fixed)


def _link_range(links: slice | None) -> tuple[int, int] | None:
    """`links` as `(start, stop)`, the form a law keeps (a slice is not walked by the layer's
    scan for held tensors)."""
    return None if links is None else (int(links.start), int(links.stop))


def open_valve_loss(q: Tensor, minor: Tensor, gradient_floor: float) -> Tensor:
    """The head loss of an OPEN valve, ``minor q |q|``: EPANET 2.2's `valvecoeff`
    (`hydcoeffs.c`), the valve's minor loss with its gradient ``2 minor |q|`` floored at
    ``gradient_floor`` (EPANET's ``RQtol``). Below the floor the loss is the line
    ``(gradient_floor / 2) q``, continuous with the quadratic where they meet (EPANET's own
    line there, ``RQtol q``, jumps by a factor 2; the same choice as
    `HazenWilliams.head_loss`). A valve without a minor loss is that line alone."""
    a = q.abs()
    return torch.where(2.0 * minor * a < gradient_floor, 0.5 * gradient_floor * q,
                       minor * q * a)


class HazenWilliams(ConstitutiveLaw):
    """``h_L = K q^1.852 + m q^2`` with ``K = 10.666829500036 C^-1.852 d^-4.871 L``.

    ``m = K_minor / (2 g A^2)`` is the optional minor-loss term of the same pipe (EPANET
    2.2 Manual p.19, ``h_m = K v^2 / 2g``); with ``m == 0`` (the usual case, and the only
    one in both committed fixtures) the inversion is the closed form
    ``q = sign(dp) (|dp| / K)^(1/1.852)``, laminar-blended below ``dp_transition`` exactly
    as ``PowerLaw`` does. With any nonzero ``m`` the law is inverted by a batched monotone
    root on ``[0, (|dp|/K)^(1/1.852)]``; that bracket is justified: ``h_L(q)`` is strictly
    increasing for ``q >= 0`` and the minor-loss term is non-negative, so the pure
    Hazen-Williams flow is an upper bound on the true one.

    ``dp_transition`` defaults to ``1e-9`` m. MEASURED, and NOT for the reason one would
    guess: on the committed two-loop fixture the choice is irrelevant -- the smallest head
    loss there is 5.8188e-3 m (pipe P7, the near-stagnant loop-closer), above every
    candidate, and the whole `test_two_loop_heads_and_flows` comparison is bit-identical
    at 1e-3, 1e-6, 1e-9 and 1e-12.
    Where it bites is a pipe carrying EXACTLY zero flow, which Net1 produces the moment a
    control closes its pump and leaves pipe 10 dead-ended: there ``dp`` is exactly 0, the
    blend is the only thing keeping the Jacobian finite, and its tangent slope
    ``(dp_transition/K)^(1/1.852) / dp_transition`` grows as the transition shrinks. Over
    Net1's 24 h duty cycle the worst Newton iteration count runs 50, 93, 136, 179 at
    1e-3, 1e-6, 1e-9, 1e-12, while the Net1 tank trajectory
    (`test_net1_extended_period_tank_level`) CONVERGES with respect to the
    transition at 1e-6 and below (8.1813e-5 m against EPANET at 1e-6, 1e-9 and 1e-12;
    6.4922e-5 m at 1e-3, which is a coincidence of EPANET's own unquantified low-flow
    linearisation, not a better answer). 1e-9 is therefore the converged, faithful choice,
    and it is what fixes ``water_steady``'s ``max_iter`` at 200.
    """

    def __init__(
        self,
        length,
        diameter,
        roughness,
        *,
        minor_loss=0.0,
        dp_transition: float = 1e-9,
        kind: str = "pipe",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        self.length = self._param(length, False)
        self.diameter = self._param(diameter, False)
        self.roughness = self._param(roughness, learnable)
        self.minor_loss = self._param(minor_loss, False)
        self.dp_transition = float(dp_transition)
        self._has_minor = bool(torch.any(torch.as_tensor(minor_loss) != 0))

    def resistance(self) -> Tensor:
        """``K`` in ``h_L = K q^1.852`` (s^1.852 m^-4.556)."""
        return (
            HW_SI
            * self.roughness ** (-1.852)
            * self.diameter ** (-4.871)
            * self.length
        )

    def _minor(self) -> Tensor:
        """``m`` in ``h_m = m q^2``: ``K_minor / (2 g A^2)``."""
        area = torch.pi * self.diameter**2 / 4.0
        return self.minor_loss / (2.0 * G * area**2)

    def head_loss(self, q: Tensor, gradient_floor: float = 0.0) -> Tensor:
        """Signed head loss (m) at signed flow ``q`` (m^3/s): ``sgn(q) (K |q|^1.852 + m q^2)``.

        Where the friction gradient ``1.852 K |q|^0.852`` is below ``gradient_floor`` (m per
        m^3/s) the friction loss is linear instead, as in EPANET's low-flow linearisation
        (`hydcoeffs.c`, `pipecoeff`, ``hgrad < RQtol``), which keeps the gradient positive at
        zero flow. EPANET's line, ``RQtol |q|``, is 1.852 times the formula's loss where they
        meet, a jump on which Newton can cycle; this one is ``(gradient_floor / 1.852) |q|``,
        continuous there. The two differ only below that flow (about 1e-12 m^3/s on Net1)."""
        k = self.resistance()
        a = q.abs()
        live = a > 0
        a_safe = torch.where(live, a, torch.ones_like(a))   # no 0^(negative) in either branch
        friction = torch.where(live, k * a_safe**1.852, torch.zeros_like(a))
        gradient = torch.where(live, 1.852 * k * a_safe**0.852, torch.zeros_like(a))
        # The line is written in `q`, not `sign(q) |q|`, whose derivative is zero at `q = 0`
        # exactly: a closed check valve holds every pipe in series with it there.
        loss = torch.where(gradient < gradient_floor, gradient_floor / 1.852 * q,
                           torch.sign(q) * friction)
        if not self._has_minor:   # (adding an exact zero would change nothing but the cost)
            return loss
        return loss + torch.sign(q) * self._minor() * a * a

    def head_gradient(self, q: Tensor, gradient_floor: float = 0.0, cols=None) -> Tensor:
        """``d head_loss / d q`` (m per m^3/s) at ``q``, with the same ``gradient_floor``;
        of the pipes ``cols`` only (``q`` then theirs) when given."""
        k = self.resistance()
        minor = self._minor() * torch.ones_like(k)
        if cols is not None:
            k, minor = k[..., cols], minor[..., cols]
        a = q.abs()
        a_safe = torch.where(a > 0, a, torch.ones_like(a))
        gradient = torch.where(a > 0, 1.852 * k * a_safe**0.852, torch.zeros_like(a))
        friction = torch.where(gradient < gradient_floor,
                               torch.full_like(gradient, gradient_floor / 1.852), gradient)
        return friction + 2.0 * minor * a

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        k = self.resistance()
        dpt = self.dp_transition
        mask = dp.abs() < dpt
        # The un-taken branch must never see |dp| == 0: |dp|^(1/1.852 - 1) is infinite
        # there, and where()'s backward would turn inf * 0 into nan. Substituting the
        # constant transition value on the masked entries keeps both branches finite.
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        if not self._has_minor:
            sharp = (dp_safe / k) ** (1.0 / 1.852)
            slope = (dpt / k) ** (1.0 / 1.852) / dpt
            return torch.where(mask, slope * dp, torch.sign(dp) * sharp)
        m = self._minor()
        hi = (dp_safe / k) ** (1.0 / 1.852)

        def residual(q, h_t, k_t, m_t):
            return k_t * q**1.852 + m_t * q**2 - h_t

        sharp = solve_monotone(
            residual,
            torch.zeros_like(hi),
            hi,
            dp_safe,
            k * torch.ones_like(hi),
            m * torch.ones_like(hi),
            tol=1e-15,
            max_iter=100,
        )
        hi_t = (torch.full_like(dp, dpt) / k) ** (1.0 / 1.852)
        q_t = solve_monotone(
            residual,
            torch.zeros_like(hi_t),
            hi_t,
            torch.full_like(dp, dpt),
            k * torch.ones_like(hi_t),
            m * torch.ones_like(hi_t),
            tol=1e-15,
            max_iter=100,
        )
        return torch.where(mask, q_t * dp / dpt, torch.sign(dp) * sharp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        k = self.resistance()
        dpt = self.dp_transition
        mask = dp.abs() < dpt
        q = self.flow(dp, drivers).abs()
        q_safe = torch.where(mask, torch.full_like(q, 1e-30), q)
        slope_sharp = 1.0 / (1.852 * k * q_safe**0.852 + 2.0 * self._minor() * q_safe)
        if not self._has_minor:
            slope_lam = (dpt / k) ** (1.0 / 1.852) / dpt
        else:
            slope_lam = self.flow(torch.full_like(dp, dpt), drivers) / dpt
        return torch.where(mask, slope_lam * torch.ones_like(dp), slope_sharp)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        k = self.resistance()
        dpt = self.dp_transition
        slope = (dpt / k) ** (1.0 / 1.852) / dpt
        zero = torch.zeros_like(slope)
        return zero, slope + zero


def three_point_curve(q_design: Tensor, h_design: Tensor) -> tuple[Tensor, Tensor, float]:
    """EPANET's single-point pump-curve fit: ``(h0, r, n)`` for ``h = h0 - r q^n``.

    EPANET 2.2 Manual p.19: "EPANET adds two more points to the curve by assuming a shutoff
    head at zero flow equal to 133% of the design head and a maximum flow at zero head equal
    to twice the design flow", then fits ``h = A - B q^C`` through the three points. Those
    three points force the exponent EXACTLY: ``B q_d^C = H/3`` and ``B (2 q_d)^C = 4H/3``
    give ``2^C = 4``, so ``C = 2`` and ``h0 = 4 H / 3``. (Confirmed against wntr's own
    `get_head_curve_coefficients` on Net1: (101.6, 2836.1385287628063, 2) for the file's
    1500 GPM / 250 ft point.)
    """
    h0 = 4.0 * h_design / 3.0
    r = (h_design / 3.0) / q_design**2
    return h0, r, 2.0


class PumpCurve(ConstitutiveLaw):
    """Constant-speed pump on its own edge kind: ``h_gain = w^2 (h0 - r (q/w)^n)``.

    The edge is oriented SUCTION -> DISCHARGE, so the layer's own
    ``dp = phi_suction - phi_discharge`` is the NEGATIVE of the head gain and the inversion
    is closed form:

        q = w ((h0 + dp / w^2) / r)^(1/n)      for  h0 + dp / w^2 > 0
        q = 0                                  otherwise (shut off)

    ``w`` is the relative speed setting (EPANET's affinity laws, Manual p.108). With
    ``links`` (the pumps' columns of the per-link drivers, a slice of `model.link_names`) the
    status is read from ``"water.status"``: a ``CLOSED`` pump delivers nothing. That is how
    ``[STATUS]`` and a ``[CONTROLS]`` line reach the element. The fitted curve is a strict
    power law, so no root solve is needed; a monotone root solve would only be needed for a
    MULTI-point (piecewise-linear) curve, which is out of scope.

    Flow is monotone increasing in ``dp`` (``dq/ddp > 0``), so the Newton Jacobian stays
    SPD-certifiable. At shut-off the slope of the sharp branch diverges, so the law is
    laminar-blended over ``h0 + dp/w^2 < dp_transition`` with a safe input substituted on
    BOTH branches of the ``where``. Beyond shut-off the slope is exactly zero, which is why
    `build_model` enters every pump through `FlowControlledPump` (the head-gain form,
    ``head_gain``) instead of this inversion; ``flow`` remains the pump's closed form.

    ``q_max`` is the flow at zero head gain,
    ``w (h0/r)^(1/n)`` -- EPANET's own curve is only DEFINED up to this point (its own
    "two more points" construction stops there), and beyond it a positive ``dp`` (the
    pump run backwards, discharge below suction) still has a well-defined closed-form
    ``flow``, silently EXTRAPOLATING the power law with no physical curve behind it. This
    element keeps that extrapolation -- refusing it here would abort Newton on a merely
    transient iterate, before the solve has had a chance to walk back inside the curve's
    domain -- and leaves the refusal to ``water_steady``, which checks only the CONVERGED
    flow against ``q_max`` and raises by name (no clamp, no warning, matching EPANET's own
    silent extrapolation being exactly what this application declines to reproduce
    uncontrolled).
    """

    def __init__(
        self,
        h0,
        r,
        n=2.0,
        *,
        speed=1.0,
        dp_transition: float = 1e-6,
        links: slice | None = None,
        kind: str = "pump",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        self.h0 = self._param(h0, learnable)
        self.r = self._param(r, learnable)
        self.n = self._param(n, False)
        self.speed = self._param(speed, False)
        self.dp_transition = float(dp_transition)
        self.link_range = _link_range(links)

    def _status(self, dp: Tensor, drivers) -> Tensor:
        """1 where the pump runs, 0 where its `"water.status"` is CLOSED."""
        if self.link_range is None:
            return torch.ones_like(dp)
        return (link_driver(self, drivers, STATUS_KEY, dp) != CLOSED).to(dp.dtype)

    def q_max(self) -> Tensor:
        """The flow at zero head gain, ``w (h0/r)^(1/n)``; see the class docstring."""
        return self.speed * (self.h0 / self.r) ** (1.0 / self.n)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        w = self.speed
        head = self.h0 + dp / w**2
        dpt = self.dp_transition
        shut = head < dpt
        head_safe = torch.where(shut, torch.full_like(head, dpt), head)
        sharp = w * (head_safe / self.r) ** (1.0 / self.n)
        # Below the transition the law is the straight line through (0, 0) and
        # (dp_transition, q(dp_transition)): continuous in value, finite in slope, and it
        # reaches exactly zero at the shut-off head instead of the infinite slope the sharp
        # branch has there.
        q_t = w * (torch.full_like(head, dpt) / self.r) ** (1.0 / self.n)
        laminar = q_t * torch.clamp(head, min=0.0) / dpt
        return self._status(dp, drivers) * torch.where(shut, laminar, sharp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        w = self.speed
        head = self.h0 + dp / w**2
        dpt = self.dp_transition
        shut = head < dpt
        head_safe = torch.where(shut, torch.full_like(head, dpt), head)
        sharp = (w / (self.n * self.r * w**2)) * (head_safe / self.r) ** (1.0 / self.n - 1.0)
        q_t = w * (torch.full_like(head, dpt) / self.r) ** (1.0 / self.n)
        # `/ dpt / w**2`, not `/ (dpt * w**2)`: with a float32 `speed` (a Python float given)
        # the product would round `dpt` to float32, 2.5e-9 off the flow's own slope.
        laminar = torch.where(head > 0.0, q_t / dpt / w**2, torch.zeros_like(head))
        return self._status(dp, drivers) * torch.where(shut, laminar, sharp)

    def head_gain(self, q: Tensor, gradient_floor: float, closed_slope: float) -> Tensor:
        """The head gain ``w^2 h0 - r w^(2-n) q^n`` at flow ``q``, for `FlowControlledPump`.

        The curve's slope ``n r w^(2-n) q^(n-1)`` vanishes at zero flow. Below the flow
        ``q_bar`` where it falls to ``gradient_floor`` the gain is the curve's TANGENT there
        instead (continuous in value and slope): wntr's own construction
        (`get_pump_line_params`) with EPANET's ``RQtol`` as the slope, so the gain never has
        a zero gradient.

        Below zero flow the pump is shut: a pump does not run backwards, and EPANET closes
        one whose head gain would exceed the shut-off head. The gain there rises with slope
        ``closed_slope``, EPANET's resistance of a CLOSED link (``CBIG``), so a pump held
        beyond shut-off converges to EPANET's closed-link leak of ``excess / CBIG`` (about
        3e-8 m^3/s for 50 m) rather than to a reverse flow. The gain is strictly decreasing
        in ``q`` everywhere, which makes the head-gain form well posed for every head.
        """
        w = self.speed
        coeff = self.r * w ** (2.0 - self.n)
        q_bar = (gradient_floor / (self.n * coeff)) ** (1.0 / (self.n - 1.0))
        on_curve = q > q_bar
        q_safe = torch.where(on_curve, q, q_bar.expand_as(q))   # no negative base below
        curve = w**2 * self.h0 - coeff * q_safe**self.n
        h_bar = w**2 * self.h0 - coeff * q_bar**self.n
        tangent = h_bar - gradient_floor * (q - q_bar)
        shut = h_bar + gradient_floor * q_bar - closed_slope * q
        return torch.where(on_curve, curve, torch.where(q < 0, shut, tangent))

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        # Tangent at dp = 0 (the pump running on its own curve at zero head difference).
        zero = torch.zeros_like(self.h0 + self.r)
        return self.flow(zero, drivers), self.dflow(zero, drivers)


class MinorLoss(ConstitutiveLaw):
    """``h = m q |q|`` with ``m = K / (2 g A^2)``: any pure minor-loss link, in its potential
    form. Inverted in closed form, ``q = sign(h) sqrt(|h| / m)``, with the same laminar blend
    and both-branch safe substitution as ``HazenWilliams``. (`build_model` writes a throttle
    control valve as `ThrottleValve`, with EPANET's own coefficient and its status.)"""

    def __init__(
        self,
        setting,
        diameter,
        *,
        dp_transition: float = 1e-9,
        kind: str = "valve",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        self.setting = self._param(setting, learnable)
        self.diameter = self._param(diameter, False)
        self.dp_transition = float(dp_transition)

    def coefficient(self) -> Tensor:
        area = torch.pi * self.diameter**2 / 4.0
        return self.setting / (2.0 * G * area**2)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        m = self.coefficient()
        dpt = self.dp_transition
        mask = dp.abs() < dpt
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        sharp = torch.sign(dp) * torch.sqrt(dp_safe / m)
        # `torch.as_tensor(dpt)` on a bare Python float casts with
        # `torch.get_default_dtype()` (float32 in this repo, `elements/duct.py`'s
        # documented gotcha), which would silently downcast a float64 element; `dtype=
        # dp.dtype` keeps it in the caller's own precision, as `Headspace` does.
        slope = torch.sqrt(torch.as_tensor(dpt, dtype=dp.dtype) / m) / dpt
        return torch.where(mask, slope * dp, sharp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        m = self.coefficient()
        dpt = self.dp_transition
        mask = dp.abs() < dpt
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        sharp = 0.5 / torch.sqrt(m * dp_safe)
        slope = torch.sqrt(torch.as_tensor(dpt, dtype=dp.dtype) / m) / dpt
        return torch.where(mask, slope * torch.ones_like(dp), sharp)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        m = self.coefficient()
        dpt = self.dp_transition
        slope = torch.sqrt(torch.as_tensor(dpt, dtype=m.dtype) / m) / dpt
        zero = torch.zeros_like(slope)
        return zero, slope + zero


# ------------------------------------------------------------- composite (EPANET) D-W
#: EPANET 2.2's own constants for the Darcy-Weisbach friction factor, copied digit for
#: digit from `src/hydcoeffs.c` (v2.2), where they are ``A1``, ``A2``, ``A8``, ``A9``,
#: ``AB`` and ``AC``. ``w = q / (nu d) = Re pi / 4``, so ``W_RE_4000`` (``A1``) is
#: Re = 4000 and ``W_RE_2000`` (``A2``) is Re = 2000.
#: Aliases: `EPANET_A1`, `EPANET_A2`, `EPANET_A8`, `EPANET_A9`, `EPANET_AB`, `EPANET_AC` (EPANET).
W_RE_4000 = 3.14159265358979323850e03  # 1000 pi
W_RE_2000 = 1.57079632679489661930e03  # 500 pi
SWAMEE_JAIN_W_COEFF = 4.61841319859066668690e00  # 5.74 (pi/4)^0.9
SWAMEE_JAIN_LOG_COEFF = -8.68588963806503655300e-01  # -2 / ln(10)
DUNLOP_AB = 3.28895476345399058690e-03  # 5.74 / 4000^0.9
DUNLOP_AC = -5.14214965799093883760e-03  # -2 * 0.9 * 2 / ln(10) * AB

#: EPANET computes in feet and cfs. ``MperFT`` (`types.h`) is exact; the gravity in the
#: D-W resistance ``R = L / (2 * 32.2 * d * A^2)`` (`hydcoeffs.c`, `resistcoeff`) is
#: 32.2 ft/s^2, NOT standard gravity's 32.174, and the minor-loss conversion
#: ``0.02517 K / d^4`` (`input1.c`, `convertunits`) is ``8 / (32.2 pi^2)`` ROUNDED to
#: four figures. Both are part of what EPANET computes, so both are reproduced here.
#: Aliases: `EPANET_FOOT`, `EPANET_G_FT`, `EPANET_KM` (EPANET).
M_PER_FT = 0.3048
G_FT_S2 = 32.2
MINOR_LOSS_K_FT = 0.02517
#: EPANET's default kinematic viscosity of water, ``VISCOS = 1.1e-5`` ft^2/s (`types.h`),
#: = 1.0219e-6 m^2/s; ``[OPTIONS] VISCOSITY`` multiplies it.
#: Aliases: `EPANET_VISCOS_FT2`, `EPANET_NU` (EPANET).
NU_WATER_FT2 = 1.1e-5
NU_WATER_REF = NU_WATER_FT2 * M_PER_FT**2
#: EPANET's own flow-unit factors, file units per cfs (`types.h`, ``GPMperCFS`` ...).
#: Several are rounded (``LPSperCFS = 28.317`` against the exact 28.3168466), and EPANET
#: divides a file's flows by exactly these numbers on the way in.
EPANET_QCF = {
    "CFS": 1.0,
    "GPM": 448.831,
    "MGD": 0.64632,
    "IMGD": 0.5382,
    "AFD": 1.9837,
    "LPS": 28.317,
    "LPM": 1699.0,
    "MLD": 2.4466,
    "CMH": 101.94,
    "CMD": 2446.6,
}


def _dunlop(w: Tensor, e: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Dunlop's cubic coefficients ``x1..x4`` in ``r = w / A2`` (`frictionFactor`)."""
    y2 = e / 3.7 + DUNLOP_AB
    y3 = SWAMEE_JAIN_LOG_COEFF * torch.log(y2)
    fa = 1.0 / (y3 * y3)
    fb = (2.0 + DUNLOP_AC / (y2 * y3)) * fa
    x1 = 7.0 * fa - fb
    x2 = 0.128 - 17.0 * fa + 2.5 * fb
    x3 = -0.128 + 13.0 * fa - (fb + fb)
    x4 = 0.032 - 3.0 * fa + 0.5 * fb
    return x1, x2, x3, x4


def composite_friction_factor(q: Tensor, e: Tensor, s: Tensor) -> tuple[Tensor, Tensor]:
    """Composite friction factor for ``Re > 2000``: ``(f, df/dq)``.

    This is EPANET 2.2's ``frictionFactor`` (`hydcoeffs.c`).

    ``q`` is ``|flow|``, ``e`` the relative roughness ``eps / d`` and ``s = nu d`` (any
    consistent units: ``w = q / s = Re pi / 4``). Swamee-Jain for Re >= 4000, Dunlop's
    cubic in ``r = Re / 2000`` between 2000 and 4000. The laminar ``f = 64 / Re`` is not
    here because EPANET does not compute it as a friction factor either: `DWpipecoeff`
    applies Hagen-Poiseuille to the head loss directly.

    Both branches are evaluated everywhere and selected with ``where``; the Swamee-Jain
    branch sees ``w`` clamped to at least ``A1`` so its un-taken values stay finite.

    Alias: `epanet_friction_factor` (EPANET).
    """
    w = q / s
    turbulent = w >= W_RE_4000
    w_sj = torch.clamp(w, min=W_RE_4000)
    q_sj = w_sj * s
    y1 = SWAMEE_JAIN_W_COEFF / w_sj**0.9
    y2 = e / 3.7 + y1
    y3 = SWAMEE_JAIN_LOG_COEFF * torch.log(y2)
    f_sj = 1.0 / (y3 * y3)
    dfdq_sj = 1.8 * f_sj * y1 * SWAMEE_JAIN_LOG_COEFF / y2 / y3 / q_sj
    x1, x2, x3, x4 = _dunlop(w, e)
    r = w / W_RE_2000
    f_tr = x1 + r * (x2 + r * (x3 + r * x4))
    dfdq_tr = (x2 + r * (2.0 * x3 + r * 3.0 * x4)) / s / W_RE_2000
    return torch.where(turbulent, f_sj, f_tr), torch.where(turbulent, dfdq_sj, dfdq_tr)


class CompositeDarcyWeisbach(ConstitutiveLaw):
    """Composite Darcy-Weisbach pipe: Hagen-Poiseuille, Dunlop's cubic, then Swamee-Jain.

    This is EPANET 2.2's Darcy-Weisbach pipe, reproduced operation for operation
    (`DWpipecoeff`).

    Head loss in EPANET's units (feet, cfs; ``q`` signed)::

        h = (16 pi nu d R + m |q|) q        Re <= 2000   (Hagen-Poiseuille)
        h = (f(Re) R + m) |q| q             Re >  2000   (Dunlop cubic, then Swamee-Jain)

    with ``R = L / (2 * 32.2 * d * A^2)`` and ``m = 0.02517 K / d^4``. The composite is
    C^1 in exact arithmetic: Dunlop's cubic matches ``64 / Re`` in value and slope at
    Re = 2000 and Swamee-Jain in value and slope at Re = 4000. EPANET has NO low-flow
    linearisation for D-W (the ``RQtol`` floor in `pipecoeff` is on the H-W / C-M branch
    only): the laminar branch is already linear through ``q = 0``, which is also why this
    element needs no transition blend and has a finite slope at ``dp = 0``.

    Inputs are SI (m, m^3/s, m^2/s) and are converted to feet internally. ``cfs_per_m3s``
    is the flow conversion EPANET itself applies: it reads flows in the file's units and
    divides by its own ROUNDED factor (``LPSperCFS = 28.317``, ``GPMperCFS = 448.831``, ...
    in `types.h`), so the internal cfs of an LPS file are ``1000 / 28.317`` per m^3/s,
    not ``1 / 0.3048^3``. ``None`` selects the exact conversion.

    The layer potential may be head times ``scale`` (the water application's D-W path
    works in pressure, ``scale = rho g``): ``dp = scale * h`` with ``h`` in metres.

    ``flow(dp)`` inverts the odd, strictly increasing ``h(q)`` by Newton's method with
    EPANET's own analytic gradient, safeguarded by bisection on ``[0, q_hi]`` (``q_hi``
    doubled until it brackets the root) and run without autograd; ONE final Newton step is
    then taken on the graph, whose value is the converged flow and whose first derivative
    in ``dp`` and in every parameter is the implicit-function one, ``-(dh/dtheta) / h'(q)``.
    It is taken on the SIGNED flow, so the derivative at ``q = 0`` is the finite laminar
    ``1 / h'(0)`` rather than the zero a ``sign(dp)`` factor would give. (A generic
    ``solve_monotone`` inversion is deliberately not used: it needs an autograd
    derivative per iteration and a collapsed bracket to stop; see
    ``docs/design/water_street.md#headloss-inversion``.)

    Alias: `EpanetDarcyWeisbach` (EPANET).
    """

    def __init__(
        self,
        length,
        diameter,
        roughness,
        *,
        minor_loss=0.0,
        nu: float = NU_WATER_REF,
        cfs_per_m3s: float | None = None,
        scale: float = 1.0,
        rtol: float = 1e-15,
        max_iter: int = 100,
        kind: str = "pipe",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        self.length = self._param(length, False)
        self.diameter = self._param(diameter, False)
        self.roughness = self._param(roughness, learnable)
        self.minor_loss = self._param(minor_loss, False)
        self.nu = float(nu)
        self.cfs_per_m3s = (
            1.0 / M_PER_FT**3 if cfs_per_m3s is None else float(cfs_per_m3s)
        )
        self.scale = float(scale)
        self.rtol = float(rtol)
        self.max_iter = int(max_iter)

    def _coefficients(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(R, m, e, s)`` in feet/cfs: resistance, minor-loss coefficient, relative
        roughness and ``nu d``, broadcast to a common shape."""
        d = self.diameter / M_PER_FT
        length = self.length / M_PER_FT
        area = torch.pi * d**2 / 4.0
        r = length / 2.0 / G_FT_S2 / d / area**2
        m = MINOR_LOSS_K_FT * self.minor_loss / d**2 / d**2
        e = (self.roughness / M_PER_FT) / d
        s = (self.nu / M_PER_FT**2) * d
        return torch.broadcast_tensors(r, m, e, s)

    @staticmethod
    def _head_ft(q: Tensor, r: Tensor, m: Tensor, e: Tensor, s: Tensor) -> Tensor:
        """Signed head loss (ft) at signed flow ``q`` (cfs)."""
        a = q.abs()
        laminar = a <= W_RE_2000 * s
        f, _ = composite_friction_factor(torch.where(laminar, W_RE_2000 * s, a), e, s)
        h_lam = (16.0 * torch.pi * s * r + m * a) * q
        h_turb = (f * r + m) * a * q
        return torch.where(laminar, h_lam, h_turb)

    def head_loss(self, q: Tensor) -> Tensor:
        """Signed head loss in METRES at signed flow ``q`` in m^3/s (EPANET's ``hloss``)."""
        r, m, e, s = self._coefficients()
        return M_PER_FT * self._head_ft(q * self.cfs_per_m3s, r, m, e, s)

    @staticmethod
    def _grad_ft(a: Tensor, r: Tensor, m: Tensor, e: Tensor, s: Tensor) -> Tensor:
        """EPANET's ``hgrad`` (ft per cfs) at ``|q| = a`` (cfs)."""
        laminar = a <= W_RE_2000 * s
        a_t = torch.where(laminar, W_RE_2000 * s, a)
        f, dfdq = composite_friction_factor(a_t, e, s)
        g_lam = 16.0 * torch.pi * s * r + 2.0 * m * a
        g_turb = 2.0 * (f * r + m) * a + dfdq * r * a * a
        return torch.where(laminar, g_lam, g_turb)

    def head_gradient(self, q: Tensor, cols=None) -> Tensor:
        """``dh/dq`` in m per (m^3/s), from EPANET's own analytic ``hgrad``; of the pipes
        ``cols`` only (``q`` then theirs) when given."""
        r, m, e, s = self._coefficients()
        if cols is not None:
            r, m, e, s = r[..., cols], m[..., cols], e[..., cols], s[..., cols]
        a = (q * self.cfs_per_m3s).abs()
        return M_PER_FT * self.cfs_per_m3s * self._grad_ft(a, r, m, e, s)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        r, m, e, s = self._coefficients()
        c = self.cfs_per_m3s
        h_ft, r, m, e, s = torch.broadcast_tensors(
            dp / self.scale / M_PER_FT, r, m, e, s
        )
        with torch.no_grad():
            r0, m0, e0, s0 = r.detach(), m.detach(), e.detach(), s.detach()
            target = h_ft.detach().abs()
            # Upper bound on |q|: the larger of the laminar flow and the flow at an
            # optimistic f = 0.005, doubled until the head loss there reaches the target.
            hi = torch.maximum(
                torch.sqrt(target / (0.005 * r0 + m0)),
                target / (16.0 * torch.pi * s0 * r0),
            )
            for _ in range(200):
                short = self._head_ft(hi, r0, m0, e0, s0) < target
                if not bool(short.any()):
                    break
                hi = torch.where(short, 2.0 * hi, hi)
            # Safeguarded Newton with EPANET's analytic gradient, from the upper bound.
            lo = torch.zeros_like(hi)
            a = hi.clone()
            for _ in range(self.max_iter):
                res = self._head_ft(a, r0, m0, e0, s0) - target
                hi = torch.where(res > 0, a, hi)
                lo = torch.where(res < 0, a, lo)
                step = res / self._grad_ft(a, r0, m0, e0, s0)
                nxt = a - step
                inside = (nxt >= lo) & (nxt <= hi) & torch.isfinite(nxt)
                nxt = torch.where(inside, nxt, 0.5 * (lo + hi))
                converged = (res == 0) | ((nxt - a).abs() <= self.rtol * a)
                a = torch.where(res == 0, a, nxt)
                if bool(converged.all()):
                    break
            else:
                raise RuntimeError(
                    f"CompositeDarcyWeisbach (kind {self.kind!r}): the head-loss inversion "
                    f"did not converge in {self.max_iter} iterations"
                )
            q_star = torch.sign(h_ft.detach()) * a
            g_star = self._grad_ft(a, r0, m0, e0, s0)
        # One Newton step taken WITH the autograd graph: its value is q_star (the residual
        # is zero to rounding) and its first derivative in every input is the implicit-
        # function one, -(dh/dtheta - dh_t/dtheta) / h'(q), finite at q = 0.
        q_cfs = q_star - (self._head_ft(q_star, r, m, e, s) - h_ft) / g_star
        return q_cfs / c

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        q = self.flow(dp, drivers)
        return 1.0 / (self.scale * self.head_gradient(q))

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        # The laminar tangent at q = 0, as `Duct.linear_init` uses its laminar line.
        slope = self.dflow(torch.zeros_like(self.length * self.diameter), drivers)
        return torch.zeros_like(slope), slope


# EPANET 2.2 names, kept as aliases: they name the reference model (and its C identifiers)
# this law is transcribed from.
EpanetDarcyWeisbach = CompositeDarcyWeisbach  # alias, the EPANET name
epanet_friction_factor = composite_friction_factor  # alias, the EPANET name
EPANET_A1 = W_RE_4000  # alias, the EPANET name
EPANET_A2 = W_RE_2000  # alias, the EPANET name
EPANET_A8 = SWAMEE_JAIN_W_COEFF  # alias, the EPANET name
EPANET_A9 = SWAMEE_JAIN_LOG_COEFF  # alias, the EPANET name
EPANET_AB = DUNLOP_AB  # alias, the EPANET name
EPANET_AC = DUNLOP_AC  # alias, the EPANET name
EPANET_FOOT = M_PER_FT  # alias, the EPANET name
EPANET_G_FT = G_FT_S2  # alias, the EPANET name
EPANET_KM = MINOR_LOSS_K_FT  # alias, the EPANET name
EPANET_VISCOS_FT2 = NU_WATER_FT2  # alias, the EPANET name
EPANET_NU = NU_WATER_REF  # alias, the EPANET name


def epanet_minor_coefficient(minor_loss: Tensor, diameter: Tensor,
                             cfs_per_m3s: float | None = None) -> Tensor:
    """EPANET's minor-loss coefficient as a head (m) per (m3/s)^2: ``Km = 0.02517 K / d^4``
    in ft per cfs^2 (`input1.c`, `convertunits`, with EPANET's rounded constant), converted
    with ``cfs_per_m3s`` (EPANET's own rounded flow factor; None: the exact
    ``1 / 0.3048^3``)."""
    cfs = 1.0 / M_PER_FT**3 if cfs_per_m3s is None else float(cfs_per_m3s)
    d_ft = diameter / M_PER_FT
    return M_PER_FT * MINOR_LOSS_K_FT * minor_loss / d_ft**4 * cfs**2


class _Valve(ImplicitLaw):
    """What every valve shares: its status. ``links`` (a slice of `model.link_names`) are the
    valves' columns of the per-link drivers; with it, ``"water.status"`` selects the valve's
    equation per edge, as EPANET 2.2's `valvecoeff` (`hydcoeffs.c`) does:

    - ``ACTIVE``: the valve acts on its setting (`active`, per subclass);
    - ``OPEN``: the valve is its minor loss, ``dp = minor q |q|`` (`open_valve_loss`),
      whatever its setting;
    - ``CLOSED``: ``kappa q = 0``, no flow. ``kappa`` (the layer's potential per m^3/s) only
      scales the row; the solution does not depend on it.

    Without ``links`` the valve is always ``ACTIVE`` and reads no driver. ``minor`` is in
    the layer's potential unit per (m^3/s)^2 and ``gradient_floor`` (EPANET's ``RQtol``) per
    m^3/s.
    """

    def __init__(self, *, minor=0.0, kappa=1.0, gradient_floor: float = 0.0,
                 links: slice | None = None, kind: str) -> None:
        super().__init__(kind)
        self.minor = self._param(minor, False)
        self.kappa = self._param(kappa, False)
        self.gradient_floor = float(gradient_floor)
        self.link_range = _link_range(links)

    def active(self, dp, q, drivers):
        """The residual of the ACTIVE valve."""
        raise NotImplementedError

    def _active(self, dp, q, drivers, ctx):
        """The residual of the ACTIVE valve, given the context too (a valve that reads the
        heads at its ends needs it)."""
        return self.active(dp, q, drivers)

    def _setting(self, own, drivers, like: Tensor) -> Tensor:
        """The valve's setting: the driver ``"water.setting"`` with ``links``, else ``own``."""
        if self.link_range is None:
            if own is None:
                raise ValueError(
                    f"{type(self).__name__} (kind {self.kind!r}): give a setting, or links "
                    f"to read it from the driver {SETTING_KEY!r}")
            return own
        return link_driver(self, drivers, SETTING_KEY, like)

    def residual(self, dp, q, drivers, ctx):
        active = self._active(dp, q, drivers, ctx)
        if self.link_range is None:
            return active
        # `None` when every valve is ACTIVE, the usual case: no `where` in the graph
        status = link_flags(self, drivers, (STATUS_KEY,), dp, lambda status: (
            None if bool((status == ACTIVE).all()) else status))
        if status is None:
            return active
        open_ = dp - open_valve_loss(q, self.minor, self.gradient_floor)
        return torch.where(status == CLOSED, closed_row(self, dp, q, drivers),
                           torch.where(status == OPEN, open_, active))


class ThrottleValve(_Valve):
    """A throttle control valve (TCV): EPANET 2.2's `tcvcoeff` (`hydcoeffs.c`).

    Active, it is a minor loss whose coefficient is its setting ``K``: ``dp = m q |q|`` with
    ``m = 0.02517 K / d^4`` in EPANET's units (`epanet_minor_coefficient`, ``cfs_per_m3s``
    its flow conversion), times ``scale`` (``rho g`` on a pressure layer), and the gradient
    floor of `open_valve_loss`. The setting is ``setting``, or the driver ``"water.setting"``
    with ``links``; see `_Valve` for the status. Implicit: the flow is an unknown of the
    layer.
    """

    def __init__(self, diameter, *, setting=None, cfs_per_m3s: float | None = None,
                 scale: float = 1.0, minor=0.0, kappa=1.0, gradient_floor: float = 0.0,
                 links: slice | None = None, kind: str = "valve") -> None:
        super().__init__(minor=minor, kappa=kappa, gradient_floor=gradient_floor,
                         links=links, kind=kind)
        self.diameter = self._param(diameter, False)
        self.setting = None if setting is None else self._param(setting, False)
        self.cfs_per_m3s = cfs_per_m3s
        self.scale = float(scale)

    def active(self, dp, q, drivers):
        k = self._setting(self.setting, drivers, dp)
        m = self.scale * epanet_minor_coefficient(k, self.diameter, self.cfs_per_m3s)
        return dp - open_valve_loss(q, m, self.gradient_floor)


class FlowControlValve(_Valve):
    """A flow control valve (FCV), active: its flow is its setting (m^3/s),
    ``kappa (q - setting) = 0``, whatever the heads. The setting is ``setting``, or the
    driver ``"water.setting"`` with ``links``; see `_Valve` for the status. Implicit: the
    flow is an unknown of the layer.
    """

    def __init__(self, *, setting=None, minor=0.0, kappa=1.0, gradient_floor: float = 0.0,
                 links: slice | None = None, kind: str = "fcv") -> None:
        super().__init__(minor=minor, kappa=kappa, gradient_floor=gradient_floor,
                         links=links, kind=kind)
        self.setting = None if setting is None else self._param(setting, False)

    def active(self, dp, q, drivers):
        return self.kappa * (q - self._setting(self.setting, drivers, q))


class BreakerValve(_Valve):
    """A pressure breaker valve (PBV): EPANET 2.2's `pbvcoeff` (`hydcoeffs.c`).

    Active, it holds the head drop from its start node to its end node at ``setting``,
    whichever way the water flows: ``dp = setting``. It is an open valve, ``dp = m q |q|``
    with ``m`` its minor-loss coefficient (`epanet_minor_coefficient`), where that loss
    exceeds the setting (``m q^2 > setting``) or the setting is 0. Implicit: the flow is
    an unknown of the layer. ``setting`` and ``m`` are in the layer's potential unit (head,
    or pressure on a Darcy-Weisbach pressure layer: both scaled by ``rho g``). The setting is
    ``setting``, or the driver ``"water.setting"`` with ``links``; see `_Valve` for the
    status, whose ``OPEN`` minor loss is this same ``m``.
    """

    def __init__(self, setting, minor, *, kappa=1.0, gradient_floor: float = 0.0,
                 links: slice | None = None, kind: str = "pbv") -> None:
        super().__init__(minor=minor, kappa=kappa, gradient_floor=gradient_floor,
                         links=links, kind=kind)
        self.setting = None if setting is None else self._param(setting, False)

    def active(self, dp, q, drivers):
        setting = self._setting(self.setting, drivers, dp)
        loss = self.minor * q * q.abs()
        open_ = (setting == 0) | (self.minor * q * q > setting)
        return dp - torch.where(open_, loss, setting)


class GeneralPurposeValve(_Valve):
    """A general purpose valve (GPV): EPANET 2.2's `gpvcoeff` (`hydcoeffs.c`).

    The head loss follows the valve's head-loss curve, ``dp = sgn(q) (h0 + r |q|)``, with
    ``h0`` and ``r`` the intercept and slope of the curve segment that brackets ``|q|``
    (`curvecoeff`: the first or last segment, extended, outside the curve) and
    ``sgn(0) = +1``. ``flows`` (m3/s) and ``heads`` (the layer's potential unit) are
    ``(n_valves, n_points)``, each row increasing in flow and padded at the end with
    ``inf`` flows to the longest curve; ``n_points`` gives each row's own length (at least
    2). Implicit: the flow is an unknown of the layer. See `_Valve` for the status.
    """

    def __init__(self, flows, heads, n_points, *, minor=0.0, kappa=1.0,
                 gradient_floor: float = 0.0, links: slice | None = None,
                 kind: str = "gpv") -> None:
        super().__init__(minor=minor, kappa=kappa, gradient_floor=gradient_floor,
                         links=links, kind=kind)
        self.flows = self._param(flows, False)
        self.heads = self._param(heads, False)
        self.register_buffer("n_points", torch.as_tensor(n_points, dtype=torch.long))

    def active(self, dp, q, drivers):
        a = q.abs()
        # The bracketing segment: k2 = number of points below |q|, kept within the curve.
        k2 = (self.flows < a.unsqueeze(-1)).sum(-1).clamp(min=1)
        k2 = torch.minimum(k2, self.n_points - 1)
        k1 = k2 - 1
        x1 = self.flows.expand(a.shape + self.flows.shape[-1:]).gather(-1, k1.unsqueeze(-1))
        x2 = self.flows.expand(a.shape + self.flows.shape[-1:]).gather(-1, k2.unsqueeze(-1))
        y1 = self.heads.expand(a.shape + self.heads.shape[-1:]).gather(-1, k1.unsqueeze(-1))
        y2 = self.heads.expand(a.shape + self.heads.shape[-1:]).gather(-1, k2.unsqueeze(-1))
        r = ((y2 - y1) / (x2 - x1)).squeeze(-1)
        h0 = y1.squeeze(-1) - r * x1.squeeze(-1)
        sign = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
        return dp - sign * (h0 + r * a)


#: A flow (m^3/s) below which a pressure valve's no-flow row does not win a near tie
#: (`_PressureValve`, THE SELECTION).
NO_FLOW_TIE = 1e-10

#: The per-link driver `water_steady` sets on a PRV or PSV with a dead end (`_dead_ends` in
#: `network.py`): +1 where the group of junctions behind its END node has no demand and no
#: other way to a reservoir or tank, -1 behind its START node, 0 elsewhere. Absent: none.
DEAD_END_KEY = "water.dead_end"


class _PressureValve(_Valve):
    """What a PRV and a PSV share: one complementarity condition on the absolute head at one
    end (`docs/plans/water-prv-psv-design.md`, section 2).

    The valve ``u -> v`` holds the head at its controlled node at ``Hset = elevation +
    setting``: ``elevation`` is the controlled node's, in the layer's potential unit, and the
    setting is the driver ``"water.setting"`` with ``links``, else ``setting``. With
    ``held`` the controlled head beyond its setting (PRV ``H_v - Hset``, PSV
    ``Hset - H_u``), ``loss = open_valve_loss(q) - dp`` (the open valve's loss less the
    drop; ``minor`` its coefficient, as for every valve, EPANET's `valvecoeff`) and
    ``g = max(held, loss)``, the residual of the ACTIVE valve is::

        r = min(kappa q, g)        that is   q >= 0,  g >= 0,  q g = 0

    It selects EPANET 2.2's three states with no memory of the previous one; EPANET's own
    transition rules (`hydstatus.c`, `prvstatus` and `psvstatus`) reach the same states
    outside their ``HTOL`` hysteresis band. The ``kappa q`` row, ``q = 0``, is CLOSED
    (reverse flow, or the controlled head beyond its setting with no flow); the ``held``
    row is ACTIVE (the controlled head at its setting); the ``loss`` row is OPEN (a minor
    loss, the controlled head short of its setting). ``kappa`` (the layer's potential per
    m^3/s) scales the no-flow row only; the solution does not depend on it. A
    ``"water.status"`` of OPEN or CLOSED overrides the condition, as for every valve
    (`_Valve`).

    THE SELECTION. The rows are taken with ``torch.where``, never ``torch.maximum`` or
    ``minimum``, whose gradient splits 0.5/0.5 at a tie: at ``held == loss`` the ``held``
    row (ACTIVE), and the ``kappa q`` row only where the flow is not forward, ``q <= 0``,
    and ``kappa q < g - kappa NO_FLOW_TIE``. At a solution these agree with the ``min``
    (a CLOSED valve has ``q = 0``, an ACTIVE or OPEN one ``g = 0``), so the solution set,
    the converged row and its gradient are the ``min``'s. Away from one, the fixed
    ``kappa`` alone misjudges an iterate that flows forward: MEASURED, a PRV line (20 L/s
    to a junction downstream, ``kappa`` = 1 m per m^3/s) whose first iterate leaves the
    downstream head 7 m above the setting picks the no-flow row, which cuts that junction
    and its demand off, and the next step is singular; with ``q <= 0`` it takes the ACTIVE
    row and converges in two steps. EPANET, too, closes an active or open PRV/PSV only on
    reverse flow (`prvstatus`, ``q < -Qtol``). The margin ``kappa NO_FLOW_TIE`` matters at a
    solution with no flow and ``g = 0``: both rows vanish there, the no-flow row would
    leave a side without demand without an equation, and rounding alone would pick it. It
    moves no solution by more than ``kappa NO_FLOW_TIE`` in head.

    A DEAD END. Where ``"water.dead_end"`` (`DEAD_END_KEY`) marks a side of the valve whose
    junctions have no demand and no other way to a reservoir or tank, no water flows
    through the valve, and the no-flow row would leave that side's head without an
    equation (a singular step). The valve then takes the row that ties that side and gives
    EPANET's answer (MEASURED, wntr 1.5.0, a dead end behind the valve): ``g`` where the
    dead side is the controlled node (a PRV's end: ACTIVE at the setting, or OPEN at the
    upstream head when that is below it), else the open row ``loss`` (a PSV's end, OPEN at
    the upstream head; a PRV's start). The ``held`` row of a PSV ties only its start node.

    Newton's method solves the row each iterate selects (a semismooth Newton method), and
    the Jacobian and the implicit-function gradient at the solution are the selected row's:
    ``dH_v/dsetting = 1`` through an ACTIVE PRV, 0 through an OPEN or CLOSED one. Across a
    switch the solution is continuous and its derivative jumps.

    ``smoothing = eps > 0`` (the layer's potential unit) replaces the ``min`` by the
    Fischer-Burmeister function ``a + b - sqrt(a^2 + b^2 + 2 eps^2)`` (its root: ``a, b > 0``
    with ``a b = eps^2``) and the ``max`` by ``(x + y + sqrt((x - y)^2 + 4 eps^2)) / 2``.
    The solution moves by ``O(eps)`` near a switch, and its derivative is continuous across
    it (``dH_v/dsetting`` rises from 0 to 1 over a band of width ``O(eps)``), for an
    optimiser that must cross one.
    """

    reads_heads = True
    #: The side of the valve whose head it holds: +1 the end node (PRV), -1 the start (PSV).
    HELD_SIDE = 0

    def __init__(self, elevation, minor, *, kappa, smoothing: float = 0.0, setting=None,
                 gradient_floor: float = 0.0, links: slice | None = None,
                 kind: str) -> None:
        super().__init__(minor=minor, kappa=kappa, gradient_floor=gradient_floor,
                         links=links, kind=kind)
        self.elevation = self._param(elevation, False)
        self.setting = None if setting is None else self._param(setting, False)
        self.smoothing = float(smoothing)
        if not self.smoothing >= 0:
            raise ValueError(f"{type(self).__name__} (kind {kind!r}): smoothing must be >= 0, "
                             f"got {smoothing!r}")

    def _held(self, ctx, hset: Tensor) -> Tensor:
        """The controlled head beyond its setting, positive where it is a reason to close."""
        raise NotImplementedError

    def _rows(self, dp, q, drivers, ctx) -> tuple[Tensor, Tensor, Tensor]:
        """`(kappa q, held, loss)`: the no-flow, ACTIVE and OPEN rows."""
        if ctx is None or ctx.phi_from is None or ctx.phi_to is None:
            raise ValueError(
                f"{type(self).__name__} (kind {self.kind!r}) reads the heads at its end "
                f"nodes: its context must carry phi_from and phi_to")
        hset = self.elevation + self._setting(self.setting, drivers, dp)
        loss = open_valve_loss(q, self.minor, self.gradient_floor) - dp
        return self.kappa * q, self._held(ctx, hset), loss

    def _active(self, dp, q, drivers, ctx):
        flow, held, loss = self._rows(dp, q, drivers, ctx)
        g = torch.where(held >= loss, held, loss)
        eps = self.smoothing
        if eps > 0:
            smooth = 0.5 * (held + loss + torch.sqrt((held - loss) ** 2 + 4.0 * eps * eps))
            r = flow + smooth - torch.sqrt(flow * flow + smooth * smooth + 2.0 * eps * eps)
        else:
            r = torch.where(self._closed(q, flow, g), flow, g)
        dead = None if drivers is None else drivers.get(DEAD_END_KEY)
        if dead is None or self.link_range is None:
            return r
        dead = dead[..., slice(*self.link_range)]
        if not bool(dead.any()):
            return r
        tie = torch.where(dead == self.HELD_SIDE, g, loss)
        return torch.where(dead != 0, tie, r)

    def _closed(self, q, flow, g) -> Tensor:
        """Where the no-flow row is taken (THE SELECTION)."""
        return (q <= 0) & (flow < g - self.kappa * NO_FLOW_TIE)

    def states(self, dp, q, drivers, ctx) -> Tensor:
        """The state the ACTIVE residual selects at ``(dp, q)`` and the heads in ``ctx``, per
        valve: `CLOSED`, `ACTIVE` or `OPEN` (`noodl.apps.water.controls`), by the rule of THE
        SELECTION; a smoothed valve, whose closed flow is small but positive, is CLOSED where
        ``kappa q < g``."""
        with torch.no_grad():
            flow, held, loss = self._rows(dp, q, drivers, ctx)
            g = torch.where(held >= loss, held, loss)
            closed = self._closed(q, flow, g) if self.smoothing == 0 else flow < g
            return torch.where(closed, torch.full_like(g, CLOSED),
                               torch.where(held >= loss, torch.full_like(g, ACTIVE),
                                           torch.full_like(g, OPEN)))


class PressureReducingValve(_PressureValve):
    """A pressure reducing valve (PRV): it holds the head at its END node at no more than
    ``Hset = elevation + setting`` (``elevation`` the end node's). EPANET 2.2's PRV
    (`hydstatus.c`, `prvstatus`; `hydcoeffs.c`, `prvcoeff`), with ``held = H_v - Hset``: a
    downstream head above the setting is the reason for it to close. See `_PressureValve`.
    """

    HELD_SIDE = 1

    def __init__(self, elevation, minor, *, kappa, smoothing: float = 0.0, setting=None,
                 gradient_floor: float = 0.0, links: slice | None = None,
                 kind: str = "prv") -> None:
        super().__init__(elevation, minor, kappa=kappa, smoothing=smoothing, setting=setting,
                         gradient_floor=gradient_floor, links=links, kind=kind)

    def _held(self, ctx, hset):
        return ctx.phi_to - hset


class PressureSustainingValve(_PressureValve):
    """A pressure sustaining valve (PSV): it holds the head at its START node at no less
    than ``Hset = elevation + setting`` (``elevation`` the start node's). EPANET 2.2's PSV
    (`hydstatus.c`, `psvstatus`; `hydcoeffs.c`, `psvcoeff`), with ``held = Hset - H_u``: an
    upstream head below the setting is the reason for it to close. See `_PressureValve`.
    """

    HELD_SIDE = -1

    def __init__(self, elevation, minor, *, kappa, smoothing: float = 0.0, setting=None,
                 gradient_floor: float = 0.0, links: slice | None = None,
                 kind: str = "psv") -> None:
        super().__init__(elevation, minor, kappa=kappa, smoothing=smoothing, setting=setting,
                         gradient_floor=gradient_floor, links=links, kind=kind)

    def _held(self, ctx, hset):
        return hset - ctx.phi_from


#: EPANET's floor on a head-loss gradient, ``RQTOL = 1e-7`` ft per cfs (`hydraul.c`,
#: `RQtol` default, Manual Appendix C ``[OPTIONS]``).
RQTOL_FT_PER_CFS = 1e-7

#: EPANET's head-loss gradient of a CLOSED link, ``CBIG = 1e8`` ft per cfs (`hydcoeffs.c`).
CBIG_FT_PER_CFS = 1e8


class FlowControlledPipe(ImplicitLaw):
    """A pipe in its resistance form, ``dp = scale h(q)``, with the flow an unknown of the
    layer (`build_model`'s ``pipe_form="flow"``): ``pipe`` is the pipe's potential-form law
    (`HazenWilliams` or `CompositeDarcyWeisbach`), whose ``head_loss`` gives ``h``.

    The potential form inverts ``h`` (``q = g(dp)``), whose slope grows without bound as the
    flow goes to zero; the resistance form's slope goes to zero there instead, which the
    solve with flow unknowns carries as a small diagonal entry. A Hazen-Williams pipe
    keeps EPANET's ``RQtol`` floor on that slope (``gradient_floor``, m per m^3/s), so a
    loop of pipes at zero flow still has a nonsingular Jacobian.

    With ``links`` (the pipes' columns of the per-link drivers, a slice of
    `model.link_names`) the pipe reads ``"water.status"`` and ``"water.one_way"``. With
    ``loss = dp - scale h(q)`` its residual is

    - ``kappa q`` when ``CLOSED`` (no flow);
    - ``loss`` both ways (``one_way = 0``);
    - forward only (``+1``, a check valve), ``max(-kappa q, loss)``: either ``q > 0`` and
      ``dp = scale h(q)``, or ``q = 0`` and ``dp <= 0``;
    - reverse only (``-1``), ``min(-kappa q, loss)``: either ``q < 0`` and
      ``dp = scale h(q)``, or ``q = 0`` and ``dp >= 0``.

    One exact equation per pipe, with no memory of the previous status, solved by Newton's
    method on the branch each iterate selects (`docs/plans/water-prv-psv-design.md`,
    section 2); EPANET reaches the same states with an outer loop over statuses
    (`hydstatus.c`, `cvstatus`). The branches are selected with ``torch.where``, never
    ``torch.maximum``/``minimum``, which split the gradient 0.5/0.5 at a tie; a tie takes
    the open branch, ``loss``. ``kappa`` (the layer's potential per m^3/s, per pipe) scales
    the closed row; `build_model` sets it to the pipe's own head-loss slope at 1 ft/s,
    EPANET's first iterate.

    THE SELECTION. A forward-only pipe takes its no-flow branch where ``loss`` is below
    BOTH ``-kappa q`` and ``-h'(q) q``, ``h'`` the pipe's own slope at the iterate
    (reverse only: above both). The second test says that the open equation's own Newton
    step, the heads held, would take the flow below zero. At a solution the two tests agree
    with ``max(-kappa q, loss)`` (``q > 0`` with ``loss = 0`` is open under both, ``q = 0``
    with ``dp < 0`` closed under both), so the solution set, the converged branch and its
    gradient are the max's. Away from a solution the fixed ``kappa`` alone misjudges an
    iterate that overshoots forward: ``h`` grows as ``q^1.852`` and ``kappa q`` only
    linearly, so ``loss < -kappa q`` closes an open check valve; the closed row then puts
    every pipe in series with it at exactly zero flow, where the next step, on the
    ``RQtol`` slope, overshoots by six orders and closes it again (measured: a check valve
    between two reservoirs cycles so for 200 iterations, and converges in 5 with the
    second test).

    A CUT-OFF ITERATE. An iterate can select the no-flow branch of every check valve that
    feeds a junction, cutting the junction off from every reservoir and tank; its head then
    has no equation and the step is singular (measured: the two-loop with every pipe a
    check valve at five times its demands, which EPANET solves). Given ``ends`` (the
    ``(start, end)`` node indices of every link, in `model.link_names` order) and ``fixed``
    (the reservoir and tank nodes), the pipe takes the open branch instead for each check
    valve it would close that touches such a cut-off group, so the step is taken; the
    solution is still exact, and `water_steady` refuses a converged check valve that carries
    flow the wrong way.
    """

    def __init__(self, pipe, *, gradient_floor: float = 0.0, kappa=1.0,
                 links: slice | None = None, ends=None, fixed=(),
                 kind: str = "pipe") -> None:
        super().__init__(kind)
        self.pipe = pipe
        self.gradient_floor = float(gradient_floor)
        self.scale = float(getattr(pipe, "scale", 1.0))
        self.kappa = self._param(kappa, False)
        self.link_range = _link_range(links)
        self.ends = None if ends is None else tuple((int(u), int(v)) for u, v in ends)
        self.fixed = tuple(int(n) for n in fixed)

    def _reopen(self, held: Tensor, ahead: Tensor, drivers) -> Tensor:
        """`held` (the check valves this iterate closes), with the open branch taken again
        where closing would cut a group of nodes off from every reservoir and tank (see A
        CUT-OFF ITERATE): per such group, the one check valve touching it whose open
        equation predicts the most flow in its own direction (`ahead`), until no group is
        cut off. Links marked by `PIN_KEY` count as joined."""
        if self.ends is None or not bool(held.any()):
            return held
        n_links = len(self.ends)
        lo, hi = self.link_range
        status = drivers[STATUS_KEY].detach()
        pin = drivers.get(PIN_KEY)
        batch = held.shape[:-1]
        joined = (status != CLOSED).expand(batch + (n_links,))
        if pin is not None:
            joined = joined | (pin.detach() != 0).expand(batch + (n_links,))
        joined = joined.reshape(-1, n_links).tolist()
        rows = held.reshape(-1, hi - lo)
        scores = ahead.expand(held.shape).reshape(-1, hi - lo).tolist()
        out = rows.clone()
        for b, row in enumerate(rows.tolist()):
            while any(row):
                links = tuple(joined[b][j] and not (lo <= j < hi and row[j - lo])
                              for j in range(n_links))
                group, fed = _joined_groups(self.ends, self.fixed, links)
                best: dict[int, int] = {}   # cut-off group -> the check valve to reopen
                for k, closing in enumerate(row):
                    if not closing:
                        continue
                    for end in self.ends[lo + k]:
                        g = group[end]
                        if g not in fed and (g not in best
                                             or scores[b][k] > scores[b][best[g]]):
                            best[g] = k
                if not best:
                    break
                for k in set(best.values()):
                    row[k] = False
                    out[b, k] = False
        return out.reshape(held.shape)

    def _loss(self, dp, q):
        if isinstance(self.pipe, HazenWilliams):
            h = self.pipe.head_loss(q, self.gradient_floor)
        else:
            h = self.pipe.head_loss(q)
        return dp - self.scale * h

    def _flags(self, drivers, like: Tensor):
        """`(closed, one_way, cols)` of the law's columns: where the status is CLOSED (`None`
        when nowhere), the direction (`None` when every pipe takes both) and the one-way
        columns as a long tensor when every instance has the same ones (else `None`);
        remembered per set of drivers (`link_flags`)."""
        def summarise(status, one_way):
            closed = status == CLOSED
            if not bool(one_way.any()):
                return (closed if bool(closed.any()) else None), None, None
            flat = one_way.reshape(-1, one_way.shape[-1])
            cols = ((flat[0] != 0).nonzero().flatten() if bool((flat == flat[:1]).all())
                    else None)
            return (closed if bool(closed.any()) else None), one_way, cols

        return link_flags(self, drivers, (STATUS_KEY, ONE_WAY_KEY), like, summarise)

    def _held(self, loss, q, one_way, cols, drivers) -> Tensor:
        """Where a one-way pipe takes its no-flow branch (see THE SELECTION, and A CUT-OFF
        ITERATE); never at a tie. `cols` are the one-way columns (`_flags`), `None` for all.
        Computed without a graph: it only selects."""
        with torch.no_grad():
            q, loss = torch.broadcast_tensors(q.detach(), loss.detach())
            one_way = one_way.expand(q.shape)
            q_c, loss_c, way_c = ((q, loss, one_way) if cols is None else
                                  (q[..., cols], loss[..., cols], one_way[..., cols]))
            if isinstance(self.pipe, HazenWilliams):
                slope = self.pipe.head_gradient(q_c, self.gradient_floor, cols)
            else:
                slope = self.pipe.head_gradient(q_c, cols)
            slope = self.scale * slope
            kappa = self.kappa.detach() * torch.ones(q.shape[-1], dtype=q.dtype)
            by_kappa = -(kappa if cols is None else kappa[cols]) * q_c
            forward = (loss_c < by_kappa) & (loss_c < -slope * q_c)
            reverse = (loss_c > by_kappa) & (loss_c > -slope * q_c)
            held_c = ((way_c > 0) & forward) | ((way_c < 0) & reverse)
            # the flow the open equation's Newton step reaches, in the pipe's own direction
            ahead_c = way_c * (q_c + loss_c / slope)
            if cols is None:
                held, ahead = held_c, ahead_c
            else:
                held = torch.zeros(q.shape, dtype=torch.bool)
                ahead = torch.zeros_like(q)
                held[..., cols], ahead[..., cols] = held_c, ahead_c
            return self._reopen(held, ahead, drivers)

    def residual(self, dp, q, drivers, ctx):
        loss = self._loss(dp, q)
        if self.link_range is None:
            return loss
        closed, one_way, cols = self._flags(drivers, dp)
        if closed is None and one_way is None:   # the usual case: no `where` in the graph
            return loss
        if one_way is not None:
            held = self._held(loss, q, one_way, cols, drivers)
            closed = held if closed is None else closed | held
            if not bool(closed.any()):
                return loss
        return torch.where(closed, closed_row(self, dp, q, drivers), loss)


class FlowControlledPump(ImplicitLaw):
    """A pump in its head-gain form, ``phi_discharge - phi_suction = gain(q)``, with the flow
    an unknown of the layer, as EPANET's and wntr's own solvers write it. ``pump`` is the
    pump's `PumpCurve`, whose ``head_gain`` gives the gain (with ``gradient_floor``, EPANET's
    ``RQtol``, and ``closed_slope``, its ``CBIG``, both in the layer's potential per m^3/s).
    `build_model` writes every pump in this form, whatever the ``pipe_form``.

    The potential form, ``PumpCurve.flow``, inverts the curve: its slope is infinite at the
    shut-off head and EXACTLY zero beyond it, where the pump delivers nothing whatever the
    heads. A Newton iterate that lands beyond shut-off then has no Jacobian entry at all
    between the pump's two sides. With fixed demands the next iterate recovers; with a
    pressure-driven demand downstream, the only thing left tying those heads to a boundary
    is the demand curve's own slope, which is small (and zero above the required pressure),
    so the next step throws the heads far outside the curve and the system turns singular
    (measured: a reservoir, a pump and a pipe feeding a PDA junction, singular from the
    fifth iterate on). In this form the pump's row ties its two heads together on every
    iterate, as a head source with the curve's slope in series, and the same network
    converges quadratically, in 5 Newton iterations (19 with `pipe_form="potential"`).

    The status is the pump's own (`PumpCurve`, ``links``): an open pump's residual is
    ``dp + gain(q)``, a ``CLOSED`` one's is ``kappa q`` (no flow). ``kappa`` (the layer's
    potential per m^3/s) only scales the row; `build_model` sets it to 1 m per m^3/s (its
    comment gives the measured reason).
    """

    def __init__(self, pump: PumpCurve, *, gradient_floor: float, closed_slope: float,
                 kappa=1.0, kind: str = "pump") -> None:
        super().__init__(kind)
        self.pump = pump
        self.gradient_floor = float(gradient_floor)
        self.closed_slope = float(closed_slope)
        self.kappa = self._param(kappa, False)

    @property
    def h0(self) -> Tensor:
        return self.pump.h0

    @property
    def r(self) -> Tensor:
        return self.pump.r

    def q_max(self) -> Tensor:
        """The flow at zero head gain; see `PumpCurve.q_max`."""
        return self.pump.q_max()

    def residual(self, dp, q, drivers, ctx):
        gain = self.pump.head_gain(q, self.gradient_floor, self.closed_slope)
        if self.link_range is None:
            return dp + gain
        # `None` when every pump runs, the usual case: no `where` in the graph
        running = link_flags(self, drivers, (STATUS_KEY,), dp, lambda status: (
            None if bool((status != CLOSED).all()) else status != CLOSED))
        if running is None:
            return dp + gain
        return torch.where(running, dp + gain, closed_row(self, dp, q, drivers))

    @property
    def link_range(self) -> tuple[int, int] | None:
        """The pump's columns of the per-link drivers (`PumpCurve`'s ``links``)."""
        return self.pump.link_range
