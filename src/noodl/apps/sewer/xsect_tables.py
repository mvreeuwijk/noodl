"""SWMM 5.2's own circular cross-section, reproduced operation for operation, in torch.

`geometry.py` evaluates the exact circular-segment identities. SWMM 5 does not: for a
``CIRCULAR`` conduit it reads 51-point tables (`xsect.dat`: ``A_Circ``, ``Y_Circ``,
``S_Circ``), interpolates them (`xsect.c`: ``lookup``, ``invLookup``, ``locate``), and near
the invert switches to closed forms solved by a Newton iteration with its OWN starting
guesses and stopping rule (``getThetaOfAlpha``, ``getThetaOfPsi``, ``getAcircular``,
``getScircular``, ``getYcircular``). The tables differ from the true circle by 1e-4 to 5e-3
relative in depth (more in velocity near the invert), which is the whole of the
depth/volume/velocity gap between the analytic model and SWMM (SWMM Reference Manual Vol. II
section 5.1.3).

Everything here is a transcription of SWMM 5.2.4 (`src/solver/xsect.c`, `xsect.dat`,
`kinwave.c`, `flowrout.c`, `link.c`, `swmm5.c`), including its unit handling: SWMM
computes internally in feet, converting lengths with 0.3048 m/ft but flows AND volumes with
``Qcf[CMS] = Ucf[VOLUME][SI] = 0.02832`` m3/ft3 (not 0.3048^3 = 0.0283168...), and Manning
with ``PHI = 1.486``. The combination is an effective SI Manning coefficient of
``0.02832 * 1.486 / 0.3048^(8/3) = 1.000156`` and a reported area/volume scale of
``0.02832 / 0.3048^3 = 1.000118``; both are reproduced because they are what SWMM computes.

Kinematic-wave steady state (`kinwave.c::kinwave_execute`, `flowrout.c::setNewLinkState`).
A conduit carrying ``q`` has the inlet area ``a1 = xsect_getAofS(q / beta)`` and the outlet
area ``a2`` that solves ``beta * xsect_getSofA(a2) = q`` (the continuity root, which at a
steady state is exactly this); SWMM reports the depth ``(Y(a1) + Y(a2)) / 2``, the volume
``(a1 + a2) / 2 * L`` and the velocity ``q / xsect_getAofY(depth)``. ``a1 == a2`` exactly
wherever both come from the table (``psi > S_Circ[2]``); below that, ``a1`` comes from
``getAcircular`` and ``a2`` from inverting ``getScircular``, and the two differ slightly --
reproduced as SWMM has it.

Differentiability. Table interpolation is piecewise linear (the segment index is a
non-differentiable integer, the value is linear in the input); the Newton loops are
unrolled with per-element termination masks, so autograd differentiates through them; the
one inverse SWMM obtains by a root solve (``a2`` near the invert) uses
`solvers.scalar.solve_monotone` (implicit-function gradient).
"""

from __future__ import annotations

import torch

from noodl.solvers.scalar import solve_monotone

Tensor = torch.Tensor
F64 = torch.float64

#: SWMM's Manning constant, applied to feet (`consts.h`).
PHI = 1.486
#: SWMM's length conversion, m per ft (`swmm5.c` ``Ucf[LENGTH][SI]``).
LCF = 0.3048
#: SWMM's flow AND volume conversion for CMS, m3 per ft3 (`swmm5.c` ``Qcf[CMS]``,
#: ``Ucf[VOLUME][SI]``).
QCF = 0.02832
#: SWMM's ``PI`` (`consts.h`).
_PI = 3.141592654

# --- xsect.dat, CIRCULAR SHAPE (SWMM 5.2.4), verbatim.
A_CIRC = (
    0.0, .00471, .0134, .024446, .0374, .05208, .0680, .08505, .1033, .12236,
    .1423, .16310, .1845, .20665, .2292, .25236, .2759, .29985, .3242, .34874,
    .3736, .39878, .4237, .44907, .4745, .500, .5255, .55093, .5763, .60135,
    .6264, .65126, .6758, .70015, .7241, .74764, .7708, .79335, .8154, .83690,
    .8576, .87764, .8967, .91495, .9320, .94792, .9626, .97555, .9866, .99516, 1.000,
)
Y_CIRC = (
    0.0, 0.05236, 0.08369, 0.11025, 0.13423, 0.15643, 0.17755, 0.19772, 0.21704,
    0.23581, 0.25412, 0.27194, 0.28948, 0.30653, 0.32349, 0.34017, 0.35666,
    0.37298, 0.38915, 0.40521, 0.42117, 0.43704, 0.45284, 0.46858, 0.4843,
    0.50000, 0.51572, 0.53146, 0.54723, 0.56305, 0.57892, 0.59487, 0.61093,
    0.62710, 0.64342, 0.65991, 0.67659, 0.69350, 0.71068, 0.72816, 0.74602,
    0.76424, 0.78297, 0.80235, 0.82240, 0.84353, 0.86563, 0.88970, 0.91444,
    0.94749, 1.0,
)
S_CIRC = (
    0.0, 0.00529, 0.01432, 0.02559, 0.03859, 0.05304, 0.06877, 0.08551, 0.10326,
    0.12195, 0.14144, 0.16162, 0.18251, 0.2041, 0.22636, 0.24918, 0.27246,
    0.29614, 0.32027, 0.34485, 0.36989, 0.39531, 0.42105, 0.44704, 0.47329,
    0.4998, 0.52658, 0.55354, 0.58064, 0.60777, 0.63499, 0.66232, 0.68995,
    0.7177, 0.74538, 0.77275, 0.79979, 0.82658, 0.8532, 0.87954, 0.90546,
    0.93095, 0.95577, 0.97976, 1.00291, 1.02443, 1.04465, 1.06135, 1.08208,
    1.07662, 1.0,
)
_N = 51


def _table(values) -> Tensor:
    return torch.tensor(values, dtype=F64)


_A = _table(A_CIRC)
_Y = _table(Y_CIRC)
_S = _table(S_CIRC)


def lookup(x: Tensor, table: Tensor) -> Tensor:
    """`xsect.c::lookup`: linear interpolation in an equally spaced table on ``[0, 1]``,
    with SWMM's quadratic correction on the first two segments (kept only if positive)."""
    n = table.shape[0]
    delta = 1.0 / (float(n) - 1.0)
    i = torch.clamp(torch.floor(x / delta), min=0).to(torch.long)
    top = i >= n - 1
    i_safe = torch.clamp(i, max=n - 2)
    x0 = i_safe.to(F64) * delta
    x1 = (i_safe.to(F64) + 1.0) * delta
    t0, t1 = table[i_safe], table[i_safe + 1]
    t2 = table[torch.clamp(i_safe + 2, max=n - 1)]
    y = t0 + (x - x0) * (t1 - t0) / delta
    y2 = y + (x - x0) * (x - x1) / (delta * delta) * (t0 / 2.0 - t1 + t2 / 2.0)
    y = torch.where((i_safe < 2) & (y2 > 0.0), y2, y)
    y = torch.clamp(y, min=0.0)
    return torch.where(top, table[n - 1].expand_as(y), y)


def inv_lookup(y: Tensor, table: Tensor) -> Tensor:
    """`xsect.c::invLookup` restricted to a table's INCREASING part (every call here has
    ``y <= 1`` on ``S_Circ``, whose maximum sits third from last, so SWMM's
    decreasing-tail branch is never taken): ``locate`` by bisection, then linear inverse."""
    n_items = table.shape[0]
    dx = 1.0 / (float(n_items) - 1.0)
    n = n_items - 2 if float(table[n_items - 3]) > float(table[n_items - 1]) else n_items
    if bool(torch.any(y > table[n_items - 1])) and n < n_items:
        raise ValueError("xsect_tables.inv_lookup: the decreasing table tail is not modelled")
    inc = table[:n]
    # `locate(y, table, n-1)`: the highest j with table[j] <= y, clamped to [0, n-1].
    i = torch.searchsorted(inc, y.contiguous(), right=True) - 1
    i = torch.clamp(i, 0, n - 1)
    top = i >= n - 1
    i_safe = torch.clamp(i, max=n - 2)
    t0, t1 = inc[i_safe], inc[i_safe + 1]
    dy = t1 - t0
    dy_safe = torch.where(dy == 0.0, torch.ones_like(dy), dy)
    x0 = i_safe.to(F64) * dx
    x = torch.where(dy == 0.0, x0, x0 + (y - t0) * dx / dy_safe)
    x = torch.clamp(x, 0.0, 1.0)
    return torch.where(top, torch.full_like(x, (n - 1) * dx), x)


def _newton(theta0: Tensor, step, max_iter: int = 40) -> Tensor:
    """SWMM's fixed-form Newton loop: up to 40 steps, each element stops after the first
    step with ``|d| <= 1e-4``; an element that never stops returns its starting guess."""
    theta = theta0
    done = torch.zeros_like(theta0, dtype=torch.bool)
    for _ in range(max_iter):
        new, d = step(theta)
        theta = torch.where(done, theta, new)
        done = done | (torch.abs(d) <= 1e-4)
        if bool(done.all()):
            return theta
    return torch.where(done, theta, theta0)


def theta_of_alpha(alpha: Tensor) -> Tensor:
    """`xsect.c::getThetaOfAlpha`."""
    alpha_pos = torch.clamp(alpha, min=0.0)
    theta = torch.where(
        alpha > 0.04,
        1.2 + 5.08 * (alpha - 0.04) / 0.96,
        0.031715 - 12.79384 * alpha + 8.28479 * torch.sqrt(alpha_pos),
    )
    ap = (2.0 * _PI) * alpha

    def step(t):
        d = -(ap - t + torch.sin(t)) / (1.0 - torch.cos(t))
        d = torch.where(d > 1.0, torch.ones_like(d), d)
        return t - d, d

    return _newton(theta, step)


def theta_of_psi(psi: Tensor) -> Tensor:
    """`xsect.c::getThetaOfPsi`."""
    psi_pos = torch.clamp(psi, min=0.0)
    theta = torch.where(
        psi > 0.90,
        4.17 + 1.12 * (psi - 0.90) / 0.176,
        torch.where(
            psi > 0.5,
            3.14 + 1.03 * (psi - 0.5) / 0.4,
            torch.where(
                psi > 0.015,
                1.2 + 1.94 * (psi - 0.015) / 0.485,
                0.12103 - 55.5075 * psi + 15.62254 * torch.sqrt(psi_pos),
            ),
        ),
    )
    ap = (2.0 * _PI) * psi

    def step(t):
        t = torch.abs(t)
        tt = t - torch.sin(t)
        tt23 = tt ** (2.0 / 3.0)
        t3 = t ** (1.0 / 3.0)
        d = ap * t / t3 - tt * tt23
        d = d / (ap * (2.0 / 3.0) / t3 - (5.0 / 3.0) * tt23 * (1.0 - torch.cos(t)))
        return t - d, d

    return _newton(theta, step)


def _guard(x: Tensor, lo: float, hi: float, safe: float) -> tuple[Tensor, Tensor, Tensor]:
    """``(below, above, x_safe)``: the masks for SWMM's early returns, and ``x`` with those
    entries replaced by ``safe`` so the un-taken branch stays finite (and back-propagates
    a finite, zero-weighted gradient)."""
    below = x <= lo
    above = x >= hi
    return below, above, torch.where(below | above, torch.full_like(x, safe), x)


def y_circular(alpha: Tensor) -> Tensor:
    """`xsect.c::getYcircular`: ``Y/Yfull`` for ``A/Afull`` near the invert."""
    below, above, a = _guard(alpha, 0.0, 1.0, 0.5)
    tiny = a <= 1.0e-5
    a_tiny = torch.where(tiny, a, torch.full_like(a, 1.0e-5))
    t_tiny = (37.6911 * a_tiny) ** (1.0 / 3.0)
    theta = theta_of_alpha(torch.where(tiny, torch.full_like(a, 0.5), a))
    y = torch.where(tiny, t_tiny * t_tiny / 16.0, (1.0 - torch.cos(theta / 2.0)) / 2.0)
    y = torch.where(below, torch.zeros_like(y), y)
    return torch.where(above, torch.ones_like(y), y)


def s_circular(alpha: Tensor) -> Tensor:
    """`xsect.c::getScircular`: ``S/Sfull`` for ``A/Afull`` near the invert."""
    below, above, a = _guard(alpha, 0.0, 1.0, 0.5)
    tiny = a <= 1.0e-5
    a_tiny = torch.where(tiny, a, torch.full_like(a, 1.0e-5))
    t_tiny = (37.6911 * a_tiny) ** (1.0 / 3.0)
    theta = theta_of_alpha(torch.where(tiny, torch.full_like(a, 0.5), a))
    s = torch.where(
        tiny,
        t_tiny ** (13.0 / 3.0) / 124.4797,
        (theta - torch.sin(theta)) ** (5.0 / 3.0) / (2.0 * _PI) / theta ** (2.0 / 3.0),
    )
    s = torch.where(below, torch.zeros_like(s), s)
    return torch.where(above, torch.ones_like(s), s)


def a_circular(psi: Tensor) -> Tensor:
    """`xsect.c::getAcircular`: ``A/Afull`` for ``S/Sfull`` near the invert."""
    below, above, p = _guard(psi, 0.0, 1.0, 0.5)
    tiny = p <= 1.0e-6
    p_tiny = torch.where(tiny, p, torch.full_like(p, 1.0e-6))
    t_tiny = (124.4797 * p_tiny) ** (3.0 / 13.0)
    theta = theta_of_psi(torch.where(tiny, torch.full_like(p, 0.5), p))
    a = torch.where(tiny, t_tiny**3 / 37.6911, (theta - torch.sin(theta)) / (2.0 * _PI))
    a = torch.where(below, torch.zeros_like(a), a)
    return torch.where(above, torch.ones_like(a), a)


def y_of_a(alpha: Tensor) -> Tensor:
    """`circ_getYofA` / ``yFull``: table ``Y_Circ`` above ``alpha = 0.04``."""
    small = alpha < 0.04
    return torch.where(
        small,
        y_circular(torch.where(small, alpha, torch.zeros_like(alpha))),
        lookup(alpha, _Y),
    )


def s_of_a(alpha: Tensor) -> Tensor:
    """`circ_getSofA` / ``sFull``: table ``S_Circ`` above ``alpha = 0.04``."""
    small = alpha < 0.04
    return torch.where(
        small,
        s_circular(torch.where(small, alpha, torch.zeros_like(alpha))),
        lookup(alpha, _S),
    )


def a_of_s(psi: Tensor) -> Tensor:
    """`circ_getAofS` / ``aFull``: ``getAcircular`` up to ``psi = 0.015``, else the
    inverse of ``S_Circ``; ``0`` at ``0`` and ``1`` at ``psi >= 1``."""
    small = psi <= 0.015
    a = torch.where(
        small,
        a_circular(torch.where(small, psi, torch.zeros_like(psi))),
        inv_lookup(torch.where(small, torch.full_like(psi, 0.5), psi), _S),
    )
    a = torch.where(psi == 0.0, torch.zeros_like(a), a)
    return torch.where(psi >= 1.0, torch.ones_like(a), a)


def a_of_y(y_norm: Tensor) -> Tensor:
    """`circ_getAofY` / ``aFull`` (``xsect_getAofY``): table ``A_Circ``, quadratic on the
    first two segments; ``0`` at ``y <= 0``."""
    a = lookup(y_norm, _A)
    return torch.where(y_norm <= 0.0, torch.zeros_like(a), a)


#: ``S_Circ`` at ``alpha = 0.04``: above it the outlet area comes from the table alone.
_S_TABLE_FLOOR = S_CIRC[2]


def _outlet_alpha(psi: Tensor) -> Tensor:
    """The continuity root ``a2``: ``s_of_a(a2) == psi``.

    For ``psi > S_Circ[2]`` the root lies on the table (``alpha >= 0.04``, linear
    segments), where ``inv_lookup`` IS the exact inverse of ``lookup``. Below, ``s_of_a``
    is ``getScircular`` (smooth and increasing), inverted by `solve_monotone` on
    ``[0, 0.04]``; SWMM's own Newton (`findroot_Newton`, restarted every step from the
    previous area) converges to the same root.
    """
    table = psi > _S_TABLE_FLOOR
    a_table = inv_lookup(torch.where(table, psi, torch.full_like(psi, 0.5)), _S)
    low = (~table) & (psi > 0.0)
    if not bool(low.any()):
        return torch.where(psi > 0.0, a_table, torch.zeros_like(a_table))
    p_low = torch.where(low, psi, torch.full_like(psi, 0.5 * _S_TABLE_FLOOR))

    def residual(a, p):
        return s_circular(a) - p

    hi_s = float(s_circular(torch.tensor([0.04], dtype=F64)))
    p_low = torch.clamp(p_low, max=hi_s)
    a_low = solve_monotone(
        residual, torch.zeros_like(p_low), torch.full_like(p_low, 0.04), p_low,
        tol=1e-15, max_iter=200,
    )
    a = torch.where(table, a_table, a_low)
    return torch.where(psi > 0.0, a, torch.zeros_like(a))


def conduit_constants(diameter: Tensor, roughness: Tensor, slope: Tensor):
    """``(y_full, a_full, s_full, beta)`` in SWMM's internal feet units
    (`xsect.c::xsect_setParams` CIRCULAR, `link.c::conduit_validate`)."""
    y_full = diameter / LCF
    a_full = _PI / 4.0 * y_full * y_full
    r_full = 0.2500 * y_full
    s_full = a_full * r_full ** (2.0 / 3.0)
    beta = PHI * torch.sqrt(torch.abs(slope)) / roughness
    return y_full, a_full, s_full, beta


def psi_of_flow(q: Tensor, diameter: Tensor, roughness: Tensor, slope: Tensor) -> Tensor:
    """``S/Sfull`` SWMM's kinematic wave assigns to the discharge ``q`` (m3/s), in its own
    order of operations (`kinwave_execute`: ``qin = q/Qfull``, ``s = qin/Beta1``)."""
    _, _, s_full, beta = conduit_constants(diameter, roughness, slope)
    q_full = s_full * beta
    beta1 = beta / q_full
    qin = (q / QCF) / q_full
    return (qin / beta1) / s_full


def kinwave_steady(
    q: Tensor, diameter: Tensor, roughness: Tensor, slope: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """``(depth m, area m2, velocity m/s)`` SWMM 5.2 reports for a circular conduit at a
    kinematic-wave steady state carrying ``q`` (m3/s). ``area`` is the mean of the inlet and
    outlet areas, in SWMM's reporting convention (ft2 times ``QCF / LCF``), so
    ``area * length`` is SWMM's reported link volume. ``q > Qfull`` is the caller's to refuse.
    """
    y_full, a_full, _, _ = conduit_constants(diameter, roughness, slope)
    psi = psi_of_flow(q, diameter, roughness, slope)
    a1 = a_of_s(psi)
    a2 = _outlet_alpha(psi)
    depth_ft = 0.5 * (y_full * y_of_a(a1) + y_full * y_of_a(a2))
    area_ft2 = 0.5 * (a1 * a_full + a2 * a_full)
    # `link_getVelocity`: zero at depth <= 0.01 ft, else q / AofY(depth) (ft/s).
    a_at_depth = a_full * a_of_y(depth_ft / y_full)
    wet = (depth_ft > 0.01) & (a_at_depth > 1.0e-6)
    a_safe = torch.where(wet, a_at_depth, torch.ones_like(a_at_depth))
    velocity_fps = torch.where(wet, (q / QCF) / a_safe, torch.zeros_like(q))
    return depth_ft * LCF, area_ft2 * (QCF / LCF), velocity_fps * LCF


def full_flow(diameter: Tensor, roughness: Tensor, slope: Tensor) -> Tensor:
    """SWMM's ``Link.qFull = sFull * beta``, in m3/s: the discharge at which its kinematic
    wave declares the conduit full (inlet area ``aFull``)."""
    _, _, s_full, beta = conduit_constants(diameter, roughness, slope)
    return s_full * beta * QCF


#: `xsect.dat` W_Circ: top width / max width against depth / full depth (51 points).
W_CIRC = (
    0.0, .2800, .3919, .4750, .5426, .6000, .6499, .6940, .7332,
    .7684, .8000, .8285, .8542, .8773, .8980, .9165, .9330,
    .9474, .9600, .9708, .9798, .9871, .9928, .9968, .9992,
    1.000, .9992, .9968, .9928, .9871, .9798, .9708, .9600,
    .9474, .9330, .9165, .8980, .8773, .8542, .8285, .8000,
    .7684, .7332, .6940, .6499, .6000, .5426, .4750, .3919,
    .2800, .0,
)
#: `xsect.dat` R_Circ: hydraulic radius / full radius against depth / full depth.
R_CIRC = (
    .0100, .0528, .1048, .1556, .2052, .254, .3016, .3484, .3944, .4388,
    .4824, .5248, .5664, .6064, .6456, .6836, .7204, .7564, .7912, .8244,
    .8568, .888, .9176, .9464, .9736, 1.00, 1.024, 1.048, 1.070, 1.0912, 1.110, 1.1272,
    1.144, 1.1596, 1.174, 1.1848, 1.194, 1.2024, 1.210, 1.2148,
    1.217, 1.2172, 1.215, 1.2104, 1.203, 1.192, 1.178, 1.1584, 1.132, 1.094, 1.000,
)
_W = _table(W_CIRC)
_R = _table(R_CIRC)

#: SWMM's gravity in ft/s2 (`consts.h:37`).
GRAVITY_FT = 32.2


def area_of_depth(y: Tensor, y_full: Tensor, a_full: Tensor) -> Tensor:
    """`xsect.c::xsect_getAofY` (xsect.c:857) for CIRCULAR: flow area (ft2) at depth ``y`` (ft)."""
    y_safe = torch.clamp(y, min=0.0)
    return torch.where(y <= 0.0, torch.zeros_like(y_safe),
                       a_full * lookup(y_safe / y_full, _A))


def width_of_depth(y: Tensor, y_full: Tensor) -> Tensor:
    """`xsect.c::xsect_getWofY` (xsect.c:943) for CIRCULAR (``wMax = yFull``): top width (ft)."""
    y_safe = torch.clamp(y, min=0.0)
    return torch.where(y <= 0.0, torch.zeros_like(y_safe), y_full * lookup(y_safe / y_full, _W))


def radius_of_depth(y: Tensor, y_full: Tensor) -> Tensor:
    """`xsect.c::xsect_getRofY` (xsect.c:1031) for CIRCULAR (``rFull = yFull/4``):
    hydraulic radius (ft)."""
    y_safe = torch.clamp(y, min=0.0)
    return torch.where(y <= 0.0, torch.zeros_like(y_safe),
                       0.25 * y_full * lookup(y_safe / y_full, _R))


_FUDGE = 1.0e-4   # consts.h:72


def froude_number(v: Tensor, y: Tensor, y_full: Tensor, a_full: Tensor) -> Tensor:
    """`link.c::link_getFroude` (link.c:847-871): |v| / sqrt(g A/W) at depth y (ft, ft/s); 0 when
    ``y <= FUDGE`` or within FUDGE of the crown (closed conduit)."""
    # link.c:861-863: empty, or closed conduit within FUDGE of the crown
    dead = (y <= _FUDGE) | (y_full - y <= _FUDGE)
    y_safe = torch.where(dead, 0.5 * y_full, y)
    # link.c:866-869: hydraulic depth A/W, Fr = |v| / sqrt(g A/W)
    hd = area_of_depth(y_safe, y_full, a_full) / width_of_depth(y_safe, y_full)
    return torch.where(dead, torch.zeros_like(v), torch.abs(v) / torch.sqrt(GRAVITY_FT * hd))


def _critical_flow(y: Tensor, y_full: Tensor, a_full: Tensor) -> Tensor:
    """`xsect.c::getQcritical` (xsect.c:1612-1630): A sqrt(g A / W), 0 where W <= 0."""
    a = area_of_depth(y, y_full, a_full)
    w = width_of_depth(y, y_full)
    w_safe = torch.where(w > 0.0, w, torch.ones_like(w))
    return torch.where(w > 0.0, a * torch.sqrt(GRAVITY_FT * a / w_safe), torch.zeros_like(a))


def critical_depth(q: Tensor, y_full: Tensor, a_full: Tensor) -> Tensor:
    """`xsect.c::xsect_getYcrit` (xsect.c:1257-1319) -> `getYcritEnum` (xsect.c:1634-1696)
    for CIRCULAR: SWMM's 25-interval enumeration with linear interpolation (ft, from
    ``q >= 0`` in cfs)."""
    q, y_full, a_full = torch.broadcast_tensors(q, y_full, a_full)
    q2g = q * q / GRAVITY_FT                                # xsect.c:1267
    zero = q2g == 0.0
    # xsect.c:1305-1306 (CIRCULAR is the equivalent circle: r = 1, enumeration path)
    y0 = 1.01 * torch.pow(torch.where(zero, torch.ones_like(q2g), q2g) / y_full, 0.25)
    y0 = torch.where(y0 >= y_full, 0.97 * y_full, y0)
    dy = y_full / 25.0                                      # xsect.c:1660
    i1 = torch.floor(y0 / dy)                               # (int)(y0/dy)
    stations = torch.arange(26, dtype=q.dtype)              # i = 0..25
    qc = _critical_flow(stations * dy.unsqueeze(-1), y_full.unsqueeze(-1),
                        a_full.unsqueeze(-1))               # (..., 26)
    q_i1 = torch.gather(qc, -1, i1.long().unsqueeze(-1)).squeeze(-1)
    # xsect.c:1670-1686: scan UP from i1+1 when Qc(i1) < q, first i with qc[i] >= q;
    # SWMM's q0 is the previous station's Qc, i.e. qc[i-1]; none found -> yFull
    up = q_i1 < q
    yc_up = y_full.clone()
    found = torch.zeros_like(up)
    for i in range(1, 26):
        hit = up & ~found & (stations[i] > i1) & (qc[..., i] >= q)
        q_prev = qc[..., i - 1]
        frac = (q - q_prev) / torch.where(hit, qc[..., i] - q_prev, torch.ones_like(q))
        yc_up = torch.where(hit, (frac + (i - 1)) * dy, yc_up)
        found = found | hit
    # xsect.c:1690-1707 scan DOWN from i1-1 otherwise: first i with qc[i] < q;
    # SWMM's q0 is the previous (upper) station's Qc, i.e. qc[i+1]; none found -> 0
    yc_dn = torch.zeros_like(q)
    found_dn = torch.zeros_like(up)
    for i in range(24, -1, -1):
        hit = ~up & ~found_dn & (stations[i] < i1) & (qc[..., i] < q)
        q_next = qc[..., i + 1]
        frac = (q - qc[..., i]) / torch.where(hit, q_next - qc[..., i], torch.ones_like(q))
        yc_dn = torch.where(hit, (frac + i) * dy, yc_dn)
        found_dn = found_dn | hit
    yc = torch.where(up, yc_up, yc_dn)
    # xsect.c:1318 MIN(y, yFull); xsect.c:1268 q2g == 0 -> 0
    return torch.where(zero, torch.zeros_like(yc), torch.minimum(yc, y_full))


def normal_depth_ft(q: Tensor, y_full: Tensor, a_full: Tensor, s_full: Tensor,
                    beta: Tensor) -> Tensor:
    """`link.c::link_getYnorm` (link.c:783-804) for CIRCULAR: q clamped to
    ``qMax = 1.08 sFull beta``, ``A(S)`` by `circ_getAofS` (``aFull`` for ``psi >= 1``),
    ``Y(A)`` by `circ_getYofA`."""
    # link.c:797-798 (qMax = sMax beta, sMax = 1.08 sFull: xsect.c:248, link.c:1136)
    q = torch.minimum(torch.abs(q), 1.08 * s_full * beta)
    # link.c:800-802 (circ_getAofS: aFull for psi >= 1)
    psi = q / beta / s_full
    alpha = torch.where(psi >= 1.0, torch.ones_like(psi),
                        a_of_s(torch.clamp(psi, max=1.0)))
    y = y_full * y_of_a(alpha)
    return torch.where(q <= 0.0, torch.zeros_like(y), y)


__all__ = [
    "A_CIRC", "GRAVITY_FT", "LCF", "PHI", "QCF", "R_CIRC", "S_CIRC", "W_CIRC", "Y_CIRC",
    "a_of_s", "a_of_y", "area_of_depth", "conduit_constants", "critical_depth", "froude_number",
    "full_flow", "inv_lookup", "kinwave_steady", "lookup", "normal_depth_ft", "psi_of_flow",
    "radius_of_depth", "s_of_a", "width_of_depth", "y_of_a",
]
