"""EPANET 2.2's water-quality reactions, transcribed (`qualreact.c`, `input1.c`, tag v2.2): the
reaction coefficients of every pipe and tank (`adjustdata`), the bulk rate (`bulkrate`), the
wall's mass-transfer-limited rate coefficient (`piperate`, once per hydraulic step through
`ratecoeffs`) and the wall rate (`wallrate`).

Every function works on tensors (one value per pipe, tank or segment) and keeps their autograd
graph: the coefficients are differentiable inputs of the quality run (`QualityRouter`'s
`quality.bulk_coefficient`, `quality.wall_coefficient`, `quality.tank_coefficient`). The
branches EPANET takes on a value (`Re < 1`, `|kf| < |kw|`, a negative potential) are
`torch.where`s with the unselected branch kept finite, so no NaN reaches a gradient.

UNITS. Concentrations are in the file's quality unit (mg/L, ug/L); coefficients are SI per
second as `WaterQuality` documents them: bulk (quality unit)^(1-n)/s for the order n, wall m/s
(order 1) or quality unit x litre per m2 per s (order 0: mg/m2/s for mg/L). EPANET computes in
ft and its internal mass per ft3 (`c / Ucf[QUALITY]`, `Ucf[QUALITY] = 1 / LperFT3`): its bulk
unit factor `Bucf` (`getucf`) cancels exactly in the file's unit, but an order-0 wall rate
crosses from mass per ft2 to mass per ft3 through `LperFT3 = 28.317` against the exact ft3, so
its litres per m3 are `EPANET_LITRES_PER_M3 = 28.317 / 0.3048^3` (1000.0054), transcribed.
"""

from __future__ import annotations

import math

import torch

from noodl.apps.water.units import M_PER_FT

Tensor = torch.Tensor
F64 = torch.float64

#: EPANET's litres per ft3 (`types.h`, ``LperFT3 = 28.317``).
L_PER_FT3 = 28.317
#: The litres per m3 of EPANET's order-0 wall rate (module docstring): ``LperFT3`` over the
#: exact ft3, 1000.0054.
EPANET_LITRES_PER_M3 = L_PER_FT3 / M_PER_FT**3
#: `bulkrate`'s floor of a Michaelis-Menten denominator (`types.h`, ``TINY = 1e-6``, in
#: EPANET's internal mass per ft3) in the file's quality unit: ``1e-6 / LperFT3``.
BULK_TINY = 1e-6 / L_PER_FT3
#: `piperate`'s order-0 mass-transfer coefficient without a mass transfer (``BIG = 1e10``
#: ft/s, `types.h`), in m/s.
WALL_BIG = 1e10 * M_PER_FT
#: Millifeet over inches as a ratio of the same length: ``1000 * 0.0254 / 0.3048``.
_MILLIFEET_PER_INCH_RATIO = 1000.0 * 0.0254 / M_PER_FT


def _sgn(x: Tensor) -> Tensor:
    """EPANET's `SGN` (`types.h`): -1 below 0, else +1 (`SGN(0) = 1`)."""
    return torch.where(x < 0, -1.0, 1.0).to(F64)


def _wall_from_roughness(net, pipe, factor: float) -> float:
    """`adjustdata`'s wall coefficient from the ROUGHNESS CORRELATION `factor` (`input1.c`):
    H-W `F / C`, D-W `F / |ln(e/d)|`, C-M `F n`, 0 when the roughness or the diameter is not
    positive. EPANET takes it BEFORE `convertunits`, so D-W's `e/d` is in the FILE's units:
    mm over mm in an SI file, millifeet over inches in a US one (`e/d * 1000 * 0.0254 /
    0.3048`, with `net.options.flow_units`; SI without it). A D-W roughness not below the
    diameter (in the file's units) is refused by name: at `e = d` EPANET divides by
    `log(1) = 0`."""
    from noodl.apps.water.inp import _US_UNITS  # the reader imports the network module
    if not (pipe.roughness > 0.0 and pipe.diameter > 0.0):
        return 0.0
    if net.headloss == "H-W":
        return factor / pipe.roughness
    if net.headloss == "C-M":
        return factor * pipe.roughness
    ratio = pipe.roughness / pipe.diameter
    if net.options.flow_units in _US_UNITS:
        ratio *= _MILLIFEET_PER_INCH_RATIO
    if ratio >= 1.0:
        raise ValueError(
            f"quality: ROUGHNESS CORRELATION on pipe {pipe.name!r}, whose Darcy-Weisbach "
            f"roughness is not below its diameter in the file's units (e/d = {ratio:g}): "
            f"EPANET's adjustdata divides by |log(e/d)|, 0 at e = d")
    return factor / abs(math.log(ratio))


def pipe_coefficients(net, quality) -> tuple[Tensor, Tensor]:
    """`(kb, kw)` (n_pipes, `net.pipes` order, SI per second) of every pipe, check-valve pipes
    included (`adjustdata`, `input1.c`: `Type <= PIPE`): its own `[REACTIONS] BULK` and
    `WALL`, else the global; a pipe without its own WALL under a nonzero ROUGHNESS
    CORRELATION takes `_wall_from_roughness`."""
    kb, kw = [], []
    factor = quality.roughness_correlation
    for pipe in net.pipes:
        kb.append(quality.pipe_bulk.get(pipe.name, quality.global_bulk))
        if pipe.name in quality.pipe_wall:
            kw.append(quality.pipe_wall[pipe.name])
        elif factor == 0.0:
            kw.append(quality.global_wall)
        else:
            kw.append(_wall_from_roughness(net, pipe, factor))
    return torch.tensor(kb, dtype=F64), torch.tensor(kw, dtype=F64)


def tank_coefficients(net, quality) -> Tensor:
    """Every tank's bulk coefficient (n_tanks, `net.tanks` order, SI per second): its own
    `[REACTIONS] TANK`, else the global BULK (`adjustdata`, `input1.c`)."""
    return torch.tensor([quality.tank_bulk.get(t.name, quality.global_bulk) for t in net.tanks],
                        dtype=F64)


def bulk_rate(c: Tensor, kb: Tensor, order: float, limit: float, tiny: float) -> Tensor:
    """`bulkrate` (`qualreact.c`): the bulk reaction rate (quality unit per s) at the
    concentration `c` for the coefficient `kb`, the reaction `order` and the limiting
    potential `limit` (`Climit`, quality unit). Order 0: `kb`. Michaelis-Menten (order < 0):
    `kb c / c1`, `c1 = Climit + sgn(kb) c`, `c1` set to `sgn(c1) tiny` when `|c1| < tiny`
    (`tiny`: EPANET's `TINY` in the file's unit, `BULK_TINY`). Order n otherwise: `c1 = c`
    without a limiting potential, else `max(0, sgn(kb) (Climit - c))`; the potential `c1`
    (order 1), `c1 c` (order 2), `c1 max(0, c)^(n-1)`. A negative potential is 0. Where
    `kb` is 0 the rate is exactly 0 even if the potential is not finite (EPANET's `0 < n <
    1` at `c = 0`, which the router refuses), its derivative in `kb` kept: the potential."""
    if order == 0.0:
        potential = torch.ones_like(c)
    elif order < 0.0:
        c1 = limit + _sgn(kb) * c
        c1 = torch.where(c1.abs() < tiny, _sgn(c1) * tiny, c1)
        potential = c / c1
    else:
        if limit == 0.0:
            c1 = c
        else:
            c1 = torch.clamp(_sgn(kb) * (limit - c), min=0.0)
        if order == 1.0:
            potential = c1
        elif order == 2.0:
            potential = c1 * c
        else:
            base = torch.clamp(c, min=0.0)
            safe = torch.where(base > 0, base, 1.0)
            at_zero = 0.0 if order > 1.0 else (1.0 if order == 1.0 else math.inf)
            potential = c1 * torch.where(base > 0, safe ** (order - 1.0), at_zero)
    potential = torch.where(potential < 0, 0.0, potential)
    potential = torch.where((kb == 0) & ~torch.isfinite(potential), 0.0, potential)
    return kb * potential


def wall_coefficients(flow: Tensor, diameter: Tensor, length: Tensor, kw: Tensor, *,
                      order: float, viscosity: float, diffusivity: float) -> Tensor:
    """`ratecoeffs` and `piperate` (`qualreact.c`): every pipe's wall rate coefficient `Rc`
    from its `flow` (m3/s as EPANET's quality reads it, 0 on a closed link), `diameter` and
    `length` (m) and wall coefficient `kw`, for the wall `order` (0 or 1), the kinematic
    `viscosity` and the chemical's `diffusivity` (m2/s); 0 where `kw` is 0. With the
    Schmidt number `Sc = viscosity / diffusivity` (0 when the diffusivity is 0) and `Re =
    |q| / (pi d^2 / 4) d / viscosity`, the Sherwood number is 2 below Re 1, `0.0149 Re^0.88
    Sc^0.333` (Notter-Sleicher) from Re 2300, else `3.65 + 0.0668 y / (1 + 0.04 y^0.667)`
    (Graetz), `y = (d/L) Re Sc`; the mass transfer `kf = Sh D / d` (m/s). Order 0 gives `kf`;
    order 1 the apparent coefficient `(4/d) kw kf / (kf + |kw|)` (1/s). Without a mass
    transfer (`Sc = 0`), order 0 gives `BIG` (`WALL_BIG`), order 1 `kw 4 / d`. Order 1's
    forms are 0 at `kw = 0` by themselves, and are left so: their derivative there is
    kept."""
    sc = viscosity / diffusivity if diffusivity > 0.0 else 0.0
    if sc == 0.0:
        if order == 0.0:
            return torch.where(kw != 0, WALL_BIG, 0.0).to(F64)
        return kw * (4.0 / diameter)
    area = math.pi * diameter * diameter / 4.0
    re = flow.abs() / area * diameter / viscosity
    turbulent = re >= 2300.0
    laminar = (re >= 1.0) & ~turbulent
    re_safe = torch.where(re >= 1.0, re, 1.0)
    y = diameter / length * re_safe * sc
    sh = torch.where(
        turbulent, 0.0149 * re_safe**0.88 * sc**0.333,
        torch.where(laminar, 3.65 + 0.0668 * y / (1.0 + 0.04 * y**0.667), 2.0))
    kf = sh * diffusivity / diameter
    if order == 0.0:
        return torch.where(kw != 0, kf, 0.0)
    return (4.0 / diameter) * kw * kf / (kf + kw.abs())


def wall_rate(c: Tensor, diameter: Tensor, kw: Tensor, rc: Tensor, *, order: float) -> Tensor:
    """`wallrate` (`qualreact.c`): the wall reaction rate (quality unit per s) at the
    concentration `c` of a pipe of `diameter` (m) with the wall coefficient `kw` and the rate
    coefficient `rc` (`wall_coefficients`). Order 0: the mass transfer `sgn(kw) c kf` or the
    reaction `kw / EPANET_LITRES_PER_M3`, whichever is smaller in magnitude (the reaction
    mass-transfer limited), times `4 / d`; order 1 `c Rc`. 0 where `kw` or `d` is 0:
    EPANET's `if (kw == 0.0 || d == 0.0) return 0` is kept for `d` only, since both forms
    are exactly 0 at `kw = 0` for a finite `c` by themselves (order 0 takes the reaction
    `kw / L = 0`; order 1's `Rc` is 0) and are left so, keeping their derivative there; the
    test would matter only for a non-finite `c`, which the router never holds."""
    if order == 0.0:
        kf = _sgn(kw) * c * rc
        reaction = kw / EPANET_LITRES_PER_M3
        limited = torch.where(kf.abs() < reaction.abs(), kf, reaction)
        safe_d = torch.where(diameter != 0, diameter, 1.0)
        rate = limited * 4.0 / safe_d
    else:
        rate = c * rc
    return torch.where(diameter != 0, rate, 0.0)


__all__ = ["BULK_TINY", "EPANET_LITRES_PER_M3", "L_PER_FT3", "WALL_BIG", "bulk_rate",
           "pipe_coefficients", "tank_coefficients", "wall_coefficients", "wall_rate"]
