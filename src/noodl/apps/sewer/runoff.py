"""Subcatchment hydrology of a SWMM model: rain gages, runoff from the three subareas of each
subcatchment and infiltration, as SWMM 5.2.4 computes them, and the lateral inflow they give
the drainage network.

``Hydrology`` holds the data (`RainGage`, `Subcatchment`, `Evaporation`, `Adjustments` and the
`Horton`, `GreenAmpt` and `CurveNumber` infiltration parameters); `RunoffModel` runs it;
`RunoffInflows` combines it with the file's `InflowSchedule` and is the ``inflow`` driver of
`simulate_dynamic_wave` (``model.inflow_schedule`` of a model built from a file with
``[SUBCATCHMENTS]``). `inp_hydrology.read_hydrology` reads the sections. What follows is
SWMM 5.2.4's behaviour, operation for operation (`subcatch.c`, `runoff.c`, `infil.c`,
`gage.c`, `climate.c`, `odesolve.c`, `routing.c`); the tests in
``tests/verification/test_sewer_runoff_parity.py`` compare it with the engine through pyswmm.

Clocks. SWMM keeps a runoff clock (``OldRunoffTime``, ``NewRunoffTime``, ms) that is NOT the
routing clock. Whenever the next routing step ends after ``NewRunoffTime``, runoff steps are
taken (`execRouting`, swmm5.c:546), so the runoff steps follow the weather alone: a step is
``WET_STEP`` when it is raining or a subcatchment had runoff at the previous step, else
``DRY_STEP``, shortened to the next change of rainfall or of evaporation (`runoff_getTimeStep`,
runoff.c:323), and the last one is cut at the end of the simulation. A routing step then
adds the runoff of each subcatchment to its outlet node, interpolated linearly between the
runoff at ``OldRunoffTime`` and at ``NewRunoffTime`` at the fraction
``f = (OldRoutingTime - OldRunoffTime) / (NewRunoffTime - OldRunoffTime)`` clamped to [0, 1]
(`addWetWeatherInflows`, routing.c:579), with ``OldRoutingTime`` the START of the routing step
(no 1 ms offset, unlike the time series of `inflows.py`). A runoff step sees the date
``START + elapsed + 1 ms`` (`getDateTime`, swmm5.c:1551). The runoff therefore depends on the
routing step through its length only: ``RunoffInflows`` is called by `simulate_dynamic_wave`
with each step's start and length (``step_aware``) and takes the runoff steps that routing
step needs.

Rain gage (`gage.c`). The series gives a rainfall per recording interval ``rainInterval``
starting at each entry (``INTENSITY`` in/hr or mm/hr; ``VOLUME`` a depth per interval;
``CUMULATIVE`` the running total, restarting when it falls): the intensity applies from an
entry's date for ``rainInterval``, and is 0 until the next non-zero entry (zero entries are
skipped, ``getNextRainfall``). A record that starts after the simulation starts is 0 before it.
Gages that share a series (co-gages) take the rainfall of the first of them. The snow catch
factor is 1 (it multiplies snowfall only; a file that needs temperatures is refused), and the
largest WET_STEP is the smallest ``rainInterval`` (`gage_validate`).

Subareas (`subcatch.c`). Each subcatchment is split in an impervious subarea without
depression storage (fraction ``f_imp * PctZero/100``), an impervious one with it
(``f_imp * (1 - PctZero/100)``) and a pervious one (``1 - f_imp``); runoff is a nonlinear
reservoir ``q = alpha (d - d_s)^(5/3)`` with ``alpha = 1.49 W sqrt(S) / (A n)`` per unit area
(SWMM's constant 1.49 in feet and cfs, ``MEXP`` = 1.6666667), integrated over the step by the
Cash-Karp Runge-Kutta of `odesolve.c` with tolerance ``ODETOL`` = 1e-4 and its adaptive
step. ``RouteTo`` IMPERVIOUS or PERVIOUS sends the fraction ``PctRouted`` of one subarea's
runoff of the previous step onto the other; ``OUTLET`` sends it to the subcatchment's outlet
node or, when the outlet is a subcatchment, onto it as runon (the previous step's runoff
too: runon is lagged by one runoff step, so the order of subcatchments never matters).
Evaporation takes ``min(depth / dt, rate)`` from each subarea's ponded water first, then
infiltration (pervious only) takes what the model allows of ``rain + runon + ponded water``.

Infiltration (`infil.c`): `Horton`, modified Horton, `GreenAmpt` (also modified) and
`CurveNumber`, each with SWMM's state (cumulative infiltration, time on the curve,
upper-zone moisture) and its soil recovery in dry weather. A subcatchment with no
``[INFILTRATION]`` line has none (SWMM's zeroed Horton parameters).

Units. The engine runs in SWMM's own units (ft, ft2, ft/s, cfs, with its conversion factors,
``Ucf`` in swmm5.c) so that every threshold (ZERO = 1e-10 ft, MIN_TOTAL_DEPTH = 0.004167 ft)
is SWMM's; the data classes hold SI values (m, m2, m/s, s) and the inflow is m3/s
(cfs times ``QCF``).

Differentiability. The runoff of a ``Hydrology`` whose parameters are torch tensors is
a differentiable function of them: the engine is plain arithmetic in which tensors flow,
and the branches (rain on or off, ponding above depression storage, saturation of the
Green-Ampt or Curve Number surface, the adaptive step of the Runge-Kutta, which is held
fixed when differentiating) are taken on the values, so the gradient is that of the branch
that was taken -- exact in each smooth region and a one-sided derivative at a switch.
``Hydrology.with_gradients`` makes leaf tensors of the chosen parameters. The rain
(series values, interval, snow catch factor) and the step sequence are not differentiable.
``simulate_dynamic_wave`` carries the gradient of the runoff into the routed depths and flows
(the lateral inflow is a differentiable driver). Not differentiable: the runoff clock and
every comparison on it, and the Curve Number clamp 10..99 outside its range.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from noodl.apps.sewer.inflows import SAMPLE_OFFSET, InflowSchedule, TimeSeries
from noodl.apps.sewer.xsect_tables import LCF, QCF

if TYPE_CHECKING:
    from noodl.apps.sewer.rdii import Rdii

F64 = torch.float64
Number = float | torch.Tensor

#: subcatch.c `MCOEFF`, `MEXP` and `ODETOL`.
MCOEFF = 1.49
MEXP = 1.6666667
ODETOL = 0.0001
#: consts.h `ZERO`, `TINY`, `MIN_TOTAL_DEPTH` (ft).
ZERO = 1.0e-10
TINY = 1.0e-6
MIN_TOTAL_DEPTH = 0.004167
#: odesolve.c.
_MAXSTP = 10000
_ODE_TINY = 1.0e-30
_SAFETY = 0.9
_PGROW = -0.2
_PSHRNK = -0.25
_ERRCON = 1.89e-4
#: swmm5.c ``Ucf``: (US, SI) divisors to feet, ft/s ...
UCF_RAINFALL = (43200.0, 1097280.0)         # in/hr, mm/hr -> ft/s
UCF_RAINDEPTH = (12.0, 304.8)               # in, mm -> ft
UCF_EVAPRATE = (1036800.0, 26334720.0)      # in/day, mm/day -> ft/s
UCF_LANDAREA = (2.2956e-5, 0.92903e-5)      # acres, hectares per ft2
UCF_LENGTH = (1.0, 0.3048)                  # ft, m per ft
#: gage.c `OneSecond` (days) in seconds.
ONE_SECOND = 1.1574074e-5 * 86400.0
#: SWMM's 60 s * ... `SECperDAY`.
SEC_PER_DAY = 86400.0

ROUTE_OUTLET, ROUTE_IMPERVIOUS, ROUTE_PERVIOUS = "outlet", "impervious", "pervious"
RAIN_KINDS = ("intensity", "volume", "cumulative")
_IMPERV0, _IMPERV1, _PERV = 0, 1, 2


# --------------------------------------------------------------------------------------
# numeric helpers that work on floats and on torch scalars
# --------------------------------------------------------------------------------------
def _exp(x):
    return torch.exp(x) if isinstance(x, torch.Tensor) else math.exp(x)


def _log(x):
    return torch.log(x) if isinstance(x, torch.Tensor) else math.log(x)


def _sqrt(x):
    return torch.sqrt(x) if isinstance(x, torch.Tensor) else math.sqrt(x)


def _val(x) -> float:
    """The float value of a number or torch scalar (detached)."""
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Horton:
    """Horton infiltration (`[INFILTRATION]` ``HORTON`` or ``MODIFIED_HORTON``).

    `f0` and `fmin` are the maximum and minimum infiltration rates (m/s), `decay` the decay
    constant (1/s), `dry_time` the time (s) for a fully wet soil to dry (SWMM's drying time:
    to 98 % along the exponential recovery curve) and `max_volume` the optional maximum
    infiltration volume (m; 0 for none). `modified` selects SWMM's modified Horton method
    (cumulative infiltration instead of time on the curve). Any field may be a torch scalar."""

    f0: Number
    fmin: Number
    decay: Number
    dry_time: Number
    max_volume: Number = 0.0
    modified: bool = False


@dataclasses.dataclass(frozen=True)
class GreenAmpt:
    """Green-Ampt infiltration (``GREEN_AMPT`` or ``MODIFIED_GREEN_AMPT``): `suction` the
    capillary suction head (m), `conductivity` the saturated hydraulic conductivity (m/s) and
    `initial_deficit` the initial moisture deficit (fraction of the upper zone's volume,
    0 to 1). `modified` selects SWMM's modified method (a new event does not reset the
    moisture deficit while the rain stays below the conductivity)."""

    suction: Number
    conductivity: Number
    initial_deficit: Number
    modified: bool = False


@dataclasses.dataclass(frozen=True)
class CurveNumber:
    """SCS Curve Number infiltration (``CURVE_NUMBER``): `curve_number` (clamped to 10..99, as
    SWMM does) and `dry_time` (s), the time for the soil's storage to regenerate; the
    obsolete conductivity of the file's line is not used by SWMM and is not kept."""

    curve_number: Number
    dry_time: Number


Infiltration = Horton | GreenAmpt | CurveNumber


@dataclasses.dataclass(frozen=True)
class RainGage:
    """A rain gage with a time series (`[RAINGAGES]` ``TIMESERIES``): `series` holds the rainfall
    in the file's rain units (in or mm per interval, or per hour) at the dates (s since the
    simulation START) of its entries, `kind` is ``intensity``, ``volume`` or ``cumulative``
    and `interval` the recording interval (s). `snow_factor` (SWMM's snow catch deficiency
    factor) multiplies snowfall only and is 1 here (there is no snowfall)."""

    name: str
    series: TimeSeries
    kind: str = "volume"
    interval: float = 3600.0
    snow_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in RAIN_KINDS:
            raise ValueError(f"rain gage {self.name!r}: kind must be one of {RAIN_KINDS}, "
                             f"got {self.kind!r}")
        if not self.interval > 0:
            raise ValueError(f"rain gage {self.name!r}: the recording interval must be > 0")


@dataclasses.dataclass(frozen=True)
class Subareas:
    """A subcatchment's `[SUBAREAS]` line: Manning roughness `n_imperv` and `n_perv`,
    depression storage `s_imperv` and `s_perv` (m), `pct_zero` (percent of the impervious area
    with no depression storage), `route_to` (``outlet``, ``impervious`` or ``pervious``) and
    `pct_routed` (percent of the routed subarea's runoff that goes to the other subarea).
    Any number may be a torch scalar."""

    n_imperv: Number
    n_perv: Number
    s_imperv: Number
    s_perv: Number
    pct_zero: Number = 25.0
    route_to: str = ROUTE_OUTLET
    pct_routed: Number = 100.0


@dataclasses.dataclass(frozen=True)
class Subcatchment:
    """A subcatchment (`[SUBCATCHMENTS]` with its `[SUBAREAS]` and `[INFILTRATION]` lines).

    `area` (m2), `pct_imperv` (percent, capped at 100), `width` (m, the characteristic width of
    overland flow) and `slope` (m/m) are the geometry; `gage` names the rain gage; `outlet` is a
    node, or a subcatchment when `outlet_is_subcatchment`. `subareas` is ``None`` when the file
    has no `[SUBAREAS]` line for it (SWMM then has no runoff from it). `infiltration` is
    ``None`` for no infiltration. `n_perv_pattern`, `storage_pattern` and `infiltration_pattern`
    are the monthly adjustment factors (12 numbers) of `[ADJUSTMENTS]` ``N-PERV``,
    ``DSTORE`` and ``INFIL``. Numbers may be torch scalars (see the module docstring)."""

    name: str
    gage: str
    outlet: str
    area: Number
    pct_imperv: Number
    width: Number
    slope: Number
    subareas: Subareas | None = None
    infiltration: Infiltration | None = None
    outlet_is_subcatchment: bool = False
    n_perv_pattern: tuple[float, ...] | None = None
    storage_pattern: tuple[float, ...] | None = None
    infiltration_pattern: tuple[float, ...] | None = None


@dataclasses.dataclass(frozen=True)
class Evaporation:
    """Potential evaporation of the subcatchments (`[EVAPORATION]`): `kind` ``constant`` or
    ``monthly`` (`monthly` holds the 12 monthly rates, in/day or mm/day) or ``timeseries``
    (`series`: dates in s since START, rates in/day or mm/day, a step function), `dry_only`
    allows evaporation only when it is not raining and `recovery` the optional monthly
    factors of ``RECOVERY`` that scale the soil recovery of the infiltration models."""

    kind: str = "constant"
    monthly: tuple[float, ...] = (0.0,) * 12
    series: TimeSeries | None = None
    dry_only: bool = False
    recovery: tuple[float, ...] | None = None


@dataclasses.dataclass(frozen=True)
class Adjustments:
    """`[ADJUSTMENTS]`: monthly `evaporation` (added to the rate, in/day or mm/day), `rainfall`
    and `conductivity` (multipliers, 12 each)."""

    evaporation: tuple[float, ...] = (0.0,) * 12
    rainfall: tuple[float, ...] = (1.0,) * 12
    conductivity: tuple[float, ...] = (1.0,) * 12


@dataclasses.dataclass(frozen=True)
class Hydrology:
    """The hydrology of a SWMM file: `gages`, `subcatchments`, `evaporation` and `adjustments`,
    on the calendar that starts at `start` and lasts `total_duration` s, with the runoff steps
    `wet_step` and `dry_step` (s; `wet_step` already lowered to the smallest rain interval,
    SWMM's ``gage_validate``, and `dry_step` raised to it). `us_units` says whether the rain
    and evaporation values of the file are in inches (else millimetres). `ignore_rainfall` is
    SWMM's IGNORE_RAINFALL. `rdii` is the file's RDII (`rdii.Rdii`; its unit hydrograph sets
    read rain gages, which SWMM then counts as used), or ``None``."""

    gages: tuple[RainGage, ...]
    subcatchments: tuple[Subcatchment, ...]
    start: datetime.datetime
    total_duration: float
    wet_step: float = 300.0
    dry_step: float = 3600.0
    us_units: bool = True
    evaporation: Evaporation = Evaporation()
    adjustments: Adjustments = Adjustments()
    ignore_rainfall: bool = False
    rdii: Rdii | None = None

    def __post_init__(self) -> None:
        gages = {g.name for g in self.gages}
        names = [s.name for s in self.subcatchments]
        if self.rdii is not None:
            for name in self.rdii.gage_names():
                if name not in gages:
                    raise ValueError(f"unit hydrograph set reads the rain gage {name!r}, "
                                     f"which is not in `gages`")
        if len(set(names)) != len(names):
            raise ValueError("Hydrology: subcatchment names must be unique")
        for s in self.subcatchments:
            if s.gage not in gages:
                raise ValueError(f"subcatchment {s.name!r} names the rain gage {s.gage!r}, "
                                 f"which is not in `gages`")
            if s.outlet_is_subcatchment and s.outlet not in names:
                raise ValueError(f"subcatchment {s.name!r}: outlet {s.outlet!r} is not a "
                                 f"subcatchment")
        if not self.total_duration > 0:
            raise ValueError("Hydrology: total_duration must be > 0 s")

    def outlet_nodes(self) -> tuple[str, ...]:
        """The nodes that receive runoff, in order of first appearance."""
        return tuple(dict.fromkeys(s.outlet for s in self.subcatchments
                                   if not s.outlet_is_subcatchment))

    def with_duration(self, total_duration: float) -> Hydrology:
        """The same hydrology over `total_duration` s: SWMM cuts the last runoff step at the
        end of the simulation (``TotalDuration``, runoff.c:158), so a shorter run differs from
        a prefix of a longer one in its last runoff step."""
        return dataclasses.replace(self, total_duration=float(total_duration))

    def with_gradients(self, fields: Sequence[str] = ("area", "pct_imperv", "width", "slope"),
                       subareas: Sequence[str] = ("n_imperv", "n_perv", "s_imperv", "s_perv"),
                       infiltration: bool = True, only: Sequence[str] | None = None
                       ) -> tuple[Hydrology, dict[str, torch.Tensor]]:
        """`(hydrology, leaves)`: this hydrology with the listed parameters of every
        subcatchment (or of the subcatchments `only` names) replaced by float64 leaf tensors
        that require gradients. `fields` are `Subcatchment` fields, `subareas` `Subareas`
        fields and `infiltration` the infiltration parameters (every numeric field of its
        model). `leaves` maps ``"<subcatchment>.<field>"`` to the tensor, e.g. ``"S1.area"``,
        ``"S1.subareas.n_perv"``, ``"S1.infiltration.f0"``."""
        leaves: dict[str, torch.Tensor] = {}

        def leaf(key: str, value) -> torch.Tensor:
            t = torch.as_tensor(_val(value), dtype=F64).clone().requires_grad_(True)
            leaves[key] = t
            return t

        new = []
        for s in self.subcatchments:
            if only is not None and s.name not in only:
                new.append(s)
                continue
            changes = {f: leaf(f"{s.name}.{f}", getattr(s, f)) for f in fields}
            if s.subareas is not None:
                sub = {f: leaf(f"{s.name}.subareas.{f}", getattr(s.subareas, f))
                       for f in subareas}
                changes["subareas"] = dataclasses.replace(s.subareas, **sub)
            if infiltration and s.infiltration is not None:
                inf = {f.name: leaf(f"{s.name}.infiltration.{f.name}",
                                    getattr(s.infiltration, f.name))
                       for f in dataclasses.fields(s.infiltration) if f.name != "modified"}
                changes["infiltration"] = dataclasses.replace(s.infiltration, **inf)
            new.append(dataclasses.replace(s, **changes))
        return dataclasses.replace(self, subcatchments=tuple(new)), leaves


# --------------------------------------------------------------------------------------
# the Runge-Kutta solver of odesolve.c (one equation)
# --------------------------------------------------------------------------------------
_B21 = 0.2
_B31, _B32 = 3.0 / 40.0, 9.0 / 40.0
_B41, _B42, _B43 = 0.3, -0.9, 1.2
_B51, _B52, _B53, _B54 = -11.0 / 54.0, 2.5, -70.0 / 27.0, 35.0 / 27.0
_B61, _B62, _B63 = 1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0
_B64, _B65 = 44275.0 / 110592.0, 253.0 / 4096.0
_C1, _C3, _C4, _C6 = 37.0 / 378.0, 250.0 / 621.0, 125.0 / 594.0, 512.0 / 1771.0
_DC5 = -277.0 / 14336.0
_DC1, _DC3 = _C1 - 2825.0 / 27648.0, _C3 - 18575.0 / 48384.0
_DC4, _DC6 = _C4 - 13525.0 / 55296.0, _C6 - 0.25


def _rkck(y, dydx, h: float, derivs):
    """One Cash-Karp step of size `h` from `y` (derivative `dydx`): `(y_new, error)`."""
    ak2 = derivs(y + _B21 * h * dydx)
    ak3 = derivs(y + h * (_B31 * dydx + _B32 * ak2))
    ak4 = derivs(y + h * (_B41 * dydx + _B42 * ak2 + _B43 * ak3))
    ak5 = derivs(y + h * (_B51 * dydx + _B52 * ak2 + _B53 * ak3 + _B54 * ak4))
    ak6 = derivs(y + h * (_B61 * dydx + _B62 * ak2 + _B63 * ak3 + _B64 * ak4 + _B65 * ak5))
    y_new = y + h * (_C1 * dydx + _C3 * ak3 + _C4 * ak4 + _C6 * ak6)
    err = h * (_DC1 * dydx + _DC3 * ak3 + _DC4 * ak4 + _DC5 * ak5 + _DC6 * ak6)
    return y_new, err


def odesolve_integrate(y, x1: float, x2: float, eps: float, h1: float,
                       derivs: Callable[[Number], Number]):
    """Integrate ``dy/dx = derivs(y)`` from `x1` to `x2` with accuracy `eps` and the first step
    guess `h1`, by Cash-Karp Runge-Kutta with adaptive step control: SWMM's
    ``odesolve_integrate`` (odesolve.c:85) for one equation. Returns the new `y`; if the
    integration fails (a step underflows, or more than 10000 steps) `y` is returned unchanged,
    as SWMM leaves ``ystart``. The step sizes are decided on the values (held fixed when `y`
    is a tensor with a graph)."""
    y_start = y
    x, h = x1, h1
    for _ in range(_MAXSTP):
        dydx = derivs(y)
        yscal = abs(_val(y)) + abs(_val(dydx) * h) + _ODE_TINY
        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x
        xold = x
        while True:                                             # rkqs
            y_new, err = _rkck(y, dydx, h, derivs)
            errmax = abs(_val(err) / yscal) / eps
            if errmax > 1.0:
                htemp = _SAFETY * h * errmax ** _PSHRNK
                if h >= 0:
                    h = htemp if htemp > 0.1 * h else 0.1 * h
                else:
                    h = htemp if htemp < 0.1 * h else 0.1 * h
                if xold + h == xold:
                    return y_start
                continue
            hnext = _SAFETY * h * errmax ** _PGROW if errmax > _ERRCON else 5.0 * h
            x = x + h
            y = y_new
            break
        if (x - x2) * (x2 - x1) >= 0.0:
            return y
        if abs(hnext) <= 0.0:
            return y_start
        h = hnext
    return y_start


# --------------------------------------------------------------------------------------
# infiltration states (infil.c)
# --------------------------------------------------------------------------------------
class _Env:
    """The climate values an infiltration call reads: the conductivity factor of the month
    (``InfilFactor``) and the soil recovery factor (``Evap.recoveryFactor``)."""

    __slots__ = ("hydcon", "recovery")

    def __init__(self) -> None:
        self.hydcon = 1.0
        self.recovery = 1.0


class _HortonState:
    """infil.c `horton_getInfil` and `modHorton_getInfil` with their state."""

    def __init__(self, p: Horton) -> None:
        self.modified = p.modified
        self.f0 = p.f0 / LCF
        self.fmin = p.fmin / LCF
        self.decay = p.decay
        dry = p.dry_time
        if not (dry != 0.0):
            dry = TINY * SEC_PER_DAY
        self.regen = -math.log(1.0 - 0.98) / dry
        self.fmax = p.max_volume / LCF
        self.tp = 0.0
        self.fe = 0.0

    def get_infil(self, tstep: float, rainfall, runon, depth, env: _Env):
        irate = rainfall + runon
        f0 = self.f0 * env.hydcon
        fmin = self.fmin * env.hydcon
        df = f0 - fmin
        kd = self.decay
        kr = self.regen * env.recovery
        if df < 0.0 or kd < 0.0 or kr < 0.0:
            return 0.0
        if df == 0.0 or kd == 0.0:
            fp = f0
            fa = irate + depth / tstep
            if fp > fa:
                fp = fa
            return max(0.0, fp)
        fa = irate + depth / tstep
        return (self._modified(tstep, fa, f0, fmin, df, kd, kr) if self.modified
                else self._classic(tstep, fa, f0, fmin, df, kd, kr))

    def _classic(self, tstep, fa, f0, fmin, df, kd, kr):
        fp = 0.0
        tp = self.tp
        fmax = self.fmax
        if fa > ZERO:
            t1 = tp + tstep
            tlim = 16.0 / kd
            if tp >= tlim:
                big_fp = fmin * tp + df / kd
                f1 = big_fp + fmin * tstep
            else:
                big_fp = fmin * tp + df / kd * (1.0 - _exp(-kd * tp))
                f1 = fmin * t1 + df / kd * (1.0 - _exp(-kd * t1))
            fp = (f1 - big_fp) / tstep
            fp = max(fp, fmin)
            if fp > fa:
                fp = fa
            if t1 > tlim:
                tp = t1
            elif fp < fa:
                tp = t1
            else:
                f1 = big_fp + fp * tstep
                tp = tp + tstep / 2.0
                for _ in range(20):
                    kt = min(60.0, kd * tp)
                    ex = _exp(-kt)
                    ff = fmin * tp + df / kd * (1.0 - ex) - f1
                    ff1 = fmin + df * ex
                    r = ff / ff1
                    tp = tp - r
                    if abs(_val(r)) <= 0.001 * tstep:
                        break
            if _val(fmax) > 0.0:
                if self.fe + fp * tstep > fmax:
                    fp = (fmax - self.fe) / tstep
                fp = max(fp, 0.0)
                self.fe = self.fe + fp * tstep
        elif kr > 0.0:
            r = math.exp(-_val(kr) * tstep) if not isinstance(kr, torch.Tensor) \
                else torch.exp(-kr * tstep)
            tp = 1.0 - _exp(-kd * tp)
            tp = -_log(1.0 - r * tp) / kd
            if _val(fmax) > 0.0:
                self.fe = fmin * tp + (df / kd) * (1.0 - _exp(-kd * tp))
        self.tp = tp
        return fp

    def _modified(self, tstep, fa, f0, fmin, df, kd, kr):
        f = 0.0
        if fa > ZERO:
            if _val(self.fmax) > 0.0 and self.fe >= self.fmax:
                return 0.0
            fp = f0 - kd * self.fe
            fp = max(fp, fmin)
            f = min(fa, fp)
            self.fe = self.fe + max(f - fmin, 0.0) * tstep
            if _val(self.fmax) > 0.0:
                self.fe = max(self.fe, self.fmax)
        elif kr > 0.0:
            self.fe = self.fe * _exp(-kr * tstep)
            self.fe = max(self.fe, 0.0)
        return f


def _grnampt_f2(f1, c1, ks, ts):
    """infil.c `grnampt_getF2`: the cumulative infiltration after `ts` s from `f1`."""
    f2 = f1
    f2min = f1 + ks * ts
    if c1 == 0.0:
        return f2min
    if ts < 10.0 and f1 > 0.01 * c1:
        f2 = f1 + ks * (1.0 + c1 / f1) * ts
        return max(f2, f2min)
    c2 = c1 * _log(f1 + c1) - ks * ts
    for _ in range(20):
        df2 = (f2 - f1 - c1 * _log(f2 + c1) + c2) / (1.0 - c1 / (f2 + c1))
        if abs(_val(df2)) < 0.00001:
            return max(f2, f2min)
        f2 = f2 - df2
    return f2min


class _GreenAmptState:
    """infil.c `grnampt_getInfil` and its two helpers, with the state."""

    def __init__(self, p: GreenAmpt) -> None:
        self.modified = p.modified
        self.s = p.suction / LCF
        self.ks = p.conductivity / LCF
        self.imd_max = p.initial_deficit
        ksat = self.ks * 12.0 * 3600.0
        self.lu = 4.0 * _sqrt(ksat) / 12.0
        self.imd = self.imd_max
        self.fu = 0.0
        self.f = 0.0
        self.sat = False
        self.t = 0.0

    def get_infil(self, tstep: float, rainfall, runon, depth, env: _Env):
        irate = rainfall + runon
        fumax = self.imd_max * self.lu * math.sqrt(env.hydcon)
        self.t = self.t - tstep
        if self.sat:
            return self._sat(tstep, irate, depth, env)
        return self._unsat(tstep, irate, depth, env, fumax)

    def _unsat(self, tstep, irate, depth, env, fumax):
        ks = self.ks * env.hydcon
        lu = self.lu * math.sqrt(env.hydcon)
        ia = irate + depth / tstep
        if ia < ZERO:
            ia = 0.0
        if ia == 0.0:
            if self.fu <= 0.0:
                return 0.0
            kr = lu / 90000.0 * env.recovery
            d_f = kr * fumax * tstep
            self.f = self.f - d_f
            self.fu = self.fu - d_f
            if self.fu <= 0.0:
                self.fu = 0.0
                self.f = 0.0
                self.imd = self.imd_max
                return 0.0
            if self.t <= 0.0:
                self.imd = (fumax - self.fu) / lu
                self.f = 0.0
            return 0.0
        if ia <= ks:
            d_f = ia * tstep
            self.f = self.f + d_f
            self.fu = min(self.fu + d_f, fumax)
            if (not self.modified) and self.t <= 0.0:
                self.imd = (fumax - self.fu) / lu
                self.f = 0.0
            return ia
        self.t = 5400.0 / lu / env.recovery
        fs = ks * (self.s + depth) * self.imd / (ia - ks)
        if self.f > fs:
            self.sat = True
            return self._sat(tstep, irate, depth, env)
        if self.f + ia * tstep < fs:
            d_f = ia * tstep
            self.f = self.f + d_f
            self.fu = min(self.fu + d_f, fumax)
            return ia
        ts = tstep - (fs - self.f) / ia
        if ts <= 0.0:
            ts = 0.0
        c1 = (self.s + depth) * self.imd
        f2 = _grnampt_f2(fs, c1, ks, ts)
        if f2 > fs + ia * ts:
            f2 = fs + ia * ts
        d_f = f2 - self.f
        self.f = f2
        self.fu = min(self.fu + d_f, fumax)
        self.sat = True
        return d_f / tstep

    def _sat(self, tstep, irate, depth, env):
        ks = self.ks * env.hydcon
        lu = self.lu * math.sqrt(env.hydcon)
        fumax = self.imd_max * self.lu * math.sqrt(env.hydcon)
        ia = irate + depth / tstep
        if ia < ZERO:
            return 0.0
        self.t = 5400.0 / lu / env.recovery
        c1 = (self.s + depth) * self.imd
        f2 = _grnampt_f2(self.f, c1, ks, tstep)
        d_f = f2 - self.f
        if d_f > ia * tstep:
            d_f = ia * tstep
            self.sat = False
        self.f = self.f + d_f
        self.fu = min(self.fu + d_f, fumax)
        return d_f / tstep


class _CurveNumberState:
    """infil.c `curvenum_getInfil` with its state."""

    def __init__(self, p: CurveNumber) -> None:
        cn = p.curve_number
        cn = max(cn, 10.0)
        cn = min(cn, 99.0)
        self.smax = (1000.0 / cn - 10.0) / 12.0
        self.regen = 1.0 / p.dry_time
        self.tmax = 0.06 / self.regen
        self.s = self.smax
        self.p = 0.0
        self.f = 0.0
        self.t = 0.0
        self.se = self.smax
        self.rate = 0.0

    def get_infil(self, tstep: float, rainfall, runon, depth, env: _Env):
        depth = depth + runon * tstep
        irate = rainfall
        fa = irate + depth / tstep
        f1 = 0.0
        if irate > ZERO:
            if self.t >= self.tmax:
                self.p = 0.0
                self.f = 0.0
                self.rate = 0.0
                self.se = self.s
            self.t = 0.0
            self.p = self.p + irate * tstep
            big_f1 = self.p * (1.0 - self.p / (self.p + self.se))
            f1 = (big_f1 - self.f) / tstep
            if f1 < 0.0 or self.s <= 0.0:
                f1 = 0.0
        else:
            if depth > MIN_TOTAL_DEPTH and self.s > 0.0:
                f1 = self.rate
                if f1 * tstep > self.s:
                    f1 = self.s / tstep
            else:
                self.t = self.t + tstep
        if f1 > 0.0:
            f1 = min(f1, fa)
            f1 = max(f1, 0.0)
            self.f = self.f + f1 * tstep
            if self.regen > 0.0:
                self.s = self.s - f1 * tstep
                if self.s < 0.0:
                    self.s = 0.0
        else:
            self.s = self.s + self.regen * self.smax * tstep * env.recovery
            if self.s > self.smax:
                self.s = self.smax
        self.rate = f1
        return f1


class _NoInfiltration:
    """A subcatchment without an ``[INFILTRATION]`` line: SWMM's zeroed Horton object, which
    infiltrates nothing."""

    def get_infil(self, tstep, rainfall, runon, depth, env):
        return 0.0


def _infiltration_state(p: Infiltration | None):
    if p is None:
        return _NoInfiltration()
    if isinstance(p, Horton):
        return _HortonState(p)
    if isinstance(p, GreenAmpt):
        return _GreenAmptState(p)
    if isinstance(p, CurveNumber):
        return _CurveNumberState(p)
    raise TypeError(f"unknown infiltration model {p!r}")


# --------------------------------------------------------------------------------------
# rain gage state (gage.c)
# --------------------------------------------------------------------------------------
class _GageState:
    """gage.c: the gage's position in its rainfall record. Dates are seconds since START."""

    def __init__(self, gage: RainGage, used: bool, ignore: bool, rain_factor: Callable[[], float]
                 ) -> None:
        self.gage = gage
        self.used = used
        self.co_gage: _GageState | None = None
        self.times = gage.series.times
        self.values = gage.series.values
        self.rain_factor = rain_factor
        self.rain_accum = 0.0
        self.pos = 0
        self.rainfall = 0.0               # user units / hr
        self.start_date: float | None = None
        self.end_date = 0.0
        self.next_date: float | None = 0.0
        self.next_rainfall = 0.0
        if ignore:
            return
        if self._first():
            self.end_date = float(_round_half_up(self.start_date)) + gage.interval
            if self.start_date > 0.0:
                self.next_date = self.start_date
                self.next_rainfall = self.rainfall
                self.start_date = 0.0
                self.end_date = self.next_date
                self.rainfall = 0.0
            elif not self._next():
                self.next_date = None
        else:
            self.start_date = None

    def _convert(self, r: float) -> float:
        g = self.gage
        if g.kind == "intensity":
            r1 = r
        elif g.kind == "volume":
            r1 = r / g.interval * 3600.0
        else:
            if r < self.rain_accum:
                r1 = r / g.interval * 3600.0
            else:
                r1 = (r - self.rain_accum) / g.interval * 3600.0
            self.rain_accum = r
        return r1 * self.rain_factor()

    def _first(self) -> bool:
        self.start_date = None
        self.rainfall = 0.0
        self.rain_accum = 0.0
        if not self.times:
            return False
        self.pos = 0
        self.start_date = self.times[0]
        self.rainfall = self._convert(self.values[0])
        return True

    def _next(self) -> bool:
        self.next_rainfall = 0.0
        while True:
            self.pos += 1
            if self.pos >= len(self.times):
                return False
            self.next_date = self.times[self.pos]
            r_next = self._convert(self.values[self.pos])
            if r_next != 0.0:
                break
        self.next_rainfall = r_next
        return True

    def set_state(self, t: float) -> None:
        """gage_setState: the rainfall at `t` s (the runoff clock's date, plus a second)."""
        if not self.used:
            return
        if self.co_gage is not None:
            self.rainfall = self.co_gage.rainfall
            return
        t = t + ONE_SECOND
        while True:
            if self.start_date is None:
                self.rainfall = 0.0
                return
            if t < self.start_date:
                self.rainfall = 0.0
                return
            if t < self.end_date:
                return
            if self.next_date is None:
                self.rainfall = 0.0
                return
            if t < self.next_date:
                self.rainfall = 0.0
                return
            self.start_date = self.next_date
            self.end_date = float(_round_half_up(self.start_date)) + self.gage.interval
            self.rainfall = self.next_rainfall
            if not self._next():
                self.next_date = None

    def next_rain_date(self, t: float) -> float | None:
        """gage_getNextRainDate: the next date at which the rainfall changes (None when the
        record gives none)."""
        if not self.used:
            return t
        t = t + ONE_SECOND
        if self.start_date is not None and t < self.start_date:
            return self.start_date
        if t < self.end_date:
            return self.end_date
        return self.next_date


# --------------------------------------------------------------------------------------
# climate (climate.c): evaporation, recovery and monthly adjustments
# --------------------------------------------------------------------------------------
class _Climate:
    """climate.c ``climate_setState`` for evaporation, the soil recovery factor and the
    monthly adjustments; dates are seconds since START."""

    def __init__(self, hyd: Hydrology) -> None:
        self.hyd = hyd
        self.evap = hyd.evaporation
        self.adjust = hyd.adjustments
        self.ucf = UCF_EVAPRATE[0 if hyd.us_units else 1]
        self.start = hyd.start
        midnight = datetime.datetime.combine(hyd.start.date(), datetime.time())
        self.start_date = -(hyd.start - midnight).total_seconds()      # StartDate
        self.end = hyd.total_duration
        self.rate = 0.0                  # ft/s
        self.recovery_factor = 1.0
        self.rain_factor = 1.0
        self.hydcon_factor = 1.0
        self.next_date = self.start_date
        self.next_rate = 0.0
        self.pos = 0
        if self.evap.kind == "timeseries":
            s = self.evap.series
            self.next_date, self.next_rate = s.times[0], s.values[0]
            self.pos = 0
            if self.next_date < self.start_date:
                self._set_next(self.start_date)
            self.rate = self.next_rate / self.ucf
            self._set_next(self.next_date)

    def _clock(self, t: float) -> datetime.datetime:
        return self.start + datetime.timedelta(seconds=t)

    def set_state(self, t: float) -> None:
        month = self._clock(t).month
        k = self.evap.kind
        if k == "constant":
            self.rate = self.evap.monthly[0] / self.ucf
        elif k == "monthly":
            self.rate = self.evap.monthly[month - 1] / self.ucf
        elif k == "timeseries":
            if t >= self.next_date:
                self.rate = self.next_rate / self.ucf
        else:
            self.rate = 0.0
        self.rate += self.adjust.evaporation[month - 1] / self.ucf
        self.recovery_factor = 1.0
        if self.evap.recovery is not None:
            self.recovery_factor = self.evap.recovery[month - 1]
        self.rain_factor = self.adjust.rainfall[month - 1]
        self.hydcon_factor = self.adjust.conductivity[month - 1]
        self._set_next(t)

    def _set_next(self, t: float) -> None:
        """climate.c setNextEvapDate: the next date at which the evaporation rate changes."""
        if self.next_date > t:
            return
        k = self.evap.kind
        if k == "constant":
            self.next_date = t + 365.0 * SEC_PER_DAY
        elif k == "monthly":
            c = self._clock(t)
            year, month = (c.year + 1, 1) if c.month == 12 else (c.year, c.month + 1)
            self.next_date = (datetime.datetime(year, month, 1) - self.start).total_seconds()
        elif k == "timeseries":
            times, values = self.evap.series.times, self.evap.series.values
            self.next_date = t + 365.0 * SEC_PER_DAY
            while self.pos + 1 < len(times):
                self.pos += 1
                d, e = times[self.pos], values[self.pos]
                if d > self.end:
                    break
                if d >= t:
                    self.next_date, self.next_rate = d, e
                    break
        else:
            self.next_date = t + 365.0 * SEC_PER_DAY


# --------------------------------------------------------------------------------------
# the runoff model (subcatch.c, runoff.c)
# --------------------------------------------------------------------------------------
class _Sub:
    """A subcatchment's runoff state, in SWMM's units (ft, cfs)."""

    def __init__(self, s: Subcatchment, hyd: Hydrology) -> None:
        self.name = s.name
        self.area = s.area / (LCF * LCF)
        frac = min(s.pct_imperv, 100.0) / 100.0
        self.frac_imperv = frac
        width = s.width / LCF
        slope = s.slope
        self.n = [0.0, 0.0, 0.0]
        self.d_store = [0.0, 0.0, 0.0]
        self.f_area = [0.0, 0.0, 0.0]
        self.f_outlet = [1.0, 1.0, 1.0]
        self.route_to = [ROUTE_OUTLET] * 3
        self.alpha = [0.0, 0.0, 0.0]
        a = s.subareas
        if a is not None:
            self.n = [a.n_imperv, a.n_imperv, a.n_perv]
            self.d_store = [0.0, a.s_imperv / LCF, a.s_perv / LCF]
            self.f_area = [frac * a.pct_zero / 100.0, frac * (1.0 - a.pct_zero / 100.0),
                           1.0 - frac]
            k = a.route_to
            if frac == 0.0 or frac == 1.0:
                k = ROUTE_OUTLET
            if k == ROUTE_IMPERVIOUS and frac:
                self.route_to[_PERV] = k
                self.f_outlet[_PERV] = 1.0 - a.pct_routed / 100.0
            if k == ROUTE_PERVIOUS:
                self.route_to[_IMPERV0] = self.route_to[_IMPERV1] = k
                self.f_outlet[_IMPERV0] = self.f_outlet[_IMPERV1] = 1.0 - a.pct_routed / 100.0
        for i in (_IMPERV0, _IMPERV1, _PERV):
            area = (1.0 - frac) * self.area if i == _PERV else frac * self.area
            if area > 0.0 and self.n[i] > 0.0:
                self.alpha[i] = MCOEFF * width / area * _sqrt(slope) / self.n[i]
        self.infil = _infiltration_state(s.infiltration)
        self.depth = [0.0, 0.0, 0.0]
        self.inflow = [0.0, 0.0, 0.0]
        self.runoff = [0.0, 0.0, 0.0]
        self.runon = 0.0
        self.rainfall = 0.0
        self.old_runoff = 0.0
        self.new_runoff = 0.0
        self.out_node = None if s.outlet_is_subcatchment else s.outlet
        self.out_sub: int | None = None
        self.gage = -1
        self.n_perv_pattern = s.n_perv_pattern
        self.storage_pattern = s.storage_pattern
        self.infil_pattern = s.infiltration_pattern


class RunoffModel:
    """SWMM's runoff computation for a `Hydrology`, advanced runoff step by runoff step.

    ``advance_to(t_ms)`` takes the runoff steps SWMM takes before a routing step that ends at
    `t_ms` (milliseconds since START); ``weighted(t_ms)`` is the runoff of each subcatchment
    (cfs) interpolated to the routing time `t_ms` between the last two runoff times
    (routing.c `addWetWeatherInflows`); ``outflow_by_node`` sums it per outlet node in m3/s.
    ``old_ms`` and ``new_ms`` are SWMM's ``OldRunoffTime`` and ``NewRunoffTime``; ``steps`` lists
    ``(start s, length s)`` of the runoff steps taken and ``runoff`` (``runoff_m3s``) the
    subcatchments' runoff in cfs (m3/s) after the last step, by subcatchment name."""

    def __init__(self, hydrology: Hydrology) -> None:
        self.hydrology = hydrology
        h = hydrology
        self.climate = _Climate(h)
        self.env = _Env()
        names = [s.name for s in h.subcatchments]
        self.subs = [_Sub(s, h) for s in h.subcatchments]
        gage_index = {g.name: i for i, g in enumerate(h.gages)}
        used = {gage_index[s.gage] for s in h.subcatchments}
        if h.rdii is not None:
            used |= {gage_index[g] for g in h.rdii.gage_names()}
        if h.rdii is not None and any(f != 1.0 for f in h.adjustments.rainfall):
            # SWMM's RDII processing leaves ``Adjust.rainFactor`` at the month of its last
            # rain interval, and the runoff's gages read their first records with it
            # (project_init does not reset it)
            model = h.rdii.model(h)
            if model.active:
                model.run_to_end()
                self.climate.rain_factor = model.rain_factor
        self.gages = [
            _GageState(g, i in used, h.ignore_rainfall, lambda: self.climate.rain_factor)
            for i, g in enumerate(h.gages)]
        for sub, s in zip(self.subs, h.subcatchments, strict=True):
            sub.gage = gage_index[s.gage]
            if s.outlet_is_subcatchment:
                sub.out_sub = names.index(s.outlet)
        # gage_validate: co-gages (a used gage sharing the series of a lower-index used gage)
        for j, g in enumerate(self.gages):
            if not g.used:
                continue
            for i in range(j):
                if self.gages[i].used and h.gages[i].series == h.gages[j].series:
                    g.co_gage = self.gages[i]
                    break
        self.ucf_rain = UCF_RAINFALL[0 if h.us_units else 1]
        self.wet_step = h.wet_step
        self.dry_step = h.dry_step
        self.total_ms = 1000.0 * h.total_duration
        self.old_ms = 0.0
        self.new_ms = 0.0
        self.is_raining = False
        self.has_runoff = False
        self.steps: list[tuple[float, float]] = []

    # --- the runoff step ---------------------------------------------------------------
    def _time_step(self, now: float) -> float:
        """runoff.c `runoff_getTimeStep`."""
        max_step = float(self.dry_step)
        evap_diff = _round_half_up(self.climate.next_date) - _round_half_up(now)
        if 0 < evap_diff < max_step:
            max_step = float(evap_diff)
        for g in self.gages:
            nd = g.next_rain_date(now)
            if nd is None:
                continue
            diff = _round_half_up(nd) - _round_half_up(now)
            if 0 < diff < max_step:
                max_step = float(diff)
        step = self.wet_step if (self.is_raining or self.has_runoff) else self.dry_step
        return min(float(step), max_step)

    def step(self) -> None:
        """Take one runoff step (runoff.c `runoff_execute`)."""
        now = (self.new_ms + 1.0) / 1000.0                 # getDateTime(NewRunoffTime)
        self.climate.set_state(now)
        self.is_raining = False
        for g in self.gages:
            g.set_state(now)
            if g.rainfall > 0.0:
                self.is_raining = True
        tstep = self._time_step(now)
        if tstep <= 0.0:
            raise RuntimeError("runoff time step is not positive (SWMM's ERR_TIMESTEP)")
        self.old_ms = self.new_ms
        self.new_ms = self.new_ms + 1000.0 * tstep
        if self.new_ms > self.total_ms:
            tstep = (self.total_ms - self.old_ms) / 1000.0
            self.new_ms = self.total_ms
        self.steps.append((self.old_ms / 1000.0, tstep))
        for sub in self.subs:                              # subcatch_setOldState
            sub.old_runoff = sub.new_runoff
            sub.inflow = [0.0, 0.0, 0.0]
        for sub in self.subs:
            if sub.area == 0.0:
                continue
            self._get_runon(sub)
        self.has_runoff = False
        for sub in self.subs:
            if sub.area == 0.0:
                continue
            if self._get_runoff(sub, tstep, now) > 0.0:
                self.has_runoff = True
        for sub in self.subs:
            sub.runon = 0.0

    def advance_to(self, t_ms: float) -> None:
        """Take runoff steps until the runoff clock reaches `t_ms` or the end of the
        simulation (SWMM's ``while (NewRunoffTime < nextRoutingTime) runoff_execute()``)."""
        while self.new_ms < t_ms and self.new_ms < self.total_ms:
            self.step()

    # --- subcatch.c ----------------------------------------------------------------------
    def _add_runon(self, k: _Sub, q) -> None:
        """subcatch_addRunonFlow: `q` cfs of runon onto subcatchment `k`."""
        if k.area <= 0.0:
            return
        q = q / k.area
        k.runon = k.runon + q
        for i in range(3):
            k.inflow[i] = k.inflow[i] + q

    def _get_runon(self, sub: _Sub) -> None:
        """subcatch_getRunon."""
        if sub.out_sub is not None and self.subs[sub.out_sub] is not sub:
            self._add_runon(self.subs[sub.out_sub], sub.old_runoff)
        frac = sub.frac_imperv
        if frac < 1.0 and sub.route_to[_IMPERV0] == ROUTE_PERVIOUS:
            q1 = sub.runoff[_IMPERV0] * sub.f_area[_IMPERV0]
            q2 = sub.runoff[_IMPERV1] * sub.f_area[_IMPERV1]
            q = q1 + q2
            sub.inflow[_PERV] = sub.inflow[_PERV] + q * (1.0 - sub.f_outlet[_IMPERV0]) \
                / sub.f_area[_PERV]
        if frac > 0.0 and sub.route_to[_PERV] == ROUTE_IMPERVIOUS and \
                sub.f_area[_IMPERV1] > 0.0:
            q = sub.runoff[_PERV]
            sub.inflow[_IMPERV1] = sub.inflow[_IMPERV1] + q * (1.0 - sub.f_outlet[_PERV]) \
                * sub.f_area[_PERV] / sub.f_area[_IMPERV1]

    def _month_factor(self, pattern: tuple[float, ...] | None) -> float | None:
        if pattern is None:
            return None
        month = (self.hydrology.start
                 + datetime.timedelta(seconds=(self.old_ms + 1.0) / 1000.0)).month
        return pattern[month - 1]

    def _get_runoff(self, sub: _Sub, tstep: float, now: float):
        """subcatch_getRunoff: returns the area-averaged runoff before internal re-routing
        (ft/s); sets ``new_runoff`` (cfs)."""
        g = self.gages[sub.gage]
        rainfall = g.rainfall / self.ucf_rain          # gage_getPrecip, no snow
        sub.rainfall = rainfall
        evap_rate = 0.0 if (self.hydrology.evaporation.dry_only and rainfall > 0.0) \
            else self.climate.rate
        # infil_setInfilFactor
        self.env.hydcon = self.climate.hydcon_factor
        f = self._month_factor(sub.infil_pattern)
        if f is not None:
            self.env.hydcon = f
        self.env.recovery = self.climate.recovery_factor
        runoff = 0.0
        v_outflow = 0.0
        for i in range(3):
            area = sub.area * sub.f_area[i]
            sub.runoff[i] = self._subarea_runoff(sub, i, area, rainfall, evap_rate, tstep)
            sub_area_runoff = sub.runoff[i] * area
            v_outflow = v_outflow + sub.f_outlet[i] * sub.runoff[i] * area * tstep
            runoff = runoff + sub_area_runoff
        sub.new_runoff = v_outflow / tstep
        return runoff / sub.area

    def _subarea_runoff(self, sub: _Sub, i: int, area, precip, evap, tstep: float):
        """subcatch.c getSubareaRunoff: runoff (ft/s over the subarea) after `tstep`."""
        if area == 0.0:
            return 0.0
        t_runoff = tstep
        surf_moisture = sub.depth[i] / tstep
        surf_evap = min(surf_moisture, evap)
        infil = 0.0
        if i == _PERV:
            infil = sub.infil.get_infil(tstep, precip, sub.inflow[i], sub.depth[i], self.env)
        sub.inflow[i] = sub.inflow[i] + precip
        surf_moisture = surf_moisture + sub.inflow[i]
        d_store = sub.d_store[i]
        alpha = sub.alpha[i]
        if i == _PERV:
            f = self._month_factor(sub.storage_pattern)
            if f is not None and f >= 0.0:
                d_store = d_store * f
            f = self._month_factor(sub.n_perv_pattern)
            if f is not None:
                alpha = 0.0 if f <= 0.0 else alpha / f
        if surf_evap + infil >= surf_moisture:
            sub.depth[i] = 0.0
        else:
            sub.inflow[i] = sub.inflow[i] - (surf_evap + infil)
            t_runoff = self._update_depth(sub, i, d_store, alpha, t_runoff)
        # findSubareaRunoff
        x_depth = sub.depth[i] - d_store
        if x_depth > ZERO:
            if sub.n[i] > 0.0:
                runoff = alpha * x_depth ** MEXP
            else:
                runoff = x_depth / t_runoff
                sub.depth[i] = d_store
        else:
            runoff = 0.0
        return runoff

    @staticmethod
    def _update_depth(sub: _Sub, i: int, d_store, alpha, dt: float) -> float:
        """subcatch.c updatePondedDepth: the new ponded depth; returns the time over which it
        is above the depression storage."""
        ix = sub.inflow[i]
        tx = dt
        depth = sub.depth[i]
        if depth + ix * tx <= d_store:
            depth = depth + ix * tx
        else:
            dx = d_store - depth
            if dx > 0.0 and ix > 0.0:
                tx = tx - dx / ix
                depth = d_store
            tx_val = _val(tx)
            if alpha > 0.0 and tx_val > 0.0:
                def derivs(d):
                    rx = d - d_store
                    rx = 0.0 if rx < 0.0 else alpha * rx ** MEXP
                    return ix - rx
                depth = odesolve_integrate(depth, 0.0, tx_val, ODETOL, tx_val, derivs)
                if isinstance(tx, torch.Tensor) and tx.requires_grad:
                    # the length of the interval depends on the parameters (the time left
                    # after the depression storage fills): d y(T) / dT = f(y(T))
                    depth = depth + (tx - tx.detach()) * derivs(depth)
            else:
                if tx < 0.0:
                    tx = 0.0
                depth = depth + ix * tx
        if depth < 0.0:
            depth = 0.0
        sub.depth[i] = depth
        return tx

    # --- results ---------------------------------------------------------------------------
    @property
    def runoff(self) -> dict[str, Number]:
        """The runoff of each subcatchment (cfs) at the last runoff step, by name."""
        return {s.name: s.new_runoff for s in self.subs}

    @property
    def runoff_m3s(self) -> dict[str, Number]:
        """The runoff of each subcatchment (m3/s) at the last runoff step, by name."""
        return {s.name: s.new_runoff * QCF for s in self.subs}

    def weighted(self, routing_ms: float) -> list[Number]:
        """The runoff of each subcatchment (cfs) at the routing time `routing_ms`:
        ``(1 - f) * old + f * new`` with ``f = (t - OldRunoffTime) / (NewRunoffTime -
        OldRunoffTime)`` clamped to [0, 1] (routing.c `addWetWeatherInflows`,
        `subcatch_getWtdOutflow`)."""
        span = self.new_ms - self.old_ms
        f = (routing_ms - self.old_ms) / span if span else 0.0
        f = min(max(f, 0.0), 1.0)
        return [0.0 if s.area == 0.0 else (1.0 - f) * s.old_runoff + f * s.new_runoff
                for s in self.subs]

    def outflow_by_node(self, routing_ms: float) -> dict[str, Number]:
        """`{node: m3/s}` of the runoff at the routing time `routing_ms`, summed over the
        subcatchments whose outlet is that node (one that drains to a subcatchment has none)."""
        out: dict[str, Number] = {}
        for sub, q in zip(self.subs, self.weighted(routing_ms), strict=True):
            if sub.out_node is not None:
                out[sub.out_node] = out.get(sub.out_node, 0.0) + q * QCF
        return out


# --------------------------------------------------------------------------------------
# the lateral-inflow driver
# --------------------------------------------------------------------------------------
class RunoffDriver:
    """``inflow(t, step)`` for `simulate_dynamic_wave`: the ``(n,)`` m3/s lateral inflow of the
    routing step that starts at `t` s and lasts `step` s -- the file's `InflowSchedule`
    (if any) plus the runoff of its subcatchments. It owns one `RunoffModel` and must be
    called with increasing `t` (as a simulation does); ``step_aware`` tells
    `simulate_dynamic_wave` to pass the step. Called with `t` alone, the runoff is that at
    `t` (the runoff clock advanced to the first runoff time past `t`)."""

    step_aware = True

    def __init__(self, inflows: RunoffInflows, names: Sequence[str]) -> None:
        self.inflows = inflows
        index = {n: i for i, n in enumerate(names)}
        missing = sorted(set(inflows.nodes) - set(index))
        if missing:
            raise ValueError(f"RunoffInflows: node(s) {missing} are not among the model's "
                             f"nodes")
        self.n = len(index)
        self.index = index
        h = inflows.hydrology
        self.model = RunoffModel(h) if h.subcatchments else None
        self.runoff_nodes = frozenset(inflows._runoff_nodes)
        self.rdii = h.rdii.model(h) if h.rdii is not None else None
        self.rdii_nodes = frozenset(inflows._rdii_nodes)

    def runoff(self, t: float, step: float | None = None) -> dict[str, Number]:
        """`{node: m3/s}` of the runoff for the routing step that starts at `t` s with the
        length `step` s (``None``: the runoff at `t`)."""
        if self.model is None:
            return {}
        start_ms = round(t * 1000.0, 6)
        end_ms = round((t + step) * 1000.0, 6) if step is not None \
            else start_ms + 1.0
        self.model.advance_to(end_ms)
        flows = self.model.outflow_by_node(start_ms)
        return {n: q for n, q in flows.items() if n in self.runoff_nodes}

    def __call__(self, t: float, step: float | None = None) -> torch.Tensor:
        t = float(t)
        base = {}
        if self.inflows.schedule is not None:
            base = self.inflows.schedule.flows(t)
        wet = self.runoff(t, step)
        if self.rdii is not None:
            for node, value in self.rdii.flows(t).items():
                if node in self.rdii_nodes:
                    wet[node] = wet[node] + value if node in wet else value
        q = torch.zeros(self.n, dtype=F64)
        for node, value in base.items():
            q[self.index[node]] = value
        if wet:
            idx = torch.tensor([self.index[n] for n in wet], dtype=torch.long)
            vals = torch.stack([torch.as_tensor(v, dtype=F64) for v in wet.values()])
            q = q.index_add(0, idx, vals)
        return q


class RunoffInflows:
    """The lateral inflow of a SWMM model with subcatchments: the `hydrology` and the file's
    `schedule` (`InflowSchedule` of time series, patterns, dry-weather flow; ``None`` if
    none). It stands where an `InflowSchedule` does (``net.inflows``,
    ``model.inflow_schedule``): `start`, `nodes`, ``driver(names)``, ``flows(t)``, ``at``.
    ``driver(names)`` is the ``inflow`` callable of `simulate_dynamic_wave` (it makes a fresh
    `RunoffModel` each time, so every run starts from dry subcatchments). Runoff to a node
    that is not in `nodes` of the network (an outfall) is dropped by the reader, as the
    dynamic wave ignores lateral inflow at an outfall; so is RDII (`rdii.py`) to a node outside
    `rdii_nodes`, which defaults to the nodes of ``hydrology.rdii``. The RDII of the file joins
    the runoff in the driver, whether or not the file has subcatchments."""

    def __init__(self, hydrology: Hydrology, schedule: InflowSchedule | None = None,
                 nodes: Sequence[str] | None = None,
                 rdii_nodes: Sequence[str] | None = None) -> None:
        self.hydrology = hydrology
        self.schedule = schedule
        runoff_nodes = tuple(nodes) if nodes is not None else hydrology.outlet_nodes()
        self._runoff_nodes = runoff_nodes
        self._rdii_nodes = (tuple(rdii_nodes) if rdii_nodes is not None
                            else hydrology.rdii.nodes if hydrology.rdii is not None else ())
        self._cached: tuple[RunoffDriver, float] | None = None

    @property
    def start(self) -> datetime.datetime:
        """The date and time of the start of the run."""
        return self.hydrology.start

    @property
    def nodes(self) -> tuple[str, ...]:
        """The nodes that receive an inflow: the schedule's, then the subcatchment outlets."""
        sched = self.schedule.nodes if self.schedule is not None else ()
        return tuple(dict.fromkeys(list(sched) + list(self._runoff_nodes)
                                   + list(self._rdii_nodes)))

    def with_duration(self, total_duration: float) -> RunoffInflows:
        """The same inflows with the hydrology cut at `total_duration` s (see
        `Hydrology.with_duration`)."""
        return RunoffInflows(self.hydrology.with_duration(total_duration), self.schedule,
                             self._runoff_nodes, self._rdii_nodes)

    def driver(self, names: Sequence[str]) -> RunoffDriver:
        """``inflow(t, step)`` over the model's nodes `names`; see `RunoffDriver`."""
        return RunoffDriver(self, names)

    def flows(self, t: float, step: float | None = None) -> dict[str, float]:
        """`{node: m3/s}` for the routing step that starts `t` s after START (and lasts
        `step` s; without it, the runoff at `t`). Queries must not go back in time without
        starting over: an earlier `t` re-runs the runoff from the start."""
        names = list(self.nodes)
        if self._cached is None or t < self._cached[1]:
            self._cached = (self.driver(names), t)
        driver = self._cached[0]
        self._cached = (driver, t)
        vec = driver(t, step)
        return {n: float(vec[i]) for i, n in enumerate(names) if float(vec[i]) != 0.0}

    def at(self, t: float, names: Sequence[str], step: float | None = None) -> torch.Tensor:
        """The ``(len(names),)`` m3/s inflow of the step that starts at `t`."""
        return self.driver(names)(t, step)


__all__ = [
    "Adjustments", "CurveNumber", "Evaporation", "GreenAmpt", "Horton", "Hydrology",
    "RainGage", "RunoffDriver", "RunoffInflows", "RunoffModel", "Subareas", "Subcatchment",
    "odesolve_integrate", "SAMPLE_OFFSET",
]
