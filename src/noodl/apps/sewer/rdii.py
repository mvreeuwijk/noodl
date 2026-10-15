"""Rainfall-dependent infiltration and inflow (RDII) of a SWMM model: the triangular unit
hydrographs of `[HYDROGRAPHS]` convolved with the rainfall of a rain gage and the sewershed areas
of `[RDII]`, as SWMM 5.2.4 computes them (``rdii.c``; the lateral inflow joins the routing in
``routing.c`` `addRdiiInflows`). The tests in ``tests/verification/test_sewer_rdii_parity.py``
compare it with the engine through pyswmm.

SWMM computes the RDII of the whole run before the first step (`rain_open` ->
`rdii_openRdii` -> `createRdiiFile`, swmm5.c:366) and stores it in an interface file that the
routing reads; RDII is a function of the rainfall alone, so `RdiiModel` computes it directly,
record by record, as the routing asks for it, and keeps what SWMM keeps. What follows is
SWMM's, operation for operation.

Unit hydrograph sets. A set (`UnitHydrographSet`) has a rain gage and, for each month and each
of three responses (SHORT, MEDIUM, LONG), a triangle of ratio R, time to peak T and recession
ratio K (hours; `setUnitHydParams`): the time to peak is ``(long)(T * 3600)`` and the base
``(long)(T * (1 + K) * 3600)`` whole seconds, the ordinate at time ``t`` the triangle of area
``R`` (``2/base`` at its peak, in 1/hr). A set with no parameters for a month has no response
(base 0). The optional ``Dmax``, ``Drecov`` and ``Dinit`` are the initial abstraction (rain
depth units; ``Drecov`` per day) of the response.

Rainfall processing. The rain interval of a set is the WET_STEP (already lowered to the smallest
rain interval of the used gages, ``gage_validate``), lowered to the rising and falling limb of
every response of every month (`getRainInterval`); the RDII step ``RdiiStep`` is the WET_STEP.
At each record ``n * RdiiStep`` s into the run (``n = 0, 1, ...``) every set (also one no node
uses) takes the gage's rainfall over each of its rain intervals that starts before the record's
date (`getRainfall`): the rain depth is ``intensity * interval / 3600``, reduced by the
unused initial abstraction of the response of the interval's month (`applyIA`; in dry
weather the used abstraction recovers by ``interval / 86400 * Drecov``), and kept in a ring
of the past excess rainfall of the response. A dry period of more than the response's base
begins a new event, which clears the ring (`updateDryPeriod`). The month of the stored rain is
that of the RECORD's date, while the abstraction takes the month of the interval's own date.
The flow of the set is the sum over the three responses of the convolution of the stored
rainfall with the response's ordinates at the middle of each past interval (`getUnitHydConvol`,
`getUnitHydOrd`), times the ratio R of the month the rain was stored in; a node's flow is that
times its sewershed area over the unit factor of rainfall, in cfs, zero below ``ZERO_RDII`` of
1e-4 cfs (`getNodeRdii`).

Quirks of SWMM that this reproduces: the gage state is set once per record for a gage, even when
a set with a short rain interval takes several intervals of one record (the later intervals
reuse the rainfall of the first: `Gage.isCurrent`, `getRainfall`); the sets that share a gage
(or a series) advance the one state of the gage; the dates are SWMM's double-precision days
(``datetime_addSeconds`` rounds to the second), compared as SWMM compares them; the flow of a
node is stored as a 4-byte float (``REAL4``) in the interface file; a record with no RDII at
any node is not stored and gives zero flow; and the record read for a routing step is the one
whose RdiiStep contains the step's date plus 1 ms (``rdii_getNumRdiiFlows``): the flow is
constant over the RdiiStep, not interpolated. The routing then drops a node's RDII flow below
``FLOW_TOL`` of 1e-5 cfs.

Differentiability. R, T, K, the initial abstraction parameters and the sewershed area may be
torch tensors (`Rdii.with_gradients`). The gradient of the area and of R is exact; the times
are truncated to whole seconds as SWMM does, and T and K carry a straight-through gradient (the
value is the truncated one, the derivative is that of the untruncated seconds, since the exact
derivative of a truncation is zero). The comparisons (rain on or off, the abstraction used up,
the dry period, the triangle's limbs), the rain interval, the number of past intervals and the
rounding of the node flow to 4 bytes (a straight-through cast) are decided on the values. Not
differentiable: the rainfall and the dates.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
import struct
from collections.abc import Sequence

import torch

from noodl.apps.sewer.runoff import (
    UCF_RAINFALL,
    Hydrology,
    Number,
    _GageState,
    _val,
)
from noodl.apps.sewer.xsect_tables import LCF, QCF

F64 = torch.float64

#: rdii.c `ZERO_RDII`: a node's RDII flow below this (cfs) is zero.
ZERO_RDII = 0.0001
#: routing.c `FLOW_TOL`: an inflow below this (cfs) is dropped by the routing.
FLOW_TOL_CFS = 1e-5
#: The response names in SWMM's order (keywords.c `UHTypeWords`).
RESPONSES = ("SHORT", "MEDIUM", "LONG")
MONTH_NAMES = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV",
               "DEC")
_DATE_DELTA = 693594            # datetime.c DateDelta: days from 0000-01-00 to 1899-12-30
_SECS_PER_DAY = 86400.0


# --------------------------------------------------------------------------------------
# SWMM's dates (datetime.c): doubles in days
# --------------------------------------------------------------------------------------
def encode_date(day: datetime.date) -> float:
    """datetime_encodeDate: `day` as SWMM's day number."""
    return float(day.toordinal() - _DATE_DELTA)


def month_of_year(date: float) -> int:
    """datetime_monthOfYear: the month (1..12) of the SWMM date `date`."""
    t = math.floor(date) + _DATE_DELTA
    if t <= 0:
        return 1
    return datetime.date.fromordinal(t).month


def add_seconds(date: float, seconds: float) -> float:
    """datetime_addSeconds: `date` plus `seconds`, the time of day of `date` first rounded to
    the second (datetime_decodeTime)."""
    d = math.floor(date)
    frac = (date - d) * _SECS_PER_DAY
    secs = int(math.floor(frac + 0.5))
    if secs >= 86400:
        secs = 86399
    mins, s = divmod(secs, 60)
    h, m = divmod(mins, 60)
    if h > 23:
        h = 0
    return d + (3600.0 * h + 60.0 * m + s + seconds) / _SECS_PER_DAY


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True, eq=False)
class UnitHydrograph:
    """One triangular unit hydrograph (SHORT, MEDIUM or LONG) of a month: response ratio `r`
    (the fraction of the rainfall that becomes RDII), time to peak `t` and recession ratio `k`
    (hours; the base is ``t * (1 + k)``) and the initial abstraction `ia_max` (rain depth, in
    or mm), `ia_recovery` (depth per day) and `ia_initial` (the depth used at the start of
    the run). Any number may be a torch scalar."""

    r: Number = 0.0
    t: Number = 0.0
    k: Number = 0.0
    ia_max: Number = 0.0
    ia_recovery: Number = 0.0
    ia_initial: Number = 0.0

    def exists(self) -> bool:
        """Whether the triangle has a base (SWMM treats a base of 0 s as no response)."""
        return _seconds(_val(self.t) * (1.0 + _val(self.k)))[0] != 0


_NO_RESPONSE = UnitHydrograph()


@dataclasses.dataclass(frozen=True, eq=False)
class UnitHydrographSet:
    """A `[HYDROGRAPHS]` set: the rain `gage` it reads (``None`` when the file gives none) and
    `responses`, twelve months (January first) of three `UnitHydrograph` (short, medium, long)."""

    name: str
    gage: str | None
    responses: tuple[tuple[UnitHydrograph, UnitHydrograph, UnitHydrograph], ...]

    def __post_init__(self) -> None:
        if len(self.responses) != 12 or any(len(m) != 3 for m in self.responses):
            raise ValueError(f"unit hydrograph set {self.name!r}: needs 12 months of 3 "
                             f"responses")

    @staticmethod
    def blank(name: str, gage: str | None = None) -> UnitHydrographSet:
        """A set with no response in any month."""
        return UnitHydrographSet(name, gage, tuple((_NO_RESPONSE,) * 3 for _ in range(12)))

    def with_response(self, response: UnitHydrograph, kind: int, month: int | None = None
                      ) -> UnitHydrographSet:
        """This set with `response` as response `kind` (0 short, 1 medium, 2 long) of `month`
        (1..12; ``None``: every month)."""
        months = range(12) if month is None else [month - 1]
        rows = [list(m) for m in self.responses]
        for m in months:
            rows[m][kind] = response
        return dataclasses.replace(self, responses=tuple(tuple(m) for m in rows))


@dataclasses.dataclass(frozen=True, eq=False)
class RdiiInflow:
    """An `[RDII]` line: the `node` that receives the RDII of the unit hydrograph set
    `unit_hydrograph` over the sewershed `area` (m2; a torch scalar allowed)."""

    node: str
    unit_hydrograph: str
    area: Number


@dataclasses.dataclass(frozen=True, eq=False)
class Rdii:
    """The RDII of a SWMM file: `sets` (`UnitHydrographSet`) and `inflows` (`RdiiInflow`, one
    per node). `ignore` is IGNORE_RDII. A set that no `[RDII]` line names still advances the
    rain gage (SWMM processes every set), so it stays here."""

    sets: tuple[UnitHydrographSet, ...]
    inflows: tuple[RdiiInflow, ...]
    ignore: bool = False

    def __post_init__(self) -> None:
        names = [s.name for s in self.sets]
        if len(set(names)) != len(names):
            raise ValueError("Rdii: unit hydrograph set names must be unique")
        for inflow in self.inflows:
            if inflow.unit_hydrograph not in names:
                raise ValueError(f"RDII at node {inflow.node!r} names the unit hydrograph "
                                 f"{inflow.unit_hydrograph!r}, which is not in `sets`")

    @property
    def nodes(self) -> tuple[str, ...]:
        """The nodes that receive RDII."""
        return tuple(dict.fromkeys(i.node for i in self.inflows))

    def gage_names(self) -> tuple[str, ...]:
        """The rain gages the sets read (SWMM marks them used, whether or not a node uses the
        set)."""
        return tuple(dict.fromkeys(s.gage for s in self.sets if s.gage is not None))

    def validate(self) -> None:
        """SWMM's ``validateRdii`` (run when a node has RDII): a response with a negative time
        to peak or ratio, or ratios that add to more than 1.01 in a month, raise ValueError."""
        for s in self.sets:
            for m, month in enumerate(s.responses):
                total = 0.0
                for resp in month:
                    tpeak, tbase = _seconds(_val(resp.t))[0], _seconds(
                        _val(resp.t) * (1.0 + _val(resp.k)))[0]
                    if tbase == 0:
                        continue
                    if tpeak < 0:
                        raise ValueError(f"unit hydrograph {s.name!r}: a negative time to "
                                         f"peak in {MONTH_NAMES[m]} (SWMM's ERR_UNITHYD_TIMES)")
                    if _val(resp.r) < 0.0:
                        raise ValueError(f"unit hydrograph {s.name!r}: a negative response "
                                         f"ratio in {MONTH_NAMES[m]} (SWMM's "
                                         f"ERR_UNITHYD_RATIOS)")
                    total += _val(resp.r)
                if total > 1.01:
                    raise ValueError(f"unit hydrograph {s.name!r}: the response ratios of "
                                     f"{MONTH_NAMES[m]} add to {total:g} > 1.01 (SWMM's "
                                     f"ERR_UNITHYD_RATIOS)")

    def with_gradients(self, fields: Sequence[str] = ("r", "t", "k"), area: bool = True,
                       ia: bool = False, only: Sequence[str] | None = None
                       ) -> tuple[Rdii, dict[str, torch.Tensor]]:
        """`(rdii, leaves)`: this RDII with the listed `UnitHydrograph` `fields` of every response
        that exists (of the sets `only` names) and, if `area`, every sewershed area replaced by
        float64 leaf tensors that require gradients; `ia` adds ``ia_max``, ``ia_recovery`` and
        ``ia_initial``. Months that share the same response share a leaf. `leaves` maps
        ``"<set>.<months>.<SHORT|MEDIUM|LONG>.<field>"`` (``months`` is ``ALL`` or the month
        names joined with ``+``) and ``"<node>.area"`` to the tensors; see the module docstring
        for what the gradient of ``t`` and ``k`` is."""
        names = list(fields) + (["ia_max", "ia_recovery", "ia_initial"] if ia else [])
        leaves: dict[str, torch.Tensor] = {}

        def leaf(key: str, value) -> torch.Tensor:
            t = torch.as_tensor(_val(value), dtype=F64).clone().requires_grad_(True)
            leaves[key] = t
            return t

        new_sets = []
        for s in self.sets:
            if only is not None and s.name not in only:
                new_sets.append(s)
                continue
            rows = [list(m) for m in s.responses]
            for kind in range(3):
                groups: dict[tuple, list[int]] = {}
                for m in range(12):
                    resp = s.responses[m][kind]
                    if resp.exists():
                        key = tuple(_val(getattr(resp, f.name)) for f in dataclasses.fields(resp))
                        groups.setdefault(key, []).append(m)
                for months in groups.values():
                    label = "ALL" if len(months) == 12 else "+".join(MONTH_NAMES[m]
                                                                      for m in months)
                    base = s.responses[months[0]][kind]
                    changes = {f: leaf(f"{s.name}.{label}.{RESPONSES[kind]}.{f}",
                                       getattr(base, f)) for f in names}
                    for m in months:
                        rows[m][kind] = dataclasses.replace(s.responses[m][kind], **changes)
            new_sets.append(dataclasses.replace(
                s, responses=tuple(tuple(m) for m in rows)))
        inflows = []
        for i in self.inflows:
            if area and (only is None or i.unit_hydrograph in only):
                inflows.append(dataclasses.replace(i, area=leaf(f"{i.node}.area", i.area)))
            else:
                inflows.append(i)
        return dataclasses.replace(self, sets=tuple(new_sets), inflows=tuple(inflows)), leaves

    def model(self, hydrology: Hydrology) -> RdiiModel:
        """The `RdiiModel` of this RDII under the rain gages and the calendar of `hydrology`."""
        return RdiiModel(self, hydrology)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _seconds(hours: Number) -> tuple[int, Number]:
    """`((long)(hours * 3600.), the same as a number)`: the truncated whole seconds, and for a
    tensor `hours` that value with the gradient of the untruncated product (straight-through)."""
    v = hours * 3600.0
    whole = int(_val(v))                       # C's (long) cast truncates toward zero
    if isinstance(v, torch.Tensor):
        return whole, v - v.detach() + float(whole)
    return whole, float(whole)


def _float32(q: Number) -> Number:
    """`q` rounded to a 4-byte float (SWMM stores node flows as REAL4); a tensor keeps its
    gradient (straight-through)."""
    value = _val(q)
    rounded = struct.unpack("f", struct.pack("f", value))[0] if abs(value) < 3.0e38 else value
    if isinstance(q, torch.Tensor):
        return q - q.detach() + rounded
    return rounded


def _ordinate(tpeak: Number, tbase: Number, t: float) -> Number:
    """rdii.c getUnitHydOrd: the ordinate (1/hr) at `t` s of the triangle of base `tbase` and
    time to peak `tpeak` (both whole seconds)."""
    if t >= _val(tbase):
        return 0.0
    qpeak = 2.0 / tbase * 3600.0
    t2 = tbase - tpeak
    f = t / tpeak if t <= _val(tpeak) else 1.0 - (t - tpeak) / t2
    return f * qpeak if _val(f) > 0.0 else 0.0


class _Uh:
    """The processing data of one response of one set (rdii.c TUHData)."""

    def __init__(self, max_periods: int, interval: int, ia_initial: Number) -> None:
        self.max_periods = max_periods
        self.past_rain: list[Number] = [0.0] * max_periods
        self.past_month = [0] * max_periods
        # the first recorded rainfall begins a new event
        self.dry_seconds = max_periods * interval + 1
        self.period = max_periods + 1
        self.has_past_rain = False
        self.ia_used: Number = ia_initial


class _Group:
    """The processing data of one set (rdii.c TUHGroup)."""

    def __init__(self, s: UnitHydrographSet, interval_cap: int, start_month: int) -> None:
        self.set = s
        self.tpeak = [[_seconds(s.responses[m][k].t) for k in range(3)] for m in range(12)]
        self.tbase = [[_seconds(s.responses[m][k].t * (1.0 + s.responses[m][k].k))
                       for k in range(3)] for m in range(12)]
        ri = interval_cap
        for m in range(12):
            for k in range(3):
                if self.tpeak[m][k][0] > 0:
                    limb = self.tpeak[m][k][0]
                    ri = min(ri, limb)
                    limb = self.tbase[m][k][0] - limb
                    if limb > 0:
                        ri = min(ri, limb)
        self.interval = ri
        self.uh = []
        for k in range(3):
            n = max(int(self.tbase[m][k][0] / ri) + 1 for m in range(12))
            self.uh.append(_Uh(n, ri, s.responses[start_month][k].ia_initial))
        self.used = False
        self.rdii: Number = 0.0
        self.gage_date = 0.0
        self.last_date = 0.0
        self._ord: dict[tuple[int, int, int], Number] = {}

    def ordinate_times_r(self, m: int, k: int, p: int) -> Number:
        """The response's ordinate at the middle of past interval `p` times its ratio R."""
        key = (m, k, p)
        got = self._ord.get(key)
        if got is None:
            t = (float(p) - 0.5) * float(self.interval)
            got = _ordinate(self.tpeak[m][k][1], self.tbase[m][k][1], t) \
                * self.set.responses[m][k].r
            self._ord[key] = got
        return got


class RdiiModel:
    """SWMM's RDII computation for an `Rdii` under the rain gages, calendar, WET_STEP and
    rainfall adjustments of a `Hydrology`. ``flows(t)`` is the RDII ``{node: m3/s}`` that SWMM
    adds to the lateral inflow of a routing step that starts `t` s into the run (sampled at
    ``t + 1 ms``, as the interface file is read); `t` must not decrease between calls by more
    than the record in force (the records are computed in order and kept). ``records`` holds
    ``{record number: {node: cfs}}`` for the records with some RDII, a record being
    ``RdiiStep`` long."""

    def __init__(self, rdii: Rdii, hydrology: Hydrology) -> None:
        self.rdii = rdii
        self.hydrology = hydrology
        us = hydrology.us_units
        self.active = bool(rdii.inflows) and not rdii.ignore and not hydrology.ignore_rainfall
        self.step = int(math.floor(hydrology.wet_step + 0.5))
        self.records: dict[int, dict[str, Number]] = {}
        self._next = 0                                 # next record to compute
        self.rain_factor = 1.0                         # Adjust.rainFactor
        if not self.active:
            return
        rdii.validate()
        h = hydrology
        midnight = datetime.datetime.combine(h.start.date(), datetime.time())
        self.start_days = encode_date(h.start.date()) + (h.start - midnight).total_seconds() \
            / _SECS_PER_DAY
        self.ucf_rain = UCF_RAINFALL[0 if us else 1]
        # --- gages: used = those of a subcatchment or of a set, co-gages (gage_validate)
        names = [g.name for g in h.gages]
        used = {s.gage for s in h.subcatchments} | set(rdii.gage_names())
        self.gages = [_GageState(g, g.name in used, False, lambda: self.rain_factor)
                      for g in h.gages]
        for j, g in enumerate(self.gages):
            if not g.used:
                continue
            for i in range(j):
                if self.gages[i].used and h.gages[i].series == h.gages[j].series:
                    g.co_gage = self.gages[i]
                    break
        # --- groups
        start_month = month_of_year(self.start_days)
        self.groups = [_Group(s, self.step, start_month - 1) for s in rdii.sets]
        self.gage_of: list[int] = []
        for grp in self.groups:
            if grp.set.gage is None:
                raise ValueError(f"unit hydrograph set {grp.set.name!r} has no rain gage "
                                 f"(SWMM then reads its rainfall from the first gage of the "
                                 f"file, which is not modelled)")
            j = names.index(grp.set.gage)
            self.gage_of.append(j)
        # a set whose gage shares a series with an earlier used gage reads that gage
        for i, j in enumerate(self.gage_of):
            if self.gages[j].co_gage is not None:
                self.gage_of[i] = self.gages.index(self.gages[j].co_gage)
        sets = {grp.set.name: i for i, grp in enumerate(self.groups)}
        by_node: dict[str, RdiiInflow] = {}
        for inflow in rdii.inflows:
            by_node[inflow.node] = inflow                    # a later line replaces
        self.node_inflows = list(by_node.values())
        for inflow in self.node_inflows:
            grp = self.groups[sets[inflow.unit_hydrograph]]
            grp.used = True
            grp.last_date = self.start_days
        for grp in self.groups:
            grp.gage_date = self.start_days
        self.node_group = [sets[i.unit_hydrograph] for i in self.node_inflows]
        self.area_ft2 = [i.area / (LCF * LCF) for i in self.node_inflows]

    # --- the processing of one record (createRdiiFile's loop body) -------------------------
    def _gage_time(self, date: float) -> float:
        """Seconds since the start of the run for the SWMM date `date`."""
        return (date - self.start_days) * _SECS_PER_DAY

    def _get_rainfall(self, current: float) -> None:
        month = month_of_year(current) - 1
        adjust = self.hydrology.adjustments.rainfall
        current_flags = [False] * len(self.gages)
        for j, grp in enumerate(self.groups):
            g = self.gage_of[j]
            ri = grp.interval
            while grp.gage_date < current:
                gage_date = grp.gage_date
                self.rain_factor = adjust[month_of_year(gage_date) - 1]
                if not current_flags[g]:
                    self.gages[g].set_state(self._gage_time(gage_date))
                    current_flags[g] = True
                depth = self.gages[g].rainfall * float(ri) / 3600.0
                for k in range(3):
                    excess = self._apply_ia(grp, k, gage_date, float(ri), depth)
                    self._update_dry_period(grp, k, excess, ri)
                    uh = grp.uh[k]
                    i = uh.period
                    if i >= uh.max_periods:
                        i = 0
                    uh.past_rain[i] = excess
                    uh.past_month[i] = month
                    uh.period = i + 1
                grp.gage_date = add_seconds(gage_date, float(ri))

    def _apply_ia(self, grp: _Group, k: int, date: float, dt: float, rain: float) -> Number:
        m = month_of_year(date) - 1
        resp = grp.set.responses[m][k]
        uh = grp.uh[k]
        ia = resp.ia_max - uh.ia_used
        if not _val(ia) > 0.0:
            ia = 0.0
        if rain > 0.0:
            net = rain - ia
            if not _val(net) > 0.0:
                net = 0.0
            ia = rain - net
            uh.ia_used = uh.ia_used + ia
        else:
            uh.ia_used = uh.ia_used - dt / 86400.0 * resp.ia_recovery
            if not _val(uh.ia_used) > 0.0:
                uh.ia_used = 0.0
            net = 0.0
        return net

    @staticmethod
    def _update_dry_period(grp: _Group, k: int, rain: Number, interval: int) -> None:
        uh = grp.uh[k]
        if _val(rain) > 0.0:
            if uh.dry_seconds >= interval * uh.max_periods:
                uh.past_rain = [0.0] * uh.max_periods
                uh.period = 0
            uh.dry_seconds = 0
            uh.has_past_rain = True
        else:
            uh.dry_seconds += interval
            uh.has_past_rain = uh.dry_seconds < interval * uh.max_periods

    def _convolve(self, grp: _Group, k: int) -> Number:
        uh = grp.uh[k]
        rdii: Number = 0.0
        i = uh.period - 1
        if i < 0:
            i = uh.max_periods - 1
        p_max = uh.max_periods
        p = 1
        while p < p_max:
            v = uh.past_rain[i]
            if _val(v) > 0.0:
                rdii = rdii + grp.ordinate_times_r(uh.past_month[i], k, p) * v
            p += 1
            i -= 1
            if i < 0:
                i = uh.max_periods - 1
        return rdii

    def _get_unit_hyd_rdii(self, current: float) -> None:
        for grp in self.groups:
            if not grp.used or current < grp.last_date:
                continue
            grp.last_date = grp.gage_date
            grp.rdii = 0.0
            for k in range(3):
                if grp.uh[k].has_past_rain:
                    grp.rdii = grp.rdii + self._convolve(grp, k)

    def _node_rdii(self) -> tuple[bool, dict[str, Number]]:
        flows: dict[str, Number] = {}
        has = False
        for inflow, g, area in zip(self.node_inflows, self.node_group, self.area_ft2,
                                   strict=True):
            q = self.groups[g].rdii * area / self.ucf_rain
            if _val(q) < ZERO_RDII:
                q = 0.0
            else:
                has = True
            flows[inflow.node] = _float32(q)
        return has, flows

    def _compute_to(self, n: int) -> None:
        while self._next <= n:
            elapsed = float(self._next * self.step)
            current = self.start_days + elapsed / _SECS_PER_DAY
            self._get_rainfall(current)
            self._get_unit_hyd_rdii(current)
            has, flows = self._node_rdii()
            if has:
                self.records[self._next] = flows
            self._next += 1

    def run_to_end(self) -> None:
        """Compute every record of the run (createRdiiFile's whole loop, ``n * RdiiStep`` up to
        the total duration); ``rain_factor`` is then what SWMM leaves in ``Adjust.rainFactor``."""
        if self.active:
            self._compute_to(int(self.hydrology.total_duration // self.step))

    # --- the flow of a routing step ---------------------------------------------------------
    def flows_cfs(self, t: float) -> dict[str, Number]:
        """`{node: cfs}` of the RDII record in force at `t` s into the run plus 1 ms (empty when
        there is none)."""
        if not self.active:
            return {}
        ts = t + 0.001
        if ts < 0.0:
            return {}
        n = int(ts // self.step)
        self._compute_to(n)
        return dict(self.records.get(n, {}))

    def flows(self, t: float) -> dict[str, Number]:
        """`{node: m3/s}` added to the lateral inflow of the routing step that starts at `t`
        s: the stored flows at or above the routing's ``FLOW_TOL``."""
        return {node: q * QCF for node, q in self.flows_cfs(t).items()
                if abs(_val(q)) >= FLOW_TOL_CFS}


__all__ = [
    "FLOW_TOL_CFS", "Rdii", "RdiiInflow", "RdiiModel", "UnitHydrograph", "UnitHydrographSet",
    "ZERO_RDII", "add_seconds", "encode_date", "month_of_year",
]
