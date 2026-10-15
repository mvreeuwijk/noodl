"""SWMM 5.2.4's evaporation and seepage losses of the dynamic wave.

Three kinds of loss take water out of the network, all from the evaporation rate and the
conductivity adjustment of the file's climate (`ClimateClock`):

- an **open conduit evaporates** ``topWidth * length * Evap.rate`` and **any conduit seeps**
  ``seepRate * width * length * hydconFactor`` (``[LOSSES]`` Seepage; the width is the section's
  width at the depth limited to its widest point), both from the mean of the step-start and
  current link depth, together capped at the link's volume over the step
  (`link.c::conduit_getLossRate`, link.c:1334-1399). The rate enters the momentum equation as
  ``dq6 = rate 2.5 dt v / L`` (dwflow.c:236, `dynwave_terms.conduit_terms`) and is taken,
  times the barrels, half from each end node, or all from the one that is not an outfall
  (dynwave.c:555-572, `DynamicWaveHydraulics`). Both are therefore functions of the pass, like
  the other conduit terms; `seepage_depth_limit` is the depth the width is capped at.
- a **storage unit evaporates** ``Fevap * Evap.rate * area`` (``area`` at the step-start depth,
  and ``Fevap * Evap.rate`` itself when the volume is FUDGE or less) and **exfiltrates**
  through its bottom and banks (`exfil.c::exfil_getLoss`): Ksat times the conductivity
  adjustment over the bottom area plus the bank area (the surface area beyond the bottom),
  or, with a positive IMDmax, SWMM's modified Green-Ampt model (`runoff._GreenAmptState`, which
  keeps its state from step to step) at the depth of the water above the bottom and half the
  depth above the banks' foot. The two are computed once per routing step from the step-start
  depth and volume and capped together at the stored volume over the step
  (`node.c::storage_getLosses`, node.c:1052-1100, routing.c:363-365); `StorageLosses.rates`.

The climate is whatever `climate.c::climate_setState` last set: in a file with subcatchments the
last runoff step taken before the routing step (`runoff.RunoffModel`), else the state at the
start of the routing step (swmm5.c:556). ``DRY_ONLY`` applies to the subcatchments alone
(subcatch.c:685) and so does not affect any loss here.

Gradients. The loss rates are functions of the conduit depths and volumes and of the storage
depths and volumes, all carried by the autograd graph (a storage unit's from the step-start
state), so a gradient with respect to a parameter that moves them -- an inflow, a roughness,
an invert -- includes the loss. The evaporation rate, the conductivity adjustment and the
Green-Ampt state are constants of a step (plain floats from the climate) and carry no gradient.

Every source reference is to SWMM 5.2.4 ``src/solver``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from noodl.apps.sewer.runoff import (
    GreenAmpt,
    Hydrology,
    RunoffModel,
    _Climate,
    _Env,
    _GreenAmptState,
)
from noodl.apps.sewer.storage import StorageCurves
from noodl.apps.sewer.xsect_tables import LCF, QCF

if TYPE_CHECKING:
    from noodl.apps.sewer.network import Exfiltration, Manhole

Tensor = torch.Tensor
F64 = torch.float64
#: SWMM's ``FUDGE`` (consts.h:72), ft3 for a volume.
FUDGE = 1.0e-4
#: SWMM's ``BIG`` (consts.h): the bank area and depth of a storage shape without a curve.
BIG = 1.0e10

#: Drivers of a network with losses (`DynamicWaveHydraulics`): the evaporation rate (m/s), the
#: conductivity adjustment (-) and each junction's storage loss (m3/s).
EVAP_DRIVER = "evap_rate"
HYDCON_DRIVER = "hydcon"
STORAGE_DRIVER = "storage_loss"

#: ``ywMax / yFull`` of the shapes whose widest point SWMM sets as a fraction of the height
#: (xsect.c:249-630); the open shapes with their widest point at the top are 1.
_WIDEST_AT = {
    "CIRCULAR": 0.5, "FORCE_MAIN": 0.5, "FILLED_CIRCULAR": 0.5, "EGGSHAPED": 0.64,
    "HORSESHOE": 0.5, "GOTHIC": 0.45, "CATENARY": 0.25, "SEMIELLIPTICAL": 0.15,
    "BASKETHANDLE": 0.2, "SEMICIRCULAR": 0.15, "RECT_CLOSED": 1.0, "RECT_OPEN": 1.0,
    "RECT_TRIANG": 1.0, "RECT_ROUND": 1.0, "TRAPEZOIDAL": 1.0, "TRIANGULAR": 1.0,
    "PARABOLIC": 1.0, "POWERFUNC": 1.0, "HORIZ_ELLIPSE": 0.48, "VERT_ELLIPSE": 0.48,
    "ARCH": 0.28,
}


def seepage_depth_limit(geo) -> Tensor:
    """Each conduit's ``xsect.ywMax`` (ft): the depth of the lowest of its widest points, at
    which the seepage width stops growing (link.c:1370-1374). A fraction of the full height for
    the closed-form shapes (xsect.c:249-630), the full height less the filled bottom for a
    MODBASKETHANDLE (xsect.c:462), and for a CUSTOM or IRREGULAR section the first maximum of its
    width table (``xsect_setCustomXsectParams``, ``getTransectParams``: ``yFull iMax / 50``)."""
    yf = geo.y_full
    if geo.xsect is None:
        return 0.5 * yf
    xs = geo.xsect
    out = torch.zeros_like(yf)
    for i, kind in enumerate(xs.kinds):
        if kind in _WIDEST_AT:
            out[i] = _WIDEST_AT[kind] * yf[i]
        elif kind == "MODBASKETHANDLE":
            out[i] = yf[i] - xs.values["y_bot"][i]
        elif kind in ("CUSTOM", "IRREGULAR"):
            row = xs.tables["width"][i]
            top = 0
            widest = float(row[0])
            for j in range(1, row.shape[0]):
                if float(row[j]) < widest:
                    break
                widest = float(row[j])
                top = j
            out[i] = yf[i] * top / (row.shape[0] - 1.0)
        else:
            raise ValueError(
                f"conduit with the {kind} section seeps: SWMM sets no depth of widest point for "
                f"that shape, so its seepage width is not reproduced")
    return out


class ClimateValues(NamedTuple):
    """What a routing step reads of the climate: ``evap`` `Evap.rate` (ft/s), ``hydcon``
    `Adjust.hydconFactor`, ``infil`` the `InfilFactor` the Green-Ampt model of a storage unit
    reads (the last subcatchment's, else 1) and ``recovery`` `Evap.recoveryFactor`."""

    evap: float
    hydcon: float
    infil: float
    recovery: float


class ClimateClock:
    """`climate.c::climate_setState` as the routing steps see it.

    `source` is the `runoff.Hydrology` that gives the evaporation and adjustments (a network's
    `climate_source`; ``None``: zero evaporation and unit factors). With subcatchments the
    climate is that of the runoff steps SWMM takes before each routing step: the clock runs a
    `runoff.RunoffModel` (the given `runoff`, shared with the lateral-inflow driver, else its
    own); without, the state at the start of each routing step (swmm5.c:556), which is read at
    ``getDateTime(NewRoutingTime)``, one millisecond late."""

    def __init__(self, source: Hydrology | None, runoff: RunoffModel | None = None) -> None:
        self.source = source
        self.model: RunoffModel | None = None
        self.climate: _Climate | None = None
        if source is not None and source.subcatchments:
            self.model = runoff if runoff is not None else RunoffModel(source)
            self.climate = self.model.climate
        elif source is not None:
            self.climate = _Climate(source)

    def at(self, start_ms: float, end_ms: float) -> ClimateValues:
        """The climate for the routing step from `start_ms` to `end_ms` (ms since START).
        Calls must come with increasing times."""
        if self.climate is None:
            return ClimateValues(0.0, 1.0, 1.0, 1.0)
        if self.model is not None:
            self.model.advance_to(end_ms)
            c = self.climate
            return ClimateValues(c.rate, c.hydcon_factor, self.model.env.hydcon,
                                 c.recovery_factor)
        self.climate.set_state((start_ms + 1.0) / 1000.0)
        c = self.climate
        return ClimateValues(c.rate, c.hydcon_factor, 1.0, c.recovery_factor)


def exfiltration_geometry(curve: tuple, us_units: bool) -> tuple[float, float, float, float]:
    """``(bottom area, bank min depth, bank max depth, bank max area)`` in ft2 and ft of a
    storage unit's exfiltration (`exfil.c::exfil_initState`, exfil.c:78-137).

    A TABULAR curve gives its bottom area at depth 0 (0 if the first depth is above 0, as
    `table_lookupEx`), the depth the area stays at its first value (the bank's foot) and the
    last depth and area at which it still grows (the banks' top), converted from the curve's
    user units. The other shapes take the bottom area as the unit's constant area ``a0``
    (``a0 + a1`` for a FUNCTIONAL one of exponent 0), and the banks run from depth 0 up
    without limit. SWMM does not convert that ``a0`` from the file's length unit: it is read as
    ft2, so for an SI file it is the number of m2 taken as ft2 (``us_units`` False)."""
    kind = curve[0]
    if kind == "tabular":
        depths, areas = (list(map(float, v)) for v in curve[1:3])
        bottom = areas[0] if depths[0] <= 0.0 else 0.0
        min_depth = max_depth = max_area = 0.0
        last = areas[0]
        for d, a in zip(depths[1:], areas[1:], strict=True):
            if a < last:
                break
            if a > last:
                max_area, max_depth = a, d
            elif max_area == 0.0:
                min_depth = d
            else:
                break
            last = a
        return (bottom / LCF**2, min_depth / LCF, max_depth / LCF, max_area / LCF**2)
    if kind == "functional":
        _, a1, a2, a0 = curve
        bottom = a0 + (a1 if a2 == 0.0 else 0.0)
    elif kind in ("cylindrical", "conical", "pyramidal"):
        from noodl.apps.sewer.storage import _coefficients
        bottom = _coefficients(curve)[1]
    else:
        raise ValueError(f"a {kind} storage unit has no exfiltration geometry in SWMM 5.2.4")
    return (bottom / LCF**2 if us_units else bottom, 0.0, BIG, BIG)


class StorageLosses:
    """The evaporation and exfiltration of the storage units of one network, per routing step
    (`node.c::storage_getLosses`, `exfil.c::exfil_getLoss`; the module docstring).

    `manholes` are the network's junctions in order (only the storage units with an
    `evaporation_factor` or `exfiltration` lose water), `curves` the `StorageCurves` of the
    junctions. ``rates`` is the cfs lost from every junction over a step of `dt` s, from the
    step-start depth. The Green-Ampt state of an exfiltration with a positive IMDmax is kept
    here, one object for the bottom and one for the banks of each unit, and advanced by every
    call of ``rates``; ``reset`` returns it to the start of a run."""

    def __init__(self, manholes: list[Manhole], curves: StorageCurves | None,
                 us_units: bool = False) -> None:
        n = len(manholes)
        self.curves = curves
        #: SWMM's ft3 per true m3 of the volume `storage_getVolume` returns: it divides the
        #: curve's volume in user units by UCF(VOLUME), which is 1 for US units (so the true ft3)
        #: but 0.02832 for SI (node.c:953, 962, 972), 1.1e-4 more ft3 than the true 35.3147 per m3
        self.volume_per_m3 = 1.0 / LCF**3 if us_units else 1.0 / QCF
        self.fevap = torch.tensor([m.evaporation_factor for m in manholes], dtype=F64)
        self.has_exfil = torch.tensor([m.exfiltration is not None for m in manholes],
                                      dtype=torch.bool)
        self.any_evap = bool((self.fevap != 0.0).any())
        self.any_exfil = bool(self.has_exfil.any())
        self.ks = torch.zeros(n, dtype=F64)               # ft/s
        self.constant = torch.zeros(n, dtype=torch.bool)  # IMDmax == 0: the rate is Ks hydcon
        geometry = [(0.0, 0.0, 0.0, 0.0)] * n
        self._params: dict[int, Exfiltration] = {}
        for i, m in enumerate(manholes):
            x = m.exfiltration
            if x is None:
                continue
            self.ks[i] = x.conductivity / LCF
            self.constant[i] = x.initial_deficit == 0.0
            geometry[i] = exfiltration_geometry(m.storage, x.us_units)
            if x.initial_deficit != 0.0:
                self._params[i] = x
        cols = list(zip(*geometry, strict=True)) if n else [(), (), (), ()]
        (self.bottom_area, self.bank_min_depth, self.bank_max_depth,
         self.bank_max_area) = (torch.tensor(c, dtype=F64) for c in cols)
        self._ga: dict[int, tuple[_GreenAmptState, _GreenAmptState]] = {}
        self.reset()

    def reset(self) -> None:
        """Return the Green-Ampt states to the start of a run (`grnampt_initState`)."""
        self._ga = {}
        for i, x in self._params.items():
            p = GreenAmpt(suction=x.suction, conductivity=x.conductivity,
                          initial_deficit=x.initial_deficit, modified=True)
            self._ga[i] = (_GreenAmptState(p), _GreenAmptState(p))

    def rates(self, depth_ft: Tensor, dt: float, climate: ClimateValues) -> Tensor:
        """The loss rate (cfs, >= 0 unless the evaporation rate is negative) of every junction
        over a step of `dt` s that starts at the depths `depth_ft` (ft, ``(..., n)``).

        The stored volume and surface area are the storage curves' at that depth. A unit that
        loses nothing (no Fevap or exfiltration) gets 0."""
        curves = self.curves
        if curves is None:
            return torch.zeros_like(depth_ft)
        depth_m = depth_ft * LCF
        volume = curves.volume(depth_m) * self.volume_per_m3
        area = curves.area(depth_m) / LCF**2
        evap_node = climate.evap * self.fevap                          # node.c:1072
        active = (evap_node > 0.0) | self.has_exfil                    # node.c:1076
        evap = torch.where(active & (volume > FUDGE), area * evap_node, evap_node)  # :1085
        exfil = torch.zeros_like(evap)
        if self.any_exfil:
            exfil = self._exfiltration(depth_ft, area, dt, climate)
        total = (evap + exfil) * dt                                    # node.c:1096
        over = active & (total > volume)
        ratio = torch.where(over, volume / torch.where(over, total, torch.ones_like(total)),
                            torch.ones_like(total))                    # node.c:1099-1101
        return (evap + exfil) * ratio

    def _exfiltration(self, depth: Tensor, area: Tensor, dt: float, climate: ClimateValues
                      ) -> Tensor:
        """`exfil.c::exfil_getLoss` (cfs) for every unit with an exfiltration, 0 elsewhere."""
        zero = torch.zeros_like(area)
        # exfil.c:158-161 / 178-182: Ks hydcon where IMDmax is 0
        constant_rate = self.ks * climate.hydcon
        btm_rate = torch.where(self.constant, constant_rate, zero)
        bank_depth = torch.where(
            depth > self.bank_max_depth,
            depth - self.bank_max_depth + (self.bank_max_depth - self.bank_min_depth) / 2.0,
            (depth - self.bank_min_depth) / 2.0)                       # exfil.c:187-194
        bank_area = torch.minimum(area, self.bank_max_area) - self.bottom_area  # exfil.c:173
        banks_on = self.has_exfil & (depth > self.bank_min_depth) & (bank_area > 0.0)
        bank_rate = torch.where(self.constant, constant_rate, zero)
        if self._ga:
            if depth.dim() != 1:
                raise ValueError("a storage exfiltration with a positive IMDmax keeps a state "
                                 "per run; it cannot be batched")
            env = _Env()
            env.hydcon, env.recovery = climate.infil, climate.recovery
            btm_list = btm_rate.tolist()
            bank_list = bank_rate.tolist()
            for i, (bottom, bank) in self._ga.items():
                d = float(depth[i])
                btm_list[i] = bottom.get_infil(dt, 0.0, 0.0, d, env)   # exfil.c:163-164
                if bool(banks_on[i]):
                    bank_list[i] = bank.get_infil(dt, 0.0, 0.0, float(bank_depth[i]), env)
            btm_rate = torch.tensor(btm_list, dtype=F64)
            bank_rate = torch.tensor(bank_list, dtype=F64)
        flow = btm_rate * self.bottom_area                             # exfil.c:166
        flow = flow + torch.where(banks_on, bank_area * bank_rate, zero)
        return torch.where(self.has_exfil, flow, zero)


__all__ = ["ClimateClock", "ClimateValues", "StorageLosses", "exfiltration_geometry",
           "seepage_depth_limit"]
