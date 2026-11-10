"""Pump energy and cost over an extended-period run, as EPANET 2.2 accumulates and reports it.

Transcribed from EPANET 2.2 (tag v2.2):

- `hydraul.c`, `addenergy`: at the start of every hydraulic step of `hstep` s (`nexthyd`, whose
  step is not shortened to the duration: the last step of a run whose duration is not a
  multiple of the step counts whole, and a run of duration 0 counts one hour) every pump
  whose status is above CLOSED adds `dt` (hours) to its time on line, `e dt` to its
  efficiency, `p/q dt`, `p dt` and `c p dt` to its energy per flow, energy and cost, and
  its peak power is the largest `p`; the network's peak total power `Emax` is the largest sum
  of the running pumps' `p`. The price `c` is the pump's own price, else the global one,
  times the pump's price pattern, else the global price pattern, at the pattern period
  `(t + Pstart) / Pstep`.
- `hydraul.c`, `getenergy`: no energy for a pump whose status is CLOSED (or shut by the
  solve at its shut-off head); `p = dh q SpGrav / 8.814 / e kw_per_hp` in feet, cfs and
  horsepower, the same for a constant-power pump (its power enters through the solved head
  gain, not directly); the efficiency `e` is the pump's efficiency curve at the flow
  `q / speed` (`interp`: linear, the end values beyond the curve), adjusted for the pump's
  speed (Sarbu and Borza, `100 - (100 - e) (1 / speed)^0.1`), or the global efficiency, and
  is limited to 1 to 100 percent.
- `output.c`, `saveenergy`: what is reported, per pump: the percent of the run it is on line,
  its average efficiency, kWh per volume and average power over the time on line, its peak
  power, and its cost per day; `Emax` times the demand charge is the network's demand
  charge.

`PumpEnergy` holds those in SI: kWh per m3 (EPANET reports kWh per million gallons for a file
in US units: multiply by 3785.411784 m3 per million gallons).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from noodl.apps.water.controls import CLOSED
from noodl.apps.water.tanks import QZERO

Tensor = torch.Tensor
F64 = torch.float64

#: EPANET's `KWperHP` (`epanet2_enums.h`/`types.h`), kW per horsepower.
KW_PER_HP = 0.7457
#: The constant of `getenergy`: horsepower = `dh q SpGrav / 8.814` in feet and cfs.
_HP_FT_CFS = 8.814
#: Metres per foot (EPANET's `MperFT`), the head conversion of every SI file.
_M_PER_FT = 0.3048


@dataclass
class PumpEnergy:
    """Each pump's energy use over a run (`pump_names` in `WaterNetwork.pumps` order), as
    EPANET reports it (`output.c`, `saveenergy`). `...` is the batch shape of the run.

    `utilisation` (..., n_pumps) the percent of the run's duration the pump is on line (a
    run whose duration is not a multiple of the step counts its last step whole, as EPANET
    does, so this can exceed 100); `efficiency` (..., n_pumps) the average efficiency in
    percent over the time on line; `energy_per_volume` (..., n_pumps) kWh per m3 pumped,
    averaged over the time on line; `average_power` and `peak_power` (..., n_pumps) kW, the
    first over the time on line; `cost` (..., n_pumps) the cost per day; `demand_charge`
    (...,) the network's peak total power times `[ENERGY] DEMAND CHARGE`. A pump that is
    never on line reads 0 throughout."""

    pump_names: list[str]
    utilisation: Tensor
    efficiency: Tensor
    energy_per_volume: Tensor
    average_power: Tensor
    peak_power: Tensor
    cost: Tensor
    demand_charge: Tensor


def interpolate_curves(x: Tensor, y: Tensor, xx: Tensor) -> Tensor:
    """EPANET's `interp` (`hydraul.c`/`input1.c`) for one curve per row: `x` and `y`
    (n_rows, L) with `x` rising along each row (a shorter curve padded by repeating its last
    point), `xx` (n_rows, B): the linear interpolation, the first value at or below the first
    `x` and the last value beyond the last."""
    top = x.shape[-1] - 1
    k = torch.searchsorted(x, xx.detach().contiguous())     # first x[k] >= xx
    upper = k.clamp(1, top) if top > 0 else k.clamp(0, 0)
    lower = (upper - 1).clamp(min=0)
    x0, x1 = x.gather(1, lower), x.gather(1, upper)
    y0, y1 = y.gather(1, lower), y.gather(1, upper)
    dx = x1 - x0
    inside = y1 - (x1 - xx) * (y1 - y0) / torch.where(dx.abs() < 1e-12, torch.ones_like(dx), dx)
    inside = torch.where(dx.abs() < 1e-12, y1, inside)
    return torch.where(k <= 0, y[:, :1], torch.where(k > top, y[:, -1:], inside))


class EnergyMeter:
    """The accumulation of `addenergy` over a run of `model`: `add` once per hydraulic step,
    `result` at the end."""

    def __init__(self, model, duration: int) -> None:
        from noodl.apps.water.network import _rounded_cfs_per_m3s, _topology

        net = model.water_network
        self.net = net
        pumps = net.pumps
        self.names = [p.name for p in pumps]
        self.cols = torch.tensor([model.link_names.index(p.name) for p in pumps],
                                 dtype=torch.long)
        ends = torch.tensor(_topology(model)[0], dtype=torch.long)
        self.from_node, self.to_node = ends[self.cols, 0], ends[self.cols, 1]
        self.head_scale = float(model.head_scale)
        cfs = _rounded_cfs_per_m3s(net.options.flow_units)
        self.cfs_per_m3s = 1.0 / _M_PER_FT**3 if cfs is None else cfs
        self.qzero = QZERO if cfs is None else 1e-6 / self.cfs_per_m3s
        self.kw_scale = (net.options.specific_gravity / _HP_FT_CFS * KW_PER_HP
                         * self.cfs_per_m3s / _M_PER_FT)         # kW per (m head m3/s)
        self.duration = int(duration)
        energy = net.energy
        n = len(pumps)
        with_curve = [p.efficiency_curve is not None for p in pumps]
        self.has_curve = torch.tensor(with_curve, dtype=torch.bool)
        self.any_curve = any(with_curve)
        global_eff = float(energy.global_efficiency)
        curves = [net.curves[p.efficiency_curve] if p.efficiency_curve is not None
                  else ((0.0, global_eff), (1.0, global_eff)) for p in pumps]
        width = max((len(c) for c in curves), default=1)
        xs = torch.zeros(n, width, dtype=F64)
        ys = torch.zeros(n, width, dtype=F64)
        for i, c in enumerate(curves):
            for j in range(width):
                x, y = c[min(j, len(c) - 1)]
                xs[i, j], ys[i, j] = x, y
        self.curve_x, self.curve_y = xs, ys
        # EPANET limits the efficiency to 1..100 percent (`getenergy`)
        self.global_fraction = min(max(global_eff, 1.0), 100.0) / 100.0
        self._periods: dict[int, Tensor] = {}
        zeros = torch.zeros(n, dtype=F64)
        self.on = self.eff = self.per_flow = self.kwh = self.cost = zeros
        self.peak = zeros
        self.emax = torch.zeros((), dtype=F64)

    def _price(self, period: int) -> Tensor:
        """Each pump's price at the pattern period `period` (`addenergy`: a pump's own price,
        else the global one, times its own pattern, else the global pattern)."""
        if period not in self._periods:
            net = self.net
            energy = net.energy

            def factor(name):
                values = net.patterns[name]
                return float(values[period % len(values)])

            f0 = 1.0 if energy.global_pattern is None else factor(energy.global_pattern)
            out = []
            for p in net.pumps:
                c = p.price if p.price is not None and p.price > 0.0 else energy.global_price
                out.append(c * (f0 if p.price_pattern is None else factor(p.price_pattern)))
            self._periods = {period: torch.tensor(out, dtype=F64)}
        return self._periods[period]

    def add(self, period: int, hours: float, solved, states: Tensor,
            speed: Tensor | None) -> None:
        """Add a step of `hours` at pattern period `period` from the solved state `solved`;
        `states` (..., n_pumps) are the pumps' states (`CLOSED` or above) and `speed`
        (..., n_pumps) their relative speeds (needed only for efficiency curves)."""
        on = (states != CLOSED).to(F64)
        phi = solved["water.phi"]
        q = solved["water.q"][..., self.cols].abs()
        if self.any_curve:
            speed = torch.where(on > 0, speed, torch.ones_like(speed))
            xx = (q / speed).movedim(-1, 0)
            shape = xx.shape
            e = interpolate_curves(self.curve_x, self.curve_y, xx.reshape(shape[0], -1))
            e = e.reshape(shape).movedim(0, -1)
            sarbu = 100.0 - (100.0 - e) * (1.0 / speed) ** 0.1
            e = torch.where(self.has_curve, sarbu, e).clamp(1.0, 100.0) / 100.0
        else:
            e = self.global_fraction
        # a pump that is not on line (`on` = 0) draws nothing: `getenergy` returns 0 for it
        kw = (phi[..., self.from_node] - phi[..., self.to_node]).abs() * q * on * (
            self.kw_scale / self.head_scale / e)
        self.on = self.on + on * hours
        self.eff = self.eff + on * e * hours
        self.per_flow = self.per_flow + kw / q.clamp(min=self.qzero) * hours
        self.kwh = self.kwh + kw * hours
        self.peak = torch.maximum(self.peak, kw)
        self.cost = self.cost + self._price(period) * kw * hours
        self.emax = torch.maximum(self.emax, kw.sum(-1))

    def result(self) -> PumpEnergy:
        """The accumulated statistics, converted as `saveenergy` does."""
        hdur = self.duration / 3600.0
        on = self.on
        used = on > 0.0

        def mean(x):
            return torch.where(used, x / torch.where(used, on, torch.ones_like(on)),
                               torch.zeros_like(x))

        if hdur == 0.0:
            fraction, cost = on, self.cost * 24.0
            eff, per_flow, kwh = self.eff, self.per_flow, self.kwh
        else:
            fraction, cost = on / hdur, self.cost * 24.0 / hdur
            eff, per_flow, kwh = mean(self.eff), mean(self.per_flow), mean(self.kwh)
        return PumpEnergy(
            pump_names=list(self.names), utilisation=fraction * 100.0,
            efficiency=eff * 100.0, energy_per_volume=per_flow / 3600.0,
            average_power=kwh, peak_power=self.peak, cost=cost,
            demand_charge=self.emax * self.net.energy.demand_charge)


__all__ = ["EnergyMeter", "PumpEnergy", "interpolate_curves"]
