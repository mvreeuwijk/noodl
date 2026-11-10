"""Pump energy and cost over an extended-period run, as EPANET 2.2 accumulates and reports it.

Transcribed from EPANET 2.2 (tag v2.2):

- `hydraul.c`, `addenergy`: once per hydraulic step of `hstep` s (`nexthyd`, whose step is not
  shortened to the duration: the last step of a run whose duration is not a multiple of the
  step counts whole, and a run of duration 0 counts one hour), called after `timestep` has
  moved the tanks to the step's end and taken the actions of the rules checked there
  (`simulate._as_addenergy_sees`), every pump whose status is then above CLOSED adds `dt`
  (hours) to its time on line, `e dt` to its efficiency, `p/q dt`, `p dt` and `c p dt` to
  its energy per flow, energy and cost, and its peak power is the largest `p`; the network's
  peak total power `Emax` is the largest sum of the running pumps' `p`. The price `c` is
  the pump's own price, else the global one, times the pump's price pattern, else the
  global price pattern, at the pattern period `(t + Pstart) / Pstep`.
- `hydraul.c`, `getenergy`: no energy for a pump whose status is CLOSED (or shut by the
  solve at its shut-off head, `XHEAD`, unless a rule's action opened it at the step's end,
  `takeactions`); `p = dh q SpGrav / 8.814 / e kw_per_hp` in feet, cfs and
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
from noodl.apps.water.elements import CBIG_FT_PER_CFS
from noodl.apps.water.tanks import QZERO

Tensor = torch.Tensor
F64 = torch.float64

#: EPANET's `KWperHP` (`epanet2_enums.h`/`types.h`), kW per horsepower.
KW_PER_HP = 0.7457
#: The constant of `getenergy`: horsepower = `dh q SpGrav / 8.814` in feet and cfs.
_HP_FT_CFS = 8.814
#: EPANET's `TINY` (`types.h`): `interp` returns `y[k]` across a step in `x` below it.
TINY = 1e-6
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


def interpolate_curves(x: Tensor, y: Tensor, xx: Tensor, tiny: float = TINY) -> Tensor:
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
    inside = y1 - (x1 - xx) * (y1 - y0) / torch.where(dx.abs() < tiny, torch.ones_like(dx), dx)
    inside = torch.where(dx.abs() < tiny, y1, inside)
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
        # EPANET's `TINY` is in the file's flow units (`getenergy` interpolates `q * Ucf`)
        from noodl.apps.water.inp import FLOW_UNITS

        self.tiny = TINY * (1.0 if net.options.flow_units is None
                            else FLOW_UNITS[net.options.flow_units])
        self.duration = int(duration)
        energy = net.energy
        n = len(pumps)
        with_curve = [p.efficiency_curve is not None for p in pumps]
        self.has_curve = torch.tensor(with_curve, dtype=torch.bool)
        self.constant_power = torch.tensor([p.power is not None for p in pumps],
                                           dtype=torch.bool)
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
        # price and price pattern per pump (`addenergy`): a pump's own price when positive,
        # else the global; its own pattern, else the global one, else a constant 1. The
        # patterns used are rows of one table, so a period's prices are one gather.
        rows = [(1.0,)]
        index = {None: 0}
        for name in [energy.global_pattern] + [p.price_pattern for p in pumps]:
            if name is not None and name not in index:
                index[name] = len(rows)
                rows.append(tuple(net.patterns[name]))
        self._pattern_length = torch.tensor([len(r) for r in rows], dtype=torch.long)
        self._pattern_table = torch.ones(len(rows), max(len(r) for r in rows), dtype=F64)
        for i, r in enumerate(rows):
            self._pattern_table[i, :len(r)] = torch.tensor(r, dtype=F64)
        self._pattern_row = torch.tensor(
            [index[p.price_pattern if p.price_pattern is not None else energy.global_pattern]
             for p in pumps], dtype=torch.long)
        self._base_price = torch.tensor(
            [p.price if p.price is not None and p.price > 0.0 else energy.global_price
             for p in pumps], dtype=F64)
        self._steps: list = []

    def add(self, period: int, hours: float, solved, states: Tensor,
            speed: Tensor | None, opened: Tensor | None = None) -> None:
        """Take a step of `hours` at pattern period `period` from the solved state `solved`;
        `states` (..., n_pumps) are the pumps' states (`CLOSED` or above) and `speed`
        (..., n_pumps) their relative speeds (needed only for efficiency curves). The
        statistics are accumulated over all the steps at once, in `result`.

        `opened` (..., n_pumps) marks the pumps that were CLOSED in the solve (closed, or shut
        at their shut-off head, EPANET's `XHEAD`) and that a rule's action opened at the
        step's end (a STATUS OPEN, or a changed positive SETTING). EPANET charges them with
        the `LinkFlow` they then have (`hydraul.c`, `getenergy`):

        - a constant-power pump: its `Q0`, 1 cfs, which `setlinkstatus` (from `CLOSED`
          proper) and `setlinksetting` (from any status `<= CLOSED`) write when they reopen
          it (`resetpumpflow`; `input2.c`, `updatepumpparams`: ``Q0 = 1.0``, "Init. flow =
          1 cfs"). A STATUS OPEN on an `XHEAD` pump would leave its flow, but a
          constant-power pump is never `XHEAD` (`hydstatus.c`, `pumpstatus`: its `hmax` is
          `BIG`), so the two cases do not part here;
        - any other pump (`resetpumpflow` leaves it): the flow its closed link carried in the
          solve, ``|dh| / CBIG`` (`hydcoeffs.c`, `pumpcoeff`: a closed or `XHEAD` pump's
          conductance ``1 / CBIG``, CBIG = 1e8 ft per cfs), where this model's closed link
          carries none. Its power is negligible, but its energy per volume ``p / q`` is not.

        A control or a pattern that reopens a pump acts at the start of a step, before its
        solve (`runhyd`), so the solve's flow is the one charged."""
        phi = solved["water.phi"]
        u, v = phi[..., self.from_node], phi[..., self.to_node]
        q = solved["water.q"][..., self.cols]
        if opened is not None and bool(opened.any()):
            leak = (u - v).abs() / self.head_scale / _M_PER_FT / CBIG_FT_PER_CFS
            reset = torch.where(self.constant_power, 1.0, leak)   # `resetpumpflow`: Q0
            q = torch.where(opened, reset / self.cfs_per_m3s, q)
        self._steps.append((period, hours, u, v, q, states, speed))

    def _price(self, periods: Tensor) -> Tensor:
        """Each pump's price (T, n_pumps) at the pattern periods `periods` (T,)
        (`addenergy`)."""
        row = self._pattern_row
        factor = self._pattern_table[row, periods[:, None] % self._pattern_length[row]]
        return self._base_price * factor

    def result(self) -> PumpEnergy:
        """The statistics over the steps taken, accumulated and converted as `addenergy` and
        `saveenergy` do."""
        n = len(self.names)
        steps = self._steps
        if not steps:
            zero = torch.zeros(n, dtype=F64)
            return PumpEnergy(list(self.names), zero, zero, zero, zero, zero, zero,
                              torch.zeros((), dtype=F64))
        batch = torch.broadcast_shapes(*(x.shape[:-1] for step in steps for x in step[2:6]))

        def stack(k):
            return torch.stack([step[k].expand(batch + (n,)) for step in steps], dim=-2)

        hours = torch.tensor([step[1] for step in steps], dtype=F64)[:, None]
        periods = torch.tensor([step[0] for step in steps], dtype=torch.long)
        dh = (stack(2) - stack(3)).abs() / self.head_scale
        q = stack(4).abs()
        on = (stack(5) != CLOSED).to(F64)
        if self.any_curve:
            speed = torch.where(on > 0, stack(6), torch.ones_like(on))
            xx = (q / speed).movedim(-1, 0)
            shape = xx.shape
            e = interpolate_curves(self.curve_x, self.curve_y, xx.reshape(shape[0], -1),
                                   self.tiny)
            e = e.reshape(shape).movedim(0, -1)
            sarbu = 100.0 - (100.0 - e) * (1.0 / speed) ** 0.1
            e = torch.where(self.has_curve, sarbu, e).clamp(1.0, 100.0) / 100.0
        else:
            e = self.global_fraction
        # a pump that is not on line (`on` = 0) draws nothing: `getenergy` returns 0 for it
        kw = dh * q * on * (self.kw_scale / e)
        on_hours = (on * hours).sum(-2)
        eff = (on * e * hours).sum(-2)
        per_flow = (kw / q.clamp(min=self.qzero) * hours).sum(-2)
        kwh = (kw * hours).sum(-2)
        peak = kw.amax(-2)
        cost = (self._price(periods) * kw * hours).sum(-2)
        emax = kw.sum(-1).amax(-1)
        used = on_hours > 0.0

        def mean(x):
            return torch.where(used, x / torch.where(used, on_hours, torch.ones_like(on_hours)),
                               torch.zeros_like(x))

        hdur = self.duration / 3600.0
        if hdur == 0.0:
            fraction, cost = on_hours, cost * 24.0
        else:
            fraction, cost = on_hours / hdur, cost * 24.0 / hdur
            eff, per_flow, kwh = mean(eff), mean(per_flow), mean(kwh)
        return PumpEnergy(
            pump_names=list(self.names), utilisation=fraction * 100.0,
            efficiency=eff * 100.0, energy_per_volume=per_flow / 3600.0,
            average_power=kwh, peak_power=peak, cost=cost,
            demand_charge=emax * self.net.energy.demand_charge)


__all__ = ["EnergyMeter", "PumpEnergy", "interpolate_curves"]
