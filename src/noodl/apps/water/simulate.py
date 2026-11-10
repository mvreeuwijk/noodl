"""EPANET 2.2's extended-period run: `water_run` and its results (`WaterResults`).

The run is EPANET's hydraulic loop (`epanet.c`, `EN_solveH`: `runhyd` then `nexthyd` until
the duration), transcribed from `hydraul.c` (tag v2.2):

1. at the step's start time `t` (whole seconds) the pattern values are applied
   (`demands`): junction demands (`pattern_factor`, each junction's own pattern or the
   default one), reservoir heads, and pump speeds -- a speed pattern sets the pump's carried
   speed, 0 closing it and a positive speed opening a closed one (`setlinksetting`);
2. the simple controls are applied (`controls`, inside `TankLevels`) and the network is
   solved, the junction-pressure controls re-solving it up to `MAX_PRESSURE_SWITCHES` times
   (`water_steady`, EPANET's `pswitch`);
3. the step is the hydraulic step shortened to the next pattern period (EPANET's own
   arithmetic, `n Pstep - t` with `n = (t + Pstart) / Pstep + 1`, `timestep`), the next
   hydraulic report time (`Rtime`, every REPORT TIMESTEP from 0), the time a tank fills or
   empties (`tanktimestep`, `TankLevels.time_to_limit`) and the time to the next control
   that would change a link (`controltimestep`, `TankLevels.control_time`), both rounded to
   whole seconds as EPANET rounds them, and the run's end;
4. the tanks move (`tanklevels`): `V += q dt`, a tank within one second's inflow of a limit
   put on it. With `[RULES]` the step is walked in rule steps (`ruletimestep`): the first
   to the next multiple of RULE TIMESTEP, the tanks moved and the rules checked after each
   (`apply_rules`); a rule that takes an action ends the step there.

Results are kept at the report times (`quality.c`, `runqual`, which writes EPANET's output):
every solve at a time at or past the next report time `Rtime`, which starts at REPORT START
and grows by REPORT TIMESTEP. They are labelled with those report times. The run ends at the
duration (EPANET solves once more past it when the duration is not a multiple of the step,
which reports nothing; here the last step ends at the duration). The pumps' energy
(`energy.py`, `addenergy`) is accumulated over every step that starts before the duration,
the last one at EPANET's own length, not shortened to the duration.

BATCHES. A batch of instances (a leading dimension of the state or the drivers) takes one
shared step: the shortest any instance needs (the spec's "Batches"), so an instance takes
the step boundaries of the others' events as well as its own; a rule that acts in one
instance ends the step for all. A single run follows EPANET's step sequence exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from noodl.apps.water.controls import CLOSED, OPEN
from noodl.apps.water.demand import EmitterOutflow, PressureDrivenDemand
from noodl.apps.water.elements import (
    DEMAND_FACTOR_KEY,
    RESERVOIR_HEAD_KEY,
    SETTING_KEY,
    TIME_KEY,
)
from noodl.apps.water.energy import EnergyMeter, PumpEnergy
from noodl.apps.water.network import (
    Pump,
    WaterPipe,
    apply_rules_with_states,
    initial_drivers,
    initial_state,
    link_states_from_cut_off,
    settle_tanks,
    solved_link_states,
    water_steady_and_cut_off,
)
from noodl.apps.water.tanks import TankVolumeCurve

Tensor = torch.Tensor
F64 = torch.float64


@dataclass
class WaterResults:
    """A run's results at its report times. `...` is the batch shape of the run.

    `time` (T,) s since the start; `head` and `pressure` (..., T, n_nodes) m, the pressure
    the head less the node's elevation (a reservoir's base head, a tank's bottom); `flow`
    (..., T, n_links) m3/s; `status` (..., T, n_links) the links' states as the solve found
    them (`link_states`: `CLOSED`, `OPEN`, `ACTIVE`); `setting` (..., T, n_links) in SI: a
    pump's relative speed (0 when its status is CLOSED, as EPANET's `setlinkstatus` sets
    it), a PRV's, PSV's or PBV's head (m), an FCV's flow (m3/s), a TCV's loss coefficient;
    `tank_level` (..., T, n_tanks) m above the bottom; `tank_overflow` (..., T, n_tanks)
    m3/s, the flow a tank that may overflow spills over the step that starts at the report
    time (its inflow less its volume change; at the last report time, a full tank's whole
    inflow); `demand` (..., T, n_junctions) m3/s the junctions' delivered demands (the
    requested ones, or the pressure-driven deliveries) and `emitter_flow` (..., T,
    n_junctions) m3/s their emitters' outflow (EPANET reports the sum of the two as the
    demand). `energy` is the pumps' energy and cost over the run (`PumpEnergy`;
    `None` for a network without pumps). Heads, flows, levels, demands and emitter flows
    keep their autograd graph."""

    time: Tensor
    head: Tensor
    pressure: Tensor
    flow: Tensor
    status: Tensor
    setting: Tensor
    tank_level: Tensor
    tank_overflow: Tensor
    demand: Tensor
    emitter_flow: Tensor
    energy: PumpEnergy | None = None
    node_names: list[str] = field(default_factory=list)
    link_names: list[str] = field(default_factory=list)


def pattern_factor(net, pattern: str | None, time: float) -> float:
    """The multiplier of `pattern` (a `net.patterns` name; `None` is the constant 1) at
    `time` s since the start: EPANET's `F[p mod length]` with the pattern period
    `p = (t + Pstart) / Pstep` in whole seconds (`hydraul.c`, `demands`; `net.pattern_start`,
    `net.pattern_timestep`)."""
    if pattern is None:
        return 1.0
    if pattern not in net.patterns:
        raise KeyError(f"pattern_factor: {pattern!r} is not a pattern of this network "
                       f"({sorted(net.patterns)})")
    values = net.patterns[pattern]
    return float(values[_period(net, int(round(time))) % len(values)])


def _period(net, t: int) -> int:
    """EPANET's elapsed pattern periods at `t` s (`demands`: `(Htime + Pstart) / Pstep`)."""
    return (t + int(round(net.pattern_start))) // int(round(net.pattern_timestep))


def _default_pattern(net) -> str | None:
    """The pattern of a demand that names none: `[OPTIONS] PATTERN`, else a pattern named
    "1", else none (EPANET's `DefPat`; the reader's rule)."""
    if net.demand_pattern is not None:
        return net.demand_pattern
    return "1" if "1" in net.patterns else None


def _round(x: Tensor) -> Tensor:
    """EPANET's `ROUND` (`types.h`): `(long)(x + 0.5)` for `x >= 0`."""
    return torch.floor(x + 0.5)


class _Clock:
    """The run's times in whole seconds, as EPANET keeps them (`input1.c`, `adjustdata`:
    the hydraulic step no longer than the pattern or the report step; the rule step a tenth
    of the hydraulic step unless given, and no longer than it)."""

    def __init__(self, net, duration: float | None) -> None:
        def whole(value, name):
            seconds = int(round(float(value)))
            if seconds <= 0:
                raise ValueError(f"water_run: {name} must be at least one second, got "
                                 f"{value!r} s")
            return seconds

        self.duration = int(round(float(net.duration if duration is None else duration)))
        if self.duration < 0:
            raise ValueError(f"water_run: the duration must not be negative, got {duration!r}")
        self.pattern_step = whole(net.pattern_timestep, "PATTERN TIMESTEP")
        self.report_step = whole(net.report_timestep, "REPORT TIMESTEP")
        self.hydraulic_step = min(whole(net.hydraulic_timestep, "HYDRAULIC TIMESTEP"),
                                  self.pattern_step, self.report_step)
        rule = (self.hydraulic_step // 10 if net.explicit_rule_timestep is None
                else int(round(float(net.explicit_rule_timestep))))
        self.rule_step = min(rule, self.hydraulic_step)
        if net.rules and self.rule_step <= 0:
            raise ValueError(f"water_run: the rule time step is {self.rule_step} s; a run "
                             f"with [RULES] needs a RULE TIMESTEP of at least one second")
        self.pattern_start = int(round(float(net.pattern_start)))
        self.report_start = int(round(float(net.report_start)))
        if self.report_start > self.duration:
            raise ValueError(
                f"water_run: REPORT START ({self.report_start} s) is after the duration "
                f"({self.duration} s), so nothing would be reported")

    def free_step(self, t: int, report_time: int) -> int:
        """EPANET's `timestep` before the tanks and controls: the hydraulic step, shortened
        to the next pattern period and the next hydraulic report time. EPANET does not
        shorten it to the end of the run (`addenergy` takes the step whole)."""
        step = self.hydraulic_step
        n = (t + self.pattern_start) // self.pattern_step + 1
        to_pattern = n * self.pattern_step - t     # EPANET's arithmetic, Pstart as it is
        if 0 < to_pattern < step:
            step = to_pattern
        to_report = report_time - t
        if 0 < to_report < step:
            step = to_report
        return step

    def step(self, t: int, report_time: int) -> int:
        """`free_step`, shortened to the end of the run."""
        return min(self.free_step(t, report_time), self.duration - t)


class _Patterns:
    """The drivers and the carried pump speeds at a time, from the patterns: the caller's
    drivers are the base values (demands, reservoir heads), scaled per pattern period."""

    def __init__(self, model, drivers) -> None:
        net = model.water_network
        self.net = net
        self.model = model
        self.drivers = dict(drivers)
        names = model.node_names
        default = _default_pattern(net)
        self.junction_nodes = [names.index(j.name) for j in net.junctions]
        self.junction_patterns = [j.pattern if j.pattern is not None else default
                                  for j in net.junctions]
        layer = model.kirchhoff["water"]
        self.pda = any(isinstance(w, PressureDrivenDemand) for w in layer.withdrawals)
        self.reservoir_patterns = [r.pattern for r in net.reservoirs]
        self.pumps = [(model.link_names.index(p.name), p.speed_pattern) for p in net.pumps
                      if p.speed_pattern is not None]
        self.closure = model.control_closure
        self.has_time = TIME_KEY in model.driver_template
        self.demand_patterned = any(p is not None for p in self.junction_patterns)
        self.head_patterned = any(p is not None for p in self.reservoir_patterns)
        self.sources = self.drivers.get("water.sources")
        if self.head_patterned:
            if self.closure is not None:
                self.head_key = RESERVOIR_HEAD_KEY
                base = self.drivers.get(RESERVOIR_HEAD_KEY, self.closure.reservoir_heads)
            else:
                self.head_key = "water.phi_boundary"
                base = self.drivers["water.phi_boundary"]
            self.heads = base
        self._cache: dict[int, dict] = {}

    def _factors(self, patterns, p: int) -> Tensor:
        out = []
        for name in patterns:
            if name is None:
                out.append(1.0)
            else:
                values = self.net.patterns[name]
                out.append(float(values[p % len(values)]))
        return torch.tensor(out, dtype=F64)

    def drivers_at(self, t: int) -> dict:
        p = _period(self.net, t)
        if p not in self._cache:
            d = dict(self.drivers)
            if self.demand_patterned:
                factor = self._factors(self.junction_patterns, p)
                if self.pda:
                    d[DEMAND_FACTOR_KEY] = factor
                if self.sources is not None:
                    scale = torch.ones(self.sources.shape[-1], dtype=F64)
                    scale[self.junction_nodes] = factor
                    d["water.sources"] = self.sources * scale
            if self.head_patterned:
                d[self.head_key] = self.heads * self._factors(self.reservoir_patterns, p)
            self._cache = {p: d}       # one period at a time: the run only moves forward
        d = self._cache[p]
        if self.has_time:
            d = {**d, TIME_KEY: torch.tensor(float(t), dtype=F64)}
        return d

    def speeds(self, state: dict, t: int) -> dict:
        """`state` with every patterned pump's carried speed set (`hydraul.c`, `demands`,
        `setlinksetting`): the pattern's value, the status CLOSED at 0 and OPEN for a
        positive value when it was closed."""
        if not self.pumps:
            return state
        p = _period(self.net, t)
        status = state["water.link_status"].clone()
        setting = state["water.link_setting"].clone()
        for j, name in self.pumps:
            values = self.net.patterns[name]
            v = float(values[p % len(values)])
            setting[..., j] = v
            if v > 0:
                status[..., j] = torch.where(status[..., j] <= CLOSED,
                                             torch.full_like(status[..., j], float(OPEN)),
                                             status[..., j])
            else:
                status[..., j] = float(CLOSED)
        return {**state, "water.link_status": status, "water.link_setting": setting}


def _setting_scale(model) -> Tensor:
    """Per link, the layer's unit of its setting per SI unit (a PRV's, PSV's or PBV's head
    in the layer's potential: `head_scale`; 1 otherwise)."""
    scale = model.head_scale
    return torch.tensor(
        [1.0 if isinstance(link, (WaterPipe, Pump)) or link.kind.upper() not in (
            "PBV", "PRV", "PSV") else scale for link in model.water_network.links()],
        dtype=F64)


class _Recorder:
    """What `WaterResults` keeps of a solve, per report time."""

    def __init__(self, model) -> None:
        net = model.water_network
        self.model = model
        self.closure = model.control_closure
        self.scale = model.head_scale
        names = model.node_names
        self.elevation = torch.tensor(
            [j.elevation for j in net.junctions] + [r.head for r in net.reservoirs]
            + [t.elevation for t in net.tanks], dtype=F64)
        order = [*(j.name for j in net.junctions), *(r.name for r in net.reservoirs),
                 *(t.name for t in net.tanks)]
        if order != names:
            self.elevation = self.elevation[[order.index(n) for n in names]]
        self.junction_nodes = torch.tensor([names.index(j.name) for j in net.junctions],
                                           dtype=torch.long)
        column = {n: k for k, n in enumerate(self.junction_nodes.tolist())}
        layer = model.kirchhoff["water"]
        self.withdrawals = [(w, [column[int(i)] for i in w.nodes])
                            for w in layer.withdrawals
                            if isinstance(w, (PressureDrivenDemand, EmitterOutflow))]
        self.setting_scale = _setting_scale(model)
        self.pumps = torch.tensor([isinstance(link, Pump) for link in net.links()])
        self.storage = layer.storage[0] if net.tanks else None
        self.rows: list[dict] = []

    def _setting(self, solved, drivers) -> Tensor:
        closure = self.closure
        if closure is not None:
            setting = closure.effective_setting(drivers, solved["water.link_setting"])
            shut = self.pumps & (solved["water.link_status"] <= CLOSED)
            setting = torch.where(shut, torch.zeros_like(setting), setting)
        else:
            setting = drivers[SETTING_KEY]
        return setting.detach() / self.setting_scale

    def pump_speeds(self, solved, drivers, cols) -> Tensor:
        """The pumps' (`cols`) relative speeds in this solve, as `WaterResults.setting`."""
        return self._setting(solved, drivers)[..., cols]

    def record(self, solved, drivers, cut_off, levels, overflow, status=None) -> None:
        phi = solved["water.phi"]
        head = phi / self.scale
        n_j = self.junction_nodes.numel()
        sources = drivers.get("water.sources")
        demand = (-sources[..., self.junction_nodes] if sources is not None
                  else phi.new_zeros(n_j))
        emitted = phi.new_zeros(n_j)
        for w, cols in self.withdrawals:
            flow = w.flow(phi[..., w.nodes], drivers)
            full = flow.new_zeros(flow.shape[:-1] + (n_j,))
            full[..., cols] = flow
            if isinstance(w, PressureDrivenDemand):
                demand = demand + full
            else:
                emitted = emitted + full
        if levels is None:
            levels = phi.new_zeros(phi.shape[:-1] + (0,))
        self.rows.append(dict(
            head=head, pressure=head - self.elevation, flow=solved["water.q"],
            status=(link_states_from_cut_off(self.model, solved, *cut_off)
                    if status is None else status),
            setting=self._setting(solved, drivers), tank_level=levels,
            tank_overflow=overflow if overflow is not None else torch.zeros_like(levels),
            demand=demand, emitter_flow=emitted))

    def results(self, times: list[int], energy=None) -> WaterResults:
        stacked = {}
        for key in self.rows[0]:
            values = [row[key] for row in self.rows]
            batch = torch.broadcast_shapes(*(v.shape[:-1] for v in values))
            stacked[key] = torch.stack([v.expand(batch + v.shape[-1:]) for v in values],
                                       dim=-2)
        return WaterResults(time=torch.tensor([float(t) for t in times], dtype=F64),
                            energy=energy, node_names=list(self.model.node_names),
                            link_names=list(self.model.link_names), **stacked)


def water_run(model, state=None, drivers=None, *, duration: float | None = None,
              diagnostics: dict | None = None, **solve_kwargs) -> WaterResults:
    """EPANET 2.2's extended-period hydraulic run of `model` (built with
    `tank_scheme="forward_euler"`, the default) from `state` (`initial_state(model)` when
    `None`) under `drivers` (`initial_drivers(model)` when `None`) for `duration` s (the
    network's `[TIMES] DURATION` when `None`); the results at the report times
    (`WaterResults`). The step rule and the order of patterns, controls, solve, rules and
    tank moves are EPANET's (module docstring); `solve_kwargs` go to every `water_steady`.

    The drivers are the base values: `"water.sources"` (the junctions' base demands) is
    scaled by each junction's demand pattern (its own, else `[OPTIONS] PATTERN`, else a
    pattern named "1"), the reservoir heads by their head patterns
    (`"water.reservoir_head"`, or `"water.phi_boundary"` on a network without a control
    closure), and a pressure-driven demand by `"water.demand_factor"`; a pump's speed
    pattern sets its carried speed in the state each step. `"water.time"` is the run's own:
    a value the caller passes is replaced.

    `diagnostics`, when given a dict, receives `"times"` (the time of every solve, s) and
    `"newton_iterations"` (each solve's). Gradients reach every result except the statuses
    through the solves and the tank moves; not through the step lengths or a switch.

    The implicit tank schemes (`tank_scheme="backward_euler"`, `"trapezoidal"`) are refused
    by name: they move a tank inside the solve and cannot end a step where EPANET's rule
    does (step them with `water_step`)."""
    net = model.water_network
    if net.tanks and model.tank_scheme != "forward_euler":
        raise ValueError(
            f"water_run: the model's tanks are stepped inside the solve "
            f"(tank_scheme={model.tank_scheme!r}), which cannot follow EPANET's step rule; "
            f"build it with tank_scheme='forward_euler', or step it with water_step")
    clock = _Clock(net, duration)
    state = initial_state(model) if state is None else dict(state)
    drivers = initial_drivers(model) if drivers is None else dict(drivers)
    patterns = _Patterns(model, drivers)
    recorder = _Recorder(model)
    closure = model.control_closure
    rules = closure is not None and bool(closure.rules)
    storage = recorder.storage
    if diagnostics is not None:
        diagnostics["times"], diagnostics["newton_iterations"] = [], []
    meter = EnergyMeter(model, clock.duration) if net.pumps else None
    t = 0
    hydraulic_report = clock.report_step      # EPANET's `Rtime` in the hydraulic run
    next_report = clock.report_start          # ... and in the output (`runqual`)
    reported: list[int] = []

    def plan(to_end: bool):
        """`(step, advanced)` of the step that starts at `t`: EPANET's `timestep` (`to_end`:
        shortened to the run's end), and the state with the tanks moved by it."""
        step = (clock.step if to_end else clock.free_step)(t, hydraulic_report)
        inflow = solved["water.storage_flow"] if storage is not None else None
        rate = None
        lv = levels
        if storage is not None:
            heads = solved["water.phi"][..., storage.nodes]
            if isinstance(storage, TankVolumeCurve):
                area = storage.area_toward(heads, inflow)
            else:
                area = storage.capacity(heads, heads, d) * closure.head_scale
            rate = inflow / area
            # `tanktimestep`: `t = ROUND(v / q)` to fill or empty, when positive and shorter
            fill = _round(closure.time_to_limit(lv, rate, inflow))
            fill = fill[torch.isfinite(fill) & (fill > 0)]
            if fill.numel():
                step = min(step, int(fill.min()))
        if closure is not None and closure.controls:
            if storage is None:
                lv = rate = solved["water.phi"].new_zeros(0)
            ahead = closure.control_time(lv, rate, d, solved, inflow)
            if ahead < step:                # `controltimestep`, rounded to whole seconds
                step = int(ahead)

        if rules:                           # `ruletimestep`
            links = solved_link_states(model, solved, d)
            dt = min(clock.rule_step, step)
            dt1 = min(clock.rule_step - t % clock.rule_step, step) or dt
            elapsed, current = 0, solved
            while True:
                elapsed += dt1
                if storage is not None:
                    current = {**current, "water.phi": _advance(
                        model, current["water.phi"], inflow, dt1, d)}
                current, changed = apply_rules_with_states(
                    model, current, d, time=t + elapsed, step=float(dt1), solved=links)
                if bool(changed.any()):
                    break
                dt = min(dt, step - elapsed)
                dt1 = dt
                if dt <= 0:
                    break
            return elapsed, current
        if storage is not None:
            return step, {**solved, "water.phi": _advance(model, solved["water.phi"], inflow,
                                                          step, d)}
        return step, solved

    def speeds_of(solved, d):
        return recorder.pump_speeds(solved, d, meter.cols) if meter.any_curve else None

    while next_report <= clock.duration:
        d = patterns.drivers_at(t)
        state = patterns.speeds(state, t)
        if diagnostics is not None:
            solve = {}
            solved, cut_off = water_steady_and_cut_off(model, state, d, diagnostics=solve,
                                                       **solve_kwargs)
            diagnostics["times"].append(float(t))
            diagnostics["newton_iterations"].append(
                solve.get("layers", {}).get("water", {}).get("newton_iterations"))
        else:
            solved, cut_off = water_steady_and_cut_off(model, state, d, **solve_kwargs)
        # the drivers the solve took, the level controls' tank inflow margin among them
        # (`with_tank_inflow`: the previous solve's), for everything read of this solve
        d = cut_off[0]
        levels = closure.levels(solved) if storage is not None else None
        report = t >= next_report
        if report:
            reported.append(next_report)
            next_report += clock.report_step
        # EPANET adds the energy of every step that starts before the duration (`nexthyd`),
        # and one hour at a duration of 0
        counted = meter is not None and (t < clock.duration or clock.duration == 0)
        status = None
        if report:
            status = link_states_from_cut_off(model, solved, *cut_off)
        elif counted:
            status = link_states_from_cut_off(model, solved, *cut_off, valves=False)
        if t >= clock.duration:
            if counted:
                meter.add(_period(net, t), 1.0, solved, status[..., meter.cols],
                          speeds_of(solved, d))
            if report:
                recorder.record(solved, d, cut_off, levels,
                                _final_overflow(model, storage, solved, d), status)
            break
        step, advanced = plan(True)
        if counted:
            # `addenergy` takes the step as EPANET's `timestep` finds it, past the duration
            # as well; the reported results end at the duration
            whole = plan(False)[0] if t + step >= clock.duration else step
            meter.add(_period(net, t), whole / 3600.0, solved, status[..., meter.cols],
                      speeds_of(solved, d))
        if report:
            recorder.record(solved, d, cut_off, levels,
                            _overflow(model, storage, solved, advanced, step), status)
        t += step
        if t >= hydraulic_report:
            hydraulic_report += clock.report_step
        state = advanced
    return recorder.results(reported, None if meter is None else meter.result())


def _advance(model, phi: Tensor, inflow: Tensor, dt: int, drivers) -> Tensor:
    """`phi` with the tanks moved over `dt` s at `inflow` (`hydraul.c`, `tanklevels`):
    `V += q dt` (`KirchhoffLayer.advance_storage`), a tank within one second's inflow of a
    limit put on it (`settle_tanks`)."""
    layer = model.kirchhoff["water"]
    moved = layer.advance_storage(phi, inflow, float(dt), drivers)
    return settle_tanks(model, layer.storage[0], moved, inflow, drivers)


def _overflow(model, storage, solved, advanced, step: int) -> Tensor | None:
    """The flow (m3/s) each tank that may overflow spilt over a step of `step` s from
    `solved` to `advanced`: for a tank that ends the step at its maximum (`tanklevels` put
    it there), its inflow less its volume change, at least 0; 0 for every other tank (whose
    volume took the whole inflow)."""
    if storage is None or not bool(model.control_closure.overflow.any()):
        return None
    closure = model.control_closure
    nodes = storage.nodes
    end = advanced["water.phi"][..., nodes]
    gained = storage.change(end, solved["water.phi"][..., nodes])
    spill = torch.clamp(solved["water.storage_flow"] - gained / step, min=0.0)
    top = (closure.bottom + closure.y_max) * closure.head_scale
    full = storage.amount(end.detach()) >= storage.amount(top)
    return torch.where(closure.overflow & full, spill, torch.zeros_like(spill))


def _final_overflow(model, storage, solved, drivers) -> Tensor | None:
    """At the run's last report time no step follows: a tank that may overflow and is full
    (within one second's inflow of its maximum volume, `tanklevels`) spills its whole
    inflow."""
    if storage is None or not bool(model.control_closure.overflow.any()):
        return None
    closure = model.control_closure
    heads = solved["water.phi"][..., storage.nodes]
    q = solved["water.storage_flow"]
    top = (closure.bottom + closure.y_max) * closure.head_scale
    full = storage.amount(heads.detach(), drivers) + q.detach() >= storage.amount(top, drivers)
    spill = torch.clamp(q, min=0.0)
    return torch.where(closure.overflow & full, spill, torch.zeros_like(spill))


__all__ = ["WaterResults", "pattern_factor", "water_run"]
