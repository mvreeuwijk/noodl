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

import contextlib
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
    apply_rules_with_states,
    initial_drivers,
    initial_state,
    link_setting_scale,
    link_states_from_cut_off,
    settle_tanks,
    solved_link_states,
    water_steady_and_cut_off,
)
from noodl.apps.water.quality import QualityRouter, quality_step
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
    keep their autograd graph.

    With EPANET's water quality on (`water_run(quality=True)`), `node_quality` (..., T,
    n_nodes) and `link_quality` (..., T, n_links) are the quality at the report times in
    the file's quality unit `quality_units` (mg/L, ug/L, hours, percent): a node's as
    EPANET reports it, a link's the volume-weighted mean of its segments (`quality.c`,
    `avgqual`); `None` when quality is off."""

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
    node_quality: Tensor | None = None
    link_quality: Tensor | None = None
    quality_units: str | None = None
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
    "1", provided it exists, else none: a constant 1 (EPANET's `DefPat = findpattern(DefPatID)`,
    0 when there is none, `input1.c`, `adjustdata`; the reader's rule)."""
    name = net.demand_pattern if net.demand_pattern is not None else "1"
    return name if name in net.patterns else None


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
        # the nodes in `build_model`'s order (`WaterNetwork.nodes`): junctions, reservoirs, tanks
        self.junction_nodes = torch.tensor([names.index(j.name) for j in net.junctions],
                                           dtype=torch.long)
        column = {n: k for k, n in enumerate(self.junction_nodes.tolist())}
        layer = model.kirchhoff["water"]
        self.withdrawals = [(w, [column[int(i)] for i in w.nodes])
                            for w in layer.withdrawals
                            if isinstance(w, (PressureDrivenDemand, EmitterOutflow))]
        # the closure carries the build's per-link scale; without one it is derived once here
        self.setting_scale = torch.tensor(
            self.closure.setting_scale if self.closure is not None
            else link_setting_scale(net, model.head_scale), dtype=F64)
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
        """The pumps' (`cols`) relative speeds in this solve as the solve takes them (a pump
        that is shut is not read: the energy skips it)."""
        if self.closure is not None:
            setting = self.closure.effective_setting(drivers, solved["water.link_setting"])
        else:
            setting = drivers[SETTING_KEY]
        return setting.detach()[..., cols]

    def demands(self, solved, drivers) -> tuple[Tensor, Tensor]:
        """`(demand, emitter_flow)` (..., n_junctions) m3/s of a solve: each junction's
        delivered demand (the requested one, or the pressure-driven delivery) and its
        emitter's outflow. EPANET's `NodeDemand` is their sum (`hydsolver.c`)."""
        phi = solved["water.phi"]
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
        return demand, emitted

    def record(self, solved, drivers, cut_off, levels, overflow, status=None, *,
               demands=None, quality=None) -> None:
        """Keep a solve's row: `demands` (`demands`, when the caller has them already)
        and `quality` (`(node_quality, link_quality)` of the quality run) with it."""
        phi = solved["water.phi"]
        head = phi / self.scale
        demand, emitted = self.demands(solved, drivers) if demands is None else demands
        if levels is None:
            levels = phi.new_zeros(phi.shape[:-1] + (0,))
        row = dict(
            head=head, pressure=head - self.elevation, flow=solved["water.q"],
            status=(link_states_from_cut_off(self.model, solved, *cut_off)
                    if status is None else status),
            setting=self._setting(solved, drivers), tank_level=levels,
            tank_overflow=overflow if overflow is not None else torch.zeros_like(levels),
            demand=demand, emitter_flow=emitted)
        if quality is not None:
            row["node_quality"], row["link_quality"] = quality
        self.rows.append(row)

    def results(self, times: list[int], energy=None, quality_units=None) -> WaterResults:
        stacked = {}
        for key in self.rows[0]:
            values = [row[key] for row in self.rows]
            batch = torch.broadcast_shapes(*(v.shape[:-1] for v in values))
            stacked[key] = torch.stack([v.expand(batch + v.shape[-1:]) for v in values],
                                       dim=-2)
        return WaterResults(time=torch.tensor([float(t) for t in times], dtype=F64),
                            energy=energy, quality_units=quality_units,
                            node_names=list(self.model.node_names),
                            link_names=list(self.model.link_names), **stacked)


def water_run(model, state=None, drivers=None, *, duration: float | None = None,
              quality: bool | None = None, diagnostics: dict | None = None,
              **solve_kwargs) -> WaterResults:
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

    `quality=True` runs EPANET's water quality of `net.quality` alongside
    (`noodl.apps.water.quality`, `QualityRouter`): after the solve at `t` the quality as it
    stands is kept (EPANET's `saveoutput`), then it is routed over the step with that
    solve's flows, link states and delivered demands (`runqual`, `nextqual`); it never
    feeds back into the hydraulics. `state["quality.initial"]` (n_nodes, quality units)
    replaces the file's initial qualities, and `drivers["quality.bulk_coefficient"]`,
    `["quality.wall_coefficient"]` (n_pipes) and `["quality.tank_coefficient"]` (n_tanks)
    its reaction coefficients (SI per second, `WaterQuality`), and
    `drivers["quality.source_strength"]` (n_nodes) its sources' strengths. The quality keeps its
    autograd graph when a quality input or the solve's flows require gradients, and runs
    detached otherwise.
    `None` and `False` run no quality.

    `diagnostics`, when given a dict, receives `"times"` (the time of every solve, s) and
    `"newton_iterations"` (each solve's), and with quality on `"quality"`
    (`QualityRouter`). Gradients reach every result except the statuses
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
    quality_inputs = {key: state.pop(key) for key in _QUALITY_STATE if key in state}
    quality_inputs.update({key: drivers.pop(key) for key in _QUALITY_DRIVERS
                           if key in drivers})
    if quality_inputs and not quality:
        raise ValueError(f"water_run: the state or the drivers carry "
                         f"{sorted(quality_inputs)}, but no water quality runs "
                         f"(quality={quality!r}); pass quality=True")
    if quality and (net.quality is None or net.quality.mode == "NONE"):
        raise ValueError("water_run: quality=True, but the network defines no water quality "
                         "([OPTIONS] QUALITY is NONE or absent)")
    if quality and net.tanks:
        # the tanks' initial segments hold the run's initial volumes (`initsegs`)
        quality_inputs["quality.tank_level"] = model.control_closure.levels(state)
    router = quiet = None
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

    def plan(to_end: bool, t: int, hydraulic_report: int, solved, levels, d):
        """`(step, advanced, acted)` of the step that starts at `t` from the solve `solved`
        (its tank `levels` and drivers `d`; `hydraulic_report` EPANET's `Rtime`): EPANET's
        `timestep` (`to_end`: shortened to the run's end), the state with the tanks moved
        by it and the rules' actions taken at its end, and the links those actions were
        taken on (`apply_rules_with_states`; None without rules)."""
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
                current, changed, acted = apply_rules_with_states(
                    model, current, d, time=t + elapsed, step=float(dt1), solved=links)
                if bool(changed.any()):
                    break
                dt = min(dt, step - elapsed)
                dt1 = dt
                if dt <= 0:
                    break
            return elapsed, current, acted
        if storage is not None:
            return step, {**solved, "water.phi": _advance(model, solved["water.phi"], inflow,
                                                          step, d)}, None
        return step, solved, None

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
        quality_row = demands = None
        if quality:
            if router is None:
                router = _router(model, quality_inputs, solved, clock, diagnostics)
                quiet = contextlib.nullcontext if router.attached else torch.no_grad
            with quiet():
                quality_row = router.record(t, link=report)
            demands = recorder.demands(solved, d)
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
                                _final_overflow(model, storage, solved, d), status,
                                demands=demands, quality=quality_row)
            break
        step, advanced, acted = plan(True, t, hydraulic_report, solved, levels, d)
        if counted:
            # `addenergy` takes the step as EPANET's `timestep` finds it, past the duration
            # as well; the reported results end at the duration
            whole, seen, taken = step, advanced, acted
            if t + clock.free_step(t, hydraulic_report) > clock.duration:
                # the duration cut this step: EPANET's is whole, its tanks moved and its rules
                # checked to its own end (a rule acting past the duration ends it there)
                whole, seen, taken = plan(False, t, hydraulic_report, solved, levels,
                                           d)
            pumps_now, opened = _as_addenergy_sees(seen, taken, status[..., meter.cols],
                                                   meter.cols)
            meter.add(_period(net, t), whole / 3600.0,
                      {"water.phi": seen["water.phi"], "water.q": solved["water.q"]},
                      pumps_now, speeds_of(seen, d), opened=opened)
        if report:
            recorder.record(solved, d, cut_off, levels,
                            _overflow(model, storage, solved, advanced, step), status,
                            demands=demands, quality=quality_row)
        if router is not None:
            states = (status if report
                      else link_states_from_cut_off(model, solved, *cut_off))
            with quiet():
                router.advance(t, step, flow=solved["water.q"], closed=states == CLOSED,
                               demand=demands[0] + demands[1])
        t += step
        if t >= hydraulic_report:
            hydraulic_report += clock.report_step
        state = advanced
    return recorder.results(reported, None if meter is None else meter.result(),
                            None if router is None else net.quality.units)


#: The quality inputs `water_run` reads from the state.
_QUALITY_STATE = ("quality.initial",)
#: ... and from the drivers (`QualityRouter`: the reaction coefficients, SI per second).
_QUALITY_DRIVERS = ("quality.bulk_coefficient", "quality.wall_coefficient",
                    "quality.tank_coefficient", "quality.source_strength")


def _router(model, inputs: dict, solved, clock: _Clock, diagnostics):
    """The run's `QualityRouter`, made at the first solve: attached when gradients are on
    and a quality input or the solve's flows require them (the spec's Ruling Q5)."""
    net = model.water_network
    flow = solved["water.q"]
    if flow.dim() > 1 or any(v.dim() > 1 for v in inputs.values()):
        raise ValueError("water_run: a batched run with water quality is not yet modelled")
    attached = torch.is_grad_enabled() and (
        flow.requires_grad or any(v.requires_grad for v in inputs.values()))
    return QualityRouter(model, net.quality, inputs=inputs,
                         quality_step=quality_step(net, clock.hydraulic_step),
                         attached=attached, diagnostics=diagnostics)


def _as_addenergy_sees(advanced, acted: Tensor | None, pump_states: Tensor, cols: Tensor):
    """`(pump_states, opened)` of a step as EPANET's `addenergy` reads them (`hydraul.c`,
    `nexthyd`). It runs AFTER `timestep`, so after `tanklevels` has moved the tanks to the
    step's end and `ruletimestep` has taken the actions of the rules checked there
    (`takeactions`, `setlinkstatus`/`setlinksetting`): `getenergy` reads the solve's flows and
    junction heads, the tank heads at the step's end (`advanced`, `plan`'s state) and each
    pump's `LinkStatus` and `LinkSetting` as those rules left them. A pump an action was taken
    on (`acted`, `take_actions_per_link`: EPANET's `flag`) takes its carried status as the
    action left it; every other pump keeps the solve's (`pump_states`, `link_states`). So a
    pump a rule closes at the end of a step draws nothing over it; one a rule opens there
    counts as on line, with the `LinkFlow` EPANET then has (`opened`; `EnergyMeter.add`: 1 cfs
    for a constant-power pump, the closed link's `dh / CBIG` for any other); an efficiency
    curve takes the speed the rule set. That includes a pump the solve shut at its shut-off
    head (EPANET's `XHEAD`, `<= CLOSED`), which `takeactions` opens on a STATUS OPEN or a
    changed positive SETTING though its carried status was OPEN already: the action is read,
    not the movement of the carried status. Simple controls and patterns are taken at the
    start of the next step (`runhyd`), so they do not reach it."""
    if acted is None:
        return pump_states, None
    taken = acted[..., cols]
    after = advanced["water.link_status"][..., cols].to(pump_states.dtype)
    opened = taken & (after != CLOSED) & (pump_states == CLOSED)
    return torch.where(taken, after, pump_states), opened


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
