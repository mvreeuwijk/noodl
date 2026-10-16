"""Run a building model over time: `simulate`, the run loop for both routes' models.

Per step from t0 to t1: every input is evaluated at t1, the END of the step (as ContamX does,
measured), and `drivers["time"]` is set to t1 and `drivers["time_step"]` to the step's length;
then `model.step`.
The first row is the initial state with its quasi-steady airflow at the start time (as
ContamX's steady initialisation). Two CONTAM terms are handled here, because they couple one
step to the next:

* `air_mass_change` (CONTAM's `dodMdt = 1`): each zone's air-mass change over the step,
  (M(t1) - M(t0)) / dt, is withdrawn from the airflow balance, so the net inflow into a zone
  whose temperature changes is dM/dt (measured on ContamX). M is the air mass at the zone's
  ABSOLUTE pressure, (P_ref + P_zone) V / (R T), whether or not the densities the flows see
  include the zone pressure (measured: a zone depressurised by an exhaust switching on loses
  M dP_zone / (Pb dt) in that step with CONTAM's `densZP = 0`). The end-of-step mass depends
  on the pressures the step solves for, so the term is the airflow layer's
  `ZoneAirMassChange` node source, inside the solve and its Jacobian: a step of any length
  converges. `simulate` supplies the start-of-step mass (`"<layer>.mass_start"`), at the
  temperatures and pressures the closures leave in the drivers. The default (None) switches
  the term on exactly when the model carries a `ZoneAirMassChange` source, which
  `project_to_model` attaches for a `dodMdt` project; False opts out; True on a model
  without the source is refused.
* a model with a `ZonePressureDensity` closure (CONTAM's `densZP = 1`, or the contaminants'
  zone air mass under `dodMdt`): every step (and the initial airflow) is repeated with the
  latest pressures until the zone pressures move by no more than `zone_pressure_rtol * P_ref`
  -- the fixed point ContamX iterates to (`prj.steady` does the same for a steady solve). The
  closure reads the previous pass's pressures, a coupling of relative size |P_zone| / Pb.
  The same repetition runs for a closure that reads the solved pressure (a pressure-driven
  source, on the potential its `pressure_key` names), so it sees the step's own pressures, as
  ContamX evaluates them. Closures reading different potentials are refused by name.

A sampled control network among the closures (`ControlNetwork`) is evaluated as ContamX
evaluates its control nodes: once before the initial airflow (its flow and pressure sensors
read 0 there), then once at the start of every output step, from the state that step starts
from, and held over the step's passes and sub-steps. The pressure differences its sensors
read are recorded after every airflow solve, and its outputs are part of the histories.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import torch

from noodl._broadcast import broadcast_shapes
from noodl.apps.building_physics.controls import PRESSURE_DIFFERENCE, ControlNetwork
from noodl.apps.building_physics.inputs import (
    Input,
    apply_inputs,
    bind_inputs,
    input_breakpoints,
)
from noodl.apps.building_physics.prj import (
    ZoneAirMassChange,
    ZonePressureDensity,
    _pressure_coupling,
)
from noodl.apps.building_physics.schedules import Calendar
from noodl.contracts import StepContext
from noodl.model import Drivers, Model, State
from noodl.solvers import ConvergenceError

Tensor = torch.Tensor
F64 = torch.float64


def _closure(model: Model, cls):
    found = [c for c in model.closures if isinstance(c, cls)]
    return found[0] if found else None


def _stack(rows: list[Tensor]) -> Tensor:
    """`rows` stacked on a new leading axis, each first broadcast to their common shape: the
    initial row is unbatched when only the drivers carry a batch."""
    shape = broadcast_shapes(*(r.shape for r in rows))
    return torch.stack([r.expand(shape) for r in rows])


def _network(model: Model) -> ControlNetwork | None:
    """The model's sampled control network, if it has one; more than one is refused (the
    order in which two would read each other's writes is not defined)."""
    found = [c for c in model.closures if isinstance(c, ControlNetwork) and c.sampled]
    if len(found) > 1:
        raise ValueError(f"simulate: the model has {len(found)} sampled control networks "
                         f"({', '.join(repr(c.name) for c in found)}); combine them into one")
    return found[0] if found else None


def _evaluate_controls(network: ControlNetwork, state: State, drivers: Drivers, dt: float,
                       t: float) -> State:
    """`state` with the network evaluated once at it (its outputs and memory updated)."""
    out = network(state, drivers, StepContext(dt=dt, t=t))
    return {**state, **{k: out[k] for k in network.state_keys}}


def _with_pressure_differences(model: Model, state: State, drivers: Drivers,
                               network: ControlNetwork | None) -> State:
    """`state` with every air-layer edge's driving pressure difference at it (Pa,
    `KirchhoffLayer.dp`) under `"air.dp"`, when a sensor of the network reads it."""
    if network is None or not network.reads_state(PRESSURE_DIFFERENCE):
        return state
    if "air" not in model.kirchhoff:
        raise ValueError(f"simulate: a sensor of control network {network.name!r} reads "
                         f"{PRESSURE_DIFFERENCE!r}, the pressure differences across the "
                         f"airflow layer 'air', which the model does not have (its potential "
                         f"layers are {sorted(model.kirchhoff)})")
    drv = model._apply_closures(state, drivers)
    return {**state, PRESSURE_DIFFERENCE: model.kirchhoff["air"].dp(state["air.phi"], drv)}


def _air_mass_source(model: Model) -> tuple[str, ZoneAirMassChange] | None:
    """The airflow layer carrying a `ZoneAirMassChange` node source, and the source."""
    for name, layer in model.kirchhoff.items():
        for ns in layer.node_sources:
            if isinstance(ns, ZoneAirMassChange):
                return name, ns
    return None


def _airflow(model: Model, state: State, drivers: Drivers, **kw) -> State:
    """Every closure, then the Kirchhoff layers alone at `state`: the quasi-steady flows,
    no transport layer advanced."""
    drv = model._apply_closures(state, drivers)
    new = dict(state)
    for name, layer in model.kirchhoff.items():
        phi, q = layer.solve(drv[f"{name}.phi_boundary"], drv, drv.get(f"{name}.sources"), **kw)
        new[f"{name}.phi"], new[f"{name}.q"] = phi, q
    return new


def _fixed_point(run, state: State, key: str | None, rtol: float, max_iter: int, P: float,
                 where: str, cause: str = "") -> State:
    """`run(state)` repeated with the latest zone pressures (`state[key]`) until they move by
    no more than `rtol * P`; one call when `key` is None."""
    out = run(state)
    if key is None:
        return out
    for _ in range(max_iter):
        nxt = run({**state, key: out[key]})
        change = (nxt[key] - out[key]).abs().max().item()
        out = nxt
        if change <= rtol * P:
            return out
    raise RuntimeError(
        f"simulate: {where}: the zone pressures fed back to {cause} did not converge in "
        f"{max_iter} passes (last change {change:.3e} Pa > {rtol * P:.3e} Pa)"
    )


def _refuse_unread_inputs(model: Model, drivers: Drivers, inputs: Sequence[Input],
                          where: str) -> None:
    """Refuse an input whose key is neither a driver nor an input the model declares: it
    would be written where nothing reads it (a misspelt key)."""
    declared = model.refs.inputs
    for inp in inputs:
        if inp.key not in drivers and inp.key not in declared:
            raise KeyError(
                f"{where}: input {inp._label()!r} writes driver {inp.key!r}, which is neither "
                f"in the drivers nor an input the model declares, so nothing would read it; "
                f"the declared inputs are {sorted(declared)}"
            )


def simulate(model: Model, state: State, drivers: Drivers, times, *,
             inputs: Sequence[Input] = (), calendar: Calendar | None = None,
             air_mass_change: bool | None = None, land_on_breakpoints: bool = True,
             zone_pressure_rtol: float = 1e-12, zone_pressure_max_iter: int = 10,
             diagnostics: dict | None = None, **step_kwargs) -> dict[str, Tensor]:
    """Time histories at `times` (s since the start, increasing, `times[0] == 0`; the
    calendar places that start in the year).

    `inputs` are evaluated at the end of every step (`calendar` places t = 0 in the year;
    default 1 January 00:00, day type 1). A signal with a `step_mean` (an occupant's
    presence) is written as its mean over each step, or each sub-step. `land_on_breakpoints`
    splits a step at every jump or kink of an input inside it (ContamX does not:
    `simulate_project` passes False). The histories keep one row per output time either way.
    See the module docstring for `air_mass_change` and the zone-pressure fixed point.
    `step_kwargs` reach `Model.step` and the airflow solves (`atol`, `rtol`, ...).

    Each step's airflow solve starts from the previous step's pressures. When it does not
    converge from there (a fan whose iterates swing between the ends of its curve, where its
    slope is zero, after a large speed change), the step is retried once from the airflow
    layers' linear initial guess, as the first solve of the run is (the closures still read
    the previous pressures: `Model.step(warm_start=False)`); a retry that fails too raises
    `ConvergenceError`, naming the step. Only non-convergence (`ConvergenceError`) is
    retried; any other error passes through unchanged.

    A sampled control network in `model.closures` (`ControlNetwork`) runs as ContamX runs
    its controls. It is evaluated once before the initial airflow, with the drivers at
    `times[0]` and the first step's length (`0.0` for a single time), so a control on a
    temperature or a path acts in the first row already, while its flow and pressure sensors
    read 0 there. It is evaluated again at the start of every output step, from the state
    the step starts from, with the inputs at the step's end and the output step's length;
    the step's passes (the zone-pressure fixed point, a retry) and its sub-steps (input
    breakpoints) hold it, so a sensor of `"air.phi"` reads the pressures that ended the
    previous step, not the latest pass. After every airflow solve the drivers' pressure
    differences across the air layer's edges (Pa) are recorded in the state as `"air.dp"`
    when a sensor reads them. With `air_mass_change` the zone air masses use the
    temperatures the controls write: the start mass those of the previous step, the end mass
    those of the step. The histories gain `"<network>.outputs"`, `(N, ..., n_blocks)`.

    `diagnostics`, when given, receives the RUN's diagnostics, not `Model.step`'s per-step
    `diagnostics` (layer statistics, pass counts), which `simulate` does not collect:
    `"airflow_retries"`, the end time (s) of every step that was retried, once per step.

    Returns `"time"` `(N,)`, and per row the state's `"<layer>.q"`, `"<layer>.phi"` of every
    Kirchhoff layer, `"<layer>.x"` of every transport layer, `"<layer>.capacity"` of every
    transport layer whose capacity a closure writes (the zones' air mass, which changes with
    their temperature and pressure) and the control network's `"<network>.outputs"`, stacked
    on a leading axis.
    """
    ts = [float(v) for v in torch.as_tensor(times, dtype=F64).reshape(-1).tolist()]
    if not ts or any(b <= a for a, b in zip(ts, ts[1:], strict=False)):
        raise ValueError(f"simulate: times must be non-empty and increasing, got {ts}")
    if ts[0] != 0.0:
        raise ValueError(
            f"simulate: times[0] must be 0 (the start the calendar places in the year), got "
            f"{ts[0]}; shift the run with Calendar(start_day=..., start_time=...)"
        )
    if int(zone_pressure_max_iter) < 1:
        raise ValueError(
            f"simulate: zone_pressure_max_iter must be >= 1, got {zone_pressure_max_iter}"
        )
    cal = calendar or Calendar()
    _refuse_unread_inputs(model, drivers, inputs, "simulate")
    bound = bind_inputs(model, inputs)
    zp = _closure(model, ZonePressureDensity)
    found = _air_mass_source(model) if air_mass_change is not False else None
    if air_mass_change and found is None:
        raise ValueError(
            "simulate: air_mass_change needs a ZoneAirMassChange node source on the airflow "
            "layer (project_to_model attaches one); this model has none"
        )
    mass_layer, mass_source = found if found else (None, None)
    # The zone pressures the density closure or a pressure reader reads, fed back between
    # passes of a step, and the couplings that read them.
    step_key, cause = _pressure_coupling(model, "simulate")

    def start_mass(state: State, d: Drivers) -> Tensor:
        # At the drivers the closures leave (a closure may write "T" or "P_ref"), as the
        # end-of-step mass inside the solve sees them.
        phi = state[f"{mass_layer}.phi"][..., mass_source.nodes]
        return mass_source.mass(phi, model._apply_closures(state, d))

    def drivers_at(t: float, t_start: float | None = None) -> Drivers:
        # A step-mean signal (an occupant's presence) is averaged over (t_start, t]; at an
        # instant (the initial solve, the controls' evaluation) it is its value there.
        d = apply_inputs(drivers, bound, t, cal, t_start=t_start)
        d["time"] = torch.tensor(t, dtype=F64)
        return d

    def P_of(d: Drivers) -> float:
        return float(torch.as_tensor(d.get("P_ref", zp.Pb if zp else 101325.0)).max())

    d = drivers_at(ts[0])
    network = _network(model)
    if network is not None:
        # ContamX evaluates its controls before the initial airflow (flow and pressure
        # sensors then read 0), and a PI node's t = 0 evaluation is a full step's update.
        state = _evaluate_controls(network, state, d, ts[1] - ts[0] if len(ts) > 1 else 0.0,
                                   ts[0])
    state = _fixed_point(lambda s: _airflow(model, s, d, **step_kwargs), state, step_key,
                         zone_pressure_rtol, zone_pressure_max_iter, P_of(d), "initial airflow",
                         cause)
    state = _with_pressure_differences(model, state, d, network)
    state = {**state, **model.initial_capacities(state, d)}
    mass = start_mass(state, d) if found else None

    keys = [f"{n}.{s}" for n in model.kirchhoff for s in ("q", "phi")]
    keys += [f"{n}.x" for n in model.transport]
    # A transport layer's capacity changes with the temperatures and pressure when a closure
    # writes it (a zone's air mass): its value at every row, for what is computed afterwards.
    keys += [f"{n}.capacity" for n in model.transport if f"{n}.capacity" in state]
    if network is not None:
        keys.append(network.outputs_key)
    rows: dict[str, list[Tensor]] = {k: [state[k]] for k in keys}
    retries: list[float] = []
    t = ts[0]
    for t_out in ts[1:]:
        cuts = input_breakpoints(bound, t, t_out, cal) if land_on_breakpoints else []
        # A breakpoint within round-off of the step's start or end is that instant: no
        # sliver sub-steps (a breakpoint is midnight + knot, which need not be exact).
        cuts = [c for c in cuts
                if c - t > cal.round_off(c) and t_out - c > cal.round_off(c)]
        if network is not None:
            # Once per output step, from its start state, with the inputs at its end (as
            # inside `Model.step`); the passes and sub-steps below hold it.
            d_out = drivers_at(t_out)
            d_out["time_step"] = torch.tensor(t_out - t, dtype=F64)
            state = _evaluate_controls(network, state, d_out, t_out - t, t)
        for t1 in [*cuts, t_out]:
            d, dt, t0 = drivers_at(t1, t), t1 - t, t
            d["time_step"] = torch.tensor(dt, dtype=F64)

            if mass is not None:
                d[mass_source.mass_key] = mass

            def run(s, d=d, dt=dt, t0=t0, t1=t1, start=state):
                # Every pass advances the SAME step from its start state; only the zone
                # pressures the closures read are updated between passes.
                step_from = dict(start)
                if step_key is not None and step_key in s:
                    step_from[step_key] = s[step_key]
                try:
                    return model.step(step_from, d, dt, t=t0, **step_kwargs)
                except ConvergenceError as warm:
                    # Each airflow solve starts from its layer's linear initial guess; the
                    # closures still read the previous pressures. Only non-convergence is
                    # retried; every other failure passes through as it was raised.
                    try:
                        out = model.step(step_from, d, dt, t=t0, warm_start=False,
                                         **step_kwargs)
                    except ConvergenceError as again:
                        raise ConvergenceError(
                            f"simulate: step of {dt} s ending at t = {t1} s: the airflow "
                            f"solve failed from the previous step's pressures ({warm}) and "
                            f"again from the linear initial guess ({again})") from again
                    if not retries or retries[-1] != t1:      # once per step, all passes
                        retries.append(t1)
                    return out

            with network.held() if network is not None else nullcontext():
                state = _fixed_point(run, state, step_key, zone_pressure_rtol,
                                     zone_pressure_max_iter, P_of(d),
                                     f"step of {dt} s ending at t = {t1} s", cause)
            state = _with_pressure_differences(model, state, d, network)
            if mass is not None:
                mass = start_mass(state, d)
            t = t1
        for k in keys:
            rows[k].append(state[k])
    out = {k: _stack(v) for k, v in rows.items()}
    out["time"] = torch.tensor(ts, dtype=F64)
    if diagnostics is not None:
        diagnostics["airflow_retries"] = retries
    return out


def _project_file(project, filename: str, what: str, where: str) -> Path:
    """The path of a file a project's run control names (`what`: "weather" or "ambient"), next
    to its `.prj`. A file that cannot be found, or a project read from no file, is refused by
    name."""
    if project.source_path is None:
        raise ValueError(f"{where}: the project names {what} file {filename!r}, but "
                         f"it has no source_path to resolve it next to")
    path = Path(project.source_path).parent / filename
    if not path.is_file():
        raise FileNotFoundError(f"{where}: the project's {what} file {filename!r} is "
                                f"not at {path}")
    return path


def _project_weather(project, where: str):
    """The project's own weather file (`read_wth`), next to its `.prj`; None without one.
    A file that cannot be found, or a project read from no file, is refused by name."""
    from noodl.apps.building_physics.wth import read_wth

    rc = project.run_control
    if not rc.weather_file:
        return None
    return read_wth(_project_file(project, rc.weather_file, "weather", where))


def project_calendar(project, weather=None) -> Calendar:
    """The calendar of a project's transient run: its run control's start day, time and day
    type, and the day types of its weather file when it has one. `weather` (from
    `read_wth`) overrides that file; by default the project's own is read, next to its
    `.prj`, and a weather file that cannot be found there is refused by name.
    `simulate_project` runs on this calendar; pass it where a function places times in the
    year (`occupant_exposure`)."""
    rc = project.run_control
    if rc is None:
        raise ValueError("project_calendar: the project carries no run control")
    if weather is None:
        weather = _project_weather(project, "project_calendar")
    types = weather.day_types if weather is not None and weather.day_types else None
    return Calendar(start_day=rc.start_day, start_time=rc.start_time,
                    start_day_type=rc.day_type, day_types=types)


def simulate_project(project, *, inputs: Sequence[Input] = (), diagnostics: dict | None = None,
                     **step_kwargs) -> dict[str, Tensor]:
    """Run a `.prj` project the way ContamX does: its run control (start, end, time step), its
    schedules, its weather and ambient-concentration files (resolved next to the `.prj`), with
    the time step fixed (no step split at a schedule's jump, as in ContamX) and the zone
    air-mass change when the run control asks for it. `inputs` are extra inputs of your own.
    Returns `simulate`'s histories plus `"path_flow"` (N, n_paths) in path-number order and,
    for a project with a duct network, `"terminal_flow"` (N, n_terminals): the flow out of
    each duct terminal into its zone, in junction-number order, and, for a project with
    control references, `"control"` (N, n_reports): the named report nodes' values in node order
    (ContamX's control outputs; `"controls.outputs"` holds every block's), and, for a project
    with occupants, `"exposure"` (N - 1, ..., n_occupants, K): each occupant's exposure over each
    step, in `project.occupants` order, as ContamX's exposure file (`occupant_exposure` on
    the run's `project_calendar`; zero while out of the building). The species
    histories carry the zones' rows first, then the junctions' (`project.zone_mass_fractions`
    selects the zones). `diagnostics` is `simulate`'s run-level dict, not `Model.step`'s:
    `"airflow_retries"`, the steps whose airflow solve was retried from the linear initial
    guess (none for a steady run).

    A steady airflow run (with no or steady contaminants) returns the one steady state at the
    run control's steady day, time and day type. Refused by name: occupants in a run whose
    contaminant simulation is not transient or that simulates no species (their generation
    and exposure need both; checked first, so the message names them), cyclic contaminant runs,
    integration other than the trapezoid method with weight 1 (short time step, CVODE, or a
    weight other than 1), steady airflow with transient contaminants, transient airflow with
    steady contaminants (ContamX solves the concentrations to steady state at every step;
    this integrates them), control references in a steady airflow run, and a run whose duration is
    not a whole number of time steps.
    `prj.steady` refuses a steady contaminant run with nothing passing through an air
    handler's plenum.
    """
    from noodl.apps.building_physics.ctm import read_ctm
    from noodl.apps.building_physics.prj import project_to_model, steady

    rc = project.run_control
    if rc is None:
        raise ValueError("simulate_project: the project carries no run control")
    if project.occupants and rc.contaminants != "transient":
        raise ValueError(f"simulate_project: occupants (their generation and exposure) need a "
                         f"transient contaminant simulation; this project's is "
                         f"{rc.contaminants!r}")
    if project.occupants and not project.species:
        raise ValueError("simulate_project: occupants (their generation and exposure) need a "
                         "simulated species; this project simulates none")
    if project.control_plan is not None and not rc.transient_airflow:
        raise ValueError("simulate_project: control nodes need a transient airflow "
                         "simulation; the run control's is steady")
    if rc.contaminants == "cyclic":
        raise ValueError("simulate_project: cyclic contaminant simulation is unsupported")
    if rc.integration != "trapezoid" or rc.integration_weight != 1.0:
        raise ValueError(
            f"simulate_project: integration {rc.integration!r} with weight "
            f"{rc.integration_weight} is unsupported (only the trapezoid method with weight "
            f"1, which is implicit Euler)"
        )
    if not rc.transient_airflow and rc.contaminants == "transient":
        raise ValueError("simulate_project: steady airflow with transient contaminants is "
                         "unsupported")
    if rc.transient_airflow and rc.contaminants == "steady":
        raise ValueError(
            "simulate_project: transient airflow with steady contaminants is unsupported "
            "(ContamX solves the concentrations to steady state at every step)"
        )
    n = round(rc.duration / rc.time_step)
    if rc.transient_airflow and (n < 1 or n * rc.time_step != rc.duration):
        raise ValueError(
            f"simulate_project: the run lasts {rc.duration} s, not a whole number (>= 1) of "
            f"{rc.time_step} s time steps"
        )
    model, state, drivers = project_to_model(project)
    if not rc.transient_airflow:
        _refuse_unread_inputs(model, drivers, inputs, "simulate_project")
        bound = bind_inputs(model, [*project.inputs_for(model), *inputs])
        d = apply_inputs(drivers, bound, 0.0, rc.steady_calendar())
        out_state = steady(model, state, d, **step_kwargs)
        out = {f"{n}.{s}": out_state[f"{n}.{s}"].unsqueeze(0)
               for n in model.kirchhoff for s in ("q", "phi")}
        out.update({f"{n}.x": out_state[f"{n}.x"].unsqueeze(0) for n in model.transport})
        out.update({f"{n}.capacity": out_state[f"{n}.capacity"].unsqueeze(0)
                    for n in model.transport if f"{n}.capacity" in out_state})
        out["time"] = torch.zeros(1, dtype=F64)
        if diagnostics is not None:
            diagnostics["airflow_retries"] = []
        out["path_flow"] = project.path_flows(out["air.q"])
        if project.terminals:
            out["terminal_flow"] = project.terminal_flows(out["air.q"])
        return out
    weather = _project_weather(project, "simulate_project")
    ambient = (read_ctm(_project_file(project, rc.ambient_file, "ambient", "simulate_project"))
               if rc.ambient_file else None)
    all_inputs = [*project.inputs_for(model),
                  *(weather.inputs(model, ambient=project.ambient) if weather else ()),
                  *(ambient.inputs(model) if ambient and "species" in model.transport else ()),
                  *inputs]
    calendar = project_calendar(project, weather)
    times = torch.arange(n + 1, dtype=F64) * rc.time_step
    out = simulate(model, state, drivers, times, inputs=all_inputs, calendar=calendar,
                   land_on_breakpoints=False, diagnostics=diagnostics,
                   **step_kwargs)
    out["path_flow"] = project.path_flows(out["air.q"])
    if project.terminals:
        out["terminal_flow"] = project.terminal_flows(out["air.q"])
    if project.control_plan is not None:
        out["control"] = project.control_outputs(out["controls.outputs"])
    if project.occupants and "species" in model.transport:
        from noodl.apps.building_physics.outputs import occupant_exposure

        out["exposure"] = torch.stack([occupant_exposure(out, o, model, calendar=calendar)
                                       for o in project.occupants], dim=-2)
    return out
