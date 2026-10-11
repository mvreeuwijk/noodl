"""Run a building model over time: `simulate`, the run loop for both routes' models.

Per step from t0 to t1: every input is evaluated at t1, the END of the step (as ContamX does,
measured), and `drivers["time"]` is set to t1; then `model.step`.
The first row is the initial state with its quasi-steady airflow at the start time (as
ContamX's steady initialisation). Two CONTAM terms are handled here, because they couple one
step to the next:

* `air_mass_change=True` (CONTAM's `dodMdt = 1`): each zone's air-mass change over the step,
  -(M(t1) - M(t0)) / dt, is added to the airflow balance's sources (`"air.sources"`), so the net
  inflow into a zone whose temperature changes is dM/dt (measured on ContamX). M is rho V from the
  model's density closures, at the zone pressures of the step. The term is added to the
  drivers before the step, so a closure that later writes `"air.sources"` itself (rather
  than adding to it) replaces it, as does one that changes `"T"` inside the step.
* a model with a `ZonePressureDensity` closure (`densZP = 1`): every step (and the initial
  airflow) is repeated with the latest pressures until the zone pressures move by no more than
  `zone_pressure_rtol * P_ref` -- the fixed point ContamX iterates to (`prj.steady` does the
  same for a steady solve).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from noodl._broadcast import broadcast_shapes
from noodl.apps.building_physics.inputs import (
    Input,
    apply_inputs,
    bind_inputs,
    input_breakpoints,
)
from noodl.apps.building_physics.prj import PrescribedTemperatureDensity, ZonePressureDensity
from noodl.apps.building_physics.schedules import Calendar
from noodl.model import Drivers, Model, State

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


def _zone_air_mass(model: Model, state: State, drivers: Drivers) -> Tensor:
    """The air mass rho V (kg) at every node, from the model's density closures; zero at a
    node of zero volume (the ambient)."""
    drv = dict(drivers)
    for c in model.closures:
        if isinstance(c, (PrescribedTemperatureDensity, ZonePressureDensity)):
            drv.update(c(state, drv))
    return drv["rho"] * _closure(model, PrescribedTemperatureDensity).volumes


def _airflow(model: Model, state: State, drivers: Drivers, **kw) -> State:
    """Every closure, then the potential layers alone at `state`: the quasi-steady flows,
    no transport layer advanced."""
    drv = model._apply_closures(state, drivers)
    new = dict(state)
    for name, layer in model.potential.items():
        phi, q = layer.solve(drv[f"{name}.phi_boundary"], drv, drv.get(f"{name}.sources"), **kw)
        new[f"{name}.phi"], new[f"{name}.q"] = phi, q
    return new


def _fixed_point(run, state: State, zp, rtol: float, max_iter: int, P: float, where: str) -> State:
    out = run(state)
    if zp is None:
        return out
    for _ in range(max_iter):
        nxt = run({**state, zp.key: out[zp.key]})
        change = (nxt[zp.key] - out[zp.key]).abs().max().item()
        out = nxt
        if change <= rtol * P:
            return out
    raise RuntimeError(
        f"simulate: {where}: the zone-pressure density did not converge in {max_iter} passes "
        f"(last change {change:.3e} Pa > {rtol * P:.3e} Pa)"
    )


def simulate(model: Model, state: State, drivers: Drivers, times, *,
             inputs: Sequence[Input] = (), calendar: Calendar | None = None,
             air_mass_change: bool = False, land_on_breakpoints: bool = True,
             zone_pressure_rtol: float = 1e-12, zone_pressure_max_iter: int = 10,
             **step_kwargs) -> dict[str, Tensor]:
    """Time histories at `times` (s since the start, increasing, `times[0] == 0`; the
    calendar places that start in the year).

    `inputs` are evaluated at the end of every step (`calendar` places t = 0 in the year;
    default 1 January 00:00, day type 1). `land_on_breakpoints` splits a step at every jump or
    kink of an input inside it (ContamX does not: `simulate_project` passes False). See the
    module docstring for `air_mass_change` and the zone-pressure fixed point. `step_kwargs`
    reach `Model.step` and the airflow solves (`atol`, `rtol`, ...).

    Returns `"time"` `(N,)`, and per row the state's `"<layer>.q"`, `"<layer>.phi"` of every
    potential layer and `"<layer>.x"` of every transport layer, stacked on a leading axis.
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
    bound = bind_inputs(model, inputs)
    zp = _closure(model, ZonePressureDensity)
    if air_mass_change and _closure(model, PrescribedTemperatureDensity) is None:
        raise ValueError(
            "simulate: air_mass_change needs zone air masses from prescribed temperatures "
            "(a PrescribedTemperatureDensity closure, as project_to_model builds); this model "
            "has none"
        )

    def drivers_at(t: float) -> Drivers:
        d = apply_inputs(drivers, bound, t, cal)
        d["time"] = torch.tensor(t, dtype=F64)
        return d

    def P_of(d: Drivers) -> float:
        return float(torch.as_tensor(d.get("P_ref", zp.Pb if zp else 0.0)).max())

    d = drivers_at(ts[0])
    state = _fixed_point(lambda s: _airflow(model, s, d, **step_kwargs), state, zp,
                         zone_pressure_rtol, zone_pressure_max_iter, P_of(d), "initial airflow")
    state = {**state, **model.initial_capacities(state, d)}
    mass = _zone_air_mass(model, state, d) if air_mass_change else None

    keys = [f"{n}.{s}" for n in model.potential for s in ("q", "phi")]
    keys += [f"{n}.x" for n in model.transport]
    rows: dict[str, list[Tensor]] = {k: [state[k]] for k in keys}
    t = ts[0]
    for t_out in ts[1:]:
        cuts = input_breakpoints(bound, t, t_out, cal) if land_on_breakpoints else []
        # A breakpoint within round-off of the step's start or end is that instant: no
        # sliver sub-steps (a breakpoint is midnight + knot, which need not be exact).
        cuts = [c for c in cuts
                if c - t > cal.round_off(c) and t_out - c > cal.round_off(c)]
        for t1 in [*cuts, t_out]:
            d, dt, t0 = drivers_at(t1), t1 - t, t

            def run(s, d=d, dt=dt, t0=t0, start=state, m0=mass):
                # Every pass advances the SAME step from its start state; only the zone
                # pressures the density closure reads are updated between passes.
                step_from = dict(start)
                if zp is not None and zp.key in s:
                    step_from[zp.key] = s[zp.key]
                drv = d
                if m0 is not None:
                    m1 = _zone_air_mass(model, step_from, d)
                    dM = -(m1 - m0) / dt
                    own = d.get("air.sources")
                    drv = {**d, "air.sources": dM if own is None else own + dM}
                return model.step(step_from, drv, dt, t=t0, **step_kwargs)

            state = _fixed_point(run, state, zp, zone_pressure_rtol, zone_pressure_max_iter,
                                 P_of(d), f"step ending at t = {t1} s")
            if mass is not None:
                mass = _zone_air_mass(model, state, d)
            t = t1
        for k in keys:
            rows[k].append(state[k])
    out = {k: _stack(v) for k, v in rows.items()}
    out["time"] = torch.tensor(ts, dtype=F64)
    return out
