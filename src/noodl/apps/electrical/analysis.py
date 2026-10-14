"""Circuit analyses on a model from `build_model`: the DC operating point, the consistent
initial state, a fixed-grid transient, the AC sweep, and per-device voltages and currents.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from noodl.apps.electrical.circuit import LAYER, Circuit, build_model
from noodl.layers.potential import PotentialFlowLayer
from noodl.model import Model

Tensor = torch.Tensor
F64 = torch.float64
State = dict[str, Tensor]
PHI, Q = f"{LAYER}.phi", f"{LAYER}.q"
_RULES = ("trapezoidal", "backward_euler")


def _layer(model: Model) -> PotentialFlowLayer:
    layer = model.potential.get(LAYER)
    if layer is None or not hasattr(model, "circuit_index"):
        raise ValueError(f"expected a circuit model from build_model (a potential layer "
                         f"{LAYER!r} and model.circuit_index)")
    return layer


def _time_domain(layer: PotentialFlowLayer, what: str) -> None:
    if layer.phasor:
        raise ValueError(f"{what}: the model is a phasor (AC) model, which has no time "
                         f"domain; build it with phasor=False")


def operating_point(model: Model, drivers: Mapping[str, Tensor], *, t: float = 0.0,
                    **solve_kwargs) -> State:
    """The DC operating point at time `t` (s): capacitors open, inductors shorted, sources at
    their value at `t` (`model.steady`; `solve_kwargs`, such as Newton's `atol` and `rtol`,
    reach the layer solve). On a phasor model, the AC steady state."""
    _layer(model)
    return model.steady({}, drivers, t=t, **solve_kwargs)


def initial_state(model: Model, drivers: Mapping[str, Tensor], *, t: float = 0.0,
                  **solve_kwargs) -> State:
    """The consistent initial state at time `t` (s), the start of a transient: every
    capacitor at its `v0`, every inductor (and transformer winding) at its `i0`, sources at
    their value at `t`, and the other voltages and currents consistent with those
    (`solve_kwargs`, such as Newton's `atol` and `rtol`, reach the layer solve).

    There a capacitor is a voltage source at its `v0` and an inductor a current source at its
    `i0`, so a capacitor across a voltage source, two inductors in series or an inductor in
    series with a current source make the system singular, which raises a `RuntimeError`
    saying so; start such a transient from `operating_point` instead."""
    layer = _layer(model)
    _time_domain(layer, "initial_state")
    # Model has no initial-point solve; a circuit model is this one layer with no closures,
    # so solving the layer directly, with the keys Model would read, is the same thing.
    name = layer.name
    phi, q = layer.solve(drivers[f"{name}.phi_boundary"], drivers,
                         drivers.get(f"{name}.sources"), mode="initial", t=t, **solve_kwargs)
    return {f"{name}.phi": phi, f"{name}.q": q}


def _times(times) -> list[float]:
    values = [float(t) for t in (times.tolist() if isinstance(times, Tensor) else times)]
    if not values:
        raise ValueError("transient: times is empty; give the start time and the step ends")
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"transient: times must be strictly increasing, got {values}")
    return values


def transient(model: Model, state: Mapping[str, Tensor], drivers: Mapping[str, Tensor],
              times: Sequence[float] | Tensor, *, integration: str = "trapezoidal",
              **solve_kwargs) -> dict:
    """March the circuit over the time grid `times` (s), from `state` at `times[0]` (usually
    `initial_state`), one `Model.step` per interval.

    `integration="trapezoidal"` (the default) uses the trapezoidal rule (second order)
    except on the first step and on every step whose interval contains a source breakpoint
    (a step, ramp or pulse edge), which use backward Euler, damping the ringing the
    trapezoidal rule shows after a discontinuity (the SPICE practice);
    `integration="backward_euler"` uses backward Euler throughout. The grid is fixed: a
    breakpoint inside a step is not inserted as a time point, so the step that contains it
    sees the source at the step end. A circuit without capacitors or inductors is solved at
    each step end as a DC operating point. `drivers` is one mapping for the whole run, so a
    `Signal` source holds its driver value throughout (a time-varying input is a waveform).
    `solve_kwargs`, such as Newton's `atol` and `rtol`, reach every step's layer solve.

    Returns `{"time": (T,), "electrical.phi": (T, ..., n), "electrical.q": (T, ..., b),
    "integration": [rule of each step]}`, the start state as row 0 (rows broadcast to the
    batch shape of the steps).
    """
    layer = _layer(model)
    _time_domain(layer, "transient")
    if integration not in _RULES:
        raise ValueError(f"transient: integration={integration!r}; expected one of {_RULES}")
    grid = _times(times)
    sources = [el for el in (layer.element_for(k)[0] for k in layer.kinds)
               if hasattr(el, "breakpoints")]
    phis, qs, rules = [state[PHI]], [state[Q]], []
    current = dict(state)
    # the sources are evaluated at t0 + dt (as Model.step does), so breakpoints are
    # searched over the evaluation times, never missing one between rounded grid times
    evaluated = grid[0]
    for k in range(len(grid) - 1):
        t0 = grid[k]
        dt = grid[k + 1] - t0
        end = t0 + dt
        jump = any(src.breakpoints(evaluated, end) for src in sources)
        rule = ("backward_euler" if integration == "backward_euler" or k == 0 or jump
                else "trapezoidal")
        current = model.step(current, drivers, dt, t=t0, integration=rule,
                             **solve_kwargs)
        evaluated = end
        phis.append(current[PHI])
        qs.append(current[Q])
        rules.append(rule)
    return {
        "time": torch.tensor(grid, dtype=F64),
        PHI: torch.stack(torch.broadcast_tensors(*phis)),
        Q: torch.stack(torch.broadcast_tensors(*qs)),
        "integration": rules,
    }


def ac_sweep(circuit: Circuit, frequencies, **solve_kwargs) -> dict[str, Tensor]:
    """The sinusoidal steady state of `circuit` at each of `frequencies` (Hz), one batched
    phasor solve: `{"frequency": (F,), "electrical.phi": (F, n), "electrical.q": (F, b)}`,
    complex peak-amplitude, sine-referenced phasors (`x(t) = |X| sin(omega t + arg X)`).
    `solve_kwargs`, such as `method` or Newton's `atol` and `rtol`, reach the layer solve.

    Sources are `Constant` (the AC magnitude, phase 0) or `Sine` (amplitude and phase; the
    sweep frequency governs, not the waveform's); other waveforms and nonlinear devices
    (diodes) are refused. For per-device values build the phasor model directly
    (`build_model(circuit, phasor=True, frequency=...)`, `operating_point`,
    `device_values`).
    """
    f = torch.as_tensor(frequencies, dtype=F64).reshape(-1)
    model, _, drivers = build_model(circuit, phasor=True, frequency=f)
    op = operating_point(model, drivers, **solve_kwargs)
    # a circuit whose laws do not read the frequency (resistive) solves unbatched; its rows
    # are copied out, so each frequency's row is its own memory
    phi, q = op[PHI], op[Q]
    return {"frequency": f, PHI: phi.expand(*f.shape, phi.shape[-1]).contiguous(),
            Q: q.expand(*f.shape, q.shape[-1]).contiguous()}


def device_values(model: Model, state: Mapping[str, Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
    """Per device name, its `(voltage, current)` (V, A) in `state`: `v = v_p - v_n` and the
    current entering pin `p`, each of shape `(...,)` for a one-port device and `(..., 2)`
    (port 1, port 2) for a two-port device. A source delivering power has a negative
    current.

    No power is reported: in the time domain a device absorbs `v i` (W); in a phasor model
    the values are peak-amplitude phasors and the complex power is `(1/2) v conj(i)` (W).
    """
    layer = _layer(model)
    dp = layer.dp(state[PHI], {})
    q = state[Q]
    out = {}
    for name, (_, edges) in model.circuit_index.items():
        v, i = dp[..., edges], q[..., edges]
        if edges.stop - edges.start == 1:
            v, i = v[..., 0], i[..., 0]
        out[name] = (v, i)
    return out


__all__ = ["ac_sweep", "device_values", "initial_state", "operating_point", "transient"]
