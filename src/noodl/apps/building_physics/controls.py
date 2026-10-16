"""Control networks: sensors, signal blocks and actuators.

A `ControlNetwork` is a model closure, first in `Model(closures=[network, ...])`. It reads its
sensors, evaluates its blocks in the order given, and writes every actuator's value into a
driver with `inputs.scatter`. Inputs (schedules) are applied before the closures run, so a
control overrides a schedule on the same driver entry, as in CONTAM.

A sampled network (the default) is evaluated from the state the step starts from, as ContamX
evaluates its control nodes, once per call that carries a step context: once per
`Model.step`, whatever sub-steps the transport layers take inside it (a `coupling="iterate"`
step re-evaluates it from the same step-start outputs and memory on every pass). `simulate`
evaluates it itself, before the initial airflow and at the start of every output step, and
holds it (`held`) over the step's passes and sub-steps. A flow or pressure sensor reads the
last airflow solve, and 0 before the first. A reading of a driver (a prescribed temperature,
an ambient mass fraction) gives the value the previous step used. A block whose input comes
later in the order reads that input's previous value; before the first evaluation it reads
0, or a constant's value (ContamX presets its constant nodes, measured). The network
carries its block outputs, its blocks' memory and whether it has been evaluated in the
state, under `"<name>.outputs"`, `"<name>.memory"` and `"<name>.evaluated"` (1 once
evaluated, per batch member). The memory holds what a block keeps between evaluations: a
sensor's lagged value, the previous step's driver value. Called outside a step (no step
context: a query), it writes its stored outputs again; before its first evaluation
(`"<name>.evaluated"` 0 and the outputs NaN, as `initial_state` gives them) it writes
nothing, so a query on the initial state sees the drivers as the inputs left them, never a
zero temperature. A batch member not evaluated yet keeps its drivers in the same way.

A continuous network (`sampled=False`) is algebraic in the state at every call. A block that
integrates keeps its state in `"<name>.memory"`; called with a step context, the network
advances that memory by one explicit Euler step of `rates`, evaluated at the state and
memory the step starts from.

Units are those of what is read (kg/kg, K, Pa, kg/s). Blocks are dimensionless unless a gain
or an offset says otherwise.

A block's parameters (gains, offsets, limits, weights, bands, a constant's value) are floats
or float64 tensors. A tensor is kept as given, so the outputs carry its gradient; a band or a
switch's threshold moves where a switch flips, and the gradient through it is zero wherever
it does not flip.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor

from noodl._broadcast import broadcast_shapes
from noodl.apps.building_physics.inputs import scatter

F64 = torch.float64
State = Mapping[str, Tensor]
Drivers = Mapping[str, Tensor]
# `output(i, block, u, memory, y)`: what block `i` passes on in place of its output `y`.
OutputHook = Callable[[int, "Block", Sequence[Tensor], Tensor, Tensor], Tensor]
PRESSURE_DIFFERENCE = "air.dp"


def _f64(value) -> Tensor:
    return torch.as_tensor(value, dtype=F64)


def _parameter(value) -> float | Tensor:
    """A block parameter: a float, or a tensor (as float64, on its own device) whose
    gradient the outputs carry."""
    return value.to(F64) if isinstance(value, Tensor) else float(value)


def _all(condition) -> bool:
    """Whether `condition` holds for every entry of a batched parameter."""
    return bool(torch.as_tensor(condition).all())


def _clip(x: Tensor, low, high) -> Tensor:
    """`x` clipped to `[low, high]`, the bounds floats or tensors. A bound replaces `x` only
    where `x` is strictly beyond it, so at the corner (`x` equal to a bound) the output is
    `x` itself: d/dx = 1 and the bound gets no gradient, whichever type it is
    (`torch.clamp` gives d/dx = 0 there for a float bound and splits the gradient in
    halves for a tensor one)."""
    low, high = _f64(low), _f64(high)
    y = torch.where(x < low, low, x)
    return torch.where(y > high, high, y)


def _stack(values: Sequence[Tensor]) -> Tensor:
    """`values` (scalars or batched) on a new last axis, batch dimensions broadcast."""
    values = [_f64(v) for v in values]
    shape = broadcast_shapes(*(v.shape for v in values))
    return torch.stack([v.expand(shape) for v in values], dim=-1)


def _concat(parts: Sequence[Tensor]) -> Tensor:
    """Memory pieces `(..., m_i)` joined on the last axis, batch dimensions broadcast."""
    batch = broadcast_shapes(*(p.shape[:-1] for p in parts))
    return torch.cat([p.expand(batch + p.shape[-1:]) for p in parts], dim=-1)


def _positions(index: int | Sequence[int]) -> tuple[int, ...]:
    return (int(index),) if isinstance(index, int) else tuple(int(i) for i in index)


@dataclass(frozen=True)
class Evaluation:
    """What a block sees: the state the evaluation reads, the drivers, and the step length
    in s (`None`: an algebraic evaluation that advances nothing)."""

    state: State
    drivers: Drivers
    dt: float | None


# ------------------------------------------------------------------------------ readings
class StateReading:
    """`factor * state[key][..., *index]`, or `default` while the state has no `key` (a
    flow or a pressure before the first airflow solve)."""

    previous_step = False

    def __init__(self, key: str, index: int | Sequence[int], *, factor: float = 1.0,
                 default: float = 0.0) -> None:
        self.key, self.index = str(key), _positions(index)
        self.factor, self.default = float(factor), float(default)

    def __call__(self, state: State, drivers: Drivers) -> Tensor:
        value = state.get(self.key)
        if value is None:
            return _f64(self.default)
        return self.factor * value[(..., *self.index)]


class DriverReading:
    """`factor * drivers[key][..., *index]`. In a sampled network, the value the previous
    step used (ContamX's sensors see a scheduled temperature one step late); the first
    evaluation reads the current value."""

    previous_step = True

    def __init__(self, key: str, index: int | Sequence[int], *, factor: float = 1.0) -> None:
        self.key, self.index, self.factor = str(key), _positions(index), float(factor)

    def __call__(self, state: State, drivers: Drivers) -> Tensor:
        if self.key not in drivers:
            raise KeyError(f"controls: a sensor reads driver {self.key!r}, which the model "
                           f"does not have")
        return self.factor * drivers[self.key][(..., *self.index)]


class FunctionReading:
    """`fn(state, drivers)`, for any other reading (the Modelica route's zone temperature).
    `key` names the state key it reads, when it reads one (see
    `ControlNetwork.reads_state`)."""

    def __init__(self, fn: Callable[[State, Drivers], Tensor], *,
                 previous_step: bool = False, key: str | None = None) -> None:
        self.fn, self.previous_step = fn, bool(previous_step)
        self.key = None if key is None else str(key)

    def __call__(self, state: State, drivers: Drivers) -> Tensor:
        return _f64(self.fn(state, drivers))


# -------------------------------------------------------------------------------- blocks
class Block:
    """One block. `inputs` are the positions, in the network's block list, of the blocks
    whose outputs it reads. `step` returns its output and its new memory,
    `(..., memory_size)`. A `sampled_only` block keeps a previous step's value; a
    `continuous_only` block integrates in continuous time (`rate`)."""

    inputs: tuple[int, ...] = ()
    memory_size = 0
    sampled_only = False
    continuous_only = False
    name = ""
    # What a block earlier in the order reads from this one before its first evaluation.
    preset_output = 0.0

    def initial_memory(self) -> list[float]:
        return [math.nan] * self.memory_size

    def step(self, u: Sequence[Tensor], memory: Tensor,
             ev: Evaluation) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def after(self, memory: Tensor, state: State, drivers: Drivers) -> Tensor:
        """The memory once the network's actuators have written `drivers`."""
        return memory

    def rate(self, u: Sequence[Tensor], memory: Tensor, ev: Evaluation) -> Tensor:
        """d(memory)/dt in a continuous network; zero for a block that does not integrate."""
        return torch.zeros_like(memory)

    def switching(self, u: Sequence[Tensor], memory: Tensor,
                  ev: Evaluation) -> Tensor | None:
        """Values that change sign where the block's output has a corner (a limit)."""
        return None


class Constant(Block):
    """A constant value; a block earlier in the order reads it before its first evaluation
    too (ContamX presets its constant nodes, measured)."""

    def __init__(self, value: float | Tensor, *, name: str = "") -> None:
        self.value, self.name = _parameter(value), name
        self.preset_output = self.value

    def step(self, u, memory, ev):
        return _f64(self.value), memory


class DriverValue(Block):
    """`drivers[key][..., index]` at this evaluation: a value an `Input` writes (a schedule
    into `"controls.input"`), or a driver such as `"time"`."""

    def __init__(self, key: str, index: int | None = None, *, name: str = "") -> None:
        self.key, self.index, self.name = str(key), index, name

    def step(self, u, memory, ev):
        if self.key not in ev.drivers:
            raise KeyError(f"controls: {self.name} reads driver {self.key!r}, which the "
                           f"model does not have")
        value = _f64(ev.drivers[self.key])
        return (value if self.index is None else value[..., self.index]), memory


class Identity(Block):
    """Passes its input on (a signal split, a reported value)."""

    def __init__(self, a: int, *, name: str = "") -> None:
        self.inputs, self.name = (int(a),), name

    def step(self, u, memory, ev):
        return u[0], memory


class Sensor(Block):
    """A reading converted to `x = (value - offset) * scale` and, with a `time_constant`
    (s), lagged: `y = x + (y_prev - x) exp(-dt / time_constant)`, `y = x` at the first
    evaluation. A reading with `previous_step` gives, in a sampled network, the value the
    previous evaluation's step used (`after` stores it once the actuators have written)."""

    def __init__(self, reading, *, offset: float | Tensor = 0.0, scale: float | Tensor = 1.0,
                 time_constant: float = 0.0, name: str = "") -> None:
        if time_constant < 0.0:
            raise ValueError(f"Sensor {name!r}: time_constant must be >= 0 s, got "
                             f"{time_constant}")
        self.reading, self.name = reading, name
        self.offset, self.scale = _parameter(offset), _parameter(scale)
        self.time_constant = float(time_constant)
        self._lag = self.time_constant > 0.0
        self.sampled_only = self._lag
        self._configure()

    def _configure(self) -> None:
        self._previous = bool(getattr(self.reading, "previous_step", False))
        self.memory_size = int(self._previous) + int(self._lag)

    def resolve(self, net, layers: Mapping, drivers: Sequence[str] = ()) -> None:
        """Resolve a `NamedReading` against the model's network and layers."""
        if isinstance(self.reading, NamedReading):
            self.reading.resolve(net, layers, drivers)
            self._configure()

    def step(self, u, memory, ev):
        if isinstance(self.reading, NamedReading) and self.reading.resolved is None:
            raise ValueError(f"controls: {self.name} reads {self.reading!r}, which is not "
                             f"resolved against a model; pass the network to "
                             f"build_model(controls=...)")
        raw = _f64(self.reading(ev.state, ev.drivers))
        mem = list(memory.unbind(-1))
        if self._previous and ev.dt is not None:
            raw = torch.where(torch.isnan(mem[0]), raw, mem[0])
        x = (raw - self.offset) * self.scale
        if not self._lag:
            return x, memory
        k = int(self._previous)
        decay = math.exp(-ev.dt / self.time_constant)
        y = torch.where(torch.isnan(mem[k]), x, x + (mem[k] - x) * decay)
        mem[k] = y
        return y, _stack(mem)

    def after(self, memory, state, drivers):
        if not self._previous:
            return memory
        mem = list(memory.unbind(-1))
        mem[0] = _f64(self.reading(state, drivers))
        return _stack(mem)


class _Many(Block):
    def __init__(self, *inputs: int, name: str = "") -> None:
        if not inputs:
            raise ValueError(f"{type(self).__name__} {name!r}: at least one input")
        self.inputs, self.name = tuple(int(i) for i in inputs), name


class Sum(_Many):
    """`sum_i w_i u_i` (`weights` default 1): CONTAM's add, subtract and sum nodes, and
    Modelica's Add, Add3, Sum, MultiSum and Feedback."""

    def __init__(self, *inputs: int, weights: Sequence[float | Tensor] | None = None,
                 name: str = "") -> None:
        super().__init__(*inputs, name=name)
        self.weights = ((1.0,) * len(self.inputs) if weights is None
                        else tuple(_parameter(w) for w in weights))
        if len(self.weights) != len(self.inputs):
            raise ValueError(f"Sum {name!r}: {len(self.weights)} weights for "
                             f"{len(self.inputs)} inputs")

    def step(self, u, memory, ev):
        return sum(w * v for w, v in zip(self.weights, u, strict=True)), memory


class Average(_Many):
    """The mean of its inputs."""

    def step(self, u, memory, ev):
        return _stack(u).mean(-1), memory


class Maximum(_Many):
    """The largest of its inputs."""

    def step(self, u, memory, ev):
        return _stack(u).amax(-1), memory


class Minimum(_Many):
    """The smallest of its inputs."""

    def step(self, u, memory, ev):
        return _stack(u).amin(-1), memory


class Product(_Many):
    """The product of its inputs."""

    def step(self, u, memory, ev):
        return _stack(u).prod(-1), memory


class Divide(Block):
    """`numerator / denominator`; a zero denominator raises naming the block (ContamX stops
    on it too)."""

    def __init__(self, numerator: int, denominator: int, *, name: str = "") -> None:
        self.inputs, self.name = (int(numerator), int(denominator)), name

    def step(self, u, memory, ev):
        num, den = _f64(u[0]), _f64(u[1])
        if bool((den == 0).any()):
            raise ValueError(f"controls: {self.name} divides by zero")
        return num / den, memory


class _Unary(Block):
    def __init__(self, a: int, *, name: str = "") -> None:
        self.inputs, self.name = (int(a),), name


class Absolute(_Unary):
    """`|u|`."""

    def step(self, u, memory, ev):
        return _f64(u[0]).abs(), memory


class Positive(_Unary):
    """1 where `u > 0`, else 0."""

    def step(self, u, memory, ev):
        return (_f64(u[0]) > 0).to(F64), memory


class Complement(_Unary):
    """`1 - u` (CONTAM's "inverse" node; not a logical negation)."""

    def step(self, u, memory, ev):
        return 1.0 - _f64(u[0]), memory


class Affine(Block):
    """`(u - offset) * gain`: a unit conversion or a gain."""

    def __init__(self, a: int, *, offset: float | Tensor = 0.0, gain: float | Tensor = 1.0,
                 name: str = "") -> None:
        self.inputs, self.name = (int(a),), name
        self.offset, self.gain = _parameter(offset), _parameter(gain)

    def step(self, u, memory, ev):
        return (_f64(u[0]) - self.offset) * self.gain, memory


class _Binary(Block):
    def __init__(self, a: int, b: int, *, name: str = "") -> None:
        self.inputs, self.name = (int(a), int(b)), name


class And(_Binary):
    """1 where both inputs are > 0."""

    def step(self, u, memory, ev):
        return ((_f64(u[0]) > 0) & (_f64(u[1]) > 0)).to(F64), memory


class Or(_Binary):
    """1 where either input is > 0."""

    def step(self, u, memory, ev):
        return ((_f64(u[0]) > 0) | (_f64(u[1]) > 0)).to(F64), memory


class ExclusiveOr(_Binary):
    """1 where exactly one input is > 0."""

    def step(self, u, memory, ev):
        return ((_f64(u[0]) > 0) ^ (_f64(u[1]) > 0)).to(F64), memory


class _Limit(Block):
    def __init__(self, a: int, limit: int, *, name: str = "") -> None:
        self.inputs, self.name = (int(a), int(limit)), name


class BelowLimit(_Limit):
    """1 where `u < limit` (strict), else 0."""

    def step(self, u, memory, ev):
        return (_f64(u[0]) < _f64(u[1])).to(F64), memory


class AboveLimit(_Limit):
    """1 where `u > limit` (strict), else 0."""

    def step(self, u, memory, ev):
        return (_f64(u[0]) > _f64(u[1])).to(F64), memory


class Shortfall(_Limit):
    """`max(0, limit - u)`: how far the input is below the limit."""

    def step(self, u, memory, ev):
        return (_f64(u[1]) - _f64(u[0])).clamp(min=0.0), memory


class Excess(_Limit):
    """`max(0, u - limit)`: how far the input is above the limit."""

    def step(self, u, memory, ev):
        return (_f64(u[0]) - _f64(u[1])).clamp(min=0.0), memory


class LowerBandSwitch(Block):
    """On (1) once `u > limit + band`, off (0) once `u < limit`, unchanged in between;
    off at the start (CONTAM's lower band switch)."""

    memory_size = 1
    sampled_only = True

    def __init__(self, a: int, limit: int, *, band: float | Tensor, name: str = "") -> None:
        self.inputs, self.band, self.name = (int(a), int(limit)), _parameter(band), name

    def initial_memory(self) -> list[float]:
        return [0.0]

    def step(self, u, memory, ev):
        x, lim = _f64(u[0]), _f64(u[1])
        prev = memory[..., 0]
        y = torch.where(x > lim + self.band, torch.ones_like(prev),
                        torch.where(x < lim, torch.zeros_like(prev), prev))
        return y, y.unsqueeze(-1)


class UpperBandSwitch(Block):
    """On (1) once `u < limit - band`, off (0) once `u > limit`, unchanged in between;
    off at the start (CONTAM's upper band switch)."""

    memory_size = 1
    sampled_only = True

    def __init__(self, a: int, limit: int, *, band: float | Tensor, name: str = "") -> None:
        self.inputs, self.band, self.name = (int(a), int(limit)), _parameter(band), name

    def initial_memory(self) -> list[float]:
        return [0.0]

    def step(self, u, memory, ev):
        x, lim = _f64(u[0]), _f64(u[1])
        prev = memory[..., 0]
        y = torch.where(x < lim - self.band, torch.ones_like(prev),
                        torch.where(x > lim, torch.zeros_like(prev), prev))
        return y, y.unsqueeze(-1)


class Hysteresis(Block):
    """ContamX's hysteresis: with `x` the input clipped to 0..1, the output follows the
    rising line `max(0, (x - slack) slope)` and the falling line `min(1, x slope)` and stays
    between them: `y = min(falling, max(rising, y_prev))`. The previous output starts at
    `initial * slope`. ContamX's files give the slope; the default `1 / (1 - slack)`, the
    slope that makes the falling line reach 1 where the rising line does, is a convenience
    for networks written in Python."""

    memory_size = 1
    sampled_only = True

    def __init__(self, a: int, *, slack: float | Tensor, slope: float | Tensor | None = None,
                 initial: float | Tensor = 0.0, name: str = "") -> None:
        slack = _parameter(slack)
        if not (_all((slack >= 0.0) & (slack < 1.0)) if isinstance(slack, Tensor)
                else 0.0 <= slack < 1.0):
            raise ValueError(f"Hysteresis {name!r}: slack must be in [0, 1), got {slack}")
        self.inputs, self.name = (int(a),), name
        self.slack = slack
        self.slope = 1.0 / (1.0 - self.slack) if slope is None else _parameter(slope)
        self.initial = _parameter(initial)

    def initial_memory(self) -> list[float]:
        return [self.initial * self.slope]

    def step(self, u, memory, ev):
        x = _f64(u[0]).clamp(0.0, 1.0)
        rising = ((x - self.slack) * self.slope).clamp(min=0.0)
        falling = (x * self.slope).clamp(max=1.0)
        y = torch.minimum(falling, torch.maximum(rising, memory[..., 0]))
        return y, y.unsqueeze(-1)


class ProportionalControl(Block):
    """`clip(gain u, 0, 1)` (CONTAM's proportional controller: the input is an error)."""

    def __init__(self, a: int, *, gain: float | Tensor, name: str = "") -> None:
        self.inputs, self.gain, self.name = (int(a),), _parameter(gain), name

    def step(self, u, memory, ev):
        return (self.gain * _f64(u[0])).clamp(0.0, 1.0), memory


class IncrementalPI(Block):
    """ContamX's PI controller on an error `e`, evaluated once per step of length `dt` (s). It
    is incremental: the output changes by a proportional part on the change of the error and
    an integral part on the trapezoid of the error over the step,
    `y = clip(y_prev + kp (e - e_prev) + ki dt (e + e_prev) / 2, 0, 1)`, `ki` in 1/s. The
    stored previous output is the clipped one (which limits windup); the first evaluation
    updates from `initial_output` and `initial_error`."""

    memory_size = 2
    sampled_only = True

    def __init__(self, a: int, *, kp: float | Tensor, ki: float | Tensor,
                 initial_output: float = 0.0, initial_error: float = 0.0,
                 name: str = "") -> None:
        self.inputs, self.name = (int(a),), name
        self.kp, self.ki = _parameter(kp), _parameter(ki)
        self.initial_output, self.initial_error = float(initial_output), float(initial_error)

    def initial_memory(self) -> list[float]:
        return [self.initial_output, self.initial_error]

    def step(self, u, memory, ev):
        e = _f64(u[0])
        y_prev, e_prev = memory[..., 0], memory[..., 1]
        y = (y_prev + self.kp * (e - e_prev)
             + self.ki * ev.dt * 0.5 * (e + e_prev)).clamp(0.0, 1.0)
        return y, _stack([y, e])


class LimitedPID(Block):
    """A P or PI controller with a limited output and anti-windup, in continuous time, as
    MSL 4.1.0's `LimPID` without its derivative part (`Modelica/Blocks/Continuous.mo`,
    `LimPID`): `v = gain (setpoint_weight u_s - u_m + xi)`, `y = clamp(v, y_min, y_max)`
    and, with an `integral_time` Ti (s), `d xi/dt = (u_s - u_m + (y - v)/(gain windup_ratio))
    / Ti`, `xi(0) = initial_integral`; without one, `xi = 0` (a P controller). `y_min`
    defaults to `-y_max`. Its switching values, `(v - y_max)/w` and `(v - y_min)/w` with
    `w = y_max - y_min`, cross zero at the limiter's corners."""

    continuous_only = True

    def __init__(self, setpoint: int, measurement: int, *, gain: float | Tensor,
                 y_max: float | Tensor, y_min: float | Tensor | None = None,
                 integral_time: float | None = None,
                 setpoint_weight: float | Tensor = 1.0, windup_ratio: float = 0.9,
                 initial_integral: float = 0.0, name: str = "") -> None:
        y_max = _parameter(y_max)
        y_min = -y_max if y_min is None else _parameter(y_min)
        gain, setpoint_weight = _parameter(gain), _parameter(setpoint_weight)
        if not _all(y_max >= y_min):
            raise ValueError(f"LimitedPID {name!r}: y_max {y_max} is below y_min {y_min}")
        if not _all(gain != 0.0):
            raise ValueError(f"LimitedPID {name!r}: the gain must be nonzero")
        if integral_time is not None and not integral_time > 0.0:
            raise ValueError(f"LimitedPID {name!r}: integral_time must be > 0 s, got "
                             f"{integral_time}")
        if not windup_ratio > 0.0:
            raise ValueError(f"LimitedPID {name!r}: windup_ratio must be > 0, got "
                             f"{windup_ratio}")
        if not _all(setpoint_weight >= 0.0):
            raise ValueError(f"LimitedPID {name!r}: setpoint_weight must be >= 0, got "
                             f"{setpoint_weight}")
        self.inputs, self.name = (int(setpoint), int(measurement)), name
        self.gain, self.y_max, self.y_min = gain, y_max, y_min
        self.integral_time = None if integral_time is None else float(integral_time)
        self.setpoint_weight, self.windup_ratio = setpoint_weight, float(windup_ratio)
        self.initial_integral = float(initial_integral)
        self.memory_size = 0 if self.integral_time is None else 1

    def initial_memory(self) -> list[float]:
        return [self.initial_integral] if self.memory_size else []

    def unlimited(self, u, memory) -> Tensor:
        """`v`, the output before the limiter, from the inputs `u` and the memory."""
        xi = memory[..., 0] if self.memory_size else 0.0
        return self.gain * (self.setpoint_weight * _f64(u[0]) - _f64(u[1]) + xi)

    def step(self, u, memory, ev):
        return _clip(self.unlimited(u, memory), self.y_min, self.y_max), memory

    def rate(self, u, memory, ev):
        if not self.memory_size:
            return memory
        v = self.unlimited(u, memory)
        y = _clip(v, self.y_min, self.y_max)
        track = (y - v) / (self.gain * self.windup_ratio)
        return ((_f64(u[0]) - _f64(u[1]) + track) / self.integral_time).unsqueeze(-1)

    def switching(self, u, memory, ev):
        v = self.unlimited(u, memory)
        width = self.y_max - self.y_min
        if isinstance(width, Tensor):
            width = torch.where(width > 0, width, torch.ones_like(width))
        elif not width > 0:
            width = 1.0
        return _stack([(v - self.y_max) / width, (v - self.y_min) / width])


# ----------------------------------------------------------------------------- actuators
@dataclass(frozen=True)
class Actuator:
    """Writes block `block`'s output into driver `key` at positions `index` on its last
    axis (the whole driver when `None`); `mode="add"` adds it to those entries instead (a
    heat flow into `"thermal.sources"`, W)."""

    block: int
    key: str
    index: tuple[int, ...] | None = None
    mode: str = "set"
    name: str = ""


def _add_at(base: Tensor, index: Sequence[int], value: Tensor) -> Tensor:
    """`base` with `value` added at `index` on the last axis, batch dimensions broadcast; a
    position listed twice receives both additions."""
    idx = torch.as_tensor(list(index), dtype=torch.long)
    batch = broadcast_shapes(base.shape[:-1], value.shape[:-1])
    out = base.to(F64).expand(batch + base.shape[-1:])
    return out.index_add(-1, idx, value.expand(batch + (idx.numel(),)))


def _kept(written: Tensor, base: Tensor | None, evaluated: Tensor, a: Actuator) -> Tensor:
    """`written` where `evaluated` is 1, `base` (the driver before the write) for a batch
    member not evaluated yet."""
    done = evaluated > 0
    if bool(done.all()):
        return written
    if base is None:
        raise ValueError(f"controls: actuator {a.name or a.key!r} writes driver {a.key!r}, "
                         f"which the model does not have, for a batch in which some members "
                         f"are not evaluated yet (they have no value to keep)")
    done = done.reshape(done.shape + (1,)) if done.ndim else done
    shape = broadcast_shapes(written.shape, base.shape, done.shape)
    return torch.where(done.expand(shape), written.expand(shape), base.to(F64).expand(shape))


# ------------------------------------------------------------------------------- network
class ControlNetwork:
    """Blocks evaluated in order and the actuators that write their outputs into drivers:
    a model closure that keeps the block outputs in the state. Blocks without a name are
    named `"block <i>"`."""

    integrates = True

    def __init__(self, blocks: Sequence[Block], actuators: Sequence[Actuator] = (), *,
                 sampled: bool = True, name: str = "controls") -> None:
        self.blocks, self.actuators = list(blocks), list(actuators)
        self.sampled, self.name = bool(sampled), str(name)
        if not self.blocks:
            raise ValueError("ControlNetwork: no blocks")
        n = len(self.blocks)
        for i, block in enumerate(self.blocks):
            if not block.name:
                block.name = f"block {i}"
            bad = [j for j in block.inputs if not 0 <= j < n]
            if bad:
                raise ValueError(f"ControlNetwork: {block.name} reads blocks {bad}, which do "
                                 f"not exist (the network has {n})")
            kind = type(block).__name__
            if self.sampled and block.continuous_only:
                raise ValueError(f"ControlNetwork: {block.name} ({kind}) integrates in "
                                 f"continuous time; it needs sampled=False")
            if not self.sampled and block.sampled_only:
                raise ValueError(f"ControlNetwork: {block.name} ({kind}) keeps a previous "
                                 f"step's value; it needs a sampled network")
        for a in self.actuators:
            if not 0 <= a.block < n:
                raise ValueError(f"ControlNetwork: actuator {a.name or a.key!r} writes block "
                                 f"{a.block}, which does not exist (the network has {n})")
            if a.mode not in ("set", "add") or (a.mode == "add" and a.index is None):
                raise ValueError(f"ControlNetwork: actuator {a.name or a.key!r}: mode must be "
                                 f"'set' or 'add', and 'add' needs positions")
            if a.mode == "set" and a.index is not None and len(set(a.index)) < len(a.index):
                raise ValueError(f"ControlNetwork: actuator {a.name or a.key!r} sets positions "
                                 f"{list(a.index)} of driver {a.key!r}, listing one twice; "
                                 f"'set' needs distinct positions ('add' adds each listing)")
        self.outputs_key, self.memory_key = f"{self.name}.outputs", f"{self.name}.memory"
        self.evaluated_key = f"{self.name}.evaluated"
        self.input_key = f"{self.name}.input"
        self.state_keys = (self.outputs_key, self.memory_key, self.evaluated_key)
        self.outputs = tuple(sorted({a.key for a in self.actuators})) + self.state_keys
        self._held = False
        self._resolved_for: tuple | None = None
        self._layout()

    def _readings(self):
        """Every sensor's reading, a resolved `NamedReading` replaced by what it resolved
        to."""
        for b in self.blocks:
            if isinstance(b, Sensor):
                r = b.reading
                yield r.resolved if isinstance(r, NamedReading) else r

    def _layout(self) -> None:
        """Memory slots and the drivers read, from the blocks as they are now."""
        self._slots: list[slice] = []
        start = 0
        for block in self.blocks:
            self._slots.append(slice(start, start + block.memory_size))
            start += block.memory_size
        self.memory_size = start
        read = {b.key for b in self.blocks if isinstance(b, DriverValue)}
        read |= {r.key for r in self._readings() if isinstance(r, DriverReading)}
        self.inputs = tuple(sorted(read))

    def resolve(self, net, layers: Mapping, drivers: Sequence[str] = ()) -> None:
        """Resolve every sensor's `NamedReading` against `net`, the model's `layers` (by
        name) and the names of the `drivers` the model declares; `build_model(controls=)`
        and `project_to_model` call it before they build the model. One network serves one
        model: resolving it again against another network or other layer objects is
        refused, as the first model would silently read the second's rows."""
        model = (net, tuple((name, id(layer)) for name, layer in layers.items()))
        if self._resolved_for is not None and not (
                model[0] is self._resolved_for[0] and model[1] == self._resolved_for[1]):
            raise ValueError(f"ControlNetwork {self.name!r} is already resolved for another "
                             f"model; one network per model (build a new network for this "
                             f"one)")
        self._resolved_for = (net, model[1], dict(layers))
        for b in self.blocks:
            if isinstance(b, Sensor):
                b.resolve(net, layers, drivers)
        self._layout()

    def __repr__(self) -> str:
        mode = "sampled" if self.sampled else "continuous"
        return (f"ControlNetwork({self.name!r}, {len(self.blocks)} blocks, "
                f"{len(self.actuators)} actuators, {mode})")

    @property
    def input_specs(self) -> dict[str, dict]:
        """The values written into the network (a schedule node's schedule)."""
        if self.input_key not in self.inputs:
            return {}
        return {self.input_key: dict(description="values written into the control network "
                                     "each step (a schedule's value)", unit="-")}

    def reads_state(self, key: str) -> bool:
        """Whether a sensor reads state `key`: a `StateReading`, or a `FunctionReading`
        declaring that key, directly or resolved from a `NamedReading` (`simulate` records
        `"air.dp"` when one does)."""
        return any(isinstance(r, (StateReading, FunctionReading)) and r.key == key
                   for r in self._readings())

    def initial_state(self) -> dict[str, Tensor]:
        """Not evaluated yet (`"<name>.evaluated"` 0, outputs NaN: the blocks read 0 or a
        constant's value, and the actuators write nothing until the first evaluation) and
        every block's initial memory."""
        memory = [v for b in self.blocks for v in b.initial_memory()]
        return {self.outputs_key: torch.full((len(self.blocks),), math.nan, dtype=F64),
                self.memory_key: _stack(memory) if memory else torch.zeros(0, dtype=F64),
                self.evaluated_key: torch.tensor(0.0, dtype=F64)}

    def initial_drivers(self) -> dict[str, Tensor]:
        """Zeros for `"<name>.input"` when a block reads it (the inputs write it each step)."""
        if self.input_key not in self.inputs:
            return {}
        n = [b.index for b in self.blocks
             if isinstance(b, DriverValue) and b.key == self.input_key and b.index is not None]
        return {self.input_key: torch.zeros(max(n, default=0) + 1, dtype=F64)}

    def _stored(self, state: State) -> tuple[Tensor, Tensor, Tensor]:
        """The outputs, memory and evaluated flag in `state`, the initial ones where it has
        none."""
        init = self.initial_state()
        return tuple(state[k] if k in state else init[k] for k in self.state_keys)

    def _run(self, state: State, drivers: Drivers, memory: Tensor, dt: float | None,
             want: str = "", output: OutputHook | None = None,
             ) -> tuple[Tensor, Tensor, Tensor | None]:
        """One pass over the blocks: outputs, new memory and, for `want` "rate" or
        "switching", that quantity of every block. `output(i, block, u, memory, y)`, when
        given, returns what block `i` passes on in place of its output `y` (see
        `evaluate`)."""
        ev = Evaluation(state, drivers, dt)
        stored, _, evaluated = self._stored(state)
        preset = _stack([b.preset_output for b in self.blocks])
        seen = torch.where((evaluated > 0).unsqueeze(-1), stored, preset)
        out = list(seen.unbind(-1))
        mem, extra = [], []
        for i, block in enumerate(self.blocks):
            u = [out[j] for j in block.inputs]
            m = memory[..., self._slots[i]]
            y, m_new = block.step(u, m, ev)
            if output is not None:
                y = output(i, block, u, m, _f64(y))
            out[i] = _f64(y)
            mem.append(m_new)
            if want == "rate":
                extra.append(block.rate(u, m, ev))
            elif want == "switching":
                s = block.switching(u, m, ev)
                if s is not None:
                    extra.append(s)
        empty = memory[..., :0]
        rest = _concat([*extra, empty]) if want == "rate" else (
            _concat(extra) if extra else None)
        return _stack(out), _concat([*mem, empty]), rest

    def evaluate(self, state: State, drivers: Drivers, memory: Tensor | None = None, *,
                 output: OutputHook | None = None) -> Tensor:
        """Every block's output at `state`, advancing nothing (continuous networks).
        `output(i, block, u, memory, y)`, when given, is called after block `i` with its
        inputs `u`, its memory and its output `y`, and returns what the block passes on to
        the blocks after it and into the result: a hook that holds a limiter on one side
        of its corner, or offsets a sensor's reading."""
        memory = self._stored(state)[1] if memory is None else memory
        return self._run(state, drivers, memory, None, output=output)[0]

    def rates(self, state: State, drivers: Drivers, memory: Tensor | None = None) -> Tensor:
        """d(memory)/dt at `state` (continuous networks; zero outside integrators)."""
        memory = self._stored(state)[1] if memory is None else memory
        return self._run(state, drivers, memory, None, "rate")[2]

    def switching(self, state: State, drivers: Drivers,
                  memory: Tensor | None = None) -> Tensor | None:
        """Every block's switching values at `state`, or `None` when no block has any."""
        memory = self._stored(state)[1] if memory is None else memory
        return self._run(state, drivers, memory, None, "switching")[2]

    def write(self, drivers: Drivers, outputs: Tensor,
              evaluated: Tensor | None = None) -> dict[str, Tensor]:
        """Every actuator's value written into its driver (a later actuator on the same key
        sees the earlier ones' writes); the given drivers are not modified. A batch member
        whose `evaluated` flag is 0 keeps its drivers as they are."""
        out: dict[str, Tensor] = {}
        for a in self.actuators:
            value = outputs[..., a.block]
            base = out.get(a.key, drivers.get(a.key))
            written = self._write_one(a, base, value)
            out[a.key] = written if evaluated is None else _kept(written, base, evaluated, a)
        return out

    @staticmethod
    def _write_one(a: Actuator, base: Tensor | None, value: Tensor) -> Tensor:
        """`base` with actuator `a`'s `value` written in."""
        if base is None and a.index is not None:
            raise KeyError(f"controls: actuator {a.name or a.key!r} writes positions "
                           f"{list(a.index)} of driver {a.key!r}, which the model does "
                           f"not have")
        if a.index is not None and base is not None and base.ndim == 0:
            raise ValueError(f"controls: actuator {a.name or a.key!r} writes positions "
                             f"{list(a.index)} of driver {a.key!r}, which is a single "
                             f"value (no positions)")
        if a.index is not None and base is not None and max(a.index) >= base.shape[-1]:
            raise ValueError(f"controls: actuator {a.name or a.key!r} writes positions "
                             f"{list(a.index)} of driver {a.key!r}, which has "
                             f"{base.shape[-1]} entries")
        if a.index is None:
            # One value per batch member, the same for every entry of the driver (a
            # driver that does not exist yet gets one entry).
            whole = value if value.ndim == 0 else value.unsqueeze(-1)
            return scatter(base, None, whole)
        v = value.unsqueeze(-1).expand(*value.shape, len(a.index))
        return (scatter(base, a.index, v) if a.mode == "set"
                else _add_at(base, a.index, v))

    @contextmanager
    def held(self) -> Iterator[None]:
        """Within it, a sampled network called with a step context is not evaluated again:
        it writes its stored outputs and records the drivers the step uses. `simulate`
        evaluates the network once per output step and holds it over the step's passes and
        sub-steps. Nesting is safe (leaving an inner hold keeps the outer one), but the hold
        is a flag on the network, not on a run: two runs of one network must not overlap
        (one network per model, one run of it at a time)."""
        previous, self._held = self._held, True
        try:
            yield
        finally:
            self._held = previous

    def __call__(self, state: State, drivers: Drivers, ctx=None) -> dict[str, Tensor]:
        outputs, memory, evaluated = self._stored(state)
        stepping = ctx is not None and getattr(ctx, "dt", None) is not None
        advance = stepping and not (self.sampled and self._held)
        if self.sampled:
            if advance:
                outputs, memory, _ = self._run(state, drivers, memory, float(ctx.dt))
                evaluated = torch.ones(outputs.shape[:-1], dtype=F64)
            elif not bool((evaluated > 0).any()):
                # Not evaluated yet: nothing to write.
                return {self.outputs_key: outputs, self.memory_key: memory,
                        self.evaluated_key: evaluated}
        else:
            if advance:
                memory = memory + float(ctx.dt) * self.rates(state, drivers, memory)
            outputs = self.evaluate(state, drivers, memory)
            evaluated = torch.ones(outputs.shape[:-1], dtype=F64)
        writes = self.write(drivers, outputs, evaluated)
        if self.sampled and stepping:
            after = {**drivers, **writes}
            memory = _concat([*(b.after(memory[..., s], state, after)
                                for b, s in zip(self.blocks, self._slots, strict=True)),
                              memory[..., :0]])
        return {**writes, self.outputs_key: outputs, self.memory_key: memory,
                self.evaluated_key: evaluated}


# ------------------------------------------------------------------------ sensors by name
class NamedReading:
    """A reading named by what it reads (a node, an edge of a kind, a species), not by its
    position in the state. `resolve(net, layers, drivers)` finds the position in the model
    the network is built into; `build_model(controls=)` and `project_to_model` resolve every
    sensor's reading before they build the model, so the rows and columns are those of that
    model's own layers. Evaluating one before it is resolved raises, naming the sensor."""

    def __init__(self, label: str, resolver) -> None:
        self.label, self._resolver = str(label), resolver
        self.resolved = None

    def __repr__(self) -> str:
        return self.label

    @property
    def previous_step(self) -> bool:
        return bool(getattr(self.resolved, "previous_step", False))

    def resolve(self, net, layers: Mapping, drivers: Sequence[str] = ()) -> None:
        self.resolved = self._resolver(net, layers, tuple(drivers))

    def __call__(self, state: State, drivers: Drivers) -> Tensor:
        if self.resolved is None:
            raise ValueError(f"controls: {self.label} is not resolved against a model; "
                             f"pass the network to build_model(controls=...)")
        return self.resolved(state, drivers)


def _layer(layers: Mapping, name: str, label: str):
    if name not in layers:
        raise KeyError(f"controls: {label} needs the model's {name!r} layer; its layers "
                       f"are {list(layers)}")
    return layers[name]


def _row(layer, net, node, label: str) -> int:
    """`node`'s row in transport layer `layer`'s state."""
    hit = (layer.interior_idx == net.node_index(node)).nonzero()
    if hit.numel() != 1:
        raise KeyError(f"controls: {label}: {node!r} is not an interior node of layer "
                       f"{layer.name!r} (a boundary value is a driver)")
    return int(hit[0, 0])


def node_mass_fraction(node, species: int | str = 0) -> NamedReading:
    """A zone's mass fraction (kg/kg) of `species` (position or name) in the species
    layer's state, read through the layer's own layout (`(..., n_interior)` for one
    species, `(..., n_interior, K)` for more)."""
    label = f"node_mass_fraction({node!r}, {species!r})"

    def resolve(net, layers, drivers):
        layer = _layer(layers, "species", label)
        i = _row(layer, net, node, label)
        k = species
        if isinstance(species, str):
            if layer.species_names is None or species not in layer.species_names:
                raise KeyError(f"controls: {label}: the species layer has no species "
                               f"{species!r} (its species are {layer.species_names})")
            k = layer.species_names.index(species)
        k = int(k)
        if not 0 <= k < layer.n_species:
            raise IndexError(f"controls: {label}: the species layer has {layer.n_species} "
                             f"species, not position {k}")
        column = k * layer.n_i + i

        def read(state: State, drivers: Drivers) -> Tensor:
            stacked, _ = layer._to_stacked(state["species.x"], layer.n_i, "species.x")
            return stacked[..., column]

        return FunctionReading(read, key="species.x")

    return NamedReading(label, resolve)


def node_temperature(node) -> NamedReading:
    """A node's temperature (K). With a thermal layer: its state at an interior node, its
    boundary temperature (the driver `"thermal.x_boundary"`) at a boundary node. Without
    one: the prescribed temperature `"T"`, which the model must declare. A driver is read
    one step late in a sampled network."""
    label = f"node_temperature({node!r})"

    def resolve(net, layers, drivers):
        if "thermal" in layers:
            layer = layers["thermal"]
            n = net.node_index(node)
            boundary = (layer.boundary_idx == n).nonzero()
            if boundary.numel() == 1:
                return DriverReading("thermal.x_boundary", (int(boundary[0, 0]),))
            return StateReading("thermal.x", (_row(layer, net, node, label),))
        if "T" not in drivers:
            raise ValueError(f"controls: {label}: the model has no thermal layer and "
                             f"declares no prescribed temperature driver 'T'")
        return DriverReading("T", (net.node_index(node),))

    return NamedReading(label, resolve)


def node_pressure(node) -> NamedReading:
    """A node's gauge pressure (Pa): the air layer's potential."""
    label = f"node_pressure({node!r})"

    def resolve(net, layers, drivers):
        _layer(layers, "air", label)
        return StateReading("air.phi", (net.node_index(node),))

    return NamedReading(label, resolve)


def _column(layers, kind: str, position: int, label: str) -> int:
    sl = _layer(layers, "air", label).kind_slice(kind)
    if not 0 <= position < sl.stop - sl.start:
        raise IndexError(f"controls: {label}: kind {kind!r} has {sl.stop - sl.start} edges, "
                         f"not position {position}")
    return sl.start + int(position)


def edge_flow(kind: str, position: int = 0) -> NamedReading:
    """The mass flow (kg/s) of edge `position` of `kind`, along the edge."""
    label = f"edge_flow({kind!r}, {position})"

    def resolve(net, layers, drivers):
        return StateReading("air.q", (_column(layers, kind, position, label),))

    return NamedReading(label, resolve)


def edge_pressure_difference(kind: str, position: int = 0) -> NamedReading:
    """The driving pressure difference (Pa) of an edge, from-node minus to-node with the
    edge's drives (`PotentialFlowLayer.dp`), which `simulate` records as `"air.dp"`."""
    label = f"edge_pressure_difference({kind!r}, {position})"

    def resolve(net, layers, drivers):
        return StateReading(PRESSURE_DIFFERENCE, (_column(layers, kind, position, label),))

    return NamedReading(label, resolve)
