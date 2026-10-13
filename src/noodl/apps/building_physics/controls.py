"""Control networks: sensors, signal blocks and actuators.

A `ControlNetwork` is a model closure, first in `Model(closures=[network, ...])`. It reads its
sensors, evaluates its blocks in the order given, and writes every actuator's value into a
driver with `inputs.scatter`. Inputs (schedules) are applied before the closures run, so a
control overrides a schedule on the same driver entry, as in CONTAM.

A sampled network (the default) is evaluated from the state the step starts from, as ContamX
evaluates its control nodes, once per call that carries a step context: once per
`Model.step`, whatever sub-steps the transport layers take inside it (a `coupling="iterate"`
step re-evaluates it from the same step-start outputs and memory on every pass). A flow or
pressure sensor reads the last airflow solve, and 0 before the first. A reading of a driver
(a prescribed temperature, an ambient mass fraction) gives the value the previous step used.
A block whose input comes later in the order reads that input's previous value, 0 before the
first evaluation. The network carries its block outputs and its blocks' memory in the state,
under `"<name>.outputs"` and `"<name>.memory"`. The memory holds what a block keeps between
evaluations: a sensor's lagged value, the previous step's driver value. Called outside a
step (no step context: a query, or the initial airflow), it writes its stored outputs again.

A continuous network (`sampled=False`) is algebraic in the state at every call. A block that
integrates keeps its state in `"<name>.memory"`; called with a step context, the network
advances that memory by one explicit Euler step of `rates`, evaluated at the state and
memory the step starts from.

Units are those of what is read (kg/kg, K, Pa, kg/s). Blocks are dimensionless unless a gain
or an offset says otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from noodl._broadcast import broadcast_shapes
from noodl.apps.building_physics.inputs import scatter

F64 = torch.float64
State = Mapping[str, Tensor]
Drivers = Mapping[str, Tensor]
PRESSURE_DIFFERENCE = "air.dp"


def _f64(value) -> Tensor:
    return torch.as_tensor(value, dtype=F64)


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
    """`fn(state, drivers)`, for any other reading (the Modelica route's zone temperature)."""

    def __init__(self, fn: Callable[[State, Drivers], Tensor], *,
                 previous_step: bool = False) -> None:
        self.fn, self.previous_step = fn, bool(previous_step)

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
    """A constant value."""

    def __init__(self, value: float, *, name: str = "") -> None:
        self.value, self.name = float(value), name

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

    def __init__(self, reading, *, offset: float = 0.0, scale: float = 1.0,
                 time_constant: float = 0.0, name: str = "") -> None:
        if time_constant < 0.0:
            raise ValueError(f"Sensor {name!r}: time_constant must be >= 0 s, got "
                             f"{time_constant}")
        self.reading, self.name = reading, name
        self.offset, self.scale = float(offset), float(scale)
        self.time_constant = float(time_constant)
        self._previous = bool(getattr(reading, "previous_step", False))
        self._lag = self.time_constant > 0.0
        self.memory_size = int(self._previous) + int(self._lag)
        self.sampled_only = self._lag

    def step(self, u, memory, ev):
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

    def __init__(self, *inputs: int, weights: Sequence[float] | None = None,
                 name: str = "") -> None:
        super().__init__(*inputs, name=name)
        self.weights = ((1.0,) * len(self.inputs) if weights is None
                        else tuple(float(w) for w in weights))
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

    def __init__(self, a: int, *, offset: float = 0.0, gain: float = 1.0,
                 name: str = "") -> None:
        self.inputs, self.name = (int(a),), name
        self.offset, self.gain = float(offset), float(gain)

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


# ------------------------------------------------------------------------------- network
class ControlNetwork:
    """Blocks evaluated in order and the actuators that write their outputs into drivers:
    a model closure (module docstring). Blocks without a name are named `"block <i>"`."""

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
        self._slots: list[slice] = []
        start = 0
        for block in self.blocks:
            self._slots.append(slice(start, start + block.memory_size))
            start += block.memory_size
        self.memory_size = start
        self.outputs_key, self.memory_key = f"{self.name}.outputs", f"{self.name}.memory"
        self.input_key = f"{self.name}.input"
        self.state_keys = (self.outputs_key, self.memory_key)
        read = {b.key for b in self.blocks if isinstance(b, DriverValue)}
        read |= {b.reading.key for b in self.blocks
                 if isinstance(b, Sensor) and isinstance(b.reading, DriverReading)}
        self.inputs = tuple(sorted(read))
        self.outputs = tuple(sorted({a.key for a in self.actuators})) + self.state_keys

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
        """Whether a sensor reads state `key` (`simulate` records `"air.dp"` when one does)."""
        return any(isinstance(b, Sensor) and isinstance(b.reading, StateReading)
                   and b.reading.key == key for b in self.blocks)

    def initial_state(self) -> dict[str, Tensor]:
        """Outputs 0 and every block's initial memory."""
        memory = [v for b in self.blocks for v in b.initial_memory()]
        return {self.outputs_key: torch.zeros(len(self.blocks), dtype=F64),
                self.memory_key: torch.tensor(memory, dtype=F64)}

    def initial_drivers(self) -> dict[str, Tensor]:
        """Zeros for `"<name>.input"` when a block reads it (the inputs write it each step)."""
        if self.input_key not in self.inputs:
            return {}
        n = [b.index for b in self.blocks
             if isinstance(b, DriverValue) and b.key == self.input_key and b.index is not None]
        return {self.input_key: torch.zeros(max(n, default=0) + 1, dtype=F64)}

    def _stored(self, state: State) -> tuple[Tensor, Tensor]:
        outputs, memory = state.get(self.outputs_key), state.get(self.memory_key)
        if outputs is None or memory is None:
            init = self.initial_state()
            outputs = init[self.outputs_key] if outputs is None else outputs
            memory = init[self.memory_key] if memory is None else memory
        return outputs, memory

    def _run(self, state: State, drivers: Drivers, memory: Tensor, dt: float | None,
             want: str = "") -> tuple[Tensor, Tensor, Tensor | None]:
        """One pass over the blocks: outputs, new memory and, for `want` "rate" or
        "switching", that quantity of every block."""
        ev = Evaluation(state, drivers, dt)
        out = list(self._stored(state)[0].unbind(-1))
        mem, extra = [], []
        for i, block in enumerate(self.blocks):
            u = [out[j] for j in block.inputs]
            m = memory[..., self._slots[i]]
            y, m_new = block.step(u, m, ev)
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

    def evaluate(self, state: State, drivers: Drivers, memory: Tensor | None = None) -> Tensor:
        """Every block's output at `state`, advancing nothing (continuous networks)."""
        memory = self._stored(state)[1] if memory is None else memory
        return self._run(state, drivers, memory, None)[0]

    def rates(self, state: State, drivers: Drivers, memory: Tensor | None = None) -> Tensor:
        """d(memory)/dt at `state` (continuous networks; zero outside integrators)."""
        memory = self._stored(state)[1] if memory is None else memory
        return self._run(state, drivers, memory, None, "rate")[2]

    def switching(self, state: State, drivers: Drivers,
                  memory: Tensor | None = None) -> Tensor | None:
        """Every block's switching values at `state`, or `None` when no block has any."""
        memory = self._stored(state)[1] if memory is None else memory
        return self._run(state, drivers, memory, None, "switching")[2]

    def write(self, drivers: Drivers, outputs: Tensor) -> dict[str, Tensor]:
        """Every actuator's value written into its driver (a later actuator on the same key
        sees the earlier ones' writes); the given drivers are not modified."""
        out: dict[str, Tensor] = {}
        for a in self.actuators:
            value = outputs[..., a.block]
            base = out.get(a.key, drivers.get(a.key))
            if base is None and a.index is not None:
                raise KeyError(f"controls: actuator {a.name or a.key!r} writes positions "
                               f"{list(a.index)} of driver {a.key!r}, which the model does "
                               f"not have")
            if a.index is None:
                # One value per batch member, the same for every entry of the driver.
                whole = value if base is None or value.ndim == 0 else value.unsqueeze(-1)
                out[a.key] = scatter(base, None, whole)
                continue
            v = value.unsqueeze(-1).expand(*value.shape, len(a.index))
            out[a.key] = (scatter(base, a.index, v) if a.mode == "set"
                          else _add_at(base, a.index, v))
        return out

    def __call__(self, state: State, drivers: Drivers, ctx=None) -> dict[str, Tensor]:
        outputs, memory = self._stored(state)
        advance = ctx is not None and getattr(ctx, "dt", None) is not None
        if self.sampled:
            if advance:
                outputs, memory, _ = self._run(state, drivers, memory, float(ctx.dt))
        else:
            if advance:
                memory = memory + float(ctx.dt) * self.rates(state, drivers, memory)
            outputs = self.evaluate(state, drivers, memory)
        writes = self.write(drivers, outputs)
        if self.sampled and advance:
            after = {**drivers, **writes}
            memory = _concat([*(b.after(memory[..., s], state, after)
                                for b, s in zip(self.blocks, self._slots, strict=True)),
                              memory[..., :0]])
        return {**writes, self.outputs_key: outputs, self.memory_key: memory}
