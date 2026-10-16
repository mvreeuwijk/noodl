"""Inputs: time-varying values (schedules, time series, weather, ambient files) written into
the drivers at each step.

An `Input` ties a `Signal` to a driver key and, optionally, to positions on that driver's last
axis (`index`), or to the labels the model declares for it (`at`: node names for `"T"`, edge
labels for a `"flow_multiplier.<kind>"`, source names). `bind_inputs` resolves the labels once
and refuses two inputs that write the same entry. `apply_inputs` evaluates every input at `t`
and returns new drivers; the caller's dict and tensors are never modified. A closure that
actuates the same keys (a control network) uses `scatter` too, and runs after the inputs
inside the step, so it overrides a schedule, which is CONTAM's precedence rule.
"""

from __future__ import annotations

import numbers
import operator
from collections.abc import Hashable, Mapping, Sequence

import torch

from noodl._broadcast import broadcast_shapes
from noodl.apps.building_physics.schedules import Calendar, Signal

Tensor = torch.Tensor
F64 = torch.float64


def _labels(at) -> tuple[Hashable, ...] | None:
    if at is None:
        return None
    if isinstance(at, (str, tuple)) or not isinstance(at, Sequence):
        return (at,)
    return tuple(at)


class Input:
    """`signal`'s value written into `drivers[key]` at each step.

    `index` (positions on the driver's last axis) or `at` (labels the model declares for
    `key`, resolved by `bind_inputs`) selects the entries written. With neither, the whole
    driver is replaced (broadcast to its shape). `name` labels the input in error messages.
    The last axis of the signal's value is positions, any axes before it batch: a batched
    value at one position has shape `(B, 1)` (a `TimeSeries` of it, `(N, B, 1)`).
    """

    def __init__(self, signal: Signal, key: str, *, index=None, at=None, name: str = "") -> None:
        if index is not None and at is not None:
            raise ValueError(f"Input {name or key!r}: give index and at, not both")
        self.signal, self.key, self.name = signal, str(key), str(name)
        if index is not None and (isinstance(index, numbers.Integral)
                                  or (isinstance(index, Tensor) and index.ndim == 0)):
            index = (index,)                # one position: a Python, numpy or torch integer
        self.index = None if index is None else tuple(operator.index(i) for i in index)
        self.at = _labels(at)

    def __repr__(self) -> str:
        where = f", index={self.index}" if self.index is not None else (
            f", at={self.at}" if self.at is not None else "")
        return f"Input({self.name or self.signal!r} -> {self.key!r}{where})"

    def _label(self) -> str:
        return self.name or repr(self.signal)


def bind_inputs(model, inputs: Sequence[Input]) -> list[Input]:
    """`inputs` with every `at` resolved to positions through `model.refs`, checked for
    overlap: two inputs may not write the same entry of the same driver (raises naming both).
    A key given with `at` must be one the model declares (`model.refs.field(key)`)."""
    bound: list[Input] = []
    for inp in inputs:
        if inp.at is None:
            bound.append(inp)
            continue
        field = model.refs.field(inp.key)
        if field is None:
            raise KeyError(
                f"Input {inp._label()!r}: the model declares no input {inp.key!r}, so its "
                f"labels {list(inp.at)} cannot be resolved; give positions with index="
            )
        bound.append(Input(inp.signal, inp.key, index=[field.index(lab) for lab in inp.at],
                           name=inp.name))
    owners: dict[tuple[str, int | None], Input] = {}
    for inp in bound:
        slots = [None] if inp.index is None else list(inp.index)
        for slot in slots:
            clash = [o for (k, s), o in owners.items()
                     if k == inp.key and (s == slot or s is None or slot is None)]
            if clash:
                raise ValueError(
                    f"inputs {clash[0]._label()!r} and {inp._label()!r} both write driver "
                    f"{inp.key!r}" + ("" if slot is None else f" at position {slot}")
                )
            owners[(inp.key, slot)] = inp
    return bound


def scatter(base: Tensor | None, index: Sequence[int] | None, value: Tensor) -> Tensor:
    """A new tensor: `base` with `value` written at `index` on the last axis (all of it when
    `index` is None), batch dimensions broadcast. Differentiable in `value`; `base` is never
    modified. The result has `base`'s floating dtype (so a model moved to float32 by
    `Model.to` stays float32); with `base` None, `index` must be None and `value` is
    returned as float64."""
    value = torch.as_tensor(value)
    dtype = base.dtype if base is not None and base.is_floating_point() else F64
    if value.dtype != dtype:
        value = value.to(dtype)
    if base is None:
        if index is not None:
            raise KeyError("scatter: positions given for a driver that does not exist")
        return value
    if index is None:
        shape = broadcast_shapes(base.shape, value.shape)
        return value.expand(shape).clone()
    idx = torch.as_tensor(list(index), dtype=torch.long)
    vbatch = value.shape[:-1] if value.ndim else ()
    batch = broadcast_shapes(base.shape[:-1], vbatch)
    out = base.expand(batch + base.shape[-1:]).clone()
    out[..., idx] = value.expand(batch + (idx.numel(),)) if value.ndim else value
    return out


def apply_inputs(drivers: Mapping[str, Tensor], inputs: Sequence[Input], t: float,
                 calendar: Calendar, *, t_start: float | None = None) -> dict[str, Tensor]:
    """New drivers with every input's value at `t` written in (inputs bound first).

    A signal with a `step_mean(t0, t1, calendar)` method (an occupant's `Presence`) is
    written as its mean over the step (t_start, t] when `t_start` (s since the start) is
    given and below `t`, and as its value at `t` otherwise."""
    out = dict(drivers)
    for inp in inputs:
        if inp.at is not None:
            raise ValueError(f"Input {inp._label()!r} is not bound; call bind_inputs first")
        if inp.index is not None and inp.key not in out:
            raise KeyError(
                f"Input {inp._label()!r}: driver {inp.key!r} does not exist, so positions "
                f"{inp.index} cannot be written"
            )
        mean = getattr(inp.signal, "step_mean", None)
        if mean is not None and t_start is not None and float(t_start) < float(t):
            value = mean(float(t_start), float(t), calendar)
        else:
            value = inp.signal.value(t, calendar)
        out[inp.key] = scatter(out.get(inp.key), inp.index, value)
    return out


def input_breakpoints(inputs: Sequence[Input], lo: float, hi: float,
                      calendar: Calendar) -> list[float]:
    """Every input's jumps and kinks in (lo, hi), merged and sorted."""
    pts: set[float] = set()
    for inp in inputs:
        pts.update(inp.signal.breakpoints(lo, hi, calendar))
    return sorted(pts)
