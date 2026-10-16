"""Which signal drives which input, and its value on the time grid.

A document's `signals` each drive one or more inputs `"<instance>.<input>"`. `SignalResolver`
keys them by the input they drive and evaluates them on the grid (`signals.evaluate`), once
each however many inputs they feed. A `Modelica.Blocks.Math` block (`signals.combine`) takes
the value of each of its inputs from the signal that drives it, recursively, so a chain such
as `ramp -> add.u1`, `add -> bouA.p_in` is evaluated from its sources outwards. Two signals
driving one input, a block input nothing drives, a loop of blocks, a signal that cannot be
evaluated and a signal driving an input the reader never read (`check_unused`) are each
appended to the reader's problem list by name; the reader raises them together.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from typing import Protocol

import torch

from noodl.modelica import signals
from noodl.modelica.errors import ModelicaImportError
from noodl.modelica.schema import Signal

Tensor = torch.Tensor

__all__ = ["EVALUATION_ERRORS", "SignalResolver", "evaluation_problem"]

# What evaluating a signal block can raise: the evaluator's own refusals, and the errors a
# malformed parameter (a table column out of range, a non-numeric value) produces in it.
EVALUATION_ERRORS = (ModelicaImportError, ValueError, TypeError, IndexError, KeyError,
                      ArithmeticError)


class _Instance(Protocol):
    name: str
    cls: str


def evaluation_problem(sig: Signal, exc: Exception) -> str:
    """The problem text for a signal that could not be evaluated, naming the signal."""
    if isinstance(exc, ModelicaImportError):    # the evaluator refusing
        return str(exc).removeprefix("modelica: ")
    return f"{sig.name} ({sig.cls}): cannot be evaluated ({type(exc).__name__}: {exc})"


class SignalResolver:
    """The signals of one document keyed by the input they drive, evaluated on `grid`.

    `problems` is the reader's list; every refusal is appended to it. `closed` names blocks
    that are never evaluated here (the building reader's feedback controllers, whose inputs
    are the state's); a subclass may fill it.
    """

    def __init__(self, sigs: Iterable[Signal], grid: Tensor, problems: list[str]) -> None:
        self.grid = grid
        self.problems = problems
        self.by_target: dict[str, Signal] = {}
        self.used: set[str] = set()
        self.closed: set[str] = set()
        sigs = tuple(sigs)
        self.signal_names = {s.name for s in sigs}
        self._values: dict[str, Tensor | None] = {}
        for s in sigs:
            for target in s.drives:  # one block output may feed several inputs
                if target in self.by_target:
                    problems.append(f"{s.name} ({s.cls}): drives {target}, which "
                                    f"{self.by_target[target].name} already drives")
                    continue
                self.by_target[target] = s

    def series(self, target: str) -> Tensor | None:
        """The value of input `target` at every grid time, `(n_grid,)`; `None` when nothing
        drives it, or (with a problem) when it cannot be evaluated."""
        s = self.by_target.get(target)
        if s is None:
            return None
        self.used.add(target)
        return self._series(s, ())

    def require(self, instance: _Instance, port: str) -> Tensor | None:
        """`series` of the enabled input `port` of `instance`; a problem when nothing
        drives it."""
        target = f"{instance.name}.{port}"
        if target not in self.by_target:
            self.problems.append(f"{instance.name} ({instance.cls}): input {port} is enabled "
                                 f"but no supported signal drives it")
            return None
        return self.series(target)

    def _series(self, s: Signal, chain: tuple[str, ...]) -> Tensor | None:
        if s.name in self._values:
            return self._values[s.name]
        if s.name in self.closed:
            return None
        if s.name in chain:
            loop = " -> ".join((*chain[chain.index(s.name):], s.name))
            self.problems.append(f"{s.name} ({s.cls}): signal loop {loop} (feedback is not "
                                 f"supported)")
            return None
        y: Tensor | None = None
        try:
            if signals.is_math(s):
                inputs: dict[str, Tensor] = {}
                for port in signals.math_inputs(s):
                    target = f"{s.name}.{port}"
                    src = self.by_target.get(target)
                    if src is None:
                        self.problems.append(f"{s.name} ({s.cls}): input {port} is not "
                                             f"driven by any supported signal")
                        continue
                    self.used.add(target)
                    value = self._series(src, (*chain, s.name))
                    if value is not None:
                        inputs[port] = value
                if len(inputs) == len(signals.math_inputs(s)):
                    y = signals.combine(s, inputs).expand(self.grid.shape).clone()
            else:
                y = signals.evaluate(s, self.grid, t_start=float(self.grid[0]))
        except EVALUATION_ERRORS as exc:
            self.problems.append(evaluation_problem(s, exc))
            y = None
        self._values[s.name] = y
        return y

    def values_at(self, target: str, t: Tensor) -> Tensor:
        """The value of input `target` at arbitrary times `t`. Only for an input `series`
        returned a value for: it has been evaluated, and any problem recorded, already."""
        s = self.by_target[target]
        if signals.is_math(s):
            inputs = {port: self.values_at(f"{s.name}.{port}", t)
                      for port in signals.math_inputs(s)}
            return signals.combine(s, inputs).expand(t.shape)
        return signals.evaluate(s, t)

    def breaks(self, target: str) -> set[float]:
        """The breakpoints of input `target`'s value inside the grid
        (`signals.breakpoints`)."""
        s = self.by_target[target]
        if signals.is_math(s):
            out: set[float] = set()
            for port in signals.math_inputs(s):
                out |= self.breaks(f"{s.name}.{port}")
            return out
        return set(signals.breakpoints(s, float(self.grid[0]), float(self.grid[-1])))

    def check_unused(self, instances: Collection[str] | None = None,
                     *, ignore: Callable[[str, Signal], bool] | None = None) -> None:
        """Refuse every signal driving an input nothing read: only the inputs of `instances`
        when given, and not those `ignore(instance, signal)` excuses."""
        for target, s in self.by_target.items():
            if target in self.used:
                continue
            inst = target.split(".", 1)[0]
            if instances is not None and inst not in instances:
                continue
            if inst in self.signal_names:
                continue  # feeds a Math block that is itself unused: reported for that block
            if inst in self.closed or s.name in self.closed:
                continue  # read, or refused by name, by the building feedback network
            if ignore is not None and ignore(inst, s):
                continue
            self.problems.append(f"{s.name} ({s.cls}): drives {target}, which this reader "
                                 f"does not read (the input is disabled or not supported)")
