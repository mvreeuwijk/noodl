"""The parts of the iterated couplings `Model._iterate` and `CoupledModel._iterate` share.

`run_passes` is the one primal loop of both: it counts the passes, runs pass 1 with grad
enabled (when grad mode is on) to learn whether anything differentiable reaches the state
and then drops that pass's graph, runs every later pass with no graph, asks for a verdict on
detached values, and stops on convergence, on the caller's early-stop rule or at the pass
budget. `confirm_gradient_dependence` is the gradient-dependence rule (#22), in one place:
when pass 1 saw nothing differentiable, ask again with one probe pass at the certified
point. Each caller keeps its own convergence measure, relaxation, messages and adjoint
behind `PassProblem`.

`detached` and `unsolved_mask` are the two diagnostics helpers both loops need.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

import torch

from noodl._grad_reach import any_requires_grad

Tensor = torch.Tensor
Z = TypeVar("Z")
Out = TypeVar("Out")


@dataclass
class Verdict:
    """A pass's convergence verdict: `converged` per instance (a bool tensor of the batch
    shape), the measured `change` per key, and `stop_early` -- the caller's own rule for
    giving up before the pass budget (the Model's "failed again from within tolerance")."""

    converged: Tensor
    change: dict[str, Tensor] = field(default_factory=dict)
    stop_early: bool = False


class PassProblem(Protocol[Z, Out]):
    """What `run_passes` needs from an iterated coupling. `z` is the fed iterate."""

    def run(self, z: Z) -> Out:
        """One primal pass at the fed iterate `z` (under the grad mode the loop chose)."""
        ...

    def depends_on_grad(self, out: Out) -> bool:
        """Whether any RECOMPUTED tensor of `out` requires grad."""
        ...

    def detach(self, z: Z, out: Out) -> tuple[Z, Out]:
        """`z` and `out` with pass 1's graph dropped, all of it."""
        ...

    def judge(self, z: Z, out: Out, *, needs_adjoint: bool) -> Verdict | None:
        """The verdict on the pass that ran at `z` and returned `out` (under `no_grad`);
        `None` for no verdict on this pass."""
        ...

    def advance(self, z: Z, out: Out, verdict: Verdict | None) -> Z:
        """The next fed iterate (under `no_grad`)."""
        ...


@dataclass
class LoopOutcome(Generic[Z, Out]):
    """What `run_passes` ends with.

    `certified` is the fed iterate of the last pass run (the certified one when converged,
    also on an early stop), after pass 1's detach; `out` is that pass's output.
    `needs_adjoint` is pass 1's observation; `passes` counts primal passes only."""

    passes: int
    verdict: Verdict | None
    certified: Any
    out: Any
    needs_adjoint: bool
    stopped_early: bool

    @property
    def converged(self) -> bool:
        """Whether every instance converged on the last pass."""
        return self.verdict is not None and bool(self.verdict.converged.all())


def run_passes(problem: PassProblem, z0, *, max_passes: int) -> LoopOutcome:
    """Run `problem`'s passes from the fed iterate `z0`, at most `max_passes` of them.

    Pass 1 runs with grad ENABLED when grad mode is on, to learn early whether anything
    differentiable reaches the state; its graph is dropped right after (`problem.detach`).
    Every later pass runs with no graph at all. The verdict and the next fed iterate are
    computed under `no_grad`. The loop stops when every instance converged, when the
    verdict asks to stop early, or after `max_passes` passes (no next iterate is formed
    after the last one).
    """
    passes = 0
    z = z0
    certified = z0
    out = None
    verdict: Verdict | None = None
    needs_adjoint = False
    stopped_early = False
    while passes < max_passes:
        passes += 1
        with torch.set_grad_enabled(passes == 1 and torch.is_grad_enabled()):
            out = problem.run(z)
            if passes == 1:
                needs_adjoint = problem.depends_on_grad(out)
                z, out = problem.detach(z, out)
        certified = z
        with torch.no_grad():
            verdict = problem.judge(z, out, needs_adjoint=needs_adjoint)
        if verdict is not None:
            if bool(verdict.converged.all()):
                break
            if verdict.stop_early:
                stopped_early = True
                break
        if passes < max_passes:
            with torch.no_grad():
                z = problem.advance(z, out, verdict)
    return LoopOutcome(passes=passes, verdict=verdict, certified=certified, out=out,
                       needs_adjoint=needs_adjoint, stopped_early=stopped_early)


def confirm_gradient_dependence(problem: PassProblem, outcome: LoopOutcome, *,
                                roots: tuple) -> bool:
    """Whether the returned state depends on something differentiable.

    Pass 1's answer when it was yes. Otherwise -- pass 1 ran at the START point, and a
    closure may take a parameter-dependent branch only once the state has moved -- ask
    again, once, under grad at `outcome.certified`, the point whose derivative is returned,
    and drop that probe pass's output and graph. Skipped (no) when grad mode is off or when
    no tensor requiring grad is reachable from `roots` at all
    (`noodl._grad_reach.any_requires_grad`), where no pass can produce a dependency. The
    probe pass is not counted in `outcome.passes`.
    """
    if outcome.needs_adjoint:
        return True
    if not torch.is_grad_enabled() or not any_requires_grad(*roots):
        return False
    with torch.enable_grad():
        probe = problem.run(outcome.certified)
        found = problem.depends_on_grad(probe)
    del probe
    return found


def detached(diag):
    """The same (nested) diagnostics structure with every tensor detached.

    Diagnostics are a REPORT. A tensor in them that is still attached to a pass's graph
    offers a second, silent route around the fixed-point adjoint: a loss touching it would
    be differentiated through that single pass, which is exactly the truncated derivative
    the implicit adjoint replaces. `_iterate` therefore detaches the diagnostics of both the pass it
    runs only to discover whether anything differentiable reaches the state and the one the
    adjoint is attached to.
    """
    if isinstance(diag, Tensor):
        return diag.detach()
    if isinstance(diag, dict):
        return {k: detached(v) for k, v in diag.items()}
    return diag


def unsolved_mask(diag: Mapping, names: Iterable[str]) -> Tensor | None:
    """Per instance, whether any potential solve of a pass (its `diag`) stopped short of
    its own tolerance (`on_failure="return_primal"`); ``None`` when every one converged.
    `names` are the Kirchhoff layers whose `diag[name]["converged"]` is read."""
    failed: Tensor | None = None
    for name in names:
        ok = diag.get(name, {}).get("converged")
        if ok is None:
            continue
        ok = torch.as_tensor(ok)
        if bool(ok.all()):
            continue
        failed = ~ok if failed is None else (failed | ~ok)
    return failed
