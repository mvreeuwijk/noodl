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

`fixed_point` is the engine behind `Model.fixed_point`: the same loop for an arbitrary pass
function on a state, with its own `PassProblem` (`_FixedPointPasses`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

import torch

from noodl.solvers.fixed_point import differentiate_fixed_point

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


def confirm_gradient_dependence(problem: PassProblem, outcome: LoopOutcome) -> bool:
    """Whether the returned state depends on something differentiable.

    Pass 1's answer when it was yes. Otherwise -- pass 1 ran at the START point, and a
    closure may take a parameter-dependent branch only once the state has moved -- ask
    again, once, under grad at `outcome.certified`, the point whose derivative is returned,
    and drop that probe pass's output and graph. Skipped (no) only when grad mode is off:
    no inspection of the model can prove that nothing a pass reads requires grad (a closure
    may read a tensor through a module attribute, a global, a cell ...), so in grad mode the
    probe always runs. A forward-only run avoids it with `torch.no_grad()`. The probe pass
    is not counted in `outcome.passes`.
    """
    if outcome.needs_adjoint:
        return True
    if not torch.is_grad_enabled():
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


class _FixedPointPasses:
    """`fixed_point`'s side of `PassProblem`.

    The fed iterate `z` is a state; a pass's output is the state `pass_fn` returns. The
    verdict compares the watched keys of the fed state with the output's, per instance, and
    `advance` relaxes the watched keys toward the output (the other keys come from it).
    """

    def __init__(self, pass_fn: Callable[[dict], dict], watch: Sequence[str], tol: float,
                 relaxation: float, measure: Callable[[dict, dict], Tensor] | None,
                 where: str) -> None:
        self.pass_fn = pass_fn
        self.watch = list(watch)
        self.tol = tol
        self.relaxation = relaxation
        self.measure = measure if measure is not None else self._largest_change
        self.where = where
        self.changes: list[float] = []

    def _largest_change(self, fed: dict, out: dict) -> Tensor:
        """The largest absolute change over every watched key (one number: the state is
        one instance; a `measure` gives the per-instance shape)."""
        worst = None
        for key in self.watch:
            c = (out[key] - fed[key]).abs().amax()
            worst = c if worst is None else torch.maximum(worst, c)
        return worst

    def run(self, z: dict) -> dict:
        out = self.pass_fn(z)
        missing = [k for k in self.watch if k not in out]
        if missing:
            raise KeyError(
                f"{self.where}: the pass did not return the watched key(s) {missing}"
            )
        return out

    def depends_on_grad(self, out: dict) -> bool:
        return any(isinstance(v, Tensor) and v.requires_grad for v in out.values())

    def detach(self, z: dict, out: dict) -> tuple[dict, dict]:
        return z, {k: v.detach() if isinstance(v, Tensor) else v for k, v in out.items()}

    def judge(self, z: dict, out: dict, *, needs_adjoint: bool) -> Verdict:
        if any(k not in z for k in self.watch):
            # The first pass of a state that does not carry the iterate has nothing to
            # compare with: not converged, whatever the pass returned.
            self.changes.append(float("inf"))
            return Verdict(converged=torch.zeros((), dtype=torch.bool))
        change = torch.as_tensor(self.measure(z, out))
        self.changes.append(float(change.max()))
        return Verdict(converged=change <= self.tol, change={"change": change})

    def advance(self, z: dict, out: dict, verdict: Verdict | None) -> dict:
        fed = dict(out)
        if self.relaxation != 1.0:
            for key in self.watch:
                if key in z:
                    fed[key] = z[key] + self.relaxation * (out[key] - z[key])
        return fed


def fixed_point(
    pass_fn: Callable[[dict], dict], state: dict, *, watch: Sequence[str], tol: float,
    max_passes: int, relaxation: float, measure: Callable[[dict, dict], Tensor] | None,
    adjoint_rtol: float, adjoint_restart: int | None, diagnostics: dict | None,
    where: str,
) -> dict:
    """Repeat `pass_fn` from `state` until the watched keys stop changing; see
    `Model.fixed_point`, which documents every argument."""
    watch = list(watch)
    if not watch:
        raise ValueError(f"{where}: watch must name at least one state key")
    if int(max_passes) < 1:
        raise ValueError(f"{where}: max_passes must be >= 1, got {max_passes!r}")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError(f"{where}: relaxation must be in (0, 1], got {relaxation!r}")
    problem = _FixedPointPasses(pass_fn, watch, tol, relaxation, measure, where)
    outcome = run_passes(problem, dict(state), max_passes=int(max_passes))
    report: dict = {}
    adjoint: str | None = None
    changes = problem.changes
    if diagnostics is not None:
        diagnostics.update({
            "passes": outcome.passes, "changes": changes,
            "contraction": _contraction(changes), "adjoint": None,
            "adjoint_batched": False,
        })
    if not outcome.converged:
        raise RuntimeError(
            f"{where}: the fixed point did not converge within {max_passes} passes; the "
            f"largest change on the last pass was {changes[-1]:.3e} against tol {tol:.1e} "
            f"(measured contraction {_contraction(changes):.3g}; pass relaxation < 1 if "
            f"it is >= 1)"
        )
    out = outcome.out
    if outcome.needs_adjoint:
        fed = outcome.certified
        interface = []
        for key in watch:
            z = fed[key].detach()
            if z.shape != out[key].shape:
                z = torch.broadcast_to(z, out[key].shape).contiguous()
            interface.append(z)
        out_keys: list[str] = []

        def adjoint_pass(z: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
            f = dict(fed)
            f.update(zip(watch, z, strict=True))
            result = pass_fn(f)
            out_keys[:] = list(result)
            z_next = [result[k] for k in watch]
            echoed = [k for k, n, t in zip(watch, z_next, z, strict=True) if n is t]
            if echoed:
                raise RuntimeError(
                    f"{where}: cannot differentiate the fixed point because the pass hands "
                    f"the watched key(s) {echoed} back unchanged instead of recomputing "
                    f"them, which makes the interface Jacobian singular"
                )
            return [result[k] for k in out_keys], z_next

        flat = differentiate_fixed_point(
            interface, adjoint_pass, rtol=adjoint_rtol, restart=adjoint_restart,
            where=where, batch_shape=tuple(outcome.verdict.converged.shape), report=report,
        )
        out = dict(zip(out_keys, flat, strict=True))
        adjoint = "implicit"
    if diagnostics is not None:
        diagnostics.update({"adjoint": adjoint,
                            "adjoint_batched": bool(report.get("batched", False))})
    return out


def _contraction(changes: list[float]) -> float:
    """The ratio of the last two changes (NaN before two passes, 0 once the earlier one is
    exactly zero): the measured spectral radius of the iteration."""
    if len(changes) < 2:
        return float("nan")
    return changes[-1] / changes[-2] if changes[-2] > 0 else 0.0
