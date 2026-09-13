"""LinearOperator protocol and the solver's per-instance result contract.

An operator is defined by its ACTION (`matvec`, `rmatvec`) rather than by how it is stored;
`assemble()` is an optional explicit form and may return `None`. `symmetric` and
`spd_certificate()` are DECLARATIONS an operator makes about itself, never inferred here --
trustworthy only once a caller has checked them (the symmetry tests in
`tests/operators/test_base.py`, and `solvers.grounding.spd_certificate` for the SPD case).

`SolveResult` is what every `solvers.*` entry point returns; it never raises on
non-convergence (see the milestone design, section 3.2). Only `raise_on_failure` turns a
per-instance failure into an exception, and only when a caller -- a public simulation or
gradient entry point -- chooses to call it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable

import torch

from tellegen.operators.dense import DenseOperator

Tensor = torch.Tensor


class SolverStatus(IntEnum):
    """Per-instance solver outcome, carried in `SolveResult.status`."""

    CONVERGED = 0
    MAX_ITER = 1
    BREAKDOWN = 2
    SINGULAR = 3
    # RESERVED: no `solvers.*` entry point currently produces this status. An eligibility
    # refusal (an explicit method="cg" that cannot certify SPD, or method="auto" seeing a
    # mixed-certification batch) raises RuntimeError instead of returning a SolveResult at
    # all, so there is no per-instance status to set -- see solvers/select.py's module
    # docstring. Kept for a future per-instance (rather than whole-call) certification
    # failure, should one be added.
    NOT_CERTIFIED = 4


@dataclass
class SolveResult:
    """Per-instance solver outcome. Never raises on its own; see `raise_on_failure`."""

    x: Tensor
    converged: Tensor
    iterations: Tensor
    residual: Tensor
    status: Tensor

    def raise_on_failure(self, where: str) -> SolveResult:
        """Raise `RuntimeError` naming every non-converged flat batch index, its status
        (by `SolverStatus` name) and its residual, if any instance failed to converge.
        Returns `self` unchanged when every instance converged, so this chains:
        `result = pcg(op, b).raise_on_failure("PotentialFlowLayer.solve")`.
        """
        if bool(torch.all(self.converged)):
            return self
        flat_converged = self.converged.reshape(-1)
        flat_status = self.status.reshape(-1)
        flat_residual = self.residual.reshape(-1)
        bad = torch.nonzero(~flat_converged, as_tuple=False).flatten().tolist()
        statuses = [SolverStatus(int(flat_status[i])).name for i in bad]
        residuals = [float(flat_residual[i]) for i in bad]
        raise RuntimeError(
            f"{where}: batch indices {bad} failed to converge; "
            f"statuses {statuses}, residuals {residuals}"
        )


def as_operator(op: LinearOperator | Tensor) -> LinearOperator:
    """Auto-wrap a plain dense tensor as a ``DenseOperator``; pass a ``LinearOperator`` through.

    The single compatibility shim that makes every pre-Milestone-1b caller -- each of which
    passes a callable returning a dense ``(..., m, m)`` tensor rather than a ``LinearOperator``
    -- invisible to `newton.newton` and `implicit.adjoint`, the two entry points that call it.
    ``DenseOperator``'s ``symmetric=False`` default is load-bearing: it lets this wrap happen
    with no keyword at all, and it makes no symmetry claim on a tensor that is in general
    nonsymmetric (a Newton Jacobian, an affine system's matrix) -- so an explicit
    ``method="cg"`` is correctly refused downstream, and ``"auto"`` falls through to GMRES for
    anything not separately routed to the direct path.

    The "a bare dense tensor resolves to ``method='direct'`` under ``method='auto'``" rule
    lives in each CALLER (`newton.newton`, `implicit.adjoint`), not here: this function only
    wraps, it never chooses a solver.
    """
    if isinstance(op, torch.Tensor):
        return DenseOperator(op)
    return op


@runtime_checkable
class LinearOperator(Protocol):
    """A batched linear operator, defined by its action rather than its storage.

    `@runtime_checkable` is required (not merely decorative) so that
    `isinstance(op, LinearOperator)` -- the protocol-conformance test in
    `tests/operators/test_base.py` -- can be written at all; `Protocol` classes are not
    usable with `isinstance` without it. As with every `runtime_checkable` protocol, the
    check is structural (attribute and method NAMES only) and does not verify signatures.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    symmetric: bool

    def matvec(self, x: Tensor) -> Tensor: ...

    def rmatvec(self, x: Tensor) -> Tensor: ...

    def diagonal(self) -> Tensor: ...

    def assemble(self) -> Tensor | None: ...

    def spd_certificate(self) -> Tensor | None: ...
