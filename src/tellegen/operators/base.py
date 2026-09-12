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

import torch

Tensor = torch.Tensor


class SolverStatus(IntEnum):
    """Per-instance solver outcome, carried in `SolveResult.status`."""

    CONVERGED = 0
    MAX_ITER = 1
    BREAKDOWN = 2
    SINGULAR = 3
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
