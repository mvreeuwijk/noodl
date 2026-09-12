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
