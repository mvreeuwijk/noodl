"""The parts of the iterated couplings `Model._iterate` and `CoupledModel._iterate` share.

`detached` and `unsolved_mask` are the two diagnostics helpers both loops need.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

Tensor = torch.Tensor


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
