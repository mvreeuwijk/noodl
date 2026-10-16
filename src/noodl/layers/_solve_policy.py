"""How `on_failure`, `differentiable`, `analytic_primal` and `diagnostics=` combine.

`KirchhoffLayer.solve` has a classic (nodal Newton) path and an MNA path. Both decide, from the
same keywords and from whether anything differentiable reaches the solve, whether the solve is
differentiated, whether it is a forward Newton alone, and what `on_failure` Newton is given.
The decisions live here, one function per path. Each reproduces the order in which the
`newton_kwargs` are mutated, because the error a caller sees depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from noodl.layers.kirchhoff import KirchhoffLayer


@dataclass(frozen=True)
class SolvePolicy:
    """The resolved path of one classic solve.

    `differentiable`: the solve is differentiated (the implicit adjoint) and not the primal
    path. `forward_only`: a differentiable solve that nothing reaches, run as its forward
    Newton alone (`on_failure="return_primal"`).
    """

    differentiable: bool
    forward_only: bool = False


def require_diagnostics(name: str, on_failure: str, diagnostics: dict | None) -> None:
    """Refuse `on_failure=<on_failure>` without a `diagnostics=` dict to carry the status."""
    if diagnostics is not None:
        return
    raise ValueError(
        f"KirchhoffLayer {name!r}: on_failure={on_failure!r} requires "
        f"diagnostics= so the status is not silently dropped; pass a dict and read "
        f"its 'converged' and 'residual_norm' entries."
    )


def resolve_mna(
    layer: KirchhoffLayer,
    newton_kwargs: dict,
    diagnostics: dict | None,
    differentiable: bool,
    drivers,
    sources,
    phi_boundary,
    history,
) -> bool:
    """The `differentiable` flag for an MNA solve; rewrites `on_failure="return_primal"`.

    `"return_primal"` is `"return"` where nothing differentiable reaches the solve (then its
    primal Newton alone) and `"raise"` where it is differentiated. Mutates `newton_kwargs`.
    """
    if newton_kwargs.get("on_failure") != "return_primal":
        return differentiable
    reaches = layer._reaches_grad(drivers or {}, sources, phi_boundary) or (
        history is not None
        and torch.is_grad_enabled()
        and any(isinstance(h, torch.Tensor) and h.requires_grad for h in history)
    )
    require_diagnostics(layer.name, "return_primal", diagnostics)
    if differentiable and not reaches:
        # the differentiable path's own refusal still applies to the primal one
        layer._check_no_unreachable_differentiable_tensors()
        differentiable = False
    newton_kwargs["on_failure"] = "raise" if differentiable else "return"
    return differentiable


def resolve_classic(
    layer: KirchhoffLayer,
    newton_kwargs: dict,
    diagnostics: dict | None,
    differentiable: bool,
    drivers,
    sources,
    phi_boundary,
) -> SolvePolicy:
    """The path of a classic solve; completes `newton_kwargs` in place.

    Order of the mutations: `method`, `on_failure` (`return_primal` to `return`), `where`, the
    diagnostics refusal, then `analytic_primal`, `forward_only` and the final `"raise"`.
    """
    newton_kwargs.setdefault("method", layer.linear_solver)
    primal_only = newton_kwargs.get("on_failure") == "return_primal"
    if primal_only:
        # decided below, once the path is known: "return" on the primal path, "raise"
        # on the differentiable one
        newton_kwargs["on_failure"] = "return"
    # Name this layer on every error the Newton solve or its INNER linear solves raise.
    # Grounding is certified at phi0, but slopes change between Newton iterates, so an
    # instance can lose it mid-iteration; without this the refusal read "newton:
    # method='auto' refuses to split the batch ..." with no way to tell which layer of a
    # composed model over one network produced it.
    newton_kwargs.setdefault("where", f"KirchhoffLayer {layer.name!r} solve")
    if newton_kwargs.get("on_failure") == "return":
        # `solve` returns (phi, q) tensors; without a diagnostics dict there is nowhere
        # for `converged`/`residual_norm` to go, and a non-converged phi would be
        # indistinguishable from a converged one. The escape hatch is never silent: the
        # result carries the status.
        require_diagnostics(layer.name, "return", diagnostics)
    return _classic_path(
        layer, newton_kwargs, primal_only, differentiable, drivers, sources, phi_boundary
    )


def _classic_path(
    layer, newton_kwargs, primal_only, differentiable, drivers, sources, phi_boundary
) -> SolvePolicy:
    if differentiable and layer.analytic_primal and not layer._reaches_grad(
        drivers, sources, phi_boundary
    ):
        # Nothing differentiable reaches this solve, so the implicit path would return
        # the same detached root through an autograd Jacobian (`analytic_primal`).
        differentiable = False
    # "return_primal": "return" unless something differentiable reaches the solve; a
    # differentiable solve that nothing reaches is its forward Newton alone.
    forward_only = (
        primal_only
        and differentiable
        and not layer._reaches_grad(drivers, sources, phi_boundary)
    )
    if primal_only and differentiable and not forward_only:
        newton_kwargs["on_failure"] = "raise"
    return SolvePolicy(differentiable=differentiable, forward_only=forward_only)
