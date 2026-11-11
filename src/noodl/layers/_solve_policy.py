"""How `on_failure`, `differentiable` and `diagnostics=` combine.

`KirchhoffLayer.solve` decides, from these keywords and from whether anything differentiable
reaches the solve, whether the solve is differentiated or is the primal solve, and what
`on_failure` Newton is given: one function, `resolve`, for both paths (the fast path, where
every flow follows from the potentials, and the path with flow unknowns).

`resolve` completes the `newton_kwargs` in place, in this order, because the error a caller
sees depends on it: `method` (the layer's `linear_solver` unless given), `on_failure`
(`"return_primal"` becomes `"return"`), `where`, the refusal of `"return"` without
`diagnostics=`, then `forward_only` and, where the solve is differentiated, the final
`"raise"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from noodl.layers.kirchhoff import KirchhoffLayer


@dataclass(frozen=True)
class SolvePolicy:
    """The resolved path of one solve.

    `differentiable`: the solve is differentiated (the implicit adjoint) and not the primal
    path. `forward_only`: a differentiable solve that nothing reaches
    (`on_failure="return_primal"`). Both paths run it as their primal solve, the Newton of
    `differentiable=False` on the layer's own slopes, with the closure check at every
    residual.
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


def resolve(
    layer: KirchhoffLayer,
    newton_kwargs: dict,
    diagnostics: dict | None,
    differentiable: bool,
    drivers,
    sources,
    phi_boundary,
    step_start=(),
) -> SolvePolicy:
    """The path of one solve; completes `newton_kwargs` in place (order: module docstring).

    `step_start` are the step-start tensors the differentiable solve threads
    (`FunctionalLaws.step_start`). `on_failure="return_primal"` is `"return"` where nothing
    differentiable reaches the solve (`forward_only`, or the primal path) and `"raise"` where
    it is differentiated. Its refusal without `diagnostics=` names `"return"` on the fast
    path and `"return_primal"` on a layer with flow unknowns (the messages each path gave).
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
        named = ("return_primal" if primal_only and layer._unknowns is not None
                 else "return")
        require_diagnostics(layer.name, named, diagnostics)
    # "return_primal": "return" unless something differentiable reaches the solve; a
    # differentiable solve that nothing reaches is not differentiated.
    forward_only = (
        primal_only
        and differentiable
        and not layer._reaches_grad(drivers, sources, phi_boundary, step_start)
    )
    if primal_only and differentiable and not forward_only:
        newton_kwargs["on_failure"] = "raise"
    return SolvePolicy(differentiable=differentiable, forward_only=forward_only)
