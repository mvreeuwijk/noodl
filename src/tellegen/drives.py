"""Drive protocol: additive potential-difference terms added to an element's dp.

A drive is a function of the `drivers` mapping ONLY (milestone 2 spec section 4.1): it never
sees the potential `phi`. The Newton Jacobian `A_I diag(g') A_I^T` is therefore exact and stays
symmetric, which is what the SPD certificate and the conjugate-gradient path rely on. Anything a
drive needs that depends on the state -- zone densities for a stack term -- is computed by a
`Model` closure before the solve and written into `drivers`. CONTAM does the same: densities
from barometric pressure and temperature before the iteration, no density-pressure term in the
Jacobian (TN 1887r1 section 3.18, eq. 8-9).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Drive(Protocol):
    """A drive must read every differentiable quantity from `drivers`, never own one.

    `PotentialFlowLayer.solve(differentiable=True)` only threads gradients through
    `Function.apply` for values reachable via `phi_boundary`, `sources`, `drivers`, and each
    Element's own registered `nn.Parameter`s. A Drive instance is captured by closure inside
    the differentiable solve, so any tensor it holds with `requires_grad=True` is invisible to
    the backward pass; `PotentialFlowLayer._check_no_unreachable_differentiable_tensors`
    raises on that. Geometry and index tensors (no grad) may be held as attributes.
    """

    kind: str

    def __call__(self, drivers: Mapping[str, torch.Tensor]) -> torch.Tensor: ...


def check_drive_signature(drive, *, where: str) -> None:
    """Raise `TypeError` unless `drive` is callable as `drive(drivers)`.

    Loud migration guard for the pre-milestone-2 `(phi, drivers)` form: a two-argument drive
    would otherwise fail deep inside `PotentialFlowLayer.dp` with an unattributed TypeError.
    """
    try:
        sig = inspect.signature(drive.__call__)
    except (TypeError, ValueError):  # builtins without a signature: nothing to check
        return
    positional = [
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) != 1:
        raise TypeError(
            f"{where}: drive {type(drive).__name__} (kind {getattr(drive, 'kind', '?')!r}) "
            f"must be callable as drive(drivers) -- a Drive is a function of the drivers "
            f"mapping only (milestone 2 spec section 4.1); got parameters {positional}. A "
            f"drive that used to take (phi, drivers) must drop phi."
        )


class ConstantDrive:
    """A drive that reads a pre-computed, already batched value from `drivers`."""

    def __init__(self, kind: str, key: str) -> None:
        self.kind = kind
        self.key = key

    def __call__(self, drivers: Mapping[str, torch.Tensor]) -> torch.Tensor:
        try:
            return drivers[self.key]
        except KeyError as exc:
            raise KeyError(f"driver {self.key!r} not found") from exc


class Stack:
    """Hydrostatic stack term (CONTAM TN 1887r1 eq. 17 with eq. 64-65 at the path elevation).

    For edge e from node i to node j at elevation z_path, with node reference heights z_ref
    and densities rho (a full-node driver, `drivers[rho_key]`, shape (..., n)):

        value_e = g * ( rho_i (z_ref_i - z_path_e) - rho_j (z_ref_j - z_path_e) )

    so that dp_e = phi_i - phi_j + value_e is the pressure difference AT the opening.
    Geometry is read once at construction and is not differentiable; rho is.
    """

    def __init__(
        self, kind: str, *, src, tgt, z_path, z_ref, rho_key: str = "rho", g: float = 9.80665
    ) -> None:
        self.kind = kind
        self.src = torch.as_tensor(src, dtype=torch.long)
        self.tgt = torch.as_tensor(tgt, dtype=torch.long)
        self.z_path = torch.as_tensor(z_path)
        self.z_ref = torch.as_tensor(z_ref)
        if self.z_path.shape != self.src.shape:
            raise ValueError(
                f"Stack (kind {kind!r}): z_path has shape {tuple(self.z_path.shape)}, "
                f"expected {tuple(self.src.shape)} (one value per edge of the kind)"
            )
        self.rho_key = rho_key
        self.g = float(g)

    @classmethod
    def from_network(
        cls,
        net,
        kind: str,
        *,
        z_path: str = "z_path",
        z_ref: str = "z_ref",
        rho_key: str = "rho",
        g: float = 9.80665,
    ) -> Stack:
        src, tgt = net.endpoints(kind)
        return cls(
            kind,
            src=src,
            tgt=tgt,
            z_path=net.edge_attr(z_path, kind),
            z_ref=net.node_attr(z_ref, default=0.0),
            rho_key=rho_key,
            g=g,
        )

    def __call__(self, drivers: Mapping[str, torch.Tensor]) -> torch.Tensor:
        try:
            rho = drivers[self.rho_key]
        except KeyError as exc:
            raise KeyError(
                f"Stack drive (kind {self.kind!r}): driver {self.rho_key!r} not found"
            ) from exc
        z_path = self.z_path.to(rho.dtype)
        z_ref = self.z_ref.to(rho.dtype)
        head_src = rho[..., self.src] * (z_ref[self.src] - z_path)
        head_tgt = rho[..., self.tgt] * (z_ref[self.tgt] - z_path)
        return self.g * (head_src - head_tgt)
