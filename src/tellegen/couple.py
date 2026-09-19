"""Orchestration-level coupling between two independently-built `Model`s (framework spec
section 4.3, design spec `docs/superpowers/specs/2026-09-19-milestone-5-coupling-design.md`).

`union` never merges networks or rebuilds layers/closures: it exchanges named driver/state
values between two ordinary `Model.step` calls each outer step, iterated to a fixed point
when a link is two-way. This module must never import `tellegen.apps.*`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from tellegen.model import Drivers, Model, State
from tellegen.topology import Network

Tensor = torch.Tensor

CONCENTRATION_TO_MASS_FRACTION = "concentration_to_mass_fraction"
MASS_FRACTION_TO_CONCENTRATION = "mass_fraction_to_concentration"

_CONVERSIONS: dict[str, Callable[[Tensor, Mapping[str, Tensor]], Tensor]] = {
    CONCENTRATION_TO_MASS_FRACTION: lambda value, drivers: value / drivers["rho_amb"],
    MASS_FRACTION_TO_CONCENTRATION: lambda value, drivers: value * drivers["rho_amb"],
}


def apply_conversion(name: str | None, value: Tensor, drivers: Mapping[str, Tensor]) -> Tensor:
    """`value` unchanged if `name` is None; the registered conversion otherwise.

    Raises `KeyError` naming `name` and the known conversions if it is not registered --
    an unregistered conversion must never be silently treated as identity (design spec
    section 3 point 1, section 8).
    """
    if name is None:
        return value
    try:
        convert = _CONVERSIONS[name]
    except KeyError as exc:
        raise KeyError(
            f"couple: unknown unit conversion {name!r}; registered conversions are "
            f"{sorted(_CONVERSIONS)}"
        ) from exc
    return convert(value, drivers)


def transport_boundary_inflow(
    net: Network,
    q: Tensor,
    flow_kinds: Sequence[str],
    x_interior: Tensor,
    x_boundary: Tensor,
    interior_idx: Tensor,
    boundary_idx: Tensor,
    node_position: int,
) -> Tensor:
    """Net mass INFLOW, `(...,)`, at the boundary node `boundary_idx[node_position]`, in a
    transport layer's own units, for a SINGLE-SPECIES layer (`x_interior`/`x_boundary` are
    `(..., n_i)`/`(..., n_b)`, never `(..., n_i, K)` -- milestone 5's global constraint).

    `TransportLayer` never exposes this: its own `rate()` has no row for boundary nodes at
    all (design spec section 3, `Model.ports()` reports boundary flows only for potential
    layers). Built here from `net.accumulate`/`net.upwind` alone -- `net.accumulate(w, kind)`
    is the scatter-add form of `incidence(kind) @ w` (net OUTFLOW: +at source, -at target).
    """
    kind = flow_kinds[0] if len(flow_kinds) == 1 else None
    if kind is None:
        raise NotImplementedError(
            f"transport_boundary_inflow: multiple flow_kinds {flow_kinds!r} not supported "
            f"(milestone 5 scope is single-flow-kind transport layers)"
        )
    batch_shape = torch.broadcast_shapes(x_interior.shape[:-1], x_boundary.shape[:-1], q.shape[:-1])
    full = torch.zeros(*batch_shape, net.n, dtype=x_interior.dtype, device=x_interior.device)
    full[..., interior_idx] = x_interior.expand(*batch_shape, interior_idx.numel())
    full[..., boundary_idx] = x_boundary.expand(*batch_shape, boundary_idx.numel())
    upwind_selector = net.upwind(q, kind)          # (..., b_kind, n)
    upstream_value = torch.einsum("...bn,...n->...b", upwind_selector, full)
    mass_flux = q * upstream_value                  # (..., b_kind)
    net_outflow = net.accumulate(mass_flux, kind)    # (..., n)
    node_idx = int(boundary_idx[node_position])
    return -net_outflow[..., node_idx]


@dataclass(frozen=True)
class ValueLink:
    """One shared-node value relationship (design spec section 3 points 1, 2, 6)."""

    from_model: str
    from_key: str
    from_index: int
    to_model: str
    to_key: str
    to_index: int = 0
    convert: str | None = None
    convert_back: str | None = None
    two_way: bool = False
    sources_key: str | None = None
    flow_layer: str | None = None
    flow_kinds: tuple[str, ...] = ()
    interior_key: str | None = None
    interior_idx_of: Callable[[Model], Tensor] | None = None
    boundary_idx_of: Callable[[Model], Tensor] | None = None

    def __post_init__(self) -> None:
        if self.two_way:
            missing = [
                name for name in ("sources_key", "flow_layer", "interior_key",
                                  "interior_idx_of", "boundary_idx_of")
                if getattr(self, name) is None or getattr(self, name) == ()
            ]
            if not self.flow_kinds:
                missing.append("flow_kinds")
            if missing:
                raise ValueError(
                    f"ValueLink: two_way=True requires {missing}, all None/empty"
                )


@dataclass(frozen=True)
class DriverAlias:
    """One driver key aliased to the SAME value across models (design spec section 3 point 3)."""

    keys: tuple[tuple[str, str], ...]


def _apply_links_forward(
    links: Sequence[ValueLink], state: Mapping[str, State], drivers: dict[str, Drivers],
) -> None:
    """Write each link's `to_key`/`to_index` driver entry from its `from_key`/`from_index`
    state entry, in place on `drivers[to_model]`. Never touches a `to_model`'s state keys."""
    for link in links:
        source_state = state[link.from_model][link.from_key]
        value = source_state[..., link.from_index]
        value = apply_conversion(link.convert, value, drivers[link.to_model])
        target = drivers[link.to_model][link.to_key].clone()
        target[..., link.to_index] = value
        drivers[link.to_model][link.to_key] = target


def _apply_aliases(aliases: Sequence[DriverAlias], drivers: dict[str, Drivers]) -> None:
    """Every alias's FIRST (model, key) pair is authoritative; every other pair in the same
    alias is overwritten to match it, each pass."""
    for alias in aliases:
        (first_model, first_key), *rest = alias.keys
        value = drivers[first_model][first_key]
        for model_tag, key in rest:
            drivers[model_tag][key] = value


class CoupledModel:
    """Returned by `union`. `.step`/`.steady` take and return `{model_tag: State}`/
    `{model_tag: Drivers}` dicts, never a merged one -- each model's own keys never collide
    with the other's in this milestone's headline pairing (design spec section 4)."""

    def __init__(
        self,
        models: Mapping[str, Model],
        links: Sequence[ValueLink],
        aliases: Sequence[DriverAlias],
        substeps: Mapping[str, int],
        *,
        coupling: str = "iterate",
        iterate_tol: float = 1e-9,
        iterate_max: int = 20,
    ) -> None:
        self.models = dict(models)
        self.links = list(links)
        self.aliases = list(aliases)
        self.substeps = dict(substeps)
        self.coupling = coupling
        self.iterate_tol = iterate_tol
        self.iterate_max = iterate_max

    def step(
        self, state: Mapping[str, State], drivers: Mapping[str, Drivers], dt: float,
    ) -> dict[str, State]:
        state = {tag: dict(s) for tag, s in state.items()}
        drivers = {tag: dict(d) for tag, d in drivers.items()}
        two_way = [link for link in self.links if link.two_way]
        if not two_way:
            _apply_aliases(self.aliases, drivers)
            _apply_links_forward(self.links, state, drivers)
            return self._step_all(state, drivers, dt)
        return self._iterate(state, drivers, dt)

    def _step_all(
        self, state: dict[str, State], drivers: dict[str, Drivers], dt: float,
    ) -> dict[str, State]:
        new_state: dict[str, State] = {}
        for tag, model in self.models.items():
            k = self.substeps.get(tag, 1)
            s = state[tag]
            for _ in range(k):
                s = model.step(s, drivers[tag], dt / k)
            new_state[tag] = s
        return new_state


def union(
    models: Mapping[str, tuple[Model, State, Drivers]],
    shared: Sequence[ValueLink | DriverAlias],
    *,
    substeps: Mapping[str, int] | None = None,
    coupling: str = "iterate",
    iterate_tol: float = 1e-9,
    iterate_max: int = 20,
) -> tuple[CoupledModel, dict[str, State], dict[str, Drivers]]:
    """Couple `models` by exchanging the driver/state values `shared` names, WITHOUT merging
    any model's `Network`, layers, or closures (design spec section 3). Never modifies the
    `Model`/`State`/`Drivers` objects passed in -- returns fresh copies."""
    model_map = {tag: m for tag, (m, _s, _d) in models.items()}
    state_map = {tag: dict(s) for tag, (_m, s, _d) in models.items()}
    drivers_map = {tag: dict(d) for tag, (_m, _s, d) in models.items()}
    links = [item for item in shared if isinstance(item, ValueLink)]
    aliases = [item for item in shared if isinstance(item, DriverAlias)]
    city = CoupledModel(
        model_map, links, aliases, substeps or {},
        coupling=coupling, iterate_tol=iterate_tol, iterate_max=iterate_max,
    )
    return city, state_map, drivers_map
