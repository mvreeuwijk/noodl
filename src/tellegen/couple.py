"""Orchestration-level coupling between two independently-built `Model`s (framework spec
section 4.3, design spec `docs/superpowers/specs/2026-09-19-milestone-5-coupling-design.md`).

`union` never merges networks or rebuilds layers/closures: it exchanges named driver/state
values between two ordinary `Model.step` calls each outer step, iterated to a fixed point
when a link is two-way. This module must never import `tellegen.apps.*`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from tellegen.model import Drivers, Model, State
from tellegen.topology import Network

Tensor = torch.Tensor

CONCENTRATION_TO_MASS_FRACTION = "concentration_to_mass_fraction"
MASS_FRACTION_TO_CONCENTRATION = "mass_fraction_to_concentration"
STREET_RAD_TO_CONTAM_DEG = "street_rad_to_contam_deg"
CONTAM_DEG_TO_STREET_RAD = "contam_deg_to_street_rad"

_CONVERSIONS: dict[str, Callable[[Tensor, Mapping[str, Tensor]], Tensor]] = {
    CONCENTRATION_TO_MASS_FRACTION: lambda value, drivers: value / drivers["rho_amb"],
    MASS_FRACTION_TO_CONCENTRATION: lambda value, drivers: value * drivers["rho_amb"],
    # CONTAM's Wd: degrees clockwise from north, the direction the wind blows FROM. The
    # street app's theta_w: radians counter-clockwise from east, the direction it blows
    # TOWARD (design spec A3). West wind: Wd=270 <-> theta=0.
    STREET_RAD_TO_CONTAM_DEG: lambda value, drivers: torch.remainder(
        270.0 - torch.rad2deg(value), 360.0),
    CONTAM_DEG_TO_STREET_RAD: lambda value, drivers: torch.remainder(
        torch.deg2rad(270.0 - value), 2.0 * math.pi),
}


def _reduced(x: Tensor, n_last: int) -> Tensor:
    """A single-species tensor in REDUCED layout `(..., n_last)`. `TransportLayer` accepts a
    single-species state either reduced or stacked `(..., n_last, 1)` (the CONTAM reader's
    `x0` and `x_boundary` are stacked, the street app's are reduced -- design spec A5); the
    glue works in the reduced form. A batched reduced `(1, 1)` and a stacked `(1, 1)` hold
    the same single number, so the ambiguity when `n_last == 1` is harmless."""
    if x.dim() >= 2 and x.shape[-1] == 1 and x.shape[-2] == n_last:
        return x.squeeze(-1)
    if x.shape[-1] != n_last:
        raise ValueError(
            f"couple: expected a tensor with {n_last} entries on its node axis, got shape "
            f"{tuple(x.shape)}"
        )
    return x


def _write_at(target: Tensor, n_last: int, position: int, value: Tensor) -> Tensor:
    """`target` with `value` written at `position` on its node axis, in the caller's own
    layout (reduced or stacked). Never writes into `target` itself."""
    reduced = _reduced(target, n_last).clone()
    reduced[..., position] = value
    return reduced.reshape(target.shape)


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
    """One shared-node value relationship (design spec section 3 points 1-2, A2, A5).

    `from_key` is a transport layer's state key ("<layer>.x") and `from_index` a position on
    that layer's ACTIVE INTERIOR axis; `to_key` is a transport layer's boundary driver
    ("<layer>.x_boundary") and `to_index` a position on its BOUNDARY axis. Two-way: the
    to-layer's net boundary INFLOW at `to_index` is added into `sources_key` (default
    "<from layer>.sources", FULL node order) at the from-layer's interior node.
    """

    from_model: str
    from_key: str
    from_index: int
    to_model: str
    to_key: str
    to_index: int = 0
    convert: str | None = None
    two_way: bool = False
    convert_back: str | None = None
    sources_key: str | None = None

    @property
    def from_layer(self) -> str:
        return self.from_key.split(".")[0]

    @property
    def to_layer(self) -> str:
        return self.to_key.split(".")[0]


@dataclass(frozen=True)
class DriverAlias:
    """One driver value shared across models (design spec section 3 point 3, A3): `source`
    is authoritative; every target is overwritten from it each pass, through its own
    registered conversion (or none)."""

    source: tuple[str, str]
    targets: tuple[tuple[str, str, str | None], ...]


class CoupledModel:
    """Returned by `union`. `.step` takes and returns `{model_tag: State}` /
    `{model_tag: Drivers}` -- each model keeps its own dicts; nothing is merged."""

    def __init__(
        self,
        models: Mapping[str, Model],
        links: Sequence[ValueLink],
        aliases: Sequence[DriverAlias],
        substeps: Mapping[str, int],
        *,
        relaxation: float = 0.5,
        iterate_rtol: float = 1e-8,
        iterate_atol: float = 0.0,
        iterate_max: int = 20,
    ) -> None:
        self.models = dict(models)
        self.links = list(links)
        self.aliases = list(aliases)
        self.substeps = {tag: int(k) for tag, k in substeps.items()}
        self.relaxation = float(relaxation)
        self.iterate_rtol, self.iterate_atol = float(iterate_rtol), float(iterate_atol)
        self.iterate_max = int(iterate_max)
        for tag, k in self.substeps.items():
            if tag not in self.models or k < 1:
                raise ValueError(
                    f"CoupledModel: substeps[{tag!r}] = {k!r} (unknown model or < 1)"
                )
        for link in self.links:
            for name in (link.convert, link.convert_back):
                if name is not None and name not in _CONVERSIONS:
                    raise KeyError(
                        f"CoupledModel: link {link.from_model}:{link.from_key} -> "
                        f"{link.to_model}:{link.to_key} names conversion {name!r}; registered "
                        f"conversions are {sorted(_CONVERSIONS)}"
                    )
            self._layer(link.from_model, link.from_layer)
            self._layer(link.to_model, link.to_layer)
        for alias in self.aliases:
            for _model, _key, name in alias.targets:
                if name is not None and name not in _CONVERSIONS:
                    raise KeyError(
                        f"CoupledModel: alias of {alias.source} names conversion {name!r}; "
                        f"registered conversions are {sorted(_CONVERSIONS)}"
                    )
        self._two_way = [link for link in self.links if link.two_way]
        self._one_way = [link for link in self.links if not link.two_way]
        if self._two_way and self.iterate_max < 2:
            raise ValueError(
                f"CoupledModel: a two-way link needs iterate_max >= 2, got {iterate_max!r}; "
                f"one pass has no predecessor to judge convergence against"
            )

    def _layer(self, tag: str, name: str):
        try:
            return self.models[tag].transport[name]
        except KeyError as exc:
            raise KeyError(
                f"CoupledModel: model {tag!r} has no transport layer {name!r} (has "
                f"{sorted(self.models[tag].transport) if tag in self.models else 'no such model'})"
            ) from exc

    # ------------------------------------------------------------------- glue
    def _apply_aliases(self, drivers: dict[str, Drivers]) -> None:
        for alias in self.aliases:
            src_model, src_key = alias.source
            value = drivers[src_model][src_key]
            for model, key, name in alias.targets:
                drivers[model][key] = apply_conversion(name, value, drivers[model])

    def _forward_value(
        self, link: ValueLink, latest: Mapping[str, State], drivers: dict[str, Drivers],
    ) -> Tensor:
        from_layer = self._layer(link.from_model, link.from_layer)
        value = _reduced(latest[link.from_model][link.from_key], from_layer.n_i)[
            ..., link.from_index
        ]
        return apply_conversion(link.convert, value, drivers[link.to_model])

    def _write_forward(
        self, link: ValueLink, value: Tensor, drivers: dict[str, Drivers],
    ) -> Tensor:
        to_layer = self._layer(link.to_model, link.to_layer)
        boundary = _write_at(
            drivers[link.to_model][link.to_key], to_layer.n_b, link.to_index, value
        )
        drivers[link.to_model][link.to_key] = boundary
        return boundary

    def _feed_back(
        self, link: ValueLink, latest: Mapping[str, State], boundary: Tensor,
        drivers: dict[str, Drivers],
    ) -> None:
        """Add the to-layer's net boundary inflow at `to_index` into the from-layer's sources
        (FULL node order, ADDED to whatever the caller supplied -- never overwritten)."""
        to_model = self.models[link.to_model]
        to_layer = self._layer(link.to_model, link.to_layer)
        q = to_model.current_flows(link.to_layer, latest[link.to_model], drivers[link.to_model])
        inflow = transport_boundary_inflow(
            to_model.net, q, to_layer.flow_kinds,
            _reduced(latest[link.to_model][f"{link.to_layer}.x"], to_layer.n_i),
            _reduced(boundary, to_layer.n_b),
            to_layer.interior_idx, to_layer.boundary_idx, node_position=link.to_index,
        )
        inflow = apply_conversion(link.convert_back, inflow, drivers[link.to_model])
        from_layer = self._layer(link.from_model, link.from_layer)
        from_net = self.models[link.from_model].net
        key = link.sources_key or f"{link.from_layer}.sources"
        existing = drivers[link.from_model].get(key)
        if existing is None:
            existing = torch.zeros(from_net.n, dtype=inflow.dtype, device=inflow.device)
        batch = torch.broadcast_shapes(existing.shape[:-1], inflow.shape)
        updated = existing.expand(*batch, existing.shape[-1]).clone()
        node = int(from_layer.interior_idx[link.from_index])  # interior_idx is FULL node order
        updated[..., node] = updated[..., node] + inflow
        drivers[link.from_model][key] = updated

    # ------------------------------------------------------------------- step
    def step(
        self, state: Mapping[str, State], drivers: Mapping[str, Drivers], dt: float,
        *, diagnostics: dict | None = None,
    ) -> dict[str, State]:
        start = {tag: dict(s) for tag, s in state.items()}
        drivers = {tag: dict(d) for tag, d in drivers.items()}
        self._apply_aliases(drivers)
        if not self._two_way:
            pass_drivers = {tag: dict(d) for tag, d in drivers.items()}
            for link in self._one_way:
                self._write_forward(
                    link, self._forward_value(link, start, pass_drivers), pass_drivers
                )
            new = self._step_all(start, pass_drivers, dt)
            if diagnostics is not None:
                diagnostics.update({"passes": 1, "converged": True, "max_change": 0.0})
            return new
        return self._iterate(start, drivers, dt, diagnostics)

    def _step_all(
        self, start: Mapping[str, State], drivers: Mapping[str, Drivers], dt: float,
    ) -> dict[str, State]:
        """Every model stepped ONCE from `start` over `dt` -- in `k` sub-steps of `dt/k` for a
        model named in `substeps`, its glue-derived drivers held constant across them
        (design spec section 3 point 4)."""
        new: dict[str, State] = {}
        for tag, model in self.models.items():
            k = self.substeps.get(tag, 1)
            s = start[tag]
            for _ in range(k):
                s = model.step(s, drivers[tag], dt / k)
            new[tag] = s
        return new

    def _iterate(self, start, drivers, dt, diagnostics) -> dict[str, State]:
        """Successive substitution on every two-way link's FORWARD value, damped by
        `relaxation` (0.5 mirrors `Model._iterate`, `model.py:540-611`). EVERY pass steps
        every model from `start` (the N1 rule, design spec A1); only the glue values -- read
        from `latest`, the previous pass's OUTPUT (the start state on pass 1) -- carry over.
        Converged when `|f_k - f_{k-1}| <= atol + rtol |f_k|` for every link (A7)."""
        latest: Mapping[str, State] = start
        prev: dict[int, Tensor] | None = None
        passes, worst, converged = 0, float("inf"), False
        new: dict[str, State] = dict(start)
        while passes < self.iterate_max:
            passes += 1
            pass_drivers = {tag: dict(d) for tag, d in drivers.items()}
            for link in self._one_way:
                self._write_forward(
                    link, self._forward_value(link, latest, pass_drivers), pass_drivers
                )
            now: dict[int, Tensor] = {}
            for i, link in enumerate(self._two_way):
                value = self._forward_value(link, latest, pass_drivers)
                if prev is not None:
                    value = prev[i] + self.relaxation * (value - prev[i])
                now[i] = value
                boundary = self._write_forward(link, value, pass_drivers)
                self._feed_back(link, latest, boundary, pass_drivers)
            new = self._step_all(start, pass_drivers, dt)
            if prev is not None:
                worst, ok = 0.0, True
                for i, f in now.items():
                    delta = (f - prev[i]).abs()
                    worst = max(worst, float(delta.max()))
                    ok = ok and bool(
                        (delta <= self.iterate_atol + self.iterate_rtol * f.abs()).all()
                    )
                if ok:
                    converged = True
                    break
            prev = now
            latest = new
        if not converged:
            raise RuntimeError(
                f"CoupledModel: two-way coupling did not converge within {self.iterate_max} "
                f"passes; largest change in the shared value(s) was {worst}, tolerance "
                f"atol={self.iterate_atol} rtol={self.iterate_rtol}"
            )
        if diagnostics is not None:
            diagnostics.update({"passes": passes, "converged": True, "max_change": worst})
        return new


def union(
    models: Mapping[str, tuple[Model, State, Drivers]],
    shared: Sequence[ValueLink | DriverAlias],
    *,
    substeps: Mapping[str, int] | None = None,
    relaxation: float = 0.5,
    iterate_rtol: float = 1e-8,
    iterate_atol: float = 0.0,
    iterate_max: int = 20,
) -> tuple[CoupledModel, dict[str, State], dict[str, Drivers]]:
    """Couple `models` by exchanging the driver/state values `shared` names, WITHOUT merging
    any model's `Network`, layers, or closures (design spec section 3). Never modifies the
    `Model`/`State`/`Drivers` objects passed in -- returns fresh dict copies."""
    model_map = {tag: m for tag, (m, _s, _d) in models.items()}
    state_map = {tag: dict(s) for tag, (_m, s, _d) in models.items()}
    drivers_map = {tag: dict(d) for tag, (_m, _s, d) in models.items()}
    links = [item for item in shared if isinstance(item, ValueLink)]
    aliases = [item for item in shared if isinstance(item, DriverAlias)]
    city = CoupledModel(
        model_map, links, aliases, substeps or {}, relaxation=relaxation,
        iterate_rtol=iterate_rtol, iterate_atol=iterate_atol, iterate_max=iterate_max,
    )
    return city, state_map, drivers_map
