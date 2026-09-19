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
    if _is_stacked(x, n_last):
        return x.squeeze(-1)
    if x.shape[-1] != n_last:
        raise ValueError(
            f"couple: expected a tensor with {n_last} entries on its node axis, got shape "
            f"{tuple(x.shape)}"
        )
    return x


def _write_at(
    target: Tensor, n_last: int, position: int, value: Tensor, *, add: bool = False,
) -> Tensor:
    """`target` with `value` written (`add=True`: ADDED) at `position` on its node axis, in
    the caller's own LAYOUT (reduced `(..., n_last)` or stacked `(..., n_last, 1)`) -- but not
    necessarily its own SHAPE: a batched `value` written into an unbatched `target`
    broadcasts the target up to the value's batch. That is the ensemble case: the street
    model runs a batch of `B` forcings while the CONTAM reader's `x_boundary` stays `(1, 1)`.
    Never writes into `target` itself.

    `add=True` is what a two-way link's feedback term needs: the caller's own entry at
    `position` (another source of pollutant at that node) must survive, with the coupling's
    contribution added to it, never overwritten.
    """
    reduced = _reduced(target, n_last)
    batch = torch.broadcast_shapes(reduced.shape[:-1], value.shape)
    reduced = reduced.expand(*batch, n_last).clone()
    reduced[..., position] = reduced[..., position] + value if add else value
    stacked = _is_stacked(target, n_last)
    return reduced.reshape(*batch, n_last, 1) if stacked else reduced.reshape(*batch, n_last)


def _is_stacked(x: Tensor, n_last: int) -> bool:
    """True when a single-species tensor is in STACKED layout `(..., n_last, 1)` rather than
    reduced `(..., n_last)` -- the one rule `_reduced` and `_write_at` both decide by."""
    return x.dim() >= 2 and x.shape[-1] == 1 and x.shape[-2] == n_last


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

    WHICH state a ONE-WAY link reads depends on the company it keeps, deliberately. In a
    union with no two-way link there is one pass, and the forward value is read from the
    STEP-START state -- the explicit-coupling ping-pong the design spec asks for. In a union
    that also has a two-way link, the iteration's passes read every link (one-way included)
    from the previous pass's OUTPUT, so a one-way link there carries an END-of-step value,
    consistent with the two-way glue beside it. One-way links are NOT part of the
    convergence criterion: only two-way forward values are measured, because only they close
    a loop that can fail to converge.
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
    is authoritative; every target is overwritten from it through that target's own
    registered conversion (or none).

    Applied ONCE per `step`, before any pass -- not per pass. An alias relates two DRIVERS,
    and drivers do not change between passes; only the glue values derived from the models'
    own states do. Applying it per pass would be the same write repeated.
    """

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
            # The suffixes are not decoration: `to_key` is written with `_write_at` on the
            # to-layer's BOUNDARY axis and `sources_key` with an ADD on the from-model's FULL
            # node axis. A key naming any other driver would be written with the wrong length
            # and the wrong semantics -- refuse it here rather than at the first shape clash.
            if not link.to_key.endswith(".x_boundary"):
                raise ValueError(
                    f"CoupledModel: link to_key {link.to_key!r} must name a transport layer's "
                    f"boundary driver, '<layer>.x_boundary'; it is written on "
                    f"{link.to_model!r}'s boundary axis at index {link.to_index}"
                )
            if link.sources_key is not None and not link.sources_key.endswith(".sources"):
                raise ValueError(
                    f"CoupledModel: link sources_key {link.sources_key!r} must name a "
                    f"transport layer's source term, '<layer>.sources'; the feedback flux is "
                    f"ADDED into it on {link.from_model!r}'s FULL node axis"
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
        (FULL node order, ADDED to whatever the caller supplied -- never overwritten).

        The sources tensor goes through `_write_at` like every other glue write, so a STACKED
        single-species `(n, 1)` sources tensor is reduced before the node index is applied and
        restored afterwards; indexing it directly would silently address the species axis.
        """
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
        node = int(from_layer.interior_idx[link.from_index])  # interior_idx is FULL node order
        drivers[link.from_model][key] = _write_at(existing, from_net.n, node, inflow, add=True)

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
                # One pass, nothing iterated: `converged` is true for every instance by
                # construction, and `max_change` is EMPTY rather than 0.0 -- no change was
                # measured, and reporting a number nothing measured would be a claim. Same
                # types as the iterated path (a 0-d bool tensor, a per-link dict).
                diagnostics.update({
                    "passes": 1,
                    "converged": torch.ones((), dtype=torch.bool),
                    "max_change": {},
                })
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

    @staticmethod
    def _link_key(link: ValueLink) -> str:
        """The name a link answers to in diagnostics and in the non-convergence message."""
        return (
            f"{link.from_model}:{link.from_key}[{link.from_index}]->"
            f"{link.to_model}:{link.to_key}"
        )

    def _iterate(self, start, drivers, dt, diagnostics) -> dict[str, State]:
        """Successive substitution on every two-way link's FORWARD value, damped by
        `relaxation` (0.5 mirrors `Model._iterate`, `model.py:540-611`). EVERY pass steps
        every model from `start` (the N1 rule, design spec A1); only the glue values -- read
        from `latest`, the previous pass's OUTPUT (the start state on pass 1) -- carry over.

        Convergence is judged on the UNRELAXED residual of the fixed-point map,
        `|f(latest_k-1) - f_k-1|` (A7: `<= atol + rtol |f|`), not on the relaxed increment
        actually applied: the relaxed step is `relaxation` times the residual, so measuring
        it would make the effective tolerance scale with `relaxation` and let a heavily
        damped run "converge" while still far from the fixed point. It is judged PER
        INSTANCE, on detached copies inside `torch.no_grad()`; the passes themselves stay on
        the autograd graph.

        The returned state is the last pass's own output, whose glue values came from pass
        k-1 -- so it is a fixed point of "one step of each model from `start`" only to within
        the tolerance, which is exactly what the criterion certifies. The fixed point is
        therefore differentiated by UNROLLING: every pass stays on the graph and memory grows
        with the pass count. An implicit-function treatment (one adjoint solve at the
        converged state) is a follow-up, as it is for `Model._iterate`.
        """
        latest: Mapping[str, State] = start
        prev: dict[int, Tensor] | None = None
        change: dict[str, Tensor] = {}
        # A placeholder the second pass always replaces: a two-way link is refused at
        # construction unless `iterate_max >= 2`, so the loop cannot end without a real
        # per-instance verdict of the right shape, dtype and device.
        converged: Tensor = torch.zeros((), dtype=torch.bool)
        passes = 0
        new: dict[str, State] = dict(start)
        while passes < self.iterate_max:
            passes += 1
            pass_drivers = {tag: dict(d) for tag, d in drivers.items()}
            for link in self._one_way:
                self._write_forward(
                    link, self._forward_value(link, latest, pass_drivers), pass_drivers
                )
            now: dict[int, Tensor] = {}
            raw: dict[int, Tensor] = {}
            for i, link in enumerate(self._two_way):
                raw[i] = self._forward_value(link, latest, pass_drivers)
                value = raw[i] if prev is None else prev[i] + self.relaxation * (raw[i] - prev[i])
                now[i] = value
                boundary = self._write_forward(link, value, pass_drivers)
                self._feed_back(link, latest, boundary, pass_drivers)
            new = self._step_all(start, pass_drivers, dt)
            if prev is not None:
                with torch.no_grad():
                    ok: Tensor | None = None
                    for i, link in enumerate(self._two_way):
                        # A forward value is ONE number per instance (one node, one species),
                        # so its own shape IS the batch shape and nothing is reduced away;
                        # `.amax(-1)` here would collapse the instance axis itself.
                        d = (raw[i] - prev[i]).abs()
                        change[self._link_key(link)] = d
                        this = d <= self.iterate_atol + self.iterate_rtol * raw[i].abs()
                        ok = this if ok is None else (ok & this)
                    converged = ok
                if bool(converged.all()):
                    break
            prev = now
            latest = new
        if not bool(converged.all()):
            failing = (~converged).nonzero().flatten().tolist() if converged.dim() else "all"
            worst = {name: float(c.max()) for name, c in change.items()}
            raise RuntimeError(
                f"CoupledModel: two-way coupling did not converge within {self.iterate_max} "
                f"passes for instances {failing}; largest change per link {worst}, tolerance "
                f"atol={self.iterate_atol} rtol={self.iterate_rtol}"
            )
        if diagnostics is not None:
            diagnostics.update(
                {"passes": passes, "converged": converged, "max_change": change}
            )
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
