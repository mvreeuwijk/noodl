"""Orchestration-level coupling between two independently-built `Model`s (framework spec
section 4.3, design spec `docs/superpowers/specs/2026-09-19-milestone-5-coupling-design.md`).

`union` never merges networks or rebuilds layers/closures: it exchanges named driver/state
values between two ordinary `Model.step` calls each outer step, iterated to a fixed point
when a link is two-way. This module must never import `noodl.apps.*`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

from noodl.model import Drivers, Model, State
from noodl.solvers.fixed_point import differentiate_fixed_point
from noodl.topology import Network

Tensor = torch.Tensor

CONCENTRATION_TO_MASS_FRACTION = "concentration_to_mass_fraction"
MASS_FRACTION_TO_CONCENTRATION = "mass_fraction_to_concentration"
STREET_RAD_TO_CONTAM_DEG = "street_rad_to_contam_deg"
CONTAM_DEG_TO_STREET_RAD = "contam_deg_to_street_rad"


class Conversion(NamedTuple):
    """One registered conversion and the UNITS it maps between, `(from_unit, to_unit, fn)`.

    The units are not decoration: `CoupledModel.__init__` checks a `ValueLink`'s forward
    `convert` against the two transport layers' own `unit` metadata, so a link that names the
    wrong conversion (or none at all between mismatched units) is refused at construction
    rather than silently off by a density. `fn(value, drivers)` is the conversion itself,
    with `drivers` the TO model's own driver mapping.
    """

    from_unit: str
    to_unit: str
    fn: Callable[[Tensor, Mapping[str, Tensor]], Tensor]


_CONVERSIONS: dict[str, Conversion] = {
    CONCENTRATION_TO_MASS_FRACTION: Conversion(
        "kg/m3", "kg/kg", lambda value, drivers: value / drivers["rho_amb"]),
    MASS_FRACTION_TO_CONCENTRATION: Conversion(
        "kg/kg", "kg/m3", lambda value, drivers: value * drivers["rho_amb"]),
    # CONTAM's Wd: degrees clockwise from north, the direction the wind blows FROM. The
    # street app's theta_w: radians counter-clockwise from east, the direction it blows
    # TOWARD (design spec A3). West wind: Wd=270 <-> theta=0. These two are used by
    # `DriverAlias` targets, and a DRIVER carries no unit metadata to check against -- their
    # "rad"/"deg" are recorded here for the reader, not enforced anywhere.
    STREET_RAD_TO_CONTAM_DEG: Conversion(
        "rad", "deg",
        lambda value, drivers: torch.remainder(270.0 - torch.rad2deg(value), 360.0)),
    CONTAM_DEG_TO_STREET_RAD: Conversion(
        "deg", "rad",
        lambda value, drivers: torch.remainder(torch.deg2rad(270.0 - value), 2.0 * math.pi)),
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
        conversion = _CONVERSIONS[name]
    except KeyError as exc:
        raise KeyError(
            f"couple: unknown unit conversion {name!r}; registered conversions are "
            f"{sorted(_CONVERSIONS)}"
        ) from exc
    return conversion.fn(value, drivers)


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
    layers). Built here with `net.endpoints` + gather + `net.accumulate` to avoid assembling
    a dense topology selector.
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
    src, tgt = net.endpoints(kind)
    upstream = torch.where(q >= 0, src, tgt)                       # (..., b_kind)
    mass_flux = q * torch.gather(full, -1, upstream.expand(*batch_shape, upstream.shape[-1]))
    net_outflow = net.accumulate(mass_flux, kind)
    node_idx = int(boundary_idx[node_position])
    return -net_outflow[..., node_idx]


@dataclass(frozen=True)
class ValueLink:
    """One shared-node value relationship (design spec section 3 points 1-2, A2, A5).

    `from_key` is a transport layer's state key ("<layer>.x") and `from_index` a position on
    that layer's ACTIVE INTERIOR axis; `to_key` is a transport layer's boundary driver
    ("<layer>.x_boundary") and `to_index` a position on its BOUNDARY axis. Two-way: the
    to-model (the RECIPIENT) is stepped first, from the step-start state, over the whole
    outer step; the amount that its own step integrated across `to_index` (`Model.step`'s
    `boundary_transfer`, summed over the recipient's sub-steps) is then divided by `dt` and
    ADDED, as a source RATE, into `sources_key` (default "<from layer>.sources", FULL node
    order) at the from-model's (the DONOR's) interior node, for the donor's own whole outer
    step. Conservation holds on every returned pass by construction: the donor receives
    exactly what the recipient's own scheme integrated, never an endpoint flux frozen from a
    stale state.

    WHICH state a ONE-WAY link reads depends on the company it keeps, deliberately. In a
    union with no two-way link there is one pass, and the forward value is read from the
    STEP-START state -- the explicit-coupling ping-pong the design spec asks for. In a union
    that also has a two-way link, the iteration's passes read every link (one-way included)
    from the previous pass's OUTPUT, so a one-way link there carries an END-of-step value,
    consistent with the two-way glue beside it. One-way links are NOT part of the
    convergence criterion: only two-way forward values are measured, because only they close
    a loop that can fail to converge.

    `sources_key` is VALIDATED on every link (it must name a "<layer>.sources" driver) but is
    USED only on a two-way one: only a two-way link feeds a flux back, so on a one-way link
    the key names a driver nothing ever writes. It is still checked, so that a link whose
    `two_way=True` is later turned on does not fail for the first time mid-run.

    UNITS. The forward `convert` is checked against the two transport layers' own `unit`
    metadata at construction: `convert=None` requires equal units, and a named conversion
    must map the from-layer's unit to the to-layer's. `convert_back` is NOT unit-checked: it
    acts on a FLUX, not on the state value, and a flux's units are the from-layer's own
    source units (kg/s of pollutant on both sides of this milestone's join, which is exactly
    why the demo leaves it `None` -- design spec A2); the layers' `unit` strings, which
    describe the STATE, say nothing about it.

    OWNERSHIP. Each boundary entry `(to_model, to_key, to_index)` has exactly one writing
    link. The recipient-first schedule is conservative because every integrated transfer has
    one donor to go to; two donors of one entry would each receive the whole transfer. Shared
    ownership needs an allocation rule and is refused rather than guessed.
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
    `{model_tag: Drivers}` -- each model keeps its own dicts; nothing is merged.

    `iterate_max=50` is a floor, not a measurement pinned here: the recipient-first schedule
    changed the per-pass cost and the pass counts a coupling needs to reach a given
    tolerance, so see `docs/applications/coupling.md` for current numbers rather than this
    docstring. The former default of 20 was below the milestone's own headline case, so
    every call site had to override it -- which hid, rather than fixed, the fact that the
    default could not run the demo.

    `iterate_atol=0.0` is a pure relative criterion: a shared value that is legitimately ZERO
    (no emission at the coupled node, say) can never satisfy `|d| <= rtol * |f|` unless `d` is
    exactly zero, so such a coupling needs a positive `iterate_atol` -- a floor in the shared
    value's own units -- to be judged converged at all.

    The two-way fixed point is differentiated IMPLICITLY, not by unrolling the passes: the
    primal passes carry no graph, and after convergence the certified pass runs once more on
    the graph with `solvers.fixed_point.differentiate_fixed_point` attaching the adjoint of
    the interface equations (`adjoint_rtol`, the GMRES tolerance of that one small solve).
    The returned gradient is therefore the CONVERGED INTERFACE's, with an error of the order
    of the primal residual (exact where the interface equations are linear in the interface)
    rather than the unrolled truncation error O(rho^passes) counted from the start state, and
    backward memory is one pass rather than all of them (P1-2). `diagnostics["adjoint"]` says
    whether that pass ran: `"implicit"` when it did, `None` when nothing differentiable
    reached the state and no extra pass was needed. `diagnostics["adjoint_batched"]` says
    whether that solve ran as one independent GMRES system per batch instance (every link's
    forward value carrying the batch as its leading dims) or, when some link's value is
    shared across instances, as today's single system flattened over the whole interface;
    `False` when `"adjoint"` is `None` too.
    """

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
        iterate_max: int = 50,
        adjoint_rtol: float = 1e-10,
    ) -> None:
        self.models = dict(models)
        self.links = list(links)
        self.aliases = list(aliases)
        self.substeps = {tag: int(k) for tag, k in substeps.items()}
        self.relaxation = float(relaxation)
        self.iterate_rtol, self.iterate_atol = float(iterate_rtol), float(iterate_atol)
        self.iterate_max = int(iterate_max)
        self.adjoint_rtol = float(adjoint_rtol)
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
            from_layer = self._layer(link.from_model, link.from_layer)
            to_layer = self._layer(link.to_model, link.to_layer)
            self._check_units(link, from_layer, to_layer)
            if link.two_way:
                self._check_two_way_scope(link, from_layer, to_layer)
        for alias in self.aliases:
            for _model, _key, name in alias.targets:
                if name is not None and name not in _CONVERSIONS:
                    raise KeyError(
                        f"CoupledModel: alias of {alias.source} names conversion {name!r}; "
                        f"registered conversions are {sorted(_CONVERSIONS)}"
                    )
        owners: dict[tuple[str, str, int], ValueLink] = {}
        for link in self.links:
            target = (link.to_model, link.to_key, link.to_index)
            first = owners.get(target)
            if first is not None:
                raise ValueError(
                    f"CoupledModel: boundary entry {link.to_model}:{link.to_key}"
                    f"[{link.to_index}] is written by two links, {self._link_key(first)} and "
                    f"{self._link_key(link)}; every boundary entry has exactly one writer. A "
                    f"second two-way link would forward the recipient's ONE transfer to two "
                    f"donors (counting it twice, P2-4) and a second one-way link would "
                    f"silently overwrite the first in list order"
                )
            owners[target] = link
        self._two_way = [link for link in self.links if link.two_way]
        self._one_way = [link for link in self.links if not link.two_way]
        recipients = {link.to_model for link in self._two_way}
        donors = {link.from_model for link in self._two_way}
        both = sorted(recipients & donors)
        if both:
            raise ValueError(
                f"CoupledModel: model(s) {both} are both a recipient and a donor of two-way "
                f"links; the conservative recipient-first schedule is defined only when no "
                f"model plays both roles (a cycle of two-way links has no such order yet)"
            )
        self._recipients = [tag for tag in self.models if tag in recipients]
        self._others = [tag for tag in self.models if tag not in recipients]
        if self._two_way and self.iterate_max < 2:
            raise ValueError(
                f"CoupledModel: a two-way link needs iterate_max >= 2, got {iterate_max!r}; "
                f"one pass rarely closes a conservative exchange to a useful tolerance, so "
                f"this floor stays as a sanity check even though the per-pass criterion below "
                f"is well-defined from the very first pass"
            )

    def _layer(self, tag: str, name: str):
        try:
            return self.models[tag].transport[name]
        except KeyError as exc:
            raise KeyError(
                f"CoupledModel: model {tag!r} has no transport layer {name!r} (has "
                f"{sorted(self.models[tag].transport) if tag in self.models else 'no such model'})"
            ) from exc

    def _check_units(self, link: ValueLink, from_layer, to_layer) -> None:
        """The forward `convert` must carry the FROM layer's unit to the TO layer's.

        Without this, a `convert=None` link between the street's `kg/m3` and CONTAM's `kg/kg`
        is accepted and is silently wrong by a factor of `rho_amb` -- a plausible mistake that
        no test of either application can catch, because both models keep running happily.
        `convert_back` is deliberately NOT checked here: it acts on a flux, whose units are
        the from-layer's own source units rather than either layer's state unit (spec A2, and
        `ValueLink`'s docstring).
        """
        what = (
            f"{link.from_model}:{link.from_layer} is {from_layer.quantity} in "
            f"{from_layer.unit!r}, {link.to_model}:{link.to_layer} is {to_layer.quantity} in "
            f"{to_layer.unit!r}"
        )
        if link.convert is None:
            if from_layer.unit != to_layer.unit:
                raise ValueError(
                    f"CoupledModel: link {self._link_key(link)} has convert=None but {what}; "
                    f"an unconverted link requires equal units -- name a conversion mapping "
                    f"{from_layer.unit!r} to {to_layer.unit!r} (registered conversions are "
                    f"{sorted(_CONVERSIONS)})"
                )
            return
        conversion = _CONVERSIONS[link.convert]   # registration already checked above
        if (from_layer.unit, to_layer.unit) != (conversion.from_unit, conversion.to_unit):
            raise ValueError(
                f"CoupledModel: link {self._link_key(link)} names conversion "
                f"{link.convert!r}, which maps {conversion.from_unit!r} to "
                f"{conversion.to_unit!r}, but {what}"
            )

    def _check_two_way_scope(self, link: ValueLink, from_layer, to_layer) -> None:
        """Milestone 5's two-way scope: one flow kind on the TO layer, one species on both.

        The recipient's own transfer is read off `Model.step`'s `boundary_transfer` at ONE
        boundary node of a single-flow-kind layer (`_apply_transfer`), so a TO layer with more
        than one flow kind has no single transfer to read there. And `_reduced`'s single-species
        layout rule is AMBIGUOUS for a multi-species state: a stacked `(n_i, 1)` and a reduced
        `(n_i, K)` with `K == n_i` are the same shape, so it would silently read the wrong axis.
        Both are therefore refused here, before any stepping, rather than discovered later.
        """
        if len(to_layer.flow_kinds) > 1:
            raise ValueError(
                f"CoupledModel: two-way link {self._link_key(link)} needs a single flow kind "
                f"on its TO layer {link.to_model}:{link.to_layer}, which has "
                f"{list(to_layer.flow_kinds)}; the recipient's own transfer is read at one "
                f"boundary node of a single-flow-kind layer"
            )
        multi = [
            f"{tag}:{layer_name} has n_species={layer.n_species}"
            for tag, layer_name, layer in (
                (link.from_model, link.from_layer, from_layer),
                (link.to_model, link.to_layer, to_layer),
            )
            if layer.n_species != 1
        ]
        if multi:
            raise ValueError(
                f"CoupledModel: two-way link {self._link_key(link)} needs n_species == 1 on "
                f"both layers ({', '.join(multi)}); `_reduced`'s single-species layout rule is "
                f"ambiguous for a multi-species state"
            )

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

    def _apply_transfer(
        self, link: ValueLink, transfer: Tensor, dt: float, drivers: dict[str, Drivers],
    ) -> Tensor:
        """Add the recipient's integrated transfer at the linked boundary node, as a source
        RATE over the donor's outer step, into the donor's sources (ADDED, never overwritten).

        `transfer` is `boundary_transfer` in the TO layer's own boundary layout, exactly as
        `Model.step(..., boundary_transfers=True)` reported it for the recipient's own step
        over the whole outer `dt` -- not a flux recomputed from a stale state, so conservation
        holds on every returned pass by construction (R1). The sources tensor goes through
        `_write_at` like every other glue write, so a STACKED single-species `(n, 1)` sources
        tensor is reduced before the node index is applied and restored afterwards.
        """
        to_layer = self._layer(link.to_model, link.to_layer)
        amount = _reduced(transfer, to_layer.n_b)[..., link.to_index]
        rate = apply_conversion(link.convert_back, amount / dt, drivers[link.to_model])
        from_layer = self._layer(link.from_model, link.from_layer)
        from_net = self.models[link.from_model].net
        key = link.sources_key or f"{link.from_layer}.sources"
        existing = drivers[link.from_model].get(key)
        if existing is None:
            existing = torch.zeros(from_net.n, dtype=rate.dtype, device=rate.device)
        node = int(from_layer.interior_idx[link.from_index])  # interior_idx is FULL node order
        drivers[link.from_model][key] = _write_at(existing, from_net.n, node, rate, add=True)
        return amount

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
            new = {}
            for tag in self.models:
                new[tag], _ = self._step_model(tag, start[tag], pass_drivers[tag], dt,
                                                 want_transfers=set())
            if diagnostics is not None:
                # One pass, nothing iterated: `converged` is true for every instance by
                # construction, and `max_change` is EMPTY rather than 0.0 -- no change was
                # measured, and reporting a number nothing measured would be a claim. Same
                # types as the iterated path (a 0-d bool tensor, a per-link dict).
                # `adjoint` is None here for the same reason `max_change` is empty: a
                # single explicit pass has no fixed point to differentiate implicitly, so no
                # adjoint was attached, and `adjoint_batched` is False along with it. The
                # keys are present so that both paths report the same set.
                diagnostics.update({
                    "passes": 1,
                    "converged": torch.ones((), dtype=torch.bool),
                    "max_change": {},
                    "transfers": {},
                    "adjoint": None,
                    "adjoint_batched": False,
                })
            return new
        return self._iterate(start, drivers, dt, diagnostics)

    def _step_model(
        self, tag: str, start: State, drivers: Drivers, dt: float,
        *, want_transfers: Collection[str],
    ) -> tuple[State, dict[str, Tensor]]:
        """Step model `tag` ONCE over `dt` -- in `k` sub-steps of `dt/k` when named in
        `substeps`, its glue-derived drivers held constant across them (design spec section 3
        point 4). `want_transfers` names the transport layer(s) to report the boundary
        transfer for (empty: none, a plain step); `Model.step` runs `step_with_transfer` on
        ONLY those layers, not every transport layer of the model -- a recipient with an
        unlinked layer must not pay `step_with_transfer`'s extra cost on a layer nothing
        reads a transfer from (task 18b). The returned dict sums each named layer's
        transfer over these `k` outer sub-steps too -- the total amount that crossed each
        boundary node during THIS model's whole `dt`, in the layer's own boundary layout."""
        k = self.substeps.get(tag, 1)
        s = start
        totals: dict[str, Tensor] = {}
        for _ in range(k):
            d: dict = {}
            s = self.models[tag].step(
                s, drivers, dt / k, diagnostics=d, boundary_transfers=want_transfers
            )
            if want_transfers:
                for name, layer_diag in d["layers"].items():
                    if "boundary_transfer" in layer_diag:
                        transfer = layer_diag["boundary_transfer"]
                        totals[name] = transfer if name not in totals else totals[name] + transfer
        return s, totals

    @staticmethod
    def _link_key(link: ValueLink) -> str:
        """The name a link answers to in diagnostics and in the non-convergence message."""
        return (
            f"{link.from_model}:{link.from_key}[{link.from_index}]->"
            f"{link.to_model}:{link.to_key}"
        )

    def _one_pass(
        self, start: Mapping[str, State], drivers: Mapping[str, Drivers], dt: float,
        values: Sequence[Tensor],
    ) -> tuple[dict[str, State], dict[str, Tensor], dict[str, Drivers]]:
        """ONE recipient-first pass with EVERY link's forward value taken from `values` (one
        entry per link in `self.links` order, one-way and two-way alike, already relaxed by
        the caller): write the values into the recipients' boundary drivers, step the
        recipients from `start`, hand each two-way link's integrated transfer to its donor as
        a source rate, then step everyone else from `start`.

        This is the whole pass, and it is the ONLY place a pass is run -- the iteration below
        calls it for each primal pass and `differentiate_fixed_point` calls it once more, at
        the converged values, for the returned differentiable one. Conservation (R1) is a
        property of THIS function, so it holds on every pass it produces, certified or
        differentiated, by construction: the donor receives exactly the amount the recipient's
        own scheme integrated across the linked boundary node in this same pass.

        The three dicts it returns are fresh: `new` (the stepped state per model), the
        per-link integrated `transfers`, and the `pass_drivers` the pass was run with (the
        boundary writes and the donor source terms), which the caller needs to read the
        pass's own forward values back out through the same conversions.
        """
        pass_drivers = {tag: dict(d) for tag, d in drivers.items()}
        # One loop over `self.links` rather than one-way-then-two-way: every boundary entry
        # has exactly one writing link (the ownership check in `__init__`), so no two writes
        # here can collide and the order between them cannot matter.
        for link, value in zip(self.links, values, strict=True):
            self._write_forward(link, value, pass_drivers)
        new: dict[str, State] = {}
        transfers: dict[str, Tensor] = {}
        for tag in self._recipients:                      # recipients first
            layers = {link.to_layer for link in self._two_way if link.to_model == tag}
            new[tag], totals = self._step_model(
                tag, start[tag], pass_drivers[tag], dt, want_transfers=layers
            )
            for link in self._two_way:
                if link.to_model == tag:
                    transfers[self._link_key(link)] = self._apply_transfer(
                        link, totals[link.to_layer], dt, pass_drivers
                    )
        for tag in self._others:                          # donors and uncoupled models
            new[tag], _ = self._step_model(
                tag, start[tag], pass_drivers[tag], dt, want_transfers=set()
            )
        return new, transfers, pass_drivers

    def _iterate(self, start, drivers, dt, diagnostics) -> dict[str, State]:
        """Successive substitution on every two-way link's FORWARD value, damped by
        `relaxation` (0.5 mirrors `Model._iterate`, `model.py:540-611`), on a RECIPIENT-FIRST
        schedule: each pass steps every recipient (`self._recipients`) from `start` first,
        with the pass's relaxed forward values written into its boundary drivers, then applies
        each two-way link's donor transfer -- the recipient's own integrated boundary
        transfer this pass, as a source RATE over the donor's whole `dt` (`_apply_transfer`,
        R1) -- into the DONOR's pass drivers, and only then steps every other model
        (`self._others`, donors and uncoupled models) from `start`. The donor therefore
        always receives exactly what the recipient's own scheme integrated THIS pass, so
        conservation holds on every returned pass by construction; no separate residual is
        needed to enforce it. One pass is `_one_pass`, which is also what the differentiable
        pass below runs, so that property is shared by both.

        Convergence is judged PER INSTANCE, on detached copies inside `torch.no_grad()`, on
        EVERY pass including the first: the donor's RETURNED forward value (computed from
        `new`, this pass's own donor output) against the relaxed forward value the recipient
        was actually stepped with in this same pass (R2) -- the returned state is what is
        certified, not the previous pass's forward value. This is the fixed-point map's own
        UNRELAXED residual, `g(v_k) - v_k`: the returned forward value `g(v_k)` against the
        relaxed iterate `v_k` the recipient was just stepped with, not the relaxed increment
        `relaxation * (raw - prev)` that produced `v_k` in the first place. Judging the
        relaxed increment instead would make the effective tolerance scale with `relaxation`,
        letting a heavily damped run falsely report convergence while `g(v_k)` still
        disagrees with `v_k` by a large, unrelaxed amount.

        Differentiation: the primal passes carry no graph. After convergence the certified
        pass runs once more on the graph at the SAME interface values and
        `solvers.fixed_point.differentiate_fixed_point` attaches the implicit adjoint of the
        interface equations (P1-2); memory is one pass. What that buys, stated no higher than
        it is: the gradient is the CONVERGED INTERFACE's, so its error is of the order of the
        primal residual at `values` -- not the unrolled truncation error O(rho^passes)
        counted from the START state, which is tied to nothing the caller controls and is
        worst exactly where the primal is cheapest. Where the interface equations are LINEAR
        in the interface the adjoint is exact whatever the residual, which is why the P1-2
        fixtures return 1/3 and 2/3 to one ulp both at `iterate_rtol=1e-12` and at 1e-3.
        Pass 1 runs on the
        graph only to learn whether anything differentiable reaches the state -- a structural
        question no inspection of `start` and `drivers` can answer, since a differentiable
        parameter may be captured inside a model's own closure and never appear in either --
        and its graph is dropped at once; if nothing does, no extra pass runs and a
        forward-only simulation costs exactly the passes it always did.

        The INTERFACE is every link's forward value, one-way links included, because that is
        what a pass actually reads from the previous pass's output. A one-way value kept out
        of the interface would have to be held fixed against the parameters, and the gradient
        would silently lose the path running from a parameter through the donor's state into
        the recipient. Only the two-way entries are MEASURED for convergence (only they close
        a loop that can fail to converge), so a one-way entry is converged only as far as the
        state it reads has settled -- exactly as true of the returned primal state itself,
        which is read from that very same pass.
        """
        two_way_pos = [i for i, link in enumerate(self.links) if link.two_way]
        latest: Mapping[str, State] = start
        prev: list[Tensor] | None = None
        change: dict[str, Tensor] = {}
        # A placeholder the FIRST pass always replaces: convergence is judged on every pass
        # including the first (see the docstring above), so the loop's own body always
        # computes a real per-instance verdict before this initial value could ever be read.
        # It exists only to give `converged` a well-typed shape/dtype/device up front.
        converged: Tensor = torch.zeros((), dtype=torch.bool)
        passes = 0
        values: list[Tensor] = []
        new: dict[str, State] = dict(start)
        last_transfers: dict[str, Tensor] = {}
        needs_adjoint = False
        while passes < self.iterate_max:
            passes += 1
            # Pass 1 runs with grad ENABLED only to learn whether anything differentiable
            # reaches the state (its graph is dropped a few lines below); every later primal
            # pass runs without a graph at all. The derivative comes from the ONE
            # differentiable pass after the loop, never from these.
            with torch.set_grad_enabled(passes == 1 and torch.is_grad_enabled()):
                scratch = {tag: dict(d) for tag, d in drivers.items()}
                raw = [self._forward_value(link, latest, scratch) for link in self.links]
                values = list(raw)
                if prev is not None:
                    for i in two_way_pos:
                        values[i] = prev[i] + self.relaxation * (raw[i] - prev[i])
                new, transfers, pass_drivers = self._one_pass(start, drivers, dt, values)
                last_transfers = {k: v.detach() for k, v in transfers.items()}
                if passes == 1:
                    needs_adjoint = any(
                        t.requires_grad for s in new.values() for t in s.values()
                    )
                    # Drop pass 1's graph here, and drop ALL of it: `values`, `new`, the pass
                    # drivers (whose boundary writes and donor source terms were built from
                    # the grad-carrying values and transfers) and the transfers themselves
                    # each anchor it, and each stays bound until the next pass overwrites it
                    # -- or until this method returns, if the loop breaks on this pass.
                    # Nothing downstream wants it: the convergence check below runs under
                    # `no_grad` and reads the pass drivers only for the conversions, and the
                    # derivative comes from the one pass after the loop.
                    values = [v.detach() for v in values]
                    new = {tag: {k: t.detach() for k, t in s.items()}
                           for tag, s in new.items()}
                    pass_drivers = {tag: {k: t.detach() for k, t in d.items()}
                                    for tag, d in pass_drivers.items()}
                    del transfers
            with torch.no_grad():
                ok: Tensor | None = None
                for i in two_way_pos:
                    link = self.links[i]
                    # A forward value is ONE number per instance (one node, one species), so
                    # its own shape IS the batch shape and nothing is reduced away.
                    returned = self._forward_value(link, new, pass_drivers)
                    d = (returned - values[i]).abs()
                    change[self._link_key(link)] = d
                    this = d <= self.iterate_atol + self.iterate_rtol * returned.abs()
                    ok = this if ok is None else (ok & this)
                converged = ok
            if bool(converged.all()):
                break
            prev = values
            latest = new
        if not bool(converged.all()):
            failing = (~converged).nonzero().flatten().tolist() if converged.dim() else "all"
            worst = {name: float(c.max()) for name, c in change.items()}
            raise RuntimeError(
                f"CoupledModel: two-way coupling did not converge within {self.iterate_max} "
                f"passes for instances {failing}; largest change per link {worst}, tolerance "
                f"atol={self.iterate_atol} rtol={self.iterate_rtol}"
            )
        # `values` are the relaxed forward values the CERTIFIED pass -- the one whose output
        # was just judged converged -- was stepped with, so running `_one_pass` on them once
        # more reproduces that pass TO SOLVER ACCURACY. Not bitwise by construction:
        # `solvers/select.py` drops the SuperLU fast path for an input that requires grad, so
        # this pass can take a different linear-solver route than primal passes 2..n did
        # (measured bitwise identical on every fixture in the suite, but that is a
        # measurement, not a guarantee). Conservation does not rest on it either way, being a
        # property of `_one_pass` itself rather than of which solver ran inside it.
        # `list(new)`, NOT `list(self.models)`: `_one_pass` builds its dict
        # recipients-first, and the no-adjoint branch below returns `new` itself, so taking
        # declaration order here would make the returned mapping's key order flip the moment
        # autograd is switched on. Reading the order off `new` also makes `keys` and the
        # `out` of every pass agree by construction.
        tags = list(new)
        keys = {tag: list(new[tag]) for tag in tags}
        adjoint_batched = False
        if needs_adjoint:
            final_transfers: dict[str, Tensor] = {}

            def pass_fn(z: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
                out, tr, pd = self._one_pass(start, drivers, dt, z)
                final_transfers.update(tr)
                # `out` is only READ here: `differentiate_fixed_point` hands its outputs back
                # as VIEWS of these tensors, and writing into one would mutate the retained
                # pass graph under autograd.
                outputs = [out[tag][k] for tag in tags for k in keys[tag]]
                # Every entry RECOMPUTED from this pass's own output, never a leaf of `z`
                # handed straight back: an unchanged entry would put a unit row in the
                # interface Jacobian and make `I - G_z` singular.
                z_next = [self._forward_value(link, out, pd) for link in self.links]
                return outputs, z_next

            # Every link's forward value is one number per instance, so its own shape IS the
            # batch shape (the docstring above) -- exactly the leading dims `converged`
            # carries, since `converged` is built by reducing over nothing but the two-way
            # links themselves. A run with a value shared across instances (a one-way link
            # fed from an unbatched driver, say) fails that check per entry and
            # `differentiate_fixed_point` falls back to the flattened solve on its own.
            adjoint_report: dict = {}
            flat = differentiate_fixed_point(
                values, pass_fn, rtol=self.adjoint_rtol,
                where="CoupledModel two-way coupling",
                batch_shape=tuple(converged.shape), report=adjoint_report,
            )
            result: dict[str, State] = {}
            it = iter(flat)
            for tag in tags:
                result[tag] = {k: next(it) for k in keys[tag]}
            # The differentiated pass's own transfers, DETACHED: diagnostics are a report,
            # and the rest of this dict (`converged`, `max_change`) is grad-free already.
            # A transfer still attached to the pass graph would offer a second, silent route
            # around the adjoint -- a loss touching it would be differentiated through the
            # single pass, which is exactly the truncated derivative P1-2 removes.
            reported = {k: v.detach() for k, v in final_transfers.items()}
            adjoint = "implicit"
            adjoint_batched = bool(adjoint_report["batched"])
        else:
            result, reported, adjoint = new, last_transfers, None
        if diagnostics is not None:
            diagnostics.update(
                {"passes": passes, "converged": converged, "max_change": change,
                 "transfers": reported, "adjoint": adjoint, "adjoint_batched": adjoint_batched}
            )
        return result


def union(
    models: Mapping[str, tuple[Model, State, Drivers]],
    shared: Sequence[ValueLink | DriverAlias],
    *,
    substeps: Mapping[str, int] | None = None,
    relaxation: float = 0.5,
    iterate_rtol: float = 1e-8,
    iterate_atol: float = 0.0,
    iterate_max: int = 50,
    adjoint_rtol: float = 1e-10,
) -> tuple[CoupledModel, dict[str, State], dict[str, Drivers]]:
    """Couple `models` by exchanging the driver/state values `shared` names, WITHOUT merging
    any model's `Network`, layers, or closures (design spec section 3). Never modifies the
    `Model`/`State`/`Drivers` objects passed in -- returns fresh dict copies.

    `iterate_max=50` is a floor, not a measurement pinned here: the recipient-first schedule
    changed the per-pass cost and the pass counts a coupling needs to reach a given
    tolerance, so see `docs/applications/coupling.md` for current numbers rather than this
    docstring. `iterate_atol=0.0` makes the criterion purely relative, so a shared value that
    is legitimately ZERO needs a positive `iterate_atol` to be judged converged at all.
    `adjoint_rtol` is the tolerance of the implicit adjoint solve at the converged interface,
    and is independent of the primal `iterate_rtol` -- that independence is the point (P1-2).
    See `CoupledModel`.
    """
    model_map = {tag: m for tag, (m, _s, _d) in models.items()}
    state_map = {tag: dict(s) for tag, (_m, s, _d) in models.items()}
    drivers_map = {tag: dict(d) for tag, (_m, _s, d) in models.items()}
    links = [item for item in shared if isinstance(item, ValueLink)]
    aliases = [item for item in shared if isinstance(item, DriverAlias)]
    city = CoupledModel(
        model_map, links, aliases, substeps or {}, relaxation=relaxation,
        iterate_rtol=iterate_rtol, iterate_atol=iterate_atol, iterate_max=iterate_max,
        adjoint_rtol=adjoint_rtol,
    )
    return city, state_map, drivers_map
