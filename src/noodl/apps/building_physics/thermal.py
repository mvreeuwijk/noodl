"""Zones, walls, the heat layer as a TransportLayer, density closures, and build_model.

Units: mass flow kg/s, temperature K, heat W, mass fraction kg/kg.
The heat layer is `TransportLayer(carrier=c_p, capacity=rho_0 c_p V [+ wall capacity],
conduction on 'wall' edges with conductance UA)`; nothing here solves anything itself.
Zone air heat capacity is held at the reference density (CONTAM likewise holds zone air mass
within a step).
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from noodl.components import Component
from noodl.elements import PrescribedFlow
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.transport import TransportLayer, active_interior
from noodl.layout import broadcast_batch, per_instance
from noodl.model import Drivers, Model, State
from noodl.refs import drivers_from, state_from
from noodl.topology import Network

if TYPE_CHECKING:
    from noodl.apps.building_physics.controls import ControlNetwork

R_AIR = 287.055
RHO_0 = 1.2041
CP_AIR = 1005.0
P_REF = 101325.0
T_REF = 293.15


@dataclass
class WallMass:
    """A lumped wall node: capacity [J/K], conductances to the zone and to ambient [W/K].

    Both conductances must be strictly positive, and are checked here rather than left to the
    layer. A NEGATIVE `ua` makes the conduction term `A_c diag(g) A_c^T` indefinite instead of
    positive semi-definite, so the heat layer's operator is anti-diffusive: heat flows UP the
    temperature gradient and a wall node can overshoot both of its neighbours, breaking the
    maximum principle `docs/theory.md` section 7 claims for the layer -- and every solve still
    reports success, because nothing downstream ever looks at the sign. A ZERO conductance is
    refused with it: it is a wall that conducts nothing, leaving the wall node with no
    conduction path to the rest of the network.
    """

    name: str
    capacity: float
    ua_zone: float
    ua_ambient: float

    def __post_init__(self) -> None:
        for attr in ("ua_zone", "ua_ambient"):
            value = float(getattr(self, attr))
            if not value > 0:
                raise ValueError(
                    f"WallMass {self.name!r}: {attr} must be strictly positive, got {value!r} "
                    f"(a non-positive conductance makes the heat layer's conduction operator "
                    f"anti-diffusive)"
                )


@dataclass
class Zone:
    """A room: its air `volume` (m3), initial temperature `T0` (K), reference height
    `z_ref` (m) and an optional lumped `wall` (`WallMass`). Add it with `add_zone`."""

    name: str
    volume: float
    T0: float = T_REF
    z_ref: float = 0.0
    wall: WallMass | None = None


def add_zone(net: Network | Component, zone: Zone, *, ambient="ambient") -> None:
    """Add the zone's air node (attributes volume, T0, z_ref, heat_capacity=0) and, if it has
    a wall, the wall node (heat_capacity) with 'wall' edges zone -> wall -> ambient (ua).

    The wall's conductances are validated by `WallMass` itself, at construction.
    `net` may be a `Network` or a `noodl.components.Component`; in a component the wall
    reaches the shared ambient through `outer(ambient)`."""
    if isinstance(net, Component):
        amb = net.outer(ambient)  # the building's shared ambient (inner/outer)
    else:
        if ambient not in net.nodes:
            net.add_node(ambient, z_ref=0.0)
        amb = ambient
    net.add_node(zone.name, volume=zone.volume, T0=zone.T0, z_ref=zone.z_ref, heat_capacity=0.0)
    if zone.wall is not None:
        w = zone.wall
        net.add_node(w.name, volume=0.0, T0=zone.T0, z_ref=zone.z_ref, heat_capacity=w.capacity)
        net.add_edge(zone.name, w.name, kind="wall", ua=w.ua_zone)
        net.add_edge(w.name, amb, kind="wall", ua=w.ua_ambient)


def thermal_layer(net: Network, *, ambient="ambient", name: str = "thermal",
                  flow_kinds=("airpath",), conduction_kind: str = "wall", c_p: float = CP_AIR,
                  rho: float = RHO_0, scheme: str | None = None,
                  fixed_temperature=(), storage_free: Sequence = ()) -> TransportLayer:
    """The heat layer: capacity rho c_p V + heat_capacity per active interior node, carrier
    c_p on the flow kinds, conduction UA on `conduction_kind` edges (if any exist).

    `storage_free` names nodes that hold no heat (a zero-volume air-handler plenum): their
    capacity may be zero, and they need the implicit scheme. `scheme` defaults to "exact",
    or to "implicit" when `storage_free` names any node; an explicit other scheme with
    storage-free nodes is refused by name.
    """
    storage_free = tuple(storage_free)
    if scheme is None:
        scheme = "implicit" if len(storage_free) else "exact"
    boundary = [ambient, *fixed_temperature]
    kinds = tuple(flow_kinds)
    has_walls = conduction_kind in net.edge_kinds()
    # `active_interior` is the ONE place the active-interior rule lives, and
    # is exactly what `TransportLayer` sizes its own interior with -- a wall-mass node has
    # conduction edges and no airpath edges, so conduction must be counted as a touch here
    # too or the capacity vector would be one entry short of the layer's rows.
    interior, _ = active_interior(
        net, kinds + ((conduction_kind,) if has_walls else ()), boundary
    )
    capacity = (rho * c_p * net.node_attr("volume", default=0.0)
                + net.node_attr("heat_capacity", default=0.0))[interior]
    free = set(net.canonical(list(storage_free))) if storage_free else set()
    bad = [
        net.nodes[i]
        for i, c in zip(interior.tolist(), capacity.tolist(), strict=True)
        if c <= 0 and net.nodes[i] not in free
    ]
    if bad:
        raise ValueError(
            f"thermal layer {name!r}: nodes {bad} have no heat capacity (zero volume and no "
            f"wall capacity){_STORAGE_FREE_HINT}"
        )
    return TransportLayer(
        net, name, capacity=capacity, flow_kind=kinds, boundary=boundary, carrier=float(c_p),
        conduction_kind=conduction_kind if has_walls else None,
        conductance=net.edge_attr("ua", conduction_kind) if has_walls else None,
        scheme=scheme, quantity="temperature", unit="K",
        source_unit="W", capacity_unit="J/K", flow_unit="kg/s",
        storage_free=tuple(storage_free),
    )


def species_layer(net: Network, *, ambient="ambient", name: str = "species",
                  flow_kinds=("airpath",), rho: float | torch.Tensor = RHO_0,
                  n_species: int = 1, scheme: str = "implicit",
                  species_names=None, transmission=None, kinetics=None,
                  storage_free: Sequence = ()) -> TransportLayer:
    """Species as mass fractions with zone air mass rho V as capacity (CONTAM convention).

    `rho` is one density for every node or a full-node `(n,)` tensor of per-node densities
    (`prj.project_to_model` passes each zone's `Pb / (R T)`, as ContamX holds it).
    `transmission` (`(b_flow, K)`, the pass fraction 1 - efficiency of each flow edge's
    filter; `filter_transmission` builds it) and `kinetics` (`(n_i, K, K)`, each zone's
    first-order reactions; `reaction_kinetics` builds it) go to the `TransportLayer`
    unchanged. `storage_free` names nodes that hold no air (a zero-volume air-handler
    plenum): their volume may be zero, and the layer must use `scheme="implicit"`.
    """
    storage_free = tuple(storage_free)
    kinds = tuple(flow_kinds)
    interior, _ = active_interior(net, kinds, [ambient])
    capacity = (rho * net.node_attr("volume", default=0.0))[interior]
    free = set(net.canonical(list(storage_free))) if storage_free else set()
    bad = [
        net.nodes[i]
        for i, c in zip(interior.tolist(), capacity.tolist(), strict=True)
        if c <= 0 and net.nodes[i] not in free
    ]
    if bad:
        raise ValueError(f"species layer {name!r}: nodes {bad} have zero volume"
                         f"{_STORAGE_FREE_HINT}")
    return TransportLayer(
        net, name, capacity=capacity, flow_kind=kinds, boundary=[ambient], n_species=n_species,
        scheme=scheme, quantity="mass_fraction", unit="kg/kg", species_names=species_names,
        source_unit="kg/s", capacity_unit="kg", flow_unit="kg/s",
        transmission=transmission, kinetics=kinetics, storage_free=tuple(storage_free),
    )


class _DensityClosure:
    # Declared driver keys (`noodl.validation`): read optionally, and written.
    inputs = ("P_ref",)
    outputs = ("rho", "rho_amb")
    output_units = {"rho": "kg/m3", "rho_amb": "kg/m3"}

    def __init__(self, thermal: TransportLayer, *, ambient_index: int) -> None:
        self.thermal = thermal
        self.ambient_index = int(ambient_index)

    def _temperatures(self, state, drivers) -> torch.Tensor:
        name = self.thermal.name
        T_i = state[f"{name}.x"]
        T_b = drivers[f"{name}.x_boundary"]
        batch, (T_i, T_b) = broadcast_batch((T_i, 1), (T_b, 1))
        n = self.thermal.net.n
        T = torch.full(batch + (n,), T_REF, dtype=T_i.dtype, device=T_i.device)
        T[..., self.thermal.interior_idx] = T_i
        T[..., self.thermal.boundary_idx] = T_b
        return T

    def _rho(self, T: torch.Tensor, drivers) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, state, drivers):
        rho = self._rho(self._temperatures(state, drivers), drivers)
        return {"rho": rho, "rho_amb": rho[..., self.ambient_index]}


class IdealGasDensity(_DensityClosure):
    """rho = P_ref / (R T); P_ref from drivers["P_ref"] if present (CONTAM section 3.18)."""

    def __init__(self, thermal, *, ambient_index: int, P_ref: float = P_REF, R: float = R_AIR):
        super().__init__(thermal, ambient_index=ambient_index)
        self.P_ref, self.R = float(P_ref), float(R)

    def _rho(self, T, drivers):
        # A batched pressure `(B, ...)` is one per instance: it gets the node axis, as the
        # CONTAM closure does, so B == n cannot pair a pressure with a node.
        P = per_instance(drivers.get("P_ref", self.P_ref), T, 1)
        return P / (self.R * T)

    @property
    def input_specs(self) -> dict[str, dict]:
        """The reference pressure this closure reads when given (`model.refs.inputs`)."""
        return {"P_ref": dict(description=f"reference pressure of the ideal-gas density "
                                          f"(default {self.P_ref:g} Pa)",
                              unit="Pa", required=False)}


class LinearDensity(_DensityClosure):
    """Boussinesq: rho = rho_0 (1 - (T - T_0) / T_0), the form the analytical natural-
    ventilation solutions (Li and Delsante 2001) assume."""

    def __init__(self, thermal, *, ambient_index: int, rho_0: float = RHO_0, T_0: float = T_REF):
        super().__init__(thermal, ambient_index=ambient_index)
        self.rho_0, self.T_0 = float(rho_0), float(T_0)

    def _rho(self, T, drivers):
        return self.rho_0 * (1.0 - (T - self.T_0) / self.T_0)


#: Appended to a zero-capacity refusal: the usual cause is an air handler's plenum.
_STORAGE_FREE_HINT = " (an air-handler plenum? pass storage_free= to build_model)"


def _prescribed_only_nodes(net: Network, air_laws, air) -> list:
    """The airflow layer's interior nodes that only `PrescribedFlow` edges touch (an air
    handler's plenums left out of `flow_balanced`): their pressure is undetermined."""
    other: set = set()
    prescribed: set = set()
    for el in air_laws:
        src, tgt = net.endpoints(el.kind)
        touched = {net.nodes[i] for i in src.tolist() + tgt.tolist()}
        (prescribed if isinstance(el, PrescribedFlow) else other).update(touched)
    interior = {net.nodes[i] for i in air.interior.tolist()}
    return [n for n in net.nodes if n in prescribed and n not in other and n in interior]


def check_flow_balanced(net: Network, air_laws, flow_balanced, *,
                        where: str = "build_model") -> None:
    """Refuse a flow-balanced node that an edge touches whose law is not a
    `PrescribedFlow` (an air handler's): the airflow solve pins such a node at zero gauge
    pressure, so a pressure-dependent edge would let it absorb an imbalance silently, and a
    fixed flow is outside the handler's balance."""
    balanced = set(net.canonical(list(flow_balanced))) if flow_balanced else set()
    if not balanced:
        return
    for el in air_laws:
        if isinstance(el, PrescribedFlow):
            continue
        src, tgt = net.endpoints(el.kind)
        for s, t in zip(src.tolist(), tgt.tolist(), strict=True):
            for node in (net.nodes[s], net.nodes[t]):
                if node in balanced:
                    raise ValueError(
                        f"{where}: flow_balanced node {node!r} is an end of the edge "
                        f"{net.nodes[s]!r} -> {net.nodes[t]!r} of kind {el.kind!r}, whose "
                        f"law is a {type(el).__name__}; the airflow solve pins a "
                        f"flow-balanced node at zero gauge pressure, so only an air "
                        f"handler's prescribed flows (PrescribedFlow) may touch it")


def build_model(net: Network, *, air_laws, drives, ambient="ambient", thermal: bool = True,
                species: int | Sequence[str] = 0, density: str = "ideal_gas",
                density_kwargs=None,
                coupling: str = "pingpong", iterate_tol=None, iterate_max: int = 20,
                thermal_scheme: str | None = None, species_scheme: str = "implicit",
                flow_kinds=None, closures: Sequence = (), flow_balanced: Sequence = (),
                storage_free: Sequence = (), controls: ControlNetwork | None = None,
                return_inputs: bool = False):
    """Layers "air" (+ "thermal", + "species"), the density closure, one Model.

    Returns the `Model`; with `return_inputs=True`, `(model, state, drivers)` -- its
    `initial_state(model)` and `initial_drivers(model)` -- like every other application's
    `build_model`.

    TWO COUPLING TRAPS, both inherited from `Model` (see `noodl.model.Model`, the class
    docstring's "Two couplings"), because this builder chooses the coupling but never calls
    `step`/`steady` itself -- solve tolerances are per-call `**solve_kwargs` on those methods,
    so neither trap is fixable here:

    1. `coupling` DEFAULTS TO `"pingpong"`, which is exactly ONE pass. A
       `build_model(...).steady(...)` at the defaults therefore solves the airflow at the
       state the closures saw -- the INITIAL temperatures -- and never re-converges it
       against the temperatures it produced, reporting a zero thermal residual for that one
       pass. On the linear-density single zone that returns T_z = 366.0327 K, exactly
       `T_o + S / (c_p F(293.15 K))`, and not the coupled answer. This is correct ping-pong
       (Hensen 1995) and it is silent; pass `coupling="iterate"` with `iterate_tol` whenever
       the airflow depends on the temperatures it carries.
    2. `iterate_tol` is an ABSOLUTE tolerance in the layer's own units (K for "thermal"), and
       it cannot be met below the potential solve's residual floor propagated through
       `dT/dF`. Newton's default residual tolerance is dtype-derived (`sqrt(eps)`, 1.49e-8 for
       float64); at `dT/dF ~ 3.6e3 K.s/kg` that is ~5e-5 K of noise, so an `iterate_tol` of
       1e-9 K stalls and `steady` raises, naming the layer, the change and the tolerance. A
       tight `iterate_tol` requires a matching `atol`/`rtol` on the `steady`/`step` call.

    `flow_kinds` NARROWS which edge kinds advect heat and species; it defaults to every air
    law's kind, which is what a building network wants. Naming a subset is deliberate
    (a duct kind that carries mass but whose heat is modelled elsewhere), so it is allowed --
    but every kind named must be one the air layer actually provides, or the narrowing is a
    typo that silently drops advection. Unknown kinds raise `KeyError` naming them. The same
    value governs BOTH the thermal and the species layer.

    `thermal_scheme` defaults to `thermal_layer`'s own default ("exact", or "implicit" when
    the layer has storage-free nodes).

    `controls`: a `ControlNetwork`, put first among the closures (its actuators write the
    drivers the other closures read). Its sensors' named readings (`node_mass_fraction`,
    `edge_flow`, ...) are resolved against this model's layers. `return_inputs=True` then
    also seeds its outputs, memory and `"controls.input"`.

    `closures` are appended after the density closure (an air handler's
    `AirHandlerFlows`). `flow_balanced` names nodes whose pressure the airflow solve does
    not need, because only prescribed, balanced flows touch them (an air handler's
    plenums): they join the air layer's boundary at zero gauge pressure, while staying
    ordinary nodes of the heat and species layers. Only edges whose law is a
    `PrescribedFlow` may touch them: a pressure-dependent edge would let the plenum, pinned
    at zero gauge pressure, absorb an imbalance silently, and a fixed flow is outside the
    handler's balance, so either is refused, naming the node and the edge. `storage_free`
    names nodes that hold no air (a zero-volume plenum); their layers must use the implicit
    scheme.
    """
    storage_free = tuple(storage_free)
    check_flow_balanced(net, air_laws, flow_balanced)
    law_kinds = tuple(el.kind for el in air_laws)
    kinds = tuple(flow_kinds) if flow_kinds else law_kinds
    unknown = [k for k in kinds if k not in law_kinds]
    if unknown:
        raise KeyError(
            f"build_model: flow_kinds names {unknown}, which no air law provides "
            f"(the air laws' kinds are {list(law_kinds)}); a kind the air layer does "
            f"not solve carries no flow, so heat and species would silently not advect on it"
        )
    air = KirchhoffLayer(net, "air", list(air_laws), drives=list(drives),
                         boundary=[ambient, *flow_balanced], quantity="pressure", unit="Pa",
                         flow_unit="kg/s")
    air.node_hints.update(dict.fromkeys(_prescribed_only_nodes(net, air_laws, air),
                                        "an air-handler plenum? pass flow_balanced= to "
                                        "build_model"))
    layers: dict = {"air": air}
    model_closures: list = [] if controls is None else [controls]
    if thermal:
        th = thermal_layer(net, ambient=ambient, flow_kinds=kinds, scheme=thermal_scheme,
                           storage_free=storage_free)
        layers["thermal"] = th
        amb = net.node_index(ambient)
        kw = dict(density_kwargs or {})
        if density == "ideal_gas":
            model_closures.append(IdealGasDensity(th, ambient_index=amb, **kw))
        elif density == "linear":
            model_closures.append(LinearDensity(th, ambient_index=amb, **kw))
        else:
            raise ValueError(
                f"build_model: density must be 'ideal_gas' or 'linear', got {density!r}"
            )
    if species:
        # A count, or the species' names (which `model.refs` then builds and reads by).
        if isinstance(species, str):
            raise TypeError(
                f"build_model: species must be a count or a sequence of names, got the "
                f"string {species!r}; pass ({species!r},)"
            )
        names = None if isinstance(species, numbers.Integral) else tuple(species)
        layers["species"] = species_layer(
            net, ambient=ambient, flow_kinds=kinds, scheme=species_scheme,
            n_species=int(species) if names is None else len(names), species_names=names,
            storage_free=storage_free,
        )
    model_closures.extend(closures)
    if controls is not None:
        controls.resolve(net, layers)
    model = Model(net, layers, closures=model_closures, coupling=coupling, iterate_tol=iterate_tol,
                  iterate_max=iterate_max)
    if return_inputs:
        state, drivers = initial_state(model), initial_drivers(model)
        if controls is not None:
            state.update(controls.initial_state())
            drivers.update(controls.initial_drivers())
        return model, state, drivers
    return model


def initial_state(model: Model, *, values: Mapping | None = None) -> State:
    """`"<layer>.x"` for every transport layer: temperatures from the node attribute `T0`
    (interior order), mass fractions zero; then `values`.

    Keyed by each layer's OWN name, and dispatched on its `quantity` tag rather than on the
    names `"thermal"`/`"species"`: a layer renamed through `thermal_layer(name=...)` must
    still get its state, and must never be answered with a silently EMPTY state that only
    surfaces later as `Model`'s "state '<name>.x' is required" from inside a step. A
    transport layer whose `quantity` this application does not know is refused by name for
    the same reason -- unless `values` gives its state.

    `values` overrides by NAME, `{layer or key: {node: value}}` (see
    `noodl.refs.state_from`); nodes not named keep the default above:
    `initial_state(model, values={"thermal": {"A": 295.0}, "species": {"A": 4e-4}})`.
    """
    values = dict(values or {})
    given = {str(k) for k in values} | {f"{k}.x" for k in map(str, values)}
    state: State = {}
    for name, layer in model.transport.items():
        if f"{name}.x" in given and layer.quantity not in ("temperature", "mass_fraction"):
            continue
        if layer.quantity == "temperature":
            state[f"{name}.x"] = model.net.node_attr("T0", default=T_REF)[layer.interior_idx]
        elif layer.quantity == "mass_fraction":
            shape = (layer.n_i,) if layer.n_species == 1 else (layer.n_i, layer.n_species)
            state[f"{name}.x"] = torch.zeros(shape, dtype=model.net.dtype)
        else:
            raise ValueError(
                f"initial_state: transport layer {name!r} has quantity "
                f"{layer.quantity!r}, which the building application has no initial value "
                f"for (it knows 'temperature' and 'mass_fraction'); build it with "
                f"thermal_layer/species_layer, or set that layer's own state key "
                f"{name + '.x'!r} yourself (initial_state(model, values={{{name!r}: ...}}))"
            )
    if values:
        state = state_from(model, values, base=state)
    return state


def initial_drivers(model: Model, *, values: Mapping | None = None) -> Drivers:
    """A complete driver set: zero boundary pressures, boundary temperatures from each
    boundary node's `T0` attribute (`T_REF` where it has none), zero boundary mass
    fractions; then `values`, by name (see `noodl.refs.drivers_from`):

        initial_drivers(model, values={"thermal.sources": {"A": 1000.0},
                                       "thermal.x_boundary": {"ambient": 283.15}})

    Sources are optional and absent unless `values` gives them.
    """
    drivers: Drivers = {}
    T0 = model.net.node_attr("T0", default=T_REF)
    for name, layer in model.kirchhoff.items():
        drivers[f"{name}.phi_boundary"] = torch.zeros(len(layer.bound), dtype=model.net.dtype)
    for name, layer in model.transport.items():
        if layer.quantity == "temperature":
            drivers[f"{name}.x_boundary"] = T0[layer.boundary_idx]
        elif layer.quantity == "mass_fraction":
            shape = (layer.n_b,) if layer.n_species == 1 else (layer.n_b, layer.n_species)
            drivers[f"{name}.x_boundary"] = torch.zeros(shape, dtype=model.net.dtype)
        else:
            raise ValueError(
                f"initial_drivers: transport layer {name!r} has quantity "
                f"{layer.quantity!r}, which the building application has no boundary value "
                f"for; set {name + '.x_boundary'!r} yourself"
            )
    if values:
        drivers = drivers_from(model, values, base=drivers)
    return drivers
