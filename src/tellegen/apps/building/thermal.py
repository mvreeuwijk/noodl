"""Zones, walls, the heat layer as a TransportLayer, density closures, and build_model.

Units (milestone 2 spec 6.4): mass flow kg/s, temperature K, heat W, mass fraction kg/kg.
The heat layer is `TransportLayer(carrier=c_p, capacity=rho_0 c_p V [+ wall capacity],
conduction on 'wall' edges with conductance UA)`; nothing here solves anything itself.
Zone air heat capacity is held at the reference density (CONTAM likewise holds zone air mass
within a step).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tellegen.layers.potential import PotentialFlowLayer
from tellegen.layers.transport import TransportLayer, active_interior
from tellegen.model import Model, State
from tellegen.topology import Network

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
    name: str
    volume: float
    T0: float = T_REF
    z_ref: float = 0.0
    wall: WallMass | None = None


def add_zone(net: Network, zone: Zone, *, ambient="ambient") -> None:
    """Add the zone's air node (attributes volume, T0, z_ref, heat_capacity=0) and, if it has
    a wall, the wall node (heat_capacity) with 'wall' edges zone -> wall -> ambient (ua).

    The wall's conductances are validated by `WallMass` itself, at construction."""
    if ambient not in net.nodes:
        net.add_node(ambient, z_ref=0.0)
    net.add_node(zone.name, volume=zone.volume, T0=zone.T0, z_ref=zone.z_ref, heat_capacity=0.0)
    if zone.wall is not None:
        w = zone.wall
        net.add_node(w.name, volume=0.0, T0=zone.T0, z_ref=zone.z_ref, heat_capacity=w.capacity)
        net.add_edge(zone.name, w.name, kind="wall", ua=w.ua_zone)
        net.add_edge(w.name, ambient, kind="wall", ua=w.ua_ambient)


def thermal_layer(net: Network, *, ambient="ambient", name: str = "thermal",
                  flow_kinds=("airpath",), conduction_kind: str = "wall", c_p: float = CP_AIR,
                  rho: float = RHO_0, scheme: str = "exact",
                  fixed_temperature=()) -> TransportLayer:
    """The heat layer: capacity rho c_p V + heat_capacity per active interior node, carrier
    c_p on the flow kinds, conduction UA on `conduction_kind` edges (if any exist)."""
    boundary = [ambient, *fixed_temperature]
    kinds = tuple(flow_kinds)
    has_walls = conduction_kind in net.edge_kinds()
    # `active_interior` (spec 14, 4.5) is the ONE place the active-interior rule lives, and
    # is exactly what `TransportLayer` sizes its own interior with -- a wall-mass node has
    # conduction edges and no airpath edges, so conduction must be counted as a touch here
    # too or the capacity vector would be one entry short of the layer's rows.
    interior, _ = active_interior(
        net, kinds + ((conduction_kind,) if has_walls else ()), boundary
    )
    capacity = (rho * c_p * net.node_attr("volume", default=0.0)
                + net.node_attr("heat_capacity", default=0.0))[interior]
    bad = [
        net.nodes[i]
        for i, c in zip(interior.tolist(), capacity.tolist(), strict=True)
        if c <= 0
    ]
    if bad:
        raise ValueError(
            f"thermal layer {name!r}: nodes {bad} have no heat capacity (zero volume and no "
            f"wall capacity)"
        )
    return TransportLayer(
        net, name, capacity=capacity, flow_kind=kinds, boundary=boundary, carrier=float(c_p),
        conduction_kind=conduction_kind if has_walls else None,
        conductance=net.edge_attr("ua", conduction_kind) if has_walls else None,
        scheme=scheme, quantity="temperature", unit="K",
    )


def species_layer(net: Network, *, ambient="ambient", name: str = "species",
                  flow_kinds=("airpath",), rho: float = RHO_0, n_species: int = 1,
                  scheme: str = "implicit") -> TransportLayer:
    """Species as mass fractions with zone air mass rho V as capacity (CONTAM convention)."""
    kinds = tuple(flow_kinds)
    interior, _ = active_interior(net, kinds, [ambient])
    capacity = (rho * net.node_attr("volume", default=0.0))[interior]
    bad = [
        net.nodes[i]
        for i, c in zip(interior.tolist(), capacity.tolist(), strict=True)
        if c <= 0
    ]
    if bad:
        raise ValueError(f"species layer {name!r}: nodes {bad} have zero volume")
    return TransportLayer(
        net, name, capacity=capacity, flow_kind=kinds, boundary=[ambient], n_species=n_species,
        scheme=scheme, quantity="mass_fraction", unit="kg/kg",
    )


class _DensityClosure:
    def __init__(self, thermal: TransportLayer, *, ambient_index: int) -> None:
        self.thermal = thermal
        self.ambient_index = int(ambient_index)

    def _temperatures(self, state, drivers) -> torch.Tensor:
        name = self.thermal.name
        T_i = state[f"{name}.x"]
        T_b = drivers[f"{name}.x_boundary"]
        batch = torch.broadcast_shapes(T_i.shape[:-1], T_b.shape[:-1])
        n = self.thermal.net.n
        T = torch.full(batch + (n,), T_REF, dtype=T_i.dtype, device=T_i.device)
        T[..., self.thermal.interior_idx] = T_i.expand(batch + (self.thermal.n_i,))
        T[..., self.thermal.boundary_idx] = T_b.expand(batch + (self.thermal.n_b,))
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
        P = drivers.get("P_ref", self.P_ref)
        return P / (self.R * T)


class LinearDensity(_DensityClosure):
    """Boussinesq: rho = rho_0 (1 - (T - T_0) / T_0), the form the analytical natural-
    ventilation solutions (Li and Delsante 2001) assume."""

    def __init__(self, thermal, *, ambient_index: int, rho_0: float = RHO_0, T_0: float = T_REF):
        super().__init__(thermal, ambient_index=ambient_index)
        self.rho_0, self.T_0 = float(rho_0), float(T_0)

    def _rho(self, T, drivers):
        return self.rho_0 * (1.0 - (T - self.T_0) / self.T_0)


def build_model(net: Network, *, air_elements, drives, ambient="ambient", thermal: bool = True,
                species: int = 0, density: str = "ideal_gas", density_kwargs=None,
                coupling: str = "pingpong", iterate_tol=None, iterate_max: int = 20,
                thermal_scheme: str = "exact", species_scheme: str = "implicit",
                flow_kinds=None) -> Model:
    """Layers "air" (+ "thermal", + "species"), the density closure, one Model.

    TWO COUPLING TRAPS, both inherited from `Model` (see `tellegen.model.Model`, the class
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
    element's kind, which is what a building network wants. Naming a subset is deliberate
    (a duct kind that carries mass but whose heat is modelled elsewhere), so it is allowed --
    but every kind named must be one the air layer actually provides, or the narrowing is a
    typo that silently drops advection. Unknown kinds raise `KeyError` naming them. The same
    value governs BOTH the thermal and the species layer.
    """
    element_kinds = tuple(el.kind for el in air_elements)
    kinds = tuple(flow_kinds) if flow_kinds else element_kinds
    unknown = [k for k in kinds if k not in element_kinds]
    if unknown:
        raise KeyError(
            f"build_model: flow_kinds names {unknown}, which no air element provides "
            f"(the air elements' kinds are {list(element_kinds)}); a kind the air layer does "
            f"not solve carries no flow, so heat and species would silently not advect on it"
        )
    air = PotentialFlowLayer(net, "air", list(air_elements), drives=list(drives),
                             boundary=[ambient], quantity="pressure", unit="Pa")
    layers: dict = {"air": air}
    closures: list = []
    if thermal:
        th = thermal_layer(net, ambient=ambient, flow_kinds=kinds, scheme=thermal_scheme)
        layers["thermal"] = th
        amb = net.node_index(ambient)
        kw = dict(density_kwargs or {})
        if density == "ideal_gas":
            closures.append(IdealGasDensity(th, ambient_index=amb, **kw))
        elif density == "linear":
            closures.append(LinearDensity(th, ambient_index=amb, **kw))
        else:
            raise ValueError(
                f"build_model: density must be 'ideal_gas' or 'linear', got {density!r}"
            )
    if species:
        layers["species"] = species_layer(net, ambient=ambient, flow_kinds=kinds,
                                          n_species=int(species), scheme=species_scheme)
    return Model(net, layers, closures=closures, coupling=coupling, iterate_tol=iterate_tol,
                 iterate_max=iterate_max)


def initial_state(model: Model) -> State:
    """`"<layer>.x"` for every transport layer: temperatures from the node attribute `T0`
    (interior order), mass fractions zero.

    Keyed by each layer's OWN name, and dispatched on its `quantity` tag rather than on the
    names `"thermal"`/`"species"`: a layer renamed through `thermal_layer(name=...)` must
    still get its state, and must never be answered with a silently EMPTY state that only
    surfaces later as `Model`'s "state '<name>.x' is required" from inside a step. A
    transport layer whose `quantity` this application does not know is refused by name for
    the same reason.
    """
    state: State = {}
    for name, layer in model.transport.items():
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
                f"{name + '.x'!r} yourself"
            )
    return state
