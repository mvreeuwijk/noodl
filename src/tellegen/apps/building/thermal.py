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
    """A lumped wall node: capacity [J/K], conductances to the zone and to ambient [W/K]."""

    name: str
    capacity: float
    ua_zone: float
    ua_ambient: float


@dataclass
class Zone:
    name: str
    volume: float
    T0: float = T_REF
    z_ref: float = 0.0
    wall: WallMass | None = None


def add_zone(net: Network, zone: Zone, *, ambient="ambient") -> None:
    """Add the zone's air node (attributes volume, T0, z_ref, heat_capacity=0) and, if it has
    a wall, the wall node (heat_capacity) with 'wall' edges zone -> wall -> ambient (ua)."""
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
    """Layers "air" (+ "thermal", + "species"), the density closure, one Model."""
    kinds = tuple(flow_kinds) if flow_kinds else tuple(el.kind for el in air_elements)
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
    """"thermal.x" from the node attribute T0 (interior order), "species.x" zeros."""
    state: State = {}
    th = model.layers.get("thermal")
    if isinstance(th, TransportLayer):
        state["thermal.x"] = model.net.node_attr("T0", default=T_REF)[th.interior_idx]
    sp = model.layers.get("species")
    if isinstance(sp, TransportLayer):
        shape = (sp.n_i,) if sp.n_species == 1 else (sp.n_i, sp.n_species)
        state["species.x"] = torch.zeros(shape, dtype=model.net.dtype)
    return state
