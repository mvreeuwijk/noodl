"""CONTAM's simple air-handling system, expanded into the network.

A supply plenum and a return plenum (nodes, with their volumes) are joined by edges of one
kind: supply points (plenum -> zone), return points (zone -> plenum), the recirculation
(return -> supply plenum), the outdoor air (ambient -> supply plenum) and the exhaust (return
plenum -> ambient). Every edge's flow is prescribed. `AirHandlerFlows` computes the flows
each step from the supply and return points' design flows times their multipliers (a
schedule or a control) and from the outdoor-air fraction, by CONTAM's algorithm (TN 1887
section 3.6, verified against ContamX 3.4.1.7):

    Ws, Wret = the supplied and the returned flow
    W'o  = max(f_o Ws, min(Ws, Wo_min))
    Wrec = min(Wret, Ws - W'o),   Wo = Ws - Wrec,   Wex = Wret - Wrec

Only these balanced flows touch the plenums, so their pressure is not solved: give them to
`build_model(flow_balanced=...)`. A plenum of zero volume holds no air and needs
`storage_free=...` too.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.apps.building_physics.thermal import T_REF
from noodl.components import Component
from noodl.elements import PrescribedFlow

Tensor = torch.Tensor
F64 = torch.float64

SUPPLY, RETURN, RECIRCULATION, OUTDOOR_AIR, EXHAUST = 0, 1, 2, 3, 4


class AirHandlerFlows:
    """The closure that prescribes every flow of one air handler each step, by CONTAM's
    algorithm (see the module) from its design flows and its outdoor-air fraction.

    `roles` (one code per edge of `kind`, in edge order) and `design_flows` (kg/s; the
    supply and return points' design flows, ignored on the other edges). Reads
    `drivers["flow_multiplier.<kind>"]` (per edge; the supply and return entries scale their
    design flows; ones when absent) and `drivers["outdoor_air_fraction.<kind>"]` (f_o;
    `outdoor_air_fraction` when absent). Writes `drivers["prescribed_flow.<kind>"]`, the
    flow of every edge in its own direction (kg/s, never negative). A negative supply or
    return point (multiplier times design flow) is refused, naming the edges; an
    outdoor-air fraction above 1 makes the recirculation negative, which ContamX refuses
    too: it raises naming the handler.
    """

    def __init__(self, kind: str, *, roles, design_flows, outdoor_air_fraction: float = 1.0,
                 min_outdoor_air: float = 0.0, name: str = "") -> None:
        self.kind, self.name = kind, name or kind
        self.roles = _role_codes(roles, f"AirHandlerFlows {self.name!r}")
        self.design_flows = torch.as_tensor(design_flows, dtype=F64).reshape(-1)
        if self.design_flows.shape != self.roles.shape:
            raise ValueError(f"AirHandlerFlows {self.name!r}: design_flows must give one "
                             f"value per edge ({self.roles.numel()} roles), got "
                             f"{self.design_flows.numel()}")
        for role, label in ((RECIRCULATION, "recirculation"), (OUTDOOR_AIR, "outdoor-air"),
                            (EXHAUST, "exhaust")):
            if int((self.roles == role).sum()) != 1:
                raise ValueError(f"AirHandlerFlows {self.name!r}: needs exactly one "
                                 f"{label} edge, got {int((self.roles == role).sum())}")
        self.outdoor_air_fraction = float(outdoor_air_fraction)
        self.min_outdoor_air = float(min_outdoor_air)
        self.multiplier_key = f"flow_multiplier.{kind}"
        self.fraction_key = f"outdoor_air_fraction.{kind}"
        self.flow_key = f"prescribed_flow.{kind}"
        self.inputs = (self.multiplier_key, self.fraction_key)
        self.outputs = (self.flow_key,)
        self.output_units = {self.flow_key: "kg/s"}

    @property
    def input_specs(self) -> dict[str, dict]:
        """The schedule- or control-driven inputs this closure reads when given."""
        return {
            self.multiplier_key: dict(
                description="multiplier of each supply and return point's design flow (a "
                            "schedule or control)", unit="-", over=("edges", self.kind),
                required=False),
            self.fraction_key: dict(
                description=f"outdoor-air fraction of the supply (default "
                            f"{self.outdoor_air_fraction:g})", unit="-", required=False),
        }

    def __call__(self, state, drivers) -> dict[str, Tensor]:
        design = self.design_flows
        m = drivers.get(self.multiplier_key)
        m = torch.ones_like(design) if m is None else torch.as_tensor(m).to(design.dtype)
        f_o = drivers.get(self.fraction_key)
        f_o = (torch.tensor(self.outdoor_air_fraction, dtype=design.dtype) if f_o is None
               else torch.as_tensor(f_o).to(design.dtype))
        # A per-instance `(B, 1)` column, not the species layout (no node axis here), so
        # `noodl.layout` does not apply.
        if f_o.dim() == 2 and f_o.shape[-1] == 1:
            f_o = f_o.squeeze(-1)                   # (B, 1), a column of instances: (B,)
        if f_o.dim() > 1:
            raise ValueError(
                f"AirHandlerFlows {self.name!r}: outdoor-air fraction must be one value per "
                f"instance (shape () or (B,), or (B, 1)), got shape {tuple(f_o.shape)}")
        points = design * m
        is_point = (self.roles == SUPPLY) | (self.roles == RETURN)
        negative = ((points.detach() < 0) & is_point).reshape(-1, points.shape[-1]).any(0)
        if bool(negative.any()):
            edges = negative.nonzero().reshape(-1).tolist()
            raise ValueError(
                f"AirHandlerFlows {self.name!r}: supply or return points {edges} (edge "
                f"positions in kind {self.kind!r}) have a negative flow (multiplier times "
                f"design flow); a point's flow must not be negative")
        supply = (self.roles == SUPPLY).to(design.dtype)
        ret = (self.roles == RETURN).to(design.dtype)
        Ws = (points * supply).sum(-1)
        Wret = (points * ret).sum(-1)
        Wo_min = torch.tensor(self.min_outdoor_air, dtype=design.dtype)
        Wo_req = torch.maximum(f_o * Ws, torch.minimum(Ws, Wo_min))
        Wrec = torch.minimum(Wret, Ws - Wo_req)
        if bool((Wrec.detach() < 0).any()):
            raise ValueError(
                f"AirHandlerFlows {self.name!r}: outdoor-air fraction "
                f"{f_o.detach().tolist()} makes the recirculation negative "
                f"({Wrec.detach().tolist()} kg/s); it must not exceed 1")
        Wo, Wex = Ws - Wrec, Wret - Wrec
        roles = self.roles
        out = torch.where(is_point, points, torch.zeros_like(points))
        out = torch.where(roles == RECIRCULATION, Wrec.unsqueeze(-1), out)
        out = torch.where(roles == OUTDOOR_AIR, Wo.unsqueeze(-1), out)
        out = torch.where(roles == EXHAUST, Wex.unsqueeze(-1), out)
        return {self.flow_key: out}


def _role_codes(roles, who: str) -> Tensor:
    """`roles` as integer codes, refusing a non-integer value or a code outside 0..4."""
    r = torch.as_tensor(roles).reshape(-1)
    if r.is_floating_point():
        bad = (~torch.isfinite(r)) | (r != r.round())
        if bool(bad.any()):
            raise ValueError(f"{who}: role codes must be integers 0..4, got "
                             f"{r[bad].tolist()} at edge positions "
                             f"{bad.nonzero().reshape(-1).tolist()}")
    r = r.to(torch.long)
    bad = (r < SUPPLY) | (r > EXHAUST)
    if bool(bad.any()):
        raise ValueError(f"{who}: unknown role codes {r[bad].tolist()} at edge positions "
                         f"{bad.nonzero().reshape(-1).tolist()}; the codes are 0 supply, "
                         f"1 return, 2 recirculation, 3 outdoor air, 4 exhaust")
    return r


def add_air_handler(target, *, supplies: Mapping, returns: Mapping, kind: str = "air_handler",
                    supply_plenum="ahs_supply", return_plenum="ahs_return",
                    outdoor_air_fraction: float = 1.0, min_outdoor_air: float = 0.0,
                    supply_volume: float = 0.0, return_volume: float = 0.0,
                    T0: float = T_REF, ambient="ambient") -> None:
    """Add an air handler to `target` (a `Network` or a `Component`): its two plenums
    (volumes in m3, temperature `T0` in K) and its edges of `kind`. `supplies` maps each
    supplied zone to its design flow (kg/s), `returns` each zone it draws from.
    `outdoor_air_fraction` and `min_outdoor_air` (kg/s) are the defaults the closure uses
    when no input sets them. In a component the outdoor air and the exhaust reach the
    shared ambient through `outer(ambient)`. Give each air handler of a building its own
    `kind`."""
    amb = target.outer(ambient) if isinstance(target, Component) else ambient
    target.add_node(supply_plenum, volume=float(supply_volume), T0=float(T0), z_ref=0.0,
                    heat_capacity=0.0)
    target.add_node(return_plenum, volume=float(return_volume), T0=float(T0), z_ref=0.0,
                    heat_capacity=0.0)
    for zone, flow in supplies.items():
        target.add_edge(supply_plenum, zone, kind=kind, role=float(SUPPLY),
                        design_flow=float(flow))
    for zone, flow in returns.items():
        target.add_edge(zone, return_plenum, kind=kind, role=float(RETURN),
                        design_flow=float(flow))
    target.add_edge(return_plenum, supply_plenum, kind=kind, role=float(RECIRCULATION),
                    design_flow=0.0, outdoor_air_fraction=float(outdoor_air_fraction),
                    min_outdoor_air=float(min_outdoor_air))
    target.add_edge(amb, supply_plenum, kind=kind, role=float(OUTDOOR_AIR), design_flow=0.0)
    target.add_edge(return_plenum, amb, kind=kind, role=float(EXHAUST), design_flow=0.0)


def air_handler_from_edges(net, kind: str) -> tuple[PrescribedFlow, AirHandlerFlows]:
    """The law and the closure of the air handler whose edges are of `kind` (as
    `add_air_handler`, or the `.prj` reader, made them)."""
    roles = _role_codes(net.edge_attr("role", kind), f"air handler {kind!r}")
    recirculation = roles == RECIRCULATION
    if int(recirculation.sum()) != 1:
        raise ValueError(f"air handler {kind!r}: needs exactly one recirculation edge "
                         f"(role 2), got {int(recirculation.sum())}")
    fraction = net.edge_attr("outdoor_air_fraction", kind, default=1.0)[recirculation]
    minimum = net.edge_attr("min_outdoor_air", kind, default=0.0)[recirculation]
    closure = AirHandlerFlows(kind, roles=roles, design_flows=net.edge_attr("design_flow", kind),
                              outdoor_air_fraction=float(fraction.reshape(-1)[0]),
                              min_outdoor_air=float(minimum.reshape(-1)[0]), name=kind)
    return PrescribedFlow(closure.flow_key, kind=kind, dtype=net.dtype), closure


def air_handler_plenums(net, kind: str) -> tuple:
    """(supply plenum, return plenum) of the air handler of `kind`: the recirculation
    edge's target and source."""
    roles = _role_codes(net.edge_attr("role", kind), f"air handler {kind!r}")
    src, tgt = net.endpoints(kind)
    found = (roles == RECIRCULATION).nonzero().reshape(-1)
    if found.numel() != 1:
        raise ValueError(f"air handler {kind!r}: its plenums are the ends of its one "
                         f"recirculation edge (role 2), but it has {found.numel()}")
    k = int(found[0])
    return net.nodes[int(tgt[k])], net.nodes[int(src[k])]
