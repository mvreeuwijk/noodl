"""SWMM 5.2.4 dynamic-wave routing for the sewer application.

`DynamicWaveOptions` holds the `[OPTIONS]` that change the physics. The model is a
`PotentialFlowLayer` ``"water"`` (quantity ``"head"``, m) whose unknowns are the junction
heads: every conduit is a `DynamicWaveConduit`, q = c0 + k (H1 - H2) with the coefficients of
SWMM's momentum update (`dynwave_terms.conduit_terms`) frozen for one pass, and every junction
carries a `JunctionStorage`, SWMM's trapezoidal continuity
``A_s (y - y_start) = 0.5 dt (N_start + N)``. `DynamicWaveHydraulics` (a closure) refreshes
the coefficients, surface areas and outfall heads from the previous pass, and
``Model(coupling="iterate")`` repeats the pass until the heads stop changing: the fixed point
is SWMM's own converged trial.

This module must not import `network.py` at module level: `network.py` imports
`DynamicWaveOptions` and `build_dynamic_wave_model` from here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from noodl.apps.sewer.dynwave_terms import (
    DAMPING,
    FUDGE,
    NORMAL_FLOW_LIMITED,
    ConduitGeometry,
    ConduitTerms,
    conduit_terms,
)
from noodl.apps.sewer.xsect_tables import (
    LCF,
    QCF,
    area_of_depth,
    critical_depth,
    normal_depth_ft,
)
from noodl.elements.base import Element
from noodl.layers.potential import PotentialFlowLayer
from noodl.model import Model, StepContext
from noodl.nodesources import NodeSource
from noodl.topology import Network

if TYPE_CHECKING:
    from noodl.apps.sewer.network import SewerNetwork

Tensor = torch.Tensor
F64 = torch.float64
State = dict[str, Tensor]
Drivers = dict[str, Tensor]

#: SWMM's dry-node flow cap (`dwflow.c:273-276`): 1e-4 cfs, in m3/s.
FUDGE_FLOW = 1.0e-4 * QCF
#: Bed stiffening factor of `JunctionStorage` (design D3): below its invert a junction's
#: storage slope gains ``(2 A_bed/dt) K_BED``, standing in for SWMM's ``yNew < 0 -> 0``.
K_BED = 1.0e6
#: Newton's absolute residual tolerance for the head solve (m3/s): a junction balance error
#: of `SOLVE_ATOL` dt m3 at most, far below the head tolerance times any surface area. It is
#: raised to the bed's round-off floor where that is larger (`solve_atol`).
SOLVE_ATOL = 1.0e-12
#: `solve_atol`'s margin over the bed's round-off floor (see there).
_BED_FLOOR_MARGIN = 4.0
#: SWMM's default junction surface-area floor (`dynwave.c:62`), ft2.
DEFAULT_MIN_SURFACE_AREA_FT2 = 12.566
#: SI area per SWMM ft2 (design D2): makes ``A_s dy = 0.5 dt N`` exact in m, m3/s.
_AREA_SI = QCF / LCF
#: SWMM's link velocity is 0 at or below this link depth (`link.c:834`), ft.
_VELOCITY_MIN_DEPTH = 0.01
_OUTFALL_CODES = {"free": 0, "normal": 1, "fixed": 2}

#: ``SURCHARGE_METHOD`` values accepted now. SWMM also has ``SLOT`` (Preissmann slot), which
#: changes surface areas below the crown and is not modelled yet.
SURCHARGE_METHODS = ("extran",)


@dataclass(frozen=True)
class DynamicWaveOptions:
    """SWMM 5.2.4's dynamic-wave [OPTIONS] that change the physics, with SWMM's defaults
    (`project.c:841-880`). `head_tolerance` and `max_trials` are SWMM's own iteration
    controls: recorded so a file round-trips, never used -- noodl solves the same
    equations to `build_model`'s own `iterate_tol` (docs: "Dynamic wave").

    `routing_step` is the fixed routing step (s); `inertial_damping` is ``"none"``,
    ``"partial"`` or ``"full"``; `normal_flow_limited` is ``"slope"``, ``"froude"``,
    ``"both"`` or ``"none"``; `surcharge_method` is ``"extran"`` (``"slot"`` is refused for
    now); `min_surface_area` is the junction surface-area floor (m2), ``None`` for SWMM's
    12.566 ft2; `head_tolerance` is SWMM's head convergence tolerance (m); `max_trials` is
    SWMM's trial limit (-).
    """

    routing_step: float = 20.0
    inertial_damping: str = "partial"
    normal_flow_limited: str = "both"
    surcharge_method: str = "extran"
    min_surface_area: float | None = None
    head_tolerance: float | None = None
    max_trials: int | None = None

    def validate(self) -> None:
        """Refuse an unknown keyword or a non-positive step or area, BY NAME."""
        if not self.routing_step > 0:
            raise ValueError(
                f"DynamicWaveOptions: routing_step must be > 0 s, got {self.routing_step!r}"
            )
        if self.surcharge_method == "slot":
            raise ValueError(
                "DynamicWaveOptions: surcharge_method 'slot' is not modelled yet: SWMM's "
                "Preissmann slot changes surface areas below the crown"
            )
        for name, allowed in (
            ("inertial_damping", DAMPING),
            ("normal_flow_limited", NORMAL_FLOW_LIMITED),
            ("surcharge_method", SURCHARGE_METHODS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"DynamicWaveOptions: {name} must be one of {allowed}, got "
                    f"{getattr(self, name)!r}"
                )
        if self.min_surface_area is not None and not self.min_surface_area > 0:
            raise ValueError("DynamicWaveOptions: min_surface_area must be > 0 m2")


class DynamicWaveConduit(Element):
    """SWMM's dynamic-wave conduit (`dwflow.c`) with its coefficients frozen for one pass.

    ``q = c0 + k dp`` (m3/s, dp = H1 - H2 in m), then ``min(q, q_norm)`` where SWMM's
    normal-flow check passed (`dwflow.c:255-258`), then the dry-node caps in SWMM's order
    (`dwflow.c:273-276`): at most +`FUDGE_FLOW` out of a dry upstream node, at least
    -`FUDGE_FLOW` out of a dry downstream node. Reads the per-pipe drivers
    ``"water.flow_offset"`` c0 (m3/s), ``"water.conveyance"`` k (m2/s),
    ``"water.normal_flow"`` q_norm (m3/s, +inf where not limited), and the 0/1 flags
    ``"water.limited"``, ``"water.dry_upstream"``, ``"water.dry_downstream"``, all written by
    `DynamicWaveHydraulics`.
    """

    def __init__(self, kind: str = "pipe") -> None:
        super().__init__(kind)

    def _parts(self, dp: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor]:
        lin = drivers["water.flow_offset"] + drivers["water.conveyance"] * dp
        limited = drivers["water.limited"] > 0.5
        q = torch.where(limited, torch.minimum(lin, drivers["water.normal_flow"]), lin)
        up = drivers["water.dry_upstream"] > 0.5
        dn = drivers["water.dry_downstream"] > 0.5
        q = torch.where(up, torch.clamp(q, max=FUDGE_FLOW), q)      # dwflow.c:273-274
        q = torch.where(dn, torch.clamp(q, min=-FUDGE_FLOW), q)     # dwflow.c:275-276
        return lin, q

    def flow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """Conduit flow (m3/s, positive from the pipe's first node to its second)."""
        return self._parts(dp, drivers)[1]

    def dflow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """dq/d(dp) (m2/s): k where no limit is active, else 0 (every limit is a constant,
        so ``q == lin`` is an exact test)."""
        lin, q = self._parts(dp, drivers)
        k = drivers["water.conveyance"].expand_as(q)
        return torch.where(q == lin, k, torch.zeros_like(k))


class JunctionStorage(NodeSource):
    """SWMM's junction continuity as a node withdrawal: trapezoidal storage over the step,
    with a stiff bed below the invert (design D3).

    ``w(H) = (2 A_s/dt)(H - H_start) - N_start + (2 A_bed/dt) K_BED min(H - z_bed, 0)``
    (m3/s, positive out of the network), so that with the conduit flows and the lateral
    inflow the node balance reads ``A_s (H - H_start) = 0.5 dt (N_start + N)``
    (`dynwave.c:694-711`). The bed term stands in for SWMM's ``yNew < 0 -> 0``
    (`dynwave.c:743`); the slope is at least ``2 A_s/dt > 0``, so every junction grounds
    itself. The bed stiffness uses the fixed `bed_area` A_bed (m2; the model passes SWMM's
    surface-area floor, which no A_s is below), not A_s, so that the head solve's round-off
    floor on the bed is known before the step (`solve_atol`). Reads the per-junction drivers
    ``"water.storage_area"`` A_s (m2, design D2's SI area), ``"water.head_start"`` (m),
    ``"water.net_inflow_start"`` (m3/s) and ``"water.storage_rate"`` 1/dt (1/s). `bed`
    holds the junction inverts (m).
    """

    def __init__(self, nodes: Tensor, bed: Tensor, bed_area: float) -> None:
        super().__init__(nodes)
        self.register_buffer("bed", torch.as_tensor(bed, dtype=F64))
        self.bed_area = float(bed_area)

    def flow(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        """Withdrawal (m3/s) at junction heads `phi_nodes` (m)."""
        rate2 = 2.0 * drivers["water.storage_rate"]
        dry = torch.clamp(phi_nodes - self.bed, max=0.0)
        return (rate2 * drivers["water.storage_area"] * (phi_nodes - drivers["water.head_start"])
                - drivers["water.net_inflow_start"] + rate2 * self.bed_area * K_BED * dry)

    def dflow(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        """d(withdrawal)/dH (m2/s), analytic and strictly positive."""
        below = (phi_nodes < self.bed).to(phi_nodes.dtype)
        rate2 = 2.0 * drivers["water.storage_rate"]
        slope = rate2 * (drivers["water.storage_area"] + self.bed_area * K_BED * below)
        return slope.expand(torch.broadcast_shapes(slope.shape, phi_nodes.shape))


class DynamicWaveHydraulics:
    """The dynamic-wave closure: the conduit coefficients, junction surface areas and
    outfall heads of one pass, from the previous pass's heads and flows.

    Reads the fed ``"water.phi"`` (m, full-node) and ``"water.q"`` (m3/s, per pipe), the
    full-node driver ``"inflow"`` (m3/s, lateral inflow, zero at outfalls) and its own
    carried state, pinned by `Model` to the step start: ``"sewer.junction_depth"`` (m),
    ``"sewer.pipe_flow"`` (m3/s), ``"sewer.mid_area"`` (SWMM's stored mid-depth area, SI
    area m2 per design D2) and ``"sewer.net_inflow"`` (m3/s, per junction). With a
    `StepContext` it writes ``"water.sources"``, ``"water.phi_boundary"`` (outfall heads, m),
    the six `DynamicWaveConduit` drivers and the four `JunctionStorage` drivers; always the
    carried keys and the reporting keys ``"sewer.q"`` (m3/s), ``"sewer.h"`` (link depth, m),
    ``"sewer.v"`` (m/s), ``"sewer.V_wet"`` (m3) and ``"sewer.junction_depth"`` (m).

    The arithmetic is SWMM's own, in feet (design D2); `layer` names the head layer.
    """

    integrates = True
    state_keys = ("sewer.junction_depth", "sewer.pipe_flow", "sewer.mid_area",
                  "sewer.net_inflow")
    edge_kinds = ("pipe",)

    def __init__(self, graph: Network, net: SewerNetwork, *, layer: str = "water") -> None:
        self.net = graph
        self.layer = layer
        self.options = net.dynamic_wave
        self.notes: dict[str, str] = {}
        nodes = list(graph.nodes)
        self.pipe_names = [p.name for p in net.pipes]
        self.junction_names = [m.name for m in net.manholes]
        self.outfall_names = [o.name for o in net.outfalls]
        self.key_labels = {
            "sewer.junction_depth": list(self.junction_names),
            "sewer.pipe_flow": list(self.pipe_names),
            "sewer.mid_area": list(self.pipe_names),
            "sewer.net_inflow": list(self.junction_names),
        }
        self.node_invert = torch.tensor(
            [float(graph.graph.nodes[n]["invert"]) for n in nodes], dtype=F64)
        self.junction_idx = torch.tensor(
            [nodes.index(n) for n in self.junction_names], dtype=torch.long)
        self.outfall_idx = torch.tensor(
            [nodes.index(n) for n in self.outfall_names], dtype=torch.long)
        is_outfall = torch.zeros(len(nodes), dtype=torch.bool)
        is_outfall[self.outfall_idx] = True
        self.is_outfall = is_outfall
        self.u = torch.tensor([nodes.index(p.u) for p in net.pipes], dtype=torch.long)
        self.v = torch.tensor([nodes.index(p.v) for p in net.pipes], dtype=torch.long)
        self.diameter = torch.tensor([p.diameter for p in net.pipes], dtype=F64)
        self.roughness = torch.tensor([p.n for p in net.pipes], dtype=F64)
        self.slope = torch.tensor([p.slope for p in net.pipes], dtype=F64)
        self.length = torch.tensor([p.length for p in net.pipes], dtype=F64)
        self.offset1 = torch.tensor([p.inlet_offset for p in net.pipes], dtype=F64)
        self.offset2 = torch.tensor([p.outlet_offset for p in net.pipes], dtype=F64)
        self._rebuild_geometry()

        # Each outfall's own conduit and the end it sits at (link.c:740-751: node2 first).
        pipe_of, at_v = [], []
        for name in self.outfall_names:
            ends = [(i, p.v == name) for i, p in enumerate(net.pipes) if name in (p.u, p.v)]
            if len(ends) != 1:
                raise ValueError(
                    f"DynamicWaveHydraulics: outfall {name!r} is attached to {len(ends)} "
                    f"conduits; dynamic-wave routing takes exactly one")
            i, downstream = ends[0]
            pipe = net.pipes[i]
            if pipe.u in self.outfall_names and pipe.v in self.outfall_names:
                raise ValueError(
                    f"DynamicWaveHydraulics: conduit {pipe.name!r} joins two outfalls "
                    f"({pipe.u!r}, {pipe.v!r}); SWMM sets the depth of only one of them")
            pipe_of.append(i)
            at_v.append(downstream)
        self.outfall_pipe = torch.tensor(pipe_of, dtype=torch.long)
        at_v_t = torch.tensor(at_v, dtype=torch.bool)
        self.outfall_offset_ft = torch.where(
            at_v_t, self.offset2[self.outfall_pipe], self.offset1[self.outfall_pipe]) / LCF
        self.outfall_kind = torch.tensor(
            [_OUTFALL_CODES[o.kind] for o in net.outfalls], dtype=torch.long)
        self.outfall_stage_ft = torch.tensor(
            [(o.stage if o.stage is not None else o.invert) / LCF for o in net.outfalls],
            dtype=F64)
        self.min_area_ft2 = (
            DEFAULT_MIN_SURFACE_AREA_FT2 if self.options.min_surface_area is None
            else self.options.min_surface_area / LCF**2)            # dynwave.c:186-187

    # ------------------------------------------------------------------ geometry
    def _rebuild_geometry(self) -> None:
        self.geo = ConduitGeometry.from_si(
            self.diameter, self.roughness, self.slope, self.length,
            self.node_invert[self.u], self.node_invert[self.v], self.offset1, self.offset2,
            self.is_outfall[self.u], self.is_outfall[self.v])

    def set_roughness(self, i: int, value) -> None:
        """Set Manning's n (s m^(-1/3)) of pipe `i` to `value`, which may be a tensor that
        requires grad; the geometry is rebuilt so the gradient reaches it. The other pipes'
        n are detached, so only the latest call's `value` carries a gradient."""
        value = torch.as_tensor(value, dtype=F64).reshape(1)
        self.roughness = torch.cat([self.roughness[:i].detach(), value,
                                    self.roughness[i + 1:].detach()])
        self._rebuild_geometry()

    # ------------------------------------------------------------------- helpers
    def _inflow(self, drivers: Mapping) -> Tensor:
        try:
            inflow = drivers["inflow"]
        except KeyError as exc:
            raise KeyError("DynamicWaveHydraulics: driver 'inflow' (m3/s, full-node lateral "
                           "inflow) is required") from exc
        if not bool(torch.isfinite(inflow).all()):
            bad = ~torch.isfinite(inflow).reshape(-1, inflow.shape[-1]).all(0)
            names = [self.net.nodes[i] for i in bad.nonzero().flatten().tolist()]
            raise ValueError(f"DynamicWaveHydraulics: non-finite inflow at {names}")
        at_outfalls = inflow[..., self.outfall_idx]
        if bool((at_outfalls != 0).any()):
            bad = (at_outfalls != 0).reshape(-1, len(self.outfall_names)).any(0)
            names = [self.outfall_names[i] for i in bad.nonzero().flatten().tolist()]
            raise ValueError(
                f"DynamicWaveHydraulics: inflow at outfall(s) {names} must be 0; an outfall "
                f"head is set by its conduit, not by a node balance")
        return inflow

    def _terms(self, state: Mapping, dt: float) -> ConduitTerms:
        heads = state[f"{self.layer}.phi"]
        q = state[f"{self.layer}.q"]
        # SWMM's node depth is never negative (dynwave.c:743); the stiff bed lets a dry
        # junction's head sit a hair below its invert.
        depth_ft = torch.clamp(heads - self.node_invert, min=0.0) / LCF
        return conduit_terms(
            self.geo, depth_ft[..., self.u], depth_ft[..., self.v], q / QCF,
            state["sewer.pipe_flow"] / QCF, state["sewer.mid_area"] / _AREA_SI, dt,
            damping=self.options.inertial_damping,
            normal_flow_limited=self.options.normal_flow_limited)

    def _carried(self, state: Mapping, terms: ConduitTerms, inflow: Tensor) -> State:
        heads = state[f"{self.layer}.phi"]
        q = state[f"{self.layer}.q"]
        jidx = self.junction_idx
        n = self.node_invert.shape[-1]
        batch = torch.broadcast_shapes(q.shape[:-1], inflow.shape[:-1])
        through = torch.zeros(batch + (n,), dtype=F64)
        q_b = q.expand(batch + q.shape[-1:])
        through = through.index_add(-1, self.v, q_b).index_add(-1, self.u, -q_b)
        net_inflow = inflow + through                                  # dynwave.c:528-541
        return {
            "sewer.junction_depth": torch.clamp(heads[..., jidx] - self.node_invert[jidx],
                                                min=0.0),
            "sewer.pipe_flow": q,
            "sewer.mid_area": terms.mid_area * _AREA_SI,
            "sewer.net_inflow": net_inflow[..., jidx],
        }

    def _outfall_heads(self, q: Tensor) -> Tensor:
        """`link.c::link_setOutfallDepth` + `node.c::outfall_setOutletDepth` (link.c:728-766,
        node.c:1413-1492), from each outfall conduit's flow `q` (m3/s): heads (m)."""
        op = self.outfall_pipe
        geo = self.geo
        q_ft = (q[..., op] / QCF).abs()
        yf, af = geo.y_full[op], geo.a_full[op]
        y_norm = normal_depth_ft(q_ft, yf, af, geo.s_full[op], geo.beta[op])
        y_crit = critical_depth(q_ft, yf, af)
        z = self.outfall_offset_ft
        zero = torch.zeros_like(y_norm)
        free = torch.where(z > 0.0, zero, torch.minimum(y_norm, y_crit))      # node.c:1431
        normal = torch.where(z > 0.0, zero, y_norm)                           # node.c:1436
        invert = self.node_invert[self.outfall_idx] / LCF
        stage = self.outfall_stage_ft
        yc = torch.minimum(y_crit, y_norm)                                    # node.c:1469
        fixed = torch.where(
            yc + z + invert < stage, (stage - invert).expand_as(yc),         # node.c:1473
            torch.where(
                z > 0.0,
                torch.where(stage < invert + z,
                            torch.clamp(stage - invert, min=0.0).expand_as(yc),
                            z + yc),                                          # node.c:1481-1485
                yc))                                                          # node.c:1490
        kind = self.outfall_kind
        depth = torch.where(kind == 0, free, torch.where(kind == 1, normal, fixed))
        return self.node_invert[self.outfall_idx] + depth * LCF

    # ------------------------------------------------------------------ protocol
    def __call__(self, state: Mapping, drivers: Mapping,
                 ctx: StepContext | None = None) -> dict[str, Tensor]:
        """One pass's drivers (see the class docstring); a query (no `ctx`, or
        ``ctx.dt is None``) returns the carried and reporting keys only."""
        dt = None if ctx is None else ctx.dt
        inflow = self._inflow(drivers)
        terms = self._terms(state, self.options.routing_step if dt is None else dt)
        out = self._carried(state, terms, inflow)
        q = state[f"{self.layer}.q"]
        depth_ft = terms.depth
        a_link = area_of_depth(depth_ft, self.geo.y_full, self.geo.a_full)
        moving = (depth_ft > _VELOCITY_MIN_DEPTH) & (a_link > FUDGE)          # link.c:834-840
        v_ft = (q / QCF) / torch.where(moving, a_link, torch.ones_like(a_link))
        out.update({
            "sewer.q": q,
            "sewer.h": depth_ft * LCF,
            "sewer.v": torch.where(moving, v_ft, torch.zeros_like(v_ft)) * LCF,
            "sewer.V_wet": terms.volume * QCF,
        })
        if dt is None:
            return out
        jidx = self.junction_idx
        n = self.node_invert.shape[-1]
        surf = torch.zeros(terms.surface_area1.shape[:-1] + (n,), dtype=F64)
        surf = surf.index_add(-1, self.u, terms.surface_area1)                # dynwave.c:582-583
        surf = surf.index_add(-1, self.v, terms.surface_area2)
        area_ft2 = torch.clamp(surf[..., jidx], min=self.min_area_ft2)        # dynwave.c:670
        out.update({
            f"{self.layer}.sources": inflow,
            f"{self.layer}.phi_boundary": self._outfall_heads(q),
            "water.flow_offset": terms.flow_offset * QCF,
            "water.conveyance": terms.conveyance * (QCF / LCF),
            "water.normal_flow": terms.normal_flow * QCF,
            "water.limited": terms.limited.to(F64),
            "water.dry_upstream": terms.dry_upstream.to(F64),
            "water.dry_downstream": terms.dry_downstream.to(F64),
            "water.storage_area": area_ft2 * _AREA_SI,
            "water.storage_rate": torch.tensor(1.0 / dt, dtype=F64),
            "water.head_start": self.node_invert[jidx] + state["sewer.junction_depth"],
            "water.net_inflow_start": state["sewer.net_inflow"],
        })
        return out

    def carried(self, state: Mapping, drivers: Mapping, *, dt: float | None = None) -> State:
        """The four carried keys evaluated AT `state` (``water.phi``, ``water.q``), for
        re-synchronising after a step of `dt` s (default: the routing step): the pass's own
        copy is its input's, which equals its output only to the iteration tolerance."""
        terms = self._terms(state, self.options.routing_step if dt is None else dt)
        return self._carried(state, terms, self._inflow(drivers))

    def initial_state(self, drivers: Mapping | None = None) -> State:
        """SWMM's cold, dry start (flowrout.c:75-102, 336-507): no flow, junctions at their
        inverts, outfalls at the depth `link_setOutfallDepth` gives at zero flow (a FIXED
        outfall at its stage, the others at their invert; flowrout.c:383), every conduit at
        the mean of its end-node depths above its offsets, clamped to [0, yFull] and at
        least FUDGE (flowrout.c:389-420), which sets its stored area (flowrout.c:497), and a
        zero junction net inflow. SWMM zeroes the lateral inflow before routing starts
        (node.c:264, flowrout.c:437), so the first step's ``oldNetInflow`` is 0
        (routing.c:236-242, node.c:303) and its trapezoid sees only the end-of-step inflow.
        `drivers`, when given, has its ``"inflow"`` validated."""
        if drivers is not None:
            self._inflow(drivers)
        q0 = torch.zeros(len(self.pipe_names), dtype=F64)
        phi = self.node_invert.clone()
        phi[self.outfall_idx] = self._outfall_heads(q0)
        geo = self.geo
        depth_ft = (phi - self.node_invert) / LCF
        y1 = torch.clamp(depth_ft[self.u] - geo.offset1, min=0.0)
        y2 = torch.clamp(depth_ft[self.v] - geo.offset2, min=0.0)
        y = torch.clamp(0.5 * (torch.minimum(y1, geo.y_full) + torch.minimum(y2, geo.y_full)),
                        min=FUDGE)                                            # :402-416
        return {
            f"{self.layer}.phi": phi,
            f"{self.layer}.q": q0,
            "sewer.junction_depth": torch.zeros(len(self.junction_names), dtype=F64),
            "sewer.pipe_flow": torch.zeros(len(self.pipe_names), dtype=F64),
            "sewer.mid_area": area_of_depth(y, geo.y_full, geo.a_full) * _AREA_SI,  # :497
            "sewer.net_inflow": torch.zeros(len(self.junction_names), dtype=F64),
        }

    def check_free_surface(self, state: Mapping, drivers: Mapping, t: float) -> None:
        """Refuse, naming the time and conduits, any conduit end whose depth reached the
        crown: surcharge is not modelled yet (phase B)."""
        heads = state[f"{self.layer}.phi"].detach() / LCF
        geo = self.geo
        z1 = geo.invert1 + geo.offset1
        z2 = geo.invert2 + geo.offset2
        y1 = torch.maximum(heads[..., self.u], z1) - z1                       # dwflow.c:110-120
        y2 = torch.maximum(heads[..., self.v], z2) - z2
        full = (y1 >= geo.y_full.detach() - 1e-12) | (y2 >= geo.y_full.detach() - 1e-12)
        if bool(full.any()):
            hit = full.reshape(-1, full.shape[-1]).any(0)
            names = [self.pipe_names[i] for i in hit.nonzero().flatten().tolist()]
            raise RuntimeError(
                f"dynamic wave: surcharge at t={t} s in conduit(s) {names} (end depth "
                f"reached the crown); surcharge is not modelled yet (phase B)")

    def junction_residual(self, before: Mapping, after: Mapping, drivers: Mapping, *,
                          dt: float | None = None) -> Tensor:
        """``A_s (y_after - y_before) - 0.5 dt (N_before + N_after)`` per junction (m3), with
        ``A_s`` (m2) and ``N_after`` (m3/s) evaluated at `after`; ``dt`` (s) defaults to the
        routing step. Zero to the iteration tolerance wherever the junction is wet."""
        dt = self.options.routing_step if dt is None else float(dt)
        drv = self(after | {k: before[k] for k in self.state_keys}, drivers,
                   StepContext(dt=dt))
        now = self.carried(after, drivers, dt=dt)
        return (drv["water.storage_area"]
                * (now["sewer.junction_depth"] - before["sewer.junction_depth"])
                - 0.5 * dt * (before["sewer.net_inflow"] + now["sewer.net_inflow"]))


def build_dynamic_wave_model(net: SewerNetwork, *, iterate_tol: float = 1e-11,
                             iterate_max: int = 200, relaxation: float = 0.5
                             ) -> tuple[Model, State, Drivers]:
    """Assemble a dynamic-wave sewer and return ``(model, state, drivers)``.

    One `PotentialFlowLayer` ``"water"`` (quantity ``"head"``, m; flows m3/s) over the pipes,
    with the outfalls as its boundary and a `JunctionStorage` at every junction, and the
    `DynamicWaveHydraulics` closure; ``coupling="iterate"`` converges on the junction heads
    to `iterate_tol` (m) within `iterate_max` passes, relaxed by `relaxation`. The state is
    SWMM's cold, dry start. Attached to the model: `junction_idx`, `junction_invert` (m),
    `outfall_pipe_idx`, `dynamic_wave` (the closure), `routing_step` (s), `pipe_names`,
    `pipe_columns`, `notes`, `input_specs`, `driver_template`.
    """
    net.validate()
    if net.routing != "dynamic_wave":
        raise ValueError(
            f"build_dynamic_wave_model: network routing is {net.routing!r}, not "
            f"'dynamic_wave'")
    graph = Network(dtype=F64)
    for manhole in net.manholes:
        graph.add_node(manhole.name, invert=manhole.invert, ground=manhole.ground)
    for outfall in net.outfalls:
        graph.add_node(outfall.name, invert=outfall.invert)
    for pipe in net.pipes:
        graph.add_edge(pipe.u, pipe.v, kind="pipe", name=pipe.name)
    boundary = [o.name for o in net.outfalls]
    closure = DynamicWaveHydraulics(graph, net)
    junction_idx = closure.junction_idx
    junction_invert = closure.node_invert[junction_idx]
    # Built once without the node source to learn the interior (a NodeSource must sit on
    # interior nodes), as the Modelica assembly does.
    probe = PotentialFlowLayer(graph, "water", [DynamicWaveConduit()], boundary=boundary,
                               quantity="head", unit="m", flow_unit="m3/s")
    interior = set(probe.interior.tolist())
    stranded = [m.name for m, i in zip(net.manholes, junction_idx.tolist(), strict=True)
                if i not in interior]
    if stranded:
        raise ValueError(f"build_dynamic_wave_model: junction(s) {stranded} touch no conduit")
    layer = PotentialFlowLayer(
        graph, "water", [DynamicWaveConduit()], boundary=boundary,
        node_sources=[JunctionStorage(junction_idx, junction_invert,
                                      bed_area=closure.min_area_ft2 * _AREA_SI)],
        quantity="head", unit="m", flow_unit="m3/s")
    if layer.bound.tolist() != closure.outfall_idx.tolist():
        raise ValueError("build_dynamic_wave_model: the layer's boundary order differs from "
                         "the outfall order the closure writes 'water.phi_boundary' in")
    model = Model(graph, {"water": layer}, closures=[closure], coupling="iterate",
                  iterate_tol={"water": iterate_tol}, iterate_max=iterate_max,
                  iterate_relaxation=relaxation)
    inflow = torch.zeros(graph.n, dtype=F64)
    inflow[junction_idx] = torch.tensor([m.inflow for m in net.manholes], dtype=F64)
    state = closure.initial_state({"inflow": inflow})
    drivers: Drivers = {
        "inflow": inflow,
        "water.phi_boundary": state["water.phi"][closure.outfall_idx].clone(),
    }
    closure.notes["routing"] = (
        f"dynamic wave (SWMM 5.2.4 fixed point): heads converged to {iterate_tol} m within "
        f"{iterate_max} passes; SWMM's HEAD_TOLERANCE and MAX_TRIALS are not used")
    model.notes = closure.notes
    model.junction_idx = junction_idx
    model.junction_invert = junction_invert
    model.outfall_pipe_idx = closure.outfall_pipe
    model.dynamic_wave = closure
    model.routing_step = float(net.dynamic_wave.routing_step)
    model.pipe_names = list(closure.pipe_names)
    model.pipe_columns = ("q", "h", "v", "V_wet")
    model.input_specs = {
        "inflow": dict(description="lateral inflow at each manhole", unit="m3/s",
                       over="nodes"),
    }
    model.driver_template = dict(drivers)
    return model, state, drivers


def solve_atol(model: Model, dt: float | None = None) -> float:
    """The head solve's default absolute tolerance (m3/s) for steps of `dt` s (default: the
    routing step): `SOLVE_ATOL`, raised to 4x the round-off floor of a junction on its stiff
    bed where that is larger. There the Jacobian slope is ``(2/dt)(A_s + K_BED A_bed)``,
    S ~ (2/dt) K_BED A_bed, and Newton cannot move the head by less than half an ulp,
    ``<= eps |H| / 2``, so the residual cannot go below about ``S eps |H| / 2`` -- 6e-10 m3/s
    on a 12 m invert at SWMM's default 12.566 ft2 floor and a 5 s step, giving an atol of
    2.5e-9 m3/s. A wet junction's head is then solved to within ``atol dt / (2 A_s)`` m, at
    most 5e-9 m in that case."""
    dt = model.routing_step if dt is None else float(dt)
    storage = next(ns for ns in model.layers["water"].node_sources
                   if isinstance(ns, JunctionStorage))
    slope = 2.0 * storage.bed_area * (1.0 + K_BED) / dt
    height = float(storage.bed.abs().max())
    floor = slope * torch.finfo(F64).eps * height / 2.0
    return max(SOLVE_ATOL, _BED_FLOOR_MARGIN * floor)


def simulate_dynamic_wave(model: Model, state: State, drivers: Drivers, *, steps: int,
                          dt: float | None = None, inflow: Tensor | None = None,
                          record: tuple[str, ...] = ("water.phi", "water.q"),
                          diagnostics: list | None = None, **solve_kwargs
                          ) -> tuple[State, dict[str, Tensor]]:
    """Run `steps` routing steps of `dt` s (default: the model's routing step) from `state`.

    `inflow` (m3/s, full-node) overrides ``drivers["inflow"]``: ``(n,)`` is one constant
    inflow; ``(B, n)`` a batch of B constant inflows -- a 2-D inflow is ALWAYS a batch,
    never a series; ``(steps, B, n)`` an inflow per step for a batch of B, the lateral
    inflow at the END of each step (one network's series is ``(steps, 1, n)``). Any other
    rank is refused. A batched inflow expands an unbatched `state` to the batch. After every
    step the carried keys are re-evaluated at the returned state, and a conduit end reaching
    its crown is refused by name. `diagnostics`, when a list, receives each step's
    `Model.step` diagnostics. `solve_kwargs` reach the head solve (Newton); `atol` defaults
    to `solve_atol` (m3/s: `SOLVE_ATOL`, or the bed's round-off floor where larger) and
    `rtol` to 0, because Newton's own default (sqrt(eps), 1.5e-8 m3/s) would stop a
    warm-started pass before it moves and leave the junction balance off by that much.
    Returns the final state and the recorded `record` keys stacked over steps (leading
    dimension `steps`).
    """
    hyd = model.dynamic_wave
    dt = model.routing_step if dt is None else float(dt)
    solve_kwargs.setdefault("atol", solve_atol(model, dt))
    solve_kwargs.setdefault("rtol", 0.0)
    n = model.net.n
    batch: tuple[int, ...] = ()
    if inflow is not None:
        inflow = torch.as_tensor(inflow, dtype=F64)
        if inflow.dim() not in (1, 2, 3) or inflow.shape[-1] != n:
            raise ValueError(
                f"simulate_dynamic_wave: inflow must be (n,), (B, n) or (steps, B, n) with "
                f"n = {n} nodes, got shape {tuple(inflow.shape)}")
        if inflow.dim() == 3 and inflow.shape[0] < steps:
            raise ValueError(
                f"simulate_dynamic_wave: inflow has {inflow.shape[0]} steps, {steps} asked")
        batch = tuple(inflow.shape[:-1]) if inflow.dim() == 2 else tuple(inflow.shape[1:-1])
    state = dict(state)
    if batch:
        state = {k: (v.expand(batch + v.shape[-1:]).clone() if v.dim() == 1 else v)
                 for k, v in state.items()}
    rows: dict[str, list[Tensor]] = {key: [] for key in record}
    t = 0.0
    for k in range(steps):
        if inflow is not None:
            drivers = dict(drivers, inflow=inflow[k] if inflow.dim() == 3 else inflow)
        diag: dict = {}
        try:
            state = model.step(state, drivers, dt, t=t, diagnostics=diag, **solve_kwargs)
        except RuntimeError as err:
            raise RuntimeError(f"dynamic wave step at t={t} s: {err}") from err
        state.update(hyd.carried(state, drivers, dt=dt))   # resync, as modelica/run.py does
        t += dt
        hyd.check_free_surface(state, drivers, t)
        if diagnostics is not None:
            diagnostics.append(diag)
        for key in record:
            rows[key].append(state[key])
    return state, {key: torch.stack(v) for key, v in rows.items()}
