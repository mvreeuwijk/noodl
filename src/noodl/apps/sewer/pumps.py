"""SWMM 5.2.4 pumps under dynamic wave: the pump curves and the flow a pump takes from its
inlet node, as pure functions of one pass's node heads, in SWMM's own feet units (ft, ft3,
cfs).

`pump_curve_flow` transcribes ``link.c::pump_getInflow`` (link.c:1548-1634) for the curve
types: PUMP1 (step in the inlet node's volume, ``table_intervalLookup``), PUMP2 (step in the
inlet depth), PUMP3 (continuous in the head DELIVERED, ``H2 - H1``), PUMP4 (continuous in the
inlet depth) and PUMP5 (PUMP3 at a speed setting ``s``: the head is divided by ``s^2``).
The flow is ``qIn * setting`` and never negative (link.c:1631-1633). Its ``dqdh`` is what SWMM
adds to the end nodes' ``sumdqdh`` (dynwave.c:566-577): PUMP3 and PUMP5 the negated slope of
the curve in the head delivered (so >= 0 for the usual falling curve), PUMP4 the forward
difference ``(Q(y + 0.001) - Q(y)) / 0.001`` in the inlet depth (an inlet-only quantity), PUMP1
and PUMP2 none. An IDEAL pump (no curve) takes its inlet node's inflow: the caller supplies it
(`dynwave.c::findNonConduitFlow` gives it as the node's inflow so far in the trial).

`mod_pump_flow` is ``dynwave.c::getModPumpFlow`` (dynwave.c:411-450): the curve flow limited
to the water available at the inlet node over the step, from the node's inflow, outflow and
surface area so far in the trial.

Curves are held in SI (m3 or m for the x values, m3/s for the flows), converted at read time
with the file's unit factors; the arithmetic converts to feet and cfs with SWMM's own factors
(`xsect_tables.LCF`, `QCF`). Every source reference is to SWMM 5.2.4 ``src/solver``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from noodl.apps.sewer.structures import table_lookup
from noodl.apps.sewer.xsect_tables import LCF, QCF

Tensor = torch.Tensor
F64 = torch.float64

#: SWMM's pump types (enums.h ``PumpType``): the four curve kinds, the variable-speed one and
#: the ideal pump (no curve).
TYPE1_PUMP, TYPE2_PUMP, TYPE3_PUMP, TYPE4_PUMP, TYPE5_PUMP, IDEAL_PUMP = range(6)
#: The `Pump.kind` keywords, in that order.
PUMP_KINDS = ("type1", "type2", "type3", "type4", "type5", "ideal")
#: SWMM's depth increment for PUMP4's slope (link.c:1555), ft.
SLOPE_STEP = 0.001


def interval_lookup(x: Tensor, curve_x: Tensor, curve_y: Tensor) -> Tensor:
    """`table.c::table_intervalLookup` (table.c:503-520) per row: the y of the first entry
    whose x exceeds `x`, the last y beyond the last x. ``curve_x`` is padded with +inf and
    ``curve_y`` with the last y (rows of `PumpSet`); ``x`` is (..., pumps)."""
    k = curve_x.shape[-1]
    count = (curve_x <= x.unsqueeze(-1)).sum(-1)          # entries at or below x
    idx = torch.clamp(count, max=k - 1)
    return curve_y.expand(x.shape + (k,)).gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def table_slope(x: Tensor, curve_x: Tensor, curve_y: Tensor, n_points: Tensor) -> Tensor:
    """`table.c::table_getSlope` (table.c:430-460) per row: the slope of the segment whose
    right end is the first entry at or above `x` (the first segment below the curve, the last
    above it); 0 for a one-point curve. `n_points` is each row's entry count."""
    k = curve_x.shape[-1]
    below = (curve_x < x.unsqueeze(-1)).sum(-1)
    last = torch.clamp(n_points - 1, min=0)
    hi = torch.minimum(torch.clamp(below, min=1), last)
    lo = torch.clamp(hi - 1, min=0)
    shape = x.shape + (k,)
    cx, cy = curve_x.expand(shape), curve_y.expand(shape)
    x1, x2 = (cx.gather(-1, i.unsqueeze(-1)).squeeze(-1) for i in (lo, hi))
    y1, y2 = (cy.gather(-1, i.unsqueeze(-1)).squeeze(-1) for i in (lo, hi))
    dx = x2 - x1
    ok = dx > 0.0
    return torch.where(ok, (y2 - y1) / torch.where(ok, dx, torch.ones_like(dx)),
                       torch.zeros_like(dx))


@dataclass(frozen=True)
class PumpSet:
    """Pumps (`link.c` ``Pump``), per device: ``kind`` (long, `TYPE1_PUMP` ... `IDEAL_PUMP`),
    the curves padded to one width, ``curve_x`` (SI: m3 for PUMP1, m for the others; +inf
    beyond a curve's points), ``curve_y`` (m3/s; the last flow beyond the points) and
    ``n_points`` (long)."""

    kind: Tensor
    curve_x: Tensor
    curve_y: Tensor
    n_points: Tensor

    @classmethod
    def build(cls, pumps) -> PumpSet:
        """From `network.Pump` devices (an ideal pump's curve is empty)."""
        n = len(pumps)
        width = max([len(p.curve[0]) for p in pumps if p.curve] or [1])
        x = torch.full((n, width), torch.inf, dtype=F64)
        y = torch.zeros((n, width), dtype=F64)
        count = torch.zeros(n, dtype=torch.long)
        kind = torch.tensor([PUMP_KINDS.index(p.kind) for p in pumps], dtype=torch.long)
        for i, p in enumerate(pumps):
            if p.curve:
                k = len(p.curve[0])
                x[i, :k] = torch.tensor(p.curve[0], dtype=F64)
                y[i, :k] = torch.tensor(p.curve[1], dtype=F64)
                y[i, k:] = float(p.curve[1][-1])
                count[i] = k
        return cls(kind, x, y, count)


def pump_curve_flow(ps: PumpSet, setting: Tensor, y1: Tensor, volume1: Tensor, h1: Tensor,
                    h2: Tensor) -> tuple[Tensor, Tensor]:
    """`pump_getInflow` for the curve pumps (link.c:1574-1634): ``(q, dqdh)`` in cfs and
    cfs/ft, with `setting` the pump's setting, `y1` and `volume1` the inlet node's depth (ft)
    and volume (ft3) and `h1`, `h2` the inlet and outlet node heads (ft). The row of an IDEAL
    pump is 0 (the caller supplies its flow). 0 flow where the setting is 0
    (link.c:1566-1568)."""
    kind = ps.kind
    cx, cy = ps.curve_x, ps.curve_y
    speed = torch.where(kind == TYPE5_PUMP, setting, torch.ones_like(setting))     # :1598
    speed = torch.where(speed > 0.0, speed, torch.ones_like(speed))              
    depth_m = y1 * LCF
    q_vol = interval_lookup(volume1 * QCF, cx, cy)                              # :1581
    q_dep = interval_lookup(depth_m, cx, cy)                                       # :1590
    head = torch.clamp((h2 - h1) / speed / speed, min=0.0)                         # :1599-1600
    head_m = head * LCF
    q_head = table_lookup(head_m, cx, cy)                                          # :1602
    d_head = -table_slope(head_m, cx, cy, ps.n_points) * (LCF / QCF) / speed       # :1606-1607
    q_cont = table_lookup(depth_m, cx, cy)                                         # :1615
    q_up = table_lookup((y1 + SLOPE_STEP) * LCF, cx, cy)                           # :1618
    d_cont = (q_up - q_cont) / QCF / SLOPE_STEP                                    # :1619
    by_head = (kind == TYPE3_PUMP) | (kind == TYPE5_PUMP)
    q = torch.where(kind == TYPE1_PUMP, q_vol,
                    torch.where(kind == TYPE2_PUMP, q_dep,
                                torch.where(by_head, q_head,
                                            torch.where(kind == TYPE4_PUMP, q_cont,
                                                        torch.zeros_like(q_cont)))))
    dqdh = torch.where(by_head, d_head,
                       torch.where(kind == TYPE4_PUMP, d_cont, torch.zeros_like(d_cont)))
    q = torch.clamp(q / QCF, min=0.0)                                              # :1631-1633
    return q * setting, dqdh


def mod_pump_flow(kind: Tensor, q: Tensor, node_storage: Tensor, full_volume: Tensor,
                  inflow: Tensor, outflow: Tensor, old_net_inflow: Tensor, old_depth: Tensor,
                  old_volume: Tensor, surf_area: Tensor, dt: float) -> Tensor:
    """`dynwave.c::getModPumpFlow` (dynwave.c:411-450) per pump: `q` (cfs) limited by what
    its inlet node holds. At a storage unit, and for a PUMP1, the outflow is capped at the
    node's ``inflow + oldVolume / dt`` (``node_getMaxOutflow``, node.c:418-428; only where the
    node's `full_volume` (ft3) is positive) and floored at 0. For PUMP2, PUMP3 and PUMP4 at
    another node, where the depth the step would leave, ``oldDepth + 0.5 (oldNetInflow +
    inflow - outflow - q) dt / area``, is not positive, the pump takes the node's inflow.
    PUMP5 and IDEAL pumps are unlimited. `inflow`, `outflow` and `surf_area` (ft2) are the
    inlet node's so far in the trial; `old_depth` (ft), `old_volume` (ft3) and
    `old_net_inflow` (cfs) its values at the step start."""
    q_max = inflow + old_volume / dt                                               # node.c:430
    capped = torch.clamp(torch.where(full_volume > 0.0, torch.minimum(q, q_max), q), min=0.0)
    new_net = inflow - outflow - q                                                 # :437
    volume = 0.5 * (old_net_inflow + new_net) * dt                                 # :438
    area_ok = surf_area != 0.0
    y = old_depth + volume / torch.where(area_ok, surf_area, torch.ones_like(surf_area))
    y = torch.where(area_ok, y, torch.where(volume < 0.0, -torch.ones_like(y),
                                            torch.ones_like(y)))    # SWMM: +-inf (0/0: nan)
    depth_pumps = (kind == TYPE2_PUMP) | (kind == TYPE3_PUMP) | (kind == TYPE4_PUMP)
    drained = depth_pumps & ~node_storage & (y <= 0.0)
    out = torch.where(drained, inflow, q)                                          # :446-447
    by_volume = node_storage | (kind == TYPE1_PUMP)
    out = torch.where(by_volume, capped, out)
    return torch.where(q == 0.0, q, out)                                           # :425


class PumpStation:
    """What a pump sees of its inlet node DURING a trial of SWMM's dynamic-wave step.

    ``dynwave.c::findLinkFlows`` (dynwave.c:365-395) first finds every conduit's flow, adds the
    flows to the end nodes' inflow and outflow and surface area (``updateNodeFlows``), and only
    then visits the other links in link-index order, a pump among them: so at the time a pump
    asks (``getModPumpFlow``, an ideal pump's ``Node.inflow``) its inlet node holds the lateral
    inflow, every conduit's contribution and those of the non-conduit links BEFORE it in the
    file, not the links after it. This class holds, per pump, the 0/1 matrices that pick those
    links out of a vector of link flows, and the orifices' end areas, so a pass can reproduce
    the partial sums at the converged flows.

    `inlet` (long, per pump) are the inlet nodes' indices over all nodes; `edge_u`, `edge_v`
    every link's end nodes (`n_pipes` conduits first, then the regulators and pumps) and
    `key` every link's rank in SWMM's link-index order (conduits' ranks are irrelevant:
    they all precede); `columns` the pumps' link columns; `orifice_columns` and `orifice_u`,
    `orifice_v` the orifices' (possibly empty)."""

    def __init__(self, inlet: Tensor, columns: Tensor, edge_u: Tensor, edge_v: Tensor,
                 n_pipes: int, key: Tensor, orifice_columns: Tensor) -> None:
        self.inlet = inlet
        n_edges = edge_u.shape[0]
        e = torch.arange(n_edges)
        before = (key[None, :] < key[columns][:, None]) | (e[None, :] < n_pipes)
        before &= e[None, :] != columns[:, None]
        into = (edge_v[None, :] == inlet[:, None]).to(F64)
        out_of = (edge_u[None, :] == inlet[:, None]).to(F64)
        mask = before.to(F64)
        #: (pumps, links): 1 where the link is counted and ends / starts at the inlet
        self.ends_here = mask * into
        self.starts_here = mask * out_of
        o_before = (key[orifice_columns][None, :] < key[columns][:, None]).to(F64)
        o_u, o_v = edge_u[orifice_columns], edge_v[orifice_columns]
        self.orifice_first = o_before * (o_u[None, :] == inlet[:, None]).to(F64)
        self.orifice_second = o_before * (o_v[None, :] == inlet[:, None]).to(F64)

    def inflow_outflow(self, q: Tensor, lateral: Tensor) -> tuple[Tensor, Tensor]:
        """The inlet nodes' inflow and outflow (cfs) so far in the trial: `q` the link flows
        (cfs, ``(..., links)``), `lateral` the nodes' lateral inflow (cfs, ``(..., nodes)``;
        a negative one is an outflow, dynwave.c:314-321)."""
        into, out_of = torch.relu(q), torch.relu(-q)
        lat = lateral[..., self.inlet]
        inflow = torch.relu(lat) + into @ self.ends_here.T + out_of @ self.starts_here.T
        outflow = torch.relu(-lat) + into @ self.starts_here.T + out_of @ self.ends_here.T
        return inflow, outflow

    def surface_area(self, own: Tensor, conduit: Tensor, first: Tensor | None,
                     second: Tensor | None) -> Tensor:
        """The inlet nodes' surface area (ft2) so far in the trial: `own` and `conduit`
        per pump (the node's own storage area and the conduits' ends', ft2) plus the ends of
        the orifices before the pump, `first` and `second` their areas at their first and
        second nodes (ft2, per orifice; None without orifices)."""
        area = own + conduit
        if first is not None:
            area = area + first @ self.orifice_first.T + second @ self.orifice_second.T
        return area
