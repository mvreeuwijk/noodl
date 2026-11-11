"""EPANET 2.2's water-quality routing, transcribed (`quality.c`, `qualroute.c`, tag v2.2).

EPANET's answer is a discrete Lagrangian procedure, not the advection equation's: every pipe
and tank holds a chain of volume segments (`initsegs`); once per hydraulic step the flow
directions are taken (`flowdirchanged`), a chain is reversed when its flow changes sign
(`reversesegs`) and the nodes are sorted from upstream to downstream (`sortnodes`,
`selectnonstacknode`); then every quality step walks the nodes in that order (`transport`),
taking each link's inflow from its leading segments (`evalnodeinflow`), mixing it at the
node (`findnodequal`, a tank by its mixing model, `quality_tanks`) and releasing the node's water
into its outflow links, merged into their last segment when within `Ctol` of it
(`evalnodeoutflow`). `QualityRouter` is that procedure; `water_run` drives it
after every solve with that solve's flows, link states and delivered demands (EPANET's
`runqual` then `nextqual`), and reports the quality as it stands before the transport
(`saveoutput`).

STATE. Every segment of every chain lives in one `SegmentPool` (EPANET's `mempool`, its
`FreeSeg` list and `Sseg`'s `prev` link): two 1-D float64 tensors, concentration `c` and
volume `v`, and Python index chains. Chain `k` is link `k` of `model.link_names`; chain
`n_links + i` is tank `i` of `net.tanks`. Concentrations stay in the file's quality unit
(mg/L, ug/L, hours, percent) and volumes are m3: EPANET's own unit factors cancel in every
ratio it takes.

DIFFERENTIABILITY. Volumes and concentrations keep their autograd graph in an attached run;
every decision (a direction, `v > 0`, `vseg >= seg.v`, `|c_seg - c| < Ctol`) is taken on a
detached `float()`. A detached run (no gradient asked for) works on Python floats under
`torch.no_grad()`.

ORDER. EPANET's node and link indices (`WaterQuality.node_order`, `link_order`: junctions,
then reservoirs and tanks in file order; links in file order across `[PIPES] [PUMPS]
[VALVES]`) decide the adjacency order (`project.c`, `buildadjlists`, which prepends: a node's
links in decreasing link index) and the sort's ties; the results stay in `model.node_names`
and `model.link_names` order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from noodl.apps.water.elements import NU_WATER_FT2
from noodl.apps.water.network import pattern_factor
from noodl.apps.water.quality_data import WaterQuality
from noodl.apps.water.quality_react import (
    BULK_TINY,
    EPANET_LITRES_PER_M3,
    bulk_rate,
    pipe_coefficients,
    tank_coefficients,
    wall_coefficients,
    wall_rate,
)
from noodl.apps.water.quality_tanks import (
    QualityTolerance,
    detached,
    initial_segments,
    mix_tank,
)
from noodl.apps.water.units import M_PER_FT, Q_STAGNANT_CFS, rounded_cfs_per_m3s

Tensor = torch.Tensor
F64 = torch.float64

#: `qualroute.c`, `LINKVOL`: EPANET's pi/4, to six digits (`0.785398 L D^2`).
LINKVOL_FACTOR = 0.785398
#: EPANET's tracer concentration at the trace node (`quality.c`, `initqual`;
#: `qualroute.c`, `findnodequal`).
TRACE_QUALITY = 100.0
_M3_PER_FT3 = 0.3048**3
#: The mass a CHEMICAL's `c v` (quality x m3) is in the chemical's mass unit: exact litres
#: per m3 (Ruling Q28; EPANET's `LperFT3 = 28.317` is 5.4e-6 above it and is not carried).
EXACT_LITRES_PER_M3 = 1000.0


def flow_volume_scale(flow_units: str | None) -> float:
    """The factor from a flow in m3/s to the volume per second EPANET's quality moves, in
    m3: EPANET divides a file's flows by its own ROUNDED factors (`types.h`, e.g.
    `LPSperCFS = 28.317` against the exact 28.3168466; `input1.c`, `convertunits`) but
    its lengths by the exact foot, so in cfs times the exact ft3 a flow moves
    `rounded_cfs_per_m3s(units) * 0.3048^3` of the volume it carries in SI (1 - 5.4e-6 for
    LPS). Without a file unit, 1."""
    cfs = rounded_cfs_per_m3s(flow_units)
    return 1.0 if cfs is None else cfs * _M3_PER_FT3


@dataclass
class QualityMassBalance:
    """EPANET's water-quality mass balance (`MassBalance`, `quality.c`; reported by
    `writemassbalance`, `report.c`): the mass `initial` in the network's pipes and tanks,
    the `inflow` the sources and reservoirs added, the `outflow` the demands and the
    reservoirs' inflow removed, the mass `reacted` away (negative when a reaction made
    mass), the `final` mass stored, and the `ratio` of what left to what came in
    (`evalmassbalance`: `(outflow + final + max(reacted, 0)) / (initial + inflow -
    min(reacted, 0))`, 1 when that denominator is 0). Each a 0-d tensor (`()` for one router;
    a batched run stacks them to the batch shape) that keeps the autograd graph of an
    attached run (the spec's Ruling Q5; only decisions are detached). Mass is the physical
    mass: a CHEMICAL's
    `c v 1000` (`EXACT_LITRES_PER_M3`: 1 m3 at 1 mg/L is 1000 mg, the chemical's mass unit);
    AGE's and TRACE's `c v` (quality x m3; the ratio is unit-free)."""

    initial: Tensor
    inflow: Tensor
    outflow: Tensor
    reacted: Tensor
    final: Tensor
    ratio: Tensor


def quality_step(net, hydraulic_step: int) -> int:
    """The quality time step (s) of `net` (`input1.c`, `adjustdata`): the file's `QUALITY
    TIMESTEP` (`net.quality.quality_timestep`), else a tenth of the hydraulic step in whole
    seconds (`Hstep / 10`, the step after its pattern and report caps), and never longer
    than the hydraulic step. A step of 0 s is refused by name: EPANET's `nextqual` would
    never advance (`dt = MIN(Qstep, hydstep - qtime)`)."""
    explicit = net.quality.quality_timestep if net.quality is not None else None
    step = int(hydraulic_step) // 10 if explicit is None else int(round(float(explicit)))
    step = min(step, int(hydraulic_step))
    if step <= 0:
        raise ValueError(
            f"quality: the QUALITY TIMESTEP is {step} s (a tenth of the hydraulic step of "
            f"{hydraulic_step} s, in whole seconds, when the file gives none); the quality "
            f"needs a step of at least one second: set [TIMES] QUALITY TIMESTEP")
    return step


class SegmentPool:
    """EPANET's segment memory (`mempool.c`; `qualroute.c`, `addseg`, `reversesegs`, the
    `FreeSeg` list): every segment's concentration `c` and volume `v` in two 1-D float64
    tensors (their capacity doubles as segments are allocated; slots past the allocated
    ones are unused), each segment's `prev` (the segment behind it, upstream; -1 is NULL)
    and `owner` chain, and per chain its `first` (leading, downstream: `FirstSeg`) and
    `last` (upstream: `LastSeg`) segment, -1 when the chain is empty. A used-up segment is
    recycled through the free list (`free`, chained by `prev`), as EPANET recycles it, its
    `owner` set to -1.

    The quality step works on scalars (`open`/`close`): `open` takes the values out once
    (detached, one `tolist`; `attached`, 0-d tensors that keep the graph), the node loop
    reads and writes them as `cs`, `vs`, and `close` writes them back in one tensor
    (detached, one `torch.tensor`; attached, one out-of-place `index_put`)."""

    def __init__(self, n_chains: int, *, attached: bool) -> None:
        self.attached = attached
        self.capacity = 16
        self.c = torch.zeros(self.capacity, dtype=F64)
        self.v = torch.zeros(self.capacity, dtype=F64)
        self.prev: list[int] = []
        self.owner: list[int] = []
        self.first = [-1] * n_chains
        self.last = [-1] * n_chains
        self.free = -1
        self.cs: list | None = None
        self.vs: list | None = None

    @property
    def size(self) -> int:
        """The segments allocated so far, live or free."""
        return len(self.prev)

    def scalars(self) -> tuple[list, list]:
        """Every allocated segment's `c` and `v` as scalars (Python floats, or 0-d tensors
        in an attached pool)."""
        n = self.size
        if self.attached:
            return list(self.c[:n].unbind()), list(self.v[:n].unbind())
        return self.c[:n].tolist(), self.v[:n].tolist()

    def open(self) -> None:
        """Take every allocated segment's `c` and `v` out as scalars (`cs`, `vs`)."""
        self.cs, self.vs = self.scalars()

    def close(self) -> None:
        """Write the scalars back into `c` and `v`, doubling the capacity as needed."""
        n = self.size
        while self.capacity < n:
            self.capacity *= 2
        if self.attached:
            index = (torch.arange(n),)
            self.c = torch.zeros(self.capacity, dtype=F64).index_put(
                index, torch.stack([torch.as_tensor(x, dtype=F64) for x in self.cs]))
            self.v = torch.zeros(self.capacity, dtype=F64).index_put(
                index, torch.stack([torch.as_tensor(x, dtype=F64) for x in self.vs]))
        else:
            pad = [0.0] * (self.capacity - n)
            self.c = torch.tensor(self.cs + pad, dtype=F64)
            self.v = torch.tensor(self.vs + pad, dtype=F64)
        self.cs = self.vs = None

    def add(self, chain: int, v, c) -> None:
        """`addseg` (`qualroute.c`): a segment of volume `v` and quality `c` added upstream
        of `chain`'s last segment (the chain's first when it was empty), taken from the free
        list when one is there."""
        closed = self.cs is None
        if closed:
            self.open()
        if self.free >= 0:
            seg = self.free
            self.free = self.prev[seg]
            self.cs[seg], self.vs[seg] = c, v
            self.owner[seg] = chain
        else:
            seg = self.size
            self.prev.append(-1)
            self.owner.append(chain)
            self.cs.append(c)
            self.vs.append(v)
        self.prev[seg] = -1
        if self.first[chain] < 0:
            self.first[chain] = seg
        if self.last[chain] >= 0:
            self.prev[self.last[chain]] = seg
        self.last[chain] = seg
        if closed:
            self.close()

    def remove_first(self, chain: int) -> None:
        """`evalnodeinflow`'s removal of a used-up leading segment: the one behind it leads,
        and the segment goes to the free list."""
        seg = self.first[chain]
        self.first[chain] = self.prev[seg]
        if self.first[chain] < 0:
            self.last[chain] = -1
        self.prev[seg] = self.free
        self.owner[seg] = -1
        self.free = seg

    def reverse(self, chain: int) -> None:
        """`reversesegs` (`qualroute.c`): `chain` re-oriented for a flow reversal, its last
        segment leading."""
        seg = self.first[chain]
        self.first[chain], self.last[chain] = self.last[chain], seg
        behind = -1
        while seg >= 0:
            ahead = self.prev[seg]
            self.prev[seg] = behind
            behind = seg
            seg = ahead

    def segments(self, chain: int) -> list[int]:
        """`chain`'s segments from its leading one (`FirstSeg`) to its last (`LastSeg`)."""
        out, seg = [], self.first[chain]
        while seg >= 0:
            out.append(seg)
            seg = self.prev[seg]
        return out

    def counts(self) -> tuple[int, ...]:
        """The number of segments in every chain."""
        return tuple(len(self.segments(k)) for k in range(len(self.first)))


def _adjacency(n_nodes: int, ends: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """`buildadjlists` (`project.c`): per node, `(link, other node)` of every link at it;
    the lists are prepended to link by link, so they run in decreasing link index."""
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for k in range(len(ends) - 1, -1, -1):
        u, v = ends[k]
        adjacency[u].append((k, v))
        adjacency[v].append((k, u))
    return adjacency


def sort_nodes(n_nodes: int, ends: Sequence[tuple[int, int]], direction: Sequence[int]
               ) -> tuple[list[int], int]:
    """`sortnodes` and `selectnonstacknode` (`qualroute.c`): the nodes `0 .. n_nodes - 1`
    (EPANET's indices less one) from upstream to downstream along the links `ends` (`(N1,
    N2)` per link, EPANET's link order) whose flow `direction` is +1, -1 or 0 (stagnant
    links are left out), and the number of cycles broken. The nodes without inflow are
    pushed on a stack in index order and popped last in, first out; a popped node lowers
    the in-degree of its links' downstream nodes in its adjacency order (`buildadjlists`),
    pushing each that reaches 0. When the stack runs empty before every node is sorted, a
    cycle is broken (`selectnonstacknode`): the first node with inflow left that is
    adjacent to a sorted node, the most recently sorted first, else the first such node in
    index order, is pushed with its in-degree set to 0. `(order, cycle breaks)`."""
    adjacency = _adjacency(n_nodes, ends)
    indegree = [0] * n_nodes
    for k, d in enumerate(direction):
        if d > 0:
            indegree[ends[k][1]] += 1
        elif d < 0:
            indegree[ends[k][0]] += 1
    stack = [i for i in range(n_nodes) if indegree[i] == 0]
    order: list[int] = []
    breaks = 0
    while len(order) < n_nodes:
        if not stack:
            j = _select_non_stack_node(order, indegree, adjacency)
            if j < 0:
                raise RuntimeError("sort_nodes: no node left to break the cycle with "
                                   "(EPANET error 120)")
            breaks += 1
            indegree[j] = 0
            stack.append(j)
        i = stack.pop()
        order.append(i)
        for k, _ in adjacency[i]:
            d = direction[k]
            if d == 0:
                continue
            n = ends[k][0] if d < 0 else ends[k][1]
            if n != i and indegree[n] > 0:
                indegree[n] -= 1
                if indegree[n] == 0:
                    stack.append(n)
    return order, breaks


def _select_non_stack_node(order: list[int], indegree: list[int],
                           adjacency: list[list[tuple[int, int]]]) -> int:
    """`selectnonstacknode` (`qualroute.c`): the first node with inflow links left that is
    adjacent to a sorted node, the sorted nodes taken last in, first out, else the first
    node with inflow left; -1 when there is none."""
    for m in reversed(order):
        for _, n in adjacency[m]:
            if indegree[n] > 0:
                return n
    for i, d in enumerate(indegree):
        if d > 0:
            return i
    return -1


def _tank_volume(net, tank, level):
    """A tank's volume (m3) at `level` (m above its bottom; a float or a 0-d tensor, whose
    graph is kept), EPANET's `tankvolume` (`hydraul.c`), which gives `V0` for the initial
    level (`input1.c`, `inittanks`; `epanet.c`, `EN_setnodevalue(EN_TANKLEVEL)`): on a
    volume curve, the curve at the level (`interp`); otherwise `Vmin + A (h - Hmin)`,
    `Vmin` the file's MinVol when positive, else `A Hmin`, `A = pi D^2 / 4`."""
    if tank.volume_curve is not None:
        points = net.curves[tank.volume_curve]
        return _interp(points, level)
    area = torch.pi * tank.diameter**2 / 4.0
    vmin = tank.min_volume if tank.min_volume > 0.0 else area * tank.min_level
    return vmin + area * (level - tank.min_level)


def _interp(points, x):
    """EPANET's `interp` (`project.c`): linear in the curve's points, its end segments
    extended beyond them."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(xs) == 1:
        return ys[0]
    m = 1
    while m < len(xs) - 1 and detached(x) > xs[m]:
        m += 1
    dx = xs[m] - xs[m - 1]
    if abs(dx) < 1e-6:
        return ys[m]
    return ys[m] - (xs[m] - x) * (ys[m] - ys[m - 1]) / dx


def _refuse_unanswerable_orders(net, quality: WaterQuality) -> None:
    """The reaction orders EPANET itself has no finite answer for, refused by name
    (Ruling Q20)."""
    if quality.mode == "CHEMICAL":
        for which, order in (("BULK", quality.bulk_order), ("TANK", quality.tank_order)):
            if 0.0 < order < 1.0:
                raise ValueError(
                    f"quality: ORDER {which} {order:g} lies between 0 and 1: EPANET 2.2's "
                    f"bulkrate (qualreact.c) takes pow(max(0, c), n - 1) = inf at zero "
                    f"concentration and returns NaN (inf with a limiting potential), so "
                    f"there is no finite answer to reproduce; use an order of 0, 1 or "
                    f"more, or a negative (Michaelis-Menten) order")


class QualityRouter:
    """EPANET 2.2's quality routing of `model`'s network under `quality` (module
    docstring): `initqual` and `initsegs` at construction, then `advance` once per
    hydraulic step.

    `inputs["quality.initial"]` (n_nodes, `model.node_names`, quality units) replaces the
    network's initial qualities (`quality.initial`, 0 where absent);
    `inputs["quality.tank_level"]` (n_tanks, m) the tanks' initial levels (`water_run`
    passes its initial state's), from which their initial segment volumes are taken.
    `inputs["quality.bulk_coefficient"]`, `["quality.wall_coefficient"]` (n_pipes,
    `net.pipes` order) and `["quality.tank_coefficient"]` (n_tanks, `net.tanks`), SI per
    second (`WaterQuality`), replace the network's reaction coefficients
    (`quality_react.pipe_coefficients`, `tank_coefficients`), the wall's after its roughness
    correlation; they keep their graph. `inputs["quality.source_strength"]` (n_nodes,
    `model.node_names`, read at the nodes that have a `[SOURCES]` entry) replaces their
    strengths (`QualitySource.strength`: quality units, a MASS source's per second); it
    keeps its graph. Sources act in a CHEMICAL run only
    (`findsourcequal`). A reactive substance (EPANET's `Reactflag`,
    `setreactflag`: AGE, or a CHEMICAL with a nonzero pipe or tank coefficient) reacts in
    every quality step before its transport (`react`), the mass it lost kept in `reacted`
    (quality x m3; EPANET's `MassBalance.reacted`). A CHEMICAL whose coefficients require a
    gradient reacts even at zero coefficients (adding exactly 0), so their derivative is
    kept; `noflowqual` still follows `Reactflag`. `mass_balance` is EPANET's mass balance
    (`QualityMassBalance`): the sources' and reservoirs' mass in and the demands' and
    reservoirs' out are accumulated in `transport` (`updatemassbalance`).
    `quality_step` is the quality step (s, `quality_step`). `attached` keeps the autograd
    graph (module docstring). `diagnostics`, a dict, receives under `"quality"` a list
    with one dict for this run: `"times"` (s) and `"node_quality"`, `"link_quality"`
    (`record`, per solve, before the transport), `"node_order"` (`(t, node names)` per
    sort), `"cycle_breaks"` (per sort), `"reversals"` (`(t, link name)` per reversed
    chain), `"segments"` (`SegmentPool.counts()` after every quality step) and
    `"inflow_volume"`, `"inflow_mass"`, `"outflow_volume"` (per quality step, per node of
    `model.node_names`: m3, quality x m3, m3)."""

    def __init__(self, model, quality: WaterQuality, *, inputs: Mapping[str, Tensor],
                 quality_step: int, attached: bool, diagnostics: dict | None = None) -> None:
        net = model.water_network
        _refuse_unanswerable_orders(net, quality)
        self.net = net
        self.quality = quality
        self.mode = quality.mode
        self.qstep = int(quality_step)
        self.attached = attached
        self.ctol = QualityTolerance(quality.tolerance, quality.mode)
        # flows and demands as EPANET's quality reads them (cfs, as m3 per s), and its
        # `Q_STAGNANT` (0.005 gpm in cfs, `quality.c`) in the same units
        self.flow_scale = flow_volume_scale(net.options.flow_units)
        self.q_stagnant = Q_STAGNANT_CFS * _M3_PER_FT3
        node_names = list(model.node_names)
        link_names = list(model.link_names)
        self.node_names, self.link_names = node_names, link_names
        n_nodes, n_links = len(node_names), len(link_names)
        self.n_nodes, self.n_links = n_nodes, n_links
        node_index = {n: i for i, n in enumerate(node_names)}
        links = net.links()
        self.ends = [(node_index[k.u], node_index[k.v]) for k in links]
        # EPANET's indices: the order of the sort and of every node's adjacency list
        epanet_nodes = list(quality.node_order) or node_names
        epanet_links = list(quality.link_order) or link_names
        self._node_of_epanet = [node_index[n] for n in epanet_nodes]
        epanet_node = {m: e for e, m in enumerate(self._node_of_epanet)}
        link_index = {n: k for k, n in enumerate(link_names)}
        self._link_of_epanet = [link_index[n] for n in epanet_links]
        self._epanet_ends = [(epanet_node[self.ends[k][0]], epanet_node[self.ends[k][1]])
                             for k in self._link_of_epanet]
        adjacency = _adjacency(n_nodes, self._epanet_ends)
        self.adjacency = [
            [self._link_of_epanet[k] for k, _ in adjacency[epanet_node[m]]]
            for m in range(n_nodes)]
        # node kinds, and each tank's chain
        self.junction_nodes = [node_index[j.name] for j in net.junctions]
        self.junction = [False] * n_nodes
        for m in self.junction_nodes:
            self.junction[m] = True
        self.tank_of = {node_index[t.name]: i for i, t in enumerate(net.tanks)}
        self.trace = (node_index[quality.trace_node] if self.mode == "TRACE" else -1)

        # `initqual` (`quality.c`): NodeQual = C0 (TRACE: 0), the tank's C its node's, the
        # trace node 100
        given = inputs.get("quality.initial")
        if given is None:
            c0 = [float(quality.initial.get(n, 0.0)) for n in node_names]
        elif attached:
            c0 = list(given.unbind())
        else:
            c0 = given.detach().tolist()
        self.node = [0.0] * n_nodes if self.mode == "TRACE" else list(c0)
        self.tank_c = [self.node[m] for m in sorted(self.tank_of, key=self.tank_of.get)]
        if self.trace >= 0:
            self.node[self.trace] = TRACE_QUALITY

        # `initsegs` (`qualroute.c`)
        self.pool = SegmentPool(n_links + len(net.tanks), attached=attached)
        self.pool.open()
        for k, pipe in enumerate(net.pipes):
            if pipe.status.upper() == "CV":
                continue      # a check-valve pipe is a CVPIPE, not a PIPE: no segment
            self.pool.add(k, LINKVOL_FACTOR * pipe.length * pipe.diameter**2,
                          self.node[self.ends[k][1]])
        # each tank at the run's initial level (`inputs["quality.tank_level"]`, else the
        # file's) and the file's initial quality, even in TRACE
        levels = inputs.get("quality.tank_level")
        if levels is None:
            levels = [tank.init_level for tank in net.tanks]
        else:
            levels = list(levels.unbind()) if attached else levels.detach().tolist()
        self.tank_mix: list[tuple[str, float]] = []
        for i, tank in enumerate(net.tanks):
            model, fraction = quality.mixing.get(tank.name, ("MIXED", 1.0))
            # `V1max = fraction x Vmax` (`input1.c`, `convertunits`), Vmax the volume at the
            # maximum level (the curve there, or `Vmin + A (max - min)`)
            vmz = fraction * detached(_tank_volume(net, tank, tank.max_level))
            self.tank_mix.append((model, vmz))
            initial_segments(self.pool, n_links + i, model, _tank_volume(net, tank, levels[i]),
                             c0[node_index[tank.name]], vmz)
        self.pool.close()

        self._init_reactions(net, quality, inputs, attached)
        self._init_sources(quality, inputs, node_index)
        # `initqual`: `MassBalance.initial = findstoredmass`, the rest 0 (`reacted` is
        # `_init_reactions`'s)
        self.mass_in = self.mass_out = 0.0
        self.mass_initial = self._stored_mass()
        self.direction = [0] * n_links         # `FlowDir`, ZERO_FLOW at the start
        self.order: list[int] = []              # `SortedNodes`, none until the first sort
        self.diagnostics = None
        if diagnostics is not None:
            self.diagnostics = {
                "times": [], "node_quality": [], "link_quality": [], "node_order": [],
                "cycle_breaks": [], "reversals": [], "segments": [], "inflow_volume": [],
                "inflow_mass": [], "outflow_volume": []}
            diagnostics.setdefault("quality", []).append(self.diagnostics)

    def _init_reactions(self, net, quality: WaterQuality, inputs: Mapping[str, Tensor],
                        attached: bool) -> None:
        """The reaction coefficients (`adjustdata`, or `inputs`), `Reactflag`
        (`setreactflag`, `qualreact.c`: TRACE never reacts, AGE always does, a CHEMICAL when
        any pipe's (check-valve pipes included) bulk or wall coefficient or any tank's is
        nonzero, read detached) and the chain kinds the reactions gather by: 1 a pipe
        (`Type == PIPE`: not a check-valve pipe, a pump or a valve), 2 a tank, 0 none; one
        more entry 0 for a free segment's owner -1."""
        n_pipes, n_tanks = len(net.pipes), len(net.tanks)
        kb, kw = pipe_coefficients(net, quality)
        kt = tank_coefficients(net, quality)
        given = []
        for key, default, what, count in (("quality.bulk_coefficient", kb, "pipe", n_pipes),
                                          ("quality.wall_coefficient", kw, "pipe", n_pipes),
                                          ("quality.tank_coefficient", kt, "tank", n_tanks)):
            value = inputs.get(key)
            if value is None:
                given.append(default)
                continue
            if tuple(value.shape) != (count,):
                raise ValueError(f"quality: {key} has the shape {tuple(value.shape)}; it "
                                 f"takes one value per {what}, ({count},)")
            value = value.to(F64)
            given.append(value if attached else value.detach())
        self.kb, self.kw, self.kt = given
        if self.mode == "TRACE":
            self.reactive = False
        elif self.mode == "AGE":
            self.reactive = True
        else:
            self.reactive = any(bool((x.detach() != 0).any()) for x in given)
        # the reactions run when EPANET's do, and in a CHEMICAL run whose coefficients ask
        # for a gradient: at a zero coefficient they add exactly 0 and carry its derivative
        self.reacts = self.reactive or (self.mode == "CHEMICAL" and attached and any(
            key in inputs and inputs[key].requires_grad
            for key in ("quality.bulk_coefficient", "quality.wall_coefficient",
                        "quality.tank_coefficient")))
        self.limit = float(quality.limiting_potential)
        self.bulk_order = float(quality.bulk_order)
        self.wall_order = float(quality.wall_order)
        self.tank_order = float(quality.tank_order)
        self.diffusivity = float(quality.diffusivity)
        self.viscosity = float(net.options.viscosity) * NU_WATER_FT2 * M_PER_FT**2
        self.pipe_diameter = torch.tensor([p.diameter for p in net.pipes], dtype=F64)
        self.pipe_length = torch.tensor([p.length for p in net.pipes], dtype=F64)
        self.rc = torch.zeros(n_pipes, dtype=F64)      # `Rc`, set by `ratecoeffs`
        kinds = [0 if p.status.upper() == "CV" else 1 for p in net.pipes]
        kinds += [0] * (self.n_links - n_pipes) + [2] * n_tanks + [0]
        self._chain_kind = torch.tensor(kinds, dtype=torch.long)
        self.n_pipes = n_pipes
        self.reacted = torch.zeros((), dtype=F64)

    def _init_sources(self, quality: WaterQuality, inputs: Mapping[str, Tensor],
                      node_index: dict[str, int]) -> None:
        """`[SOURCES]` (`sourcedata`, `input3.c`) of a CHEMICAL run, by node: `source_kind`,
        `source_pattern` and `source_base` (the strength, `C0`, from
        `inputs["quality.source_strength"]` when given; a scalar that keeps its graph when
        attached). Other modes ignore them (`findsourcequal`: `Qualflag != CHEM`)."""
        self.source_kind: dict[int, str] = {}
        self.source_pattern: dict[int, str | None] = {}
        self.source_base: dict[int, object] = {}
        self.source_c: dict[int, object] = {}
        given = inputs.get("quality.source_strength")
        if given is not None and tuple(given.shape) != (self.n_nodes,):
            raise ValueError(f"quality: quality.source_strength has the shape "
                             f"{tuple(given.shape)}; it takes one value per node, "
                             f"({self.n_nodes},)")
        if self.mode != "CHEMICAL":
            return
        for source in quality.sources:
            m = node_index[source.node]
            self.source_kind[m] = source.kind
            self.source_pattern[m] = source.pattern
            if given is None:
                self.source_base[m] = float(source.strength)
            else:
                self.source_base[m] = given[m] if self.attached else float(given[m].detach())

    def _source_strengths(self, t: int) -> None:
        """Every source's strength at the step starting at `t`: `sourcequal` (`quality.c`),
        the base strength times its pattern's factor at `Qtime`, which is the step's start
        for every quality step of it (`nextqual` advances it after the transport)."""
        self.source_c = {
            m: (base if self.source_pattern[m] is None
                else base * pattern_factor(self.net, self.source_pattern[m], t))
            for m, base in self.source_base.items()}

    def _source_quality(self, n: int, volout, dt: int):
        """`findsourcequal` (`quality.c`): the concentration node `n`'s source adds to the
        water leaving it, 0.0 for none. Nothing when the base strength is 0 or the outflow
        `volout / dt` (m3/s as EPANET's flows) is not above `Q_STAGNANT`. CONCEN at a
        junction needs a negative demand: `-c demand dt / volout`; MASS (per second)
        `c dt / volout` over the litres per m3 EPANET's volumes carry (`LperFT3` over the
        exact ft3, `EPANET_LITRES_PER_M3`, the factor `quality_react` takes: the volumes are
        in `flow_volume_scale`'s EPANET m3); SETPOINT `max(c - quality, 0)`; FLOWPACED `c`.
        Concentrations keep their graph; the decisions are on detached values."""
        base = self.source_base.get(n)
        if base is None or detached(base) == 0.0:
            return 0.0
        if detached(volout) / dt <= self.q_stagnant:
            return 0.0
        c = self.source_c[n]
        kind = self.source_kind[n]
        if kind == "CONCEN":
            if self.junction[n]:
                if detached(self.demand[n]) < 0.0:
                    return -c * self.demand[n] * dt / volout
                return 0.0
            return c
        if kind == "MASS":
            return c * dt / (volout * EPANET_LITRES_PER_M3)
        if kind == "SETPOINT":
            gap = c - self.node[n]
            return gap if detached(gap) > 0.0 else 0.0
        return c

    # ---- results -------------------------------------------------------------------

    def node_quality(self) -> Tensor:
        """Every node's quality (n_nodes, `model.node_names`, quality units): `NodeQual`."""
        return self._tensor(self.node)

    def link_quality(self) -> Tensor:
        """Every link's quality (n_links, `model.link_names`): `avgqual` (`quality.c`), the
        volume-weighted mean of its segments, the mean of its end nodes' qualities when it
        holds no volume."""
        pool = self.pool
        cs, vs = pool.scalars()
        out = []
        for k in range(self.n_links):
            vsum, msum = 0.0, 0.0
            for seg in pool.segments(k):
                vsum = vsum + vs[seg]
                msum = msum + cs[seg] * vs[seg]
            if detached(vsum) > 0.0:
                out.append(msum / vsum)
            else:
                u, v = self.ends[k]
                out.append((self.node[u] + self.node[v]) / 2.0)
        return self._tensor(out)

    def _stored_mass(self):
        """`findstoredmass` (`quality.c`): the mass `c v` of every segment of every link
        and tank chain (reservoirs have none), quality x m3."""
        pool = self.pool
        cs, vs = pool.scalars()
        total = 0.0
        for chain in range(len(pool.first)):
            for seg in pool.segments(chain):
                total = total + cs[seg] * vs[seg]
        return total

    def mass_balance(self) -> QualityMassBalance:
        """The mass balance now (`evalmassbalance`, which EPANET runs after every
        `nextqual`: the state after the last `advance`), in `QualityMassBalance`'s mass
        unit. Sums of tensors: an attached run keeps their graph."""
        scale = EXACT_LITRES_PER_M3 if self.mode == "CHEMICAL" else 1.0
        initial, inflow, outflow, reacted, final = (
            torch.as_tensor(x, dtype=F64) * scale for x in (
                self.mass_initial, self.mass_in, self.mass_out, self.reacted,
                self._stored_mass()))
        massin, massout = initial + inflow, outflow + final
        if detached(reacted) > 0.0:
            massout = massout + reacted
        else:
            massin = massin - reacted
        if detached(massin) == 0.0:
            ratio = torch.ones((), dtype=F64)
        else:
            ratio = massout / massin
        return QualityMassBalance(initial, inflow, outflow, reacted, final, ratio)

    def _update_mass_balance(self, n: int, massin, volout, dt: int) -> None:
        """`updatemassbalance` (`qualroute.c`) for node `n` after its outflow was released:
        a junction loses `max(0, demand) dt` of its quality and gains `SourceQual x volout`;
        a reservoir loses its inflow mass `massin` and gains `SourceQual x volout` when that
        is positive, else its quality `x volout`; a tank gains `SourceQual x volout`."""
        if self.junction[n]:
            lost = max(0.0, self.demand[n]) * dt * self.node[n]
            added = self.source_qual * volout
        elif n in self.tank_of:
            lost = 0.0
            added = self.source_qual * volout
        else:
            lost = massin
            added = (self.source_qual if detached(self.source_qual) > 0.0
                     else self.node[n]) * volout
        self.mass_out = self.mass_out + lost
        self.mass_in = self.mass_in + added

    def _tensor(self, values: list) -> Tensor:
        if self.attached:
            return torch.stack([torch.as_tensor(x, dtype=F64) for x in values])
        return torch.tensor([float(x) for x in values], dtype=F64)

    def record(self, t: int, *, link: bool = True) -> tuple[Tensor, Tensor | None]:
        """`(node_quality, link_quality)` at the solve at `t`, before its transport (EPANET's
        `saveoutput` in `runqual`), the link quality only when `link` asks for it; both
        noted in the diagnostics."""
        node = self.node_quality()
        links = self.link_quality() if link or self.diagnostics is not None else None
        if self.diagnostics is not None:
            self.diagnostics["times"].append(float(t))
            self.diagnostics["node_quality"].append(node.detach().tolist())
            self.diagnostics["link_quality"].append(links.detach().tolist())
        return node, links if link else None

    # ---- one hydraulic step ----------------------------------------------------------

    def advance(self, t: int, step: int, *, flow: Tensor, closed: Tensor,
                demand: Tensor) -> None:
        """EPANET's `runqual` at `t` then `nextqual` over `[t, t + step]` (`quality.c`):
        the flow directions (`flowdirchanged`, the chains reversed and the nodes re-sorted
        on a change), then the transport (`transport`) in quality steps of `dt =
        min(Qstep, step - qtime)`. `flow` (n_links, m3/s) is the solve's, `closed`
        (n_links, bool) its links whose state is CLOSED (EPANET's `LINKFLOW` is 0 there),
        `demand` (n_junctions, `net.junctions` order, m3/s) each junction's delivered
        demand plus its emitter's outflow (`hydsolver.c`: `NodeDemand`). Both are taken in
        EPANET's own flow units (`flow_volume_scale`)."""
        shut = closed.tolist()
        flow, demand = flow * self.flow_scale, demand * self.flow_scale
        if self.reacts and self.mode != "AGE":
            # `ratecoeffs` (`qualreact.c`), once per hydraulic step, from EPANET's `LinkFlow`
            # (0 when CLOSED, as `piperate` takes it)
            pipes = flow[: self.n_pipes]
            pipes = torch.where(closed[: self.n_pipes], torch.zeros_like(pipes), pipes)
            self.rc = wall_coefficients(pipes, self.pipe_diameter, self.pipe_length, self.kw,
                                        order=self.wall_order, viscosity=self.viscosity,
                                        diffusivity=self.diffusivity)
        if self.attached:
            values = list(flow.unbind())
            demands = list(demand.unbind())
        else:
            values = flow.detach().tolist()
            demands = demand.detach().tolist()
        self.flow = [0.0 if shut[k] else values[k] for k in range(self.n_links)]
        node_demand = [0.0] * self.n_nodes
        for m, d in zip(self.junction_nodes, demands, strict=True):
            node_demand[m] = d
        self.demand = node_demand
        if self._directions_changed(t):
            self._sort(t)
        self._source_strengths(t)
        qtime = 0
        while qtime < step:
            dt = min(self.qstep, step - qtime)
            qtime += dt
            self.transport(dt)

    def _directions_changed(self, t: int) -> bool:
        """`flowdirchanged` (`quality.c`): each link's direction, the sign of its flow (0
        on a closed link, and below `Q_STAGNANT` in magnitude, in EPANET's cfs:
        `Q_STAGNANT_CFS / rounded_cfs_per_m3s(units)` m3/s); a link's chain is reversed
        only when the new direction times the old one is negative (positive, then stagnant,
        then negative is never reversed); True when any direction changed."""
        changed = False
        for k in range(self.n_links):
            q = detached(self.flow[k])
            new = (q > 0.0) - (q < 0.0)
            if abs(q) < self.q_stagnant:
                new = 0
            old = self.direction[k]
            if new * old < 0:
                self.pool.reverse(k)
                if self.diagnostics is not None:
                    self.diagnostics["reversals"].append((t, self.link_names[k]))
            if new != old:
                changed = True
            self.direction[k] = new
        return changed

    def _sort(self, t: int) -> None:
        """`sortnodes` over EPANET's indices, the order kept in `model.node_names`
        indices."""
        direction = [self.direction[k] for k in self._link_of_epanet]
        order, breaks = sort_nodes(self.n_nodes, self._epanet_ends, direction)
        self.order = [self._node_of_epanet[e] for e in order]
        if self.diagnostics is not None:
            self.diagnostics["node_order"].append(
                (t, [self.node_names[m] for m in self.order]))
            self.diagnostics["cycle_breaks"].append(breaks)

    def transport(self, dt: int) -> None:
        """`transport` (`qualroute.c`) over one quality step of `dt` s: each node in the
        sorted order takes the inflow of its links whose flow ends there (`evalnodeinflow`)
        and counts the outflow of the others (`|LINKFLOW|`) and, at a junction, its
        demand when positive; its quality follows (`findnodequal`) and is released into
        its outflow links (`evalnodeoutflow`). A stagnant link (direction 0) counts as
        flowing from its first node to its second, with its actual flow. A reactive
        substance reacts first (`reactpipes`, `reacttanks`: `react`)."""
        if self.reacts:
            self.react(dt)
        pool = self.pool
        pool.open()
        record = self.diagnostics is not None
        if record:
            inflow_v = [0.0] * self.n_nodes
            inflow_m = [0.0] * self.n_nodes
            outflow_v = [0.0] * self.n_nodes
        for n in self.order:
            volin, massin, volout = 0.0, 0.0, 0.0
            for k in self.adjacency[n]:
                u, v = self.ends[k]
                if (u if self.direction[k] < 0 else v) == n:
                    volin, massin = self._inflow(k, dt, volin, massin)
                else:
                    volout = volout + abs(self.flow[k])
            if self.junction[n]:
                volout = volout + max(0.0, self.demand[n])
            volout = volout * dt
            quality = self._node_quality(n, volin, massin, volout, dt)
            for k in self.adjacency[n]:
                u, v = self.ends[k]
                if (v if self.direction[k] < 0 else u) == n:
                    self._outflow(k, quality, dt)
            if record:
                inflow_v[n], inflow_m[n], outflow_v[n] = (detached(volin), detached(massin),
                                                          detached(volout))
            self._update_mass_balance(n, massin, volout, dt)
        pool.close()
        if record:
            self.diagnostics["segments"].append(pool.counts())
            self.diagnostics["inflow_volume"].append(inflow_v)
            self.diagnostics["inflow_mass"].append(inflow_m)
            self.diagnostics["outflow_volume"].append(outflow_v)

    def react(self, dt: int) -> None:
        """`reactpipes` and `reacttanks` (`qualreact.c`) over `dt` s, one vectorised update
        of the pool: every segment of a pipe (`Type == PIPE`) and of a tank takes
        `max(0, c + dc)` (`pipereact`, `tankreact`; explicit Euler), its coefficients
        gathered by its chain. AGE: `dc = dt / 3600` h. A CHEMICAL: a pipe's `dc =
        bulk_rate(c, kb, ORDER BULK) dt + wall_rate(c, d, kw, Rc, ORDER WALL) dt`, a tank's
        `bulk_rate(c, kt, ORDER TANK) dt`. The mass lost, `(c - c_new) v`, adds to
        `reacted`."""
        pool = self.pool
        n = pool.size
        if n == 0:
            return
        owner = torch.tensor(pool.owner, dtype=torch.long)
        kind = self._chain_kind[owner]
        pipe = (kind == 1).nonzero().squeeze(1)
        tank = (kind == 2).nonzero().squeeze(1)
        index = torch.cat([pipe, tank])
        if index.numel() == 0:
            return
        c = pool.c[:n]
        if self.mode == "AGE":
            new = torch.clamp(c[index] + dt / 3600.0, min=0.0)
        else:
            k = owner[pipe]
            cp = c[pipe]
            rbulk = bulk_rate(cp, self.kb[k], self.bulk_order, self.limit, BULK_TINY)
            rwall = wall_rate(cp, self.pipe_diameter[k], self.kw[k], self.rc[k],
                              order=self.wall_order)
            new_pipe = torch.clamp(cp + (rbulk * dt + rwall * dt), min=0.0)
            i = owner[tank] - self.n_links
            ct = c[tank]
            rtank = bulk_rate(ct, self.kt[i], self.tank_order, self.limit, BULK_TINY)
            new_tank = torch.clamp(ct + rtank * dt, min=0.0)
            new = torch.cat([new_pipe, new_tank])
        old = c[index]
        self.reacted = self.reacted + ((old - new) * pool.v[index]).sum()
        pool.c = pool.c.index_put((index,), new)

    def _inflow(self, k: int, dt: int, volin, massin):
        """`evalnodeinflow` (`qualroute.c`): the volume `|q| dt` of link `k` taken from its
        leading segments into its downstream node, each used-up segment removed; the
        node's inflow volume and mass with it added."""
        pool = self.pool
        cs, vs = pool.cs, pool.vs
        v = abs(self.flow[k]) * dt
        while detached(v) > 0.0:
            seg = pool.first[k]
            if seg < 0:
                break
            vseg = vs[seg]
            vseg = vseg if detached(vseg) < detached(v) else v        # MIN(vseg, v)
            volin = volin + vseg
            massin = massin + vseg * cs[seg]
            v = v - vseg
            if detached(v) >= 0.0 and detached(vseg) >= detached(vs[seg]):
                pool.remove_first(k)
            else:
                vs[seg] = vs[seg] - vseg
        return volin, massin

    def _node_quality(self, n: int, volin, massin, volout, dt: int):
        """`findnodequal` (`qualroute.c`): a junction's quality is its inflow's mass over
        its volume, the volume diluted by a negative demand (`volin -= MIN(0, demand)
        dt`); with no inflow it keeps its quality, or, for a reactive substance, takes its
        links' adjoining segments' mean (`noflowqual`). A tank mixes its inflow by its model
        (`mixtank`: `quality_tanks`). A reservoir keeps its quality. In TRACE the trace
        node holds 100 (`SourceQual` 100 at a reservoir, else `max(100 - quality, 0)`).
        An external source adds its concentration (`findsourcequal`, CHEMICAL only): onto a
        junction's quality, onto a tank's outflow only, and in place of a reservoir's (which
        keeps its previous quality when 0 is added). The quality released downstream is
        returned."""
        if self.junction[n]:
            volin = volin - min(0.0, self.demand[n]) * dt
            if detached(volin) > 0.0:
                self.node[n] = massin / volin
            elif self.reactive:
                self.node[n] = self._no_flow_quality(n)
        elif n in self.tank_of:
            self.node[n] = self._mix_tank(self.tank_of[n], volin, massin, volin - volout)
        self.source_qual = 0.0          # `SourceQual`, for `updatemassbalance`
        if self.mode == "TRACE":
            if n == self.trace:
                # the tracer added: 100 at a reservoir, else what the node lacks of it
                self.source_qual = (TRACE_QUALITY if n not in self.tank_of
                                    and not self.junction[n]
                                    else max(TRACE_QUALITY - self.node[n], 0.0))
                self.node[n] = TRACE_QUALITY
            return self.node[n]
        added = self._source_quality(n, volout, dt)
        self.source_qual = added
        if detached(added) == 0.0:
            return self.node[n]
        if self.junction[n]:
            self.node[n] = self.node[n] + added
            return self.node[n]
        if n in self.tank_of:
            return self.node[n] + added
        self.node[n] = added
        return added

    def _no_flow_quality(self, n: int):
        """`noflowqual` (`qualroute.c`): the mean, over node `n`'s links (its adjacency
        order), of the segment at `n`'s end of each: the leading one (`FirstSeg`) of a link
        whose flow ends at `n` (`N2 == n` and direction >= 0, or `N1 == n` and direction <
        0), else the last one (`LastSeg`); a link without segments is left out, and 0 when
        none has one."""
        pool = self.pool
        total, count = 0.0, 0
        for k in self.adjacency[n]:
            u, v = self.ends[k]
            d = self.direction[k]
            inflow = (v == n and d >= 0) or (u == n and d < 0)
            if inflow and pool.first[k] >= 0:
                total = total + pool.cs[pool.first[k]]
                count += 1
            elif not inflow and pool.last[k] >= 0:
                total = total + pool.cs[pool.last[k]]
                count += 1
        return total / count if count else 0.0

    def _mix_tank(self, i: int, vin, win, vnet):
        """`mixtank` (`qualreact.c`) for tank `i` by its `[MIXING]` model
        (`quality_tanks.mix_tank`); the tank's quality is kept in `tank_c` (`Stank.C`)."""
        model, vmz = self.tank_mix[i]
        self.tank_c[i] = mix_tank(self.pool, self.n_links + i, model, vmz, self.ctol, vin,
                                  win, vnet, self.tank_c[i])
        return self.tank_c[i]

    def _outflow(self, k: int, c, dt: int) -> None:
        """`evalnodeoutflow` (`qualroute.c`): the volume `|q| dt` released into link `k` at
        quality `c`, merged into its last (upstream) segment when within `Ctol` of its
        quality, else added as a new segment."""
        v = abs(self.flow[k]) * dt
        if detached(v) == 0.0:
            return
        pool = self.pool
        seg = pool.last[k]
        if seg >= 0 and self.ctol.merges(pool.cs[seg], c):
            cs, vs = pool.cs, pool.vs
            cs[seg] = (cs[seg] * vs[seg] + c * v) / (vs[seg] + v)
            vs[seg] = vs[seg] + v
        else:
            pool.add(k, v, c)


__all__ = ["EXACT_LITRES_PER_M3", "LINKVOL_FACTOR", "QualityMassBalance", "QualityRouter",
           "SegmentPool", "TRACE_QUALITY", "flow_volume_scale", "quality_step", "sort_nodes"]
