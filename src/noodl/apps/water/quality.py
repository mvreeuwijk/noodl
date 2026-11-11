"""EPANET 2.2's water-quality routing, transcribed (`quality.c`, `qualroute.c`, tag v2.2).

EPANET's answer is a discrete Lagrangian procedure, not the advection equation's: every pipe
and tank holds a chain of volume segments (`initsegs`); once per hydraulic step the flow
directions are taken (`flowdirchanged`), a chain is reversed when its flow changes sign
(`reversesegs`) and the nodes are sorted from upstream to downstream (`sortnodes`,
`selectnonstacknode`); then every quality step walks the nodes in that order (`transport`),
taking each link's inflow from its leading segments (`evalnodeinflow`), mixing it at the
node (`findnodequal`, a tank by its mixing model `tankmix1`) and releasing the node's water
into its outflow links, merged into their last segment when within `Ctol` of it
(`evalnodeoutflow`). `QualityRouter` is that procedure; `water_run(quality=True)` drives it
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

import torch

from noodl.apps.water.quality_data import WaterQuality
from noodl.apps.water.units import Q_STAGNANT_CFS, rounded_cfs_per_m3s

Tensor = torch.Tensor
F64 = torch.float64

#: `qualroute.c`, `LINKVOL`: EPANET's pi/4, to six digits (`0.785398 L D^2`).
LINKVOL_FACTOR = 0.785398
#: EPANET's tracer concentration at the trace node (`quality.c`, `initqual`;
#: `qualroute.c`, `findnodequal`).
TRACE_QUALITY = 100.0
_M3_PER_FT3 = 0.3048**3


def _value(x) -> float:
    """A decision's operand, detached: `x` as a Python float (Ruling Q5)."""
    return float(x.detach()) if isinstance(x, Tensor) else float(x)


def flow_volume_scale(flow_units: str | None) -> float:
    """The factor from a flow in m3/s to the volume per second EPANET's quality moves, in
    m3: EPANET divides a file's flows by its own ROUNDED factors (`types.h`, e.g.
    `LPSperCFS = 28.317` against the exact 28.3168466; `input1.c`, `convertunits`) but
    its lengths by the exact foot, so in cfs times the exact ft3 a flow moves
    `rounded_cfs_per_m3s(units) * 0.3048^3` of the volume it carries in SI (1 - 5.4e-6 for
    LPS). Without a file unit, 1."""
    cfs = rounded_cfs_per_m3s(flow_units)
    return 1.0 if cfs is None else cfs * _M3_PER_FT3


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
    recycled through the free list (`free`, chained by `prev`), as EPANET recycles it.

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
    while m < len(xs) - 1 and _value(x) > xs[m]:
        m += 1
    dx = xs[m] - xs[m - 1]
    if abs(dx) < 1e-6:
        return ys[m]
    return ys[m] - (xs[m] - x) * (ys[m] - ys[m - 1]) / dx


def _refuse_later_features(net, quality: WaterQuality) -> None:
    """The features of later tasks, refused by name until they are modelled."""
    if quality.mode == "AGE":
        raise ValueError("quality: QUALITY AGE (water age) is not yet modelled")
    reactive = (quality.global_bulk, quality.global_wall, quality.roughness_correlation,
                *quality.pipe_bulk.values(), *quality.pipe_wall.values(),
                *quality.tank_bulk.values())
    if quality.mode == "CHEMICAL" and any(k != 0.0 for k in reactive):
        raise ValueError("quality: [REACTIONS] with a nonzero bulk, wall, tank or "
                         "roughness-correlation coefficient are not yet modelled")
    if quality.sources:
        raise ValueError("quality: [SOURCES] (quality sources) are not yet modelled")
    for tank in net.tanks:
        model, _ = quality.mixing.get(tank.name, ("MIXED", 1.0))
        if model != "MIXED":
            raise ValueError(f"quality: [MIXING] {tank.name} {model}: tank mixing models "
                             f"other than MIXED are not yet modelled")


class QualityRouter:
    """EPANET 2.2's quality routing of `model`'s network under `quality` (module
    docstring): `initqual` and `initsegs` at construction, then `advance` once per
    hydraulic step.

    `inputs["quality.initial"]` (n_nodes, `model.node_names`, quality units) replaces the
    network's initial qualities (`quality.initial`, 0 where absent);
    `inputs["quality.tank_level"]` (n_tanks, m) the tanks' initial levels (`water_run`
    passes its initial state's), from which their initial segment volumes are taken.
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
        _refuse_later_features(net, quality)
        self.net = net
        self.quality = quality
        self.mode = quality.mode
        self.qstep = int(quality_step)
        self.attached = attached
        self.ctol = float(quality.tolerance)
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
        for i, tank in enumerate(net.tanks):
            self.pool.add(n_links + i, _tank_volume(net, tank, levels[i]),
                          c0[node_index[tank.name]])
        self.pool.close()

        self.direction = [0] * n_links         # `FlowDir`, ZERO_FLOW at the start
        self.order: list[int] = []              # `SortedNodes`, none until the first sort
        self.diagnostics = None
        if diagnostics is not None:
            self.diagnostics = {
                "times": [], "node_quality": [], "link_quality": [], "node_order": [],
                "cycle_breaks": [], "reversals": [], "segments": [], "inflow_volume": [],
                "inflow_mass": [], "outflow_volume": []}
            diagnostics.setdefault("quality", []).append(self.diagnostics)

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
            if _value(vsum) > 0.0:
                out.append(msum / vsum)
            else:
                u, v = self.ends[k]
                out.append((self.node[u] + self.node[v]) / 2.0)
        return self._tensor(out)

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
            q = _value(self.flow[k])
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
        flowing from its first node to its second, with its actual flow."""
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
                inflow_v[n], inflow_m[n], outflow_v[n] = (_value(volin), _value(massin),
                                                          _value(volout))
        pool.close()
        if record:
            self.diagnostics["segments"].append(pool.counts())
            self.diagnostics["inflow_volume"].append(inflow_v)
            self.diagnostics["inflow_mass"].append(inflow_m)
            self.diagnostics["outflow_volume"].append(outflow_v)

    def _inflow(self, k: int, dt: int, volin, massin):
        """`evalnodeinflow` (`qualroute.c`): the volume `|q| dt` of link `k` taken from its
        leading segments into its downstream node, each used-up segment removed; the
        node's inflow volume and mass with it added."""
        pool = self.pool
        cs, vs = pool.cs, pool.vs
        v = abs(self.flow[k]) * dt
        while _value(v) > 0.0:
            seg = pool.first[k]
            if seg < 0:
                break
            vseg = vs[seg]
            vseg = vseg if _value(vseg) < _value(v) else v        # MIN(vseg, v)
            volin = volin + vseg
            massin = massin + vseg * cs[seg]
            v = v - vseg
            if _value(v) >= 0.0 and _value(vseg) >= _value(vs[seg]):
                pool.remove_first(k)
            else:
                vs[seg] = vs[seg] - vseg
        return volin, massin

    def _node_quality(self, n: int, volin, massin, volout, dt: int):
        """`findnodequal` (`qualroute.c`): a junction's quality is its inflow's mass over
        its volume, the volume diluted by a negative demand (`volin -= MIN(0, demand)
        dt`); with no inflow it keeps its quality (nothing reacts: `noflowqual` is for a
        reactive substance). A tank mixes its inflow by its model (`mixtank`; MIXED:
        `tankmix1`). A reservoir keeps its quality. In TRACE the trace node holds 100
        (`SourceQual` 100 at a reservoir, else `max(100 - quality, 0)`). The quality
        released downstream is returned."""
        if self.junction[n]:
            volin = volin - min(0.0, self.demand[n]) * dt
            if _value(volin) > 0.0:
                self.node[n] = massin / volin
        elif n in self.tank_of:
            self.node[n] = self._mix_tank(self.tank_of[n], volin, massin, volin - volout)
        if self.mode == "TRACE" and n == self.trace:
            self.node[n] = TRACE_QUALITY
        return self.node[n]

    def _mix_tank(self, i: int, vin, win, vnet):
        """`mixtank` and `tankmix1` (`qualreact.c`), a completely mixed tank: its segment's
        quality `(c v + win) / (v + vin)` when `v + vin > 0`, its volume `max(0, v +
        vnet)`; the tank's quality is its segment's (unchanged without one)."""
        pool = self.pool
        seg = pool.first[self.n_links + i]
        if seg >= 0:
            cs, vs = pool.cs, pool.vs
            vnew = vs[seg] + vin
            if _value(vnew) > 0.0:
                cs[seg] = (cs[seg] * vs[seg] + win) / vnew
            vs[seg] = vs[seg] + vnet
            if _value(vs[seg]) < 0.0:          # MAX(0.0, seg->v)
                vs[seg] = 0.0
            self.tank_c[i] = cs[seg]
        return self.tank_c[i]

    def _outflow(self, k: int, c, dt: int) -> None:
        """`evalnodeoutflow` (`qualroute.c`): the volume `|q| dt` released into link `k` at
        quality `c`, merged into its last (upstream) segment when within `Ctol` of its
        quality, else added as a new segment."""
        v = abs(self.flow[k]) * dt
        if _value(v) == 0.0:
            return
        pool = self.pool
        seg = pool.last[k]
        if seg >= 0 and abs(_value(pool.cs[seg]) - _value(c)) < self.ctol:
            cs, vs = pool.cs, pool.vs
            cs[seg] = (cs[seg] * vs[seg] + c * v) / (vs[seg] + v)
            vs[seg] = vs[seg] + v
        else:
            pool.add(k, v, c)


__all__ = ["LINKVOL_FACTOR", "QualityRouter", "SegmentPool", "TRACE_QUALITY",
           "flow_volume_scale", "quality_step", "sort_nodes"]
