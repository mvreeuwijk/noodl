"""What closed links, check valves and full or empty tanks do to one solve of a water
network: the cut-off analysis around `water_steady`, `water_step` and `link_states`.
Internal to `noodl.apps.water`.

Before a solve, `cut_off_plan` refuses by name what cannot be solved (a demand cut off from
every reservoir and tank, by closed links or by the directions check valves leave open) and
marks one closed link per cut-off group without demand, so the Jacobian is not singular
(`PIN_KEY`), and every PRV or PSV with a dead end (`DEAD_END_KEY`). After it, its `limit`
puts every part the solve leaves without a head of its own at EPANET's limit head (the
convex program of `_limit`, solved by `noodl.solvers.qp.solve_qp`), and `refuse_wrong_way`
refuses a converged check valve or pump carrying flow the wrong way.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from noodl._unionfind import UnionFind
from noodl.apps.water.controls import ACTIVE, CLOSED, OPEN
from noodl.apps.water.demand import PressureDrivenDemand
from noodl.apps.water.elements import (
    DEAD_END_KEY,
    ONE_WAY_KEY,
    PIN_KEY,
    SETTING_KEY,
    STATUS_KEY,
    FlowControlledPipe,
)
from noodl.apps.water.graph import breadth_first, joined_groups, link_topology
from noodl.model import Model
from noodl.solvers.qp import InfeasibleConstraints, solve_qp

Tensor = torch.Tensor
F64 = torch.float64
State = dict[str, Tensor]
Drivers = dict[str, Tensor]


def solve_status(model: Model, state: State, drivers: Drivers) -> Tensor:
    """The `"water.status"` the solve sees: the drivers' own, or, on a network with tanks,
    the tank closure's (its controls switch the pumps; the closure remembers its last
    answer, so the solve's own call that follows costs nothing), with every pump at speed 0
    (`"water.setting"`) CLOSED."""
    if model.control_closure is not None:
        status = model.control_closure(state, drivers)[STATUS_KEY]
    elif STATUS_KEY not in drivers:
        raise KeyError(
            f"water: driver {STATUS_KEY!r} is required and was not given; build the drivers "
            f"with initial_drivers(model)")
    else:
        status = drivers[STATUS_KEY]
    layer = model.kirchhoff["water"]
    speed = drivers.get(SETTING_KEY)
    if "pump" not in layer.kinds or speed is None:
        return status
    cols = layer.law_for("pump")[1]
    stopped = speed[..., cols].detach() == 0
    if not bool(stopped.any()):
        return status
    # a pump at speed 0 is closed (`hydcoeffs.c`, `pumpcoeff`)
    status = torch.broadcast_to(status, stopped.shape[:-1] + status.shape[-1:]).clone()
    status[..., cols] = torch.where(stopped, float(CLOSED), status[..., cols])
    return status


@dataclass(frozen=True)
class _CutOff:
    """What closed links do to one network for one set of open links and demands
    (`cut_off_plan`): the junctions cut off with a demand (`fed_none`) and the closed links
    whose row ties a cut-off group to its neighbour during the solve (`ties`)."""

    fed_none: tuple[str, ...]
    ties: tuple[int, ...]


def _cut_off_groups(model: Model, open_links: tuple, demand: tuple) -> _CutOff:
    """`_CutOff` for one set of open links (per link) and demands (per junction; junction
    `i` is node `i`, `build_model`'s node order).

    The groups are the connected parts of the open links. A group with no reservoir or tank
    and no demand has no head of its own in the exact equations (`_limit` gives it EPANET's
    after the solve); so that the solve's Jacobian is not singular, one closed link per
    group, chosen breadth-first from the reservoirs and tanks, ties the group to its
    neighbour during the solve (`ties`, the links `PIN_KEY` marks)."""
    ends, fixed = link_topology(model)
    group, fed = joined_groups(ends, fixed, open_links)
    junctions = model.water_network.junctions
    fed_none = tuple(j.name for i, (j, d) in enumerate(zip(junctions, demand, strict=True))
                     if d and group[i] not in fed)
    if fed_none or fed.issuperset(group.values()):
        return _CutOff(fed_none, ())
    neighbours: dict[int, list] = {}
    for j, ((u, v), is_open) in enumerate(zip(ends, open_links, strict=True)):
        gu, gv = group[u], group[v]
        if not is_open and gu != gv:
            neighbours.setdefault(gu, []).append((j, gv))
            neighbours.setdefault(gv, []).append((j, gu))
    # Every group is reached: the layer refuses, when built, a node no link joins to a
    # reservoir or tank.
    reached = breadth_first(fed, neighbours)
    return _CutOff((), tuple(j for j in reached.values() if j is not None))


def _groups(model: Model, open_links: tuple, demand: tuple) -> _CutOff:
    """`_cut_off_groups`, remembered per set of open links and demands: an extended-period
    run meets the same few sets again and again."""
    cache = model.__dict__.setdefault("_cut_off", {})
    key = (open_links, demand)
    if key not in cache:
        cache[key] = _cut_off_groups(model, open_links, demand)
    return cache[key]


def cut_off_plan(model: Model, state: State, drivers: Drivers, caller: str):
    """`(drivers, limit)` for one solve by `caller`, after refusing what cannot be solved.

    Refused by name: a junction with a demand that closed links cut off from every
    reservoir and tank (EPANET reports it disconnected), one that no water reaches through
    the open links in the directions check valves and full or empty tanks leave them
    (`_one_way_cut_off`), and a status or direction on a
    pipe in its potential form, which reads neither. The demands are the drivers'
    `"water.sources"`, or, under PDA, the build's demands: a pressure-driven demand is not a
    driver, so a PDA demand changed after the build is not seen here.

    A cut-off group without demand is solved, as EPANET solves it: `drivers` gains
    `PIN_KEY` on one closed link per group (`_cut_off_groups`), which ties the group's head
    to that link's far end with every flow zero to round-off, and `limit(solution)` then gives
    the heads with every part cut off AT THE SOLUTION at EPANET's limit, and which check
    valves holding no flow are open there (`_limit`). `limit` is `None` when no link is
    closed and none is one-way."""
    net = model.water_network
    layer = model.kirchhoff["water"]
    status = solve_status(model, state, drivers).detach()
    one_way = drivers.get(ONE_WAY_KEY)
    if model.control_closure is not None:
        # a full or empty tank's links (`TankLevels`): the closure's direction, which the
        # solve's own call of it gives again
        given = model.control_closure(state, drivers).get(ONE_WAY_KEY)
        if given is not None:
            one_way = given
            drivers = {**drivers, ONE_WAY_KEY: given}
    if "pipe" in layer.kinds and not isinstance(layer.law_for("pipe")[0], FlowControlledPipe):
        cols = layer.law_for("pipe")[1]
        if bool((status[..., cols] != OPEN).any()) or (
                one_way is not None and bool((one_way[..., cols] != 0).any())):
            raise ValueError(
                f"{caller}: a pipe is closed or one-way in 'water.status'/'water.one_way' (a "
                f"CLOSED or CV pipe, or a pipe to a full or empty tank), which pipes take "
                f"only in their resistance form; build the model with pipe_form='flow'")
    n_j = len(net.junctions)
    n_l = len(model.link_names)
    closed = status == CLOSED
    one_way_any = one_way is not None and bool(one_way.any())
    pressure_valves = "prv" in layer.kinds or "psv" in layer.kinds
    if not n_j or not (bool(closed.any()) or one_way_any or pressure_valves):
        return drivers, None
    if any(isinstance(w, PressureDrivenDemand) for w in layer.withdrawals):
        demand = torch.tensor([j.demand != 0 for j in net.junctions])
    else:
        demand = drivers["water.sources"][..., :n_j].detach() != 0
    if pressure_valves:
        dead = _dead_ends(model, status, demand)
        if dead is not None:
            drivers = {**drivers, DEAD_END_KEY: dead}
        if not (bool(closed.any()) or one_way_any):
            return drivers, None
    if bool(closed.any()):
        open_, wanted = torch.broadcast_tensors(~closed[..., None, :], demand[..., :, None])
        batch = open_.shape[:-2]
        pattern = torch.cat([open_[..., 0, :], wanted[..., 0]], dim=-1).reshape(-1, n_l + n_j)
        rows, inverse = torch.unique(pattern, dim=0, return_inverse=True)
        plans = [_groups(model, tuple(row[:n_l]), tuple(row[n_l:]))
                 for row in rows.tolist()]
        fed_none = sorted({name for p in plans for name in p.fed_none})
        if fed_none:
            raise ValueError(
                f"{caller}: junction(s) {fed_none} are cut off from every reservoir and tank "
                f"by closed links, yet have a demand: no head can deliver it")
        if any(p.ties for p in plans):
            ties = torch.zeros(len(plans), n_l, dtype=F64)
            for k, plan in enumerate(plans):
                ties[k, list(plan.ties)] = 1.0
            drivers = {**drivers, PIN_KEY: ties[inverse].reshape(batch + (n_l,))}
    if one_way_any:
        if any(isinstance(w, PressureDrivenDemand) for w in layer.withdrawals):
            withdrawal = torch.tensor([float(j.demand) for j in net.junctions], dtype=F64)
        else:
            withdrawal = -drivers["water.sources"][..., :n_j].detach()
        unfed = _one_way_cut_off(model, closed, one_way.detach(), withdrawal)
        if unfed:
            raise ValueError(
                f"{caller}: junction(s) {unfed} have a demand that no reservoir, tank or "
                f"inflow reaches through the open links in the direction they let water "
                f"flow: check valves (CV pipes, or the pipes of a full or empty tank) cut "
                f"them off, so no head can deliver it")

    def limit(solution: State) -> tuple[Tensor, Tensor]:
        return _limit(model, solution, closed, demand, one_way, caller)

    return drivers, limit


def _one_way_cut_off(model: Model, closed: Tensor, one_way: Tensor,
                     withdrawal: Tensor) -> list[str]:
    """The junctions with a demand (``withdrawal > 0``, m3/s) that no water can reach, over
    every batch instance (in node order; empty when none): those no path of links that are not
    CLOSED, each taken in a direction its ``one_way`` allows, leads to from a reservoir, a
    tank or a junction with an inflow (``withdrawal < 0``). Every flow that delivers a
    demand runs along such a path, so these demands have no solution; the undirected test
    of `_cut_off_groups` misses them when the only links to a reservoir or tank are check
    valves pointing away (a pipe into an empty tank fills it only). Exact, with pumps and
    valves taken both ways (a superset of their paths), and remembered per set of closed
    links, directions and demand signs."""
    closed, one_way, withdrawal = torch.broadcast_tensors(
        closed[..., None, :], one_way[..., None, :], withdrawal[..., :, None])
    closed, one_way, withdrawal = closed[..., 0, :], one_way[..., 0, :], withdrawal[..., 0]
    n_l = closed.shape[-1]
    pattern = torch.cat([closed.to(F64), one_way.to(F64), torch.sign(withdrawal)],
                        dim=-1).reshape(-1, 2 * n_l + withdrawal.shape[-1])
    cache = model.__dict__.setdefault("_one_way_cut_off", {})
    unfed: set[str] = set()
    for row in torch.unique(pattern, dim=0).tolist():
        key = tuple(row)
        if key not in cache:
            cache[key] = _one_way_unfed(model, row[:n_l], row[n_l:2 * n_l], row[2 * n_l:])
        unfed.update(cache[key])
    return [name for name in model.node_names if name in unfed]


def _one_way_unfed(model: Model, closed: list, one_way: list, sign: list) -> tuple:
    """`_one_way_cut_off` for one set of closed links, directions and demand signs.

    `sign` is per junction, and junction `i` is node `i`: `build_model` puts the junctions
    first in `model.node_names`, in `net.junctions` order (the `"water.sources"` columns
    `[:n_j]` are theirs). Checked here, once per pattern."""
    junctions = [j.name for j in model.water_network.junctions]
    assert list(model.node_names[:len(sign)]) == junctions, "junctions first in node_names"
    ends, fixed = link_topology(model)
    ahead: dict[int, list[tuple]] = {}
    for j, ((u, v), shut, way) in enumerate(zip(ends, closed, one_way, strict=True)):
        if shut:
            continue
        if way >= 0:
            ahead.setdefault(u, []).append((j, v))
        if way <= 0:
            ahead.setdefault(v, []).append((j, u))
    reached = breadth_first([*fixed, *(i for i, s in enumerate(sign) if s < 0)], ahead)
    return tuple(model.node_names[i] for i, s in enumerate(sign)
                 if s > 0 and i not in reached)


def _dead_ends(model: Model, status: Tensor, demand: Tensor) -> Tensor | None:
    """`DEAD_END_KEY` for one solve: per link, +1 where a PRV or PSV that is ACTIVE has its
    end node in a dead end, -1 its start node, 0 elsewhere; `None` when no valve has one.
    A valve with both sides dead (in a part without demand that CLOSED links cut off from
    every reservoir and tank) is refused by name.

    A dead end of a valve is the group of junctions on one side of it that only the valve
    joins to a reservoir or tank (through links that are not CLOSED) and that has no
    demand. No water can flow through the valve then, and the exact condition leaves the
    group's head anywhere that keeps the valve shut; the valve's law then ties the group
    (`_PressureValve`, A DEAD END), as EPANET's answer does. Remembered per set of statuses
    and demands."""
    layer = model.kirchhoff["water"]
    status, demand = torch.broadcast_tensors(
        status.detach()[..., None, :], demand[..., :, None])
    status, demand = status[..., 0, :], demand[..., 0]
    batch = status.shape[:-1]
    n_l = status.shape[-1]
    pattern = torch.cat([status, demand.to(status.dtype)], dim=-1).reshape(-1, n_l
                                                                          + demand.shape[-1])
    rows, inverse = torch.unique(pattern, dim=0, return_inverse=True)
    cache = model.__dict__.setdefault("_dead_ends", {})
    cols = [layer.law_for(kind)[1] for kind in ("prv", "psv") if kind in layer.kinds]
    out = []
    for row in rows.tolist():
        key = tuple(row)
        if key not in cache:
            cache[key] = _dead_ends_of(model, row[:n_l], row[n_l:], cols)
        out.append(cache[key])
    if not any(any(r) for r in out):
        return None
    table = torch.tensor(out, dtype=F64)
    return table[inverse].reshape(batch + (n_l,))


def _dead_ends_of(model: Model, status: list, demand: list, cols: list) -> list[float]:
    """`_dead_ends` for one set of link statuses and junction demands (lists)."""
    ends, fixed = link_topology(model)
    n_j = len(demand)
    out = [0.0] * len(status)
    for sl in cols:
        for j in range(sl.start, sl.stop):
            if status[j] != ACTIVE:
                continue
            group, fed = joined_groups(ends, fixed, tuple(
                k != j and status[k] != CLOSED for k in range(len(status))))
            for side, node in ((1.0, ends[j][1]), (-1.0, ends[j][0])):
                g = group[node]
                if g in fed:
                    continue
                if not any(demand[i] for i in range(n_j) if group.get(i) == g):
                    if out[j]:
                        raise ValueError(
                            f"water: valve {model.link_names[j]!r} joins two parts of the "
                            f"network without demand that CLOSED links cut off from every "
                            f"reservoir and tank, so its state and those heads are "
                            f"undetermined; open one of those links, or close the valve")
                    out[j] = side
    return out


#: A flow below this (m3/s) is zero for a check valve: it holds no flow.
NO_FLOW = 1e-12


def _limit(model: Model, solution: State, closed: Tensor, demand: Tensor,
           one_way: Tensor | None, caller: str) -> tuple[Tensor, Tensor]:
    """`(heads, open)`: `solution`'s heads with every zero-demand part of the network that
    the solve leaves without a head of its own set to EPANET's limit head, and, per link,
    whether a check valve holding no flow is open there.

    A link carrying flow, or open in both directions, ties its ends' heads (it is "solid");
    a node joined by solid links to a reservoir or tank is "reliable" and keeps the solve's
    own head. Every other node is "free": it sits behind closed links (status CLOSED) and
    check valves holding no flow, where the exact equations leave its head anywhere that
    keeps those valves shut. EPANET gives each closed link the conductance `1 / CBIG`, and
    opens a check valve whose upstream head is the higher; the free heads of this model are
    the limit of EPANET's as `CBIG` grows: the solution of the convex quadratic program

        minimise   1/2 sum over the closed links and the check valves holding no flow,
                   (a, b), of (H_a - H_b)^2
        subject to H_u <= H_v for every such check valve u -> v

    (reverse-only pipes with the ends swapped), the heads outside the structure held at the
    solve's. Each link of the structure is a resistor of one conductance (EPANET's shut
    check valve has `1 / CBIG` too), and each check valve is in addition an ideal diode in
    parallel (open, EPANET's is `1 / RQtol`, 1e15 times larger): the KKT conditions are
    the node balances of that resistor-and-diode network, with the constraints' multipliers
    the diodes' flows (in units of `1 / CBIG`), so a check valve is open where its multiplier
    is positive and shut where it is zero. The objective is strictly convex on the free
    heads (every free node reaches a reliable one through the structure), so the heads are
    unique, and `solve_qp` (`noodl.solvers.qp`) finds them in finitely many steps. The heads are a
    linear map of the solve's heads for the active set found, so a gradient flows through
    them. Solid links among free nodes carry no flow and tie their ends.

    The multipliers, and so the states, are unique only where the active constraints'
    normals are independent. Parallel check valves (the same ends, the same way) share one
    constraint row and one multiplier, so they are open or shut together, as EPANET has
    them. Other dependence (a triangle Ra -> J1, J1 -> J2, Ra -> J2, J1 and J2 both tied to
    Ra) leaves a split of the flow among the valves undetermined; the method never adds a
    constraint the active ones already satisfy, so such a valve has multiplier 0 and is
    CLOSED, and the water goes through the valves it added first, the most violated as it
    went (MEASURED: that triangle's states are EPANET's).

    A junction with a demand among the free nodes is refused by name: no head delivers it.
    At a converged solution that cannot happen where EPANET solves (a demand junction is
    reliable: its water reaches it through links carrying flow); it fires where a
    converged check valve holds no flow against a demand behind it, which the solve's
    branch selection refuses first (`refuse_wrong_way`)."""
    phi = solution["water.phi"]
    net = model.water_network
    n, n_j, n_l = phi.shape[-1], len(net.junctions), len(model.link_names)
    q = solution["water.q"].detach()
    if one_way is None:
        one_way = torch.zeros(n_l, dtype=q.dtype)
    still = (one_way != 0) & (q.abs() <= NO_FLOW) & ~closed
    solid = ~closed & ~still
    batch = torch.broadcast_shapes(phi.shape[:-1], solid.shape[:-1], demand.shape[:-1],
                                   one_way.shape[:-1])
    flat_in = phi.expand(batch + (n,)).reshape(-1, n)
    out = flat_in.clone()
    open_ = torch.zeros(flat_in.shape[0], n_l, dtype=torch.bool)
    solid = solid.expand(batch + (n_l,)).reshape(-1, n_l).tolist()
    still = still.expand(batch + (n_l,)).reshape(-1, n_l).tolist()
    closed = closed.expand(batch + (n_l,)).reshape(-1, n_l).tolist()
    demand = demand.expand(batch + (n_j,)).reshape(-1, n_j).tolist()
    direction = one_way.expand(batch + (n_l,)).reshape(-1, n_l).tolist()
    ends, fixed = link_topology(model)
    tol = 1e-9 * model.head_scale
    for b in range(out.shape[0]):
        group, fed = joined_groups(ends, fixed, tuple(solid[b]))
        free = [i for i in range(n) if group.get(i, i) not in fed]
        if not free:
            continue
        for i in free:
            if i < n_j and demand[b][i]:
                raise ValueError(
                    f"{caller}: junction {net.junctions[i].name!r} has a demand but is "
                    f"joined to every reservoir and tank only through closed links and "
                    f"check valves holding no flow")
        structure = [j for j in range(n_l) if closed[b][j] or still[b][j]]
        rows, far, coefficients, opened = _limit_heads(
            ends, free, structure, [j for j in structure if still[b][j]], direction[b],
            solid[b], flat_in[b].detach(), tol, model, caller)
        out[b, rows] = coefficients.to(phi.dtype) @ flat_in[b][far]
        open_[b, opened] = True
    return out.reshape(batch + (n,)), open_.reshape(batch + (n_l,))


def _limit_heads(ends, free, structure, valves, direction, solid, values, tol, model,
                 caller):
    """`(rows, far, coefficients, opened)` of `_limit` for one instance: the free nodes
    `rows`, whose limit heads are `coefficients @ values[far]` (`far` the reliable nodes
    the structure touches), and the check valves `opened` (the links among `valves` whose
    multiplier is positive).

    The unknowns are the tie groups of the free nodes (joined by solid links, which carry no
    flow there). The program is assembled on them (`G`, `a` from the `structure` links;
    one constraint per check valve whose two ends are not tied or both reliable) and solved
    by `solve_qp` (`noodl.solvers.qp`); its active set, as ties, then gives the heads
    exactly: a group tied to a reliable node takes that node's head, the others solve the
    Laplacian over the structure links with unit weights."""
    is_free = [False] * len(values)
    for i in free:
        is_free[i] = True
    ties: UnionFind[int] = UnionFind(free)
    for j, (u, v) in enumerate(ends):
        if solid[j] and is_free[u] and is_free[v]:
            ties.union(u, v)
    variables = list(dict.fromkeys(ties.find(i) for i in free))
    k_of = {g: k for k, g in enumerate(variables)}
    of_node = {i: k_of[ties.find(i)] for i in free}
    far: list[int] = []
    far_of: dict[int, int] = {}

    def far_index(node: int) -> int:
        if node not in far_of:
            far_of[node] = len(far)
            far.append(node)
        return far_of[node]

    # the links as (variable or None, variable or None, far index or None): the free ends
    # as variables, a reliable end as a far node
    links = []
    for j in structure:
        u, v = ends[j]
        ku = of_node.get(u) if is_free[u] else None
        kv = of_node.get(v) if is_free[v] else None
        if ku is None and kv is None:
            links.append(None)
        elif ku is not None and kv is not None:
            links.append((ku, kv, None) if ku != kv else None)
        elif ku is not None:
            links.append((ku, None, far_index(v)))
        else:
            links.append((None, kv, far_index(u)))
    n_var = len(variables)
    # one constraint per distinct row (variable +1 or None, variable -1 or None, far +1,
    # far -1), with the check valves that share it: parallel valves give one row, else the
    # multipliers would not be unique and a duplicate would never enter the active set
    constraints: dict[tuple, list[int]] = {}
    for j in valves:
        u, v = ends[j]
        lo, hi = (u, v) if direction[j] > 0 else (v, u)
        k_lo = of_node.get(lo) if is_free[lo] else None
        k_hi = of_node.get(hi) if is_free[hi] else None
        if (k_lo is None and k_hi is None) or (k_lo is not None and k_lo == k_hi):
            continue
        # H_hi - H_lo >= 0, as c x >= b with the fixed heads on the right
        constraints.setdefault((k_hi, k_lo, far_index(lo) if k_lo is None else None,
                                far_index(hi) if k_hi is None else None), []).append(j)
    n_far = len(far)
    g = torch.zeros(n_var, n_var, dtype=F64)
    b_far = torch.zeros(n_var, n_far, dtype=F64)   # the objective's `a` is `-b_far h_far`
    for link in links:
        if link is None:
            continue
        ku, kv, f = link
        if f is None:
            g[ku, ku] += 1.0
            g[kv, kv] += 1.0
            g[ku, kv] -= 1.0
            g[kv, ku] -= 1.0
        else:
            k = ku if ku is not None else kv
            g[k, k] += 1.0
            b_far[k, f] += 1.0
    h_far = values[far]
    active: list[int] = []
    multipliers: list[float] = []
    rows = list(constraints)
    if rows:
        c = torch.zeros(len(rows), n_var, dtype=F64)
        d = torch.zeros(len(rows), n_far, dtype=F64)
        for row, (k_hi, k_lo, f_lo, f_hi) in enumerate(rows):
            if k_hi is not None:
                c[row, k_hi] += 1.0
            if k_lo is not None:
                c[row, k_lo] -= 1.0
            if f_lo is not None:
                d[row, f_lo] += 1.0
            if f_hi is not None:
                d[row, f_hi] -= 1.0
        try:
            _, active, multipliers = solve_qp(g, -b_far @ h_far, c, d @ h_far, tol)
        except InfeasibleConstraints as exc:
            names = [model.link_names[j] for row in exc.rows for j in constraints[rows[row]]]
            raise ValueError(
                f"{caller}: the check valves {names} hold no flow, yet no heads keep them "
                f"all shut: the solve's state is not a solution") from exc
    opened = [j for row, u in zip(active, multipliers, strict=True) if u > tol
              for j in constraints[rows[row]]]
    # The heads from the active set, as ties: exact where tied to a reliable node.
    merged: UnionFind[int] = UnionFind(range(n_var))
    anchor: dict[int, int] = {}
    for row in active:
        k_hi, k_lo, f_lo, f_hi = rows[row]
        if k_hi is not None and k_lo is not None:
            merged.union(k_hi, k_lo)
        elif k_hi is not None:
            anchor[k_hi] = f_lo
        else:
            anchor[k_lo] = f_hi
    root = [merged.find(k) for k in range(n_var)]
    anchored = {}
    for k, f in anchor.items():
        anchored.setdefault(root[k], f)
    unknown = [r for r in dict.fromkeys(root) if r not in anchored]
    u_of = {r: m for m, r in enumerate(unknown)}
    lap = torch.zeros(len(unknown), len(unknown), dtype=F64)
    rhs = torch.zeros(len(unknown), n_far, dtype=F64)
    for link in links:
        if link is None:
            continue
        ku, kv, f = link
        ru = None if ku is None else root[ku]
        rv = None if kv is None else root[kv]
        if ru is not None and ru == rv:
            continue
        for a, other, f_other in ((ru, rv, f), (rv, ru, f)):
            if a not in u_of:
                continue
            lap[u_of[a], u_of[a]] += 1.0
            if other in u_of:
                lap[u_of[a], u_of[other]] -= 1.0
            else:
                rhs[u_of[a], f_other if other is None else anchored[other]] += 1.0
    per_unknown = torch.linalg.solve(lap, rhs) if unknown else rhs
    coefficients = torch.zeros(n_var, n_far, dtype=F64)
    for k in range(n_var):
        if root[k] in anchored:
            coefficients[k, anchored[root[k]]] = 1.0
        else:
            coefficients[k] = per_unknown[u_of[root[k]]]
    return free, far, coefficients[[of_node[i] for i in free]], opened


def refuse_wrong_way(model: Model, state: State, drivers: Drivers, caller: str) -> None:
    """Refuse a converged one-way pipe carrying flow against its direction, or a pump
    carrying flow backwards. A check valve or pump an iterate would close is kept open
    where closing it would cut a junction off (`FlowControlledPipe`,
    `FlowControlledPump`); where the solution itself needs it closed with the junction
    then cut off, it ends with that wrong-way flow, and the network cannot be solved."""
    layer = model.kirchhoff["water"]
    if "pump" in layer.kinds:
        cols = layer.law_for("pump")[1]
        q = state["water.q"][..., cols].detach()
        wrong = (q < -1e-9).reshape(-1, q.shape[-1]).any(0)
        if bool(wrong.any()):
            names = [model.link_names[cols.start + j] for j in range(len(wrong)) if wrong[j]]
            raise ValueError(
                f"{caller}: pump(s) {names} carry flow backwards: with them shut a junction "
                f"is cut off from every reservoir and tank")
    one_way = drivers.get(ONE_WAY_KEY)
    if one_way is None or "pipe" not in layer.kinds or not bool(one_way.any()):
        return
    cols = layer.law_for("pipe")[1]
    q = state["water.q"][..., cols].detach()
    wrong = (q * one_way[..., cols] < -1e-9).reshape(-1, q.shape[-1]).any(0)
    if bool(wrong.any()):
        names = [model.link_names[cols.start + j] for j in range(len(wrong)) if wrong[j]]
        raise ValueError(
            f"{caller}: check valve(s) {names} carry flow against their direction (a CV "
            f"pipe, or a pipe to a full or empty tank): with them closed a junction is cut "
            f"off from every reservoir and tank")
