"""The AC power flow: a `PowerNetwork` as one phasor `PotentialFlowLayer` in per unit.

Build rules (MATPOWER's power-flow semantics: `makeYbus`, `bustypes`, `makeSbus`):

- Units: powers in p.u. on the network's `base_mva` (MW / `base_mva`, MVAr / `base_mva`),
  voltages in p.u. (root-mean-square phasors), angles in radians.
- Nodes: `str(bus.id)` for every bus in table order, then `"ground"`. The layer's boundary
  is `["ground", str(slack)]`: ground at 0 and the slack bus at `vm_set e^{j va}` (its
  generators' voltage setpoint, the angle from the bus table).
- Edges, by kind, in this order:
  `"branch"`: for each in-service branch in file order, `str(from) -> ground` then
  `str(to) -> ground`, one `PiBranch` pair per branch;
  `"load"`: one `ConstantPowerLoad` edge per bus with a nonzero net fixed demand
  `(sum Pd - sum fixed Pg) + j (sum Qd - sum fixed Qg)`, where the fixed generators are the
  in-service generators of the buses solved as PQ;
  `"shunt"`: one `Shunt` edge per bus with a nonzero fixed admittance (`Gs`, `Bs` over the
  base);
  `"generator"`: one `VoltageControlledGenerator` edge per voltage-controlled (PV) bus with
  in-service generation, at the sum of its generators' real power and their common
  voltage setpoint.
  A kind with no edges is left out.
- Generators and branches out of service are dropped; generators on one bus are summed; a
  PV bus without an in-service generator is solved as PQ. Reactive-power limits are not
  enforced.
- State (a flat start): `"power.phi"` is the voltage setpoint at the regulated buses (the
  slack and the PV buses with generation) and 1 elsewhere, every bus at the slack angle,
  0 at ground; `"power.q"` is zero. Drivers: `"power.phi_boundary"` (in the layer's
  boundary order) and `"power.frequency"` (Hz; no power-flow law reads it, the phasor layer
  requires it).

`check_power_network` refuses, by name, a network these rules cannot solve: not exactly one
slack bus, a slack bus without an in-service generator, a regulated bus whose in-service
generators disagree on the voltage setpoint, a bus not connected to the slack bus through
in-service branches, and a branch of zero impedance (`r = x = 0`; `x = 0` for the DC power
flow).
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from noodl.apps.electrical.loads import ConstantPowerLoad
from noodl.apps.electrical.power.laws import PiBranch, Shunt, VoltageControlledGenerator
from noodl.apps.electrical.power.network import PowerNetwork
from noodl.layers.potential import PotentialFlowLayer
from noodl.model import Model
from noodl.topology import Network

Tensor = torch.Tensor
F64 = torch.float64
C128 = torch.complex128
State = dict[str, Tensor]
Drivers = dict[str, Tensor]

#: The name of the AC power-flow layer: state keys `"power.phi"` (bus voltages, p.u., the
#: last node is ground) and `"power.q"` (edge currents, p.u.); drivers
#: `"power.phi_boundary"` and `"power.frequency"`.
POWER_LAYER = "power"
_GROUND = "ground"


@dataclass(frozen=True)
class PowerFlowIndex:
    """How a `PowerNetwork` maps onto a power-flow model.

    `bus_ids` in table order; `slack_bus`; `pv_buses`, the voltage-controlled buses with
    in-service generation (table order); `gen_buses`, the buses with at least one in-service
    generator (table order); `branch_rows`, the positions in `net.branches` of the
    in-service branches; `base_mva`; `fixed_generation`, the summed in-service generation
    (p.u.) of every bus solved as PQ that has some; `load_buses` and `shunt_buses`, the
    buses carrying a load and a shunt edge (table order).
    """

    bus_ids: tuple[int, ...]
    slack_bus: int
    pv_buses: tuple[int, ...]
    gen_buses: tuple[int, ...]
    branch_rows: tuple[int, ...]
    base_mva: float
    fixed_generation: Mapping[int, complex]
    load_buses: tuple[int, ...]
    shunt_buses: tuple[int, ...]


@dataclass(frozen=True)
class PowerFlowSolution:
    """A solved power flow in per unit; tensors may carry leading batch dimensions.

    `vm` (p.u.) and `va` (radians) per bus of `bus_ids`; `pg`, `qg` (p.u.) the total
    generation per bus of `gen_buses`; `sf`, `st` the complex power (p.u.) entering each
    branch of `branch_rows` at its from and to end.
    """

    bus_ids: tuple[int, ...]
    vm: Tensor
    va: Tensor
    gen_buses: tuple[int, ...]
    pg: Tensor
    qg: Tensor
    branch_rows: tuple[int, ...]
    sf: Tensor
    st: Tensor


def _regulated_setpoints(net: PowerNetwork, gens_at: dict[int, list], regulated) -> dict:
    setpoints = {}
    for bus in regulated:
        values = sorted({g.vm_set for g in gens_at[bus]})
        if len(values) > 1:
            raise ValueError(
                f"PowerNetwork {net.name!r}: bus {bus}: its in-service generators have "
                f"different voltage setpoints {values}; give them one setpoint"
            )
        setpoints[bus] = values[0]
    return setpoints


def check_power_network(net: PowerNetwork, *, dc: bool = False) -> PowerFlowIndex:
    """Check that `net` is a solvable power-flow network and return its `PowerFlowIndex`.

    Refused with a `ValueError`: not exactly one slack bus (the ones found are listed); a
    slack bus without an in-service generator; a regulated bus (the slack, a PV bus with
    in-service generation) whose in-service generators have different voltage setpoints
    (bus and values named); an in-service branch of zero impedance, `r = x = 0` for the AC
    power flow and `x = 0` for the DC one (`dc=True`; row, from and to bus named); buses not
    connected to the slack bus through in-service branches (listed). A PV bus without an
    in-service generator is solved as PQ, as MATPOWER does.
    """
    slacks = [b.id for b in net.buses if b.kind == "slack"]
    if len(slacks) != 1:
        raise ValueError(
            f"PowerNetwork {net.name!r}: a power flow needs exactly one slack bus, found "
            f"{len(slacks)}: {slacks}"
        )
    slack = slacks[0]
    gens_at: dict[int, list] = {b.id: [] for b in net.buses}
    for g in net.generators:
        if g.in_service:
            gens_at[g.bus].append(g)
    if not gens_at[slack]:
        raise ValueError(
            f"PowerNetwork {net.name!r}: the slack bus {slack} has no in-service generator"
        )
    pv = tuple(b.id for b in net.buses if b.kind == "pv" and gens_at[b.id])
    _regulated_setpoints(net, gens_at, (slack, *pv))

    rows = tuple(k for k, br in enumerate(net.branches) if br.in_service)
    for k in rows:
        br = net.branches[k]
        zero = br.x == 0 if dc else (br.r == 0 and br.x == 0)
        if zero:
            what = "zero reactance (x = 0)" if dc else "zero impedance (r = x = 0)"
            raise ValueError(
                f"PowerNetwork {net.name!r}: branch {k} ({br.from_bus}-{br.to_bus}) has "
                f"{what}; merge its buses or give it an impedance"
            )

    parent = {b.id: b.id for b in net.buses}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k in rows:
        br = net.branches[k]
        parent[find(br.from_bus)] = find(br.to_bus)
    root = find(slack)
    island = [b.id for b in net.buses if find(b.id) != root]
    if island:
        raise ValueError(
            f"PowerNetwork {net.name!r}: buses not connected to the slack bus {slack} "
            f"through in-service branches: {island}"
        )

    regulated = {slack, *pv}
    base = net.base_mva
    fixed: dict[int, complex] = {}
    for b in net.buses:
        if b.id not in regulated and gens_at[b.id]:
            fixed[b.id] = complex(sum(g.p_mw for g in gens_at[b.id]),
                                  sum(g.q_mvar for g in gens_at[b.id])) / base
    demand = _net_demand(net, fixed)
    shunts = _shunt_admittance(net)
    return PowerFlowIndex(
        bus_ids=net.bus_ids(),
        slack_bus=slack,
        pv_buses=pv,
        gen_buses=tuple(b.id for b in net.buses if gens_at[b.id]),
        branch_rows=rows,
        base_mva=float(base),
        fixed_generation=fixed,
        load_buses=tuple(bus for bus, s in demand.items() if s != 0),
        shunt_buses=tuple(bus for bus, y in shunts.items() if y != 0),
    )


def _net_demand(net: PowerNetwork, fixed: Mapping[int, complex]) -> dict[int, complex]:
    """Per bus (table order): the absorbed power of its loads less its fixed generation,
    p.u."""
    mw = {b.id: 0j for b in net.buses}
    for ld in net.loads:
        mw[ld.bus] += complex(ld.p_mw, ld.q_mvar)
    return {bus: s / net.base_mva - fixed.get(bus, 0j) for bus, s in mw.items()}


def _shunt_admittance(net: PowerNetwork) -> dict[int, complex]:
    """Per bus (table order): `g + jb` of its fixed shunts, p.u."""
    mw = {b.id: 0j for b in net.buses}
    for sh in net.shunts:
        mw[sh.bus] += complex(sh.g_mw, sh.b_mvar)
    return {bus: y / net.base_mva for bus, y in mw.items()}


def _f64(values) -> Tensor:
    return torch.tensor([float(v) for v in values], dtype=F64)


def build_power_flow(
    net: PowerNetwork, *, frequency=50.0, learnable: bool = False,
    enforce_q_limits: bool = False,
) -> tuple[Model, State, Drivers]:
    """The AC power flow of `net` as `(model, state, drivers)`, by the build rules of
    `noodl.apps.electrical.power.build`; the network is checked first
    (`check_power_network`).

    `learnable=True` registers every constitutive-law coefficient (branch impedances,
    charging, taps and shifts, load powers, shunt admittances, generator powers and
    setpoints) as an `nn.Parameter`, reached as
    `model.potential["power"].element_for(kind)[0]`. `model.power_index` is the
    `PowerFlowIndex`, `model.power_network` the network. `enforce_q_limits=True` (switching
    a PV bus to PQ at a reactive-power limit) is refused: it is not implemented.
    """
    if enforce_q_limits:
        raise ValueError(
            "reactive-power limits (switching a PV bus to PQ at its limit) are not implemented"
        )
    index = check_power_network(net)
    base = net.base_mva
    buses = {b.id: b for b in net.buses}

    graph = Network(dtype=F64)
    for bus in net.buses:
        graph.add_node(str(bus.id))
    graph.add_node(_GROUND)
    laws = []
    branches = [net.branches[k] for k in index.branch_rows]
    if branches:
        for br in branches:
            graph.add_edge(str(br.from_bus), _GROUND, kind="branch")
            graph.add_edge(str(br.to_bus), _GROUND, kind="branch")
        laws.append(PiBranch(
            _f64(br.r for br in branches), _f64(br.x for br in branches),
            _f64(br.b for br in branches), _f64(br.ratio for br in branches),
            _f64(math.radians(br.shift_deg) for br in branches), learnable=learnable))
    demand = _net_demand(net, index.fixed_generation)
    if index.load_buses:
        for bus in index.load_buses:
            graph.add_edge(str(bus), _GROUND, kind="load")
        laws.append(ConstantPowerLoad(
            _f64(demand[b].real for b in index.load_buses),
            _f64(demand[b].imag for b in index.load_buses), kind="load", learnable=learnable))
    shunts = _shunt_admittance(net)
    if index.shunt_buses:
        for bus in index.shunt_buses:
            graph.add_edge(str(bus), _GROUND, kind="shunt")
        laws.append(Shunt(_f64(shunts[b].real for b in index.shunt_buses),
                          _f64(shunts[b].imag for b in index.shunt_buses),
                          learnable=learnable))
    gens_at: dict[int, list] = {b.id: [] for b in net.buses}
    for g in net.generators:
        if g.in_service:
            gens_at[g.bus].append(g)
    setpoint = {bus: gens_at[bus][0].vm_set for bus in (index.slack_bus, *index.pv_buses)}
    if index.pv_buses:
        for bus in index.pv_buses:
            graph.add_edge(str(bus), _GROUND, kind="generator")
        laws.append(VoltageControlledGenerator(
            _f64(sum(g.p_mw for g in gens_at[b]) / base for b in index.pv_buses),
            _f64(setpoint[b] for b in index.pv_buses), learnable=learnable))

    slack = str(index.slack_bus)
    layer = PotentialFlowLayer(graph, POWER_LAYER, laws, boundary=[_GROUND, slack],
                               phasor=True, quantity="voltage", unit="p.u.", flow_unit="p.u.")
    model = Model(graph, [layer])
    model.power_index = index
    model.power_network = net

    angle = math.radians(buses[index.slack_bus].va_deg)
    v_slack = cmath.rect(setpoint[index.slack_bus], angle)
    nodes = graph.nodes
    boundary = torch.tensor([v_slack if nodes[i] == slack else 0j
                             for i in layer.bound.tolist()], dtype=C128)
    start = [cmath.rect(setpoint.get(b.id, 1.0), angle) for b in net.buses] + [0j]
    state: State = {
        f"{POWER_LAYER}.phi": torch.tensor(start, dtype=C128),
        f"{POWER_LAYER}.q": torch.zeros(graph.b, dtype=C128),
    }
    drivers: Drivers = {
        f"{POWER_LAYER}.phi_boundary": boundary,
        layer.frequency_key: torch.tensor(float(frequency), dtype=F64),
    }
    return model, state, drivers


def solve_power_flow(model: Model, state: State, drivers: Drivers, *, atol: float = 1e-11,
                     rtol: float = 0.0, max_iter: int = 30, **solve_kwargs) -> State:
    """The solved power flow: `model.steady(state, drivers, atol=atol, rtol=rtol,
    max_iter=max_iter, **solve_kwargs)`. Newton stops when the residual's max-norm (p.u.)
    is below `atol`; `state` is its starting point (the flat start `build_power_flow`
    returns, or an earlier solution). Non-convergence raises a `RuntimeError` naming the
    layer."""
    return model.steady(state, drivers, atol=atol, rtol=rtol, max_iter=max_iter,
                        **solve_kwargs)


def power_flow_solution(model: Model, state: State) -> PowerFlowSolution:
    """Bus voltages, generation per generator bus and branch end powers of a solved AC
    power flow (`model` from `build_power_flow`).

    The slack bus's generation is `V conj(I)` with `I` the current its edges carry away
    (its loads and shunts are edges, so this is the generation); a PV bus's is
    `-V conj(i)` of its generator edge; a generator bus solved as PQ keeps its fixed
    generation. Branch end powers are `V_f conj(I_f)` and `V_t conj(I_t)`.
    """
    index: PowerFlowIndex = model.power_index
    net: PowerNetwork = model.power_network
    layer = model.potential[POWER_LAYER]
    phi, q = state[f"{POWER_LAYER}.phi"], state[f"{POWER_LAYER}.q"]
    position = {bus: k for k, bus in enumerate(index.bus_ids)}
    v = phi[..., : len(index.bus_ids)]

    outflow = layer._accumulate(q)
    gen_slot = {}
    if index.pv_buses:
        gen_q = q[..., layer.kind_slice("generator")]
        gen_slot = {bus: k for k, bus in enumerate(index.pv_buses)}
    powers = []
    for bus in index.gen_buses:
        vb = v[..., position[bus]]
        if bus == index.slack_bus:
            powers.append(vb * outflow[..., position[bus]].conj())
        elif bus in gen_slot:
            powers.append(-vb * gen_q[..., gen_slot[bus]].conj())
        else:
            powers.append(torch.full_like(vb, index.fixed_generation[bus]))
    gen = torch.stack(powers, dim=-1)

    if index.branch_rows:
        ends = q[..., layer.kind_slice("branch")].unflatten(-1, (-1, 2))
        f = [position[net.branches[k].from_bus] for k in index.branch_rows]
        t = [position[net.branches[k].to_bus] for k in index.branch_rows]
        sf = v[..., f] * ends[..., 0].conj()
        st = v[..., t] * ends[..., 1].conj()
    else:
        sf = st = v[..., :0]
    return PowerFlowSolution(
        bus_ids=index.bus_ids, vm=v.abs(), va=v.angle(), gen_buses=index.gen_buses,
        pg=gen.real, qg=gen.imag, branch_rows=index.branch_rows, sf=sf, st=st,
    )


__all__ = [
    "POWER_LAYER",
    "PowerFlowIndex",
    "PowerFlowSolution",
    "build_power_flow",
    "check_power_network",
    "power_flow_solution",
    "solve_power_flow",
]
