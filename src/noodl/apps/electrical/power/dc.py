"""The DC power flow: a `PowerNetwork` as plain nodal analysis on a classic real
`PotentialFlowLayer` in per unit.

The DC approximation (all voltage magnitudes 1 p.u., no branch resistance or charging,
small angle differences) makes the real power on a branch linear in the voltage angles.
MATPOWER's `makeBdc` writes it as `Pf = Bf θ + Pfinj` with, per in-service branch from bus
`f` to bus `t` of series reactance `x`, tap ratio `τ` (0 means 1) and phase shift `σ`
(radians),

    b    = 1 / (x τ)
    Pf   = b (θ_f - θ_t) - b σ,

and `dcpf` solves the bus balance `Bbus θ = Pbus - Pbusinj - Gs / base_mva` for every bus
but the slack, whose angle is fixed. Here that is one classic layer `"power_dc"`:

- Potential: the voltage angle θ (rad) at the nodes `str(bus.id)`, in bus-table order; no
  ground node. The slack bus is the layer's one boundary node, at `radians(Va_slack)`.
- Edges: one kind `"branch"`, an edge `from -> to` per in-service branch in file order with
  a `Conductance(1 / (x τ))`, and one `ConstantDrive("branch", "power_dc.shift")` holding
  `-σ` per branch, so `dp = θ_f - θ_t - σ` and the edge flow `q = (θ_f - θ_t - σ) / (x τ)`
  is MATPOWER's `Pf` (p.u.). The phase shifts' injections `Pbusinj` are thereby part of the
  branch flows, not of the sources.
- Sources (p.u., the bus's net injection): `(Σ Pg - Σ Pd - Gs) / base_mva` over the bus's
  in-service generators (on PV, PQ and slack buses alike), loads and fixed shunts; zero at
  the slack bus, whose balance the boundary condition replaces.
- `r`, `b`, the bus `Bs` and the reactive demand play no part.

The layer has only `Conductance` laws and a drive, so it is solved by one linear solve of
the weighted graph Laplacian (`layer.mna` is `False`). Driver keys:
`"power_dc.phi_boundary"` (`[θ_slack]`, rad), `"power_dc.sources"` (full-node injections,
p.u.) and `"power_dc.shift"` (per branch, rad); the state is empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from noodl.apps.electrical.power.build import PowerFlowIndex, _analyse
from noodl.apps.electrical.power.network import PowerNetwork
from noodl.drives import ConstantDrive
from noodl.elements.conductance import Conductance
from noodl.layers.potential import PotentialFlowLayer
from noodl.model import Model
from noodl.topology import Network

Tensor = torch.Tensor
F64 = torch.float64
State = dict[str, Tensor]
Drivers = dict[str, Tensor]

#: The name of the DC power-flow layer: state keys `"power_dc.phi"` (bus voltage angles,
#: rad) and `"power_dc.q"` (branch real powers, p.u.); drivers `"power_dc.phi_boundary"`,
#: `"power_dc.sources"` and `"power_dc.shift"`.
DC_LAYER = "power_dc"


@dataclass(frozen=True)
class DcPowerFlowSolution:
    """A solved DC power flow in per unit; tensors may carry leading batch dimensions.

    `va` (radians) per bus of `bus_ids`; `pg` (p.u.) the total real generation per bus of
    `gen_buses`; `pf` (p.u.) the real power entering each branch of `branch_rows` at its
    from end (in the DC approximation the same power leaves at its to end).
    """

    bus_ids: tuple[int, ...]
    va: Tensor
    gen_buses: tuple[int, ...]
    pg: Tensor
    branch_rows: tuple[int, ...]
    pf: Tensor


def _f64(values) -> Tensor:
    return torch.tensor([float(v) for v in values], dtype=F64)


def dc_power_flow(net: PowerNetwork) -> tuple[Model, State, Drivers]:
    """The DC power flow of `net` as `(model, state, drivers)`, by the build rules of
    `noodl.apps.electrical.power.dc`; solve it with `solve_power_flow`.

    The network is checked first, as `check_power_network(net, dc=True)` does: exactly one
    slack bus with an in-service generator, no in-service branch of zero reactance, every
    bus connected to the slack bus through in-service branches (otherwise a `ValueError`
    naming the offending branch or buses). A network without an in-service branch is
    refused too. `model.power_index` is the `PowerFlowIndex`,
    `model.power_network` the network; the state is empty (the solve is linear).
    """
    index = _analyse(net, dc=True).index
    base = net.base_mva
    slack = index.slack_bus

    graph = Network(dtype=F64)
    for bus in net.buses:
        graph.add_node(str(bus.id))
    branches = [net.branches[k] for k in index.branch_rows]
    if not branches:
        raise ValueError(
            f"PowerNetwork {net.name!r}: the DC power flow needs at least one in-service "
            f"branch; a network of the slack bus alone has nothing to solve"
        )
    for br in branches:
        graph.add_edge(str(br.from_bus), str(br.to_bus), kind="branch")
    law = Conductance(_f64(1.0 / (br.x * (br.ratio or 1.0)) for br in branches), kind="branch")
    shift = ConstantDrive("branch", f"{DC_LAYER}.shift")
    layer = PotentialFlowLayer(graph, DC_LAYER, [law], drives=[shift], boundary=[str(slack)],
                               quantity="voltage angle", unit="rad", flow_unit="p.u.")
    model = Model(graph, [layer])
    model.power_index = index
    model.power_network = net

    injection = {b.id: 0.0 for b in net.buses}
    for g in net.generators:
        if g.in_service:
            injection[g.bus] += g.p_mw
    for ld in net.loads:
        injection[ld.bus] -= ld.p_mw
    for sh in net.shunts:
        injection[sh.bus] -= sh.g_mw
    injection[slack] = 0.0
    va_slack = next(b.va_deg for b in net.buses if b.id == slack)
    drivers: Drivers = {
        f"{DC_LAYER}.phi_boundary": _f64([math.radians(va_slack)]),
        f"{DC_LAYER}.sources": _f64(injection[b.id] / base for b in net.buses),
        f"{DC_LAYER}.shift": _f64(-math.radians(br.shift_deg) for br in branches),
    }
    return model, {}, drivers


def dc_power_flow_solution(model: Model, state: State) -> DcPowerFlowSolution:
    """Bus angles, generation per generator bus and branch flows of a solved DC power flow
    (`model` from `dc_power_flow`).

    The slack bus's generation is the power its branches carry away plus its own demand
    and shunt conductance, `Σ_out Pf - Σ_in Pf + (Σ Pd + Gs) / base_mva`; every other
    generator bus keeps its in-service generators' summed setpoint.
    """
    index: PowerFlowIndex = model.power_index
    net: PowerNetwork = model.power_network
    phi, q = state[f"{DC_LAYER}.phi"], state[f"{DC_LAYER}.q"]
    slack = index.slack_bus
    base = net.base_mva

    rows = index.branch_rows
    out = [k for k, row in enumerate(rows) if net.branches[row].from_bus == slack]
    into = [k for k, row in enumerate(rows) if net.branches[row].to_bus == slack]
    demand = (sum(ld.p_mw for ld in net.loads if ld.bus == slack)
              + sum(sh.g_mw for sh in net.shunts if sh.bus == slack))
    slack_pg = q[..., out].sum(-1) - q[..., into].sum(-1) + demand / base

    powers = []
    for bus in index.gen_buses:
        if bus == slack:
            powers.append(slack_pg)
        else:
            pg = sum(g.p_mw for g in net.generators if g.in_service and g.bus == bus)
            powers.append(torch.full_like(slack_pg, pg / base))
    return DcPowerFlowSolution(
        bus_ids=index.bus_ids, va=phi, gen_buses=index.gen_buses,
        pg=torch.stack(powers, dim=-1), branch_rows=rows, pf=q,
    )


__all__ = ["DC_LAYER", "DcPowerFlowSolution", "dc_power_flow", "dc_power_flow_solution"]
