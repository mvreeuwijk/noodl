# Power flow

How to build, solve and differentiate the power flow (load flow) of a transmission or
distribution network: the balanced, positive-sequence, sinusoidal steady state of buses
joined by lines and transformers, with constant-power loads, fixed shunts and generators
that hold their bus voltage. The model and its MATPOWER semantics are in
[Physics](physics.md#power-flow-model); to read a case file see
[MATPOWER files](matpower.md).

Everything is in **per unit**: powers on the network's base `base_mva` (MW / `base_mva`,
MVAr / `base_mva`), voltages in p.u. as root-mean-square phasors, angles in radians. The
network is a `PowerNetwork` of `Bus`, `Branch`, `Generator`, `Load` and `FixedShunt` records
in MATPOWER's file units (MW, MVAr, degrees), read from a MATPOWER case file by
`read_matpower` or built by hand.

Everything lives in `noodl.apps.electrical.power`; besides the records and builders used
below it exports `read_matpower`, `check_power_network`, the three power-flow laws
`PiBranch`, `Shunt` and `VoltageControlledGenerator`, and `matpower_case`, which fetches
MATPOWER's standard cases on first use.

## Solve an AC and a DC power flow

A small network built by hand: a 132 kV triangle of lines (buses 1 to 3) feeding a 33 kV bus
(4) through a transformer with an off-nominal tap, bus 1 the slack, a generator on bus 2
holding 1.01 p.u., loads on buses 2 to 4 and a capacitor bank on bus 4. It is solved in full
and in the DC approximation:

```python
from noodl.apps.electrical.power import (
    Branch, Bus, FixedShunt, Generator, Load, PowerNetwork, build_power_flow, dc_power_flow,
    dc_power_flow_solution, power_flow_solution, solve_power_flow,
)

net = PowerNetwork(
    buses=(Bus(1, "slack"), Bus(2, "pv"), Bus(3, "pq"), Bus(4, "pq")),
    branches=(Branch(1, 2, r=0.01, x=0.06, b=0.03), Branch(1, 3, r=0.02, x=0.08, b=0.04),
              Branch(2, 3, r=0.015, x=0.07, b=0.02), Branch(3, 4, r=0.0, x=0.1, b=0.0,
                                                           ratio=0.975)),
    generators=(Generator(1, 0.0, vm_set=1.02), Generator(2, 50.0, vm_set=1.01)),
    loads=(Load(2, 20.0, 5.0), Load(3, 60.0, 20.0), Load(4, 45.0, 15.0)),
    shunts=(FixedShunt(4, 0.0, 8.0),),
    base_mva=100.0,
)
model, state, drivers = build_power_flow(net)
sol = power_flow_solution(model, solve_power_flow(model, state, drivers))
print(" ".join(f"{v:.4f}" for v in sol.vm.tolist()))
print([round(100 * p, 3) for p in sol.pg.tolist()])
dc_model, dc_state, dc_drivers = dc_power_flow(net)
dc = dc_power_flow_solution(dc_model, solve_power_flow(dc_model, dc_state, dc_drivers))
print(" ".join(f"{100 * p:.2f}" for p in dc.pf.tolist()))
```

```text
1.0200 1.0100 0.9951 1.0129
[76.121, 50.0]
18.57 56.43 48.57 45.00
```

The first line is the voltage magnitude of each bus (p.u.), the second the generation at the
two generator buses (MW: the slack bus 1 takes up the 125 MW of load less bus 2's 50 MW, plus
the losses), the third the DC real power entering each branch at its from end (MW,
lossless). The same network as a MATPOWER case file is the example of
[MATPOWER files](matpower.md). A `PowerFlowSolution` holds `vm`, `va` per bus of `bus_ids`,
`pg`, `qg` per bus of `gen_buses` (the sum of the bus's generators) and the complex powers
`sf`, `st` entering each in-service branch (`branch_rows`) at its two ends; a
`DcPowerFlowSolution` holds `va`, `pg` and `pf`.

`solve_power_flow(model, state, drivers)` runs `model.steady` with Newton to a residual
max-norm below `atol = 1e-11` p.u. (`rtol = 0`, at most 30 iterations). The five cases
verified against pandapower converge from the flat start in 4 or 5 iterations. A solve that
fails, because Newton does not converge or the system is singular (the loading exceeds what
the network can transfer, so no solution exists, or the case is malformed), raises a
`RuntimeError` that says so, chained to the solver's own error; any other error passes
through unchanged.

## What the builders refuse

`check_power_network(net, dc=False)` (called by both builders) refuses, with a `ValueError`
naming what it found: not exactly **one slack bus**; a slack bus without an in-service
generator; a regulated bus whose generators have different voltage setpoints (MATPOWER
silently takes the last), or a setpoint that is not positive; a branch of zero impedance
(`r = x = 0`; `x = 0` for DC), named by its row; for DC (`dc=True`), a branch whose `x tau` is
negative, such as a series-compensated line (the DC power flow's conductance layer needs
positive branch susceptance; the AC power flow takes it); and buses not connected to the
slack bus through in-service branches (an **island**). `dc_power_flow` also refuses a network
without an in-service branch, and `PowerNetwork` itself refuses a `base_mva` that is not
positive, a duplicate bus id and an element on a bus that does not exist. What is not
implemented is listed on the [limitations page](limitations.md#power-flow).

## Gradients

`build_power_flow(net, learnable=True)` registers every constitutive-law coefficient (branch
`r`, `x`, `b`, taps and shifts; load `P`, `Q`; shunt `g`, `b`; generator `P`, `V`) as an
`nn.Parameter`, reached as `model.kirchhoff["power"].law_for(kind)[0]`, and the solve is
differentiable through the implicit-function adjoint. The sensitivity of bus 4's voltage
magnitude to the three loads' real power (p.u./p.u.):

```python
import torch

model, state, drivers = build_power_flow(net, learnable=True)
vm = power_flow_solution(model, solve_power_flow(model, state, drivers)).vm
(grad,) = torch.autograd.grad(vm[3], model.kirchhoff["power"].law_for("load")[0].P)
print(model.power_index.load_buses, [round(g, 4) for g in grad.tolist()])
```

```text
(2, 3, 4) [-0.0005, -0.0112, -0.0192]
```

Load at bus 4 lowers its voltage most, load at bus 2, next to the voltage-controlled
generator, hardly at all. The adjoint agrees with central differences to 1e-6 relative on
this network (see the [verification page](../../reference/verification/electrical.md#verification)).
