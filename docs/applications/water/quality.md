# Water distribution: water quality

How to run EPANET 2.2's water-quality simulation (water age, a chemical, or the trace of a source) along with the hydraulics, what it reads and what it refuses. The hydraulics it follows are described on [Extended-period simulation](extended-period.md); importing the quality sections of a file is on [Importing EPANET files](epanet.md). The overview is [Water distribution](index.md).

## Overview

`water_run` carries EPANET 2.2's water-quality simulation along with the hydraulics: the
Lagrangian procedure of `quality.c`, `qualroute.c` and `qualreact.c`, transcribed, so it
reproduces EPANET's answer and not merely the advection-reaction equation's. EPANET moves
volumes of water through pipes as segments, merges neighbours that differ by less than a
tolerance and reacts each by an explicit Euler step; an Eulerian discretisation would differ
from it however fine. The hydraulics are untouched: quality reads each step's flows, link
states and delivered demands and never feeds back, so a run is bit-identical with it on and
off. The quality is read from the `.inp` file ([importing EPANET files](epanet.md)) into
`WaterNetwork.quality` (a `WaterQuality`, in SI) and run by `QualityRouter`
(`noodl.apps.water.quality`).

## Run a quality simulation

`quality=None` (the default) runs it when the network defines a mode and the
duration is positive, [as described for `water_run`](extended-period.md#running-over-time). The results carry:

| Field | Shape | Meaning |
| --- | --- | --- |
| `node_quality` | `(..., T, n_nodes)` | each node's quality at the report times, in the file's units; a tank's is the quality of the water it releases |
| `link_quality` | `(..., T, n_links)` | the volume-weighted mean over the link's segments (the mean of its end nodes for a link with none) |
| `quality_units` | `str` | `mg/L`, `hrs`, `%` or the file's |
| `mass_balance` | `QualityMassBalance`, fields of shape `(...,)` | initial, inflow, outflow, reacted and final mass and their ratio, as EPANET's report writes them, in physical units (a chemical in its mass unit, AGE and TRACE as quality times m³) |

All are `None` when no quality ran. `diagnostics={}` additionally receives
`diagnostics["quality"]`, one entry per instance: the node order of every sort, the cycle
breaks, the flow reversals, the segment counts per link and quality step, and the per-node
inflow and outflow.

```python
from noodl.apps.water import read_epanet_inp, build_model, water_run

net = read_epanet_inp("Net1.inp")        # chlorine, bulk -0.5 per day, wall -1 ft per day
model, state, drivers = build_model(net)
run = water_run(model, duration=6 * 3600)

print(run.quality_units, tuple(run.node_quality.shape), tuple(run.link_quality.shape))
print([round(float(c), 3) for c in run.node_quality[:4, run.node_names.index("11")]])
print(round(float(run.mass_balance.ratio), 4))
```

```text
mg/L (7, 11) (7, 13)
[0.5, 0.444, 0.866, 0.864]
1.0
```

## What is modelled

### Modes

(`[OPTIONS] QUALITY`). `CHEMICAL` (named, with its units, `Chlorine mg/L`;
concentrations stay in the file's unit), `AGE` (hours: every segment gains the quality step as it
moves) and `TRACE <node>` (percent of the water that came from that node: the node is held at
100 and every other starts at 0 whatever the file's initial qualities say, except a tank's own
segments). `NONE`, or a duration of 0, runs no quality.

### Reactions

(`[REACTIONS]`). In the bulk water, order `n` with a limiting concentration
(`bulkrate`: growth towards or decay to the limit; a negative order is Michaelis-Menten). At
the wall, order 0 (mass per area per day) or 1 (length per day), limited by mass transfer from
the Sherwood number (laminar, transitional, turbulent) with the diffusivity relative to
chlorine's, `DIFFUSIVITY 0` switching the limitation off; a pipe with no wall coefficient of
its own takes one from the roughness correlation (Hazen-Williams, Darcy-Weisbach,
Chezy-Manning). In tanks, one global order and a coefficient per tank (default: the global bulk
coefficient). A reaction step is `c = max(0, c + (rbulk + rwall) dt)`; pumps and valves never
react. The wall coefficients are recomputed once per hydraulic step from that step's flow.

### Sources

(`[SOURCES]`). `CONCEN` (at a junction only where its demand is negative), `MASS`
(per minute), `SETPOINT` and `FLOWPACED`, each with an optional pattern. A reservoir whose
source goes to zero keeps its last quality, as in EPANET.

### Tank mixing

(`[MIXING]`). `MIXED`, `2COMP` (a mixing zone and a stagnant zone, the zone
fraction given), `FIFO` and `LIFO`. A tank's quality tracks its own segment volumes, not the
hydraulic tank volume.

## Learnable inputs

Optional overrides of what the file says: `state["quality.initial"]`
(n_nodes, quality units) and the drivers `"quality.bulk_coefficient"` and
`"quality.wall_coefficient"` (n_pipes, in `net.pipes` order; (quality unit)^(1-n)/s for a
bulk reaction of order n, 1/s for order 1; m/s for a wall coefficient of order 1, quality
units per m² per s of order 0), `"quality.tank_coefficient"` (n_tanks, (quality
unit)^(1-n)/s for the bulk order n, 1/s for order 1) and `"quality.source_strength"`
(n_nodes, scaled by the source's pattern; a MASS source in mass per second). The file gives
its coefficients per day (and in its own length units); the reader converts them to these
per-second SI values. A wrong shape, or any of them on a run without quality, is refused
by name.

## Differentiability

Volumes and concentrations stay in the autograd graph; only the
decisions are made on detached values: whether a segment is used up, whether two segments merge
(`|c1 - c2| < Ctol`), the flow direction and the node order. Segment volumes are `|q| dt` from
the solve's flows, so the quality is piecewise differentiable with respect to the reaction
coefficients, the source strengths, the initial quality and the hydraulic parameters (demands,
roughnesses, a pump's speed) through the travel times. It jumps where a decision flips: where a
merge, a reversal or the node order differs between `p - h` and `p + h`, a central difference
is not the derivative. Autograd is on only when a quality input or the solve's flows require
gradients (`differentiable=True` for the hydraulics); otherwise the router runs detached, since
the graph grows linearly with the quality steps (about 60 nodes per step for a coefficient, 135
for a hydraulic parameter, on the verification network).

## Batches

A batched run builds one router per instance (each has its own segments and node
order) and stacks the results `(..., T, n)`; the `mass_balance` fields have shape `(...,)`. The
hydraulic step is shared by the batch, [as described for `water_run`](extended-period.md#running-over-time): an instance takes the
others' step boundaries, so its quality is its own single run's only when no instance has an
event (a tank filling, a control) that the others lack. Run instances one at a time to
reproduce EPANET's sequence for each.

## EPANET behaviours transcribed as they are

These are reproduced as EPANET has them, not corrected:

- A check-valve pipe (`CV`) holds no initial segment and never reacts; a pump's or valve's
  chain is filled and emptied within the step and never reacts.
- A flow below `Q_STAGNANT` (0.005 gpm) has no direction: the link is left out of the node
  order and its water moves from its first node. A closed link has zero flow. Segments are
  reversed only when the direction changes from positive to negative or the reverse, not
  through a stagnant interval.
- A `2COMP` tank with no net flow loses the mass of its stagnant zone (the zone's volume is
  set to 0), so the mass balance ratio is below 1 there, in EPANET and here alike.
- The last hydraulic step is transported only if it ends within the duration: EPANET does not
  cut it at `DURATION`, so a last step that overruns it is not routed.
- A reservoir's `[QUALITY]` is its quality, kept until a source overwrites it; a reservoir
  without a source adds `quality x outflow` to the mass balance.
- EPANET's rounded unit factors. It divides a file's flows by a rounded factor (28.317 L/s per
  cfs) but its lengths by the exact foot, so the volume its quality moves is 5.4e-6 smaller than
  the exact SI volume (in LPS files; other units differ), and a `MASS` source and an order 0 wall
  coefficient use 28.317 L per ft³ against 1000 L per m³. These are transcribed. The mass
  balance reports physical mass, so it carries no unit factor; EPANET's terms are divided by it
  when compared.
- Segments merge when their quality differs by less than `Ctol` (`TOLERANCE`, default 0.01 in
  the quality unit). The decision is made in EPANET's internal unit (a chemical divided by
  28.317 L per ft³), where a tie at exactly `Ctol` can be decided otherwise than in mg/L.

## Refused by name

The following are refused by name: a bulk or tank reaction order between 0 and 1 (EPANET's `bulkrate` raises
zero to a negative power at zero concentration and gives NaN or infinity: there is no finite
behaviour to reproduce), `ORDER WALL` other than 0 or 1, a Darcy-Weisbach roughness not below
the diameter under a roughness correlation (EPANET runs it, with an unphysical value), a
quality step that rounds to 0, a quality input on a run without quality, and the steady tracer
alongside the EPANET quality (give `quality=False` for that model).

