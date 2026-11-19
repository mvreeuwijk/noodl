# Verification: water quality

Evidence for the water quality of the [water distribution application](../../applications/water/index.md) ([Water quality](../../applications/water/quality.md)); the hydraulics are on [Verification: water distribution](water.md) and the summary is on the [verification overview](index.md). Limitations are on [Limitations](../../applications/water/limitations.md).

The comparisons run in `tests/verification/test_water_quality.py` and `tests/verification/test_water_quality_networks.py`.

The reference is EPANET 2.2 through the toolkit on the committed file, with the same settings as
the hydraulics page (`ACCURACY 1e-7` through `EN_setoption`, `TRIALS 400`, 5000 and `DAMPLIMIT 1e-2` for ky10; any warning fails the
test unless it is named). Quality is read at every hydraulic step (the time of each `EN_runQ`; `EN_nextQ`
advances the quality to the next one, the loop `EN_solveQ` runs; `EN_stepQ` splits a step at a hydraulic event, which EPANET's own
runs do not). The bound, over every node and link at every hydraulic step time: each point within 1e-5 of
the quality's scale (1 for a chemical, 100 for a trace, the largest age of the run for an
age: 24 h on the base network, 6.92 h on ky10) and the worst within `Ctol`; mass-balance terms within 2e-5 relative and the ratio within
1e-6. A point beyond 1e-5 is admitted only with a measured cause asserted in the test. EPANET reads its hydraulics back as
float32 reals, a floor of about 6e-8 relative.

Per feature, on a small base network (`quality_base.inp`: a reservoir, a tank with a pump line,
a dead end), each case compared at 25 hourly times; the 54 cases also have their mass balance
compared:

| Feature | Measured (worst, of the quality's scale, away from the admitted points) |
|---|---|
| Base, CHEMICAL, TRACE at a junction, a reservoir and a tank; a dead end below `Q_STAGNANT`; a tank at its limit; a check valve; a closed link; a flow cycle | 1.3e-7 to 4.4e-7 |
| Bulk order 0, 1, 2, Michaelis-Menten, limiting potential (growth and decay) | 4.9e-8 to 2.2e-7 |
| Wall order 0 (SI and US), order 1, mass-transfer limited, `DIFFUSIVITY 0`, roughness correlation (Hazen-Williams; Darcy-Weisbach in US units) | 5.7e-8 to 2.8e-7; 2.3e-6 for Darcy-Weisbach (the hydraulics of that file, 1e-6 class) |
| Tank reaction; water age; reactive dead end and check valve (`noflowqual`) | 2.0e-7 to 2.2e-7 |
| Sources: each type, with and without a pattern, at a reservoir, a junction (positive and negative demand), a tank; SETPOINT below the node's quality; a reservoir source whose pattern reaches 0 | 1.7e-5 to 2.5e-5 at the admitted dead-end points, none beyond elsewhere |
| Mixing: MIXED, 2COMP, FIFO, LIFO; 2COMP crossing its mixing-zone volume; FIFO and LIFO emptying through several segments; 2COMP with no net flow | 3.3e-8 to 4.2e-7 |
| Mass balance, all 54 cases | terms 3.8e-6 (2e-5 bound), ratio 8.3e-8 (1e-6) |
| A duration past the last whole hydraulic step (5:30; the last step not routed) | 1.98e-6; terms 2.9e-7, ratio 2.6e-8 |
| A batch against its single runs (bulk batch bit-equal; demands 0.8 and 1.2 over 23 h) | equal; 1e-8 of scale |
| Gradients of a node's quality with respect to bulk, wall, source, initial quality (a junction and a tank) and a pipe roughness, against central differences (h = 1e-4 of the value; the +h, -h and base runs have identical segment counts and node order) | relative difference 8.5e-12 to 1.1e-8 (bound 1e-5) |

The admitted points on the base network are all at its dead-end pipe P7 (1.2e-5 to 6.3e-5 of the
scale): EPANET's solver leaves a residual flow of at most 3.7e-9 m³/s in it, and it pushes the
junction's water into the pipe while noodl's P7 carries 1e-25 m³/s. Each point is checked
against a bound computed from EPANET's cumulative flow through P7, and admitted only while that flow stays
below 1e-8 m³/s.

EPANET's example networks:

| Network | Compared | Worst | Notes |
|---|---|---|---|
| Net1 (chlorine, 24 h) | all nodes and links, 27 times | 1.6e-6 mg/L (p99.9 1.4e-6); mass terms 7.7e-7, ratio 1.8e-8 | |
| Net2 (fluoride, a `CONCEN` source on a pattern, 55 h, one tank), hydraulics | 56 steps | heads 1.1e-7 relative, levels 8.9e-6 m, flows 1.9e-8 m³/s | states and step times equal |
| Net2, `Tolerance` 0.01 moved by 1e-8 either way | all nodes and links | 4.2e-6 mg/L (p99.9 6.6e-7) | |
| Net2 as shipped (`Tolerance` 0.01) | | 2.83e-3 mg/L, within `Ctol`; 667 of 4256 points beyond 1e-5, all from 32 h | a tie at exactly `Ctol`, below |
| Net3 (TRACE from the lake, 168 h, 1226 flow reversals, 154 node sorts) | 183 times | 2.5e-4 % (p99.9 9.7e-5 %); mass terms 1.4e-6, ratio 1.3e-8 | |
| Net6 as shipped (quality identically 0) | smoke | exactly 0 in both; ratio exactly 1 | |
| Net6 with a source added, before 18922 s | 40 steps | 5.9e-6 mg/L; mass terms (whole run) 1.9e-6 | hydraulic cause below |
| Net6 with a source added, noodl's router on EPANET's saved hydraulics | all 609 times | 5.2e-11 | |
| ky4 (TRACE, DURATION edited to 24 h) | 65782 points, 31 times | 9.35e-3 % (p99.9 5.5e-4 %); terms 1.8e-6, ratio 9.0e-7; router on EPANET's hydraulics 2.0e-9 % | 33 points admitted, below |
| ky10 (AGE, DURATION edited to 6:54:59) | 35928 points, 18 times | 3.53e-3 h (p99.9 1.5e-5 h); terms 3.2e-6, ratio 6.1e-9; router on EPANET's hydraulics 1.4e-12 h | 12 points admitted (10 beyond on Windows, 11 on Linux), below |

The causes admitted, each asserted in its test:

- **Net2 as shipped: a decision at an exact tie.** The pattern steps from 0.09 to 0.08 mg/L at 31 h,
  and pipe 1's last segment then meets the new outflow exactly `Ctol` = 0.01 apart. noodl computes
  `0.09 - 0.08 = 0.009999999999999995`, below `Ctol`, and merges; EPANET, in its internal unit, gets a
  difference one ulp above `Ctol` and starts a new segment. The test checks the decision (the
  segment counts, and that EPANET at 0.01 equals EPANET at 0.00999999) and does not count it as accuracy.
  noodl's own run flips those 667 points; its router fed EPANET's saved hydraulics in mg/L  flips 791. Accuracy is tested with `Tolerance` moved by 1e-8.
  A tie at 49 h does resolve as EPANET's, since the decisions are made in EPANET's unit.
- **Net6 with a source: closed-link leaks.** The step plans part at 18922 s (EPANET's PUMP-3845
  switch; noodl's is at 18921 s, see [closed links](../../applications/water/limitations.md#differences-from-epanet)), after which 252 of
  EPANET's 609 step times are 1 to 55 s from noodl's and the fronts arrive at shifted times (a 1 s
  shift of one late switch moves EPANET's own quality by 2e-2). The test asserts the bounds up to 18922 s and
  the mass balance over the whole run; after it the comparison is the router on EPANET's
  hydraulics, 5.2e-11.
- **ky4: closed-link leaks and EPANET's own convergence.** 30 of the 33 points are leaks of closed links, which EPANET
  conducts at `1/CBIG` and noodl does not: the closed pump's 9e-8 m³/s back to the reservoir through
  P-977 (below `Q_STAGNANT`, so EPANET routes it from its first node and the reservoir's 100 % enters the pipe) and the
  closed tank pipes' 4e-9 to 2e-8 m³/s. With those leaks injected into noodl as sources, 3 points
  remain (J-703 and P-696 at 83882 and 86400 s) against EPANET at `ACCURACY 1e-7`, and none against
  EPANET at 1e-8: EPANET's own convergence moves its flow by 3e-4 relative there. With wntr's
  Linux build two of the three are beyond at 1e-7 and J-703 still at 1e-8 (2.4e-3 %, from
  4.3e-3 %): every one shrinks with EPANET's accuracy, on both builds.
  The mass-balance ratio differs by 9.0e-7: the router on EPANET's saved hydraulics reproduces EPANET's
  ratio to 1e-15, and 89 % of the deficit is the pump's leak routed into dead junctions, whose
  mass counts as neither outflow nor storage.
- **ky10: EPANET's unconverged continuity, and closed-link leaks.** 10 points are P-310, the only link into four
  dead-end junctions: noodl's flow equals their demand to 1e-12, EPANET's is up to 0.5 % off
  (a 1.3e-9 to 5.0e-8 m³/s residual at its accuracy, `DAMPLIMIT 1e-2`), and 1e-8 is unbalanced at the start even
  with 5000 trials; nine of them are beyond the bound with wntr's Windows build, all ten with
  its Linux build. Two points (J-433 at 23050 s; J-145 at 14400 s, Windows only) are closed-link leaks.
  The run stops at 6:54:59 because the hydraulics diverge after it ([Limitations](../../applications/water/limitations.md)).

A TRACE comparison covers one traced source at a time; runs with several traced sources are not compared.
