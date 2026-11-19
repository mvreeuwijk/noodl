# Water distribution: theoretical background

Background for the [water distribution application](index.md), moved from the framework theory page ([theory](../../concepts/theory.md)). The sewer sections are on the [sewer background page](../sewer/background.md).

A pressurised water-distribution network is the opposite case to the kinematic-wave sewer, and
exactly the framework's own: every pipe runs full, and head loss is a monotone function of the
head difference at each pipe, pump and valve, so `noodl.apps.water` is an ordinary
`KirchhoffLayer` (§1, §3 above) with Hazen-Williams or Darcy-Weisbach pipes, `PumpCurve` and
minor-loss elements, solved by the same batched, damped Newton iteration as every other
Kirchhoff layer in this codebase. Section 3's own table says why SWMM's dynamic-wave engine is
the heavier comparison: it solves the sewer's full 1-D Saint-Venant equations with surcharging
and reverse flow, "a genuinely dynamic (not quasi-steady) network". SWMM's KINWAVE routing, which
shares the continuity-first assumption, is the comparison target for the kinematic-wave path and
its live dynamic-wave runs for the dynamic-wave path.
