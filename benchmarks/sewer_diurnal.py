"""Milestone 4 demonstration: a diurnal day on the committed tree.

Runs 24 h at dt = 60 s with storage on, batched over 8 instances that VARY `f_i` and the
leak area across the batch (FR-18: `Drag.f_i` and the leak `PowerLaw`'s `C` already accept
a tensor carrying a leading batch dimension without any core or element change -- ordinary
broadcasting and `Element._param`'s pass-through rule do the rest -- so `build_sewer_model`
is handed `f_i`/`leak_area` shaped `(n_instances, 1)` rather than the plain floats every
other caller uses), and prints the wall time and the peak headspace H2S per manhole in ppm.
Budget (spec section 7): the run must finish in under 60 s on the CPU used; if it does not,
a FOLLOW-UP is recorded rather than the budget loosened.

MEASURED 18 Sep 2026 (before FR-18/FR-21/FR-19): 1440 steps x 8 identical instances in
248.04 s -> FAIL (about 0.17 s per step); peak H2S printed 0.000 ppm because `bod_in`/
`sulfide_in` were read by no code (FR-21).

FR-19 PROFILE (cProfile, 20 warmed-up steps of this benchmark's own 8-instance batch,
varied f_i/leak area): of one step's ~0.097 s, `SewerHydraulics`'s closure (the storage
sweep's own `solve_monotone` Newton root-finds, one per tree LEVEL) is ~56%, the two
`TransportLayer.step` GMRES solves are ~31%, the air layer's Newton solve is ~12%, and the
two-film closure is negligible. The single biggest hot spot (measured ~40% of one step
BEFORE this fix) was `SewerHydraulics.__call__` calling `geom.normal_depth(q, ...)` -- a
SECOND, redundant `solve_monotone` Newton solve over every pipe -- to recover a depth the
storage sweep had ALREADY solved for directly (`state["sewer.H"]`, spec 3.2's own "manhole
level equals the outgoing pipe's entrance depth"): fixed by remapping the storage sweep's
own `levels` from manhole into pipe order instead of re-solving for them.

MEASURED after FR-18 (varied instances)/FR-21 (lateral loads)/FR-19 (the fix above): 1440
steps x 8 instances in 235.28 s -> still FAIL, RECORDED, not loosened (down from 282.21 s
with FR-18/FR-21 alone, itself slower than the original 248.04 s because the batch now
varies element parameters and the lateral BOD load now actually drives the sulfide
reaction and the two-film transfer every step). Peak headspace H2S is now non-zero (row
below).

Run: `.venv/Scripts/python benchmarks/sewer_diurnal.py`
"""

from __future__ import annotations

import time

import torch

from noodl.apps.sewer.air import F_I_DEFAULT
from noodl.apps.sewer.network import build_sewer_model, tree_steady
from noodl.apps.sewer.report import to_ppm

F64 = torch.float64


def main() -> None:
    net = tree_steady()
    n_instances = 8
    # FR-18: vary f_i and the leak area 0.5x-1.5x of their defaults across the 8 instances.
    scale = torch.linspace(0.5, 1.5, n_instances, dtype=F64).unsqueeze(-1)
    model, state, drivers = build_sewer_model(
        net, storage=True, dt_storage=60.0,
        f_i=F_I_DEFAULT * scale, leak_area=8e-4 * scale,
    )
    # `SewerHydraulics._storage_sweep` derives `lateral` from `drivers["inflow"]` alone
    # (`lateral = inflow.index_select(-1, self.manhole_idx)`, hydraulics.py:185), so once
    # `state["sewer.H"]` below carries a leading batch dimension of `n_instances`, every
    # per-node driver the closures read must carry the SAME batch dimension too, or the
    # storage sweep's `index_add` sees a rank-1 `lateral` against a rank-2 `q_out` and
    # raises (`RuntimeError: index_add_(): Number of indices ... for dim: 0`, reproduced
    # before this fix) -- `inflow` and `bod_in` are batched here for that reason.
    base_inflow = drivers["inflow"].unsqueeze(0).expand(n_instances, -1).clone()
    for key in ("air.phi", "air.q", "water_quality.x", "air_quality.x", "sewer.H"):
        state[key] = state[key].unsqueeze(0).expand((n_instances,) + state[key].shape).clone()
    drivers = dict(drivers)
    drivers["air.phi_boundary"] = drivers["air.phi_boundary"].expand(n_instances, 1)
    drivers["bod_in"] = base_inflow * 0.0
    peak = torch.zeros(n_instances, model.transport["air_quality"].n_i, dtype=F64)
    started = time.perf_counter()
    steps = 24 * 60
    for step in range(steps):
        hour = step / 60.0
        # Amplitude 0.25, not the plan's 0.5 (ruling M4-R21): J2's outgoing pipe C2 has a
        # Manning capacity of 0.10402 m3/s against a base inflow of 0.08 m3/s, so any
        # factor above 1.3003 surcharges it and the storage sweep refuses the step by name
        # (measured: the first refusal came at factor 1.3009). The peak is now 0.100 m3/s.
        factor = 1.0 + 0.25 * torch.sin(torch.tensor(2.0 * torch.pi * hour / 24.0))
        drivers["inflow"] = base_inflow * factor
        drivers["bod_in"] = torch.where(
            base_inflow > 0, torch.full_like(base_inflow, 0.30 * float(factor)),
            torch.zeros_like(base_inflow),
        )
        state = model.step(state, drivers, 60.0)
        peak = torch.maximum(peak, state["air_quality.x"])
    elapsed = time.perf_counter() - started
    print(f"sewer_diurnal: {steps} steps x {n_instances} instances in {elapsed:.2f} s")
    ppm = to_ppm(peak, drivers["T_head"])
    print("peak headspace H2S (ppm) per manhole, instance 0:")
    print("  " + "  ".join(f"{v:.3f}" for v in ppm[0].tolist()))
    print(f"BUDGET: under 60 s -> {'PASS' if elapsed < 60.0 else 'FAIL (record a follow-up)'}")


if __name__ == "__main__":
    main()
