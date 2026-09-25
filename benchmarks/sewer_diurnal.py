"""Sewer demonstration: a diurnal day on the committed tree.

Runs 24 h at dt = 60 s with storage on, batched over 8 instances that VARY `f_i` and the
leak area across the batch (`Drag.f_i` and the leak `PowerLaw`'s `C` accept
a tensor carrying a leading batch dimension without any core or element change -- ordinary
broadcasting and `Element._param`'s pass-through rule do the rest -- so `build_model`
is handed `f_i`/`leak_area` shaped `(n_instances, 1)` rather than the plain floats every
other caller uses), and prints the wall time and the peak headspace H2S per manhole in ppm.
Budget: the run should finish in under 60 s on the CPU used; a miss is reported, never
hidden by loosening the budget.

PROFILE (cProfile, 20 warmed-up steps of this benchmark's own 8-instance batch,
varied f_i/leak area): of one step's ~0.097 s, `SewerHydraulics`'s closure (the storage
sweep's own `solve_monotone` Newton root-finds, one per tree LEVEL) is ~56%, the two
`TransportLayer.step` GMRES solves are ~31%, the air layer's Newton solve is ~12%, and the
two-film closure is negligible. The closure does NOT re-solve for the pipe depths with
`geom.normal_depth` (that redundant second Newton solve over every pipe measured ~40% of a
step): the manhole level equals the outgoing pipe's entrance depth, so the storage sweep's
own `levels` are remapped from manhole into pipe order instead.

MEASURED: 1440 steps x 8 varied instances in 235.28 s -> FAIL against the 60 s budget.
Peak headspace H2S is non-zero (the lateral BOD load drives the sulfide reaction and the
two-film transfer every step).

Run: `.venv/Scripts/python benchmarks/sewer_diurnal.py`
"""

from __future__ import annotations

import time

import torch

from noodl.apps.sewer.air import F_I_DEFAULT
from noodl.apps.sewer.network import build_model, tree_steady
from noodl.apps.sewer.report import to_ppm

F64 = torch.float64


def main() -> None:
    net = tree_steady()
    n_instances = 8
    # Vary f_i and the leak area 0.5x-1.5x of their defaults across the 8 instances.
    scale = torch.linspace(0.5, 1.5, n_instances, dtype=F64).unsqueeze(-1)
    model, state, drivers = build_model(
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
        # Amplitude 0.25, not 0.5: J2's outgoing pipe C2 has a
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
    print(f"BUDGET: under 60 s -> {'PASS' if elapsed < 60.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
