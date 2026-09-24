"""Run the street model on the real AQ_DT Leiden data and record what it costs.

Spec section 11 budgets nothing for this milestone: the run is RECORDED, and the recorded
number is what decides whether the conditional follow-up (a street row in the composed
scaling gate) is triggered.

    .venv/Scripts/python -m benchmarks.street_leiden --domain leiden_small
    .venv/Scripts/python -m benchmarks.street_leiden --domain leiden --steps 1

The data directory comes from `NOODL_AQDT_DATA`, defaulting to the location on the
machine this milestone was written on. Nothing is written into it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from noodl.apps.street_aq.loader import read_aqdt
from noodl.apps.street_aq.network import build_street_model
from noodl.apps.street_aq.report import to_ug_m3, write_network_concentration

DEFAULT_DATA = Path(os.environ.get(
    "NOODL_AQDT_DATA", r"<workspace>\tmp\2026_AQ_DT\data"
))
DTYPE = torch.float64


def run(domain: str = "leiden_small", *, year: int = 2024, steps: int | None = None,
        chunk: int = 96, data: Path = DEFAULT_DATA, out: Path | None = None) -> dict:
    """Load `domain`, solve every forcing step quasi-steadily in chunks, and report.

    `chunk` is the number of forcing steps solved in one batched call. Every chunk is one
    sparse solve per instance, so the memory cost is linear in it and the Python overhead
    is inversely so; 96 (twelve days at the 3-hourly step) is a compromise measured to fit
    comfortably at the `leiden_small` size.
    """
    stage1 = Path(data) / "stage1_geometry" / domain
    stage2 = Path(data) / "stage2_inputs" / domain
    started = time.perf_counter()
    aqdt = read_aqdt(stage1, stage2, year=year, wind_height_m=10.0,
                     trust_file_height=True)
    load_seconds = time.perf_counter() - started
    n_time = int(aqdt.emission.shape[0]) if steps is None else int(steps)
    started = time.perf_counter()
    model, _state, _drivers = build_street_model(
        aqdt.net, canyon_wind="soulhac", exchange="sirane", routing="sirane",
        direction_averaging="none", kappa=0.4, pblh_floor=True, z_ref=10.0,
    )
    build_seconds = time.perf_counter() - started
    net = model.net
    columns = torch.tensor([net.node_index(s.name) for s in aqdt.net.streets],
                           dtype=torch.long)
    pieces, velocities = [], []
    started = time.perf_counter()
    for lo in range(0, n_time, chunk):
        hi = min(lo + chunk, n_time)
        sources = torch.zeros(hi - lo, net.n, dtype=DTYPE)
        sources[:, columns] = aqdt.emission[lo:hi]
        drivers = {
            "street.x_boundary": aqdt.forcing.background[lo:hi].reshape(-1, 1),
            "street.sources": sources,
            "U_ref": aqdt.forcing.u_ref[lo:hi],
            "theta_w": aqdt.forcing.theta_w[lo:hi],
            "h_abl": aqdt.forcing.h_abl[lo:hi],
        }
        solved = model.steady({}, drivers)
        pieces.append(solved["street.x"].detach())
        velocities.append(
            model._apply_closures(solved, drivers)["street.u_canyon"].detach()
        )
    solve_seconds = time.perf_counter() - started
    concentration = torch.cat(pieces)
    canyon = torch.cat(velocities)
    report = {
        "domain": domain,
        "year": year,
        "streets": len(aqdt.net.streets),
        "junctions": len(aqdt.net.junctions),
        "forcing_steps": n_time,
        "chunk": chunk,
        "load_seconds": round(load_seconds, 3),
        "build_seconds": round(build_seconds, 3),
        "solve_seconds": round(solve_seconds, 3),
        "seconds_per_step": round(solve_seconds / max(n_time, 1), 5),
        "max_ug_m3": float(to_ug_m3(concentration).max()),
        "mean_ug_m3": float(to_ug_m3(concentration).mean()),
        "notes": aqdt.notes,
    }
    if out is not None:
        write_network_concentration(
            out, time_hours=aqdt.forcing.time_hours[:n_time],
            feature_index=aqdt.feature_index, osmid=aqdt.osmid,
            background=aqdt.forcing.background[:n_time], canyon_velocity=canyon,
            concentration=concentration, year=year,
        )
        report["written"] = str(out)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="leiden_small")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=96)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.domain, year=arguments.year, steps=arguments.steps,
                         chunk=arguments.chunk, data=arguments.data, out=arguments.out),
                     indent=2))


if __name__ == "__main__":
    main()
