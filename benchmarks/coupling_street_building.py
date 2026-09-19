"""Milestone 5 demonstration: batched throughput of the coupled street-building model
(design spec section 6). Not a speedup claim against anything -- no existing tool runs this
pair at all -- a sanity number in the spirit of `benchmarks/wsimod_oxford.py`. Batches over
B independent copies of the SAME pair (both apps broadcast a leading batch axis: the street
parity test batches its forcing, `project_to_model`'s drives take `(B,)` winds), with the
wind varied per instance so the batch axis does real work.

Run: `.venv/Scripts/python benchmarks/coupling_street_building.py [batch_size ...]`
(default batch sizes: 1 10 100).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

from tellegen.apps.building.prj import project_to_model, read_prj
from tellegen.apps.street.network import Street, StreetNetwork, build_street_model, street_index
from tellegen.couple import (
    CONCENTRATION_TO_MASS_FRACTION,
    STREET_RAD_TO_CONTAM_DEG,
    DriverAlias,
    ValueLink,
    union,
)

F64 = torch.float64
PRJ = (
    Path(__file__).resolve().parent.parent
    / "tests" / "data" / "contam" / "valThreeZonesWthCtm-UseApi.prj"
)
N_HOURS = 6


def small_street() -> StreetNetwork:
    x = {"n0": 0.0, "n1": 30.0, "n2": 60.0, "n3": 30.0}
    y = {"n0": 40.0, "n1": 30.0, "n2": 40.0, "n3": 0.0}
    spec = [("r1", "n0", "n1"), ("r2", "n1", "n2"), ("r3", "n3", "n1")]
    return StreetNetwork(
        streets=[
            Street(n, a, b, math.hypot(x[b] - x[a], y[b] - y[a]), 2.0, 3.0) for n, a, b in spec
        ],
        x=x, y=y,
    )


def build_city(batch_size: int):
    net = small_street()
    street_model, street_state, street_drivers = build_street_model(
        net, species=("nox",), z_ref=10.0
    )
    graph = street_model.net
    sources = torch.zeros(batch_size, graph.n, dtype=F64)
    for s in net.streets:
        sources[:, graph.node_index(s.name)] = 1.0e-7
    street_drivers = dict(street_drivers)
    street_drivers.update({
        "street.x_boundary": torch.full((batch_size, 1), 2.0e-8, dtype=F64),
        "street.sources": sources,
        "U_ref": torch.linspace(1.0, 4.0, batch_size, dtype=F64),
        "theta_w": torch.zeros(batch_size, dtype=F64),
        "h_abl": torch.full((batch_size,), 1200.0, dtype=F64),
    })
    street_state = {"street.x": torch.zeros(batch_size, len(net.streets), dtype=F64)}
    project = read_prj(PRJ)
    building_model, building_state, building_drivers = project_to_model(project)
    building_state = {k: v.expand(batch_size, *v.shape).clone() for k, v in building_state.items()}
    link = ValueLink(
        from_model="street", from_key="street.x", from_index=street_index(street_model)["r2"],
        to_model="building", to_key="species.x_boundary", to_index=0,
        convert=CONCENTRATION_TO_MASS_FRACTION, two_way=True,
    )
    aliases = [
        DriverAlias(source=("street", "U_ref"), targets=(("building", "V_met", None),)),
        DriverAlias(
            source=("street", "theta_w"),
            targets=(("building", "theta_w", STREET_RAD_TO_CONTAM_DEG),),
        ),
    ]
    return union(
        {"street": (street_model, street_state, street_drivers),
         "building": (building_model, building_state, building_drivers)},
        shared=[link, *aliases], substeps={"building": 60}, iterate_rtol=1e-8, iterate_max=50,
    )


def run(batch_size: int) -> tuple[float, int]:
    city, state, drivers = build_city(batch_size)
    passes = 0
    start = time.perf_counter()
    for _ in range(N_HOURS):
        diag: dict = {}
        state = city.step(state, drivers, dt=3600.0, diagnostics=diag)
        passes += diag["passes"]
    return time.perf_counter() - start, passes


if __name__ == "__main__":
    batch_sizes = [int(a) for a in sys.argv[1:]] or [1, 10, 100]
    for b in batch_sizes:
        print(f"batch_size={b}: running {N_HOURS} coupled hours (60 building sub-steps each) ...",
              flush=True)
        elapsed, passes = run(b)
        print(f"batch_size={b}: {N_HOURS} coupled hours (60 building sub-steps each) in "
              f"{elapsed:.3f} s, {passes} outer passes total", flush=True)
