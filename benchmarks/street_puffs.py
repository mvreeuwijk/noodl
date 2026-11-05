"""Cost of the unsteady above-roof puff street model at city scale (design Sect. 4.10).

One pulse release on a street near the upwind edge of the network, the case's first fully
stable hour's meteorology (hour 0 is partly unstable; the puff tables have no unstable
regime yet) held constant, dt = 5 s, one simulated hour (720 steps), batch 1
and 8 (the batch varies the wind direction by a few degrees). Prints ms/step for the
forward run (under `torch.no_grad()`) and for forward + backward (a loss on the final
street concentrations, checkpointed puff steps), the peak number of live puffs over the
batch and the number of puffs evicted (capacity reached). The puffs are emitted every
`--puff-dt` seconds (default `none`: one puff per street step) and dropped below
`--drop-below` of the instance's peak (default 1e-3), the `PuffOptions` defaults.

The network is the 577-street Le Perreux-sur-Marne MUNICH case when `NOODL_PARIS_CASE`
points to its directory (or `--case`), else a 24 x 24 lattice of 100 m streets.

Run: `.venv/Scripts/python benchmarks/street_puffs.py [--batch 1 8] [--steps 720]
[--capacity 8192] [--case DIR] [--hour K] [--puff-dt none|30] [--drop-below 1e-3]
[--forward-only] [--shear] [--profile]`.
"""

from __future__ import annotations

import argparse
import cProfile
import math
import os
import pstats
import time
from pathlib import Path

import numpy as np
import torch

from noodl.apps.street_aq import StreetCase, read_case
from noodl.apps.street_aq.case import drivers_at
from noodl.apps.street_aq.network import Street, StreetNetwork, build_model
from noodl.apps.street_aq.puff import PuffOptions, PulseSource
from noodl.model import Model

F64 = torch.float64
DT = 5.0
PULSE_MASS_KG = 1.0
PULSE_DURATION_S = 30.0
SIGMA_THETA = math.radians(6.0)   # the case carries no direction spread driver


def lattice_case(n: int = 24, spacing: float = 100.0) -> StreetCase:
    """An `n x n` lattice of junctions joined by `2 n (n - 1)` streets, one-hour synthetic
    meteorology (3 m/s from the west, neutral), the fallback when no case is available."""
    x, y, streets = {}, {}, []
    for i in range(n):
        for j in range(n):
            x[f"n{i}_{j}"], y[f"n{i}_{j}"] = i * spacing, j * spacing
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                if i + di < n and j + dj < n:
                    streets.append(Street(f"s{i}_{j}_{di}{dj}", f"n{i}_{j}",
                                          f"n{i + di}_{j + dj}", spacing, 15.0, 20.0))
    net = StreetNetwork(streets=streets, x=x, y=y)
    meteo = {"wind_dir_from_deg": 270.0, "wind_speed": 3.0, "u_star": 0.4, "h_abl": 800.0,
             "lmo": 1.0e5, "temperature": 288.0}
    return StreetCase.synthetic(net, species=["NO2"], times=[0.0], meteo=meteo,
                                emissions=0.0, background=0.0)


def load_case(path: str | None) -> tuple[StreetCase, str]:
    path = path or os.environ.get("NOODL_PARIS_CASE")
    if path:
        return read_case(Path(path)), f"MUNICH case {path}"
    return lattice_case(), "24 x 24 lattice (NOODL_PARIS_CASE unset)"


def first_stable_hour(case: StreetCase) -> int:
    """The first hour whose Obukhov length is positive at every street: `puff_tables` has no
    unstable regime yet (the Paris case's hours 0 and 1 are partly unstable)."""
    return next(k for k in range(len(case.times)) if bool((case.meteo["lmo"][k] > 0).all()))


def build(case: StreetCase, batch: int, capacity: int, hour: int = 0, *,
          puff_dt: float | None = PuffOptions.puff_dt,
          drop_below: float = PuffOptions.drop_below, shear: bool = False):
    """`(model, state, drivers, source_street)` for `batch` instances of the case's `hour`,
    with a `PulseSource` on the street nearest the upwind edge."""
    species = case.species[0]
    net = case.network
    model, state, _ = build_model(
        net, species=(species,), background="per_street",
        puffs=PuffOptions(capacity=capacity, sigma_theta=SIGMA_THETA, puff_dt=puff_dt,
                          drop_below=drop_below, shear=shear), **case.model_options())
    drivers = drivers_at(case, model, hour, species=[species])
    sources = drivers.pop("street.sources")
    names = [s.name for s in net.streets]
    pulsed = Model(model.net, model.layers,
                   closures=[PulseSource(model.net, names), *model.closures])
    model.closures[-1].bind(pulsed)
    theta = float(drivers["theta_w"])
    toward = np.array([math.cos(theta), math.sin(theta)])
    mid = np.array([[(net.x[s.u] + net.x[s.v]) / 2, (net.y[s.u] + net.y[s.v]) / 2]
                    for s in net.streets])
    along = mid @ toward
    cross = mid @ np.array([-toward[1], toward[0]])
    # near the upwind edge, and near the middle across the wind
    street = int(np.argmin(along - along.min() + 0.5 * np.abs(cross - np.median(cross))))
    for key in ("u_star", "h_abl", "lmo", "U_ref", "sigma_theta", "theta_w"):
        if key in drivers and batch > 1:
            drivers[key] = drivers[key].expand(batch).clone()
    if batch > 1:
        drivers["theta_w"] = drivers["theta_w"] + torch.linspace(-0.05, 0.05, batch, dtype=F64)
    drivers["pulse.weights"] = torch.nn.functional.one_hot(
        torch.full((batch,), street), len(names)).to(F64)
    for key, value in (("rate", PULSE_MASS_KG / PULSE_DURATION_S), ("start", 0.0),
                       ("duration", PULSE_DURATION_S)):
        drivers[f"pulse.{key}"] = torch.full((batch,), value, dtype=F64)
    drivers["street.sources"] = torch.zeros_like(sources)  # the case's own emissions off
    state = {**state, "street.x": drivers["street.x_boundary"].clone()}
    return pulsed, state, drivers, names[street]


def run(model, state, drivers, steps: int, *, grad: bool = False):
    """Step `steps` times (a backward pass at the end when `grad`); `(seconds, peak live
    puffs, puffs evicted)`."""
    peak, evicted = 0, 0
    closure = model.closures[-1]
    t0 = time.perf_counter()
    for k in range(steps):
        state = model.step(state, drivers, DT, t=DT * k)
        peak = max(peak, int(closure.last["live"].max()))
        evicted += int(closure.last["evicted"].sum())
    if grad:
        state["street.x"].sum().backward()
    return time.perf_counter() - t0, peak, evicted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--capacity", type=int, default=8192)
    ap.add_argument("--case")
    default = PuffOptions()
    ap.add_argument("--puff-dt", default=str(default.puff_dt).lower(),
                    help="puff emission interval, s, or 'none' (one puff per step)")
    ap.add_argument("--drop-below", type=float, default=default.drop_below)
    ap.add_argument("--forward-only", action="store_true", help="skip forward + backward")
    ap.add_argument("--shear", action="store_true", help="wind-shear stretching on")
    ap.add_argument("--hour", type=int, help="case hour (default: the first fully stable one)")
    ap.add_argument("--profile", action="store_true",
                    help="cProfile of 20 forward steps at the first batch size "
                         "(top 10 by cumulative time)")
    args = ap.parse_args()
    case, label = load_case(args.case)
    hour = first_stable_hour(case) if args.hour is None else args.hour
    puff_dt = None if args.puff_dt.lower() == "none" else float(args.puff_dt)
    opts = {"puff_dt": puff_dt, "drop_below": args.drop_below, "shear": args.shear}
    print(f"{label}, hour {hour}: {len(case.network.streets)} streets, dt = {DT} s, "
          f"{args.steps} steps, capacity {args.capacity}, puff_dt {puff_dt}, drop_below "
          f"{args.drop_below}, shear {args.shear}", flush=True)
    for i, batch in enumerate(args.batch):
        model, state, drivers, street = build(case, batch, args.capacity, hour, **opts)
        if args.profile and i == 0:
            prof = cProfile.Profile()
            with torch.no_grad():
                prof.enable()
                run(model, state, drivers, 20)
                prof.disable()
            pstats.Stats(prof).sort_stats("cumulative").print_stats(10)
            model, state, drivers, street = build(case, batch, args.capacity, hour, **opts)
        with torch.no_grad():
            sec, peak, ev = run(model, state, drivers, args.steps)
        print(f"B={batch} pulse on street {street}: forward {1e3 * sec / args.steps:.1f} "
              f"ms/step, peak live {peak}, evicted {ev}", flush=True)
        if args.forward_only:
            continue
        model, state, drivers, _ = build(case, batch, args.capacity, hour, **opts)
        drivers["pulse.rate"].requires_grad_(True)
        sec, peak, ev = run(model, state, drivers, args.steps, grad=True)
        print(f"B={batch}: forward+backward {1e3 * sec / args.steps:.1f} ms/step, "
              f"peak live {peak}, evicted {ev}", flush=True)


if __name__ == "__main__":
    main()
