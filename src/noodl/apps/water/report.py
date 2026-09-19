"""Reporting helpers for the water application: pressure head, kPa and a link table."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

Tensor = torch.Tensor
F64 = torch.float64

#: Water at 20 C and standard gravity, the pair `to_kilopascal` converts with.
RHO_W = 998.2
G = 9.80665

_COLUMNS = ("link", "flow", "headloss", "velocity")


def pressure_head(model, state, elevations=None) -> Tensor:
    """Pressure head (m) at every node: the solved head minus the node's elevation.

    `elevations` defaults to the elevations the builder stored on the graph (a reservoir's
    is its own fixed head and a tank's is its bottom, so both report zero pressure head
    unless a caller overrides them). `model.head_scale` divides out the `rho g` the
    Darcy-Weisbach path carries, so the result is metres of head either way.
    """
    if elevations is None:
        elevations = torch.tensor(
            [model.net.graph.nodes[n].get("elevation", 0.0) for n in model.net.nodes],
            dtype=F64,
        )
    else:
        elevations = torch.as_tensor(elevations, dtype=F64)
    return state["water.phi"] / model.head_scale - elevations


def to_kilopascal(head: Tensor) -> Tensor:
    """Metres of head -> kPa, at `rho = 998.2 kg/m3` and `g = 9.80665 m/s2`."""
    return head * RHO_W * G / 1000.0


def link_table(model, state, names=None, *, path=None) -> list[dict]:
    """One row per link: flow (m3/s), head loss (m) and velocity (m/s); optionally to CSV.

    Head loss is the layer's own `dp` divided by `model.head_scale`, so it is metres on
    both the Hazen-Williams and the Darcy-Weisbach path, and it is NEGATIVE across a pump
    (a head gain), exactly as EPANET reports it. Velocity is `q / A` on the pipe block and
    is reported as `0.0` on a pump or a valve, which have no length to average over.
    """
    layer = model.potential["water"]
    names = list(model.link_names if names is None else names)
    q = state["water.q"]
    dp = layer.dp(state["water.phi"], {}) / model.head_scale
    net = model.water_network
    area = {
        pipe.name: torch.pi * pipe.diameter**2 / 4.0 for pipe in net.pipes
    }
    rows: list[dict] = []
    for i, name in enumerate(names):
        flow = float(q[..., i])
        rows.append(
            {
                "link": name,
                "flow": flow,
                "headloss": float(dp[..., i]),
                "velocity": flow / area[name] if name in area else 0.0,
            }
        )
    if path is not None:
        with Path(path).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
    return rows
