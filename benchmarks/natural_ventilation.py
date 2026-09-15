"""The natural-ventilation demo: a heated room and an unheated room joined by a two-way
doorway, the unheated room open to an ambient whose temperature follows a daily sinusoid.
Runs both coupling modes and writes benchmarks/natural_ventilation.json (and a PNG if
matplotlib is importable). `python -m benchmarks.natural_ventilation`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tellegen.apps.building.elements import add_large_opening, orifice_elements_from_edges
from tellegen.apps.building.thermal import Zone, add_zone, build_model, initial_state
from tellegen.drives import Stack
from tellegen.topology import Network

F64 = torch.float64
T_O = 283.15


def build(coupling: str):
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    add_zone(net, Zone("A", volume=60.0, T0=T_O + 5.0))
    add_zone(net, Zone("B", volume=60.0, T0=T_O + 2.0))
    add_large_opening(net, "A", "B", H=2.0, W=0.9, z_mid=1.0, Cd=0.78)
    net.add_edge("ambient", "B", kind="airpath", z_path=0.2, Cd=0.6, area=0.05)
    net.add_edge("ambient", "B", kind="airpath", z_path=1.8, Cd=0.6, area=0.05)
    el = orifice_elements_from_edges(net, "airpath")
    model = build_model(
        net, air_elements=[el], drives=[Stack.from_network(net, "airpath")], coupling=coupling,
        iterate_tol={"thermal": 0.01}, iterate_max=50,
    )
    sources = torch.zeros(net.n, dtype=F64)
    sources[net.node_index("A")] = 1000.0
    return net, model, initial_state(model), sources


def run(coupling: str, dt: float, hours: float = 24.0) -> dict:
    net, model, state, sources = build(coupling)
    t_list, TA, TB, door = [], [], [], []
    for k in range(int(round(hours * 3600.0 / dt))):
        t = (k + 1) * dt
        T_amb = T_O + 6.0 * math.sin(2.0 * math.pi * (t - 9.0 * 3600.0) / 86400.0)
        drivers = {
            "air.phi_boundary": torch.zeros(1, dtype=F64),
            "thermal.x_boundary": torch.tensor([T_amb], dtype=F64),
            "thermal.sources": sources,
        }
        state = model.step(state, drivers, dt)
        t_list.append(t / 3600.0)
        TA.append(state["thermal.x"][0].item())
        TB.append(state["thermal.x"][1].item())
        door.append(abs(state["air.q"][0].item()))
    return {"coupling": coupling, "dt": dt, "t_h": t_list, "T_A": TA, "T_B": TB, "door_kg_s": door}


def main() -> None:
    results = {
        f"{c}_{int(dt)}": run(c, dt)
        for c in ("pingpong", "iterate")
        for dt in (3600.0, 600.0)
    }
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    for key, r in results.items():
        ax[0].plot(r["t_h"], r["T_A"], label=f"T_A {key}")
        ax[1].plot(r["t_h"], r["door_kg_s"], label=key)
    ax[0].set_ylabel("T_A [K]")
    ax[1].set_ylabel("doorway exchange [kg/s]")
    ax[1].set_xlabel("hour")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.savefig(out.with_suffix(".png"), dpi=120)


if __name__ == "__main__":
    main()
