"""Regenerate tests/golden/contam_airflow.json from the current solver implementation.

This is an explicit action, not run automatically: run it only after an intentional change
to element laws or solver defaults that is expected to change the reference numbers.

    .venv/Scripts/python benchmarks/regenerate_golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from golden import save_golden  # noqa: E402

from tellegen.drives import ConstantDrive  # noqa: E402
from tellegen.elements.fan import FanCurve  # noqa: E402
from tellegen.elements.fixed import FixedFlow  # noqa: E402
from tellegen.elements.powerlaw import PowerLaw  # noqa: E402
from tellegen.layers.potential import PotentialFlowLayer  # noqa: E402
from tellegen.topology import Network  # noqa: E402

DTYPE = torch.float64


def _series_case() -> dict:
    net = Network(dtype=DTYPE)
    for name in ("ambient_w", "z1", "z2", "ambient_l"):
        net.add_node(name)
    net.add_edge("ambient_w", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient_l", kind="airpath")
    element = PowerLaw(
        torch.tensor([0.010, 0.008, 0.012], dtype=DTYPE),
        torch.tensor(0.65, dtype=DTYPE),
        dp_transition=1e-6,
    )
    drive = ConstantDrive(kind="airpath", key="wind")
    layer = PotentialFlowLayer(
        net, "series", [element], [drive], boundary=["ambient_w", "ambient_l"]
    )
    drivers = {"wind": torch.tensor([12.0, 0.0, 0.0], dtype=DTYPE)}
    phi, q = layer.solve(torch.zeros(2, dtype=DTYPE), drivers, differentiable=False)
    return {"phi": phi.tolist(), "q": q.tolist()}


def _parallel_case() -> dict:
    net = Network(dtype=DTYPE)
    net.add_node("a")
    net.add_node("c")
    net.add_edge("a", "c", kind="airpath")
    net.add_edge("a", "c", kind="airpath")
    element = PowerLaw(
        torch.tensor([0.020, 0.015], dtype=DTYPE),
        torch.tensor(0.6, dtype=DTYPE),
        dp_transition=1e-6,
    )
    layer = PotentialFlowLayer(net, "parallel", [element], boundary=["a", "c"])
    phi, q = layer.solve(torch.tensor([8.0, 0.0], dtype=DTYPE), differentiable=False)
    return {"phi": phi.tolist(), "q": q.tolist()}


def _fan_driven_case() -> dict:
    net = Network(dtype=DTYPE)
    net.add_node("zone")
    net.add_node("ambient")
    net.add_edge("zone", "ambient", kind="airpath")
    net.add_edge("zone", "ambient", kind="airpath")
    net.add_edge("zone", "ambient", kind="fan")
    leak = PowerLaw(
        torch.tensor([0.020, 0.010], dtype=DTYPE),
        torch.tensor(0.65, dtype=DTYPE),
        dp_transition=1e-6,
    )
    fan = FixedFlow(torch.tensor(0.05, dtype=DTYPE), kind="fan")
    layer = PotentialFlowLayer(net, "fan_driven", [leak, fan], boundary=["ambient"])
    phi, q = layer.solve(torch.zeros(1, dtype=DTYPE), differentiable=False)
    return {"phi": phi.tolist(), "q": q.tolist()}


def _fan_curve_case() -> dict:
    net = Network(dtype=DTYPE)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="fan")
    net.add_edge("z", "ambient", kind="airpath")
    coeffs = torch.tensor([150.0, -100.0, -80.0, 40.0], dtype=DTYPE)
    fan = FanCurve(coeffs, torch.tensor(1.0, dtype=DTYPE), kind="fan")
    leak = PowerLaw(
        torch.tensor(0.05, dtype=DTYPE), torch.tensor(0.5, dtype=DTYPE), dp_transition=1e-6
    )
    layer = PotentialFlowLayer(net, "fan_curve", [fan, leak], boundary=["ambient"])
    phi, q = layer.solve(torch.zeros(1, dtype=DTYPE), differentiable=False)
    return {"phi": phi.tolist(), "q": q.tolist()}


def main() -> None:
    data = {
        "series": _series_case(),
        "parallel": _parallel_case(),
        "fan_driven": _fan_driven_case(),
        "fan_curve": _fan_curve_case(),
    }
    save_golden("contam_airflow", data)
    print("wrote tests/golden/contam_airflow.json")


if __name__ == "__main__":
    main()
