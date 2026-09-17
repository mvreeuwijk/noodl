"""Regenerate the stored golden results from the current implementation.

Writes `tests/golden/contam_airflow.json` (the closed-form airflow cases) and
`tests/golden/natural_ventilation.json` (the coupled airflow-heat demo trajectory).

This is an explicit action, not run automatically: run it only after an intentional change
to element laws, coupling or solver defaults that is expected to change the reference
numbers.

    .venv/Scripts/python benchmarks/regenerate_golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
# `tests` for `golden`, and the repository root so `benchmarks.natural_ventilation` imports
# whether this file is run as a path (sys.path[0] is `benchmarks/`) or as `-m`.
sys.path.insert(0, str(_REPO_ROOT / "tests"))
sys.path.insert(0, str(_REPO_ROOT))

from golden import save_golden  # noqa: E402

from benchmarks.natural_ventilation import run as run_natural_ventilation  # noqa: E402
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


def _natural_ventilation_case() -> dict:
    """Two simulated hours (12 steps of 600 s) of the coupled airflow-heat demo, `iterate`.

    `benchmarks.natural_ventilation.run` is deterministic -- no RNG anywhere in it -- so this
    is a genuine bit-for-bit regression reference for the whole milestone-2 stack at once:
    `build_model`, the stack drive, the density closure, the large opening, and the iterated
    coupling of the air and thermal layers, step by step.
    """
    result = run_natural_ventilation("iterate", 600.0, hours=2.0)
    return {key: result[key] for key in ("T_A", "T_B", "door_kg_s")}


def _impaq_test_network_case() -> dict:
    """The IMPAQ prototype's own answer on `build_test_network`, so the port's fidelity
    keeps being checked where the AQ_DT repository is not installed."""
    import math

    from tellegen.apps.street.impaq import (
        build_test_network,
        canyon_velocity,
        compute_boundary_layer,
        solve_steady_state,
    )

    network = build_test_network()
    layer = compute_boundary_layer(network, 1e-4, 2.0, 0.25 * math.pi)
    network.roads.canyon_velocity_mps = canyon_velocity(
        network.roads, layer.friction_velocity_mps, layer.wind_angle_rad
    )
    return {
        "friction_velocity_mps": float(layer.friction_velocity_mps),
        "canyon_velocity_mps": [float(v) for v in network.roads.canyon_velocity_mps],
        "prototype": [float(v) for v in solve_steady_state(network, layer)],
        "fix_ab": [
            float(v) for v in solve_steady_state(network, layer, fix_a=True, fix_b=True)
        ],
    }


def _munich_idealised_case() -> dict:
    """The model's own 12x6 answer on the published idealised case, so that a change in
    the routing, the direction averaging or the canyon wind shows up as a diff."""
    from tests.verification.test_munich import _fixture, _run

    out, _names = _run(_fixture())
    return {panel: {street: float(value) for street, value in row.items()}
            for panel, row in out.items()}


def main() -> None:
    data = {
        "series": _series_case(),
        "parallel": _parallel_case(),
        "fan_driven": _fan_driven_case(),
        "fan_curve": _fan_curve_case(),
    }
    save_golden("contam_airflow", data)
    print("wrote tests/golden/contam_airflow.json")
    save_golden("natural_ventilation", _natural_ventilation_case())
    print("wrote tests/golden/natural_ventilation.json")
    save_golden("impaq_test_network", _impaq_test_network_case())
    print("wrote tests/golden/impaq_test_network.json")
    save_golden("munich_idealised", _munich_idealised_case())
    print("wrote tests/golden/munich_idealised.json")


if __name__ == "__main__":
    main()
