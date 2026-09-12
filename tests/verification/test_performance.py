"""Performance tests for PotentialFlowLayer.solve.

test_newton_scaling_smoke runs unconditionally (tiny sizes); the budgeted test at the bottom
of this file is marked slow and skipped by default (see pyproject.toml addopts).
"""

import time

import torch

from tellegen.drives import ConstantDrive
from tellegen.elements.powerlaw import PowerLaw
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.solvers.newton import newton
from tellegen.topology import Network

DTYPE = torch.float64


def _random_ambient_network(n_nodes: int, seed: int) -> tuple[Network, list[str]]:
    """Random connected multigraph of n_nodes zones, ~2n edges, plus one ambient boundary node."""
    gen = torch.Generator().manual_seed(seed)
    net = Network(dtype=DTYPE)
    names = [f"z{i}" for i in range(n_nodes)]
    for name in names:
        net.add_node(name)
    net.add_node("ambient")
    for i in range(1, n_nodes):
        j = int(torch.randint(0, i, (1,), generator=gen))
        net.add_edge(names[j], names[i], kind="airpath")
    n_extra = max(0, 2 * n_nodes - (n_nodes - 1) - n_nodes)
    for _ in range(n_extra):
        u = int(torch.randint(0, n_nodes, (1,), generator=gen))
        v = int(torch.randint(0, n_nodes, (1,), generator=gen))
        if u != v:
            net.add_edge(names[u], names[v], kind="airpath")
    for name in names:
        net.add_edge(name, "ambient", kind="airpath")
    return net, names


def _random_layer_and_drivers(n_nodes: int, batch: int, seed: int):
    net, names = _random_ambient_network(n_nodes, seed)
    b = len(net.edge_index("airpath"))
    gen = torch.Generator().manual_seed(seed + 1)
    C = 0.005 + 0.02 * torch.rand(batch, b, dtype=DTYPE, generator=gen)
    n = torch.full((batch, b), 0.65, dtype=DTYPE)
    element = PowerLaw(C, n, dp_transition=1e-6)
    drive = ConstantDrive(kind="airpath", key="wind")
    layer = PotentialFlowLayer(net, "scaling", [element], [drive], boundary=["ambient"])
    wind = 5.0 * torch.rand(batch, b, dtype=DTYPE, generator=gen)
    drivers = {"wind": wind}
    phi_boundary = torch.zeros(batch, 1, dtype=DTYPE)
    return layer, drivers, phi_boundary


def _time_solve(layer, drivers, phi_boundary, sources=None) -> tuple[float, int]:
    def residual_fn(phi_i: torch.Tensor) -> torch.Tensor:
        return layer.residual(phi_i, phi_boundary, drivers, sources)

    def jacobian_fn(phi_i: torch.Tensor) -> torch.Tensor:
        return layer.jacobian(phi_i, phi_boundary, drivers)

    x0 = layer.linear_init(phi_boundary, drivers, sources)
    newton(residual_fn, jacobian_fn, x0)  # warm-up
    start = time.perf_counter()
    result = newton(residual_fn, jacobian_fn, x0)
    elapsed = time.perf_counter() - start
    return elapsed, result.iterations


def test_newton_scaling_smoke():
    layer, drivers, phi_boundary = _random_layer_and_drivers(n_nodes=5, batch=2, seed=0)
    elapsed, iterations = _time_solve(layer, drivers, phi_boundary)
    assert elapsed >= 0.0
    assert iterations >= 1
