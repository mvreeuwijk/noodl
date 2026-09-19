"""Benchmark: batched Newton solve time and iteration count vs network size and batch size.

Usage:
    .venv/Scripts/python benchmarks/newton_scaling.py
    .venv/Scripts/python benchmarks/newton_scaling.py --device cuda
"""

from __future__ import annotations

import argparse
import time

import torch

from noodl.drives import ConstantDrive
from noodl.elements.powerlaw import PowerLaw
from noodl.layers.potential import PotentialFlowLayer
from noodl.solvers.newton import newton
from noodl.topology import Network

DTYPE = torch.float64


def random_ambient_network(n_nodes: int, seed: int) -> Network:
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
    return net


def build_layer(n_nodes: int, batch: int, seed: int, device: torch.device):
    net = random_ambient_network(n_nodes, seed)
    net.to(device=device, dtype=DTYPE)
    b = len(net.edge_index("airpath"))
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    C = (0.005 + 0.02 * torch.rand(batch, b, generator=gen)).to(device=device, dtype=DTYPE)
    n = torch.full((batch, b), 0.65, dtype=DTYPE, device=device)
    element = PowerLaw(C, n, dp_transition=1e-6)
    drive = ConstantDrive(kind="airpath", key="wind")
    layer = PotentialFlowLayer(net, "scaling", [element], [drive], boundary=["ambient"])
    wind = (5.0 * torch.rand(batch, b, generator=gen)).to(device=device, dtype=DTYPE)
    drivers = {"wind": wind}
    phi_boundary = torch.zeros(batch, 1, dtype=DTYPE, device=device)
    return layer, drivers, phi_boundary


def time_solve(layer, drivers, phi_boundary, sources=None) -> tuple[float, int]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    sizes = (10, 30, 100, 300)
    batches = (1, 10, 100, 1000)
    print(f"{'n_nodes':>8} {'batch':>8} {'seconds':>10} {'iterations':>10}")
    for n_nodes in sizes:
        for batch in batches:
            layer, drivers, phi_boundary = build_layer(
                n_nodes, batch, seed=n_nodes * 1000 + batch, device=device
            )
            elapsed, iterations = time_solve(layer, drivers, phi_boundary)
            print(f"{n_nodes:>8} {batch:>8} {elapsed:>10.4f} {iterations:>10}")


if __name__ == "__main__":
    main()
