"""Milestone 4b demonstration: batched `CapacitatedTransferLayer` throughput on the
`oxford_demo` topology (design spec section 7).

NOT a fair batched-vs-unbatched speedup claim: WSIMOD itself runs exactly one instance of
this topology in ~4 s (its own recursive push/pull message-chaining, single-threaded). This
script instead reports tellegen's own wall-clock for `B` BATCHED instances of the SAME
21-arc, 18-node topology and the SAME 1,456-timestep length, run through
`CapacitatedTransferLayer` in hard-clip mode -- a sanity number in the same spirit as
milestone 4's `benchmarks/sewer_diurnal.py` and `benchmarks/water_eps.py` rows, stated as
such rather than framed as a speedup over WSIMOD (no budget is set for this row, per spec
section 7 and section 8, which reserves a hard budget for the sewer benchmark only).

Uses random per-step requests scaled to each arc's own capacity (`torch.rand(...) * c_arc`),
not the committed WSIMOD fixture data: the parity claim itself (W1/W2) is already made by
`tests/verification/test_wsimod_parity.py` against the real captured requests; this script
only exercises the SAME topology's shape and step count for a throughput number, so random
per-instance requests are enough and let every batch instance differ (unlike replaying one
fixture's requests identically B times, which would not exercise the batch dimension's own
broadcasting any differently to B=1).

Run: `.venv/Scripts/python benchmarks/wsimod_oxford.py`
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from tellegen.layers.capacitated import CapacitatedTransferLayer
from tellegen.topology import Network

F64 = torch.float64
N_STEPS = 1456
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "wsimod"


def build_network() -> tuple[Network, torch.Tensor]:
    topology = json.loads((FIXTURE_DIR / "oxford_topology.json").read_text())
    net = Network()
    for node in topology["nodes"]:
        net.add_node(node["name"])
    for arc in topology["arcs"]:
        net.add_edge(arc["source"], arc["target"], kind="link", name=arc["name"])
    c_arc = torch.tensor([a["capacity"] for a in topology["arcs"]], dtype=F64)
    return net, c_arc


def run(batch_size: int) -> float:
    net, c_arc = build_network()
    s_max = torch.full((net.n,), float("inf"), dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    s = torch.zeros(batch_size, net.n, dtype=F64)
    torch.manual_seed(0)
    r = torch.rand(batch_size, c_arc.numel(), dtype=F64) * c_arc
    started = time.perf_counter()
    for _ in range(N_STEPS):
        s, _f = layer.step(s, {"cap.requests": r}, dt=1.0)
    return time.perf_counter() - started


def main() -> None:
    for batch_size in (1, 10, 100):
        elapsed = run(batch_size)
        print(
            f"wsimod_oxford: batch_size={batch_size}: {elapsed:.3f} s for {N_STEPS} steps "
            "(WSIMOD's own single-instance run: ~4 s, not a fair comparison, see module "
            "docstring)"
        )


if __name__ == "__main__":
    main()
