"""Water-quality demonstration: EPANET's quality routing through `water_run`.

Three runs, each reporting the wall time (hydraulics and quality together), the number of
quality steps and the most segments the pool held at once (`diagnostics["quality"][0]
["segments"]`, `SegmentPool.counts()` after every quality step, summed over the chains):

* Net1, 24 h (chlorine, bulk and wall decay), detached: the router runs without autograd;
* Net3, 168 h (TRACE from the lake; reversing flows, three tanks), detached: the long run;
* Net1, 6 h, attached: `quality.bulk_coefficient` requires grad and the run is
  `differentiable=True`, so the router builds the autograd graph; the time of the backward
  pass to a node's quality is reported too.

Nothing is asserted. Net3 is read from `wntr`'s network library (test-only dependency, as in the
verification suite); without `wntr` that run is skipped.

Run: `.venv/Scripts/python benchmarks/water_quality.py`
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model
from noodl.apps.water.simulate import water_run

DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "water"


def _run(name: str, path: Path, *, hours: float | None = None, attached: bool = False) -> None:
    net = read_epanet_inp(path)
    model, state, drivers = build_model(net)
    hours_run = net.duration / 3600 if hours is None else hours
    kwargs: dict = {}
    if attached:
        n_pipes = len(net.pipes)
        base = net.quality.global_bulk
        drivers = {**drivers, "quality.bulk_coefficient":
                   torch.full((n_pipes,), base, dtype=torch.float64, requires_grad=True)}
        kwargs["differentiable"] = True
    diagnostics: dict = {}
    started = time.perf_counter()
    results = water_run(model, state, drivers, duration=hours_run * 3600, quality=True,
                        diagnostics=diagnostics, **kwargs)
    elapsed = time.perf_counter() - started
    entry = diagnostics["quality"][0]
    steps = len(entry["segments"])
    peak = max(sum(counts) for counts in entry["segments"])
    line = (f"water_quality: {name}, {hours_run:g} h, {'attached' if attached else 'detached'}: "
            f"{elapsed:.2f} s, {steps} quality steps, peak {peak} segments")
    if attached:
        started = time.perf_counter()
        results.node_quality[-1].sum().backward()
        line += f"; backward {time.perf_counter() - started:.2f} s"
    print(line)


def main() -> None:
    _run("Net1", DATA / "Net1.inp")
    try:
        import wntr
    except ImportError:
        print("water_quality: Net3 skipped (wntr is not installed)")
    else:
        _run("Net3", Path(wntr.__file__).parent / "library" / "networks" / "Net3.inp")
    _run("Net1", DATA / "Net1.inp", hours=6, attached=True)


if __name__ == "__main__":
    main()
