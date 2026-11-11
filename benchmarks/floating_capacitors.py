"""What eliminating floating capacitors would save in transient stepping (decision D6).

A capacitor between two unknown nodes (a floating, or series, capacitor) is today's
`Capacitance`, a relation with a flow unknown, so its layer solves on the sparse operator.
Written in flow form, `q = C (dp - dp_prev) / dt` on a backward-Euler step, it adds no flow
unknown and the layer stays on the symmetric fast path (`_FlowFormCap`, the test-only law of
`tests/layers/test_context_laws_fast_path.py`, copied here).

The circuit is two coupled RC interconnect lines (a crosstalk model): line `a` is driven by a
1 V source through a source resistor, line `b` is quiet and terminated to ground at both
ends. Each line has `n` series resistors and a grounded capacitor at every node
(`LinearStorage`, the same in both forms), and a coupling capacitor joins `a_i` to `b_i`:
`n` floating capacitors. The `diode` variant ends line `a` in a `JunctionDiode` to ground,
in its exponential range (`Maxexp=40`), which makes the layer nonlinear.

Each case steps `--steps` backward-Euler steps of the layer (`KirchhoffLayer.solve` with
`history`), warmed once. With `--start warm` (default) each step's Newton starts from the
previous step (`phi0`, and `q0` for the flow unknowns), as `Model.step` does; with
`--start cold` from the layer's own start (the fast path's linear guess, the general path's
zeros). `--omega 1` gives both forms an undamped first Newton step (by default only a linear
layer with flow unknowns takes one). The two forms run interleaved, in alternating order,
over `--rounds` rounds; the median wall time per step is reported, with the Newton
iterations per step and the largest difference between the two forms' last potentials.
`grad` steps differentiably with a learnable coupling capacitance and back-propagates the
sum of the last potentials; otherwise the steps run with `differentiable=False`.

Timings depend on the machine and its load, so they are recorded in
`floating_capacitors_results.txt`, never asserted in a test.

    PYTHONPATH=src .venv/Scripts/python -m benchmarks.floating_capacitors
    PYTHONPATH=src .venv/Scripts/python -m benchmarks.floating_capacitors --omega 1
    PYTHONPATH=src .venv/Scripts/python -m benchmarks.floating_capacitors --start cold
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time

import torch

from noodl.apps.electrical import Capacitance, JunctionDiode, Resistance
from noodl.elements import ConstitutiveLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.storage import LinearStorage
from noodl.topology import Network

F64 = torch.float64
DT = 1e-3
FORMS = ("relation", "flow")


class _FlowFormCap(ConstitutiveLaw):
    """A backward-Euler capacitor in flow form, `q = C (dp - dp_prev) / dt` on a step (open
    otherwise): it gives its flow, so it adds no flow unknown."""

    dynamic = True
    linear = True

    def __init__(self, C, kind="cap", learnable=False):
        super().__init__(kind)
        self.C = self._param(C, learnable)

    def flow(self, dp, drivers=None, ctx=None):
        if ctx is None or ctx.mode != "transient":
            return torch.zeros_like(dp) * self.C
        return self.C * (dp - ctx.dp_prev) / ctx.dt


def coupled_lines(n: int, form: str, *, diode: bool = False, learnable: bool = False):
    """The layer, its boundary potentials, its start `(phi, q)` and the coupling law."""
    net = Network(dtype=F64)
    for name in ("gnd", "s"):
        net.add_node(name)
    for line in "ab":
        for i in range(n):
            net.add_node(f"{line}{i}")
    net.add_edge("s", "a0", kind="r")
    net.add_edge("gnd", "b0", kind="r")
    for line in "ab":
        for i in range(n - 1):
            net.add_edge(f"{line}{i}", f"{line}{i + 1}", kind="r")
        net.add_edge(f"{line}{n - 1}", "gnd", kind="r")
    for i in range(n):
        net.add_edge(f"a{i}", f"b{i}", kind="cap")
    if diode:
        net.add_edge(f"a{n - 1}", "gnd", kind="diode")
    n_r = 2 * (n + 1)
    r = torch.full((n_r,), 10.0, dtype=F64)
    r[0] = r[n + 1] = 50.0                     # source and near-end terminations
    r[n] = r[2 * n + 1] = 1e3                  # far-end loads
    C = torch.full((n,), 2e-5, dtype=F64)
    cap = (Capacitance(C, kind="cap", learnable=learnable) if form == "relation"
           else _FlowFormCap(C, kind="cap", learnable=learnable))
    laws = [Resistance(r, kind="r"), cap]
    if diode:
        laws.append(JunctionDiode(Ids=1e-9, Vt=0.025, Maxexp=40.0, kind="diode"))
    storage = [LinearStorage(list(range(2, 2 + 2 * n)), torch.full((2 * n,), 1e-5, dtype=F64))]
    layer = KirchhoffLayer(net, "e", laws, boundary=["gnd", "s"], storage=storage,
                           scheme="backward_euler")
    pb = torch.tensor([0.0, 1.0], dtype=F64)
    start = (torch.zeros(2 + 2 * n, dtype=F64), torch.zeros(n_r + n + int(diode), dtype=F64))
    return layer, pb, start, cap


def _stepper(n, form, steps, *, diode, grad, warm, newton):
    layer, pb, start, cap = coupled_lines(n, form, diode=diode, learnable=grad)
    implicit = layer.implicit_edges
    assert implicit.numel() == (n if form == "relation" else 0)

    def run():
        phi, q = start
        iterations = 0
        for i in range(steps):
            kw = dict(newton)
            if warm:
                kw["phi0"] = phi[..., layer.interior]
                if implicit.numel():
                    kw["q0"] = q[..., implicit]
            diag: dict = {}
            phi, q = layer.solve(pb, dt=DT, t=DT * (i + 1), history=(phi, q),
                                 differentiable=grad, diagnostics=diag, **kw)
            iterations += diag["newton_iterations"]
        if grad:
            phi.sum().backward()
            cap.C.grad = None
        run.iterations = iterations / steps
        return phi.detach()

    return run


def measure(n: int, steps: int, rounds: int, *, diode: bool, grad: bool, warm: bool,
            newton: dict) -> dict:
    runs = {form: _stepper(n, form, steps, diode=diode, grad=grad, warm=warm, newton=newton)
            for form in FORMS}
    phis = {form: run() for form, run in runs.items()}  # warm the caches
    walls = {form: [] for form in FORMS}
    for k in range(rounds):
        for form in (FORMS if k % 2 == 0 else FORMS[::-1]):
            t0 = time.perf_counter()
            runs[form]()
            walls[form].append(time.perf_counter() - t0)
    ms = {form: 1e3 * statistics.median(w) / steps for form, w in walls.items()}
    return {
        "n": n, "diode": diode, "grad": grad, "steps": steps,
        "ms_per_step": ms,
        "iterations_per_step": {form: runs[form].iterations for form in FORMS},
        "speedup": 1.0 - ms["flow"] / ms["relation"],
        "max_dphi": float((phis["relation"] - phis["flow"]).abs().max()),
    }


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true",
                        help="n = 4, 2 steps, 1 round (a smoke run)")
    parser.add_argument("--sizes", type=int, nargs="*", default=[10, 100, 500])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--start", choices=["warm", "cold"], default="warm")
    parser.add_argument("--omega", type=float, default=None,
                        help="Newton's first-step factor for both forms (default: the layer's)")
    args = parser.parse_args(argv)
    sizes, steps, rounds = ([4], 2, 1) if args.quick else (args.sizes, args.steps, args.rounds)
    newton = {} if args.omega is None else {"omega": args.omega}
    print(f"- torch {torch.__version__}, threads {torch.get_num_threads()}, "
          f"{platform.platform()}, {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"- {args.start} start, omega {args.omega or 'default'}, {steps} steps, "
          f"{rounds} rounds")
    print("\n| n | diode | grad | relation ms/step | flow form ms/step | speed-up | "
          "Newton its/step (relation, flow) | max abs dphi |")
    print("|---|---|---|---|---|---|---|---|")
    results = []
    for diode in (False, True):
        for grad in (False, True):
            for n in sizes:
                r = measure(n, steps, rounds, diode=diode, grad=grad,
                            warm=args.start == "warm", newton=newton)
                results.append(r)
                ms, its = r["ms_per_step"], r["iterations_per_step"]
                print(f"| {n} | {diode} | {grad} | {ms['relation']:.2f} | {ms['flow']:.2f} | "
                      f"{100 * r['speedup']:+.0f} % | {its['relation']:.1f}, "
                      f"{its['flow']:.1f} | {r['max_dphi']:.1e} |", flush=True)
    return results


if __name__ == "__main__":
    main()
