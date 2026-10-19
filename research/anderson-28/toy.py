import sys

sys.path.insert(0, "C:/projects/noodl-ci3")
import torch

import noodl.solvers.anderson as A
from tests.test_model import _toy

F64 = torch.float64
eps = torch.finfo(F64).eps
for nmin in [float(a) for a in sys.argv[1:]]:
    A._ANDERSON_NOISE_MIN = nmin
    passes, fails = [], []
    for k in range(-50, 51):
        model, state, drivers, _ = _toy(
            coupling="iterate", iterate_tol={"toy": 1e-10}, iterate_max=400,
            iterate_acceleration="anderson", iterate_flow_tol={"toy": 1e-9})
        drivers["toy.sources"] = torch.tensor([20.0 * (1 + k * eps), 0.0, 0.0], dtype=F64)
        diag = {}
        try:
            model.steady(state, drivers, diagnostics=diag, atol=1e-14, rtol=1e-14)
            if not bool(diag["converged"]):
                fails.append((k, "nc"))
            passes.append(diag["passes"])
        except RuntimeError:
            fails.append((k, "raise"))
    import collections
    print(f"NMIN {nmin}: fails {fails} passes {sorted(collections.Counter(passes).items())}")
