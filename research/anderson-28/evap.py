import pathlib
import sys
import tempfile

sys.path.insert(0, "C:/projects/noodl-ci3")
import torch

import noodl.solvers.anderson as A
from noodl.apps.sewer import simulate_dynamic_wave
from tests.apps.sewer.test_losses import build

tmp = pathlib.Path(tempfile.mkdtemp())

def run(evap, bump=0.0, grad=False):
    net, (model, state, drivers) = build(tmp, evap=evap, iterate_tol=1e-11, iterate_max=500)
    inflow = drivers["inflow"].clone()
    inflow[0] = inflow[0] + bump
    leaf = inflow.clone().requires_grad_(grad)
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=24, inflow=leaf,
                                    record=("water.q",))
    return hist["water.q"][-1, 1], leaf

for nmin in [float(a) for a in sys.argv[1:]]:
    A._ANDERSON_NOISE_MIN = nmin
    q, leaf = run(20000.0, grad=True)
    (g,) = torch.autograd.grad(q, leaf)
    h = 1e-6
    qp = run(20000.0, bump=h)[0].item(); qm = run(20000.0, bump=-h)[0].item()
    fd = (qp - qm) / (2 * h)
    print(f"NMIN {nmin}: grad {g[0].item():.9e} fd {fd:.9e} q {q.item():.15e} q+ {qp:.15e} q- {qm:.15e}")
