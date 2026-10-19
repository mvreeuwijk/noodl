import sys; import os; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, ".")
ARGS = sys.argv[1:]
from pathlib import Path

import toyscan

from noodl.apps.sewer import read_swmm_inp
from noodl.apps.sewer.dynwave import build_dynamic_wave_model, simulate_dynamic_wave

EPS = 2.220446049250313e-16
data = Path("tests/data/sewer")
def case(tol, k=0):
    net, _, _ = read_swmm_inp(data / "tree_dynwave.inp")
    model, state, drivers = build_dynamic_wave_model(net, iterate_tol=tol)
    if k: drivers = dict(drivers, inflow=drivers["inflow"] * (1 + k * EPS))
    diags = []
    final, hist = simulate_dynamic_wave(model, state, drivers, steps=4, diagnostics=diags)
    return [(d.get("passes"), str(d.get("converged"))) for d in diags], model
mode = ARGS[0]
for t in map(float, ARGS[1:]):
    if mode == "margins":
        toyscan.report(f"sewer tol={t}", lambda: case(t))
        print("   flow_tol:", case(t)[1].iterate_flow_tol)
    else:
        res = set()
        for k in range(-1000, 1001, 40):
            try: res.add(str(case(t, k)[0]))
            except Exception as e: res.add("ERR " + str(e)[:60])
        print(t, len(res), sorted(res)[:4])
