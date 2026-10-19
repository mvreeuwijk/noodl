import sys; import os; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, ".")
ARGS = sys.argv[1:]
import toyscan

exec(open(os.path.join(os.path.dirname(__file__), "scan_sewer.py")).read().split("mode = ARGS[0]")[0].split("import toyscan")[1])
import numpy as np

LOG = toyscan.LOG
lo, hi, n = map(float, ARGS)
for t in [float(f"{x:.2g}") for x in np.logspace(np.log10(lo), np.log10(hi), int(n))]:
    LOG.clear(); passes = [p for p, _ in case(t)[0]]
    loops, cur = [], None
    for e in LOG:
        if e is None or cur is None:
            cur = []; loops.append(cur)
        if e is not None: cur.append(e)
    m = []; ul = 1e300
    for lp in loops:
        hist = []
        for R, U in lp:
            r = max(x.reshape(-1)[0].item() for x in R); hist.append(r)
            ul = min(ul, min(x.reshape(-1)[0].item() for x in U))
            if r <= 1: break
        m.append(min(1 / hist[-1] if hist[-1] > 0 else 1e9, hist[-2] if len(hist) > 1 else 1e9))
    print(f"tol={t:.2g} passes={passes} margins={[f'{x:.2f}' for x in m]} min={min(m):.2f} tol_ulps={ul:.3g}")
