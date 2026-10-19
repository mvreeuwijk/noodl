import sys; import os; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, ".")
import toyscan

exec(open(os.path.join(os.path.dirname(__file__), "scan_sewer.py")).read().split("mode = ARGS[0]")[0].split("import toyscan")[1])
LOG = toyscan.LOG
for k in (0, 6, -6):
    LOG.clear(); case(1e-12, k)
    loops, cur = [], None
    for e in LOG:
        if e is None or cur is None:
            cur = []; loops.append(cur)
        if e is not None: cur.append(e)
    lp = loops[2]
    print(k, [(f"{R[0].item():.4g}", f"{R[1].item():.4g}", f"{U[0].item():.3g}", f"{U[1].item():.3g}") for R, U in lp[-3:]])
