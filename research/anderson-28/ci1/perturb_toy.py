"""Pass counts under input perturbations of a few ulp (a stand-in for platform rounding)."""
import sys; sys.path.insert(0, ".")
from tests import test_golden_orchestration as G

EPS = 2.220446049250313e-16
def run(kind, tol, k):
    f = 1 + k * EPS
    pass
    if kind == "batch2":
        rec = G._toy_record("anderson", sources=[[0.5 * f, 0, 0], [40.0 * f, 0, 0]], tol=tol)
    elif kind == "anderson":
        rec = G._toy_record("anderson", sources=[[20.0 * f, 0, 0]], tol=tol)
    elif kind == "plain":
        rec = G._toy_record(None, sources=[[20.0 * f, 0, 0]], tol=tol)
    elif kind == "plain_noflow":
        rec = G._toy_record(None, sources=[[20.0 * f, 0, 0]], tol=tol, flow_tol=False)
    return rec["diagnostics"]["passes"], rec["diagnostics"]["converged"]
kind = sys.argv[1]
for tol in map(float, sys.argv[2:]):
    res = []
    for k in range(-50, 51):
        try:
            p, c = run(kind, tol, k); res.append((p, str(c["data"])))
        except RuntimeError:
            res.append(("ERR", ""))
    print(kind, tol, "passes:", sorted({str(p) for p, _ in res}), "conv:", sorted({c for _, c in res}), "k=0:", res[50][0])
