import sys; import os; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, ".")
TOL = float(sys.argv[1])
import toyscan

G = toyscan.G; LOG = toyscan.LOG
EPS = 2.220446049250313e-16

for k in (0, 4, -4, 8):
    f = 1 + k * EPS
    LOG.clear(); G.outcome(lambda: G._toy_record("anderson", sources=[[0.5 * f, 0, 0], [40.0 * f, 0, 0]], tol=TOL))
    for i in (0, 1):
        print(k, i, " ".join(f"{max(t[i].item() for t in R):.2e}" for R, U in LOG[1:]))
