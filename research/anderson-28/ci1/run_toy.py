import sys; import os; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, ".")
import toyscan

G = toyscan.G
for tol in (1e-13, 1e-12, 3e-12, 1e-11, 3e-11, 1e-10, 1e-9):
    toyscan.report(f"batch2 tol={tol}", lambda: G._toy_record("anderson", sources=[[0.5,0,0],[40.0,0,0]], tol=tol))
    toyscan.report(f"anderson tol={tol}", lambda: G._toy_record("anderson", tol=tol))
    toyscan.report(f"plain tol={tol}", lambda: G._toy_record(None, tol=tol))
    toyscan.report(f"plain_noflow tol={tol}", lambda: G._toy_record(None, tol=tol, flow_tol=False))
