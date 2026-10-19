import sys; import os; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, ".")
which = sys.argv[1]; TOLS = list(map(float, sys.argv[2:]))
import toyscan

G = toyscan.G

fns = {"batch2": lambda tol: G._toy_record("anderson", sources=[[0.5,0,0],[40.0,0,0]], tol=tol),
       "anderson": lambda tol: G._toy_record("anderson", tol=tol),
       "plain": lambda tol: G._toy_record(None, tol=tol)}
for tol in TOLS:
    toyscan.report(f"{which} tol={tol}", lambda: fns[which](tol))
