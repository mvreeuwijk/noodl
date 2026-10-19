import sys; import os; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, ".")
ARGS = sys.argv[1:]
import toyscan

G = toyscan.G
which, tols = ARGS[0], list(map(float, ARGS[1:]))
for t in tols:
    if which == "mna": toyscan.report(f"mna tol={t}", lambda: G.case_mna_dynamic_iterate_feedback(t))
    if which == "cpl": toyscan.report(f"coupled rtol={t}", lambda: G.case_coupled_two_way(t))
    if which == "cplb": toyscan.report(f"coupled_b rtol={t}", lambda: G.case_coupled_two_way_batch(t))
