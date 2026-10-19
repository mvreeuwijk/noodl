import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.argv = ["x", "__none__"]
import importlib.util

spec = importlib.util.spec_from_file_location("t2", os.path.join(os.path.dirname(__file__), "trace2.py"))
# trace2 runs on import; give it no cases
G = None
src = open(spec.origin).read().split("VERB = ")[0]
ns = {"__name__": "t2"}; exec(compile(src, spec.origin, "exec"), ns)
G = ns["G"]; LOG = ns["LOG"]
def report(label, fn):
    LOG.clear(); G.outcome(fn)
    loops, cur = [], None
    for e in LOG:
        if e is None or cur is None:
            cur = []; loops.append(cur)
        if e is not None: cur.append(e)
    for li, lp in enumerate(loops):
        if not lp: continue
        n = max(t.numel() for t in lp[0][0])
        for i in range(n):
            hist = []
            for R, U in lp:
                r = max(t[min(i, t.numel()-1)].item() for t in R)
                u = min(t[min(i, t.numel()-1)].item() for t in U)
                hist.append((r, u))
                if r <= 1: break
            rl, ul = hist[-1]; rp = hist[-2][0] if len(hist) > 1 else float("nan")
            print(f"{label:40s} loop{li} inst{i} pass{len(hist)+1} r_last={rl:.3g} r_prev={rp:.3g} tol_ulps={ul:.3g}  hist={[f'{r:.2g}' for r,_ in hist[-4:]]}")
