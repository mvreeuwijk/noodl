"""Per case and instance: the stopping decision's margins.

For each fixed-point loop and instance: r_last = (decision quantity / threshold) at the pass
that converged (or the last pass), r_prev = the same at the pass before, and tol_ulps = the
threshold in ulps of the judged state (how far the threshold sits above round-off)."""
import math
import sys

import torch

sys.path.insert(0, ".")
import noodl.couple as C
import noodl.model as M
from tests import test_golden_orchestration as G

EPS = 2.220446049250313e-16
LOG = []

orig = M._ModelPasses.judge
def judge(self, z, out, *, needs_adjoint):
    v = orig(self, z, out, needs_adjoint=needs_adjoint)
    if v is None:
        LOG.append(None); return v
    m = self.model; new = out[0]
    R, U = [], []
    for name, tol in m.iterate_tol.items():
        key = f"{name}.x" if name in m.transport else f"{name}.phi"
        ch = v.change[name].detach()
        sc = new[key].detach().abs().reshape(*ch.shape, -1).amax(-1)
        R.append((ch / tol).reshape(-1)); U.append((tol / (EPS * sc)).reshape(-1))
        if name in m.kirchhoff and name in m.iterate_flow_tol:
            ft = m.iterate_flow_tol[name]; ch = v.change[name + ".q"].detach()
            sc = new[f"{name}.q"].detach().abs().reshape(*ch.shape, -1).amax(-1)
            R.append((ch / ft).reshape(-1)); U.append((ft / (EPS * sc)).reshape(-1))
    LOG.append((R, U))
    return v
M._ModelPasses.judge = judge

cls = [c for c in vars(C).values() if isinstance(c, type) and "judge" in vars(c)][0]
co = cls.judge
def cjudge(self, z, out, *, needs_adjoint):
    coupled = self.coupled
    new, transfers, pass_drivers = out
    v = co(self, z, out, needs_adjoint=needs_adjoint)
    R, U = [], []
    for i in self.two_way_pos:
        link = coupled.links[i]
        ret = coupled._forward_value(link, new, pass_drivers).detach()
        d = (ret - z[i]).abs().detach()
        thr = coupled.iterate_atol + coupled.iterate_rtol * ret.abs()
        thr = torch.as_tensor(thr, dtype=torch.float64)
        R.append(torch.where(thr > 0, d / thr, torch.where(d > 0, torch.inf, 0.0)).reshape(-1))
        U.append((thr / (EPS * ret.abs())).reshape(-1))
    LOG.append((R, U))
    return v
cls.judge = cjudge

VERB = "-v" in sys.argv
names = [a for a in sys.argv[1:] if a != "-v"] or sorted(G.CASES)
for name in names:
    LOG.clear()
    G.outcome(G.CASES[name])
    loops, cur = [], None
    for e in LOG:
        if e is None or cur is None:
            cur = []; loops.append(cur)
        if e is not None: cur.append(e)
    rows = []
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
            if VERB: print("   ", li, i, [f"{r:.3g}" for r, _ in hist])
            r_last, u_last = hist[-1]
            r_prev = hist[-2][0] if len(hist) > 1 else math.nan
            rows.append((li, i, len(hist) + 1, r_last, r_prev, u_last))
    flag = ""
    for li, i, p, rl, rp, u in rows:
        bad = (rl <= 1 and (rl > 0.1 or (rp == rp and rp < 10))) or (rl > 1 and rl < 10) or u < 1e3
        s = f"loop{li} inst{i} pass{p} r_last={rl:.3g} r_prev={rp:.3g} tol_ulps={u:.3g}"
        print(f"{'*' if bad else ' '} {name:48s} {s}")
