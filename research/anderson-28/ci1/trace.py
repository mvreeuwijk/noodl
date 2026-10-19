"""Trace per-pass stopping margins of every golden orchestration case."""
import math
import sys

import torch

sys.path.insert(0, ".")
import noodl.couple as C
import noodl.model as M
from tests import test_golden_orchestration as G

LOG = []
ULPS = {}

orig = M._ModelPasses.judge
def judge(self, z, out, *, needs_adjoint):
    v = orig(self, z, out, needs_adjoint=needs_adjoint)
    if v is None:
        LOG.append(None); return v
    m = self.model
    ratios = {}
    for name, tol in m.iterate_tol.items():
        new = out[0]
        key = f"{name}.x" if name in m.transport else f"{name}.phi"
        scale = new[key].detach().abs().flatten(-1 if new[key].dim() < 2 else -2 if name in m.transport and new[key].dim()>2 else -1).amax(-1) if False else new[key].detach().abs().reshape(*v.change[name].shape, -1).amax(-1)
        ratios[name] = (v.change[name] / tol).detach()
        ULPS[name] = ((v.change[name] - tol).abs() / (2.2e-16 * scale)).detach()
        if name in m.kirchhoff and name in m.iterate_flow_tol:
            ratios[name + ".q"] = (v.change[name + ".q"] / m.iterate_flow_tol[name]).detach()
            sq = new[f"{name}.q"].detach().abs().reshape(*v.change[name+".q"].shape, -1).amax(-1)
            ULPS[name + ".q"] = ((v.change[name + ".q"] - m.iterate_flow_tol[name]).abs() / (2.2e-16 * sq)).detach()
    LOG.append((ratios, v.converged.detach().clone(), dict(ULPS))); ULPS.clear()
    return v
M._ModelPasses.judge = judge

corig = C.judge if hasattr(C, "judge") else None
cls = [c for c in vars(C).values() if isinstance(c, type) and "judge" in vars(c)][0]
co = cls.judge
def cjudge(self, z, out, *, needs_adjoint):
    coupled = self.coupled
    new, transfers, pass_drivers = out
    v = co(self, z, out, needs_adjoint=needs_adjoint)
    ratios = {}
    for i in self.two_way_pos:
        link = coupled.links[i]
        returned = coupled._forward_value(link, new, pass_drivers)
        d = (returned - z[i]).abs()
        thr = coupled.iterate_atol + coupled.iterate_rtol * returned.abs()
        ULPS[coupled._link_key(link)] = ((d - thr).abs() / (2.2e-16 * returned.abs())).detach()
        ratios[coupled._link_key(link)] = torch.where(thr > 0, d / thr, torch.where(d > 0, torch.inf, 0.0)).detach()
    LOG.append((ratios, v.converged.detach().clone() if v.converged is not None else None, dict(ULPS))); ULPS.clear()
    return v
cls.judge = cjudge

VERB = '-v' in sys.argv
names = [a for a in sys.argv[1:] if a != '-v'] or sorted(G.CASES)
for name in names:
    LOG.clear()
    G.outcome(G.CASES[name])
    # group consecutive judge calls into loops: a None marks the first pass of a loop
    loops, cur = [], None
    for e in LOG:
        if e is None or cur is None:
            cur = []; loops.append(cur)
        if e is not None: cur.append(e)
    worst = math.inf; detail = ""; wu = math.inf; wdet = ''
    for li, lp in enumerate(loops):
        done = None
        for pi, (ratios, conv, ulps) in enumerate(lp):
            r = torch.stack([torch.as_tensor(t, dtype=torch.float64).reshape(-1) for t in ratios.values()]) if ratios else None
            if r is None: continue
            mx = r.max(0).values
            u = torch.stack([torch.as_tensor(t, dtype=torch.float64).reshape(-1) for t in ulps.values()]).min(0).values  # decision quantity per instance: ok iff mx <= 1
            n = mx.numel()
            if done is None: done = torch.zeros(n, dtype=torch.bool)
            for i in range(n):
                if done[i]: continue
                x = mx[i].item()
                if x == 0 or not math.isfinite(x):
                    marg = math.inf
                else:
                    marg = abs(math.log10(x))
                uu = u[min(i, u.numel()-1)].item()
                if VERB: print(f"   loop{li} pass{pi+2} inst{i} ratio={x:.3e} ulps_from_tol={uu:.3g}")
                if uu < wu: wu = uu; wdet = f"loop{li} pass{pi+2} inst{i} ratio={x:.3g} ulps={uu:.3g}"
                if marg < worst:
                    worst = marg; detail = f"loop{li} pass{pi+2} inst{i} ratio={x:.3g} keys=" + ",".join(f"{k}:{torch.as_tensor(t).reshape(-1)[min(i, torch.as_tensor(t).numel()-1)].item():.3g}" for k,t in ratios.items())
                if x <= 1: done[i] = True
        if lp:
            detail += f" | loop{li}: {len(lp)+1} passes"
    print(f"{name:50s} min|log10 r|={worst:5.2f} min ulps-from-tol={wu:9.3g} [{wdet}] {detail[-30:]}")
