import sys; sys.path.insert(0, ".")
import torch

import noodl.solvers.anderson as A
from tests import test_golden_orchestration as G

EPS = 2.220446049250313e-16
REC = []
om = A._anderson_mix
def mix_wrap(xs, gs, rs):
    out = om(xs, gs, rs)
    # recompute diagonal ratios
    m = min(len(xs) - 1, rs[-1].shape[-1])
    ds = []; qs = []
    for j in range(m):
        v = rs[-1 - j] - rs[-2 - j]
        for q in qs:
            v = v - (q * v).sum(-1)[..., None] * q
        d = torch.sqrt((v * v).sum(-1)); ds.append(d)
        qs.append(v / d[..., None])
    REC.append(("cols", [f"{d.reshape(-1)[0].item():.2e}" for d in ds], "orig", [f"{(rs[-1-j]-rs[-2-j]).norm(dim=-1).reshape(-1)[0].item():.2e}" for j in range(m)]))
    return out
A._anderson_mix = mix_wrap
oc = A._AndersonHistory.mix
def hist_mix(self, fed, out, w, converged):
    res = oc(self, fed, out, w, converged)
    return res
A._AndersonHistory.mix = hist_mix
tol = float(sys.argv[1])
for k in (0, 4):
    REC.clear(); f = 1 + k * EPS
    try:
        rec = G._toy_record("anderson", sources=[[20.0 * f, 0, 0]], tol=tol)
        print("k", k, "passes", rec["diagnostics"]["passes"])
    except Exception as e:
        print("k", k, "ERR", e)
    for i, r in enumerate(REC[:12]):
        print("  mix", i + 2, *r)
