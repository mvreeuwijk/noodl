import os

import torch

import noodl.solvers.anderson as A

LOG = open(os.environ.get("ANDLOG", "andlog.txt"), "w")
orig = A._anderson_mix
CALL = [0]


def logged(xs, gs, rs, ns=None):
    CALL[0] += 1
    if ns is None:
        return orig(xs, gs, rs)
    with torch.no_grad():
        m = min(len(xs) - 1, rs[-1].shape[-1])
        q = []
        rec = []
        for j in range(m):
            v = rs[-1 - j] - rs[-2 - j]
            raw = v.clone()
            for qi in q:
                v = v - (qi * v).sum(-1)[..., None] * qi
            d = torch.sqrt((v * v).sum(-1))
            noise = (v.abs() * (ns[-1 - j] + ns[-2 - j])).sum(-1)
            q.append(v / d.clamp_min(1e-300)[..., None])
            rec.append((j, d.flatten()[:2].tolist(), (d * d / noise).flatten()[:2].tolist(), raw.flatten()[:8].tolist(), (ns[-1-j]+ns[-2-j]).flatten()[:8].tolist()))
        a, k = orig(xs, gs, rs, ns)
        b, k2 = orig(xs, gs, rs, None)
        diff = (a - b).abs().max().item()
        rn = rs[-1].abs().max().item()
        if diff > 0:
            LOG.write(f"call {CALL[0]} len {len(xs)} |r|max {rn:.3e} diffmix {diff:.3e} kept {k.flatten()[:2].tolist()} vs {k2.flatten()[:2].tolist()}\n")
            for r_ in rec:
                LOG.write(f"   col {r_[0]} d {r_[1]} d2/noise {r_[2]}\n      v {r_[3]}\n      n {r_[4]}\n")
            LOG.write(f"   r {rs[-1].flatten()[:8].tolist()}\n")
            LOG.flush()
    return a, k


A._anderson_mix = logged
