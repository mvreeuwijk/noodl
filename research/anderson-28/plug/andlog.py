"""pytest plugin: log the Anderson noise-rule decisions."""
import os

import torch

import noodl.solvers.anderson as A

LOG = open(os.environ.get("ANDLOG", "andlog.txt"), "w")
orig = A._anderson_mix
CALL = [0]


def logged(xs, gs, rs, ns=None):
    CALL[0] += 1
    if ns is None or os.environ.get("ANDLOG_OFF"):
        return orig(xs, gs, rs, None if os.environ.get("ANDLOG_OFF") else ns)
    with torch.no_grad():
        m = min(len(xs) - 1, rs[-1].shape[-1])
        q = []
        rec = []
        for j in range(m):
            v = rs[-1 - j] - rs[-2 - j]
            noise = ns[-1 - j] + ns[-2 - j]
            nz = torch.sqrt((noise * noise).sum(-1))
            for qi in q:
                v = v - (qi * v).sum(-1)[..., None] * qi
            d = torch.sqrt((v * v).sum(-1))
            q.append(v / d.clamp_min(1e-300)[..., None])
            rec.append((j, d.flatten()[:4].tolist(), nz.flatten()[:4].tolist(),
                        (d / nz).flatten()[:4].tolist()))
        a, k = orig(xs, gs, rs, ns)
        b, k2 = orig(xs, gs, rs, None)
        diff = (a - b).abs().max().item()
        rn = rs[-1].abs().max().item()
        if diff > 0:
            LOG.write(f"call {CALL[0]} len {len(xs)} |r|max {rn:.3e} diffmix {diff:.3e} kept {k.flatten()[:4].tolist()} vs {k2.flatten()[:4].tolist()}\n")
            for r_ in rec:
                LOG.write(f"   col {r_[0]} d {r_[1]} noise {r_[2]} ratio {r_[3]}\n")
            # entries dominating noise
            n0 = ns[-1].flatten(0, -2)[0] if ns[-1].dim() > 1 else ns[-1]
            r0 = rs[-1].flatten(0, -2)[0] if rs[-1].dim() > 1 else rs[-1]
            LOG.write(f"   ns top {torch.topk(n0, min(5, n0.numel())).values.tolist()} idx {torch.topk(n0, min(5, n0.numel())).indices.tolist()} N {n0.numel()}\n")
            LOG.write(f"   r at those {r0[torch.topk(n0, min(5, n0.numel())).indices].tolist()}\n")
            LOG.flush()
    return a, k


A._anderson_mix = logged
