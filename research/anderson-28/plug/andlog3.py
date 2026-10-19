import os

import torch

import noodl.solvers.anderson as A

LOG = open(os.environ.get("ANDLOG", "andlog.txt"), "w")
orig = A._anderson_mix
CALL = [0]
MODE = os.environ.get("NOISE_MODE", "proj")


def ratios(xs, rs, ns):
    m = min(len(xs) - 1, rs[-1].shape[-1])
    q, out, rm = [], [], None
    for j in range(m):
        v = rs[-1 - j] - rs[-2 - j]
        for qi in q:
            v = v - (qi * v).sum(-1)[..., None] * qi
        d = torch.sqrt((v * v).sum(-1))
        rm = d if rm is None else torch.maximum(rm, d)
        nn = ns[-1 - j] + ns[-2 - j]
        if MODE == "proj":
            ratio = d * d / (v.abs() * nn).sum(-1)
        else:
            ratio = d / torch.sqrt((nn * nn).sum(-1))
        q.append(v / d.clamp_min(1e-300)[..., None])
        out.append((j, d, ratio, d * A._ANDERSON_COND_MAX > rm))
    return out


def logged(xs, gs, rs, ns=None):
    CALL[0] += 1
    if ns is None:
        return orig(xs, gs, rs)
    with torch.no_grad():
        a, k = orig(xs, gs, rs, ns)
        b, _ = orig(xs, gs, rs, None)
        diff = (a - b).abs().max().item()
        if diff > 0:
            for j, d, ratio, ok in ratios(xs, rs, ns):
                if bool(ok.all()) and float(ratio.min()) < A._ANDERSON_NOISE_MIN:
                    LOG.write(f"call {CALL[0]} |r| {rs[-1].abs().max().item():.3e} col {j} d {d.min().item():.3e} ratio {ratio.min().item():.4g} diffmix {diff:.3e}\n")
                    break
            LOG.flush()
    return a, k


A._anderson_mix = logged
