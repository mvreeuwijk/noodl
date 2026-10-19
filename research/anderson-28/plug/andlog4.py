import os

import andlog3
import torch

import noodl.solvers.anderson as A

WANT = {int(c) for c in os.environ["CALLS"].split(",")}
prev = A._anderson_mix
def dump(xs, gs, rs, ns=None):
    if andlog3.CALL[0] + 1 in WANT and ns is not None:
        torch.set_printoptions(precision=6, linewidth=200)
        print("=== call", andlog3.CALL[0] + 1)
        for k in range(len(xs)):
            print(" k", k, "x", xs[k].tolist(), "\n      g", gs[k].tolist(), "\n      r", rs[k].tolist(), "\n      n", ns[k].tolist())
        for j, d, ratio, ok in andlog3.ratios(xs, rs, ns):
            print("  col", j, "d", d.item(), "ratio", ratio.item(), "condok", ok.item())
    return prev(xs, gs, rs, ns)
A._anderson_mix = dump
