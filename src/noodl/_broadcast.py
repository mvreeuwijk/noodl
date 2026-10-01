"""`broadcast_shapes`: `torch.broadcast_shapes` for plain integer shapes, in pure Python.

`torch.broadcast_shapes` goes through `torch._refs` (symbolic-shape checks) and costs
~150-190 us a call on CPU, more than most of the small tensor operations it sizes. The
airflow and transport steps call it about a hundred times per step (the Modelica parity
runs: ~12 % of a step, profiled). The rules are the same: shapes are aligned on the right,
a dimension of 1 stretches, any other mismatch raises `RuntimeError` (as torch does).
"""

from __future__ import annotations

import torch


def broadcast_shapes(*shapes) -> torch.Size:
    """The broadcast of `shapes` (each a `torch.Size`, a tuple of ints or an int)."""
    out: list[int] = []
    for shape in shapes:
        if isinstance(shape, int):
            shape = (shape,)
        n = len(shape)
        if n > len(out):
            out = [1] * (n - len(out)) + out
        off = len(out) - n
        for i, s in enumerate(shape):
            s = int(s)
            o = out[off + i]
            if s == o or s == 1:
                continue
            if o == 1:
                out[off + i] = s
            else:
                raise RuntimeError(
                    f"Shape mismatch: objects cannot be broadcast to a single shape: "
                    f"{tuple(shapes)}")
    return torch.Size(out)
