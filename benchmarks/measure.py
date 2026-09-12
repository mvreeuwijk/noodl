"""Backward-memory measurement for autograd-differentiable solves.

`tracemalloc` cannot be used to measure PyTorch backward memory: it tracks only allocations
made through Python's own memory allocator (and any allocator that explicitly registers with
it, as numpy's does), and PyTorch's tensor storage is allocated through its own C++ allocator,
which never registers with `tracemalloc`. Measured directly on this `.venv`: an 80 MB
`torch.zeros` tensor shows as 0.000 MB of traced memory. A `tracemalloc`-based assertion about
backward memory would therefore measure Python-object bookkeeping overhead only, and would pass
identically whether or not the thing under test (the implicit adjoint keeping backward memory
independent of forward solver iterations) is actually true -- a vacuous pass, not a real
regression guard.

`saved_tensor_bytes` instead measures, deterministically and without spawning a subprocess,
exactly the set of tensors PyTorch's autograd engine will need during the backward pass: every
tensor `ctx.save_for_backward` (or the analogous internal machinery for built-in ops) retains
for a later `backward()` call. `torch.autograd.graph.saved_tensors_hooks` lets us intercept
every such tensor as it is saved, sum its footprint (`numel() * element_size()`), and return
the tensor unchanged so the forward computation is unaffected.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch

T = TypeVar("T")


def saved_tensor_bytes(fn: Callable[[], T]) -> tuple[int, T]:
    """Run `fn()` and return `(total_bytes_saved_for_backward, fn()'s return value)`.

    `total_bytes_saved_for_backward` is the sum of `t.numel() * t.element_size()` over every
    tensor autograd saves while `fn` runs, i.e. the memory backward() will need -- independent
    of any Python-level object overhead `tracemalloc` would otherwise report instead.
    """
    total = 0

    def pack(t: torch.Tensor) -> torch.Tensor:
        nonlocal total
        total += t.numel() * t.element_size()
        return t

    def unpack(t: torch.Tensor) -> torch.Tensor:
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        result = fn()
    return total, result
