"""Measurement primitives for the milestone-1b benchmarks: backward memory, wall clock, RSS.

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

`isolated_peak_rss` is the same argument applied to whole-process PEAK memory budgets (the
milestone design's section 6.1 table): since `tracemalloc` cannot see torch's allocator, the
only honest measurement of "how much memory did this workload need" is the operating system's
own working-set high-water mark, taken in a FRESH process that does nothing but import the
workload and run it. Isolation is by construction -- one process per measurement -- rather
than by resetting a counter, and the interpreter + torch import baseline is subtracted so the
figure is the workload's own footprint.

`time_call` is a thin `perf_counter` wrapper returning both the elapsed time and the
callable's own return value, so one call site can report timing without a second invocation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import torch

T = TypeVar("T")

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The child program `isolated_peak_rss` runs, as a single `-c` source string (never a shell
# command line: quoting a JSON payload through cmd.exe is exactly the kind of thing that
# breaks on Windows). Module, function and JSON-encoded kwargs arrive as argv[1:4].
_CHILD_PROGRAM = r"""
import ctypes
import importlib
import json
import os
import sys


def _working_set():
    # (current_bytes, peak_bytes) for this process, however the platform exposes them.
    if sys.platform == "win32":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # Declare the signatures: without them ctypes passes the process HANDLE as a 32-bit
        # int, and GetProcessMemoryInfo fails on 64-bit Windows with a garbage handle.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            ctypes.c_ulong,
        ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise OSError(f"GetProcessMemoryInfo failed ({ctypes.get_last_error()})")
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)

    import resource

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is kilobytes on Linux and bytes on macOS/BSD.
    peak = int(raw) * 1024 if sys.platform.startswith("linux") else int(raw)
    current = peak
    try:
        with open("/proc/self/statm") as fh:
            current = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        pass
    return current, peak


module_name, func_name, payload = sys.argv[1], sys.argv[2], sys.argv[3]
func = getattr(importlib.import_module(module_name), func_name)
before, _ = _working_set()
result = func(**json.loads(payload))
_, peak = _working_set()
sys.stdout.write("\n__ISOLATED_PEAK_RSS__" + json.dumps(
    {"before": before, "peak": peak, "result": result}
) + "\n")
"""

_MARKER = "__ISOLATED_PEAK_RSS__"


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


def time_call(fn: Callable[[], T]) -> tuple[float, T]:
    """Run `fn()` and return `(elapsed_seconds, fn()'s return value)`."""
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return elapsed, result


def isolated_peak_rss(module: str, func: str, kwargs: dict) -> tuple[int, Any]:
    """Peak resident memory (bytes) of `module.func(**kwargs)`, run in a fresh process.

    Returns `(peak_above_baseline_bytes, result)`, where the baseline is the working-set size
    the child reports immediately BEFORE the call (i.e. after the interpreter, torch and
    `module` are already imported), and `result` is whatever `func` returned -- which must
    therefore be JSON-serialisable, since it crosses a process boundary.

    `func` must be a NAMED module-level function, not a closure: the child imports it by
    name, so there is nothing to pickle and nothing about the parent's state leaks into the
    measurement. That is the isolation: every call is its own process, so a measurement can
    never inherit another one's high-water mark.
    """
    payload = json.dumps(kwargs)
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM, module, func, payload],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"isolated_peak_rss child for {module}.{func}({kwargs}) exited "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [ln for ln in completed.stdout.splitlines() if ln.startswith(_MARKER)]
    if not lines:
        raise RuntimeError(
            f"isolated_peak_rss child for {module}.{func}({kwargs}) printed no measurement "
            f"line\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    measurement = json.loads(lines[-1][len(_MARKER) :])
    return int(measurement["peak"]) - int(measurement["before"]), measurement["result"]
