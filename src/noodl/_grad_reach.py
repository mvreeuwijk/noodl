"""Whether any tensor requiring grad is reachable from a set of Python objects.

The iterated couplings (`Model._iterate`, `CoupledModel._iterate`) use this to skip the
extra grad-tracked pass that confirms, at the converged point, that nothing differentiable
reaches the state: if no tensor anywhere in the model, its closures or its inputs requires
grad, no pass can produce one, and the pass is skipped.

The answer must never be a false "no", since that would drop a gradient without a word.
It may be a false "yes", which costs one pass. The walk therefore follows
everything a pass could read: attributes (`__dict__` and `__slots__`), registered parameters
and buffers, containers, a function's closure cells, defaults and the module globals its
code names, bound methods, `functools.partial`, and the class attributes and methods of
user classes. Library code (`noodl` itself, the standard library, and installed packages
under `site-packages`) is assumed to hold no tensor requiring grad at module or class
level, so its functions' globals and its classes' attributes are not walked. Its INSTANCES
still are, because they hold the caller's tensors. Without this rule the walk would wander
through torch's dependencies (sympy, ...) on every call. A walk that reaches more than
`_MAX_OBJECTS` objects stops and counts as "yes".
"""

from __future__ import annotations

import functools
import os
import sys
import sysconfig
import types

import torch

_MAX_OBJECTS = 200_000
# Values that hold no tensor and need no walk.
_ATOMS = (str, bytes, bytearray, int, float, complex, bool, type(None), range,
          types.ModuleType, types.BuiltinFunctionType, types.BuiltinMethodType)
_LIBRARY_ROOTS = ("noodl", "builtins")
_LIBRARY_DIRS = tuple(
    os.path.normcase(os.path.abspath(p))
    for p in {sysconfig.get_paths().get(k) for k in ("stdlib", "platstdlib", "purelib",
                                                     "platlib")}
    if p
)


@functools.cache
def _library(module: str | None) -> bool:
    """Whether `module` is library code: `noodl`, a builtin, or a module whose file lies
    in the standard library or `site-packages`."""
    if not module:
        return False
    if module.split(".", 1)[0] in _LIBRARY_ROOTS:
        return True
    mod = sys.modules.get(module)
    path = getattr(mod, "__file__", None)
    if mod is not None and path is None:
        return True                               # a builtin or frozen module
    if not path:
        return False
    path = os.path.normcase(os.path.abspath(path))
    return any(path.startswith(d + os.sep) for d in _LIBRARY_DIRS)


def any_requires_grad(*roots) -> bool:
    """Whether a tensor with `requires_grad=True` may be reachable from `roots`.

    False only when the walk finished and found none; True when it found one or could not
    finish. Runs in time linear in the number of objects reached.
    """
    if not torch.is_grad_enabled():
        return False
    seen: set[int] = set()
    stack: list = list(roots)
    while stack:
        obj = stack.pop()
        if isinstance(obj, torch.Tensor):
            if obj.requires_grad:
                return True
            continue
        if isinstance(obj, _ATOMS):
            continue
        key = id(obj)
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > _MAX_OBJECTS:
            return True                           # could not finish: "maybe"
        stack.extend(_children(obj))
    return False


def _code_globals(code: types.CodeType, glb: dict, out: list) -> None:
    for name in code.co_names:
        if name in glb:
            out.append(glb[name])
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _code_globals(const, glb, out)


def _children(obj) -> list:
    out: list = []
    if isinstance(obj, dict):
        out.extend(obj.keys())
        out.extend(obj.values())
        return out
    if isinstance(obj, (list, tuple, set, frozenset)):
        out.extend(obj)
        return out
    if isinstance(obj, types.MethodType):
        return [obj.__self__, obj.__func__]
    if isinstance(obj, functools.partial):
        return [obj.func, obj.args, obj.keywords]
    if isinstance(obj, (staticmethod, classmethod)):
        return [obj.__func__]
    if isinstance(obj, property):
        return [obj.fget, obj.fset, obj.fdel]
    if isinstance(obj, types.FunctionType):
        out.extend(c.cell_contents for c in (obj.__closure__ or ())
                   if _cell_filled(c))
        out.extend(obj.__defaults__ or ())
        out.extend((obj.__kwdefaults__ or {}).values())
        if not _library(obj.__module__):
            _code_globals(obj.__code__, obj.__globals__, out)
        return out
    if isinstance(obj, type):
        if not _library(obj.__module__):
            out.extend(vars(obj).values())
            out.extend(b for b in obj.__bases__ if b is not object)
        return out
    if isinstance(obj, torch.nn.Module):
        out.extend(obj.parameters())
        out.extend(obj.buffers())
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        out.extend(d.values())
    for cls in type(obj).__mro__:
        slots = vars(cls).get("__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if hasattr(obj, slot):
                out.append(getattr(obj, slot))
    out.append(type(obj))
    return out


def _cell_filled(cell) -> bool:
    try:
        cell.cell_contents  # noqa: B018 -- an empty cell raises
    except ValueError:
        return False
    return True
