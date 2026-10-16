"""Moving a built model, or any part of it, to another device or dtype (`Model.to`).

A model is a tree of plain Python objects (layers, closures, reactions, drives, application
records) around `torch.nn.Module`s (laws, node sources) and tensors. `Mover` walks that
tree once and moves every tensor it reaches, in place where the holder is mutable:

* a tensor goes to the new device, and a floating tensor to the new precision; a complex
  tensor keeps being complex at that precision (float32 <-> complex64,
  float64 <-> complex128), an integer or boolean tensor keeps its dtype (index tensors);
* a `torch.nn.Module` moves its parameters and buffers in place (`Module._apply`, so a
  parameter stays the same object and an optimiser keeps it) and its plain tensor
  attributes too;
* a `Network` is moved with `Network.to`, once, however many layers refer to it;
* dicts and lists are updated in place (so sharing is kept: a `notes` dict that a closure
  and the model both hold stays one dict), tuples are rebuilt;
* a `torch.dtype` or `torch.device` stored as an attribute is mapped like a tensor's;
* an object of a class defined in noodl is walked attribute by attribute (frozen dataclasses
  included); an object from elsewhere is moved only when it defines `to(device=, dtype=)`,
  which is the hook a user closure implements. Anything else (a plain function, a lambda, a
  `functools.partial`, a NumPy array) is left as it is.

Every object is visited once (by identity), so cycles and shared parts are safe, and moving
twice to the same place changes nothing.
"""

from __future__ import annotations

import torch

_COMPLEX_OF = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}
_REAL_OF = {c: r for r, c in _COMPLEX_OF.items()}

# What an `nn.Module` keeps in its `__dict__` for itself; `Module._apply` moves the tensors
# among them (parameters and buffers) and the rest holds none.
_MODULE_INTERNALS = frozenset(vars(torch.nn.Module()))

# Caches that are recomputed on demand: emptied rather than moved (their keys are tensor
# identities, which a move changes).
_CACHES = frozenset({"_memo_cache"})


def real_of(dtype: torch.dtype) -> torch.dtype:
    """The real dtype of the same precision (`complex64 -> float32`); a real dtype itself."""
    return _REAL_OF.get(dtype, dtype)


def complex_of(dtype: torch.dtype) -> torch.dtype:
    """The complex dtype of the same precision (`float32 -> complex64`); a complex dtype
    itself."""
    return dtype if dtype.is_complex else _COMPLEX_OF[dtype]


class Mover:
    """One move to `device` and/or the precision of `dtype` (either may be None: kept)."""

    def __init__(self, device=None, dtype: torch.dtype | None = None) -> None:
        self.device = None if device is None else torch.device(device)
        if dtype is not None and not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError(f"to(): dtype must be a floating or complex dtype, got {dtype}")
        self.precision = None if dtype is None else real_of(dtype)
        self._seen: dict[int, object] = {}
        self._keep: list[object] = []   # keeps visited objects alive, so ids stay unique

    # ------------------------------------------------------------------ dtypes
    def dtype_for(self, dtype: torch.dtype) -> torch.dtype:
        """Where a tensor of `dtype` goes: a real or complex dtype changes precision, an
        integer or boolean one stays."""
        if self.precision is None:
            return dtype
        if dtype.is_complex:
            return complex_of(self.precision)
        if dtype.is_floating_point:
            return self.precision
        return dtype

    def tensor(self, t: torch.Tensor) -> torch.Tensor:
        dtype = self.dtype_for(t.dtype)
        device = t.device if self.device is None else self.device
        if t.dtype == dtype and t.device == device:
            return t
        if isinstance(t, torch.nn.Parameter):
            # In place, as `Module._apply` does: the parameter stays the same object.
            with torch.no_grad():
                t.data = t.data.to(device=device, dtype=dtype)
            if t.grad is not None:
                t.grad = t.grad.to(device=device, dtype=dtype)
            return t
        if t.is_leaf and t.requires_grad:
            # A leaf someone optimises: moved, and a leaf again.
            return t.detach().to(device=device, dtype=dtype).requires_grad_(True)
        return t.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ walking
    def visit(self, obj):
        """`obj` moved: the same object for a mutable holder, a new one for a tensor or a
        tuple whose contents moved."""
        if obj is None or isinstance(obj, (bool, int, float, complex, str, bytes, type)):
            return obj
        key = id(obj)
        if key in self._seen:
            return self._seen[key]
        self._keep.append(obj)
        if isinstance(obj, torch.Tensor):
            out = self._seen[key] = self.tensor(obj)
            return out
        if isinstance(obj, torch.dtype):
            out = self._seen[key] = self.dtype_for(obj)
            return out
        if isinstance(obj, torch.device):
            out = self._seen[key] = obj if self.device is None else self.device
            return out
        self._seen[key] = obj          # before recursing: a cycle comes back to `obj`
        if isinstance(obj, torch.nn.Module):
            self._module(obj)
            return obj
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                new = self.visit(v)
                if new is not v:
                    obj[k] = new
            return obj
        if isinstance(obj, list):
            for i, v in enumerate(obj):
                new = self.visit(v)
                if new is not v:
                    obj[i] = new
            return obj
        if isinstance(obj, tuple):
            items = [self.visit(v) for v in obj]
            if all(a is b for a, b in zip(items, obj, strict=True)):
                return obj
            out = type(obj)(*items) if hasattr(obj, "_fields") else type(obj)(items)
            self._seen[key] = out
            return out
        from noodl.topology import Network

        if isinstance(obj, Network):
            obj.to(device=self.device, dtype=None if self.precision is None
                   else self.dtype_for(obj.dtype))
            return obj
        hook = getattr(obj, "_move_with", None)
        if callable(hook):
            hook(self)
            return obj
        if _is_noodl(obj):
            self.attributes(obj)
            return obj
        to = getattr(type(obj), "to", None)
        if callable(to):
            # A foreign object's declared hook (a user closure holding tensors).
            moved = obj.to(device=self.device, dtype=self.precision)
            if moved is None or moved is obj:      # moved in place
                return obj
            # A replacement: every later reference to `obj` gets this same object (`obj`
            # stays alive in `_keep`, so its id is not reused during the walk).
            self._seen[key] = moved
            self._seen[id(moved)] = moved
            self._keep.append(moved)
            return moved
        return obj

    def attributes(self, obj, skip: frozenset = frozenset()) -> None:
        """Move every attribute of `obj` (a plain object, a frozen dataclass), in place."""
        state = getattr(obj, "__dict__", None)
        if state is None:
            return
        for name, value in list(state.items()):
            if name in skip:
                continue
            if name in _CACHES and isinstance(value, dict):
                value.clear()
                continue
            new = self.visit(value)
            if new is not value:
                state[name] = new

    def _raw(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(device=t.device if self.device is None else self.device,
                    dtype=self.dtype_for(t.dtype))

    def _module(self, module: torch.nn.Module) -> None:
        module._apply(self._raw)   # parameters (in place), their gradients and buffers
        for sub in module.modules():
            self._seen[id(sub)] = sub
            self._keep.append(sub)
            self.attributes(sub, skip=_MODULE_INTERNALS)


def _is_noodl(obj) -> bool:
    module = type(obj).__module__ or ""
    return module == "noodl" or module.startswith("noodl.")


class Movable:
    """Mixin: `to(device=None, dtype=None)` moves every tensor the object holds (see
    `Mover`), in place, and returns the object."""

    def to(self, device=None, dtype: torch.dtype | None = None):
        Mover(device, dtype).visit(self)
        return self

    def _move_with(self, mover: Mover) -> None:
        mover.attributes(self)
