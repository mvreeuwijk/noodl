"""Checks that a differentiable solve sees every tensor it should.

Answers: does a law, storage or closure hold a tensor that requires grad but that the
solve would not thread (`_held_tensors`, `_check_no_unreachable_differentiable_tensors`),
and does the residual reach the threaded parameters (`_require_threaded`,
`_threaded_residual`)? A mixin of `KirchhoffLayer`.
"""

from __future__ import annotations

import types
from collections.abc import Mapping

import torch

from noodl._tracking import bulk_values
from noodl.elements.base import ConstitutiveLaw, ImplicitLaw
from noodl.storage import NodeStorage
from noodl.withdrawals import NodeWithdrawal

#: Containers longer than this are not entered by `_held_tensors`: a long list or dict on a
#: law is a numeric table, not a parameter holder, and walking it item by item would cost
#: more than the solve (`KirchhoffLayer._require_threaded`, which evaluates the residual,
#: needs no walk).
_SCAN_MAX_ITEMS = 1024


_ATOMS = (str, bytes, int, float, complex, bool, type(None))


_OPAQUE = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType,
           types.MethodType)


def _detached(params) -> list:
    """`params` with every tensor detached."""
    return [p.detach() if isinstance(p, torch.Tensor) else p for p in params]


def plain_call(owner) -> bool:
    """Whether calling `owner` (a law, a withdrawal or a storage) as a module is calling its
    own method (`flow`, an implicit law's `residual`, `rate`, and their slopes): its class
    keeps its base's `forward` and it carries no forward hook."""
    base = (ImplicitLaw if isinstance(owner, ImplicitLaw)
            else ConstitutiveLaw if isinstance(owner, ConstitutiveLaw)
            else NodeWithdrawal if isinstance(owner, NodeWithdrawal) else NodeStorage)
    return not (type(owner).forward is not base.forward or owner._forward_hooks
                or owner._forward_pre_hooks)


def _same_objects(a, b) -> bool:
    """Whether the sequences `a` and `b` hold the same objects (`is`), position by position."""
    return a is not None and len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))


def _shallow(obj):
    """What one object of a scan holds, by identity: a container's items (its length alone
    past `_SCAN_MAX_ITEMS`, where the scan does not enter it), an object's attributes. Equal
    shallow signatures of every object a scan visited mean the scan would find the same
    tensors (`KirchhoffLayer._held_scan`)."""
    if isinstance(obj, dict):
        if len(obj) > _SCAN_MAX_ITEMS:
            return len(obj)
        return tuple(obj), tuple(map(id, obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)):
        if len(obj) > _SCAN_MAX_ITEMS:
            return len(obj)
        return tuple(map(id, obj))
    held = vars(obj)
    return tuple(held), tuple(map(id, held.values()))


def _held_tensors(obj, *, registered_threaded: bool = True, visited: list | None = None):
    """Yield `(name, tensor)` for every tensor `obj` holds: its attributes, its registered
    buffers (when a module), and tensors nested at any depth (up to 8) in lists, tuples,
    dicts, sets and the attributes of plain objects and modules held there.

    With `registered_threaded`, `obj`'s own registered parameters (`_parameters`) and
    submodules (`_modules`, scanned as holders of their own) are not yielded: the
    differentiable solve threads them. A module reached through a container or a plain
    attribute is not registered, so its parameters ARE yielded. Each object is visited once
    (cycles end), and a container longer than `_SCAN_MAX_ITEMS` is not entered. `visited`,
    when given, collects every object whose contents the scan read (`_shallow`)."""
    seen: set[int] = set()
    record = visited.append if visited is not None else (lambda _o: None)

    def walk(label, value, depth):
        if isinstance(value, torch.Tensor):
            yield label, value
            return
        if depth >= 8 or isinstance(value, _ATOMS + _OPAQUE) or id(value) in seen:
            return
        seen.add(id(value))
        record(value)
        if isinstance(value, dict):
            if len(value) <= _SCAN_MAX_ITEMS:
                for k, v in value.items():
                    yield from walk(f"{label}[{k!r}]", v, depth + 1)
        elif isinstance(value, (list, tuple, set, frozenset)):
            if len(value) <= _SCAN_MAX_ITEMS:
                for i, v in enumerate(value):
                    yield from walk(f"{label}[{i}]", v, depth + 1)
        elif hasattr(value, "__dict__"):
            for k, v in vars(value).items():
                if k != "_memo_cache":
                    yield from walk(f"{label}.{k}", v, depth + 1)

    skip = {"_memo_cache"}
    record(obj)
    if registered_threaded:
        skip |= {"_parameters", "_buffers", "_modules"}
        if isinstance(obj, torch.nn.Module):
            # what is registered decides what is skipped: a change there changes the scan
            for registry in (obj._parameters, obj._buffers, obj._modules):
                record(registry)
    for name, value in vars(obj).items():
        if name not in skip:
            yield from walk(name, value, 0)
    if registered_threaded and isinstance(obj, torch.nn.Module):
        for name, value in obj._buffers.items():
            if value is not None:
                yield name, value


class _GuardsMixin:
    """The `_guards` methods of `KirchhoffLayer`; see the module docstring."""

    def _check_no_unreachable_differentiable_tensors(self) -> None:
        """Guard against a gradient that would be silently wrong or absent: the differentiable
        solve threads only two kinds of tensor into
        `Function.apply` -- each ConstitutiveLaw's own registered `nn.Parameter`s (found via
        `named_parameters()`, substituted in via `functional_call`) and whatever is reachable
        through the `drivers`/`sources`/`phi_boundary` arguments to `solve()`. Any OTHER
        tensor a ConstitutiveLaw happens to hold with `requires_grad=True` is captured only
        by closure (the law object itself, not its tensor payload) and is
        invisible to `_Implicit.backward`: for a ConstitutiveLaw, this happens when it was
        constructed with `learnable=False` on a tensor that already had `requires_grad=True`
        (`ConstitutiveLaw._param`'s documented pass-through exception, which is still a correct
        construction -- its graph is reachable through this layer's own component calls
        (`flows`, `dflows`, `residual`, `jacobian`, `linear_init`), though NOT through
        `solve(differentiable=False)`, which returns detached tensors).
        This raises now, before dispatching, rather than returning a gradient that is silently
        wrong (if some other path happens to also touch the same value) or silently `None` (if
        it doesn't).
        """
        for role, el, name, value in self._held_scan():
            if not value.requires_grad:
                continue
            if role == "storage":
                raise ValueError(
                    f"storage {el!r} has a tensor attribute {name!r} with "
                    f"requires_grad=True that is not a registered nn.Parameter, so "
                    f"the differentiable solve cannot reach it through "
                    f"Function.apply and its gradient would be silently wrong or "
                    f"absent. Construct it with learnable=True, or register it as an "
                    f"nn.Parameter in __init__, or use differentiable=False."
                )
            if role == "source":
                # A NodeWithdrawal has no `learnable=` constructor kwarg (unlike
                # ConstitutiveLaw) -- its fix is to register the tensor as an nn.Parameter
                # directly, so it must be named and fixed as a withdrawal, not
                # rendered (and told to construct itself) as if it were a ConstitutiveLaw.
                raise ValueError(
                    f"withdrawal {el!r} has a tensor attribute {name!r} with "
                    f"requires_grad=True that is not a registered nn.Parameter, so "
                    f"the differentiable solve cannot reach it through "
                    f"Function.apply and its gradient would be silently wrong or "
                    f"absent. Register it as an nn.Parameter "
                    f"(self.<name> = torch.nn.Parameter(...)) in __init__ so it is "
                    f"reachable via named_parameters() (buffers and tensors inside "
                    f"containers are not), or use differentiable=False."
                )
            raise ValueError(
                f"law {el!r} (kind {el.kind!r}) has a "
                f"tensor attribute {name!r} with requires_grad=True that is not a "
                f"registered nn.Parameter (a buffer or a tensor inside a container "
                f"is not one either), so the differentiable solve cannot reach "
                f"it through Function.apply and its gradient would be silently wrong "
                f"or absent. Construct this law with learnable=True (so the tensor "
                f"is registered and reachable via named_parameters()), or use "
                f"differentiable=False."
            )

    def _held_scan(self) -> list:
        """`(role, owner, name, tensor)` for every tensor the laws, withdrawals, storages
        (every submodule: a law may wrap another, `Scaled`) hold where the
        differentiable solve does not thread it (`_held_tensors`). The scan is cached: it is
        repeated only when an object it read holds something else (`_shallow`), and the tensors'
        `requires_grad` is read live."""
        cache = getattr(self, "_held_cache", None)
        if cache is not None and all(_shallow(o) == sig for o, sig in cache[1]):
            return cache[0]
        visited: list = [self._laws, self._withdrawals, self._storage]
        found = []
        for role, owners in (("law", self._laws), ("source", self._withdrawals),
                             ("storage", self._storage)):
            for el in owners:
                holders = el.modules() if isinstance(el, torch.nn.Module) else (el,)
                for h in holders:
                    found += [(role, el, n, v) for n, v in _held_tensors(h, visited=visited)]
        # the visited objects are kept alive by the cache, so their ids are not reused
        self._held_cache = (found, [(o, _shallow(o)) for o in visited])
        return found

    def _require_threaded(self, residual, params) -> None:
        """Refuse a differentiable solve whose residual depends on a tensor that is not threaded.

        `residual(*params)` is the residual at one point (the starting guess before the
        solve, the converged root after it) as a function of exactly the tensors handed to
        `Function.apply`. It is evaluated once, under `enable_grad`, with every one of them
        detached: if the result still requires grad, some tensor that requires grad reaches
        the residual by another route (held by a law, a withdrawal in a plain
        object, a module in a plain list, or captured by a closure), and the implicit
        adjoint would return a gradient for it that is silently wrong or absent.

        What this establishes is structural and holds AT THAT POINT only: a law that reads
        an un-threaded tensor on a branch inactive there is not seen. The differentiable
        solve therefore applies it to every residual its forward Newton evaluates
        (`_newton_differentiable`, `_threaded_residual`, at no extra evaluation), from the
        starting guess to the root, which is the point the adjoint linearises at (its
        residual VJP and Jacobian operator are both evaluated there); the Jacobian operator
        is not checked separately: it is the same laws' slopes, their exact `dflow` or the
        autograd derivative of their flows (`_law_slopes`), a constant of the
        adjoint.
        Tensors held as attributes, buffers or in containers are refused before solving
        by `_check_no_unreachable_differentiable_tensors`, which also names them; this
        check catches the rest (a closure capture) and describes the layer.
        """
        self._threaded_residual(residual, params)

    def _threaded_residual(self, residual, params, *, detach: bool = True):
        """`residual(*params)` evaluated with every one of `params` detached (already, with
        `detach=False`), under grad mode, and refused (`_require_threaded`) if it still
        requires grad; returned otherwise (it then carries no graph)."""
        detached = _detached(params) if detach else params
        with torch.enable_grad():
            r = residual(*detached)
        if not (isinstance(r, torch.Tensor) and r.requires_grad):
            return r
        raise ValueError(
            f"KirchhoffLayer {self.name!r}: the residual depends on a tensor with "
            f"requires_grad=True that is not a registered nn.Parameter of a law or node "
            f"source, a driver, the sources or the boundary potentials (it is captured by "
            f"closure or held where no attribute scan reaches), so the differentiable solve "
            f"cannot reach it through Function.apply and its gradient would be silently "
            f"wrong or absent. Register it as an nn.Parameter of the law, pass it through "
            f"the `drivers` mapping, or use differentiable=False."
        )

    # ------------------------------------------------------------------ solve
    def _reaches_grad(self, drivers: Mapping, sources, phi_boundary, extra=()) -> bool:
        """Whether anything a differentiable solve threads through `Function.apply` requires
        grad under the current grad mode: a driver, `sources`, `phi_boundary`, a law's, node
        source's or storage's parameter, or one of `extra` (the step-start tensors of a step
        of a layer with storage)."""
        return torch.is_grad_enabled() and self._threads_grad(drivers, sources, phi_boundary,
                                                              extra)

    def _plain_calls(self) -> bool:
        """Whether calling every owner as a module (what the threaded solve does) is calling
        its own method (what the primal solve does): no law, withdrawal or storage overrides
        `forward` (its alias for `flow` or `rate`) or carries a forward hook."""
        return all(plain_call(o) for o in (*self._laws, *self._withdrawals, *self._storage))

    def _threads_grad(self, drivers: Mapping, sources, phi_boundary, extra=()) -> bool:
        """`_reaches_grad` whatever the grad mode: whether any of those tensors requires
        grad."""
        values = bulk_values(drivers, list(drivers.keys()))
        values += [sources, phi_boundary, *extra]
        for owner in (*self._laws, *self._withdrawals, *self._storage):
            values += list(owner.parameters())
        return any(isinstance(v, torch.Tensor) and v.requires_grad for v in values)
