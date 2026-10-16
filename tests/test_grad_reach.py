"""`any_requires_grad`: never a false "no" wherever a pass could read a tensor from."""

from __future__ import annotations

import functools

import pytest
import torch

from noodl._grad_reach import any_requires_grad

F64 = torch.float64
GLOBAL_THETA = torch.tensor(1.0, dtype=F64, requires_grad=True)


def _leaf():
    return torch.tensor(2.0, dtype=F64, requires_grad=True)


def _plain():
    return torch.tensor(2.0, dtype=F64)


class _Holder:
    def __init__(self, value):
        self.value = value


class _Slotted:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class _ClassLevel:
    scale = torch.tensor(3.0, dtype=F64, requires_grad=True)


class _ReadsGlobal:
    def __call__(self, state, drivers):
        return {"x": GLOBAL_THETA}


class _Module(torch.nn.Module):
    def __init__(self, learnable):
        super().__init__()
        if learnable:
            self.p = torch.nn.Parameter(torch.ones(1, dtype=F64))
        else:
            self.register_buffer("b", torch.ones(1, dtype=F64))


def _closure_over(value):
    def closure(state, drivers):
        return {"x": value}
    return closure


@pytest.mark.parametrize("make", [
    lambda: _leaf(),
    lambda: {"k": [(_leaf(),)]},
    lambda: _Holder(_Holder(_leaf())),
    lambda: _Slotted(_leaf()),
    lambda: _ClassLevel(),
    lambda: _ReadsGlobal(),
    lambda: _Module(True),
    lambda: _closure_over(_leaf()),
    lambda: functools.partial(lambda s, d, v: v, v=_leaf()),
    lambda: _Holder(_closure_over(_leaf())).value,
    lambda: _leaf() * 2.0,                        # a non-leaf requiring grad
], ids=["tensor", "containers", "attributes", "slots", "class attribute", "module global",
        "parameter", "closure cell", "partial", "bound closure", "non-leaf"])
def test_a_reachable_tensor_requiring_grad_is_found(make):
    assert any_requires_grad(make())


def test_nothing_requiring_grad_is_a_no():
    roots = ({"k": [_plain()]}, _Holder(_plain()), _Module(False), _closure_over(_plain()),
             [1, 2.0, "s", None])
    assert not any_requires_grad(*roots)


def test_under_no_grad_the_answer_is_no():
    with torch.no_grad():
        assert not any_requires_grad(_leaf())


def test_a_reference_cycle_terminates():
    a = _Holder(None)
    a.value = [a, _plain()]
    assert not any_requires_grad(a)
