"""Unit tests of `noodl.layers._solve_policy`, called on the functions directly."""

from __future__ import annotations

import pytest
import torch

from noodl.layers._solve_policy import (
    SolvePolicy,
    require_diagnostics,
    resolve_classic,
    resolve_mna,
)
from tests.layers.test_kirchhoff_solve_policy import _classic, _mna


def test_require_diagnostics_messages():
    require_diagnostics("e", "return", {})
    with pytest.raises(ValueError, match=r"'e': on_failure='return' requires diagnostics="):
        require_diagnostics("e", "return", None)
    with pytest.raises(ValueError, match="on_failure='return_primal' requires"):
        require_diagnostics("e", "return_primal", None)


def _args(layer, call, reaches):
    (phi_b, drivers, sources, *_), _, _ = call(reaches)
    return drivers, sources, phi_b


def test_classic_mutation_order_and_defaults():
    layer, call = _classic(False)
    drivers, sources, phi_b = _args(layer, call, False)
    kw = {}
    policy = resolve_classic(layer, kw, None, True, drivers, sources, phi_b)
    assert policy == SolvePolicy(differentiable=True, forward_only=False)
    assert kw == {"method": layer.linear_solver, "where": f"KirchhoffLayer {layer.name!r} solve"}


def test_classic_return_without_diagnostics_refused():
    layer, call = _classic(False)
    drivers, sources, phi_b = _args(layer, call, False)
    for value in ("return", "return_primal"):
        with pytest.raises(ValueError, match="on_failure='return' requires"):
            resolve_classic(layer, {"on_failure": value}, None, True, drivers, sources, phi_b)


@pytest.mark.parametrize("reaches", [True, False])
def test_classic_return_primal(reaches):
    layer, call = _classic(False)
    drivers, sources, phi_b = _args(layer, call, reaches)
    kw = {"on_failure": "return_primal"}
    policy = resolve_classic(layer, kw, {}, True, drivers, sources, phi_b)
    assert policy.differentiable
    assert policy.forward_only is (not reaches)
    assert kw["on_failure"] == ("raise" if reaches else "return")


def test_classic_analytic_primal_and_primal_path():
    layer, call = _classic(True)
    drivers, sources, phi_b = _args(layer, call, False)
    kw = {"on_failure": "return_primal"}
    policy = resolve_classic(layer, kw, {}, True, drivers, sources, phi_b)
    assert policy == SolvePolicy(differentiable=False, forward_only=False)
    assert kw["on_failure"] == "return"
    drivers, sources, phi_b = _args(layer, call, True)
    assert resolve_classic(layer, {}, None, True, drivers, sources, phi_b).differentiable
    assert not resolve_classic(layer, {}, None, False, drivers, sources, phi_b).differentiable


@pytest.mark.parametrize("reaches", [True, False])
def test_mna_return_primal(reaches):
    layer, call = _mna(False)
    drivers, sources, phi_b = _args(layer, call, reaches)
    kw = {"on_failure": "return_primal"}
    with pytest.raises(ValueError, match="return_primal"):
        resolve_mna(layer, dict(kw), None, True, drivers, sources, phi_b, None)
    differentiable = resolve_mna(layer, kw, {}, True, drivers, sources, phi_b, None)
    assert differentiable is reaches
    assert kw["on_failure"] == ("raise" if reaches else "return")


def test_mna_other_on_failure_untouched():
    layer, call = _mna(False)
    drivers, sources, phi_b = _args(layer, call, False)
    kw = {"on_failure": "raise"}
    assert resolve_mna(layer, kw, None, True, drivers, sources, phi_b, None) is True
    assert kw == {"on_failure": "raise"}


def test_mna_history_reaches():
    layer, call = _mna(False)
    drivers, sources, phi_b = _args(layer, call, False)
    h = (torch.ones(2, dtype=torch.float64, requires_grad=True),)
    kw = {"on_failure": "return_primal"}
    assert resolve_mna(layer, kw, {}, True, drivers, sources, phi_b, h) is True
    assert kw["on_failure"] == "raise"
