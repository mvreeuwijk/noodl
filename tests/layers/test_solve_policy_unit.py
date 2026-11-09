"""Unit tests of `noodl.layers._solve_policy`, called on `resolve` directly, on a fast-path
layer (`_classic`) and on a layer with flow unknowns (`_mna`)."""

from __future__ import annotations

import pytest
import torch

from noodl.layers._solve_policy import SolvePolicy, require_diagnostics, resolve
from tests.layers.test_kirchhoff_solve_policy import _classic, _mna

LAYERS = {"fast": _classic, "flow_unknowns": _mna}


def test_require_diagnostics_messages():
    require_diagnostics("e", "return", {})
    with pytest.raises(ValueError, match=r"'e': on_failure='return' requires diagnostics="):
        require_diagnostics("e", "return", None)
    with pytest.raises(ValueError, match="on_failure='return_primal' requires"):
        require_diagnostics("e", "return_primal", None)


def _args(call, reaches):
    (phi_b, drivers, sources, *_), _, _ = call(reaches)
    return drivers, sources, phi_b


@pytest.mark.parametrize("kind", sorted(LAYERS))
def test_mutation_order_and_defaults(kind):
    layer, call = LAYERS[kind]()
    drivers, sources, phi_b = _args(call, False)
    kw = {}
    policy = resolve(layer, kw, None, True, drivers, sources, phi_b)
    assert policy == SolvePolicy(differentiable=True, forward_only=False)
    assert kw == {"method": layer.linear_solver, "where": f"KirchhoffLayer {layer.name!r} solve"}


@pytest.mark.parametrize("kind", sorted(LAYERS))
def test_other_on_failure_untouched(kind):
    layer, call = LAYERS[kind]()
    drivers, sources, phi_b = _args(call, False)
    kw = {"on_failure": "raise", "method": "direct"}
    assert resolve(layer, kw, None, True, drivers, sources, phi_b).differentiable
    assert kw["on_failure"] == "raise" and kw["method"] == "direct"


def test_fast_path_return_without_diagnostics_refused_naming_return():
    layer, call = _classic()
    drivers, sources, phi_b = _args(call, False)
    for value in ("return", "return_primal"):
        with pytest.raises(ValueError, match="on_failure='return' requires"):
            resolve(layer, {"on_failure": value}, None, True, drivers, sources, phi_b)


def test_flow_unknowns_return_primal_without_diagnostics_refused_naming_it():
    layer, call = _mna()
    drivers, sources, phi_b = _args(call, False)
    with pytest.raises(ValueError, match="on_failure='return' requires"):
        resolve(layer, {"on_failure": "return"}, None, True, drivers, sources, phi_b)
    with pytest.raises(ValueError, match="on_failure='return_primal' requires"):
        resolve(layer, {"on_failure": "return_primal"}, None, True, drivers, sources, phi_b)


@pytest.mark.parametrize("kind", sorted(LAYERS))
@pytest.mark.parametrize("reaches", [True, False])
def test_return_primal(kind, reaches):
    """Differentiated ("raise") where something reaches the solve, else `forward_only`
    ("return"): the fast path's threaded forward Newton, the other path's primal solve."""
    layer, call = LAYERS[kind]()
    drivers, sources, phi_b = _args(call, reaches)
    kw = {"on_failure": "return_primal"}
    policy = resolve(layer, kw, {}, True, drivers, sources, phi_b)
    assert policy.differentiable
    assert policy.forward_only is (not reaches)
    assert kw["on_failure"] == ("raise" if reaches else "return")


@pytest.mark.parametrize("kind", sorted(LAYERS))
def test_return_primal_off_the_differentiable_path_is_return(kind):
    layer, call = LAYERS[kind]()
    drivers, sources, phi_b = _args(call, True)
    kw = {"on_failure": "return_primal"}
    assert resolve(layer, kw, {}, False, drivers, sources, phi_b) == SolvePolicy(False)
    assert kw["on_failure"] == "return"


@pytest.mark.parametrize("kind", sorted(LAYERS))
def test_differentiable_flag_is_the_callers(kind):
    """`differentiable=True` stays the differentiable path whether or not anything reaches the
    solve (no primal shortcut); `differentiable=False` is the primal path."""
    layer, call = LAYERS[kind]()
    for reaches in (False, True):
        drivers, sources, phi_b = _args(call, reaches)
        assert resolve(layer, {}, None, True, drivers, sources, phi_b).differentiable
        assert not resolve(layer, {}, None, False, drivers, sources, phi_b).differentiable


@pytest.mark.parametrize("kind", sorted(LAYERS))
def test_step_start_reaches(kind):
    layer, call = LAYERS[kind]()
    drivers, sources, phi_b = _args(call, False)
    h = (torch.ones(2, dtype=torch.float64, requires_grad=True),)
    kw = {"on_failure": "return_primal"}
    assert not resolve(layer, kw, {}, True, drivers, sources, phi_b, h).forward_only
    assert kw["on_failure"] == "raise"
    with torch.no_grad():
        kw = {"on_failure": "return_primal"}
        assert resolve(layer, kw, {}, True, drivers, sources, phi_b, h).forward_only
