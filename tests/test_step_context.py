"""The model owns the clock. Integrating closures declare themselves and receive it."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from noodl.layers.transport import TransportLayer
from noodl.model import Model, StepContext
from noodl.topology import Network

F64 = torch.float64


def _t(values):
    return torch.tensor(values, dtype=F64)


def _net_and_layer(scheme="implicit"):
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("zone", "ambient", kind="flow")
    layer = TransportLayer(net, "c", capacity=_t([1.0]), flow_kind="flow",
                           boundary=["ambient"], scheme=scheme)
    return net, layer


class Clock:
    """Integrates: advances its state by the interval it is given."""

    state_keys = ("clock.t",)
    integrates = True

    def __init__(self):
        self.calls = []

    def __call__(self, state, drivers, ctx):
        self.calls.append(ctx)
        if ctx is None:
            return {"clock.t": state["clock.t"], "c.q": _t([1.0])}
        if ctx.dt is None:
            raise ValueError("Clock: steady() has no interval to integrate over")
        return {"clock.t": state["clock.t"] + ctx.dt, "c.q": _t([1.0])}


class Undeclared:
    state_keys = ("u.n",)

    def __call__(self, state, drivers):
        return {"u.n": state["u.n"], "c.q": _t([1.0])}


def _drivers():
    return {"c.x_boundary": _t([0.0]), "c.sources": _t([0.0, 0.0])}


def test_an_integrating_closure_receives_the_step_interval_once_per_step():
    net, layer = _net_and_layer()
    clock = Clock()
    model = Model(net, {"c": layer}, closures=[clock], coupling="iterate",
                  iterate_tol={"c": 1e-14}, iterate_max=10)
    state = {"c.x": _t([1.0]), "clock.t": _t(0.0)}
    new = model.step(state, _drivers(), 7.0)
    assert new["clock.t"].item() == pytest.approx(7.0)      # once per step, whatever the pass count
    assert all(isinstance(c, StepContext) and c.dt == 7.0 for c in clock.calls)
    assert len(clock.calls) >= 2                              # iterate took more than one pass


def test_two_different_intervals_advance_by_two_different_amounts():
    net, layer = _net_and_layer()
    model = Model(net, {"c": layer}, closures=[Clock()])
    state = {"c.x": _t([1.0]), "clock.t": _t(0.0)}
    assert model.step(state, _drivers(), 1.0)["clock.t"].item() == pytest.approx(1.0)
    assert model.step(state, _drivers(), 60.0)["clock.t"].item() == pytest.approx(60.0)


def test_steady_refuses_an_integrating_closure_by_name():
    net, layer = _net_and_layer()
    model = Model(net, {"c": layer}, closures=[Clock()])
    with pytest.raises(ValueError, match="Clock.*steady"):
        model.steady({"c.x": _t([1.0]), "clock.t": _t(0.0)}, _drivers())


def test_a_query_does_not_advance_the_clock():
    net, layer = _net_and_layer()
    clock = Clock()
    model = Model(net, {"c": layer}, closures=[clock])
    state = {"c.x": _t([1.0]), "clock.t": _t(3.0)}
    model.residuals(state, _drivers())
    model.current_flows("c", state, _drivers())
    assert all(c is None for c in clock.calls)
    assert state["clock.t"].item() == 3.0


def test_a_closure_with_state_keys_must_declare_whether_it_integrates():
    net, layer = _net_and_layer()
    with pytest.raises(ValueError, match="Undeclared.*integrates"):
        Model(net, {"c": layer}, closures=[Undeclared()])


def test_an_integrating_closure_must_accept_a_context_argument():
    net, layer = _net_and_layer()

    class Bad:
        integrates = True

        def __call__(self, state, drivers):
            return {}

    with pytest.raises(ValueError, match="Bad.*ctx"):
        Model(net, {"c": layer}, closures=[Bad()])


def test_plain_algebraic_closures_are_unchanged():
    net, layer = _net_and_layer()
    model = Model(net, {"c": layer}, closures=[lambda state, drivers: {"c.q": _t([1.0])}])
    assert model.step({"c.x": _t([1.0])}, _drivers(), 1.0)["c.x"].item() == pytest.approx(0.5)


def test_step_context_is_frozen_and_carries_t():
    ctx = StepContext(dt=2.0, t=10.0)
    assert (ctx.dt, ctx.t) == (2.0, 10.0)
    # `pytest.raises(Exception)` is ruff B017 ("blind exception"); the concrete
    # exception a frozen dataclass raises is `dataclasses.FrozenInstanceError` -- more
    # precise, and it is what actually gets raised here.
    with pytest.raises(FrozenInstanceError):
        ctx.dt = 3.0  # type: ignore[misc]
