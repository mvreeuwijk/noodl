"""Independent oracles for R5: a step across a changing capacity conserves the stored amount.

One tank `zone` with a single edge zone->ambient. The closure prescribes the edge flow q
(q < 0 is inflow from ambient at x_boundary = 0, i.e. clean water). The state carries the
capacity at the start of the step under "c.capacity"; the driver "c.capacity" is the
capacity at the end of it. The conserved amount is V * x.
"""

from __future__ import annotations

import pytest
import torch

from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64


def _t(values):
    return torch.tensor(values, dtype=F64)


def _step(scheme, *, q, v_old, v_new, x0=1.0, dt=1.0, carry_capacity=True):
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("zone", "ambient", kind="flow")
    layer = TransportLayer(
        net, "c", capacity=_t([1.0]), flow_kind="flow", boundary=["ambient"], scheme=scheme,
    )
    model = Model(net, {"c": layer}, closures=[lambda state, drivers: {"c.q": _t([q])}])
    state = {"c.x": _t([x0])}
    drivers = {"c.x_boundary": _t([0.0]), "c.sources": _t([0.0, 0.0])}
    if carry_capacity:
        state["c.capacity"] = _t([v_old])
        drivers["c.capacity"] = _t([v_new])
    return model.step(state, drivers, dt)["c.x"].item()


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal"])
@pytest.mark.parametrize(
    "q, v_old, v_new, expected",
    [
        pytest.param(-1.0, 1.0, 2.0, 0.5, id="clean-filling"),
        pytest.param(1.0, 2.0, 1.0, 1.0, id="draining"),
        pytest.param(0.0, 1.0, 2.0, 0.5, id="prescribed-dilution"),
    ],
)
def test_amount_balance_across_a_changing_capacity(scheme, q, v_old, v_new, expected):
    """V_new x_new - V_old x_old = dt * (flux at the new state), the amount form of the step.

    clean-filling: unit volume at x=1 receives one unit of clean water -> V=2, mass 1, x=0.5.
    draining: two units at x=1 lose one unit at their own concentration -> V=1, mass 1, x=1
      (draining never changes a concentration; the implicit and trapezoidal amount forms
      both give exactly 1 because the outflow is linear in x).
    prescribed-dilution: no transport, volume doubles -> the added volume carries nothing,
      mass 1, x=0.5. This is the stated interpretation of a storage change with no flow.
    """
    x = _step(scheme, q=q, v_old=v_old, v_new=v_new)
    assert x == pytest.approx(expected, rel=1e-10)
    assert v_new * x == pytest.approx(v_old * 1.0 - (1.0 if q > 0 else 0.0) * x, rel=1e-10)


@pytest.mark.parametrize("scheme", ["exact", "implicit", "trapezoidal"])
def test_a_fixed_capacity_layer_is_unchanged_by_the_storage_contract(scheme):
    """Control: without capacity keys the step is the classic one. x1 = 1/(1+1) = 0.5 for
    implicit, e^{-1} for exact, (1-0.5)/(1+0.5) = 1/3 for trapezoidal (q = 1, V = 1)."""
    expected = {"exact": 0.36787944117144233, "implicit": 0.5, "trapezoidal": 1 / 3}[scheme]
    assert _step(scheme, q=1.0, v_old=1.0, v_new=1.0, carry_capacity=False) == pytest.approx(
        expected, rel=1e-12
    )


def _bare_layer(scheme):
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("zone", "ambient", kind="flow")
    return TransportLayer(net, "c", capacity=_t([1.0]), flow_kind="flow",
                          boundary=["ambient"], scheme=scheme)


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal"])
def test_layer_step_with_capacity_prev_conserves_the_amount(scheme):
    layer = _bare_layer(scheme)
    x = layer.step(_t([1.0]), _t([-1.0]), _t([0.0, 0.0]), _t([0.0]), 1.0,
                   capacity=_t([2.0]), capacity_prev=_t([1.0]))
    assert x.item() == pytest.approx(0.5, rel=1e-12)


def test_the_exact_scheme_refuses_a_changing_capacity_by_name():
    layer = _bare_layer("exact")
    with pytest.raises(ValueError, match="'c'.*exact.*capacity"):
        layer.step(_t([1.0]), _t([-1.0]), _t([0.0, 0.0]), _t([0.0]), 1.0,
                   capacity=_t([2.0]), capacity_prev=_t([1.0]))


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal"])
def test_gradients_flow_through_both_capacities(scheme):
    layer = _bare_layer(scheme)

    def f(v_old, v_new):
        return layer.step(_t([1.0]), _t([0.5]), _t([0.0, 0.0]), _t([0.0]), 1.0,
                          capacity=v_new, capacity_prev=v_old)

    v_old = _t([1.0]).requires_grad_(True)
    v_new = _t([1.5]).requires_grad_(True)
    assert torch.autograd.gradcheck(f, (v_old, v_new), eps=1e-6, atol=1e-7)


def test_substeps_interpolate_the_capacity_and_conserve_the_amount():
    """k = 4 substeps from V = 1 to V = 2 with clean inflow: the amount is 1 throughout."""
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("zone", "ambient", kind="flow")
    layer = TransportLayer(net, "c", capacity=_t([1.0]), flow_kind="flow",
                           boundary=["ambient"], scheme="implicit")
    model = Model(net, {"c": layer}, closures=[lambda s, d: {"c.q": _t([-1.0])}],
                  substeps={"c": 4})
    new = model.step({"c.x": _t([1.0]), "c.capacity": _t([1.0])},
                     {"c.x_boundary": _t([0.0]), "c.sources": _t([0.0, 0.0]),
                      "c.capacity": _t([2.0])}, 1.0)
    assert new["c.x"].item() == pytest.approx(0.5, rel=1e-12)
    assert new["c.capacity"].item() == 2.0


def test_a_written_capacity_driver_requires_the_capacity_in_the_start_state():
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("zone", "ambient", kind="flow")
    layer = TransportLayer(net, "c", capacity=_t([1.0]), flow_kind="flow",
                           boundary=["ambient"], scheme="implicit")
    model = Model(net, {"c": layer}, closures=[lambda s, d: {"c.q": _t([0.0]),
                                                             "c.capacity": _t([2.0])}])
    with pytest.raises(KeyError, match="c.capacity.*initial_capacities"):
        model.step({"c.x": _t([1.0])}, {"c.x_boundary": _t([0.0])}, 1.0)
    caps = model.initial_capacities({"c.x": _t([1.0])}, {"c.x_boundary": _t([0.0])})
    assert set(caps) == {"c.capacity"}
    assert torch.equal(caps["c.capacity"], _t([2.0]))
