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


R5 = pytest.mark.xfail(
    strict=True,
    reason=(
        "R5: the step keeps the old concentration as its initial condition "
        "while the volume changes"
    ),
)


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal"])
@pytest.mark.parametrize(
    "q, v_old, v_new, expected",
    [
        pytest.param(-1.0, 1.0, 2.0, 0.5, id="clean-filling"),
        pytest.param(1.0, 2.0, 1.0, 1.0, id="draining"),
        pytest.param(0.0, 1.0, 2.0, 0.5, id="prescribed-dilution"),
    ],
)
@R5
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
