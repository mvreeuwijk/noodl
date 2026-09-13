"""Sparse-path tests for PotentialFlowLayer (Task 11): grounding via the per-instance
certificate from solvers.grounding, and linear_init/solve routed through
GraphLaplacianOperator + solvers.select.solve rather than a dense einsum Jacobian.
"""

import pytest
import torch

from tellegen.elements import Conductance, FixedFlow
from tellegen.elements.fan import FanCurve
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.topology import Network


def _three_node_chain() -> Network:
    """ambient (boundary) -- z1 -- z2, both edges kind "conduction"."""
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z1")
    net.add_node("z2")
    net.add_edge("ambient", "z1", kind="conduction")
    net.add_edge("z1", "z2", kind="conduction")
    return net


def test_batched_counterexample_raises_naming_instance_1_not_empty_list():
    # The verified defect (spec section 3.1): per-instance slopes [[1, 1], [0, 1]] on a
    # grounded 3-node chain. Instance 0 (slopes [1, 1]) is fully grounded (min eigenvalue
    # 0.382); instance 1 (slopes [0, 1]) has its ambient-z1 edge closed (slope 0), so {z1,
    # z2} floats as a group (min eigenvalue 0.0) even though z2 alone still "looks" fine
    # from its own nonzero z1-z2 edge. The OLD _floating_group_nodes ORs "is this edge
    # nonzero" across the WHOLE batch before checking connectivity, so it sees the edge as
    # present for BOTH instances (nonzero somewhere in the batch) and returns [] -- passing
    # a genuinely singular instance. spd_certificate is per-instance and must catch this.
    #
    # Naming instance 1 is not on its own enough to distinguish the fix from the defect:
    # before the fix the call DID raise here, but only downstream, from
    # solvers.linear.solve's own singular-system retry ("singular system at batch indices
    # [1]") -- after a dense (n_I, n_I) Jacobian had been assembled and factorised, with no
    # statement of WHY instance 1 is singular and no check at all on any path that does not
    # end in a dense factorisation. The assertions below therefore pin the message to the
    # grounding check itself and to the per-instance diagnosis (amendment A2) it must carry.
    net = _three_node_chain()
    g = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    layer = PotentialFlowLayer(
        net, "chain", [Conductance(g, kind="conduction")], boundary=["ambient"]
    )
    phi_b = torch.zeros(2, 1, dtype=torch.float64)

    with pytest.raises(RuntimeError, match=r"\[1\]") as excinfo:
        layer.linear_init(phi_b, {}, None)

    message = str(excinfo.value)
    assert "linear_init" in message
    assert "ungrounded interior nodes" in message
    assert "z1" in message and "z2" in message
    assert "singular system" not in message


def test_unbatched_floating_group_error_still_names_nodes():
    # Companion to the above: the existing (unbatched) tests in test_potential.py must keep
    # naming actual node identifiers, not a batch index -- this is the ndim==0 certificate
    # branch, checked directly here rather than only implicitly via the unchanged old tests.
    net = Network(dtype=torch.float64)
    for name in ("ambient", "z0", "f1", "f2"):
        net.add_node(name)
    net.add_edge("ambient", "z0", kind="conduction")
    net.add_edge("f1", "f2", kind="conduction")
    net.add_edge("z0", "f1", kind="duct")

    layer = PotentialFlowLayer(
        net,
        "grp",
        [
            Conductance(torch.tensor([1.0, 1.0], dtype=torch.float64), kind="conduction"),
            FixedFlow(torch.tensor([0.1], dtype=torch.float64), kind="duct"),
        ],
        boundary=["ambient"],
    )
    phi_b = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(RuntimeError, match=r"f1.*f2|f2.*f1"):
        layer.linear_init(phi_b, {}, None)


def _shutoff_fan_layer() -> tuple[PotentialFlowLayer, torch.Tensor, torch.Tensor]:
    """ambient (boundary) -- z, one FanCurve edge, plus a source that pulls 0.2 through it.

    P(q) = 100 - 200 q on 0 <= q <= 1: shutoff pressure 100 at q = 0, -100 at q = q_max, so
    the fan is on the smooth part of its curve at dp = 0 (slope -1/P'(q) = 0.005 > 0) and the
    whole network is grounded there. Past its shutoff point (back-pressure -dp > 100, i.e.
    phi_z > 100) FanCurve.dflow is EXACTLY zero -- a physical fan that is shut contributes no
    slope at all -- so at such a point the only edge tying z to the boundary is inactive and
    the operator is genuinely singular.
    """
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("ambient", "z", kind="fan")
    element = FanCurve(
        torch.tensor([100.0, -200.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        kind="fan",
    )
    layer = PotentialFlowLayer(net, "fan", [element], boundary=["ambient"])
    phi_b = torch.zeros(1, dtype=torch.float64)
    sources = torch.tensor([0.0, -0.2], dtype=torch.float64)
    return layer, phi_b, sources


def test_grounding_is_checked_on_actual_slopes_even_when_phi0_is_supplied():
    # The pre-Task-11 code checked grounding only inside linear_init, on linear_init's own
    # tangent-at-zero slopes, and skipped it entirely whenever a caller supplied phi0 (`if
    # phi0 is None: phi0 = self.linear_init(...)` never runs, and with it neither does its
    # check). FanCurve is what makes the two distinguishable: its slope is dp-dependent AND
    # exactly zero past shutoff, so the SAME network is grounded at its linear_init point and
    # ungrounded at a supplied phi0 beyond the fan's shutoff pressure.
    layer, phi_b, sources = _shutoff_fan_layer()

    # Non-vacuity: from its own linear_init the network is grounded and the solve succeeds.
    phi, q = layer.solve(phi_b, {}, sources, differentiable=False)
    torch.testing.assert_close(q, torch.tensor([0.2], dtype=torch.float64), atol=1e-9, rtol=0)

    phi0_supplied = torch.tensor([200.0], dtype=torch.float64)  # past shutoff: dflow == 0
    with pytest.raises(RuntimeError, match=r"solve: floating nodes"):
        layer.solve(phi_b, {}, sources, phi0=phi0_supplied, differentiable=False)


def test_grounding_at_a_supplied_phi0_is_checked_on_the_differentiable_path_too():
    # Same check, same place: solve() runs it once, before the differentiable/
    # non-differentiable branch, so neither path can be the one that skips it.
    layer, phi_b, sources = _shutoff_fan_layer()
    phi0_supplied = torch.tensor([200.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match=r"solve: floating nodes"):
        layer.solve(phi_b, {}, sources, phi0=phi0_supplied, differentiable=True)
