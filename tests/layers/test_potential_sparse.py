"""Sparse-path tests for PotentialFlowLayer (Task 11): grounding via the per-instance
certificate from solvers.grounding, and linear_init/solve routed through
GraphLaplacianOperator + solvers.select.solve rather than a dense einsum Jacobian.
"""

import pytest
import torch

from tellegen.elements import Conductance, FixedFlow
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
