"""Tests for AdvectionOperator: the nonsymmetric transport spatial operator.

Every test compares the operator's ACTION against TransportLayer's existing dense
assembly (the retained dense reference), never against a second copy of the gather/scatter
formulas -- a bug shared between the operator and its own test would otherwise be
invisible.
"""

import pytest
import torch
from torch.autograd import gradcheck

from noodl.layers.transport import TransportLayer
from noodl.operators.advection import AdvectionOperator
from noodl.topology import Network


def _interior_of_node(net: Network, boundary: list) -> torch.Tensor:
    interior_idx = net.interior_index(boundary)
    out = torch.full((net.n,), -1, dtype=torch.long)
    out[interior_idx] = torch.arange(interior_idx.shape[0], dtype=torch.long)
    return out


def flow_through_zone() -> Network:
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("Z")
    net.add_edge("Z", "ambient", kind="airpath")
    net.add_edge("ambient", "Z", kind="airpath")
    return net


def three_node_chain() -> Network:
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("A")
    net.add_node("B")
    net.add_edge("ambient", "A", kind="airpath")
    net.add_edge("A", "B", kind="airpath")
    net.add_edge("B", "ambient", kind="airpath")
    return net


def test_matvec_matches_dense_transport_operator_forward_flow():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, 0.2, 0.25], dtype=torch.float64)
    M, _ = layer.operator(q)

    src, tgt = net.endpoints("airpath")
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=_interior_of_node(net, ["ambient"]),
    )
    x = torch.tensor([12.0, -4.0], dtype=torch.float64)
    torch.testing.assert_close(op.matvec(x), M @ x, rtol=1e-9, atol=1e-12)


def test_matvec_matches_dense_transport_operator_reversed_and_mixed_sign_flow():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    for q in (
        torch.tensor([-0.3, -0.2, -0.25], dtype=torch.float64),
        torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64),
    ):
        M, _ = layer.operator(q)
        src, tgt = net.endpoints("airpath")
        op = AdvectionOperator(
            src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
            capacity=cap, n_interior=layer.n_i,
            interior_of_node=_interior_of_node(net, ["ambient"]),
        )
        x = torch.tensor([7.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(op.matvec(x), M @ x, rtol=1e-9, atol=1e-12)


def test_matvec_batched_matches_looped():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    n = 6
    q = 0.2 + 0.6 * torch.rand(n, 2, dtype=torch.float64) - 0.3
    src, tgt = net.endpoints("airpath")
    interior_of_node = _interior_of_node(net, ["ambient"])
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=interior_of_node,
    )
    x = torch.rand(n, 1, dtype=torch.float64)
    out = op.matvec(x)
    ref = []
    for i in range(n):
        op_i = AdvectionOperator(
            src, tgt, flow=layer.carrier.to(q.dtype) * q[i], transmission=layer.transmission,
            capacity=cap, n_interior=layer.n_i, interior_of_node=interior_of_node,
        )
        ref.append(op_i.matvec(x[i]))
    torch.testing.assert_close(out, torch.stack(ref), rtol=1e-10, atol=1e-12)


def test_rmatvec_is_the_true_transpose_of_matvec():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    src, tgt = net.endpoints("airpath")
    interior_of_node = _interior_of_node(net, ["ambient"])
    torch.manual_seed(0)
    for q_sign in (1.0, -1.0):
        q = q_sign * torch.tensor([0.3, 0.2, 0.25], dtype=torch.float64)
        op = AdvectionOperator(
            src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
            capacity=cap, n_interior=layer.n_i, interior_of_node=interior_of_node,
        )
        x = torch.randn(2, dtype=torch.float64)
        y = torch.randn(2, dtype=torch.float64)
        lhs = (op.matvec(x) * y).sum()
        rhs = (x * op.rmatvec(y)).sum()
        torch.testing.assert_close(lhs, rhs, rtol=1e-9, atol=1e-12)


def test_rmatvec_matches_dense_transpose():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    M, _ = layer.operator(q)
    src, tgt = net.endpoints("airpath")
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=_interior_of_node(net, ["ambient"]),
    )
    y = torch.tensor([5.0, -2.0], dtype=torch.float64)
    torch.testing.assert_close(op.rmatvec(y), M.transpose(-1, -2) @ y, rtol=1e-9, atol=1e-12)


def test_matvec_and_rmatvec_genuinely_differ_on_the_same_vector():
    """Symmetry check (NOT the transpose-correctness check above): matvec(x) and
    rmatvec(x) evaluated at the SAME x, for a problem with nonzero flow. This is a
    DIFFERENT property from the transpose identity `matvec(x).y == x.rmatvec(y)`,
    which holds for any correct rmatvec whether or not the operator is symmetric
    (it holds for a random nonsymmetric matrix too). This test instead asserts the
    operator FAILS `matvec(x) == rmatvec(x)`: since upwinding is direction-dependent
    (matvec gathers at the upwind node and scatters to the downwind node; rmatvec
    swaps that), the two must give genuinely different vectors whenever flow is
    nonzero. A passing (i.e. equal) result here would mean the upwind/downwind swap
    was not actually implemented -- a real bug that the transpose-identity test
    alone cannot see, because a `rmatvec` that is silently just `matvec` again
    would fail the transpose-identity test AND this one together in the general
    case, but could coincidentally satisfy the identity on a single random (x, y)
    pair for a small enough problem; asserting inequality directly closes that gap.
    """
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)  # nonzero flow throughout
    src, tgt = net.endpoints("airpath")
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=_interior_of_node(net, ["ambient"]),
    )
    x = torch.tensor([12.0, -4.0], dtype=torch.float64)
    assert not torch.allclose(op.matvec(x), op.rmatvec(x), rtol=1e-6, atol=1e-8)


def test_matvec_with_conduction_matches_dense():
    net = Network(dtype=torch.float64)
    net.add_node("Tb")
    net.add_node("T1")
    net.add_node("T2")
    net.add_edge("T1", "Tb", kind="conduction")
    net.add_edge("T2", "T1", kind="conduction")
    layer = TransportLayer(
        net, "heat", capacity=torch.tensor([1000.0, 500.0], dtype=torch.float64),
        flow_kind="conduction", boundary=["Tb"], conduction_kind="conduction",
        conductance=torch.tensor([5.0, 3.0], dtype=torch.float64),
    )
    q = torch.zeros(2, dtype=torch.float64)
    M, _ = layer.operator(q)

    csrc, ctgt = net.endpoints("conduction")
    src, tgt = net.endpoints("conduction")  # flow_kind == conduction_kind here
    interior_of_node = _interior_of_node(net, ["Tb"])
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=layer.capacity, n_interior=layer.n_i, interior_of_node=interior_of_node,
        conduction=(csrc, ctgt, torch.tensor([5.0, 3.0], dtype=torch.float64)),
    )
    x = torch.tensor([20.0, -6.0], dtype=torch.float64)
    torch.testing.assert_close(op.matvec(x), M @ x, rtol=1e-9, atol=1e-12)

    y = torch.tensor([3.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(op.rmatvec(y), M.transpose(-1, -2) @ y, rtol=1e-9, atol=1e-12)


def test_matvec_with_kinetics_and_removal_matches_dense_three_species():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    l1, l2 = 0.01, 0.02
    kinetics = torch.zeros(3, 3, dtype=torch.float64)
    kinetics[0, 0] = -l1
    kinetics[1, 0] = l1
    kinetics[1, 1] = -l2
    kinetics[2, 1] = l2
    removal = torch.tensor([0.0, 0.0, 0.005], dtype=torch.float64)
    layer = TransportLayer(
        net, "chain", capacity=cap, flow_kind="airpath", boundary=["ambient"], n_species=3,
        kinetics=kinetics, removal=removal,
    )
    q = torch.tensor([0.4, 0.4], dtype=torch.float64)
    M, _ = layer.operator(q)

    src, tgt = net.endpoints("airpath")
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=_interior_of_node(net, ["ambient"]),
        kinetics=layer.kinetics, removal=layer.removal,
    )
    x = torch.tensor([1.0, 0.3, 0.1], dtype=torch.float64)  # (n_i*K,), species-major
    torch.testing.assert_close(op.matvec(x), M @ x, rtol=1e-9, atol=1e-12)

    y = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float64)
    torch.testing.assert_close(op.rmatvec(y), M.transpose(-1, -2) @ y, rtol=1e-9, atol=1e-12)


def test_boundary_forcing_matches_dense_N_block():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, 0.2, 0.25], dtype=torch.float64)
    _, N = layer.operator(q)

    src, tgt = net.endpoints("airpath")
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=_interior_of_node(net, ["ambient"]),
    )
    x_b = torch.tensor([420.0], dtype=torch.float64)
    torch.testing.assert_close(op.boundary_forcing(x_b), (N @ x_b), rtol=1e-9, atol=1e-12)


def test_diagonal_matches_torch_diagonal_of_assemble():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    src, tgt = net.endpoints("airpath")
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=_interior_of_node(net, ["ambient"]),
    )
    torch.testing.assert_close(
        op.diagonal(), torch.diagonal(op.assemble(), dim1=-2, dim2=-1), rtol=1e-9, atol=1e-12
    )


def test_diagonal_matches_dense_M_diagonal():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    M, _ = layer.operator(q)
    src, tgt = net.endpoints("airpath")
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=_interior_of_node(net, ["ambient"]),
    )
    torch.testing.assert_close(
        op.diagonal(), torch.diagonal(M, dim1=-2, dim2=-1), rtol=1e-9, atol=1e-12
    )


def test_assemble_matches_dense_transport_operator():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    M, _ = layer.operator(q)
    src, tgt = net.endpoints("airpath")
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=_interior_of_node(net, ["ambient"]),
    )
    torch.testing.assert_close(op.assemble(), M, rtol=1e-9, atol=1e-12)


def test_symmetric_is_false_and_spd_certificate_is_none():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    src, tgt = net.endpoints("airpath")
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=_interior_of_node(net, ["ambient"]),
    )
    assert op.symmetric is False
    assert op.spd_certificate() is None
    # n_interior=1 (only "Z" is interior; "ambient" is the sole boundary node) and
    # n_species=1 (default single-species transmission), so m = n_interior * n_species = 1;
    # this is not b_flow (2 edges) -- shape is (m, m), matching M from
    # TransportLayer.operator(q), not the edge count.
    assert op.shape == (1, 1)
    assert op.dtype == torch.float64


def test_diagonal_and_assemble_match_dense_with_kinetics_removal_and_conduction():
    """Coverage fix (fix round 1): every prior test calling .diagonal()/.assemble() used the
    plain three_node_chain() CO2 layer, so the kinetics/removal/conduction branches inside
    those two methods (as opposed to matvec/rmatvec, which already exercised them) were never
    hit. This test combines all three terms -- three-species kinetics + removal (as in
    test_matvec_with_kinetics_and_removal_matches_dense_three_species) AND a conduction_kind
    edge (as in test_matvec_with_conduction_matches_dense) -- on ONE layer, with a BATCHED,
    mixed-sign flow q and a BATCHED conductance so assemble()'s `L.dim() > 2` branch (as well
    as its plain conduction/removal/kinetics branches) is also exercised, not just the
    unbatched path already covered by the conduction-only test above.
    """
    net = flow_through_zone()
    net.add_edge("Z", "ambient", kind="conduction")
    cap = torch.tensor([1000.0], dtype=torch.float64)
    l1, l2 = 0.01, 0.02
    kinetics = torch.zeros(3, 3, dtype=torch.float64)
    kinetics[0, 0] = -l1
    kinetics[1, 0] = l1
    kinetics[1, 1] = -l2
    kinetics[2, 1] = l2
    removal = torch.tensor([0.0, 0.0, 0.005], dtype=torch.float64)
    conductance = torch.tensor([[2.0], [3.0]], dtype=torch.float64)  # (2, b_c=1), batched
    layer = TransportLayer(
        net, "chain", capacity=cap, flow_kind="airpath", boundary=["ambient"], n_species=3,
        kinetics=kinetics, removal=removal,
        conduction_kind="conduction", conductance=conductance,
    )
    q = torch.tensor([[0.4, -0.3], [-0.2, 0.5]], dtype=torch.float64)  # (2, b_flow), mixed sign
    M, _ = layer.operator(q)

    src, tgt = net.endpoints("airpath")
    csrc, ctgt = net.endpoints("conduction")
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(q.dtype) * q, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=_interior_of_node(net, ["ambient"]),
        kinetics=layer.kinetics, removal=layer.removal,
        conduction=(csrc, ctgt, conductance),
    )
    torch.testing.assert_close(op.assemble(), M, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(
        op.diagonal(), torch.diagonal(op.assemble(), dim1=-2, dim2=-1), rtol=1e-9, atol=1e-12
    )


def test_matvec_source_has_no_python_loop_over_edges():
    """Construction-time loops are fine; a per-solve hot path (matvec is called every solver
    iteration) must not loop over edges or nodes. Reads the actual source of matvec's method
    body and asserts no `for` appears in it -- a static, cheap proxy for "vectorised", checked
    here rather than merely asserted in prose because a future edit could silently reintroduce
    a loop otherwise. Mirrors tests/operators/test_graph.py's identical check for
    GraphLaplacianOperator.
    """
    import inspect

    from noodl.operators.advection import AdvectionOperator as A

    source = inspect.getsource(A.matvec)
    assert "for " not in source


def test_rmatvec_source_has_no_python_loop_over_edges():
    """Same check as test_matvec_source_has_no_python_loop_over_edges, but for rmatvec."""
    import inspect

    from noodl.operators.advection import AdvectionOperator as A

    source = inspect.getsource(A.rmatvec)
    assert "for " not in source


def test_gradcheck_matvec_wrt_flow_transmission_capacity_x():
    net = three_node_chain()
    src, tgt = net.endpoints("airpath")
    interior_of_node = _interior_of_node(net, ["ambient"])
    n_i = 2

    flow0 = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64, requires_grad=True)
    transmission0 = torch.tensor([[1.0, 0.6, 0.9]], dtype=torch.float64, requires_grad=True)
    capacity0 = torch.tensor([50.0, 80.0], dtype=torch.float64, requires_grad=True)
    x0 = torch.tensor([12.0, -4.0], dtype=torch.float64, requires_grad=True)

    def f(flow, transmission, capacity, x):
        op = AdvectionOperator(
            src, tgt, flow=flow, transmission=transmission, capacity=capacity,
            n_interior=n_i, interior_of_node=interior_of_node,
        )
        return op.matvec(x)

    assert gradcheck(f, (flow0, transmission0, capacity0, x0), eps=1e-6, atol=1e-5)


def test_gradcheck_rmatvec_wrt_flow_and_y():
    net = three_node_chain()
    src, tgt = net.endpoints("airpath")
    interior_of_node = _interior_of_node(net, ["ambient"])
    n_i = 2
    transmission0 = torch.tensor([[1.0, 0.6, 0.9]], dtype=torch.float64)
    capacity0 = torch.tensor([50.0, 80.0], dtype=torch.float64)

    flow0 = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64, requires_grad=True)
    y0 = torch.tensor([5.0, -2.0], dtype=torch.float64, requires_grad=True)

    def f(flow, y):
        op = AdvectionOperator(
            src, tgt, flow=flow, transmission=transmission0, capacity=capacity0,
            n_interior=n_i, interior_of_node=interior_of_node,
        )
        return op.rmatvec(y)

    assert gradcheck(f, (flow0, y0), eps=1e-6, atol=1e-5)


def test_matvec_broadcasts_an_unbatched_state_against_a_batched_flow():
    """Final-review finding C1: the operator's own batch must broadcast against the state.

    `x` is one initial condition; `flow` is an ensemble of 5 realisations. The result must
    be (5, m) and equal the explicitly-batched call entry for entry.
    """
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    src, tgt = net.endpoints("airpath")
    base = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    flow = base * torch.linspace(0.5, 1.5, 5, dtype=torch.float64).unsqueeze(-1)  # (5, 3)
    op = AdvectionOperator(
        src, tgt, flow=layer.carrier.to(flow.dtype) * flow, transmission=layer.transmission,
        capacity=cap, n_interior=layer.n_i, interior_of_node=_interior_of_node(net, ["ambient"]),
    )
    x = torch.tensor([12.0, -4.0], dtype=torch.float64)
    y = op.matvec(x)
    assert y.shape == (5, 2)
    torch.testing.assert_close(y, op.matvec(x.expand(5, 2)), rtol=1e-12, atol=1e-14)

    xb = torch.tensor([420.0], dtype=torch.float64)
    yb = op.boundary_forcing(xb)
    assert yb.shape == (5, 2)
    torch.testing.assert_close(
        yb, op.boundary_forcing(xb.expand(5, 1)), rtol=1e-12, atol=1e-14
    )

    yt = op.rmatvec(x)
    assert yt.shape == (5, 2)
    torch.testing.assert_close(yt, op.rmatvec(x.expand(5, 2)), rtol=1e-12, atol=1e-14)


# ------------------------------------------------- I2: constructor shape validation
# Global Constraint: "ValueError for bad shapes, naming the offender". Before this wave a
# wrong edge count in `transmission` constructed silently and then raised an opaque
# broadcast RuntimeError at the first matvec, and an `n_interior` inconsistent with
# `interior_of_node` was not detected at all.


def _valid_args():
    net = three_node_chain()
    src, tgt = net.endpoints("airpath")
    return {
        "src": src,
        "tgt": tgt,
        "flow": torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64),
        "transmission": torch.ones(1, 3, dtype=torch.float64),
        "capacity": torch.tensor([50.0, 80.0], dtype=torch.float64),
        "n_interior": 2,
        "interior_of_node": _interior_of_node(net, ["ambient"]),
    }


def test_transmission_edge_count_mismatch_raises_value_error_naming_it():
    args = _valid_args() | {"transmission": torch.ones(1, 2, dtype=torch.float64)}
    with pytest.raises(ValueError, match=r"AdvectionOperator.*transmission"):
        AdvectionOperator(**args)


def test_capacity_length_mismatch_raises_value_error_naming_it():
    args = _valid_args() | {"capacity": torch.ones(3, dtype=torch.float64)}
    with pytest.raises(ValueError, match=r"AdvectionOperator.*capacity"):
        AdvectionOperator(**args)


def test_n_interior_inconsistent_with_interior_of_node_raises_value_error():
    args = _valid_args() | {"n_interior": 3, "capacity": torch.ones(3, dtype=torch.float64)}
    with pytest.raises(ValueError, match=r"AdvectionOperator.*n_interior"):
        AdvectionOperator(**args)


def test_src_and_tgt_shape_mismatch_raises_value_error_naming_them():
    args = _valid_args() | {"tgt": torch.tensor([0, 1])}
    with pytest.raises(ValueError, match=r"AdvectionOperator.*src.*tgt"):
        AdvectionOperator(**args)


def test_kinetics_trailing_shape_mismatch_raises_value_error_naming_it():
    args = _valid_args() | {"kinetics": torch.zeros(3, 1, 1, dtype=torch.float64)}
    with pytest.raises(ValueError, match=r"AdvectionOperator.*kinetics"):
        AdvectionOperator(**args)


def test_removal_trailing_shape_mismatch_raises_value_error_naming_it():
    args = _valid_args() | {"removal": torch.zeros(3, 1, dtype=torch.float64)}
    with pytest.raises(ValueError, match=r"AdvectionOperator.*removal"):
        AdvectionOperator(**args)
