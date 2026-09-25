"""Tests for GraphLaplacianOperator: A_I diag(g) A_I^T represented matvec-free.

Fixture used throughout ("the chain fixture"): 3 nodes, node 2 is the boundary (grounded)
node, nodes 0 and 1 are interior; edges (0,1) and (1,2). Slopes are batched (2, 2):
instance 0 = [1, 1] (grounded through both edges), instance 1 = [0, 1] (edge (0,1) has zero
slope, so {0, 1} has no path to the boundary through strictly positive slope). This is the
exact counterexample design section 3.1 cites for `_floating_group_nodes`'s defect, and its
minimum eigenvalues (0.381966..., 0.0) are re-derived by hand below and checked against
`torch.linalg.eigvalsh` directly, independently of Task 3's own tests of spd_certificate.
"""

import pytest
import torch

from noodl.operators.dense import DenseOperator
from noodl.operators.graph import GraphLaplacianOperator


@pytest.fixture(autouse=True)
def _set_float64_dtype():
    """Set default dtype to float64 for this module's tests, then restore."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old_dtype)


def _chain_op(slopes: torch.Tensor) -> GraphLaplacianOperator:
    src = torch.tensor([0, 1])
    tgt = torch.tensor([1, 2])
    interior_of_node = torch.tensor([0, 1, -1])  # node 2's compact index is -1 (boundary)
    boundary_mask = torch.tensor([False, False, True])
    return GraphLaplacianOperator(
        src, tgt, slopes, 2, interior_of_node, boundary_mask=boundary_mask
    )


def test_assemble_matches_hand_derived_chain_fixture():
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    op = _chain_op(slopes)
    A = op.assemble()
    expect0 = torch.tensor([[1.0, -1.0], [-1.0, 2.0]])
    expect1 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    torch.testing.assert_close(A[0], expect0)
    torch.testing.assert_close(A[1], expect1)


def test_matvec_matches_dense_reference_on_random_batched_inputs():
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    op = _chain_op(slopes)
    dense = DenseOperator(op.assemble(), symmetric=True)
    torch.manual_seed(0)
    x = torch.randn(5, 2, 2)  # extra leading batch dims broadcast against the (2, 2) op batch
    torch.testing.assert_close(op.matvec(x), dense.matvec(x), rtol=1e-9, atol=1e-12)


def test_transpose_correctness_matvec_dot_y_equals_x_dot_rmatvec():
    """matvec(x).y == x.rmatvec(y) for DIFFERENT random x, y: the definition of a transpose.

    This holds for ANY correctly implemented rmatvec, symmetric or not -- it is not a test of
    symmetry (see test_symmetry_matvec_equals_rmatvec_on_same_x below for that), it is a test
    that rmatvec actually computes the adjoint of matvec and would equally catch a wrong
    rmatvec on a nonsymmetric operator.
    """
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    op = _chain_op(slopes)
    torch.manual_seed(1)
    x = torch.randn(4, 2, 2)
    y = torch.randn(4, 2, 2)
    lhs = (op.matvec(x) * y).sum(-1)
    rhs = (x * op.rmatvec(y)).sum(-1)
    torch.testing.assert_close(lhs, rhs, rtol=1e-9, atol=1e-12)


def test_symmetry_matvec_equals_rmatvec_on_same_x():
    """matvec(x) == rmatvec(x) on the SAME x: what actually discriminates a symmetric
    operator from a nonsymmetric one, and what method="auto" eligibility (Task 7) relies on
    when it trusts `symmetric = True`. A random nonsymmetric matrix fails this identity; this
    operator, being A_I diag(g) A_I^T, must pass it.
    """
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    op = _chain_op(slopes)
    torch.manual_seed(2)
    x = torch.randn(4, 2, 2)
    torch.testing.assert_close(op.matvec(x), op.rmatvec(x), rtol=1e-9, atol=1e-12)


def test_rmatvec_is_a_distinct_method_from_matvec():
    """rmatvec must be its OWN implementation, tested to agree -- never `rmatvec = matvec`."""
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    op = _chain_op(slopes)
    assert op.matvec.__func__ is not op.rmatvec.__func__


def test_diagonal_matches_assembled_diagonal():
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    op = _chain_op(slopes)
    diag = op.diagonal()
    diag_ref = torch.diagonal(op.assemble(), dim1=-2, dim2=-1)
    torch.testing.assert_close(diag, diag_ref, rtol=1e-9, atol=1e-12)


def test_positive_semidefinite_for_nonnegative_slopes():
    torch.manual_seed(3)
    slopes = torch.rand(6, 2)  # non-negative by construction (torch.rand is in [0, 1))
    op = _chain_op(slopes)
    x = torch.randn(6, 2)
    quad = (x * op.matvec(x)).sum(-1)
    assert bool((quad >= -1e-10).all())


def test_spd_certificate_matches_eigvalsh_positivity():
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    op = _chain_op(slopes)
    A = op.assemble()
    eig_min = torch.linalg.eigvalsh(A).amin(dim=-1)
    torch.testing.assert_close(eig_min, torch.tensor([0.38196601125, 0.0]), atol=1e-8, rtol=0)
    cert = op.spd_certificate()
    expect = eig_min > 0
    assert torch.equal(cert, expect)
    assert cert.tolist() == [True, False]


def test_batching_matches_looped():
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    op = _chain_op(slopes)
    torch.manual_seed(4)
    x = torch.randn(3, 2, 2)
    mv_batched = op.matvec(x)
    for i in range(2):
        op_i = _chain_op(slopes[i])
        torch.testing.assert_close(
            op_i.matvec(x[:, i, :]), mv_batched[:, i, :], rtol=1e-9, atol=1e-12
        )


def test_gradcheck_matvec_wrt_slopes_and_x():
    src = torch.tensor([0, 1])
    tgt = torch.tensor([1, 2])
    interior_of_node = torch.tensor([0, 1, -1])
    boundary_mask = torch.tensor([False, False, True])
    slopes = torch.tensor([1.3, 0.7], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([0.5, -0.2], dtype=torch.float64, requires_grad=True)

    def f(s, xx):
        op = GraphLaplacianOperator(src, tgt, s, 2, interior_of_node, boundary_mask=boundary_mask)
        return op.matvec(xx)

    assert torch.autograd.gradcheck(f, (slopes, x), eps=1e-6, atol=1e-6)


def test_matvec_source_has_no_python_loop_over_edges():
    """Construction-time loops are fine; a per-solve hot path (matvec is called every solver
    iteration) must not loop over edges or nodes. Reads the actual source of matvec's method
    body and asserts no `for` appears in it -- a static, cheap proxy for "vectorised", checked
    here rather than merely asserted in prose because a future edit could silently reintroduce
    a loop otherwise.
    """
    import inspect

    from noodl.operators.graph import GraphLaplacianOperator as G

    source = inspect.getsource(G.matvec)
    assert "for " not in source


def test_rmatvec_source_has_no_python_loop_over_edges():
    """Same check as test_matvec_source_has_no_python_loop_over_edges, but for rmatvec.

    A8 amendment: rmatvec must also be vectorised (no Python loop over edges).
    """
    import inspect

    from noodl.operators.graph import GraphLaplacianOperator as G

    source = inspect.getsource(G.rmatvec)
    assert "for " not in source


def test_apply_source_has_no_python_loop_over_edges():
    """`matvec`/`rmatvec` are now one-line delegators to `_apply` (the consolidation this
    review flagged): checking only their own source, as the two tests above do, would never
    catch a loop introduced INSIDE `_apply` itself, where all the actual arithmetic now
    lives. Checked for both `for` and `while` loops.
    """
    import inspect

    from noodl.operators.graph import GraphLaplacianOperator as G

    source = inspect.getsource(G._apply)
    assert "for " not in source
    assert "while " not in source


def test_spd_diagnosis_on_ungrounded_chain_fixture():
    """A2 amendment: add one test on the chain fixture asserting `op.spd_diagnosis()`
    returns exactly one record for instance 1 with `reason == "ungrounded"`.
    """
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    op = _chain_op(slopes)
    diagnosis = op.spd_diagnosis()
    assert len(diagnosis) == 1
    assert diagnosis[0]["instance"] == 1
    assert diagnosis[0]["reason"] == "ungrounded"
    # In instance 1: edge 0 (0,1) has slope 0.0 (inactive), edge 1 (1,2) has slope 1.0 (active).
    # Node 1 reaches boundary through edge 1, but node 0 cannot (edge 0 is inactive).
    # So only node 0 is ungrounded.
    assert diagnosis[0]["nodes"] == [0]
    assert diagnosis[0]["edges"] == []


def _chain_op_with_negative_parallel_edge() -> GraphLaplacianOperator:
    """The chain fixture plus a parallel 0--1 edge whose slope is -5: every interior node
    is still grounded through strictly positive slopes, so the pre-I1 certificate (which
    tested grounding only) said True while the operator is in fact indefinite."""
    src = torch.tensor([0, 1, 0])
    tgt = torch.tensor([1, 2, 1])
    interior_of_node = torch.tensor([0, 1, -1])
    boundary_mask = torch.tensor([False, False, True])
    slopes = torch.tensor([1.0, 1.0, -5.0])
    return GraphLaplacianOperator(
        src, tgt, slopes, 2, interior_of_node, boundary_mask=boundary_mask
    )


def test_a_grounded_but_negative_slope_operator_is_genuinely_indefinite():
    """WHY the certificate must refuse a negative slope even when grounding holds: the
    assembled operator has a negative eigenvalue, so it is not SPD and CG has no business
    being dispatched to it.
    """
    op = _chain_op_with_negative_parallel_edge()
    eig = torch.linalg.eigvalsh(op.assemble())
    assert bool((eig < 0).any()), eig
    assert not bool(op.spd_certificate())


# ------------------------------------------------- I2: constructor shape validation
# Global Constraint: "ValueError for bad shapes, naming the offender". The final review
# found the last of these four produced not an opaque torch error but a SILENT WRONG
# ANSWER: `interior_nodes = torch.empty(n_interior)` leaves an uninitialised slot that is
# then used as a gather index, so the operator constructed, reported shape (3, 3), and
# `matvec` returned all zeros.


def test_src_and_tgt_shape_mismatch_raises_value_error_naming_them():
    with pytest.raises(ValueError, match=r"GraphLaplacianOperator.*src.*tgt"):
        GraphLaplacianOperator(
            torch.tensor([0, 1, 0]),
            torch.tensor([1, 2]),
            torch.tensor([1.0, 1.0]),
            2,
            torch.tensor([0, 1, -1]),
            boundary_mask=torch.tensor([False, False, True]),
        )


def test_slopes_edge_count_mismatch_raises_value_error_naming_slopes():
    with pytest.raises(ValueError, match=r"GraphLaplacianOperator.*slopes"):
        GraphLaplacianOperator(
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            torch.tensor([1.0, 1.0, 1.0]),
            2,
            torch.tensor([0, 1, -1]),
            boundary_mask=torch.tensor([False, False, True]),
        )


def test_interior_of_node_and_boundary_mask_length_mismatch_raises_value_error():
    with pytest.raises(ValueError, match=r"GraphLaplacianOperator.*interior_of_node"):
        GraphLaplacianOperator(
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            torch.tensor([1.0, 1.0]),
            2,
            torch.tensor([0, 1, -1, -1]),
            boundary_mask=torch.tensor([False, False, True]),
        )


def test_n_interior_inconsistent_with_boundary_mask_raises_instead_of_zeroing():
    # The silent-wrong-answer case: 3 interior rows claimed, 2 actually unmasked.
    with pytest.raises(ValueError, match=r"GraphLaplacianOperator.*n_interior"):
        GraphLaplacianOperator(
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            torch.tensor([1.0, 1.0]),
            3,
            torch.tensor([0, 1, -1]),
            boundary_mask=torch.tensor([False, False, True]),
        )
