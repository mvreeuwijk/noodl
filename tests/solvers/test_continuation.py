"""`noodl.solvers.continuation`: the storage ramp, a starting point for a failed Newton."""

import torch

from noodl.operators.graph import GraphLaplacianOperator
from noodl.operators.shifted import ShiftedOperator
from noodl.operators.sparse import SparseOperator
from noodl.solvers.continuation import shift_diagonal, storage_ramp
from noodl.solvers.newton import newton

F64 = torch.float64


def _cubic(x):
    """x^3 - 2x + 2: undamped Newton from x = 1 cycles 1 -> 0 -> 1 for ever."""
    return x**3 - 2.0 * x + 2.0


def _cubic_jacobian(x):
    return torch.diag_embed(3.0 * x**2 - 2.0)


def test_plain_undamped_newton_cycles_from_the_start():
    x0 = torch.tensor([1.0], dtype=F64)
    result = newton(_cubic, _cubic_jacobian, x0, omega=1.0, switch_ratio=0.0, max_iter=50,
                    on_failure="return")
    assert not bool(result.converged)
    assert float(result.x) in (0.0, 1.0)


def test_newton_from_the_storage_ramp_converges_where_plain_newton_cycles():
    x0 = torch.tensor([1.0], dtype=F64)
    report: dict = {}
    start = storage_ramp(_cubic, _cubic_jacobian, x0, omega=1.0, diagnostics=report)
    assert not start.requires_grad
    assert report["newton_iterations"] > 0
    result = newton(_cubic, _cubic_jacobian, start, omega=1.0, switch_ratio=0.0,
                    max_iter=50, atol=1e-13, rtol=0.0)
    assert bool(result.converged)
    root = float(result.x)
    assert abs(root**3 - 2.0 * root + 2.0) < 1e-12
    assert abs(root + 1.7692923542386) < 1e-10


def test_the_storage_is_zero_on_rows_left_out():
    """Two independent rows; the second is left out, so its stages are plain Newton on it:
    a linear row lands on its root in the first stage."""
    def residual(x):
        return torch.stack([x[..., 0] ** 3 - 2.0 * x[..., 0] + 2.0, 2.0 * x[..., 1] - 3.0],
                           dim=-1)

    def jacobian(x):
        return torch.diag_embed(torch.stack(
            [3.0 * x[..., 0] ** 2 - 2.0, torch.full_like(x[..., 1], 2.0)], dim=-1))

    rows = torch.tensor([True, False])
    out = storage_ramp(residual, jacobian, torch.tensor([1.0, 0.0], dtype=F64), rows=rows,
                       scales=(1.0,), omega=1.0, atol=1e-14, rtol=0.0)
    assert float(out[1]) == 1.5
    # the stored row moved toward its own root only part of the way
    assert -1.77 < float(out[0]) < 1.0


def _laplacian(slopes):
    # boundary node 0, interior nodes 1 and 2: 0 - 1 - 2
    src = torch.tensor([0, 1])
    tgt = torch.tensor([1, 2])
    interior_of_node = torch.tensor([-1, 0, 1])
    return GraphLaplacianOperator(src, tgt, slopes, 2, interior_of_node,
                                  boundary_mask=torch.tensor([True, False, False]))


def test_a_shifted_laplacian_keeps_its_spd_certificate():
    op = _laplacian(torch.tensor([1.0, 2.0], dtype=F64))
    g = torch.tensor([0.5, 0.25], dtype=F64)
    shifted = shift_diagonal(op, g)
    assert isinstance(shifted, ShiftedOperator)
    assert bool(torch.as_tensor(shifted.spd_certificate()).all())
    torch.testing.assert_close(shifted.assemble(), op.assemble() + torch.diag(g))


def test_a_shift_on_a_shifted_laplacian_adds_to_its_shift_and_keeps_the_certificate():
    base = _laplacian(torch.tensor([1.0, 2.0], dtype=F64))
    op = ShiftedOperator(base, torch.tensor([0.1, 0.0], dtype=F64))   # a withdrawal's slope
    g = torch.tensor([0.5, 0.25], dtype=F64)
    shifted = shift_diagonal(op, g)
    assert shifted.base is base
    assert bool(torch.as_tensor(shifted.spd_certificate()).all())
    torch.testing.assert_close(shifted.assemble(), op.assemble() + torch.diag(g))


def test_a_shifted_sparse_operator_adds_diagonal_entries():
    op = SparseOperator(torch.tensor([0, 1, 1]), torch.tensor([1, 0, 1]),
                        torch.tensor([[2.0, -1.0, 3.0], [1.0, 1.0, 1.0]], dtype=F64), 2)
    g = torch.tensor([0.5, 0.0], dtype=F64)
    shifted = shift_diagonal(op, g)
    assert isinstance(shifted, SparseOperator)
    torch.testing.assert_close(shifted.assemble(), op.assemble() + torch.diag(g))
    torch.testing.assert_close(shifted.diagonal(), op.diagonal() + g)
