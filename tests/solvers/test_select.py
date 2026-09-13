"""Tests for solvers.select.solve: the method="auto" eligibility table (design section 3.1)
and the raise/return failure boundary (design section 3.2).

Uses a small test-only operator double, _FakeOperator, rather than DenseOperator or
GraphLaplacianOperator: these tests are about select.solve's OWN branching on whatever
spd_certificate() and symmetric happen to say, not about whether a real operator computes
its certificate correctly (Tasks 3 and 4 already cover that).
"""

import pytest
import torch

from tellegen.operators.base import SolverStatus
from tellegen.operators.graph import GraphLaplacianOperator
from tellegen.solvers.select import solve


@pytest.fixture(autouse=True)
def _set_float64_dtype():
    """Set default dtype to float64 for this module's tests, then restore -- a bare
    module-level `torch.set_default_dtype` would leak into every test module collected
    afterwards, since PyTorch's default dtype is process-global state.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old_dtype)


class _FakeOperator:
    def __init__(self, A, *, symmetric, certificate):
        self.A = A
        self.symmetric = symmetric
        self._certificate = certificate
        self.shape = A.shape
        self.dtype = A.dtype
        self.device = A.device

    def matvec(self, x):
        return torch.einsum("...ij,...j->...i", self.A, x)

    def rmatvec(self, x):
        return torch.einsum("...ji,...j->...i", self.A, x)

    def diagonal(self):
        return torch.diagonal(self.A, dim1=-2, dim2=-1)

    def assemble(self):
        return self.A

    def spd_certificate(self):
        return self._certificate


_A_SPD = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
_B_SPD = torch.tensor([1.0, 2.0], dtype=torch.float64)
_A_NS = torch.tensor([[3.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
_B_NS = torch.tensor([1.0, 2.0], dtype=torch.float64)


def _chain_op(slopes: torch.Tensor) -> GraphLaplacianOperator:
    """The chain fixture used throughout the milestone: 3 nodes, node 2 is the boundary
    (grounded) node, nodes 0 and 1 are interior; edges (0,1) and (1,2). slopes = [[1,1],[0,1]]
    certifies instance 0 (grounded through both edges) but not instance 1 (edge (0,1) has
    zero slope, so {0, 1} has no path to the boundary through strictly positive slope).
    """
    src = torch.tensor([0, 1])
    tgt = torch.tensor([1, 2])
    interior_of_node = torch.tensor([0, 1, -1])
    boundary_mask = torch.tensor([False, False, True])
    return GraphLaplacianOperator(
        src, tgt, slopes, 2, interior_of_node, boundary_mask=boundary_mask
    )


def _spy(monkeypatch, target, name):
    """Wrap tellegen.solvers.select.<name> to count calls while still delegating to the
    real implementation, so the eligibility mechanism is verified by WHICH solver actually
    ran (not merely by inspecting internal state) without losing correctness checking.
    """
    calls = {"count": 0}
    real = getattr(target, name)

    def wrapper(*args, **kwargs):
        calls["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, wrapper)
    return calls


def test_all_instances_certify_selects_pcg(monkeypatch):
    import tellegen.solvers.select as select_module

    pcg_calls = _spy(monkeypatch, select_module, "pcg")
    gmres_calls = _spy(monkeypatch, select_module, "gmres")
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    result = solve(op, _B_SPD)
    assert pcg_calls["count"] == 1
    assert gmres_calls["count"] == 0
    x_ref = torch.linalg.solve(_A_SPD, _B_SPD)
    torch.testing.assert_close(result.x, x_ref, atol=1e-8, rtol=1e-8)


def test_symmetric_but_cannot_certify_selects_gmres(monkeypatch):
    import tellegen.solvers.select as select_module

    pcg_calls = _spy(monkeypatch, select_module, "pcg")
    gmres_calls = _spy(monkeypatch, select_module, "gmres")
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=None)
    result = solve(op, _B_SPD)
    assert gmres_calls["count"] == 1
    assert pcg_calls["count"] == 0
    x_ref = torch.linalg.solve(_A_SPD, _B_SPD)
    torch.testing.assert_close(result.x, x_ref, atol=1e-6, rtol=1e-6)


def test_nonsymmetric_selects_gmres_and_never_uses_rmatvec_for_the_forward_solve(monkeypatch):
    import tellegen.solvers.select as select_module

    pcg_calls = _spy(monkeypatch, select_module, "pcg")
    gmres_calls = _spy(monkeypatch, select_module, "gmres")
    op = _FakeOperator(_A_NS, symmetric=False, certificate=None)
    result = solve(op, _B_NS)
    assert gmres_calls["count"] == 1
    assert pcg_calls["count"] == 0
    x_ref = torch.linalg.solve(_A_NS, _B_NS)
    torch.testing.assert_close(result.x, x_ref, atol=1e-6, rtol=1e-6)


def test_none_certify_selects_gmres_not_a_raise(monkeypatch):
    # Uniformly ineligible (certificate all False, not a mix) is not "some but not all": there
    # is no subset to split off, so this routes to gmres exactly like certificate=None does.
    import tellegen.solvers.select as select_module

    pcg_calls = _spy(monkeypatch, select_module, "pcg")
    gmres_calls = _spy(monkeypatch, select_module, "gmres")
    op = _FakeOperator(_A_NS, symmetric=False, certificate=torch.tensor(False))
    solve(op, _B_NS)
    assert gmres_calls["count"] == 1
    assert pcg_calls["count"] == 0


def test_mixed_certification_raises_naming_the_non_certifying_instances():
    A_batch = torch.stack([_A_SPD, _A_SPD])
    b_batch = torch.stack([_B_SPD, _B_SPD])
    op = _FakeOperator(A_batch, symmetric=True, certificate=torch.tensor([True, False]))
    with pytest_raises_containing("[1]"):
        solve(op, b_batch)


def test_mixed_certification_does_not_split_the_batch(monkeypatch):
    # Confirm the refusal happens BEFORE either solver runs -- neither pcg nor gmres is
    # ever called on a mixed-certification batch, since silently splitting it is exactly
    # what design section 3.1 says would hide a modelling error.
    import tellegen.solvers.select as select_module

    pcg_calls = _spy(monkeypatch, select_module, "pcg")
    gmres_calls = _spy(monkeypatch, select_module, "gmres")
    A_batch = torch.stack([_A_SPD, _A_SPD])
    b_batch = torch.stack([_B_SPD, _B_SPD])
    op = _FakeOperator(A_batch, symmetric=True, certificate=torch.tensor([True, False]))
    try:
        solve(op, b_batch)
    except RuntimeError:
        pass
    assert pcg_calls["count"] == 0
    assert gmres_calls["count"] == 0


def test_explicit_cg_on_noncertifying_operator_raises():
    op = _FakeOperator(_A_NS, symmetric=False, certificate=torch.tensor(False))
    with pytest_raises_containing("cg"):
        solve(op, _B_NS, method="cg")


def test_explicit_cg_on_operator_with_no_certificate_at_all_raises():
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=None)
    with pytest_raises_containing("cg"):
        solve(op, _B_SPD, method="cg")


def test_explicit_cg_on_fully_certifying_operator_succeeds():
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    result = solve(op, _B_SPD, method="cg")
    x_ref = torch.linalg.solve(_A_SPD, _B_SPD)
    torch.testing.assert_close(result.x, x_ref, atol=1e-8, rtol=1e-8)


def test_on_failure_return_does_not_raise_and_carries_failing_status():
    # A is exactly rank 1; b has a component outside A's range, so no method converges.
    A_sing = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    b_sing = torch.tensor([1.0, 3.0])
    op = _FakeOperator(A_sing, symmetric=False, certificate=None)
    result = solve(op, b_sing, on_failure="return", max_iter=5)  # must not raise
    assert not bool(result.converged)
    assert int(result.status) != int(SolverStatus.CONVERGED)


def test_on_failure_raise_is_the_default_and_does_raise():
    A_sing = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    b_sing = torch.tensor([1.0, 3.0])
    op = _FakeOperator(A_sing, symmetric=False, certificate=None)
    with pytest_raises_containing(""):
        solve(op, b_sing, max_iter=5)  # on_failure="raise" by default


def test_error_message_names_instances_status_and_residual():
    # select.solve raises via SolveResult.raise_on_failure (Task 1), which is where the
    # naming of instances/status/residual actually happens per design section 3.2 -- this
    # test exercises that content THROUGH select.solve's default on_failure="raise" path,
    # rather than re-implementing the naming here. If this assertion ever fails, the fix
    # belongs in Task 1's raise_on_failure, not in this file's solve().
    A_sing = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    b_sing = torch.tensor([1.0, 3.0])
    op = _FakeOperator(A_sing, symmetric=False, certificate=None)
    try:
        solve(op, b_sing, max_iter=5)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        message = str(exc).lower()
        assert "0" in message  # the single (flat) failing batch index
        assert "residual" in message or "status" in message


# -- A2: spd_diagnosis wired into select.solve's refusal messages --------------------------


def test_mixed_certification_on_graph_laplacian_names_ungrounded_interior_nodes():
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    op = _chain_op(slopes)
    b = torch.zeros(2, 2)
    with pytest_raises_containing("ungrounded interior nodes"):
        solve(op, b)
    with pytest_raises_containing("instance 1"):
        solve(op, b)


def test_spd_diagnosis_is_never_called_on_the_success_path(monkeypatch):
    calls = _spy(monkeypatch, GraphLaplacianOperator, "spd_diagnosis")
    slopes = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # both instances certify
    op = _chain_op(slopes)
    b = torch.ones(2, 2)
    result = solve(op, b)
    assert bool(torch.all(result.converged))
    assert calls["count"] == 0


# -- A3.1: method="direct" -------------------------------------------------------------------


def test_direct_matches_torch_linalg_solve_on_spd_batch():
    A_batch = torch.stack([_A_SPD, _A_SPD])
    b_batch = torch.stack([_B_SPD, _B_SPD])
    op = _FakeOperator(A_batch, symmetric=True, certificate=torch.tensor(False))
    result = solve(op, b_batch, method="direct")
    x_ref = torch.linalg.solve(A_batch, b_batch)
    torch.testing.assert_close(result.x, x_ref, atol=1e-8, rtol=1e-8)
    assert bool(torch.all(result.converged))
    # SolveResult's contract and both iterative backends use int64 throughout; _direct's
    # `info` from lu_factor_ex is int32, so iterations/status must be cast, not inherited.
    assert result.iterations.dtype == torch.int64
    assert result.status.dtype == torch.int64


def test_direct_matches_torch_linalg_solve_on_nonsymmetric_system():
    op = _FakeOperator(_A_NS, symmetric=False, certificate=None)
    result = solve(op, _B_NS, method="direct")
    x_ref = torch.linalg.solve(_A_NS, _B_NS)
    torch.testing.assert_close(result.x, x_ref, atol=1e-8, rtol=1e-8)


def test_explicit_gmres_is_honoured_unconditionally_even_on_a_certifying_operator(monkeypatch):
    # method="gmres" makes no SPD assumption to violate, so it never even asks the operator
    # to certify -- unlike method="cg", it is honoured regardless of what spd_certificate says.
    # An SPD operator gives the same numeric answer through either backend, so the assertion
    # that matters is WHICH backend actually ran, not just the result value.
    import tellegen.solvers.select as select_module

    pcg_calls = _spy(monkeypatch, select_module, "pcg")
    gmres_calls = _spy(monkeypatch, select_module, "gmres")
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    result = solve(op, _B_SPD, method="gmres")
    assert gmres_calls["count"] == 1
    assert pcg_calls["count"] == 0
    x_ref = torch.linalg.solve(_A_SPD, _B_SPD)
    torch.testing.assert_close(result.x, x_ref, atol=1e-6, rtol=1e-6)


def test_direct_reports_per_instance_singular_status_without_raising():
    A_batch = torch.stack([torch.tensor([[1.0, 2.0], [2.0, 4.0]]), _A_SPD])
    b_batch = torch.stack([torch.tensor([1.0, 3.0]), _B_SPD])
    op = _FakeOperator(A_batch, symmetric=False, certificate=None)
    result = solve(op, b_batch, method="direct", on_failure="return")
    assert result.status[0] == int(SolverStatus.SINGULAR)
    assert result.status[1] == int(SolverStatus.CONVERGED)
    assert bool(result.converged[0]) is False
    assert bool(result.converged[1]) is True


def test_direct_on_failure_raise_names_the_singular_instance():
    A_batch = torch.stack([torch.tensor([[1.0, 2.0], [2.0, 4.0]]), _A_SPD])
    b_batch = torch.stack([torch.tensor([1.0, 3.0]), _B_SPD])
    op = _FakeOperator(A_batch, symmetric=False, certificate=None)
    with pytest_raises_containing("0"):
        solve(op, b_batch, method="direct")


def test_direct_ignores_certificate_and_never_calls_spd_diagnosis():
    # certificate=False would refuse method="cg" and (for a mixed batch) method="auto", but
    # "direct" needs no SPD-ness at all: it must solve successfully and never even ask.
    calls = {"count": 0}
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(False))
    op.spd_diagnosis = lambda: calls.__setitem__("count", calls["count"] + 1) or []
    result = solve(op, _B_SPD, method="direct")
    x_ref = torch.linalg.solve(_A_SPD, _B_SPD)
    torch.testing.assert_close(result.x, x_ref, atol=1e-8, rtol=1e-8)
    assert calls["count"] == 0


def test_direct_raises_valueerror_when_assemble_returns_none():
    class _NoAssemble(_FakeOperator):
        def assemble(self):
            return None

    op = _NoAssemble(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    try:
        solve(op, _B_SPD, method="direct")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "assemble" in str(exc)


def test_direct_is_not_selected_by_auto(monkeypatch):
    import tellegen.solvers.select as select_module

    pcg_calls = _spy(monkeypatch, select_module, "pcg")
    gmres_calls = _spy(monkeypatch, select_module, "gmres")
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    solve(op, _B_SPD)  # method="auto"
    assert pcg_calls["count"] == 1
    assert gmres_calls["count"] == 0


# -- A3.2: explicit kwarg forwarding and on_failure applying only to numerical failure --------


def test_x0_is_forwarded_to_pcg_and_gmres():
    # x0 is common to both backends. Starting from the exact solution should converge in
    # zero iterations for either -- if solve() dropped x0 instead of forwarding it, the
    # solver would start from zero and take at least one iteration.
    x_star_spd = torch.linalg.solve(_A_SPD, _B_SPD)
    op_spd = _FakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    result_spd = solve(op_spd, _B_SPD, method="auto", x0=x_star_spd)  # routes to pcg
    assert bool(torch.all(result_spd.iterations == 0))

    x_star_ns = torch.linalg.solve(_A_NS, _B_NS)
    op_ns = _FakeOperator(_A_NS, symmetric=False, certificate=None)
    result_ns = solve(op_ns, _B_NS, method="auto", x0=x_star_ns)  # routes to gmres
    assert bool(torch.all(result_ns.iterations == 0))


def test_solve_forwards_preconditioner_to_pcg_only():
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    result = solve(op, _B_SPD, method="cg", preconditioner=None)
    x_ref = torch.linalg.solve(_A_SPD, _B_SPD)
    torch.testing.assert_close(result.x, x_ref, atol=1e-8, rtol=1e-8)


def test_solve_drops_preconditioner_kwarg_when_gmres_is_selected():
    # A reviewer confirmed solve(op_nonsym, b, preconditioner="jacobi") used to raise TypeError
    # from gmres, which does not accept that kwarg. solve must silently drop it instead.
    op = _FakeOperator(_A_NS, symmetric=False, certificate=None)
    result = solve(op, _B_NS, preconditioner="jacobi")
    x_ref = torch.linalg.solve(_A_NS, _B_NS)
    torch.testing.assert_close(result.x, x_ref, atol=1e-6, rtol=1e-6)


def test_solve_drops_restart_kwarg_when_pcg_is_selected():
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    result = solve(op, _B_SPD, method="cg", restart=5)
    x_ref = torch.linalg.solve(_A_SPD, _B_SPD)
    torch.testing.assert_close(result.x, x_ref, atol=1e-8, rtol=1e-8)


def test_direct_accepts_and_ignores_every_kwarg():
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(False))
    result = solve(
        op,
        _B_SPD,
        method="direct",
        rtol=1e-3,
        atol=1e-3,
        max_iter=1,
        x0=torch.zeros(2),
        preconditioner="jacobi",
        restart=1,
    )
    x_ref = torch.linalg.solve(_A_SPD, _B_SPD)
    torch.testing.assert_close(result.x, x_ref, atol=1e-8, rtol=1e-8)


def test_on_failure_return_does_not_suppress_an_eligibility_refusal():
    # on_failure applies to NUMERICAL failure only; an eligibility refusal is a contract
    # violation and raises regardless of on_failure.
    op = _FakeOperator(_A_NS, symmetric=False, certificate=torch.tensor(False))
    with pytest_raises_containing("cg"):
        solve(op, _B_NS, method="cg", on_failure="return")


def test_unknown_method_raises_valueerror():
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    try:
        solve(op, _B_SPD, method="bogus")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "bogus" in str(exc)


def test_unknown_on_failure_raises_valueerror():
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    try:
        solve(op, _B_SPD, on_failure="bogus")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "bogus" in str(exc)


class pytest_raises_containing:
    """Small local helper: assert a RuntimeError is raised whose message contains `text`
    (case-sensitive substring), without pulling in a separate `pytest.raises(match=...)`
    regex-escaping discussion for plain literal substrings like "[1]".
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        assert exc_type is not None, f"expected a RuntimeError containing {self.text!r}"
        assert issubclass(exc_type, RuntimeError)
        assert self.text in str(exc), f"{self.text!r} not in {str(exc)!r}"
        return True


def test_negative_slope_refusal_message_names_the_offending_edges():
    """Finding I1: before the certificate tested section 3.1's condition 2, a
    grounded-but-negative-slope instance certified True, so `select.solve`'s negative-slope
    message branch was unreachable in practice and untested.
    """
    src = torch.tensor([0, 1, 0])
    tgt = torch.tensor([1, 2, 1])  # a parallel 0--1 edge, so grounding survives
    interior_of_node = torch.tensor([0, 1, -1])
    boundary_mask = torch.tensor([False, False, True])
    slopes = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, -5.0]])
    op = GraphLaplacianOperator(
        src, tgt, slopes, 2, interior_of_node, boundary_mask=boundary_mask
    )
    b = torch.ones(2, 2)
    with pytest_raises_containing("negative slope on edges"):
        solve(op, b)
    with pytest_raises_containing("[2]"):
        solve(op, b)


# -- section 6.2 step 2: method="sparse_direct" ----------------------------------------------


class _SparseFakeOperator(_FakeOperator):
    """`_FakeOperator` plus the optional `assemble_sparse` member, built from its own dense
    `A` by listing every structurally nonzero entry. Test-only: a real sparse operator
    derives its COO pattern from its topology (see `GraphLaplacianOperator`), but what
    `select.solve`'s sparse-direct branch needs to be exercised on is the CONTRACT --
    shared int64 indices, batch-leading values -- not any particular operator's derivation.
    """

    def assemble_sparse(self):
        if self.A.dim() > 2:
            pattern = (self.A != 0).any(dim=tuple(range(self.A.dim() - 2)))
        else:
            pattern = self.A != 0
        row, col = torch.nonzero(pattern, as_tuple=True)
        values = self.A[..., row, col]
        return row.to(torch.int64), col.to(torch.int64), values


class _NoSparseFormOperator(_FakeOperator):
    def assemble_sparse(self):
        return None


def test_sparse_direct_matches_torch_linalg_solve_on_an_spd_system():
    op = _SparseFakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    result = solve(op, _B_SPD, method="sparse_direct")
    torch.testing.assert_close(
        result.x, torch.linalg.solve(_A_SPD, _B_SPD), rtol=1e-12, atol=1e-14
    )
    assert bool(torch.all(result.converged))
    assert int(result.iterations) == 1
    assert float(result.residual) < 1e-12
    assert int(result.status) == int(SolverStatus.CONVERGED)


def test_sparse_direct_matches_torch_linalg_solve_on_a_nonsymmetric_system():
    op = _SparseFakeOperator(_A_NS, symmetric=False, certificate=None)
    result = solve(op, _B_NS, method="sparse_direct")
    torch.testing.assert_close(
        result.x, torch.linalg.solve(_A_NS, _B_NS), rtol=1e-12, atol=1e-14
    )


def test_sparse_direct_solves_a_batch_instance_by_instance():
    A = torch.stack([_A_SPD, 2.0 * _A_SPD, _A_NS])
    b = torch.stack([_B_SPD, _B_SPD, _B_NS])
    op = _SparseFakeOperator(A, symmetric=False, certificate=None)
    result = solve(op, b, method="sparse_direct")
    torch.testing.assert_close(result.x, torch.linalg.solve(A, b), rtol=1e-12, atol=1e-14)
    assert result.iterations.shape == (3,)
    assert result.status.shape == (3,)


def test_sparse_direct_broadcasts_an_unbatched_operator_against_a_batched_rhs():
    b = torch.stack([_B_SPD, 3.0 * _B_SPD])
    op = _SparseFakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    result = solve(op, b, method="sparse_direct")
    torch.testing.assert_close(
        result.x, torch.linalg.solve(_A_SPD.expand(2, 2, 2), b), rtol=1e-12, atol=1e-14
    )


def test_sparse_direct_reports_per_instance_singular_status_without_raising():
    """The per-instance failure contract (design section 3.2) on the sparse path: one
    singular instance is reported as SINGULAR with x = 0, and its siblings are still solved.
    """
    A = torch.stack([_A_SPD, torch.zeros(2, 2)])
    b = torch.stack([_B_SPD, _B_SPD])
    op = _SparseFakeOperator(A, symmetric=False, certificate=None)
    result = solve(op, b, method="sparse_direct", on_failure="return")
    assert bool(result.converged[0]) and not bool(result.converged[1])
    assert int(result.status[1]) == int(SolverStatus.SINGULAR)
    torch.testing.assert_close(result.x[1], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(
        result.x[0], torch.linalg.solve(_A_SPD, _B_SPD), rtol=1e-12, atol=1e-14
    )


def test_sparse_direct_on_failure_raise_names_the_singular_instance():
    A = torch.stack([_A_SPD, torch.zeros(2, 2)])
    b = torch.stack([_B_SPD, _B_SPD])
    op = _SparseFakeOperator(A, symmetric=False, certificate=None)
    with pytest_raises_containing("SINGULAR"):
        solve(op, b, method="sparse_direct", where="sparse")


def test_sparse_direct_refuses_when_assemble_sparse_returns_none():
    op = _NoSparseFormOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    try:
        solve(op, _B_SPD, method="sparse_direct")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "assemble_sparse" in str(exc)


def test_sparse_direct_refuses_an_operator_without_the_optional_member_at_all():
    op = _FakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    try:
        solve(op, _B_SPD, method="sparse_direct")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "assemble_sparse" in str(exc)


def test_sparse_direct_refuses_to_run_under_grad_mode_with_a_grad_requiring_input():
    """The backend goes out to SciPy and back, so nothing it computes is connected to the
    autograd graph. Both layer paths solve under `no_grad` (`_LinearSolve` and `_Implicit`),
    so this never fires there -- it fires for a caller who would otherwise receive a
    silently detached answer.
    """
    b = _B_SPD.clone().requires_grad_(True)
    op = _SparseFakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    with pytest_raises_containing("sparse_direct"):
        solve(op, b, method="sparse_direct")


def test_sparse_direct_refuses_when_the_operators_own_values_require_grad():
    A = _A_SPD.clone().requires_grad_(True)
    op = _SparseFakeOperator(A, symmetric=True, certificate=torch.tensor(True))
    with pytest_raises_containing("sparse_direct"):
        solve(op, _B_SPD, method="sparse_direct")


def test_sparse_direct_runs_happily_under_no_grad_on_grad_requiring_inputs():
    A = _A_SPD.clone().requires_grad_(True)
    op = _SparseFakeOperator(A, symmetric=True, certificate=torch.tensor(True))
    with torch.no_grad():
        result = solve(op, _B_SPD, method="sparse_direct")
    torch.testing.assert_close(
        result.x, torch.linalg.solve(_A_SPD, _B_SPD), rtol=1e-12, atol=1e-14
    )
    assert not result.x.requires_grad


def test_sparse_direct_raises_importerror_naming_scipy_when_it_is_missing(monkeypatch):
    """SciPy is a DEV dependency, so the backend imports it lazily inside the function and
    must say so by name when it is absent, rather than failing with a bare ModuleNotFound
    from somewhere inside the solver.
    """
    import builtins

    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.split(".")[0] == "scipy":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    op = _SparseFakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    try:
        solve(op, _B_SPD, method="sparse_direct")
        raise AssertionError("expected ImportError")
    except ImportError as exc:
        assert "scipy" in str(exc).lower()


def test_sparse_direct_result_tensors_carry_the_inputs_dtype_and_device():
    A = _A_SPD.to(torch.float32)
    b = _B_SPD.to(torch.float32)
    op = _SparseFakeOperator(A, symmetric=True, certificate=torch.tensor(True))
    result = solve(op, b, method="sparse_direct")
    assert result.x.dtype is torch.float32
    assert result.x.device == b.device
    assert result.residual.dtype is torch.float32
    torch.testing.assert_close(result.x, torch.linalg.solve(A, b), rtol=1e-5, atol=1e-6)


def test_sparse_direct_reports_zero_residual_for_a_zero_right_hand_side():
    """`_direct`'s own convention, preserved: a zero rhs has no relative residual to report,
    so the reported residual is 0 rather than 0/0.
    """
    op = _SparseFakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(True))
    result = solve(op, torch.zeros(2, dtype=torch.float64), method="sparse_direct")
    assert float(result.residual) == 0.0


def test_sparse_direct_matches_direct_on_the_graph_laplacian_chain_fixture():
    slopes = torch.tensor([[1.0, 1.0], [2.0, 3.0]])
    op = _chain_op(slopes)
    b = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    sparse = solve(op, b, method="sparse_direct")
    direct = solve(op, b, method="direct")
    torch.testing.assert_close(sparse.x, direct.x, rtol=1e-9, atol=1e-12)


def test_sparse_direct_accepts_and_ignores_every_iterative_kwarg():
    op = _SparseFakeOperator(_A_SPD, symmetric=True, certificate=torch.tensor(False))
    result = solve(
        op,
        _B_SPD,
        method="sparse_direct",
        rtol=1e-3,
        atol=1e-3,
        max_iter=1,
        x0=torch.zeros(2),
        preconditioner="jacobi",
        restart=1,
    )
    torch.testing.assert_close(
        result.x, torch.linalg.solve(_A_SPD, _B_SPD), rtol=1e-12, atol=1e-14
    )


def test_sparse_direct_ignores_the_certificate_entirely(monkeypatch):
    """An explicit `sparse_direct` makes no SPD assumption, so it neither consults nor
    diagnoses the certificate -- the same contract `method="direct"` has.
    """
    calls = _spy(monkeypatch, GraphLaplacianOperator, "spd_diagnosis")
    slopes = torch.tensor([[1.0, 1.0], [0.0, 1.0]])  # instance 1 does not certify
    op = _chain_op(slopes)
    b = torch.ones(2, 2)
    result = solve(op, b, method="sparse_direct", on_failure="return")
    assert calls["count"] == 0
    assert bool(result.converged[0])
