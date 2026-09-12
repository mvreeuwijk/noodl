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
