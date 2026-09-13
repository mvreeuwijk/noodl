"""Parity tests between TransportLayer's new operator-based paths and its retained
dense oracle (`operator()`, unchanged since milestone 1). Every test here compares
the SPARSE result against the DENSE one on the same problem; `tests/layers/
test_transport.py` is untouched and re-verifies the dense/analytic behaviour on
its own.
"""

import math

import pytest
import torch
from torch.autograd import gradcheck

from benchmarks.measure import saved_tensor_bytes
from tellegen.layers.transport import (
    TransportLayer,
    _expm_action,
    _linear_solve,
    _van_loan_step_dense,
)
from tellegen.operators.base import SolverStatus
from tellegen.solvers.select import solve as _solve_operator
from tellegen.topology import Network


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


def test_advection_operator_matvec_matches_dense_operator():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    M, _ = layer.operator(q)
    op = layer._advection_operator(q)
    x = torch.tensor([12.0, -4.0], dtype=torch.float64)
    torch.testing.assert_close(op.matvec(x), M @ x, rtol=1e-9, atol=1e-12)


def test_advection_operator_boundary_forcing_matches_dense_N():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    _, N = layer.operator(q)
    op = layer._advection_operator(q)
    x_b = torch.tensor([420.0], dtype=torch.float64)
    torch.testing.assert_close(op.boundary_forcing(x_b), N @ x_b, rtol=1e-9, atol=1e-12)


def test_steady_sparse_matches_dense_oracle():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    x_sparse = layer.steady(q, source, c_out)

    M, N = layer.operator(q)
    b0 = (N @ c_out.unsqueeze(-1)).squeeze(-1) + source / layer.capacity
    x_dense = torch.linalg.solve(M, -b0.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(x_sparse, x_dense, rtol=1e-9, atol=1e-12)


def test_steady_singular_system_raises_naming_instance():
    net = Network(dtype=torch.float64)
    net.add_node("A")
    net.add_node("B")
    net.add_edge("A", "B", kind="airpath")
    net.add_edge("B", "A", kind="airpath")
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([100.0, 100.0]), flow_kind="airpath", boundary=[]
    )
    q = torch.zeros(2, dtype=torch.float64)  # no flow, no boundary: M is exactly the zero matrix
    # A nonzero source is required: with source == 0 too, the system 0 = 0 is trivially (if
    # non-uniquely) satisfied by x = 0, which GMRES correctly reports as CONVERGED -- unlike
    # torch.linalg.solve, an iterative solver has no obligation to detect that M itself is
    # singular when the particular right-hand side it was asked to solve happens to be
    # consistent with it. A nonzero source makes the system exactly singular AND
    # inconsistent (no x solves M x = b when M = 0 and b != 0), which is what genuinely
    # fails to converge.
    source = torch.ones(2, dtype=torch.float64)
    x_b = torch.zeros(0, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="co2"):
        layer.steady(q, source, x_b)


def test_steady_on_failure_return_gives_failing_status_instead_of_raising():
    net = Network(dtype=torch.float64)
    net.add_node("A")
    net.add_node("B")
    net.add_edge("A", "B", kind="airpath")
    net.add_edge("B", "A", kind="airpath")
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([100.0, 100.0]), flow_kind="airpath", boundary=[]
    )
    q = torch.zeros(2, dtype=torch.float64)
    source = torch.ones(2, dtype=torch.float64)  # see the fixture note above: must be nonzero
    x_b = torch.zeros(0, dtype=torch.float64)
    result = layer.steady(q, source, x_b, on_failure="return")
    assert bool(torch.any(result.status != SolverStatus.CONVERGED))


def test_gradcheck_steady_wrt_q_sources_boundary():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([0.4, 0.4], dtype=torch.float64, requires_grad=True)
    sources = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    x_b = torch.tensor([420.0], dtype=torch.float64, requires_grad=True)

    def f(q, sources, x_b):
        return layer.steady(q, sources, x_b)

    assert gradcheck(f, (q, sources, x_b), eps=1e-6, atol=1e-5)


def test_steady_adjoint_gradient_matches_unrolled_reference_on_small_problem():
    """Proves the implicit adjoint is RIGHT, not merely self-consistent with finite
    differences of itself (all `gradcheck` alone would show): on a problem small enough
    that unrolling GMRES's own iteration is affordable, compare the production
    adjoint-based gradient (through `TransportLayer.steady`) against a gradient obtained
    by calling `solvers.select.solve` DIRECTLY and differentiating straight through its
    ordinary (non-`no_grad`) iteration -- bypassing `_LinearSolve` entirely. Agreement
    here is the check that actually tells the adjoint derivation is correct.
    """
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([0.4, 0.4], dtype=torch.float64, requires_grad=True)
    sources = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    x_b = torch.tensor([420.0], dtype=torch.float64, requires_grad=True)

    x_adjoint = layer.steady(q, sources, x_b)
    grad_adjoint = torch.autograd.grad(x_adjoint.sum(), (q, sources, x_b))

    q2 = q.detach().clone().requires_grad_(True)
    sources2 = sources.detach().clone().requires_grad_(True)
    x_b2 = x_b.detach().clone().requires_grad_(True)
    op = layer._advection_operator(q2)
    xb_s, _ = layer._to_stacked(x_b2, layer.n_b, "x_boundary")
    src_s, _ = layer._to_stacked(sources2, layer.n_i, "sources")
    cap = layer._capacity_stacked(torch.float64)
    b0 = op.boundary_forcing(xb_s) + src_s / cap
    x_unrolled = _solve_operator(op, -b0, method="auto", where="unrolled reference").x
    grad_unrolled = torch.autograd.grad(x_unrolled.sum(), (q2, sources2, x_b2))

    for g_a, g_u in zip(grad_adjoint, grad_unrolled, strict=True):
        torch.testing.assert_close(g_a, g_u, rtol=1e-6, atol=1e-8)


def test_backward_memory_independent_of_solver_iterations():
    """A loose and a tight GMRES tolerance on the SAME, deliberately ill-conditioned
    problem (widely disparate capacities along a long chain) give substantially different
    FORWARD iteration counts -- asserted and printed below, so a future regression that
    accidentally makes the two tolerances equally cheap is visible.

    Backward memory is measured with `saved_tensor_bytes` (amendment A4(a)), not
    `tracemalloc`: `tracemalloc` sees zero bytes of PyTorch tensor allocations on this
    `.venv` (measured; see `benchmarks/measure.py`'s module docstring), so a
    `tracemalloc`-based assertion here would pass vacuously regardless of whether the
    implicit adjoint actually decouples backward memory from forward iteration count.
    `saved_tensor_bytes` instead counts exactly the tensors autograd will need for
    `backward()`, deterministically.

    The bytes `_linear_solve` (the implicit adjoint) saves for backward must be EQUAL at
    the loose and tight tolerance -- the whole point of the adjoint over unrolling. That
    assertion alone would still be vacuous if `saved_tensor_bytes` simply always returned
    the same number regardless of what ran; the second half of this test rules that out by
    showing the UNROLLED reference (autograd tracing straight through `solvers.select.solve`'s
    own iteration, bypassing `_LinearSolve` entirely) saves MORE at the tight tolerance than
    the loose one, on the exact same problem -- proving `saved_tensor_bytes` is sensitive to
    iteration count in general, so the adjoint's flat count is a real property of the
    adjoint, not an artifact of the measurement.

    Fixture note: the brief's original 3-node fixture (`three_node_chain`, capacities
    [50, 8000], `q0 = [0.05, -0.03, 0.02]`) reverses the middle edge, which makes node A a
    pure sink with no outgoing advective edge at all -- the resulting 2x2 M is EXACTLY
    singular (confirmed against the dense `operator()` oracle too, independent of
    `AdvectionOperator`), and even with the sign fixed to a natural forward flow, a 2x2
    system is far too small: GMRES's restart cycle length defaults to `min(restart, m) = m`
    for `m = 2`, so ONE full-size cycle always covers the entire Krylov space regardless of
    `rtol`, and `saved_tensor_bytes` (which reflects actual computation, not just the
    reported `iterations` count) is IDENTICAL for both tolerances -- vacuous for exactly the
    reason A4(a) rejects `tracemalloc`, just one level deeper. A genuinely restart-bound
    problem needs `m` large enough, relative to GMRES's default `restart=30`, that a loose
    tolerance converges within the first restart cycle while a tight one needs several more
    -- a 40-node chain with capacities spanning 4 orders of magnitude does this.
    """

    def long_chain(n: int) -> Network:
        net = Network(dtype=torch.float64)
        net.add_node("ambient")
        for i in range(n):
            net.add_node(f"n{i}")
        net.add_edge("ambient", "n0", kind="airpath")
        for i in range(n - 1):
            net.add_edge(f"n{i}", f"n{i + 1}", kind="airpath")
        net.add_edge(f"n{n - 1}", "ambient", kind="airpath")
        return net

    n = 40
    net = long_chain(n)
    cap = torch.logspace(0, 4, n, dtype=torch.float64)  # 4 orders of magnitude: ill-conditioned
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q0 = 0.05 * torch.ones(n + 1, dtype=torch.float64)  # natural forward flow around the loop
    sources0 = torch.ones(n, dtype=torch.float64)
    x_b0 = torch.tensor([420.0], dtype=torch.float64)
    max_iter = 400  # exceeds default restart=30 several times over, so tolerance genuinely
    #                 controls how many restart cycles GMRES needs (not just whether the one
    #                 cycle a small problem gets already covers the whole Krylov space).

    def build_system(q_, sources_, xb_):
        op = layer._advection_operator(q_)
        xb_s, _ = layer._to_stacked(xb_, layer.n_b, "x_boundary")
        src_s, _ = layer._to_stacked(sources_, layer.n_i, "sources")
        cap_s = layer._capacity_stacked(torch.float64)
        b0 = op.boundary_forcing(xb_s) + src_s / cap_s
        return op, -b0

    op0, rhs0 = build_system(q0, sources0, x_b0)
    iters_loose = _solve_operator(
        op0, rhs0, method="auto", rtol=1e-2, max_iter=max_iter, where="probe"
    ).iterations
    iters_tight = _solve_operator(
        op0, rhs0, method="auto", rtol=1e-6, max_iter=max_iter, where="probe"
    ).iterations
    print(
        f"GMRES iterations: loose(rtol=1e-2)={iters_loose.tolist()}, "
        f"tight(rtol=1e-6)={iters_tight.tolist()}"
    )
    assert int(iters_tight.max()) > int(iters_loose.max())

    def adjoint_saved_bytes(rtol):
        q = q0.clone().requires_grad_(True)
        sources = sources0.clone().requires_grad_(True)
        x_b = x_b0.clone().requires_grad_(True)

        def run():
            return _linear_solve(
                build_system, "probe", q, sources, x_b, rtol=rtol, max_iter=max_iter
            )

        return saved_tensor_bytes(run)[0]

    def unrolled_saved_bytes(rtol):
        q = q0.clone().requires_grad_(True)
        sources = sources0.clone().requires_grad_(True)
        x_b = x_b0.clone().requires_grad_(True)

        def run():
            op, rhs = build_system(q, sources, x_b)
            return _solve_operator(
                op, rhs, method="auto", rtol=rtol, max_iter=max_iter, where="unrolled probe"
            ).x

        return saved_tensor_bytes(run)[0]

    bytes_loose = adjoint_saved_bytes(1e-2)
    bytes_tight = adjoint_saved_bytes(1e-6)
    bytes_unrolled_loose = unrolled_saved_bytes(1e-2)
    bytes_unrolled_tight = unrolled_saved_bytes(1e-6)
    print(
        f"saved tensor bytes: adjoint loose={bytes_loose}, adjoint tight={bytes_tight}, "
        f"unrolled loose={bytes_unrolled_loose}, unrolled tight={bytes_unrolled_tight}"
    )
    assert bytes_tight == bytes_loose
    assert bytes_unrolled_tight > bytes_unrolled_loose


def test_implicit_step_sparse_matches_dense_oracle():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath",
                            boundary=["ambient"], scheme="implicit")
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    dt = 200.0

    x_sparse = layer.step(c0, q, source, c_out, dt)

    M, N = layer.operator(q)
    b0 = (N @ c_out.unsqueeze(-1)).squeeze(-1) + source / cap
    m = M.shape[-1]
    eye = torch.eye(m, dtype=torch.float64)
    rhs = c0 + dt * b0
    x_dense = torch.linalg.solve(eye - dt * M, rhs.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(x_sparse, x_dense, rtol=1e-9, atol=1e-12)


def test_gradcheck_implicit_step_wrt_x_q_sources_boundary():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit",
    )
    x = torch.tensor([150.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.4, 0.4], dtype=torch.float64, requires_grad=True)
    sources = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    x_b = torch.tensor([420.0], dtype=torch.float64, requires_grad=True)

    def f(x, q, sources, x_b):
        return layer.step(x, q, sources, x_b, 300.0)

    assert gradcheck(f, (x, q, sources, x_b), eps=1e-6, atol=1e-5)


def test_trapezoidal_step_sparse_matches_dense_oracle():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath",
                            boundary=["ambient"], scheme="trapezoidal")
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    dt = 200.0

    x_sparse = layer.step(c0, q, source, c_out, dt)

    M, N = layer.operator(q)
    b0 = (N @ c_out.unsqueeze(-1)).squeeze(-1) + source / cap
    m = M.shape[-1]
    eye = torch.eye(m, dtype=torch.float64)
    rhs = ((eye + 0.5 * dt * M) @ c0.unsqueeze(-1)).squeeze(-1) + dt * b0
    x_dense = torch.linalg.solve(eye - 0.5 * dt * M, rhs.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(x_sparse, x_dense, rtol=1e-9, atol=1e-12)


def test_two_sealed_zones_conserve_total_amount_sparse_path():
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("A")
    net.add_node("B")
    net.add_edge("A", "B", kind="airpath")
    net.add_edge("B", "A", kind="airpath")
    cap = torch.tensor([100.0, 300.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath",
                            boundary=["ambient"], scheme="implicit")
    q = torch.tensor([0.05, 0.05], dtype=torch.float64)
    c = torch.tensor([1000.0, 400.0], dtype=torch.float64)
    total0 = (cap * c).sum()
    for _ in range(10):
        c = layer.step(c, q, torch.zeros(2, dtype=torch.float64), torch.tensor([420.0]), 900.0)
    total = (cap * c).sum()
    torch.testing.assert_close(total, total0, rtol=1e-8, atol=1e-8)


def test_implicit_step_on_failure_return_does_not_raise():
    net = Network(dtype=torch.float64)
    net.add_node("A")
    net.add_node("B")
    net.add_edge("A", "B", kind="airpath")
    net.add_edge("B", "A", kind="airpath")
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([100.0, 100.0]), flow_kind="airpath",
        boundary=[], scheme="implicit",
    )
    q = torch.zeros(2, dtype=torch.float64)
    x = torch.tensor([10.0, 5.0], dtype=torch.float64)
    source = torch.zeros(2, dtype=torch.float64)
    x_b = torch.zeros(0, dtype=torch.float64)
    result, _ = layer._implicit_step_sparse(x, q, source, x_b, 1.0, "return")
    assert result.converged.all()  # backward Euler with dt=1 is well posed here; sanity check


def test_step_and_steady_reject_unknown_on_failure():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="on_failure"):
        layer.step(c0, q, source, c_out, 300.0, on_failure="bogus")
    with pytest.raises(ValueError, match="on_failure"):
        layer.steady(q, source, c_out, on_failure="bogus")


def test_expm_action_matches_dense_matrix_exp_small_dt():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    x0 = torch.tensor([12.0, -4.0], dtype=torch.float64)
    xb = torch.tensor([420.0], dtype=torch.float64)
    sources = torch.zeros(2, dtype=torch.float64)
    dt = 30.0

    M, N = layer.operator(q)
    b0 = (N @ xb.unsqueeze(-1)).squeeze(-1) + sources / cap
    dense = _van_loan_step_dense(M, x0, b0, dt)

    op = layer._advection_operator(q)
    sparse, substeps = _expm_action(op, x0, b0, dt)
    assert substeps == 1  # this problem is not stiff at dt=30
    torch.testing.assert_close(sparse, dense, rtol=1e-9, atol=1e-12)


def test_expm_action_matches_dense_zero_flow_singular_M():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.zeros(2, dtype=torch.float64)  # M is exactly the zero matrix here
    x0 = torch.tensor([100.0], dtype=torch.float64)
    xb = torch.tensor([420.0], dtype=torch.float64)
    sources = torch.tensor([3.0], dtype=torch.float64)
    dt = 500.0

    M, N = layer.operator(q)
    b0 = (N @ xb.unsqueeze(-1)).squeeze(-1) + sources / cap
    dense = _van_loan_step_dense(M, x0, b0, dt)
    op = layer._advection_operator(q)
    sparse, _ = _expm_action(op, x0, b0, dt)
    torch.testing.assert_close(sparse, dense, rtol=1e-9, atol=1e-12)
    # zero flow, zero M: x should simply grow linearly in dt from the constant source term
    torch.testing.assert_close(sparse, x0 + dt * b0, rtol=1e-9, atol=1e-12)


def test_expm_action_matches_dense_large_dt():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    x0 = torch.tensor([100.0], dtype=torch.float64)
    xb = torch.tensor([420.0], dtype=torch.float64)
    sources = torch.tensor([2.0], dtype=torch.float64)
    dt = 1e6

    M, N = layer.operator(q)
    b0 = (N @ xb.unsqueeze(-1)).squeeze(-1) + sources / cap
    dense = _van_loan_step_dense(M, x0, b0, dt)
    op = layer._advection_operator(q)
    sparse, substeps = _expm_action(op, x0, b0, dt)
    assert substeps > 1
    torch.testing.assert_close(sparse, dense, rtol=1e-9, atol=1e-12)


def test_expm_action_matches_dense_three_species_kinetics():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    l1, l2 = 0.01, 0.02
    kinetics = torch.zeros(3, 3, dtype=torch.float64)
    kinetics[0, 0] = -l1
    kinetics[1, 0] = l1
    kinetics[1, 1] = -l2
    kinetics[2, 1] = l2
    layer = TransportLayer(
        net, "chain", capacity=cap, flow_kind="airpath", boundary=["ambient"], n_species=3,
        kinetics=kinetics,
    )
    q = torch.zeros(2, dtype=torch.float64)
    x0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    xb = torch.zeros(3, dtype=torch.float64)
    sources = torch.zeros(3, dtype=torch.float64)
    dt = 200.0

    M, N = layer.operator(q)
    b0 = (N @ xb.unsqueeze(-1)).squeeze(-1) + sources / layer._capacity_stacked(torch.float64)
    dense = _van_loan_step_dense(M, x0, b0, dt)
    op = layer._advection_operator(q)
    sparse, _ = _expm_action(op, x0, b0, dt)
    torch.testing.assert_close(sparse, dense, rtol=1e-9, atol=1e-12)


def test_expm_action_flow_reversal_matches_dense():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    x0 = torch.tensor([100.0], dtype=torch.float64)
    xb = torch.tensor([420.0], dtype=torch.float64)
    sources = torch.zeros(1, dtype=torch.float64)
    dt = 300.0
    for q in (torch.tensor([0.5, 0.5], dtype=torch.float64),
              torch.tensor([-0.5, -0.5], dtype=torch.float64)):
        M, N = layer.operator(q)
        b0 = (N @ xb.unsqueeze(-1)).squeeze(-1) + sources / cap
        dense = _van_loan_step_dense(M, x0, b0, dt)
        op = layer._advection_operator(q)
        sparse, _ = _expm_action(op, x0, b0, dt)
        torch.testing.assert_close(sparse, dense, rtol=1e-9, atol=1e-12)


def test_exact_scheme_conserves_total_amount_sparse_path():
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("A")
    net.add_node("B")
    net.add_edge("A", "B", kind="airpath")
    net.add_edge("B", "A", kind="airpath")
    cap = torch.tensor([100.0, 300.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = torch.tensor([0.05, 0.05], dtype=torch.float64)
    c = torch.tensor([1000.0, 400.0], dtype=torch.float64)
    total0 = (cap * c).sum()
    for _ in range(10):
        c = layer.step(c, q, torch.zeros(2, dtype=torch.float64), torch.tensor([420.0]), 900.0)
    total = (cap * c).sum()
    torch.testing.assert_close(total, total0, rtol=1e-9, atol=1e-9)


def test_exact_scheme_preserves_positivity_sparse_path():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c = torch.tensor([0.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    source = torch.zeros(1, dtype=torch.float64)
    for _ in range(20):
        c = layer.step(c, q, source, c_out, 300.0)
        assert torch.all(c >= 0.0)


def test_error_control_triggers_substepping_on_a_stiff_case():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1.0]), flow_kind="airpath", boundary=["ambient"],
        removal=torch.tensor([500.0], dtype=torch.float64),  # dt * rate will be huge
    )
    q = torch.zeros(2, dtype=torch.float64)
    x0 = torch.tensor([10.0], dtype=torch.float64)
    xb = torch.tensor([0.0], dtype=torch.float64)
    sources = torch.zeros(1, dtype=torch.float64)
    dt = 50.0  # dt * rate = 25000: far past the Taylor series' single-step radius

    op = layer._advection_operator(q)
    M, N = layer.operator(q)
    b0 = (N @ xb.unsqueeze(-1)).squeeze(-1) + sources / layer.capacity
    sparse, substeps = _expm_action(op, x0, b0, dt)
    assert substeps > 1

    expected = x0 * math.exp(-500.0 * dt)
    torch.testing.assert_close(sparse, expected, rtol=1e-6, atol=1e-9)


def test_error_control_raises_naming_instances_when_max_substeps_exceeded():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1.0]), flow_kind="airpath", boundary=["ambient"],
        removal=torch.tensor([1e6], dtype=torch.float64),
    )
    q = torch.zeros(2, dtype=torch.float64)
    x0 = torch.tensor([10.0], dtype=torch.float64)
    xb = torch.tensor([0.0], dtype=torch.float64)
    sources = torch.zeros(1, dtype=torch.float64)
    op = layer._advection_operator(q)
    M, N = layer.operator(q)
    b0 = (N @ xb.unsqueeze(-1)).squeeze(-1) + sources / layer.capacity
    with pytest.raises(RuntimeError, match="TransportLayer 'co2' exact step.*failed to converge"):
        _expm_action(
            op, x0, b0, dt=1e9, max_substeps=3,
            where=f"TransportLayer '{layer.name}' exact step",
        )


def test_gradcheck_expm_action_wrt_x_flow_sources_boundary():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    x0 = torch.tensor([150.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.4, 0.4], dtype=torch.float64, requires_grad=True)
    sources = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    x_b = torch.tensor([420.0], dtype=torch.float64, requires_grad=True)

    def f(x, q, sources, x_b):
        op = layer._advection_operator(q)
        b0 = op.boundary_forcing(x_b) + sources / layer.capacity
        result, _ = _expm_action(op, x, b0, 300.0)
        return result

    assert gradcheck(f, (x0, q, sources, x_b), eps=1e-6, atol=1e-5)


def test_expm_action_backward_memory_scales_with_substep_count():
    """Measures (does not gate) how backward memory through _expm_action's own unrolled
    iteration grows from a non-stiff case (1 substep) to a genuinely stiff one (several
    substeps, forced exactly as in test_error_control_triggers_substepping_on_a_stiff_case
    above). Unlike Task 9's linear-solve adjoint, no O(state) bound is asserted here --
    see this task's Mathematics section for why none is expected. `tracemalloc` cannot be
    used (amendment A4(a)): it sees zero bytes of PyTorch allocations on this `.venv`, so a
    tracemalloc-based assertion would pass vacuously regardless of whether more sub-steps
    genuinely cost more backward memory. `saved_tensor_bytes` instead counts exactly the
    tensors autograd will need for backward(), deterministically. The printed numbers are
    what a future composed-model-gate report (design spec section 6.1) would compare
    against its timestep-history memory budget; if that gate is ever breached, section
    6.2 already names checkpointing the sub-steps as the follow-up, not a change to this
    test.
    """
    net = flow_through_zone()
    layer_mild = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"],
    )
    layer_stiff = TransportLayer(
        net, "co2", capacity=torch.tensor([1.0]), flow_kind="airpath", boundary=["ambient"],
        removal=torch.tensor([500.0], dtype=torch.float64),
    )

    def run_and_measure(layer, dt):
        q = torch.zeros(2, dtype=torch.float64)
        x0 = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
        xb = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        sources = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        op = layer._advection_operator(q)
        b0 = op.boundary_forcing(xb) + sources / layer.capacity

        substeps_holder = {}

        def run():
            result, substeps = _expm_action(op, x0, b0, dt)
            substeps_holder["substeps"] = substeps
            return result

        peak, result = saved_tensor_bytes(run)
        result.sum().backward()
        return substeps_holder["substeps"], peak

    substeps_mild, peak_mild = run_and_measure(layer_mild, 30.0)
    substeps_stiff, peak_stiff = run_and_measure(layer_stiff, 50.0)
    print(
        f"_expm_action backward memory (saved_tensor_bytes): "
        f"mild(substeps={substeps_mild})={peak_mild}, "
        f"stiff(substeps={substeps_stiff})={peak_stiff}"
    )
    assert substeps_stiff > substeps_mild  # confirms the two cases are genuinely different
    assert peak_stiff > peak_mild  # more sub-steps genuinely save more for backward


def test_expm_action_mixed_stiffness_batch_matches_standalone_within_tolerance():
    """ONE `_expm_action` call on a batch of two instances -- instance 0 (removal rate
    0.01) converges in a handful of Taylor terms, instance 1 (removal rate 500.0, the
    same rate/dt as test_error_control_triggers_substepping_on_a_stiff_case above) needs
    several dt-halvings. Because the WHOLE batch is halved together whenever ANY instance
    has not converged (the batched shape never changes, matching newton.py's masking
    idiom), the mild instance is recomputed at the halved dt too, by a DIFFERENT sequence
    of Taylor evaluations than its own standalone (single-instance) call would use. This
    test therefore asserts agreement with the standalone result and with the dense oracle
    to rtol=1e-9/atol=1e-12 -- a TOLERANCE bound, not bit-for-bit identity, which the
    algorithm does not guarantee by design.
    """
    net = flow_through_zone()
    cap = torch.tensor([1.0], dtype=torch.float64)
    rates = [0.01, 500.0]
    layers = [
        TransportLayer(
            net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"],
            removal=torch.tensor([r], dtype=torch.float64),
        )
        for r in rates
    ]
    q = torch.zeros(2, dtype=torch.float64)
    dt = 50.0
    xb = torch.tensor([0.0], dtype=torch.float64)
    sources = torch.zeros(1, dtype=torch.float64)
    x0_val = 10.0

    # Build ONE batched operator: same net/topology, but a per-instance removal rate
    # stacked along a new leading batch dimension -- everything else (flow, transmission,
    # capacity) is shared, un-batched, and broadcasts against it.
    op_batch = layers[1]._advection_operator(q)
    op_batch.removal = torch.stack([layer.removal for layer in layers], dim=0)  # (2, 1, 1)

    x_batch = torch.tensor([[x0_val], [x0_val]], dtype=torch.float64, requires_grad=True)
    b0_batch = torch.zeros(2, 1, dtype=torch.float64)

    sparse, substeps = _expm_action(op_batch, x_batch, b0_batch, dt)
    assert substeps > 1

    x_standalone = [
        torch.tensor([x0_val], dtype=torch.float64, requires_grad=True) for _ in rates
    ]
    standalone_results = []
    for layer, xi in zip(layers, x_standalone, strict=True):
        op = layer._advection_operator(q)  # a FRESH operator per instance, never op_batch
        M, N = layer.operator(q)
        b0 = (N @ xb.unsqueeze(-1)).squeeze(-1) + sources / layer.capacity
        dense = _van_loan_step_dense(M, xi.detach(), b0, dt)
        result_i, _ = _expm_action(op, xi, b0, dt)
        torch.testing.assert_close(result_i.detach(), dense, rtol=1e-9, atol=1e-12)
        standalone_results.append(result_i)

    for i in range(2):
        torch.testing.assert_close(
            sparse[i], standalone_results[i].detach(), rtol=1e-9, atol=1e-12
        )

    sparse.sum().backward()
    assert torch.isfinite(x_batch.grad).all()
    for i in range(2):
        standalone_results[i].backward()
        torch.testing.assert_close(
            x_batch.grad[i], x_standalone[i].grad, rtol=1e-8, atol=1e-8
        )


def _conduction_chain_layer() -> TransportLayer:
    """ambient (boundary) -- A -- B with three airpath edges and ONE conduction edge A->B.

    Used by the Task 15 structural tests: the conduction branch is the one that used to
    build an (n, n) `self.L` in `__init__`, and the no-conduction branch is the one that
    allocated an (n, n) block of ZEROS for nothing at all.
    """
    net = Network(dtype=torch.float64)
    for name in ("ambient", "A", "B"):
        net.add_node(name)
    net.add_edge("ambient", "A", kind="airpath")
    net.add_edge("A", "B", kind="airpath")
    net.add_edge("B", "ambient", kind="airpath")
    net.add_edge("A", "B", kind="conduction")
    return TransportLayer(
        net,
        "heat",
        capacity=torch.tensor([50.0, 80.0], dtype=torch.float64),
        flow_kind="airpath",
        boundary=["ambient"],
        conduction_kind="conduction",
        conductance=torch.tensor([2.5], dtype=torch.float64),
    )


def test_transport_layer_allocates_no_dense_L_at_construction():
    # Task 15: `self.L` is (n, n) and grew 4x per node doubling in the composed-model
    # memory gate, and in the NO-conduction case it was a block of zeros that operator()
    # subtracted for nothing. Construction must hold no (n, n) tensor on either branch.
    net = three_node_chain()
    plain = TransportLayer(
        net,
        "co2",
        capacity=torch.tensor([50.0, 80.0], dtype=torch.float64),
        flow_kind="airpath",
        boundary=["ambient"],
    )
    for layer in (plain, _conduction_chain_layer()):
        n = layer.net.n
        square = {
            name: tuple(v.shape)
            for name, v in vars(layer).items()
            if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[-2:] == (n, n)
        }
        assert square == {}, f"TransportLayer holds an (n, n) tensor after __init__: {square}"


def test_operator_oracle_still_includes_conduction():
    # The dense oracle keeps its exact values. These are the (M, N) this fixture produced
    # with the pre-Task-15 code (conduction folded in via the construction-time `self.L`),
    # recorded before the change and hard-coded here, so a conduction term silently dropped
    # when L stopped being an attribute would fail this test rather than pass a
    # self-consistent comparison.
    layer = _conduction_chain_layer()
    q = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    M, N = layer.operator(q)

    expected_M = torch.tensor(
        [[-0.05, 0.054000000000000006], [0.03125, -0.036875000000000005]],
        dtype=torch.float64,
    )
    expected_N = torch.tensor([[0.006], [0.0]], dtype=torch.float64)
    torch.testing.assert_close(M, expected_M, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(N, expected_N, rtol=1e-12, atol=1e-14)

    # ... and the conduction term is genuinely IN there: the same network without the
    # conduction edge gives a different M (2.5 W/K on both diagonal entries, scaled by
    # capacity), so the assertions above are not merely pinning an advection-only oracle.
    no_conduction = TransportLayer(
        layer.net,
        "heat",
        capacity=torch.tensor([50.0, 80.0], dtype=torch.float64),
        flow_kind="airpath",
        boundary=["ambient"],
    )
    M_plain, _ = no_conduction.operator(q)
    torch.testing.assert_close(
        M - M_plain,
        torch.tensor([[-2.5 / 50.0, 2.5 / 50.0], [2.5 / 80.0, -2.5 / 80.0]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-14,
    )


# ---------------------------------------------------------------- batch broadcasting
# Final-review finding C1: an unbatched state against a BATCHED flow (one initial
# condition against an ensemble of flow realisations -- the milestone's own calibration
# use case) used to work at the pre-milestone dense `transport.py` and regressed to a raw
# RuntimeError out of `AdvectionOperator._raw_action`, which expanded the state only to
# its OWN batch shape rather than to the operator's.
def _ensemble_flow() -> torch.Tensor:
    base = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    scale = torch.linspace(0.5, 1.5, 5, dtype=torch.float64).unsqueeze(-1)
    return base * scale


@pytest.mark.parametrize("scheme", ["exact", "implicit", "trapezoidal"])
def test_step_broadcasts_unbatched_state_against_batched_flow(scheme):
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(
        net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"], scheme=scheme
    )
    q = _ensemble_flow()
    x = torch.tensor([12.0, -4.0], dtype=torch.float64)
    sources = torch.tensor([1.0, 2.0], dtype=torch.float64)
    x_boundary = torch.tensor([420.0], dtype=torch.float64)

    out = layer.step(x, q, sources, x_boundary, 60.0)
    assert out.shape == (5, 2)

    expanded = layer.step(
        x.expand(5, 2), q, sources.expand(5, 2), x_boundary.expand(5, 1), 60.0
    )
    assert torch.allclose(out, expanded)


def test_steady_broadcasts_unbatched_state_against_batched_flow():
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(
        net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"]
    )
    # A mass-conserving circulation ambient -> A -> B -> ambient, so the steady system is
    # nonsingular; the ensemble varies its magnitude.
    q = torch.tensor([0.3, 0.3, 0.3], dtype=torch.float64) * torch.linspace(
        0.5, 1.5, 5, dtype=torch.float64
    ).unsqueeze(-1)
    sources = torch.tensor([1.0, 2.0], dtype=torch.float64)
    x_boundary = torch.tensor([420.0], dtype=torch.float64)

    out = layer.steady(q, sources, x_boundary)
    assert out.shape == (5, 2)

    expanded = layer.steady(q, sources.expand(5, 2), x_boundary.expand(5, 1))
    assert torch.allclose(out, expanded)
