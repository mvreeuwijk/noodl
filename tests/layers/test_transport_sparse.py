"""Parity tests between TransportLayer's new operator-based paths and its retained
dense oracle (`operator()`, unchanged since milestone 1). Every test here compares
the SPARSE result against the DENSE one on the same problem; `tests/layers/
test_transport.py` is untouched and re-verifies the dense/analytic behaviour on
its own.
"""

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
