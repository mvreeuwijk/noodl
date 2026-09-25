"""Parity tests between TransportLayer's new operator-based paths and its retained
dense reference (`operator()`). Every test here compares
the SPARSE result against the DENSE one on the same problem; `tests/layers/
test_transport.py` is untouched and re-verifies the dense/analytic behaviour on
its own.
"""

import math
import time

import pytest
import torch
from torch.autograd import gradcheck

from benchmarks.measure import saved_tensor_bytes
from noodl.layers import transport as transport_module
from noodl.layers.transport import (
    TransportLayer,
    _AffineSystemOperator,
    _expm_action,
    _linear_solve,
    _van_loan_step_dense,
)
from noodl.operators.base import SolveResult, SolverStatus
from noodl.solvers.select import solve as _solve_operator
from noodl.topology import Network


def _full(layer, s_interior, node_dim=-1):
    """Interior-order sources -> FULL node order with zeros on boundary nodes."""
    shape = list(s_interior.shape)
    shape[node_dim] = layer.net.n
    full = torch.zeros(shape, dtype=s_interior.dtype)
    return full.index_copy(node_dim % full.dim(), layer.interior_idx, s_interior)


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


def test_steady_sparse_matches_dense_reference():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"]
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    x_sparse = layer.steady(q, _full(layer, source), c_out)

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
        layer.steady(q, _full(layer, source), x_b)


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
    result = layer.steady(q, _full(layer, source), x_b, on_failure="return")
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
        return layer.steady(q, _full(layer, sources), x_b)

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

    x_adjoint = layer.steady(q, _full(layer, sources), x_b)
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

    Backward memory is measured with `saved_tensor_bytes`, not
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

    Fixture note: a 3-node fixture (`three_node_chain`, capacities
    [50, 8000], `q0 = [0.05, -0.03, 0.02]`) reverses the middle edge, which makes node A a
    pure sink with no outgoing advective edge at all -- the resulting 2x2 M is EXACTLY
    singular (confirmed against the dense `operator()` reference too, independent of
    `AdvectionOperator`), and even with the sign fixed to a natural forward flow, a 2x2
    system is far too small: GMRES's restart cycle length defaults to `min(restart, m) = m`
    for `m = 2`, so ONE full-size cycle always covers the entire Krylov space regardless of
    `rtol`, and `saved_tensor_bytes` (which reflects actual computation, not just the
    reported `iterations` count) is IDENTICAL for both tolerances -- vacuous for exactly the
    reason `tracemalloc` is rejected, just one level deeper. A genuinely restart-bound
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


def test_implicit_step_sparse_matches_dense_reference():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath",
                            boundary=["ambient"], scheme="implicit")
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    dt = 200.0

    x_sparse = layer.step(c0, q, _full(layer, source), c_out, dt)

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
        return layer.step(x, q, _full(layer, sources), x_b, 300.0)

    assert gradcheck(f, (x, q, sources, x_b), eps=1e-6, atol=1e-5)


def test_trapezoidal_step_sparse_matches_dense_reference():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath",
                            boundary=["ambient"], scheme="trapezoidal")
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    dt = 200.0

    x_sparse = layer.step(c0, q, _full(layer, source), c_out, dt)

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
    full_source = _full(layer, torch.zeros(2, dtype=torch.float64))
    for _ in range(10):
        c = layer.step(c, q, full_source, torch.tensor([420.0]), 900.0)
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
    full_source = _full(layer, source)
    with pytest.raises(ValueError, match="on_failure"):
        layer.step(c0, q, full_source, c_out, 300.0, on_failure="bogus")
    with pytest.raises(ValueError, match="on_failure"):
        layer.steady(q, full_source, c_out, on_failure="bogus")


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
    out = _expm_action(op, x0, b0, dt)
    assert out.substeps == 1  # this problem is not stiff at dt=30
    torch.testing.assert_close(out.x, dense, rtol=1e-9, atol=1e-12)


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
    out = _expm_action(op, x0, b0, dt)
    torch.testing.assert_close(out.x, dense, rtol=1e-9, atol=1e-12)
    # zero flow, zero M: x should simply grow linearly in dt from the constant source term
    torch.testing.assert_close(out.x, x0 + dt * b0, rtol=1e-9, atol=1e-12)


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
    out = _expm_action(op, x0, b0, dt)
    assert out.substeps > 1
    torch.testing.assert_close(out.x, dense, rtol=1e-9, atol=1e-12)


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
    out = _expm_action(op, x0, b0, dt)
    torch.testing.assert_close(out.x, dense, rtol=1e-9, atol=1e-12)


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
        out = _expm_action(op, x0, b0, dt)
        torch.testing.assert_close(out.x, dense, rtol=1e-9, atol=1e-12)


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
    full_source = _full(layer, torch.zeros(2, dtype=torch.float64))
    for _ in range(10):
        c = layer.step(c, q, full_source, torch.tensor([420.0]), 900.0)
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
    full_source = _full(layer, source)
    for _ in range(20):
        c = layer.step(c, q, full_source, c_out, 300.0)
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
    # A nonzero source keeps b0 != 0, so `_expm_action`'s mean-diagonal shift -- which makes
    # a genuinely zero-forcing pure decay collapse to one term by design (see
    # test_pure_decay_with_no_forcing_costs_one_term_after_the_diagonal_shift in
    # test_expm_schedule.py) -- does not apply, and this stays a genuinely stiff, multi-
    # substep case (the shift is correct and intentional, not a bug to work
    # around; this test's own intent -- exercising real substepping -- needs forcing now).
    sources = torch.tensor([50.0], dtype=torch.float64)
    dt = 50.0  # dt * rate = 25000: far past the Taylor series' single-step radius

    op = layer._advection_operator(q)
    M, N = layer.operator(q)
    b0 = (N @ xb.unsqueeze(-1)).squeeze(-1) + sources / layer.capacity
    out = _expm_action(op, x0, b0, dt)
    assert out.substeps > 1

    # dx/dt = -500 x + 50: steady state 50/500 = 0.1, and exp(-500*50) underflows to 0.
    expected = x0 * math.exp(-500.0 * dt) + (50.0 / 500.0) * (1.0 - math.exp(-500.0 * dt))
    torch.testing.assert_close(out.x, expected, rtol=1e-6, atol=1e-9)


def test_error_control_refuses_a_step_that_exceeds_the_matvec_budget():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1.0]), flow_kind="airpath", boundary=["ambient"],
        removal=torch.tensor([1e6], dtype=torch.float64),
    )
    q = torch.zeros(2, dtype=torch.float64)
    x0 = torch.tensor([10.0], dtype=torch.float64)
    xb = torch.tensor([0.0], dtype=torch.float64)
    # Nonzero source disables the b0 == 0 diagonal shift (see the test above), so this stays
    # far too stiff for the tiny budget below.
    sources = torch.tensor([1.0], dtype=torch.float64)
    op = layer._advection_operator(q)
    M, N = layer.operator(q)
    b0 = (N @ xb.unsqueeze(-1)).squeeze(-1) + sources / layer.capacity
    with pytest.raises(RuntimeError, match="matvecs"):
        _expm_action(
            op, x0, b0, dt=1e9, max_matvecs=100,
            where=f"TransportLayer '{layer.name}' exact step",
        )


def test_step_passes_the_structural_shift_verdict_computed_from_xb_and_sources(monkeypatch):
    """`b0 = boundary_forcing(x_boundary) + sources / capacity` has `requires_grad=True`
    whenever `q` requires grad, even with `x_boundary` and `sources` constant zero, which
    would silently disable `_expm_action`'s own (value/requires_grad-based) default shift
    inference under training -- a ~139k-matvec budget cliff for what is structurally still an
    exact, one-term pure decay. `step()`'s exact branch must instead pass a `shift` verdict
    computed from `x_boundary`/`sources` themselves, unaffected by whether `q` requires grad.
    Observed by monkeypatching `_expm_action` with a spy that records the `shift` kwarg it
    received and forwards the call unchanged.
    """
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1.0]), flow_kind="airpath", boundary=["ambient"],
        removal=torch.tensor([500.0], dtype=torch.float64),
    )
    q = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    x0 = torch.tensor([10.0], dtype=torch.float64)
    xb = torch.tensor([0.0], dtype=torch.float64)          # constant zero, not grad-tracked
    sources = torch.zeros(1, dtype=torch.float64)          # constant zero, not grad-tracked
    full_source = _full(layer, sources)

    recorded = {}
    real_expm_action = transport_module._expm_action

    def spy(*args, **kwargs):
        recorded["shift"] = kwargs.get("shift")
        return real_expm_action(*args, **kwargs)

    monkeypatch.setattr(transport_module, "_expm_action", spy)
    y = layer.step(x0, q, full_source, xb, 50.0)

    assert recorded["shift"] is True
    assert y.item() == pytest.approx(10.0 * math.exp(-25_000.0), abs=1e-300)


def test_step_records_shift_false_when_sources_require_grad(monkeypatch):
    """Negative case: `sources` requiring grad (even though it is 0 at this call) must record
    `shift=False`. The existing reference (test_transport_derivatives.py) already covers the
    resulting gradient's correctness; this only pins the recorded flag."""
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1.0]), flow_kind="airpath", boundary=["ambient"],
        removal=torch.tensor([500.0], dtype=torch.float64),
    )
    q = torch.zeros(2, dtype=torch.float64)
    x0 = torch.tensor([10.0], dtype=torch.float64)
    xb = torch.tensor([0.0], dtype=torch.float64)
    sources = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    full_source = _full(layer, sources)

    recorded = {}
    real_expm_action = transport_module._expm_action

    def spy(*args, **kwargs):
        recorded["shift"] = kwargs.get("shift")
        return real_expm_action(*args, **kwargs)

    monkeypatch.setattr(transport_module, "_expm_action", spy)
    layer.step(x0, q, full_source, xb, 50.0)

    assert recorded["shift"] is False


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
        return _expm_action(op, x, b0, 300.0).x

    assert gradcheck(f, (x0, q, sources, x_b), eps=1e-6, atol=1e-5)


def test_expm_action_backward_memory_scales_with_substep_count():
    """Measures (does not gate) how backward memory through _expm_action's own unrolled
    iteration grows from a non-stiff case (1 substep) to a genuinely stiff one (several
    substeps). Both `run_and_measure` calls below build `b0` from `xb`/`sources` with
    `requires_grad=True`, so even though their VALUES are 0, `_expm_action`'s default
    (`shift=None`) inference disables the diagonal shift on `b0.requires_grad` alone
    -- unlike test_error_control_triggers_substepping_on_a_stiff_case (which now needs an
    explicit nonzero `sources=50` to stay stiff, since IT calls through a b0 with
    `requires_grad=False`), the stiffness here comes from the grad-tracked zeros, not from
    forcing. Unlike the linear-solve adjoint, no O(state) bound is asserted here: the
    exact scheme is differentiated through its sub-steps, so its backward memory grows with
    them. `tracemalloc` cannot be used: it sees zero bytes of PyTorch allocations on this `.venv`,
    so a tracemalloc-based assertion would pass vacuously regardless of whether more sub-steps
    genuinely cost more backward memory. `saved_tensor_bytes` instead counts exactly the
    tensors autograd will need for backward(), deterministically. The printed numbers are
    what a future composed-model-gate report would compare
    against its timestep-history memory budget; if that budget is ever breached, the remedy
    is checkpointing the sub-steps, not a change to this test.
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
            out = _expm_action(op, x0, b0, dt)
            substeps_holder["substeps"] = out.substeps
            return out.x

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
    test therefore asserts agreement with the standalone result and with the dense reference
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
    # Nonzero source disables the b0 == 0 diagonal shift for both instances (see
    # test_error_control_triggers_substepping_on_a_stiff_case above), keeping the removal=500
    # instance genuinely stiff enough to force substepping under the new norm-based schedule.
    sources = torch.tensor([1.0], dtype=torch.float64)
    x0_val = 10.0

    # Build ONE batched operator: same net/topology, but a per-instance removal rate
    # stacked along a new leading batch dimension -- everything else (flow, transmission,
    # capacity) is shared, un-batched, and broadcasts against it.
    op_batch = layers[1]._advection_operator(q)
    op_batch.removal = torch.stack([layer.removal for layer in layers], dim=0)  # (2, 1, 1)

    x_batch = torch.tensor([[x0_val], [x0_val]], dtype=torch.float64, requires_grad=True)
    b0_batch = (sources / cap).expand(2, 1).clone()

    start = time.perf_counter()
    out = _expm_action(op_batch, x_batch, b0_batch, dt)
    elapsed = time.perf_counter() - start
    print(
        f"test_expm_action_mixed_stiffness_batch_matches_standalone_within_tolerance: "
        f"wall time={elapsed:.4f}s, out.matvecs={out.matvecs} (substeps={out.substeps}, "
        f"terms={out.terms})"
    )
    # No wall-clock assertion here (a wall-clock bound would be flaky, and would guard only
    # this ~16 s action call, not the ~57 s the whole test actually takes with the standalone
    # comparisons and backward passes below) -- the print above is the record.
    assert out.substeps > 1
    sparse = out.x

    x_standalone = [
        torch.tensor([x0_val], dtype=torch.float64, requires_grad=True) for _ in rates
    ]
    standalone_results = []
    for layer, xi in zip(layers, x_standalone, strict=True):
        op = layer._advection_operator(q)  # a FRESH operator per instance, never op_batch
        M, N = layer.operator(q)
        b0 = (N @ xb.unsqueeze(-1)).squeeze(-1) + sources / layer.capacity
        dense = _van_loan_step_dense(M, xi.detach(), b0, dt)
        result_i = _expm_action(op, xi, b0, dt).x
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

    Used by the structural tests: the conduction branch is the one that could build an
    (n, n) `self.L` in `__init__`, and the no-conduction branch the one that could allocate
    an (n, n) block of ZEROS for nothing at all.
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
    # A dense `self.L` is (n, n) and grows 4x per node doubling in the composed-model
    # memory budget, and in the NO-conduction case it would be a block of zeros that
    # operator() subtracts for nothing. Construction must hold no tensor whose shape scales
    # with the full node count `n` -- not just "exactly (n, n)" but ANY 2-D+ tensor with `n`
    # anywhere in its shape, since an (n, b) or (b, n) matrix reintroduces exactly the same O(n)
    # scaling an (n, n) one does; it just isn't square. `transmission` is exempted BY NAME: its own
    # shape is (K, b_flow), and in this fixture's network b_flow (three airpath edges) happens to
    # equal n (three nodes) --  a coincidental collision with the quantity this test guards against,
    # not evidence of a node-count-shaped tensor.
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
        offending = {
            name: tuple(v.shape)
            for name, v in vars(layer).items()
            if name != "transmission"
            and isinstance(v, torch.Tensor)
            and v.dim() >= 2
            and n in v.shape
        }
        assert offending == {}, (
            f"TransportLayer holds a tensor with node count {n} in its shape after "
            f"__init__: {offending}"
        )


def test_operator_reference_still_includes_conduction():
    # The dense reference keeps its exact values. These are the (M, N) this fixture produced
    # with the earlier code (conduction folded in via the construction-time `self.L`),
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
    # capacity), so the assertions above are not merely pinning an advection-only reference.
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
# An unbatched state against a BATCHED flow (one initial condition against an ensemble of
# flow realisations -- the calibration use case) must work: expanding the state only to
# its OWN batch shape rather than to the operator's raises a raw RuntimeError out of
# `AdvectionOperator._raw_action`.
def _ensemble_flow() -> torch.Tensor:
    base = torch.tensor([0.3, -0.2, 0.25], dtype=torch.float64)
    scale = torch.linspace(0.5, 1.5, 5, dtype=torch.float64).unsqueeze(-1)
    return base * scale


def test_matvec_rmatvec_boundary_forcing_broadcast_match_dense_reference_per_instance():
    """The C1 broadcast case (unbatched `x`/`x_b`, ensemble `flow` `(5, b)`), checked against
    the dense `TransportLayer.operator(q)` reference PER INSTANCE -- not merely self-consistent
    with an explicitly-expanded call (that is `test_matvec_broadcasts_an_unbatched_state_
    against_a_batched_flow` in `tests/operators/test_advection.py`), but numerically equal to
    `M_i @ x`, `M_i.T @ x` and `N_i @ x_b` for each of the 5 flow realisations' own dense
    `M_i`, `N_i`.
    """
    net = three_node_chain()
    cap = torch.tensor([50.0, 80.0], dtype=torch.float64)
    layer = TransportLayer(net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"])
    q = _ensemble_flow()  # (5, 3)
    op = layer._advection_operator(q)

    x = torch.tensor([12.0, -4.0], dtype=torch.float64)
    x_b = torch.tensor([420.0], dtype=torch.float64)

    y = op.matvec(x)
    yt = op.rmatvec(x)
    yb = op.boundary_forcing(x_b)
    assert y.shape == (5, 2)
    assert yt.shape == (5, 2)
    assert yb.shape == (5, 2)

    for i in range(5):
        M_i, N_i = layer.operator(q[i])
        torch.testing.assert_close(y[i], M_i @ x, rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(yt[i], M_i.transpose(-1, -2) @ x, rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(yb[i], N_i @ x_b, rtol=1e-9, atol=1e-12)


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
    full_sources = _full(layer, sources)

    out = layer.step(x, q, full_sources, x_boundary, 60.0)
    assert out.shape == (5, 2)

    expanded = layer.step(
        x.expand(5, 2), q, full_sources.expand(5, 3), x_boundary.expand(5, 1), 60.0
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
    full_sources = _full(layer, sources)

    out = layer.steady(q, full_sources, x_boundary)
    assert out.shape == (5, 2)

    expanded = layer.steady(q, full_sources.expand(5, 3), x_boundary.expand(5, 1))
    assert torch.allclose(out, expanded)


# ------------------------------------ K=2, kinetics + removal + conduction, all at once
def _k2_full_layer(scheme: str, linear_solver: str = "auto") -> TransportLayer:
    """ambient (boundary) -- A -- B, three airpath edges plus one conduction edge A->B, TWO
    species with both inter-species kinetics and a per-species removal rate -- every term
    `operator()` assembles (advection, conduction, removal, kinetics) present at once,
    jointly exercised through a batched, mixed-sign flow.
    """
    net = Network(dtype=torch.float64)
    for name in ("ambient", "A", "B"):
        net.add_node(name)
    net.add_edge("ambient", "A", kind="airpath")
    net.add_edge("A", "B", kind="airpath")
    net.add_edge("B", "ambient", kind="airpath")
    net.add_edge("A", "B", kind="conduction")
    kinetics = torch.tensor([[-0.01, 0.02], [0.01, -0.02]], dtype=torch.float64)
    removal = torch.tensor([0.001, 0.002], dtype=torch.float64)
    return TransportLayer(
        net,
        "gas",
        capacity=torch.tensor([50.0, 80.0], dtype=torch.float64),
        flow_kind="airpath",
        boundary=["ambient"],
        n_species=2,
        kinetics=kinetics,
        removal=removal,
        conduction_kind="conduction",
        conductance=torch.tensor([2.5], dtype=torch.float64),
        scheme=scheme,
        linear_solver=linear_solver,
    )


def _batched_mixed_sign_flow() -> torch.Tensor:
    """(4, 3): four instances, each with a mix of positive and negative branch flows."""
    return torch.tensor(
        [
            [0.30, -0.20, 0.25],
            [-0.15, 0.10, -0.05],
            [0.05, -0.30, 0.20],
            [-0.40, 0.35, -0.10],
        ],
        dtype=torch.float64,
    )


def _k2_state(batch: int, n_i: int) -> torch.Tensor:
    torch.manual_seed(0)
    return 50.0 + 10.0 * torch.rand(batch, n_i, 2, dtype=torch.float64)


@pytest.mark.parametrize("scheme", ["implicit", "trapezoidal"])
def test_k2_kinetics_removal_conduction_batched_mixed_sign_step_matches_dense_reference(scheme):
    layer = _k2_full_layer(scheme)
    q = _batched_mixed_sign_flow()
    batch = q.shape[0]
    x = _k2_state(batch, layer.n_i)
    sources = 0.1 * _k2_state(batch, layer.n_i)
    x_boundary = torch.tensor([[10.0, 5.0]], dtype=torch.float64).expand(batch, 1, 2)
    dt = 30.0

    x_sparse = layer.step(x, q, _full(layer, sources, node_dim=-2), x_boundary, dt)
    assert x_sparse.shape == (batch, layer.n_i, 2)

    cap = layer._capacity_stacked(torch.float64)
    for i in range(batch):
        M_i, N_i = layer.operator(q[i])
        x0_i, _ = layer._to_stacked(x[i], layer.n_i, "x")
        src_i, _ = layer._to_stacked(sources[i], layer.n_i, "sources")
        xb_i, _ = layer._to_stacked(x_boundary[i], layer.n_b, "x_boundary")
        b0_i = (N_i @ xb_i.unsqueeze(-1)).squeeze(-1) + src_i / cap
        m = M_i.shape[-1]
        eye = torch.eye(m, dtype=torch.float64)
        if scheme == "implicit":
            rhs = x0_i + dt * b0_i
            x_dense_i = torch.linalg.solve(eye - dt * M_i, rhs.unsqueeze(-1)).squeeze(-1)
        else:  # trapezoidal
            rhs = ((eye + 0.5 * dt * M_i) @ x0_i.unsqueeze(-1)).squeeze(-1) + dt * b0_i
            x_dense_i = torch.linalg.solve(eye - 0.5 * dt * M_i, rhs.unsqueeze(-1)).squeeze(-1)
        x_dense_i_unstacked = layer._from_stacked(x_dense_i, layer.n_i, False)
        torch.testing.assert_close(x_sparse[i], x_dense_i_unstacked, rtol=1e-8, atol=1e-10)


def test_k2_kinetics_removal_conduction_batched_mixed_sign_steady_matches_dense_reference():
    layer = _k2_full_layer("implicit")  # scheme is irrelevant to steady()
    q = _batched_mixed_sign_flow()
    batch = q.shape[0]
    sources = 0.1 * _k2_state(batch, layer.n_i)
    x_boundary = torch.tensor([[10.0, 5.0]], dtype=torch.float64).expand(batch, 1, 2)

    x_sparse = layer.steady(q, _full(layer, sources, node_dim=-2), x_boundary)
    assert x_sparse.shape == (batch, layer.n_i, 2)

    cap = layer._capacity_stacked(torch.float64)
    for i in range(batch):
        M_i, N_i = layer.operator(q[i])
        src_i, _ = layer._to_stacked(sources[i], layer.n_i, "sources")
        xb_i, _ = layer._to_stacked(x_boundary[i], layer.n_b, "x_boundary")
        b0_i = (N_i @ xb_i.unsqueeze(-1)).squeeze(-1) + src_i / cap
        x_dense_i = torch.linalg.solve(M_i, -b0_i.unsqueeze(-1)).squeeze(-1)
        x_dense_i_unstacked = layer._from_stacked(x_dense_i, layer.n_i, False)
        torch.testing.assert_close(x_sparse[i], x_dense_i_unstacked, rtol=1e-8, atol=1e-10)


# ------------------------------------------------------------- linear_solver option
def test_unknown_linear_solver_refused_by_name_at_construction():
    net = flow_through_zone()
    with pytest.raises(ValueError, match="bogus"):
        TransportLayer(
            net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath",
            boundary=["ambient"], linear_solver="bogus",
        )


@pytest.mark.parametrize("name", ["gmres_jacobi", "gmres_ilu"])
def test_gmres_preconditioner_names_match_the_dense_reference(name):
    """`gmres_jacobi`/`gmres_ilu` are real preconditioners. Same problem, dense reference solved by
    hand -- the same comparison
    `test_k2_kinetics_removal_conduction_batched_mixed_sign_step_matches_dense_reference` already
    makes for the default solver, repeated here for both new preconditioner names.
    """
    layer = _k2_full_layer("implicit", linear_solver=name)
    q = _batched_mixed_sign_flow()
    batch = q.shape[0]
    x = _k2_state(batch, layer.n_i)
    sources = 0.1 * _k2_state(batch, layer.n_i)
    x_boundary = torch.tensor([[10.0, 5.0]], dtype=torch.float64).expand(batch, 1, 2)
    dt = 30.0

    x_sparse = layer.step(x, q, _full(layer, sources, node_dim=-2), x_boundary, dt)
    assert x_sparse.shape == (batch, layer.n_i, 2)

    cap = layer._capacity_stacked(torch.float64)
    for i in range(batch):
        M_i, N_i = layer.operator(q[i])
        x0_i, _ = layer._to_stacked(x[i], layer.n_i, "x")
        src_i, _ = layer._to_stacked(sources[i], layer.n_i, "sources")
        xb_i, _ = layer._to_stacked(x_boundary[i], layer.n_b, "x_boundary")
        b0_i = (N_i @ xb_i.unsqueeze(-1)).squeeze(-1) + src_i / cap
        m = M_i.shape[-1]
        eye = torch.eye(m, dtype=torch.float64)
        rhs = x0_i + dt * b0_i
        x_dense_i = torch.linalg.solve(eye - dt * M_i, rhs.unsqueeze(-1)).squeeze(-1)
        x_dense_i_unstacked = layer._from_stacked(x_dense_i, layer.n_i, False)
        torch.testing.assert_close(x_sparse[i], x_dense_i_unstacked, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("name", ["gmres_jacobi", "gmres_ilu"])
def test_gradcheck_gmres_preconditioner_names_through_implicit_step(name):
    """Both names must be differentiable end to end through `layer.step`: gradients come
    from `_LinearSolve`'s implicit adjoint (forward AND backward run under `no_grad`),
    so this proves the adjoint's own resolved kwargs (`method='gmres'`,
    `preconditioner=name.removeprefix('gmres_')`) produce a correct gradient -- not merely
    that the forward value is right.
    """
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit", linear_solver=name,
    )
    x = torch.tensor([150.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.4, 0.4], dtype=torch.float64, requires_grad=True)
    sources = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    x_b = torch.tensor([420.0], dtype=torch.float64, requires_grad=True)

    def f(x, q, sources, x_b):
        return layer.step(x, q, _full(layer, sources), x_b, 300.0)

    assert gradcheck(f, (x, q, sources, x_b), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("name", ["gmres_jacobi", "gmres_ilu"])
def test_diagnostics_linear_reports_the_resolved_preconditioner_backend(name):
    """Both names resolve to backend='gmres' (preconditioning is an internal detail of how
    gmres converges, not a distinct backend name) with a real, positive residual/iteration
    count -- unlike sparse_direct's trivial iterations==1.
    """
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit", linear_solver=name,
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    diag: dict = {}
    layer.step(c0, q, _full(layer, source), c_out, 200.0, diagnostics=diag)
    assert diag["linear"]["backend"] == "gmres"
    assert diag["linear"]["iterations"] >= 1
    assert diag["linear"]["residual"] < 1e-8


def test_gmres_resolves_bit_identical_to_the_bare_select_solve_auto():
    """Regression proof for the EXPLICIT `linear_solver="gmres"` option (`"auto"` differs --
    see the two tests below): builds the exact same affine system `TransportLayer.step` does and
    solves it directly with `solvers.select.solve(method="auto", ...)` and no other kwargs -- this
    operator never certifies SPD (`_AffineSystemOperator.spd_certificate()` is `None`), so
    `select.solve`'s OWN "auto" (a separate policy) always routes it
    to plain gmres too, making this comparison bit-identical either way -- then compares
    against `step()`'s own result under `linear_solver="gmres"`.
    """
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    layer = TransportLayer(
        net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"],
        scheme="implicit", linear_solver="gmres",
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    dt = 200.0

    x_new = layer.step(c0, q, _full(layer, source), c_out, dt)

    op = layer._advection_operator(q.to(torch.float64))
    xb_s, _ = layer._to_stacked(c_out.to(torch.float64), layer.n_b, "x_boundary")
    src_s, _ = layer._to_stacked(source.to(torch.float64), layer.n_i, "sources")
    cap_s = layer._capacity_stacked(torch.float64)
    b0 = op.boundary_forcing(xb_s) + src_s / cap_s
    x0_s, _ = layer._to_stacked(c0.to(torch.float64), layer.n_i, "x")
    rhs = x0_s + dt * b0
    system = _AffineSystemOperator(op, dt)
    x_old = _solve_operator(system, rhs, method="auto", where="regression probe").x

    assert torch.equal(x_new, x_old)


def test_auto_resolves_to_sparse_direct_at_or_under_the_batch_cap():
    """Pinned directly against `_resolve_solver`: the batch-1 and batch-cap
    (`_SPARSE_DIRECT_MAX_BATCH == 32`) rows of `benchmarks/transport_solver_bench.py`
    both favour `sparse_direct` by a wide, spread-clear margin, so `"auto"` resolves to it
    at or under the cap.
    """
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"],
    )
    assert layer._resolve_solver(1) == {
        "method": "sparse_direct", "preconditioner": None, "restart": 30,
    }
    assert layer._resolve_solver(transport_module._SPARSE_DIRECT_MAX_BATCH) == {
        "method": "sparse_direct", "preconditioner": None, "restart": 30,
    }


def test_auto_resolves_to_gmres_above_the_batch_cap():
    """Above `_SPARSE_DIRECT_MAX_BATCH`, `sparse_direct` is not applicable at all (the
    benchmark's e=100 rows exclude it before construction, not tried-then-caught), and the gmres
    family's own internal ordering was not clean enough within the measured spread to prefer a
    preconditioner over plain gmres, so `"auto"` falls back to plain `gmres` -- never
    `gmres_jacobi`/`gmres_ilu`.
    """
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"],
    )
    assert layer._resolve_solver(transport_module._SPARSE_DIRECT_MAX_BATCH + 1) == {
        "method": "gmres", "preconditioner": None, "restart": 30,
    }


def test_implicit_step_sparse_direct_matches_gmres_to_high_precision():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    dt = 200.0
    layer_gmres = TransportLayer(
        net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"],
        scheme="implicit", linear_solver="gmres",
    )
    layer_sd = TransportLayer(
        net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"],
        scheme="implicit", linear_solver="sparse_direct",
    )
    x_gmres = layer_gmres.step(c0, q, _full(layer_gmres, source), c_out, dt)
    x_sd = layer_sd.step(c0, q, _full(layer_sd, source), c_out, dt)
    torch.testing.assert_close(x_sd, x_gmres, rtol=1e-10, atol=1e-10)


def test_trapezoidal_step_sparse_direct_matches_gmres_to_high_precision():
    net = flow_through_zone()
    cap = torch.tensor([1000.0], dtype=torch.float64)
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    dt = 200.0
    layer_gmres = TransportLayer(
        net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"],
        scheme="trapezoidal", linear_solver="gmres",
    )
    layer_sd = TransportLayer(
        net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"],
        scheme="trapezoidal", linear_solver="sparse_direct",
    )
    x_gmres = layer_gmres.step(c0, q, _full(layer_gmres, source), c_out, dt)
    x_sd = layer_sd.step(c0, q, _full(layer_sd, source), c_out, dt)
    torch.testing.assert_close(x_sd, x_gmres, rtol=1e-10, atol=1e-10)


def test_steady_sparse_direct_matches_gmres_to_high_precision():
    net = flow_through_zone()
    cap = torch.tensor([1000.0])
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    layer_gmres = TransportLayer(
        net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"],
        linear_solver="gmres",
    )
    layer_sd = TransportLayer(
        net, "co2", capacity=cap, flow_kind="airpath", boundary=["ambient"],
        linear_solver="sparse_direct",
    )
    x_gmres = layer_gmres.steady(q, _full(layer_gmres, source), c_out)
    x_sd = layer_sd.steady(q, _full(layer_sd, source), c_out)
    torch.testing.assert_close(x_sd, x_gmres, rtol=1e-10, atol=1e-10)


def test_gradcheck_implicit_step_sparse_direct_adjoint_through_superlu():
    """The backward's transposed solve must go through SuperLU too when the forward did --
    proof, not merely a claim, that `_LinearSolve.backward` uses the SAME resolved kwargs as
    `forward`."""
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit", linear_solver="sparse_direct",
    )
    x = torch.tensor([150.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.4, 0.4], dtype=torch.float64, requires_grad=True)
    sources = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    x_b = torch.tensor([420.0], dtype=torch.float64, requires_grad=True)

    def f(x, q, sources, x_b):
        return layer.step(x, q, _full(layer, sources), x_b, 300.0)

    assert gradcheck(f, (x, q, sources, x_b), eps=1e-6, atol=1e-5)


def test_diagnostics_linear_reports_sparse_direct_backend_and_one_iteration():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit", linear_solver="sparse_direct",
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    diag: dict = {}
    layer.step(c0, q, _full(layer, source), c_out, 200.0, diagnostics=diag)
    assert diag["linear"]["backend"] == "sparse_direct"
    assert diag["linear"]["iterations"] == 1
    assert diag["linear"]["residual"] < 1e-10


def test_diagnostics_linear_reports_gmres_backend_and_at_least_one_iteration():
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit",
        linear_solver="gmres",  # the default "auto" resolves to sparse_direct here
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    diag: dict = {}
    layer.step(c0, q, _full(layer, source), c_out, 200.0, diagnostics=diag)
    assert diag["linear"]["backend"] == "gmres"
    assert diag["linear"]["iterations"] >= 1


def test_diagnostics_linear_reports_sparse_direct_backend_under_the_default_auto():
    """The default `linear_solver="auto"` at a small batch (1 instance, well under
    `_SPARSE_DIRECT_MAX_BATCH`) reports `sparse_direct`, not `gmres` -- the companion of
    the test above, which pins the same problem's diagnostics under the explicit spelling.
    """
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit",  # default linear_solver="auto"
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    diag: dict = {}
    layer.step(c0, q, _full(layer, source), c_out, 200.0, diagnostics=diag)
    assert diag["linear"]["backend"] == "sparse_direct"
    assert diag["linear"]["iterations"] == 1
    assert diag["linear"]["residual"] < 1e-10


def test_auto_falls_back_to_gmres_and_warns_once_when_scipy_is_not_importable(monkeypatch):
    """Hazard 1: the one fall-back that is an ENVIRONMENT fault (SciPy, the
    `noodl[sparse]` extra, missing) rather than a modelling or grad-safety fact, so it is
    the only one that warns, and only once per process.
    """
    import builtins

    monkeypatch.setattr(transport_module, "_WARNED_AUTO_NEEDS_SCIPY", False)
    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.split(".")[0] == "scipy":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath", boundary=["ambient"],
    )
    with pytest.warns(RuntimeWarning, match="scipy"):
        resolved = layer._resolve_solver(1)
    assert resolved == {"method": "gmres", "preconditioner": None, "restart": 30}
    # Once per process: a second resolution under the same missing-scipy condition must not
    # warn again.
    import warnings as warnings_module
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error")
        again = layer._resolve_solver(1)
    assert again == {"method": "gmres", "preconditioner": None, "restart": 30}


def test_auto_falls_back_to_gmres_for_a_grad_requiring_on_failure_return_solve():
    """Hazard 2: `steady`/`_implicit_step_sparse`/`_trapezoidal_step_sparse`'s
    `on_failure="return"` paths call `_resolve_solver` OUTSIDE `_LinearSolve`'s `no_grad`
    (unlike the differentiable forward/backward path, which always resolves under
    `no_grad`). An explicit `method="sparse_direct"` reaching `solvers.select.solve` there
    would raise `RuntimeError` when grad is enabled and an input requires grad (see
    `select._sparse_direct`) -- an eligibility refusal `on_failure="return"` does not catch.
    `"auto"` must resolve to `gmres` instead so `step(..., on_failure="return")` with a
    grad-requiring `q` neither raises nor silently detaches the answer.
    """
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit", linear_solver="auto",
    )
    q = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
    c0 = torch.tensor([100.0], dtype=torch.float64)
    source = torch.tensor([2.0], dtype=torch.float64)
    c_out = torch.tensor([420.0], dtype=torch.float64)
    result = layer.step(c0, q, _full(layer, source), c_out, 200.0, on_failure="return")
    assert isinstance(result, SolveResult)
    assert bool(torch.all(result.converged))
    assert result.x.requires_grad, "gmres, not sparse_direct, must have run so grad survives"


def test_differentiable_step_builds_the_system_exactly_once_per_pass(monkeypatch):
    """Regression guard: resolving `linear_solver` must not need a separate probe
    `build_system` call (purely to learn `rhs.shape[:-1].numel()`) before invoking
    `_linear_solve`, which would double the cost of `_advection_operator`'s own
    `boundary_forcing` (an embed/gather/`scatter_add_` over every edge, the same cost order
    as a matvec) on the differentiable path. Counts
    `TransportLayer._advection_operator` invocations directly: `build_system` calls it
    exactly once per invocation (for a fixed, non-changing capacity), so this is a direct
    proxy for `build_system`'s own call count.

    FORWARD must call it exactly ONCE (never a probe-and-solve pair). BACKWARD calls it
    TWICE, not once -- `_LinearSolve.backward` rebuilds the system twice by design: once for the
    adjoint solve's operator (`op`, under `no_grad`) and once more for the residual pass that
    produces the parameter gradients (`op_p`, under `enable_grad`), per `solvers/implicit.py`'s
    `_Implicit` structure this class mirrors. So a correct forward+backward pass totals THREE calls
    (1 + 2) -- not two, and a probe-based implementation would make it FOUR (2 + 2) by adding a
    redundant probe call in forward alone.
    """
    net = flow_through_zone()
    layer = TransportLayer(
        net, "co2", capacity=torch.tensor([1000.0]), flow_kind="airpath",
        boundary=["ambient"], scheme="implicit",
    )
    x = torch.tensor([150.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.4, 0.4], dtype=torch.float64, requires_grad=True)
    sources = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    x_b = torch.tensor([420.0], dtype=torch.float64, requires_grad=True)

    calls = {"n": 0}
    real_advection_operator = TransportLayer._advection_operator

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return real_advection_operator(self, *args, **kwargs)

    monkeypatch.setattr(TransportLayer, "_advection_operator", counting)

    y = layer.step(x, q, _full(layer, sources), x_b, 300.0)
    assert calls["n"] == 1  # forward: exactly one build, no probe

    y.sum().backward()
    assert calls["n"] == 3  # + backward's own two (adjoint op, residual-pass op_p)
