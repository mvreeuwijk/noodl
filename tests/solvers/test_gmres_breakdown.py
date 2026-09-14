"""Regression tests for gmres near-breakdown (Task 8b).

A multi-species `TransportLayer` operator is block diagonal with `K` identical blocks, so
its minimal polynomial has degree `n_i`, not `K * n_i`: the Arnoldi process reaches an
INVARIANT Krylov subspace after `n_i` steps and `h_next` collapses to rounding noise there
(measured: `h_next / ||A v_k|| ~ 2e-16`, i.e. about `eps`, never anywhere near
`finfo.tiny`). Treating that as a live Arnoldi direction corrupts the basis and made
ordinary three-node, two- and three-species steady solves return MAX_ITER/SINGULAR with
relative residuals around 0.6 on a system whose condition number is 2.44.
"""

import pytest
import torch

from tellegen.layers.transport import TransportLayer
from tellegen.operators.base import SolverStatus
from tellegen.operators.dense import DenseOperator
from tellegen.solvers.iterative import gmres
from tellegen.topology import Network


def _three_node_network() -> Network:
    """ambient -> A -> B -> ambient, one airpath edge each."""
    net = Network(dtype=torch.float64)
    net.add_node("ambient")
    net.add_node("A")
    net.add_node("B")
    net.add_edge("ambient", "A", kind="airpath")
    net.add_edge("A", "B", kind="airpath")
    net.add_edge("B", "ambient", kind="airpath")
    return net


def _reference_steady(layer: TransportLayer, q: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
    """`torch.linalg.solve` on the assembled stacked system, in the layer's own x layout."""
    op = layer._advection_operator(q)
    xb_s, _ = layer._to_stacked(xb, layer.n_b, "x_boundary")
    rhs = -op.boundary_forcing(xb_s)
    x_stacked = torch.linalg.solve(op.assemble(), rhs)
    if layer.n_species == 1:
        return x_stacked
    return x_stacked.reshape(layer.n_species, layer.n_i).transpose(0, 1)


@pytest.mark.parametrize("n_species", [1, 2, 3])
def test_steady_flow_sweep_matches_direct_solve_at_every_species_count(n_species):
    """60 ordinary flow magnitudes x K = 1, 2, 3 species, every solve exact.

    Before the fix this swept 0/60, 7/60 and 13/60 failures at K = 1, 2, 3 -- the K > 1
    ones raising `RuntimeError ... statuses ['MAX_ITER'] / ['SINGULAR']` with residuals
    around 0.6.
    """
    net = _three_node_network()
    layer = TransportLayer(
        net,
        "co2",
        capacity=torch.tensor([50.0, 80.0], dtype=torch.float64),
        flow_kind="airpath",
        boundary=["ambient"],
        n_species=n_species,
    )
    src_shape = (net.n, n_species) if n_species > 1 else (net.n,)
    xb_shape = (1, n_species) if n_species > 1 else (1,)
    sources = torch.zeros(src_shape, dtype=torch.float64)
    xb = torch.full(xb_shape, 1e-3, dtype=torch.float64)

    for i in range(60):
        w = 0.005 + i * (0.0345 - 0.005) / 59
        q = torch.full((3,), w, dtype=torch.float64)
        x = layer.steady(q, sources, xb)
        torch.testing.assert_close(
            x, _reference_steady(layer, q, xb), rtol=1e-9, atol=1e-12,
            msg=lambda s, w=w: f"flow magnitude w = {w!r}, n_species = {n_species}: {s}",
        )


def test_gmres_repeated_spectrum_converges_within_m_iterations():
    """`block_diag(A0, A0)` with the same right-hand side in both blocks: m = 6 unknowns,
    but the minimal polynomial has degree 3, so the Krylov space is invariant after 3
    Arnoldi steps.

    GMRES terminates in at most `m` steps in exact arithmetic, so the default
    `max_iter = m` must be enough. Before the fix this returned status MAX_ITER with a
    relative residual of 6.25e-01 and x off by 2.86e-01 in the max norm: step 3 produced
    `h_next` at rounding level, which was divided through anyway, and the degenerate
    diagonal that left in the rotated Hessenberg amplified the back-substitution.
    """
    A0 = torch.tensor(
        [[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]], dtype=torch.float64
    )
    A = torch.block_diag(A0, A0)
    m = A.shape[-1]
    b = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)

    result = gmres(DenseOperator(A, symmetric=False), b)  # max_iter defaults to m
    assert bool(result.converged)
    assert int(result.status) == int(SolverStatus.CONVERGED)
    assert int(result.iterations) <= m
    torch.testing.assert_close(result.x, torch.linalg.solve(A, b), rtol=1e-9, atol=1e-12)


def test_gmres_breakdown_is_per_instance_and_leaves_siblings_untouched():
    """Breakdown freezes ONE instance's Arnoldi, not the batch's.

    Instance 0 breaks down at step 3 (repeated spectrum), instance 1 needs all 6 steps,
    instance 2 has b = 0 and is already converged before any Arnoldi work happens. Each
    must come out of the batched solve exactly as it comes out of its own solve, and the
    frozen instance's x must never be written at all.
    """
    torch.manual_seed(3)
    A0 = torch.tensor(
        [[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]], dtype=torch.float64
    )
    A_break = torch.block_diag(A0, A0)
    A_full = torch.eye(6, dtype=torch.float64) + 0.4 * torch.randn(6, 6, dtype=torch.float64)
    A_frozen = 2.0 * torch.eye(6, dtype=torch.float64)
    As = [A_break, A_full, A_frozen]
    bs = [
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64),
        torch.randn(6, dtype=torch.float64),
        torch.zeros(6, dtype=torch.float64),
    ]

    batched = gmres(DenseOperator(torch.stack(As), symmetric=False), torch.stack(bs))
    assert bool(torch.all(batched.converged))
    assert batched.iterations.tolist() == [3, 6, 0]
    assert bool(torch.all(batched.x[2] == 0.0))  # frozen at iteration 0, never written

    for i, (A_i, b_i) in enumerate(zip(As, bs, strict=True)):
        single = gmres(DenseOperator(A_i, symmetric=False), b_i)
        assert int(single.iterations) == int(batched.iterations[i])
        torch.testing.assert_close(batched.x[i], single.x, rtol=1e-9, atol=1e-12)


def test_steady_gradients_survive_the_breakdown_at_two_species():
    """`gmres` runs inside `_LinearSolve`, which takes its gradients from the implicit
    adjoint (forward AND adjoint solve under `no_grad`), so truncating the Arnoldi cycle at
    a breakdown cannot perturb them -- but the adjoint solve hits the SAME repeated
    spectrum on `A^T`, so it has to survive the breakdown too. `w = 0.0055` is the
    magnitude the sweep above used to fail at.
    """
    net = _three_node_network()
    layer = TransportLayer(
        net,
        "co2",
        capacity=torch.tensor([50.0, 80.0], dtype=torch.float64),
        flow_kind="airpath",
        boundary=["ambient"],
        n_species=2,
    )
    sources = torch.zeros(net.n, 2, dtype=torch.float64)
    q = torch.full((3,), 0.0055, dtype=torch.float64, requires_grad=True)
    xb = torch.full((1, 2), 1e-3, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda q_, xb_: layer.steady(q_, sources, xb_), (q, xb),
        eps=1e-7, atol=1e-6, rtol=1e-3,
    )
