"""Bounded independent checks of the sparse report; no production code changes.

Run from the repository root: .venv/Scripts/python benchmarks/sparse_review_checks.py
CPU-only. Timings are local observations, not GPU predictions.
"""

from __future__ import annotations

import time

import sparse_scaling as original
import torch
from torch.utils.benchmark import Timer

from noodl.elements.conductance import Conductance
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network


def shared_incidence(edge_index, n, layout):
    # Construct directly from edges: no dense n-by-b temporary.
    src, dst = edge_index
    e = torch.arange(src.numel())
    rows = torch.cat((src, dst))
    cols = torch.cat((e, e))
    values = torch.cat((torch.ones_like(src), -torch.ones_like(dst))).double()
    keep = rows != 0
    A = torch.sparse_coo_tensor(
        torch.stack((rows[keep] - 1, cols[keep])), values[keep],
        (n - 1, src.numel()),
    ).coalesce()
    AT = A.transpose(0, 1).coalesce()
    if layout == "csr":
        A, AT = A.to_sparse_csr(), AT.to_sparse_csr()
    return A, AT


def shared_mv(A, AT, g, x):
    # The batch is a set of dense RHS columns; g differs in every column.
    d = torch.sparse.mm(AT, x.T)
    return torch.sparse.mm(A, g.T * d).T


def checks():
    torch.manual_seed(20260912)
    torch.set_num_threads(1)
    print("environment", torch.__version__, "CUDA", torch.cuda.is_available(), flush=True)
    print("spsolve", original.try_spsolve_once(), flush=True)
    dense = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    S = dense.to_sparse_csr()
    v = torch.ones(2, 3, 1, dtype=torch.float64)
    for name, fn in (("@", lambda: S @ v), ("bmm", lambda: torch.bmm(S, v)),
                     ("sparse.mm", lambda: torch.sparse.mm(S, v))):
        try:
            print("batched CSR", name, "WORKS", fn().shape)
        except Exception as exc:
            print("batched CSR", name, type(exc).__name__, str(exc).splitlines()[0][:160])

    ei = original.build_graph(6, 6)
    for layout in ("coo", "csr"):
        A, AT = shared_incidence(ei, 6, layout)
        g = original.random_conductances(2, ei.shape[1], 2, 10).requires_grad_()
        x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
        def fn(g, x, A=A, AT=AT):
            return shared_mv(A, AT, g, x)
        torch.testing.assert_close(fn(g, x), original.matvec_gather_scatter(ei, g, 6, x))
        print("shared", layout, "different g per batch: gradcheck",
              torch.autograd.gradcheck(fn, (g, x)), flush=True)

    print("matvec median milliseconds, fixed topology; construction excluded", flush=True)
    for n, B in ((300, 100), (300, 1000), (1000, 100)):
        ei = original.build_graph(n, n)
        g = original.random_conductances(B, ei.shape[1], 6, n + B)
        x = torch.randn(B, n - 1, dtype=torch.float64)
        coo, coot = shared_incidence(ei, n, "coo")
        csr, csrt = shared_incidence(ei, n, "csr")
        funcs = {
            "gather_scatter": lambda ei=ei, g=g, n=n, x=x: original.matvec_gather_scatter(
                ei, g, n, x
            ),
            "shared_coo": lambda coo=coo, coot=coot, g=g, x=x: shared_mv(coo, coot, g, x),
            "shared_csr": lambda csr=csr, csrt=csrt, g=g, x=x: shared_mv(csr, csrt, g, x),
        }
        reference = funcs["gather_scatter"]()
        for fn in funcs.values():
            torch.testing.assert_close(fn(), reference, atol=1e-9, rtol=1e-10)
        for threads in (1, 14):
            timings = {name: round(1000 * Timer("fn()", globals={"fn": fn},
                       num_threads=threads).blocked_autorange(min_run_time=0.2).median, 4)
                       for name, fn in funcs.items()}
            print(n, B, "threads", threads, timings, flush=True)

    # Audit original stopping criterion with a separately computed true residual.
    torch.set_num_threads(1)
    n, B = 300, 100
    ei = original.build_graph(n, 999)
    rhs = torch.randn(B, n - 1, dtype=torch.float64)
    print(
        "spread: decades, method, iterations, true max relative residual, failed batches",
        flush=True,
    )
    for decades in (0, 2, 6, 8):
        g = original.random_conductances(B, ei.shape[1], decades, 7)

        def mv(x, g=g):
            return original.matvec_gather_scatter(ei, g, n, x)

        inv_diag = original.jacobi_diag(ei, g, n).reciprocal()
        for name, pre in (("CG", None), ("PCG", lambda r, inv_diag=inv_diag: r * inv_diag)):
            t0 = time.perf_counter()
            sol, it = original.cg_solve(mv, rhs, precond=pre, max_iter=3000)
            rel = (rhs - mv(sol)).abs().amax(-1) / rhs.abs().amax(-1)
            print(decades, name, it, rel.max().item(), int((rel >= 1e-6).sum()),
                  "seconds", round(time.perf_counter() - t0, 3), flush=True)

    # An eight-decade ratio alone does not imply a hard preconditioned problem.
    ei = torch.tensor([[0, 0], [1, 2]])
    g = torch.tensor([[1.0, 1e-8]], dtype=torch.float64)
    rhs = torch.ones(1, 2, dtype=torch.float64)
    def mv(x):
        return original.matvec_gather_scatter(ei, g, 3, x)

    sol, it = original.cg_solve(mv, rhs, precond=lambda r: r / g)
    print("8-decade grounded star PCG", it, "residual", (rhs - mv(sol)).abs().max().item())

    # Structural connectivity across a batch is weaker than connectivity per instance.
    net = Network(dtype=torch.float64)
    for node in range(3):
        net.add_node(node)
    net.add_edge(0, 1, kind="air")
    net.add_edge(1, 2, kind="air")
    layer = PotentialFlowLayer(net, "review", [Conductance(torch.ones(2), kind="air")],
                               boundary=[0])
    k = torch.tensor([[1., 1.], [0., 1.]], dtype=torch.float64)
    print("floating group check for connected + disconnected batch:",
          layer._floating_group_nodes(k), flush=True)
    J = original.dense_jacobian(original.dense_incidence_interior(
        torch.tensor([[0, 1], [1, 2]]), 3), k)
    print("per-instance minimum eigenvalues", torch.linalg.eigvalsh(J)[:, 0].tolist())

    # Retain the original dense assembly but chunk the ensemble: bounded intermediates.
    n, B, chunk = 300, 1000, 32
    ei = original.build_graph(n, n)
    g = original.random_conductances(B, ei.shape[1], 6, n + B)
    rhs = torch.randn(B, n - 1, dtype=torch.float64)
    A = original.dense_incidence_interior(ei, n)
    t0 = time.perf_counter()
    out = torch.cat([torch.linalg.solve(original.dense_jacobian(A, g[i:i + chunk]),
                     rhs[i:i + chunk].unsqueeze(-1)).squeeze(-1)
                     for i in range(0, B, chunk)])
    rel = (rhs - original.matvec_gather_scatter(ei, g, n, out)).abs().amax(-1) / rhs.abs().amax(-1)
    print("chunked original dense n=300 B=1000 chunk=32", "seconds",
          round(time.perf_counter() - t0, 3), "max relative residual", rel.max().item(),
          "weighted temporary MB", chunk * (n - 1) * ei.shape[1] * 8 / 1e6, flush=True)

    # Sparse-direct CPU fallback is a separate backend decision from tensor storage.
    import scipy
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    A_sp = sp.csc_matrix(A.numpy())
    t0 = time.perf_counter()
    out_sp = torch.from_numpy(__import__("numpy").stack([
        spla.spsolve(A_sp @ sp.diags(g[i].numpy()) @ A_sp.T, rhs[i].numpy())
        for i in range(B)
    ]))
    rel = (rhs - original.matvec_gather_scatter(ei, g, n, out_sp)).abs().amax(
        -1
    ) / rhs.abs().amax(-1)
    print("SciPy", scipy.__version__, "sparse direct n=300 B=1000", "seconds",
          round(time.perf_counter() - t0, 3), "max relative residual", rel.max().item(),
          "sequential CPU loop; no custom backward in this check", flush=True)


if __name__ == "__main__":
    checks()
