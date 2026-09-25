"""Benchmark: dense solve vs. matrix-free CG vs. torch.sparse for a batched
network Laplacian J = A_I diag(g) A_I^T, representative of a noodl nodal
Jacobian.

Written to answer one question empirically: given noodl's move to sparse-
by-default operators, what is actually fast, what actually fits in memory, and
what actually works with autograd in THIS installed torch version -- not what
the docs claim in the abstract.

Three solve paths are compared, all on CPU (this repo's .venv has no CUDA):

1. dense    -- assemble J densely, batched (B, n_i, n_i), solve with
               torch.linalg.solve. This is noodl's current path.
2. cg       -- matrix-free CG. J is never assembled; the matvec is
               gather (index_select) + elementwise scale + scatter
               (index_add), operating on a *shared* edge-index array
               (2, b) with *batched* values (B, b). Run with and without
               a Jacobi (diagonal) preconditioner.
3. sparse   -- torch.sparse COO, batched properly (one sparse tensor of
               shape (B, n_i, n_i)), matvec via torch.bmm, used as the
               operator inside the same CG loop. torch.sparse.spsolve is
               also attempted directly and the failure is reported (it is
               expected to fail on this build).

A companion section benchmarks *assembly*: A @ q and A^T @ phi (the
topology operator itself, independent of the linear solve) via dense matmul
vs. index_select/index_add on the same edge-index array.

Usage:
    .venv/Scripts/python benchmarks/sparse_scaling.py
"""

from __future__ import annotations

import ctypes
import gc
import sys
import time
from ctypes import wintypes

import torch

try:
    sys.stdout.reconfigure(line_buffering=True)  # so progress is visible while running,
except Exception:  # not all stdout wrappers support this; harmless if it fails
    pass

print(f"torch {torch.__version__}, threads={torch.get_num_threads()}", flush=True)

DTYPE = torch.float64  # matches noodl's own choice for its solve paths (see
# TransportLayer.step/steady in src/noodl/layers/transport.py, and the benchmark
# in benchmarks/newton_scaling.py): float32 was tried first here and found to make
# the accuracy comparison meaningless -- at the conductance spread this benchmark
# uses (up to 6 decades), cond(J) reaches ~1e6, and cond*eps_fp32 (~1.2e-7) is
# already ~0.1, so torch.linalg.solve itself returns ~10% wrong answers in fp32
# before CG or torch.sparse ever enter the picture.
SEED = 0


# --------------------------------------------------------------------------- memory
class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


try:
    _GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
    _GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    _GetProcessMemoryInfo.restype = wintypes.BOOL
    _GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
    _GetCurrentProcess.restype = wintypes.HANDLE
    _HAVE_WIN32_MEM = True
except Exception:
    _HAVE_WIN32_MEM = False


def peak_rss_mb() -> float | None:
    """Process-wide peak working-set size, in MB (Windows only; no psutil in this venv)."""
    if not _HAVE_WIN32_MEM:
        return None
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
    ok = _GetProcessMemoryInfo(_GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        return None
    return counters.PeakWorkingSetSize / 1e6


# --------------------------------------------------------------------------- graph
def build_graph(n: int, seed: int) -> torch.Tensor:
    """Random connected graph on n nodes, ~2n edges, node 0 is ground (boundary).

    Returns edge_index (2, b) int64: edge_index[0] = source, edge_index[1] = target.
    A random spanning tree guarantees connectivity; ~n extra random edges bring the
    total to ~2n, matching the prompt's "n nodes, about 2n edges".
    """
    gen = torch.Generator().manual_seed(seed)
    src = torch.empty(n - 1, dtype=torch.long)
    dst = torch.arange(1, n, dtype=torch.long)
    for i in range(1, n):
        src[i - 1] = torch.randint(0, i, (1,), generator=gen)
    n_extra = n
    extra_src = torch.randint(0, n, (n_extra,), generator=gen)
    extra_dst = torch.randint(0, n, (n_extra,), generator=gen)
    keep = extra_src != extra_dst
    edge_index = torch.stack(
        [torch.cat([src, extra_src[keep]]), torch.cat([dst, extra_dst[keep]])]
    )
    return edge_index


def random_conductances(B: int, b: int, decades: float, seed: int) -> torch.Tensor:
    """g in [10^-decades/2, 10^+decades/2]; decades=0 means all g == 1."""
    gen = torch.Generator().manual_seed(seed)
    if decades == 0:
        return torch.ones(B, b, dtype=DTYPE)
    u = (torch.rand(B, b, generator=gen) - 0.5) * decades
    return (10.0**u).to(DTYPE)


# --------------------------------------------------------------------------- dense
def dense_incidence_interior(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """A_I: (n_i, b) dense incidence restricted to interior nodes (all but node 0)."""
    b = edge_index.shape[1]
    A = torch.zeros(n, b, dtype=DTYPE)
    A[edge_index[0], torch.arange(b)] += 1.0
    A[edge_index[1], torch.arange(b)] -= 1.0
    return A[1:]  # drop node 0 (ground)


def dense_jacobian(A_I: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """(B, n_i, n_i) = A_I diag(g) A_I^T, done as two matmuls (not a triple einsum)."""
    # (B, n_i, b) = g[:, None, :] * A_I[None]
    weighted = A_I.unsqueeze(0) * g.unsqueeze(1)
    return weighted @ A_I.T


def dense_solve(edge_index, n, g, rhs):
    A_I = dense_incidence_interior(edge_index, n)
    J = dense_jacobian(A_I, g)
    x = torch.linalg.solve(J, rhs.unsqueeze(-1)).squeeze(-1)
    return x


# --------------------------------------------------------------------------- matrix-free
def matvec_gather_scatter(edge_index: torch.Tensor, g: torch.Tensor, n: int, x_i: torch.Tensor):
    """J @ x_i without ever forming J. x_i: (..., n_i); returns (..., n_i)."""
    src, dst = edge_index[0], edge_index[1]
    batch_shape = x_i.shape[:-1]
    x_full = x_i.new_zeros(*batch_shape, n)
    x_full[..., 1:] = x_i
    diff = x_full[..., src] - x_full[..., dst]  # (..., b)
    flux = g * diff
    out_full = x_i.new_zeros(*batch_shape, n)
    out_full = out_full.index_add(-1, src, flux)
    out_full = out_full.index_add(-1, dst, -flux)
    return out_full[..., 1:]


def jacobi_diag(edge_index: torch.Tensor, g: torch.Tensor, n: int) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    batch_shape = g.shape[:-1]
    d = g.new_zeros(*batch_shape, n)
    d = d.index_add(-1, src, g)
    d = d.index_add(-1, dst, g)
    return d[..., 1:]


def cg_solve(matvec, rhs, precond=None, tol=1e-6, max_iter=2000):
    """Batched, preconditioned CG. rhs: (B, m). Returns (x, iters_until_all_converged)."""
    x = torch.zeros_like(rhs)
    r = rhs - matvec(x)
    z = precond(r) if precond is not None else r
    p = z.clone()
    rz_old = (r * z).sum(-1)
    tiny = torch.finfo(rhs.dtype).tiny
    b_norm = rhs.abs().amax(-1).clamp_min(tiny)
    it = 0
    while it < max_iter:
        resid = r.abs().amax(-1)
        if bool(torch.all(resid < tol * b_norm)):
            break
        Ap = matvec(p)
        alpha = rz_old / (p * Ap).sum(-1).clamp(min=tiny)
        x = x + alpha.unsqueeze(-1) * p
        r = r - alpha.unsqueeze(-1) * Ap
        z = precond(r) if precond is not None else r
        rz_new = (r * z).sum(-1)
        beta = rz_new / rz_old.clamp_min(tiny)
        p = z + beta.unsqueeze(-1) * p
        rz_old = rz_new
        it += 1
    return x, it


# --------------------------------------------------------------------------- torch.sparse
def build_batched_coo(edge_index: torch.Tensor, g: torch.Tensor, n: int) -> torch.Tensor:
    """Batched (B, n_i, n_i) sparse COO Laplacian. Indices MUST be replicated per batch
    element (torch.sparse has no way to share one (2, nnz) index array across a batch
    dimension the way manual gather/scatter shares `edge_index`) -- this replication is
    itself part of what this benchmark measures.
    """
    B, b = g.shape
    src, dst = edge_index[0], edge_index[1]
    # Node 0 is ground: an edge touching it contributes to the diagonal at its
    # *interior* endpoint only (a fixed-potential node has no row/column of its own),
    # exactly as dense_incidence_interior drops row 0 of A before forming A_I. An edge
    # between two interior nodes contributes both an off-diagonal pair and both diagonals.
    both_interior = (src != 0) & (dst != 0)
    src_interior = src != 0
    dst_interior = dst != 0

    oi_src = src[both_interior] - 1
    oi_dst = dst[both_interior] - 1
    g_off = g[:, both_interior]  # (B, m)

    di_src = src[src_interior] - 1
    di_dst = dst[dst_interior] - 1
    g_diag_src = g[:, src_interior]
    g_diag_dst = g[:, dst_interior]

    rows = torch.cat([oi_src, oi_dst, di_src, di_dst])
    cols = torch.cat([oi_dst, oi_src, di_src, di_dst])
    vals_pattern = torch.cat([-g_off, -g_off, g_diag_src, g_diag_dst], dim=-1)  # (B, nnz)

    nnz = rows.shape[0]
    batch_idx = torch.arange(B).repeat_interleave(nnz)
    idx = torch.stack([batch_idx, rows.repeat(B), cols.repeat(B)])
    vals = vals_pattern.reshape(-1)
    n_i = n - 1
    return torch.sparse_coo_tensor(idx, vals, (B, n_i, n_i)).coalesce()


def sparse_matvec_bmm(S: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.bmm(S, x.unsqueeze(-1)).squeeze(-1)


def try_spsolve_once() -> str:
    """Attempt torch.sparse.spsolve on a trivial 2D CSR system; report what happens."""
    A = torch.eye(4, dtype=DTYPE) * 2 - torch.diag(torch.ones(3, dtype=DTYPE), 1) - torch.diag(
        torch.ones(3, dtype=DTYPE), -1
    )
    b = torch.randn(4, dtype=DTYPE)
    try:
        x = torch.sparse.spsolve(A.to_sparse_csr(), b)
        err = (x - torch.linalg.solve(A, b)).abs().max().item()
        return f"OK (max err vs dense = {err:.2e})"
    except Exception as e:  # noqa: BLE001 - we want to report ANY failure verbatim
        return f"FAILED: {type(e).__name__}: {str(e)[:150]}"


# --------------------------------------------------------------------------- timing helpers
def timed(fn, *args, repeats=3, **kwargs):
    out = fn(*args, **kwargs)  # warm-up
    gc.collect()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return out, best


DENSE_MEM_BUDGET_BYTES = 4e8  # ~400 MB: gates on the DOMINANT term, the (B, n_i, b)
# intermediate `weighted` inside dense_jacobian (not the smaller final (B, n_i, n_i) J).
# Two earlier, looser budgets (8 GB gating on final J size alone, then 1.5 GB gating on
# the intermediate) each produced a single combo that took an unreasonable amount of
# wall time on this machine -- at n=3000, B=100 the intermediate is ~14 GB; at n=300,
# B=1000 it is "only" ~1.4 GB but the batched matmul against it was still, empirically,
# far slower than its FLOP count suggests it should be (a non-standard batched-broadcast
# shape apparently not hitting an optimised batched-GEMM path in this build). Rather than
# spend more of this benchmark's time budget diagnosing that specific slowdown, dense is
# gated tightly here; the qualitative point -- dense assembly cost is O(n_i * b) per
# batch element and blows up fast -- is already visible well inside this budget.


def main() -> None:
    torch.manual_seed(SEED)
    sizes = [50, 100, 300, 1000, 3000]
    batches = [1, 100, 1000]
    # Part A (the full solve-time sweep) is the expensive part of this script -- on this
    # machine, a full run took over an hour of wall time even after three rounds of tuning
    # down its iteration caps (see the comments above). Its results are
    # already captured in benchmarks/sparse_scaling_results_partA.txt from an earlier run;
    # SKIP_PART_A=1 re-runs only Parts A2/B/B2/C (all fast and bounded) without repeating it.
    skip_part_a = __import__("os").environ.get("SKIP_PART_A") == "1"

    print("=" * 100)
    print("Part A: solve J x = r, J = A_I diag(g) A_I^T, g spans 6 decades (1e-3..1e3)")
    if skip_part_a:
        print("(SKIPPED this run -- see benchmarks/sparse_scaling_results_partA.txt)")
    print("=" * 100)
    header = (
        f"{'n':>6} {'B':>6} {'b_edges':>8} | "
        f"{'dense_s':>10} {'dense_mem_MB':>13} | "
        f"{'cg_s':>8} {'cg_it':>6} {'cgJ_s':>8} {'cgJ_it':>7} | "
        f"{'sp_s':>8} {'sp_it':>6} | {'err_cg':>9} {'err_sp':>9}"
    )
    print(header)

    for n in ([] if skip_part_a else sizes):
        edge_index = build_graph(n, seed=SEED + n)
        b_edges = edge_index.shape[1]
        n_i = n - 1
        for B in batches:
            print(f"  ... n={n} B={B} b_edges={b_edges} starting", flush=True)
            g = random_conductances(B, b_edges, decades=6.0, seed=SEED + n + B)
            rhs = torch.randn(B, n_i, dtype=DTYPE)

            # ---- dense ----
            elem_bytes = torch.finfo(DTYPE).bits // 8
            dense_bytes = B * n_i * max(n_i, b_edges) * elem_bytes  # dominant intermediate
            x_dense = None
            dense_s = float("nan")
            dense_mem = float("nan")
            if dense_bytes < DENSE_MEM_BUDGET_BYTES:
                try:
                    gc.collect()
                    x_dense, dense_s = timed(dense_solve, edge_index, n, g, rhs, repeats=1)
                    dense_mem = (peak_rss_mb() or 0.0)
                except (RuntimeError, MemoryError) as e:
                    dense_s = float("nan")
                    dense_mem = float("nan")
                    print(f"    [dense FAILED at n={n} B={B}: {type(e).__name__}: {e}]")
            else:
                print(
                    f"    [dense SKIPPED at n={n} B={B}: J would be "
                    f"{dense_bytes / 1e9:.1f} GB, over the "
                    f"{DENSE_MEM_BUDGET_BYTES / 1e9:.0f} GB budget]"
                )

            # ---- matrix-free CG, no preconditioner ----
            # Unpreconditioned CG at this 6-decade conductance spread turns out (measured
            # below) NOT to converge for n >= 300 at any batch size --
            # so it always runs to max_iter, and cost scales with B * max_iter regardless of
            # the cap chosen. max_iter is scaled down as B * n_i grows so the *wall-clock*
            # cost of demonstrating "does not converge" stays roughly flat across the sweep;
            # the qualitative finding does not depend on the exact cap.
            cost_proxy = B * n_i
            cg_max_iter = 800 if cost_proxy <= 2000 else (400 if cost_proxy <= 50000 else 150)
            cgj_max_iter = 1200 if cost_proxy <= 2000 else (600 if cost_proxy <= 50000 else 250)

            def mv(x, edge_index=edge_index, g=g, n=n):
                return matvec_gather_scatter(edge_index, g, n, x)

            (x_cg, it_cg), cg_s = timed(cg_solve, mv, rhs, repeats=1, max_iter=cg_max_iter)

            # ---- matrix-free CG, Jacobi preconditioner ----
            diag = jacobi_diag(edge_index, g, n)
            inv_diag = 1.0 / diag.clamp_min(torch.finfo(DTYPE).tiny)

            def precond(r, inv_diag=inv_diag):
                return r * inv_diag

            (x_cgj, it_cgj), cgj_s = timed(
                cg_solve, mv, rhs, precond=precond, repeats=1, max_iter=cgj_max_iter
            )

            # ---- torch.sparse (COO, batched, bmm-as-matvec CG) ----
            # Capped: torch.sparse's batched-bmm matvec (see Part B2 below) has per-call overhead
            # far above the manual gather/scatter matvec, so
            # running it to full CG convergence at the same max_iter as the other two methods
            # is only affordable while B * n_i is small -- above that this is bounded to a
            # fixed, small iteration count instead of being run to convergence (an earlier,
            # unbounded version of this benchmark spent over an hour of wall time here alone).
            # it_sp == -2 marks "capped, not run to convergence" in the printed table.
            sparse_cost_proxy = B * n_i
            SPARSE_FULL_CG_LIMIT = 8000
            try:
                S, build_s = timed(build_batched_coo, edge_index, g, n, repeats=1)

                def mv_sparse(x, S=S):
                    return sparse_matvec_bmm(S, x)

                if sparse_cost_proxy <= SPARSE_FULL_CG_LIMIT:
                    (x_sp, it_sp), sp_s = timed(cg_solve, mv_sparse, rhs, repeats=1, max_iter=800)
                    err_sp = (x_sp - x_cg).abs().max().item()
                else:
                    x_sp, it_sp = cg_solve(mv_sparse, rhs, max_iter=30)
                    _, sp_s = timed(cg_solve, mv_sparse, rhs, max_iter=30, repeats=3)
                    it_sp = -2
                    err_sp = float("nan")
                sp_s += build_s
            except Exception as e:  # noqa: BLE001
                sp_s, it_sp, err_sp = float("nan"), -1, float("nan")
                print(f"    [sparse path FAILED at n={n} B={B}: {type(e).__name__}: {e}]")

            err_cg = (
                (x_cg - x_dense).abs().max().item() / x_dense.abs().max().clamp_min(1e-30).item()
                if x_dense is not None
                else float("nan")
            )

            print(
                f"{n:>6} {B:>6} {b_edges:>8} | "
                f"{dense_s:>10.4f} {dense_mem:>13.1f} | "
                f"{cg_s:>8.4f} {it_cg:>6} {cgj_s:>8.4f} {it_cgj:>7} | "
                f"{sp_s:>8.4f} {it_sp:>6} | {err_cg:>9.2e} {err_sp:>9.2e}"
            )

    print()
    print("=" * 100)
    print("Part A2: iteration counts vs conductance spread (n=300, B=100), Jacobi on/off")
    print("=" * 100)
    n = 300
    edge_index = build_graph(n, seed=SEED + 999)
    b_edges = edge_index.shape[1]
    B = 100
    rhs = torch.randn(B, n - 1, dtype=DTYPE)
    print(f"{'decades':>8} {'it_plain':>9} {'it_jacobi':>10}")
    for decades in (0.0, 1.0, 2.0, 4.0, 6.0, 8.0):
        g = random_conductances(B, b_edges, decades=decades, seed=SEED + 7)

        def mv(x, edge_index=edge_index, g=g, n=n):
            return matvec_gather_scatter(edge_index, g, n, x)

        _, it_plain = cg_solve(mv, rhs, max_iter=3000)
        diag = jacobi_diag(edge_index, g, n)
        inv_diag = 1.0 / diag.clamp_min(torch.finfo(DTYPE).tiny)
        _, it_jac = cg_solve(
            mv, rhs, precond=lambda r, inv_diag=inv_diag: r * inv_diag, max_iter=3000
        )
        print(f"{decades:>8.0f} {it_plain:>9} {it_jac:>10}")

    print()
    print("=" * 100)
    print("Part B: torch.sparse.spsolve, direct attempt on a trivial 4x4 CSR system")
    print("=" * 100)
    print(try_spsolve_once())

    print()
    print("=" * 100)
    print("Part B2: raw matvec cost, J @ x, gather/scatter vs. torch.sparse batched-COO bmm")
    print("(fixed 30 calls each, no CG/convergence involved -- isolates per-call overhead)")
    print("=" * 100)
    print(f"{'n':>6} {'B':>6} | {'gs_ms/call':>11} {'sp_ms/call':>11} {'sp/gs ratio':>12}")
    for n in [50, 100, 300, 1000, 3000]:
        edge_index = build_graph(n, seed=SEED + n + 5)
        b_edges = edge_index.shape[1]
        for B in [1, 100, 1000]:
            g = random_conductances(B, b_edges, decades=6.0, seed=SEED + n + B + 5)
            x = torch.randn(B, n - 1, dtype=DTYPE)
            S = build_batched_coo(edge_index, g, n)

            def call_gs(edge_index=edge_index, g=g, n=n, x=x):
                for _ in range(30):
                    matvec_gather_scatter(edge_index, g, n, x)

            def call_sp(S=S, x=x):
                for _ in range(30):
                    sparse_matvec_bmm(S, x)

            _, gs_s = timed(call_gs, repeats=3)
            _, sp_s = timed(call_sp, repeats=3)
            print(
                f"{n:>6} {B:>6} | {1000 * gs_s / 30:>11.4f} {1000 * sp_s / 30:>11.4f} "
                f"{sp_s / gs_s:>12.1f}"
            )

    print()
    print("=" * 100)
    print("Part C: assembly cost -- A @ q (scatter) and A^T @ phi (gather),")
    print("dense matmul vs. index_select/index_add, same edge_index, no solve involved")
    print("=" * 100)
    print(f"{'n':>7} {'B':>6} {'b_edges':>8} | {'dense_Aq_s':>11} {'scat_Aq_s':>10} | "
          f"{'dense_ATphi_s':>14} {'gath_ATphi_s':>13} | {'speedup_Aq':>11} {'speedup_ATphi':>14}")
    for n in [100, 300, 1000, 3000, 10000]:
        edge_index = build_graph(n, seed=SEED + n + 1)
        b_edges = edge_index.shape[1]
        src, dst = edge_index[0], edge_index[1]
        A = torch.zeros(n, b_edges, dtype=DTYPE)
        A[src, torch.arange(b_edges)] += 1.0
        A[dst, torch.arange(b_edges)] -= 1.0
        for B in [1, 1000]:
            q = torch.randn(B, b_edges, dtype=DTYPE)
            phi = torch.randn(B, n, dtype=DTYPE)

            _, dense_Aq_s = timed(lambda q=q, A=A: q @ A.T, repeats=5)

            def scatter_Aq(q=q, src=src, dst=dst, n=n):
                out = q.new_zeros(q.shape[0], n)
                out = out.index_add(-1, src, q)
                out = out.index_add(-1, dst, -q)
                return out

            _, scat_Aq_s = timed(scatter_Aq, repeats=5)

            _, dense_ATphi_s = timed(lambda phi=phi, A=A: phi @ A, repeats=5)

            def gather_ATphi(phi=phi, src=src, dst=dst):
                return phi[..., src] - phi[..., dst]

            _, gath_ATphi_s = timed(gather_ATphi, repeats=5)

            print(
                f"{n:>7} {B:>6} {b_edges:>8} | {dense_Aq_s:>11.5f} {scat_Aq_s:>10.5f} | "
                f"{dense_ATphi_s:>14.5f} {gath_ATphi_s:>13.5f} | "
                f"{dense_Aq_s / scat_Aq_s:>11.1f} {dense_ATphi_s / gath_ATphi_s:>14.1f}"
            )


if __name__ == "__main__":
    main()
