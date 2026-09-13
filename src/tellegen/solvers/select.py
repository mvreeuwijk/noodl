"""method="auto" eligibility resolution (design section 3.1) and the raise/return failure
boundary (design section 3.2), on top of solvers.iterative's pcg and gmres, plus the retained
dense LU reference (`method="direct"`, amendment A3.1) and the SciPy sparse LU reference
(`method="sparse_direct"`, spec section 6.2 step 2).

solvers.iterative.pcg/gmres never raise; this is the one layer where a failed NUMERICAL solve
becomes an exception by default. `on_failure="return"` is the explicit, narrow, non-default
escape hatch for that numerical-failure case only (never applies to a backward pass -- Task 12
always raises unconditionally there, bypassing this function's on_failure entirely). It does
NOT apply to an eligibility refusal: requesting `method="cg"` on an operator that cannot
certify SPD, or letting `method="auto"` see a batch where some but not all instances certify,
is a contract violation and raises regardless of `on_failure` -- there is no SolveResult to
return in that case, only a modelling error to report.

THE ELIGIBILITY TABLE for `method="auto"` (design section 3.1, amended by spec section 6.2
step 2). Read top to bottom; the first matching row wins:

  spd_certificate() mixed over the batch          -> RuntimeError (never split a batch)
  certified SPD, and sparse-direct is applicable  -> sparse_direct  (SciPy SuperLU per instance)
  certified SPD, otherwise                        -> pcg            (Jacobi-preconditioned CG)
  certificate None, or uniformly False            -> gmres          (no symmetry assumption)

"sparse-direct is applicable" is the conjunction `_auto_sparse_triplet` tests: the operator
declares the optional `assemble_sparse` member AND returns a sparse form from it, SciPy is
importable, the solve is grad-safe (not `is_grad_enabled()` with a grad-requiring input), AND
the flat batch size is at most `_SPARSE_DIRECT_MAX_BATCH` (see that constant: above it the
per-instance Python loop has lost to PCG's batched arithmetic). Every one of those is a
FALL-BACK to pcg when it fails, never a refusal: `auto` is a promise to choose a backend that
works.

WHY sparse-direct is the certified-SPD default (spec section 2, "selected on evidence, per
platform"; the platform here is CPU + SciPy). Measured in process on the reference composed
model -- 1028 unknowns, ~5300 nonzeros, float64, 14 threads -- median of 3 warm runs via
`benchmarks.profile_forward --compare-solvers`:

  forward  ensemble 1     pcg 128.4 ms    sparse_direct   28.0 ms    4.59x faster
  forward  ensemble 100   pcg 2399 ms     sparse_direct   2182 ms    1.10x (a tie: repeated,
                                                                     interleaved, pcg ranged
                                                                     2068-2434 ms against
                                                                     2113-2182 ms)
  forward  ensemble 1000  pcg 29489 ms    sparse_direct   19263 ms   1.53x faster
  backward ensemble 1     pcg  44.9 ms    sparse_direct   13.8 ms    3.25x faster
  backward ensemble 100   pcg 1741 ms     sparse_direct    476 ms    3.66x faster

PCG needed 168-180 iterations per Newton step at this conditioning; the factorisation needs
one. The per-instance Python loop is its real cost (see `_sparse_direct`), which is why the
ensemble-100 forward is only a tie, why the rule DOES carry an ensemble threshold
(`_SPARSE_DIRECT_MAX_BATCH`, measured below) and why a batched vendor backend stays
admissible: it would replace the loop, not the algorithm.
"""

from __future__ import annotations

import warnings

import torch

from tellegen.operators.base import SolveResult, SolverStatus
from tellegen.solvers.iterative import gmres, pcg

Tensor = torch.Tensor

# The largest FLAT batch (product of the leading dims of `b`) for which `method="auto"` still
# routes a certified-SPD, sparse-capable operator to sparse-direct. Above it, `auto` uses PCG.
#
# WHY THERE IS A THRESHOLD AT ALL. `_sparse_direct` factorises instance by instance in a
# PYTHON LOOP (SuperLU has no batched entry point), so its cost is linear in the ensemble
# size at ~21 ms/instance on the reference model; PCG's cost is sub-linear, because its
# arithmetic is batched torch. The two therefore cross. Measured in process on the reference
# composed model -- 1028 unknowns, float64, 14 threads, median of 3, forward solve -- as the
# ratio sparse_direct/cg (below 1.0 sparse-direct wins):
#
#   batch      1      2      4      8     16     32     64
#   ratio   0.24   0.14   0.26   0.30   0.54   0.68   1.00
#
# and in the child-process acceptance gate (`benchmarks/composed_scaling_report.json`, whole
# forward pass) cg is already AHEAD of an unthresholded auto at ensemble 100 (1.593 s vs
# 1.686 s), 100x24 (39.4 s vs 48.5 s) and 1000 (12.9 s vs 15.2 s), while auto wins 2.2x at
# ensemble 1 (0.097 s vs 0.210 s). 32 is the last measured point at which the factorisation
# is still clearly ahead (0.68) and 64 is the tie; the whole 32-64 band is within 1.0-1.5x,
# so the cost of choosing 32 rather than 64 is small either way. Like the rest of this rule
# (spec section 2, "selected on evidence, per platform") it is a CPU + SciPy measurement at
# one problem size, and is the thing to re-measure when either changes.
_SPARSE_DIRECT_MAX_BATCH = 32

# Set once, the first time `auto` falls back to PCG for the ONE reason that is an environment
# fault rather than a modelling fact: a certified-SPD operator offered a sparse form, the
# batch was within the threshold, the solve was grad-safe, and SciPy -- the `tellegen[sparse]`
# extra -- was the only thing missing. Warning on EVERY such solve would be unusable noise on
# a machine that simply has not installed the extra, and PCG is a correct answer; warning
# never at all leaves a user silently on the 4.6x-slower path with `diagnostics["backend"]`
# as the only signal. Once per process is the compromise. It is module state rather than a
# `warnings`-module filter because `warnings.warn`'s own "once" registry is keyed on the
# message and can be reset out from under us by `catch_warnings`.
_WARNED_SPARSE_DIRECT_NEEDS_SCIPY = False


def _describe_uncertified(op, cert: Tensor, where: str) -> str:
    """Message fragment naming which instances fail to certify SPD, and why.

    Prefers `op.spd_diagnosis()` (present on GraphLaplacianOperator, absent elsewhere) to name
    the actual negative-slope edges or ungrounded interior nodes per failing instance; falls
    back to the generic "instances {bad} do not certify SPD" when the operator has no
    `spd_diagnosis` (e.g. the test-only `_FakeOperator`, or any future non-diagnosing operator).
    """
    bad = torch.nonzero(~cert.reshape(-1), as_tuple=False).flatten().tolist()
    diagnose = getattr(op, "spd_diagnosis", None)
    if diagnose is None:
        return f"instances {bad} do not certify SPD"
    lines = []
    for rec in diagnose():
        if rec["reason"] == "negative_slope":
            lines.append(f"instance {rec['instance']}: negative slope on edges {rec['edges']}")
        else:
            lines.append(f"instance {rec['instance']}: ungrounded interior nodes {rec['nodes']}")
    return "; ".join(lines) if lines else f"instances {bad} do not certify SPD"


def _direct(A: Tensor, b: Tensor) -> SolveResult:
    """LU solve of the assembled operator with PER-INSTANCE singularity status.

    torch.linalg.solve raises for the whole batch if any one instance is singular, which is
    the very behaviour the milestone-1 Newton had to work around with an identity
    substitution. lu_factor_ex reports singularity per instance through `info` instead.

    `converged` HERE MAKES NO RESIDUAL CLAIM: it is `finite & ~singular`, so a factorisation
    that returns a finite answer is reported CONVERGED whatever its residual, and `residual`
    is informational rather than a gate (unlike `pcg`/`gmres`, where convergence IS a
    residual test). LU is backward stable, so the residual is genuinely ~0 wherever the
    factorisation succeeds; what the caller does not get is a forward-accuracy guarantee on
    an ill-conditioned system. `_sparse_direct` shares this contract, and since section 6.2
    step 2 made it the default for small certified batches it is the SHIPPED failure
    signature: a badly-conditioned certified system that PCG would have reported as
    MAX_ITER or BREAKDOWN now returns a finite, backward-stable, forward-inaccurate answer
    marked CONVERGED, and Newton's own residual test will not catch it either.
    """
    batch = torch.broadcast_shapes(A.shape[:-2], b.shape[:-1])
    m = A.shape[-1]
    A_b = A.expand(*batch, m, m)
    b_b = b.expand(*batch, m)
    LU, pivots, info = torch.linalg.lu_factor_ex(A_b)
    singular = info != 0  # (...,) per instance; info>0 = zero pivot
    x = torch.linalg.lu_solve(LU, pivots, b_b.unsqueeze(-1)).squeeze(-1)
    x = torch.where(singular.unsqueeze(-1), torch.zeros_like(x), x)
    r = torch.einsum("...ij,...j->...i", A_b, x) - b_b
    b_norm = torch.linalg.vector_norm(b_b, dim=-1)
    residual = torch.linalg.vector_norm(r, dim=-1) / b_norm.clamp_min(torch.finfo(b.dtype).tiny)
    residual = torch.where(b_norm > 0, residual, torch.zeros_like(residual))
    finite = torch.isfinite(x).all(dim=-1)
    converged = finite & ~singular
    status = torch.where(
        converged,
        torch.full_like(info, int(SolverStatus.CONVERGED)),
        torch.full_like(info, int(SolverStatus.SINGULAR)),
    )
    return SolveResult(
        x=x,
        converged=converged,
        iterations=torch.ones(info.shape, dtype=torch.int64, device=info.device),
        residual=residual,
        status=status.to(torch.int64),
    )


def _sparse_triplet(op):
    """`op.assemble_sparse()` if the operator declares the optional member, else None."""
    assemble_sparse = getattr(op, "assemble_sparse", None)
    return assemble_sparse() if assemble_sparse is not None else None


def _auto_sparse_triplet(op, b: Tensor):
    """The COO triplet `method="auto"` should factorise for a CERTIFIED-SPD operator, or
    `None` to stay on PCG. Never raises: every "no" here is a fall-back, not a refusal.

    This is where the section 6.2 step 2 default lives. `auto` is a promise to pick a
    backend that works, so each of the five ways sparse-direct can be inapplicable makes it
    return None and leaves `solve` on the Krylov path it had before:

    1. the flat batch exceeds `_SPARSE_DIRECT_MAX_BATCH` -- a PERFORMANCE fall-back rather
       than a capability one (sparse-direct would answer correctly, just slower), measured
       and justified at that constant. It is checked first because it is the cheapest of the
       five and, unlike the others, it is a property of this call rather than of the
       environment;
    2. the operator does not declare `assemble_sparse` (the member is OPTIONAL);
    3. the solve is not grad-safe -- grad mode is on AND some input requires grad -- so a
       non-differentiable backend would silently detach the answer. Both layer paths solve
       under `no_grad`, so this is the non-differentiable-`solve`-with-learnable-elements
       case and a handful of direct callers, not the implicit adjoint;
    4. the operator declares the member but has no sparse form to give (returns None);
    5. SciPy is not importable (it is an OPTIONAL extra, `tellegen[sparse]` -- an explicit
       `method="sparse_direct"` raises ImportError, but `auto` must not).

    ORDER MATTERS, and 5 is deliberately last even though it is the cheapest test after 1.
    Reaching it means every OTHER condition held, which is exactly the predicate the
    once-per-process warning needs: "this solve would have been factorised if the extra were
    installed". Testing SciPy earlier would make that indistinguishable from the fall-backs
    that are modelling facts, which must never warn -- telling someone to install SciPy for
    an operator that has no sparse form to give would be wrong advice. The price is that a
    SciPy-less installation assembles a COO triplet per solve and discards it; that is one
    O(edges) vectorised expression against a PCG solve of ~170 matvecs, so well under 1%.

    The SciPy import is repeated per solve rather than cached in a module global. After the
    first one it is a `sys.modules` lookup costing microseconds against a solve costing
    milliseconds, and a cached answer would be wrong for exactly the case the fall-back
    exists for (a process that can or cannot import scipy is not a property this module gets
    to memoise -- and it would make the behaviour untestable without process isolation).
    """
    global _WARNED_SPARSE_DIRECT_NEEDS_SCIPY

    if b.shape[:-1].numel() > _SPARSE_DIRECT_MAX_BATCH:
        return None
    if getattr(op, "assemble_sparse", None) is None:
        return None
    grad_on = torch.is_grad_enabled()
    if grad_on and b.requires_grad:
        return None
    triplet = _sparse_triplet(op)
    if triplet is None:
        return None
    if grad_on and triplet[2].requires_grad:
        return None
    try:
        import scipy.sparse.linalg  # noqa: F401
    except ImportError:
        if not _WARNED_SPARSE_DIRECT_NEEDS_SCIPY:
            _WARNED_SPARSE_DIRECT_NEEDS_SCIPY = True
            # `stacklevel=2` points at `solve`, the nearest frame that means anything: this
            # sits under `solve` under `newton` under a layer, so no fixed depth reaches the
            # user's own call site. The message is therefore self-contained, and
            # `diagnostics["backend"]` is the per-solve, non-noisy answer.
            warnings.warn(
                "tellegen: method='auto' would have factorised this certified-SPD system "
                "with SciPy's sparse LU, but scipy could not be imported, so it fell back "
                "to preconditioned CG. The answer is correct; it is roughly 4.6x slower at "
                "small ensembles. Install the extra to get the documented default: "
                "pip install tellegen[sparse]. This warning is issued once per process; "
                "PotentialFlowLayer.solve's diagnostics['backend'] reports the backend that "
                "actually ran on every solve.",
                RuntimeWarning,
                stacklevel=2,
            )
        return None
    return triplet


def _sparse_direct(op, b: Tensor, where: str, triplet=None) -> SolveResult:
    """SciPy sparse LU (SuperLU) of `op.assemble_sparse()`, PER INSTANCE, with per-instance
    singularity status -- the spec's section 6.2 sparse-direct reference path.

    KNOWN LIMITATION, and the reason a vendor backend stays open (spec section 2, "a vendor
    sparse-direct path stays admissible and is selected on evidence, per platform"): the
    factorisation is a PYTHON LOOP over the flat batch. SciPy's SuperLU bindings factor one
    matrix at a time and have no batched entry point, so an ensemble of `N` realisations
    costs `N` independent `splu` calls with `N` round trips through the interpreter. That is
    exactly the cost this backend was added to MEASURE; a cuDSS/MKL-style batched vendor
    backend would replace the loop rather than the algorithm, and nothing above this function
    would change.

    The batch is flattened once, not instance by instance: `row`/`col` are shared across the
    whole batch by the `SparseAssembling` contract, so only `values[i]` changes between
    iterations and the index arrays are converted to NumPy exactly once.

    SciPy is a DEV dependency of this project, so both imports are LAZY and inside this
    function: an installation without SciPy must be able to import `tellegen.solvers` and use
    every other backend, and must get an `ImportError` naming scipy (not a bare
    `ModuleNotFoundError` from somewhere inside a solve) if it asks for this one.

    NOT DIFFERENTIABLE: the solve leaves torch entirely, so `x` carries no autograd history.
    Rather than return a silently detached tensor, this refuses when grad mode is on AND some
    input actually requires grad. Both layer paths reach it under `torch.no_grad`
    (`layers.transport._LinearSolve.forward` and `solvers.implicit._Implicit.forward`/
    `.backward`), so a differentiable `PotentialFlowLayer.solve` configured with
    `linear_solver="sparse_direct"` is unaffected: its gradients come from the implicit
    adjoint, which never differentiates through the linear solver's own arithmetic.

    `converged` makes no residual claim here either -- `finite & ~singular`, with `residual`
    informational. See `_direct`, which states the contract and what it means now that this
    backend is the default for small certified batches.
    """
    try:
        import scipy.sparse
        import scipy.sparse.linalg
    except ImportError as exc:
        raise ImportError(
            f"{where}: method='sparse_direct' requires scipy (scipy.sparse and "
            f"scipy.sparse.linalg), which could not be imported: {exc}. Install scipy, or "
            f"use method='auto', 'cg', 'gmres' or 'direct'."
        ) from exc

    if triplet is None:
        # Not pre-fetched by the "auto" branch, so this is an EXPLICIT method="sparse_direct":
        # an operator with no sparse form is a caller error here, not a reason to fall back.
        assemble_sparse = getattr(op, "assemble_sparse", None)
        triplet = _sparse_triplet(op)
        if triplet is None:
            raise ValueError(
                f"{where}: method='sparse_direct' requires op.assemble_sparse() to return a "
                f"(row, col, values) COO triplet, but this operator "
                f"{'returned None' if assemble_sparse is not None else 'has no assemble_sparse'}"
            )
    row, col, values = triplet
    if torch.is_grad_enabled() and (b.requires_grad or values.requires_grad):
        raise RuntimeError(
            f"{where}: method='sparse_direct' is not differentiable (the factorisation and "
            f"solve happen in SciPy, outside autograd) but grad mode is enabled and an input "
            f"requires grad; refusing rather than returning a silently detached answer. Solve "
            f"under torch.no_grad(), or use method='auto', 'cg' or 'direct'."
        )

    m = int(b.shape[-1])
    nnz = int(row.shape[-1])
    batch = torch.broadcast_shapes(values.shape[:-1], b.shape[:-1])
    # The flat instance count is computed, never inferred with a -1: a system with zero
    # unknowns (every node of a network is a boundary node -- the CONTAM parallel-combination
    # fixtures are exactly that) has m = 0 AND nnz = 0, and `reshape(-1, 0)` is ambiguous
    # rather than empty.
    n_flat = int(batch.numel())
    values_flat = values.expand(*batch, nnz).reshape(n_flat, nnz)
    b_flat = b.expand(*batch, m).reshape(n_flat, m)

    # SuperLU has single- and double-precision kernels only; anything narrower is factorised
    # in float64 and cast back, which is strictly better than refusing (and than silently
    # truncating the factorisation to a dtype SciPy would have rejected outright).
    work_dtype = b.dtype if b.dtype in (torch.float32, torch.float64) else torch.float64
    row_np = row.detach().cpu().numpy()
    col_np = col.detach().cpu().numpy()
    values_np = values_flat.detach().cpu().to(work_dtype).numpy()
    b_np = b_flat.detach().cpu().to(work_dtype).numpy()
    x_np = b_np.copy()
    singular_flat = torch.zeros(n_flat, dtype=torch.bool)
    # `m == 0` is a system with no unknowns: already solved, and SuperLU has nothing to
    # factorise. The empty `x_np` below IS the unique solution, and the residual machinery
    # that follows reports it as converged with a zero residual, exactly as pcg did.
    for i in range(n_flat if m > 0 else 0):
        A_i = scipy.sparse.csc_matrix((values_np[i], (row_np, col_np)), shape=(m, m))
        try:
            lu = scipy.sparse.linalg.splu(A_i)
        except RuntimeError:
            # SuperLU reports an exactly singular factor by raising, for THIS instance only;
            # its siblings are independent matrices and are still solved (design section 3.2:
            # per-instance status, never a whole-batch abort).
            singular_flat[i] = True
            x_np[i] = 0.0
            continue
        x_np[i] = lu.solve(b_np[i])

    x = torch.from_numpy(x_np).to(device=b.device, dtype=b.dtype).reshape(*batch, m)
    singular = singular_flat.to(b.device).reshape(batch)
    # Residual through the operator's OWN action rather than a re-assembled matrix: same
    # definition as `_direct`'s (relative, zero for a zero right-hand side), one matvec
    # instead of an (..., m, m) einsum, and it cannot drift from the operator being solved.
    r = op.matvec(x) - b.expand(*batch, m)
    b_norm = torch.linalg.vector_norm(b.expand(*batch, m), dim=-1)
    residual = torch.linalg.vector_norm(r, dim=-1) / b_norm.clamp_min(torch.finfo(b.dtype).tiny)
    residual = torch.where(b_norm > 0, residual, torch.zeros_like(residual))
    finite = torch.isfinite(x).all(dim=-1)
    converged = finite & ~singular
    status = torch.where(
        converged,
        torch.full(batch, int(SolverStatus.CONVERGED), dtype=torch.int64, device=b.device),
        torch.full(batch, int(SolverStatus.SINGULAR), dtype=torch.int64, device=b.device),
    )
    return SolveResult(
        x=x,
        converged=converged,
        iterations=torch.ones(batch, dtype=torch.int64, device=b.device),
        residual=residual,
        status=status,
    )


_METHODS = ("auto", "cg", "gmres", "direct", "sparse_direct")
_ON_FAILURE = ("raise", "return")


def solve(
    op,
    b: Tensor,
    *,
    method: str = "auto",
    on_failure: str = "raise",
    where: str = "solve",
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_iter: int | None = None,
    x0: Tensor | None = None,
    preconditioner: str | None = "jacobi",
    restart: int = 30,
    backend_out: dict | None = None,
) -> SolveResult:
    """Resolve `method="auto"` per the eligibility table (see the module docstring), or honour
    an explicit method, then apply the raise/return boundary on the NUMERICAL outcome.

    Accepts the explicit union of backend keyword arguments -- `rtol`, `atol`, `max_iter`,
    `x0` (both pcg and gmres), `preconditioner` (pcg only), `restart` (gmres only) -- and
    forwards to the chosen backend only the ones it accepts, silently dropping the rest (no
    `**kw`: a reviewer found `solve(op_nonsym, b, preconditioner="jacobi")` raising `TypeError`
    from gmres before this signature was made explicit). `method="direct"` and
    `method="sparse_direct"` accept and ignore all of them.

    `method="sparse_direct"` is the spec's section 6.2 sparse-direct reference: SciPy SuperLU
    of `op.assemble_sparse()`, per instance. It is a DIRECT method, so it makes no SPD or
    symmetry assumption and never consults the certificate; it requires the optional
    `assemble_sparse` member (`ValueError` otherwise) and SciPy (`ImportError` otherwise);
    and it is not differentiable (see `_sparse_direct`).

    `backend_out`, when a dict is passed, is filled with `{"backend": name}` naming the
    backend that actually RAN -- one of "sparse_direct", "pcg", "gmres", "direct" -- as
    distinct from the `method` that was requested. Since the `"auto"` default is conditional
    on runtime predicates (SciPy's presence, the batch size, grad-safety) the two genuinely
    differ, and nothing else exposes which way a given solve went: `linear_iterations` is an
    indirect signal (1 vs ~170) and there is no other. It is an out-parameter rather than a
    field on `SolveResult` so that the result type -- which every backend constructs and
    every caller unpacks -- is unchanged. It is written only once a backend has been chosen,
    so an ELIGIBILITY REFUSAL leaves it untouched: there is no backend behind a refusal.
    """
    if method not in _METHODS:
        raise ValueError(f"{where}: unknown method {method!r}; expected one of {_METHODS}")
    if on_failure not in _ON_FAILURE:
        raise ValueError(
            f"{where}: unknown on_failure {on_failure!r}; expected one of {_ON_FAILURE}"
        )

    if method == "cg":
        cert = op.spd_certificate()
        if cert is None:
            raise RuntimeError(
                f"{where}: method='cg' requested explicitly but this operator's "
                f"spd_certificate() is None (it cannot certify SPD at all); refusing "
                f"rather than returning a plausible wrong answer."
            )
        if not bool(torch.all(cert)):
            raise RuntimeError(
                f"{where}: method='cg' requested explicitly but "
                f"{_describe_uncertified(op, cert, where)}; refusing rather than returning a "
                f"plausible wrong answer."
            )
        backend = "pcg"
        result = pcg(
            op, b, rtol=rtol, atol=atol, max_iter=max_iter, preconditioner=preconditioner, x0=x0
        )
    elif method == "gmres":
        backend = "gmres"
        result = gmres(op, b, rtol=rtol, atol=atol, max_iter=max_iter, restart=restart, x0=x0)
    elif method == "direct":
        A = op.assemble()
        if A is None:
            raise ValueError(f"{where}: method='direct' requires op.assemble() to return a matrix")
        backend = "direct"
        result = _direct(A, b)
    elif method == "sparse_direct":
        backend = "sparse_direct"
        result = _sparse_direct(op, b, where)
    else:  # method == "auto"
        cert = op.spd_certificate()
        if cert is not None and bool(torch.any(cert)) and not bool(torch.all(cert)):
            raise RuntimeError(
                f"{where}: method='auto' refuses to split the batch; "
                f"{_describe_uncertified(op, cert, where)} while other instances certify. "
                f"Certify all instances, or pass an explicit method."
            )
        if cert is not None and bool(torch.all(cert)):
            # Spec section 6.2 step 2, decided on the Task C measurement (see this module's
            # docstring): a certified-SPD operator that can hand over a sparse form is
            # factorised rather than iterated, up to `_SPARSE_DIRECT_MAX_BATCH` instances.
            # `_auto_sparse_triplet` returns None -- and never raises -- whenever that is not
            # applicable, which is what keeps every other certified-SPD operator, every
            # SciPy-less installation, and every larger ensemble on PCG.
            triplet = _auto_sparse_triplet(op, b)
            if triplet is not None:
                backend = "sparse_direct"
                result = _sparse_direct(op, b, where, triplet)
            else:
                backend = "pcg"
                result = pcg(
                    op,
                    b,
                    rtol=rtol,
                    atol=atol,
                    max_iter=max_iter,
                    preconditioner=preconditioner,
                    x0=x0,
                )
        else:
            # cert is None (cannot certify at all) or cert is uniformly False (no instance
            # is eligible for cg, so there is nothing to split off): both route to gmres,
            # which makes no symmetry or SPD assumption to violate. rmatvec, if this
            # operator declares one, is reserved for the adjoint and is never called here.
            backend = "gmres"
            result = gmres(op, b, rtol=rtol, atol=atol, max_iter=max_iter, restart=restart, x0=x0)

    # After the branches, so every path that reaches here has actually chosen and run a
    # backend; the raises above (unknown method, eligibility refusal, no assembled matrix)
    # leave `backend_out` untouched.
    if backend_out is not None:
        backend_out["backend"] = backend

    if on_failure == "raise":
        return result.raise_on_failure(where)
    return result
