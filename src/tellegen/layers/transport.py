"""Multi-species transport on nodal scalars advected by signed branch flows.

For interior capacity ``V`` (volume, or heat capacity), signed branch flow ``q``
on edges of ``flow_kind``, a carrier factor and a transmission fraction per edge:

    V dx/dt = (In(q) - Out(q)) x + N x_b + sources

``Out`` is the total weighted outflow leaving the upstream node of every edge and
``In`` is the transmitted weighted inflow arriving at the downstream node; the
full-node generator ``In - Out`` is split into an interior/interior block ``M``
and an interior/boundary block ``N``, both already divided by capacity. Species
are stacked species-major: for ``K`` species the stacked row/column index is
``k * n_i + i`` for interior node ``i`` (see the module docstring of
``TransportLayer.operator`` for why the stacked shape is used even when species
do not interact).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from tellegen.operators.advection import AdvectionOperator
from tellegen.solvers.select import solve as _solve_operator
from tellegen.topology import Network, Node


class _TransposeView:
    """A LinearOperator-shaped view exposing `op`'s TRANSPOSE: matvec and rmatvec swapped,
    everything else passed through. Used only to solve the adjoint system `A^T lam =
    grad_x` via the ordinary `solvers.select.solve` entry point -- the adjoint needs no
    solver of its own, it reuses GMRES/PCG against the swapped action.
    """

    def __init__(self, op) -> None:
        self._op = op
        self.shape = op.shape
        self.dtype = op.dtype
        self.device = op.device
        self.symmetric = op.symmetric

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self._op.rmatvec(x)

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self._op.matvec(x)

    def diagonal(self) -> torch.Tensor:
        return self._op.diagonal()  # diagonal entries are invariant under transpose

    def assemble(self):
        dense = self._op.assemble()
        return None if dense is None else dense.transpose(-1, -2)

    def spd_certificate(self):
        return None


class _LinearSolve(torch.autograd.Function):
    """Differentiate a linear solve `A(params) x = rhs(params)` via the IMPLICIT ADJOINT,
    never by unrolling the forward solver's iteration -- see this task's Design decisions
    section for why this is required rather than optional.

    Forward: run `solvers.select.solve` under `no_grad`. Backward: solve the ADJOINT system
    `A(params)^T lam = grad_x` via `_TransposeView` (i.e. via `op.rmatvec`, never
    `op.matvec` -- the entire reason `rmatvec` is part of this contract), then obtain
    gradients wrt every parameter tensor by one more autograd pass through the residual
    `A(params) x - rhs(params)` evaluated at the converged `x`, weighted by `-lam`. This is
    `solvers/implicit.py`'s `_Implicit` structure, specialised to a LINEAR residual, so
    there is no fixed-point iteration to differentiate through: `A` already IS the
    residual's exact Jacobian everywhere, not just at convergence.

    Every differentiable input is passed EXPLICITLY as a `*params` tensor to `apply`, never
    captured by closure: `build_system` is a plain (non-tensor) Python callable, stored on
    `ctx` for its STRUCTURE only, and it is called AGAIN inside `backward` on the SAVED
    params (or fresh detached-and-`requires_grad_`-ed copies, for the residual pass) --
    never on tensors implicitly captured from the enclosing scope, which `backward` cannot
    see. This is `solvers/implicit.py`'s own module-docstring warning, almost verbatim,
    because it is the identical trap.

    The adjoint solve inside `backward` RAISES unconditionally on non-convergence --
    independent of whatever `on_failure` the forward call was given -- because a wrong
    gradient is worse than no gradient (design spec section 3.2).
    """

    @staticmethod
    def forward(ctx, build_system, where, solver_kwargs, *params):
        with torch.no_grad():
            op, rhs = build_system(*params)
            result = _solve_operator(
                op, rhs, method="auto", on_failure="raise", where=where, **solver_kwargs
            )
        ctx.build_system = build_system
        ctx.where = where
        ctx.solver_kwargs = solver_kwargs
        ctx.save_for_backward(result.x, *params)
        return result.x

    @staticmethod
    def backward(ctx, grad_x):
        if torch.is_grad_enabled():
            # See solvers/implicit.py's module docstring: grad mode enabled here means the
            # caller requested create_graph=True (second-order differentiation), which
            # this adjoint does not support.
            raise RuntimeError(
                f"{ctx.where}: second-order differentiation (create_graph=True) is not "
                f"supported by this implicit linear adjoint; detach the first-order "
                f"gradient before using it in a further differentiable loss."
            )
        saved = ctx.saved_tensors
        x, params = saved[0], list(saved[1:])
        with torch.no_grad():
            op, _ = ctx.build_system(*params)
            lam = _solve_operator(
                _TransposeView(op), grad_x, method="auto", on_failure="raise",
                where=f"{ctx.where} backward (adjoint)", **ctx.solver_kwargs,
            ).x
        with torch.enable_grad():
            p = [t.detach().requires_grad_(t.requires_grad) for t in params]
            op_p, rhs_p = ctx.build_system(*p)
            residual = op_p.matvec(x.detach()) - rhs_p
            needs_grad = [t for t in p if t.requires_grad]
            grads = (
                torch.autograd.grad(residual, needs_grad, grad_outputs=-lam, allow_unused=True)
                if needs_grad else []
            )
        grads_aligned = []
        it = iter(grads)
        for t in p:
            grads_aligned.append(next(it) if t.requires_grad else None)
        return (None, None, None, *grads_aligned)


def _linear_solve(build_system, where: str, *params, **solver_kwargs) -> torch.Tensor:
    return _LinearSolve.apply(build_system, where, solver_kwargs, *params)


class _AffineSystemOperator:
    """(I - alpha * M): built from an AdvectionOperator, satisfying the LinearOperator
    duck type solvers.select.solve reads (matvec, rmatvec, diagonal, shape, dtype,
    device, symmetric, assemble, spd_certificate). Used for both the implicit scheme
    (alpha = dt) and the trapezoidal scheme (alpha = dt / 2), and passed to
    _linear_solve exactly like a bare AdvectionOperator is for steady() -- the adjoint
    derivation does not care what concrete operator "A" is, only that it exposes
    matvec/rmatvec/diagonal/assemble/spd_certificate.
    """

    symmetric = False

    def __init__(self, M: AdvectionOperator, alpha: float) -> None:
        self.M = M
        self.alpha = alpha
        self.shape = M.shape
        self.dtype = M.dtype
        self.device = M.device

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.alpha * self.M.matvec(x)

    def rmatvec(self, y: torch.Tensor) -> torch.Tensor:
        return y - self.alpha * self.M.rmatvec(y)

    def diagonal(self) -> torch.Tensor:
        return 1.0 - self.alpha * self.M.diagonal()

    def assemble(self):
        dense = self.M.assemble()
        m = dense.shape[-1]
        eye = torch.eye(m, dtype=dense.dtype, device=dense.device)
        return eye - self.alpha * dense

    def spd_certificate(self):
        return None


class TransportLayer:
    """dx/dt = M x + N x_b + sources / capacity on interior nodes."""

    def __init__(
        self,
        net: Network,
        name: str,
        *,
        capacity: torch.Tensor,
        flow_kind: str,
        boundary: Sequence[Node],
        n_species: int = 1,
        carrier: torch.Tensor | float = 1.0,
        transmission: torch.Tensor | None = None,
        kinetics: torch.Tensor | None = None,
        removal: torch.Tensor | None = None,
        conduction_kind: str | None = None,
        conductance: torch.Tensor | None = None,
        scheme: Literal["exact", "implicit", "trapezoidal"] = "exact",
    ) -> None:
        self.net = net
        self.name = name
        self.flow_kind = flow_kind
        self.boundary = list(boundary)
        self.n_species = n_species
        self.scheme = scheme
        self.interior_idx = net.interior_index(self.boundary)
        self.boundary_idx = net.boundary_index(self.boundary)
        self.n_i = int(self.interior_idx.shape[0])
        self.n_b = int(self.boundary_idx.shape[0])

        self.capacity = torch.as_tensor(capacity, dtype=net.dtype)
        if self.capacity.dim() == 0 or self.capacity.shape[-1] != self.n_i:
            cap_len = self.capacity.shape[-1] if self.capacity.dim() >= 1 else 0
            raise ValueError(
                f"TransportLayer '{name}': capacity has {cap_len} entries, "
                f"expected {self.n_i} interior nodes"
            )
        self.carrier = torch.as_tensor(carrier, dtype=net.dtype)

        b_flow = int(net.edge_index(flow_kind).shape[0])
        K = n_species
        if transmission is None:
            transmission = torch.ones(b_flow, dtype=net.dtype)
        transmission = torch.as_tensor(transmission, dtype=net.dtype)
        if transmission.dim() >= 2 and transmission.shape[-2:] == (b_flow, K):
            transmission = transmission.transpose(-1, -2)
        elif transmission.shape[-1] == b_flow:
            transmission = transmission.unsqueeze(-2).expand(*transmission.shape[:-1], K, b_flow)
        else:
            raise ValueError(
                f"TransportLayer '{name}': transmission must have shape ({b_flow},) or "
                f"({b_flow}, {K}), got {tuple(transmission.shape)}"
            )
        self.transmission = transmission  # (..., K, b_flow)

        if kinetics is not None:
            kinetics = torch.as_tensor(kinetics, dtype=net.dtype)
            if kinetics.shape[-2:] != (K, K):
                raise ValueError(
                    f"TransportLayer '{name}': kinetics must have trailing shape "
                    f"({K}, {K}), got {tuple(kinetics.shape)}"
                )
            if kinetics.dim() == 2:
                kinetics = kinetics.unsqueeze(0).expand(self.n_i, K, K)
            elif kinetics.shape[-3] != self.n_i:
                raise ValueError(
                    f"TransportLayer '{name}': kinetics must have {self.n_i} node rows, "
                    f"got {kinetics.shape[-3]}"
                )
        self.kinetics = kinetics

        if removal is not None:
            removal = torch.as_tensor(removal, dtype=net.dtype)
            if removal.dim() == 1:
                if removal.shape[-1] != K:
                    raise ValueError(
                        f"TransportLayer '{name}': removal must have shape ({K},) or "
                        f"({self.n_i}, {K}), got {tuple(removal.shape)}"
                    )
                removal = removal.unsqueeze(0).expand(self.n_i, K)
            elif removal.shape[-1] != K or removal.shape[-2] != self.n_i:
                raise ValueError(
                    f"TransportLayer '{name}': removal must have shape ({K},) or "
                    f"({self.n_i}, {K}), got {tuple(removal.shape)}"
                )
        self.removal = removal

        self.conduction_kind = conduction_kind
        if conduction_kind is not None:
            if conductance is None:
                raise ValueError(
                    f"TransportLayer '{name}': conductance is required when "
                    f"conduction_kind is given"
                )
            A_c = net.incidence(conduction_kind)
            b_c = A_c.shape[-1]
            g = torch.as_tensor(conductance, dtype=net.dtype)
            # Validate explicitly rather than let a mismatched length reach einsum: with a
            # single conduction_kind edge (b_c == 1), einsum's size-1 broadcasting for the
            # repeated "e" subscript would otherwise silently accept a wrongly-shaped g
            # (e.g. length 2) and sum it into L instead of raising, giving a silently wrong
            # conductance matrix rather than a ValueError naming the offender.
            if g.dim() == 0 or g.shape[-1] != b_c:
                raise ValueError(
                    f"TransportLayer '{name}': conductance must have shape ({b_c},), "
                    f"got {tuple(g.shape)}"
                )
            self.L = torch.einsum("ne,...e,me->...nm", A_c, g, A_c)
        else:
            self.L = torch.zeros(net.n, net.n, dtype=net.dtype)

        if not torch.all(self.carrier > 0):
            raise ValueError(
                f"TransportLayer '{name}': carrier must be strictly positive everywhere "
                f"(AdvectionOperator folds carrier into flow as flow = carrier * q, which "
                f"only preserves sign(q) and |carrier * q| == carrier * |q| when carrier > "
                f"0); got minimum value {self.carrier.min().item()}"
            )
        self._interior_of_node = torch.full((net.n,), -1, dtype=torch.long)
        self._interior_of_node[self.interior_idx] = torch.arange(self.n_i, dtype=torch.long)
        if conduction_kind is not None:
            csrc, ctgt = net.endpoints(conduction_kind)
            self._conduction_edges = (csrc, ctgt, torch.as_tensor(conductance, dtype=net.dtype))
        else:
            self._conduction_edges = None

    def _advection_operator(self, q: torch.Tensor) -> AdvectionOperator:
        dtype = q.dtype
        src, tgt = self.net.endpoints(self.flow_kind)
        conduction = None
        if self._conduction_edges is not None:
            csrc, ctgt, g = self._conduction_edges
            conduction = (csrc, ctgt, g.to(dtype))
        return AdvectionOperator(
            src, tgt,
            flow=self.carrier.to(dtype) * q,
            transmission=self.transmission.to(dtype),
            capacity=self.capacity.to(dtype),
            n_interior=self.n_i,
            interior_of_node=self._interior_of_node,
            kinetics=self.kinetics.to(dtype) if self.kinetics is not None else None,
            removal=self.removal.to(dtype) if self.removal is not None else None,
            conduction=conduction,
        )

    # ------------------------------------------------------------ assembly
    def _capacity_stacked(self, dtype: torch.dtype) -> torch.Tensor:
        K, n_i = self.n_species, self.n_i
        c = self.capacity.to(dtype)
        c = c.unsqueeze(-2).expand(*c.shape[:-1], K, n_i)
        return c.reshape(*c.shape[:-2], K * n_i)

    def operator(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        net, K, n_i, n_b = self.net, self.n_species, self.n_i, self.n_b
        dtype = q.dtype
        Up = net.upwind(q, self.flow_kind).to(dtype)    # (..., b_flow, n)
        Dn = net.downwind(q, self.flow_kind).to(dtype)  # (..., b_flow, n)
        w = self.carrier.to(dtype) * q.abs()            # (..., b_flow)
        Out = torch.einsum("...ei,...e,...ej->...ij", Up, w, Up)          # (..., n, n)
        weight = self.transmission.to(dtype) * w.unsqueeze(-2)           # (..., K, b_flow)
        In = torch.einsum("...ei,...ke,...ej->...kij", Dn, weight, Up)   # (..., K, n, n)
        L = self.L.to(dtype)
        G = In - Out.unsqueeze(-3) - L.unsqueeze(-3)                     # (..., K, n, n)

        idx_i, idx_b = self.interior_idx, self.boundary_idx
        Gii = G.index_select(-2, idx_i).index_select(-1, idx_i)   # (..., K, n_i, n_i)
        Gib = G.index_select(-2, idx_i).index_select(-1, idx_b)   # (..., K, n_i, n_b)

        # Divide the transport (advection + conduction) block by interior capacity here,
        # before removal/kinetics are added: those are caller-supplied per-second rate
        # constants (e.g. a deposition rate, a reaction rate constant) that already act
        # directly on the intensive state x, unlike the advective/conductive terms in Gii,
        # Gib, which are extensive flow rates that must be divided by capacity to become a
        # concentration/temperature rate. Dividing the *whole* stacked M (as a literal
        # reading of the spec's "finally, every row is divided by capacity" would do) instead
        # rescales removal/kinetics by 1/capacity too, which is wrong: e.g. the decay-chain
        # kinetics test below expects rate constants l1, l2 unchanged by capacity=1000, and
        # the removal test expects exp(-rate * t) with capacity=500 not entering at all.
        cap = self.capacity.to(dtype).unsqueeze(-2).unsqueeze(-1)  # (..., 1, n_i, 1)
        Gii = Gii / cap
        Gib = Gib / cap

        if self.removal is not None:
            Gii = Gii - torch.diag_embed(self.removal.to(dtype).transpose(-1, -2))

        eyeK = torch.eye(K, dtype=dtype)
        M_block = torch.einsum("kl,...kij->...kilj", eyeK, Gii)   # (..., K, n_i, K, n_i)
        N_block = torch.einsum("kl,...kij->...kilj", eyeK, Gib)   # (..., K, n_i, K, n_b)

        if self.kinetics is not None:
            eye_i = torch.eye(n_i, dtype=dtype)
            M_block = M_block + torch.einsum("ikl,ij->kilj", self.kinetics.to(dtype), eye_i)

        batch = M_block.shape[:-4]
        M = M_block.reshape(*batch, K * n_i, K * n_i)
        N = N_block.reshape(*N_block.shape[:-4], K * n_i, K * n_b)
        return M, N

    # ------------------------------------------------------------ stacking
    def _to_stacked(
        self, x: torch.Tensor, n_nodes: int, arg_name: str
    ) -> tuple[torch.Tensor, bool]:
        K = self.n_species
        if x.dim() >= 2 and x.shape[-2] == n_nodes and x.shape[-1] == K:
            return x.transpose(-1, -2).reshape(*x.shape[:-2], K * n_nodes), False
        if K == 1 and x.shape[-1] == n_nodes:
            return x, True
        raise ValueError(
            f"TransportLayer '{self.name}': {arg_name} must have trailing shape "
            f"({n_nodes}, {K}) or, for n_species=1, ({n_nodes},); got {tuple(x.shape)}"
        )

    def _from_stacked(self, x: torch.Tensor, n_nodes: int, reduced: bool) -> torch.Tensor:
        if reduced:
            return x
        K = self.n_species
        x = x.reshape(*x.shape[:-1], K, n_nodes)
        return x.transpose(-1, -2)

    def _forcing(
        self, sources: torch.Tensor, x_boundary: torch.Tensor, N: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, bool]:
        src_s, reduced = self._to_stacked(sources, self.n_i, "sources")
        xb_s, _ = self._to_stacked(x_boundary, self.n_b, "x_boundary")
        src_s, xb_s = src_s.to(dtype), xb_s.to(dtype)
        cap = self._capacity_stacked(dtype)
        b0 = (N @ xb_s.unsqueeze(-1)).squeeze(-1) + src_s / cap
        return b0, reduced

    # ------------------------------------------------------------ stepping
    def step(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        sources: torch.Tensor,
        x_boundary: torch.Tensor,
        dt: float,
        *,
        on_failure: str = "raise",
    ) -> torch.Tensor:
        """Advance one timestep under `self.scheme`.

        `on_failure` (keyword-only, default `"raise"`) is threaded to the `"implicit"` and
        `"trapezoidal"` schemes' underlying linear solve; on `"return"` those two schemes
        return the raw, stacked `SolveResult` instead of a plain `Tensor` (return type
        `torch.Tensor | SolveResult`, amendment A8). `"exact"` does not accept a failure
        mode here: it has no linear solve at all (Task 10's augmented matrix exponential
        controls its own error via sub-stepping, and raises `RuntimeError` directly on
        failure, as it always has).
        """
        if on_failure not in ("raise", "return"):
            raise ValueError(
                f"TransportLayer '{self.name}': unknown on_failure {on_failure!r}; "
                f"expected 'raise' or 'return'"
            )
        out_dtype = x.dtype
        dtype = torch.float64
        x_s, reduced = self._to_stacked(x, self.n_i, "x")
        x_s = x_s.to(dtype)
        if self.scheme == "exact":
            op = self._advection_operator(q.to(dtype))
            xb_s, _ = self._to_stacked(x_boundary.to(dtype), self.n_b, "x_boundary")
            src_s, _ = self._to_stacked(sources.to(dtype), self.n_i, "sources")
            cap = self._capacity_stacked(dtype)
            b0 = op.boundary_forcing(xb_s) + src_s / cap
            result, _substeps = _expm_action(
                op, x_s, b0, dt, where=f"TransportLayer '{self.name}' exact step"
            )
        elif self.scheme == "implicit":
            result, reduced = self._implicit_step_sparse(
                x, q, sources, x_boundary, dt, on_failure
            )
            if on_failure == "return":
                return result
        elif self.scheme == "trapezoidal":
            result, reduced = self._trapezoidal_step_sparse(
                x, q, sources, x_boundary, dt, on_failure
            )
            if on_failure == "return":
                return result
        else:
            raise ValueError(
                f"TransportLayer '{self.name}': unknown scheme {self.scheme!r}; "
                f"valid schemes are 'exact', 'implicit', 'trapezoidal'"
            )
        return self._from_stacked(result.to(out_dtype), self.n_i, reduced)

    def steady(
        self, q: torch.Tensor, sources: torch.Tensor, x_boundary: torch.Tensor,
        *, on_failure: str = "raise",
    ) -> torch.Tensor:
        """Solve `0 = M x + N x_b + sources / capacity` for the steady-state `x`.

        `on_failure="raise"` (the default) returns a plain `torch.Tensor`, matching every
        existing call site's expectation. `on_failure="return"` bypasses the differentiable
        `_linear_solve` path entirely and returns the raw, STACKED `SolveResult` from
        `solvers.select.solve` directly (not reshaped by `_from_stacked`): a `SolveResult`
        is not a plain `Tensor`, so it cannot be a single `torch.autograd.Function`'s output
        the way the default `Tensor` return is. Return type on that path is therefore
        `torch.Tensor | SolveResult` (amendment A8).
        """
        if on_failure not in ("raise", "return"):
            raise ValueError(
                f"TransportLayer '{self.name}': unknown on_failure {on_failure!r}; "
                f"expected 'raise' or 'return'"
            )
        dtype = torch.float64
        q = q.to(dtype)
        sources = sources.to(dtype)
        x_boundary = x_boundary.to(dtype)
        _, reduced = self._to_stacked(sources, self.n_i, "sources")

        def build_system(q_, sources_, xb_):
            op = self._advection_operator(q_)
            xb_s, _ = self._to_stacked(xb_, self.n_b, "x_boundary")
            src_s, _ = self._to_stacked(sources_, self.n_i, "sources")
            cap = self._capacity_stacked(dtype)
            b0 = op.boundary_forcing(xb_s) + src_s / cap
            return op, -b0

        if on_failure == "return":
            op, rhs = build_system(q, sources, x_boundary)
            return _solve_operator(
                op, rhs, method="auto", on_failure="return",
                where=f"TransportLayer '{self.name}' steady",
            )
        x_s = _linear_solve(
            build_system, f"TransportLayer '{self.name}' steady", q, sources, x_boundary
        )
        return self._from_stacked(x_s, self.n_i, reduced)

    def _implicit_step_sparse(
        self, x: torch.Tensor, q: torch.Tensor, sources: torch.Tensor,
        x_boundary: torch.Tensor, dt: float, on_failure: str,
    ) -> torch.Tensor:
        """Backward Euler `(I - dt M) x_{n+1} = x_n + dt b0` on the operator contract."""
        dtype = torch.float64
        x = x.to(dtype)
        q = q.to(dtype)
        sources = sources.to(dtype)
        x_boundary = x_boundary.to(dtype)
        _, reduced = self._to_stacked(x, self.n_i, "x")

        def build_system(x_, q_, sources_, xb_):
            op = self._advection_operator(q_)
            xb_s, _ = self._to_stacked(xb_, self.n_b, "x_boundary")
            src_s, _ = self._to_stacked(sources_, self.n_i, "sources")
            x_s, _ = self._to_stacked(x_, self.n_i, "x")
            cap = self._capacity_stacked(dtype)
            b0 = op.boundary_forcing(xb_s) + src_s / cap
            rhs = x_s + dt * b0
            system = _AffineSystemOperator(op, dt)
            return system, rhs

        if on_failure == "return":
            system, rhs = build_system(x, q, sources, x_boundary)
            result = _solve_operator(
                system, rhs, method="auto", on_failure="return",
                where=f"TransportLayer '{self.name}' implicit step",
            )
            return result, reduced
        x_s = _linear_solve(
            build_system, f"TransportLayer '{self.name}' implicit step",
            x, q, sources, x_boundary,
        )
        return x_s, reduced

    def _trapezoidal_step_sparse(
        self, x: torch.Tensor, q: torch.Tensor, sources: torch.Tensor,
        x_boundary: torch.Tensor, dt: float, on_failure: str,
    ) -> torch.Tensor:
        """Crank-Nicolson `(I - dt/2 M) x_{n+1} = (I + dt/2 M) x_n + dt b0` on the operator
        contract."""
        dtype = torch.float64
        x = x.to(dtype)
        q = q.to(dtype)
        sources = sources.to(dtype)
        x_boundary = x_boundary.to(dtype)
        _, reduced = self._to_stacked(x, self.n_i, "x")

        def build_system(x_, q_, sources_, xb_):
            op = self._advection_operator(q_)
            xb_s, _ = self._to_stacked(xb_, self.n_b, "x_boundary")
            src_s, _ = self._to_stacked(sources_, self.n_i, "sources")
            x_s, _ = self._to_stacked(x_, self.n_i, "x")
            cap = self._capacity_stacked(dtype)
            b0 = op.boundary_forcing(xb_s) + src_s / cap
            rhs = x_s + 0.5 * dt * op.matvec(x_s) + dt * b0
            system = _AffineSystemOperator(op, 0.5 * dt)
            return system, rhs

        if on_failure == "return":
            system, rhs = build_system(x, q, sources, x_boundary)
            result = _solve_operator(
                system, rhs, method="auto", on_failure="return",
                where=f"TransportLayer '{self.name}' trapezoidal step",
            )
            return result, reduced
        x_s = _linear_solve(
            build_system, f"TransportLayer '{self.name}' trapezoidal step",
            x, q, sources, x_boundary,
        )
        return x_s, reduced


def _van_loan_step_dense(
    M: torch.Tensor, x: torch.Tensor, b0: torch.Tensor, dt: float
) -> torch.Tensor:
    """Exact linear step via the augmented matrix exponential (Van Loan, 1978).

    Retained as the O(m^2) dense ORACLE for tests and for the small-system path; the
    operational path is `_expm_action`, which never forms this (2m, 2m) block.
    """
    m = M.shape[-1]
    batch = M.shape[:-2]
    Z = torch.zeros(*batch, 2 * m, 2 * m, dtype=M.dtype)
    Z[..., :m, :m] = M * dt
    Z[..., :m, m:] = torch.eye(m, dtype=M.dtype) * dt
    E = torch.linalg.matrix_exp(Z)
    Ed = E[..., :m, :m]
    Phi = E[..., :m, m:]
    return (Ed @ x.unsqueeze(-1)).squeeze(-1) + (Phi @ b0.unsqueeze(-1)).squeeze(-1)


def _expm_action(
    M: AdvectionOperator,
    x: torch.Tensor,
    b0: torch.Tensor,
    dt: float,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_terms: int = 60,
    max_substeps: int = 20,
    where: str = "TransportLayer exact step",
    _depth: int = 0,
) -> tuple[torch.Tensor, int]:
    """expm(dt * [[M, b0], [0, 0]]) @ [x, 1], as a scaling-and-squaring-free Taylor
    action in M -- see the module docstring / Task 10's plan for the derivation.
    Never forms a (2m, 2m), or even an (m, m), dense object.
    """
    u = M.matvec(x) + b0
    result = x.clone()
    term = u
    coef = dt
    converged = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
    j = 1
    while j <= max_terms:
        increment = coef * term
        result = torch.where(
            converged.unsqueeze(-1), result, result + increment
        )
        tol = atol + rtol * result.abs().amax(dim=-1, keepdim=True).squeeze(-1)
        finite = torch.isfinite(increment).all(dim=-1) & torch.isfinite(result).all(dim=-1)
        newly_converged = finite & (increment.abs().amax(dim=-1) <= tol)  # amendment A6
        converged = converged | newly_converged
        if bool(torch.all(converged)):
            return result, 1  # one leaf Taylor evaluation (amendment A6)
        term = M.matvec(term)
        j += 1
        coef = coef * dt / (j)
    if _depth >= max_substeps:
        bad = torch.nonzero(~converged.reshape(-1), as_tuple=False).flatten()
        raise RuntimeError(
            f"{where}: batch indices {bad.tolist()} failed to converge "
            f"the exponential action after {max_substeps} dt-halvings"
        )
    half = dt / 2
    x_mid, substeps_a = _expm_action(
        M, x, b0, half, rtol=rtol, atol=atol, max_terms=max_terms,
        max_substeps=max_substeps, where=where, _depth=_depth + 1,
    )
    x_end, substeps_b = _expm_action(
        M, x_mid, b0, half, rtol=rtol, atol=atol, max_terms=max_terms,
        max_substeps=max_substeps, where=where, _depth=_depth + 1,
    )
    return x_end, substeps_a + substeps_b  # amendment A6


