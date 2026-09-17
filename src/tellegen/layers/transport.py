"""Multi-species transport on nodal scalars advected by signed branch flows.

For interior capacity ``V`` (volume, or heat capacity), signed branch flow ``q``
on the edges of ``flow_kind`` (one edge kind, or several: ``q`` then carries the
kinds' flows concatenated in ``flow_kinds`` order, which is what
``PotentialFlowLayer.flows_of_kind(q, layer.flow_kinds)`` returns), a carrier
factor and a transmission fraction per edge:

    V dx/dt = (In(q) - Out(q)) x + N x_b + sources

``Out`` is the total weighted outflow leaving the upstream node of every edge and
``In`` is the transmitted weighted inflow arriving at the downstream node; the
full-node generator ``In - Out`` is split into an interior/interior block ``M``
and an interior/boundary block ``N``, both already divided by capacity. Species
are stacked species-major: for ``K`` species the stacked row/column index is
``k * n_i + i`` for interior node ``i`` (see the module docstring of
``TransportLayer.operator`` for why the stacked shape is used even when species
do not interact).

``sources`` is given in FULL node order (spec 4.2: trailing shape ``(n, K)``, or ``(n,)``
for ``n_species == 1``), covering every node of the network, not just this layer's
interior ones -- the same full-node order every other layer's inputs use, so a caller
(``Model``) can hand every layer the same per-node source tensor without slicing it per
layer. Boundary rows must be zero (refused by name, not silently dropped); ``step``,
``steady`` and ``rate`` all pull out the interior rows internally via
``_sources_interior``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from tellegen.operators.advection import AdvectionOperator
from tellegen.solvers.implicit import TransposeOperator as _TransposeView
from tellegen.solvers.select import solve as _solve_operator
from tellegen.topology import Network, Node

# `_TransposeView` used to be its own LinearOperator-shaped adjoint-view class, duplicating
# `solvers.implicit.TransposeOperator` method for method except for `spd_certificate` (this
# module's version always returned None; `TransposeOperator`'s forwards the wrapped
# operator's certificate iff it declares itself symmetric). Both this layer's operators
# (`AdvectionOperator`, `_AffineSystemOperator` below) declare `symmetric = False`, so
# `TransposeOperator.spd_certificate()` returns None for them exactly as the old local class
# did -- this alias changes nothing observable here, it only removes the duplicate.


def active_interior(
    net: Network, kinds: Sequence[str], boundary: Sequence[Node]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the non-boundary nodes of `net` into the ones `kinds` touch and the ones it does
    not: `(interior_idx, inactive_idx)`, both in NODE order, both excluding `boundary`.

    Spec 14, 4.5: a node that no edge of a layer's kinds touches is INACTIVE for that layer
    -- not an unknown, and not a singular row. A wall-mass node in a building network is the
    motivating case: it carries conduction edges and no airpath edges, so it belongs to the
    thermal layer's interior but not to a species layer's. The same holds a scale up: in the
    composed reference model the street and sewer nodes carry no `airpath` edge at all, so
    the `co2` layer over that network has 960 unknowns, 68 fewer than its 1028 non-boundary
    nodes.

    This is the ONE place the rule is decided. `TransportLayer` sizes its interior with it,
    `physics.species.SpeciesTransport` sizes its `interior`/`volumes` with it, and a caller
    building a layer's capacity vector (`benchmarks.composed_model`, and the building
    application's thermal builder) must size that vector with it too rather than re-deriving
    "every node that is not a boundary node" -- which is what the layer's capacity check
    compares against, and what it names in its error when the two disagree.

    Raises `KeyError` (via `Network.endpoints`) naming any kind that carries no edge at all,
    and via `Network.interior_index` any unknown boundary node. Construction-time only: the
    Python loop is over KINDS (a handful), never over nodes or edges.
    """
    touched = torch.zeros(net.n, dtype=torch.bool, device=net.device)
    for kind in kinds:
        src, tgt = net.endpoints(kind)
        touched[src] = True
        touched[tgt] = True
    all_interior = net.interior_index(boundary)
    active = touched[all_interior]
    return all_interior[active], all_interior[~active]


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
    """dx/dt = M x + N x_b + sources / capacity on this layer's ACTIVE interior nodes.

    The interior is the non-boundary nodes an edge of ``flow_kind`` (or of
    ``conduction_kind``) touches; the rest are ``inactive_idx`` and have no row here at all
    (spec 14, 4.5, and ``active_interior``). ``capacity`` is indexed by that active interior,
    NOT by every non-boundary node -- a caller sizing it must use ``active_interior`` too.

    ``quantity``/``unit`` are metadata a ``Model`` reports ("temperature"/"K",
    "concentration"/"ppm"); nothing in the numerics reads them.
    """

    def __init__(
        self,
        net: Network,
        name: str,
        *,
        capacity: torch.Tensor,
        flow_kind: str | Sequence[str],
        boundary: Sequence[Node],
        n_species: int = 1,
        carrier: torch.Tensor | float = 1.0,
        transmission: torch.Tensor | None = None,
        kinetics: torch.Tensor | None = None,
        removal: torch.Tensor | None = None,
        conduction_kind: str | None = None,
        conductance: torch.Tensor | None = None,
        scheme: Literal["exact", "implicit", "trapezoidal"] = "exact",
        quantity: str = "scalar",
        unit: str = "",
    ) -> None:
        self.net = net
        self.name = name
        self.flow_kind = flow_kind
        # One or several advecting edge kinds. `flow_kinds` is the tuple form every internal
        # site uses; `flow_kind` keeps the caller's own spelling. `q` is then the kinds'
        # flows CONCATENATED in this order (`_flow_slices` holds the block boundaries), which
        # is exactly what `PotentialFlowLayer.flows_of_kind(q, layer.flow_kinds)` produces:
        # a thermal layer advected by both "airpath" and "door" edges needs no new operator,
        # only both kinds' endpoints in one array.
        self.flow_kinds = (flow_kind,) if isinstance(flow_kind, str) else tuple(flow_kind)
        # Metadata only (spec 4.4): what this layer's state IS and what it is measured in.
        self.quantity, self.unit = quantity, unit
        self.boundary = list(boundary)
        self.n_species = n_species
        self.scheme = scheme
        # Spec 14, 4.5: nodes no edge of this layer's kinds touches are INACTIVE -- excluded
        # from the interior rather than left as an all-zero (singular) row. Conduction counts
        # as a touch: a wall-mass node with conduction edges and no airpath edge IS an
        # unknown of a thermal layer, and is not one of a species layer over the same
        # network. See `active_interior`, which is also what a caller must size `capacity`
        # with.
        kinds = self.flow_kinds if conduction_kind is None else (*self.flow_kinds, conduction_kind)
        self.interior_idx, self.inactive_idx = active_interior(net, kinds, self.boundary)
        self.boundary_idx = net.boundary_index(self.boundary)
        self.n_i = int(self.interior_idx.shape[0])
        self.n_b = int(self.boundary_idx.shape[0])
        self._inactive_names = [net.nodes[i] for i in self.inactive_idx.tolist()]

        self.capacity = torch.as_tensor(capacity, dtype=net.dtype)
        if self.capacity.dim() == 0 or self.capacity.shape[-1] != self.n_i:
            cap_len = self.capacity.shape[-1] if self.capacity.dim() >= 1 else 0
            raise ValueError(
                f"TransportLayer '{name}': capacity has {cap_len} entries, "
                f"expected {self.n_i} active interior nodes"
            )
        self.carrier = torch.as_tensor(carrier, dtype=net.dtype)

        b_flow = 0
        self._flow_slices: list[tuple[int, int]] = []
        for kind in self.flow_kinds:
            n_kind = int(net.edge_index(kind).numel())
            self._flow_slices.append((b_flow, b_flow + n_kind))
            b_flow += n_kind
        # The flow edges' endpoints, concatenated once here in `flow_kinds` order so that
        # every `_advection_operator` call is a plain attribute read (and so that a
        # multi-kind layer is one AdvectionOperator over all its flow edges, not several).
        src_parts, tgt_parts = zip(
            *(net.endpoints(kind) for kind in self.flow_kinds), strict=True
        )
        self._flow_src = torch.cat(src_parts)
        self._flow_tgt = torch.cat(tgt_parts)
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
            # The conduction edges' ENDPOINTS, never the (n, b_c) incidence matrix and never
            # the (n, n) Laplacian it used to build here: since Task 15 this tuple is the
            # layer's whole representation of its conduction topology. `_advection_operator`
            # (Task 9) already consumed exactly this; `operator()`, the dense oracle, now
            # forms its (n, n) `L` from it on demand (`_conduction_matrix`). The (n, n)
            # matrix was 8.5 MB at the composed model's reference size, grew 4x per node
            # doubling, and -- with no conduction configured, as in that model -- was a block
            # of ZEROS that `operator()` subtracted for nothing.
            csrc, ctgt = net.endpoints(conduction_kind)
            b_c = len(csrc)
            g = torch.as_tensor(conductance, dtype=net.dtype)
            # Validate explicitly rather than let a mismatched length reach the assembly:
            # with a single conduction_kind edge (b_c == 1), the einsum that used to build L
            # broadcast the repeated "e" subscript, so a wrongly-shaped g (e.g. length 2)
            # was silently summed in instead of raising, giving a silently wrong conductance
            # matrix rather than a ValueError naming the offender.
            if g.dim() == 0 or g.shape[-1] != b_c:
                raise ValueError(
                    f"TransportLayer '{name}': conductance must have shape ({b_c},), "
                    f"got {tuple(g.shape)}"
                )
            self._conduction_edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = (
                csrc,
                ctgt,
                g,
            )
        else:
            self._conduction_edges = None

        if not torch.all(self.carrier > 0):
            raise ValueError(
                f"TransportLayer '{name}': carrier must be strictly positive everywhere "
                f"(AdvectionOperator folds carrier into flow as flow = carrier * q, which "
                f"only preserves sign(q) and |carrier * q| == carrier * |q| when carrier > "
                f"0); got minimum value {self.carrier.min().item()}"
            )
        self._interior_of_node = torch.full((net.n,), -1, dtype=torch.long)
        self._interior_of_node[self.interior_idx] = torch.arange(self.n_i, dtype=torch.long)

    def _conduction_matrix(self, dtype: torch.dtype) -> torch.Tensor | None:
        """The (..., n, n) conduction Laplacian `A_c diag(g) A_c^T`, or None if no conduction.

        Built HERE, on demand, from the endpoint tuple rather than held as `self.L`: only
        `operator()` -- the dense oracle, which is (..., K, n, n) anyway -- wants a matrix,
        and every other path goes through `AdvectionOperator`'s own sparse conduction term.
        When there is no conduction this returns None rather than an (n, n) block of zeros,
        so the oracle skips a subtraction instead of allocating n^2 doubles to subtract
        nothing.

        `index_add` on a flattened (n*n,) view gives the same four entries per edge the
        einsum form did (+g at (s, s) and (t, t), -g at (s, t) and (t, s)), out of place and
        broadcasting over any leading batch dims `g` carries.
        """
        if self._conduction_edges is None:
            return None
        csrc, ctgt, g = self._conduction_edges
        g = g.to(dtype)
        n = self.net.n
        flat = torch.zeros(g.shape[:-1] + (n * n,), dtype=dtype, device=g.device)
        for rows, cols, sign in (
            (csrc, csrc, 1.0),
            (ctgt, ctgt, 1.0),
            (csrc, ctgt, -1.0),
            (ctgt, csrc, -1.0),
        ):
            flat = flat.index_add(-1, rows * n + cols, sign * g)
        return flat.reshape(g.shape[:-1] + (n, n))

    def _advection_operator(self, q: torch.Tensor) -> AdvectionOperator:
        dtype = q.dtype
        src, tgt = self._flow_src, self._flow_tgt
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
            # This layer's PRESCRIBED nodes, in the caller's own `boundary` order. Passed
            # explicitly because "not interior" is no longer the same set: an inactive node
            # is not interior and is not a boundary value either, and the operator would
            # otherwise expect an `x_boundary` entry for it.
            boundary_idx=self.boundary_idx,
        )

    # ------------------------------------------------------------ assembly
    def _capacity_stacked(self, dtype: torch.dtype) -> torch.Tensor:
        K, n_i = self.n_species, self.n_i
        c = self.capacity.to(dtype)
        c = c.unsqueeze(-2).expand(*c.shape[:-1], K, n_i)
        return c.reshape(*c.shape[:-2], K * n_i)

    def _selectors(self, selector, q: torch.Tensor) -> torch.Tensor:
        """`net.upwind`/`net.downwind` over every flow kind, concatenated along the EDGE
        dimension in `flow_kinds` order -- the order `q`'s blocks are in (`_flow_slices`).
        """
        if len(self.flow_kinds) == 1:
            return selector(q, self.flow_kinds[0])
        return torch.cat(
            [
                selector(q[..., lo:hi], kind)
                for kind, (lo, hi) in zip(self.flow_kinds, self._flow_slices, strict=True)
            ],
            dim=-2,
        )

    def operator(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        net, K, n_i, n_b = self.net, self.n_species, self.n_i, self.n_b
        dtype = q.dtype
        Up = self._selectors(net.upwind, q).to(dtype)    # (..., b_flow, n)
        Dn = self._selectors(net.downwind, q).to(dtype)  # (..., b_flow, n)
        w = self.carrier.to(dtype) * q.abs()            # (..., b_flow)
        Out = torch.einsum("...ei,...e,...ej->...ij", Up, w, Up)          # (..., n, n)
        weight = self.transmission.to(dtype) * w.unsqueeze(-2)           # (..., K, b_flow)
        In = torch.einsum("...ei,...ke,...ej->...kij", Dn, weight, Up)   # (..., K, n, n)
        G = In - Out.unsqueeze(-3)                                       # (..., K, n, n)
        L = self._conduction_matrix(dtype)
        if L is not None:
            G = G - L.unsqueeze(-3)

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

    def _sources_interior(self, sources: torch.Tensor) -> torch.Tensor:
        """Full-node `sources` -> interior rows (spec 4.2). Boundary rows must be zero.

        Accepts trailing shape `(n, K)` or, for `n_species == 1`, `(n,)`, in NODE order. A
        nonzero entry on a boundary node is refused by name rather than dropped: a source on
        a node whose value is prescribed is a modelling error, not a value to ignore. The
        same holds for an INACTIVE node (spec 14, 4.5): it has no row in this layer at all,
        so a source there could not be balanced by anything -- a CO2 source placed on a
        wall-mass node is a wiring mistake, and is named rather than silently discarded.
        """
        n, K = self.net.n, self.n_species
        if sources.dim() >= 2 and sources.shape[-2] == n and sources.shape[-1] == K:
            node_dim = sources.dim() - 2
        elif K == 1 and sources.shape[-1] == n:
            node_dim = sources.dim() - 1
        else:
            raise ValueError(
                f"TransportLayer '{self.name}': sources must be in FULL node order with "
                f"trailing shape ({n}, {K}) or, for n_species=1, ({n},); got "
                f"{tuple(sources.shape)}. Boundary rows must be zero."
            )
        self._refuse_nonzero_rows(sources, node_dim, self.boundary_idx, self.boundary, "boundary")
        self._refuse_nonzero_rows(
            sources, node_dim, self.inactive_idx, self._inactive_names, "inactive"
        )
        return sources.index_select(node_dim, self.interior_idx)

    def _refuse_nonzero_rows(self, sources, node_dim, idx, names, role) -> None:
        if idx.numel() == 0:
            return
        rows = sources.index_select(node_dim, idx)
        nonzero = rows != 0
        if node_dim == sources.dim() - 2:
            nonzero = nonzero.any(-1)
        nonzero = nonzero.reshape(-1, nonzero.shape[-1]).any(0)
        if bool(nonzero.any()):
            bad = [names[i] for i in nonzero.nonzero().flatten().tolist()]
            raise ValueError(
                f"TransportLayer '{self.name}': sources must be zero on {role} nodes; "
                f"nonzero at {bad}"
            )

    def rate(self, x, q, sources, x_boundary) -> torch.Tensor:
        """dx/dt = M x + N x_b + sources / capacity at (x, q); x's layout and dtype.

        The balance `Model.residuals` reports for a transport layer, and the oracle the
        energy-balance tests check against. Zero at the fixed point of `steady`.
        """
        dtype = torch.float64
        x_s, reduced = self._to_stacked(x.to(dtype), self.n_i, "x")
        op = self._advection_operator(q.to(dtype))
        xb_s, _ = self._to_stacked(x_boundary.to(dtype), self.n_b, "x_boundary")
        src_s, _ = self._to_stacked(
            self._sources_interior(sources).to(dtype), self.n_i, "sources"
        )
        r = op.matvec(x_s) + op.boundary_forcing(xb_s) + src_s / self._capacity_stacked(dtype)
        return self._from_stacked(r.to(x.dtype), self.n_i, reduced)

    def _forcing(
        self, sources: torch.Tensor, x_boundary: torch.Tensor, N: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, bool]:
        src_s, reduced = self._to_stacked(self._sources_interior(sources), self.n_i, "sources")
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

        `sources` is in FULL node order (spec 4.2), not just this layer's interior nodes;
        see the module docstring and `_sources_interior`. `on_failure` (keyword-only,
        default `"raise"`) is threaded to the `"implicit"` and
        `"trapezoidal"` schemes' underlying linear solve; on `"return"` those two schemes
        return the raw, stacked `SolveResult` instead of a plain `Tensor` (return type
        `torch.Tensor | SolveResult`, amendment A8). `"exact"` has no linear solve at all
        (Task 10's augmented matrix exponential controls its own error via sub-stepping, and
        raises `RuntimeError` directly on failure, as it always has), so it does not accept
        `on_failure="return"`: there is no `SolveResult` for it to produce, and silently
        falling back to `"raise"` behaviour would make the argument look like it had an
        effect it does not have. `on_failure="return"` with `scheme="exact"` therefore
        raises `ValueError` naming the layer.
        """
        if on_failure not in ("raise", "return"):
            raise ValueError(
                f"TransportLayer '{self.name}': unknown on_failure {on_failure!r}; "
                f"expected 'raise' or 'return'"
            )
        if on_failure == "return" and self.scheme == "exact":
            # Both `on_failure` checks are ARGUMENT VALIDITY and both belong here, before any
            # work. This one used to sit inside the `scheme == "exact"` branch below, after
            # `_to_stacked` had already validated and reshaped `x`, so a caller who passed
            # both a bad shape and this unusable combination was told about the shape (final
            # review M9). `on_failure='return'` is wrong for this scheme whatever the shapes.
            raise ValueError(
                f"TransportLayer '{self.name}': on_failure='return' has no effect for "
                f"scheme='exact' (there is no linear solve to return the status of; "
                f"a sub-stepping failure raises RuntimeError directly). Use the default "
                f"on_failure='raise', or a scheme with a linear solve ('implicit', "
                f"'trapezoidal')."
            )
        out_dtype = x.dtype
        dtype = torch.float64
        x_s, reduced = self._to_stacked(x, self.n_i, "x")
        x_s = x_s.to(dtype)
        if self.scheme == "exact":
            op = self._advection_operator(q.to(dtype))
            xb_s, _ = self._to_stacked(x_boundary.to(dtype), self.n_b, "x_boundary")
            src_s, _ = self._to_stacked(
                self._sources_interior(sources.to(dtype)), self.n_i, "sources"
            )
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

        `sources` is in FULL node order (spec 4.2), not just this layer's interior nodes;
        see the module docstring and `_sources_interior`. `on_failure="raise"` (the
        default) returns a plain `torch.Tensor`, matching every
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
        _, reduced = self._to_stacked(self._sources_interior(sources), self.n_i, "sources")

        def build_system(q_, sources_, xb_):
            op = self._advection_operator(q_)
            xb_s, _ = self._to_stacked(xb_, self.n_b, "x_boundary")
            src_s, _ = self._to_stacked(self._sources_interior(sources_), self.n_i, "sources")
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
            src_s, _ = self._to_stacked(self._sources_interior(sources_), self.n_i, "sources")
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
            src_s, _ = self._to_stacked(self._sources_interior(sources_), self.n_i, "sources")
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


