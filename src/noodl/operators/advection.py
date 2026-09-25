"""AdvectionOperator: the nonsymmetric transport spatial operator, defined by its action.

Two points of the derivation are not obvious from the code: rmatvec swaps the
upwind/downwind roles of the advective IN term, and the capacity division moves from the
output (matvec) to the input (rmatvec).
"""

from __future__ import annotations

import torch


def _bcast_index(idx: torch.Tensor, batch_shape: torch.Size, k: int) -> torch.Tensor:
    """Reshape a (b,) index tensor to broadcast against a (*batch_shape, k, b) tensor."""
    leading = (1,) * (len(batch_shape) + 1)
    return idx.view(*leading, -1).expand(*batch_shape, k, -1)


class AdvectionOperator:
    """M in dx/dt = M x + boundary_forcing(x_boundary); nonsymmetric, no dense n x n form."""

    symmetric = False

    def __init__(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        flow: torch.Tensor,
        transmission: torch.Tensor,
        capacity: torch.Tensor,
        n_interior: int,
        interior_of_node: torch.Tensor,
        *,
        kinetics: torch.Tensor | None = None,
        removal: torch.Tensor | None = None,
        conduction: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        boundary_idx: torch.Tensor | None = None,
    ) -> None:
        # ValueError for bad shapes, naming the offender. Without these, a wrong edge
        # count in `transmission` would construct silently and surface as an opaque
        # broadcast RuntimeError at the first matvec, and an `n_interior` inconsistent
        # with `interior_of_node` would never be detected at all.
        transmission = (
            transmission if transmission.dim() >= 2 else transmission.unsqueeze(0)
        )
        if src.shape != tgt.shape:
            raise ValueError(
                f"AdvectionOperator: src and tgt must have the same shape, got "
                f"src {tuple(src.shape)} and tgt {tuple(tgt.shape)}"
            )
        if transmission.shape[-1] != src.shape[-1]:
            raise ValueError(
                f"AdvectionOperator: transmission has {transmission.shape[-1]} edges in its "
                f"last dimension (shape {tuple(transmission.shape)}) but src/tgt have "
                f"{src.shape[-1]}"
            )
        n_interior_actual = int((interior_of_node >= 0).sum())
        if n_interior != n_interior_actual:
            raise ValueError(
                f"AdvectionOperator: n_interior is {n_interior} but interior_of_node "
                f"(shape {tuple(interior_of_node.shape)}) marks {n_interior_actual} nodes "
                f"as interior"
            )
        if capacity.shape[-1] != n_interior:
            raise ValueError(
                f"AdvectionOperator: capacity has {capacity.shape[-1]} entries in its last "
                f"dimension (shape {tuple(capacity.shape)}) but n_interior is {n_interior}"
            )
        n_species = transmission.shape[-2]
        if kinetics is not None and tuple(kinetics.shape[-3:]) != (
            n_interior,
            n_species,
            n_species,
        ):
            raise ValueError(
                f"AdvectionOperator: kinetics must have trailing shape "
                f"({n_interior}, {n_species}, {n_species}), got {tuple(kinetics.shape)}"
            )
        if removal is not None and tuple(removal.shape[-2:]) != (n_interior, n_species):
            raise ValueError(
                f"AdvectionOperator: removal must have trailing shape "
                f"({n_interior}, {n_species}), got {tuple(removal.shape)}"
            )

        self._src = src
        self._tgt = tgt
        self.flow = flow
        self.transmission = transmission
        self.capacity = capacity
        self.n_interior = n_interior
        self.kinetics = kinetics
        self.removal = removal
        self.conduction = conduction
        self.n_species = self.transmission.shape[-2]

        self._n = interior_of_node.shape[0]
        self._interior_idx = torch.nonzero(interior_of_node >= 0, as_tuple=True)[0]
        # Which nodes `boundary_forcing`'s `x_boundary` covers, and in WHICH ORDER. It
        # defaults to "every node that is not interior, in node order", which is what a
        # two-way interior/boundary split means. A caller whose node set has a third class --
        # `TransportLayer`'s INACTIVE nodes, which no edge of the layer's kinds touches, so
        # they are neither unknowns nor prescribed values -- passes its own
        # boundary index instead, so that `x_boundary` stays one entry per PRESCRIBED node
        # and is embedded at the node each entry actually names.
        self._boundary_idx = (
            torch.nonzero(interior_of_node < 0, as_tuple=True)[0]
            if boundary_idx is None
            else boundary_idx
        )
        self._n_edges = src.shape[0]

        # Per-shape caches for `_raw_action`, which runs once per transport matvec and
        # otherwise rebuilds the same index views every time. `_upwind_cache` holds the (up, down)
        # endpoint arrays per dtype; `_bcast_cache` holds their `_bcast_index` expansions per (name,
        # batch shape, K). Both hold only LONG index tensors -- derived from the SIGN of `flow` and
        # from `_src`/`_tgt`, never from `flow`'s values -- so nothing cached here can capture an
        # autograd graph, which a cached expansion of a float tensor (`flow`, `transmission`)
        # would: `flow` is a solved `q` on the differentiable path and does require grad.
        # Neither dict evicts, and neither needs to: an AdvectionOperator is rebuilt once
        # per transport step from that step's own `flow`, so its key set is the one or two
        # shapes that step uses and it is then discarded. The values are stride-0 views of
        # (b,) index arrays, so the cache holds no memory beyond those arrays either way.
        self._upwind_cache: dict[torch.dtype, tuple[torch.Tensor, torch.Tensor]] = {}
        self._bcast_cache: dict[tuple, torch.Tensor] = {}
        # Lazily built COO index pattern for `assemble_sparse` (row, col, the positions to
        # keep from the raw per-family value concatenation, and the node-index arrays used
        # to gather capacity per entry): built on first use, since topology is fixed but not
        # every AdvectionOperator is ever assembled sparsely. See `assemble_sparse`.
        self._coo_cache: dict | None = None

        self.dtype = flow.dtype
        self.device = flow.device
        shapes = [flow.shape[:-1], self.transmission.shape[:-2], capacity.shape[:-1]]
        if conduction is not None and conduction[2].dim() >= 1:
            shapes.append(conduction[2].shape[:-1])
        if removal is not None:
            shapes.append(removal.shape[:-2])
        if kinetics is not None:
            shapes.append(kinetics.shape[:-3])
        # EVERY coefficient family contributes to the operator's batch: an ensemble
        # batched in conductance, removal or kinetics alone is a first-class shape, exactly
        # like one batched in flow or capacity alone.
        self.batch_shape = torch.broadcast_shapes(*shapes)
        m = n_interior * self.n_species
        self.shape = (*self.batch_shape, m, m)

    # ---------------------------------------------------------------- raw action
    def _upwind(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """`(up, down)` endpoint arrays at the given dtype, cached per dtype.

        The dtype is part of the key rather than ignored because the sign test is made on
        `flow.to(dtype)`: a value that underflows to -0.0 in a narrower dtype tests
        `>= 0` as TRUE where the wider value tested false, so the two dtypes genuinely can
        disagree about which end of an edge is upwind.
        """
        cached = self._upwind_cache.get(dtype)
        if cached is None:
            positive = self.flow.to(dtype) >= 0
            cached = (
                torch.where(positive, self._src, self._tgt),      # (..., b) upwind
                torch.where(positive, self._tgt, self._src),      # (..., b) downwind
            )
            self._upwind_cache[dtype] = cached
        return cached

    def _expanded(
        self, name: tuple, idx: torch.Tensor, batch_shape: torch.Size, k: int
    ) -> torch.Tensor:
        """`idx` broadcast against a (*batch_shape, k, b) tensor, cached under `name`.

        `_bcast_index` for a shared (b,) index array, and the equivalent unsqueeze/expand
        for an already-batched one. The result is a stride-0 VIEW, so the cache costs
        nothing beyond the (b,) array it views; `name` must identify the index array
        (including any dtype it was derived at), since that is what the key cannot see.
        """
        key = (name, tuple(batch_shape), k)
        cached = self._bcast_cache.get(key)
        if cached is None:
            cached = (
                _bcast_index(idx, batch_shape, k)
                if idx.dim() == 1
                else idx.unsqueeze(-2).expand(*batch_shape, k, idx.shape[-1])
            )
            self._bcast_cache[key] = cached
        return cached

    def _raw_action(self, v: torch.Tensor, *, transpose: bool) -> torch.Tensor:
        dtype = v.dtype
        flow = self.flow.to(dtype)
        up, down = self._upwind(dtype)
        w = flow.abs()

        # The state must broadcast against the OPERATOR's batch, not only against its own:
        # one initial condition against an ensemble of flow realisations (an unbatched `v`
        # with a batched `flow`) is a first-class shape here, per the framework's
        # "arbitrary leading batch dimensions, torch.broadcast_tensors semantics" rule.
        # Taking `v.shape[:-2]` alone made the index tensors (which carry `flow`'s batch)
        # disagree with `v_b`, raising a raw RuntimeError out of `_bcast_index`. When every
        # batch shape already agrees, the broadcast and the expand
        # below are both no-ops.
        batch_shape = torch.broadcast_shapes(v.shape[:-2], self.batch_shape)
        K = v.shape[-2]
        gather_name = ("down", dtype) if transpose else ("up", dtype)
        scatter_name = ("up", dtype) if transpose else ("down", dtype)
        gather_idx = down if transpose else up
        scatter_idx = up if transpose else down
        gather_idx_b = self._expanded(gather_name, gather_idx, batch_shape, K)
        scatter_idx_b = self._expanded(scatter_name, scatter_idx, batch_shape, K)

        v_b = v.expand(*batch_shape, K, self._n)
        x_g = torch.gather(v_b, -1, gather_idx_b)
        weight = self.transmission.to(dtype) * w.unsqueeze(-2)   # (..., K, b)
        in_val = weight.expand(*batch_shape, K, self._n_edges) * x_g
        in_action = torch.zeros(*batch_shape, K, self._n, dtype=dtype, device=v.device)
        in_action.scatter_add_(-1, scatter_idx_b, in_val)

        up_b = self._expanded(("up", dtype), up, batch_shape, K)
        w_b = w.unsqueeze(-2).expand(*batch_shape, K, self._n_edges)
        out_weight = torch.zeros(*batch_shape, K, self._n, dtype=dtype, device=v.device)
        out_weight.scatter_add_(-1, up_b, w_b)
        out_action = out_weight * v_b

        l_action = torch.zeros(*batch_shape, K, self._n, dtype=dtype, device=v.device)
        if self.conduction is not None:
            csrc, ctgt, g = self.conduction
            g = g.to(dtype)
            diff = v_b.index_select(-1, csrc) - v_b.index_select(-1, ctgt)
            weighted = g.unsqueeze(-2).expand(*batch_shape, K, csrc.shape[-1]) * diff \
                if g.dim() >= 1 and g.shape[-1] == csrc.shape[-1] else g * diff
            csrc_b = self._expanded(("csrc",), csrc, batch_shape, K)
            ctgt_b = self._expanded(("ctgt",), ctgt, batch_shape, K)
            l_action.scatter_add_(-1, csrc_b, weighted)
            l_action.scatter_add_(-1, ctgt_b, -weighted)

        return in_action - out_action - l_action

    def _embed(self, x_kn: torch.Tensor, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        shape = x_kn.shape[:-1] + (self._n,)
        v = torch.zeros(shape, dtype=dtype, device=x_kn.device)
        v[..., idx] = x_kn
        return v

    # ---------------------------------------------------------------- LinearOperator
    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        K, n_i = self.n_species, self.n_interior
        dtype = x.dtype
        x_kn = x.reshape(*x.shape[:-1], K, n_i)
        v = self._embed(x_kn, self._interior_idx, dtype)
        raw = self._raw_action(v, transpose=False)
        y_i = raw.index_select(-1, self._interior_idx)
        cap = self.capacity.to(dtype).unsqueeze(-2)
        y_i = y_i / cap
        if self.removal is not None:  # a rate on x itself, not divided by capacity
            y_i = y_i - self.removal.to(dtype).transpose(-1, -2) * x_kn
        if self.kinetics is not None:  # likewise
            y_i = y_i + torch.einsum("...ikl,...li->...ki", self.kinetics.to(dtype), x_kn)
        return y_i.reshape(*y_i.shape[:-2], K * n_i)

    def boundary_forcing(self, x_boundary: torch.Tensor) -> torch.Tensor:
        K, n_i = self.n_species, self.n_interior
        dtype = x_boundary.dtype
        n_b = self._boundary_idx.shape[0]
        xb_kn = x_boundary.reshape(*x_boundary.shape[:-1], K, n_b)
        v = self._embed(xb_kn, self._boundary_idx, dtype)
        raw = self._raw_action(v, transpose=False)
        y_i = raw.index_select(-1, self._interior_idx)
        cap = self.capacity.to(dtype).unsqueeze(-2)
        y_i = y_i / cap
        return y_i.reshape(*y_i.shape[:-2], K * n_i)

    def boundary_net_inflow(
        self, x_interior: torch.Tensor, x_boundary: torch.Tensor
    ) -> torch.Tensor:
        """Net rate at which AMOUNT enters each boundary node from this layer: the boundary
        rows of the capacity-free generator applied to the full state (interior values at
        interior nodes, prescribed values at boundary nodes). Positive means the interior is
        losing amount to that boundary node. Carrier, transmission and conduction enter
        exactly as they do in `matvec`; there is no capacity division, since a boundary node
        holds no storage of this layer. Stacked layouts on both sides: `(..., K*n_i)` in,
        `(..., K*n_b)` out."""
        K, n_i = self.n_species, self.n_interior
        n_b = self._boundary_idx.shape[0]
        dtype = x_interior.dtype

        # Reshape and embed interior values
        xi_kn = x_interior.reshape(*x_interior.shape[:-1], K, n_i)
        v_interior = self._embed(xi_kn, self._interior_idx, dtype)

        # Reshape and embed boundary values
        xb_kn = x_boundary.to(dtype).reshape(*x_boundary.shape[:-1], K, n_b)
        v_boundary = self._embed(xb_kn, self._boundary_idx, dtype)

        # If the two embedded tensors have different batch shapes due to broadcasting,
        # broadcast both to a common batch shape first
        batch_shape = torch.broadcast_shapes(v_interior.shape[:-2], v_boundary.shape[:-2])
        if v_interior.shape[:-2] != batch_shape:
            v_interior = v_interior.expand(*batch_shape, K, self._n)
        if v_boundary.shape[:-2] != batch_shape:
            v_boundary = v_boundary.expand(*batch_shape, K, self._n)

        # Combine full state
        v = v_interior + v_boundary

        # Apply raw action and select boundary rows
        raw = self._raw_action(v, transpose=False)
        y_b = raw.index_select(-1, self._boundary_idx)

        return y_b.reshape(*y_b.shape[:-2], K * n_b)

    def rmatvec(self, y: torch.Tensor) -> torch.Tensor:
        K, n_i = self.n_species, self.n_interior
        dtype = y.dtype
        y_kn = y.reshape(*y.shape[:-1], K, n_i)
        cap = self.capacity.to(dtype).unsqueeze(-2)
        z_kn = y_kn / cap
        v = self._embed(z_kn, self._interior_idx, dtype)
        raw_t = self._raw_action(v, transpose=True)
        out_i = raw_t.index_select(-1, self._interior_idx)
        if self.removal is not None:  # transpose of the removal term
            out_i = out_i - self.removal.to(dtype).transpose(-1, -2) * y_kn
        if self.kinetics is not None:  # transpose of the kinetics block
            out_i = out_i + torch.einsum(
                "...ikl,...li->...ki", self.kinetics.to(dtype).transpose(-1, -2), y_kn
            )
        return out_i.reshape(*out_i.shape[:-2], K * n_i)

    def diagonal(self) -> torch.Tensor:
        K, n_i = self.n_species, self.n_interior
        dtype = self.flow.dtype
        flow = self.flow.to(dtype)
        w = flow.abs()
        batch_shape = self.batch_shape
        w_b = w.unsqueeze(-2).expand(*batch_shape, K, self._n_edges)

        up = torch.where(flow >= 0, self._src, self._tgt)
        up_b = _bcast_index(up, batch_shape, K) if up.dim() == 1 \
            else up.unsqueeze(-2).expand(*batch_shape, K, up.shape[-1])
        out_diag_full = torch.zeros(*batch_shape, K, self._n, dtype=dtype, device=w.device)
        out_diag_full.scatter_add_(-1, up_b, w_b)

        self_loop = (self._src == self._tgt)
        weight = self.transmission.to(dtype) * w.unsqueeze(-2)
        weight = weight.expand(*batch_shape, K, self._n_edges)
        masked = torch.where(
            self_loop.expand(*batch_shape, K, self._n_edges), weight, torch.zeros_like(weight)
        )
        src_b = _bcast_index(self._src, batch_shape, K)
        in_diag_full = torch.zeros(*batch_shape, K, self._n, dtype=dtype, device=w.device)
        in_diag_full.scatter_add_(-1, src_b, masked)

        l_diag_full = torch.zeros(*batch_shape, K, self._n, dtype=dtype, device=w.device)
        if self.conduction is not None:
            csrc, ctgt, g = self.conduction
            g_b = g.to(dtype)
            g_b = g_b.unsqueeze(-2).expand(*batch_shape, K, csrc.shape[-1]) if g_b.dim() >= 1 \
                else g_b
            csrc_b = _bcast_index(csrc, batch_shape, K)
            ctgt_b = _bcast_index(ctgt, batch_shape, K)
            l_diag_full.scatter_add_(-1, csrc_b, g_b)
            l_diag_full.scatter_add_(-1, ctgt_b, g_b)

        raw_diag = in_diag_full - out_diag_full - l_diag_full
        diag_i = raw_diag.index_select(-1, self._interior_idx)
        cap = self.capacity.to(dtype).unsqueeze(-2)
        diag_i = diag_i / cap
        if self.removal is not None:
            diag_i = diag_i - self.removal.to(dtype).transpose(-1, -2)
        if self.kinetics is not None:
            kdiag = torch.diagonal(self.kinetics.to(dtype), dim1=-2, dim2=-1)  # (n_i, K)
            diag_i = diag_i + kdiag.transpose(-1, -2)
        return diag_i.reshape(*diag_i.shape[:-2], K * n_i)

    def abs_column_sums(self) -> torch.Tensor:
        """sum_i |M_ij| for every column j (stacked index k * n_interior + i), so that
        `.amax(-1)` is the exact 1-norm of the interior operator. O(edges), no dense form.

        The DIAGONAL entry of column (node u, species k) is `self.diagonal()`'s own entry:
        `diagonal()` already combines, SIGNED, everything that lands there -- the Out term,
        a self-loop edge's In term, conduction's own-node term, `removal`, and kinetics'
        own-species term `K_{u,k,k}` -- so taking `.abs()` of that one combined number is
        exact even when, e.g., a signed kinetics self-term partially cancels the (always
        non-positive) Out/removal diagonal; summing each piece's `.abs()` separately would
        overcount that cancellation and fail to reproduce the dense assembly bit for bit.

        The OFF-diagonal entries never share a matrix position with each other or with the
        diagonal, so their magnitudes add directly: the IN term t_{k,e} |w_e| / cap[d] of
        every non-self-loop edge whose downwind node d is interior (a boundary d is not a
        row of M and contributes nothing); a conduction edge (u, v, g)'s cross term, g /
        cap[v] in u's column and g / cap[u] in v's; and the kinetics column's off-species
        terms sum_{l != k} |K_{u,l,k}|.
        """
        K, n_i, n = self.n_species, self.n_interior, self._n
        dtype = self.flow.dtype
        flow = self.flow.to(dtype)
        w = flow.abs()
        batch_shape = self.batch_shape
        up = torch.where(flow >= 0, self._src, self._tgt).expand(*batch_shape, self._n_edges)
        down = torch.where(flow >= 0, self._tgt, self._src).expand(*batch_shape, self._n_edges)
        w = w.expand(*batch_shape, self._n_edges)
        inv_cap = torch.zeros(*batch_shape, n, dtype=dtype, device=w.device)
        inv_cap[..., self._interior_idx] = 1.0 / self.capacity.to(dtype).expand(*batch_shape, n_i)

        # Off-diagonal In term: exclude self-loop edges (up == down), whose contribution is
        # already inside the signed diagonal above -- adding its magnitude again here would
        # both double-count it and defeat any cancellation already resolved there.
        not_self_loop = (self._src != self._tgt).expand(*batch_shape, self._n_edges)
        weight = self.transmission.to(dtype).expand(*batch_shape, K, self._n_edges)
        weight = weight * w.unsqueeze(-2)
        in_val = weight * torch.gather(inv_cap, -1, down).unsqueeze(-2)          # (..., K, b)
        in_val = torch.where(not_self_loop.unsqueeze(-2), in_val, torch.zeros_like(in_val))
        up_k = up.unsqueeze(-2).expand(*batch_shape, K, self._n_edges)
        in_col = torch.zeros(*batch_shape, K, n, dtype=dtype, device=w.device)
        in_col = in_col.scatter_add(-1, up_k, in_val)

        cond_col = torch.zeros(*batch_shape, n, dtype=dtype, device=w.device)
        if self.conduction is not None:
            csrc, ctgt, g = self.conduction
            g = g.to(dtype).expand(*batch_shape, csrc.shape[-1])
            csrc_b = csrc.expand(*batch_shape, -1)
            ctgt_b = ctgt.expand(*batch_shape, -1)
            cond_col = cond_col.scatter_add(-1, csrc_b, g * torch.gather(inv_cap, -1, ctgt_b))
            cond_col = cond_col.scatter_add(-1, ctgt_b, g * torch.gather(inv_cap, -1, csrc_b))

        off = cond_col.unsqueeze(-2) + in_col                                    # (..., K, n)
        off = off.index_select(-1, self._interior_idx)                            # (..., K, n_i)
        if self.kinetics is not None:
            kin_abs = self.kinetics.to(dtype).abs()
            kin_self = torch.diagonal(kin_abs, dim1=-2, dim2=-1)                  # (..., n_i, K)
            kin_off = kin_abs.sum(-2) - kin_self                                  # sum rows l != k
            off = off + kin_off.transpose(-1, -2)
        off = off.reshape(*off.shape[:-2], K * n_i)
        return off + self.diagonal().abs()

    def norm1_bound(self) -> torch.Tensor:
        """||M||_1 per batch instance: a norm, so it bounds the Taylor truncation error for
        non-normal and nilpotent operators alike (a power iteration would not)."""
        return self.abs_column_sums().amax(-1)

    def assemble(self) -> torch.Tensor:
        K, n_i, n = self.n_species, self.n_interior, self._n
        dtype = self.flow.dtype
        flow = self.flow.to(dtype)
        batch_shape = self.batch_shape
        eye_n = torch.eye(n, dtype=dtype, device=flow.device)
        up = torch.where(flow >= 0, self._src, self._tgt)
        down = torch.where(flow >= 0, self._tgt, self._src)
        Up = eye_n[up]      # (..., b, n) one-hot
        Dn = eye_n[down]    # (..., b, n) one-hot
        w = flow.abs()
        Out = torch.einsum("...ei,...e,...ej->...ij", Up, w, Up)
        weight = self.transmission.to(dtype) * w.unsqueeze(-2)     # (..., K, b)
        In = torch.einsum("...ei,...ke,...ej->...kij", Dn, weight, Up)
        L = torch.zeros(n, n, dtype=dtype, device=flow.device)
        if self.conduction is not None:
            csrc, ctgt, g = self.conduction
            A_c = torch.zeros(n, csrc.shape[-1], dtype=dtype, device=flow.device)
            A_c[csrc, torch.arange(csrc.shape[-1])] += 1
            A_c[ctgt, torch.arange(ctgt.shape[-1])] -= 1
            L = torch.einsum("nc,...c,mc->...nm", A_c, g.to(dtype), A_c)
        G = In - Out.unsqueeze(-3) - L.unsqueeze(-3) if L.dim() > 2 else \
            In - Out.unsqueeze(-3) - L.expand(n, n).unsqueeze(-3)
        Gii = G.index_select(-2, self._interior_idx).index_select(-1, self._interior_idx)
        cap = self.capacity.to(dtype).unsqueeze(-2).unsqueeze(-1)
        Gii = Gii / cap
        if self.removal is not None:
            Gii = Gii - torch.diag_embed(self.removal.to(dtype).transpose(-1, -2))
        eyeK = torch.eye(K, dtype=dtype, device=flow.device)
        M_block = torch.einsum("kl,...kij->...kilj", eyeK, Gii)
        if self.kinetics is not None:
            eye_i = torch.eye(n_i, dtype=dtype, device=flow.device)
            M_block = M_block + torch.einsum(
                "...ikl,ij->...kilj", self.kinetics.to(dtype), eye_i
            )
        return M_block.expand(*batch_shape, K, n_i, K, n_i).reshape(
            *batch_shape, K * n_i, K * n_i
        )

    def assemble_sparse(self):
        """COO `(row, col, values)` for the interior generator M.

        The optional `operators.base.SparseAssembling` member. `M` is nonsymmetric (its
        `spd_certificate()` is `None`, so `method="auto"` still routes it to GMRES whatever
        this returns), but `method="sparse_direct"` and the ILU preconditioner both need this
        form, and
        so does `_AffineSystemOperator.assemble_sparse` (`I - alpha * M`), which just adds an
        identity diagonal to it.

        THE STENCIL, derived from `_raw_action`/`diagonal` and matching `assemble()` bit for
        bit (interior rows and columns only -- a boundary column is forcing, not part of
        `M`; every entry divided by the ROW node's interior capacity except removal and
        kinetics, which act on the intensive state directly). Species are stacked as
        `k * n_interior + i`:

          * Out, flow edge e, orientation `up = src` (active where `flow >= 0`): for every
            species k, `(k*n_i + up, k*n_i + up) -= abs(w_e) / cap[up]`, where `up` interior.
          * In, same orientation (`down = tgt`): `(k*n_i + down, k*n_i + up)
            += t[k, e] * abs(w_e) / cap[down]`, where `up` AND `down` interior.
          * Out and In, orientation `up = tgt, down = src` (active where `flow < 0`): as
            above with the endpoint roles swapped.
          * conduction edge (u, v, g): `(u, u)` and `(v, v)` each `-= g / cap[row]`; `(u, v)`
            and `(v, u)` each `+= g / cap[row]`; every species; row and col interior.
          * removal: `(k*n_i + i, k*n_i + i) -= removal[i, k]`.
          * kinetics: `(k*n_i + i, l*n_i + i) += kinetics[i, k, l]` for all k, l.

        BOTH flow orientations are emitted for every edge, so the index pattern (`row`,
        `col`) never depends on the SIGN of `flow`: only the inactive orientation's VALUES
        are multiplied by 0.0 per batch instance (`positive = flow >= 0`, then `positive`/
        `~positive` as float masks). This is what lets one index set serve a whole batch of
        signed flow fields, at the cost of at most doubling nnz for the flow terms. A
        self-loop edge (`src == tgt`) degenerates harmlessly: both orientations land on the
        same (row, col), and the batch mask still picks exactly one contribution, matching
        `assemble()`'s single-orientation self-loop term. An edge with `flow == 0` is zeroed
        by `abs(w_e) == 0` regardless of the mask, matching `assemble()` exactly (which is
        indifferent to which "orientation" a zero-weight edge is assigned).

        CACHING mirrors `GraphLaplacianOperator.assemble_sparse`: topology is fixed once an
        instance is built (only coefficient VALUES vary from one call to the next), so the
        interior compact-index map, the final filtered `(row, col)` and the per-edge/
        per-conduction-edge node-index arrays needed to gather values are all built ONCE, on
        first call, and reused. Only `values` -- one O(edges) vectorised expression, no
        Python loop -- is recomputed every call.
        """
        K, n_i = self.n_species, self.n_interior
        dtype, device = self.dtype, self.device
        cache = self._coo_cache
        if cache is None:
            compact = torch.full((self._n,), -1, dtype=torch.long, device=device)
            compact[self._interior_idx] = torch.arange(n_i, device=device)

            up_c_A, down_c_A = compact[self._src], compact[self._tgt]
            # Orientation B is the mirror of A: up/down simply swap roles.

            k_idx = torch.arange(K, device=device)

            def tile(node_idx: torch.Tensor) -> torch.Tensor:
                """`k * n_i + node_idx[e]`, flattened k-major (matches `.repeat(K)`)."""
                return (k_idx.view(K, 1) * n_i + node_idx.view(1, -1)).reshape(-1)

            row_parts, col_parts, keep_parts = [], [], []

            # Out, both orientations (diagonal in (row, col)).
            row_parts += [tile(up_c_A), tile(down_c_A)]
            col_parts += [tile(up_c_A), tile(down_c_A)]
            keep_parts += [(up_c_A >= 0).repeat(K), (down_c_A >= 0).repeat(K)]

            # In, both orientations.
            keep_in = (up_c_A >= 0) & (down_c_A >= 0)
            row_parts += [tile(down_c_A), tile(up_c_A)]
            col_parts += [tile(up_c_A), tile(down_c_A)]
            keep_parts += [keep_in.repeat(K), keep_in.repeat(K)]

            u_c = v_c = None
            if self.conduction is not None:
                csrc, ctgt, _ = self.conduction
                u_c, v_c = compact[csrc], compact[ctgt]
                keep_uv = (u_c >= 0) & (v_c >= 0)
                row_parts += [tile(u_c), tile(v_c), tile(u_c), tile(v_c)]
                col_parts += [tile(u_c), tile(v_c), tile(v_c), tile(u_c)]
                keep_parts += [
                    (u_c >= 0).repeat(K), (v_c >= 0).repeat(K),
                    keep_uv.repeat(K), keep_uv.repeat(K),
                ]

            if self.removal is not None:
                i_idx = torch.arange(n_i, device=device)
                row_parts.append(tile(i_idx))
                col_parts.append(tile(i_idx))
                keep_parts.append(torch.ones(K * n_i, dtype=torch.bool, device=device))

            if self.kinetics is not None:
                i_idx = torch.arange(n_i, device=device)
                l_idx = torch.arange(K, device=device)
                row_kli = (k_idx.view(K, 1, 1) * n_i + i_idx.view(1, 1, -1)).expand(K, K, n_i)
                col_kli = (l_idx.view(1, K, 1) * n_i + i_idx.view(1, 1, -1)).expand(K, K, n_i)
                row_parts.append(row_kli.reshape(-1))
                col_parts.append(col_kli.reshape(-1))
                keep_parts.append(torch.ones(K * K * n_i, dtype=torch.bool, device=device))

            row_raw = torch.cat(row_parts).to(torch.int64)
            col_raw = torch.cat(col_parts).to(torch.int64)
            keep_idx = torch.nonzero(torch.cat(keep_parts), as_tuple=False).flatten()
            cache = {
                "row": row_raw[keep_idx], "col": col_raw[keep_idx], "keep_idx": keep_idx,
                "up_c_A": up_c_A, "down_c_A": down_c_A, "u_c": u_c, "v_c": v_c,
            }
            self._coo_cache = cache

        row, col, keep_idx = cache["row"], cache["col"], cache["keep_idx"]
        up_c_A, down_c_A = cache["up_c_A"], cache["down_c_A"]
        batch_shape = self.batch_shape

        def cap_at(idx: torch.Tensor) -> torch.Tensor:
            return self.capacity.to(dtype).index_select(-1, idx.clamp_min(0))

        def tile_species(x: torch.Tensor, n: int) -> torch.Tensor:
            """A (..., n) value with no species axis, broadcast across all K species."""
            x = x.expand(*batch_shape, n)
            return x.unsqueeze(-2).expand(*batch_shape, K, n).reshape(*batch_shape, K * n)

        def flat_species(x: torch.Tensor, n: int) -> torch.Tensor:
            """A (..., K, n) value, expanded to batch_shape and flattened k-major."""
            return x.expand(*batch_shape, K, n).reshape(*batch_shape, K * n)

        flow = self.flow.to(dtype)
        w = flow.abs()
        pos = (flow >= 0).to(dtype)
        neg = 1.0 - pos
        t = self.transmission.to(dtype)
        n_e = self._n_edges

        cap_up, cap_down = cap_at(up_c_A), cap_at(down_c_A)
        val_parts = [
            tile_species(-(w / cap_up) * pos, n_e),        # Out, orientation A
            tile_species(-(w / cap_down) * neg, n_e),       # Out, orientation B
            flat_species(t * (w / cap_down * pos).unsqueeze(-2), n_e),   # In, orientation A
            flat_species(t * (w / cap_up * neg).unsqueeze(-2), n_e),    # In, orientation B
        ]

        if self.conduction is not None:
            _csrc, _ctgt, g = self.conduction
            g = g.to(dtype)
            n_c = g.shape[-1]
            cap_u, cap_v = cap_at(cache["u_c"]), cap_at(cache["v_c"])
            val_parts += [
                tile_species(-(g / cap_u), n_c),
                tile_species(-(g / cap_v), n_c),
                tile_species(g / cap_u, n_c),
                tile_species(g / cap_v, n_c),
            ]

        if self.removal is not None:
            removal = self.removal.to(dtype).transpose(-1, -2)  # (..., K, n_i)
            val_parts.append(flat_species(removal.neg(), n_i))

        if self.kinetics is not None:
            kin = self.kinetics.to(dtype).movedim(-3, -1)  # (..., K(k), K(l), n_i)
            kin = kin.expand(*batch_shape, K, K, n_i)
            val_parts.append(kin.reshape(*batch_shape, K * K * n_i))

        values = torch.cat(val_parts, dim=-1).index_select(-1, keep_idx)
        return row, col, values

    def spd_certificate(self):
        return None
