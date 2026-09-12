"""AdvectionOperator: the nonsymmetric transport spatial operator, defined by its action.

See docs/superpowers/plans/2026-09-12-milestone-1b-sparse.md, Task 8, for the full
derivation, including why rmatvec swaps the upwind/downwind roles of the advective IN
term and why the capacity division moves from output (matvec) to input (rmatvec).
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
    ) -> None:
        self._src = src
        self._tgt = tgt
        self.flow = flow
        self.transmission = transmission if transmission.dim() >= 2 else transmission.unsqueeze(0)
        self.capacity = capacity
        self.n_interior = n_interior
        self.kinetics = kinetics
        self.removal = removal
        self.conduction = conduction
        self.n_species = self.transmission.shape[-2]

        self._n = interior_of_node.shape[0]
        self._interior_idx = torch.nonzero(interior_of_node >= 0, as_tuple=True)[0]
        self._boundary_idx = torch.nonzero(interior_of_node < 0, as_tuple=True)[0]
        self._n_edges = src.shape[0]

        self.dtype = flow.dtype
        self.device = flow.device
        batch = torch.broadcast_shapes(
            flow.shape[:-1], self.transmission.shape[:-2], capacity.shape[:-1]
        )
        m = n_interior * self.n_species
        self.shape = (*batch, m, m)

    # ---------------------------------------------------------------- raw action
    def _raw_action(self, v: torch.Tensor, *, transpose: bool) -> torch.Tensor:
        dtype = v.dtype
        flow = self.flow.to(dtype)
        positive = flow >= 0
        up = torch.where(positive, self._src, self._tgt)      # (..., b)
        down = torch.where(positive, self._tgt, self._src)    # (..., b)
        w = flow.abs()

        batch_shape = v.shape[:-2]
        K = v.shape[-2]
        gather_idx = down if transpose else up
        scatter_idx = up if transpose else down
        gather_idx_b = _bcast_index(gather_idx, batch_shape, K) if gather_idx.dim() == 1 \
            else gather_idx.unsqueeze(-2).expand(*batch_shape, K, gather_idx.shape[-1])
        scatter_idx_b = _bcast_index(scatter_idx, batch_shape, K) if scatter_idx.dim() == 1 \
            else scatter_idx.unsqueeze(-2).expand(*batch_shape, K, scatter_idx.shape[-1])

        v_b = v.expand(*batch_shape, K, self._n)
        x_g = torch.gather(v_b, -1, gather_idx_b)
        weight = self.transmission.to(dtype) * w.unsqueeze(-2)   # (..., K, b)
        in_val = weight.expand(*batch_shape, K, self._n_edges) * x_g
        in_action = torch.zeros(*batch_shape, K, self._n, dtype=dtype, device=v.device)
        in_action.scatter_add_(-1, scatter_idx_b, in_val)

        up_b = _bcast_index(up, batch_shape, K) if up.dim() == 1 \
            else up.unsqueeze(-2).expand(*batch_shape, K, up.shape[-1])
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
            csrc_b = _bcast_index(csrc, batch_shape, K)
            ctgt_b = _bcast_index(ctgt, batch_shape, K)
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
        if self.removal is not None:  # amendment A5
            y_i = y_i - self.removal.to(dtype).transpose(-1, -2) * x_kn
        if self.kinetics is not None:  # amendment A5
            y_i = y_i + torch.einsum("ikl,...li->...ki", self.kinetics.to(dtype), x_kn)
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

    def rmatvec(self, y: torch.Tensor) -> torch.Tensor:
        K, n_i = self.n_species, self.n_interior
        dtype = y.dtype
        y_kn = y.reshape(*y.shape[:-1], K, n_i)
        cap = self.capacity.to(dtype).unsqueeze(-2)
        z_kn = y_kn / cap
        v = self._embed(z_kn, self._interior_idx, dtype)
        raw_t = self._raw_action(v, transpose=True)
        out_i = raw_t.index_select(-1, self._interior_idx)
        if self.removal is not None:  # amendment A5: transpose of the removal term
            out_i = out_i - self.removal.to(dtype).transpose(-1, -2) * y_kn
        if self.kinetics is not None:  # amendment A5: transpose of the kinetics block
            out_i = out_i + torch.einsum(
                "ikl,...li->...ki", self.kinetics.to(dtype).transpose(-1, -2), y_kn
            )
        return out_i.reshape(*out_i.shape[:-2], K * n_i)

    def diagonal(self) -> torch.Tensor:
        K, n_i = self.n_species, self.n_interior
        dtype = self.flow.dtype
        flow = self.flow.to(dtype)
        w = flow.abs()
        batch_shape = torch.broadcast_shapes(w.shape[:-1], self.capacity.shape[:-1])
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

    def assemble(self) -> torch.Tensor:
        K, n_i, n = self.n_species, self.n_interior, self._n
        dtype = self.flow.dtype
        flow = self.flow.to(dtype)
        batch_shape = torch.broadcast_shapes(
            flow.shape[:-1], self.transmission.shape[:-2], self.capacity.shape[:-1]
        )
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
            M_block = M_block + torch.einsum("ikl,ij->kilj", self.kinetics.to(dtype), eye_i)
        return M_block.reshape(*batch_shape, K * n_i, K * n_i)

    def spd_certificate(self):
        return None
