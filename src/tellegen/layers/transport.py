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

from tellegen.topology import Network, Node


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
        if self.capacity.shape[-1] != self.n_i:
            raise ValueError(
                f"TransportLayer '{name}': capacity has {self.capacity.shape[-1]} entries, "
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

        # Set by later steps in this task (removal, conduction); left inert here.
        self.removal: torch.Tensor | None = None
        self.L = torch.zeros(net.n, net.n, dtype=net.dtype)

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
    ) -> torch.Tensor:
        out_dtype = x.dtype
        dtype = torch.float64
        x_s, reduced = self._to_stacked(x, self.n_i, "x")
        x_s = x_s.to(dtype)
        M, N = self.operator(q.to(dtype))
        b0, _ = self._forcing(sources, x_boundary, N, dtype)
        if self.scheme == "exact":
            result = _van_loan_step(M, x_s, b0, dt)
        else:
            raise ValueError(f"TransportLayer '{self.name}': unknown scheme {self.scheme!r}")
        return self._from_stacked(result.to(out_dtype), self.n_i, reduced)

    def steady(
        self, q: torch.Tensor, sources: torch.Tensor, x_boundary: torch.Tensor
    ) -> torch.Tensor:
        dtype = torch.float64
        M, N = self.operator(q.to(dtype))
        b0, reduced = self._forcing(sources, x_boundary, N, dtype)
        try:
            x_s = torch.linalg.solve(M, -b0.unsqueeze(-1)).squeeze(-1)
        except torch.linalg.LinAlgError as err:
            raise RuntimeError(
                f"TransportLayer '{self.name}': steady-state system is singular (no "
                f"outflow anywhere on some interior node): {err}"
            ) from err
        return self._from_stacked(x_s, self.n_i, reduced)


def _van_loan_step(
    M: torch.Tensor, x: torch.Tensor, b0: torch.Tensor, dt: float
) -> torch.Tensor:
    """Exact linear step via the augmented matrix exponential (Van Loan, 1978)."""
    m = M.shape[-1]
    batch = M.shape[:-2]
    Z = torch.zeros(*batch, 2 * m, 2 * m, dtype=M.dtype)
    Z[..., :m, :m] = M * dt
    Z[..., :m, m:] = torch.eye(m, dtype=M.dtype) * dt
    E = torch.linalg.matrix_exp(Z)
    Ed = E[..., :m, :m]
    Phi = E[..., :m, m:]
    return (Ed @ x.unsqueeze(-1)).squeeze(-1) + (Phi @ b0.unsqueeze(-1)).squeeze(-1)
