"""The held initial point: storage that holds a potential at the first row.

Answers: what are the held rows (`_Held`) of a solve at `mode="initial"`, in which layout
is the held system solved, how is it checked for grounding, and which flows does the
storage deliver there (`_initial_storage_flows`)? A mixin of `KirchhoffLayer`.
"""

from __future__ import annotations

import torch

from noodl._broadcast import broadcast_shapes
from noodl.layers._flow_unknowns import (
    FlowUnknowns,
    _Held,
    flow_unknowns,
)
from noodl.operators.sparse import SparseOperator


class _HeldMixin:
    """The `_held` methods of `KirchhoffLayer`; see the module docstring."""

    def _held_values(self, drivers, call, *, check: bool = False):
        """`(rows, value)`: the interior rows held at the initial point ((n_I,) bool) and
        the potential each is held at ((..., n_I), zero elsewhere); `None` when no storage
        holds an initial potential. `initial` is evaluated through `call`, so on the
        differentiable path with the threaded parameters. `check` refuses entries at one
        node with different initial values, by name."""
        n_I = len(self.interior)
        base = len(self._laws) + len(self._withdrawals)
        held = []
        for j, _st in enumerate(self._storage):
            v = call(base + j, (drivers,), {"part": "initial"})
            if v is not None:
                pos = self._storage_positions[j]
                v = torch.as_tensor(v, device=pos.device)
                held.append((j, pos, v.expand(v.shape[:-1] + pos.shape) if v.dim()
                             else v.expand(pos.shape)))
        if not held:
            return None
        batch = broadcast_shapes(*(v.shape[:-1] for _, _, v in held))
        dtype = held[0][2].dtype
        value = torch.zeros(batch + (n_I,), dtype=dtype, device=held[0][2].device)
        rows = torch.zeros(n_I, dtype=torch.bool, device=value.device)
        for _, pos, v in held:
            value = value.index_copy(-1, pos, v.to(dtype).expand(batch + pos.shape))
            rows[pos] = True
        if check:
            for _, pos, v in held:
                differ = (v.to(dtype) != value[..., pos]).reshape(-1, pos.numel()).any(0)
                if bool(differ.any()):
                    row = pos[differ.nonzero()[0, 0]]
                    names = [f"{type(self._storage[k]).__name__} (storage {k})"
                             for k, p, _ in held if bool((p == row).any())]
                    node = self.net.nodes[int(self.interior[row])]
                    raise ValueError(
                        f"KirchhoffLayer {self.name!r}: the storage entries at node "
                        f"{node!r} hold different initial potentials ({', '.join(names)}); "
                        f"mode='initial' holds a node at one value")
        return rows, value

    def _held(self, found, u: FlowUnknowns) -> _Held:
        """The held rows of `mode="initial"` (`found`, from `_held_values`) and their masks
        on the values of the Jacobian of layout `u`."""
        rows = found[0].to(u.row.device)
        full = torch.zeros(u.n, dtype=torch.bool, device=rows.device)
        full[: u.n_I] = rows
        diag = torch.arange(len(u.row) - u.n_I, len(u.row), device=rows.device)
        return _Held(rows=rows, clear=full[u.row], ones=diag[rows])

    def _held_layout(self) -> FlowUnknowns:
        """The layout of the initial point with held storage on a layer without flow
        unknowns: no unknown flows, the balance rows and their diagonal slots (`flow_unknowns`
        of this layer). Built on the first such solve and kept; no other solve reads it."""
        if self._held_unknowns is None:
            self._held_unknowns = flow_unknowns(self)
        return self._held_unknowns

    def _hold(self, op: SparseOperator, store) -> SparseOperator:
        """`op` with each row `mode="initial"` holds replaced by the row of
        `phi_i - initial_i`: its values zeroed, 1 on its diagonal slot."""
        if not isinstance(store, _Held):
            return op
        values = op.values.masked_fill(store.clear, 0.0)
        values[..., store.ones] = 1.0
        return SparseOperator(op.row, op.col, values, op.shape[-1])

    def _held_grounding_check(self, x0, phi_boundary, drivers, step) -> None:
        """The grounding certificate of the held initial point of a layer without flow
        unknowns, at its starting point `x0`: the law slopes there, with every held row
        grounded (a positive slope on its diagonal), so a node that is neither held nor
        joined to a boundary or a held node is refused by name."""
        phi = self.assemble(x0, phi_boundary)
        call = self._module_call()
        dq = self._dflows(phi, drivers, step.ctx, call)
        rows = step.storage.rows
        ns = self._node_slopes(phi, drivers, None, call)
        held = rows.to(dtype=dq.dtype)
        ns = held if ns is None else torch.where(rows, held, ns)
        self._grounding_check(dq, where="solve", node_slopes=ns)

    def _initial_storage_flows(self, phi, q, sources, drivers, zeros) -> torch.Tensor:
        """`storage_flows` at the initial point (`mode="initial"`): each held node's flow
        imbalance, split among its entries by capacity."""
        held = self._held_values(drivers, self._owner_call)
        if held is None:
            return zeros
        if q is None:
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: storage_flows(mode='initial') needs q, every "
                f"edge's flow at the initial point (as solve returns it)"
            )
        rows, _ = held
        phi_i = phi[..., self.interior]
        imbalance = self._source_interior(sources, phi_i) - self._accumulate_interior(q)
        w = self._node_withdrawal(phi, drivers, None, self._module_call())
        if w is not None:
            imbalance = imbalance - w
        imbalance = torch.where(rows, imbalance, torch.zeros_like(imbalance))
        caps = [st.capacity(phi[..., st.nodes], phi[..., st.nodes], drivers)
                for st in self._storage]
        total = self._scatter_storage(lambda j, st: caps[j])
        count = self._scatter_storage(lambda j, st: torch.ones_like(caps[j]))
        parts = []
        for cap, pos in zip(caps, self._storage_positions, strict=True):
            tot = total[..., pos]
            share = torch.where(tot != 0, cap / torch.where(tot != 0, tot, 1.0),
                                1.0 / count[..., pos])
            parts.append(share * imbalance[..., pos])
        batch = broadcast_shapes(*(p.shape[:-1] for p in parts))
        return torch.cat([p.expand(batch + p.shape[-1:]) for p in parts], dim=-1)
