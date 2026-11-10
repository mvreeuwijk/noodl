"""Storage at boundary nodes, stepped explicitly (`scheme="forward_euler"`).

Answers: which boundary potentials does a step hold, and how does a boundary node advance
from its inflow (`hold_potentials`, `advance_storage`)? The construction checks of that
scheme live here too. Held values come in through `history`, as for every other step
start. A mixin of `KirchhoffLayer`.
"""

from __future__ import annotations

import torch

from noodl._broadcast import broadcast_shapes


class _BoundaryStorageMixin:
    """The `_boundary_storage` methods of `KirchhoffLayer`; see the module docstring."""

    def _check_boundary_storage(self, net, name, scheme) -> list[torch.Tensor]:
        """Construction checks of storage under `scheme="forward_euler"`, and each boundary
        storage's columns in `"<name>.phi_boundary"` (the positions of its nodes in
        `bound`)."""
        if scheme != "forward_euler":
            return []
        if not self._boundary_storage:
            raise ValueError(
                f"KirchhoffLayer {name!r}: scheme='forward_euler' steps storage at boundary "
                f"nodes, and this layer has none; give it storage=[LinearStorage at boundary "
                f"nodes], or use 'backward_euler' or 'trapezoidal'"
            )
        dynamic = [el.kind for el in self._laws if el.dynamic]
        if dynamic:
            raise ValueError(
                f"KirchhoffLayer {name!r}: scheme='forward_euler' steps storage at boundary "
                f"nodes only, and kinds {dynamic} have dynamic laws; use 'backward_euler' or "
                f"'trapezoidal'"
            )
        column = {int(n): j for j, n in enumerate(self.bound.tolist())}
        cols, seen = [], set()
        for st in self._boundary_storage:
            nodes = st.nodes.tolist()
            interior = [net.nodes[i] for i in nodes if i not in column]
            if interior:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: storage {st!r} is at {interior}, which are not "
                    f"boundary nodes; scheme='forward_euler' steps storage at boundary nodes "
                    f"only (storage at interior nodes steps with 'backward_euler' or "
                    f"'trapezoidal')"
                )
            if not st.linear:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: storage {type(st).__name__} at boundary nodes "
                    f"is not linear; scheme='forward_euler' steps a node by dt * inflow / C "
                    f"with a constant capacity C (a LinearStorage)"
                )
            again = [net.nodes[i] for i in nodes if i in seen]
            if again:
                raise ValueError(
                    f"KirchhoffLayer {name!r}: boundary storage is given twice at {again}; "
                    f"scheme='forward_euler' takes one storage entry per node"
                )
            seen.update(nodes)
            cols.append(torch.tensor([column[i] for i in nodes], dtype=torch.long))
        return cols

    def _held_nodes(self) -> torch.Tensor:
        """The full-node positions of every boundary storage node, in entry order."""
        return torch.cat([st.nodes for st in self._boundary_storage])

    def hold_potentials(self, phi, phi_start) -> torch.Tensor:
        """`phi` (full-node) with every boundary storage node (`scheme="forward_euler"`) at
        its potential in `phi_start`: what a closure reading the state sees of such a node
        during a step, the value the solve holds it at. Unchanged on a layer without
        boundary storage."""
        if not self._boundary_storage:
            return phi
        idx = self._held_nodes().to(phi.device)
        held = phi_start[..., idx]
        batch = broadcast_shapes(phi.shape[:-1], held.shape[:-1])
        out = phi.expand(batch + phi.shape[-1:]).clone()
        out[..., idx] = held.to(out.dtype)
        return out

    def _hold_boundary(self, phi_boundary, history, dt):
        """`(phi_boundary, history)` as the rest of a call takes them. On a layer that holds
        its storage at boundary nodes, `phi_boundary` gives the `prescribed` nodes only and
        the storage nodes are held at their potentials in `history[0]` (full-node, the step
        start); the result is every boundary node's potential, and the `history` a step of
        the layer's laws reads (`None` unless the layer steps with `dt`). Unchanged on any
        other layer."""
        if not self._boundary_storage:
            return phi_boundary, history
        name = self.name
        nodes = self.net.nodes
        held = [nodes[i] for i in self.bound[self._held_cols].tolist()]
        given = [nodes[i] for i in self.prescribed.tolist()]
        if history is None:
            raise ValueError(
                f"KirchhoffLayer {name!r} holds its storage nodes {held} at their potentials "
                f"at the step start, and reads them from history=(phi, None), phi being "
                f"every node's potential then (the state's {name + '.phi'!r}); pass history"
            )
        if not isinstance(history, (tuple, list)) or len(history) != 2:
            raise ValueError(
                f"KirchhoffLayer {name!r}: history must be the pair (phi, q) at the step "
                f"start; got {type(history).__name__}"
            )
        phi_start = history[0]
        if phi_start.shape[-1] != len(nodes):
            raise ValueError(
                f"KirchhoffLayer {name!r}: history's phi must hold every node's potential "
                f"({len(nodes)}); got last dimension {phi_start.shape[-1]}"
            )
        if phi_boundary.shape[-1] != len(given):
            raise ValueError(
                f"KirchhoffLayer {name!r}: phi_boundary must give the potentials of the "
                f"boundary nodes {given} only ({len(given)}), the storage nodes {held} being "
                f"held at their potentials in history; got last dimension "
                f"{phi_boundary.shape[-1]}"
            )
        start = phi_start[..., self.bound[self._held_cols].to(phi_start.device)]
        batch = broadcast_shapes(phi_boundary.shape[:-1], start.shape[:-1])
        out = phi_boundary.new_empty(batch + (len(self.bound),))
        out[..., self._prescribed_cols.to(out.device)] = phi_boundary.expand(
            batch + phi_boundary.shape[-1:])
        out[..., self._held_cols.to(out.device)] = start.to(out.dtype)
        return out, history if dt is not None and self._takes_step else None

    def advance_storage(self, phi, storage_flow, dt, drivers=None) -> torch.Tensor:
        """The potentials `phi` (full-node) after one forward-Euler step of `dt` of the
        storage at boundary nodes (`scheme="forward_euler"`): each storage node moves by
        `dt * storage_flow / C`, with `storage_flow` the flow into every entry over the step
        (`storage_flows`, `"<name>.storage_flow"`) and `C` the storage's capacity. Every
        other potential is unchanged. Differentiable in `phi`, `storage_flow`, `dt` and the
        capacities."""
        if not self._boundary_storage:
            raise ValueError(
                f"KirchhoffLayer {self.name!r} has no storage at boundary nodes "
                f"(scheme='forward_euler'); advance_storage steps nothing"
            )
        if isinstance(dt, (int, float)):
            if not dt > 0:
                raise ValueError(
                    f"KirchhoffLayer {self.name!r}: dt must be positive, got {dt!r}")
        elif not bool((torch.as_tensor(dt) > 0).all()):
            raise ValueError(f"KirchhoffLayer {self.name!r}: dt must be positive, got {dt!r}")
        if storage_flow.shape[-1] != self._n_entries:
            raise ValueError(
                f"KirchhoffLayer {self.name!r}: storage_flow must hold one flow per storage "
                f"entry ({self._n_entries}); got last dimension {storage_flow.shape[-1]}"
            )
        nodes, moves, offset = [], [], 0
        for st in self._boundary_storage:
            n = st.nodes.numel()
            start = phi[..., st.nodes]
            moves.append(dt * storage_flow[..., offset:offset + n]
                         / st.capacity(start, start, drivers))
            nodes.append(st.nodes)
            offset += n
        move = torch.cat(moves, dim=-1)
        idx = torch.cat(nodes)
        batch = broadcast_shapes(phi.shape[:-1], move.shape[:-1])
        out = phi.expand(batch + phi.shape[-1:]).clone()
        out[..., idx] = out[..., idx] + move
        return out
