"""Typed network topology with the operators of algebraic graph theory.

A :class:`Network` is a directed multigraph whose edges carry a ``kind``
(``"airpath"``, ``"conduction"``, ``"hydronic"``, ``"storage"``, ...). It exposes,
as PyTorch tensors:

* the incidence matrix ``incidence()`` (n x b): +1 at the source node of each
  edge, -1 at the target, so that ``incidence() @ q`` is the net outflow at
  every node and ``incidence() @ q == 0`` is conservation (Kirchhoff's current
  law);
* the branch potential difference ``difference() = incidence().T`` (b x n), so
  that ``difference() @ phi`` is the source-minus-target difference of a nodal
  potential on every edge. This is the sign convention the rest of the
  framework actually uses: ``PotentialFlowLayer.dp()`` calls this operator, a
  positive ``difference()`` at an edge's source drives a positive flow ``q``
  from source to target under every Element law in this package, and the
  nodal Jacobian ``A_I diag(g') A_I^T`` is positive definite under exactly
  this convention;
* the gradient ``gradient() = -incidence().T`` (b x n) -- the *opposite* sign
  convention (target minus source). It is kept only because existing tests
  assert its shape and sign; nothing in this package's solve path reads it,
  and a caller computing ``net.gradient() @ phi`` and treating the result as
  "the potential difference driving flow" gets the wrong sign. Use
  ``difference()`` instead;
* a basis of the cycle space ``cycle_basis()`` (l x b) with
  ``incidence() @ cycle_basis().T == 0``; any flow ``q = cycle_basis().T @ m``
  is divergence free for any amplitudes ``m``;
* the upwind selection operator ``upwind(q)`` (b x n) that picks, for each
  edge, the value of a nodal scalar at its upstream node;
* Tellegen's residual ``power_residual(p, q) = sum(p * q)``, which vanishes
  identically for ``p`` in the cut space and ``q`` in the cycle space.

The conventions follow John Craske's 2019 ``Tellegen`` package (after Brayton
and Moser, 1964), rebuilt on ``networkx.MultiDiGraph`` and ``torch``.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence

import networkx as nx
import torch

Node = Hashable
EdgeKey = tuple[Hashable, Hashable, int]


class Network:
    """Directed multigraph with typed edges and tensor-valued topology operators."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        self.graph = nx.MultiDiGraph()
        self.dtype = dtype
        self.device = torch.device("cpu")
        # Branch order is insertion order (networkx iterates edges by adjacency).
        self._edges: list[EdgeKey] = []
        self._cache: dict[tuple, torch.Tensor] = {}

    # ------------------------------------------------------------------ building
    def add_node(self, name: Node, **attrs) -> None:
        self.graph.add_node(name, **attrs)
        self._cache.clear()

    def add_edge(self, source: Node, target: Node, *, kind: str, **attrs) -> EdgeKey:
        """Add a directed edge of the given kind and return its (source, target, key)."""
        for node in (source, target):
            if node not in self.graph:
                raise KeyError(f"unknown node {node!r}; add nodes before edges")
        key = self.graph.add_edge(source, target, kind=kind, **attrs)
        edge = (source, target, key)
        self._edges.append(edge)
        self._cache.clear()
        return edge

    def with_ambient(self, name: Node = "ambient", *, kind: str = "storage") -> Network:
        """Return a copy with one extra node joined to every existing node (the cospan)."""
        other = Network(dtype=self.dtype)
        other.device = self.device
        other.graph = self.graph.copy()
        other._edges = list(self._edges)
        other.add_node(name)
        for node in self.nodes:
            other.add_edge(node, name, kind=kind)
        return other

    # ------------------------------------------------------------------ counting
    @property
    def nodes(self) -> list[Node]:
        return list(self.graph.nodes)

    @property
    def edges(self) -> list[EdgeKey]:
        """Edges as (source, target, key), in insertion order; this is the branch order."""
        return list(self._edges)

    @property
    def n(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def b(self) -> int:
        return self.graph.number_of_edges()

    @property
    def n_components(self) -> int:
        return nx.number_connected_components(self.graph.to_undirected(as_view=True))

    @property
    def n_cycles(self) -> int:
        """Dimension of the cycle space: b - n + number of components."""
        return self.b - self.n + self.n_components

    # ------------------------------------------------------------------ indexing
    def _node_index(self) -> dict[Node, int]:
        return {node: i for i, node in enumerate(self.graph.nodes)}

    def node_index(self, node: Node) -> int:
        """Position of `node` in node order."""
        index = self._node_index()
        if node not in index:
            raise KeyError(f"unknown node {node!r}")
        return index[node]

    def interior_index(self, boundary: Sequence[Node]) -> torch.Tensor:
        """Positions, in node order, of all nodes not listed in `boundary`."""
        index = self._node_index()
        missing = [n for n in boundary if n not in index]
        if missing:
            raise KeyError(f"unknown boundary nodes {missing}")
        boundary_set = set(boundary)
        idx = [index[n] for n in self.graph.nodes if n not in boundary_set]
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def boundary_index(self, boundary: Sequence[Node]) -> torch.Tensor:
        """Positions of `boundary` nodes, in the order given."""
        index = self._node_index()
        missing = [n for n in boundary if n not in index]
        if missing:
            raise KeyError(f"unknown boundary nodes {missing}")
        return torch.tensor([index[n] for n in boundary], dtype=torch.long, device=self.device)

    def node_attr(self, name: str, default: float | None = None) -> torch.Tensor:
        """Node attribute values in node order, as a (n,) tensor.

        Raises `KeyError` naming the offending nodes if any node lacks the
        attribute and no `default` is given.
        """
        values = []
        missing = []
        for node in self.graph.nodes:
            data = self.graph.nodes[node]
            if name in data:
                values.append(float(data[name]))
            elif default is not None:
                values.append(float(default))
            else:
                missing.append(node)
        if missing:
            raise KeyError(f"node attribute {name!r} missing for nodes {missing}")
        return torch.tensor(values, dtype=self.dtype, device=self.device)

    def component_labels(self, kind: str | None = None) -> torch.Tensor:
        """Connected-component label (0..components-1) of every node, in node order.

        With `kind` given, connectivity is restricted to edges of that kind: a node
        touched by no edge of `kind` gets its own singleton component. Raises
        `KeyError` (via `edge_index`) naming the unknown kind if `kind` matches no
        edge.
        """
        key = ("component_labels", kind)
        if key in self._cache:
            return self._cache[key]
        cols = self.edge_index(kind)
        edges = self.edges
        undirected = nx.Graph()
        undirected.add_nodes_from(self.graph.nodes)
        for col in cols.tolist():
            u, v, _ = edges[col]
            undirected.add_edge(u, v)
        label_of: dict[Node, int] = {}
        for label, component in enumerate(nx.connected_components(undirected)):
            for node in component:
                label_of[node] = label
        result = torch.tensor(
            [label_of[n] for n in self.graph.nodes], dtype=torch.long, device=self.device
        )
        self._cache[key] = result
        return result

    def n_components_of(self, kind: str | None = None) -> int:
        """Number of connected components among edges of one kind (or the whole graph).

        Does not affect `n_components`, which always describes the whole graph.
        """
        labels = self.component_labels(kind)
        if labels.numel() == 0:
            return 0
        return int(labels.max().item()) + 1

    def edge_attr(
        self, name: str, kind: str | None = None, default: float | None = None
    ) -> torch.Tensor:
        """Edge attribute values of one kind, in branch order, as a (b_kind,) tensor."""
        cols = self.edge_index(kind)
        edges = self.edges
        values = []
        missing = []
        for col in cols.tolist():
            u, v, k = edges[col]
            data = self.graph.edges[u, v, k]
            if name in data:
                values.append(float(data[name]))
            elif default is not None:
                values.append(float(default))
            else:
                missing.append((u, v, k))
        if missing:
            raise KeyError(f"edge attribute {name!r} missing for edges {missing}")
        return torch.tensor(values, dtype=self.dtype, device=self.device)

    def edge_index(self, kind: str | None = None) -> torch.Tensor:
        """Column indices (into the full edge list) of the edges of one kind, or all.

        Raises `KeyError` naming the unknown kind (and the kinds that do exist) if
        `kind` is given but no edge carries it. `kind=None` always means "all
        edges" and never raises, even on an empty network.
        """
        key = ("edge_index", kind)
        if key in self._cache:
            return self._cache[key]
        kinds = self.edge_kinds()
        if kind is not None and kind not in kinds:
            raise KeyError(f"unknown edge kind {kind!r}; network has kinds {sorted(set(kinds))}")
        idx = [i for i, k in enumerate(kinds) if kind is None or k == kind]
        result = torch.tensor(idx, dtype=torch.long, device=self.device)
        self._cache[key] = result
        return result

    def edge_kinds(self) -> list[str]:
        return [self.graph.edges[u, v, k].get("kind") for (u, v, k) in self._edges]

    # ------------------------------------------------------------------ operators
    def incidence(self, kind: str | None = None) -> torch.Tensor:
        """Incidence matrix (n x b_kind): +1 at source, -1 at target of each edge."""
        key = ("incidence", kind)
        if key in self._cache:
            return self._cache[key]
        index = self._node_index()
        cols = self.edge_index(kind)
        d = torch.zeros(self.n, len(cols), dtype=self.dtype, device=self.device)
        edges = self.edges
        for j, col in enumerate(cols.tolist()):
            source, target, _ = edges[col]
            d[index[source], j] += 1
            d[index[target], j] -= 1
        self._cache[key] = d
        return d

    def gradient(self, kind: str | None = None) -> torch.Tensor:
        """Gradient (b_kind x n): target minus source of a nodal potential.

        This is the *opposite* sign convention from ``difference()``, which is what
        ``PotentialFlowLayer.dp()`` and the rest of this package's solve path actually use.
        ``gradient()`` is not called anywhere in this package outside its own definition; it
        is kept only because existing tests assert its shape and sign. Computing
        ``net.gradient() @ phi`` and treating the result as "the potential difference that
        drives flow" gives the wrong sign -- use ``difference()`` for that.
        """
        key = ("gradient", kind)
        if key in self._cache:
            return self._cache[key]
        result = -self.incidence(kind).T
        self._cache[key] = result
        return result

    def difference(self, kind: str | None = None) -> torch.Tensor:
        """Branch potential difference (b_kind x n): source minus target of a nodal potential.

        ``difference() @ phi`` gives, for every edge, ``phi[source] - phi[target]``: this is
        the sign convention the whole framework uses (the opposite of ``gradient()``, which
        gives ``target - source`` and is not used by the solve path). A positive value here
        at an edge's source drives a positive flow ``q`` from source to target under every
        Element law in this package, and the nodal Jacobian ``A_I diag(g') A_I^T`` built from
        this convention is positive definite. ``PotentialFlowLayer.dp()`` calls this operator
        (restricted to its own layer's edge columns) rather than repeating the einsum inline.
        """
        key = ("difference", kind)
        if key in self._cache:
            return self._cache[key]
        result = self.incidence(kind).T
        self._cache[key] = result
        return result

    def endpoints(self, kind: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(source_node_index, target_node_index), each (b_kind,), shared across the batch.

        A DISTINCT accessor from `edge_index(kind)`, which returns the columns of `kind`
        into the full edge list -- `endpoints` returns, for those same columns, the node
        POSITION of every edge's source and target: exactly the row positions
        `incidence(kind)` puts a +1 and a -1 at. `difference_ep` and `accumulate` are
        gather/scatter forms of `difference(kind)` and `incidence(kind) @ w` built from this
        pair, so `endpoints` and `incidence` always describe the same graph by construction
        (both derive from `edge_index(kind)` and `self.edges` in node order). Cached under
        `("endpoints", kind)`; cleared by `add_node`, `add_edge` and `to()` like every other
        cached tensor. Raises `KeyError` (via `edge_index`) naming the unknown kind, exactly
        as `incidence(kind)` and `difference(kind)` do.
        """
        key = ("endpoints", kind)
        if key in self._cache:
            return self._cache[key]
        cols = self.edge_index(kind)
        edges = self.edges
        index = self._node_index()
        src = torch.tensor(
            [index[edges[c][0]] for c in cols.tolist()], dtype=torch.long, device=self.device
        )
        tgt = torch.tensor(
            [index[edges[c][1]] for c in cols.tolist()], dtype=torch.long, device=self.device
        )
        result = (src, tgt)
        self._cache[key] = result
        return result

    def difference_ep(self, phi: torch.Tensor, kind: str | None = None) -> torch.Tensor:
        """Gather form of `difference(kind) @ phi`: `phi[..., src] - phi[..., tgt]`.

        `phi` has shape `(..., n)`; the result has shape `(..., b_kind)`, broadcasting over
        arbitrary leading batch dimensions exactly like `difference(kind) @ phi` does on a
        batched `phi`. Never forms the `(b_kind, n)` `difference()` matrix: this is the
        gather primitive the milestone's sparse operators are built from instead.
        """
        if phi.dim() == 0:
            raise ValueError(
                f"Network.difference_ep: phi is a scalar (shape {tuple(phi.shape)}); "
                f"expected shape (..., {self.n}) with {self.n} nodes (kind={kind!r})"
            )
        if phi.shape[-1] != self.n:
            raise ValueError(
                f"Network.difference_ep: phi has trailing size {phi.shape[-1]} but the network "
                f"has {self.n} nodes (kind={kind!r}); got shape {tuple(phi.shape)}"
            )
        src, tgt = self.endpoints(kind)
        return phi[..., src] - phi[..., tgt]

    def accumulate(self, w: torch.Tensor, kind: str | None = None) -> torch.Tensor:
        """Scatter-add form of `incidence(kind) @ w`: `+w` at `src`, `-w` at `tgt`.

        `w` has shape `(..., b_kind)`; the result has shape `(..., n)`. Built with
        `index_add` (out of place) on a FRESH zero tensor allocated inside this call, never
        `index_add_` (in place) on an input or on anything aliased with one: an in-place
        scatter into a tensor autograd needs unmodified for its own backward pass would
        corrupt the gradient of any earlier operation sharing that storage, whereas an
        out-of-place `index_add` on a tensor this call itself just created has no such alias
        to protect, and is exactly as differentiable w.r.t. `w` as `incidence(kind) @ w` is.
        Never forms the `(n, b_kind)` `incidence()` matrix.
        """
        src, tgt = self.endpoints(kind)
        b_kind = len(src)
        if w.dim() == 0:
            raise ValueError(
                f"Network.accumulate: w is a scalar (shape {tuple(w.shape)}); "
                f"expected shape (..., {b_kind}) with {b_kind} edges (kind={kind!r})"
            )
        if w.shape[-1] != b_kind:
            raise ValueError(
                f"Network.accumulate: w has trailing size {w.shape[-1]} but the network "
                f"has {b_kind} edges (kind={kind!r}); got shape {tuple(w.shape)}"
            )
        batch_shape = w.shape[:-1]
        out = torch.zeros(batch_shape + (self.n,), dtype=w.dtype, device=w.device)
        out = out.index_add(-1, src, w)
        out = out.index_add(-1, tgt, -w)
        return out

    def spanning_forest(self, kind: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Spanning forest of the edges of one kind: (tree_cols, chord_cols).

        Both are `LongTensor`s indexing columns of `incidence(kind)`. One tree per
        connected component, built by union-find over insertion order; `cycle_basis`
        uses this same forest so a chord's fundamental cycle closes on it.
        """
        key = ("spanning_forest", kind)
        if key in self._cache:
            return self._cache[key]
        cols = self.edge_index(kind).tolist()
        edges = [self.edges[c] for c in cols]
        uf = _UnionFind(self.graph.nodes)
        tree: list[int] = []
        chord: list[int] = []
        for j, (u, v, _) in enumerate(edges):
            if u != v and uf.union(u, v):
                tree.append(j)
            else:
                chord.append(j)
        result = (
            torch.tensor(tree, dtype=torch.long, device=self.device),
            torch.tensor(chord, dtype=torch.long, device=self.device),
        )
        self._cache[key] = result
        return result

    def cycle_basis(self, kind: str | None = None) -> torch.Tensor:
        """Integer basis of the cycle space (l x b_kind) from the spanning forest.

        Each chord e = (u, v) gives one basis vector: unit flow along e from u to
        v, returning from v to u along the unique tree path, with +1 on tree
        edges traversed in their own direction and -1 otherwise.
        """
        key = ("cycle_basis", kind)
        if key in self._cache:
            return self._cache[key]
        cols = self.edge_index(kind).tolist()
        edges = [self.edges[c] for c in cols]
        b = len(edges)
        tree_cols, chord_cols = self.spanning_forest(kind)

        forest = nx.Graph()
        forest.add_nodes_from(self.graph.nodes)
        tree_edges: dict[frozenset, tuple[int, Node, Node]] = {}
        for j in tree_cols.tolist():
            u, v, _ = edges[j]
            forest.add_edge(u, v)
            tree_edges[frozenset((u, v))] = (j, u, v)

        chord_list = chord_cols.tolist()
        rows = torch.zeros(len(chord_list), b, dtype=self.dtype, device=self.device)
        for r, j in enumerate(chord_list):
            u, v, _ = edges[j]
            rows[r, j] = 1
            if u == v:
                continue  # self loop closes on its own
            path = nx.shortest_path(forest, v, u)
            for a, c in zip(path[:-1], path[1:], strict=True):
                jt, s, _ = tree_edges[frozenset((a, c))]
                rows[r, jt] += 1 if s == a else -1
        self._cache[key] = rows
        return rows

    def source_selector(self, kind: str | None = None) -> torch.Tensor:
        """One-hot selector (b_kind x n): row e is 1 at the source node of edge e."""
        key = ("source_selector", kind)
        if key in self._cache:
            return self._cache[key]
        cols = self.edge_index(kind)
        index = self._node_index()
        edges = self.edges
        s = torch.zeros(len(cols), self.n, dtype=self.dtype, device=self.device)
        for j, col in enumerate(cols.tolist()):
            source, _target, _ = edges[col]
            s[j, index[source]] = 1
        self._cache[key] = s
        return s

    def target_selector(self, kind: str | None = None) -> torch.Tensor:
        """One-hot selector (b_kind x n): row e is 1 at the target node of edge e."""
        key = ("target_selector", kind)
        if key in self._cache:
            return self._cache[key]
        cols = self.edge_index(kind)
        index = self._node_index()
        edges = self.edges
        t = torch.zeros(len(cols), self.n, dtype=self.dtype, device=self.device)
        for j, col in enumerate(cols.tolist()):
            _source, target, _ = edges[col]
            t[j, index[target]] = 1
        self._cache[key] = t
        return t

    def upwind(self, q: torch.Tensor, kind: str | None = None) -> torch.Tensor:
        """Sign-aware selector: where(q >= 0, source_selector, target_selector).

        `q` has shape (..., b_kind); the result has shape (..., b_kind, n). With
        `kind=None` and `q` of shape (b,) this reproduces the original edge-by-edge
        selection over the whole graph unchanged.
        """
        S = self.source_selector(kind)
        T = self.target_selector(kind)
        positive = (q >= 0).unsqueeze(-1)
        return torch.where(positive, S, T)

    def downwind(self, q: torch.Tensor, kind: str | None = None) -> torch.Tensor:
        """Sign-aware selector: where(q >= 0, target_selector, source_selector)."""
        S = self.source_selector(kind)
        T = self.target_selector(kind)
        positive = (q >= 0).unsqueeze(-1)
        return torch.where(positive, T, S)

    def to(self, device=None, dtype=None) -> Network:
        """Move cached and future tensors to `device`/`dtype`, in place; clears the cache."""
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self._cache.clear()
        return self

    def power_residual(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Tellegen's residual sum(p * q); zero for consistent potentials and flows."""
        return (p * q).sum()

    # ------------------------------------------------------------------ misc
    def __repr__(self) -> str:
        return f"Network(n={self.n}, b={self.b}, kinds={sorted(set(self.edge_kinds()))})"


class _UnionFind:
    def __init__(self, items: Iterable[Node]) -> None:
        self.parent = {item: item for item in items}

    def find(self, x: Node) -> Node:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: Node, b: Node) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True
