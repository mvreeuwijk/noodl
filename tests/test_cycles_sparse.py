"""Sparse-path tests for cycles.py (Task 13): particular_flow and branch_flows migrated off
a dense per-component tree solve and a materialised cycle-basis matmul, onto a level-
synchronous elimination over the spanning forest. Each test compares the migrated function
to an independent DENSE reference (the pre-Task-13 algorithm, reproduced here verbatim) on
a multi-kind, multi-component network, plus a gradcheck through both.
"""

import time
import unittest.mock

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tellegen.cycles import branch_flows, particular_flow
from tellegen.topology import Network


def _dense_particular_flow_reference(net, sources, kind=None):
    """The exact pre-Task-13 algorithm: a dense per-component tree solve. Kept here, not in
    production code, purely as this test file's independent oracle."""
    A = net.incidence(kind)
    tree_cols, chord_cols = net.spanning_forest(kind)
    labels = net.component_labels(kind)
    n_components = net.n_components_of(kind)
    col_component = (net.source_selector(kind) @ labels.to(A.dtype)).round().long()
    batch_shape = sources.shape[:-1]
    q = torch.zeros(*batch_shape, A.shape[1], dtype=sources.dtype)
    for c in range(n_components):
        node_idx = torch.nonzero(labels == c, as_tuple=False).flatten()
        if node_idx.numel() <= 1:
            continue
        rest = node_idx[1:]
        tree_cols_c = tree_cols[col_component[tree_cols] == c]
        A_c = A[rest][:, tree_cols_c]
        rhs_c = sources[..., rest]
        q_tree_c = torch.linalg.solve(A_c, rhs_c.unsqueeze(-1)).squeeze(-1)
        q[..., tree_cols_c] = q_tree_c
    return q


def _multi_kind_multi_component_network() -> Network:
    """Two disjoint airpath trees (a 4-node star and a 3-node chain) plus one hydronic
    bridge edge between them: airpath-kind operations must see 2 components, matching the
    component-labelling behaviour test_cycles.py already exercises elsewhere.

    Amendment A7.2: the star also carries one extra airpath chord (h1 -> h2) so the
    airpath sub-network has a nonzero cycle-space dimension (l = 1); b_air becomes 6 and
    n_air_components stays 2.
    """
    net = Network(dtype=torch.float64)
    for name in ("h0", "h1", "h2", "h3", "c0", "c1", "c2"):
        net.add_node(name)
    net.add_edge("h0", "h1", kind="airpath")
    net.add_edge("h0", "h2", kind="airpath")
    net.add_edge("h0", "h3", kind="airpath")
    net.add_edge("h1", "h2", kind="airpath")  # chord: closes a cycle in the star
    net.add_edge("c0", "c1", kind="airpath")
    net.add_edge("c1", "c2", kind="airpath")
    net.add_edge("h3", "c0", kind="hydronic")
    return net


def test_particular_flow_matches_dense_reference_on_multi_kind_multi_component_network():
    net = _multi_kind_multi_component_network()
    torch.manual_seed(0)
    raw = torch.rand(3, dtype=torch.float64)  # 3 leaves (amendment A7)
    star_sources = torch.cat([-(raw.sum()).unsqueeze(0), raw])  # zero-sum on the star
    chain_sources = torch.tensor([0.3, -0.1, -0.2], dtype=torch.float64)  # zero-sum
    sources = torch.cat([star_sources, chain_sources])

    q = particular_flow(net, sources, kind="airpath")
    q_ref = _dense_particular_flow_reference(net, sources, kind="airpath")
    torch.testing.assert_close(q, q_ref, rtol=1e-9, atol=1e-12)

    A_air = net.incidence(kind="airpath")
    torch.testing.assert_close(A_air @ q, sources, atol=1e-12, rtol=0.0)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000), batch=st.integers(min_value=1, max_value=8))
def test_particular_flow_matches_dense_reference_batched_random_trees(seed, batch):
    rng = torch.Generator().manual_seed(seed)
    n = 7
    net = Network(dtype=torch.float64)
    for i in range(n):
        net.add_node(i)
    for i in range(1, n):
        j = int(torch.randint(0, i, (1,), generator=rng))
        net.add_edge(i, j, kind="x")
    raw = torch.rand(batch, n, generator=rng, dtype=torch.float64)
    sources = raw - raw.mean(dim=-1, keepdim=True)

    q = particular_flow(net, sources)
    q_ref = _dense_particular_flow_reference(net, sources)
    torch.testing.assert_close(q, q_ref, rtol=1e-9, atol=1e-12)


def test_gradcheck_particular_flow():
    net = Network(dtype=torch.float64)
    for name in ("a", "b", "c"):
        net.add_node(name)
    net.add_edge("a", "b", kind="airpath")
    net.add_edge("b", "c", kind="airpath")
    raw = torch.tensor([1.0, -0.3], dtype=torch.float64, requires_grad=True)

    def f(raw_):
        sources = torch.cat([-(raw_.sum()).unsqueeze(0), raw_])
        return particular_flow(net, sources)

    assert torch.autograd.gradcheck(f, (raw,), eps=1e-6, atol=1e-5)


def test_tree_elimination_helpers_are_importable():
    from tellegen.cycles import _tree_elimination_levels, _tree_solve  # noqa: F401


def _dense_branch_flows_reference(net, amplitudes, kind=None):
    """The exact pre-Task-13 algorithm: materialise cycle_basis(kind) and matmul."""
    J = net.cycle_basis(kind).to(amplitudes.dtype)
    return amplitudes @ J


def test_branch_flows_matches_dense_reference_on_multi_kind_network():
    net = _multi_kind_multi_component_network()
    # airpath sub-network here has two components (star + chain): l = b - n + n_components.
    b_air = net.edge_index("airpath").numel()
    n_air_components = net.n_components_of("airpath")
    l_dim = b_air - 7 + n_air_components  # 7 airpath-touching nodes across both components
    torch.manual_seed(1)
    amplitudes = torch.tensor([0.7], dtype=torch.float64)
    assert amplitudes.shape[-1] == l_dim  # non-vacuous: l_dim must be 1 after amendment A7.2

    q = branch_flows(net, amplitudes, kind="airpath")
    q_ref = _dense_branch_flows_reference(net, amplitudes, kind="airpath")
    torch.testing.assert_close(q, q_ref, rtol=1e-9, atol=1e-12)
    assert not torch.all(q == 0)  # amendment A7.2: a nonzero chord amplitude is really injected

    A_air = net.incidence(kind="airpath")
    torch.testing.assert_close(
        A_air @ q, torch.zeros(7, dtype=torch.float64), atol=1e-12, rtol=0.0
    )


def test_gradcheck_branch_flows(triangle):
    m = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

    def f(m_):
        return branch_flows(triangle, m_)

    assert torch.autograd.gradcheck(f, (m,), eps=1e-6, atol=1e-5)


def test_branch_flows_never_materialises_the_dense_cycle_basis_matrix():
    # Structural guard: a correct migration must not call Network.cycle_basis at all from
    # inside branch_flows (that is precisely the dense O(l * b) materialisation being
    # removed). Patch it to raise if touched, and confirm branch_flows still works.
    #
    # NOTE: the brief's literal version of this test used kind="hydronic", but the
    # network's only hydronic edge (h3 -> c0) is itself a spanning-tree edge (it is the
    # first edge to connect two previously separate hydronic components), so
    # spanning_forest("hydronic") has zero chords; amplitudes=[0.5] would then mismatch
    # chord_cols.numel() == 0 and *even the dense* branch_flows raises ValueError on that
    # input (confirmed by direct execution against the pre-Task-13 code). Using
    # kind="airpath" instead exercises the one real chord amendment A7.2 introduced,
    # which is what this test needs to inject a nonzero amplitude at all.
    net = _multi_kind_multi_component_network()
    amplitudes = torch.tensor([0.5], dtype=torch.float64)
    with unittest.mock.patch.object(
        Network, "cycle_basis", side_effect=AssertionError("cycle_basis must not be called")
    ):
        q = branch_flows(net, amplitudes, kind="airpath")
    assert torch.isfinite(q).all()


@pytest.mark.slow
def test_particular_flow_1000_node_chain_is_faster_than_dense():
    """Reports, does not gate on, wall-clock time for a 1000-node chain (worst case for
    tree depth: a chain's single component has depth == n, so this is also the WORST case
    for this migration's O(depth) elimination -- a star or balanced tree would show a far
    larger improvement).

    Memory is not measured here: tracemalloc does not see PyTorch's C-level allocator (an
    80 MB torch.zeros shows as 0.000 MB under tracemalloc, measured on this machine), so a
    tracemalloc-based memory comparison would be a vacuous pass on Python-object overhead
    only (amendment A4). This test therefore reports TIME only.
    """
    n = 1000
    net = Network(dtype=torch.float64)
    for i in range(n):
        net.add_node(i)
    for i in range(1, n):
        net.add_edge(i - 1, i, kind="x")
    torch.manual_seed(0)
    raw = torch.rand(n, dtype=torch.float64)
    sources = raw - raw.mean()

    particular_flow(net, sources)  # warm the topology cache before timing either path

    t0 = time.perf_counter()
    q_sparse = particular_flow(net, sources)
    t_sparse = time.perf_counter() - t0

    t0 = time.perf_counter()
    q_dense = _dense_particular_flow_reference(net, sources)
    t_dense = time.perf_counter() - t0

    torch.testing.assert_close(q_sparse, q_dense, rtol=1e-9, atol=1e-9)
    print(
        f"\nparticular_flow 1000-node chain: sparse {t_sparse * 1e3:.2f} ms vs "
        f"dense {t_dense * 1e3:.2f} ms"
    )
