"""Tests for Network.endpoints, difference_ep and accumulate: the gather/scatter primitives
every sparse operator is built from, checked for exact agreement with the
existing dense incidence()/difference() they replace.
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from noodl.topology import Network


def _random_connected_multigraph(n, extra, seed, dtype=torch.float64):
    rng = torch.Generator().manual_seed(seed)
    net = Network(dtype=dtype)
    for i in range(n):
        net.add_node(i)
    for i in range(1, n):
        j = int(torch.randint(0, i, (1,), generator=rng))
        net.add_edge(i, j, kind="x")
    for _ in range(extra):
        u = int(torch.randint(0, n, (1,), generator=rng))
        v = int(torch.randint(0, n, (1,), generator=rng))
        if u != v:
            net.add_edge(u, v, kind="x")
    return net


def test_endpoints_shapes_and_dtype_on_triangle(triangle):
    src, tgt = triangle.endpoints()
    assert src.shape == (3,) and tgt.shape == (3,)
    assert src.dtype == torch.long and tgt.dtype == torch.long
    assert src.tolist() == [0, 1, 2]  # a->b, b->c, c->a in node order a, b, c
    assert tgt.tolist() == [1, 2, 0]


def test_endpoints_and_incidence_describe_the_same_graph(triangle):
    src, tgt = triangle.endpoints()
    A = triangle.incidence()
    for j in range(triangle.b):
        assert A[src[j], j] == 1
        assert A[tgt[j], j] == -1


def test_endpoints_is_kind_restricted_on_a_multi_kind_network():
    net = Network(dtype=torch.float64)
    for name in ("a", "b", "c", "d"):
        net.add_node(name)
    net.add_edge("a", "b", kind="x")
    net.add_edge("c", "d", kind="x")
    net.add_edge("b", "c", kind="y")
    src_x, tgt_x = net.endpoints("x")
    assert src_x.tolist() == [0, 2]  # a, c
    assert tgt_x.tolist() == [1, 3]  # b, d
    src_y, tgt_y = net.endpoints("y")
    assert src_y.tolist() == [1] and tgt_y.tolist() == [2]


def test_endpoints_returns_the_same_cached_tensor_object(triangle):
    first = triangle.endpoints()
    second = triangle.endpoints()
    assert first[0] is second[0] and first[1] is second[1]


def test_endpoints_cache_invalidated_by_add_node_and_add_edge(triangle):
    before_src, _ = triangle.endpoints()
    triangle.add_node("d")
    triangle.add_edge("a", "d", kind="airpath")
    after_src, after_tgt = triangle.endpoints()
    assert after_src is not before_src
    assert after_src.shape == (4,)


def test_endpoints_cache_invalidated_by_to(triangle):
    before_src, _ = triangle.endpoints()
    triangle.to(dtype=torch.float64)
    after_src, _ = triangle.endpoints()
    assert after_src is not before_src
    assert after_src.tolist() == before_src.tolist()


def test_endpoints_unknown_kind_raises_keyerror(triangle):
    import pytest

    with pytest.raises(KeyError, match="airpaths"):
        triangle.endpoints("airpaths")


def test_difference_ep_matches_difference_matmul_on_triangle_batched(triangle):
    torch.manual_seed(0)
    phi = torch.randn(5, 7, 3, dtype=torch.float64)
    got = triangle.difference_ep(phi)
    expected = torch.einsum("en,...n->...e", triangle.difference(), phi)
    torch.testing.assert_close(got, expected)


def test_difference_ep_matches_difference_matmul_on_two_zone_batched(two_zone):
    torch.manual_seed(1)
    phi = torch.randn(4, 3, dtype=torch.float64)
    got = two_zone.difference_ep(phi)
    expected = torch.einsum("en,...n->...e", two_zone.difference(), phi)
    torch.testing.assert_close(got, expected)


def test_difference_ep_matches_looped_single_instance_calls(triangle):
    torch.manual_seed(2)
    phi = torch.randn(6, 3, dtype=torch.float64)
    batched = triangle.difference_ep(phi)
    looped = torch.stack([triangle.difference_ep(phi[i]) for i in range(6)])
    torch.testing.assert_close(batched, looped)


def test_difference_ep_is_kind_restricted():
    net = Network(dtype=torch.float64)
    for name in ("a", "b", "c", "d"):
        net.add_node(name)
    net.add_edge("a", "b", kind="x")
    net.add_edge("c", "d", kind="x")
    net.add_edge("b", "c", kind="y")
    phi = torch.randn(4, 4, dtype=torch.float64)
    got = net.difference_ep(phi, kind="x")
    expected = torch.einsum("en,...n->...e", net.difference("x"), phi)
    torch.testing.assert_close(got, expected)


def test_difference_ep_unknown_kind_raises_keyerror(triangle):
    import pytest

    phi = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(KeyError, match="airpaths"):
        triangle.difference_ep(phi, kind="airpaths")


def test_gradcheck_difference_ep(triangle):
    phi = torch.randn(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p: triangle.difference_ep(p), (phi,))


def test_accumulate_matches_incidence_matmul_batched(triangle):
    torch.manual_seed(3)
    w = torch.randn(5, 7, 3, dtype=torch.float64)
    got = triangle.accumulate(w)
    expected = torch.einsum("ne,...e->...n", triangle.incidence(), w)
    torch.testing.assert_close(got, expected)


def test_accumulate_matches_looped_single_instance_calls(triangle):
    torch.manual_seed(4)
    w = torch.randn(6, 3, dtype=torch.float64)
    batched = triangle.accumulate(w)
    looped = torch.stack([triangle.accumulate(w[i]) for i in range(6)])
    torch.testing.assert_close(batched, looped)


def test_accumulate_is_kind_restricted():
    net = Network(dtype=torch.float64)
    for name in ("a", "b", "c", "d"):
        net.add_node(name)
    net.add_edge("a", "b", kind="x")
    net.add_edge("c", "d", kind="x")
    net.add_edge("b", "c", kind="y")
    w = torch.randn(4, 2, dtype=torch.float64)
    got = net.accumulate(w, kind="x")
    expected = torch.einsum("ne,...e->...n", net.incidence("x"), w)
    torch.testing.assert_close(got, expected)


def test_accumulate_unknown_kind_raises_keyerror(triangle):
    import pytest

    w = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(KeyError, match="airpaths"):
        triangle.accumulate(w, kind="airpaths")


def test_gradcheck_accumulate(triangle):
    w = torch.randn(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda ww: triangle.accumulate(ww), (w,))


@settings(max_examples=30, deadline=None)
@given(
    n=st.integers(min_value=2, max_value=8),
    extra=st.integers(min_value=0, max_value=10),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_accumulate_of_difference_ep_matches_incidence_times_difference_on_random_graphs(
    n, extra, seed
):
    net = _random_connected_multigraph(n, extra, seed)
    torch.manual_seed(seed)
    phi = torch.randn(4, n, dtype=torch.float64)
    got = net.accumulate(net.difference_ep(phi))
    expected = torch.einsum(
        "ne,...e->...n", net.incidence(), torch.einsum("en,...n->...e", net.difference(), phi)
    )
    torch.testing.assert_close(got, expected)


def test_difference_ep_rejects_wrong_node_count_with_value_error(triangle):
    import pytest

    phi_wrong = torch.randn(5, 2, dtype=torch.float64)  # 2 nodes instead of 3
    with pytest.raises(ValueError, match="2.*3"):  # match wrong vs expected count
        triangle.difference_ep(phi_wrong)


def test_accumulate_rejects_wrong_edge_count_with_value_error(triangle):
    import pytest

    w_wrong = torch.randn(5, 2, dtype=torch.float64)  # 2 edges instead of 3
    with pytest.raises(ValueError, match="2.*3"):  # match wrong vs expected count
        triangle.accumulate(w_wrong)
