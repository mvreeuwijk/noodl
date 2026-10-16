"""`noodl.layout`: the batch and species-layout rules, and equality with the private rules
they replace (copied here verbatim before those are deleted)."""

from __future__ import annotations

import itertools

import pytest
import torch

from noodl import layout
from noodl._broadcast import broadcast_shapes

# ---- the old private rules, copied -------------------------------------------------------


def _old_is_stacked(x, n_last):
    return x.dim() >= 2 and x.shape[-1] == 1 and x.shape[-2] == n_last


def _old_reduced(x, n_last):
    if _old_is_stacked(x, n_last):
        return x.squeeze(-1)
    if x.shape[-1] != n_last:
        raise ValueError(
            f"couple: expected a tensor with {n_last} entries on its node axis, got shape "
            f"{tuple(x.shape)}"
        )
    return x


def _old_write_at(target, n_last, position, value, *, add=False):
    reduced = _old_reduced(target, n_last)
    batch = broadcast_shapes(reduced.shape[:-1], value.shape)
    reduced = reduced.expand(*batch, n_last).clone()
    reduced[..., position] = reduced[..., position] + value if add else value
    stacked = _old_is_stacked(target, n_last)
    return reduced.reshape(*batch, n_last, 1) if stacked else reduced.reshape(*batch, n_last)


def _old_to_stacked(x, n_nodes, K):
    """TransportLayer._to_stacked's rule: (stacked?, reduced?) or ValueError."""
    if x.dim() >= 2 and x.shape[-2] == n_nodes and x.shape[-1] == K:
        return False
    if K == 1 and x.shape[-1] == n_nodes:
        return True
    raise ValueError("bad")


def _old_species_history(x, n_i, K):
    if x.dim() >= 3 and tuple(x.shape[-2:]) == (n_i, K):
        return False
    if K == 1 and x.dim() >= 2 and x.shape[-1] == n_i:
        return True
    raise ValueError("bad")


SHAPES = [s for r in range(1, 5) for s in itertools.product([1, 2, 3], repeat=r)]


# ---- property tests against the old rules -----------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n", [1, 2, 3])
def test_is_stacked_and_reduced_single_equal_the_couple_rules(shape, n):
    x = torch.arange(float(torch.Size(shape).numel())).reshape(shape)
    assert layout.is_stacked(x, n) == _old_is_stacked(x, n)
    try:
        old = _old_reduced(x, n)
    except ValueError as exc:
        with pytest.raises(ValueError) as err:
            layout.reduced_single(x, n, where="couple")
        assert str(err.value) == str(exc)
    else:
        assert torch.equal(layout.reduced_single(x, n, where="couple"), old)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n,K", [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (1, 2), (3, 3)])
def test_as_stacked_accepts_exactly_what_to_stacked_accepts(shape, n, K):
    x = torch.zeros(shape)
    try:
        reduced = _old_to_stacked(x, n, K)
    except ValueError:
        with pytest.raises(ValueError, match="must have trailing shape"):
            layout.as_stacked(x, n, K, name="x", where="w")
        return
    out, got_reduced = layout.as_stacked(x, n, K, name="x", where="w")
    assert got_reduced == reduced
    assert out.shape[-2:] == (n, K)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n,K", [(1, 1), (2, 1), (3, 1), (2, 2), (3, 3)])
def test_as_stacked_min_dim_3_equals_species_history(shape, n, K):
    x = torch.zeros(shape)
    try:
        reduced = _old_species_history(x, n, K)
    except ValueError:
        with pytest.raises(ValueError):
            layout.as_stacked(x, n, K, name="x", where="w", min_dim=3)
        return
    _, got = layout.as_stacked(x, n, K, name="x", where="w", min_dim=3)
    assert got == reduced


@pytest.mark.parametrize("target_shape", [(3,), (1, 3), (3, 1), (2, 3), (2, 3, 1), (1,), (1, 1)])
@pytest.mark.parametrize("value_shape", [(), (2,), (4, 2)])
@pytest.mark.parametrize("add", [False, True])
def test_write_at_equals_the_couple_rule(target_shape, value_shape, add):
    n = 1 if target_shape in ((1,), (1, 1)) else 3
    target = torch.arange(float(torch.Size(target_shape).numel())).reshape(target_shape) + 1
    value = torch.arange(float(torch.Size(value_shape).numel())).reshape(value_shape) + 10
    try:
        old = _old_write_at(target, n, 0, value, add=add)
    except (RuntimeError, ValueError) as exc:
        with pytest.raises(type(exc)):
            layout.write_at(target, n, 0, value, add=add, where="couple")
        return
    new = layout.write_at(target, n, 0, value, add=add, where="couple")
    assert new.shape == old.shape and torch.equal(new, old)


# ---- batch_shape / broadcast_batch / per_instance ---------------------------------------


def test_batch_shape():
    assert layout.batch_shape(torch.zeros(4, 3, 2), 2) == (4,)
    assert layout.batch_shape(torch.zeros(3), 1) == ()
    assert layout.batch_shape(torch.zeros(5), 0) == (5,)
    with pytest.raises(ValueError, match="at least"):
        layout.batch_shape(torch.zeros(3), 2)


def test_broadcast_batch_unbatched_and_batched_inputs():
    a, b = torch.zeros(3), torch.zeros(4, 2, 5)
    batch, (ea, eb) = layout.broadcast_batch((a, 1), (b, 1))
    assert batch == (4, 2)
    assert ea.shape == (4, 2, 3) and eb.shape == (4, 2, 5)
    assert ea.data_ptr() == a.data_ptr()  # a view, not a copy


def test_broadcast_batch_batch_equal_to_node_count_is_not_a_node_axis():
    # B == n: the shape (n, n) must be read as (batch n, nodes n), never aligned on the node axis
    n = 3
    per_node = torch.arange(float(n))            # (n,): one instance
    batched = torch.zeros(n, n)                   # (B=n, n)
    batch, (p, q) = layout.broadcast_batch((per_node, 1), (batched, 1))
    assert batch == (n,)
    assert p.shape == (n, n) and torch.equal(p[1], per_node)
    assert q.shape == (n, n)


def test_broadcast_batch_scalar_per_instance_and_node_field():
    t = torch.tensor([1.0, 2.0])                  # (B=2,), 0 trailing
    x = torch.zeros(2, 4)                         # (B=2, n=4)
    batch, (te, xe) = layout.broadcast_batch((t, 0), (x, 1))
    assert batch == (2,) and te.shape == (2,) and xe.shape == (2, 4)


def test_broadcast_batch_mismatch_names_the_shapes():
    with pytest.raises(RuntimeError, match=r"\(2, 3\).*\(4, 3\)"):
        layout.broadcast_batch((torch.zeros(2, 3), 1), (torch.zeros(4, 3), 1))


def test_per_instance_float_and_tensor():
    like = torch.zeros(5, 3, dtype=torch.float64)
    v = layout.per_instance(2.0, like, 1)
    assert v.shape == (1,) and v.dtype == torch.float64
    t = layout.per_instance(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32), like, 1)
    assert t.shape == (5, 1) and t.dtype == torch.float64
    assert (like + t).shape == (5, 3)


def test_per_instance_cannot_broadcast_against_a_physical_axis():
    # B == n == 3: a (3,) per-instance value must scale rows (instances), not columns (nodes)
    like = torch.zeros(3, 3)
    t = layout.per_instance(torch.tensor([1.0, 2.0, 3.0]), like, 1)
    out = like + t
    assert torch.equal(out[:, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(out[0], torch.ones(3))


def test_per_instance_two_trailing_dims_and_mismatch():
    like = torch.zeros(4, 3, 2)
    assert layout.per_instance(torch.ones(4), like, 2).shape == (4, 1, 1)
    with pytest.raises(RuntimeError):
        layout.per_instance(torch.ones(5), like, 2)


# ---- stacked / reduced -------------------------------------------------------------------


def test_is_stacked_edge_cases():
    assert layout.is_stacked(torch.zeros(3, 1), 3)
    assert not layout.is_stacked(torch.zeros(3), 3)
    assert not layout.is_stacked(torch.zeros(1), 1)                   # a vector is never stacked
    assert layout.is_stacked(torch.zeros(1, 1), 1)                    # the n == 1 ambiguity
    assert layout.is_stacked(torch.zeros(3, 2), 3, 2)
    assert not layout.is_stacked(torch.zeros(3, 2), 2, 3)
    assert not layout.is_stacked(torch.zeros(3, 1), 3, min_dim=3)
    assert layout.is_stacked(torch.zeros(7, 3, 1), 3, min_dim=3)


def test_as_stacked_stacked_wins_over_reduced_when_both_fit():
    x = torch.zeros(2, 1)                # n = 1: stacked (2 instances) is not (n, 1) unless n=2...
    out, reduced = layout.as_stacked(x, 1, 1, name="x", where="w")
    assert reduced is True and out.shape == (2, 1, 1)   # (2, 1) is reduced with batch 2, n = 1
    x = torch.zeros(1, 1)
    out, reduced = layout.as_stacked(x, 1, 1, name="x", where="w")
    assert reduced is False and out is x                # (1, 1) is stacked: tested first


def test_as_stacked_column_and_reduced_roundtrip():
    x = torch.arange(6.0).reshape(2, 3)
    stacked, reduced = layout.as_stacked(x, 3, 1, name="x", where="w")
    assert reduced and stacked.shape == (2, 3, 1)
    assert torch.equal(layout.as_layout(stacked, reduced), x)
    col = torch.arange(3.0).reshape(3, 1)
    stacked, reduced = layout.as_stacked(col, 3, 1, name="x", where="w")
    assert not reduced and layout.as_layout(stacked, reduced) is col


def test_as_stacked_multi_species_refuses_reduced():
    with pytest.raises(ValueError, match=r"w: 'x' must have trailing shape \(3, 2\), got \(3,\)"):
        layout.as_stacked(torch.zeros(3), 3, 2, name="x", where="w")


def test_as_stacked_message_mentions_the_reduced_form_for_one_species():
    with pytest.raises(ValueError, match=r"or, for one species, \(\.\.\., 3\)"):
        layout.as_stacked(torch.zeros(4), 3, 1, name="x", where="w")


def test_as_stacked_min_dim_3_needs_a_leading_axis():
    with pytest.raises(ValueError):
        layout.as_stacked(torch.zeros(3, 1), 3, 1, name="x", where="w", min_dim=3)
    out, reduced = layout.as_stacked(torch.zeros(5, 3), 3, 1, name="x", where="w", min_dim=3)
    assert reduced and out.shape == (5, 3, 1)


def test_reduced_single_squeezes_a_column_and_refuses_a_wrong_axis():
    assert layout.reduced_single(torch.zeros(3, 1), 3, where="couple").shape == (3,)
    assert layout.reduced_single(torch.zeros(2, 3), 3, where="couple").shape == (2, 3)
    with pytest.raises(ValueError, match="^couple: expected a tensor with 3 entries"):
        layout.reduced_single(torch.zeros(4), 3, where="couple")


def test_write_at_keeps_the_layout_and_never_writes_into_the_target():
    target = torch.zeros(3, 1)
    out = layout.write_at(target, 3, 1, torch.tensor(5.0))
    assert out.shape == (3, 1) and out[1, 0] == 5.0 and target.sum() == 0
    out = layout.write_at(torch.ones(3), 3, 0, torch.tensor(2.0), add=True)
    assert torch.equal(out, torch.tensor([3.0, 1.0, 1.0]))


def test_write_at_batched_value_broadcasts_an_unbatched_target():
    out = layout.write_at(torch.zeros(1, 1), 1, 0, torch.tensor([1.0, 2.0, 3.0]))
    assert out.shape == (3, 1, 1)


def test_write_at_where_prefixes_the_error():
    with pytest.raises(ValueError, match="^couple: expected"):
        layout.write_at(torch.zeros(4), 3, 0, torch.tensor(1.0), where="couple")


# ---- species_columns ---------------------------------------------------------------------


def test_species_columns_any_order_and_subset():
    names = ("bod", "sulfide", "o2")
    assert layout.species_columns(names, ["o2", "bod"], where="w") == [2, 0]
    assert layout.species_columns(names, [], where="w") == []


def test_species_columns_unknown_and_repeated_names_are_named():
    with pytest.raises(ValueError, match=r"w: unknown species \['x'\]"):
        layout.species_columns(("a", "b"), ["a", "x"], where="w")
    with pytest.raises(ValueError, match=r"species \['a'\] are named more than once"):
        layout.species_columns(("a", "b"), ["a", "a"], where="w")
    with pytest.raises(ValueError, match="repeat a name"):
        layout.species_columns(("a", "a"), ["a"], where="w")


def test_species_columns_exact_order():
    assert layout.species_columns(("a", "b"), ["a", "b"], where="w", order="exact") == [0, 1]
    with pytest.raises(ValueError, match="exactly"):
        layout.species_columns(("a", "b"), ["b", "a"], where="w", order="exact")
    with pytest.raises(ValueError, match="exactly"):
        layout.species_columns(("a", "b"), ["a"], where="w", order="exact")
    with pytest.raises(ValueError, match="order must be"):
        layout.species_columns(("a",), ["a"], where="w", order="sorted")


def test_species_columns_without_declared_names_is_the_identity_for_all_species():
    assert layout.species_columns(None, ["p", "q"], where="w", n_species=2) == [0, 1]
    with pytest.raises(ValueError, match="no species_names"):
        layout.species_columns(None, ["p"], where="w", n_species=2)
    with pytest.raises(ValueError, match="no species_names"):
        layout.species_columns(None, ["p"], where="w")
