"""`noodl.solvers.anderson`: the Anderson mixing step moved out of `noodl.model`."""

import torch

from noodl.solvers.anderson import _ANDERSON_COND_MAX, _anderson_mix

F64 = torch.float64


def _linear_history(a, b, xs):
    gs = [a @ x + b for x in xs]
    return gs, [g - x for g, x in zip(gs, xs, strict=True)]


def test_model_re_exports_the_moved_names():
    import noodl.model
    import noodl.solvers.anderson

    assert noodl.model._anderson_mix is noodl.solvers.anderson._anderson_mix
    assert noodl.model._AndersonHistory is noodl.solvers.anderson._AndersonHistory
    assert noodl.model._ANDERSON_COND_MAX == _ANDERSON_COND_MAX == 1.0e8


def test_recorded_two_entry_history_of_a_linear_map():
    """One history column: the value recorded from the function before the move."""
    a = torch.tensor([[0.5, 0.1, 0.0], [0.2, 0.3, 0.1], [0.0, 0.1, 0.4]], dtype=F64)
    b = torch.tensor([1.0, -2.0, 0.5], dtype=F64)
    xs = [torch.zeros(3, dtype=F64), torch.tensor([1.0, 0.5, -1.0], dtype=F64)]
    gs, rs = _linear_history(a, b, xs)
    mixed, kept = _anderson_mix(xs, gs, rs)
    assert bool(kept)
    assert mixed.tolist() == [0.6999999999999998, -2.1363636363636362, 0.6909090909090909]


def test_a_single_entry_history_keeps_no_column_and_returns_the_output():
    x = [torch.tensor([0.0, 1.0], dtype=F64)]
    g = [torch.tensor([2.0, 3.0], dtype=F64)]
    mixed, kept = _anderson_mix(x, g, [g[0] - x[0]])
    assert not bool(kept)
    assert torch.equal(mixed, g[0])


def test_a_collinear_history_keeps_no_column():
    x = [torch.tensor([0.0, 0.0], dtype=F64), torch.tensor([1.0, 1.0], dtype=F64)]
    g = [torch.tensor([1.0, 1.0], dtype=F64), torch.tensor([1.0, 1.0], dtype=F64)]
    r = [torch.tensor([1.0], dtype=F64), torch.tensor([1.0], dtype=F64)]
    _, kept = _anderson_mix(x, g, r)
    assert not bool(kept)


def test_batch_instances_are_mixed_independently():
    """A batch of two histories gives, per instance, exactly the unbatched result."""
    torch.manual_seed(3)
    a = torch.tensor([[0.4, 0.1], [0.2, 0.3]], dtype=F64)
    hist = []
    for b in (torch.tensor([1.0, -1.0], dtype=F64), torch.tensor([0.3, 2.0], dtype=F64)):
        xs = [torch.zeros(2, dtype=F64), torch.rand(2, dtype=F64), torch.rand(2, dtype=F64)]
        gs, rs = _linear_history(a, b, xs)
        hist.append((xs, gs, rs))
    stacked = [[torch.stack([hist[0][j][i], hist[1][j][i]]) for i in range(3)]
               for j in range(3)]
    mixed, kept = _anderson_mix(*stacked)
    assert kept.shape == (2,)
    for i in range(2):
        single, k_i = _anderson_mix(*hist[i])
        assert torch.equal(mixed[i], single)
        assert bool(kept[i]) == bool(k_i)
