"""`noodl.solvers.anderson`: the Anderson mixing step moved out of `noodl.model`."""

import pytest
import torch

from noodl.solvers.anderson import _ANDERSON_COND_MAX, _ANDERSON_NOISE_MIN, _anderson_mix

F64 = torch.float64


def _linear_history(a, b, xs):
    gs = [a @ x + b for x in xs]
    return gs, [g - x for g, x in zip(gs, xs, strict=True)]


def test_the_conditioning_bound():
    assert _ANDERSON_COND_MAX == 1.0e8


def test_the_noise_bound():
    assert _ANDERSON_NOISE_MIN == 10.0


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


def _two_column_history():
    """Newest column [-1, 0] (norm 1); the older one [0, 1e-7], orthogonal to it and well
    inside the conditioning bound. The newest residual's 3e-8 along it gives that column a
    coefficient of 0.3: kept, it moves the mixed state by 0.3 of its output difference."""
    xs = [torch.tensor([0.0, 0.0], dtype=F64), torch.tensor([0.3, -0.2], dtype=F64),
          torch.tensor([0.1, 0.4], dtype=F64)]
    rs = [torch.tensor([1.5, -7e-8], dtype=F64), torch.tensor([1.5, 3e-8], dtype=F64),
          torch.tensor([0.5, 3e-8], dtype=F64)]
    gs = [x + 2 * r for x, r in zip(xs, rs, strict=True)]
    return xs, gs, rs


#: per-entry noise at which the older column's norm (1e-7) is exactly `_ANDERSON_NOISE_MIN`
#: times its noise, the norm of its two entries' summed noise: 2 * noise * sqrt(2)
_AT_BOUND = 1e-7 / (_ANDERSON_NOISE_MIN * 2 * 2 ** 0.5)


@pytest.mark.parametrize(("factor", "dropped"), [(2.0, True), (0.5, False)])
def test_a_column_within_its_rounding_noise_is_dropped(factor, dropped):
    """The older column against `factor` times the noise that puts it at the bound. Dropped,
    the mixing is exactly that of the history without its oldest entry; kept, exactly the
    mixing without a noise estimate (which keeps both columns)."""
    xs, gs, rs = _two_column_history()
    ns = [torch.full((2,), factor * _AT_BOUND, dtype=F64)] * 3
    mixed, kept = _anderson_mix(xs, gs, rs, ns)
    assert bool(kept)
    both, _ = _anderson_mix(xs, gs, rs)
    newest, _ = _anderson_mix(xs[1:], gs[1:], rs[1:])
    assert not torch.equal(both, newest)
    assert torch.equal(mixed, newest if dropped else both)


def test_a_newest_column_within_its_noise_keeps_no_column():
    xs, gs, rs = _two_column_history()
    mixed, kept = _anderson_mix(xs, gs, rs, [torch.full((2,), 0.1, dtype=F64)] * 3)
    assert not bool(kept)
    assert torch.equal(mixed, gs[-1])


def test_the_noise_rule_is_per_instance():
    xs, gs, rs = _two_column_history()
    ns = [torch.tensor([[2.0], [0.5]], dtype=F64).expand(2, 2) * _AT_BOUND] * 3
    mixed, kept = _anderson_mix(xs, gs, rs, ns)
    assert kept.tolist() == [True, True]
    assert torch.equal(mixed[0], _anderson_mix(xs[1:], gs[1:], rs[1:])[0])
    assert torch.equal(mixed[1], _anderson_mix(xs, gs, rs)[0])
