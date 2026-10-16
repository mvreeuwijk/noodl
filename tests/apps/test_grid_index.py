"""`noodl.modelica.timeline.grid_index` (shared by the building and electrical Modelica
runners): the sorted-grid lookup equals the old all-pairs algorithm."""

from __future__ import annotations

import pytest
import torch

from noodl.modelica.timeline import grid_index as _shared_grid_index

F64 = torch.float64


@pytest.fixture(params=["simulate", "simulate_circuit"])
def grid_index(request):
    """The shared helper, as each runner calls it."""
    return lambda grid, times: _shared_grid_index(grid, times, caller=request.param)


def _reference(grid, times):
    """The quadratic algorithm `_grid_index` replaced, kept here as the specification."""
    tol = 1e-9 * max(1.0, float(grid.abs().max()))
    diff = (times.unsqueeze(1) - grid.unsqueeze(0)).abs()
    idx = diff.argmin(dim=1)
    off = diff.gather(1, idx.unsqueeze(1)).squeeze(1) > tol
    if bool(off.any()):
        return ("off", int(torch.nonzero(off)[0]))
    out = idx.tolist()
    if any(b <= a for a, b in zip(out, out[1:], strict=False)):
        return ("order", None)
    return ("ok", out)


def _new(grid_index, grid, times):
    try:
        return ("ok", grid_index(grid, times))
    except ValueError as exc:
        msg = str(exc)
        if "not a grid time" in msg:
            return ("off", int(msg.split("times[")[1].split("]")[0]))
        assert "must be increasing" in msg
        return ("order", None)


def _t(*v):
    return torch.tensor(v, dtype=F64)


def test_exact_grid_times_and_a_subset(grid_index):
    grid = torch.arange(0.0, 101.0, dtype=F64) * 0.5
    assert grid_index(grid, grid) == list(range(101))
    assert grid_index(grid, _t(0.0, 5.0, 50.0)) == [0, 10, 100]
    assert grid_index(grid, _t()) == []


def test_tolerance_edges_match_the_reference(grid_index):
    grid = _t(0.0, 1.0, 2.0, 1e6)
    tol_small, tol_big = 1e-9 * 1e6, 1e-9 * 1e6                  # the grid's max sets tol
    cases = [
        _t(1.0 + 0.5 * tol_small), _t(1.0 + tol_small), _t(1.0 + 1.001 * tol_small),
        _t(1.0 - 0.999 * tol_big), _t(1.0 - 1.001 * tol_big), _t(-0.5 * tol_small),
        _t(1e6 + 0.5 * tol_small), _t(1e6 + 2 * tol_small), _t(-1.0), _t(2e6),
    ]
    for times in cases:
        assert _new(grid_index, grid, times) == _reference(grid, times), times.tolist()


def test_a_time_off_the_grid_names_the_first_offender(grid_index):
    grid = _t(0.0, 1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match=r"times\[1\] = 1\.5 is not a grid time"):
        grid_index(grid, _t(0.0, 1.5, 2.5))
    with pytest.raises(ValueError, match=r"times\[0\] = 7\.0 is not a grid time"):
        grid_index(grid, _t(7.0))


def test_times_must_increase(grid_index):
    grid = _t(0.0, 1.0, 2.0)
    for times in (_t(1.0, 0.0), _t(1.0, 1.0), _t(0.0, 2.0, 1.0)):
        with pytest.raises(ValueError, match="times must be increasing"):
            grid_index(grid, times)


def test_unsorted_and_duplicate_grids_follow_the_reference(grid_index):
    for grid in (_t(3.0, 0.0, 2.0, 1.0), _t(0.0, 1.0, 1.0, 2.0), _t(2.0, 2.0, 2.0)):
        for times in (_t(0.0, 2.0), _t(2.0), _t(1.0), _t(0.0, 1.0, 3.0), _t(3.0, 0.0)):
            assert _new(grid_index, grid, times) == _reference(grid, times), (grid.tolist(),
                                                                  times.tolist())


def test_random_small_grids_agree_with_the_reference(grid_index):
    gen = torch.Generator().manual_seed(7)
    for _ in range(300):
        m = int(torch.randint(1, 9, (1,), generator=gen))
        n = int(torch.randint(0, 7, (1,), generator=gen))
        grid = torch.randint(-4, 12, (m,), generator=gen).to(F64) * 0.5
        # times near grid points, half a step away (a tie between two entries) or exact
        times = (torch.randint(-5, 14, (n,), generator=gen).to(F64) * 0.25
                 + torch.randint(0, 2, (n,), generator=gen).to(F64) * 1e-10)
        assert _new(grid_index, grid, times) == _reference(grid, times), (
            grid.tolist(), times.tolist())
        sorted_times = torch.sort(times).values
        assert _new(grid_index, grid, sorted_times) == _reference(grid, sorted_times)
