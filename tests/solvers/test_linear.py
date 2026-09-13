"""Tests for the batched linear solve wrapper."""

import pytest
import torch

from tellegen.solvers.linear import solve


def test_solve_matches_torch_linalg_solve_on_a_regular_system():
    A = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    b = torch.tensor([4.0, 9.0])
    x = solve(A, b)
    torch.testing.assert_close(x, torch.tensor([2.0, 3.0]))


def test_solve_raises_runtime_error_naming_the_singular_batch_index():
    A = torch.stack(
        [
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
            torch.tensor([[1.0, 2.0], [2.0, 4.0]]),  # singular: rows proportional
        ]
    )
    b = torch.stack([torch.tensor([4.0, 9.0]), torch.tensor([1.0, 2.0])])
    with pytest.raises(RuntimeError, match=r"singular system.*\[1\]"):
        solve(A, b)
