"""`noodl._fixed_point_loop`: the pieces `Model._iterate` and `CoupledModel._iterate` share."""

import torch

from noodl._fixed_point_loop import detached, unsolved_mask

F64 = torch.float64


def test_detached_detaches_every_nested_tensor_and_keeps_the_rest():
    p = torch.tensor(2.0, dtype=F64, requires_grad=True)
    diag = {"a": {"x": p * 3.0, "n": 4, "s": "text"}, "b": p + 1.0, "c": [p]}
    out = detached(diag)
    assert not out["a"]["x"].requires_grad and out["a"]["x"].item() == 6.0
    assert out["a"]["n"] == 4 and out["a"]["s"] == "text"
    assert not out["b"].requires_grad
    assert out["c"] is diag["c"]                 # only dicts are walked, as before the move
    assert diag["b"].requires_grad               # the input is not modified


def test_unsolved_mask_is_none_when_every_named_solve_converged():
    diag = {"a": {"converged": torch.tensor([True, True])}, "b": {}, "t": {"substeps": 1}}
    assert unsolved_mask(diag, ["a", "b", "missing"]) is None


def test_unsolved_mask_ors_the_failures_of_the_named_layers_only():
    diag = {
        "a": {"converged": torch.tensor([True, False, True])},
        "b": {"converged": torch.tensor([False, True, True])},
        "c": {"converged": torch.tensor([False, False, False])},     # not named
    }
    assert unsolved_mask(diag, ["a", "b"]).tolist() == [True, True, False]
    assert unsolved_mask({"a": {"converged": False}}, ["a"]).item() is True
