"""Unit tests of `noodl.layers._functional.FunctionalLaws` against the layer's own methods."""

from __future__ import annotations

import torch

from noodl.layers._functional import FunctionalLaws
from tests.layers.test_kirchhoff_solve_policy import _classic


def test_residual_and_operator_match_the_layer():
    layer, call = _classic(False)
    (phi_b, drivers, sources), _, _ = call(True)
    fn = FunctionalLaws(layer, drivers, sources, phi_b)
    # leaf order: law parameters, drivers (sorted), sources, phi_boundary
    assert fn.all_params[-1] is phi_b
    assert fn.all_params[-2].shape == (layer._n_nodes,)
    assert fn.driver_keys == sorted(drivers)
    x = torch.tensor([3.0, 1.0], dtype=torch.float64)
    ref, _, _ = layer._residual_parts(x, phi_b, drivers, None)
    assert torch.allclose(fn.residual_fn(x, *fn.all_params), ref, rtol=0, atol=1e-14)
    phi = layer.assemble(x, phi_b)
    dq = layer.dflows(phi, drivers)
    got = fn.operator_fn(x, *fn.all_params).matvec(torch.ones(2, dtype=torch.float64))
    want = layer._operator_at(dq, None).matvec(torch.ones(2, dtype=torch.float64))
    assert torch.allclose(got, want, rtol=0, atol=1e-14)


def test_residual_is_differentiable_in_the_driver():
    layer, call = _classic(False)
    (phi_b, drivers, sources), _, leaf = call(True)
    fn = FunctionalLaws(layer, drivers, sources, phi_b)
    x = torch.tensor([3.0, 1.0], dtype=torch.float64)
    fn.residual_fn(x, *fn.all_params).sum().backward()
    assert leaf.grad is not None
