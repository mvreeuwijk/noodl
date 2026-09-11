"""Tests for PotentialFlowLayer: assembly, non-differentiable solve, power residual."""

import pytest
import torch
from scipy.optimize import brentq

from tellegen.drives import ConstantDrive
from tellegen.elements import Conductance, FixedFlow, PowerLaw
from tellegen.layers.potential import PotentialFlowLayer
from tellegen.solvers.newton import newton
from tellegen.topology import Network


def _closed_form_series_flow(C, n, drive):
    def f(q):
        return sum((q / c) ** (1.0 / n) for c in C) - drive

    return brentq(f, 1e-9, 1e3)


def test_duplicate_element_kind_raises_value_error():
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")
    with pytest.raises(ValueError, match="airpath"):
        PotentialFlowLayer(
            net,
            "dup",
            [
                Conductance(torch.tensor([1.0], dtype=torch.float64), kind="airpath"),
                Conductance(torch.tensor([1.0], dtype=torch.float64), kind="airpath"),
            ],
        )


def test_element_kind_with_no_edges_raises_value_error():
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")
    with pytest.raises(ValueError, match="conduction"):
        PotentialFlowLayer(
            net,
            "missing",
            [Conductance(torch.tensor([1.0], dtype=torch.float64), kind="conduction")],
        )


def test_unknown_boundary_node_raises_key_error():
    net = Network(dtype=torch.float64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")
    element = PowerLaw(torch.tensor([1.0], dtype=torch.float64), 0.5, kind="airpath")
    with pytest.raises(KeyError, match="zzz"):
        PotentialFlowLayer(net, "bad", [element], boundary=["zzz"])
