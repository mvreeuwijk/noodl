"""Tests for the Element base class: abstract flow, autograd dflow default, linear_init."""

import pytest
import torch

from tellegen.elements.base import Element


class _Linear(Element):
    """q = k * dp, the minimal concrete Element used to test the base-class defaults."""

    def __init__(self, k, *, learnable=False):
        super().__init__(kind="test")
        self.k = self._param(k, learnable)

    def flow(self, dp, drivers=None):
        return self.k * dp


class _Cubic(Element):
    """q = a * dp**3, used because its derivative is not constant."""

    def __init__(self, a, *, learnable=False):
        super().__init__(kind="test")
        self.a = self._param(a, learnable)

    def flow(self, dp, drivers=None):
        return self.a * dp**3


def test_element_kind_is_stored():
    assert _Linear(2.0).kind == "test"


def test_abstract_flow_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        Element(kind="airpath").flow(torch.tensor([1.0]))


def test_dflow_default_matches_known_derivative_of_linear_element():
    el = _Linear(3.5)
    dp = torch.linspace(-4.0, 4.0, 9)
    torch.testing.assert_close(el.dflow(dp), torch.full_like(dp, 3.5))


def test_dflow_default_matches_known_derivative_of_cubic_element():
    el = _Cubic(2.0)
    dp = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    torch.testing.assert_close(el.dflow(dp), 3 * 2.0 * dp**2)


def test_dflow_default_does_not_perturb_the_callers_tensor():
    el = _Cubic(1.0)
    dp = torch.tensor([1.0, 2.0], requires_grad=True)
    el.dflow(dp)
    assert dp.grad is None  # dflow works on a private clone, not the caller's leaf


def test_dflow_default_builds_a_graph_only_when_grad_is_enabled():
    el = _Cubic(1.0)
    dp = torch.tensor([1.0, 2.0])
    assert el.dflow(dp).requires_grad
    with torch.no_grad():
        assert not el.dflow(dp).requires_grad


def test_linear_init_default_matches_flow_and_dflow_at_zero():
    c, k = _Linear(7.0).linear_init()
    torch.testing.assert_close(c, torch.tensor(0.0))
    torch.testing.assert_close(k, torch.tensor(7.0))


def test_param_marks_learnable_tensor_as_a_parameter():
    el = _Linear(2.0, learnable=True)
    assert isinstance(el.k, torch.nn.Parameter)
    assert el.k.requires_grad
    assert list(el.parameters()) == [el.k]


def test_param_leaves_non_learnable_tensor_without_grad():
    assert not _Linear(2.0, learnable=False).k.requires_grad


def test_forward_delegates_to_the_subclass_flow_override_not_the_base_class():
    """Guards against a naive `forward = flow` class-body assignment.

    That form binds the base class's own `flow` function object at class-definition time,
    so subclasses overriding `flow` would silently be unreachable through `__call__`/
    `forward` (and therefore through `torch.func.functional_call`, which Task 8 depends on).
    """
    el = _Cubic(2.0)
    dp = torch.tensor([-2.0, 0.5, 3.0])
    torch.testing.assert_close(el(dp), el.flow(dp))
    torch.testing.assert_close(el.forward(dp), el.flow(dp))
    with pytest.raises(NotImplementedError):
        Element(kind="airpath").forward(torch.tensor([1.0]))
