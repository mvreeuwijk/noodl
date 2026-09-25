"""Tests for the Element base class: abstract flow, autograd dflow default, linear_init."""

import pytest
import torch

from noodl.elements.base import Element


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


def test_param_learnable_true_yields_a_registered_parameter_even_for_a_grad_tracking_input():
    """Regression: the pass-through exception for an already-requires_grad input tensor must
    be gated on learnable=False. With learnable=True the result must always be a real,
    module-registered nn.Parameter -- so .parameters(), state_dict(), and
    torch.func.functional_call name-based substitution all see it -- even if that means
    detaching from whatever graph the incoming tensor carried."""
    c = torch.tensor(1.3, requires_grad=True)
    el = _Linear(c, learnable=True)
    assert isinstance(el.k, torch.nn.Parameter)
    assert list(el.parameters()) != []
    assert "k" in el.state_dict()


class _ConstantIgnoringDp(Element):
    """Buggy element: flow does not depend on dp at all (no learnable state either), so
    the returned tensor never requires grad."""

    def __init__(self, q0):
        super().__init__(kind="test")
        self.q0 = q0

    def flow(self, dp, drivers=None):
        return self.q0 + 0.0 * dp.detach()


def test_dflow_default_raises_a_clear_error_naming_the_element_when_flow_ignores_dp():
    # Same hazard as `_dflows_functional`'s guard, one call earlier: `Element.linear_init`'s
    # default reaches this default `dflow` (via `PotentialFlowLayer.linear_init`, called
    # before every solve()), and a `flow()` that silently drops the autograd graph must
    # raise a clear, element-naming error here too rather than a raw, unattributed one.
    el = _ConstantIgnoringDp(torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="_ConstantIgnoringDp"):
        el.dflow(torch.tensor([0.0, 1.0]))


class _DetachedButLearnable(Element):
    """Buggy element: flow requires grad (through its own learnable parameter `k`), but
    never actually uses its dp input, so autograd.grad(flow.sum(), dp) raises "not used in
    the graph" rather than "does not require grad"."""

    def __init__(self, k):
        super().__init__(kind="test")
        self.k = self._param(k, learnable=True)

    def flow(self, dp, drivers=None):
        return self.k + 0.0 * dp.detach()


def test_dflow_default_raises_a_clear_error_naming_the_element_when_dp_is_unused():
    el = _DetachedButLearnable(torch.tensor(2.0))
    with pytest.raises(RuntimeError, match="_DetachedButLearnable"):
        el.dflow(torch.tensor([0.0, 1.0]))


def test_forward_delegates_to_the_subclass_flow_override_not_the_base_class():
    """Guards against a naive `forward = flow` class-body assignment.

    That form binds the base class's own `flow` function object at class-definition time,
    so subclasses overriding `flow` would silently be unreachable through `__call__`/
    `forward` (and therefore through `torch.func.functional_call`, which the differentiable
    solve depends on).
    """
    el = _Cubic(2.0)
    dp = torch.tensor([-2.0, 0.5, 3.0])
    torch.testing.assert_close(el(dp), el.flow(dp))
    torch.testing.assert_close(el.forward(dp), el.flow(dp))
    with pytest.raises(NotImplementedError):
        Element(kind="airpath").forward(torch.tensor([1.0]))
