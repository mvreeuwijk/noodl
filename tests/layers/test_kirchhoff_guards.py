"""Refusals a KirchhoffLayer owes its caller: a stale topology and unreachable gradients."""

import dataclasses
import pickle

import pytest
import torch

from noodl.elements import Conductance
from noodl.elements.base import ConstitutiveLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

F64 = torch.float64


def _net() -> Network:
    net = Network(dtype=F64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="conduction")
    return net


def _layer(net: Network) -> KirchhoffLayer:
    return KirchhoffLayer(
        net, "heat", [Conductance(torch.tensor(1.0, dtype=F64))], boundary=["a"]
    )


# ------------------------------------------------------------------ R08
def test_a_layer_refuses_to_solve_after_an_edge_is_added():
    net = _net()
    layer = _layer(net)
    net.add_node("c")
    net.add_edge("b", "c", kind="conduction")
    with pytest.raises(RuntimeError, match="topology.*changed"):
        layer.solve(torch.ones(1, dtype=F64), differentiable=False)


def test_a_layer_refuses_component_calls_after_a_node_is_added():
    net = _net()
    layer = _layer(net)
    net.add_node("c")
    with pytest.raises(RuntimeError, match="topology.*changed"):
        layer.flows(torch.zeros(2, dtype=F64), {})


def test_a_new_layer_over_the_changed_network_works():
    net = _net()
    _layer(net)
    net.add_node("c")
    net.add_edge("b", "c", kind="conduction")
    fresh = KirchhoffLayer(
        net, "heat", [Conductance(torch.ones(2, dtype=F64))], boundary=["a"]
    )
    phi, q = fresh.solve(torch.ones(1, dtype=F64), differentiable=False)
    assert torch.allclose(phi, torch.ones(3, dtype=F64))
    assert q.shape == (2,)


def test_re_adding_an_existing_node_with_attributes_is_not_a_topology_change():
    net = _net()
    layer = _layer(net)
    net.add_node("b", volume=3.0)
    phi, q = layer.solve(torch.ones(1, dtype=F64), differentiable=False)
    assert torch.allclose(q, torch.zeros(1, dtype=F64))


# ------------------------------------------------------------------ H01
class _BufferLaw(ConstitutiveLaw):
    def __init__(self, g):
        super().__init__("conduction")
        self.register_buffer("g", g)

    def flow(self, dp, drivers=None):
        return self.g * dp

    def dflow(self, dp, drivers=None):
        return self.g * torch.ones_like(dp)

    def linear_init(self, drivers=None):
        return torch.zeros_like(self.g), self.g


class _ContainerLaw(_BufferLaw):
    def __init__(self, g):
        ConstitutiveLaw.__init__(self, "conduction")
        self.coeffs = {"g": [g]}
        self.g = g.detach()

    def flow(self, dp, drivers=None):
        return self.coeffs["g"][0] * dp

    def dflow(self, dp, drivers=None):
        return self.coeffs["g"][0] * torch.ones_like(dp)

    def linear_init(self, drivers=None):
        g = self.coeffs["g"][0]
        return torch.zeros_like(g), g


@pytest.mark.parametrize("law_class", [_BufferLaw, _ContainerLaw])
def test_a_gradient_tensor_in_a_buffer_or_container_is_refused(law_class):
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    layer = KirchhoffLayer(_net(), "heat", [law_class(g)], boundary=["a"])
    with pytest.raises(ValueError, match="not a registered nn.Parameter"):
        layer.solve(torch.zeros(1, dtype=F64), sources=torch.tensor([0.0, 1.0], dtype=F64))


# H01: the exact check (`KirchhoffLayer._require_threaded`) evaluates the residual with every
# threaded tensor detached, so a tensor reached by any other route is refused.
@dataclasses.dataclass
class _Holder:
    g: torch.Tensor


class _ObjectLaw(ConstitutiveLaw):
    """Holds its coefficient inside a plain object (a dataclass)."""

    def __init__(self, g):
        super().__init__("conduction")
        self.h = _Holder(g)

    def flow(self, dp, drivers=None):
        return self.h.g * dp

    def linear_init(self, drivers=None):
        return torch.zeros_like(self.h.g), self.h.g.detach()


class _ModuleListLaw(ConstitutiveLaw):
    """Holds a learnable law in a plain Python list: its parameter is not registered."""

    def __init__(self, g):
        super().__init__("conduction")
        self.parts = [Conductance(torch.nn.Parameter(g))]

    def flow(self, dp, drivers=None):
        return self.parts[0].flow(dp)

    def linear_init(self, drivers=None):
        return self.parts[0].linear_init()


class _NestedDrive:
    """A drive that owns a gradient tensor two levels down."""

    kind = "conduction"

    def __init__(self, g):
        self.table = {"coeffs": [_Holder(g)]}

    def __call__(self, drivers):
        return self.table["coeffs"][0].g * torch.ones(1, dtype=F64)


class _ClosureLaw(ConstitutiveLaw):
    """Captures its coefficient in a closure, where no attribute scan can see it."""

    def __init__(self, g):
        super().__init__("conduction")
        self._coefficient = lambda: g

    def flow(self, dp, drivers=None):
        return self._coefficient() * dp

    def linear_init(self, drivers=None):
        g = self._coefficient().detach()
        return torch.zeros_like(g), g


def _solve_unit_source(layer):
    return layer.solve(torch.zeros(1, dtype=F64), sources=torch.tensor([0.0, 1.0], dtype=F64))


def test_a_gradient_tensor_in_a_plain_object_attribute_is_refused():
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    layer = KirchhoffLayer(_net(), "heat", [_ObjectLaw(g)], boundary=["a"])
    with pytest.raises(ValueError, match=r"'h\.g'.*not a registered nn.Parameter"):
        _solve_unit_source(layer)


def test_a_module_in_a_plain_list_is_refused():
    layer = KirchhoffLayer(
        _net(), "heat", [_ModuleListLaw(torch.tensor([2.0], dtype=F64))], boundary=["a"]
    )
    with pytest.raises(ValueError, match=r"parts\[0\]\._parameters\['g'\]"):
        _solve_unit_source(layer)


def test_a_gradient_tensor_nested_in_a_drive_is_refused():
    g = torch.tensor(0.5, dtype=F64, requires_grad=True)
    layer = KirchhoffLayer(
        _net(), "heat", [Conductance(torch.tensor(1.0, dtype=F64))], boundary=["a"],
        drives=[_NestedDrive(g)],
    )
    with pytest.raises(ValueError, match=r"drive .*table\['coeffs'\]\[0\]\.g"):
        _solve_unit_source(layer)


class _ModuleDrive(torch.nn.Module):
    """A drive with a registered parameter: drives get no `functional_call`, so even a
    registered parameter of a drive is not threaded (it gave dq/dh = 1.0, true 0.5)."""

    kind = "conduction"

    def __init__(self):
        super().__init__()
        self.h = torch.nn.Parameter(torch.tensor(1.0, dtype=F64))

    def __call__(self, drivers):
        return self.h * torch.tensor([1.0, 0.0], dtype=F64)


def test_a_registered_parameter_of_a_module_drive_is_refused():
    net = Network(dtype=F64)
    for n in "abc":
        net.add_node(n)
    net.add_edge("a", "b", kind="conduction")
    net.add_edge("b", "c", kind="conduction")
    layer = KirchhoffLayer(
        net, "heat", [Conductance(torch.tensor(1.0, dtype=F64))], boundary=["a", "c"],
        drives=[_ModuleDrive()],
    )
    with pytest.raises(ValueError, match=r"drive .*_parameters\['h'\]"):
        layer.solve(torch.zeros(2, dtype=F64))


def test_a_gradient_tensor_captured_by_a_closure_is_refused():
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    layer = KirchhoffLayer(_net(), "heat", [_ClosureLaw(g)], boundary=["a"])
    with pytest.raises(ValueError, match="residual depends on a tensor"):
        _solve_unit_source(layer)


def test_threaded_tensors_are_not_refused():
    """No false positive: a ParameterList, a grad-requiring driver and boundary, and a
    numeric table too long to scan all reach the residual only through `Function.apply`."""

    class _ParameterListLaw(ConstitutiveLaw):
        def __init__(self, g):
            super().__init__("conduction")
            self.gs = torch.nn.ParameterList([torch.nn.Parameter(g)])
            self.table = [float(i) for i in range(100_000)]

        def flow(self, dp, drivers=None):
            return self.gs[0] * drivers["scale"] * dp

        def linear_init(self, drivers=None):
            return torch.zeros_like(self.gs[0]), (self.gs[0] * drivers["scale"]).detach()

    law = _ParameterListLaw(torch.tensor([2.0], dtype=F64))
    layer = KirchhoffLayer(_net(), "heat", [law], boundary=["a"])
    scale = torch.tensor([1.0], dtype=F64, requires_grad=True)
    pb = torch.tensor([0.5], dtype=F64, requires_grad=True)
    phi, _ = layer.solve(pb, {"scale": scale}, sources=torch.tensor([0.0, 1.0], dtype=F64))
    # phi_b = pb + 1 / (g * scale)
    g_grad, s_grad, pb_grad = torch.autograd.grad(phi[1], (law.gs[0], scale, pb))
    assert torch.allclose(g_grad, torch.tensor([-0.25], dtype=F64))
    assert torch.allclose(s_grad, torch.tensor([-0.5], dtype=F64))
    assert torch.allclose(pb_grad, torch.tensor([1.0], dtype=F64))


# ------------------------------------------------------------------ R08: old pickles
def _as_if_pickled_before_the_guard(layer: KirchhoffLayer) -> KirchhoffLayer:
    """Round-trip `layer` through pickle with the attributes older versions lacked removed."""
    del layer.net.__dict__["_version"]
    del layer.__dict__["_topology_version"]
    return pickle.loads(pickle.dumps(layer))


def test_an_old_pickle_without_topology_versions_still_solves():
    layer = _as_if_pickled_before_the_guard(_layer(_net()))
    assert layer.net.topology_version == 0
    phi, q = layer.solve(torch.ones(1, dtype=F64), differentiable=False)
    assert torch.allclose(q, torch.zeros(1, dtype=F64))


def test_an_old_pickled_network_counts_changes_and_its_layer_refuses_them():
    layer = _as_if_pickled_before_the_guard(_layer(_net()))
    layer.net.add_node("c")
    assert layer.net.topology_version == 1
    with pytest.raises(RuntimeError, match="topology.*changed"):
        layer.solve(torch.ones(1, dtype=F64), differentiable=False)
